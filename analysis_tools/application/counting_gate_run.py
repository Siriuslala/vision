"""Experiment 7.3 Part C stage drivers (counting-gate fine-tuning).

Results doc: docs/analysis/application/results_2026-11-06_counting_gate_finetune.md.

Stages (comma list via --stages):
  sweep    (GPU, base model, before any training) per Gated DeltaNet layer: prefill-only gate edits at the image
           tokens of that single layer (no forgetting in the non-integrator heads; no writing in all heads) and the
           resulting long-axis ordinal accuracy on 1.5 H64 / V64 base images and 3.8 8x8 grids;
  prepare  (CPU) training set (random 1.5-style slot grids with held-out templates + TextVQA) ;
  train    LoRA on --scope (counting_gate.parse_scope); gate scopes are restricted to image tokens;
  eval     1.5 ordinal cases (direct, all variants; think subset), 8x8 grids, grid counting, 7.1 coordinate subset,
           general benchmarks + text-only, BabyVision Find the different; count probe features at the target;
  gatestat alpha / beta of every DeltaNet head at the image tokens of 1.5 base / insL images.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.application import counting_gate as cg
from analysis_tools.application import interp_opt as io_
from analysis_tools.application import interp_opt_run as ior
from analysis_tools.application import located_finetune_run as lfr
from analysis_tools.behavior import ordinal_mechanism as om
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import rowcol_mechanism as rm
from analysis_tools.position import sequence_order as so
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_run as tfr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

SEED = 20261106
BIG = 1e4
N_ODD = 1200
SWEEP_SETS = ('H64', 'V64', 'g8')
SWEEP_EDITS = ('nodecay', 'nowrite')
# Pre-registered block-choice rule (fixed before the sweep was run): effect_l = max over the two edits of the mean over
# the three sets of (base accuracy - edited accuracy); chosen = layers with effect >= 0.10, at most 4, ranked by
# effect (if fewer than 2 qualify, the top 2); wrong-location control = the same number of layers with the smallest
# effect.
EFFECT_MIN, K_MAX, K_MIN = 0.10, 4, 2
PROBE_LAYERS = (8, 16, 24)
THINK_CONDITIONS = ('H64', 'V64')
THINK_VARIANTS = ('base', 'insL')


# ----------------------------------------------------------------------------- items

def ordinal_cases(run: Path, mode: str = 'direct') -> list[dict]:
    return [c for c in read_jsonl(Path(run) / 'cases.jsonl') if c['mode'] == mode]


def g8_items() -> list[dict]:
    return [{'id': f'g8_{t}_r{r}c{c}', 'template': t, 'target': [r, c]} for t in ('person', 'ring')
            for r in range(8) for c in range(8)]


def g8_image(it: dict) -> Image.Image:
    return so.grid_image(it['template'], 8, 8, tuple(it['target']))[0]


def long_correct(case: dict, text: str) -> dict:
    rc = om.parse_answer(text)
    long_, short = om.probed_answer(case, rc)
    t = case['truth']
    return {'valid': rc is not None, 'answer': long_, 'answer_other': short, 'correct': long_ == t['count'],
            'match_gap': long_ == t['gap'], 'match_slot': long_ == t['slot'], 'other_correct': short == t['other'],
            'err': None if long_ is None else long_ - t['count']}


def g8_score(it: dict, text: str) -> dict:
    rc = om.parse_answer(text)
    r, c = it['target']
    return {'valid': rc is not None, 'row_ok': rc is not None and rc[0] == r + 1,
            'col_ok': rc is not None and rc[1] == c + 1,
            'correct': rc is not None and rc[0] == r + 1 and rc[1] == c + 1}


# ----------------------------------------------------------------------------- gate edits (analysis tool)

@contextmanager
def gate_edit(rt, layer: int, edit: str, heads: list[int] | None):
    """Prefill-only edit at every image token of one DeltaNet layer: 'nodecay' (alpha = 1) or 'nowrite' (beta = 0)."""
    tok = rt.model.config.image_token_id
    state = {}

    def pre(m, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is not None and ids.shape[1] > 1:
            state['mask'] = ids == tok

    def hook(m, a, o):
        if o.shape[1] <= 1 or 'mask' not in state:
            return o
        o = o.clone()
        sub = o[state['mask']]
        idx = torch.arange(o.shape[-1], device=o.device) if heads is None else torch.tensor(heads, device=o.device)
        sub[:, idx] = -BIG
        o[state['mask']] = sub
        return o
    mod = rt.layers[layer - 1].linear_attn
    hs = [rt.model.model.register_forward_pre_hook(pre, with_kwargs=True),
          (mod.in_proj_a if edit == 'nodecay' else mod.in_proj_b).register_forward_hook(hook)]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def sweep_items(args) -> list[tuple[str, dict, Image.Image, str]]:
    out = []
    for c in ordinal_cases(args.ordinal_run):
        if c['variant'] == 'base' and c['condition'] in ('H64', 'V64'):
            out.append((c['condition'], c, Image.open(Path(args.ordinal_run) / c['image']).convert('RGB'), c['prompt']))
    for it in g8_items()[:64]:
        out.append(('g8', it, g8_image(it), om.DIRECT_PROMPT))
    return out


def run_batch(rt, items, max_new, bs=16):
    res = []
    groups: dict = {}
    for i, (_, _, img, p) in enumerate(items):
        groups.setdefault((img.size, p), []).append(i)
    out = [None] * len(items)
    for _, idx in groups.items():
        for a in range(0, len(idx), bs):
            ch = idx[a:a + bs]
            o = ior._orig_generate_batch(rt, [([items[i][2]], items[i][3]) for i in ch], max_new)
            for i, r in zip(ch, o):
                out[i] = r
    return out


def score_item(set_name, it, text) -> bool:
    if set_name == 'g8':
        s = g8_score(it, text)
        return (s['row_ok'] + s['col_ok']) / 2
    return float(long_correct(it, text)['correct'])


def stage_sweep(rt, out: Path, args) -> None:
    d = out / 'sweep'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'sweep.jsonl'
    done = {(r['layer'], r['edit']) for r in read_jsonl(path)} if path.exists() else set()
    items = sweep_items(args)
    integ = rm.head_sets(Path(args.gate_stats))['integrator']
    jobs = [(0, 'none')] + [(l, e) for l in cg.DN_LAYERS for e in SWEEP_EDITS]
    for layer, edit in jobs:
        if (layer, edit) in done:
            continue
        t0 = time.time()
        if layer == 0:
            res = run_batch(rt, items, 24)
        else:
            heads = [h for h in range(cg.N_HEADS) if h not in integ.get(layer, [])] if edit == 'nodecay' else None
            with gate_edit(rt, layer, edit, heads):
                res = run_batch(rt, items, 24)
        acc = {s: float(np.mean([score_item(s, it, r['text']) for (s2, it, _, _), r in zip(items, res) if s2 == s]))
               for s in SWEEP_SETS}
        err = {s: float(np.mean([long_correct(it, r['text'])['err'] or 0 for (s2, it, _, _), r in zip(items, res) if s2 == s]))
               for s in ('H64', 'V64')}
        append_jsonl(path, [{'layer': layer, 'edit': edit, 'acc': acc, 'err': err,
                             'answers': [r['text'] for r in res]}])
        print(f'[sweep] L{layer} {edit}: {acc} ({time.time() - t0:.0f}s)', flush=True)


def choose_layers(rows: list[dict]) -> dict:
    """Apply the pre-registered rule to the sweep rows."""
    base = next(r for r in rows if r['layer'] == 0)['acc']
    eff = {}
    for r in rows:
        if r['layer'] == 0:
            continue
        drop = float(np.mean([base[s] - r['acc'][s] for s in SWEEP_SETS]))
        eff[r['layer']] = max(eff.get(r['layer'], -9), drop)
    ranked = sorted(eff, key=lambda l: -eff[l])
    chosen = [l for l in ranked if eff[l] >= EFFECT_MIN][:K_MAX]
    if len(chosen) < K_MIN:
        chosen = ranked[:K_MIN]
    wrong = sorted(eff, key=lambda l: eff[l])[:len(chosen)]
    return {'effect': eff, 'chosen': sorted(chosen), 'wrong': sorted(wrong), 'base': base}


# ----------------------------------------------------------------------------- prepare / train

def stage_prepare(out: Path, args) -> None:
    rng = np.random.default_rng(SEED)
    items = []
    for k in range(N_ODD):
        spec, count, other = cg.random_slot_grid(rng)
        img, meta = om.render(spec)
        p = out / 'images' / 'train' / f'slot_{k:05d}.png'
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        r, c = cg.rowcol_answer(spec, count, other)
        items.append(io_.Item(f'slot_{k:05d}', 'odd', 'train', str(p), om.DIRECT_PROMPT,
                              io_.ROWCOL_TEMPLATE.format(r=r, c=c), {'row': r, 'col': c},
                              {'axis': spec.axis, 'cell': spec.cell, 'n_present': len(spec.present)}))
    vqa = [it for it in io_.read_items(Path(args.interp_run) / 'train.jsonl') if it.task == 'vqa']
    vqa = [vqa[int(i)] for i in rng.permutation(len(vqa))][:N_ODD]
    io_.dump_items(out / 'train_odd.jsonl', items)
    io_.dump_items(out / 'train_vqa.jsonl', vqa)
    print(json.dumps({'n_odd': len(items), 'n_vqa': len(vqa)}), flush=True)


def build_lora(rt, scope: str, rank: int, alpha: float) -> tf.LoRA:
    return tf.LoRA(cg.dn_modules(rt.layers, scope), rank, alpha)


def attach(rt, lora: tf.LoRA, scope: str) -> None:
    lora.attach()
    if cg.parse_scope(scope)['kind'] == 'gate':
        tf.attach_image_mask(rt.model, lora, rt.model.config.image_token_id)


def load_adapter(rt, out: Path, name: str):
    if name == 'base':
        return None
    st = torch.load(out / 'adapters' / f'{name}.pt', map_location='cpu', weights_only=False)
    lora = build_lora(rt, st['scope'], st['rank'], st['alpha'])
    lora.load(st)
    attach(rt, lora, st['scope'])
    return lora


def stage_train(rt, out: Path, args) -> None:
    # Same loop as located_finetune_run.stage_train with this module's adapters (image-token restricted gates).
    orig_build = lfr.build_lora
    lfr.build_lora = lambda rt_, scope, rank, alpha: _AttachingLoRA(rt_, scope, rank, alpha)
    try:
        lfr.stage_train(rt, out, args)
    finally:
        lfr.build_lora = orig_build


class _AttachingLoRA(tf.LoRA):
    """LoRA whose attach() also installs the image-token mask for gate scopes."""

    def __init__(self, rt, scope, rank, alpha):
        super().__init__(cg.dn_modules(rt.layers, scope), rank, alpha)
        self._rt, self._scope = rt, scope

    def attach(self) -> None:
        super().attach()
        if cg.parse_scope(self._scope)['kind'] == 'gate':
            tf.attach_image_mask(self._rt.model, self, self._rt.model.config.image_token_id)


# ----------------------------------------------------------------------------- eval

def stage_eval(rt, out: Path, args) -> None:
    name = args.name
    d = out / 'eval' / name
    d.mkdir(parents=True, exist_ok=True)
    sets = args.sets.split(',')
    lora = load_adapter(rt, out, name)
    t0 = time.time()
    try:
        if 'ord' in sets and not (d / 'ord.jsonl').exists():
            cases = ordinal_cases(args.ordinal_run)
            items = [(c['condition'], c, Image.open(Path(args.ordinal_run) / c['image']).convert('RGB'), c['prompt'])
                     for c in cases]
            res = run_batch(rt, items, 24)
            append_jsonl(d / 'ord.jsonl', [{'id': c['id'], 'condition': c['condition'], 'variant': c['variant'],
                                            'c': c['c'], 'text': r['text'], **long_correct(c, r['text'])}
                                           for (_, c, _, _), r in zip(items, res)])
            print(f'ord {name} {time.time() - t0:.0f}s', flush=True)
        if 'think' in sets and not (d / 'think.jsonl').exists():
            cases = [c for c in ordinal_cases(args.ordinal_run, 'think')
                     if c['condition'] in THINK_CONDITIONS and c['variant'] in THINK_VARIANTS]
            items = [(c['condition'], c, Image.open(Path(args.ordinal_run) / c['image']).convert('RGB'), c['prompt'])
                     for c in cases]
            res = run_batch(rt, items, args.max_new_think)
            append_jsonl(d / 'think.jsonl', [{'id': c['id'], 'condition': c['condition'], 'variant': c['variant'],
                                              'c': c['c'], 'text': r['text'], 'finish_reason': r['finish_reason'],
                                              **long_correct(c, r['text'])} for (_, c, _, _), r in zip(items, res)])
            print(f'think {name} {time.time() - t0:.0f}s', flush=True)
        if 'g8' in sets and not (d / 'g8.jsonl').exists():
            its = g8_items()
            items = [('g8', it, g8_image(it), om.DIRECT_PROMPT) for it in its]
            res = run_batch(rt, items, 24)
            append_jsonl(d / 'g8.jsonl', [{'id': it['id'], 'target': it['target'], 'text': r['text'],
                                           **g8_score(it, r['text'])} for it, r in zip(its, res)])
            print(f'g8 {name} {time.time() - t0:.0f}s', flush=True)
        if 'probe' in sets and not (d / 'probe.npz').exists():
            stage_probe_features(rt, args, d)
            print(f'probe {name} {time.time() - t0:.0f}s', flush=True)
        rest = [s for s in sets if s in ('count', 'coord', 'general')]
        if rest:
            ns = Namespace(**{**vars(args), 'sets': ','.join(rest)})
            _with_lora_eval(rt, out, ns)
        if 'babyvision' in sets and not (d / 'babyvision.jsonl').exists():
            from eva.benchmark import score as bv_score
            cases = [c for c in lfr.babyvision_cases(Path(args.data_dir)) if c['subtype'] == 'Find the different']
            outs = ior._orig_generate_batch(rt, [([io_.fit_area(Image.open(c['image']), 1024 * 1024)], c['prompt'])
                                                 for c in cases], args.bv_max_new)
            append_jsonl(d / 'babyvision.jsonl', [{'id': c['id'], 'subtype': c['subtype'], 'text': o['text'],
                                                   'finish_reason': o['finish_reason'],
                                                   'score': bv_score(c['row'], o['text'], 'babyvision')}
                                                  for c, o in zip(cases, outs)])
            print(f'babyvision {name} {time.time() - t0:.0f}s', flush=True)
    finally:
        if lora is not None:
            lora.detach()


def _with_lora_eval(rt, out: Path, args) -> None:
    """located_finetune_run.stage_eval for count / coord / general with the adapter already attached."""
    orig = lfr.load_adapter
    lfr.load_adapter = lambda rt_, out_, name: None
    try:
        lfr.stage_eval(rt, out, args)
    finally:
        lfr.load_adapter = orig


def stage_probe_features(rt, args, d: Path) -> None:
    """Residual stream at the target cell (mean over its tokens) after decoder layers PROBE_LAYERS, all 1.5 direct
    images: the running-count probe of experiment 3.10 restricted to the target."""
    cases = ordinal_cases(args.ordinal_run)
    feats = {l: [] for l in PROBE_LAYERS}
    ids, ks, gaps, conds, variants = [], [], [], [], []
    with torch.inference_mode():
        for c in cases:
            img = Image.open(Path(args.ordinal_run) / c['image']).convert('RGB')
            batch = qr.build(rt, [img], c['prompt'])
            span = qr.image_spans(rt, batch)[0]
            W = c['truth']['merged_grid'][1]
            x0, y0, x1, y1 = c['truth']['target_cell_box']
            toks = [r * W + q for r in range(y0 // 32, max(y0 // 32 + 1, y1 // 32)) for q in range(x0 // 32, max(x0 // 32 + 1, x1 // 32))]
            pos = torch.tensor([span['start'] + t for t in toks])
            with qr.capture(rt, list(PROBE_LAYERS)) as store:
                rt.model(**batch, use_cache=False, logits_to_keep=1)
            for l in PROBE_LAYERS:
                feats[l].append(store[l][pos].float().mean(0).half().numpy())
            ids.append(c['id']); ks.append(c['truth']['count']); gaps.append(c['truth']['gap'])
            conds.append(c['condition']); variants.append(c['variant'])
    np.savez(d / 'probe.npz', ids=np.array(ids), k=np.array(ks), gap=np.array(gaps), cond=np.array(conds),
             variant=np.array(variants), **{f'L{l}': np.stack(v) for l, v in feats.items()})


def stage_gatestat(rt, out: Path, args) -> None:
    """Mean alpha and beta of every DeltaNet head over image tokens that start a figure line vs other figure tokens
    vs blank tokens, on the 1.5 base / insL images of H64 and V64 (with the adapter, if any)."""
    d = out / 'eval' / args.name
    d.mkdir(parents=True, exist_ok=True)
    lora = load_adapter(rt, out, args.name)
    cases = [c for c in ordinal_cases(args.ordinal_run) if c['condition'] in ('H64', 'V64') and c['variant'] in ('base', 'insL')]
    store = {}
    hs = []
    for l in cg.DN_LAYERS:
        mod = rt.layers[l - 1].linear_attn
        hs.append(mod.in_proj_a.register_forward_hook(lambda m, a, o, l=l: store.__setitem__(('a', l), o.detach())))
        hs.append(mod.in_proj_b.register_forward_hook(lambda m, a, o, l=l: store.__setitem__(('b', l), o.detach())))
    acc = {}
    try:
        with torch.inference_mode():
            for c in cases:
                img = Image.open(Path(args.ordinal_run) / c['image']).convert('RGB')
                batch = qr.build(rt, [img], c['prompt'])
                span = qr.image_spans(rt, batch)[0]
                H, W = c['truth']['merged_grid']
                cls = token_classes(c, H, W)                       # (H*W,) 0 blank, 1 line onset, 2 other figure
                rt.model(**batch, use_cache=False, logits_to_keep=1)
                for l in cg.DN_LAYERS:
                    mod = rt.layers[l - 1].linear_attn
                    a = store[('a', l)][0, span['start']:span['end']].float()
                    b = store[('b', l)][0, span['start']:span['end']].float()
                    alpha = torch.exp(-mod.A_log.float().exp() * torch.nn.functional.softplus(a + mod.dt_bias.float()))
                    beta = b.sigmoid()
                    for k in (0, 1, 2):
                        m = torch.tensor(cls == k, device=alpha.device)
                        if m.any():
                            key = (c['condition'], c['variant'], l, k)
                            s = acc.setdefault(key, [torch.zeros(cg.N_HEADS, device=alpha.device), torch.zeros(cg.N_HEADS, device=alpha.device), 0])
                            s[0] += alpha[m].sum(0); s[1] += beta[m].sum(0); s[2] += int(m.sum())
                store.clear()
    finally:
        for h in hs:
            h.remove()
        if lora is not None:
            lora.detach()
    res = [{'condition': k[0], 'variant': k[1], 'layer': k[2], 'cls': k[3], 'alpha': (v[0] / v[2]).tolist(),
            'beta': (v[1] / v[2]).tolist(), 'n': v[2]} for k, v in acc.items()]
    (d / 'gatestat.json').write_text(json.dumps(res))


def token_classes(case: dict, H: int, W: int) -> np.ndarray:
    """Per merged token: 0 = blank or margin, 1 = first token of a figure line along the long axis, 2 = other figure
    token.  For H (long axis horizontal) figure lines are columns and the onset is the first token column of a
    present slot; for V lines are rows and the onset is the first token row of a present slot."""
    t = case['truth']
    cell = t['cell']
    x0, y0 = t['target_cell_box'][:2]
    slot, other = t['target_slot'], t['other'] - 1
    gx, gy = (x0 - slot * cell, y0 - other * cell) if case['axis'] == 'H' else (x0 - other * cell, y0 - slot * cell)
    n_long, n_short = om.SLOTS, om.SHORT
    present = set(t['present'])
    cls = np.zeros((H, W), np.int64)
    for r in range(H):
        for q in range(W):
            px, py = q * 32 - gx, r * 32 - gy
            if px < 0 or py < 0:
                continue
            ci, ri = px // cell, py // cell
            long_i, short_i = (ci, ri) if case['axis'] == 'H' else (ri, ci)
            if long_i >= n_long or short_i >= n_short or long_i not in present:
                continue
            off = px % cell if case['axis'] == 'H' else py % cell
            cls[r, q] = 1 if off == 0 else 2
    return cls.reshape(-1)


STAGES = {'sweep': stage_sweep, 'train': stage_train, 'eval': stage_eval, 'gatestat': stage_gatestat}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--name', default='base')
    p.add_argument('--scope', default='')
    p.add_argument('--n-odd', type=int, default=N_ODD)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--sets', default='ord,think,g8,probe,count,coord,general,babyvision')
    p.add_argument('--ordinal-run', required=True)
    p.add_argument('--gate-stats', required=True)
    p.add_argument('--interp-run', required=True)
    p.add_argument('--finetune-run', required=True)
    p.add_argument('--data-dir', default='')
    p.add_argument('--vcot-gqa', default='')
    p.add_argument('--mmvp-csv', default='')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=32.0)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--step-samples', type=int, default=16)
    p.add_argument('--micro-tokens', type=int, default=3072)
    p.add_argument('--token-budget', type=int, default=16000)
    p.add_argument('--general-n', type=int, default=200)
    p.add_argument('--general-seed', type=int, default=20261020)
    p.add_argument('--general-batch', type=int, default=16)
    p.add_argument('--general-max-new', type=int, default=96)
    p.add_argument('--bv-batch', type=int, default=16)
    p.add_argument('--bv-max-new', type=int, default=1536)
    p.add_argument('--max-new-think', type=int, default=768)
    p.add_argument('--max-memory', default='')
    args = p.parse_args()
    torch.set_num_threads(16)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(',')
    if stages == ['prepare']:
        stage_prepare(out, args)
        return
    rt = tfr.load_model(args)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages, 'name': args.name,
                         'gpu': torch.cuda.get_device_name(), 'args': vars(args),
                         'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]})
    meta.write_text(json.dumps(info, indent=1))
    for s in stages:
        t0 = time.time()
        STAGES[s](rt, out, args)
        print(f'stage {s} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
