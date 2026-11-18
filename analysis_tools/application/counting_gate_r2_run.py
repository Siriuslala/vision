"""Experiment 7.3 Part C round 2 stage drivers (gate modifications: training and inference-time edits).

Results doc: docs/analysis/application/results_2026-11-11_counting_gate_r2.md.

A condition is either an inference-time edit (--spec with only 'fix' parts, no training) or a trained gate
modification (--spec with trainable parts; stage 'train' saves adapters/<name>.pt).  Round-1 adapters of
experiment 7.3C are evaluated under the names 'r1:<name>'.

Stages (comma list via --stages):
  headsweep  (base) per head of --sweep-layer: forgetting removed in that single head at image tokens; long-axis
             ordinal answers on 1.5 H64 / V64 base images;
  train      trainable --spec on the 7.3C training mix (same loop and data as round 1);
  eval       sets of counting_gate_run.stage_eval (ord, think, g8, probe, count, coord, general, babyvision) plus
             gmulti (6x6 / 10x10 / 12x12 grids), ansprobe (answer-position residuals and candidate log-probs on the
             1.5 base images) and gatestat (alpha / beta per head at line-onset vs other figure tokens).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.application import counting_gate_r2 as cg2
from analysis_tools.application import counting_gate_run as cgr
from analysis_tools.application import interp_opt_run as ior
from analysis_tools.application import located_finetune_run as lfr
from analysis_tools.behavior import ordinal_mechanism as om
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import rowcol_mechanism as rm
from analysis_tools.position import sequence_order as so
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_run as tfr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

SEED = 20261111
CAND_OFFSETS = (-2, -1, 0, 1, 2)
GMULTI_SIZES = (6, 10, 12)
GMULTI_PER_SIZE = 48


# ----------------------------------------------------------------------------- conditions

def integrators(args) -> dict[int, list[int]]:
    return rm.head_sets(Path(args.gate_stats))['integrator']


def make_mod(rt, spec: str, args) -> cg2.GateMod:
    mod = cg2.GateMod(rt.layers, spec, integrators(args))
    return mod


def attach_mod(rt, mod: cg2.GateMod) -> None:
    mod.attach()
    if mod.positions == 'img':
        tf.attach_image_mask(rt.model, mod, rt.model.config.image_token_id)


def load_condition(rt, out: Path, args):
    """Attached modification for args.name (None for 'base')."""
    name = args.name
    if name == 'base' or name.endswith('_base'):          # unmodified model (e.g. 'vl_base' for Qwen3-VL)
        return None
    if name.startswith('r1:'):
        # 'r1:<adapter>' or 'r1:<adapter>@<layers>' (only the listed layers' modules apply their update)
        adapter, _, keep = name[3:].partition('@')
        lora = cgr.load_adapter(rt, Path(args.r1_run), adapter)
        if keep:
            layers = {int(x) for x in keep.split(',')}
            lora.active = {n for n in lora.names if int(n.split('.')[0][1:]) in layers}
            if not lora.active:
                raise ValueError(f'{name}: no adapted module in layers {sorted(layers)}')
        return lora
    path = out / 'adapters' / f'{name}.pt'
    if path.exists():
        st = torch.load(path, map_location='cpu', weights_only=False)
        mod = make_mod(rt, st['spec'], args)
        mod.load(st)
    else:
        if not args.spec:
            raise ValueError(f'{name}: no trained adapter and no --spec')
        if cg2.n_trainable(args.spec):
            raise ValueError(f'{name}: spec {args.spec!r} has trainable parts but no adapter file {path}')
        mod = make_mod(rt, args.spec, args)
    attach_mod(rt, mod)
    return mod


@contextmanager
def patched_loader(mod):
    """Make counting_gate_run / located_finetune_run use the already attached modification."""
    o1, o2 = cgr.load_adapter, lfr.load_adapter
    cgr.load_adapter = lambda *a, **k: None
    lfr.load_adapter = lambda *a, **k: None
    try:
        yield
    finally:
        cgr.load_adapter, lfr.load_adapter = o1, o2


# ----------------------------------------------------------------------------- items

def gmulti_items() -> list[dict]:
    rng = np.random.default_rng(SEED)
    out = []
    for n in GMULTI_SIZES:
        for i in range(GMULTI_PER_SIZE):
            r, c = int(rng.integers(n)), int(rng.integers(n))
            out.append({'id': f'g{n}_{i:02d}', 'n': n, 'template': ('person', 'ring')[i % 2], 'target': [r, c]})
    return out


def base_cases(args) -> list[dict]:
    return [c for c in cgr.ordinal_cases(args.ordinal_run) if c['variant'] == 'base']


def answer_prefix(case: dict, lead: str = '') -> str:
    """Model's own direct-answer format up to the long-axis number (the short-axis value is the truth); `lead` is
    the text the model writes before the JSON object (empty for Qwen3.5, e.g. a code fence for other models)."""
    return lead + ('{"row": ' if case['axis'] == 'V' else '{"row": %d, "column": ' % case['truth']['other'])


def answer_lead(d: Path) -> str:
    """Most common text before '{' in this condition's own direct answers (ord.jsonl), '' if none."""
    rows = read_jsonl(d / 'ord.jsonl') if (d / 'ord.jsonl').exists() else []
    leads = [r['text'].split('{')[0] for r in rows if '{' in r['text']]
    return max(set(leads), key=leads.count) if leads else ''


# ----------------------------------------------------------------------------- stages

def stage_headsweep(rt, out: Path, args) -> None:
    d = out / 'headsweep'
    d.mkdir(parents=True, exist_ok=True)
    path = d / f'L{args.sweep_layer}.jsonl'
    done = {r['head'] for r in read_jsonl(path)} if path.exists() else set()
    items = [(c['condition'], c, Image.open(Path(args.ordinal_run) / c['image']).convert('RGB'), c['prompt'])
             for c in base_cases(args) if c['condition'] in ('H64', 'V64')]
    for head in [-1] + list(range(cg2.cg.N_HEADS)):
        if head in done:
            continue
        t0 = time.time()
        mod = None
        if head >= 0:
            mod = make_mod(rt, f'fix:{args.sweep_layer}:{head}:1@img', args)
            attach_mod(rt, mod)
        try:
            res = cgr.run_batch(rt, items, 24)
        finally:
            if mod is not None:
                mod.detach()
        append_jsonl(path, [{'head': head, 'ids': [c['id'] for _, c, _, _ in items],
                             'err': [cgr.long_correct(c, r['text'])['err'] for (_, c, _, _), r in zip(items, res)]}])
        print(f'[headsweep] L{args.sweep_layer} H{head} ({time.time() - t0:.0f}s)', flush=True)


class _TrainMod(cg2.GateMod):
    def __init__(self, rt, spec, integ):
        super().__init__(rt.layers, spec, integ)
        self._rt = rt

    def attach(self) -> None:
        super().attach()
        if self.positions == 'img':
            tf.attach_image_mask(self._rt.model, self, self._rt.model.config.image_token_id)


def stage_train(rt, out: Path, args) -> None:
    if not cg2.n_trainable(args.spec):
        raise ValueError(f'spec {args.spec!r} has nothing to train')
    integ = integrators(args)
    orig = lfr.build_lora
    lfr.build_lora = lambda rt_, scope, rank, alpha: _TrainMod(rt_, scope, integ)
    args.scope = args.spec
    try:
        lfr.stage_train(rt, out, args)
    finally:
        lfr.build_lora = orig


def stage_eval(rt, out: Path, args) -> None:
    d = out / 'eval' / args.name
    d.mkdir(parents=True, exist_ok=True)
    sets = args.sets.split(',')
    mod = load_condition(rt, out, args)
    t0 = time.time()
    try:
        with patched_loader(mod):
            std = [s for s in sets if s in ('ord', 'think', 'g8', 'probe', 'count', 'coord', 'general', 'babyvision')]
            if std:
                cgr.stage_eval(rt, out, argparse.Namespace(**{**vars(args), 'sets': ','.join(std)}))
            if 'gatestat' in sets and not (d / 'gatestat.json').exists():
                cgr.stage_gatestat(rt, out, args)
        if 'gmulti' in sets and not (d / 'gmulti.jsonl').exists():
            its = gmulti_items()
            items = [('g', it, so.grid_image(it['template'], it['n'], it['n'], tuple(it['target']))[0], om.DIRECT_PROMPT)
                     for it in its]
            res = cgr.run_batch(rt, items, 24)
            append_jsonl(d / 'gmulti.jsonl', [{'id': it['id'], 'n': it['n'], 'target': it['target'], 'text': r['text'],
                                               **cgr.g8_score(it, r['text'])} for it, r in zip(its, res)])
            print(f'gmulti {args.name} {time.time() - t0:.0f}s', flush=True)
        if 'headout' in sets and not (d / f'headout_L{args.sweep_layer}.npz').exists():
            stage_headout(rt, args, d)
            print(f'headout {args.name} {time.time() - t0:.0f}s', flush=True)
        if 'ansprobe' in sets and not (d / 'ansprobe.npz').exists():
            stage_ansprobe(rt, args, d)
            print(f'ansprobe {args.name} {time.time() - t0:.0f}s', flush=True)
    finally:
        if mod is not None:
            mod.detach()


@torch.inference_mode()
def stage_ansprobe(rt, args, d: Path) -> None:
    """On the 1.5 base images (5 conditions x 64): residual stream at the last answer-prefix token (the position
    that emits the long-axis number) and the mean over the target cell's tokens after every 4th decoder layer, and the
    log-probability of the candidate numbers k + CAND_OFFSETS followed by the format's next character."""
    tok = rt.tokenizer
    n_layers = len(rt.layers)
    layers = [l for l in range(4, n_layers + 1, 4)]
    lead = answer_lead(d)
    rows = []
    ans_feats, tgt_feats = [], []
    for c in base_cases(args):
        img = Image.open(Path(args.ordinal_run) / c['image']).convert('RGB')
        head = qr.prompt_text(rt, 1, c['prompt']) + answer_prefix(c, lead)
        term = ',' if c['axis'] == 'V' else '}'
        k = c['truth']['count']
        base_batch = rt.processor(text=[head], images=[img], return_tensors='pt').to(rt.device)
        n_head = base_batch['input_ids'].shape[1]
        span = qr.image_spans(rt, base_batch)[0]
        W = c['truth']['merged_grid'][1]
        x0, y0, x1, y1 = c['truth']['target_cell_box']
        toks = [r * W + q for r in range(y0 // 32, max(y0 // 32 + 1, y1 // 32))
                for q in range(x0 // 32, max(x0 // 32 + 1, x1 // 32))]
        tpos = torch.tensor([span['start'] + t for t in toks])
        logp = {}
        for off in CAND_OFFSETS:
            n = k + off
            if n < 1:
                continue
            batch = rt.processor(text=[head + str(n) + term], images=[img], return_tensors='pt').to(rt.device)
            ids = batch['input_ids'][0]
            if not torch.equal(ids[:n_head], base_batch['input_ids'][0]):
                raise ValueError('candidate tokenisation changed the answer prefix')
            if off == 0:
                with qr.capture(rt, layers) as store:
                    logits = rt.model(**batch, use_cache=False).logits[0].float()
                ans_feats.append(np.stack([store[l][n_head - 1].half().numpy() for l in layers]))
                tgt_feats.append(np.stack([store[l][tpos].mean(0).half().numpy() for l in layers]))
            else:
                logits = rt.model(**batch, use_cache=False).logits[0].float()
            lp = torch.log_softmax(logits[n_head - 1:-1], -1)
            logp[n] = float(lp[torch.arange(len(ids) - n_head), ids[n_head:]].sum())
        rows.append({'id': c['id'], 'condition': c['condition'], 'k': k, 'logp': logp,
                     'n_tokens_number': len(tok.encode(str(k), add_special_tokens=False))})
    np.savez(d / 'ansprobe.npz', ans=np.stack(ans_feats), tgt=np.stack(tgt_feats), layers=np.array(layers), lead=np.array(lead),
             ids=np.array([r['id'] for r in rows]), k=np.array([r['k'] for r in rows]),
             cond=np.array([r['condition'] for r in rows]))
    (d / 'ansprobe.json').write_text(json.dumps(rows))


@torch.inference_mode()
def stage_headout(rt, args, d: Path) -> None:
    """Per-head output of DeltaNet layer --sweep-layer (gated RMSNorm output, the input of out_proj, 32 heads x 128)
    averaged over the target cell's tokens, and the residual stream after the layer and after the next layer at the
    same tokens, on the 1.5 base images; out_proj weights are saved once so the per-head written vectors
    out_proj[:, h] u_h can be formed offline."""
    l = args.sweep_layer
    mod = rt.layers[l - 1].linear_attn
    grab = {}
    h = mod.out_proj.register_forward_pre_hook(lambda m, a: grab.__setitem__('u', a[0].detach()))
    U, R0, R1, ids = [], [], [], []
    try:
        for c in base_cases(args):
            img = Image.open(Path(args.ordinal_run) / c['image']).convert('RGB')
            batch = qr.build(rt, [img], c['prompt'])
            span = qr.image_spans(rt, batch)[0]
            W = c['truth']['merged_grid'][1]
            x0, y0, x1, y1 = c['truth']['target_cell_box']
            toks = [r * W + q for r in range(y0 // 32, max(y0 // 32 + 1, y1 // 32))
                    for q in range(x0 // 32, max(x0 // 32 + 1, x1 // 32))]
            pos = torch.tensor([span['start'] + t for t in toks])
            with qr.capture(rt, [l - 1, l, l + 1]) as store:
                rt.model(**batch, use_cache=False, logits_to_keep=1)
            U.append(grab['u'][0, pos.to(grab['u'].device)].float().mean(0).view(cg2.cg.N_HEADS, -1).cpu().half().numpy())
            R0.append(np.stack([store[x][pos].mean(0).half().numpy() for x in (l - 1, l, l + 1)]))
            ids.append(c['id'])
    finally:
        h.remove()
    np.savez(d / f'headout_L{l}.npz', u=np.stack(U), resid=np.stack(R0), resid_layers=np.array([l - 1, l, l + 1]),
             ids=np.array(ids))
    wpath = d.parent / f'out_proj_L{l}.npy'
    if not wpath.exists():
        np.save(wpath, mod.out_proj.weight.detach().float().cpu().half().numpy())


STAGES = {'headsweep': stage_headsweep, 'train': stage_train, 'eval': stage_eval}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--name', default='base')
    p.add_argument('--spec', default='')
    p.add_argument('--sweep-layer', type=int, default=15)
    p.add_argument('--r1-run', required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--sets', default='ord,g8,probe,ansprobe')
    p.add_argument('--ordinal-run', required=True)
    p.add_argument('--gate-stats', required=True)
    p.add_argument('--interp-run', required=True)
    p.add_argument('--finetune-run', required=True)
    p.add_argument('--data-dir', default='')
    p.add_argument('--vcot-gqa', default='')
    p.add_argument('--mmvp-csv', default='')
    p.add_argument('--n-odd', type=int, default=1200)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--step-samples', type=int, default=16)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--alpha', type=float, default=0.0)
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
    rt = tfr.load_model(args)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages, 'name': args.name,
                         'spec': args.spec, 'gpu': torch.cuda.get_device_name(), 'args': vars(args), 'batch_size': 16,
                         'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]})
    meta.write_text(json.dumps(info, indent=1))
    for s in args.stages.split(','):
        t0 = time.time()
        STAGES[s](rt, out, args)
        print(f'stage {s} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
