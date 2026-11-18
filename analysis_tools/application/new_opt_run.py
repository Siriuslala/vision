"""Experiment 7.5 stage drivers (see new_opt.py for the two pilots and their interpretability basis).

Stages (comma list via --stages):
  link      (CPU) link the 7.3 training sets, probe and base-model evaluation into this run (identical data and
            recipe, so the 7.3 ViT adapters are directly comparable);
  premask   (GPU, base model, before any training) teacher-forced 6 x 6 probe under own-entity key masks of
            PREMASK_CONDITIONS (or --premask list); records answer correctness, the own-entity attention mass of every
            layer-8 head for every image-token query, and the reader-head attention on the target cell;
  train     decoder LoRA of one scope (new_opt.SCOPES) at image tokens, with the 7.3 recipe (located_finetune_run);
  eval      7.3 evaluation sets (probe, count, coord incl. RefCOCO, POPE / TextVQA / MMVP / GQA, text-only,
            BabyVision find-the-different / find-the-same) for one adapter;
  attn      the premask measurements (no mask) for one adapter: did the layer-8 own-entity mass fall at early
            entities, and did the reader attention on early targets rise?
  format    (GPU, base model, no training) pilot B: the 3.13 coordinate test set under three answer formats.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.application import interp_opt as io_
from analysis_tools.application import interp_opt_run as ior
from analysis_tools.application import located_finetune_run as lfr
from analysis_tools.application import new_opt as no
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry_run as sr

LINKED = ('train_odd.jsonl', 'train_vqa.jsonl', 'probe.jsonl', 'prepare.json')


# ----------------------------------------------------------------------------- adapters (decoder, image tokens)

class ImageLoRA(tf.LoRA):
    """tf.LoRA whose update is restricted to image-token positions (pre-hook on the multimodal backbone)."""

    def __init__(self, model, modules, rank, alpha, image_token_id):
        super().__init__(modules, rank, alpha)
        self._model_ref = [model]          # list: keep the model out of nn.Module registration
        self._image_token_id = image_token_id

    def attach(self) -> None:
        super().attach()
        tf.attach_image_mask(self._model_ref[0], self, self._image_token_id)


def build_lora(rt, scope: str, rank: int, alpha: float) -> tf.LoRA:
    """Drop-in replacement of located_finetune_run.build_lora for the decoder scopes of new_opt.SCOPES."""
    s = no.SCOPES[scope]
    if rank != s['rank']:
        raise ValueError(f'scope {scope} is registered with rank {s["rank"]}, got {rank}')
    mods = {}
    for l in s['layers']:
        named = dict(rt.layers[l - 1].named_modules())
        for m in s['modules']:
            if not isinstance(named.get(m), torch.nn.Linear):
                raise ValueError(f'layer {l} has no linear module {m}')
            mods[f'L{l}.{m}'] = named[m]
    lora = ImageLoRA(rt.model, mods, rank, alpha, rt.model.config.image_token_id)
    if lora.n_params() != no.scope_params(scope):
        raise RuntimeError(f'parameter count {lora.n_params()} != registered {no.scope_params(scope)}')
    return lora


def install() -> None:
    lfr.build_lora = build_lora          # stage_train / load_adapter of 7.3 look the builder up at call time


# ----------------------------------------------------------------------------- link

def stage_link(out: Path, args) -> None:
    src = Path(args.vit_run)
    for name in LINKED:
        dst = out / name
        if not dst.exists():
            os.symlink(src / name, dst)
    (out / 'eval').mkdir(exist_ok=True)
    if not (out / 'eval' / 'base').exists():
        os.symlink(src / 'eval' / 'base', out / 'eval' / 'base')
    print('linked', [p.name for p in out.iterdir()], flush=True)


# ----------------------------------------------------------------------------- measurement passes

def parse_condition(cond: str, n_layers: int) -> list[tuple[int, tuple[int, ...] | None]]:
    """'none' | registered names | 'own_l{a}-{b}_all' (all heads of layers a..b)."""
    if cond in no.PREMASK_CONDITIONS:
        return no.PREMASK_CONDITIONS[cond]
    if cond.startswith('own_l') and cond.endswith('_all'):
        a, _, b = cond[5:-4].partition('-')
        layers = list(range(int(a), int(b or a) + 1))
        if min(layers) < 1 or max(layers) > n_layers:
            raise ValueError(cond)
        return [(l, None) for l in layers]
    raise ValueError(f'unknown mask condition {cond!r}')


def _attn_module(rt, layer: int):
    blk = rt.layers[layer - 1]
    att = getattr(blk, 'self_attn', None)
    if att is None or not hasattr(att, 'q_norm'):
        raise ValueError(f'layer {layer} is not a full-attention layer')
    return att


@contextmanager
def instrument(rt, L: int, B: int, mask_spec, groups, record_layers: tuple[int, ...]):
    """Own-entity key masks for the listed (layer, heads) and capture of post-norm q / k plus RoPE at record_layers."""
    store = {'q': {}, 'k': {}, 'pe': {}}
    hs = []
    own = no.own_entity_mask(L, groups)
    by_layer = defaultdict(list)
    for l, heads in mask_spec:
        by_layer[l].append(heads)
    for l, hl in by_layer.items():
        att = _attn_module(rt, l)
        H = att.config.num_attention_heads
        dev = att.o_proj.weight.device
        m = torch.ones(L, L, dtype=torch.bool, device=dev).tril()[None].repeat(H, 1, 1)
        for heads in hl:
            for h in (range(H) if heads is None else heads):
                m[h] = own.to(dev)
        mask = m[None].expand(B, H, L, L)

        def pre(mod, a, kw, mask=mask):
            if kw['hidden_states'].shape[1] != L:
                return a, kw
            kw['attention_mask'] = mask
            return a, kw
        hs.append(att.register_forward_pre_hook(pre, with_kwargs=True))
    for l in record_layers:
        att = _attn_module(rt, l)

        def pre_pe(mod, a, kw, l=l):
            if kw['hidden_states'].shape[1] == L:
                store['pe'][l] = kw['position_embeddings']
            return a, kw

        def hq(mod, a, o, l=l):
            if o.shape[1] == L:
                store['q'][l] = o.detach()

        def hk(mod, a, o, l=l):
            if o.shape[1] == L:
                store['k'][l] = o.detach()
        hs += [att.register_forward_pre_hook(pre_pe, with_kwargs=True), att.q_norm.register_forward_hook(hq),
               att.k_norm.register_forward_hook(hk)]
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


def attention_probs(rt, store, layer: int, qpos: list[int], mask: torch.Tensor | None) -> torch.Tensor:
    """(B, H, nq, L) attention probabilities of the queries at qpos recomputed from the captured q / k (post-norm,
    pre-RoPE, LoRA included); mask (L, L) bool (True = may attend) or None for causal."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    att = _attn_module(rt, layer)
    q, k = store['q'][layer].float(), store['k'][layer].float()        # (B, L, H, hd), (B, L, KV, hd)
    cos, sin = (t.float() for t in store['pe'][layer])
    q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)   # (B, H, L, hd)
    k = k.repeat_interleave(att.num_key_value_groups, dim=1)
    qq = q[:, :, qpos]
    s = torch.einsum('bhqd,bhkd->bhqk', qq, k) * att.scaling
    L = k.shape[2]
    allowed = torch.ones(L, L, dtype=torch.bool, device=s.device).tril() if mask is None else mask.to(s.device)
    s = s.masked_fill(~allowed[qpos][None, None], float('-inf'))
    return s.softmax(-1)


def measure(rt, items: list[io_.Item], cond: str, batch_size: int = 12) -> list[dict]:
    """Teacher-forced probe answers plus attention measurements under one mask condition."""
    n_layers = len(rt.layers)
    spec = parse_condition(cond, n_layers)
    readers = [(l, h) for l, h in no.READERS if l <= n_layers] if rt.family == 'qwen3_5' else []
    rec_layers = tuple(sorted({8} | {l for l, _ in readers})) if rt.family == 'qwen3_5' else ()
    digits = sr.digit_ids(rt, no.GRID)
    rows = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        answers, i_row, i_col = [], None, None
        for it in chunk:
            a, i_row, i_col = sr.answer_ids(rt, tuple(it.truth['target']))
            answers.append(a)
        batch = sr.build_rows(rt, [Image.open(it.image).convert('RGB') for it in chunk], answers)
        s, e = sr.image_span(rt, batch)
        L = batch['input_ids'].shape[1]
        n_ans = len(answers[0])
        q_row = L - n_ans + i_row - 1
        metas = [it.meta['meta'] for it in chunk]
        groups = [[s + t for t in toks] for toks in oe.entity_tokens(metas[0])]
        if any(oe.entity_tokens(m) != oe.entity_tokens(metas[0]) for m in metas):
            raise ValueError('probe entities do not share one token layout')
        with torch.no_grad(), instrument(rt, L, len(chunk), spec, groups, rec_layers) as st:
            logits = rt.model(**{k: v.to(rt.device) if torch.is_tensor(v) else v for k, v in batch.items()},
                              use_cache=False, logits_to_keep=n_ans + 1).logits
            scores = sr.score_rows(logits, answers, i_row, i_col, n_ans, digits)
            extra = [{} for _ in chunk]
            if rec_layers:
                # Causal (unmasked) layer-8 probabilities; under a layer-8 mask condition the masked heads' values
                # describe the scores, not the executed attention (analysis uses them only for unmasked runs).
                img_q = list(range(s, e))
                p8 = attention_probs(rt, st, 8, img_q, None)              # (B, H, nq, L)
                ent_of = {t: gi for gi, g in enumerate(groups) for t in g}
                own_cols = torch.zeros(len(img_q), L, dtype=torch.bool)
                prev_cols = torch.zeros(len(img_q), L, dtype=torch.bool)
                for qi, t in enumerate(img_q):
                    g = ent_of.get(t)
                    if g is None:
                        continue
                    own_cols[qi, groups[g]] = True
                    for gj in range(g):
                        prev_cols[qi, groups[gj]] = True
                own_mass = (p8 * own_cols.to(p8.device)[None, None]).sum(-1)     # (B, H, nq)
                prev_mass = (p8 * prev_cols.to(p8.device)[None, None]).sum(-1)
                # per-entity mean over the entity's tokens: (B, H, n_ent)
                ent_idx = [[img_q.index(t) for t in g] for g in groups]
                own_ent = torch.stack([own_mass[..., ix].mean(-1) for ix in ent_idx], -1)
                prev_ent = torch.stack([prev_mass[..., ix].mean(-1) for ix in ent_idx], -1)
                rd = []
                for l, h in readers:
                    p = attention_probs(rt, st, l, [q_row], None)[:, h, 0]          # (B, L)
                    rd.append(p)
                rd = torch.stack(rd, 1)                                             # (B, n_readers, L)
                for b, it in enumerate(chunk):
                    tg = it.truth['target']
                    tcell = groups[tg[0] * no.GRID + tg[1]]
                    extra[b] = {'own8': np.round(own_ent[b].cpu().numpy(), 4).tolist(),
                                'prev8': np.round(prev_ent[b].cpu().numpy(), 4).tolist(),
                                'reader_target': np.round(rd[b][:, tcell].sum(-1).cpu().numpy(), 4).tolist(),
                                'reader_image': np.round(rd[b][:, s:e].sum(-1).cpu().numpy(), 4).tolist()}
        for b, it in enumerate(chunk):
            tg = it.truth['target']
            sc = scores[b]
            rows.append({'id': it.id, 'cond': cond, 'pair': it.meta['pair'], 'variant': it.meta['variant'],
                         'split': it.meta['split'], 'family': it.meta['family'], 'target': tg,
                         'bin': no.raster_bin(tg), 'correct': bool(sc['arg_row'] == tg[0] + 1 and
                                                                  sc['arg_col'] == tg[1] + 1),
                         'row_ok': bool(sc['arg_row'] == tg[0] + 1), 'p_row': sc['p_row'], 'p_col': sc['p_col'],
                         **extra[b]})
    return rows


def stage_premask(rt, out: Path, args) -> None:
    d = out / 'premask'
    d.mkdir(exist_ok=True)
    items = io_.read_items(out / 'probe.jsonl')
    tag = rt.name.replace('/', '_')
    for cond in args.premask.split(','):
        path = d / f'{tag}__{cond}.jsonl'
        if path.exists():
            continue
        t0 = time.time()
        rows = measure(rt, items, cond, args.batch)
        append_jsonl(path, rows)
        acc = np.mean([r['correct'] for r in rows])
        print(f'[premask] {tag} {cond} acc={acc:.3f} {time.time() - t0:.0f}s', flush=True)


def stage_attn(rt, out: Path, args) -> None:
    d = out / 'eval' / args.name
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'attn.jsonl'
    if path.exists():
        return
    lora = lfr.load_adapter(rt, out, args.name)
    try:
        rows = measure(rt, io_.read_items(out / 'probe.jsonl'), 'none', args.batch)
    finally:
        if lora is not None:
            lora.detach()
    append_jsonl(path, rows)


# ----------------------------------------------------------------------------- decomposition of a q/k adapter

DECOMP_PARTS = ('full', 'gate', 'query_key')


def _restrict(lora: tf.LoRA, part: str, head_dim: int) -> None:
    """Zero the LoRA B rows outside one part. q_proj output = per head [query (hd), output gate (hd)];
    'gate' keeps only the gate rows of q_proj; 'query_key' keeps the query rows of q_proj and all of k_proj."""
    if part == 'full':
        return
    with torch.no_grad():
        for n in lora.names:
            B = lora.B[lora._key(n)]
            if n.endswith('q_proj'):
                j = torch.arange(B.shape[0], device=B.device) % (2 * head_dim)
                is_gate = j >= head_dim
                B[~is_gate if part == 'gate' else is_gate] = 0
            elif n.endswith('k_proj') and part == 'gate':
                B.zero_()


def stage_decomp(rt, out: Path, args) -> None:
    """For one q/k adapter: probe answers and reader attention with the full update, only its output-gate part
    and only its query/key part; plus the mean output gate sigmoid(g) per head at image tokens (base and full)."""
    d = out / 'eval' / args.name
    path = d / 'decomp.jsonl'
    if path.exists():
        return
    items = io_.read_items(out / 'probe.jsonl')
    state = torch.load(out / 'adapters' / f'{args.name}.pt', map_location='cpu', weights_only=False)
    layer = no.SCOPES[state['scope']]['layers'][0]
    att = _attn_module(rt, layer)
    hd = att.head_dim
    gate_sum = {}

    def gate_hook(mod, a, o, key=None):
        ids_mask = gate_ctx.get('mask')
        if ids_mask is None or o.shape[1] != ids_mask.shape[1]:
            return
        g = torch.sigmoid(o.float().view(*o.shape[:2], -1, 2 * hd)[..., hd:]).mean(-1)     # (B, L, H)
        m = ids_mask.to(g.device)[..., None]
        gate_sum[gate_ctx['part']] = gate_sum.get(gate_ctx['part'], 0) + (g * m).sum((0, 1)).cpu()
        gate_sum[gate_ctx['part'] + '_n'] = gate_sum.get(gate_ctx['part'] + '_n', 0) + float(m.sum())
    gate_ctx = {}

    def ids_pre(mod, a, kw):
        ids = kw.get('input_ids')
        if ids is not None:
            gate_ctx['mask'] = (ids == rt.model.config.image_token_id).float()
    h_ids = rt.model.model.register_forward_pre_hook(ids_pre, with_kwargs=True)
    rows = []
    try:
        for part in ('base',) + DECOMP_PARTS:
            gate_ctx['part'] = part
            lora = None
            if part != 'base':
                lora = build_lora(rt, state['scope'], state['rank'], state['alpha'])
                lora.load(state)
                _restrict(lora, part, hd)
                lora.attach()
            hg = att.q_proj.register_forward_hook(gate_hook)       # after the LoRA hook: sees the adapted output
            try:
                res = measure(rt, items, 'none', args.batch)
            finally:
                hg.remove()
                if lora is not None:
                    lora.detach()
            for r in res:
                r['part'] = part
                for k in ('own8', 'prev8'):
                    r.pop(k, None)
            rows += res
            sw = [r['correct'] for r in res if r['variant'] == 'swap']
            print(f'[decomp] {args.name} {part} swap TF acc {np.mean(sw):.3f}', flush=True)
    finally:
        h_ids.remove()
    append_jsonl(path, rows)
    gates = {p: (gate_sum[p] / gate_sum[p + '_n']).numpy().round(4).tolist() for p in ('base',) + DECOMP_PARTS}
    (d / 'decomp_gates.json').write_text(json.dumps({'layer': layer, 'mean_gate_image_tokens': gates}, indent=1))


# ----------------------------------------------------------------------------- pilot B: answer formats

def stage_format(rt, out: Path, args) -> None:
    d = out / 'format'
    d.mkdir(exist_ok=True)
    samples = tf.read_samples(Path(args.finetune_run) / 'test.jsonl')
    gen = ior._orig_generate_batch
    for style in ('list', 'key', 'bare'):
        path = d / f'{style}.jsonl'
        if path.exists():
            continue
        t0 = time.time()
        items = [io_.Item(s.id, 'coord', 'test', s.image, no.format_prompt(s.prompt, s.fmt, style)) for s in samples]
        res = ior.run_items(rt, gen, items, args.token_budget, 64)
        rows = []
        for s, it in zip(samples, items):
            text = res[s.id]['text']
            vals = no.parse_values(text, s.fmt, style)
            W, H = s.truth['canvas']
            if s.fmt == 'point':
                truth = [1000 * s.truth['center'][0] / W, 1000 * s.truth['center'][1] / H]
            else:
                b = s.truth['box']
                truth = [1000 * b[0] / W, 1000 * b[1] / H, 1000 * b[2] / W, 1000 * b[3] / H]
            row = {'id': s.id, 'family': s.family, 'fmt': s.fmt, 'style': style, 'prompt': it.prompt, 'text': text,
                   'finish_reason': res[s.id]['finish_reason'], 'batch_size': res[s.id]['batch_size'],
                   'values': vals, 'truth': truth, 'valid': vals is not None}
            if vals is not None:
                row['abs_err'] = [abs(v - t) for v, t in zip(vals, truth)]
                if s.fmt == 'bbox':
                    row['iou'] = tf.box_iou([v * (W if j % 2 == 0 else H) / 1000 for j, v in enumerate(vals)],
                                            s.truth['box'])
                else:
                    px = [vals[0] * W / 1000, vals[1] * H / 1000]
                    if 'diameter' in s.truth:
                        row['hit'] = bool(np.hypot(px[0] - s.truth['center'][0], px[1] - s.truth['center'][1])
                                          <= s.truth['diameter'] / 2)
                    else:
                        bx = s.truth['box']
                        row['hit'] = bool(bx[0] <= px[0] <= bx[2] and bx[1] <= px[1] <= bx[3])
            rows.append(row)
        append_jsonl(path, rows)
        print(f'[format] {style} valid={np.mean([r["valid"] for r in rows]):.3f} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--name', default='base')
    p.add_argument('--scope', default='')
    p.add_argument('--vit-run', required=True, help='7.3 run directory (training sets, probe, base evaluation)')
    p.add_argument('--premask', default=','.join(no.PREMASK_CONDITIONS))
    p.add_argument('--batch', type=int, default=12)
    # 7.3 recipe and evaluation arguments (forwarded to located_finetune_run stages)
    p.add_argument('--n-odd', type=int, default=1200)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--sets', default='probe,count,coord,general,babyvision')
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
    p.add_argument('--bv-batch', type=int, default=11)
    p.add_argument('--bv-max-new', type=int, default=1536)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(',')
    if stages == ['link']:
        stage_link(out, args)
        return
    install()
    if args.scope:
        # Rank is fixed by the registered scope; alpha = 4 r keeps the 7.3 LoRA scale (alpha / r = 4).
        args.rank = no.SCOPES[args.scope]['rank']
        args.alpha = 4.0 * args.rank
    torch.set_num_threads(16)
    rt = qr.load(args.model)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': stages, 'name': args.name,
                         'scope': args.scope, 'model': qr.describe(rt), 'gpu': torch.cuda.get_device_name(),
                         'args': vars(args)})
    meta.write_text(json.dumps(info, indent=1))
    for stage in stages:
        t0 = time.time()
        if stage == 'premask':
            stage_premask(rt, out, args)
        elif stage == 'train':
            lfr.stage_train(rt, out, args)
        elif stage == 'eval':
            lfr.stage_eval(rt, out, args)
        elif stage == 'attn':
            stage_attn(rt, out, args)
        elif stage == 'decomp':
            stage_decomp(rt, out, args)
        elif stage == 'format':
            stage_format(rt, out, args)
        else:
            raise ValueError(f'unknown stage {stage}')
        print(f'stage {stage} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
