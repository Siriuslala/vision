"""Experiment 3.14 round 2: answer-time reading heads of Qwen3.5-9B and Qwen3-VL-8B (GPU stage).

Results: docs/analysis/position/results_2026-11-03_causal_vs_linear_r2.md (section on attention patterns).

For each answer of the 3.11 cases already decoded by experiment 3.14 (gen_<tag>.jsonl in the round-1 run),
one record holds two measurements at the query that predicts each coordinate value's first digit:

* `maps`: the softmax attention of every head of every full-attention layer over the image tokens
  ((layers, heads, Q, Nv) float16, saved as one npz per answer);
* head-output patching: for the decoder-only splice pair of each axis (run B = run A with only the changed
  tokens D taking run-B merged embeddings, and DeepStack features for Qwen3-VL), the output of one head
  (the slice of the o_proj input) at the answer positions is copied from run B into run A; the logit difference
  of the first digit gives the recovered fraction r of that head. Conditions: every single head, every
  whole layer and all heads together.

Heads with large r are the "reading heads" of the model; their maps are analysed on CPU by
causal_vs_linear_heads_analysis. Layers are 1-based.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import causal_vs_linear as cv
from analysis_tools.position import causal_vs_linear_run as cvr
from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import coordinate_readout as cr

sr = cf.sr
read_jsonl, append_jsonl, done_ids = sr.read_jsonl, sr.append_jsonl, sr.done_ids


def head_projection(rt: qr.Runtime, layer: int) -> tuple[torch.nn.Module, int]:
    """Output projection of a full-attention layer and its per-head input width (both model families)."""
    if rt.family == 'qwen3_5':
        return cr.mixer_projection(rt, layer)
    att = rt.layers[layer - 1].self_attn
    return att.o_proj, att.head_dim


def image_share_heads(round1: Path, tag: str, conditions: list[str], top: int) -> list[tuple[int, int]]:
    """The `top` heads with the largest mean image-attention share at the first-digit queries in round 1.

    Uses the round-1 attention masses (kv_attn_<tag>/<id>.npz, mass[..., -1] = text share) of the listed
    conditions. A head that hardly attends to image tokens cannot read the target from them, so single-head
    patching is restricted to these heads when the model has too many heads to patch all (Qwen3-VL: 1152).
    """
    files = sorted(f for c in conditions for f in (round1 / f'kv_attn_{tag}').glob(f'*_{c.split("_", 1)[1]}.npz')
                   if f.name.startswith(c.split('_', 1)[0]))
    if not files:
        raise FileNotFoundError(f'No round-1 attention masses for {tag} {conditions}')
    share = np.stack([1.0 - np.load(f)['mass'][..., -1] for f in files]).mean((0, 3))   # (layers, heads)
    order = np.argsort(-share.ravel())[:top]
    return sorted((int(i // share.shape[1]) + 1, int(i % share.shape[1])) for i in order)


def head_conditions(rt: qr.Runtime, subset: list[tuple[int, int]] | None = None
                    ) -> list[tuple[str, list[tuple[int, int]]]]:
    """(name, [(layer, head), ...]) of every head-output patching row; 'none' is the unedited run A.

    subset: (layer index among full-attention layers, 1-based; head) pairs that get a single-head row;
    None = every head. Whole-layer rows and the all-heads row always cover every head.
    """
    H = cv.n_heads(rt)
    FL = cv.full_layers(rt)
    full = [(l, h) for l in FL for h in range(H)]
    singles = full if subset is None else [(FL[i - 1], h) for i, h in subset]
    conds = [('none', [])]
    conds += [(f'h{l}_{h}', [(l, h)]) for l, h in singles]
    conds += [(f'L{l}', [(l, h) for h in range(H)]) for l in cv.full_layers(rt)]
    conds += [('full_all', full)]
    return conds


class HeadEdit:
    """Copy head output slices (o_proj inputs) from batch row 0 (run B) into chosen rows at positions >= start.

    ops: list of (row, layer, head). Row 0 must not be edited. Works for Qwen3.5 and Qwen3-VL.
    """

    def __init__(self, rt: qr.Runtime, ops: list[tuple[int, int, int]], start: int):
        self.handles = []
        by_layer: dict[int, list] = {}
        for op in ops:
            if op[0] == 0:
                raise ValueError('Row 0 is the run-B source and cannot be edited.')
            by_layer.setdefault(op[1], []).append(op)
        for layer, lops in by_layer.items():
            module, hd = head_projection(rt, layer)

            def pre(module, args, lops=lops, hd=hd):
                x = args[0]
                y = x.clone()
                for row, _, h in lops:
                    sl = slice(h * hd, (h + 1) * hd)
                    y[row, start:, sl] = x[0, start:, sl]
                return (y,)
            self.handles.append(module.register_forward_pre_hook(pre))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def image_maps(rt, x, pos, queries: list[int], span: tuple[int, int], ds) -> np.ndarray:
    """(layers, heads, Q, Nv) float16 attention over the image tokens at the given queries."""
    s, e = span
    rows = cv.attention_rows(rt, x, pos, queries, ds=ds, span=span)
    return np.stack([rows[l][:, :, s:e].numpy() for l in cv.full_layers(rt)]).astype(np.float16)


@torch.inference_mode()
def stage_heads(rt, out: Path, args) -> None:
    tag = cvr.tag_of(args)
    path = out / f'heads_{tag}.jsonl'
    mdir = out / f'maps_{tag}'
    mdir.mkdir(exist_ok=True)
    cases = cf.load_cases(Path(args.flow_run))
    gen = read_jsonl(Path(args.round1) / f'gen_{tag}.jsonl')
    todo = cf.limit_per_condition(gen, args.conditions, args.n_answers)
    digits = cf.digit_ids(rt)
    use_ds = cvr.ds_on(rt, args)
    subset = image_share_heads(Path(args.round1), tag, args.conditions, args.subset_top) if args.subset_top else None
    conds = head_conditions(rt, subset)
    for g in todo:
        if g['id'] in done_ids(path) or not g['score']['valid']:
            continue
        c = cases[g['id']]
        t0 = time.time()
        prep = cv.prepare(rt, sr.load_image(c), c['prompt'])
        L = len(prep.ids)
        s, e = prep.span['start'], prep.span['end']
        ans = g['ids']
        target = cr.tokens_overlapping(c['truth']['box'], prep.grid)
        xa, dsa, pairs = cvr.pair_inputs(rt, c, prep, splice=True, use_ds=use_ds)
        pairs = [p for p in pairs if p['kind'] == 'splice']
        xa_tf, pos = cr.teacher_forced(rt, prep, ans[:-1], x=xa)
        queries = cf.first_digit_queries(g['values'], L)
        maps = image_maps(rt, xa_tf, pos, queries, (s, e), dsa)
        np.savez_compressed(mdir / f"{g['id']}.npz", maps=maps, target=np.array(target),
                            grid=np.array(prep.grid), keys=np.array([v['key'] for v in g['values']]))
        recs = []
        keep = slice(L - 1, None)
        for pr in pairs:
            xb_tf, _ = cr.teacher_forced(rt, prep, ans[:-1], x=pr['xb'])
            chunks = []
            for i in range(0, len(conds), args.head_rows):
                part = conds[i:i + args.head_rows]
                x = torch.cat([xb_tf] + [xa_tf] * len(part), 0)
                p = pos.expand(-1, x.shape[0], -1).contiguous()
                ds = None if dsa is None else [pr['dsb']] + [dsa] * len(part)
                ops = [(1 + j, l, h) for j, (_, heads) in enumerate(part) for l, h in heads]
                with ExitStack() as st:
                    st.enter_context(cv.deepstack_rows(rt, ds, (s, e)))
                    st.enter_context(HeadEdit(rt, ops, L - 1))
                    lg = rt.model(inputs_embeds=x, position_ids=p, use_cache=False, logits_to_keep=keep).logits
                chunks.append(cf.ld_table(lg, ans, g['values'], digits, b_row=0))
            recs.append({'tag': pr['tag'], 'kind': pr['kind'], 'axis': pr['axis'], 'target': pr['target'],
                         'moved': pr['moved'], 'values': cf._merge(chunks, [n for n, _ in conds], args.head_rows)})
        append_jsonl(path, [{'id': g['id'], 'cond': c['cond'], 'mode': args.mode, 'correct': g['score']['correct'],
                             'grid': list(prep.grid), 'span': [s, e], 'target': target, 'queries': queries,
                             'values_meta': g['values'], 'pairs': recs, 'seconds': round(time.time() - t0, 1)}])
        print(f"heads {tag} {g['id']} {len(pairs)} pairs {time.time() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--mode', required=True, help='ds_on (Qwen3-VL) or none (Qwen3.5)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--round1', type=Path, required=True, help='3.14 round-1 run directory (gen_<tag>.jsonl)')
    p.add_argument('--flow-run', type=Path, required=True, help='3.11 run directory (cases.json)')
    p.add_argument('--conditions', nargs='+', default=['t6g10_rowcol', 't6g10_point'])
    p.add_argument('--n-answers', type=int, default=16)
    p.add_argument('--head-rows', type=int, default=32)
    p.add_argument('--subset-top', type=int, default=0,
                   help='single-head rows only for the N heads with the largest round-1 image-attention share (0 = all)')
    p.add_argument('--max-memory', default='')
    p.add_argument('--torch-deltanet', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.torch_deltanet:
        cvr.use_torch_deltanet()
    rt = cf.load_runtime(args.model, args.max_memory)
    if (rt.family, args.mode) not in (('qwen3_vl', 'ds_on'), ('qwen3_5', 'none')):
        raise ValueError('Supported: Qwen3-VL with --mode ds_on, Qwen3.5 with --mode none.')
    append_jsonl(args.output / 'run_log.jsonl', [{
        'stage': 'heads', 'tag': cvr.tag_of(args), 'args': {k: str(v) for k, v in vars(args).items()},
        'runtime': qr.describe(rt), 'device': torch.cuda.get_device_name(0),
        'started': datetime.now(timezone.utc).isoformat()}])
    stage_heads(rt, args.output, args)


if __name__ == '__main__':
    main()
