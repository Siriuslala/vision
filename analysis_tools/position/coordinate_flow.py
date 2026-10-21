"""Experiment 3.11: information flow of coordinate readout in Qwen3.5-9B (GPU stages).

Design: docs/analysis/position/design_coordinate_flow.md; results: results_2026-10-20_coordinate_flow.md.

Every causal measurement is a paired, teacher-forced comparison. Run A is the original input, run B
moves the target (synthetic images: target redrawn k tokens away; RefCOCO: the target's merged
embeddings swapped with an equally sized region, the 3.5 content swap). Both runs are teacher-forced on
run A's own greedy answer, so they share every text token. For each coordinate value k of the answer,
at the position whose next-token prediction is the first digit of k,

    LD = z[d_B] - z[d_A],

with d_A run A's first digit and d_B run B's argmax digit at the same position (pairs with d_A == d_B
are excluded for that value). An intervention on run A gives LD_e and the recovered fraction
r = (LD_e - LD_A) / (LD_B - LD_A); r = 1 means the edited component carries all of the difference.

Stages (--stages, run in order inside one process; all resumable by record id):
  cases     CPU: case list from the 2.5e run (disc, T6 10x10, RefCOCO) times answer formats that change
            the emission order ([x, y] list, {"x","y"}, {"y","x"}, bbox list, {"y_min",...}, row/column,
            column/row).
  gen       greedy answers of run A (manual equal-length batches, experiment 3.5 decoder).
  spotcheck answer-level agreement of the manual decoder with qwen_runtime.generate.
  heads     answer-position output patching of every single full-attention head (128), every layer,
            leave-one-out sets and the full set (Q1).
  kv        image-KV subset patching from run B into run A's prompt cache (token groups, token rows,
            token columns, rings, layer x group), plus attention mass on token groups at every answer
            position (Q3, Q4).
  flow      on the 2.5e answers: residual states at all 33 sites for key positions and the image share of
            attention for every post-image query (Q5).
  field     on the 2.5e images: ridge decoders from each visual token's state to its own position at
            sites 0/4/8/12/16/20, image-parity two-fold, and the decoded position field of every image (Q6).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position.coordinate_readout_supp import refcoco_swaps
from analysis_tools.representation import sees_right as sr

FULL_LAYERS = cr.FULL_LAYERS
COORD_NOTE = cr.COORD_NOTE
JSON_ONLY = cr.JSON_ONLY
T6_QUERY = sr.T6_QUERY
REFCOCO_QUERY = sr.REFCOCO_QUERY
ROWCOL_TAIL = ', counting rows from top to bottom and columns from left to right, starting at 1.'
FORMAT_SUFFIX = {
    'point': ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].' + COORD_NOTE + JSON_ONLY,
    'point_xyk': ' Give the center of its bounding box. Return only {"x": x, "y": y} (x first, then y), not a list.'
                 + COORD_NOTE + JSON_ONLY,
    'point_yxk': ' Give the center of its bounding box. Return only {"y": y, "x": x} (y first, then x), not a list.'
                 + COORD_NOTE + JSON_ONLY,
    'bbox': ' Give its tight bounding box. Return only [{"bbox_2d": [x_min, y_min, x_max, y_max]}].'
            + COORD_NOTE + JSON_ONLY,
    'bbox_yxk': ' Give its tight bounding box as a JSON object with the y coordinates first: {"y_min": y_min, '
                '"x_min": x_min, "y_max": y_max, "x_max": x_max}.' + COORD_NOTE + JSON_ONLY,
    'rowcol': ' Give its position. Return only {"row": r, "column": c}' + ROWCOL_TAIL + JSON_ONLY,
    'colrow': ' Give its position. Return only {"column": c, "row": r}' + ROWCOL_TAIL + JSON_ONLY,
}
# Keys of each format in requested order; list formats name their positional slots.
FORMAT_KEYS = {
    'point': ['x', 'y'], 'point_xyk': ['x', 'y'], 'point_yxk': ['y', 'x'],
    'bbox': ['x_min', 'y_min', 'x_max', 'y_max'], 'bbox_yxk': ['y_min', 'x_min', 'y_max', 'x_max'],
    'rowcol': ['row', 'column'], 'colrow': ['column', 'row'],
}
LIST_FORMATS = {'point': 'point', 'bbox': 'bbox'}
AXIS = {'x': 'h', 'x_min': 'h', 'x_max': 'h', 'column': 'h', 'y': 'v', 'y_min': 'v', 'y_max': 'v', 'row': 'v'}
SET_FORMATS = {'disc': ['point', 'point_xyk', 'point_yxk', 'bbox', 'bbox_yxk'],
               't6g10': ['rowcol', 'colrow', 'point', 'point_yxk'],
               'rc': ['point', 'point_yxk']}
FIELD_SITES = (0, 4, 8, 12, 16, 20)
FIELD_LAMBDAS = (1e1, 1e2, 1e3, 1e4)
read_jsonl, append_jsonl, done_ids = sr.read_jsonl, sr.append_jsonl, sr.done_ids


# ----------------------------------------------------------------------------- answers

def value_spans(text: str, fmt: str) -> list[tuple[str, tuple[int, int]]] | None:
    """(key, character span of the digits) of every coordinate value, in emission order.

    List formats take the first point_2d / bbox_2d list; key formats take each requested key once and
    order the values by where the model wrote them (the model may ignore the requested order).
    """
    keys = FORMAT_KEYS[fmt]
    if fmt in LIST_FORMATS:
        m = re.search(r'"(?:point_2d|bbox_2d)":\s*\[([^\]]*)\]', text)
        if not m:
            return None
        nums = list(re.finditer(r'\d+', m.group(1)))
        if len(nums) != len(keys):
            return None
        return [(k, (m.start(1) + n.start(), m.start(1) + n.end())) for k, n in zip(keys, nums)]
    out = []
    for k in keys:
        m = re.search(rf'"{k}":\s*(\d+)', text)
        if not m:
            return None
        out.append((k, m.span(1)))
    return sorted(out, key=lambda kv: kv[1][0])


def answer_values(rt, ids: list[int], fmt: str) -> list[dict] | None:
    """Coordinate values with their generated-token indices: [{'key', 'axis', 'value', 'toks'}]."""
    text = rt.tokenizer.decode(ids, skip_special_tokens=False)
    spans = value_spans(text, fmt)
    if spans is None:
        return None
    ends = [len(rt.tokenizer.decode(ids[:i + 1], skip_special_tokens=False)) for i in range(len(ids))]
    starts = [0] + ends[:-1]
    out = []
    for key, (a, b) in spans:
        toks = [i for i in range(len(ids)) if starts[i] >= a and ends[i] <= b]
        if not toks or ''.join(rt.tokenizer.decode([ids[i]]) for i in toks) != text[a:b]:
            return None
        out.append({'key': key, 'axis': AXIS[key], 'value': int(text[a:b]), 'toks': toks})
    return out


def score_values(case: dict, values: list[dict] | None) -> dict:
    """Correctness of run A: disc point inside the disc / bbox IoU >= 0.5; grid cell exact; RefCOCO point in box."""
    if values is None:
        return {'valid': False, 'correct': False}
    v = {d['key']: d['value'] for d in values}
    t = case['truth']
    W, H = t['canvas']
    if case['format'] in ('rowcol', 'colrow'):
        pred = (v['row'] - 1, v['column'] - 1)
        tr, tc = t['target_rc']
        return {'valid': True, 'correct': pred == (tr, tc), 'd_row': pred[0] - tr, 'd_col': pred[1] - tc}
    if 'x' in v:
        c = np.array([v['x'], v['y']], float)
        box = None
    else:
        box = [v['x_min'], v['y_min'], v['x_max'], v['y_max']]
        c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], float)
    px = c * np.array([W, H]) / 1000
    out = {'valid': True, 'out_units': c.tolist()}
    if case['set'] == 'disc':
        if box is None:
            out['correct'] = bool(np.linalg.norm(px - np.array(t['center'])) <= t['diameter'] / 2)
        else:
            pb = [box[0] * W / 1000, box[1] * H / 1000, box[2] * W / 1000, box[3] * H / 1000]
            out['iou'] = cr.box_iou(pb, t['box'])
            out['correct'] = out['iou'] >= 0.5
    elif case['set'] == 't6':
        pred = sr.point_cell(t, c)
        tr, tc = t['target_rc']
        out.update({'correct': pred == (tr, tc), 'd_row': pred[0] - tr, 'd_col': pred[1] - tc})
    else:
        b = t['box']
        out['correct'] = bool(b[0] <= px[0] <= b[2] and b[1] <= px[1] <= b[3])
    return out


# ----------------------------------------------------------------------------- token groups

def token_groups(grid: tuple[int, int], target: list[int], moved: list[int]) -> dict[str, list[int]]:
    """Image-token groups (raster indices) relative to the changed tokens D = T_A U T_B.

    With a true move (run B redrawn) every merged embedding can differ because the ViT attends globally.
    With a splice or content swap (only D's embeddings differ) tokens strictly before the first token of D
    in raster order have identical keys and values in both runs (causal decoder), so 'above' and the part
    of 'row_left' in D's first row are exact zero controls. Rings use the Chebyshev token distance to D.
    """
    H, W = grid
    D = sorted(set(target) | set(moved))
    rr, cc = np.divmod(np.arange(H * W), W)
    dr = np.array([t // W for t in D])
    dc = np.array([t % W for t in D])
    r0, r1, c0, c1 = dr.min(), dr.max(), dc.min(), dc.max()
    inD = np.zeros(H * W, bool)
    inD[D] = True
    cheb = np.min(np.maximum(np.abs(rr[:, None] - dr[None]), np.abs(cc[:, None] - dc[None])), 1)
    rows_in = (rr >= r0) & (rr <= r1)
    cols_in = (cc >= c0) & (cc <= c1)
    below = rr > r1
    g = {
        'obj': inD,
        'inbox': rows_in & cols_in & ~inD,
        'row_left': rows_in & (cc < c0),
        'row_right': rows_in & (cc > c1),
        'below_cols': below & cols_in,
        'below_left': below & (cc < c0),
        'below_right': below & (cc > c1),
        'above': rr < r0,
        'ring1': cheb == 1, 'ring2': cheb == 2, 'ring3_4': (cheb >= 3) & (cheb <= 4), 'ring5p': cheb >= 5,
        'all_but_obj': ~inD,
        'row_same': rows_in & ~inD,
        'below': below,
    }
    return {k: np.flatnonzero(v).tolist() for k, v in g.items()}


GROUPS_ALL_LAYERS = ('obj', 'inbox', 'row_left', 'row_right', 'below_cols', 'below_left', 'below_right', 'above',
                     'ring1', 'ring2', 'ring3_4', 'ring5p', 'all_but_obj', 'row_same', 'below')
GROUPS_BY_LAYER = ('obj', 'row_same', 'below', 'all_but_obj', 'all_img')


# ----------------------------------------------------------------------------- cases and pairs

def stage_cases(out: Path, args) -> None:
    src = {c['id']: c for c in json.loads((Path(args.source) / 'cases.json').read_text())}
    picks = {'disc': [f'disc_point_{i:03d}' for i in range(args.n_disc)],
             't6g10': [f't6g10_rowcol_{i:03d}' for i in range(args.n_grid)]}
    # RefCOCO: referents whose content-swap region fits on both axes, smallest first; at least one axis.
    rc = []
    for c in src.values():
        if c['cond'] != 'rc_point':
            continue
        W, H = c['truth']['canvas']
        grid = (H // cr.TOKEN_PIXELS, W // cr.TOKEN_PIXELS)
        target = cr.tokens_overlapping(c['truth']['box'], grid)
        n_ok = sum(r['valid'] for r in refcoco_swaps(target, grid))
        if n_ok:
            rc.append((-n_ok, len(target), c['id']))
    picks['rc'] = [cid for _, _, cid in sorted(rc)[:args.n_rc]]
    cases = []
    for set_name, ids in picks.items():
        for cid in ids:
            c = src[cid]
            if set_name == 'disc':
                query = cr.DISC_QUERY
            elif set_name == 't6g10':
                query = T6_QUERY
            else:
                query = REFCOCO_QUERY.format(expr=c['expression'])
            for fmt in SET_FORMATS[set_name]:
                cases.append({**{k: v for k, v in c.items() if k not in ('id', 'cond', 'prompt', 'format', 'style')},
                              'id': f"{c['image_key']}_{fmt}", 'source_id': cid, 'group': set_name,
                              'cond': f'{set_name}_{fmt}', 'format': fmt, 'prompt': query + FORMAT_SUFFIX[fmt]})
    (out / 'cases.json').write_text(json.dumps(cases, indent=1))
    counts = {}
    for c in cases:
        counts[c['cond']] = counts.get(c['cond'], 0) + 1
    print(json.dumps(counts, indent=1))


def load_cases(out: Path) -> dict:
    return {c['id']: c for c in json.loads((out / 'cases.json').read_text())}


def stage_gen(rt, out: Path, args) -> None:
    path = out / 'gen.jsonl'
    finished = done_ids(path)
    todo = [c for c in load_cases(out).values() if c['id'] not in finished]
    for i in range(0, len(todo), args.prep_chunk):
        chunk = todo[i:i + args.prep_chunk]
        preps = [cr.prepare(rt, sr.load_image(c), c['prompt']) for c in chunk]
        by_len: dict[int, list] = {}
        for c, p in zip(chunk, preps):
            by_len.setdefault(len(p.ids), []).append((c, p))
        for L, items in by_len.items():
            for j in range(0, len(items), args.batch_size):
                part = items[j:j + args.batch_size]
                res = cr.run_conditions(rt, [cr.embeds(rt, p) for _, p in part], [p.pos for _, p in part],
                                        args.max_new_tokens, batch_size=len(part))
                rows = []
                for (c, _), r in zip(part, res):
                    vals = answer_values(rt, r['ids'], c['format'])
                    rows.append({'id': c['id'], 'cond': c['cond'], 'text': r['text'], 'ids': r['ids'],
                                 'prompt_len': L, 'batch_size': len(part), 'values': vals,
                                 'score': score_values(c, vals)})
                append_jsonl(path, rows)
        print(f'gen {i + len(chunk)}/{len(todo)}', flush=True)
        del preps
        torch.cuda.empty_cache()


def stage_spotcheck(rt, out: Path, args) -> None:
    path = out / 'spotcheck.jsonl'
    finished = done_ids(path)
    cases = load_cases(out)
    gen = read_jsonl(out / 'gen.jsonl')
    rng = np.random.default_rng(args.seed)
    for k in sorted(rng.choice(len(gen), min(args.spotcheck, len(gen)), replace=False)):
        g = gen[k]
        if g['id'] in finished:
            continue
        c = cases[g['id']]
        ref = qr.generate(rt, qr.build(rt, [sr.load_image(c)], c['prompt']), args.max_new_tokens)
        ref_ids = rt.tokenizer.encode(ref['text'], add_special_tokens=False)
        vr = answer_values(rt, ref_ids, c['format'])
        val = lambda v: None if v is None else [(d['key'], d['value']) for d in v]
        append_jsonl(path, [{'id': g['id'], 'reference': ref['text'], 'manual': g['text'],
                             'text_equal': ref['text'].strip() == g['text'].strip(),
                             'answer_equal': val(vr) == val(g['values'])}])


def pair_inputs(rt, case: dict, prep, splice: bool = False):
    """Run-B inputs for every feasible displacement: [{'tag', 'axis', 'xb', 'target', 'moved', 'd'}]."""
    xa = cr.embeds(rt, prep)
    target = cr.tokens_overlapping(case['truth']['box'], prep.grid)
    out = []
    if case['set'] in ('disc', 't6'):
        for mv in sr.moves(case, prep.grid, target, 2, 4):
            img = sr.run_b_image(case, mv)
            pb = cr.prepare(rt, img, case['prompt'])
            if len(pb.ids) != len(prep.ids):
                raise ValueError(f"Run B length differs for {case['id']}")
            xb = cr.embeds(rt, pb)
            tag = f"{mv['axis']}{mv['sign']:+d}"
            out.append({'tag': tag, 'kind': 'move', 'axis': mv['axis'], 'xb': xb, 'target': target,
                        'moved': mv['shifted']})
            if splice:
                # Decoder-only variant: run A's merged embeddings everywhere except the changed tokens D, which
                # take run B's embeddings. ViT-propagated context differences are removed, so any difference
                # at a token outside D can only arise inside the causal decoder (hence only after D).
                D = torch.tensor(sorted(set(target) | set(mv['shifted']))) + prep.span['start']
                xs = xa.clone()
                xs[0, D] = xb[0, D]
                out.append({'tag': tag + '_splice', 'kind': 'splice', 'axis': mv['axis'], 'xb': xs, 'target': target,
                            'moved': mv['shifted']})
    else:
        for reg in refcoco_swaps(target, prep.grid):
            if not reg['valid']:
                continue
            order = cr.swap_index(prep.grid[0] * prep.grid[1], target, reg['region'])
            out.append({'tag': f"{reg['axis']}swap", 'kind': 'swap', 'axis': reg['axis'], 'xb': cr.embeds(rt, prep, order=order),
                        'target': target, 'moved': reg['region']})
    return xa, out


def digit_ids(rt) -> list[int]:
    return [rt.tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]


def limit_per_condition(gen: list[dict], conds: list[str], n: int) -> list[dict]:
    """The first n answers (case order) of each listed condition, interleaved by image index."""
    per = {c: [g for g in gen if g['cond'] == c][:n] for c in conds}
    out = []
    for i in range(n):
        out += [per[c][i] for c in conds if i < len(per[c])]
    return out


def first_digit_queries(values: list[dict], L: int) -> list[int]:
    """Absolute query position (in prompt + answer[:-1]) predicting each value's first digit."""
    return [L - 1 + v['toks'][0] for v in values]


def ld_table(logits: torch.Tensor, ans: list[int], values: list[dict], digits: list[int], b_row: int = 0) -> dict:
    """Per value: d_A, d_B and LD for every row. logits: (R, n_tail, V) over tail positions L-1 .. ; row b_row = run B."""
    out = []
    dig = torch.tensor(digits, device=logits.device)
    for k, v in enumerate(values):
        i = v['toks'][0]
        dA = ans[i]
        zb = logits[b_row, i, dig]
        dB = digits[int(zb.argmax())]
        ld = (logits[:, i, dB] - logits[:, i, dA]).float().cpu().tolist()
        out.append({'k': k, 'key': v['key'], 'axis': v['axis'], 'dA': rt_digit(dA, digits), 'dB': rt_digit(dB, digits),
                    'ld': ld})
    return out


def rt_digit(tok: int, digits: list[int]) -> int | None:
    return digits.index(tok) if tok in digits else None


# ----------------------------------------------------------------------------- Q1: head patching

def head_conditions() -> list[tuple[str, list[tuple[int, int]]]]:
    full = [(l, h) for l in FULL_LAYERS for h in range(16)]
    conds = [('none', [])]
    conds += [(f'h{l}_{h}', [(l, h)]) for l, h in full]
    conds += [(f'L{l}', [(l, h) for h in range(16)]) for l in FULL_LAYERS]
    conds += [('full_all', full)]
    conds += [(f'loo{l}_{h}', [x for x in full if x != (l, h)]) for l in (16, 20, 24) for h in range(16)]
    conds += [('full_minus_L20', [x for x in full if x[0] != 20])]
    return conds


@torch.inference_mode()
def stage_heads(rt, out: Path, args) -> None:
    path = out / 'heads.jsonl'
    finished = done_ids(path)
    cases = load_cases(out)
    digits = digit_ids(rt)
    conds = head_conditions()
    for g in limit_per_condition(read_jsonl(out / 'gen.jsonl'), args.head_conditions, args.n_heads):
        c = cases[g['id']]
        if g['id'] in finished or not g['score']['valid']:
            continue
        t0 = time.time()
        prep = cr.prepare(rt, sr.load_image(c), c['prompt'])
        L = len(prep.ids)
        ans = g['ids']
        xa, pairs = pair_inputs(rt, c, prep)
        recs = []
        for pr in pairs:
            xa_tf, pos = cr.teacher_forced(rt, prep, ans[:-1], x=xa)
            xb_tf, _ = cr.teacher_forced(rt, prep, ans[:-1], x=pr['xb'])
            keep = slice(L - 1, None)          # a slice is not moved between devices by accelerate
            lds = []
            for i in range(0, len(conds), args.head_rows):
                part = conds[i:i + args.head_rows]
                x = torch.cat([xb_tf] + [xa_tf] * len(part), 0)
                p = pos.expand(-1, x.shape[0], -1).contiguous()
                ops = [(1 + j, l, h, 0) for j, (_, heads) in enumerate(part) for l, h in heads]
                with cr.MixerEdit(rt, ops, L - 1):
                    lg = rt.model(inputs_embeds=x, position_ids=p, use_cache=False, logits_to_keep=keep).logits
                lds.append(_lds(lg, ans, g['values'], digits))
            table = _merge(lds, [n for n, _ in conds], args.head_rows)
            recs.append({'tag': pr['tag'], 'kind': pr['kind'], 'axis': pr['axis'], 'values': table})
        append_jsonl(path, [{'id': g['id'], 'cond': c['cond'], 'correct': g['score']['correct'], 'pairs': recs,
                             'seconds': round(time.time() - t0, 1)}])
        print(f"heads {g['id']} {len(pairs)} pairs {time.time() - t0:.1f}s", flush=True)


def _lds(lg: torch.Tensor, ans: list[int], values: list[dict], digits: list[int]) -> list[dict]:
    return ld_table(lg, ans, values, digits, b_row=0)


def _merge(chunks: list[list[dict]], names: list[str], rows: int) -> list[dict]:
    """Join chunked LD tables (each chunk: row 0 = run B, rows 1.. = conditions) into name -> LD per value."""
    out = []
    for k in range(len(chunks[0])):
        v = {kk: chunks[0][k][kk] for kk in ('key', 'axis', 'dA', 'dB')}
        v['LD_B'] = chunks[0][k]['ld'][0]
        lds = {}
        for ci, ch in enumerate(chunks):
            for j, ld in enumerate(ch[k]['ld'][1:]):
                lds[names[ci * rows + j]] = round(ld, 4)
        v['LD'] = lds
        out.append(v)
    return out


# ----------------------------------------------------------------------------- Q3/Q4: KV subset patching

def kv_conditions(grid, target, moved, s: int, e: int, L: int, full: bool = True) -> list[tuple[str, list[int], tuple[int, ...]]]:
    """(name, absolute positions, full-attention layers) copied from run B into run A's prompt cache.

    full=False keeps the reduced set used for the emission-order variants: the main groups and the
    layer-resolved target / non-target / whole-image edits.
    """
    H, W = grid
    g = token_groups(grid, target, moved)
    img = list(range(H * W))
    ab = lambda idx: [s + i for i in idx]
    conds = [('none', [], ()), ('all_img', ab(img), FULL_LAYERS), ('text', list(range(e, L - 1)), FULL_LAYERS)]
    allg = {**g, 'all_img': img}
    if not full:
        conds += [(name, ab(g[name]), FULL_LAYERS) for name in ('obj', 'all_but_obj', 'row_same', 'below')]
        conds += [(f'{name}@{l}', ab(allg[name]), (l,)) for name in ('obj', 'all_but_obj', 'all_img') for l in FULL_LAYERS]
        return [c for c in conds if c[0] == 'none' or c[1]]
    conds += [(name, ab(g[name]), FULL_LAYERS) for name in GROUPS_ALL_LAYERS]
    conds += [(f'row{r}', ab([r * W + c for c in range(W)]), FULL_LAYERS) for r in range(H)]
    conds += [(f'col{c}', ab([r * W + c for r in range(H)]), FULL_LAYERS) for c in range(W)]
    conds += [(f'{name}@{l}', ab(allg[name]), (l,)) for name in GROUPS_BY_LAYER for l in FULL_LAYERS]
    return [c for c in conds if c[0] == 'none' or c[1]]


@torch.inference_mode()
def tf_kv_logits(rt, xb: torch.Tensor, xa: torch.Tensor, pos: torch.Tensor, L: int,
                 edits: list[tuple[list[int], tuple[int, ...]]]) -> torch.Tensor:
    """Teacher-forced logits over the tail (positions L-1 ..) with run-B keys/values copied into run A's cache.

    Rows: 0 = run B, 1.. = run A with edits[i-1] = (absolute prompt positions, full-attention layers).
    The prompt except its last token is prefilled; the cache of rows >= 1 is edited; the tail (last prompt
    token and the teacher-forced answer) is then processed in one cached forward.
    """
    R = 1 + len(edits)
    x = torch.cat([xb] + [xa] * len(edits), 0)
    p = pos.expand(-1, R, -1).contiguous()
    o = rt.model(inputs_embeds=x[:, :L - 1], position_ids=p[:, :, :L - 1], use_cache=True, logits_to_keep=1)
    cache = o.past_key_values
    idx = sr.kv_layers(rt)
    for b, (positions, layers) in enumerate(edits, start=1):
        if not positions:
            continue
        for l in layers:
            keys, values = sr.cache_kv(cache, idx[l])
            ps = torch.tensor(positions, device=keys.device)
            keys[b, :, ps] = keys[0, :, ps]
            values[b, :, ps] = values[0, :, ps]
    o = rt.model(inputs_embeds=x[:, L - 1:], position_ids=p[:, :, L - 1:], past_key_values=cache, use_cache=True)
    return o.logits


@torch.inference_mode()
def attention_rows(rt, x: torch.Tensor, pos: torch.Tensor, queries: list[int], layers=FULL_LAYERS) -> cr.Readout:
    """Softmax rows of full-attention layers at the given queries (device-safe version of cr.attention_readout).

    Same recomputation as cr.attention_readout (pre-hooks capture the normalized input and rotary tables of
    each self_attn module; q and k are recomputed) but every index tensor follows the device of the layer,
    so the model may be split over several GPUs. contrib is not computed. check is the relative max error
    of the recomputed gated output against the module output at the queries.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    store, handles = {}, []
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn
        def pre(module, args, kwargs, layer=layer):
            store.setdefault(layer, {})['h'] = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            store[layer]['pe'] = kwargs['position_embeddings']
        def post(module, args, kwargs, output, layer=layer):
            store[layer]['out'] = output[0]
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(mod.register_forward_hook(post, with_kwargs=True))
    try:
        rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    attn, check = {}, {}
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn
        h, (cos, sin), out = store[layer]['h'], store[layer]['pe'], store[layer]['out']
        dev = h.device
        q_idx = torch.tensor(queries, device=dev)
        cos, sin = cos.to(dev), sin.to(dev)
        L, hd = h.shape[1], mod.head_dim
        q, gate = torch.chunk(mod.q_proj(h[:, q_idx]).view(1, len(queries), -1, hd * 2), 2, dim=-1)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(1, L, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(1, L, -1, hd).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, q_idx], sin[:, q_idx])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        k, v = repeat_kv(k, mod.num_key_value_groups), repeat_kv(v, mod.num_key_value_groups)
        sc = (q.float() @ k.float().transpose(2, 3)) * mod.scaling
        sc = sc.masked_fill(torch.arange(L, device=dev)[None, :] > q_idx[:, None], float('-inf'))
        a = sc.softmax(-1)
        o = ((a @ v.float()) * torch.sigmoid(gate.float()).transpose(1, 2)).transpose(1, 2).reshape(1, len(queries), -1)
        recon = mod.o_proj(o.to(h.dtype))
        ref = out[0, q_idx.to(out.device)].to(dev)
        check[layer] = float((recon[0] - ref).abs().max() / ref.abs().max())
        attn[layer] = a[0].cpu()
    return cr.Readout(attn, {}, check)


def attention_groups(grid, target: list[int], s: int, e: int, Lfull: int) -> dict[str, torch.Tensor]:
    """Masks over the full sequence for attention bookkeeping, defined by run A's target only."""
    H, W = grid
    rows = {t // W for t in target}
    cols = {t % W for t in target}
    m = {k: torch.zeros(Lfull, dtype=torch.bool) for k in ('target', 'target_rows', 'target_cols', 'other_img', 'text')}
    for j in range(H * W):
        r, c = divmod(j, W)
        if j in target:
            m['target'][s + j] = True
        elif r in rows:
            m['target_rows'][s + j] = True
        elif c in cols:
            m['target_cols'][s + j] = True
        else:
            m['other_img'][s + j] = True
    m['text'][:s] = True
    m['text'][e:] = True
    return m


@torch.inference_mode()
def stage_kv(rt, out: Path, args) -> None:
    path = out / 'kv.jsonl'
    finished = done_ids(path)
    cases = load_cases(out)
    digits = digit_ids(rt)
    (out / 'kv_attn').mkdir(exist_ok=True)
    gen = read_jsonl(out / 'gen.jsonl')
    todo = limit_per_condition(gen, args.kv_full_conditions, args.n_kv_full) + \
        limit_per_condition(gen, [c for c in args.kv_conditions if c not in args.kv_full_conditions], args.n_kv)
    if args.reverse:
        todo = todo[::-1]
    for g in todo:
        c = cases[g['id']]
        finished = done_ids(path)              # re-read: a second process may share the work list
        if g['id'] in finished or not g['score']['valid']:
            continue
        t0 = time.time()
        prep = cr.prepare(rt, sr.load_image(c), c['prompt'])
        L = len(prep.ids)
        s, e = prep.span['start'], prep.span['end']
        ans = g['ids']
        full = c['cond'] in args.kv_full_conditions
        xa, pairs = pair_inputs(rt, c, prep, splice=full)
        xa_tf, pos = cr.teacher_forced(rt, prep, ans[:-1], x=xa)
        # Attention mass on token groups at every tail query (run A only).
        tail = list(range(L - 1, xa_tf.shape[1]))
        ro = attention_rows(rt, xa_tf, pos, tail)
        target = cr.tokens_overlapping(c['truth']['box'], prep.grid)
        masks = attention_groups(prep.grid, target, s, e, xa_tf.shape[1])
        mass = np.stack([np.stack([ro.attn[l][:, :, m].sum(-1).numpy() for m in masks.values()], -1)
                         for l in FULL_LAYERS])                                      # (8, 16, Q, 5)
        l20 = ro.attn[20][:, [v['toks'][0] for v in g['values']], s:e].numpy()   # (16, K, Nv)
        np.savez_compressed(out / 'kv_attn' / f"{g['id']}.npz", mass=mass.astype(np.float32),
                            l20=l20.astype(np.float16), groups=np.array(list(masks)))
        recs = []
        for pr in pairs:
            xb_tf, _ = cr.teacher_forced(rt, prep, ans[:-1], x=pr['xb'])
            conds = kv_conditions(prep.grid, pr['target'], pr['moved'], s, e, L, full=full)
            chunks = []
            for i in range(0, len(conds), args.kv_rows):
                part = conds[i:i + args.kv_rows]
                lg = tf_kv_logits(rt, xb_tf, xa_tf, pos, L, [(pp, ll) for _, pp, ll in part])
                chunks.append(_lds(lg, ans, g['values'], digits))
            recs.append({'tag': pr['tag'], 'kind': pr['kind'], 'axis': pr['axis'], 'target': pr['target'], 'moved': pr['moved'],
                         'values': _merge(chunks, [n for n, _, _ in conds], args.kv_rows)})
        append_jsonl(path, [{'id': g['id'], 'cond': c['cond'], 'correct': g['score']['correct'], 'grid': list(prep.grid),
                             'span': [s, e], 'prompt_len': L, 'tail_tokens': [rt.tokenizer.decode([t]) for t in ans],
                             'values_meta': g['values'], 'pairs': recs, 'attn_check': ro.check,
                             'seconds': round(time.time() - t0, 1)}])
        print(f"kv {g['id']} {len(pairs)} pairs {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- Q5: whole-process states

@torch.inference_mode()
def stage_flow(rt, out: Path, args) -> None:
    """Residuals at 33 sites for key positions and image attention share for post-image queries (2.5e answers)."""
    src = Path(args.source)
    cases = {c['id']: c for c in json.loads((src / 'cases.json').read_text())}
    root = out / 'flow'
    root.mkdir(exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    rng = np.random.default_rng(args.seed)
    gen = [g for g in read_jsonl(src / 'gen.jsonl') if cases[g['id']]['cond'] in args.flow_conditions
           and g['score'].get('valid')]
    for g in gen:
        if g['id'] in finished:
            continue
        c = cases[g['id']]
        t0 = time.time()
        prep = cr.prepare(rt, sr.load_image(c), c['prompt'])
        coord = cr.coordinate_tokens(rt, g['ids'], c['format'])
        if len(coord) != 2:
            append_jsonl(path, [{'id': g['id'], 'status': 'no_coordinate_tokens'}])
            continue
        L = len(prep.ids)
        s, e = prep.span['start'], prep.span['end']
        x, pos = cr.teacher_forced(rt, prep, g['ids'][:-1])
        target = cr.tokens_overlapping(c['truth']['box'], prep.grid)
        H, W = prep.grid
        tr = sorted({t // W for t in target})
        # One context token: the first visual token after the target's last row-run (same row, right side)
        ctx = [t for t in range(max(target) + 1, H * W)][:1]
        rand_img = sorted(rng.choice(H * W, 4, replace=False).tolist())
        named = {'target_mean': [s + t for t in target], 'img_mean': list(range(s, e)),
                 'img_last': [e - 1], 'after_target': [s + t for t in ctx] or [e - 1],
                 'text_mean': list(range(e, L - 1)), 'prompt_last': [L - 1],
                 'q_x': [L - 1 + coord[0][0]], 'q_y': [L - 1 + coord[1][0]]}
        store = {}
        handles = []
        def mk(layer):
            def hook(module, inputs, output):
                h = (output[0] if isinstance(output, tuple) else output)[0].float()
                store[layer] = torch.stack([h[idx].mean(0) for idx in named.values()]).cpu()
            return hook
        for layer in range(1, len(rt.layers) + 1):
            handles.append(rt.layers[layer - 1].register_forward_hook(mk(layer)))
        try:
            queries = list(range(e, x.shape[1]))
            ro = attention_rows(rt, x, pos, queries)
        finally:
            for h in handles:
                h.remove()
        x0 = x[0].float()
        store[0] = torch.stack([x0[idx].mean(0) for idx in named.values()]).cpu()
        resid = torch.stack([store[l] for l in range(len(rt.layers) + 1)]).numpy()      # (33, P, D)
        img_share = np.stack([ro.attn[l][:, :, s:e].sum(-1).numpy() for l in FULL_LAYERS])  # (8, 16, Q)
        np.savez_compressed(root / f"{g['id']}.npz", resid=resid.astype(np.float16), img_share=img_share.astype(np.float16))
        append_jsonl(path, [{'id': g['id'], 'cond': c['cond'], 'status': 'ok', 'positions': list(named), 'grid': [H, W],
                             'span': [s, e], 'prompt_len': L, 'query_start': e, 'q_x': named['q_x'][0], 'q_y': named['q_y'][0],
                             'target_rows': tr, 'correct': g['score']['correct'], 'true_units': g['score']['true_units'],
                             'out_units': g['score'].get('out_units'), 'check': ro.check,
                             'seconds': round(time.time() - t0, 2)}])
        print(f"flow {g['id']} {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- Q6: position field

@torch.inference_mode()
def visual_states(rt, prep, sites=FIELD_SITES) -> dict[int, torch.Tensor]:
    """States (Nv, D) float32 of the visual tokens at decoder sites (0 = merger output / decoder input)."""
    s, e = prep.span['start'], prep.span['end']
    x = cr.embeds(rt, prep)
    store = {0: x[0, s:e].float()}
    handles = []
    for l in sites:
        if l == 0:
            continue
        def hook(module, inputs, output, l=l):
            store[l] = (output[0] if isinstance(output, tuple) else output)[0, s:e].float().to(rt.device)
        handles.append(rt.layers[l - 1].register_forward_hook(hook))
    try:
        rt.model(inputs_embeds=x[:, :e], position_ids=prep.pos[:, :, :e], use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return store


@torch.inference_mode()
def stage_field(rt, out: Path, args) -> None:
    src = Path(args.source)
    cases = {c['id']: c for c in json.loads((src / 'cases.json').read_text())}
    gen = {g['id']: g for g in read_jsonl(src / 'gen.jsonl')}
    root = out / 'field'
    root.mkdir(exist_ok=True)
    dev = torch.device(rt.device)
    for cond in args.field_conditions:
        if (root / f'{cond}.npz').exists():
            continue
        ids = sorted(i for i, c in cases.items() if c['cond'] == cond and gen.get(i, {}).get('score', {}).get('valid'))
        D = rt.model.config.text_config.hidden_size
        acc = {(l, f): [torch.zeros(D + 1, D + 1, dtype=torch.float64, device=dev),
                        torch.zeros(D + 1, 2, dtype=torch.float64, device=dev)] for l in FIELD_SITES for f in (0, 1)}
        t0 = time.time()
        for n, cid in enumerate(ids):
            c = cases[cid]
            prep = cr.prepare(rt, sr.load_image(c), c['prompt'])
            y = torch.tensor(cr.token_centers(prep.grid), dtype=torch.float64, device=dev)
            st = visual_states(rt, prep)
            for l, h in st.items():
                X = torch.cat([h.double(), torch.ones(len(h), 1, dtype=torch.float64, device=dev)], 1)
                G, B = acc[(l, n % 2)]
                G += X.T @ X
                B += X.T @ y
        print(f'field {cond} pass 1 {time.time() - t0:.0f}s', flush=True)
        weights = {}
        for (l, f), (G, B) in acc.items():
            for lam in FIELD_LAMBDAS:
                reg = lam * torch.eye(D + 1, dtype=torch.float64, device=dev)
                reg[-1, -1] = 0
                weights[(l, f, lam)] = torch.linalg.solve(G + reg, B).float()
        del acc
        decoded = {f'{l}_{lam}': [] for l in FIELD_SITES for lam in FIELD_LAMBDAS}
        grids = []
        for n, cid in enumerate(ids):
            c = cases[cid]
            prep = cr.prepare(rt, sr.load_image(c), c['prompt'])
            st = visual_states(rt, prep)
            grids.append(list(prep.grid))
            for l, h in st.items():
                X = torch.cat([h, torch.ones(len(h), 1, device=dev)], 1)
                for lam in FIELD_LAMBDAS:
                    w = weights[(l, 1 - n % 2, lam)]           # decoder fitted on the other image parity
                    decoded[f'{l}_{lam}'].append((X @ w).cpu().numpy().astype(np.float32))
        print(f'field {cond} pass 2 {time.time() - t0:.0f}s', flush=True)
        np.savez_compressed(root / f'{cond}.npz', ids=np.array(ids), grids=np.array(grids),
                            **{k: np.stack(v) for k, v in decoded.items()})
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- main

def load_runtime(name: str, max_memory: str | None) -> qr.Runtime:
    """qr.load on one GPU, or the same checkpoint split over several visible GPUs.

    max_memory: e.g. '0:15GiB,1:9GiB' (local device indices) when no single shared GPU has room for the
    model; inputs are placed on cuda:0 and accelerate moves activations between devices.
    """
    if not max_memory:
        return qr.load(name)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    mm = {int(k): v for k, v in (item.split(':') for item in max_memory.split(','))}
    checkpoint = resolve_model(name, load_paths()['HF_HUB_CACHE'])
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, device_map='auto', max_memory=mm,
        attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    return qr.Runtime(name, checkpoint, model, processor, 'cuda:0')


STAGES = {'gen': stage_gen, 'spotcheck': stage_spotcheck, 'heads': stage_heads, 'kv': stage_kv, 'flow': stage_flow,
          'field': stage_field}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stages', nargs='+', required=True, choices=['cases', *STAGES])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True, help='2.5e run directory (cases, images, answers)')
    p.add_argument('--n-disc', type=int, default=48)
    p.add_argument('--n-grid', type=int, default=40)
    p.add_argument('--n-rc', type=int, default=48)
    p.add_argument('--head-conditions', nargs='+', default=[])
    p.add_argument('--kv-conditions', nargs='+', default=[])
    p.add_argument('--reverse', action='store_true', help='process the kv work list from the end (second process)')
    p.add_argument('--n-heads', type=int, default=24, help='answers per condition in the head stage')
    p.add_argument('--n-kv-full', type=int, default=24, help='answers per full-set condition in the kv stage')
    p.add_argument('--n-kv', type=int, default=20, help='answers per order-variant condition in the kv stage')
    p.add_argument('--kv-full-conditions', nargs='+', default=[], help='conditions with the full group set and splice pairs')
    p.add_argument('--flow-conditions', nargs='+', default=[])
    p.add_argument('--field-conditions', nargs='+', default=[])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--prep-chunk', type=int, default=64)
    p.add_argument('--head-rows', type=int, default=48)
    p.add_argument('--kv-rows', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261020)
    p.add_argument('--max-memory', default='', help="split the model over visible GPUs, e.g. '0:15GiB,1:9GiB'")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = list(args.stages)
    if 'cases' in stages:
        stage_cases(args.output, args)
        stages.remove('cases')
    if not stages:
        return
    rt = load_runtime(args.model, args.max_memory)
    for st in stages:
        meta = {'stage': st, 'args': {k: str(v) for k, v in vars(args).items()}, 'runtime': qr.describe(rt),
                'code_sha256': sr.sha256(Path(__file__)), 'started': datetime.now(timezone.utc).isoformat(),
                'device': torch.cuda.get_device_name(0)}
        append_jsonl(args.output / 'run_log.jsonl', [meta])
        STAGES[st](rt, args.output, args)


if __name__ == '__main__':
    main()
