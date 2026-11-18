"""Experiment 3.14 round 3: why Qwen3-VL's y-reading heads attend along the target's token row (GPU stages).

Results: docs/analysis/position/results_2026-11-05_row_attention.md.

Both models (Qwen3-VL-8B, DeepStack on; Qwen3.5-9B) are teacher-forced on an answer; every measurement is
taken at the query that predicts the first digit of each coordinate value. Two stages:

* `maps`: for the reading heads of round 2 (top heads per format and axis), the softmax attention over the
  image tokens and an exact decomposition of every pre-softmax score q . R k into MRoPE sections. With
  rotated vectors q~, k~ the score is sum_d q~_d k~_d; the rotary dimension pairs (f, f + r/2) belong to the
  temporal (T), height (H) or width (W) section of interleaved MRoPE, the non-rotary dimensions of Qwen3.5
  to section N. The same sums are recomputed with every key rotated as if it sat at the target's first
  token ("content" part); actual minus content is the positional part of each section.
* `edits`: answer-time edits of the image keys / values cached by the prompt (all full-attention layers;
  the prompt itself is computed unedited). Values of a token group are replaced by (a) values of the same
  tokens displaced by the target move (the group's own image, positions shifted by the move offset), (b) the
  values of the moved target in run B (target group only), or (c) the image-token mean. The recovered fraction
  r = (LD_e - LD_A) / (LD_B - LD_A) (3.11 definition, run B = the image with the target moved) measures how
  much of the coordinate change each group's values can induce.

Answers: Qwen3-VL from the round-1 gen file, Qwen3.5 from the 3.11 gen file; sweep images (target moved along
one token row / column of a 10x10 grid) are teacher-forced on the true answer. Layers are 1-based.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.common.stimuli import generate
from analysis_tools.position import causal_vs_linear as cv
from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import coordinate_readout as cr

sr = cf.sr
read_jsonl, append_jsonl, done_ids = sr.read_jsonl, sr.append_jsonl, sr.done_ids

MAP_CONDS = ('t6g10_rowcol', 't6g10_colrow', 't6g10_point', 'disc_point', 'rc_point')
EDIT_CONDS = ('t6g10_rowcol', 't6g10_colrow', 't6g10_point', 'disc_point')
SWEEP_TEMPLATES = ('person', 'ring')
SECTIONS = ('T', 'H', 'W', 'N')
COMPONENTS = tuple(f'{s}_act' for s in SECTIONS) + tuple(f'{s}_con' for s in SECTIONS[:3])
ORDER_CONDS = ('t6g10_rowcol', 't6g10_colrow', 't6g10_point', 'disc_point')
ORDERS = ('raster', 'rev', 'colmajor')
EDIT_ROWS = ('none', 'vT', 'vR', 'vRa', 'vRl', 'vRr', 'vC', 'mT', 'mR', 'mRa', 'mC')


# ----------------------------------------------------------------------------- pure helpers (tested)

def section_labels(rotary_dim: int, head_dim: int, mrope_section: list[int]) -> np.ndarray:
    """Section index (0 T, 1 H, 2 W, 3 N) of every head dimension under interleaved MRoPE.

    Mirrors `recomposition_frequencies` of the Qwen3-VL / Qwen3.5 rotary embedding: frequency index f of the
    rotary half starts as T; f = 1, 4, ... < 3 * section[1] becomes H, f = 2, 5, ... < 3 * section[2] becomes W.
    Dimension d < rotary_dim uses frequency d mod (rotary_dim / 2) (rotate-half layout); d >= rotary_dim is N.
    """
    half = rotary_dim // 2
    lab = np.zeros(half, dtype=np.int64)
    for sec, offset in ((1, 1), (2, 2)):
        lab[offset:mrope_section[sec] * 3:3] = sec
    return np.concatenate([lab, lab, np.full(head_dim - rotary_dim, 3, dtype=np.int64)])


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on the first cos.shape[-1] dimensions of x (..., hd); the rest pass through."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., :r // 2], xr[..., r // 2:]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * cos + rot * sin, xp], -1)


def section_scores(q: torch.Tensor, k: torch.Tensor, labels: torch.Tensor, scaling: float) -> torch.Tensor:
    """(..., 4, Nk) per-section score sums of rotated q (..., hd) against rotated keys (Nk, hd)."""
    prod = q[..., None, :] * k                                                  # (..., Nk, hd)
    out = torch.stack([prod[..., labels == s].sum(-1) for s in range(4)], -2)
    return out * scaling


def token_groups(grid: tuple[int, int], target: list[int]) -> dict[str, list[int]]:
    """Image-token groups (raster indices) relative to the target tokens T.

    R: the target's token rows without T; Ra: the first token column of those rows (row-start tokens) without
    T; Rl / Rr: R tokens left / right of the target columns, Ra excluded; C: the target's token columns
    without T.
    """
    H, W = grid
    T = set(target)
    rows = {t // W for t in target}
    cols = {t % W for t in target}
    R = [i * W + j for i in sorted(rows) for j in range(W) if i * W + j not in T]
    Ra = [i * W for i in sorted(rows) if i * W not in T]
    Rl = [t for t in R if t % W < min(cols) and t not in Ra]
    Rr = [t for t in R if t % W > max(cols)]
    C = [i * W + j for i in range(H) for j in sorted(cols) if i not in rows]
    return {'T': sorted(T), 'R': R, 'Ra': Ra, 'Rl': Rl, 'Rr': Rr, 'C': C}


def shifted(tokens: list[int], grid: tuple[int, int], dr: int, dc: int) -> tuple[list[int], list[int]]:
    """(destinations, sources): tokens whose source (row + dr, col + dc) lies inside the grid."""
    H, W = grid
    dst, src = [], []
    for t in tokens:
        i, j = divmod(t, W)
        if 0 <= i + dr < H and 0 <= j + dc < W:
            dst.append(t)
            src.append((i + dr) * W + j + dc)
    return dst, src


# ----------------------------------------------------------------------------- inputs

def load_runtime(name: str) -> qr.Runtime:
    return cf.load_runtime(name, None)


def prepare(rt, image, prompt: str):
    return cv.prepare(rt, image, prompt)


def ds_of(rt, prep):
    return prep.meta.get('ds') if rt.family == 'qwen3_vl' else None


def sweep_cases(flow_cases: dict) -> list[dict]:
    """Synthetic sweep: target at (r, 4) for r = 0..9 and (4, c) for c = 0..9 in two 10x10 grid templates.

    The base specs are taken from the 3.11 grid cases of each template (colour change, level 4) so that the
    sweep images match the main grid set except for the target position.
    """
    base = {}
    for c in flow_cases.values():
        if c['cond'] == 't6g10_rowcol' and c['spec']['template'] in SWEEP_TEMPLATES:
            base.setdefault(c['spec']['template'], c)
    cases = []
    for tpl in SWEEP_TEMPLATES:
        c0 = base[tpl]
        spec = sr.spec_from_dict(c0['spec'])
        rcs = [(r, 4) for r in range(10)] + [(4, c) for c in range(10) if c != 4]
        fmts = ('rowcol', 'point') if tpl == 'person' else ('rowcol',)
        for rc in rcs:
            image, meta = generate(replace(spec, target=rc))
            truth = sr.grid_truth(meta, rc)
            for fmt in fmts:
                cases.append({'id': f'sweep_{tpl}_{rc[0]}{rc[1]}_{fmt}', 'cond': f'sweep_{fmt}', 'set': 't6',
                              'format': fmt, 'truth': truth, 'spec': {**c0['spec'], 'target': list(rc)},
                              'prompt': cf.T6_QUERY + cf.FORMAT_SUFFIX[fmt], '_image': image})
    return cases


def truth_answer(case: dict) -> str:
    """Teacher-forced answer text of a sweep case (true cell for rowcol, true centre for point)."""
    r, c = case['truth']['target_rc']
    if case['format'] == 'rowcol':
        return f'{{"row": {r + 1}, "column": {c + 1}}}'
    x, y = case['truth']['center']
    W, H = case['truth']['canvas']
    return f'[{{"point_2d": [{round(1000 * x / W)}, {round(1000 * y / H)}]}}]'


def answers(rt, args) -> dict[str, dict]:
    """{case id: gen record} of the model's own answers (valid ones only)."""
    path = args.vl_gen if rt.family == 'qwen3_vl' else args.q35_gen
    return {g['id']: g for g in read_jsonl(Path(path)) if g['score']['valid']}


def reading_heads(summary: Path, model_key: str, top: int) -> list[tuple[int, int]]:
    """Union of the top round-2 reading heads over formats and axes, as (layer, head) with 1-based layers."""
    s = json.loads(summary.read_text())['models'][model_key]['keys']
    heads = set()
    for v in s.values():
        for h in v['reading_heads'][:top]:
            l, k = h['head'][1:].split('_')
            heads.add((int(l), int(k)))
    return sorted(heads)


# ----------------------------------------------------------------------------- stage maps

class AttnInputs:
    """Capture (hidden_states, (cos, sin)) at the input of chosen full-attention layers in one forward."""

    def __init__(self, rt, layers):
        self.store, self.handles = {}, []
        for layer in layers:
            mod = rt.layers[layer - 1].self_attn

            def pre(module, args, kwargs, layer=layer):
                if layer not in self.store:
                    self.store[layer] = (kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0],
                                         kwargs['position_embeddings'])
            self.handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def head_qk(rt, layer: int, hidden: torch.Tensor, queries: list[int]):
    """Pre-RoPE q at the queries (heads, Q, hd) and k at every position (kv_heads, L, hd), after q/k norm."""
    mod = rt.layers[layer - 1].self_attn
    hd = mod.head_dim
    qi = torch.tensor(queries, device=hidden.device)
    if rt.family == 'qwen3_5':
        q, _ = torch.chunk(mod.q_proj(hidden[:, qi]).view(1, len(queries), -1, hd * 2), 2, dim=-1)
    else:
        q = mod.q_proj(hidden[:, qi]).view(1, len(queries), -1, hd)
    q = mod.q_norm(q)[0].transpose(0, 1).float()
    k = mod.k_norm(mod.k_proj(hidden).view(1, hidden.shape[1], -1, hd))[0].transpose(0, 1).float()
    return q, k, mod


@torch.inference_mode()
def case_maps(rt, prep, ans_ids: list[int], values: list[dict], heads, target: list[int],
              order: torch.Tensor | None = None) -> dict:
    """Attention rows and MRoPE section decomposition of the chosen heads at each value's first-digit query.

    order: optional (Nv,) slot -> raster-token permutation of the image span (each token keeps its MRoPE ids
    and DeepStack features, only its place in the sequence changes). Per-token outputs are returned in raster
    order. The digit log-probabilities at the queries are returned as `digit_logp` (Q, 10).
    """
    L = len(prep.ids)
    s, e = prep.span['start'], prep.span['end']
    Nv = e - s
    order = torch.arange(Nv) if order is None else order.cpu()
    inv = torch.argsort(order)                                                   # raster token -> slot
    pos0 = cr.permute_pos(prep, prep.pos, order)
    x, pos = cr.teacher_forced(rt, prep, ans_ids[:-1] if len(ans_ids) > 1 else ans_ids,
                               x=cr.embeds(rt, prep, order=order), pos=pos0)
    queries = cf.first_digit_queries(values, L)
    layers = sorted({l for l, _ in heads})
    ds = ds_of(rt, prep)
    if ds is not None:
        ds = [d[order.to(d.device)] for d in ds]
    with AttnInputs(rt, layers) as cap, cv.deepstack_rows(rt, None if ds is None else [ds], (s, e)):
        lg = rt.model(inputs_embeds=x, position_ids=pos, use_cache=False,
                      logits_to_keep=x.shape[1] - min(queries)).logits[0]
    dig = torch.tensor(cf.digit_ids(rt), device=lg.device)
    digit_logp = torch.log_softmax(lg[[q - min(queries) for q in queries]][:, dig].float(), -1).cpu().numpy()
    mrope_section = rt.model.model.language_model.rotary_emb.mrope_section
    out = {k: [] for k in ('att', 'text', 'comp', 'q', 'kmean')}
    ref = s + int(inv[min(target)])
    back = inv.to('cuda' if torch.cuda.is_available() else 'cpu')
    for layer in layers:
        hidden, (cos, sin) = cap.store[layer]
        q, k, mod = head_qk(rt, layer, hidden, queries)
        cos, sin = cos[0].float(), sin[0].float()                               # (Lx, r)
        labels = torch.as_tensor(section_labels(cos.shape[-1], mod.head_dim, mrope_section), device=q.device)
        qi = torch.tensor(queries, device=q.device)
        qr_ = rotate(q, cos[qi], sin[qi])                                       # (heads, Q, hd)
        kr = rotate(k, cos, sin)                                                # (kvh, Lx, hd)
        kc = rotate(k[:, s:e], cos[ref].expand(e - s, -1), sin[ref].expand(e - s, -1))
        groups = mod.num_key_value_groups
        for l2, h in heads:
            if l2 != layer:
                continue
            g = h // groups
            sc = (qr_[h] @ kr[g].T) * mod.scaling                               # (Q, Lx)
            sc = sc.masked_fill(torch.arange(sc.shape[1], device=sc.device)[None] > qi[:, None], float('-inf'))
            a = sc.softmax(-1)
            act = section_scores(qr_[h], kr[g, s:e], labels, mod.scaling)       # (Q, 4, Nv) in slot order
            con = section_scores(qr_[h], kc[g], labels, mod.scaling)[:, :3]
            b = back.to(a.device)
            act, con = act[..., b], con[..., b]
            out['att'].append(a[:, s:e][:, b].cpu().numpy())
            out['text'].append((1 - a[:, s:e].sum(-1)).cpu().numpy())
            out['comp'].append(torch.cat([act, con], 1).cpu().numpy())
            col0 = torch.zeros(e - s, dtype=torch.bool, device=k.device)
            col0[::prep.grid[1]] = True
            kimg = k[g, s:e][back.to(k.device)]
            out['q'].append(q[h].cpu().numpy())
            out['kmean'].append(torch.stack([kimg.mean(0), kimg[col0].mean(0), kimg[~col0].mean(0)]).cpu().numpy())
    head_order = [(l, h) for layer in layers for (l, h) in heads if l == layer]
    return {'heads': np.array(head_order), 'att': np.stack(out['att']).astype(np.float16),
            'text': np.stack(out['text']).astype(np.float32), 'comp': np.stack(out['comp']).astype(np.float16),
            'q': np.stack(out['q']).astype(np.float16), 'kmean': np.stack(out['kmean']).astype(np.float16),
            'pos_img': pos[1:, 0, s:e][:, back.to(pos.device)].cpu().numpy(), 'pos_q': pos[1:, 0, queries].cpu().numpy(),
            'digit_logp': digit_logp.astype(np.float32), 'order': order.numpy(),
            'queries': np.array(queries), 'keys': np.array([v['key'] for v in values]),
            'target': np.array(target), 'grid': np.array(prep.grid)}


def stage_maps(rt, out: Path, args, cases: dict, gens: dict, heads, sweep: list[dict]) -> None:
    mdir = out / f'maps_{args.tag}'
    mdir.mkdir(exist_ok=True)
    todo, per = [], {}
    for i, g in gens.items():
        if i in cases and g['cond'] in MAP_CONDS and per.get(g['cond'], 0) < args.n_maps:
            per[g['cond']] = per.get(g['cond'], 0) + 1
            todo.append((cases[i], g))
    for c in sweep:
        ids = rt.tokenizer(truth_answer(c), add_special_tokens=False)['input_ids']
        vals = cf.answer_values(rt, ids, c['format'])
        if vals is None:
            raise ValueError(f"Cannot parse the teacher-forced answer of {c['id']}")
        todo.append((c, {'id': c['id'], 'cond': c['cond'], 'ids': ids + [0], 'values': vals}))
    for c, g in todo:
        f = mdir / f"{g['id']}.npz"
        if f.exists():
            continue
        t0 = time.time()
        image = c['_image'] if '_image' in c else sr.load_image(c)
        prep = prepare(rt, image, c['prompt'])
        target = cr.tokens_overlapping(c['truth']['box'], prep.grid)
        rec = case_maps(rt, prep, g['ids'], g['values'], heads, target)
        np.savez_compressed(f, **rec)
        if '_image' in c:
            idir = out / 'sweep_images'
            idir.mkdir(exist_ok=True)
            if not (idir / f"{c['id'].rsplit('_', 1)[0]}.png").exists():
                image.save(idir / f"{c['id'].rsplit('_', 1)[0]}.png")
        print(f"maps {g['id']} {time.time() - t0:.1f}s", flush=True)


def raster_orders(grid: tuple[int, int]) -> dict[str, torch.Tensor]:
    """Slot -> raster-token orders of the image span.

    raster: the native left-to-right, top-to-bottom order; rev: every token row read right to left (rows
    still top to bottom); colmajor: token columns read top to bottom, columns left to right.
    """
    H, W = grid
    idx = torch.arange(H * W).view(H, W)
    return {'raster': idx.reshape(-1), 'rev': idx.flip(1).reshape(-1), 'colmajor': idx.T.reshape(-1)}


def stage_order(rt, out: Path, args, cases: dict, gens: dict, heads) -> None:
    mdir = out / f'order_{args.tag}'
    mdir.mkdir(exist_ok=True)
    per = {}
    for i, g in gens.items():
        if i not in cases or g['cond'] not in ORDER_CONDS or per.get(g['cond'], 0) >= args.n_order:
            continue
        per[g['cond']] = per.get(g['cond'], 0) + 1
        c = cases[i]
        prep = None
        for name in ORDERS:
            f = mdir / f"{g['id']}__{name}.npz"
            if f.exists():
                continue
            t0 = time.time()
            prep = prepare(rt, sr.load_image(c), c['prompt']) if prep is None else prep
            target = cr.tokens_overlapping(c['truth']['box'], prep.grid)
            rec = case_maps(rt, prep, g['ids'], g['values'], heads, target, raster_orders(prep.grid)[name])
            rec.pop('comp')
            np.savez_compressed(f, **rec)
            print(f"order {g['id']} {name} {time.time() - t0:.1f}s", flush=True)


CROSS_CONDS = ('AA', 'AB', 'A_anchorB', 'BB')


@torch.inference_mode()
def rotated_qk(rt, prep, x, pos, queries: list[int], layers, ds):
    """{layer: (rotated q at the queries (heads, Q, hd), rotated k (kv_heads, Lx, hd), module)} of one forward."""
    s, e = prep.span['start'], prep.span['end']
    with AttnInputs(rt, layers) as cap, cv.deepstack_rows(rt, None if ds is None else [ds], (s, e)):
        rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
    out = {}
    for layer in layers:
        hidden, (cos, sin) = cap.store[layer]
        q, k, mod = head_qk(rt, layer, hidden, queries)
        cos, sin = cos[0].float(), sin[0].float()
        qi = torch.tensor(queries, device=q.device)
        out[layer] = (rotate(q, cos[qi], sin[qi]), rotate(k, cos, sin), mod)
    return out


def stage_cross(rt, out: Path, args, cases: dict, gens: dict, heads) -> None:
    """Run-A queries against run-B keys (run B = target moved two cells; identical positions, so no re-rotation).

    Conditions per head and query: AA (native), AB (every image key from run B), A_anchorB (only the row-start
    keys of run A's target rows from run B), BB (run B's own query and keys). Text keys always come from the
    query's own run. Saved as softmax rows over the image tokens (raster order).
    """
    mdir = out / f'cross_{args.tag}'
    mdir.mkdir(exist_ok=True)
    layers = sorted({l for l, _ in heads})
    per = {}
    for i, g in gens.items():
        if i not in cases or g['cond'] not in ORDER_CONDS[:3] or per.get(g['cond'], 0) >= args.n_order:
            continue
        per[g['cond']] = per.get(g['cond'], 0) + 1
        c = cases[i]
        prep = prepare(rt, sr.load_image(c), c['prompt'])
        L = len(prep.ids)
        s, e = prep.span['start'], prep.span['end']
        target = cr.tokens_overlapping(c['truth']['box'], prep.grid)
        anchors = token_groups(prep.grid, target)['Ra']
        queries = cf.first_digit_queries(g['values'], L)
        xa, pos = cr.teacher_forced(rt, prep, g['ids'][:-1], x=cr.embeds(rt, prep))
        qa = rotated_qk(rt, prep, xa, pos, queries, layers, ds_of(rt, prep))
        done_axes = set()
        for mv in sr.moves(c, prep.grid, target, 2, 4):
            f = mdir / f"{g['id']}__{mv['axis']}.npz"
            if mv['axis'] in done_axes or f.exists():
                continue
            done_axes.add(mv['axis'])
            t0 = time.time()
            pb = prepare(rt, sr.run_b_image(c, mv), c['prompt'])
            xb, _ = cr.teacher_forced(rt, pb, g['ids'][:-1], x=cr.embeds(rt, pb))
            qb = rotated_qk(rt, pb, xb, pos, queries, layers, ds_of(rt, pb))
            att = []
            for l, h in heads:
                qA, kA, mod = qa[l]
                qB, kB, _ = qb[l]
                gidx = h // mod.num_key_value_groups
                rows = []
                for name in CROSS_CONDS:
                    q = qB[h] if name == 'BB' else qA[h]
                    k = (kB if name in ('AB', 'BB') else kA)[gidx].clone()
                    if name == 'AB':
                        k[:s] = kA[gidx, :s]
                        k[e:] = kA[gidx, e:]
                    if name == 'A_anchorB' and anchors:
                        a_idx = torch.tensor(anchors, device=k.device) + s
                        k[a_idx] = kB[gidx, a_idx]
                    sc = (q @ k.T) * mod.scaling
                    qi = torch.tensor(queries, device=sc.device)
                    sc = sc.masked_fill(torch.arange(sc.shape[1], device=sc.device)[None] > qi[:, None], float('-inf'))
                    rows.append(sc.softmax(-1)[:, s:e].cpu().numpy())
                att.append(np.stack(rows))
            np.savez_compressed(f, heads=np.array(heads), att=np.stack(att).astype(np.float16),
                                conds=np.array(CROSS_CONDS), target=np.array(target), moved=np.array(mv['shifted']),
                                grid=np.array(prep.grid), keys=np.array([v['key'] for v in g['values']]),
                                anchors=np.array(anchors, dtype=np.int64))
            print(f"cross {g['id']} {mv['axis']} {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- stage edits

def edit_plan(grid, target: list[int], moved: list[int]) -> tuple[dict, tuple[int, int]]:
    """Per edit row: (kind, destinations, sources); kind 'B' (from run B), 'self' (run A shifted), 'mean'."""
    W = grid[1]
    dr = moved[0] // W - target[0] // W
    dc = moved[0] % W - target[0] % W
    g = token_groups(grid, target)
    plan = {'none': ('self', [], []), 'vT': ('B', sorted(target), sorted(moved))}
    for name in ('R', 'Ra', 'Rl', 'Rr', 'C'):
        dst, src = shifted(g[name], grid, dr, dc)
        plan[f'v{name}'] = ('self', dst, src)
    for name in ('T', 'R', 'Ra', 'C'):
        plan[f'm{name}'] = ('mean', g[name], [])
    return plan, (dr, dc)


@torch.inference_mode()
def edit_logits(rt, xa, xb, pos, L: int, span, plan: dict, ds_a, ds_b) -> torch.Tensor:
    """Teacher-forced tail logits; row 0 = run B, row 1 + i = run A with EDIT_ROWS[i] applied to the cache."""
    s, e = span
    R = 1 + len(EDIT_ROWS)
    x = torch.cat([xb] + [xa] * len(EDIT_ROWS), 0)
    p = pos.expand(-1, R, -1).contiguous()
    ds = None if ds_a is None else [ds_b] + [ds_a] * len(EDIT_ROWS)
    with cv.deepstack_rows(rt, ds, span):
        o = rt.model(inputs_embeds=x[:, :L - 1], position_ids=p[:, :, :L - 1], use_cache=True, logits_to_keep=1)
    cache = o.past_key_values
    for l, ci in cv.kv_index(rt).items():
        _, values = sr.cache_kv(cache, ci)
        ref = values[1].clone()                                                  # unedited run A
        for b, name in enumerate(EDIT_ROWS, start=1):
            kind, dst, src = plan[name]
            if not dst:
                continue
            d = torch.tensor(dst, device=values.device) + s
            if kind == 'B':
                values[b, :, d] = values[0, :, torch.tensor(src, device=values.device) + s]
            elif kind == 'self':
                values[b, :, d] = ref[:, torch.tensor(src, device=values.device) + s]
            else:
                values[b, :, d] = ref[:, s:e].mean(1, keepdim=True).to(values.dtype)
    o = rt.model(inputs_embeds=x[:, L - 1:], position_ids=p[:, :, L - 1:], past_key_values=cache, use_cache=True)
    return o.logits


def stage_edits(rt, out: Path, args, cases: dict, gens: dict) -> None:
    path = out / f'edits_{args.tag}.jsonl'
    digits = cf.digit_ids(rt)
    per = {}
    for i, g in gens.items():
        if i in cases and g['cond'] in EDIT_CONDS and per.get(g['cond'], 0) < args.n_edits:
            per[g['cond']] = per.get(g['cond'], 0) + 1
            if g['id'] in done_ids(path):
                continue
            c = cases[i]
            t0 = time.time()
            prep = prepare(rt, sr.load_image(c), c['prompt'])
            L = len(prep.ids)
            s, e = prep.span['start'], prep.span['end']
            target = cr.tokens_overlapping(c['truth']['box'], prep.grid)
            xa_tf, pos = cr.teacher_forced(rt, prep, g['ids'][:-1], x=cr.embeds(rt, prep))
            recs = []
            done_axes = set()
            for mv in sr.moves(c, prep.grid, target, 2, 4):
                if mv['axis'] in done_axes:
                    continue
                done_axes.add(mv['axis'])
                pb = prepare(rt, sr.run_b_image(c, mv), c['prompt'])
                if len(pb.ids) != L:
                    raise ValueError(f"Run B length differs for {c['id']}")
                xb_tf, _ = cr.teacher_forced(rt, pb, g['ids'][:-1], x=cr.embeds(rt, pb))
                plan, off = edit_plan(prep.grid, target, mv['shifted'])
                lg = edit_logits(rt, xa_tf, xb_tf, pos, L, (s, e), plan, ds_of(rt, prep), ds_of(rt, pb))
                tab = cf.ld_table(lg, g['ids'], g['values'], digits, b_row=0)
                dig = torch.tensor(digits, device=lg.device)
                for row in tab:
                    i0 = g['values'][row['k']]['toks'][0]
                    row['logp_dA'] = torch.log_softmax(lg[:, i0, dig].float(), -1)[:, row['dA']].cpu().tolist()
                recs.append({'axis': mv['axis'], 'offset': list(off), 'sizes': {k: len(v[1]) for k, v in plan.items()},
                             'values': tab})
            append_jsonl(path, [{'id': g['id'], 'cond': g['cond'], 'grid': list(prep.grid), 'target': target,
                                 'rows': list(EDIT_ROWS), 'moves': recs, 'seconds': round(time.time() - t0, 1)}])
            print(f"edits {g['id']} {time.time() - t0:.1f}s", flush=True)
            torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--stages', nargs='+', default=['maps', 'edits'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--flow-run', type=Path, required=True, help='3.11 run directory (cases.json, Qwen3.5 gen.jsonl)')
    p.add_argument('--vl-gen', type=Path, required=True, help='3.14 round-1 gen_vl_ds_on.jsonl')
    p.add_argument('--r2-summary', type=Path, required=True, help='3.14 round-2 summary.json (reading heads)')
    p.add_argument('--top-heads', type=int, default=5)
    p.add_argument('--n-maps', type=int, default=40, help='answers per condition in the maps stage')
    p.add_argument('--n-edits', type=int, default=40, help='answers per condition in the edits stage')
    p.add_argument('--n-order', type=int, default=40, help='answers per condition in the order stage')
    p.add_argument('--torch-deltanet', action='store_true', help='Qwen3.5 on A40: torch DeltaNet kernel')
    args = p.parse_args()
    if args.torch_deltanet:
        from analysis_tools.position import causal_vs_linear_run as cvr
        cvr.use_torch_deltanet()
    args.q35_gen = args.flow_run / 'gen.jsonl'
    args.output.mkdir(parents=True, exist_ok=True)
    rt = load_runtime(args.model)
    args.tag = 'vl' if rt.family == 'qwen3_vl' else 'q35'
    heads = reading_heads(args.r2_summary, 'vl_ds_on' if args.tag == 'vl' else 'q35_none', args.top_heads)
    cases = cf.load_cases(args.flow_run)
    gens = answers(rt, args)
    append_jsonl(args.output / 'run_log.jsonl', [{
        'stages': args.stages, 'tag': args.tag, 'heads': heads, 'args': {k: str(v) for k, v in vars(args).items()},
        'runtime': qr.describe(rt), 'device': torch.cuda.get_device_name(0),
        'started': datetime.now(timezone.utc).isoformat()}])
    if 'maps' in args.stages:
        stage_maps(rt, args.output, args, cases, gens, heads, sweep_cases(cases))
    if 'order' in args.stages:
        stage_order(rt, args.output, args, cases, gens, heads)
    if 'cross' in args.stages:
        stage_cross(rt, args.output, args, cases, gens, heads)
    if 'edits' in args.stages:
        stage_edits(rt, args.output, args, cases, gens)


if __name__ == '__main__':
    main()
