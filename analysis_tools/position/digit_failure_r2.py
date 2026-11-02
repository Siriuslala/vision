"""Experiment 3.12 round 2: refining the digit-readout mechanism of Qwen3.5-9B (GPU stages).

Design and results: docs/analysis/position/results_2026-10-21_digit_failure_r2.md. The round-1 engine
(`digit_failure.py`, `digit_failure_run.py`) provides the units, prompts, answer prefixes and stimuli.

All stages use a *prefill-then-step* scheme. The prompt and the answer prefix are prefilled once (cache
over every position before the studied step); the cache is then expanded to R rows, optionally edited
row by row, and a single decoding step is run with forward hooks. Two steps are studied:

  q0  the last answer-prefix token; it predicts the first digit (prefill ends before it);
  q1  the forced first digit k itself; it predicts the second digit (prefill ends at q0).

Stages (comma list via --stages):
  weights   digit unembedding rows, final-norm weight and per-neuron / per-head digit projections;
  attn      softmax attention of the full-attention layers at q0 and q1, split into token groups
            (pre-image text, image, post-image text, q0, the query itself);
  patch     source patching on disc sweep pairs (A, B): run B's image KV, post-image text KV, q0 KV,
            Gated DeltaNet states, or L20 H11 output are copied into run A at the studied step;
  tail      the same pairs; run B's image KV is made visible from a chosen answer position onward
            (scan over the start position), locating where along the answer the image is read;
  capture   MLP 27-31 activations, all L32 head outputs and the final RMS at q1 for every forced k;
  ablate    mean ablation at q1 of the centre-prior neurons (selected from `capture`) and of L32 H8 / H10;
  steer     add a vector to the q0 residual (layer 24 or 28 output) along the first-digit neighbour-margin
            direction (or a random control) and read the first digit at q0 and the second digit at q1.
Layer numbers are 1-based. Head h of a full-attention layer is the o_proj input slice [h*256, (h+1)*256).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import digit_failure as df
from analysis_tools.position import digit_failure_run as dr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

FULL_LAYERS = df.FULL_LAYERS
PRIOR_MLP_LAYERS = (27, 28, 29, 30, 31)
HEAD_DIM = 256
READER = (20, 11)                    # the dominant coordinate reader of experiment 3.11
PRIOR_HEADS = ((32, 8), (32, 10))
PRIOR_CASES = ((4, 9), (5, 0))       # (forced first digit k, centre-prior second digit)
DELTA_GROUPS = ((1, 8), (9, 16), (17, 24), (25, 32))   # layer ranges for the layer-resolved conv-state copies


# ----------------------------------------------------------------------------- inputs

def unit_inputs(rt, u: df.Unit) -> dict:
    """Prompt + answer prefix of a unit, cut before q1.

    Returns x (1, L0, D) inputs_embeds through q0, pos (4, 1, L0 + 1) position ids through q1, q0,
    the image span (start, end) or None, and the number of prompt tokens L.
    """
    pre = df.prefix_ids(rt, u.prefix)
    img = dr.unit_image(u)
    if img is None:
        x0, p0, L = df.text_prepare(rt, u.prompt)
        span = None
    else:
        prep = cr.prepare(rt, img, u.prompt)
        x0, p0, L = cr.embeds(rt, prep), prep.pos, len(prep.ids)
        span = (prep.span['start'], prep.span['end'])
    x, p = df.append_tokens(rt, x0, p0, pre + [df.digit_ids(rt)[1]])        # the k slot is replaced per row
    q0 = L + len(pre) - 1
    if q0 < 1:
        raise ValueError(f'Unit {u.id} has no token before q0')
    return {'x': x[:, :q0 + 1], 'pos': p, 'q0': q0, 'span': span, 'L': L}


def digit_embeds(rt, ks: list[int]) -> torch.Tensor:
    ids = torch.tensor([df.digit_ids(rt)[k] for k in ks], device=rt.model.device)
    return rt.model.model.get_input_embeddings()(ids)[:, None]                  # (R, 1, D)


@torch.inference_mode()
def prefill(rt, x: torch.Tensor, pos: torch.Tensor, rows: list[int]):
    """Prefill the distinct inputs x (N, T, D) and expand the cache to len(rows) rows (row i <- input rows[i])."""
    o = rt.model(inputs_embeds=x, position_ids=pos[:, :, :x.shape[1]].expand(-1, x.shape[0], -1).contiguous(),
                 use_cache=True, logits_to_keep=1)
    cache = o.past_key_values
    cache.reorder_cache(torch.tensor(rows, device=x.device))
    return cache


@torch.inference_mode()
def step(rt, cache, emb: torch.Tensor, pos_t: torch.Tensor) -> torch.Tensor:
    """One cached decoding step; emb (R, 1, D), pos_t (4, 1, 1). Returns digit log-probabilities (R, 10)."""
    R = emb.shape[0]
    o = rt.model(inputs_embeds=emb.to(rt.model.dtype), position_ids=pos_t.expand(-1, R, -1).contiguous(),
                 past_key_values=cache, use_cache=True)
    lp = o.logits[:, -1].float().log_softmax(-1)
    return lp[:, torch.tensor(df.digit_ids(rt), device=lp.device)]


def cache_layers(rt, cache) -> tuple[dict[int, object], dict[int, object]]:
    """1-based layer -> cache layer object, split into full-attention and Gated DeltaNet layers."""
    full, lin = {}, {}
    for l in range(1, df.N_LAYERS + 1):
        lay = cache.layers[l - 1]
        (full if l in FULL_LAYERS else lin)[l] = lay
    for l, lay in full.items():
        if not hasattr(lay, 'keys'):
            raise TypeError(f'Layer {l} cache has no keys; cache layout differs from the expected one')
    for l, lay in lin.items():
        if not hasattr(lay, 'recurrent_states'):
            raise TypeError(f'Layer {l} cache has no recurrent_states')
    return full, lin


def copy_kv(full: dict, dst: int, src: int, positions: list[int], layers) -> None:
    if not positions:
        return
    for l in layers:
        lay = full[l]
        ps = torch.tensor(positions, device=lay.keys.device)
        lay.keys[dst, :, ps] = lay.keys[src, :, ps]
        lay.values[dst, :, ps] = lay.values[src, :, ps]


def copy_delta(lin: dict, dst: int, src: int, conv: bool = True, recurrent: bool = True, last_only: bool = False,
               layers=None) -> None:
    """Copy Gated DeltaNet cache states of row src into row dst.

    The conv state of a layer holds the in_proj_qkv outputs of the last 4 cached tokens (slot -1 = the
    token right before the step); the next step convolves slots -3..-1 with its own projection, so the
    conv state is a direct short-range channel from the preceding tokens. last_only copies slot -1 only.
    """
    for l, lay in lin.items():
        if layers is not None and l not in layers:
            continue
        for i in lay.recurrent_states:
            if recurrent and lay.is_recurrent_states_initialized[i]:
                lay.recurrent_states[i][dst] = lay.recurrent_states[i][src]
            if conv and lay.is_conv_states_initialized[i]:
                if last_only:
                    lay.conv_states[i][dst, :, -1] = lay.conv_states[i][src, :, -1]
                else:
                    lay.conv_states[i][dst] = lay.conv_states[i][src]


class OprojEdit:
    """Edit head slices of the o_proj input of full-attention layers during a single decoding step.

    ops: list of (row, layer, head, source); source is an int (copy from that row, read before any edit)
    or a tensor of shape (256,) (replace, e.g. by a mean for mean ablation).
    """

    def __init__(self, rt, ops: list[tuple[int, int, int, object]]):
        self.handles = []
        by_layer: dict[int, list] = {}
        for op in ops:
            by_layer.setdefault(op[1], []).append(op)
        for layer, lops in by_layer.items():
            mod = rt.layers[layer - 1].self_attn.o_proj

            def pre(module, args, lops=lops):
                x = args[0]
                y = x.clone()
                for row, _, h, src in lops:
                    sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                    y[row, -1, sl] = x[src, -1, sl] if isinstance(src, int) else src.to(x.dtype).to(x.device)
                return (y,)
            self.handles.append(mod.register_forward_pre_hook(pre))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


class NeuronEdit:
    """Replace selected MLP intermediate activations (down_proj inputs) at the last position for chosen rows.

    edits: layer -> (rows tensor, neuron index tensor, values tensor of shape (len(neurons),)).
    """

    def __init__(self, rt, edits: dict[int, tuple[list[int], torch.Tensor, torch.Tensor]]):
        self.handles = []
        for layer, (rows, idx, vals) in edits.items():
            mod = rt.layers[layer - 1].mlp.down_proj

            def pre(module, args, rows=rows, idx=idx, vals=vals):
                x = args[0].clone()
                r = torch.tensor(rows, device=x.device)
                x[r[:, None], -1, idx.to(x.device)[None, :]] = vals.to(x.dtype).to(x.device)[None, :]
                return (x,)
            self.handles.append(mod.register_forward_pre_hook(pre))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


# ----------------------------------------------------------------------------- stage: weights

@torch.inference_mode()
def stage_weights(rt, out: Path, args) -> None:
    lm = rt.model.model.language_model
    dig = torch.tensor(df.digit_ids(rt), device=rt.model.lm_head.weight.device)
    W_U = rt.model.lm_head.weight[dig].float()                                   # (10, D)
    g = (1.0 + lm.norm.weight.float()).to(W_U.device)                           # final RMSNorm gain
    U = (W_U * g).T                                                              # (D, 10): residual -> digit logit / rms
    res = {'W_U': W_U.cpu().numpy(), 'norm_gain': g.cpu().numpy(), 'norm_eps': np.array(lm.norm.eps)}
    for l in PRIOR_MLP_LAYERS:
        Wd = rt.layers[l - 1].mlp.down_proj.weight.float()                       # (D, 12288)
        res[f'mlp{l}_proj'] = (Wd.T @ U.to(Wd.device)).cpu().numpy()             # (12288, 10)
    for l in FULL_LAYERS:
        Wo = rt.layers[l - 1].self_attn.o_proj.weight.float()                    # (D, 16*256)
        res[f'L{l}_head_proj'] = (Wo.T @ U.to(Wo.device)).view(df.N_HEADS, HEAD_DIM, 10).cpu().numpy()
    np.savez(out / 'weights.npz', **res)
    print('weights saved', flush=True)


# ----------------------------------------------------------------------------- stage: attention groups

@torch.inference_mode()
def attention_groups(rt, x: torch.Tensor, pos: torch.Tensor, queries: list[int], groups: dict[str, list[int]]) -> dict:
    """Attention mass of every full-attention head at the queries on each token group (batch 1, full prefill).

    Recomputes q/k (q/k norm, partial MRoPE, GQA) from the captured layer inputs, as df.attention_mass;
    returns name -> list over layers of (heads, Q) arrays and the recomputation errors.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    store, handles = {}, []
    for layer in FULL_LAYERS:
        mod = rt.layers[layer - 1].self_attn

        def pre(module, a, kw, layer=layer):
            store.setdefault(layer, {})['h'] = kw['hidden_states'] if 'hidden_states' in kw else a[0]
            store[layer]['pe'] = kw['position_embeddings']

        def post(module, a, kw, o, layer=layer):
            store[layer]['out'] = o[0]
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(mod.register_forward_hook(post, with_kwargs=True))
    try:
        rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    out = {name: [] for name in groups}
    out['check'] = []
    for layer in FULL_LAYERS:
        mod = rt.layers[layer - 1].self_attn
        h, (cos, sin), o_ref = store[layer]['h'], store[layer]['pe'], store[layer]['out']
        dev = h.device
        q_idx = torch.tensor(queries, device=dev)
        L, hd = h.shape[1], mod.head_dim
        q, gate = torch.chunk(mod.q_proj(h[:, q_idx]).view(1, len(queries), -1, hd * 2), 2, dim=-1)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(1, L, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(1, L, -1, hd).transpose(1, 2)
        cos, sin = cos.to(dev), sin.to(dev)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, q_idx], sin[:, q_idx])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        k, v = repeat_kv(k, mod.num_key_value_groups), repeat_kv(v, mod.num_key_value_groups)
        scores = (q.float() @ k.float().transpose(2, 3)) * mod.scaling
        mask = torch.arange(L, device=dev)[None, :] > q_idx[:, None]
        a = scores.masked_fill(mask, float('-inf')).softmax(-1)[0]               # (heads, Q, L)
        g = torch.sigmoid(gate.float()).transpose(1, 2)
        recon = mod.o_proj((((a[None] @ v.float()) * g).transpose(1, 2).reshape(1, len(queries), -1)).to(h.dtype))
        out['check'].append(float((recon[0] - o_ref[0, q_idx]).abs().max() / o_ref[0, q_idx].abs().max()))
        for name, idx in groups.items():
            ii = torch.tensor(idx, device=dev, dtype=torch.long)
            out[name].append(a[:, :, ii].sum(-1).cpu().numpy() if len(idx) else np.zeros(tuple(a.shape[:2])))
    return out


def attn_units(args) -> list[df.Unit]:
    units = dr.build_units(args)
    keep = []
    for fam, every in (('sweep_h', 3), ('sweep_v', 3), ('refcoco', 4), ('textconv', 2)):
        keep += [u for u in units if u.family == fam][::every]
    return keep


def stage_attn(rt, out: Path, args) -> None:
    path = out / 'attn.jsonl'
    done = {r['unit'] for r in read_jsonl(path)} if path.exists() else set()
    t0 = time.time()
    units = attn_units(args)
    units = units[:args.limit] if args.limit else units
    for j, u in enumerate(units):
        if u.id in done or not (100 <= u.truth < 1000):
            continue
        inp = unit_inputs(rt, u)
        q0 = inp['q0']
        k = df.hundreds(u.truth)
        x = torch.cat([inp['x'], digit_embeds(rt, [k])[:1].to(inp['x'].dtype)], 1)
        pos = inp['pos'][:, :, :q0 + 2]
        s, e = inp['span'] if inp['span'] else (0, 0)
        groups = {'pre': list(range(0, s)) if inp['span'] else list(range(0, q0)), 'image': list(range(s, e)),
                  'post': list(range(e, q0)) if inp['span'] else [], 'q0': [q0], 'q1': [q0 + 1]}
        res = attention_groups(rt, x, pos, [q0, q0 + 1], groups)
        append_jsonl(path, [{'unit': u.id, 'family': u.family, 'truth': u.truth, 'k': k, 'layers': list(FULL_LAYERS),
                             **{n: [m.tolist() for m in res[n]] for n in groups}, 'check': res['check']}])
        if j % 20 == 0:
            print(f'attn {j}/{len(units)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- stage: source patching

def patch_pairs(units: list[df.Unit]) -> list[dict]:
    """Directed (A, B) pairs on the disc sweeps.

    q1 pairs: same first digit k (100 <= x < 900), A near the low end of the bin (offset 3-30 units) and
    B near the high end (offset 70-97), plus the reversed pairs; the forced first digit is k for both.
    q0 pairs: A in the middle of bin k, B in the middle of bin k + 1 (and reversed).
    """
    pairs = []
    for fam in ('sweep_h', 'sweep_v'):
        us = sorted([u for u in units if u.family == fam], key=lambda u: u.truth)
        for k in range(1, 9):
            lo = [u for u in us if 100 * k + 3 <= u.truth < 100 * k + 30][:2]
            hi = [u for u in us if 100 * k + 70 <= u.truth < 100 * k + 97][-2:]
            for a, b in zip(lo, hi):
                pairs += [{'role': 'q1', 'A': a.id, 'B': b.id, 'k': k}, {'role': 'q1', 'A': b.id, 'B': a.id, 'k': k}]
            mid = [u for u in us if 100 * k + 35 <= u.truth < 100 * k + 65]
            nxt = [u for u in us if 100 * (k + 1) + 35 <= u.truth < 100 * (k + 1) + 65]
            if mid and nxt and k < 9:
                a, b = mid[len(mid) // 2], nxt[len(nxt) // 2]
                pairs += [{'role': 'q0', 'A': a.id, 'B': b.id, 'k': k}, {'role': 'q0', 'A': b.id, 'B': a.id, 'k': k + 1}]
    return pairs


def patch_conditions(role: str) -> list[str]:
    conds = ['A', 'B', 'img', 'post', 'delta', 'rec', 'conv', 'convlast', 'img+delta', 'post+delta', 'img+post',
             'img+rec', 'img+convlast']
    if role == 'q1':
        conds += ['q0', 'q0+delta', 'q0+convlast']
    conds += [f'convlast@{a}-{b}' for a, b in DELTA_GROUPS]
    conds += [f'img@{l}' for l in FULL_LAYERS]
    conds += ['H11<-img@20', 'H11<-B', 'L20<-B']
    return conds


@torch.inference_mode()
def run_pair(rt, ua: df.Unit, ub: df.Unit, role: str, k: int) -> dict:
    ia, ib = unit_inputs(rt, ua), unit_inputs(rt, ub)
    if ia['x'].shape != ib['x'].shape or ia['span'] != ib['span']:
        raise ValueError(f'Pair {ua.id}/{ub.id} differs in length or image span')
    q0, (s, e) = ia['q0'], ia['span']
    conds = patch_conditions(role)
    R = len(conds)
    T = q0 if role == 'q0' else q0 + 1               # number of cached positions before the studied step
    X = torch.cat([ia['x'], ib['x']], 0)[:, :T]
    cache = prefill(rt, X, ia['pos'], [1 if c == 'B' else 0 for c in conds])
    full, lin = cache_layers(rt, cache)
    b = conds.index('B')
    img, post = list(range(s, e)), list(range(e, min(q0, T)))
    for r, c in enumerate(conds):
        parts = c.split('+')
        for part in parts:
            if part == 'img':
                copy_kv(full, r, b, img, FULL_LAYERS)
            elif part.startswith('img@'):
                copy_kv(full, r, b, img, (int(part[4:]),))
            elif part == 'post':
                copy_kv(full, r, b, post, FULL_LAYERS)
            elif part == 'q0':
                copy_kv(full, r, b, [q0], FULL_LAYERS)
            elif part == 'delta':
                copy_delta(lin, r, b)
            elif part == 'rec':
                copy_delta(lin, r, b, conv=False)
            elif part == 'conv':
                copy_delta(lin, r, b, recurrent=False)
            elif part == 'convlast':
                copy_delta(lin, r, b, recurrent=False, last_only=True)
            elif part.startswith('convlast@'):
                lo, hi = (int(v) for v in part[9:].split('-'))
                copy_delta(lin, r, b, recurrent=False, last_only=True, layers=range(lo, hi + 1))
            elif part not in ('A', 'B') and not part.startswith(('H11', 'L20')):
                raise ValueError(f'Unknown patch condition part {part!r}')
    if role == 'q0':
        emb = ia['x'][:, q0:q0 + 1].expand(R, -1, -1)                            # the last prefix token
        pos_t = ia['pos'][:, :, q0:q0 + 1]
    else:
        emb = digit_embeds(rt, [k] * R)
        pos_t = ia['pos'][:, :, q0 + 1:q0 + 2]
    l20, h11 = READER
    src = conds.index('img@20')
    ops = [(conds.index('H11<-img@20'), l20, h11, src), (conds.index('H11<-B'), l20, h11, b)]
    ops += [(conds.index('L20<-B'), l20, h, b) for h in range(df.N_HEADS)]
    with OprojEdit(rt, ops):
        lp = step(rt, cache, emb, pos_t)
    return {'conds': conds, 'logp': lp.cpu().numpy()}


@torch.inference_mode()
def run_tail_scan(rt, ua: df.Unit, ub: df.Unit, role: str, k: int) -> dict:
    """Where along the answer prefix is the image read? For every start position s from the last prompt
    token to the studied step, run B's image KV (all full-attention layers) is made visible to the
    positions >= s only (positions < s keep run A's states). Rows per s: A, B, edited A.
    Returns the start positions (relative to q0) and digit log-probabilities (S, 3, 10).
    """
    ia, ib = unit_inputs(rt, ua), unit_inputs(rt, ub)
    if ia['x'].shape != ib['x'].shape or ia['span'] != ib['span']:
        raise ValueError(f'Pair {ua.id}/{ub.id} differs in length or image span')
    q0, (s0, e0), L = ia['q0'], ia['span'], ia['L']
    end = q0 if role == 'q0' else q0 + 1                  # index of the studied step
    X = torch.cat([ia['x'], ib['x']], 0)
    if role == 'q1':
        X = torch.cat([X, digit_embeds(rt, [k, k]).to(X.dtype)], 1)
    starts, out = list(range(L - 1, end + 1)), []
    for st in starts:
        cache = prefill(rt, X[:, :st], ia['pos'], [0, 1, 0])
        full, _ = cache_layers(rt, cache)
        copy_kv(full, 2, 1, list(range(s0, e0)), FULL_LAYERS)
        tail = X[[0, 1, 0], st:end + 1]
        o = rt.model(inputs_embeds=tail, position_ids=ia['pos'][:, :, st:end + 1].expand(-1, 3, -1).contiguous(),
                     past_key_values=cache, use_cache=True)
        lp = o.logits[:, -1].float().log_softmax(-1)[:, torch.tensor(df.digit_ids(rt), device=o.logits.device)]
        out.append(lp.cpu().numpy())
    return {'starts': [st - q0 for st in starts], 'logp': np.stack(out)}


def stage_tail(rt, out: Path, args) -> None:
    path = out / 'tail.jsonl'
    done = {(r['role'], r['A'], r['B']) for r in read_jsonl(path)} if path.exists() else set()
    units = {u.id: u for u in dr.build_units(args) if u.family in ('sweep_h', 'sweep_v')}
    pairs = patch_pairs(list(units.values()))
    pairs = pairs[:args.limit] if args.limit else pairs
    t0 = time.time()
    for j, pr in enumerate(pairs):
        if (pr['role'], pr['A'], pr['B']) in done:
            continue
        ua, ub = units[pr['A']], units[pr['B']]
        res = run_tail_scan(rt, ua, ub, pr['role'], pr['k'])
        append_jsonl(path, [{**pr, 'xA': ua.truth, 'xB': ub.truth, 'axis': ua.axis, 'starts': res['starts'],
                             'logp': res['logp'].round(5).tolist()}])
        if j % 10 == 0:
            print(f'tail {j}/{len(pairs)} {time.time() - t0:.0f}s', flush=True)


def stage_patch(rt, out: Path, args) -> None:
    path = out / 'patch.jsonl'
    done = {(r['role'], r['A'], r['B']) for r in read_jsonl(path)} if path.exists() else set()
    units = {u.id: u for u in dr.build_units(args) if u.family in ('sweep_h', 'sweep_v')}
    pairs = patch_pairs(list(units.values()))
    pairs = pairs[:args.limit] if args.limit else pairs
    t0 = time.time()
    for j, pr in enumerate(pairs):
        if (pr['role'], pr['A'], pr['B']) in done:
            continue
        ua, ub = units[pr['A']], units[pr['B']]
        res = run_pair(rt, ua, ub, pr['role'], pr['k'])
        append_jsonl(path, [{**pr, 'xA': ua.truth, 'xB': ub.truth, 'axis': ua.axis, 'conds': res['conds'],
                             'logp': res['logp'].round(5).tolist()}])
        if j % 10 == 0:
            print(f'patch {j}/{len(pairs)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- stage: capture at q1

def prior_units(args) -> list[df.Unit]:
    units = dr.build_units(args)
    keep = [u for u in units if u.family in ('sweep_h', 'sweep_v', 'blank')]
    keep += [u for u in units if u.family == 'textconv']
    keep += [u for u in units if u.family == 'refcoco'][:args.n_refcoco]
    keep += [u for u in units if u.family == 'count'][::3]
    keep += [u for u in units if u.family == 'grid' and u.source['grid']['n'] == 10]
    return keep


class Capture:
    """Record, at the last position of a decoding step, MLP intermediate activations of PRIOR_MLP_LAYERS,
    the o_proj inputs of layer 32 and the input of the final norm."""

    def __init__(self, rt):
        self.store, self.handles = {}, []
        for l in PRIOR_MLP_LAYERS:
            self.handles.append(rt.layers[l - 1].mlp.down_proj.register_forward_pre_hook(
                lambda m, a, l=l: self.store.__setitem__(f'mlp{l}', a[0][:, -1].float().cpu())))
        self.handles.append(rt.layers[31].self_attn.o_proj.register_forward_pre_hook(
            lambda m, a: self.store.__setitem__('L32', a[0][:, -1].float().cpu())))
        self.handles.append(rt.model.model.language_model.norm.register_forward_pre_hook(
            lambda m, a: self.store.__setitem__('final', a[0][:, -1].float().cpu())))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def stage_capture(rt, out: Path, args) -> None:
    root = Path(args.features) / 'capture'
    root.mkdir(parents=True, exist_ok=True)
    meta_path = out / 'capture.jsonl'
    done = {r['unit'] for r in read_jsonl(meta_path)} if meta_path.exists() else set()
    t0 = time.time()
    units = prior_units(args)
    units = units[:args.limit] if args.limit else units
    for j, u in enumerate(units):
        if u.id in done:
            continue
        inp = unit_inputs(rt, u)
        ks = list(u.ks)
        cache = prefill(rt, inp['x'], inp['pos'], [0] * len(ks))
        with Capture(rt) as cap:
            lp = step(rt, cache, digit_embeds(rt, ks), inp['pos'][:, :, inp['q0'] + 1:inp['q0'] + 2])
        f = cap.store['final']
        rms = torch.sqrt(f.pow(2).mean(-1) + rt.model.model.language_model.norm.eps)
        np.savez(root / f'{u.id}.npz', **{n: cap.store[n].half().numpy() for n in cap.store if n != 'final'},
                 final=f.half().numpy(), rms=rms.numpy())
        append_jsonl(meta_path, [{'unit': u.id, 'family': u.family, 'axis': u.axis, 'truth': u.truth, 'ks': ks,
                                  'logp1': lp.cpu().numpy().round(5).tolist()}])
        if j % 25 == 0:
            print(f'capture {j}/{len(units)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- stage: ablation at q1

def load_ablation_plan(out: Path) -> dict:
    """Neuron sets and mean values written by digit_failure_r2_analysis.select_prior_neurons."""
    path = out / 'ablation_plan.npz'
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing; run the CPU selection (analysis stage "select") first')
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def ablation_conditions(plan: dict) -> list[tuple[str, dict, list]]:
    """(name, neuron edits {layer: (idx, values)}, head edits [(layer, head, mean)]) per condition."""
    def neurons(n: int) -> dict:
        sel = plan['order'][:n]                     # rows: (layer, neuron)
        ed = {}
        for l in PRIOR_MLP_LAYERS:
            idx = sel[sel[:, 0] == l, 1]
            if len(idx):
                ed[l] = (torch.tensor(idx), torch.tensor(plan[f'mean_mlp{l}'][idx]))
        return ed
    h8 = [(32, 8, torch.tensor(plan['mean_L32'][8]))]
    h10 = [(32, 10, torch.tensor(plan['mean_L32'][10]))]
    conds = [('none', {}, []), ('H8', {}, h8), ('H10', {}, h10)]
    for n in plan['n_values'].tolist():
        conds.append((f'N{n}', neurons(n), []))
        conds.append((f'N{n}+H8', neurons(n), h8))
    conds.append(('rand+H8', {l: (torch.tensor(plan[f'rand_mlp{l}']), torch.tensor(plan[f'mean_mlp{l}'][plan[f'rand_mlp{l}']]))
                              for l in PRIOR_MLP_LAYERS if len(plan[f'rand_mlp{l}'])}, h8))
    return conds


def stage_ablate(rt, out: Path, args) -> None:
    plan = load_ablation_plan(out)
    conds = ablation_conditions(plan)
    path = out / 'ablate.jsonl'
    done = {r['unit'] for r in read_jsonl(path)} if path.exists() else set()
    t0 = time.time()
    units = prior_units(args)
    units = units[:args.limit] if args.limit else units
    for j, u in enumerate(units):
        if u.id in done:
            continue
        inp = unit_inputs(rt, u)
        ks = list(u.ks)
        parts = []
        per = max(1, args.max_rows // len(ks))             # conditions per cached step (bounds DeltaNet state memory)
        for c0 in range(0, len(conds), per):
            sub = conds[c0:c0 + per]
            cache = prefill(rt, inp['x'], inp['pos'], [0] * (len(ks) * len(sub)))
            emb = digit_embeds(rt, ks * len(sub))
            handles, h_ops = [], []
            for c, (_, ned, hed) in enumerate(sub):
                rows = list(range(c * len(ks), (c + 1) * len(ks)))
                for l, (idx, vals) in ned.items():
                    handles.append(NeuronEdit(rt, {l: (rows, idx, vals)}))
                h_ops += [(r, l, h, m) for r in rows for (l, h, m) in hed]
            with OprojEdit(rt, h_ops):
                try:
                    parts.append(step(rt, cache, emb, inp['pos'][:, :, inp['q0'] + 1:inp['q0'] + 2]))
                finally:
                    for h in handles:
                        h.__exit__()
            del cache
        lp = torch.cat(parts, 0)
        lp = lp.cpu().numpy().reshape(len(conds), len(ks), 10)
        append_jsonl(path, [{'unit': u.id, 'family': u.family, 'axis': u.axis, 'truth': u.truth, 'ks': ks,
                             'conds': [c[0] for c in conds], 'logp1': lp.round(5).tolist()}])
        if j % 25 == 0:
            print(f'ablate {j}/{len(units)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- stage: steering q0

STEER_LAYERS = (24, 28)
STEER_SCALES = (-0.1, -0.05, 0.05, 0.1)          # added norm relative to the residual norm at q0


class AddAtPosition:
    """Add vectors to the output of decoder layers at one position during a prefill.

    adds: layer -> (R, D) tensor added to row r at the position (zeros leave a row unchanged).
    """

    def __init__(self, rt, adds: dict[int, torch.Tensor], position: int):
        self.handles = []
        for layer, vec in adds.items():
            def hook(module, a, o, vec=vec):
                h = o[0] if isinstance(o, tuple) else o
                h = h.clone()
                h[:, position] += vec.to(h.dtype).to(h.device)
                return (h, *o[1:]) if isinstance(o, tuple) else h
            self.handles.append(rt.layers[layer - 1].register_forward_hook(hook))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def run_steer(rt, u: df.Unit, W_dir: torch.Tensor, rand: torch.Tensor) -> dict:
    """Steer the q0 residual along the first-digit neighbour-margin direction and read q1.

    Direction (unit norm, residual space): gain * (W_U[k+1] - W_U[k-1]) for the true first digit k, i.e. the
    direction that raises the first-digit logit of the upper neighbour against the lower one. Control: a random
    unit vector orthogonal to it. Rows: base, then layer x direction x scale. Returns q0 and q1 digit log-probs.
    """
    inp = unit_inputs(rt, u)
    k = df.hundreds(u.truth)
    q0 = inp['q0']
    v = W_dir[k + 1] - W_dir[k - 1]
    v = v / v.norm()
    r = rand - (rand @ v) * v
    r = r / r.norm()
    rows = [('base', 0, '', 0.0)] + [(f'{d}@{l}{c:+.2f}', l, d, c) for l in STEER_LAYERS for d in ('margin', 'random')
                                     for c in STEER_SCALES]
    R = len(rows)
    # Residual norm at q0 of each steering layer (from an unsteered pass) sets the scale.
    store = {}
    hs = [rt.layers[l - 1].register_forward_hook(lambda m, a, o, l=l: store.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0, q0].float().norm()))
          for l in STEER_LAYERS]
    try:
        rt.model(inputs_embeds=inp['x'], position_ids=inp['pos'][:, :, :q0 + 1], use_cache=False, logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
    adds = {l: torch.zeros(R, v.shape[0], device=v.device) for l in STEER_LAYERS}
    for i, (_, l, d, c) in enumerate(rows):
        if l:
            adds[l][i] = c * store[l].to(v.device) * (v if d == 'margin' else r)
    X = inp['x'].expand(R, -1, -1)
    with AddAtPosition(rt, adds, q0):
        o = rt.model(inputs_embeds=X, position_ids=inp['pos'][:, :, :q0 + 1].expand(-1, R, -1).contiguous(),
                     use_cache=True, logits_to_keep=1)
    dig = torch.tensor(df.digit_ids(rt), device=o.logits.device)
    lp0 = o.logits[:, -1].float().log_softmax(-1)[:, dig]
    lp1 = step(rt, o.past_key_values, digit_embeds(rt, [k] * R), inp['pos'][:, :, q0 + 1:q0 + 2])
    return {'rows': [x[0] for x in rows], 'k': k, 'logp0': lp0.cpu().numpy(), 'logp1': lp1.cpu().numpy()}


def stage_steer(rt, out: Path, args) -> None:
    path = out / 'steer.jsonl'
    done = {r['unit'] for r in read_jsonl(path)} if path.exists() else set()
    w = np.load(out / 'weights.npz')                   # written by stage 'weights' (works with offloaded heads)
    W_dir = torch.tensor(w['W_U'] * w['norm_gain'][None], dtype=torch.float32, device=rt.model.device)
    gen = torch.Generator().manual_seed(0)
    rand = torch.randn(W_dir.shape[1], generator=gen).to(rt.model.device)
    units = [u for u in dr.build_units(args) if u.family in ('sweep_h', 'sweep_v') and 150 <= u.truth < 850]
    units = units[:args.limit] if args.limit else units
    t0 = time.time()
    for j, u in enumerate(units):
        if u.id in done:
            continue
        res = run_steer(rt, u, W_dir, rand)
        append_jsonl(path, [{'unit': u.id, 'family': u.family, 'axis': u.axis, 'truth': u.truth, 'k': res['k'],
                             'rows': res['rows'], 'logp0': res['logp0'].round(5).tolist(),
                             'logp1': res['logp1'].round(5).tolist()}])
        if j % 20 == 0:
            print(f'steer {j}/{len(units)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- main

STAGES = {'weights': stage_weights, 'attn': stage_attn, 'patch': stage_patch, 'tail': stage_tail, 'capture': stage_capture,
          'ablate': stage_ablate, 'steer': stage_steer}


def code_digest() -> dict:
    root = Path(__file__).parent
    names = ('digit_failure_r2.py', 'digit_failure.py', 'digit_failure_run.py')
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()[:16] for n in names}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--refcoco-cases', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--n-refcoco', type=int, default=40)
    p.add_argument('--max-memory', default='', help='optional weight split, e.g. "0:11GiB,1:10GiB" (visible indices)')
    p.add_argument('--max-rows', type=int, default=27, help='rows per cached decoding step in the ablate stage')
    p.add_argument('--limit', type=int, default=0, help='smoke test: process only this many units / pairs per stage')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rt = dr.load_split(args.model, args.max_memory) if args.max_memory else qr.load(args.model)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages,
                         'runtime': qr.describe(rt), 'code': code_digest(), 'n_refcoco': args.n_refcoco,
                         'max_memory': args.max_memory,
                         'scheme': 'prefill-then-step, batch = rows of one unit or pair'})
    meta.write_text(json.dumps(info, indent=1))
    for s in args.stages.split(','):
        t0 = time.time()
        STAGES[s](rt, out, args)
        print(f'stage {s} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
