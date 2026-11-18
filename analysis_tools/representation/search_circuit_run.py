"""Experiment 2.7 round 3 GPU stages (resumable; outputs under --output).

  behave    greedy direct answers for every odd image (generate_batch, left-padded); results_behave.jsonl
  spotcheck 20 odd images re-decoded at batch size 1; spotcheck.json
  trace     one teacher-forced prefill per image (odd and corrupted) with an instrumented forward:
              * every Gated DeltaNet layer is run by an exact token-by-token recurrence that records, per image
                token and value head, the delta-rule prediction error e_t = v_t - (alpha_t S_{t-1})^T k_t and the
                prediction itself; for the target cell it also records how the prediction decomposes over the
                raster-earlier cells (unrolled delta rule, see search_circuit.prediction_sources);
              * reader heads: attention logits of the answer queries (the positions that predict the row and the
                column digit) on every image token, decomposed exactly into the contributions of the decoder input
                (merger output) and of every earlier mixer and MLP output (RMSNorm scales frozen at their actual
                values, so the decomposition is additive and sums to the actual logit);
              * attention probabilities of all full-attention heads at the two answer queries, per cell;
              * image-query attention summaries (first image token, text prefix, own cell, earlier cells) at the
                full-attention layers 4-16 and residual norms of image tokens.
            trace/<set>.npz
  knock     teacher-forced score and reader attention under causal interventions on single images:
              dn_off*   alpha_t = 0 at all image tokens in Gated DeltaNet layers (no memory of earlier tokens)
              fa_off*   image-token queries of full-attention layers cannot attend to image tokens of other cells
              prime_*   the DeltaNet states at the first image token are replaced by the states reached at the end
                        of a donor image (same-majority homogeneous image, another pair's, or a blank image)
            knock.jsonl
  patch     noising at the target tokens: one component's output (decoder input, mixer l, MLP l, or a band) is
            replaced by its value in the corrupted image; patch.jsonl

Teacher forcing uses the answer text '{"row": r, "column": c}'.  S = log p(row digit) + log p(column digit).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import time
import types
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_run as sr
from analysis_tools.representation import search_circuit as sc

FULL_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
READERS = ((16, 6), (16, 11), (16, 2), (20, 2), (20, 11), (20, 10))
N_COMP = 39                                   # decoder input + 19 mixers + 19 MLPs (enough for layer 20)


# ----------------------------------------------------------------------------- items

class Items:
    def __init__(self, output: Path):
        self.root = output
        self.items = {it['id']: it for it in read_jsonl(output / 'items.jsonl')} if (output / 'items.jsonl').exists() else {}

    def image(self, item_id: str) -> Image.Image:
        return Image.open(self.root / self.items[item_id]['path']).convert('RGB')

    def odd(self, sets=None):
        return [it for it in self.items.values() if it['role'] == 'odd' and (sets is None or it['set'] in sets)]


def _append(path: Path, rows: list[dict]) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _done(path: Path, key='key') -> set:
    return {r[key] for r in read_jsonl(path)} if path.exists() else set()


# ----------------------------------------------------------------------------- behaviour

def stage_behave(rt, args, items: Items, output: Path) -> None:
    path = output / 'results_behave.jsonl'
    done = _done(path, 'id')
    todo = [it for it in items.odd() if it['id'] not in done]
    for a in range(0, len(todo), args.batch_direct):
        chunk = todo[a:a + args.batch_direct]
        outs = qr.generate_batch(rt, [([items.image(it['id'])], sa.DIRECT_PROMPT) for it in chunk], 24)
        rows = []
        for it, o in zip(chunk, outs):
            rows.append({'id': it['id'], **o, **sa.score(it, o), 'batch_size': args.batch_direct})
        _append(path, rows)
        print(f'[behave] {a + len(chunk)}/{len(todo)}', flush=True)


def stage_spotcheck(rt, args, items: Items, output: Path) -> None:
    beh = {r['id']: r for r in read_jsonl(output / 'results_behave.jsonl')}
    rng = np.random.default_rng(args.seed)
    ids = sorted(beh)
    pick = [ids[i] for i in rng.choice(len(ids), size=min(args.spotcheck, len(ids)), replace=False)]
    rows = []
    for i in pick:
        g = qr.generate(rt, qr.build(rt, [items.image(i)], sa.DIRECT_PROMPT), 24)
        rows.append({'id': i, 'batch': beh[i]['text'], 'single': g['text'],
                     'same_answer': oe.parse_answer(g['text']) == oe.parse_answer(beh[i]['text'])})
    (output / 'spotcheck.json').write_text(json.dumps(
        {'n': len(rows), 'same_answer': sum(r['same_answer'] for r in rows), 'rows': rows}, indent=1))
    print(f'[spotcheck] {sum(r["same_answer"] for r in rows)}/{len(rows)}', flush=True)


# ----------------------------------------------------------------------------- instrumented Gated DeltaNet

class DNControl:
    """Per-forward control of the replaced Gated DeltaNet forward.

    mode: 'native' (library kernel on the whole sequence), 'trace' (exact recurrence with records),
          'split' (library kernel in segments [0, s), [s, L); optional state injection at s and state capture at e).
    """

    def __init__(self):
        self.mode = 'native'
        self.s = self.e = None
        self.init: dict[int, torch.Tensor] = {}       # layer -> (B, H, dk, dv) state injected at s
        self.keep_state = False
        self.states: dict[int, torch.Tensor] = {}     # layer -> state after token e - 1
        self.records: dict[int, dict] = {}
        self.target_tok: list[torch.Tensor] | None = None   # per row, absolute positions of the target tokens


def _conv(mod, x: torch.Tensor) -> torch.Tensor:
    """Causal depthwise convolution + SiLU, identical to the library fallback.  x: (B, C, L)."""
    L = x.shape[-1]
    return F.silu(F.conv1d(x, mod.conv1d.weight, None, padding=mod.conv_kernel_size - 1, groups=x.shape[1])[..., :L])


def _l2(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def dn_forward_factory(ctrl: DNControl, layer: int, original):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if cache_params is not None or ctrl.mode == 'native':
            return original(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
        B, L, _ = hidden_states.shape
        qkv = _conv(self, self.in_proj_qkv(hidden_states).transpose(1, 2)).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(B, L, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, L, -1, self.head_k_dim)
        k = k.reshape(B, L, -1, self.head_k_dim)
        v = v.reshape(B, L, -1, self.head_v_dim)
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        if ctrl.mode == 'trace':
            out = _trace_recurrence(ctrl, layer, q, k, v, g, beta)
        else:
            s, e = ctrl.s, ctrl.e
            kw = dict(use_qk_l2norm_in_kernel=True)
            o1, st = mq.torch_chunk_gated_delta_rule(q[:, :s], k[:, :s], v[:, :s], g=g[:, :s], beta=beta[:, :s],
                                                     initial_state=None, output_final_state=True, **kw)
            if layer in ctrl.init:
                donor, sel = ctrl.init[layer]
                st = torch.where(sel.to(st.device)[:, None, None, None], donor.to(st), st)
            if ctrl.keep_state:
                o2, st2 = mq.torch_chunk_gated_delta_rule(q[:, s:e], k[:, s:e], v[:, s:e], g=g[:, s:e],
                                                          beta=beta[:, s:e], initial_state=st,
                                                          output_final_state=True, **kw)
                ctrl.states[layer] = st2.detach().float()
                o3, _ = mq.torch_chunk_gated_delta_rule(q[:, e:], k[:, e:], v[:, e:], g=g[:, e:], beta=beta[:, e:],
                                                        initial_state=st2, output_final_state=False, **kw)
                out = torch.cat([o1, o2, o3], 1)
            else:
                o2, _ = mq.torch_chunk_gated_delta_rule(q[:, s:], k[:, s:], v[:, s:], g=g[:, s:], beta=beta[:, s:],
                                                        initial_state=st, output_final_state=False, **kw)
                out = torch.cat([o1, o2], 1)
        out = out.to(hidden_states.dtype).reshape(-1, self.head_v_dim)
        out = self.norm(out, z.reshape(-1, self.head_v_dim)).reshape(B, L, -1)
        return self.out_proj(out)
    return forward


def _trace_recurrence(ctrl: DNControl, layer: int, q, k, v, g, beta) -> torch.Tensor:
    """Exact gated delta rule (float32), recording prediction errors at image tokens.

    Records (per layer): err_norm, v_norm, pred_dot (B, H, n_img) and, for the target tokens, the unrolled
    prediction weights over all earlier tokens folded into an error-weighted contribution per earlier position:
    contrib[b, h, j] = w_tj (e_j . pred_t) / |pred_t|^2 averaged over the target tokens (sums to 1 over j < t when the
    initial state is zero).
    """
    B, L, H, dk = k.shape
    q = _l2(q.float()) / dk ** 0.5
    k = _l2(k.float())
    v = v.float()
    beta = beta.float()
    S = torch.zeros(B, H, dk, v.shape[-1], device=v.device)
    out = torch.empty_like(v)
    s, e = ctrl.s, ctrl.e
    errs = torch.empty(B, L, H, v.shape[-1], device=v.device)
    preds = torch.empty_like(errs)
    for t in range(L):
        S = S * g[:, t].exp()[..., None, None]
        p = torch.einsum('bhkv,bhk->bhv', S, k[:, t])
        d = v[:, t] - p
        S = S + torch.einsum('bhk,bhv->bhkv', k[:, t], d * beta[:, t, :, None])
        out[:, t] = torch.einsum('bhkv,bhk->bhv', S, q[:, t])
        errs[:, t], preds[:, t] = d, p
    rec = {'err_norm': errs[:, s:e].norm(dim=-1).transpose(1, 2), 'v_norm': v[:, s:e].norm(dim=-1).transpose(1, 2),
           'pred_dot': (preds[:, s:e] * v[:, s:e]).sum(-1).transpose(1, 2)}
    if ctrl.target_tok is not None:
        G = torch.cumsum(g.float(), 1)                                  # (B, L, H)
        contrib = torch.zeros(B, H, L, device=v.device)
        for b, toks in enumerate(ctrl.target_tok):
            for t in toks.tolist():
                w = (G[b, t][None] - G[b, :t]).exp() * beta[b, :t] * torch.einsum('jhk,hk->jh', k[b, :t], k[b, t])
                pt = preds[b, t]                                        # (H, dv)
                proj = torch.einsum('jhv,hv->jh', errs[b, :t], pt) / (pt.pow(2).sum(-1)[None] + 1e-9)
                contrib[b, :, :t] += (w * proj).T / len(toks)
        rec['contrib'] = contrib
    ctrl.records[layer] = rec
    return out


@contextmanager
def instrumented(rt, ctrl: DNControl):
    """Replace the forward of every Gated DeltaNet module for the duration of the context."""
    saved = []
    for i, block in enumerate(rt.layers):
        if block.block_type == 'linear_attention':
            mod = block.linear_attn
            orig = mod.forward
            mod.forward = types.MethodType(dn_forward_factory(ctrl, i + 1, orig), mod)
            saved.append((mod, orig))
    try:
        yield ctrl
    finally:
        for mod, orig in saved:
            mod.forward = orig


# ----------------------------------------------------------------------------- batches and hooks

def batch_for(rt, items: Items, ids: list[str], answers_rc: list[tuple[int, int]]):
    answers, i_row, i_col = [], None, None
    for rc in answers_rc:
        a, i_row, i_col = sr.answer_ids(rt, rc)
        answers.append(a)
    batch = sr.build_rows(rt, [items.image(i) for i in ids], answers)
    return batch, answers, i_row, i_col


def repeat_batch(batch, n: int):
    """Repeat a one-row teacher-forcing batch n times without re-running the image processor."""
    out = {}
    for k, v in batch.items():
        out[k] = v.repeat(n, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v
    return type(batch)(out) if not isinstance(batch, dict) else out


def positions(rt, batch, n_ans: int, i_row: int, i_col: int) -> dict:
    s, e = sr.image_span(rt, batch)
    L = batch['input_ids'].shape[1]
    return {'s': s, 'e': e, 'L': L, 'q_row': L - n_ans + i_row - 1, 'q_col': L - n_ans + i_col - 1}


def cell_tokens(meta: dict, s: int) -> list[list[int]]:
    return [[s + t for t in toks] for toks in sc.entity_groups(meta)]


@contextmanager
def capture_layers(rt, inputs_at: tuple[int, ...], comps_upto: int, img: slice | None):
    """Capture decoder-layer inputs (all positions) and position embeddings at `inputs_at`, and the decoder input,
    mixer and MLP outputs of layers 1..comps_upto at the image positions (bf16)."""
    store = {'inp': {}, 'pe': {}, 'comp': {}}
    hs = []
    for layer in inputs_at:
        def pre(m, args, kwargs, layer=layer):
            store['inp'][layer] = args[0].detach()
            store['pe'][layer] = kwargs['position_embeddings']
        hs.append(rt.layers[layer - 1].register_forward_pre_hook(pre, with_kwargs=True))
    if img is not None:
        def pre0(m, args, kwargs):
            store['comp'][0] = args[0][:, img].detach()
        hs.append(rt.layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
        for layer in range(1, comps_upto + 1):
            block = rt.layers[layer - 1]
            mix = block.linear_attn if block.block_type == 'linear_attention' else block.self_attn

            def hm(m, a, o, layer=layer):
                store['comp'][2 * layer - 1] = sr._out(o)[:, img].detach()

            def hp(m, a, o, layer=layer):
                store['comp'][2 * layer] = o[:, img].detach()
            hs.append(mix.register_forward_hook(hm))
            hs.append(block.mlp.register_forward_hook(hp))
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


def _rope(x, cos, sin):
    """Rotary embedding on the last dim of x (..., hd) with cos/sin (..., rot) broadcast; partial rotary aware."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import rotate_half
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    return torch.cat([xr * cos + rotate_half(xr) * sin, xp], -1)


def query_vectors(rt, layer: int, x: torch.Tensor, cos, sin, qpos: list[int]) -> torch.Tensor:
    """Post-norm, post-RoPE queries of all heads at positions qpos.  x: (L, d) layer input.  Returns (nq, H, hd)."""
    blk = rt.layers[layer - 1]
    att = blk.self_attn
    hd = att.head_dim
    xn = blk.input_layernorm(x[qpos])
    qg = att.q_proj(xn).view(len(qpos), -1, hd * 2)
    qh = att.q_norm(qg[..., :hd]).float()
    return _rope(qh, cos[qpos][:, None].float(), sin[qpos][:, None].float())


def key_vectors(rt, layer: int, x: torch.Tensor, cos, sin) -> torch.Tensor:
    blk = rt.layers[layer - 1]
    att = blk.self_attn
    hd = att.head_dim
    k = att.k_norm(att.k_proj(blk.input_layernorm(x)).view(x.shape[0], -1, hd)).float()
    return _rope(k, cos[:, None].float(), sin[:, None].float())                 # (L, KV, hd)


def attention_probs(rt, layer: int, x, cos, sin, qpos: list[int]) -> torch.Tensor:
    """Causal attention probabilities (H, nq, L) recomputed from the captured layer input."""
    att = rt.layers[layer - 1].self_attn
    q = query_vectors(rt, layer, x, cos, sin, qpos)
    k = key_vectors(rt, layer, x, cos, sin)
    k = k.repeat_interleave(att.num_key_value_groups, dim=1)                    # (L, H, hd)
    sc_ = torch.einsum('qhd,khd->hqk', q, k) * att.scaling
    L = x.shape[0]
    mask = torch.arange(L, device=x.device)[None, :] > torch.as_tensor(qpos, device=x.device)[:, None]
    return sc_.masked_fill(mask[None], float('-inf')).softmax(-1)


def logit_attribution(rt, layer: int, x: torch.Tensor, cos, sin, qpos: list[int], comps: torch.Tensor,
                      img: slice, heads: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact additive decomposition of reader-head attention logits on image tokens.

    x: (L, d) input of `layer`; comps: (n_c, n_img, d) the decoder input and the mixer/MLP outputs of earlier layers
    at image tokens (their sum is x[img]).  The two RMSNorms on the key path (input_layernorm, k_norm) are linear once
    their per-token scales are frozen at the actual values; RoPE is linear.  Returns
    contributions (len(heads), nq, n_c, n_img) and the actual logits (len(heads), nq, n_img).
    """
    blk = rt.layers[layer - 1]
    att = blk.self_attn
    hd = att.head_dim
    xi = x[img].float()
    eps_in = blk.input_layernorm.eps
    r_in = torch.rsqrt(xi.pow(2).mean(-1, keepdim=True) + eps_in)                # (n_img, 1)
    w_in = 1.0 + blk.input_layernorm.weight.float()
    Wk = att.k_proj.weight.float()                                                # (KV*hd, d)
    kc = (comps.float() * r_in[None] * w_in) @ Wk.T                               # (n_c, n_img, KV*hd)
    kc = kc.view(kc.shape[0], kc.shape[1], -1, hd)
    ktot = kc.sum(0)
    r_k = torch.rsqrt(ktot.pow(2).mean(-1, keepdim=True) + att.k_norm.eps)        # (n_img, KV, 1)
    kc = kc * r_k[None] * (1.0 + att.k_norm.weight.float())
    c_img, s_img = cos[img][:, None].float(), sin[img][:, None].float()
    kc = _rope(kc, c_img[None], s_img[None])                                      # (n_c, n_img, KV, hd)
    q = query_vectors(rt, layer, x, cos, sin, qpos)                               # (nq, H, hd)
    kv_of = [h // att.num_key_value_groups for h in heads]
    contrib = torch.stack([torch.einsum('qd,cnd->qcn', q[:, h], kc[:, :, kv]) for h, kv in zip(heads, kv_of)]) * att.scaling
    kact = key_vectors(rt, layer, x, cos, sin)[img]                               # (n_img, KV, hd)
    actual = torch.stack([torch.einsum('qd,nd->qn', q[:, h], kact[:, kv]) for h, kv in zip(heads, kv_of)]) * att.scaling
    return contrib, actual


def cell_mean(x: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
    """Average the last dim (image-token index) within each cell.  Returns (..., n_cells)."""
    return torch.stack([x[..., g].mean(-1) for g in groups], -1)


def cell_sum(x: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
    return torch.stack([x[..., g].sum(-1) for g in groups], -1)


# ----------------------------------------------------------------------------- trace

@torch.inference_mode()
def stage_trace(rt, args, items: Items, output: Path) -> None:
    out_dir = output / 'trace'
    out_dir.mkdir(exist_ok=True)
    ctrl = DNControl()
    by_set = defaultdict(list)
    for it in items.items.values():
        by_set[it['set']].append(it)
    for set_name in args.trace_sets:
        path = out_dir / f'{set_name}.npz'
        if path.exists():
            print(f'[trace] {set_name} exists', flush=True)
            continue
        its = sorted(by_set[set_name], key=lambda it: it['id'])
        if args.trace_limit:
            its = its[:args.trace_limit]
        recs = defaultdict(list)
        t0 = time.time()
        for a in range(0, len(its), args.batch_trace):
            chunk = its[a:a + args.batch_trace]
            ids = [it['id'] for it in chunk]
            batch, answers, i_row, i_col = batch_for(rt, items, ids, [tuple(it['target']) for it in chunk])
            P = positions(rt, batch, len(answers[0]), i_row, i_col)
            groups0 = sc.entity_groups(chunk[0]['meta'])
            groups = [[P['s'] + t for t in g] for g in groups0]
            img = slice(P['s'], P['e'])
            ctrl.mode, ctrl.s, ctrl.e, ctrl.records = 'trace', P['s'], P['e'], {}
            tcell = [it['target'][0] * it['cols'] + it['target'][1] for it in chunk]
            ctrl.target_tok = [torch.as_tensor(groups[c]) for c in tcell]
            with instrumented(rt, ctrl), capture_layers(rt, FULL_LAYERS[:5], 19, img) as st:
                logits = rt.model(**batch, use_cache=False, logits_to_keep=len(answers[0]) + 1).logits
            ctrl.mode = 'native'
            scores = sr.score_rows(logits, answers, i_row, i_col, len(answers[0]), sr.digit_ids(rt, 6))
            comps = torch.stack([st['comp'][c] for c in range(N_COMP)], 1)            # (B, n_c, n_img, d)
            qpos = [P['q_row'], P['q_col']]
            for b, it in enumerate(chunk):
                recs['id'].append(it['id'])
                recs['score'].append([scores[b]['s'], scores[b]['s_row'], scores[b]['s_col']])
                recs['argmax'].append([scores[b]['arg_row'], scores[b]['arg_col']])
                # attention probabilities of every full-attention head at the two answer queries, per cell
                att = []
                for layer in FULL_LAYERS[:5]:
                    cos, sin = st['pe'][layer]
                    pr = attention_probs(rt, layer, st['inp'][layer][b], cos[b], sin[b], qpos)
                    att.append(cell_sum(pr, groups))
                recs['ans_attn'].append(torch.stack(att).half().cpu().numpy())   # (5, H, 2, n_cells)
                # reader-logit attribution
                contrib_all, actual_all = [], []
                for layer in (16, 20):
                    heads = [h for l, h in READERS if l == layer]
                    cos, sin = st['pe'][layer]
                    n_c = 1 + 2 * (layer - 1)
                    c, act = logit_attribution(rt, layer, st['inp'][layer][b], cos[b], sin[b], qpos,
                                               comps[b, :n_c], img, heads)
                    pad = torch.full((c.shape[0], c.shape[1], N_COMP - n_c, c.shape[3]), float('nan'), device=c.device)
                    contrib_all.append(torch.cat([c, pad], 2))
                    actual_all.append(act)
                g_img = [[t - P['s'] for t in g] for g in groups]
                recs['logit_contrib'].append(cell_mean(torch.cat(contrib_all), g_img).half().cpu().numpy())
                recs['logit'].append(cell_mean(torch.cat(actual_all), g_img).float().cpu().numpy())
                # image-query attention summaries at FA layers 4-16
                summ = []
                for layer in FULL_LAYERS[:4]:
                    cos, sin = st['pe'][layer]
                    qimg = list(range(P['s'], P['e']))
                    pr = attention_probs(rt, layer, st['inp'][layer][b], cos[b], sin[b], qimg)  # (H, n_img, L)
                    first = pr[:, :, P['s']].mean(1)
                    prefix = pr[:, :, :P['s']].sum(-1).mean(1)
                    own = torch.zeros_like(first)
                    earlier = torch.zeros_like(first)
                    for ci, gi in enumerate(groups):
                        qi = [t - P['s'] for t in gi]
                        own += pr[:, qi][:, :, gi].sum(-1).sum(-1)
                        prev_tok = [t for g2 in groups[:ci] for t in g2]
                        if prev_tok:
                            earlier += pr[:, qi][:, :, prev_tok].sum(-1).sum(-1)
                    n = P['e'] - P['s']
                    summ.append(torch.stack([first, prefix, own / n, earlier / n]))
                recs['img_attn'].append(torch.stack(summ).half().cpu().numpy())     # (4, 4, H)
                norms = torch.stack([st['inp'][l][b, img].float().norm(dim=-1) for l in FULL_LAYERS[:5]])
                recs['res_norm'].append(norms.half().cpu().numpy())                  # (5, n_img)
                dn = sorted(ctrl.records)
                rel = torch.stack([ctrl.records[l]['err_norm'][b] / (ctrl.records[l]['v_norm'][b] + 1e-6) for l in dn])
                recs['dn_rel_err'].append(cell_mean(rel, g_img).half().cpu().numpy())   # (24, H, n_cells)
                cosp = torch.stack([ctrl.records[l]['pred_dot'][b] / (ctrl.records[l]['v_norm'][b] ** 2 + 1e-6) for l in dn])
                recs['dn_pred_v'].append(cell_mean(cosp, g_img).half().cpu().numpy())    # (24, H, n_cells)
                con = torch.stack([ctrl.records[l]['contrib'][b] for l in dn])             # (24, H, L)
                cells = cell_sum(con, groups)
                pre = con[..., :P['s']].sum(-1, keepdim=True)
                recs['dn_contrib'].append(torch.cat([cells, pre], -1).half().cpu().numpy())  # (24, H, n_cells+1)
            print(f'[trace] {set_name} {a + len(chunk)}/{len(its)} {time.time() - t0:.0f}s', flush=True)
        ctrl.target_tok = None
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in recs.items()})
        print(f'[trace] wrote {path}', flush=True)


# ----------------------------------------------------------------------------- knock / prime

DN_LAYERS_BELOW_16 = [l for l in range(1, 16) if l not in FULL_LAYERS]
KNOCKS = {
    'none': {},
    'dn_off': {'dn': DN_LAYERS_BELOW_16},
    'dn_off_1_7': {'dn': [1, 2, 3, 5, 6, 7]},
    'dn_off_9_15': {'dn': [9, 10, 11, 13, 14, 15]},
    'fa_off': {'fa': [4, 8, 12, 16]},
    'fa_off_4': {'fa': [4]},
    'fa_off_8': {'fa': [8]},
    'fa_off_12_16': {'fa': [12, 16]},
    'both_off': {'dn': DN_LAYERS_BELOW_16, 'fa': [4, 8, 12, 16]},
    'prime_same': {'prime': 'same'},
    'prime_other': {'prime': 'other'},
    'prime_blank': {'prime': 'blank'},
}
# local knockouts: only the target's tokens (or those of a control distractor, raster index t + 7 mod 36)
KNOCKS.update({
    'fa8_tgt': {'fa': [8], 'scope': 'target'}, 'fa_tgt': {'fa': [4, 8, 12, 16], 'scope': 'target'},
    'dn17_tgt': {'dn': [1, 2, 3, 5, 6, 7], 'scope': 'target'}, 'dn_tgt': {'dn': DN_LAYERS_BELOW_16, 'scope': 'target'},
    'fa_dst': {'fa': [4, 8, 12, 16], 'scope': 'distractor'}, 'dn_dst': {'dn': DN_LAYERS_BELOW_16, 'scope': 'distractor'},
})
LOCAL_KNOCKS = ('fa8_tgt', 'fa_tgt', 'dn17_tgt', 'dn_tgt', 'fa_dst', 'dn_dst')
PREV_KNOCKS = ('none', 'dn_off', 'dn_off_1_7', 'dn_off_9_15', 'fa_off', 'prime_same', 'prime_blank')
MAG_KNOCKS = ('none', 'dn_off_1_7', 'fa_off_8', 'fa8_tgt', 'dn17_tgt')
HOM_KNOCKS = ('none', 'prime_same', 'prime_blank')


def reader_readout(rt, st: dict, b: int, groups: list[list[int]], qpos: int) -> np.ndarray:
    """Attention mass of the six reader heads from the row-digit query on every cell.  Returns (6, n_cells)."""
    out = []
    for layer in (16, 20):
        cos, sin = st['pe'][layer]
        pr = attention_probs(rt, layer, st['inp'][layer][b], cos[b], sin[b], [qpos])[:, 0]
        for l, h in READERS:
            if l == layer:
                out.append(cell_sum(pr[h], groups))
    return torch.stack(out).float().cpu().numpy()


def donor_ids(items: Items, it: dict, kind: str, blank_id: str) -> str:
    """One fixed donor per (pair, majority entity): the first atlas homogeneous image of that kind (the item's own
    corrupted partner is skipped), so that donor states can be cached."""
    if kind == 'blank':
        return blank_id
    hom = 'homA' if it['odd_variant'] == 'swap' else 'homB'
    pair = it['pair_name'] if kind == 'same' else ('tilt' if it['pair_name'] != 'tilt' else 'qo')
    pool = sorted((x for x in items.items.values() if x['role'] == 'corrupted' and x['set'] == 'atlas'
                   and x['pair_name'] == pair and x['level'] == 'subtle'
                   and (x['variant'] == hom or kind == 'other') and x['id'] != it.get('partner')), key=lambda x: x['id'])
    if not pool:
        raise ValueError(f'no donor of kind {kind} for {it["id"]}')
    return pool[0]['id']


def blank_image_item(items: Items) -> str:
    """A white 6 x 6-sized image used as the 'blank' donor (written once under images/)."""
    rel = 'images/blank_canvas.png'
    key = 'images:blank_canvas'
    if key not in items.items:
        any_it = next(iter(items.items.values()))
        w, h = any_it['meta']['canvas']
        Image.new('RGB', (w, h), (255, 255, 255)).save(items.root / rel)
        items.items[key] = {'id': key, 'path': rel, 'meta': any_it['meta'], 'role': 'donor', 'set': 'donor',
                            'target': [0, 0], 'cols': 6}
    return key


@torch.inference_mode()
def donor_states(rt, items: Items, ids: list[str], ctrl: DNControl) -> dict[int, torch.Tensor]:
    batch, answers, i_row, i_col = batch_for(rt, items, ids, [(0, 0)] * len(ids))
    P = positions(rt, batch, len(answers[0]), i_row, i_col)
    ctrl.mode, ctrl.s, ctrl.e, ctrl.init, ctrl.keep_state, ctrl.states = 'split', P['s'], P['e'], {}, True, {}
    with instrumented(rt, ctrl):
        rt.model(**batch, use_cache=False, logits_to_keep=1)
    ctrl.keep_state, ctrl.mode = False, 'native'
    return dict(ctrl.states)


@torch.inference_mode()
def stage_knock(rt, args, items: Items, output: Path) -> None:
    from analysis_tools.position.rowcol_mechanism_run import override_gates_heads
    path = output / 'knock.jsonl'
    done = _done(path)
    ctrl = DNControl()
    blank_id = blank_image_item(items)
    def conds_for(it):
        if it['role'] == 'odd' and it['set'] == 'atlas':
            return args.knocks
        if it['role'] == 'odd' and it['set'] == 'mag':
            return [c for c in args.knocks if c in MAG_KNOCKS]
        if it['role'] == 'odd' and it['set'] == 'prev':
            return [c for c in args.knocks if c in PREV_KNOCKS]
        return [c for c in args.knocks if c in HOM_KNOCKS]
    targets = [it for it in items.items.values() if it['set'] in args.knock_sets
               and ((it['role'] == 'odd' and it['set'] != 'shift') or (it['set'] in ('shift', 'prev') and it['role'] == 'corrupted'))]
    if args.knock_limit:
        keep = defaultdict(list)
        for it in sorted(targets, key=lambda x: x['id']):
            keep[(it['set'], it['pair_name'], it['level'], it['variant'], it.get('n_prev'), it.get('k_blank'))].append(it)
        targets = [x for v in keep.values() for x in v[:args.knock_limit]]
    # item chunks share one processed batch; every condition runs on the whole chunk (homogeneous intervention)
    by_conds = defaultdict(list)
    for it in sorted(targets, key=lambda x: x['id']):
        cs = [c for c in conds_for(it) if f'{it["id"]}|{c}' not in done]
        if cs:
            by_conds[tuple(cs)].append(it)
    plan = [(chunk_items, cs) for cs, its in by_conds.items()
            for chunk_items in (its[i:i + args.batch_knock] for i in range(0, len(its), args.batch_knock))]
    n_jobs = sum(len(c) * len(cs) for c, cs in plan)
    cache: dict[str, dict[int, torch.Tensor]] = {}
    t0, n_done = time.time(), 0
    for chunk_items, cs in plan:
      base, answers, i_row, i_col = batch_for(rt, items, [it['id'] for it in chunk_items],
                                              [tuple(it['target']) for it in chunk_items])
      for cond in cs:
        chunk = [(it, cond) for it in chunk_items]
        batch = base
        P = positions(rt, batch, len(answers[0]), i_row, i_col)
        groups = cell_tokens(chunk[0][0]['meta'], P['s'])
        img_pos = np.arange(P['s'], P['e'])
        specs = [KNOCKS[c] for _, c in chunk]
        def scope_pos(it, sp):
            if sp.get('scope') is None:
                return img_pos
            t = it['target'][0] * it['cols'] + it['target'][1]
            cell = t if sp['scope'] == 'target' else (t + 7) % len(groups)
            return np.asarray(groups[cell])
        gates = [{'reset': {l: (scope_pos(it, sp), None) for l in sp.get('dn', [])}} for (it, _), sp in zip(chunk, specs)]
        fa_q = [None if sp.get('scope') is None else set(scope_pos(it, sp).tolist()) for (it, _), sp in zip(chunk, specs)]
        fa_layers = sorted({l for sp in specs for l in sp.get('fa', [])})
        primes = [sp.get('prime') for sp in specs]
        init = {}
        if any(primes):
            donors = [donor_ids(items, it, p, blank_id) if p else blank_id for (it, _), p in zip(chunk, primes)]
            need = sorted({d for d in donors if d not in cache})
            for k in range(0, len(need), args.batch_knock):
                sub = need[k:k + args.batch_knock]
                stt = donor_states(rt, items, sub, ctrl)
                for r, d in enumerate(sub):
                    cache[d] = {l: v[r].to(torch.bfloat16) for l, v in stt.items()}
            sel = torch.as_tensor([p is not None for p in primes])
            init = {l: (torch.stack([cache[d][l] for d in donors]).float(), sel) for l in cache[donors[0]]}
        ctrl.mode, ctrl.s, ctrl.e, ctrl.keep_state = ('split' if init else 'native'), P['s'], P['e'], False
        ctrl.init = init
        fa_rows = [set(sp.get('fa', [])) for sp in specs]
        with instrumented(rt, ctrl), override_gates_heads(rt, gates, P['L']), \
                fa_knockout_rows(rt, fa_layers, fa_rows, groups, P['L'], len(chunk), fa_q), \
                capture_layers(rt, (16, 20), 0, None) as st:
            logits = rt.model(**batch, use_cache=False, logits_to_keep=len(answers[0]) + 1).logits
        ctrl.mode, ctrl.init = 'native', {}
        scores = sr.score_rows(logits, answers, i_row, i_col, len(answers[0]), sr.digit_ids(rt, 6))
        rows = []
        for b, (it, c) in enumerate(chunk):
            rd = reader_readout(rt, st, b, groups, P['q_row'])
            rows.append({'key': f'{it["id"]}|{c}', 'id': it['id'], 'cond': c, **scores[b],
                         'reader': np.round(rd, 5).tolist()})
        _append(path, rows)
        n_done += len(chunk)
        print(f'[knock] {n_done}/{n_jobs} {time.time() - t0:.0f}s', flush=True)


@contextmanager
def fa_knockout_rows(rt, layers: list[int], rows: list[set], groups: list[list[int]], L: int, B: int,
                     queries: list[set | None] | None = None):
    """Per-row full-attention knockout: in the listed layers, image-token queries (all of them, or only the row's
    `queries` positions) cannot attend to image tokens of other cells."""
    if not layers:
        yield
        return
    dev = rt.layers[0].mlp.down_proj.weight.device
    causal = torch.ones(L, L, dtype=torch.bool, device=dev).tril()
    block = causal.clone()
    for gi in groups:
        others = torch.as_tensor([t for g in groups if g is not gi for t in g], device=dev)
        r = torch.as_tensor(gi, device=dev)
        block[r[:, None], others[None, :]] = False
    def row_mask(b):
        if queries is None or queries[b] is None:
            return block
        m = causal.clone()
        q = torch.as_tensor(sorted(queries[b]), device=dev)
        m[q] = block[q]
        return m
    hs = []
    for layer in layers:
        mask = torch.stack([row_mask(b) if layer in rows[b] else causal for b in range(B)])[:, None]   # True = may attend

        def pre(m, a, kw, mask=mask):
            h = kw.get('hidden_states', a[0] if a else None)
            if h is None or h.shape[1] != L:
                return a, kw
            kw['attention_mask'] = mask
            return a, kw
        hs.append(rt.layers[layer - 1].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


# ----------------------------------------------------------------------------- patch (noising at the target)

PATCH_GROUPS = {'embed': [0]}
for _l in range(1, 20):
    PATCH_GROUPS[f'mix{_l}'] = [2 * _l - 1]
    PATCH_GROUPS[f'mlp{_l}'] = [2 * _l]
PATCH_GROUPS.update({
    'dn_mix_1_15': [2 * l - 1 for l in DN_LAYERS_BELOW_16],
    'fa_mix_4_12': [2 * l - 1 for l in (4, 8, 12)],
    'mlp_1_15': [2 * l for l in range(1, 16)],
    'all_1_15': [c for l in range(1, 16) for c in (2 * l - 1, 2 * l)],
})


@contextmanager
def replace_components(rt, rows: list[tuple[int, list[int], dict[int, torch.Tensor]]], L: int):
    """rows: (batch row, absolute positions, {component index: values (len(positions), d)}); component 0 replaces the
    decoder input, 2l-1 the mixer output of layer l, 2l the MLP output of layer l."""
    by_comp = defaultdict(list)
    for b, pos, vals in rows:
        for c, v in vals.items():
            by_comp[c].append((b, torch.as_tensor(pos), v))
    hs = []
    for c, lst in by_comp.items():
        if c == 0:
            def pre(m, a, kw, lst=lst):
                x = a[0]
                if x.shape[1] != L:
                    return None
                x = x.clone()
                for b, pos, v in lst:
                    x[b, pos.to(x.device)] = v.to(x.device, x.dtype)
                return (x, *a[1:]), kw
            hs.append(rt.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
            continue
        layer = (c + 1) // 2
        block = rt.layers[layer - 1]
        mod = (block.linear_attn if block.block_type == 'linear_attention' else block.self_attn) if c % 2 else block.mlp

        def hook(m, a, o, lst=lst):
            h = sr._out(o)
            if h.shape[1] != L:
                return o
            for b, pos, v in lst:
                h[b, pos.to(h.device)] = v.to(h.device, h.dtype)
            return o
        hs.append(mod.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@torch.inference_mode()
def stage_patch(rt, args, items: Items, output: Path) -> None:
    path = output / 'patch.jsonl'
    done = _done(path)
    odd = [it for it in items.odd(args.patch_sets)]
    keep = defaultdict(list)
    for it in sorted(odd, key=lambda x: x['id']):
        keep[(it['set'], it['pair_name'], it['level'], it['variant'])].append(it)
    odd = [x for v in keep.values() for x in v[:args.patch_limit]]
    t0 = time.time()
    for n_it, it in enumerate(odd):
        conds = [c for c in args.patch_groups if f'{it["id"]}|{c}' not in done]
        if not conds:
            continue
        # corrupted run: component outputs at the target tokens
        batch, answers, i_row, i_col = batch_for(rt, items, [it['partner']], [tuple(it['target'])])
        P = positions(rt, batch, len(answers[0]), i_row, i_col)
        groups = cell_tokens(it['meta'], P['s'])
        tcell = it['target'][0] * it['cols'] + it['target'][1]
        tpos = groups[tcell]
        img = slice(P['s'], P['e'])
        with capture_layers(rt, (), 19, img) as st:
            rt.model(**batch, use_cache=False, logits_to_keep=1)
        rel = [p - P['s'] for p in tpos]
        corr = {c: st['comp'][c][0, rel].clone() for c in range(N_COMP)}
        one, answers1, _, _ = batch_for(rt, items, [it['id']], [tuple(it['target'])])
        answers = answers1 * args.batch_patch
        for a in range(0, len(conds), args.batch_patch):
            cc = conds[a:a + args.batch_patch]
            batch = repeat_batch(one, len(cc))
            rows = [(b, tpos, {k: corr[k] for k in PATCH_GROUPS[c]}) for b, c in enumerate(cc) if c != 'none']
            with replace_components(rt, rows, P['L']), capture_layers(rt, (16, 20), 0, None) as st2:
                logits = rt.model(**batch, use_cache=False, logits_to_keep=len(answers[0]) + 1).logits
            scores = sr.score_rows(logits, answers[:len(cc)], i_row, i_col, len(answers[0]), sr.digit_ids(rt, 6))
            out = []
            for b, c in enumerate(cc):
                rd = reader_readout(rt, st2, b, groups, P['q_row'])
                out.append({'key': f'{it["id"]}|{c}', 'id': it['id'], 'cond': c, **scores[b],
                            'reader_target': np.round(rd[:, tcell], 5).tolist()})
            _append(path, out)
        print(f'[patch] {n_it + 1}/{len(odd)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- self-test

@torch.inference_mode()
def stage_selftest(rt, args, items: Items, output: Path) -> None:
    """Instrumented forwards must reproduce the native forward; attribution must sum to the actual logits."""
    its = sorted(items.odd(['atlas']), key=lambda x: x['id'])[:4]
    ids = [it['id'] for it in its]
    batch, answers, i_row, i_col = batch_for(rt, items, ids, [tuple(it['target']) for it in its])
    P = positions(rt, batch, len(answers[0]), i_row, i_col)
    n = len(answers[0]) + 1
    ref = rt.model(**batch, use_cache=False, logits_to_keep=n).logits.float()
    ctrl = DNControl()
    res = {}
    for mode in ('split', 'trace'):
        ctrl.mode, ctrl.s, ctrl.e, ctrl.init, ctrl.target_tok = mode, P['s'], P['e'], {}, None
        with instrumented(rt, ctrl):
            lg = rt.model(**batch, use_cache=False, logits_to_keep=n).logits.float()
        res[f'{mode}_max_abs_logit_diff'] = float((lg - ref).abs().max())
        res[f'{mode}_argmax_agree'] = float((lg.argmax(-1) == ref.argmax(-1)).float().mean())
    ctrl.mode = 'native'
    groups = cell_tokens(its[0]['meta'], P['s'])
    img = slice(P['s'], P['e'])
    with capture_layers(rt, (16, 20), 19, img) as st:
        rt.model(**batch, use_cache=False, logits_to_keep=1)
    comps = torch.stack([st['comp'][c] for c in range(N_COMP)], 1)
    for layer in (16, 20):
        x = st['inp'][layer][0]
        n_c = 1 + 2 * (layer - 1)
        rec = comps[0, :n_c].float().sum(0)
        res[f'L{layer}_residual_rel_err'] = float((rec - x[img].float()).norm() / x[img].float().norm())
        cos, sin = st['pe'][layer]
        c, act = logit_attribution(rt, layer, x, cos[0], sin[0], [P['q_row']], comps[0, :n_c], img, [6, 11])
        res[f'L{layer}_logit_sum_max_abs_err'] = float((c.sum(2) - act).abs().max())
        res[f'L{layer}_logit_abs_mean'] = float(act.abs().mean())
    (output / 'selftest.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


# ----------------------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True,
                   choices=['build', 'selftest', 'behave', 'spotcheck', 'trace', 'knock', 'patch'])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--atlas-pairs', nargs='+', default=['qo', 'closure', 'tilt', 'curve', 'sat', 'size', 'term', 'plus',
                                                        'dot', 'circsq', 'plusx'])
    p.add_argument('--atlas-salient-pairs', nargs='+', default=['qo', 'closure'])
    p.add_argument('--atlas-items', type=int, default=24)
    p.add_argument('--prev-conditions', nargs='+', default=['qo:subtle:swap', 'qo:salient:swap', 'closure:subtle:swap',
                   'closure:salient:swap', 'dot:subtle:swap', 'term:subtle:swap', 'plus:subtle:swap',
                   'qo:subtle:std', 'closure:subtle:std', 'dot:subtle:std', 'term:subtle:std', 'tilt:subtle:std'])
    p.add_argument('--prev-items', type=int, default=12)
    p.add_argument('--mag', nargs='+', default=['qo:3,6,10,14', 'closure:2,4,8,12'])
    p.add_argument('--mag-items', type=int, default=36)
    p.add_argument('--shift-pairs', nargs='+', default=['qo', 'closure'])
    p.add_argument('--shift-blanks', type=int, nargs='+', default=[0, 6, 12, 18])
    p.add_argument('--shift-items', type=int, default=6)
    p.add_argument('--batch-direct', type=int, default=32)
    p.add_argument('--batch-trace', type=int, default=16)
    p.add_argument('--batch-knock', type=int, default=12)
    p.add_argument('--batch-patch', type=int, default=16)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--trace-sets', nargs='+', default=['atlas', 'prev', 'mag', 'shift'])
    p.add_argument('--trace-limit', type=int, default=0)
    p.add_argument('--knock-sets', nargs='+', default=['atlas', 'prev', 'shift', 'mag'])
    p.add_argument('--knocks', nargs='+', default=list(KNOCKS))
    p.add_argument('--knock-limit', type=int, default=0, help='max items per condition cell (0 = all)')
    p.add_argument('--patch-sets', nargs='+', default=['atlas'])
    p.add_argument('--patch-limit', type=int, default=12)
    p.add_argument('--patch-groups', nargs='+', default=['none'] + list(PATCH_GROUPS))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'build':
        cfg = {'seed': args.seed, 'atlas_pairs': args.atlas_pairs, 'atlas_items': args.atlas_items,
               'atlas_salient_pairs': args.atlas_salient_pairs,
               'prev_conditions': [c.split(':') for c in args.prev_conditions], 'prev_items': args.prev_items,
               'mag': {c.split(':')[0]: [float(x) for x in c.split(':')[1].split(',')] for c in args.mag},
               'mag_items': args.mag_items, 'shift_pairs': args.shift_pairs, 'shift_blanks': args.shift_blanks,
               'shift_items': args.shift_items}
        sc.build(args.output, cfg)
        print(f'[build] {len(read_jsonl(args.output / "items.jsonl"))} items', flush=True)
        return
    items = Items(args.output)
    rt = qr.load(args.model)
    meta = {'stage': args.stage, 'args': {k: str(v) for k, v in vars(args).items()}, **qr.describe(rt),
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    {'selftest': stage_selftest, 'behave': stage_behave, 'spotcheck': stage_spotcheck, 'trace': stage_trace,
     'knock': stage_knock, 'patch': stage_patch}[args.stage](rt, args, items, args.output)
    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (args.output / f'run_{args.stage}.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
