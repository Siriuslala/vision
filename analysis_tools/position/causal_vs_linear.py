"""Experiment 3.14: causal mask vs linear attention in row/column readout (engine, no model loading).

Design: docs/analysis/position/design_causal_vs_linear.md; results: results_2026-10-21_causal_vs_linear.md.

The experiment compares Qwen3.5-9B (24 Gated DeltaNet + 8 full-attention layers, causal) with Qwen3-VL-8B
(36 full-attention layers, causal, DeepStack) on the measurements of experiments 3.10 / 3.11, and edits the
DeltaNet layers of Qwen3.5 alone. This module holds the family-generic pieces:

* `prepare`: one tokenized prompt with merged visual embeddings and, for Qwen3-VL, the three DeepStack
  feature maps (Qwen3.5 reuses `coordinate_readout.prepare` unchanged);
* `deepstack_rows`: injects per-batch-row DeepStack features into the prefill when the model is driven
  through `inputs_embeds` (the native path only injects them when `pixel_values` are given);
* `tf_kv_logits`: teacher-forced logits after copying run-B keys / values of image-token subsets into run A's
  prompt cache (the 3.11 measurement, generic over the set of full-attention layers);
* `attention_rows`: softmax rows of full-attention layers at chosen queries (gated Qwen3.5 attention and
  ungated Qwen3-VL attention);
* `gate_override`: prefill-only Gated DeltaNet gate edits on the image span (freeze, no decay, no write);
* `attention_knockout`: prefill-only removal of image-to-image attention in chosen full-attention layers.

Layers are 1-based. Token indices are relative to the image span (raster order) unless noted otherwise.
"""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import sequence_order_r2 as s2

GATE_MODES = ('none', 'freeze', 'nodecay', 'nowrite', 'freeze_l9_16', 'nodecay_l9_16', 'nowrite_target', 'freeze_target')


# ----------------------------------------------------------------------------- architecture facts

def full_layers(rt: qr.Runtime) -> tuple[int, ...]:
    """1-based decoder layers whose token mixer is softmax attention."""
    if rt.family == 'qwen3_5':
        return cr.FULL_LAYERS
    return tuple(range(1, len(rt.layers) + 1))


def linear_layers(rt: qr.Runtime) -> tuple[int, ...]:
    """1-based Gated DeltaNet layers (empty for Qwen3-VL)."""
    return tuple(i + 1 for i, b in enumerate(rt.layers) if getattr(b, 'block_type', '') == 'linear_attention')


def n_heads(rt: qr.Runtime) -> int:
    return rt.layers[full_layers(rt)[0] - 1].self_attn.config.num_attention_heads


def kv_index(rt: qr.Runtime) -> dict[int, int]:
    """1-based full-attention layer -> cache layer index."""
    return {l: rt.layers[l - 1].self_attn.layer_idx for l in full_layers(rt)}


def layer_bins(rt: qr.Runtime) -> list[tuple[int, ...]]:
    """Groups of full-attention layers used for layer-resolved KV edits.

    Qwen3.5: each of the 8 full-attention layers alone. Qwen3-VL: nine bins of four consecutive layers,
    which keeps the number of edit rows comparable between the two models.
    """
    fl = full_layers(rt)
    if len(fl) <= 8:
        return [(l,) for l in fl]
    return [tuple(fl[i:i + 4]) for i in range(0, len(fl), 4)]


# ----------------------------------------------------------------------------- prompts with DeepStack

def prepare(rt: qr.Runtime, image, text: str) -> cr.Prepared:
    """Prepared prompt; for Qwen3-VL prep.meta['ds'] holds the DeepStack maps (list of (Nv, D))."""
    if rt.family == 'qwen3_5':
        return cr.prepare(rt, image, text)
    batch = qr.build(rt, [image], text)
    span = qr.image_spans(rt, batch)[0]
    mrope, _ = rt.model.model.get_rope_index(batch['input_ids'], mm_token_type_ids=batch['mm_token_type_ids'],
                                             image_grid_thw=batch['image_grid_thw'])
    L = batch['input_ids'].shape[1]
    seq = torch.arange(L, device=mrope.device).view(1, 1, L)
    pos = torch.cat([seq, mrope], 0)
    with torch.inference_mode():
        out = rt.model.model.get_image_features(batch['pixel_values'], batch['image_grid_thw'], return_dict=True)
    prep = cr.Prepared(batch, batch['input_ids'][0], span, None, None, pos)
    prep.vis = out.pooler_output[0]
    prep.meta['ds'] = [d for d in out.deepstack_features]
    if any(d.shape != prep.vis.shape for d in prep.meta['ds']):
        raise ValueError('DeepStack maps do not match the merged visual tokens.')
    return prep


def splice_rows(a: torch.Tensor, b: torch.Tensor, tokens: list[int]) -> torch.Tensor:
    """Copy of the (Nv, D) map a whose rows `tokens` come from b."""
    out = a.clone()
    idx = torch.as_tensor(tokens, device=a.device, dtype=torch.long)
    out[idx] = b[idx]
    return out


@contextmanager
def deepstack_rows(rt: qr.Runtime, rows: list[list[torch.Tensor]] | None, span: tuple[int, int]):
    """Inject DeepStack features per batch row whenever a forward covers the image span.

    rows: one list of DeepStack maps (each (Nv, D)) per batch row, or None (DeepStack off, or Qwen3.5).
    The decoder adds map k to the image-token states after decoder layer k (k = 1, 2, 3), as in the native
    forward. Calls whose input does not reach the end of the span (cached tail / decoding steps) are untouched.
    A forward whose batch is a multiple of len(rows) repeats the rows (row b uses rows[b % len(rows)]).
    """
    if rows is None or rt.family != 'qwen3_vl':
        yield
        return
    lm = rt.model.model.language_model
    original = lm.forward
    s, e = span

    def forward(*args, **kwargs):
        x = kwargs.get('inputs_embeds')
        if x is None or x.shape[1] < e or kwargs.get('deepstack_visual_embeds') is not None:
            return original(*args, **kwargs)
        B = x.shape[0]
        if B % len(rows):
            raise ValueError(f'Batch {B} is not a multiple of the {len(rows)} DeepStack rows.')
        mask = torch.zeros(B, x.shape[1], dtype=torch.bool, device=x.device)
        mask[:, s:e] = True
        n_maps = len(rows[0])
        kwargs['visual_pos_masks'] = mask
        kwargs['deepstack_visual_embeds'] = [torch.cat([rows[b % len(rows)][k] for b in range(B)], 0)
                                             for k in range(n_maps)]
        return original(*args, **kwargs)

    lm.forward = forward
    try:
        yield
    finally:
        del lm.forward


# ----------------------------------------------------------------------------- KV subset patching

@torch.inference_mode()
def tf_kv_logits(rt: qr.Runtime, xb: torch.Tensor, xa: torch.Tensor, pos: torch.Tensor, L: int,
                 edits: list[tuple[list[int], tuple[int, ...]]], span: tuple[int, int],
                 ds_b: list[torch.Tensor] | None = None, ds_a: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Teacher-forced tail logits with run-B keys/values copied into run A's prompt cache.

    Rows: 0 = run B, 1.. = run A with edits[i-1] = (absolute prompt positions, full-attention layers).
    The prompt except its last token is prefilled; the cache of rows >= 1 is edited; the tail (last prompt
    token and the teacher-forced answer) then runs in one cached forward. Gated DeltaNet states (Qwen3.5)
    keep run A's values, so the edit measures only the answer-time read through softmax attention.
    """
    R = 1 + len(edits)
    x = torch.cat([xb] + [xa] * len(edits), 0)
    p = pos.expand(-1, R, -1).contiguous()
    ds = None if ds_a is None else [ds_b] + [ds_a] * len(edits)
    with deepstack_rows(rt, ds, span):
        o = rt.model(inputs_embeds=x[:, :L - 1], position_ids=p[:, :, :L - 1], use_cache=True, logits_to_keep=1)
    cache = o.past_key_values
    idx = kv_index(rt)
    for b, (positions, layers) in enumerate(edits, start=1):
        if not positions:
            continue
        for l in layers:
            keys, values = cf.sr.cache_kv(cache, idx[l])
            ps = torch.tensor(positions, device=keys.device)
            keys[b, :, ps] = keys[0, :, ps]
            values[b, :, ps] = values[0, :, ps]
    o = rt.model(inputs_embeds=x[:, L - 1:], position_ids=p[:, :, L - 1:], past_key_values=cache, use_cache=True)
    return o.logits


def kv_conditions(rt: qr.Runtime, grid, target, moved, s: int, e: int, L: int, full: bool = True
                  ) -> list[tuple[str, list[int], tuple[int, ...]]]:
    """(name, absolute positions, layers) of every KV edit; the 3.11 group set over this model's layers.

    full=False keeps the main groups only (emission-order variants and gate-edited runs).
    """
    H, W = grid
    FL = full_layers(rt)
    g = cf.token_groups(grid, target, moved)
    img = list(range(H * W))
    ab = lambda idx: [s + i for i in idx]
    conds = [('none', [], ()), ('all_img', ab(img), FL), ('text', list(range(e, L - 1)), FL)]
    allg = {**g, 'all_img': img}
    main = ('obj', 'all_but_obj', 'row_same', 'row_left', 'row_right', 'below', 'below_cols', 'above', 'ring1', 'ring5p')
    if not full:
        conds += [(name, ab(g[name]), FL) for name in main]
        conds += [(f'all_img@{"-".join(map(str, b))}', ab(img), b) for b in layer_bins(rt)]
        return [c for c in conds if c[0] == 'none' or c[1]]
    conds += [(name, ab(g[name]), FL) for name in cf.GROUPS_ALL_LAYERS]
    conds += [(f'row{r}', ab([r * W + c for c in range(W)]), FL) for r in range(H)]
    conds += [(f'col{c}', ab([r * W + c for r in range(H)]), FL) for c in range(W)]
    conds += [(f'{name}@{"-".join(map(str, b))}', ab(allg[name]), b)
              for name in ('obj', 'row_same', 'all_but_obj', 'all_img') for b in layer_bins(rt)]
    return [c for c in conds if c[0] == 'none' or c[1]]


# ----------------------------------------------------------------------------- attention rows

@torch.inference_mode()
def attention_rows(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, queries: list[int],
                   layers: tuple[int, ...] | None = None, ds: list[torch.Tensor] | None = None,
                   span: tuple[int, int] = (0, 0)) -> dict[int, torch.Tensor]:
    """Softmax rows {layer: (heads, Q, L) float32 on CPU} of full-attention layers at the given queries.

    Pre-hooks capture each attention module's input and rotary tables during one forward; q and k are
    recomputed (Qwen3.5: q_proj also emits the output gate, which does not enter the softmax).
    """
    layers = full_layers(rt) if layers is None else layers
    if rt.family == 'qwen3_5':
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    else:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
    store, handles = {}, []
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn

        def pre(module, args, kwargs, layer=layer):
            if layer in store:
                return
            store[layer] = (kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0],
                            kwargs['position_embeddings'])
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        with deepstack_rows(rt, None if ds is None else [ds], span):
            rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    out = {}
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn
        h, (cos, sin) = store[layer]
        dev = h.device
        q_idx = torch.tensor(queries, device=dev)
        cos, sin = cos.to(dev), sin.to(dev)
        Lh, hd = h.shape[1], mod.head_dim
        if rt.family == 'qwen3_5':
            q, _ = torch.chunk(mod.q_proj(h[:, q_idx]).view(1, len(queries), -1, hd * 2), 2, dim=-1)
        else:
            q = mod.q_proj(h[:, q_idx]).view(1, len(queries), -1, hd)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(1, Lh, -1, hd)).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, q_idx], sin[:, q_idx])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        k = repeat_kv(k, mod.num_key_value_groups)
        sc = (q.float() @ k.float().transpose(2, 3)) * mod.scaling
        sc = sc.masked_fill(torch.arange(Lh, device=dev)[None, :] > q_idx[:, None], float('-inf'))
        out[layer] = sc.softmax(-1)[0].cpu()
    return out


def profile_masks(grid: tuple[int, int], target: list[int]) -> dict[str, np.ndarray]:
    """Boolean masks over image tokens (raster order) used to summarise one attention row.

    target: tokens of the target; row / col: the target's token rows / columns (target excluded);
    row_after / row_before: target-row tokens after / before the target's first token in raster order;
    col0: the first token column; below / above: rows strictly below / above the target's rows.
    """
    H, W = grid
    rr, cc = np.divmod(np.arange(H * W), W)
    t = np.zeros(H * W, bool)
    t[target] = True
    rows = np.isin(rr, sorted({i // W for i in target}))
    cols = np.isin(cc, sorted({i % W for i in target}))
    first = min(target)
    return {'target': t, 'row': rows & ~t, 'col': cols & ~t, 'row_after': rows & ~t & (np.arange(H * W) > first),
            'row_before': rows & ~t & (np.arange(H * W) < first), 'col0': cc == 0,
            'below': rr > max(i // W for i in target), 'above': rr < min(i // W for i in target)}


# ----------------------------------------------------------------------------- Gated DeltaNet edits

def parse_mode(mode: str) -> tuple[str, tuple[int, int] | None, bool]:
    """(base edit, layer range or None for all DeltaNet layers, target-only flag) of a gate mode name.

    Examples: 'freeze' (all layers, whole image span), 'freeze_l9_16' (DeltaNet layers 9-16, whole span),
    'nowrite_target' (all layers, only the target tokens; the caller supplies the positions).
    """
    target = mode.endswith('_target')
    base = mode[:-len('_target')] if target else mode
    rng = None
    if '_l' in base:
        base, r = base.split('_l')
        a, b = r.split('_')
        rng = (int(a), int(b))
    if base not in ('none', 'freeze', 'nodecay', 'nowrite'):
        raise ValueError(f'Unknown gate mode {mode}')
    return base, rng, target


@contextmanager
def gate_override(rt: qr.Runtime, mode: str, span: tuple[int, int], layers: tuple[int, ...] | None = None,
                  positions: list[list[int]] | None = None):
    """Prefill-only gate edit of every head of the chosen DeltaNet layers.

    freeze:  alpha = 1 and beta = 0, the recurrent state crosses the edited tokens unchanged (no write);
    nodecay: alpha = 1, the edited tokens write but nothing is forgotten;
    nowrite: beta = 0, the edited tokens do not write and the state still decays.
    The edited tokens are the whole image span, or, when `positions` is given, the absolute positions
    positions[b % len(positions)] of batch row b (target-only modes). Layers come from the mode name
    (suffix _l<a>_<b>) or from `layers`. Forwards whose input does not reach the end of the span are untouched.
    """
    base, rng, target = parse_mode(mode)
    if base == 'none':
        yield
        return
    if rt.family != 'qwen3_5':
        raise ValueError('Gate edits need Gated DeltaNet layers.')
    if target and positions is None:
        raise ValueError(f'Mode {mode} needs target positions.')
    if layers is None:
        layers = tuple(l for l in linear_layers(rt) if rng is None or rng[0] <= l <= rng[1])
    s, e = span
    handles = []

    def setter(value):
        def hook(module, args, out):
            if out.shape[1] < e:
                return out
            out = out.clone()
            if positions is None:
                out[:, s:e] = value
            else:
                for b in range(out.shape[0]):
                    out[b, torch.as_tensor(positions[b % len(positions)], device=out.device)] = value
            return out
        return hook
    for l in layers:
        mod = s2.linear_block(rt, l)
        if base in ('freeze', 'nodecay'):
            handles.append(mod.in_proj_a.register_forward_hook(setter(-s2.BIG)))
        if base in ('freeze', 'nowrite'):
            handles.append(mod.in_proj_b.register_forward_hook(setter(-s2.BIG)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ----------------------------------------------------------------------------- full-attention knockout

def knockout_mask(L: int, span: tuple[int, int], kind: str, grid: tuple[int, int], device) -> torch.Tensor:
    """Boolean (1, 1, L, L) attention mask (True = may attend) for a causal prefill.

    img:   image queries cannot attend to other image tokens (each keeps itself and all text);
    local: image queries cannot attend to earlier image tokens in the same token row within 3 columns.
    """
    m = torch.ones(L, L, dtype=torch.bool, device=device).tril()
    s, e = span
    n = e - s
    if kind == 'img':
        block = ~torch.eye(n, dtype=torch.bool, device=device)
    elif kind == 'local':
        H, W = grid
        r, c = torch.div(torch.arange(n, device=device), W, rounding_mode='floor'), torch.arange(n, device=device) % W
        dq = c[:, None] - c[None, :]
        block = (r[:, None] == r[None, :]) & (dq >= 1) & (dq <= 3)
    else:
        raise ValueError(f'Unknown knockout kind {kind}')
    m[s:e, s:e] &= ~block
    return m[None, None]


@contextmanager
def attention_knockout(rt: qr.Runtime, layers: tuple[int, ...], span: tuple[int, int], grid: tuple[int, int],
                       kind: str = 'img'):
    """Replace the attention mask of the given full-attention layers during the prefill (no cache)."""
    if not layers:
        yield
        return
    cache: dict = {}
    handles = []

    def pre(module, args, kwargs):
        h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
        L = h.shape[1]
        if L < span[1]:                          # cached decoding steps keep the native mask
            return args, kwargs
        if L not in cache:
            cache[L] = knockout_mask(L, span, kind, grid, h.device)
        kwargs['attention_mask'] = cache[L]
        return args, kwargs
    for l in layers:
        if l not in full_layers(rt):
            raise ValueError(f'Layer {l} is not a full-attention layer.')
        handles.append(rt.layers[l - 1].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ----------------------------------------------------------------------------- recovery statistics

def recovery(value: dict, cond: str) -> float | None:
    """r = (LD_e - LD_A) / (LD_B - LD_A) for one value record of a KV table; None when undefined."""
    lda, ldb = value['LD'].get('none'), value['LD_B']
    if lda is None or cond not in value['LD'] or value['dA'] is None or value['dA'] == value['dB']:
        return None
    if abs(ldb - lda) < 2:
        return None
    return (value['LD'][cond] - lda) / (ldb - lda)
