"""Experiment 3.12: failure mechanism of coordinate-digit readout in Qwen3.5-9B.

Engine of the experiment (GPU stage drivers live in `digit_failure_run.py`, CPU analysis in
`digit_failure_analysis.py`). Design: docs/analysis/position/design_digit_failure.md.

The central measurement is a *first-digit factorial*: for one input (image + question + an answer
prefix that ends right before the first digit of the studied number) every first digit k = 1..9 is
forced, and one batched prefill records

* the digit distribution at the position that predicts the first digit (role q0, independent of k);
* the digit distribution at the position that predicts the second digit given k (role q1);
* residual streams at q0 and q1 for a few layers;
* the direct logit attribution (DLA) of every token mixer, MLP and full-attention head output to the
  ten digit logits at q0 and q1, computed exactly through the final RMSNorm scale of each row.

Inputs come from several "families": the single red disc sweeps of experiment 3.6, a blank canvas,
RefCOCO point questions, T6-like red-target grids, dot counting, the text-given pixel conversion of
3.6 round 2 and a pure-text position task (an 'x' in a row of 'o' symbols).
Layer numbers are 1-based.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.common import stimuli as st
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import coordinate_value as cv

FULL_LAYERS = cr.FULL_LAYERS
N_LAYERS = 32
N_HEADS = 16
RESID_LAYERS = (8, 12, 16, 20, 24, 28, 32)     # residual layers stored at q0 and q1
FIRST_DIGITS = tuple(range(1, 10))

# Canonical answer prefix of the point task (the most frequent greedy template in experiment 3.6:
# a code block with four-space indentation). Only the tokens before the first digit of x are used;
# for y the x value and the separator ", " follow.
POINT_PREFIX_X = '```json\n[\n    {"point_2d": ['
POINT_SEPARATOR = ', '

TEXT_ROW_LENGTH = 100
TEXT_POSITION_QUERY = ('Below is a row of {n} symbols separated by spaces. Every symbol is o except a single x.\n'
                       '{row}\n'
                       'Counting from the left and starting at 1, what is the position of the x? '
                       'Answer with only the number.')

GRID_SIZES = (6, 10, 20)
GRID_TEMPLATES = ('person', 'ring', 'square', 'arrow')
GRID_PROMPT = ('One figure is drawn in red. Give the center of its bounding box. Return only '
               '[{"point_2d": [x, y]}]. All coordinates use 0-1000 normalization; x increases rightward and '
               'y increases downward. Output only the JSON, with no explanation.')


# ----------------------------------------------------------------------------- stimuli

def text_position_row(position: int, n: int = TEXT_ROW_LENGTH) -> str:
    """Row of n space-separated symbols with a single 'x' at 1-based `position`."""
    if not 1 <= position <= n:
        raise ValueError(f'position {position} outside 1..{n}')
    return ' '.join('x' if i == position else 'o' for i in range(1, n + 1))


def text_position_prompt(position: int, n: int = TEXT_ROW_LENGTH) -> str:
    return TEXT_POSITION_QUERY.format(n=n, row=text_position_row(position, n))


def grid_spec(n: int, row: int, col: int, template: str) -> st.StimulusSpec:
    """T6-like grid: n x n instances in 64-pixel cells, the target drawn in pure red."""
    return st.StimulusSpec(template=template, rows=n, cols=n, change=st.Change('color', 4), target=(row, col), cell=64)


def grid_rows(n: int) -> tuple[int, ...]:
    """Target rows used per grid size (all columns are swept in each row)."""
    return {6: (1, 4), 10: (2, 7), 20: (5, 14)}[n]


def grid_cases() -> list[dict]:
    cases = []
    for n in GRID_SIZES:
        for i, r in enumerate(grid_rows(n)):
            for c in range(n):
                template = GRID_TEMPLATES[(c + i) % len(GRID_TEMPLATES)]
                cases.append({'id': f'grid{n:02d}_r{r:02d}_c{c:02d}', 'n': n, 'row': r, 'col': c, 'template': template})
    return cases


def grid_input(case: dict) -> tuple[Image.Image, dict]:
    image, meta = st.generate(grid_spec(case['n'], case['row'], case['col'], case['template']))
    target = next(i for i in meta['instances'] if i['is_target'])
    W, H = meta['canvas']
    truth = {'canvas': [W, H], 'cell': 64, 'grid_origin': meta['grid_origin'], 'n': case['n'],
             'target_rc': [case['row'], case['col']], 'center_px': target['center'],
             'center_units': [1000 * target['center'][0] / W, 1000 * target['center'][1] / H],
             'col_edges_units': [1000 * (meta['grid_origin'][0] + 64 * j) / W for j in range(case['n'] + 1)]}
    return image, truth


# ----------------------------------------------------------------------------- units and rows

@dataclass
class Unit:
    """One input whose first digit is forced to every k in `ks`.

    family:  'sweep_h' | 'sweep_v' | 'ctx' | 'blank' | 'refcoco' | 'grid' | 'count' | 'textconv' | 'textpos'.
    axis:    0 = x, 1 = y (the coordinate whose digits are studied); 0 for scalar answers.
    truth:   true value of the studied number in the answer's own units (0-1000, a count or a position).
    prefix:  answer text before the first digit of the studied number ('' when the answer is the number).
    ctx:     x value written in the prefix when the studied number is y (else None).
    """
    id: str
    family: str
    axis: int
    truth: float
    prompt: str
    prefix: str
    ks: tuple[int, ...] = FIRST_DIGITS
    ctx: int | None = None
    source: dict = field(default_factory=dict)


def point_prefix(axis: int, ctx: int | None) -> str:
    if axis == 0:
        return POINT_PREFIX_X
    if ctx is None:
        raise ValueError('The y prefix needs the x value written before it')
    return POINT_PREFIX_X + str(int(ctx)) + POINT_SEPARATOR


def prefix_ids(rt: qr.Runtime, prefix: str) -> list[int]:
    """Token ids of an answer prefix; the token right before the first digit must not contain a digit."""
    if not prefix:
        return []
    ids = rt.tokenizer.encode(prefix, add_special_tokens=False)
    if rt.tokenizer.decode(ids) != prefix:
        raise ValueError(f'Prefix does not round-trip through the tokenizer: {prefix!r}')
    return ids


def text_prepare(rt: qr.Runtime, text: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """inputs_embeds (1, L, D), position ids (4, 1, L) and prompt length of a text-only turn.

    Without images the MRoPE ids reduce to the sequence index on all three axes.
    """
    batch = qr.build(rt, [], text)
    ids = batch['input_ids']
    x = rt.model.model.get_input_embeddings()(ids)
    L = ids.shape[1]
    pos = torch.arange(L, device=ids.device).view(1, 1, L).expand(4, 1, L).clone()
    return x, pos, L


def append_tokens(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Append answer tokens to inputs_embeds / position ids (same convention as cr.teacher_forced)."""
    if not ids:
        return x, pos
    tail = rt.model.model.get_input_embeddings()(torch.tensor(ids, device=x.device))[None]
    L, n = x.shape[1], len(ids)
    step = torch.arange(1, n + 1, device=x.device, dtype=pos.dtype)
    pa = torch.cat([(L - 1 + step).view(1, 1, n), pos[1:, :, -1:] + step.view(1, 1, n)], 0)
    return torch.cat([x, tail.to(x.dtype)], 1), torch.cat([pos, pa], 2)


# ----------------------------------------------------------------------------- batched readout pass

def digit_ids(rt: qr.Runtime) -> list[int]:
    return cv.digit_token_ids(rt)


@torch.inference_mode()
def readout_forward(rt: qr.Runtime, X: torch.Tensor, P: torch.Tensor, q0: list[int], q1: list[int],
                    resid_layers: tuple[int, ...] = RESID_LAYERS) -> dict:
    """One right-padded batched prefill with per-row readouts at q0 and q1.

    Returns numpy arrays (B rows, 2 roles in the order q0, q1):
      logp:   (B, 2, 10) digit log-probabilities under the full-vocabulary softmax;
      lp_top: (B, 2) log-probability of the arg-max token and top_id (B, 2) its id;
      resid:  (B, 2, len(resid_layers), D) float16 residual stream after each listed layer;
      dla_mixer, dla_mlp: (B, 2, 32, 10) direct digit-logit contribution of every block output;
      dla_head: (B, 2, 8, 16, 10) contribution of every full-attention head (o_proj column block);
      dla_embed: (B, 2, 10) contribution of the input embedding;
      logit: (B, 2, 10) digit logits recomputed from the final norm (for the decomposition check).
    The DLA of a vector c is W_U[digits] (s * c), with s = (1 + w) / rms(h_final) the row's frozen final
    RMSNorm scale, so the contributions sum exactly to the digit logits.
    """
    B = X.shape[0]
    b = torch.arange(B, device=X.device)
    idx = torch.stack([torch.tensor(q0), torch.tensor(q1)], 1).to(X.device)      # (B, 2)
    store: dict = {'mixer': {}, 'mlp': {}, 'head': {}, 'resid': {}}
    handles = []
    lm = rt.model.model.language_model

    def pick(h):
        h = h[0] if isinstance(h, tuple) else h
        return h[b.to(h.device)[:, None], idx.to(h.device)].to(X.device)          # (B, 2, D)

    handles.append(lm.layers[0].register_forward_pre_hook(
        lambda m, a, kw: store.__setitem__('embed', pick(kw.get('hidden_states', a[0] if a else None)).float()),
        with_kwargs=True))
    handles.append(lm.norm.register_forward_pre_hook(lambda m, a: store.__setitem__('final', pick(a[0]).float())))
    handles.append(lm.norm.register_forward_hook(lambda m, a, o: store.__setitem__('normed', pick(o))))
    for l, layer in enumerate(lm.layers, start=1):
        mixer = layer.self_attn if hasattr(layer, 'self_attn') else layer.linear_attn
        handles.append(mixer.register_forward_hook(lambda m, a, o, l=l: store['mixer'].__setitem__(l, pick(o).float())))
        handles.append(layer.mlp.register_forward_hook(lambda m, a, o, l=l: store['mlp'].__setitem__(l, pick(o).float())))
        if l in resid_layers:
            handles.append(layer.register_forward_hook(lambda m, a, o, l=l: store['resid'].__setitem__(l, pick(o).half())))
        if l in FULL_LAYERS:
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, l=l: store['head'].__setitem__(l, pick(a[0]).float())))
    try:
        rt.model(inputs_embeds=X, position_ids=P, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    head_dev = rt.model.lm_head.weight.device
    logits_all = rt.model.lm_head(store['normed'].to(head_dev)).float().to(X.device)  # (B, 2, V)
    lp_all = logits_all.log_softmax(-1)
    dig = torch.tensor(digit_ids(rt), device=X.device)
    W_U = rt.model.lm_head.weight[dig.to(head_dev)].float().to(X.device)           # (10, D)
    hf = store['final']
    scale = (1.0 + lm.norm.weight.float().to(X.device)) * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + lm.norm.eps)

    def dla(c):                                                    # (B, 2, ..., D) -> (B, 2, ..., 10)
        sc = scale.view(B, 2, *([1] * (c.dim() - 3)), -1)
        return (c * sc) @ W_U.T

    dla_mixer = torch.stack([dla(store['mixer'][l]) for l in range(1, N_LAYERS + 1)], 2)
    dla_mlp = torch.stack([dla(store['mlp'][l]) for l in range(1, N_LAYERS + 1)], 2)
    heads = []
    for l in FULL_LAYERS:
        attn = lm.layers[l - 1].self_attn
        hd = attn.head_dim
        z = store['head'][l].view(B, 2, N_HEADS, hd)                               # gated head outputs
        Wo = attn.o_proj.weight.float().to(X.device).view(-1, N_HEADS, hd)                       # (D, heads, hd)
        per_head = torch.einsum('bqkh,dkh->bqkd', z, Wo)                            # (B, 2, heads, D)
        heads.append(dla(per_head))
    top = lp_all.max(-1)
    return {
        'logp': lp_all[..., dig].cpu().numpy(),
        'lp_top': top.values.cpu().numpy(), 'top_id': top.indices.cpu().numpy(),
        'resid': torch.stack([store['resid'][l] for l in resid_layers], 2).cpu().numpy(),
        'dla_mixer': dla_mixer.cpu().numpy(), 'dla_mlp': dla_mlp.cpu().numpy(),
        'dla_head': torch.stack(heads, 2).cpu().numpy(), 'dla_embed': dla(store['embed']).cpu().numpy(),
        'logit': logits_all[..., dig].cpu().numpy(),
    }


# ----------------------------------------------------------------------------- digit bookkeeping

def hundreds(v: float) -> int:
    """First digit of a three-digit 0-1000 value (values are rounded first)."""
    return int(round(v)) // 100


def tens_of(v: float) -> int:
    return (int(round(v)) // 10) % 10


def first_and_second(v: float) -> tuple[int, int, int]:
    """(first digit, second digit, number of digits) of a rounded non-negative value."""
    s = str(int(round(v)))
    return int(s[0]), (int(s[1]) if len(s) > 1 else -1), len(s)


def dumps(obj) -> str:
    return json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


# ----------------------------------------------------------------------------- attention mass

@torch.inference_mode()
def attention_mass(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, queries: list[int], image: tuple[int, int] | None,
                   digit_pos: int, layers=FULL_LAYERS) -> dict:
    """Softmax attention of the full-attention layers at the query positions (one prefill, batch 1).

    Same recomputation as coordinate_readout.attention_readout, but every tensor stays on the device of its
    layer, so it also works when the weights are split over several GPUs. Returns per layer (heads, Q) arrays of
    the attention mass on the image span, on the token at digit_pos, and the recomputation error.
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
    out = {'image': [], 'digit': [], 'check': []}
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn
        h, (cos, sin), o_ref = store[layer]['h'], store[layer]['pe'], store[layer]['out']
        dev = h.device
        q_idx = torch.tensor(queries, device=dev)
        L, hd = h.shape[1], mod.head_dim
        qg = mod.q_proj(h[:, q_idx]).view(1, len(queries), -1, hd * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(1, L, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(1, L, -1, hd).transpose(1, 2)
        cos, sin = cos.to(dev), sin.to(dev)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, q_idx], sin[:, q_idx])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        k, v = repeat_kv(k, mod.num_key_value_groups), repeat_kv(v, mod.num_key_value_groups)
        scores = (q.float() @ k.float().transpose(2, 3)) * mod.scaling
        mask = torch.arange(L, device=dev)[None, :] > q_idx[:, None]
        a = scores.masked_fill(mask, float('-inf')).softmax(-1)[0]                    # (heads, Q, L)
        g = torch.sigmoid(gate.float()).transpose(1, 2)
        recon = mod.o_proj((((a[None] @ v.float()) * g).transpose(1, 2).reshape(1, len(queries), -1)).to(h.dtype))
        out['check'].append(float((recon[0] - o_ref[0, q_idx]).abs().max() / o_ref[0, q_idx].abs().max()))
        out['image'].append((a[:, :, image[0]:image[1]].sum(-1) if image else torch.zeros(a.shape[:2], device=dev)).cpu().numpy())
        out['digit'].append(a[:, :, digit_pos].cpu().numpy())
    return out
