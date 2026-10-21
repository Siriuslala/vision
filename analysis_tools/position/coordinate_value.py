"""Experiment 3.6: representation and conversion of coordinate values in Qwen3.5-9B.

Engine of the experiment (stage drivers live in `coordinate_value_run.py`):

* dense single-disc sweeps on a 512x512 canvas and their 0-1000 truth;
* the tokenizer audit of integers 0-1000 in several contexts;
* location of the answer "query positions" (the residual-stream positions whose next-token
  prediction is a coordinate digit);
* a teacher-forced capture pass recording residual streams, per-layer attention / MLP outputs
  and per-head attention outputs at the query positions;
* a prefix-continuation decoder that can force leading digits and add per-row steering vectors
  to one decoder layer at the positions that predict the digits of the steered number.

The point prompt, the disc renderer and the manual greedy decoder are reused from experiment 3.5
(`coordinate_readout.py`), so the two experiments share stimuli and decoding.
Layer numbers are 1-based; layer 0 denotes the input embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr

CANVAS = cr.CANVAS
DIAMETER = 64
GRID_STEP = 16                  # pixels between 2D grid centers
SWEEP_STEP = 2                  # pixels between 1D sweep centers (about 3.9 units of 0-1000)
EDGE = 32                       # smallest center coordinate, so the disc stays inside the canvas
POINT_PROMPT = cr.DISC_QUERY + cr.SUFFIX['point']
ROLES = ('last', 'x0', 'x1', 'x2', 'y0', 'y1', 'y2')     # query positions recorded per sample
COMPONENT_ROLES = ('x0', 'y0')                         # roles with per-layer attn/MLP and head outputs
FULL_LAYERS = cr.FULL_LAYERS
N_LAYERS = 32
N_HEADS = 16
HEAD_DIM = 256


# ----------------------------------------------------------------------------- stimuli

def to_unit(pixels: float, size: int = CANVAS) -> float:
    """Pixel coordinate -> 0-1000 normalized coordinate."""
    return 1000.0 * pixels / size


def disc_cases() -> list[dict]:
    """Dense single-disc cases: a 29x29 2D grid and two 225-step 1D sweeps through the canvas center."""
    cases = []
    centers = np.arange(EDGE, CANVAS - EDGE + 1, GRID_STEP)
    for cy in centers:
        for cx in centers:
            cases.append({'id': f'g2d_x{int(cx):03d}_y{int(cy):03d}', 'set': 'grid2d', 'center': [float(cx), float(cy)]})
    for axis in ('h', 'v'):
        for t in np.arange(EDGE, CANVAS - EDGE + 1, SWEEP_STEP):
            c = [float(t), CANVAS / 2] if axis == 'h' else [CANVAS / 2, float(t)]
            cases.append({'id': f'sweep_{axis}_{int(t):03d}', 'set': f'sweep_{axis}', 'center': c})
    for c in cases:
        c.update({'diameter': DIAMETER, 'canvas': CANVAS, 'format': 'point'})
    return cases


def disc_input(case: dict):
    image, truth = cr.disc_image(tuple(case['center']), case['diameter'], case['canvas'])
    return image, truth, POINT_PROMPT


def truth_units(truth: dict) -> tuple[float, float]:
    """Disc (or T6 target) center in 0-1000 units, (x, y)."""
    if 'target_center' in truth:
        w, h = truth['canvas']
        return to_unit(truth['target_center'][0], w), to_unit(truth['target_center'][1], h)
    w, h = truth['canvas']
    return to_unit(truth['center'][0], w), to_unit(truth['center'][1], h)


# ----------------------------------------------------------------------------- tokenizer audit

AUDIT_CONTEXTS = {
    'bare': '{n}',
    'leading_space': ' {n}',
    'json_x': '[{{"point_2d": [{n}, 500]}}]',
    'json_y': '[{{"point_2d": [500, {n}]}}]',
    'json_bbox_last': '[{{"bbox_2d": [10, 20, 30, {n}]}}]',
    'rowcol': '{{"row": {n}, "column": 3}}',
}


def number_pieces(tokenizer, context: str, n: int) -> list[str]:
    """Token strings that overlap the characters of the number n inside the formatted context."""
    text = context.format(n=n)
    start = text.index(str(n)) if context != '{n}' else 0
    if context.startswith(' '):
        start = 1
    end = start + len(str(n))
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tokenizer.decode([i]) for i, (a, b) in zip(enc['input_ids'], enc['offset_mapping']) if a < end and b > start]


def audit_tokenizer(tokenizer, values=range(1001)) -> dict:
    """For every context, count numbers whose overlapping tokens are not exactly one digit each."""
    out = {}
    for name, context in AUDIT_CONTEXTS.items():
        bad, examples = [], {}
        for n in values:
            pieces = number_pieces(tokenizer, context, n)
            if pieces != list(str(n)):
                bad.append(n)
            if n in (0, 7, 42, 357, 500, 1000):
                examples[n] = pieces
        full = tokenizer(context.format(n=357), add_special_tokens=False)['input_ids']
        out[name] = {'context': context, 'n_values': len(list(values)), 'n_not_one_digit_per_token': len(bad),
                     'violations': bad[:20], 'examples': examples,
                     'tokens_357': [tokenizer.decode([i]) for i in full]}
    return out


# ----------------------------------------------------------------------------- query positions

def digit_spans(rt: qr.Runtime, ids: list[int]) -> list[list[int]] | None:
    """Answer-token indices of the digits of x and y in a point answer, or None if unparseable."""
    spans = cr.coordinate_tokens(rt, ids, 'point')
    if len(spans) != 2 or any(not s for s in spans):
        return None
    return spans


def query_positions(prompt_len: int, spans: list[list[int]]) -> dict[str, int]:
    """Sequence index of every role; the hidden state at index i predicts token i + 1.

    Answer token j sits at sequence index prompt_len + j, so the state predicting it is at
    prompt_len + j - 1. Roles x2 / y2 exist only for three-digit (or longer) values.
    """
    out = {'last': prompt_len - 1}
    for axis, toks in zip('xy', spans):
        for k, j in enumerate(toks[:3]):
            out[f'{axis}{k}'] = prompt_len + j - 1
    return out


def value_of(text: str) -> int:
    return int(re.sub(r'\D', '', text))


# ----------------------------------------------------------------------------- capture pass

@dataclass
class Capture:
    """Arrays of one teacher-forced pass (float16 on CPU).

    resid:   (33, R, D) residual stream at the R = len(ROLES) roles; index 0 is the embedding,
             index l is the output of decoder layer l. Missing roles are filled with NaN.
    attn:    (32, 2, D) attention (or linear-attention) block output at roles x0, y0.
    mlp:     (32, 2, D) MLP block output at roles x0, y0.
    heads:   (8, 2, 16, 256) per-head input of o_proj (after the output gate) at full-attention layers.
    logp:    (R, 10) log-probabilities of the ten digit tokens at every role.
    next_logp: (R,) log-probability of the token actually generated next at every role.
    """
    resid: np.ndarray
    attn: np.ndarray
    mlp: np.ndarray
    heads: np.ndarray
    logp: np.ndarray
    next_logp: np.ndarray


def digit_token_ids(rt: qr.Runtime) -> list[int]:
    return [qr.token_id(rt, str(d)) for d in range(10)]


@torch.inference_mode()
def capture(rt: qr.Runtime, prep: cr.Prepared, answer_ids: list[int], roles: dict[str, int],
            visual_layers: tuple[int, ...] = ()) -> tuple[Capture, dict[int, np.ndarray]]:
    """Teacher-forced forward on prompt + answer; record states at the role positions.

    visual_layers: decoder layers whose image-token residuals are also returned (Nv, D) float16.
    """
    x, pos = cr.teacher_forced(rt, prep, answer_ids)
    index = torch.tensor([roles.get(r, -1) for r in ROLES])
    present = index >= 0
    comp_index = torch.tensor([roles[r] for r in COMPONENT_ROLES])
    D = x.shape[-1]
    resid = torch.full((N_LAYERS + 1, len(ROLES), D), float('nan'))
    attn = torch.zeros(N_LAYERS, 2, D)
    mlp = torch.zeros(N_LAYERS, 2, D)
    heads = torch.zeros(len(FULL_LAYERS), 2, N_HEADS, HEAD_DIM)
    vis = {}
    s, e = prep.span['start'], prep.span['end']
    handles = []

    def grab(h):
        return (h[0] if isinstance(h, tuple) else h)[0]

    def on_input(module, args, kwargs):
        h = kwargs.get('hidden_states', args[0] if args else None)
        resid[0, present] = h[0, index[present]].float().cpu()

    handles.append(rt.layers[0].register_forward_pre_hook(on_input, with_kwargs=True))
    for l, layer in enumerate(rt.layers, start=1):
        def on_layer(module, args, output, l=l):
            h = grab(output)
            resid[l, present] = h[index[present]].float().cpu()
            if l in visual_layers:
                vis[l] = h[s:e].to(torch.float16).cpu().numpy()
        handles.append(layer.register_forward_hook(on_layer))
        mixer = layer.self_attn if hasattr(layer, 'self_attn') else layer.linear_attn
        handles.append(mixer.register_forward_hook(
            lambda m, a, o, l=l: attn.__setitem__(l - 1, grab(o)[comp_index].float().cpu())))
        handles.append(layer.mlp.register_forward_hook(
            lambda m, a, o, l=l: mlp.__setitem__(l - 1, grab(o)[comp_index].float().cpu())))
        if l in FULL_LAYERS:
            k = FULL_LAYERS.index(l)
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, k=k: heads.__setitem__(k, a[0][0][comp_index].float().cpu().view(2, N_HEADS, HEAD_DIM))))
    try:
        logits = rt.model(inputs_embeds=x, position_ids=pos).logits[0].float()
    finally:
        for h in handles:
            h.remove()
    logp_all = logits.log_softmax(-1)
    digits = digit_token_ids(rt)
    seq = list(prep.ids.tolist()) + list(answer_ids)
    logp = np.full((len(ROLES), 10), np.nan, np.float32)
    nxt = np.full(len(ROLES), np.nan, np.float32)
    for r, i in enumerate(index.tolist()):
        if i >= 0:
            logp[r] = logp_all[i, digits].cpu().numpy()
            nxt[r] = float(logp_all[i, seq[i + 1]]) if i + 1 < len(seq) else np.nan
    cap = Capture(resid.half().numpy(), attn.half().numpy(), mlp.half().numpy(), heads.half().numpy(), logp, nxt)
    return cap, vis


# ----------------------------------------------------------------------------- prefix continuation with steering

class Steerer:
    """Adds a per-row vector to the last position of one decoder layer's output.

    The decoding loop sets `delta` (B, D) before each forward call; rows with a zero vector are
    unchanged. During the prefill the last position is the prompt's final token, during decoding
    it is the single new token, so the vector always lands on the state that predicts the next token.
    """

    def __init__(self, rt: qr.Runtime, layer: int | None):
        self.delta = None
        self.handle = None
        if layer is not None:
            self.handle = rt.layers[layer - 1].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self.delta is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h[:, -1] += self.delta.to(h.device, h.dtype)
        return output

    def close(self):
        if self.handle is not None:
            self.handle.remove()


@torch.inference_mode()
def continue_number(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, steer: Steerer | None = None,
                    deltas: torch.Tensor | None = None, max_digits: int = 5) -> list[dict]:
    """Greedy continuation of a number whose prefix (possibly with forced digits) ends the input.

    x: (B, L, D) inputs_embeds; pos: (4, B, L) position ids (equal length, no padding).
    deltas: optional (B, K, D) steering vectors; step k of the number (the state predicting its
            k-th generated token) receives deltas[:, min(k, K-1)]. Steering stops for a row once
            that row emits a non-digit token.
    Returns, per row, the generated digits (as a string), the terminating token and the
    log-probabilities of the ten digit tokens at the first step (digit order 0-9).
    """
    B, L, _ = x.shape
    digit_set = {qr.token_id(rt, str(d)): str(d) for d in range(10)}
    embed = rt.model.model.get_input_embeddings()
    digits = [''] * B
    stop = [None] * B
    active = torch.ones(B, dtype=torch.bool)

    def set_delta(k):
        if steer is not None and deltas is not None:
            d = deltas[:, min(k, deltas.shape[1] - 1)].clone()
            d[~active] = 0
            steer.delta = d

    set_delta(0)
    out = rt.model(inputs_embeds=x, position_ids=pos, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    nxt_mrope = pos[1:, :, -1] + 1
    first = out.logits[:, -1].float().log_softmax(-1)[:, list(digit_set)].cpu().numpy()
    for step in range(max_digits + 1):
        tok = out.logits[:, -1].argmax(-1).tolist()
        for b in range(B):
            if active[b]:
                if tok[b] in digit_set and step < max_digits:
                    digits[b] += digit_set[tok[b]]
                else:
                    stop[b] = rt.tokenizer.decode([tok[b]])
                    active[b] = False
        if not bool(active.any()):
            break
        set_delta(step + 1)
        p = torch.cat([torch.full((1, B), L + step, device=x.device, dtype=pos.dtype), nxt_mrope + step], 0)[:, :, None]
        out = rt.model(inputs_embeds=embed(torch.tensor(tok, device=x.device)[:, None]), position_ids=p,
                       past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    if steer is not None:
        steer.delta = None
    return [{'digits': d, 'stop': s, 'first_logp': f.tolist()} for d, s, f in zip(digits, stop, first)]


def prefix_inputs(rt: qr.Runtime, prep: cr.Prepared, prefix_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """inputs_embeds and position ids of prompt + a forced answer prefix (all prefix tokens are inputs)."""
    if not prefix_ids:
        return cr.embeds(rt, prep), prep.pos
    return cr.teacher_forced(rt, prep, list(prefix_ids))


# ----------------------------------------------------------------------------- digit bookkeeping

def leading_digits(value: int, k: int) -> list[int]:
    """First k digits of a non-negative integer as written without padding."""
    return [int(c) for c in str(value)[:k]]


def coarse_digit(value: float) -> int:
    """Hundreds digit of a 0-1000 value (0 for values below 100, 10 for 1000)."""
    return int(round(value)) // 100


class SpanPatcher:
    """Replaces one decoder layer's output on a contiguous position range during the prefill only.

    values: (B, P, D) replacement for positions [start, start + P); decoding steps (sequence length 1)
    pass through unchanged.
    """

    def __init__(self, rt: qr.Runtime, layer: int, start: int, values: torch.Tensor):
        self.start, self.values = start, values
        self.handle = rt.layers[layer - 1].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < self.start + self.values.shape[1]:
            return output
        h[:, self.start:self.start + self.values.shape[1]] = self.values.to(h.device, h.dtype)
        return output

    def close(self):
        self.handle.remove()


@torch.inference_mode()
def layer_states(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, layers: tuple[int, ...], start: int) -> dict:
    """Outputs of the given decoder layers on positions [start, L) for one prefill (1, L, D)."""
    store, handles = {}, []
    for l in layers:
        def hook(m, a, o, l=l):
            h = o[0] if isinstance(o, tuple) else o
            store[l] = h[0, start:].detach().clone()
        handles.append(rt.layers[l - 1].register_forward_hook(hook))
    try:
        rt.model(inputs_embeds=x, position_ids=pos, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return store
