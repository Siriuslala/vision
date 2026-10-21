"""Experiment 3.6 round 2: digit-level output mechanism of coordinate values in Qwen3.5-9B.

Engine of the round (stage drivers live in `coordinate_value_r2_run.py`):

* numeric-format variants of the single-disc point task (0-1000, 0-100, 0-10, pixels, two-decimal
  fractions, text-to-0-1000 conversion), dot counting and RefCOCO point questions, each with the
  string of the true answer so that a teacher-forced pass can read the digit distribution of every
  digit given the true preceding digits;
* digit-position bookkeeping for arbitrary numeric answers (integers and decimals);
* a right-padded batched forward that returns the next-token log-probabilities and residuals at
  selected positions without materializing full-vocabulary logits for every position;
* row-wise interventions at chosen positions (mean ablation, value replacement and cross-row copy)
  on MLP outputs, MLP neurons, token-mixer outputs and single attention heads, with recording of the
  same sites, used for component ablation, path patching and neuron ablation.

Layer numbers are 1-based; all interventions act during the prefill only.
Design: docs/analysis/position/design_coordinate_value.md, section "第二轮".
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np
import torch
from PIL import Image, ImageDraw

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr

CANVAS = cr.CANVAS
AXIS_NOTE = ' x increases rightward and y increases downward.'
POINT_HEAD = cr.DISC_QUERY + ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].'

# Numeric formats of the disc point task. `scale` maps a 0-1000 truth to the requested number; `fmt`
# writes the true answer string. Only the coordinate note differs from the experiment-3.5 prompt.
FORMAT_NOTES = {
    's1000': ' All coordinates use 0-1000 normalization;' + AXIS_NOTE,
    's100': (' Use percentage coordinates from 0 to 100 instead of 0-1000: the image center is [50, 50] and the'
             ' bottom-right corner is [100, 100];' + AXIS_NOTE),
    's10': (' Use integer coordinates from 0 to 10 instead of 0-1000: the image center is [5, 5] and the'
            ' bottom-right corner is [10, 10];' + AXIS_NOTE),
    'pix': (' Use absolute pixel coordinates instead of 0-1000: the image is 512x512 pixels, its center is'
            ' [256, 256] and the bottom-right corner is [512, 512];' + AXIS_NOTE),
    'frac': (' Give each coordinate as a fraction of the image width or height with exactly two decimal places'
             ' (for example 0.37);' + AXIS_NOTE),
}
TEXT_QUERY = ('There is one red circle in the image. Its center is at pixel ({px}, {py}) of this 512x512 image.'
              ' Convert this point to 0-1000 normalization. Return only [{{"point_2d": [x, y]}}].'
              + FORMAT_NOTES['s1000'] + cr.JSON_ONLY)
COUNT_QUERY = 'How many red circles are in the image? Answer with only the number.'
COUNT_DIAMETER = 24
COUNT_MIN_GAP = 6          # minimum free pixels between two counting discs


def format_prompt(fmt: str) -> str:
    return POINT_HEAD + FORMAT_NOTES[fmt] + cr.JSON_ONLY


def true_string(fmt: str, t: float) -> str:
    """True answer string of a 0-1000 coordinate t in the given numeric format."""
    if fmt in ('s1000', 'text'):
        return str(int(round(t)))
    if fmt == 's100':
        return str(int(round(t / 10)))
    if fmt == 's10':
        return str(int(round(t / 100)))
    if fmt == 'pix':
        return str(int(round(t * CANVAS / 1000)))
    if fmt == 'frac':
        return f'{t / 1000:.2f}'
    raise ValueError(f'Unknown format {fmt}')


def scale_of(fmt: str) -> float:
    """Units of the format per 0-1000 unit (used to express digit bin widths in 0-1000 units)."""
    return {'s1000': 1.0, 'text': 1.0, 's100': 0.1, 's10': 0.01, 'pix': CANVAS / 1000, 'frac': 1.0}[fmt]


# ----------------------------------------------------------------------------- counting stimuli

def count_image(n: int, seed: int, size: int = CANVAS, d: int = COUNT_DIAMETER) -> tuple[Image.Image, list]:
    """n non-overlapping red discs at random positions on a white canvas (rejection sampling)."""
    rng = np.random.default_rng(seed)
    centers: list[tuple[float, float]] = []
    tries = 0
    while len(centers) < n:
        tries += 1
        if tries > 200000:
            raise RuntimeError(f'Cannot place {n} discs')
        c = rng.uniform(d / 2 + 2, size - d / 2 - 2, 2)
        if all((c[0] - a) ** 2 + (c[1] - b) ** 2 >= (d + COUNT_MIN_GAP) ** 2 for a, b in centers):
            centers.append((float(c[0]), float(c[1])))
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    for x, y in centers:
        draw.ellipse([x - d / 2, y - d / 2, x + d / 2, y + d / 2], fill=cr.RED)
    return img, centers


# ----------------------------------------------------------------------------- numeric answer bookkeeping

NUMBER = re.compile(r'\d+(?:\.\d+)?')


def point_numbers(text: str) -> list[tuple[int, int]]:
    """Character spans of the numbers inside the first point_2d list, or [] if absent."""
    m = re.search(r'"point_2d":\s*\[([^\]]*)\]', text)
    if not m:
        return []
    return [(m.start(1) + n.start(), m.start(1) + n.end()) for n in NUMBER.finditer(m.group(1))]


def ideal_point_text(model_text: str, values: list[str]) -> tuple[str, bool]:
    """Model answer with the point numbers replaced by the true strings; canonical template if unparseable."""
    spans = point_numbers(model_text)
    if len(spans) == 2:
        (a0, b0), (a1, b1) = spans
        return model_text[:a0] + values[0] + model_text[b0:a1] + values[1] + model_text[b1:], True
    return '```json\n[\n\t{"point_2d": [' + values[0] + ', ' + values[1] + ']}\n]\n```', False


@dataclass
class DigitSlot:
    """One digit of a numeric answer: the state at `pos` predicts this digit (sequence index)."""
    number: int          # index of the number in the answer (0 = x, 1 = y; 0 for counting)
    k: int               # index among the digits of the number (0 = first written digit)
    n_digits: int        # number of digits in the number
    digit: int           # true digit
    after_point: bool    # digit lies after the decimal point
    pos: int             # sequence index of the predicting state


def digit_slots(tokenizer, prompt_len: int, answer_text: str, spans: list[tuple[int, int]]) -> tuple[list[int], list[DigitSlot]]:
    """Tokenize the answer and locate every digit of the given number spans.

    Raises if a digit is not a single token of its own, which would break the digit-wise readout.
    """
    enc = tokenizer(answer_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    slots = []
    for n, (a, b) in enumerate(spans):
        s = answer_text[a:b]
        nd = sum(c.isdigit() for c in s)
        k = 0
        for j, ch in enumerate(s):
            if not ch.isdigit():
                continue
            c = a + j
            tok = [i for i, (x, y) in enumerate(offs) if x <= c < y]
            if len(tok) != 1 or offs[tok[0]] != (c, c + 1):
                raise ValueError(f'Digit {ch!r} of {s!r} is not a single token in {answer_text!r}')
            slots.append(DigitSlot(n, k, nd, int(ch), '.' in s[:j], prompt_len + tok[0] - 1))
            k += 1
    return ids, slots


# ----------------------------------------------------------------------------- batched forward with selection

def pad_batch(xs: list[torch.Tensor], poss: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad inputs_embeds (1, L_i, D) and position ids (4, 1, L_i) to a common length.

    Causal attention and the causal Gated DeltaNet recurrence make right padding invisible to every
    real position, so no attention mask is needed.
    """
    L = max(x.shape[1] for x in xs)
    X, P = [], []
    for x, p in zip(xs, poss):
        n = L - x.shape[1]
        if n:
            x = torch.cat([x, x.new_zeros(1, n, x.shape[2])], 1)
            step = torch.arange(1, n + 1, device=p.device, dtype=p.dtype).view(1, 1, n)
            p = torch.cat([p, p[:, :, -1:] + step], 2)
        X.append(x)
        P.append(p)
    return torch.cat(X, 0), torch.cat(P, 1)


@torch.inference_mode()
def select_forward(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, select: list[list[int]],
                   resid_layers: tuple[int, ...] = (), neuron_layers: tuple[int, ...] = ()) -> list[dict]:
    """One batched forward; per row, outputs at the selected sequence indices.

    Returns per row: 'logp' (S, V) float32 log-probabilities of the next token, 'resid' {layer: (S, D)}
    float16 decoder-layer outputs, 'neurons' {layer: (S, F)} float16 MLP neuron activations (input of
    down_proj).
    """
    B = x.shape[0]
    flat_b = torch.tensor([b for b in range(B) for _ in select[b]])
    flat_p = torch.tensor([p for b in range(B) for p in select[b]])
    store: dict = {'resid': {}, 'neurons': {}}
    handles = []
    norm = rt.model.model.language_model.norm
    handles.append(norm.register_forward_hook(lambda m, a, o: store.__setitem__('normed', o[flat_b, flat_p])))
    for l in resid_layers:
        handles.append(rt.layers[l - 1].register_forward_hook(
            lambda m, a, o, l=l: store['resid'].__setitem__(l, (o[0] if isinstance(o, tuple) else o)[flat_b, flat_p].half().cpu())))
    for l in neuron_layers:
        handles.append(rt.layers[l - 1].mlp.down_proj.register_forward_pre_hook(
            lambda m, a, l=l: store['neurons'].__setitem__(l, a[0][flat_b, flat_p].half().cpu())))
    try:
        rt.model(inputs_embeds=x, position_ids=pos, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    logp = rt.model.lm_head(store['normed']).float().log_softmax(-1).cpu()
    out, i = [], 0
    for b in range(B):
        n = len(select[b])
        out.append({'logp': logp[i:i + n], 'resid': {l: v[i:i + n] for l, v in store['resid'].items()},
                    'neurons': {l: v[i:i + n] for l, v in store['neurons'].items()}})
        i += n
    return out


# ----------------------------------------------------------------------------- interventions

@dataclass
class Op:
    """One row-wise edit applied during the prefill.

    site:  'mlp' (MLP block output), 'mixer' (token-mixer block output), 'head' (one attention head's
           slice of the o_proj input) or 'neuron' (entries of the down_proj input).
    pos:   sequence index, or (start, end) range of indices.
    value: tensor broadcastable to the edited slice (replacement), or an int source row whose
           unedited slice at the same positions is copied (activation patching).
    index: head number for 'head'; neuron indices (LongTensor) for 'neuron'.
    """
    row: int
    layer: int
    site: str
    pos: int | tuple[int, int]
    value: torch.Tensor | int
    index: int | torch.Tensor | None = None


@dataclass
class Record:
    """Sites to record at one sequence index per row (after any edit at the same site)."""
    pos: list[int]
    mlp: tuple[int, ...] = ()
    mixer: tuple[int, ...] = ()
    heads: tuple[int, ...] = ()          # full-attention layers whose 16 head slices are recorded
    data: dict = field(default_factory=dict)


def _slice(pos):
    return slice(pos, pos + 1) if isinstance(pos, int) else slice(pos[0], pos[1])


class Interventions:
    """Installs row-wise edits and recordings on one model for the duration of a `with` block."""

    def __init__(self, rt: qr.Runtime, ops: list[Op], record: Record | None = None):
        self.handles = []
        by: dict = {}
        for op in ops:
            by.setdefault((op.layer, op.site), []).append(op)
        rec = record
        sites = set(by)
        if rec is not None:
            sites |= {(l, 'mlp') for l in rec.mlp} | {(l, 'mixer') for l in rec.mixer} | {(l, 'head') for l in rec.heads}
        for layer, site in sites:
            lops = by.get((layer, site), [])
            block = rt.layers[layer - 1]
            if site == 'mlp':
                self.handles.append(block.mlp.register_forward_hook(self._post(lops, rec, ('mlp', layer))))
            elif site == 'mixer':
                mixer = block.self_attn if hasattr(block, 'self_attn') else block.linear_attn
                self.handles.append(mixer.register_forward_hook(self._post(lops, rec, ('mixer', layer))))
            elif site == 'head':
                proj, hd = cr.mixer_projection(rt, layer)
                self.handles.append(proj.register_forward_pre_hook(self._pre(lops, rec, ('head', layer), hd)))
            elif site == 'neuron':
                self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self._pre(lops, None, None, None)))
            else:
                raise ValueError(f'Unknown site {site}')

    @staticmethod
    def _apply(h: torch.Tensor, lops: list[Op], hd: int | None):
        src = h.clone() if any(isinstance(o.value, int) for o in lops) else h
        for o in lops:
            sl = _slice(o.pos)
            if o.site == 'head':
                ch = slice(o.index * hd, (o.index + 1) * hd)
            elif o.site == 'neuron':
                ch = o.index.to(h.device)
            else:
                ch = slice(None)
            if isinstance(o.value, int):
                h[o.row, sl, ch] = src[o.value, sl, ch]
            else:
                h[o.row, sl, ch] = o.value.to(h.device, h.dtype)

    def _post(self, lops, rec, key):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == 1:
                return output
            if lops:
                self._apply(h, lops, None)
            if rec is not None and key[1] in (rec.mlp if key[0] == 'mlp' else rec.mixer):
                rec.data[key] = h[torch.arange(h.shape[0]), torch.tensor(rec.pos)].float().cpu()
            return output
        return hook

    def _pre(self, lops, rec, key, hd):
        def hook(module, args):
            x = args[0]
            if x.shape[1] == 1:
                return None
            y = x.clone()
            if lops:
                self._apply(y, lops, hd)
            if rec is not None and key is not None and key[1] in rec.heads:
                v = y[torch.arange(y.shape[0]), torch.tensor(rec.pos)].float().cpu()
                rec.data[key] = v.view(v.shape[0], -1, hd)
            return (y,)
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def digit_logits_at(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, at: list[int], digit_ids: list[int],
                    ops: list[Op] = (), record: Record | None = None) -> torch.Tensor:
    """(B, 10) float32 digit log-probabilities (full-vocabulary softmax) at index at[b] of each row."""
    store = {}
    norm = rt.model.model.language_model.norm
    b_idx = torch.arange(x.shape[0])
    h = norm.register_forward_hook(lambda m, a, o: store.__setitem__('n', o[b_idx, torch.tensor(at)]))
    try:
        with Interventions(rt, list(ops), record):
            rt.model(inputs_embeds=x, position_ids=pos, logits_to_keep=1)
    finally:
        h.remove()
    return rt.model.lm_head(store['n']).float().log_softmax(-1)[:, digit_ids].cpu()
