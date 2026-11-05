"""Experiment 7.1: interpretability-guided optimisation of Qwen3.5-9B (model-side components and data helpers).

Design: docs/analysis/application/design_interp_guided_optimization.md.

Components piloted (all general, none task specific):

* (A) ``image_bidirectional``: inside every image span, the full-attention layers (4, 8, ..., 32) use a
  bidirectional mask during prefill (prefix-LM style for image tokens only); text keeps the causal mask and
  the Gated DeltaNet layers are untouched.  No parameter is added.
* (E) ``ContrastAdapter``: a centre-surround channel added to the merger output.  For merged visual token
  v_i, c_i = v_i - mean of its k x k token neighbourhood; the adapter adds B phi(A c_i / s), where
  phi(z) = [GELU(z), GELU(-z)] keeps both signs of the deviation (an absent feature gives a negative
  deviation that a rectified presence detector would discard) and B starts at zero.
* (C) ``PriorContrastProcessor``: training-free prior-subtracted decoding.  The batch holds every item twice,
  with the real image and with a blank image of the same size and mean colour; the next-token distribution is
  (1 + alpha) log p(. | image) - alpha log p(. | blank), restricted to tokens that are plausible under the
  real image (VCD-style adaptive plausibility, Leng et al. 2024).

The LoRA itself (hook based, all decoder linear layers) is reused from experiment 3.13
(``analysis_tools.position.targeted_finetune``).  Layer numbers are 1-based.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------------- (A) image-block bidirectional mask


def image_segments(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """(B, L) long: 0 for non-image tokens, k >= 1 for the k-th contiguous image-token run of each row."""
    img = input_ids == image_token_id
    start = img.clone()
    start[:, 1:] &= ~img[:, :-1]
    return torch.cumsum(start.long(), dim=1) * img.long()


def prefill_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor | None, image_token_id: int,
                 bidirectional: bool = True) -> torch.Tensor:
    """Boolean (B, 1, L, L) mask for a prefill without past keys (True = may attend).

    Causal attention restricted to non-padding keys; if ``bidirectional``, every image token may additionally
    attend to every token of its own image (raster-later ones included).  The diagonal is always allowed so
    that left-padding queries never see an empty row (their outputs are discarded anyway).
    """
    B, L = input_ids.shape
    dev = input_ids.device
    valid = torch.ones(B, L, dtype=torch.bool, device=dev) if attention_mask is None else attention_mask.bool()
    m = torch.ones(L, L, dtype=torch.bool, device=dev).tril()[None] & valid[:, None, :]
    if bidirectional:
        seg = image_segments(input_ids, image_token_id)
        m = m | ((seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] > 0))
    m = m | torch.eye(L, dtype=torch.bool, device=dev)[None]
    return m[:, None]


def full_attention_layers(rt) -> list[int]:
    """1-based indices of the full-attention decoder layers."""
    return [i + 1 for i, layer in enumerate(rt.layers) if hasattr(layer, 'self_attn')]


@contextmanager
def image_bidirectional(rt, layers: list[int] | None = None):
    """Install the image-block bidirectional prefill mask on the given full-attention layers (default: all).

    A pre-hook on the multimodal model records input_ids and the 2-D padding mask of every call with more than
    one new token (prefill or training forward); a pre-hook on each selected attention module swaps in the
    corresponding 4-D mask.  Single-token decoding steps keep the default mask (a text query attends to every
    earlier token in both conditions).  The state is kept until the next prefill, so activation recomputation
    under gradient checkpointing sees the same mask.
    """
    layers = layers or full_attention_layers(rt)
    tok_id = rt.model.config.image_token_id
    state: dict = {}

    def top_pre(module, args, kwargs):
        ids = kwargs.get('input_ids')
        if ids is None and args:
            ids = args[0]
        if ids is None:
            raise RuntimeError('image_bidirectional needs input_ids in the model call')
        if ids.shape[1] > 1:
            state['mask'] = prefill_mask(ids, kwargs.get('attention_mask'), tok_id)
        return None

    def attn_pre(module, args, kwargs):
        h = kwargs.get('hidden_states', args[0] if args else None)
        m = state.get('mask')
        if h is None or h.shape[1] <= 1 or m is None:
            return None
        if m.shape[-1] != h.shape[1]:
            raise RuntimeError(f'Stored mask length {m.shape[-1]} != sequence length {h.shape[1]}')
        kwargs['attention_mask'] = m.to(h.device)
        return args, kwargs

    handles = [rt.model.model.register_forward_pre_hook(top_pre, with_kwargs=True)]
    for l in layers:
        attn = rt.layers[l - 1].self_attn
        handles.append(attn.register_forward_pre_hook(attn_pre, with_kwargs=True))
    try:
        yield state
    finally:
        for h in handles:
            h.remove()


# ----------------------------------------------------------------------------- (E) centre-surround contrast adapter


def local_mean(v: torch.Tensor, grid: tuple[int, int], k: int) -> torch.Tensor:
    """Mean of each token's k x k neighbourhood (zero padding excluded). v: (H*W, d) raster order -> (H*W, d)."""
    H, W = grid
    x = v.float().t().reshape(1, -1, H, W)
    m = nn.functional.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)
    return m.reshape(-1, H * W).t()


class ContrastAdapter(nn.Module):
    """e_i = B [GELU(A c_i / s), GELU(-A c_i / s)] with c_i = v_i - local_mean(v)_i and s the RMS of v over the image.

    A: (r, d) Kaiming-uniform, B: (d, 2r) zeros (the adapted model equals the base model at step 0).
    Parameters are float32 and the result is cast back to the merger dtype.
    """

    def __init__(self, d: int, rank: int = 64, k: int = 5, device=None):
        super().__init__()
        self.k, self.rank = k, rank
        a = torch.empty(rank, d, device=device)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.A = nn.Parameter(a)
        self.B = nn.Parameter(torch.zeros(d, 2 * rank, device=device))
        self.last: list[torch.Tensor] = []          # per-image ||e_i|| of the last call (mechanism check)

    def forward(self, v: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        c = v.float() - local_mean(v, grid, self.k)
        s = v.float().pow(2).mean().sqrt().clamp_min(1e-6)
        z = (c / s) @ self.A.t()
        e = torch.cat([nn.functional.gelu(z), nn.functional.gelu(-z)], -1) @ self.B.t()
        self.last.append(e.detach().norm(dim=-1).cpu())
        return v + e.to(v.dtype)

    def state(self) -> dict:
        return {'rank': self.rank, 'k': self.k, 'A': self.A.detach().cpu(), 'B': self.B.detach().cpu()}

    def load(self, st: dict) -> None:
        if st['rank'] != self.rank or st['k'] != self.k:
            raise ValueError('Contrast adapter state does not match (rank or window differ)')
        with torch.no_grad():
            self.A.copy_(st['A'])
            self.B.copy_(st['B'])


@contextmanager
def contrast_channel(rt, adapter: ContrastAdapter):
    """Apply the adapter to the merged visual tokens of every image (forward hook on the vision tower)."""
    visual = rt.model.model.visual
    merge = visual.spatial_merge_size

    def hook(module, args, kwargs, output):
        grid_thw = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
        if grid_thw is None:
            raise RuntimeError('contrast_channel needs grid_thw')
        adapter.last = []
        parts, start = [], 0
        for t, h, w in grid_thw.tolist():
            H, W = h // merge, w // merge
            n = t * H * W
            if t != 1:
                raise ValueError('contrast_channel supports single-frame images only')
            parts.append(adapter(output.pooler_output[start:start + n], (H, W)))
            start += n
        if start != output.pooler_output.shape[0]:
            raise RuntimeError('Merged token count does not match grid_thw')
        output.pooler_output = torch.cat(parts, 0)
        return output

    handle = visual.register_forward_hook(hook, with_kwargs=True)
    try:
        yield adapter
    finally:
        handle.remove()


# ----------------------------------------------------------------------------- (C) prior-subtracted decoding


class PriorContrastProcessor:
    """Logits processor for a doubled batch [real images; blank images] (rows i and i + B are one item).

    score = (1 + alpha) log p_real - alpha log p_blank on tokens with p_real >= beta * max p_real, -inf elsewhere.
    ``token_ids`` restricts the adjustment to a vocabulary subset (e.g. digit tokens); other entries keep
    log p_real.  Both rows of a pair receive the same scores, so greedy decoding emits the same token for both.
    """

    def __init__(self, alpha: float, beta: float, token_ids: list[int] | None = None):
        self.alpha, self.beta = alpha, beta
        self.token_ids = None if token_ids is None else torch.tensor(sorted(token_ids))

    def combine(self, real: torch.Tensor, blank: torch.Tensor) -> torch.Tensor:
        lv = real.float().log_softmax(-1)
        lb = blank.float().log_softmax(-1)
        cd = (1 + self.alpha) * lv - self.alpha * lb
        plausible = lv >= lv.max(-1, keepdim=True).values + math.log(self.beta)
        cd = cd.masked_fill(~plausible, float('-inf'))
        if self.token_ids is None:
            return cd
        out = lv.clone()
        idx = self.token_ids.to(lv.device)
        out[:, idx] = cd[:, idx]
        return out

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        n = scores.shape[0]
        if n % 2:
            raise ValueError('PriorContrastProcessor expects a doubled batch')
        b = n // 2
        if not torch.equal(input_ids[:b, -1], input_ids[b:, -1]):
            raise RuntimeError('Paired rows diverged during decoding')
        s = self.combine(scores[:b], scores[b:])
        return torch.cat([s, s], 0).to(scores.dtype)


def blank_like(image):
    """Uniform image of the same size filled with the image's mean colour (the 4.4 'Blank' construction)."""
    from PIL import Image
    arr = np.asarray(image.convert('RGB'), dtype=np.float64)
    colour = tuple(int(round(c)) for c in arr.reshape(-1, 3).mean(0))
    return Image.new('RGB', image.size, colour)


# ----------------------------------------------------------------------------- samples and supervision

ROWCOL_TEMPLATE = '```json\n{{"row": {r}, "column": {c}}}\n```'      # the base model's own direct-answer format
SHORT_TRAIN = '\nAnswer the question using a single word or phrase.'


@dataclass
class Item:
    """One training or evaluation item.

    task:   'odd' (row/column of the odd entity), 'vqa' (short answer), 'count' (rows/columns of a grid) or
            'coord' (3.13 point / bbox items).
    answer: supervised answer text (training) or '' (evaluation).
    truth:  task-specific ground truth (odd: {'target': [r, c], 'grid': g}; count: {'rows', 'cols'}; ...).
    """
    id: str
    task: str
    split: str
    image: str
    prompt: str
    answer: str = ''
    truth: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def dump_items(path: Path, items: list[Item]) -> None:
    path.write_text(''.join(json.dumps(it.__dict__) + '\n' for it in items))


def read_items(path: Path) -> list[Item]:
    return [Item(**json.loads(l)) for l in path.read_text().splitlines() if l.strip()]


def encode_batch(processor, prompt_fn, items: list[Item], images: list) -> dict:
    """Right-padded batch of prompt + answer + <|im_end|>, labels on the answer tokens only.

    Same output fields as ``targeted_finetune.encode_batch`` so that ``targeted_finetune.batch_loss`` applies.
    """
    tok = processor.tokenizer
    prompts = [prompt_fn(it.prompt) for it in items]
    answers = [it.answer + '<|im_end|>' for it in items]
    if any(not it.answer for it in items):
        raise ValueError('encode_batch needs supervised answers')
    tok.padding_side = 'right'
    batch = processor(text=[p + a for p, a in zip(prompts, answers)], images=images, padding=True, return_tensors='pt')
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist()):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError(f'Processor resized a {image.width}x{image.height} image')
    rows, idx, tgt, dig = [], [], [], []
    ids = batch['input_ids']
    lengths = batch['attention_mask'].sum(1).tolist()
    digit_ids = {tok.convert_tokens_to_ids(str(d)) for d in range(10)}
    for b, a in enumerate(answers):
        n_ans = len(tok.encode(a, add_special_tokens=False))
        L = int(lengths[b])
        full = ids[b, :L].tolist()
        if tok.decode(full[L - n_ans:]) != a:
            raise ValueError(f'Answer tokens do not align with the end of item {items[b].id}')
        for t in range(L - n_ans, L):
            rows.append(b)
            idx.append(t - 1)
            tgt.append(full[t])
            dig.append(full[t] in digit_ids)
    batch['label_row'] = torch.tensor(rows)
    batch['label_idx'] = torch.tensor(idx)
    batch['label_tok'] = torch.tensor(tgt)
    batch['is_digit'] = torch.tensor(dig)
    return batch


def fit_area(image, max_pixels: int = 512 * 512, min_pixels: int = 65536):
    """Resize to sides that are multiples of 32 with area in [min_pixels, max_pixels] (aspect kept)."""
    from PIL import Image
    w, h = image.size
    s = math.sqrt(max_pixels / (w * h)) if w * h > max_pixels else max(1.0, math.sqrt(min_pixels / (w * h)))
    W, H = (max(32, int(round(v * s / 32)) * 32) for v in (w, h))
    while W * H > max_pixels:
        if W >= H:
            W -= 32
        else:
            H -= 32
    return image.convert('RGB').resize((W, H), Image.Resampling.BICUBIC)


# ----------------------------------------------------------------------------- scoring

def score_count(text: str, truth: dict) -> dict:
    import re
    m = re.findall(r'"rows"\s*:\s*(\d+)\s*,\s*"columns"\s*:\s*(\d+)', text)
    if not m:
        return {'valid': False, 'correct': False}
    r, c = int(m[-1][0]), int(m[-1][1])
    return {'valid': True, 'pred': [r, c], 'correct': r == truth['rows'] and c == truth['cols'],
            'row_ok': r == truth['rows'], 'col_ok': c == truth['cols'], 'd_rows': r - truth['rows'],
            'd_cols': c - truth['cols']}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
