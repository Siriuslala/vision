"""Experiment 5.2: cached-feature re-read through the native zoom tool protocol.

Design: docs/analysis/reread/design_cached_reread.md. Results: results_2026-10-20_cached_reread.md.

The frozen model sees one image and a question, together with the zoom tool schema of the Qwen-VL
"think with images" recipe (`image_zoom_in_tool`, 0-1000 `bbox_2d`). It emits one tool call; the call
is parsed and the tool response carries a second visual segment whose content depends on the
condition:

* pixel crop (the native tool): the window is cropped from the image, optionally upscaled, and
  re-encoded by the ViT as a new image with default MRoPE ids;
* cached re-read (this experiment): the window's tokens are sliced from a feature cache computed on the
  whole image, either the merger output (ViT block 27) or the main-path merger applied to ViT block 9,
  at the input resolution or from a ViT pass on the 2x image. The tokens keep their raster row
  structure and, by default, the row / column MRoPE ids of the window in the original image.

Window choice (the tool call) is generated once per question and shared by every re-read condition, so
conditions differ only in what the tool returns. Oracle conditions replace the call by a window derived
from ground-truth evidence. Layer and block numbers are 1-based.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
import string

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.reread import reread_pilot as rp

TOKEN = rp.TOKEN                      # pixels per merged visual token at the input resolution
Window = rp.Window

ZOOM_TOOL = {
    'type': 'function',
    'function': {
        'name': 'image_zoom_in_tool',
        'description': 'Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) '
                       'and an optional object label.',
        'parameters': {
            'type': 'object',
            'properties': {
                'bbox_2d': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4,
                            'description': 'The bounding box of the region to zoom in, as [x1, y1, x2, y2], where '
                                           '(x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner, '
                                           'in relative coordinates 0-1000 of the image.'},
                'label': {'type': 'string', 'description': 'The name or label of the object in the specified '
                                                           'bounding box (optional).'},
            },
            'required': ['bbox_2d'],
        },
    },
}
FORCE_ZOOM = ('\n\nBefore answering, call image_zoom_in_tool exactly once on the region of the image that is most '
              'relevant to the question. After you receive the zoomed view, answer the question.')
# The tool response restates the answer format, because after a tool turn the model otherwise answers in
# free-form prose; the text is identical for every condition.
TOOL_RESPONSE = ('Zoomed view of region [{}, {}, {}, {}] of the image, in 0-1000 coordinates of the image. '
                 'Now answer the question. {}')
SUFFIX_AFTER_CALL = '<|im_end|>\n<|im_start|>user\n<tool_response>\n<|vision_start|><|image_pad|><|vision_end|>'
SUFFIX_END = '\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

MIN_WINDOW = 4                        # minimum window side in merged tokens (128 pixels)
CROP_MAX_PIXELS = 1280 * 1280         # cap on an upscaled pixel crop, like the processor max_pixels


@dataclass(frozen=True)
class Condition:
    """One way of answering. kind: 'base' | 'full2x' | 'crop' | 'cache'."""
    name: str
    kind: str
    scale: int = 1                    # crop upscaling, or 2 for a cache built on the 2x image
    site: str = 'vis'                 # cache site: 'vis' (merger output, block 27) or 'block_9'
    ids: str = 'orig'                 # cache MRoPE ids: 'orig' (window ids of the image) or 'new'
    oracle: bool = False
    order: str = 'raster'             # cache token order in the sequence: 'raster' or 'reverse' (ids follow tokens)


CONDITIONS = {c.name: c for c in [
    Condition('base', 'base'),
    Condition('full2x', 'full2x', 2),
    Condition('crop1', 'crop', 1),
    Condition('crop2', 'crop', 2),
    Condition('crop4', 'crop', 4),
    Condition('c27', 'cache', 1, 'vis'),
    Condition('c9', 'cache', 1, 'block_9'),
    Condition('h27', 'cache', 2, 'vis'),
    Condition('h9', 'cache', 2, 'block_9'),
    Condition('h9n', 'cache', 2, 'block_9', 'new'),
    Condition('o_crop2', 'crop', 2, oracle=True),
    Condition('o_crop4', 'crop', 4, oracle=True),
    Condition('o_c9', 'cache', 1, 'block_9', oracle=True),
    Condition('o_h9', 'cache', 2, 'block_9', oracle=True),
    Condition('o_h27', 'cache', 2, 'vis', oracle=True),
    Condition('h9r', 'cache', 2, 'block_9', order='reverse'),
    Condition('o_h9r', 'cache', 2, 'block_9', oracle=True, order='reverse'),
]}


# ----------------------------------------------------------------------------- images and windows

def resize_budget(image: Image.Image, max_pixels: int | None) -> Image.Image:
    """Resize to sides that are multiples of 32 pixels, keeping the aspect ratio, with area <= max_pixels.

    max_pixels None keeps the native size (rounded to multiples of 32). Small images are never upscaled
    beyond rounding.
    """
    w, h = image.size
    s = 1.0 if max_pixels is None else min(1.0, math.sqrt(max_pixels / (w * h)))
    W = max(2 * TOKEN, int(w * s / TOKEN) * TOKEN)
    H = max(2 * TOKEN, int(h * s / TOKEN) * TOKEN)
    return image if (W, H) == image.size else image.resize((W, H), Image.Resampling.BICUBIC)


def upscale(image: Image.Image, k: int) -> Image.Image:
    return image if k == 1 else image.resize((image.width * k, image.height * k), Image.Resampling.BICUBIC)


def box_to_window(box: list[float], grid: tuple[int, int], min_side: int = MIN_WINDOW) -> Window:
    """Token-aligned window covering a 0-1000 box (x1, y1, x2, y2), at least min_side tokens per side.

    The box is expanded outward to the token lattice; a side shorter than min_side grows symmetrically
    around the box centre and is shifted inward at the image border.
    """
    H, W = grid
    x1, y1, x2, y2 = (min(max(float(v), 0.0), 1000.0) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    def axis(a: float, b: float, n: int) -> tuple[int, int]:
        lo, hi = int(math.floor(a * n / 1000)), int(math.ceil(b * n / 1000))
        hi = max(hi, lo + 1)
        side = min(max(hi - lo, min_side), n)
        centre = (lo + hi) / 2
        lo = int(round(centre - side / 2))
        lo = min(max(lo, 0), n - side)
        return lo, side

    r0, h = axis(y1, y2, H)
    c0, w = axis(x1, x2, W)
    return Window(r0, c0, h, w)


def crop_scale(win: Window, k: int, max_pixels: int = CROP_MAX_PIXELS) -> int:
    """Largest upscaling factor <= k whose crop stays within max_pixels (at least 1)."""
    area = win.h * win.w * TOKEN * TOKEN
    while k > 1 and area * k * k > max_pixels:
        k -= 1
    return k


# ----------------------------------------------------------------------------- prompts

def stage1_ids(rt: qr.Runtime, question: str) -> list[int]:
    """Tool-enabled single-turn prompt (thinking off) with one image placeholder."""
    chat = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question + FORCE_ZOOM}]}]
    text = rt.processor.apply_chat_template(chat, tools=[ZOOM_TOOL], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
    return rt.tokenizer(text, add_special_tokens=False)['input_ids']


def plain_ids(rt: qr.Runtime, question: str) -> list[int]:
    """Prompt without tools, one image placeholder (base and full2x conditions)."""
    return rt.tokenizer(qr.prompt_text(rt, 1, question, False), add_special_tokens=False)['input_ids']


def suffix_ids(rt: qr.Runtime, region: list[int], fmt: str) -> list[int]:
    """Tokens after the generated tool call: end of turn, tool response with one image placeholder, new turn."""
    text = SUFFIX_AFTER_CALL + TOOL_RESPONSE.format(*region, fmt.strip()) + SUFFIX_END
    return rt.tokenizer(text, add_special_tokens=False)['input_ids']


_CALL = re.compile(r'<tool_call>.*?<function=image_zoom_in_tool>.*?<parameter=bbox_2d>\s*(.*?)\s*</parameter>',
                   re.S)


def parse_call(text: str) -> list[float] | None:
    """bbox_2d of the first image_zoom_in_tool call in the Qwen3.5 XML tool format, or None."""
    m = _CALL.search(text)
    if m is None:
        return None
    nums = re.findall(r'-?\d+(?:\.\d+)?', m.group(1))
    if len(nums) != 4:
        return None
    return [float(v) for v in nums]


# ----------------------------------------------------------------------------- sequence assembly

@dataclass
class VSeg:
    """A visual segment. emb (h*w, D) in raster order of a (h, w) grid.

    ids 'new': default ids of a new image starting at the segment's sequence position.
    ids 'orig': temporal id at the segment's own position; row / column ids continue the first segment's
    grid as first_start + origin + index * step, with step = 1 / (cache resolution factor), so a token of
    a 2x cache sits between the integer positions of the 1x image. MRoPE is applied with float positions.
    """
    emb: torch.Tensor
    grid: tuple[int, int]
    ids: str = 'new'
    origin: tuple[float, float] = (0.0, 0.0)
    step: float = 1.0
    perm: torch.Tensor | None = None  # sequence index i holds raster token perm[i]; its ids follow it

    def __post_init__(self):
        if self.emb.shape[0] != self.grid[0] * self.grid[1]:
            raise ValueError('Segment embedding count does not match its grid.')
        if self.ids not in ('new', 'orig'):
            raise ValueError(f'Unknown ids mode {self.ids!r}.')


def mrope_float(kinds: list[int], segs: list[VSeg]) -> torch.Tensor:
    """(3, L) float MRoPE ids. Text after a segment resumes at start + max(h, w), as in get_rope_index."""
    ids = torch.zeros(3, len(kinds), dtype=torch.float32)
    cur, i, first = 0.0, 0, None
    while i < len(kinds):
        k = kinds[i]
        if k < 0:
            ids[:, i] = cur
            cur += 1
            i += 1
            continue
        seg = segs[k]
        h, w = seg.grid
        n = h * w
        if kinds[i:i + n] != [k] * n:
            raise ValueError('Segment tokens are not contiguous.')
        rr, cc = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32),
                                indexing='ij')
        rr, cc = rr.flatten(), cc.flatten()
        if seg.perm is not None:
            rr, cc = rr[seg.perm], cc[seg.perm]
        if first is None:
            first = cur
            if seg.ids != 'new':
                raise ValueError('The first segment must use default ids.')
        ids[0, i:i + n] = cur
        if seg.ids == 'new':
            ids[1, i:i + n], ids[2, i:i + n] = cur + rr, cur + cc
        else:
            ids[1, i:i + n] = first + seg.origin[0] + rr * seg.step
            ids[2, i:i + n] = first + seg.origin[1] + cc * seg.step
        cur += max(h, w)
        i += n
    return ids


def assemble(rt: qr.Runtime, ids: list[int], segs: list[VSeg]) -> dict:
    """Expand image placeholders into segment tokens. Returns x (L, D) on device, pos (3, L) float, counts."""
    pad = rt.model.config.image_token_id
    if ids.count(pad) != len(segs):
        raise ValueError(f'{ids.count(pad)} image placeholders for {len(segs)} segments.')
    full, kinds, k = [], [], 0
    for t in ids:
        if t == pad:
            n = segs[k].grid[0] * segs[k].grid[1]
            full += [pad] * n
            kinds += [k] * n
            k += 1
        else:
            full.append(t)
            kinds.append(-1)
    dev = rt.device
    x = rt.model.model.get_input_embeddings()(torch.tensor(full, device=dev)).clone()
    kt = torch.tensor(kinds)
    for j, seg in enumerate(segs):
        x[(kt == j).nonzero().flatten().to(dev)] = seg.emb.to(dev, x.dtype)
    return {'x': x, 'pos': mrope_float(kinds, segs), 'L': len(full), 'n_visual': int((kt >= 0).sum()),
            'n_extra_visual': int((kt > 0).sum())}


# ----------------------------------------------------------------------------- decoding

def stop_ids(rt: qr.Runtime, tool: bool) -> set[int]:
    e = rt.model.generation_config.eos_token_id
    out = set(e if isinstance(e, list) else [e])
    if tool:
        t = rt.tokenizer.encode('</tool_call>', add_special_tokens=False)
        if len(t) != 1:
            raise ValueError('</tool_call> is expected to be a single token.')
        out.add(t[0])
    return out


@torch.inference_mode()
def greedy_padded(rt: qr.Runtime, items: list[dict], max_new_tokens: int, stops: set[int],
                  score_ids: tuple[int, int] | None = None) -> list[dict]:
    """Left-padded batched greedy decoding from inputs_embeds and float MRoPE ids.

    items: dicts with 'x' (L_i, D) and 'pos' (3, L_i). Padding tokens get zero embeddings, position 0 and
    attention-mask 0 (Gated DeltaNet layers zero masked states). New tokens continue at max(pos) + 1.
    score_ids: (yes, no) token ids whose first-step logit difference is returned as 's'.
    """
    B = len(items)
    Ls = [it['x'].shape[0] for it in items]
    Lm = max(Ls)
    D = items[0]['x'].shape[1]
    dev = rt.device
    x = torch.zeros(B, Lm, D, device=dev, dtype=items[0]['x'].dtype)
    pos = torch.zeros(3, B, Lm, device=dev, dtype=torch.float32)
    mask = torch.zeros(B, Lm, device=dev, dtype=torch.long)
    for b, it in enumerate(items):
        L = Ls[b]
        x[b, Lm - L:] = it['x']
        pos[:, b, Lm - L:] = it['pos'].to(dev)
        mask[b, Lm - L:] = 1
    out = rt.model(inputs_embeds=x, position_ids=pos, attention_mask=mask, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    logits = out.logits[:, -1].float()
    s = (logits[:, score_ids[0]] - logits[:, score_ids[1]]).tolist() if score_ids else None
    nxt = pos.amax(-1) + 1                                         # (3, B)
    toks: list[list[int]] = [[] for _ in range(B)]
    done = [False] * B
    embed = rt.model.model.get_input_embeddings()
    for step in range(max_new_tokens):
        tok = logits.argmax(-1)
        tl = tok.tolist()
        for b in range(B):
            if not done[b]:
                toks[b].append(tl[b])
                done[b] = tl[b] in stops
        if all(done) or step == max_new_tokens - 1:
            break
        mask = torch.cat([mask, torch.ones(B, 1, device=dev, dtype=mask.dtype)], 1)
        out = rt.model(inputs_embeds=embed(tok[:, None]), position_ids=(nxt + step)[:, :, None],
                       attention_mask=mask, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1].float()
    res = []
    for b in range(B):
        ids = toks[b]
        fin = bool(ids) and ids[-1] in stops
        r = {'ids': ids, 'text': rt.tokenizer.decode(ids, skip_special_tokens=False),
             'n_tokens': len(ids), 'finish_reason': 'stop' if fin else 'length'}
        r['text'] = r['text'].replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
        if s is not None:
            r['s'] = s[b]
        res.append(r)
    return res


def run_batched(rt: qr.Runtime, items: list[dict], max_new_tokens: int, stops: set[int], token_budget: int,
                max_batch: int, score_ids: tuple[int, int] | None = None) -> list[dict]:
    """Sort by length and decode in batches with at most token_budget padded prompt tokens per batch."""
    order = sorted(range(len(items)), key=lambda i: items[i]['x'].shape[0])
    res: list[dict | None] = [None] * len(items)
    i = 0
    while i < len(order):
        j = i + 1
        while (j < len(order) and j - i < max_batch
               and (j - i + 1) * items[order[j]]['x'].shape[0] <= token_budget):
            j += 1
        chunk = order[i:j]
        outs = greedy_padded(rt, [items[k] for k in chunk], max_new_tokens, stops, score_ids)
        for k, o in zip(chunk, outs):
            res[k] = {**o, 'batch_size': len(chunk)}
        i = j
    return res


def marker_cut(rt: qr.Runtime, ids: list[int], marker: str = 'final answer:') -> int | None:
    """Number of leading generated tokens up to and including the last occurrence of the answer marker."""
    cut = None
    for k in range(1, len(ids) + 1):
        if rt.tokenizer.decode(ids[:k]).lower().count(marker) > (0 if cut is None else
                                                                  rt.tokenizer.decode(ids[:cut]).lower().count(marker)):
            cut = k
    return cut


def extend_item(rt: qr.Runtime, item: dict, gen: list[int]) -> dict:
    """Append generated tokens to an assembled prompt (text positions continue at max(pos) + 1)."""
    dev = rt.device
    e = rt.model.model.get_input_embeddings()(torch.tensor(gen, device=dev))
    start = float(item['pos'].max()) + 1
    p = (start + torch.arange(len(gen), dtype=torch.float32)).expand(3, -1)
    return {'x': torch.cat([item['x'], e.to(item['x'].dtype)], 0), 'pos': torch.cat([item['pos'], p], 1)}


# ----------------------------------------------------------------------------- cost accounting

def kv_bytes(config, L: int) -> dict:
    """Decoder cache size for a sequence of L tokens.

    Only gated full-attention layers keep a per-token KV cache; Gated DeltaNet layers keep a constant
    recurrent state (mamba_ssm_dtype) plus a short-convolution state, independent of L.
    """
    t = config.text_config if hasattr(config, 'text_config') else config
    n_full = sum(1 for k in t.layer_types if k == 'full_attention')
    n_lin = len(t.layer_types) - n_full
    per_token = n_full * 2 * t.num_key_value_heads * t.head_dim * 2           # bf16 K and V
    state = n_lin * t.linear_num_value_heads * t.linear_key_head_dim * t.linear_value_head_dim * 4
    conv_dim = 2 * t.linear_num_key_heads * t.linear_key_head_dim + t.linear_num_value_heads * t.linear_value_head_dim
    conv = n_lin * conv_dim * t.linear_conv_kernel_dim * 2
    return {'kv_bytes': per_token * L, 'state_bytes': state + conv, 'per_token_bytes': per_token}


# ----------------------------------------------------------------------------- scoring

_PUNCT = set(string.punctuation) - {"'", ':', ','}
_ARTICLES = {'a', 'an', 'the'}
_NUM = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
        'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}


def vqa_normalize(text: str) -> str:
    """Simplified VQA answer normalization: lower case, strip punctuation, articles, map number words."""
    t = text.lower().replace('\n', ' ').strip()
    t = re.sub(r'(?<=\d),(?=\d)', '', t)
    t = ''.join(' ' if ch in _PUNCT else ch for ch in t)
    words = [_NUM.get(w, w) for w in t.split() if w not in _ARTICLES]
    return ' '.join(words).strip(" '.,:")


def short_answer(text: str) -> str:
    """First line of a short answer, without a trailing period or an 'Answer:' prefix."""
    line = text.strip().split('\n')[0]
    line = re.sub(r'^(final answer|answer)\s*:\s*', '', line, flags=re.I)
    return line.strip()


_FINAL = re.compile(r'final answer\s*[:：]\s*(.+)', re.I)


def final_answer(text: str) -> tuple[str, bool]:
    """Content of the last "Final answer:" line, without Markdown emphasis or a trailing period.

    Without the marker, a single-line reply is taken as the answer (parsed=False either way is recorded);
    a multi-line reply without the marker yields its last line.
    """
    m = _FINAL.findall(text)
    if m:
        return m[-1].strip().rstrip('.').strip().strip('*`').strip().rstrip('.').strip(), True
    lines = [l for l in text.strip().split('\n') if l.strip()]
    return (lines[-1].strip() if lines else ''), False


def vqa_accuracy(pred: str, answers: list[str]) -> float:
    """Standard VQA accuracy min(#matching annotators / 3, 1) on normalized strings."""
    p = vqa_normalize(short_answer(pred))
    return min(1.0, sum(vqa_normalize(a) == p for a in answers) / 3.0)


def exact_match(pred: str, answer: str) -> bool:
    return vqa_normalize(short_answer(pred)) == vqa_normalize(answer)


def yes_no_answer(text: str) -> str | None:
    m = re.match(r'\W*(yes|no)\b', text.strip().lower())
    return m.group(1) if m else None


def option_letter(text: str) -> str | None:
    """MMVP option letter from '(a)', 'a', 'A.', 'Option (b)'."""
    t = text.strip()
    m = re.match(r'^\W*(?:option\s*)?\(?([abAB])\)?(?:\W|$)', t)
    return m.group(1).lower() if m else None


@dataclass
class Case:
    """One benchmark question. image is loaded lazily by the runner."""
    id: str
    bench: str
    question: str                    # text after the image, including the answer-format instruction
    answer: object
    fmt: str = ''                    # the answer-format instruction alone, restated in the tool response
    image_path: str = ''
    oracle_box: list[float] | None = None      # 0-1000 evidence box of the input image, when available
    meta: dict = field(default_factory=dict)


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def dumps(obj) -> str:
    return json.dumps(obj, default=json_default)
