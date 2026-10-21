"""Experiment 5.1: local re-read pilot (docs/analysis/reread/plan.md, results_2026-10-16_reread_pilot.md).

This module holds the reusable, model-free parts of the experiment and a small multi-image engine:

* windows in merged-token units (one merged visual token covers 32x32 input pixels) and their
  pixel / 0-1000 boxes, random non-overlapping control windows;
* `encode`: one vision forward over several images without any processor resizing, returning the
  merger output of every image and optionally the merger projection of intermediate ViT blocks;
* `assemble`: a single-turn prompt with several visual segments whose embeddings and MRoPE ids are
  chosen freely. A segment either receives the default ids of a new image placed at its sequence
  position ('new'), or keeps the row / column ids of a window of the first image ('orig'), with its
  temporal id at its own sequence position, as for a later video frame;
* task construction (synthetic grids, RefCOCO, BabyVision), prompts, and scoring.

Every condition of the experiment is a list of segments plus a prompt; the frozen model only ever
sees inputs_embeds and explicit (4, B, L) position ids. Layer and block numbers are 1-based.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.common import stimuli as st

TOKEN = 32                  # pixels per merged visual token along each axis
PATCH = 16
CELL = 64                   # synthetic instance cell (2x2 merged tokens)

ORDINAL_NOTE = 'Rows are counted from top to bottom and columns from left to right, starting at 1.'
COORD_NOTE = ' All coordinates use 0-1000 normalization; x increases rightward and y increases downward.'
JSON_ONLY = ' Output only the JSON, with no explanation.'
ANSWER_YES_NO = ' Answer with only Yes or No.'
T3_REVISED = 'Is there anything about the figure in row {r}, column {c} that differs from the majority of the figures?'
T5_QUERY = 'One figure differs from all the others.'
T6_QUERY = 'One figure is drawn in red.'
LOCATE_SUFFIX = {
    'rowcol': ' Give its position. Return only {"row": r, "column": c}, counting rows from top to bottom '
              'and columns from left to right, starting at 1.',
    'point': ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].' + COORD_NOTE,
}
# Think-then-answer row/column prompt of experiment 1.3 (thinking off, free-form reasoning allowed).
EXPLICIT_LOCATE = (' First describe step by step what you observe and how you find this figure, including how many '
                   'rows and columns of figures there are. Then give the final answer on the last line as '
                   '{"row": r, "column": c}, counting rows from top to bottom and columns from left to right, '
                   'starting at 1.')
ORDINALS = ('second', 'third', 'fourth')


# ----------------------------------------------------------------------------- windows

@dataclass(frozen=True)
class Window:
    """Axis-aligned window in merged-token units of one image: rows r0..r0+h-1, cols c0..c0+w-1."""
    r0: int
    c0: int
    h: int
    w: int

    def pixel_box(self) -> tuple[int, int, int, int]:
        return self.c0 * TOKEN, self.r0 * TOKEN, (self.c0 + self.w) * TOKEN, (self.r0 + self.h) * TOKEN

    def norm_box(self, grid: tuple[int, int]) -> list[int]:
        """0-1000 box (x1, y1, x2, y2) of the window in an image of grid (H, W) merged tokens."""
        H, W = grid
        return [round(1000 * self.c0 / W), round(1000 * self.r0 / H),
                round(1000 * (self.c0 + self.w) / W), round(1000 * (self.r0 + self.h) / H)]

    def indices(self, grid_w: int) -> list[int]:
        return [(self.r0 + r) * grid_w + self.c0 + c for r in range(self.h) for c in range(self.w)]

    def overlaps(self, other: 'Window') -> bool:
        return not (self.r0 + self.h <= other.r0 or other.r0 + other.h <= self.r0
                    or self.c0 + self.w <= other.c0 or other.c0 + other.w <= self.c0)

    def as_list(self) -> list[int]:
        return [self.r0, self.c0, self.h, self.w]


def cell_window(r0: int, c0: int, k: int, cell_tokens: int = CELL // TOKEN, origin: tuple[int, int] = (0, 0)) -> Window:
    """k x k instance cells starting at cell (r0, c0) of a grid whose origin is given in tokens."""
    return Window(origin[0] + r0 * cell_tokens, origin[1] + c0 * cell_tokens, k * cell_tokens, k * cell_tokens)


def centered_cells(rc: tuple[int, int], k: int, rows: int, cols: int) -> tuple[int, int]:
    """Top-left cell of a k x k block centred on rc, shifted inward at the borders."""
    return (min(max(rc[0] - k // 2, 0), rows - k), min(max(rc[1] - k // 2, 0), cols - k))


def random_cells_containing(rc: tuple[int, int], k: int, rows: int, cols: int, rng: np.random.Generator) -> tuple[int, int]:
    """Top-left cell of a k x k block containing rc with uniformly random relative position (before clamping)."""
    r0 = rc[0] - int(rng.integers(0, k))
    c0 = rc[1] - int(rng.integers(0, k))
    return min(max(r0, 0), rows - k), min(max(c0, 0), cols - k)


def random_window(grid: tuple[int, int], size: tuple[int, int], avoid: list[Window], rng: np.random.Generator,
                  step: int = 1, origin: tuple[int, int] = (0, 0), extent: tuple[int, int] | None = None,
                  tries: int = 2000) -> Window:
    """Window of the given size inside the image, on a `step` lattice, overlapping none of `avoid`."""
    H, W = extent or grid
    h, w = size
    for _ in range(tries):
        r0 = origin[0] + step * int(rng.integers(0, (H - h) // step + 1))
        c0 = origin[1] + step * int(rng.integers(0, (W - w) // step + 1))
        win = Window(r0, c0, h, w)
        if r0 + h <= grid[0] and c0 + w <= grid[1] and not any(win.overlaps(a) for a in avoid):
            return win
    raise ValueError(f'No non-overlapping {size} window in a {grid} grid.')


def pixel_to_window(box: list[float], grid: tuple[int, int], rng: np.random.Generator | None = None,
                    scale: float = 3.0, jitter: float = 0.5) -> Window:
    """Square window of side `scale` x the box long side, centred on the box (plus uniform jitter of at most
    `jitter` x long side), expanded outward to the token lattice and clipped to the image. Pixel units."""
    x0, y0, x1, y1 = box
    side = scale * max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if rng is not None:
        L = max(x1 - x0, y1 - y0)
        cx += float(rng.uniform(-jitter, jitter)) * L
        cy += float(rng.uniform(-jitter, jitter)) * L
    H, W = grid
    side_t = max(2, min(math.ceil(side / TOKEN), H, W))
    c0 = int(round(cx / TOKEN - side_t / 2))
    r0 = int(round(cy / TOKEN - side_t / 2))
    c0, r0 = min(max(c0, 0), W - side_t), min(max(r0, 0), H - side_t)
    return Window(r0, c0, side_t, side_t)


def bbox_window(box: list[float], grid: tuple[int, int]) -> Window:
    """Smallest token-aligned window covering a pixel box."""
    r0, c0 = int(box[1] // TOKEN), int(box[0] // TOKEN)
    r1, c1 = min(math.ceil(box[3] / TOKEN), grid[0]), min(math.ceil(box[2] / TOKEN), grid[1])
    return Window(r0, c0, r1 - r0, c1 - c0)


def crop(image: Image.Image, win: Window, scale: int = 1) -> Image.Image:
    out = image.crop(win.pixel_box())
    if scale != 1:
        out = out.resize((out.width * scale, out.height * scale), Image.Resampling.BICUBIC)
    return out


def resize32(image: Image.Image) -> Image.Image:
    """Resize to the nearest multiple of 32 pixels on each side (at least 256 on the short side is not enforced)."""
    W, H = max(TOKEN, round(image.width / TOKEN) * TOKEN), max(TOKEN, round(image.height / TOKEN) * TOKEN)
    return image if (W, H) == image.size else image.resize((W, H), Image.Resampling.BICUBIC)


# ----------------------------------------------------------------------------- vision encoding

@torch.inference_mode()
def encode(rt: qr.Runtime, images: list[Image.Image], blocks: tuple[int, ...] = ()) -> list[dict]:
    """Vision forward without resizing. Returns per image {'vis': (n, D), 'grid': (H, W), 'block_k': (n, D)}.

    'block_k' is the main-path merger applied to the output of ViT block k (1-based); for the last block
    it equals 'vis'. Merged tokens are in raster order.
    """
    for im in images:
        if im.width % TOKEN or im.height % TOKEN:
            raise ValueError(f'Image {im.size} is not a multiple of {TOKEN} pixels.')
    out = rt.processor.image_processor(images=images, do_resize=False, return_tensors='pt')
    grid = out['image_grid_thw']
    for im, (_, h, w) in zip(images, grid.tolist()):
        if (h * PATCH, w * PATCH) != (im.height, im.width):
            raise ValueError('Image processor changed the image size.')
    visual = rt.model.model.visual
    store, handles = {}, []
    for k in blocks:
        handles.append(visual.blocks[k - 1].register_forward_hook(
            lambda m, i, o, k=k: store.__setitem__(k, o[0] if isinstance(o, tuple) else o)))
    try:
        feats = rt.model.model.get_image_features(out['pixel_values'].to(rt.device), grid.to(rt.device)).pooler_output
    finally:
        for h in handles:
            h.remove()
    sizes = [int(h * w) // 4 for _, h, w in grid.tolist()]
    res = [{'grid': (h // 2, w // 2)} for _, h, w in grid.tolist()]
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(list(feats), 0)
    for r, part in zip(res, torch.split(feats, sizes)):
        r['vis'] = part
    for k in blocks:
        merged = visual.merger(store[k])
        for r, part in zip(res, torch.split(merged, sizes)):
            r[f'block_{k}'] = part
    return res


# ----------------------------------------------------------------------------- sequence assembly

@dataclass
class Segment:
    """One visual segment of the prompt.

    emb:  (n, D) embeddings in raster order of a (h, w) token grid.
    mode: 'new' -> default ids of a new image at the segment's sequence position;
          'orig' -> row / col ids of the window `window` of the first segment (temporal id = own start).
    """
    emb: torch.Tensor
    grid: tuple[int, int]
    mode: str = 'new'
    window: Window | None = None

    def __post_init__(self):
        if self.emb.shape[0] != self.grid[0] * self.grid[1]:
            raise ValueError('Segment embedding count does not match its grid.')
        if self.mode not in ('new', 'orig') or (self.mode == 'orig' and self.window is None):
            raise ValueError('orig mode requires the source window.')


def mrope_ids(token_kinds: list[int], segs: list[Segment]) -> torch.Tensor:
    """(3, L) MRoPE ids following Qwen3.5 get_rope_index, with 'orig' segments re-using window ids.

    token_kinds: per token -1 for text, else the index of its segment (contiguous runs).
    The text after a segment resumes at start + max(h, w), as in the default rule, for both modes,
    so that 'orig' and 'new' segments differ only inside the segment.
    """
    ids = torch.zeros(3, len(token_kinds), dtype=torch.long)
    cur, i, first_start = 0, 0, None
    while i < len(token_kinds):
        k = token_kinds[i]
        if k < 0:
            ids[:, i] = cur
            cur += 1
            i += 1
            continue
        seg = segs[k]
        h, w = seg.grid
        n = h * w
        if token_kinds[i:i + n] != [k] * n:
            raise ValueError('Segment tokens are not contiguous.')
        rr, cc = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        rr, cc = rr.flatten(), cc.flatten()
        if first_start is None:
            first_start = cur
        ids[0, i:i + n] = cur
        if seg.mode == 'new':
            ids[1, i:i + n], ids[2, i:i + n] = cur + rr, cur + cc
        else:
            ids[1, i:i + n] = first_start + seg.window.r0 + rr
            ids[2, i:i + n] = first_start + seg.window.c0 + cc
        cur += max(h, w)
        i += n
    return ids


def assemble(rt: qr.Runtime, segs: list[Segment], text: str, thinking: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """inputs_embeds (1, L, D), position ids (4, 1, L) and token-count metadata of one prompt."""
    template = qr.prompt_text(rt, len(segs), text, thinking)
    raw = rt.tokenizer(template, add_special_tokens=False)['input_ids']
    pad = rt.model.config.image_token_id
    if raw.count(pad) != len(segs):
        raise ValueError('Image placeholders do not match the segments.')
    ids, kinds, k = [], [], 0
    for t in raw:
        if t == pad:
            n = segs[k].grid[0] * segs[k].grid[1]
            ids += [pad] * n
            kinds += [k] * n
            k += 1
        else:
            ids.append(t)
            kinds.append(-1)
    dev = rt.device
    id_t = torch.tensor(ids, device=dev)
    x = rt.model.model.get_input_embeddings()(id_t[None]).clone()
    kinds_t = torch.tensor(kinds)
    for j, seg in enumerate(segs):
        x[0, (kinds_t == j).nonzero().flatten().to(dev)] = seg.emb.to(dev, x.dtype)
    L = len(ids)
    pos = torch.cat([torch.arange(L).view(1, L), mrope_ids(kinds, segs)], 0).view(4, 1, L).to(dev)
    meta = {'L': L, 'n_visual': int((kinds_t >= 0).sum()), 'n_extra_visual': int((kinds_t > 0).sum())}
    return x, pos, meta


def hint_text(n_extra: int, box: list[int]) -> str:
    """Region hint naming the original-image 0-1000 box shown by the extra visual segments."""
    names = [ORDINALS[i] for i in range(n_extra)]
    if n_extra == 1:
        lead = f'The {names[0]} image shows'
    else:
        lead = 'The ' + ', '.join(names[:-1]) + f' and {names[-1]} images all show'
    return (f'{lead} region ({box[0]}, {box[1]}, {box[2]}, {box[3]}) of the first image, given in 0-1000 '
            'coordinates of the first image. ')


# ----------------------------------------------------------------------------- synthetic tasks

TEMPLATES = st.DEVELOPMENT_TEMPLATES


def det_cases(n: int, kind: str, level: int, count: int, seed: int) -> list[dict]:
    """Detail task (revised T3): half the images ask about the target, half about a normal instance at
    Chebyshev distance >= 2 from it, so a 3x3 window centred on the asked instance never contains the target."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        template = TEMPLATES[i % len(TEMPLATES)]
        target = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        positive = i % 2 == 0
        if positive:
            asked = target
        else:
            while True:
                asked = (int(rng.integers(0, n)), int(rng.integers(0, n)))
                if max(abs(asked[0] - target[0]), abs(asked[1] - target[1])) >= 2:
                    break
        site = int(rng.integers(0, 3)) if kind == 'gap' else 0
        out.append({'id': f'det_{n}_{kind}{level}_{i:02d}', 'task': 'det', 'grid_n': n, 'kind': kind, 'level': level,
                    'template': template, 'target': list(target), 'asked': list(asked), 'site': site,
                    'answer_yes': positive, 'seed': seed + i})
    return out


def loc_cases(n: int, task: str, count: int, seed: int) -> list[dict]:
    """Localization: T6 (pure red target) or T5 (gap level 3, 8 base pixels)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        target = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        out.append({'id': f'loc_{task}_{n}_{i:02d}', 'task': task, 'grid_n': n, 'template': TEMPLATES[i % len(TEMPLATES)],
                    'target': list(target), 'site': int(rng.integers(0, 3)), 'seed': seed + i})
    return out


def synth_image(case: dict) -> tuple[Image.Image, dict]:
    if case['task'] == 'det':
        change = st.Change(case['kind'], case['level'], case['site'])
    elif case['task'] == 'T6':
        change = st.Change('color', 4)
    else:
        change = st.Change('gap', 3, case['site'])
    spec = st.StimulusSpec(case['template'], case['grid_n'], case['grid_n'], change, tuple(case['target']),
                           seed=case['seed'])
    image, meta = st.generate(spec)
    if meta['grid_origin'] != [0, 0]:
        raise ValueError('Synthetic grids are expected to fill the canvas.')
    return image, meta


def det_prompt(asked: tuple[int, int]) -> str:
    return T3_REVISED.format(r=asked[0] + 1, c=asked[1] + 1) + ' ' + ORDINAL_NOTE + ANSWER_YES_NO


def loc_prompt(task: str, fmt: str, style: str = 'direct') -> str:
    query = T6_QUERY if task == 'T6' else T5_QUERY
    if style == 'think':
        if fmt != 'rowcol':
            raise ValueError('Think-then-answer uses the row/column format only.')
        return query + EXPLICIT_LOCATE
    return query + LOCATE_SUFFIX[fmt] + JSON_ONLY


# ----------------------------------------------------------------------------- parsing and scoring

def _json_values(text: str) -> list:
    """All top-level JSON values (objects or arrays) in a text, in order of appearance."""
    dec = json.JSONDecoder()
    vals, i = [], 0
    text = re.sub(r'```(?:json)?', ' ', text)
    while i < len(text):
        if text[i] in '[{':
            try:
                v, j = dec.raw_decode(text, i)
                vals.append(v)
                i = j
                continue
            except ValueError:
                pass
        i += 1
    return vals


def parse_rowcol(text: str, last: bool = False) -> tuple[int, int] | None:
    """{"row": r, "column": c} (1-based). Direct answers use the first JSON value, think answers the last."""
    vals = _json_values(text)
    cands = [v[0] if isinstance(v, list) and len(v) == 1 else v for v in vals]
    cands = [v for v in cands if isinstance(v, dict) and isinstance(v.get('row'), int) and isinstance(v.get('column'), int)]
    if not cands:
        return None
    v = cands[-1] if last else cands[0]
    return v['row'], v['column']


def parse_numbers(text: str, key: str, length: int, diagnostic: bool = False) -> list[float] | None:
    """Strict: first JSON value must be [{key: [..length numbers..]}]. Diagnostic also accepts a bare dict
    or bare number list, and a list of identical repeated points."""
    vals = _json_values(text)
    if not vals:
        return None
    v = vals[0]
    if diagnostic:
        if isinstance(v, dict):
            v = [v]
        if isinstance(v, list) and len(v) == length and all(isinstance(t, (int, float)) for t in v):
            v = [{key: v}]
        if isinstance(v, list) and len(v) > 1 and all(isinstance(t, dict) and t.get(key) == v[0].get(key) for t in v):
            v = v[:1]
    if not (isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict)):
        return None
    c = v[0].get(key)
    if not (isinstance(c, list) and len(c) == length and all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in c)):
        return None
    return [float(t) for t in c]


def to_local(win: Window, grid: tuple[int, int], xy_norm: np.ndarray) -> np.ndarray:
    """Interpret 0-1000 coordinates as local to the window and map them to original-image 0-1000 coordinates."""
    H, W = grid
    x = 1000 * (win.c0 + xy_norm[..., 0] / 1000 * win.w) / W
    y = 1000 * (win.r0 + xy_norm[..., 1] / 1000 * win.h) / H
    return np.stack([x, y], -1)


def score_point(text: str, finish: str, n: int, target: tuple[int, int], win: Window, grid: tuple[int, int]) -> dict:
    """Point answer on an n x n grid filling the canvas. hit: strict 0-1000; hit_any: also pixel convention;
    frame_error: wrong in original coordinates but correct when read as window-local coordinates."""
    out = {'valid': False, 'hit': False, 'hit_any': False, 'frame_error': False, 'in_window': False,
           'hit_tol': False, 'in_window_tol': False, 'frame_error_tol': False, 'valid_tol': False}
    if finish != 'eos':
        return {**out, 'failure': 'truncated'}
    p = parse_numbers(text, 'point_2d', 2)
    pd = parse_numbers(text, 'point_2d', 2, diagnostic=True)
    def cell(xy):
        c, r = int(xy[0] / 1000 * n), int(xy[1] / 1000 * n)
        return (r, c) if 0 <= r < n and 0 <= c < n else None
    if pd is not None:
        xy = np.array(pd)
        canvas = grid[0] * TOKEN
        out['hit_any'] = cell(xy) == tuple(target) or cell(xy / canvas * 1000) == tuple(target)
        # format-tolerant score: same 0-1000 convention, but bare dicts / bare lists are accepted
        out['hit_tol'] = cell(xy) == tuple(target)
        wx0, wy0, wx1, wy1 = win.norm_box(grid)
        out['in_window_tol'] = bool(wx0 <= xy[0] < wx1 and wy0 <= xy[1] < wy1)
        out['frame_error_tol'] = (not out['hit_tol']) and cell(to_local(win, grid, xy)) == tuple(target)
        out['valid_tol'] = True
    if p is None or not all(0 <= t <= 1000 for t in p):
        return {**out, 'failure': 'schema'}
    xy = np.array(p)
    H, W = grid
    wx0, wy0, wx1, wy1 = win.norm_box(grid)
    out.update(valid=True, pred=p, hit=cell(xy) == tuple(target),
               in_window=wx0 <= xy[0] < wx1 and wy0 <= xy[1] < wy1)
    out['frame_error'] = (not out['hit']) and cell(to_local(win, grid, xy)) == tuple(target)
    cx, cy = (target[1] + 0.5) * 1000 / n, (target[0] + 0.5) * 1000 / n
    out['err'] = float(np.hypot(xy[0] - cx, xy[1] - cy))
    return out


def window_cells(win: Window, n: int, grid: tuple[int, int]) -> tuple[int, int, int, int]:
    """Cell range (r0, c0, kr, kc) of a cell-aligned window on an n x n grid filling the canvas."""
    ct = grid[0] // n
    return win.r0 // ct, win.c0 // ct, win.h // ct, win.w // ct


def score_rowcol(text: str, finish: str, n: int, target: tuple[int, int], win: Window, grid: tuple[int, int],
                 last: bool = False) -> dict:
    out = {'valid': False, 'correct': False, 'frame_error': False, 'in_window': False}
    if finish != 'eos':
        return {**out, 'failure': 'truncated'}
    rc = parse_rowcol(text, last)
    if rc is None:
        return {**out, 'failure': 'schema'}
    r, c = rc[0] - 1, rc[1] - 1
    r0, c0, kr, kc = window_cells(win, n, grid)
    local = (target[0] - r0, target[1] - c0)
    out.update(valid=True, pred=[r, c], correct=(r, c) == tuple(target),
               in_window=r0 <= r < r0 + kr and c0 <= c < c0 + kc, d_row=r - target[0], d_col=c - target[1])
    out['frame_error'] = (not out['correct']) and (r, c) == local
    return out


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score_bbox(text: str, finish: str, truth_norm: list[float], win: Window, grid: tuple[int, int]) -> dict:
    """RefCOCO bbox answer in 0-1000 coordinates; truth_norm is the 0-1000 ground-truth box."""
    out = {'valid': False, 'correct': False, 'center_in': False, 'frame_error': False, 'in_window': False}
    if finish != 'eos':
        return {**out, 'failure': 'truncated'}
    b = parse_numbers(text, 'bbox_2d', 4, diagnostic=True)
    if b is None or b[0] >= b[2] or b[1] >= b[3]:
        return {**out, 'failure': 'schema'}
    iou = box_iou(b, truth_norm)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    wx0, wy0, wx1, wy1 = win.norm_box(grid)
    loc = to_local(win, grid, np.array([[b[0], b[1]], [b[2], b[3]]]))
    iou_local = box_iou([loc[0, 0], loc[0, 1], loc[1, 0], loc[1, 1]], truth_norm)
    out.update(valid=True, pred=b, iou=iou, correct=iou >= 0.5,
               center_in=truth_norm[0] <= cx <= truth_norm[2] and truth_norm[1] <= cy <= truth_norm[3],
               in_window=wx0 <= cx < wx1 and wy0 <= cy < wy1, iou_local=iou_local)
    out['frame_error'] = iou < 0.5 and iou_local >= 0.5
    return out


def trivial_box(win: Window, grid: tuple[int, int]) -> list[float]:
    """Box centred on the window, side 1/3 of the window side (0-1000 coordinates of the image)."""
    x0, y0, x1, y1 = win.norm_box(grid)
    cx, cy, sx, sy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 6, (y1 - y0) / 6
    return [cx - sx, cy - sy, cx + sx, cy + sy]


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]
