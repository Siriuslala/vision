"""Experiment 3.11 round 4: where the edge-reading heads read, how two edge reads combine, what L20 H11 does.

Results: docs/analysis/position/results_2026-11-05_edge_heads_r2.md. Builds on 3.11 round 3 (edge_heads.py),
whose 872 natural cases, images, masks and mean head outputs are reused.

Item sets (case['set']); every image is rendered deterministically from its spec (render_pair), so only the
specs are stored:
  nat   natural images of round 3 whose target box spans >= 5 merged tokens on both axes. Pair B is the same
        image translated by 3 tokens on both axes (towards the image centre); every coordinate value moves.
  str   natural images whose box can be expanded by 96 px on some side. Behaviour: one edge of the object
        moves by delta in {-64, -32, +32, +64, +96} px while the opposite edge stays (the object is resampled
        along one axis, content beyond the moved edge is shifted). Probe pairs: one side expanded by 96 px.
  rect  one solid rectangle on white, same one-edge moves as 'str' (no resampling needed).
  comp  composites for the L20 H11 study: object type (coloured geometric shape | natural cut-out from a round-3
        mask) x background (plain white | photo) x target size (32, 64, 128, 160 px) x distractors (0 | 2).
        Pair B moves the target by 3 tokens on both axes (where pair); with distractors a second pair keeps
        the image and names a distractor (what pair). Shape distractors share colour (d1) or shape (d2) with
        the target so that the attention on them shows which attributes the query matches.

GPU stages (one process, each resumable by record id):
  gen     greedy answers of run A (point for every item, box for nat/str/rect and comp with size >= 128).
  means   mean full-attention head outputs over answer positions per comp family (object type x background).
  behav   greedy point and box answers for every one-edge variant of str and rect items.
  ablate  greedy comp point answers with mean-ablated head sets at the answer positions.
  probe   per pair: teacher-forced run A (attention of every head at every digit query, head outputs, residuals,
          L20 H11 keys/values of the image tokens), run B, condition rows (component patching from B, freezing
          to A, mean ablation, query patching) with later-layer attention, and single-head scans.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import coordinate_flow_natural as cfn
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import edge_heads as eh
from analysis_tools.representation import sees_right as sr

FULL_LAYERS = cr.FULL_LAYERS
DN_LAYERS = cfn.DN_LAYERS
N_LAYERS = 32
TP = cr.TOKEN_PIXELS                 # one merged token covers 32 x 32 input pixels
ATT_LAYERS = (12, 16, 20, 24, 28, 32)  # attention recorded at every digit query of run A
REC_LAYERS = (16, 20, 24, 28)        # full image-attention rows stored (first-digit queries of run A)
COND_LAYERS = (20, 24, 28)           # attention summaries stored for every condition row
OUT_LAYERS = (16, 20, 24, 28, 32)    # head outputs stored at the first-digit queries of run A
RES_LAYERS = (20, 24, 28, 32)        # decoder-layer outputs stored at the first-digit queries of run A
Q_LAYERS = (20, 24, 28)              # q_proj outputs captured for query patching
read_jsonl, append_jsonl, done_ids = sr.read_jsonl, sr.append_jsonl, sr.done_ids

# ----------------------------------------------------------------------------- region protocol

REGIONS = ('interior', 'left', 'right', 'top', 'bottom', 'top_left', 'top_right', 'bottom_left', 'bottom_right',
           'out_ring', 'out_col0', 'out_far')
BAND = 2      # edge band thickness in merged tokens, measured inward from the token holding the box edge


def box_cells(box, grid: tuple[int, int], tp: int = TP) -> tuple[int, int, int, int]:
    """Inclusive token column / row range (c_lo, c_hi, r_lo, r_hi) of the tokens a pixel box overlaps."""
    H, W = grid
    c0 = int(np.clip(np.floor(box[0] / tp), 0, W - 1))
    c1 = int(np.clip(np.floor((box[2] - 1e-6) / tp), 0, W - 1))
    r0 = int(np.clip(np.floor(box[1] / tp), 0, H - 1))
    r1 = int(np.clip(np.floor((box[3] - 1e-6) / tp), 0, H - 1))
    return c0, c1, r0, r1


def box_span_ok(box, grid, band: int = BAND) -> bool:
    c0, c1, r0, r1 = box_cells(box, grid)
    return c1 - c0 + 1 >= 2 * band + 1 and r1 - r0 + 1 >= 2 * band + 1


def region_labels(box, grid: tuple[int, int], band: int = BAND) -> np.ndarray:
    """Region index (into REGIONS) of every merged token, raster order.

    Inside the box, a token is in the left band when its column is one of the `band` columns starting at the
    column that holds x_min (the boundary token and the first inner column), and analogously for right, top,
    bottom. A token in two bands is a corner; a token in one band is on that edge; else interior. Outside the
    box: the one-token ring around it, the first image column (row-start tokens), and everything else.
    Requires a box spanning >= 2 * band + 1 tokens on both axes so that opposite bands do not overlap.
    """
    H, W = grid
    c0, c1, r0, r1 = box_cells(box, grid)
    if c1 - c0 + 1 < 2 * band + 1 or r1 - r0 + 1 < 2 * band + 1:
        raise ValueError(f'Box spans fewer than {2 * band + 1} tokens on an axis; bands would overlap.')
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    r, c = r.ravel(), c.ravel()
    inside = (c >= c0) & (c <= c1) & (r >= r0) & (r <= r1)
    hl, hr = c <= c0 + band - 1, c >= c1 - band + 1
    vt, vb = r <= r0 + band - 1, r >= r1 - band + 1
    lab = np.full(H * W, REGIONS.index('out_far'))
    edge_h, edge_v = hl | hr, vt | vb
    lab[inside & ~edge_h & ~edge_v] = REGIONS.index('interior')
    for name, m in (('left', hl & ~edge_v), ('right', hr & ~edge_v), ('top', vt & ~edge_h), ('bottom', vb & ~edge_h),
                    ('top_left', hl & vt), ('top_right', hr & vt), ('bottom_left', hl & vb),
                    ('bottom_right', hr & vb)):
        lab[inside & m] = REGIONS.index(name)
    ring = ~inside & (c >= c0 - 1) & (c <= c1 + 1) & (r >= r0 - 1) & (r <= r1 + 1)
    lab[ring] = REGIONS.index('out_ring')
    lab[~inside & ~ring & (c == 0)] = REGIONS.index('out_col0')
    return lab


def region_mass(p: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Sum of attention p (..., Nv) per region -> (..., len(REGIONS))."""
    onehot = np.eye(len(REGIONS), dtype=np.float32)[labels]          # (Nv, R)
    return p.astype(np.float32) @ onehot


def box_mass(p: np.ndarray, box, grid) -> np.ndarray:
    """Attention on the tokens a pixel box overlaps (..., Nv) -> (...)."""
    c0, c1, r0, r1 = box_cells(box, grid)
    H, W = grid
    m = np.zeros((H, W), bool)
    m[r0:r1 + 1, c0:c1 + 1] = True
    return p[..., m.ravel()].sum(-1)


def peak_xy(p: np.ndarray, grid) -> np.ndarray:
    """3x3-neighbourhood attention-weighted centre of the argmax token, pixels; p (..., Nv) -> (..., 2)."""
    H, W = grid
    flat = p.reshape(-1, H, W)
    out = np.zeros((flat.shape[0], 2))
    for i, a in enumerate(flat):
        r, c = divmod(int(a.argmax()), W)
        rs, cs = slice(max(r - 1, 0), r + 2), slice(max(c - 1, 0), c + 2)
        w = a[rs, cs]
        rr, cc = np.meshgrid(np.arange(rs.start, rs.start + w.shape[0]), np.arange(cs.start, cs.start + w.shape[1]),
                             indexing='ij')
        out[i] = [((cc + 0.5) * w).sum() / w.sum() * TP, ((rr + 0.5) * w).sum() / w.sum() * TP]
    return out.reshape(*p.shape[:-1], 2)


# ----------------------------------------------------------------------------- image operations

def _resample_axis(arr: np.ndarray, src: np.ndarray, axis: int) -> np.ndarray:
    """Linear resampling of arr along `axis` at float source indices src (one per output index), clamped."""
    n = arr.shape[axis]
    s = np.clip(src, 0, n - 1)
    i0 = np.floor(s).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = s - i0
    shape = [1] * arr.ndim
    shape[axis] = -1
    w = w.reshape(shape)
    return np.take(arr, i0, axis=axis) * (1 - w) + np.take(arr, i1, axis=axis) * w


def stretch_source(n: int, lo: float, hi: float, side: str, delta: float) -> np.ndarray:
    """Source pixel index for every output pixel when one edge of the interval [lo, hi] moves.

    side 'max': the edge at hi moves to hi + delta, the edge at lo stays; content in [lo, hi] is resampled
    onto [lo, hi + delta] and content beyond hi is shifted by delta. side 'min': the edge at lo moves to
    lo - delta (delta > 0 expands the object), the edge at hi stays. Coordinates are continuous (pixel centres
    at i + 0.5).
    """
    t = np.arange(n) + 0.5
    if side == 'max':
        new = hi + delta
        s = np.where(t < lo, t, np.where(t < new, lo + (t - lo) * (hi - lo) / (new - lo), t - delta))
    elif side == 'min':
        new = lo - delta
        s = np.where(t >= hi, t, np.where(t >= new, hi - (hi - t) * (hi - lo) / (hi - new), t + delta))
    else:
        raise ValueError(side)
    return s - 0.5


SIDES = {'left': ('x', 'min'), 'right': ('x', 'max'), 'top': ('y', 'min'), 'bottom': ('y', 'max')}


def stretched_box(box, side: str, delta: float) -> list[float]:
    b = list(box)
    k = {'left': 0, 'top': 1, 'right': 2, 'bottom': 3}[side]
    b[k] = b[k] - delta if k < 2 else b[k] + delta
    return b


def stretch_feasible(box, canvas, side: str, delta: float, min_size: float = 48) -> bool:
    W, H = canvas
    nb = stretched_box(box, side, delta)
    return nb[0] >= 0 and nb[1] >= 0 and nb[2] <= W and nb[3] <= H and nb[2] - nb[0] >= min_size \
        and nb[3] - nb[1] >= min_size


def stretch_image(img: Image.Image, box, side: str, delta: float) -> Image.Image:
    """Move one edge of the object box by delta pixels (see stretch_source); returns the edited image."""
    axis_name, which = SIDES[side]
    arr = np.asarray(img, dtype=np.float32)
    if axis_name == 'x':
        src = stretch_source(arr.shape[1], box[0], box[2], which, delta)
        out = _resample_axis(arr, src, 1)
    else:
        src = stretch_source(arr.shape[0], box[1], box[3], which, delta)
        out = _resample_axis(arr, src, 0)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def translate_image(img: Image.Image, dx: int, dy: int) -> Image.Image:
    """Shift the content by (dx, dy) pixels; the uncovered band takes the image's mean colour."""
    arr = np.asarray(img)
    out = np.empty_like(arr)
    out[:] = arr.reshape(-1, arr.shape[2]).mean(0).astype(arr.dtype)
    H, W = arr.shape[:2]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = arr[ys, xs]
    return Image.fromarray(out)


def shift_box(box, dx: float, dy: float) -> list[float]:
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


# ----------------------------------------------------------------------------- composites

SHAPE_COLORS = {'red': (220, 30, 30), 'blue': (30, 70, 220), 'green': (20, 150, 50), 'orange': (245, 140, 0),
                'purple': (130, 40, 170)}
SHAPES = ('circle', 'square', 'triangle', 'diamond')
COMP_SIZES = (32, 64, 128, 160)
CUTOUT_CATEGORIES = ('dog', 'bird', 'cat', 'cow', 'horse', 'sheep', 'giraffe', 'bear', 'zebra', 'elephant',
                     'aeroplane', 'train', 'truck', 'boat', 'bicycle', 'car_(automobile)', 'car', 'motorbike', 'bus')
BG_FAMILIES = ('furniture', 'tableware', 'electronics', 'food')
CANVAS = 512


def cutout(r3case: dict) -> Image.Image:
    """RGBA crop of a round-3 object: RGB from its image, alpha from its instance mask, cropped to the mask box."""
    img = sr.load_image(r3case)
    mask = np.asarray(Image.open(r3case['mask_path'])) > 0
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError(f"Empty mask for {r3case['id']}")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgba = np.dstack([np.asarray(img)[y0:y1, x0:x1], (mask[y0:y1, x0:x1] * 255).astype(np.uint8)])
    return Image.fromarray(rgba, 'RGBA')


def background(spec: dict, r3: dict) -> Image.Image:
    if spec['bg'] == 'plain':
        return Image.new('RGB', (CANVAS, CANVAS), (255, 255, 255))
    img = sr.load_image(r3[spec['bg_src']])
    s = CANVAS / min(img.size)
    img = img.resize((max(CANVAS, round(img.width * s)), max(CANVAS, round(img.height * s))), Image.BICUBIC)
    x0, y0 = (img.width - CANVAS) // 2, (img.height - CANVAS) // 2
    return img.crop((x0, y0, x0 + CANVAS, y0 + CANVAS))


def draw_object(img: Image.Image, obj: dict, box, r3: dict) -> None:
    if obj['kind'] == 'shape':
        cfn.draw_shape(ImageDraw.Draw(img), obj['shape'], tuple(box), SHAPE_COLORS[obj['color']])
        return
    cut = cutout(r3[obj['src']])
    w, h = round(box[2] - box[0]), round(box[3] - box[1])
    cut = cut.resize((w, h), Image.LANCZOS)
    img.paste(cut.convert('RGB'), (round(box[0]), round(box[1])), cut.split()[3])


def render_comp(spec: dict, r3: dict, moved: bool) -> Image.Image:
    img = background(spec, r3)
    for i, o in enumerate(spec['objects']):
        box = shift_box(o['box'], *spec['move']) if (moved and i == 0) else o['box']
        draw_object(img, o, box, r3)
    return img


def render_rect(spec: dict, box) -> Image.Image:
    img = Image.new('RGB', (CANVAS, CANVAS), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([round(v) for v in box], fill=SHAPE_COLORS[spec['color']])
    return img


def render(case: dict, r3: dict, variant=None) -> tuple[Image.Image, list[float]]:
    """Image and target box of run A (variant None) or of a variant: 'B' (translation / moved target) or a
    (side, delta) one-edge move."""
    s = case['set']
    if s == 'comp':
        moved = variant == 'B'
        box = shift_box(case['box'], *case['move']) if moved else case['box']
        return render_comp(case, r3, moved), box
    if s == 'rect':
        box = case['box'] if variant is None else stretched_box(case['box'], *variant)
        return render_rect(case, box), box
    img = sr.load_image(r3[case['src']])
    if variant is None:
        return img, case['box']
    if variant == 'B':
        return translate_image(img, *case['shift']), shift_box(case['box'], *case['shift'])
    return stretch_image(img, case['box'], *variant), stretched_box(case['box'], *variant)


# ----------------------------------------------------------------------------- case building (CPU)

def _fits(b, others, gap, canvas=CANVAS) -> bool:
    if b[0] < 0 or b[1] < 0 or b[2] > canvas or b[3] > canvas:
        return False
    return all(b[0] >= o[2] + gap or o[0] >= b[2] + gap or b[1] >= o[3] + gap or o[1] >= b[3] + gap for o in others)


def place(rng, sizes: list[tuple[float, float]], gap: int = 16, shift: int = 3 * TP, tries: int = 300):
    """Boxes for objects of the given (w, h) and a move (dx, dy) of object 0 by `shift` pixels on both axes that
    keeps it inside the canvas and free of the others; None when no layout is found."""
    for _ in range(tries):
        boxes = []
        for w, h in sizes:
            for _ in range(100):
                x0, y0 = rng.uniform(gap, CANVAS - w - gap), rng.uniform(gap, CANVAS - h - gap)
                b = [float(x0), float(y0), float(x0 + w), float(y0 + h)]
                if _fits(b, boxes, gap):
                    boxes.append(b)
                    break
            else:
                break
        if len(boxes) != len(sizes):
            continue
        moves = [(sx * shift, sy * shift) for sx in (-1, 1) for sy in (-1, 1)]
        rng.shuffle(moves)
        for dx, dy in moves:
            if _fits(shift_box(boxes[0], dx, dy), boxes[1:], gap // 2):
                return boxes, [int(dx), int(dy)]
    return None


def translation_shift(box, canvas, k: int = 3) -> list[int] | None:
    """Shift of k tokens per axis towards the image centre that keeps the box inside the canvas."""
    W, H = canvas
    out = []
    for lo, hi, n in ((box[0], box[2], W), (box[1], box[3], H)):
        d = k * TP if (lo + hi) / 2 < n / 2 else -k * TP
        if lo + d < 0 or hi + d > n:
            d = -d
            if lo + d < 0 or hi + d > n:
                return None
        out.append(int(d))
    return out


def stage_cases(out: Path, args) -> None:
    rng = np.random.default_rng(args.seed)
    r3 = eh.load_cases(args.round3)
    cases: list[dict] = []
    groups = ('rc', 'paco', 'vocpart', 'stuff')

    def base(c: dict) -> dict:
        return {'src': c['id'], 'group': c['group'], 'family': c['family'], 'category': c['category'],
                'expr': c['expression'], 'canvas': c['canvas'], 'box': c['truth']['box'], 'area_frac': c['area_frac']}

    # nat: translation pairs on large boxes
    for g in groups:
        cand = [c for c in r3.values() if c['group'] == g and box_span_ok(c['truth']['box'], grid_of(c['canvas']))]
        cand = [c for c in cand if translation_shift(c['truth']['box'], c['canvas']) is not None]
        for c in [cand[i] for i in rng.permutation(len(cand))[:args.n_nat]]:
            cases.append({'id': f"nat_{c['id']}", 'set': 'nat', **base(c),
                          'shift': translation_shift(c['truth']['box'], c['canvas'])})
    # str: one-edge moves on natural images
    deltas = (-64, -32, 32, 64, 96)
    for g in groups:
        cand = [c for c in r3.values() if c['group'] == g and box_span_ok(c['truth']['box'], grid_of(c['canvas']))]
        cand = [c for c in cand if any(stretch_feasible(c['truth']['box'], c['canvas'], s, 96) for s in SIDES)]
        for c in [cand[i] for i in rng.permutation(len(cand))[:args.n_str]]:
            b = c['truth']['box']
            var = [[s, d] for s in SIDES for d in deltas if stretch_feasible(b, c['canvas'], s, d)]
            scan = [[s, 96] for s in SIDES if stretch_feasible(b, c['canvas'], s, 96)]
            cases.append({'id': f"str_{c['id']}", 'set': 'str', **base(c), 'variants': var, 'scan': scan})
    # rect: one solid rectangle
    for i in range(args.n_rect):
        w, h = rng.uniform(112, 224, 2)
        x0, y0 = rng.uniform(100, CANVAS - w - 100), rng.uniform(100, CANVAS - h - 100)
        b = [float(x0), float(y0), float(x0 + w), float(y0 + h)]
        color = str(rng.choice(list(SHAPE_COLORS)))
        var = [[s, d] for s in SIDES for d in deltas if stretch_feasible(b, (CANVAS, CANVAS), s, d)]
        cases.append({'id': f'rect_{i:03d}', 'set': 'rect', 'color': color, 'expr': f'the {color} rectangle',
                      'canvas': [CANVAS, CANVAS], 'box': b, 'area_frac': w * h / CANVAS ** 2, 'variants': var,
                      'scan': [['right', 96], ['bottom', 96]] if all(
                          stretch_feasible(b, (CANVAS, CANVAS), s, 96) for s in ('right', 'bottom')) else []})
    # comp: H11 study composites
    cut_src = [c for c in r3.values() if c['category'] in CUTOUT_CATEGORIES and c['group'] != 'rc'
               and c.get('mask_path') and c['area_frac'] >= 0.04 and _interior(c)]
    bg_src = [c['id'] for c in r3.values() if c['family'] in BG_FAMILIES and c['group'] in ('paco', 'stuff')]
    if len(cut_src) < 20 or len(bg_src) < 20:
        raise ValueError(f'Too few cut-out ({len(cut_src)}) or background ({len(bg_src)}) sources')
    k = 0
    for objtype in ('shape', 'cutout'):
        for bg in ('plain', 'photo'):
            for size in COMP_SIZES:
                for nd in (0, 2):
                    made, attempts = 0, 0
                    while made < args.n_comp:
                        attempts += 1
                        if attempts > 50 * args.n_comp:
                            raise RuntimeError(f'Cannot place comp cell {objtype}/{bg}/{size}/{nd}')
                        spec = comp_spec(rng, objtype, bg, size, nd, cut_src, bg_src)
                        if spec is None:
                            continue
                        spec['id'] = f'comp_{k:04d}'
                        k += 1
                        made += 1
                        cases.append(spec)
    (out / 'cases.json').write_text(json.dumps(cases))
    from collections import Counter
    print(Counter(c['set'] for c in cases))


def grid_of(canvas) -> tuple[int, int]:
    W, H = canvas
    return H // TP, W // TP


def _interior(c: dict) -> bool:
    W, H = c['canvas']
    b = c['truth']['box']
    return b[0] > 3 and b[1] > 3 and b[2] < W - 3 and b[3] < H - 3


def _cut_name(c: dict) -> str:
    return c['expression'][4:] if c['expression'].startswith('the ') else c['expression']


def comp_spec(rng, objtype: str, bg: str, size: int, nd: int, cut_src: list[dict], bg_src: list[str]) -> dict | None:
    if objtype == 'shape':
        color, shape = rng.choice(list(SHAPE_COLORS)), rng.choice(SHAPES)
        objs = [{'kind': 'shape', 'color': str(color), 'shape': str(shape), 'role': 'target'}]
        if nd:
            other_c = [c for c in SHAPE_COLORS if c != color]
            other_s = [s for s in SHAPES if s != shape]
            objs.append({'kind': 'shape', 'color': str(color), 'shape': str(rng.choice(other_s)), 'role': 'same_color'})
            objs.append({'kind': 'shape', 'color': str(rng.choice(other_c)), 'shape': str(shape), 'role': 'same_shape'})
        dims = [(size, size)] * len(objs)
        names = [f"the {o['color']} {o['shape']}" for o in objs]
    else:
        picks, used = [], set()
        for j in rng.permutation(len(cut_src)):
            c = cut_src[j]
            if _cut_name(c) in used:
                continue
            picks.append(c)
            used.add(_cut_name(c))
            if len(picks) == 1 + nd:
                break
        objs, dims = [], []
        for i, c in enumerate(picks):
            b = c['truth']['box']
            w, h = b[2] - b[0], b[3] - b[1]
            s = size / max(w, h)
            dims.append((max(8.0, round(w * s)), max(8.0, round(h * s))))
            objs.append({'kind': 'cutout', 'src': c['id'], 'name': _cut_name(c), 'role': 'target' if i == 0 else 'other'})
        names = [f"the {o['name']}" for o in objs]
    placed = place(rng, dims)
    if placed is None:
        return None
    boxes, move = placed
    for o, b in zip(objs, boxes):
        o['box'] = b
    spec = {'set': 'comp', 'objtype': objtype, 'bg': bg, 'size': size, 'ndistr': nd, 'objects': objs,
            'box': boxes[0], 'move': move, 'expr': names[0], 'canvas': [CANVAS, CANVAS],
            'bg_src': str(rng.choice(bg_src)) if bg == 'photo' else None, 'area_frac': dims[0][0] * dims[0][1] / CANVAS ** 2}
    if nd:
        spec['expr_b'] = names[1]
        spec['box_b'] = boxes[1]
    return spec


def load_cases(out: Path) -> dict:
    return {c['id']: c for c in json.loads((out / 'cases.json').read_text())}


def formats_of(case: dict) -> tuple[str, ...]:
    if case['set'] == 'comp':
        return ('point', 'bbox') if case['size'] >= 128 else ('point',)
    return ('point', 'bbox')


def comp_family(case: dict) -> str:
    return f"{case['objtype']}_{case['bg']}"


def mean_key(case: dict) -> tuple[str, str]:
    """(means file tag, family) used for mean ablation of this item."""
    if case['set'] == 'comp':
        return 'own', comp_family(case)
    if case['set'] == 'rect':
        return 'own', 'shape_plain'
    return 'r3', 'rc' if case['group'] == 'rc' else 'mc'


def score(box, values, fmt, canvas) -> dict:
    return eh.score({'truth': {'canvas': canvas, 'box': box}}, values, fmt)


# ----------------------------------------------------------------------------- hooks

class Capture:
    """Run-level capture (batch row 0) at positions >= start: token-mixer inputs of the output projection
    (all layers), MLP outputs (all layers) and q_proj outputs (Q_LAYERS)."""

    def __init__(self, rt, start: int):
        self.mixer, self.mlp, self.q = {}, {}, {}
        self.handles = []
        for layer in range(1, N_LAYERS + 1):
            module, _ = cr.mixer_projection(rt, layer)

            def pre(module, args, layer=layer):
                self.mixer[layer] = args[0][0, start:].detach().clone()

            def mlp_hook(module, args, output, layer=layer):
                self.mlp[layer] = output[0, start:].detach().clone()
            self.handles.append(module.register_forward_pre_hook(pre))
            self.handles.append(rt.layers[layer - 1].mlp.register_forward_hook(mlp_hook))
            if layer in Q_LAYERS:
                def q_hook(module, args, output, layer=layer):
                    self.q[layer] = output[0, start:].detach().clone()
                self.handles.append(rt.layers[layer - 1].self_attn.q_proj.register_forward_hook(q_hook))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


class Edits:
    """Per-row edits at positions >= start of a teacher-forced batched forward.

    ops: (row, kind, layer, head, src). kind 'head' edits one head slice of the mixer output-projection input
    (head -1: the whole mixer, i.e. a Gated DeltaNet layer or all heads of a full-attention layer); 'mlp' the MLP
    output (head ignored); 'q' the query half of one head in q_proj's output. src: 'A' or 'B' (copy from that
    run's Capture), 'mean' (mean head output) or ('mix', a) = A + a (B - A).
    """

    def __init__(self, rt, ops: list[tuple], start: int, donors: dict, means: dict | None = None):
        self.handles = []
        by: dict[tuple[str, int], list] = {}
        for op in ops:
            by.setdefault((op[1], op[2]), []).append(op)
        for (kind, layer), lops in by.items():
            if kind == 'head':
                module, hd = cr.mixer_projection(rt, layer)
                store = 'mixer'
            elif kind == 'mlp':
                module, hd, store = rt.layers[layer - 1].mlp, None, 'mlp'
            elif kind == 'q':
                module, hd, store = rt.layers[layer - 1].self_attn.q_proj, rt.layers[layer - 1].self_attn.head_dim, 'q'
            else:
                raise ValueError(kind)

            def apply(x, lops=lops, hd=hd, store=store, layer=layer, kind=kind):
                n = x.shape[1] - start
                y = x.clone()
                for row, _, _, h, src in lops:
                    if kind == 'q':
                        sl = slice(h * 2 * hd, h * 2 * hd + hd)
                    else:
                        sl = slice(None) if (h is None or h < 0) else slice(h * hd, (h + 1) * hd)
                    if src == 'mean':
                        val = means[(layer, h)].to(x.dtype)
                    elif isinstance(src, tuple):
                        a = getattr(donors['A'], store)[layer][:, sl].float()
                        b = getattr(donors['B'], store)[layer][:, sl].float()
                        val = (a + src[1] * (b - a)).to(x.dtype)
                    else:
                        val = getattr(donors[src], store)[layer][:, sl].to(x.dtype)
                        if val.shape[0] != n:
                            raise ValueError('Donor tail length differs from the edited tail.')
                    y[row, start:, sl] = val
                return y
            if kind == 'head':
                self.handles.append(module.register_forward_pre_hook(lambda m, args, apply=apply: (apply(args[0]),)))
            else:
                self.handles.append(module.register_forward_hook(lambda m, args, out, apply=apply: apply(out)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def attn_forward(rt, x: torch.Tensor, pos: torch.Tensor, queries: list[int], layers, logit_pos: list[int],
                 extras: dict | None = None):
    """One batched forward. Returns digit-position logits (B, len(logit_pos), V), attention rows
    {layer: (B, nq, 16, L)} at the queries computed from the actual (possibly edited) q_proj outputs, and the
    max relative error of the recomputed head outputs against the unedited o_proj input.

    extras (optional dict) receives 'heads' {layer: (B, nq, 16, hd)} gated head outputs at the queries,
    'resid' {layer: (B, nq, D)} decoder-layer outputs at the queries, and 'kv' {layer: (k, v)} image-span-free
    full key / value tensors of row 0 (B=1 only) for the layers in extras['kv_layers'].
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    store, handles = {}, []
    q_idx_cpu = torch.tensor(queries)
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn

        def pre(module, args, kwargs, layer=layer):
            store.setdefault(layer, {})['h'] = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            store[layer]['pe'] = kwargs['position_embeddings']

        def qh(module, args, out, layer=layer):
            store.setdefault(layer, {})['q'] = out[:, q_idx_cpu.to(out.device)].detach()

        def opre(module, args, layer=layer):
            store.setdefault(layer, {})['o_in'] = args[0][:, q_idx_cpu.to(args[0].device)].detach()
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(mod.q_proj.register_forward_hook(qh))
        handles.append(mod.o_proj.register_forward_pre_hook(opre, prepend=True))
    res_layers = extras.get('res_layers', ()) if extras is not None else ()
    for layer in res_layers:
        def rh(module, args, out, layer=layer):
            h = out[0] if isinstance(out, tuple) else out
            store.setdefault(('res', layer), h[:, q_idx_cpu.to(h.device)].detach().float().cpu())
        handles.append(rt.layers[layer - 1].register_forward_hook(rh))
    try:
        logits = rt.model(inputs_embeds=x, position_ids=pos, use_cache=False,
                          logits_to_keep=torch.tensor(logit_pos, device=x.device)).logits.float()
    finally:
        for h in handles:
            h.remove()
    attn, err = {}, 0.0
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn
        h, (cos, sin) = store[layer]['h'], store[layer]['pe']
        dev = h.device
        B, L, _ = h.shape
        q_idx = q_idx_cpu.to(dev)
        hd = mod.head_dim
        q, gate = torch.chunk(store[layer]['q'].view(B, len(queries), -1, hd * 2), 2, dim=-1)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(B, L, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(B, L, -1, hd).transpose(1, 2)
        cos, sin = cos.to(dev), sin.to(dev)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, q_idx], sin[:, q_idx])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        if extras is not None and layer in extras.get('kv_layers', ()):
            extras.setdefault('kv', {})[layer] = (k[0].float().cpu(), v[0].float().cpu())
        k, v = repeat_kv(k, mod.num_key_value_groups), repeat_kv(v, mod.num_key_value_groups)
        sc = (q.float() @ k.float().transpose(2, 3)) * mod.scaling
        sc = sc.masked_fill(torch.arange(L, device=dev)[None, :] > q_idx[:, None], float('-inf'))
        a = sc.softmax(-1)
        o = ((a @ v.float()) * torch.sigmoid(gate.float()).transpose(1, 2)).transpose(1, 2)   # (B, nq, 16, hd)
        ref = store[layer]['o_in'].float().view(B, len(queries), -1, hd)
        err = max(err, float(((o - ref).abs().amax((1, 2, 3)) / ref.abs().amax((1, 2, 3))).max()))
        attn[layer] = a.transpose(1, 2).cpu()
        if extras is not None:
            extras.setdefault('heads', {})[layer] = ref.cpu()
    if extras is not None:
        extras['resid'] = {l: store[('res', l)] for l in res_layers}
    return logits, attn, err


def plain_logits(rt, x: torch.Tensor, pos: torch.Tensor, logit_pos: list[int]) -> torch.Tensor:
    return rt.model(inputs_embeds=x, position_ids=pos, use_cache=False,
                    logits_to_keep=torch.tensor(logit_pos, device=x.device)).logits.float()


# ----------------------------------------------------------------------------- head groups and conditions

def parse_heads(spec: str) -> list[tuple[int, int]]:
    out = []
    for t in filter(None, spec.split(',')):
        l, h = t.split('.')
        if int(l) not in FULL_LAYERS or not 0 <= int(h) < 16:
            raise ValueError(f'Invalid head {t}')
        out.append((int(l), int(h)))
    return out


def head_groups(args) -> dict[str, list[tuple[int, int]]]:
    """Attention-defined reader groups of round 3 (point answers, layers >= 16, edge-regression R^2 >= 0.5)."""
    g = {k: parse_heads(getattr(args, f'group_{k}')) for k in ('minx', 'maxx', 'ctrx', 'miny', 'maxy', 'ctry')}
    for k, v in g.items():
        if (20, 11) in v:
            raise ValueError(f'L20 H11 must not be part of group {k}; it is handled separately.')
    return g


def _heads(hs, src):
    return [('head', l, h, src) for l, h in hs]


def conditions(fmt: str, kind: str, groups: dict) -> list[tuple[str, list[tuple]]]:
    """Condition rows (name, ops without row index) for one pair kind: 'where' (translation, moved target or
    one-edge move) or 'what' (other referent, same image)."""
    allh = [(l, h) for l in FULL_LAYERS for h in range(16)]
    dn = [('head', l, -1, 'B') for l in DN_LAYERS]
    c = [('none', []), ('H11_B', _heads([(20, 11)], 'B')), ('H11_mean', _heads([(20, 11)], 'mean')),
         ('L20_B', _heads([(20, h) for h in range(16)], 'B')), ('full_B', _heads(allh, 'B')), ('dn_B', dn),
         ('q11_B', [('q', 20, 11, 'B')])]
    if kind == 'what':
        c += [('L16H1H2_B', _heads([(16, 1), (16, 2)], 'B')),
              ('dn_pre20_B', [('head', l, -1, 'B') for l in DN_LAYERS if l < 20])]
        return c
    if fmt == 'bbox':
        c += [(f'L{l}_B', _heads([(l, h) for h in range(16)], 'B')) for l in (16, 24, 28)]
        return c
    for ax in ('x', 'y'):
        mn, mx, ct = groups[f'min{ax}'], groups[f'max{ax}'], groups[f'ctr{ax}']
        both = mn + mx
        late = [(l, h) for l, h in allh if l >= 24 and (l, h) not in both]
        c += [(f'min{ax}_B', _heads(mn, 'B')), (f'max{ax}_B', _heads(mx, 'B')), (f'ctr{ax}_B', _heads(ct, 'B')),
              (f'edges{ax}_B', _heads(both, 'B')),
              (f'max{ax}_mix0.5', _heads(mx, ('mix', 0.5))), (f'max{ax}_mix2', _heads(mx, ('mix', 2.0))),
              (f'edges{ax}_B_freeze_late_heads', _heads(both, 'B') + _heads(late, 'A')),
              (f'edges{ax}_B_freeze_dn21', _heads(both, 'B') + [('head', l, -1, 'A') for l in DN_LAYERS if l > 20]),
              (f'edges{ax}_B_freeze_mlp21_27', _heads(both, 'B') + [('mlp', l, None, 'A') for l in range(21, 28)]),
              (f'edges{ax}_B_freeze_mlp28_32', _heads(both, 'B') + [('mlp', l, None, 'A') for l in range(28, 33)]),
              (f'q_edges{ax}_B', [('q', l, h, 'B') for l, h in both if l in Q_LAYERS])]
    return c


def scan_conditions() -> list[tuple[str, list[tuple]]]:
    return [(f'h{l}.{h}', [('head', l, h, 'B')]) for l in FULL_LAYERS for h in range(16)]


# ----------------------------------------------------------------------------- shared GPU helpers

def _ld(lg_rows: torch.Tensor, lg_b: torch.Tensor, ans: list[int], values: list[dict], digits: list[int]) -> list[dict]:
    """First-digit logit differences. lg_rows (R, nv, V) and lg_b (nv, V) at the first-digit queries."""
    dig = torch.tensor(digits, device=lg_b.device)
    out = []
    for k, v in enumerate(values):
        dA = ans[v['toks'][0]]
        dB = digits[int(lg_b[k, dig].argmax())]
        out.append({'key': v['key'], 'dA': digits.index(dA) if dA in digits else None, 'dB': digits.index(dB),
                    'LD_B': round(float(lg_b[k, dB] - lg_b[k, dA]), 4),
                    'ld': [round(float(t), 4) for t in (lg_rows[:, k, dB] - lg_rows[:, k, dA]).cpu()],
                    'p_digits_B': [round(float(t), 4) for t in lg_b[k, dig].softmax(-1).cpu()]})
    return out


def digit_queries(values: list[dict], L: int) -> list[tuple[str, int, int]]:
    """(value key, digit slot, absolute query) for every digit of every value; slot -1 is the token before the
    query of the first digit (where 3.12 found the first digit's coordinate read starts)."""
    out = []
    for v in values:
        q0 = L - 1 + v['toks'][0]
        out.append((v['key'], -1, q0 - 1))
        for k, t in enumerate(v['toks']):
            out.append((v['key'], k, L - 1 + t))
    return out


class Ctx:
    """Per-run shared state: runtime, cases, round-3 cases, means and digit ids."""

    def __init__(self, rt, out: Path, args):
        self.rt, self.out, self.args = rt, out, args
        self.cases = load_cases(out)
        self.r3 = eh.load_cases(args.round3)
        self.digits = cf.digit_ids(rt)
        self.groups = head_groups(args)
        self._means: dict = {}

    def means(self, case: dict):
        tag, fam = mean_key(case)
        if (tag, fam) not in self._means:
            src = self.args.round3 if tag == 'r3' else self.out
            self._means[(tag, fam)] = eh.head_means(self.rt, src, fam)
        return self._means[(tag, fam)]

    def prompt(self, case: dict, fmt: str, which: str = 'a') -> str:
        return eh.prompt_for(case['expr_b'] if which == 'b' else case['expr'], fmt)


def gen_answers(ctx: Ctx) -> dict:
    return {g['id']: g for g in read_jsonl(ctx.out / 'gen.jsonl')}


# ----------------------------------------------------------------------------- stage: gen

@torch.inference_mode()
def stage_gen(ctx: Ctx) -> None:
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / 'gen.jsonl'
    finished = done_ids(path)
    items = [(c, f) for c in ctx.cases.values() for f in formats_of(c) if f"{c['id']}:{f}" not in finished]
    for i in range(0, len(items), args.prep_chunk):
        chunk = items[i:i + args.prep_chunk]
        preps = []
        for c, f in chunk:
            img, box = render(c, ctx.r3)
            preps.append(cr.prepare(rt, img, ctx.prompt(c, f)))
        by_len: dict[int, list] = {}
        for it, p in zip(chunk, preps):
            by_len.setdefault(len(p.ids), []).append((it, p))
        for L, grp in by_len.items():
            for j in range(0, len(grp), args.batch_size):
                part = grp[j:j + args.batch_size]
                res = cr.run_conditions(rt, [cr.embeds(rt, p) for _, p in part], [p.pos for _, p in part],
                                        args.max_new_tokens, batch_size=len(part))
                rows = []
                for ((c, f), p), r in zip(part, res):
                    vals = cf.answer_values(rt, r['ids'], f)
                    rows.append({'id': f"{c['id']}:{f}", 'case': c['id'], 'set': c['set'], 'format': f, 'text': r['text'],
                                 'ids': r['ids'], 'prompt_len': L, 'batch_size': len(part), 'grid': list(p.grid),
                                 'values': vals, 'score': score(c['box'], vals, f, c['canvas'])})
                append_jsonl(path, rows)
        print(f'gen {i + len(chunk)}/{len(items)}', flush=True)
        del preps
        torch.cuda.empty_cache()


@torch.inference_mode()
def stage_spotcheck(ctx: Ctx) -> None:
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / 'spotcheck.jsonl'
    finished = done_ids(path)
    gen = read_jsonl(out / 'gen.jsonl')
    rng = np.random.default_rng(args.seed)
    for k in sorted(rng.choice(len(gen), min(args.spotcheck, len(gen)), replace=False)):
        g = gen[int(k)]
        if g['id'] in finished:
            continue
        c = ctx.cases[g['case']]
        img, _ = render(c, ctx.r3)
        ref = qr.generate(rt, qr.build(rt, [img], ctx.prompt(c, g['format'])), args.max_new_tokens)
        vr = cf.answer_values(rt, rt.tokenizer.encode(ref['text'], add_special_tokens=False), g['format'])
        val = lambda v: None if v is None else [(d['key'], d['value']) for d in v]
        append_jsonl(path, [{'id': g['id'], 'reference': ref['text'], 'batched': g['text'], 'batch_size': g['batch_size'],
                             'answer_equal': val(vr) == val(g['values'])}])


# ----------------------------------------------------------------------------- stage: means

@torch.inference_mode()
def stage_means(ctx: Ctx) -> None:
    rt, out, args = ctx.rt, ctx.out, ctx.args
    if (out / 'means.npz').exists():
        return
    gen = gen_answers(ctx)
    res = {}
    for fam in ('shape_plain', 'shape_photo', 'cutout_plain', 'cutout_photo'):
        cs = [c for c in ctx.cases.values() if c['set'] == 'comp' and comp_family(c) == fam
              and gen.get(f"{c['id']}:point", {}).get('score', {}).get('valid')]
        cs = cs[::max(1, len(cs) // args.n_means)][:args.n_means]
        acc, n = {l: 0.0 for l in FULL_LAYERS}, 0
        for c in cs:
            g = gen[f"{c['id']}:point"]
            img, _ = render(c, ctx.r3)
            prep = cr.prepare(rt, img, ctx.prompt(c, 'point'))
            L = len(prep.ids)
            x, pos = cr.teacher_forced(rt, prep, g['ids'][:-1])
            with cfn.TailCapture(rt, L - 1) as cap:
                rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
            for l in FULL_LAYERS:
                acc[l] = acc[l] + cap.store[l].float().sum(0).cpu()
            n += cap.store[FULL_LAYERS[0]].shape[0]
        res[fam] = {l: (acc[l] / n).numpy() for l in FULL_LAYERS}
        print(f'means {fam}: {len(cs)} answers, {n} positions', flush=True)
    np.savez_compressed(out / 'means.npz', **{f'{f}_{l}': v for f, d in res.items() for l, v in d.items()})


# ----------------------------------------------------------------------------- stage: behav

@torch.inference_mode()
def stage_behav(ctx: Ctx) -> None:
    """Greedy answers for every one-edge variant (all variants of an image share one batch)."""
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / 'behav.jsonl'
    finished = done_ids(path)
    for c in ctx.cases.values():
        if c['set'] not in ('str', 'rect'):
            continue
        for f in ('point', 'bbox'):
            rid = f"{c['id']}:{f}"
            if rid in finished:
                continue
            variants = [None] + [tuple(v) for v in c['variants']]
            preps, boxes = [], []
            for v in variants:
                img, box = render(c, ctx.r3, v)
                preps.append(cr.prepare(rt, img, ctx.prompt(c, f)))
                boxes.append(box)
            if len({len(p.ids) for p in preps}) != 1:
                raise ValueError(f'Variant prompt lengths differ for {rid}')
            res = cr.run_conditions(rt, [cr.embeds(rt, p) for p in preps], [p.pos for p in preps],
                                    args.max_new_tokens, batch_size=args.batch_size)
            rows = []
            for v, b, r in zip(variants, boxes, res):
                vals = cf.answer_values(rt, r['ids'], f)
                rows.append({'variant': list(v) if v else None, 'box': b, 'text': r['text'],
                             'values': None if vals is None else {d['key']: d['value'] for d in vals},
                             'score': score(b, vals, f, c['canvas'])})
            append_jsonl(path, [{'id': rid, 'case': c['id'], 'set': c['set'], 'format': f, 'canvas': c['canvas'],
                                 'rows': rows}])
        print(f"behav {c['id']}", flush=True)


# ----------------------------------------------------------------------------- stage: ablate

def ablate_sets(args) -> dict[str, list[tuple[int, int]]]:
    return cfn.parse_sets(args.ablate_sets)


@torch.inference_mode()
def stage_ablate(ctx: Ctx) -> None:
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / 'ablate.jsonl'
    finished = done_ids(path)
    sets = ablate_sets(args)
    names = list(sets)
    for c in ctx.cases.values():
        if c['set'] != 'comp' or c['id'] in finished:
            continue
        img, box = render(c, ctx.r3)
        prep = cr.prepare(rt, img, ctx.prompt(c, 'point'))
        L = len(prep.ids)
        x = cr.embeds(rt, prep)
        ops = [(j, l, h, 'mean') for j, n in enumerate(names) for l, h in sets[n]]
        with cr.MixerEdit(rt, ops, L - 1, means=ctx.means(c)):
            res = cr.greedy(rt, x.expand(len(names), -1, -1).contiguous(),
                            prep.pos.expand(-1, len(names), -1).contiguous(), args.max_new_tokens)
        rows = []
        for n, r in zip(names, res):
            vals = cf.answer_values(rt, r['ids'], 'point')
            rows.append({'set': n, 'text': r['text'], 'values': None if vals is None else {d['key']: d['value'] for d in vals},
                         'score': score(box, vals, 'point', c['canvas'])})
        append_jsonl(path, [{'id': c['id'], 'results': rows}])
    print('ablate done', flush=True)


# ----------------------------------------------------------------------------- stage: probe

def probe_jobs(ctx: Ctx) -> list[tuple[dict, str, str, object]]:
    """(case, format, pair kind, variant) for every pair to probe; variant 'B', ('side', delta) or 'what'."""
    jobs = []
    for c in ctx.cases.values():
        if c['set'] == 'nat':
            jobs += [(c, f, 'where', 'B') for f in ('point', 'bbox')]
        elif c['set'] in ('str', 'rect'):
            jobs += [(c, 'point', 'where', tuple(v)) for v in c['scan']]
        elif c['set'] == 'comp':
            jobs += [(c, f, 'where', 'B') for f in formats_of(c)]
            if c['ndistr']:
                jobs.append((c, 'point', 'what', 'what'))
    return jobs


def job_id(c, f, kind, variant) -> str:
    v = variant if isinstance(variant, str) else f'{variant[0]}{variant[1]:+d}'
    return f"{c['id']}:{f}:{v}"


def wants_scan(c: dict, f: str, kind: str) -> bool:
    if kind != 'where':
        return False
    if c['set'] == 'comp':
        return c['size'] == max(COMP_SIZES) and f == 'point'
    return True


@torch.inference_mode()
def stage_probe(ctx: Ctx) -> None:
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / 'probe.jsonl'
    (out / 'probe').mkdir(exist_ok=True)
    finished = done_ids(path)
    gen = gen_answers(ctx)
    for c, f, kind, variant in probe_jobs(ctx):
        jid = job_id(c, f, kind, variant)
        if jid in finished:
            continue
        g = gen.get(f"{c['id']}:{f}")
        if g is None or not g['score']['valid']:
            append_jsonl(path, [{'id': jid, 'skipped': 'run A answer invalid'}])
            continue
        t0 = time.time()
        rec, arrays = probe_one(ctx, c, f, kind, variant, g)
        np.savez_compressed(out / 'probe' / f"{jid.replace(':', '__')}.npz", **arrays)
        rec.update({'id': jid, 'seconds': round(time.time() - t0, 1)})
        append_jsonl(path, [rec])
        print(f'probe {jid} {time.time() - t0:.1f}s', flush=True)


def probe_one(ctx: Ctx, c: dict, f: str, kind: str, variant, g: dict) -> tuple[dict, dict]:
    rt, args = ctx.rt, ctx.args
    img_a, box_a = render(c, ctx.r3)
    if kind == 'what':
        img_b, box_b = img_a, c['box_b']
        prompt_b = ctx.prompt(c, f, 'b')
    else:
        img_b, box_b = render(c, ctx.r3, variant)
        prompt_b = ctx.prompt(c, f)
    pa, pb = cr.prepare(rt, img_a, ctx.prompt(c, f)), cr.prepare(rt, img_b, prompt_b)
    La, Lb = len(pa.ids), len(pb.ids)
    ans, values = g['ids'], g['values']
    xa, pos_a = cr.teacher_forced(rt, pa, ans[:-1])
    xb, pos_b = cr.teacher_forced(rt, pb, ans[:-1])
    dq = digit_queries(values, La)
    q_all = [q for _, _, q in dq]
    first = [L for L in (La - 1 + v['toks'][0] for v in values)]
    first_b = [Lb - 1 + v['toks'][0] for v in values]
    s, e = pa.span['start'], pa.span['end']
    sb, eb = pb.span['start'], pb.span['end']
    grid = pa.grid
    # run A: captures, attention at every digit query, head outputs / residuals / H11 k, v at first-digit queries
    extras = {'res_layers': RES_LAYERS, 'kv_layers': (20,)}
    with Capture(rt, La - 1) as cap_a:
        lg_a, att_a, err_a = attn_forward(rt, xa, pos_a, q_all, ATT_LAYERS, first, extras)
    first_idx = [q_all.index(q) for q in first]
    att_img = np.stack([att_a[l][0, :, :, s:e].numpy() for l in ATT_LAYERS], 1)          # (nq, nL, 16, Nv)
    att_img = att_img / att_img.sum(-1, keepdims=True)
    heads = np.stack([extras['heads'][l][0, first_idx].numpy() for l in OUT_LAYERS], 1)   # (nv, nOut, 16, hd)
    resid = np.stack([extras['resid'][l][0, first_idx].numpy() for l in RES_LAYERS], 1)   # (nv, nRes, D)
    kv = extras['kv'][20]
    kv_head = 11 // rt.layers[19].self_attn.num_key_value_groups
    k11, v11 = kv[0][kv_head, s:e].numpy(), kv[1][kv_head, s:e].numpy()
    # run B (donor)
    with Capture(rt, Lb - 1) as cap_b:
        lg_b, att_b, _ = attn_forward(rt, xb, pos_b, first_b, REC_LAYERS, first_b)
    att_b_img = np.stack([att_b[l][0, :, :, sb:eb].numpy() for l in REC_LAYERS], 1)
    att_b_img = att_b_img / att_b_img.sum(-1, keepdims=True)
    if cap_a.mixer[1].shape[0] != cap_b.mixer[1].shape[0]:
        raise ValueError('Run A and run B tails differ in length.')
    donors = {'A': cap_a, 'B': cap_b}
    means = ctx.means(c)
    # condition rows
    conds = conditions(f, kind, ctx.groups)
    lg_rows, cond_att, errs = [], [], []
    for i in range(0, len(conds), args.cond_batch):
        part = conds[i:i + args.cond_batch]
        ops = [(j, *op) for j, (_, o) in enumerate(part) for op in o]
        R = len(part)
        with Edits(rt, ops, La - 1, donors, means):
            lg, att, err = attn_forward(rt, xa.expand(R, -1, -1).contiguous(), pos_a.expand(-1, R, -1).contiguous(),
                                        first, COND_LAYERS, first)
        lg_rows.append(lg)
        cond_att.append(np.stack([att[l][:, :, :, s:e].numpy() for l in COND_LAYERS], 2))   # (R, nv, 3, 16, Nv)
        errs.append(err)
    lg_rows = torch.cat(lg_rows, 0)
    cond_att = np.concatenate(cond_att, 0)
    cond_att = cond_att / cond_att.sum(-1, keepdims=True)
    lds = _ld(lg_rows, lg_b[0], ans, values, ctx.digits)
    rec = {'case': c['id'], 'set': c['set'], 'format': f, 'kind': kind, 'variant': variant if isinstance(variant, str) else list(variant),
           'grid': list(grid), 'box_a': box_a, 'box_b': box_b, 'values': [{k: v[k] for k in ('key', 'value', 'toks')} for v in values],
           'digit_queries': [[k, slot] for k, slot, _ in dq], 'conds': [n for n, _ in conds], 'lds': lds,
           'attn_check': max(err_a, *errs)}
    arrays = {'att_a': att_img.astype(np.float16), 'att_b': att_b_img.astype(np.float16),
              'cond_att': cond_att.astype(np.float16), 'heads': heads.astype(np.float16), 'resid': resid.astype(np.float16),
              'k11': k11.astype(np.float16), 'v11': v11.astype(np.float16)}
    if wants_scan(c, f, kind):
        sc = scan_conditions()
        rows = []
        for i in range(0, len(sc), args.scan_batch):
            part = sc[i:i + args.scan_batch]
            ops = [(j, *op) for j, (_, o) in enumerate(part) for op in o]
            R = len(part)
            with Edits(rt, ops, La - 1, donors, means):
                rows.append(plain_logits(rt, xa.expand(R, -1, -1).contiguous(), pos_a.expand(-1, R, -1).contiguous(), first))
        rec['scan'] = [n for n, _ in sc]
        rec['scan_lds'] = _ld(torch.cat(rows, 0), lg_b[0], ans, values, ctx.digits)
    return rec, arrays


# ----------------------------------------------------------------------------- stage: mediate

MED_LAYERS = tuple(range(20, 33))


def mediate_conditions(groups: dict) -> list[tuple[str, list[tuple]]]:
    """Edge groups patched from B, with single MLP layers frozen to run A (does layer l carry the effect?) and with
    every MLP of layers 20-32 frozen except layer l (is layer l alone enough?)."""
    c = [('none', [])]
    for ax in ('x', 'y'):
        both = _heads(groups[f'min{ax}'] + groups[f'max{ax}'], 'B')
        c.append((f'edges{ax}_B', both))
        c.append((f'edges{ax}_B_freeze_mlp20_32', both + [('mlp', l, None, 'A') for l in MED_LAYERS]))
        for l in MED_LAYERS:
            c.append((f'edges{ax}_B_freeze_mlp{l}', both + [('mlp', l, None, 'A')]))
            c.append((f'edges{ax}_B_only_mlp{l}', both + [('mlp', k, None, 'A') for k in MED_LAYERS if k != l]))
    return c


@torch.inference_mode()
def stage_mediate(ctx: Ctx) -> None:
    """Teacher-forced first-digit logits under mediate_conditions for every point probe pair of nat, str and rect."""
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / 'mediate.jsonl'
    finished = done_ids(path)
    gen = gen_answers(ctx)
    conds = mediate_conditions(ctx.groups)
    for c, f, kind, variant in probe_jobs(ctx):
        if c['set'] not in ('nat', 'str', 'rect') or f != 'point':
            continue
        jid = job_id(c, f, kind, variant)
        g = gen.get(f"{c['id']}:{f}")
        if jid in finished or g is None or not g['score']['valid']:
            continue
        t0 = time.time()
        img_a, _ = render(c, ctx.r3)
        img_b, _ = render(c, ctx.r3, variant)
        pa, pb = cr.prepare(rt, img_a, ctx.prompt(c, f)), cr.prepare(rt, img_b, ctx.prompt(c, f))
        La = len(pa.ids)
        xa, pos_a = cr.teacher_forced(rt, pa, g['ids'][:-1])
        xb, pos_b = cr.teacher_forced(rt, pb, g['ids'][:-1])
        first = [La - 1 + v['toks'][0] for v in g['values']]
        with Capture(rt, La - 1) as cap_a:
            rt.model(inputs_embeds=xa, position_ids=pos_a, use_cache=False, logits_to_keep=1)
        with Capture(rt, La - 1) as cap_b:
            lg_b = plain_logits(rt, xb, pos_b, first)
        rows = []
        for i in range(0, len(conds), args.scan_batch):
            part = conds[i:i + args.scan_batch]
            ops = [(j, *op) for j, (_, o) in enumerate(part) for op in o]
            R = len(part)
            with Edits(rt, ops, La - 1, {'A': cap_a, 'B': cap_b}):
                rows.append(plain_logits(rt, xa.expand(R, -1, -1).contiguous(), pos_a.expand(-1, R, -1).contiguous(), first))
        lds = _ld(torch.cat(rows, 0), lg_b[0], g['ids'], g['values'], ctx.digits)
        append_jsonl(path, [{'id': jid, 'case': c['id'], 'set': c['set'], 'variant': variant if isinstance(variant, str) else list(variant),
                             'conds': [n for n, _ in conds], 'lds': lds, 'seconds': round(time.time() - t0, 1)}])
        print(f'mediate {jid} {time.time() - t0:.1f}s', flush=True)


# ----------------------------------------------------------------------------- main

STAGES = {'gen': stage_gen, 'spotcheck': stage_spotcheck, 'means': stage_means, 'behav': stage_behav,
          'ablate': stage_ablate, 'probe': stage_probe, 'mediate': stage_mediate}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stages', nargs='+', required=True, choices=['cases', *STAGES])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--round3', type=Path, required=True, help='3.11 round-3 run directory (cases, images, means)')
    p.add_argument('--n-nat', type=int, default=30, help='nat items per round-3 group')
    p.add_argument('--n-str', type=int, default=15, help='str items per round-3 group')
    p.add_argument('--n-rect', type=int, default=24)
    p.add_argument('--n-comp', type=int, default=16, help='comp items per design cell')
    p.add_argument('--n-means', type=int, default=32)
    for k in ('minx', 'maxx', 'ctrx', 'miny', 'maxy', 'ctry'):
        p.add_argument(f'--group-{k}', default='')
    p.add_argument('--ablate-sets', default='')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--cond-batch', type=int, default=12)
    p.add_argument('--scan-batch', type=int, default=32)
    p.add_argument('--prep-chunk', type=int, default=64)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--max-memory', default='')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = list(args.stages)
    if 'cases' in stages:
        stage_cases(args.output, args)
        stages.remove('cases')
    if not stages:
        return
    rt = cf.load_runtime(args.model, args.max_memory)
    ctx = Ctx(rt, args.output, args)
    for st in stages:
        meta = {'stage': st, 'args': {k: str(v) for k, v in vars(args).items()}, 'runtime': qr.describe(rt),
                'code_sha256': sha256_file(Path(__file__)), 'started': datetime.now(timezone.utc).isoformat(),
                'device': torch.cuda.get_device_name(0)}
        append_jsonl(args.output / 'run_log.jsonl', [meta])
        STAGES[st](ctx)


if __name__ == '__main__':
    main()
