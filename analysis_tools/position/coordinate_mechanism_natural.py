"""Experiment 3.15 round 2: coordinate-value mechanism on natural and synthetic images (model-free parts).

Results: docs/analysis/position/results_2026-11-05_coordinate_mechanism_natural.md.

This module holds everything that does not need the model:

* stimulus families that separate the input properties suspected to switch the coordinate mechanism
  (identical repeated entities, row/column arrangement, natural content):
    grid10      10x10 grid of identical black figures, one red (64 px cells; the 3.15 reference stimulus);
    grid8       8x8 grid of identical black figures, one red (96 px cells; matched to the thumbnail grids);
    scatter     40 identical black figures at random non-overlapping positions, one red (no rows / columns);
    single      one red figure on a blank canvas (no repetition, no arrangement);
    thumb_mixed 8x8 grid of natural object thumbnails (GQA crops), the target named by its category and the
                other 63 cells drawn from other categories (arrangement without identical repetition);
    thumb_same  as thumb_mixed, but the 63 distractor cells are copies of one crop (identical repetition);
    refcoco     RefCOCO test images (natural scenes);
* the 'delL' manipulation: the entity that precedes the target in raster order is erased (pixels of the target
  unchanged), which moves a count-based coordinate but not a pixel-based one;
* crop-shift pairs (A, B) on RefCOCO images: two equally sized crops of one image at horizontal (or vertical)
  offsets that differ by d pixels, so the referent and its content are the same and only its absolute position
  differs; the token layout of A and B is identical, which the activation-patching stages require;
* answer-token bookkeeping (which generated tokens are the hundreds / tens / units digits of which value);
* recovery ratios, value-structure statistics (cell- and box-relative answer positions) and Wilson intervals.

Coordinates are 0-1000 normalised; boxes are [x_min, y_min, x_max, y_max] in input pixels (half-open).
"""
from __future__ import annotations

import math
import re

import numpy as np
from PIL import Image, ImageDraw

from analysis_tools.common import stimuli as st

RED_CHANGE = st.Change('color', 4)            # pure red in stimuli-v1
TEMPLATES = ('person', 'ring', 'square', 'arrow')
CANVAS = 640
SYNTH_TARGET = 'the figure drawn in red'
FAMILIES = ('grid10', 'grid8', 'scatter', 'single', 'thumb_mixed', 'thumb_same', 'refcoco')
GRID_FAMILIES = {'grid10': (10, 64), 'grid8': (8, 96), 'thumb_mixed': (8, 96), 'thumb_same': (8, 96)}

# Thumbnail categories, grouped so that a target never shares a visual group with any distractor.
THUMB_GROUPS = {
    'animal': ('giraffe', 'elephant', 'zebra', 'horse', 'dog', 'cat', 'cow', 'sheep', 'bear', 'bird'),
    'vehicle': ('car', 'train', 'bus', 'airplane', 'boat', 'motorcycle', 'truck'),
    'food': ('pizza', 'cake', 'banana', 'broccoli', 'donut', 'bread'),
    'household': ('clock', 'laptop', 'toilet', 'couch', 'sink', 'television', 'refrigerator', 'lamp', 'vase',
                  'umbrella', 'bench', 'bed', 'chair', 'bowl', 'cup', 'keyboard', 'kite'),
}
THUMB_MIN_SIDE = 60                            # minimum GQA box side (source pixels) for a usable crop

THINK_PROMPT = ('Locate the object described as: {target}. First describe step by step how you find this object and '
                'estimate the center of its bounding box. Then give the final answer on the last line as '
                '[{{"point_2d": [x, y]}}]. All coordinates use 0-1000 normalization; x increases rightward and '
                'y increases downward.')


# ----------------------------------------------------------------------------- helpers

def _rng(*parts) -> np.random.Generator:
    import hashlib
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], 'little'))


def tight_box(tile: np.ndarray, x0: int, y0: int) -> list[float]:
    """Half-open box of the non-white pixels of a tile pasted at (x0, y0)."""
    dark = np.argwhere((tile < 255).any(-1))
    if len(dark) == 0:
        raise ValueError('tile has no foreground pixels')
    (ty0, tx0), (ty1, tx1) = dark.min(0), dark.max(0) + 1
    return [float(x0 + tx0), float(y0 + ty0), float(x0 + tx1), float(y0 + ty1)]


def truth_from_box(box: list[float], canvas: tuple[int, int], cell_box: list[float] | None = None,
                   extra: dict | None = None) -> dict:
    W, H = canvas
    c = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    out = {'canvas': [W, H], 'box': [float(v) for v in box], 'center': c,
           'center_units': [1000 * c[0] / W, 1000 * c[1] / H],
           'box_units': [1000 * box[0] / W, 1000 * box[1] / H, 1000 * box[2] / W, 1000 * box[3] / H]}
    if cell_box is not None:
        out['cell_box_units'] = [1000 * cell_box[0] / W, 1000 * cell_box[1] / H,
                                 1000 * cell_box[2] / W, 1000 * cell_box[3] / H]
    if extra:
        out.update(extra)
    return out


def paste(canvas: np.ndarray, tile: np.ndarray, x0: int, y0: int) -> None:
    region = canvas[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]]
    np.minimum(region, tile, out=region)


# ----------------------------------------------------------------------------- synthetic families

def synth_grid(n: int, cell: int, target: tuple[int, int], template: str, erase: tuple[int, int] | None = None
               ) -> tuple[Image.Image, dict]:
    """n x n grid of identical black figures with a red target (stimuli-v1); `erase` blanks one cell."""
    spec = st.StimulusSpec(template=template, rows=n, cols=n, change=RED_CHANGE, target=target, cell=cell)
    img, meta = st.generate(spec)
    inst = next(i for i in meta['instances'] if i['is_target'])
    if erase is not None:
        e = next(i for i in meta['instances'] if (i['row'], i['col']) == tuple(erase))
        ImageDraw.Draw(img).rectangle([e['cell_box'][0], e['cell_box'][1], e['cell_box'][2] - 1, e['cell_box'][3] - 1],
                                      fill=(255, 255, 255))
    truth = truth_from_box(inst['tight_box'], tuple(meta['canvas']), inst['cell_box'],
                           {'row': target[0] + 1, 'col': target[1] + 1, 'n': n, 'cell': cell,
                            'grid_origin': meta['grid_origin']})
    return img, truth


def grid_cases(family: str, n_items: int, seed: int = 1105) -> list[dict]:
    """Target cells (not on the border rows/cols, so that delL and pairs stay inside the grid) and templates."""
    n, cell = GRID_FAMILIES[family]
    rng = _rng(seed, family)
    cells = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]
    pick = rng.permutation(len(cells))[:n_items]
    return [{'id': f'{family}_{i:03d}_r{cells[j][0]}c{cells[j][1]}', 'family': family, 'target': list(cells[j]),
             'template': TEMPLATES[i % len(TEMPLATES)], 'seed': int(rng.integers(1 << 30))}
            for i, j in enumerate(pick)]


def scatter_layout(seed: int, n: int = 40, size: int = CANVAS, tile: int = 64, min_gap: int = 4
                   ) -> list[tuple[int, int]]:
    """n non-overlapping top-left positions of tile x tile figures, placed by rejection sampling."""
    rng = _rng('scatter', seed)
    pos = []
    for _ in range(20000):
        x, y = (int(v) for v in rng.integers(0, size - tile, 2))
        if all(abs(x - a) >= tile + min_gap or abs(y - b) >= tile + min_gap for a, b in pos):
            pos.append((x, y))
            if len(pos) == n:
                return pos
    raise RuntimeError(f'could not place {n} figures')


def raster_order(points: list[tuple[float, float]], token: int = st.TOKEN_PIXELS) -> list[int]:
    """Indices sorted in the order the image tokens are read: by token row of the centre, then by x."""
    return sorted(range(len(points)), key=lambda i: (int(points[i][1] // token), points[i][0]))


def scatter_image(case: dict, erase_prev: bool = False) -> tuple[Image.Image, dict]:
    """Scatter family; the target is entity case['target_index'] of the layout. erase_prev drops the entity that
    precedes the target in raster order of the figure centres (the scatter version of delL)."""
    pos = scatter_layout(case['seed'])
    t = case['target_index']
    black = st.render_instance(case['template'], st.NONE, 64)
    red = st.render_instance(case['template'], RED_CHANGE, 64)
    centres = [(x + 32, y + 32) for x, y in pos]
    order = raster_order(centres)
    k = order.index(t)
    if k == 0:
        raise ValueError('target must not be the first entity in raster order')
    drop = order[k - 1] if erase_prev else None
    canvas = np.full((CANVAS, CANVAS, 3), 255, np.uint8)
    for i, (x, y) in enumerate(pos):
        if i == drop:
            continue
        paste(canvas, red if i == t else black, x, y)
    box = tight_box(red, *pos[t])
    return Image.fromarray(canvas), truth_from_box(box, (CANVAS, CANVAS), extra={'raster_rank': k + 1,
                                                                                 'n_entities': len(pos)})


def scatter_cases(n_items: int, seed: int = 1105) -> list[dict]:
    rng = _rng(seed, 'scatter')
    out = []
    for i in range(n_items):
        s = int(rng.integers(1 << 30))
        order = raster_order([(x + 32, y + 32) for x, y in scatter_layout(s)])
        t = order[int(rng.integers(3, len(order)))]          # at least 3 entities before the target
        out.append({'id': f'scatter_{i:03d}', 'family': 'scatter', 'seed': s, 'target_index': t,
                    'template': TEMPLATES[i % len(TEMPLATES)]})
    return out


def single_image(case: dict) -> tuple[Image.Image, dict]:
    red = st.render_instance(case['template'], RED_CHANGE, 64)
    canvas = np.full((CANVAS, CANVAS, 3), 255, np.uint8)
    x, y = case['xy']
    paste(canvas, red, x, y)
    return Image.fromarray(canvas), truth_from_box(tight_box(red, x, y), (CANVAS, CANVAS))


def single_cases(n_items: int, seed: int = 1105) -> list[dict]:
    rng = _rng(seed, 'single')
    return [{'id': f'single_{i:03d}', 'family': 'single', 'template': TEMPLATES[i % len(TEMPLATES)],
             'xy': [int(v) for v in rng.integers(16, CANVAS - 80, 2)]} for i in range(n_items)]


# ----------------------------------------------------------------------------- thumbnail grids (GQA crops)

def thumb_pool(graphs: dict, max_per_name: int = 40) -> dict[str, list[dict]]:
    """category -> list of {'image_id', 'box' (x, y, w, h)} of large single objects in GQA scene graphs."""
    names = {n for g in THUMB_GROUPS.values() for n in g}
    pool: dict[str, list[dict]] = {n: [] for n in names}
    for gid in sorted(graphs):
        for o in graphs[gid]['objects'].values():
            n = o['name']
            W, H = graphs[gid].get('width', 0), graphs[gid].get('height', 0)
            inside = o['x'] >= 0 and o['y'] >= 0 and o['x'] + o['w'] <= W and o['y'] + o['h'] <= H
            if n in pool and len(pool[n]) < max_per_name and min(o['w'], o['h']) >= THUMB_MIN_SIDE and inside:
                pool[n].append({'image_id': gid, 'box': [o['x'], o['y'], o['w'], o['h']]})
    return {n: v for n, v in pool.items() if len(v) >= 3}


def group_of(name: str) -> str:
    return next(g for g, ns in THUMB_GROUPS.items() if name in ns)


def thumb_crop(image: Image.Image, box: list[float], side: int) -> np.ndarray:
    """Square crop around a GQA box (padded with white), resized to side x side."""
    x, y, w, h = box
    s = max(w, h) * 1.05
    cx, cy = x + w / 2, y + h / 2
    l, t = int(round(cx - s / 2)), int(round(cy - s / 2))
    S = int(round(s))
    canvas = Image.new('RGB', (S, S), (255, 255, 255))
    x0, y0, x1, y1 = max(l, 0), max(t, 0), min(l + S, image.width), min(t + S, image.height)
    if x1 <= x0 or y1 <= y0:                       # GQA box outside the image: fail loudly upstream
        raise ValueError(f'GQA box {box} lies outside the {image.size} image')
    canvas.paste(image.crop((x0, y0, x1, y1)), (max(-l, 0), max(-t, 0)))
    return np.asarray(canvas.resize((side, side), Image.Resampling.LANCZOS))


def thumb_cases(pool: dict, family: str, n_items: int, seed: int = 1105) -> list[dict]:
    """Thumbnail-grid cases: target category + crop, and the distractor crops (63 cells)."""
    n, cell = GRID_FAMILIES[family]
    rng = _rng(seed, family)
    names = sorted(pool)
    out = []
    targets = grid_cases(family, n_items, seed)
    for i, tc in enumerate(targets):
        tname = names[int(rng.integers(len(names)))]
        others = [m for m in names if group_of(m) != group_of(tname)]
        tcrop = pool[tname][int(rng.integers(len(pool[tname])))]
        if family == 'thumb_same':
            dn = others[int(rng.integers(len(others)))]
            d = pool[dn][int(rng.integers(len(pool[dn])))]
            dis = [dict(d, name=dn)] * (n * n - 1)
        else:
            dis = []
            for _ in range(n * n - 1):
                dn = others[int(rng.integers(len(others)))]
                dis.append(dict(pool[dn][int(rng.integers(len(pool[dn])))], name=dn))
        out.append({'id': tc['id'], 'family': family, 'target': tc['target'], 'name': tname,
                    'target_crop': dict(tcrop, name=tname), 'distractors': dis})
    return out


def thumb_image(case: dict, load_image, erase: tuple[int, int] | None = None) -> tuple[Image.Image, dict]:
    """Render a thumbnail grid; load_image(image_id) -> PIL image. erase blanks one cell."""
    n, cell = GRID_FAMILIES[case['family']]
    side = cell - 8
    canvas = np.full((n * cell, n * cell, 3), 255, np.uint8)
    cache: dict[str, np.ndarray] = {}

    def crop(c):
        key = f"{c['image_id']}:{c['box']}"
        if key not in cache:
            cache[key] = thumb_crop(load_image(c['image_id']), c['box'], side)
        return cache[key]
    tr, tcol = case['target']
    k = 0
    for r in range(n):
        for c in range(n):
            if (r, c) == (tr, tcol):
                src = case['target_crop']
            else:
                src = case['distractors'][k]
                k += 1
            if erase is not None and (r, c) == tuple(erase):
                continue
            canvas[r * cell + 4:r * cell + 4 + side, c * cell + 4:c * cell + 4 + side] = crop(src)
    box = [tcol * cell + 4, tr * cell + 4, tcol * cell + 4 + side, tr * cell + 4 + side]
    cell_box = [tcol * cell, tr * cell, (tcol + 1) * cell, (tr + 1) * cell]
    truth = truth_from_box(box, (n * cell, n * cell), cell_box,
                           {'row': tr + 1, 'col': tcol + 1, 'n': n, 'cell': cell, 'grid_origin': [0, 0]})
    return Image.fromarray(canvas), truth


def target_text(case: dict) -> str:
    if case['family'] in ('thumb_mixed', 'thumb_same'):
        return f"the {case['name']}"
    if case['family'] == 'refcoco':
        return case['target']
    return SYNTH_TARGET


# ----------------------------------------------------------------------------- RefCOCO crop-shift pairs

def crop_pair(canvas: tuple[int, int], box: list[float], axis: int, d: int, shrink: int = 64
              ) -> tuple[list[int], list[int]] | None:
    """Two crop windows (left, top, right, bottom) of equal size whose offsets differ by d pixels along `axis`.

    The window is the full canvas shrunk by `shrink` pixels along the axis (a multiple of 32, so the crop keeps
    the 32-pixel token grid size). Window A starts at offset o, window B at o + d; o is chosen so that the box is
    centred in the overlap as far as possible. axis = 2 shifts both axes by d (windows shrunk on both axes).
    Returns None if the box does not fit inside both windows."""
    W, H = canvas
    if axis == 2:
        wx, wy = crop_pair(canvas, box, 0, d, shrink), crop_pair(canvas, box, 1, d, shrink)
        if wx is None or wy is None:
            return None
        return ([wx[0][0], wy[0][1], wx[0][2], wy[0][3]], [wx[1][0], wy[1][1], wx[1][2], wy[1][3]])
    L = (W if axis == 0 else H) - shrink
    if L % st.TOKEN_PIXELS or d <= 0 or d > shrink:
        raise ValueError('window size must be a multiple of 32 and 0 < d <= shrink')
    lo, hi = box[axis], box[axis + 2]
    full = W if axis == 0 else H
    # both windows [o, o + L) and [o + d, o + d + L) must contain [lo, hi)
    o_min, o_max = max(0, math.ceil(hi - L)), min(full - L - d, math.floor(lo - d))
    if o_min > o_max:
        return None
    o = int(round((o_min + o_max) / 2))
    if axis == 0:
        return [o, 0, o + L, H], [o + d, 0, o + d + L, H]
    return [0, o, W, o + L], [0, o + d, W, o + d + L]


def crop_truth(box: list[float], window: list[int]) -> dict:
    l, t, r, b = window
    nb = [box[0] - l, box[1] - t, box[2] - l, box[3] - t]
    return truth_from_box(nb, (r - l, b - t))


# ----------------------------------------------------------------------------- answer parsing

NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def parse_point(text: str) -> list[float] | None:
    m = re.search(r'point_2d"?\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)', text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    m = re.findall(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', text)
    return [float(m[-1][0]), float(m[-1][1])] if m else None


def parse_bbox(text: str) -> list[float] | None:
    m = re.search(r'bbox_2d"?\s*:\s*\[\s*' + r'\s*,\s*'.join([r'(-?\d+(?:\.\d+)?)'] * 4), text)
    return [float(m.group(i)) for i in range(1, 5)] if m else None


def digit_runs(tokens: list[str]) -> list[dict]:
    """Maximal runs of single-digit tokens in a token list: [{'start', 'len', 'value'}].

    Qwen tokenises numbers into one token per digit, so a run of k digit tokens is a k-digit number; 'start' is
    the index of its first digit token in `tokens`."""
    out, i = [], 0
    while i < len(tokens):
        if re.fullmatch(r'\d', tokens[i]):
            j = i
            while j < len(tokens) and re.fullmatch(r'\d', tokens[j]):
                j += 1
            out.append({'start': i, 'len': j - i, 'value': int(''.join(tokens[i:j]))})
            i = j
        else:
            i += 1
    return out


def answer_values(tokens: list[str], fmt: str) -> list[dict]:
    """Numbers of the JSON answer with their role: point -> x, y; bbox -> x_min, y_min, x_max, y_max.

    Only the first len(roles) digit runs after the format key are used (labels after them are ignored)."""
    text = ''.join(tokens)
    key = 'point_2d' if fmt == 'point' else 'bbox_2d'
    k = text.find(key)
    if k < 0:
        return []
    # token index where the key ends
    acc, start_tok = 0, None
    for i, t in enumerate(tokens):
        acc += len(t)
        if acc >= k + len(key):
            start_tok = i + 1
            break
    roles = ('x', 'y') if fmt == 'point' else ('x_min', 'y_min', 'x_max', 'y_max')
    runs = [dict(r, start=r['start'] + start_tok) for r in digit_runs(tokens[start_tok:])][:len(roles)]
    return [dict(r, role=role) for r, role in zip(runs, roles)]


# ----------------------------------------------------------------------------- statistics

def recovery(lA: np.ndarray, lB: np.ndarray, lX: np.ndarray) -> float:
    """Pooled recovery ratio r = sum <lX - lA, lB - lA> / sum ||lB - lA||^2 over pairs.

    l*: (pairs, 10) digit log-probabilities, each row centred over the ten digits before use."""
    c = lambda v: v - v.mean(-1, keepdims=True)
    a, b, x = c(np.asarray(lA, float)), c(np.asarray(lB, float)), c(np.asarray(lX, float))
    den = ((b - a) ** 2).sum()
    if den <= 0:
        return float('nan')
    return float(((x - a) * (b - a)).sum() / den)


def recovery_ci(lA, lB, lX, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    lA, lB, lX = (np.asarray(v, float) for v in (lA, lB, lX))
    n = len(lA)
    vals = [recovery(lA[i], lB[i], lX[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def relative_position(value: float, lo: float, hi: float) -> float:
    """Position of a value inside an interval: 0 = lower edge, 0.5 = centre, 1 = upper edge."""
    return (value - lo) / (hi - lo)


def position_class(f: float, tol: float = 0.2) -> str:
    """Classify a cell-relative position f (cell units) into 'centre', 'upper_edge' (f ~ 1), 'lower_edge' (f ~ 0)."""
    if abs(f - 0.5) <= tol:
        return 'centre'
    if abs(f - 1.0) <= tol:
        return 'upper_edge'
    if abs(f) <= tol:
        return 'lower_edge'
    return 'other'


def kl_div(p_log: np.ndarray, q_log: np.ndarray) -> float:
    """KL(p || q) for log-probability vectors over the same support."""
    p = np.exp(p_log)
    return float((p * (p_log - q_log)).sum())
