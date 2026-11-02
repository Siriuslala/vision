"""Experiment 3.9 round 3: reduced-input coordinate readout tool, zero-training attention decoders,
object extent from content features, and RefCOCO failure-mode annotations.

Design: docs/analysis/position/design_readout_tool.md, section 11.

All coordinates are in 0-1000 units of the image canvas.  A capture item stores, per visual token in
raster order of the merged-token grid (gh, gw), the softmax attention of the 128 full-attention heads
from the slot query (the token before the first digit of `approx_point`), and PCA-64 content features
of the visual-token states at decoder input (layer 0) and after layer 16 (see readout_tool_r2).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position import readout_tool_r2 as r2

ATTN_LAYERS = rt_.ATTN_LAYERS                       # (4, 8, ..., 32), 16 heads each, 1-based layers
N_HEADS = 16


def head_index(layer: int, head: int) -> int:
    """Row of head `head` of full-attention layer `layer` in the (128, Nv) capture maps."""
    return ATTN_LAYERS.index(layer) * N_HEADS + head


def head_name(index: int) -> str:
    return f'L{ATTN_LAYERS[index // N_HEADS]}H{index % N_HEADS}'


def parse_heads(spec: str) -> tuple[int, ...]:
    """'L20H11,L16H0' -> map rows; 'L20' -> the 16 heads of layer 20; 'all' -> 128 rows; '' -> none."""
    out: list[int] = []
    for tok in [t for t in spec.split(',') if t]:
        if tok == 'all':
            out += range(len(ATTN_LAYERS) * N_HEADS)
        elif 'H' in tok:
            l, h = tok[1:].split('H')
            out.append(head_index(int(l), int(h)))
        else:
            out += [head_index(int(tok[1:]), h) for h in range(N_HEADS)]
    if len(set(out)) != len(out):
        raise ValueError(f'duplicate heads in {spec!r}')
    return tuple(out)


# ----------------------------------------------------------------------------- zero-training decoders

def cell_size(grid: tuple[int, int]) -> tuple[float, float]:
    return 1000.0 / grid[1], 1000.0 / grid[0]


def peak_centroid(m: np.ndarray, grid: tuple[int, int], radius: int = 1) -> np.ndarray:
    """Experiment 3.11 decoder: attention-weighted centre of the (2r+1)^2 neighbourhood of the argmax token."""
    gh, gw = grid
    a = m.reshape(gh, gw).astype(np.float64)
    r, c = np.unravel_index(int(a.argmax()), a.shape)
    r0, r1, c0, c1 = max(r - radius, 0), min(r + radius + 1, gh), max(c - radius, 0), min(c + radius + 1, gw)
    w = a[r0:r1, c0:c1]
    rr, cc = np.mgrid[r0:r1, c0:c1]
    cw, ch = cell_size(grid)
    s = w.sum()
    return np.array([((cc + 0.5) * cw * w).sum() / s, ((rr + 0.5) * ch * w).sum() / s])


EDGE_TL = ('L20H0', 'L24H1', 'L24H2', 'L24H7', 'L24H12')            # experiment 3.11 round 2: top-left edge readers
EDGE_BR = ('L20H10', 'L24H5', 'L24H6', 'L24H8', 'L24H9', 'L28H0')   # bottom-right edge readers


def edge_group_box(maps: np.ndarray, grid: tuple[int, int], tl: tuple[int, ...], br: tuple[int, ...]) -> np.ndarray:
    """Experiment-3.11-round-2 edge-pair decoder at one query position.

    maps (128, Nv) attention shares; the maps of each edge group are averaged and the peak 3x3 centre of the
    top-left (bottom-right) average is taken as the box's top-left (bottom-right) corner.  Returns
    (x0, y0, x1, y1) in 0-1000 units, sorted so that x0 <= x1 and y0 <= y1; the box centre is the edge-pair
    centre estimate of 3.11.
    """
    a = peak_centroid(maps[list(tl)].astype(np.float64).mean(0), grid)
    b = peak_centroid(maps[list(br)].astype(np.float64).mean(0), grid)
    return np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])])


def map_centroid(m: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Attention-weighted mean of all token centres (the full first moment of the map)."""
    c = r2.token_centers(grid)
    w = m.astype(np.float64)
    return (w[:, None] * c).sum(0) / w.sum()


def _component(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """4-connected component of a boolean grid that contains `seed` (seed is always included)."""
    gh, gw = mask.shape
    out = np.zeros_like(mask)
    out[seed] = True
    q = deque([seed])
    while q:
        r, c = q.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < gh and 0 <= cc < gw and mask[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                q.append((rr, cc))
    return out


def region_box(region: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Bounding box (x0, y0, x1, y1) of the token cells in a boolean (gh, gw) region."""
    rows, cols = np.nonzero(region)
    cw, ch = cell_size(grid)
    return np.array([cols.min() * cw, rows.min() * ch, (cols.max() + 1) * cw, (rows.max() + 1) * ch], float)


def attention_extent(m: np.ndarray, grid: tuple[int, int], tau: float) -> np.ndarray:
    """Region of tokens whose attention is at least tau x the peak, connected to the peak."""
    a = m.reshape(grid).astype(np.float64)
    seed = np.unravel_index(int(a.argmax()), a.shape)
    return _component(a >= tau * a.max(), seed)


def content_similarity(m: np.ndarray, content: np.ndarray, top: int = 3) -> np.ndarray:
    """Cosine similarity (Nv,) of every token's content vector to the object prototype.

    The prototype is the mean content vector of the `top` most attended tokens: attention marks the
    salient part of the selected object, and tokens of the same object share its content features,
    so the similarity map extends the salient peak to the object's extent.  `content` is (Nv, D),
    already standardized per channel.
    """
    idx = np.argsort(-m)[:top]
    proto = content[idx].mean(0)
    x = content / (np.linalg.norm(content, axis=1, keepdims=True) + 1e-6)
    return x @ (proto / (np.linalg.norm(proto) + 1e-6))


def content_extent(m: np.ndarray, content: np.ndarray, grid: tuple[int, int], tau: float, top: int = 3) -> np.ndarray:
    """Region grown from the attention peak over tokens whose content similarity to the prototype >= tau."""
    sim = content_similarity(m, content, top).reshape(grid)
    seed = np.unravel_index(int(m.reshape(grid).argmax()), grid)
    return _component(sim >= tau, seed)


def box_center(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def box_iou_units(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


# ----------------------------------------------------------------------------- reader feature groups

def head_features(maps: np.ndarray, rows: tuple[int, ...], grid: tuple[int, int]) -> np.ndarray:
    """(128, Nv) attention -> (Nv, len(rows)) log-ratio to uniform for the selected heads."""
    return r2.attention_features(maps[list(rows)], grid)


def extent_features(maps: np.ndarray, rows: tuple[int, ...], content: np.ndarray, content_mean: np.ndarray,
                    content_std: np.ndarray, top: int = 3) -> np.ndarray:
    """(Nv, 1) content similarity to the prototype of the tokens most attended by the selected heads.

    The selected heads' maps are normalized to unit mass and averaged before picking the top tokens.
    """
    m = maps[list(rows)].astype(np.float32)
    m = (m / (m.sum(1, keepdims=True) + 1e-12)).mean(0)
    x = (content.astype(np.float32) - content_mean) / content_std
    return content_similarity(m, x, top)[:, None].astype(np.float32)


# ----------------------------------------------------------------------------- RefCOCO annotations

def refcoco_annotations(refcoco_dir: Path, question_ids: set[str]) -> dict[str, dict]:
    """question_id -> {'file_name', 'bbox_xywh', 'segmentation' (flat polygon, source pixels),
    'others': [bbox_xywh of the other RefCOCO-annotated objects in the same image]}.

    Other objects come from all RefCOCO rows (every split) of the same COCO image (file names are
    `COCO_train2014_<image id>_<k>.jpg`; the suffix k indexes the reference), deduplicated by rounded box.  RefCOCO selects images with at least two objects of the target's category, so these
    boxes are mostly same-category distractors.
    """
    import pyarrow.parquet as pq
    shards = sorted((refcoco_dir / 'data').glob('*.parquet'))
    if not shards:
        raise FileNotFoundError(f'No RefCOCO parquet shards under {refcoco_dir}/data')
    by_file: dict[str, dict[tuple, list[float]]] = {}
    rows: dict[str, dict] = {}
    for shard in shards:
        t = pq.read_table(shard, columns=['question_id', 'bbox', 'segmentation', 'file_name']).to_pydict()
        for qid, bbox, seg, name in zip(t['question_id'], t['bbox'], t['segmentation'], t['file_name']):
            key = tuple(round(v, 1) for v in bbox)
            name = coco_image_of(name)
            by_file.setdefault(name, {})[key] = [float(v) for v in bbox]
            if str(qid) in question_ids:
                rows[str(qid)] = {'file_name': name, 'bbox_xywh': [float(v) for v in bbox],
                                  'segmentation': [float(v) for v in (seg or [])]}
    missing = question_ids - set(rows)
    if missing:
        raise ValueError(f'{len(missing)} RefCOCO question ids not found, e.g. {sorted(missing)[:3]}')
    for qid, r in rows.items():
        own = tuple(round(v, 1) for v in r['bbox_xywh'])
        r['others'] = [b for k, b in by_file[r['file_name']].items() if k != own]
    return rows


def coco_image_of(file_name: str) -> str:
    """'COCO_train2014_000000296747_0.jpg' -> 'COCO_train2014_000000296747'."""
    return file_name.rsplit('_', 1)[0]


def polygon_mask(seg: list[float], source_size: tuple[int, int], canvas: tuple[int, int]) -> np.ndarray:
    """Rasterize a flat source-pixel polygon onto the input canvas -> bool (H, W)."""
    from PIL import Image, ImageDraw
    sw, sh = source_size
    W, H = canvas
    pts = [(seg[i] * W / sw, seg[i + 1] * H / sh) for i in range(0, len(seg) - 1, 2)]
    img = Image.new('1', (W, H), 0)
    if len(pts) >= 3:
        ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.asarray(img, bool)


def xywh_to_units(b: list[float], source_size: tuple[int, int]) -> np.ndarray:
    sw, sh = source_size
    return np.array([b[0] / sw, b[1] / sh, (b[0] + b[2]) / sw, (b[1] + b[3]) / sh]) * 1000.0


def inside_box(p: np.ndarray, b: np.ndarray) -> bool:
    return bool(b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3])


def selection_outcome(p: np.ndarray, target: np.ndarray, others: list[np.ndarray]) -> str:
    """'target' if the point is in the target box; 'distractor' if it lies only in another annotated
    object's box; 'background' otherwise."""
    if inside_box(p, target):
        return 'target'
    if any(inside_box(p, o) for o in others):
        return 'distractor'
    return 'background'


def occlusion_fraction(target: np.ndarray, others: list[np.ndarray]) -> float:
    """Largest fraction of the target box covered by any other annotated object's box."""
    area = (target[2] - target[0]) * (target[3] - target[1])
    best = 0.0
    for o in others:
        ix = max(0.0, min(target[2], o[2]) - max(target[0], o[0]))
        iy = max(0.0, min(target[3], o[3]) - max(target[1], o[1]))
        best = max(best, ix * iy / area if area > 0 else 0.0)
    return float(best)
