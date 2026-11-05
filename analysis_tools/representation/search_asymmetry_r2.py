"""Experiment 2.7 round 2: generality and mechanism of visual-search asymmetry (stimuli and model-free metrics).

Design and results: docs/analysis/representation/results_2026-11-03_search_asymmetry_r2.md.

Naming used throughout round 2 (replaces the round-1 labels std / swap in documents; the stored variant names
'std' / 'swap' are kept so that round-1 artifacts stay readable):

  feature-present target  ('std')   one entity A among entities B; A carries the extra part or property
  feature-absent target   ('swap')  one entity B among entities A; the odd entity lacks it
  homA / homB                       every entity is A / B (no odd entity)

Round 2 adds
  * five new feature pairs: three in which B is a pixel subset of A (line with / without an end bar, plus with /
    without its fourth arm, square with / without an inner dot) and two in which each entity has a part the
    other lacks (circle vs rounded square, upright vs rotated cross);
  * a target-index set: feature-absent targets of three pairs placed evenly on all 36 cells of the 6 x 6 grid;
  * a layout set: the same pairs on 3 x 12 and 12 x 3 grids, which decouple "number of raster-earlier
    entities" from "row index" (a row-1 target on a 3 x 12 grid can have up to 11 earlier entities);
  * salient-level quadruples (with homA / homB) for Q vs O and open vs closed circle, used to ask which neurons
    signal the feature-absent target once it becomes findable.

The model-free metrics at the end are the causal-template surprise and the oddness direction used in the
analysis; GPU stages are in `search_asymmetry_r2_run.py`, CPU analysis in `search_asymmetry_r2_analysis.py`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from analysis_tools.behavior.decomposition import unit_seed, write_jsonl
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa

GENERATOR = sa.GENERATOR + '+r2'

# New pairs.  A carries the extra part (subset pairs) or is the first member (mutually exclusive pairs).
#   term:   line of length 28 (vertical for orient 0/1, horizontal for 2/3); A adds a perpendicular end bar of
#           width p at one end
#   plus:   cross with arms of length 12; B has three arms, A adds the fourth arm with length p
#   dot:    square outline of side 28; A adds a filled inner dot of diameter p
#   circsq: A is a square outline (side 26) with corner radius 13 - p (p = 13 gives sharp corners), B is a
#           circle outline of diameter 26; corners versus constant curvature
#   plusx:  A is a cross (arm length 12) rotated by p degrees, B is an upright cross
NEW_PAIRS = {
    'term': {'A': 'line with end bar', 'B': 'plain line', 'levels': {'subtle': 6, 'salient': 14}, 'unit': 'px',
             'relation': 'subset'},
    'plus': {'A': 'four-arm cross', 'B': 'three-arm cross', 'levels': {'subtle': 5, 'salient': 12}, 'unit': 'px',
             'relation': 'subset'},
    'dot': {'A': 'square with inner dot', 'B': 'empty square', 'levels': {'subtle': 4, 'salient': 8}, 'unit': 'px',
            'relation': 'subset'},
    'circsq': {'A': 'rounded square', 'B': 'circle', 'levels': {'subtle': 6, 'salient': 13}, 'unit': 'px',
               'relation': 'exclusive'},
    'plusx': {'A': 'rotated cross', 'B': 'upright cross', 'levels': {'subtle': 15, 'salient': 45}, 'unit': 'deg',
              'relation': 'exclusive'},
}
# Feature relation of the round-1 pairs, by the same pixel criterion (B is a subset of A, A of B, or neither).
OLD_RELATION = {'qo': 'subset', 'closure': 'superset', 'tilt': 'exclusive', 'curve': 'exclusive',
                'length': 'subset', 'sat': 'exclusive', 'size': 'exclusive'}
ALL_PAIRS = {**sa.PAIRS, **NEW_PAIRS}


def render_entity(pair: str, present: bool, param: float, orient: int, offset: tuple[int, int]) -> np.ndarray:
    """One (64, 64, 3) tile for any round-1 or round-2 pair; present=True draws A."""
    if pair in sa.PAIRS:
        return sa.render_entity(pair, present, param, orient, offset)
    image, draw = sa._canvas()
    cx, cy = sa.CELL / 2 + offset[0], sa.CELL / 2 + offset[1]
    if pair == 'term':
        half = 14.0
        vertical = orient < 2
        sign = 1 if orient % 2 == 0 else -1
        if vertical:
            sa._line(draw, (cx, cy - half), (cx, cy + half))
            if present:
                y = cy + sign * half
                sa._line(draw, (cx - param / 2, y), (cx + param / 2, y))
        else:
            sa._line(draw, (cx - half, cy), (cx + half, cy))
            if present:
                x = cx + sign * half
                sa._line(draw, (x, cy - param / 2), (x, cy + param / 2))
    elif pair in ('plus', 'plusx'):
        arm = 12.0
        base = math.radians(param) if (pair == 'plusx' and present) else 0.0
        missing = orient % 4 if pair == 'plus' else -1
        for k in range(4):
            length = arm
            if k == missing:
                if not present:
                    continue
                length = param
            ang = base + k * math.pi / 2
            sa._line(draw, (cx, cy), (cx + length * math.cos(ang), cy + length * math.sin(ang)))
    elif pair == 'dot':
        h = 14.0
        draw.rectangle([(cx - h) * sa.SS, (cy - h) * sa.SS, (cx + h) * sa.SS, (cy + h) * sa.SS],
                       outline=sa.BLACK, width=sa.STROKE * sa.SS)
        if present:
            r = param / 2
            draw.ellipse([(cx - r) * sa.SS, (cy - r) * sa.SS, (cx + r) * sa.SS, (cy + r) * sa.SS], fill=sa.BLACK)
    elif pair == 'circsq':
        h = 13.0
        radius = (h - param) if present else h
        draw.rounded_rectangle([(cx - h) * sa.SS, (cy - h) * sa.SS, (cx + h) * sa.SS, (cy + h) * sa.SS],
                               radius=max(0.0, radius) * sa.SS, outline=sa.BLACK, width=sa.STROKE * sa.SS)
    else:
        raise ValueError(f'Unknown pair {pair!r}')
    return sa._finish(image)


def grid_images(pair: str, param: float, rows: int, cols: int, target: tuple[int, int], orient: int,
                jit: dict, variants=sa.VARIANTS) -> dict:
    """Render the requested variants of one item on a rows x cols grid (shared target, orientation and jitter)."""
    rule = {'std': lambda rc: rc == target, 'swap': lambda rc: rc != target,
            'homA': lambda rc: True, 'homB': lambda rc: False}
    out = {}
    for v in variants:
        tiles = {rc: render_entity(pair, rule[v](rc), param, orient, jit[rc]) for rc in jit}
        image, meta = oe.compose(rows, cols, sa.CELL, tiles)
        for inst in meta['instances']:
            inst['has_feature'] = bool(rule[v]((inst['row'], inst['col'])))
        out[v] = (image, meta)
    return out


def build(output: Path, cfg: dict) -> None:
    """Write every round-2 image and items.jsonl.

    cfg keys: seed; new_pairs; main_items {level: n} (6 x 6, both target variants); hom_items (6 x 6 subtle items of
    the new pairs, and salient items of salient_hom_pairs, that also get homA / homB); salient_hom_pairs;
    index_pairs, index_items (feature-absent targets per pair on 6 x 6); layout_pairs, layouts ['3x12', ...],
    layout_items (feature-absent), layout_present_items (feature-present controls).
    """
    output.mkdir(parents=True, exist_ok=True)
    items = []

    def add(image, meta, rel, **fields):
        digest = oe._save(image, output, rel)
        items.append({'id': rel.removesuffix('.png').replace('/', ':'), 'path': rel, 'sha256': digest,
                      'meta': meta, **fields})

    def emit(set_name, pair, level, rows, cols, n, variants_for, tag):
        param = ALL_PAIRS[pair]['levels'][level]
        rng = np.random.default_rng(unit_seed(cfg['seed'], 'search_asymmetry_r2', set_name, pair, level, rows, cols))
        for p, target in enumerate(oe.balanced_targets(rows, cols, n, rng)):
            orient = int(rng.integers(0, 4))
            jit = sa.jitter_map(rows, cols, rng)
            for v, (image, meta) in grid_images(pair, param, rows, cols, target, orient, jit, variants_for(p)).items():
                add(image, meta, f'images/{set_name}/{pair}/{tag}_p{p:03d}_{v}.png', set=set_name, group='pair',
                    pair_name=pair, level=level, param=param, grid=rows if rows == cols else f'{rows}x{cols}',
                    rows=rows, cols=cols, pair=p, variant=v, target=list(target), orient=orient,
                    n_entities=rows * cols)

    for pair in cfg['new_pairs']:
        for level, n in cfg['main_items'].items():
            hom = level == 'subtle'
            emit('pairs', pair, level, 6, 6, n,
                 lambda p, hom=hom: sa.VARIANTS if (hom and p < cfg['hom_items']) else sa.ODD_VARIANTS,
                 f'g06_{level}')
    for pair in cfg['salient_hom_pairs']:
        emit('salient_hom', pair, 'salient', 6, 6, cfg['hom_items'], lambda p: sa.VARIANTS, 'g06_salient')
    for pair in cfg['index_pairs']:
        emit('index', pair, 'subtle', 6, 6, cfg['index_items'], lambda p: ('swap',), 'g06_subtle')
    for pair in cfg['layout_pairs']:
        for lay in cfg['layouts']:
            rows, cols = (int(x) for x in lay.split('x'))
            emit('layout', pair, 'subtle', rows, cols, cfg['layout_items'], lambda p: ('swap',), f'g{lay}_absent')
            emit('layout', pair, 'subtle', rows, cols, cfg['layout_present_items'], lambda p: ('std',),
                 f'g{lay}_present')
    write_jsonl(output / 'items.jsonl', items)
    (output / 'build_config.json').write_text(json.dumps({**cfg, 'generator': GENERATOR}, indent=1))


def score(item: dict, record: dict) -> dict:
    """Round-1 strict score (row and column both correct; truncation counts as failure)."""
    return sa.score(item, record)


# ----------------------------------------------------------------------------- model-free metrics

def causal_surprise(x: np.ndarray, min_prev: int = 1) -> np.ndarray:
    """Distance of each entity from the mean of the raster-earlier entities.

    x: (n_entities, d) entity features in raster order.  Returns (n_entities,) with
    s_e = || x_e - mean_{e' < e} x_{e'} ||, and NaN for e < min_prev (no earlier entity to compare with).
    This is the oddness a strictly causal observer can compute when it compares every entity with a running
    average (template) of what it has seen so far.
    """
    x = x.astype(np.float64)
    csum = np.cumsum(x, 0)
    out = np.full(len(x), np.nan)
    for e in range(min_prev, len(x)):
        out[e] = float(np.linalg.norm(x[e] - csum[e - 1] / e))
    return out


def global_surprise(x: np.ndarray) -> np.ndarray:
    """Distance of each entity from the mean of all other entities (a bidirectional observer)."""
    x = x.astype(np.float64)
    n = len(x)
    others = (x.sum(0)[None] - x) / (n - 1)
    return np.linalg.norm(x - others, axis=1)


def oddness_direction(contrast_present: np.ndarray, contrast_absent: np.ndarray) -> np.ndarray:
    """Unit oddness direction of one feature pair: the part of the target contrast that keeps its sign when the
    roles of A and B are exchanged.  Inputs: (n_images, d) target-minus-distractor-mean contrasts."""
    u = (contrast_present.mean(0) + contrast_absent.mean(0)) / 2
    return u / (np.linalg.norm(u) + 1e-12)


def entity_zscores(scores: np.ndarray) -> np.ndarray:
    """Standardise per-entity scores within one image: (s - mean) / std over all entities.  scores: (n_ent,)."""
    return (scores - scores.mean()) / (scores.std() + 1e-12)
