"""Experiment 2.7: mechanism of visual-search asymmetry in Qwen3.5-9B (stimuli, cases, scoring, metrics).

Design: docs/analysis/representation/design_search_asymmetry.md.

Every feature pair (A, B) follows the psychophysics convention of Treisman & Souther (1985) and Treisman &
Gormican (1988): A is the item that carries an extra feature or "more" of a dimension (Q = O plus a line
segment, a tilted line, a longer line, a curved line, an open circle, a saturated colour, a larger circle),
B is the standard.  For every target cell, jitter pattern and pair orientation four images are rendered:

  std   -- one A among B   (the odd entity HAS the feature; "presence" search)
  swap  -- one B among A   (the odd entity LACKS the feature; "absence" search)
  homA  -- every entity is A (no odd entity)
  homB  -- every entity is B (no odd entity)

std and swap differ from the homogeneous images by the same pixels at the target cell, with opposite sign.
Any accuracy difference between them therefore comes from how the model treats the direction of the
difference.  A conjunction group (red circle among green circles and red squares) and its two feature-search
controls measure the set-size dependence that separates parallel from serial-like search.

This module is model-free.  GPU stages live in `search_asymmetry_run.py`, CPU analysis and figures in
`search_asymmetry_analysis.py`.  Grid composition, entity-token geometry, answer parsing and the Wilson
interval are reused from experiment 2.6 (`odd_entity.py`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from analysis_tools.behavior.decomposition import read_jsonl, unit_seed, write_jsonl
from analysis_tools.common.stimuli import GENERATOR_VERSION, lab_to_srgb
from analysis_tools.representation import odd_entity as oe

GENERATOR = GENERATOR_VERSION + '+search_asymmetry'
CELL = 64
SS = 4                      # supersampling factor for anti-aliased primitives
STROKE = 3                  # stroke width in pixels at the 64-pixel cell scale
JITTER = 3                  # maximum per-entity offset (pixels) from the cell centre
BLACK = (0, 0, 0)
RED = (210, 40, 40)
GREEN = (40, 160, 40)
SAT_L, SAT_HUE, SAT_CHROMA = 55.0, math.radians(30), 60.0

# Feature pairs: description of A and B, and the parameter of A at the two difficulty levels.
#   qo:     O radius 14; Q adds a radial line segment of length p crossing the circle (Treisman & Souther 1985)
#   tilt:   line of length 32; A is tilted by p degrees, B is vertical
#   length: vertical line; B has length 28, A has length 28 + p
#   curve:  arc with chord 32 and sagitta p; B is straight
#   closure: A is an open circle with an arc gap of p pixels, B is closed (Treisman & Gormican 1988)
#   sat:    filled disc diameter 30 at L* 55, hue 30 deg; A chroma 60, B chroma 60 - p
#   size:   outline circle; B diameter 28, A diameter 28 + p
PAIRS = {
    'qo': {'A': 'Q (O plus a line segment)', 'B': 'O', 'levels': {'subtle': 6, 'salient': 14}, 'unit': 'px'},
    'tilt': {'A': 'tilted line', 'B': 'vertical line', 'levels': {'subtle': 8, 'salient': 20}, 'unit': 'deg'},
    'length': {'A': 'long line', 'B': 'short line', 'levels': {'subtle': 4, 'salient': 12}, 'unit': 'px'},
    'curve': {'A': 'curved line', 'B': 'straight line', 'levels': {'subtle': 3, 'salient': 8}, 'unit': 'px'},
    'closure': {'A': 'open circle', 'B': 'closed circle', 'levels': {'subtle': 4, 'salient': 12}, 'unit': 'px'},
    'sat': {'A': 'saturated disc', 'B': 'desaturated disc', 'levels': {'subtle': 15, 'salient': 40}, 'unit': 'dE'},
    'size': {'A': 'large circle', 'B': 'small circle', 'levels': {'subtle': 4, 'salient': 12}, 'unit': 'px'},
}
VARIANTS = ('std', 'swap', 'homA', 'homB')
ODD_VARIANTS = ('std', 'swap')
CONJ_CONDITIONS = ('conj', 'feat_color', 'feat_shape')
DIRECT_PROMPT = oe.DIRECT_PROMPT          # identical to experiment 2.6 / 1.3 T5 (direct answer)


# ----------------------------------------------------------------------------- primitives

def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new('RGB', (CELL * SS, CELL * SS), (255, 255, 255))
    return image, ImageDraw.Draw(image)


def _finish(image: Image.Image) -> np.ndarray:
    return np.asarray(image.resize((CELL, CELL), Image.LANCZOS), dtype=np.uint8).copy()


def _line(draw, p0, p1, color=BLACK, width=STROKE):
    draw.line([(p0[0] * SS, p0[1] * SS), (p1[0] * SS, p1[1] * SS)], fill=color, width=width * SS)


def _arc_points(cx, cy, r, a0, a1, n=96):
    """Points of a circular arc from angle a0 to a1 (radians, image coordinates with y down)."""
    t = np.linspace(a0, a1, n)
    return [(cx + r * math.cos(v), cy + r * math.sin(v)) for v in t]


def _polyline(draw, pts, color=BLACK, width=STROKE):
    draw.line([(x * SS, y * SS) for x, y in pts], fill=color, width=width * SS, joint='curve')


def render_entity(pair: str, present: bool, param: float, orient: int, offset: tuple[int, int]) -> np.ndarray:
    """One (64, 64, 3) tile.  present=True draws A, False draws B.  orient (0-3) rotates the
    pair-specific direction (tail angle, gap direction, tilt sign, arc bulge) and is shared by all entities
    of an image, so A and B differ only in the manipulated feature."""
    image, draw = _canvas()
    cx, cy = CELL / 2 + offset[0], CELL / 2 + offset[1]
    ang = math.pi / 4 + orient * math.pi / 2          # diagonal directions for Q tails and gaps
    if pair in ('qo', 'closure', 'size'):
        r = 14.0
        if pair == 'size' and present:
            r = 14.0 + param / 2
        if pair == 'closure' and present:
            half = (param / 2) / r                     # half gap angle for an arc gap of `param` pixels
            _polyline(draw, _arc_points(cx, cy, r, ang + half, ang + 2 * math.pi - half))
        else:
            _polyline(draw, _arc_points(cx, cy, r, 0, 2 * math.pi, 128))
        if pair == 'qo' and present:
            # radial segment centred on the circle at angle `ang`, length `param`
            r0, r1 = r - param / 2, r + param / 2
            _line(draw, (cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)),
                  (cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)))
    elif pair in ('tilt', 'length'):
        length = 32.0 if pair == 'tilt' else 28.0 + (param if present else 0.0)
        theta = math.radians(param) * (1 if orient % 2 == 0 else -1) if (pair == 'tilt' and present) else 0.0
        dx, dy = (length / 2) * math.sin(theta), (length / 2) * math.cos(theta)
        _line(draw, (cx - dx, cy - dy), (cx + dx, cy + dy))
    elif pair == 'curve':
        half = 16.0
        if present:
            s = param * (1 if orient % 2 == 0 else -1)
            radius = (half * half + s * s) / (2 * abs(s))
            centre_x = cx - math.copysign(radius - abs(s), s)      # bulge towards +x for s > 0
            phi = math.asin(half / radius)
            base = 0.0 if s > 0 else math.pi
            _polyline(draw, _arc_points(centre_x, cy, radius, base - phi, base + phi))
        else:
            _line(draw, (cx, cy - half), (cx, cy + half))
    elif pair == 'sat':
        chroma = SAT_CHROMA if present else SAT_CHROMA - param
        color = lab_to_srgb((SAT_L, chroma * math.cos(SAT_HUE), chroma * math.sin(SAT_HUE)))
        r = 15.0
        draw.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=color)
    else:
        raise ValueError(f'Unknown pair {pair!r}')
    return _finish(image)


def render_conj(color: str, shape: str, offset: tuple[int, int]) -> np.ndarray:
    """Filled red/green circle or square (side and diameter 28 pixels) for the conjunction group."""
    image, draw = _canvas()
    cx, cy, h = CELL / 2 + offset[0], CELL / 2 + offset[1], 14.0
    fill = RED if color == 'red' else GREEN
    box = [(cx - h) * SS, (cy - h) * SS, (cx + h) * SS, (cy + h) * SS]
    (draw.ellipse if shape == 'circle' else draw.rectangle)(box, fill=fill)
    return _finish(image)


# ----------------------------------------------------------------------------- stimulus sets

def jitter_map(rows: int, cols: int, rng: np.random.Generator) -> dict[tuple[int, int], tuple[int, int]]:
    return {(r, c): (int(rng.integers(-JITTER, JITTER + 1)), int(rng.integers(-JITTER, JITTER + 1)))
            for r in range(rows) for c in range(cols)}


def pair_images(pair: str, level: str, g: int, target: tuple[int, int], orient: int,
                jit: dict, variants=VARIANTS) -> dict[str, tuple[Image.Image, dict]]:
    """Render the requested variants of one item (shared target, orientation and jitter)."""
    param = PAIRS[pair]['levels'][level]
    rule = {'std': lambda rc: rc == target, 'swap': lambda rc: rc != target,
            'homA': lambda rc: True, 'homB': lambda rc: False}
    out = {}
    for v in variants:
        tiles = {rc: render_entity(pair, rule[v](rc), param, orient, jit[rc]) for rc in jit}
        image, meta = oe.compose(g, g, CELL, tiles)
        for inst in meta['instances']:
            inst['has_feature'] = bool(rule[v]((inst['row'], inst['col'])))
        out[v] = (image, meta)
    return out


def conj_image(condition: str, g: int, target: tuple[int, int], jit: dict,
               rng: np.random.Generator) -> tuple[Image.Image, dict]:
    """Red circle target among green circles (feat_color), red squares (feat_shape) or both (conj, half each)."""
    cells = [rc for rc in jit if rc != target]
    if condition == 'conj':
        order = rng.permutation(len(cells))
        kinds = {cells[i]: (('green', 'circle') if k % 2 == 0 else ('red', 'square')) for k, i in enumerate(order)}
    elif condition == 'feat_color':
        kinds = {rc: ('green', 'circle') for rc in cells}
    elif condition == 'feat_shape':
        kinds = {rc: ('red', 'square') for rc in cells}
    else:
        raise ValueError(condition)
    kinds[target] = ('red', 'circle')
    image, meta = oe.compose(g, g, CELL, {rc: render_conj(*kinds[rc], jit[rc]) for rc in jit})
    for inst in meta['instances']:
        inst['color'], inst['shape'] = kinds[(inst['row'], inst['col'])]
    return image, meta


def build(output: Path, cfg: dict) -> None:
    """Write every image and items.jsonl.

    cfg keys: seed, pairs, main_grid, main_items {level: n}, setsize_grids, setsize_items, hom_items (number of
    main-grid subtle items that also get homA / homB donors), conj_grids, conj_items {grid: n}.
    """
    output.mkdir(parents=True, exist_ok=True)
    items = []

    def add(image, meta, rel, **fields):
        digest = oe._save(image, output, rel)
        items.append({'id': rel.removesuffix('.png').replace('/', ':'), 'path': rel, 'sha256': digest,
                      'meta': meta, **fields})

    jobs = [(cfg['main_grid'], level, n) for level, n in cfg['main_items'].items()]
    jobs += [(g, 'subtle', cfg['setsize_items']) for g in cfg['setsize_grids']]
    for pair in cfg['pairs']:
        for g, level, n in jobs:
            rng = np.random.default_rng(unit_seed(cfg['seed'], 'search_asymmetry', pair, level, g))
            for p, target in enumerate(oe.balanced_targets(g, g, n, rng)):
                orient = int(rng.integers(0, 4))
                jit = jitter_map(g, g, rng)
                main_subtle = g == cfg['main_grid'] and level == 'subtle'
                variants = VARIANTS if (main_subtle and p < cfg['hom_items']) else ODD_VARIANTS
                for v, (image, meta) in pair_images(pair, level, g, target, orient, jit, variants).items():
                    add(image, meta, f'images/{pair}/g{g:02d}_{level}_p{p:03d}_{v}.png', group='pair', pair_name=pair,
                        level=level, param=PAIRS[pair]['levels'][level], grid=g, pair=p, variant=v,
                        target=list(target), orient=orient, n_entities=g * g)
    for g in cfg['conj_grids']:
        for cond in CONJ_CONDITIONS:
            rng = np.random.default_rng(unit_seed(cfg['seed'], 'search_asymmetry', 'conj', cond, g))
            for p, target in enumerate(oe.balanced_targets(g, g, cfg['conj_items'][str(g)], rng)):
                jit = jitter_map(g, g, rng)
                image, meta = conj_image(cond, g, target, jit, rng)
                add(image, meta, f'images/conj/g{g:02d}_{cond}_p{p:03d}.png', group='conj', pair_name=cond,
                    level='-', param=0, grid=g, pair=p, variant='std', target=list(target), orient=0,
                    n_entities=g * g)
    write_jsonl(output / 'items.jsonl', items)
    (output / 'build_config.json').write_text(json.dumps({**cfg, 'generator': GENERATOR}, indent=1))


def load_items(output: Path) -> dict[str, dict]:
    return {it['id']: it for it in read_jsonl(output / 'items.jsonl')}


# ----------------------------------------------------------------------------- scoring

def score(item: dict, record: dict) -> dict:
    """Strict score (2.6 convention: last row/column pair, truncation = failure) plus a +-1 tolerance flag
    that separates localisation from row/column counting errors in large grids."""
    out = oe.score(item, record)
    out['near'] = bool(out['valid'] and abs(out['d_row']) <= 1 and abs(out['d_col']) <= 1)
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b (std only) and c (swap only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


# ----------------------------------------------------------------------------- representation metrics

def mahalanobis_salience(h: np.ndarray, target: int, ridge: float = 1e-1, rank: int = 8) -> float:
    """Salience of the target relative to the distractor distribution (Rosenholtz 1999/2001).

    h: (n_entities, d) entity states of one image.  The distractor covariance is estimated in the span of
    its top `rank` principal components plus an isotropic ridge (fraction `ridge` of the mean eigenvalue),
    because 35 distractors cannot support a full d x d covariance.  Returns sqrt((x - mu)^T S^-1 (x - mu)).
    """
    others = np.delete(h, target, axis=0).astype(np.float64)
    mu = others.mean(0)
    x = h[target].astype(np.float64) - mu
    c = others - mu
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    lam = s ** 2 / max(1, len(others) - 1)
    k = min(rank, len(lam))
    floor = ridge * float(lam.mean()) + 1e-12
    proj = vt[:k] @ x
    resid = x @ x - proj @ proj
    return float(math.sqrt(max(0.0, (proj ** 2 / (lam[:k] + floor)).sum() + resid / floor)))


def selectivity(act_a: np.ndarray, act_b: np.ndarray) -> np.ndarray:
    """Per-neuron d' between A and B entity activations.  act_*: (n_entities, n_neurons)."""
    ma, mb = act_a.mean(0), act_b.mean(0)
    sd = np.sqrt(0.5 * (act_a.var(0) + act_b.var(0))) + 1e-6
    return (ma - mb) / sd


def target_dprime(act: np.ndarray, target: int) -> np.ndarray:
    """Per-neuron standardised deviation of the target from the distractors of one image.  act: (n_ent, n)."""
    others = np.delete(act, target, axis=0)
    return (act[target] - others.mean(0)) / (others.std(0) + 1e-3)
