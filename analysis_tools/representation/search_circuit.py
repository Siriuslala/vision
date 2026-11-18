"""Experiment 2.7 round 3: the computation that locates the odd entity in Qwen3.5-9B (stimuli and model-free helpers).

Design and results: docs/analysis/representation/design_2026-11-05_search_circuit.md and
results_2026-11-05_search_circuit.md.

Every stimulus is a single-image counterfactual pair that differs only at the target cell:

  odd        one entity differs from all others (feature-present target: A among B, variant 'std';
             feature-absent target: B among A, variant 'swap')
  corrupted  the same image with the target redrawn as a majority entity (homB for 'std', homA for 'swap'),
             identical jitter, orientation and blank pattern

Sets
  atlas   6 x 6, eleven feature pairs x both target variants at the subtle level (plus Q/O and open/closed circle
          at the salient level); targets spread over all cells.  Feature-agnosticity tests use this set.
  prev    6 x 6, the target is fixed at cell (row 4, column 4) (raster index 21, 21 raster-earlier cells), and a
          random subset of the earlier cells is left blank so that the number of raster-earlier entities n_prev
          takes the values in PREV_COUNTS while the target's sequence position, row and column stay fixed.
  mag     6 x 6, Q/O tail length and open/closed gap swept over four levels, both target variants, one target per
          cell (36 per condition): dose-response in the size of the difference and in the raster index.
  shift   6 x 6 homogeneous images whose first k raster cells are blank (no odd entity): tests whether the decoder's
          start effect follows the first image tokens or the first entities.

GPU stages: search_circuit_run.py; CPU analysis and figures: search_circuit_analysis.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis_tools.behavior.decomposition import unit_seed, write_jsonl
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_r2 as r2

GENERATOR = r2.GENERATOR + '+r3'
GRID = 6
PREV_TARGET = (3, 3)                       # 0-based (row, col); raster index 21
PREV_COUNTS = (0, 2, 5, 10, 21)
CORRUPT_OF = {'std': 'homB', 'swap': 'homA'}
BLANK_TILE = np.full((sa.CELL, sa.CELL, 3), 255, np.uint8)


def level_param(pair: str, level: str | float) -> float:
    """Named level ('subtle' / 'salient') or an explicit numeric parameter."""
    if isinstance(level, str):
        return r2.ALL_PAIRS[pair]['levels'][level]
    return float(level)


def render_pair(pair: str, param: float, variant: str, target: tuple[int, int], orient: int, jit: dict,
                blank: set[tuple[int, int]]) -> dict[str, tuple]:
    """Odd image and its corrupted counterpart.  Blank cells hold a white tile and are marked in the metadata."""
    out = {}
    for name, v in (('odd', variant), ('corrupted', CORRUPT_OF[variant])):
        has = {'std': lambda rc: rc == target, 'swap': lambda rc: rc != target,
               'homA': lambda rc: True, 'homB': lambda rc: False}[v]
        tiles = {rc: (BLANK_TILE if rc in blank else r2.render_entity(pair, has(rc), param, orient, jit[rc]))
                 for rc in jit}
        image, meta = oe.compose(GRID, GRID, sa.CELL, tiles)
        for inst in meta['instances']:
            rc = (inst['row'], inst['col'])
            inst['blank'] = rc in blank
            inst['has_feature'] = (not inst['blank']) and bool(has(rc))
        out[name] = (image, meta)
    return out


def blank_cells(n_prev: int, target: tuple[int, int], rng: np.random.Generator) -> set[tuple[int, int]]:
    """Leave n_prev of the raster-earlier cells filled, chosen uniformly; all later cells stay filled."""
    t = target[0] * GRID + target[1]
    if not 0 <= n_prev <= t:
        raise ValueError(f'n_prev={n_prev} impossible for raster index {t}')
    earlier = np.arange(t)
    keep = set(rng.choice(earlier, size=n_prev, replace=False).tolist()) if n_prev else set()
    return {(i // GRID, i % GRID) for i in earlier if i not in keep}


def n_prev_entities(meta: dict, target: tuple[int, int]) -> int:
    t = target[0] * GRID + target[1]
    return sum(1 for inst in meta['instances'] if inst['index'] < t and not inst['blank'])


def build(output: Path, cfg: dict) -> None:
    """Write all images and items.jsonl.  Every odd item has a 'corrupted' partner id.

    cfg keys: seed; atlas_pairs, atlas_items, atlas_salient_pairs; prev_conditions [[pair, level, variant], ...],
    prev_items (per n_prev); mag {pair: [levels]}, mag_items; shift_pairs, shift_blanks, shift_items.
    """
    output.mkdir(parents=True, exist_ok=True)
    items = []

    def add(image, meta, rel, **fields):
        digest = oe._save(image, output, rel)
        items.append({'id': rel.removesuffix('.png').replace('/', ':'), 'path': rel, 'sha256': digest,
                      'meta': meta, **fields})

    def emit(set_name, pair, level, variant, targets, rng, tag, blanks=None, extra=None):
        param = level_param(pair, level)
        for p, target in enumerate(targets):
            orient = int(rng.integers(0, 4))
            jit = sa.jitter_map(GRID, GRID, rng)
            blank = blanks(p, rng) if blanks else set()
            imgs = render_pair(pair, param, variant, target, orient, jit, blank)
            ids = {}
            for name, (image, meta) in imgs.items():
                rel = f'images/{set_name}/{pair}/{tag}_p{p:03d}_{name}.png'
                ids[name] = rel.removesuffix('.png').replace('/', ':')
            for name, (image, meta) in imgs.items():
                fields = {'set': set_name, 'pair_name': pair, 'level': str(level), 'param': param,
                          'variant': variant if name == 'odd' else CORRUPT_OF[variant], 'role': name,
                          'odd_variant': variant, 'target': list(target), 'orient': orient, 'rows': GRID,
                          'cols': GRID, 'grid': GRID, 'n_prev': n_prev_entities(meta, target),
                          'partner': ids['corrupted' if name == 'odd' else 'odd'], **(extra or {})}
                add(image, meta, f'images/{set_name}/{pair}/{tag}_p{p:03d}_{name}.png', **fields)

    seed = cfg['seed']
    for pair in cfg['atlas_pairs']:
        levels = ['subtle'] + (['salient'] if pair in cfg['atlas_salient_pairs'] else [])
        for level in levels:
            for variant in ('std', 'swap'):
                rng = np.random.default_rng(unit_seed(seed, 'search_circuit', 'atlas', pair, level, variant))
                targets = oe.balanced_targets(GRID, GRID, cfg['atlas_items'], rng)
                emit('atlas', pair, level, variant, targets, rng, f'{level}_{variant}')
    for pair, level, variant in cfg['prev_conditions']:
        for n_prev in PREV_COUNTS:
            rng = np.random.default_rng(unit_seed(seed, 'search_circuit', 'prev', pair, level, variant, n_prev))
            emit('prev', pair, level, variant, [PREV_TARGET] * cfg['prev_items'], rng,
                 f'{level}_{variant}_n{n_prev:02d}', blanks=lambda p, g, n=n_prev: blank_cells(n, PREV_TARGET, g))
    for pair, levels in cfg['mag'].items():
        for level in levels:
            for variant in ('std', 'swap'):
                rng = np.random.default_rng(unit_seed(seed, 'search_circuit', 'mag', pair, level, variant))
                targets = oe.balanced_targets(GRID, GRID, cfg['mag_items'], rng)
                emit('mag', pair, float(level), variant, targets, rng, f'l{level:g}_{variant}')
    # shift: homogeneous images only (the 'corrupted' image of a std item is homB); the odd member is unused.
    for pair in cfg['shift_pairs']:
        for k in cfg['shift_blanks']:
            for variant in ('std', 'swap'):
                rng = np.random.default_rng(unit_seed(seed, 'search_circuit', 'shift', pair, k, variant))
                last = (GRID * GRID - 1) // GRID, (GRID * GRID - 1) % GRID
                emit('shift', pair, 'subtle', variant, [last] * cfg['shift_items'], rng, f'k{k:02d}_{variant}',
                     blanks=lambda p, g, k=k: {(i // GRID, i % GRID) for i in range(k)}, extra={'k_blank': k})
    write_jsonl(output / 'items.jsonl', items)
    (output / 'build_config.json').write_text(json.dumps({**cfg, 'generator': GENERATOR}, indent=1))


def entity_groups(meta: dict) -> list[list[int]]:
    """Merged-token indices (within the image) of each cell in raster order (blank cells included)."""
    return oe.entity_tokens(meta)


# ----------------------------------------------------------------------------- model-free helpers

def delta_rule_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, g: np.ndarray, beta: np.ndarray,
                         init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-head gated delta rule in float64, used to test the GPU tracer.

    q, k: (T, dk) (already L2-normalised; q already scaled); v: (T, dv); g: (T,) log decay; beta: (T,).
    Returns outputs (T, dv), prediction errors e_t = v_t - (alpha_t S_{t-1})^T k_t (T, dv) and the
    predictions (T, dv).  State convention as in transformers: S has shape (dk, dv).
    """
    T, dk = k.shape
    S = np.zeros((dk, v.shape[1])) if init is None else init.astype(np.float64).copy()
    out, err, pred = np.zeros_like(v, dtype=np.float64), np.zeros_like(v, dtype=np.float64), np.zeros_like(v, dtype=np.float64)
    for t in range(T):
        S = S * np.exp(g[t])
        pred[t] = S.T @ k[t]
        err[t] = v[t] - pred[t]
        S = S + beta[t] * np.outer(k[t], err[t])
        out[t] = S.T @ q[t]
    return out, err, pred


def prediction_sources(k: np.ndarray, g: np.ndarray, beta: np.ndarray, err: np.ndarray, t: int) -> np.ndarray:
    """Unrolled delta-rule prediction at token t: pred_t = sum_{j<t} w_tj e_j with
    w_tj = exp(G_t - G_j) beta_j (k_j . k_t), G the cumulative log decay.  Returns w (t,) (zero initial state)."""
    G = np.cumsum(g)
    j = np.arange(t)
    return np.exp(G[t] - G[j]) * beta[j] * (k[j] @ k[t])


def wilson(k: int, n: int) -> tuple[float, float, float]:
    return oe.wilson(k, n)
