"""Experiment 2.7 round 4: the complete pipeline from pixels to the (row, column) answer (model-free helpers).

Design and results: docs/analysis/representation/results_2026-11-11_search_pipeline.md.

Round 4 reuses the single-image counterfactual pairs of round 3 (odd image / corrupted image, which differ only at the
target cell) and adds one more image per pair:

  same-type image  every entity is drawn as the target type (homA for a feature-present target 'std', homB for a
                   feature-absent target 'swap'), with the jitter and orientation of the pair.  The target cell holds
                   exactly the pixels of the odd image's target; every other cell holds an entity of the target type.
                   It provides the "context identical to the target" counterfactual for the reference-swap
                   interventions (what the target token reads from other cells when nothing differs from it).

The jitter of round 3 was not stored, so the same-type image is rendered by replaying the round-3 build RNG
(search_circuit.build) and is verified by re-rendering the odd image and comparing its SHA-256 with the stored one.

GPU stages: search_pipeline_run.py (Qwen3.5-9B) and search_pipeline_vl.py (Qwen3-VL-8B);
CPU analysis and figures: search_pipeline_analysis.py.
"""
from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import read_jsonl, unit_seed
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_r2 as r2
from analysis_tools.representation import search_circuit as sc

SAME_TYPE_OF = {'std': 'homA', 'swap': 'homB'}


def parse_item(it: dict) -> tuple[str, str, object, str, int]:
    """(set, pair, level key used by the round-3 build, variant, pair index p) of a round-3 odd item."""
    set_name, pair = it['set'], it['pair_name']
    tag = it['id'].split(':')[-1]                      # e.g. subtle_std_p003_odd or l10_swap_p012_odd
    p = int(tag.split('_p')[1].split('_')[0])
    if set_name == 'atlas':
        level: object = it['level']
    elif set_name == 'mag':
        level = float(it['param'])
    else:
        raise ValueError(f'same-type rendering is implemented for atlas and mag items only, got {set_name}')
    return set_name, pair, level, it['odd_variant'], p


def _png_sha(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return hashlib.sha256(buf.getvalue()).hexdigest()


def replay_item(cfg: dict, it: dict) -> dict:
    """Replay the round-3 RNG for one odd item; returns orient, jitter map, param and target."""
    set_name, pair, level, variant, p = parse_item(it)
    seed = cfg['seed']
    rng = np.random.default_rng(unit_seed(seed, 'search_circuit', set_name, pair, level, variant))
    n = cfg['atlas_items'] if set_name == 'atlas' else cfg['mag_items']
    targets = oe.balanced_targets(sc.GRID, sc.GRID, n, rng)
    for q, target in enumerate(targets):
        orient = int(rng.integers(0, 4))
        jit = sa.jitter_map(sc.GRID, sc.GRID, rng)
        if q == p:
            if tuple(target) != tuple(it['target']):
                raise ValueError(f'replay of {it["id"]} gives target {target}, stored {it["target"]}')
            return {'orient': orient, 'jit': jit, 'param': sc.level_param(pair, level), 'target': tuple(target),
                    'pair': pair, 'variant': variant}
    raise ValueError(f'pair index {p} out of range for {it["id"]}')


def render_same_type(cfg: dict, it: dict) -> tuple[Image.Image, dict, bool]:
    """Same-type image of an odd item and whether the replayed odd image reproduces the stored SHA-256."""
    rp = replay_item(cfg, it)
    imgs = sc.render_pair(rp['pair'], rp['param'], rp['variant'], rp['target'], rp['orient'], rp['jit'], set())
    ok = _png_sha(imgs['odd'][0]) == it['sha256']
    has_feature = rp['variant'] == 'std'               # the target type has the feature iff the target is 'std'
    tiles = {rc: r2.render_entity(rp['pair'], has_feature, rp['param'], rp['orient'], rp['jit'][rc]) for rc in rp['jit']}
    image, meta = oe.compose(sc.GRID, sc.GRID, sa.CELL, tiles)
    for inst in meta['instances']:
        inst['blank'] = False
        inst['has_feature'] = has_feature
    return image, meta, ok


def build_same_type(round3: Path, output: Path, ids: list[str]) -> list[dict]:
    """Render and save same-type images for the given odd item ids; returns rows {id, same_type_path, sha256, ok}."""
    cfg = json.loads((round3 / 'build_config.json').read_text())
    items = {it['id']: it for it in read_jsonl(round3 / 'items.jsonl')}
    rows = []
    for i in ids:
        image, meta, ok = render_same_type(cfg, items[i])
        if not ok:
            raise ValueError(f'replayed odd image of {i} does not match the stored SHA-256')
        rel = 'same_type/' + i.replace(':', '__') + '.png'
        digest = oe._save(image, output, rel)
        rows.append({'id': i, 'path': rel, 'sha256': digest, 'replay_ok': ok})
    return rows


# ----------------------------------------------------------------------------- geometry

def vit_patches_of_tokens(tokens: list[int]) -> list[int]:
    """ViT patch indices (within one image) of merged-token indices: merged token t holds patches 4t .. 4t + 3
    (the Qwen vision tower orders patches by 2 x 2 merge window)."""
    return [4 * t + j for t in tokens for j in range(4)]


def redirect_cell(tcell: int, grid: int = sc.GRID) -> int:
    """Control cell for reader redirection: a cell in a different row and a different column than the target,
    at raster distance >= 7, chosen deterministically (row + 2, column + 3, modulo the grid)."""
    r, c = divmod(tcell, grid)
    return ((r + 2) % grid) * grid + (c + 3) % grid


# ----------------------------------------------------------------------------- statistics

def zeta(mu_t: np.ndarray, mu_m: np.ndarray, sd_m: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """Exclusive-detector score of round 2: (mean on target-type entities - mean on majority-type entities) /
    (standard deviation on majority-type entities + eps).  zeta > 5 marks an exclusive detector."""
    return (mu_t - mu_m) / (sd_m + eps)


def cosine(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(axis) / (np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + 1e-12)


def participation_ratio(x: np.ndarray) -> float:
    """Effective dimensionality (sum lambda)^2 / sum lambda^2 of the rows of x (centred)."""
    x = np.asarray(x, np.float64)
    x = x - x.mean(0, keepdims=True)
    lam = np.linalg.svd(x, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def mean_ci(x) -> tuple[float, float]:
    """Mean and half-width of the normal 95% interval."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float('nan')), float('nan')
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
