"""Experiment 7.5: new interpretability-guided optimisation pilots (pure helpers, no model state).

Results doc: docs/analysis/application/results_2026-11-11_new_optimizations.md.

Pilot A (template formation at full-attention layer 8). Experiment 2.7 round 3 (plan.md finding 50) found that the
comparison of an entity with the preceding entities starts at decoder full-attention layer 8: heads 6, 9 and 1 move a
template of the preceding entities onto each entity's tokens, and MLP 8-15 subtract it from the entity's own content.
For the first entities in raster order the softmax of layer 8 has few earlier keys and places most of its mass on the
entity's own tokens, so the template is mostly the entity itself and the difference is small. The helpers here build

  * the own-entity key mask used on the base model (analysis intervention, before training): image-token queries of
    the chosen heads may not attend to the image tokens of their own entity;
  * the decoder LoRA scopes (layer, module list, rank) of the pre-registered condition and its equal-parameter
    location controls; the update is applied only at image-token positions.

Pilot B (format tokens before a number). Experiment 3.15 round 2 (finding 52) found that in direct answers the
hundreds digit of a coordinate is read from the image mainly at the 1-3 format tokens written before the number.
The helpers here define three answer formats that differ in the number of format tokens before each coordinate value,
and a parser that returns the coordinate values in a fixed order for every format.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import torch

GRID = 6
L8_TEMPLATE_HEADS = (6, 9, 1)            # 2.7 round 3: top layer-8 heads by drop of reader attention when blocked
FULL_ATTN_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
READERS = ((16, 6), (16, 11), (16, 2), (20, 2), (20, 11), (20, 10))   # 2.7 reader heads (row query)

# Raster bins of the target cell in the 6 x 6 probe (1-based raster index): the first cell has no earlier entity,
# cells 2-6 are the rest of the first row, 7-12 the second row, 13-36 the remaining rows.
RASTER_BINS = (('c1', 1, 1), ('c2-6', 2, 6), ('c7-12', 7, 12), ('c13-36', 13, 36))


def raster_bin(target: list[int] | tuple[int, int], cols: int = GRID) -> str:
    idx = target[0] * cols + target[1] + 1
    for name, lo, hi in RASTER_BINS:
        if lo <= idx <= hi:
            return name
    raise ValueError(f'target {target} outside the {cols}-column grid')


# ----------------------------------------------------------------------------- pilot A: own-entity key mask

def own_entity_mask(L: int, groups: list[list[int]]) -> torch.Tensor:
    """(L, L) bool mask, True = may attend. Causal; image tokens of an entity may not attend to any image token of
    the same entity (including themselves). Tokens outside the entities keep the causal mask."""
    m = torch.ones(L, L, dtype=torch.bool).tril()
    for g in groups:
        t = torch.as_tensor(g)
        m[t[:, None], t[None, :]] = False
    return m


PREMASK_CONDITIONS = {
    # name: list of (layer, heads or None = all heads) whose image-token queries cannot read their own entity
    'none': [],
    'own_l8_h691': [(8, L8_TEMPLATE_HEADS)],
    'own_l8_all': [(8, None)],
    'own_l4_all': [(4, None)],
    'own_l12_all': [(12, None)],
}


# ----------------------------------------------------------------------------- pilot A: decoder LoRA scopes

HIDDEN, Q_OUT, KV_OUT, MLP_DIM = 4096, 8192, 1024, 12288

SCOPES = {
    # Pre-registered condition: query and key projections of full-attention layer 8 (template heads live here).
    'fa8qk': {'layers': [8], 'modules': ['self_attn.q_proj', 'self_attn.k_proj'], 'rank': 16},
    # Equal-parameter location controls of the same module type: layer 12 (inside the marker-writing window but
    # not the template layer) and layer 24 (after the grid reader heads of layers 16 and 20).
    'fa12qk': {'layers': [12], 'modules': ['self_attn.q_proj', 'self_attn.k_proj'], 'rank': 16},
    'fa24qk': {'layers': [24], 'modules': ['self_attn.q_proj', 'self_attn.k_proj'], 'rank': 16},
    # Component control: the MLPs of layers 8-15 that compute the difference (rank 1, 1.5x the parameters).
    'mlp8_15': {'layers': list(range(8, 16)), 'modules': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
                'rank': 1},
}

_SHAPES = {'self_attn.q_proj': (HIDDEN, Q_OUT), 'self_attn.k_proj': (HIDDEN, KV_OUT),
           'self_attn.v_proj': (HIDDEN, KV_OUT), 'self_attn.o_proj': (HIDDEN, HIDDEN),
           'mlp.gate_proj': (HIDDEN, MLP_DIM), 'mlp.up_proj': (HIDDEN, MLP_DIM), 'mlp.down_proj': (MLP_DIM, HIDDEN)}


def scope_params(scope: str) -> int:
    """Trainable LoRA parameters r (in + out) summed over the scope's linear layers."""
    s = SCOPES[scope]
    return len(s['layers']) * sum(s['rank'] * sum(_SHAPES[m]) for m in s['modules'])


# ----------------------------------------------------------------------------- pilot B: answer formats

# Styles differ in the format tokens written before each value after the first: 'list' (native) writes ', ',
# 'key' writes ', "y": ' (a key name and a colon), 'bare' writes a single ','. Token counts are recorded at run time.
FORMATS = {
    'point': {
        'list': 'Return only [{"point_2d": [x, y]}].',
        'key': 'Return only {"x": x, "y": y}.',
        'bare': 'Return only x,y (two integers separated by a comma, no spaces, no brackets).',
    },
    'bbox': {
        'list': 'Return only [{"bbox_2d": [x_min, y_min, x_max, y_max]}].',
        'key': 'Return only {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}.',
        'bare': 'Return only x_min,y_min,x_max,y_max (four integers separated by commas, no spaces, no brackets).',
    },
}
NATIVE_POINT = 'Return only [{"point_2d": [x, y]}].'
NATIVE_BBOX = 'Return only [{"bbox_2d": [x_min, y_min, x_max, y_max]}].'


def format_prompt(prompt: str, fmt: str, style: str) -> str:
    """Replace the native answer-format sentence of a 3.13 prompt by the requested style."""
    native = NATIVE_POINT if fmt == 'point' else NATIVE_BBOX
    if native not in prompt:
        raise ValueError(f'native format sentence not found in prompt: {prompt[:120]!r}')
    out = prompt.replace(native, FORMATS[fmt][style])
    if style != 'list':
        out = out.replace('Output only the JSON,', 'Output only the answer,')
    return out


_INT = re.compile(r'-?\d+(?:\.\d+)?')


def parse_values(text: str, fmt: str, style: str) -> list[float] | None:
    """Coordinate values in the order (x, y) or (x_min, y_min, x_max, y_max); None if the answer does not parse."""
    n = 2 if fmt == 'point' else 4
    t = text.strip()
    if style == 'key':
        names = ('x', 'y') if fmt == 'point' else ('x_min', 'y_min', 'x_max', 'y_max')
        try:
            obj = json.loads(t[t.index('{'):t.rindex('}') + 1])
            return [float(obj[k]) for k in names]
        except (ValueError, KeyError, TypeError):
            return None
    if style == 'list':
        key = 'point_2d' if fmt == 'point' else 'bbox_2d'
        m = re.search(rf'"{key}"\s*:\s*\[([^\]]*)\]', t)
        if not m:
            return None
        vals = _INT.findall(m.group(1))
    else:
        lines = [ln for ln in t.split('\n') if _INT.search(ln)]
        vals = _INT.findall(lines[0]) if lines else []
    return [float(v) for v in vals] if len(vals) == n else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n_boot, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
