"""Experiment 7.3: located ViT fine-tuning for features without detectors (model-free part).

Design and results: docs/analysis/application/results_2026-11-05_located_vit_finetune.md.

Contents
* Feature pairs.  Experiment 2.7 found that a target that LACKS a part (an O among Qs, a closed circle among open
  circles) has no exclusive rectified detector in the ViT MLPs and is found only by decoder comparison.  This
  module adds six new pairs of the same kind and groups all pairs into families by the kind of part the target
  lacks:
      gap       closure*, sqgap, trigap          (target is closed; distractors have a gap)
      stroke    qo*, term, sqtail, tritail       (target lacks an attached outer stroke)
      interior  dot, circdot, theta              (target lacks an inner mark)  -- held-out family
      rotation  plusx                            (upright cross among rotated crosses) -- held-out control
  (* = held-out feature of a trained family).  A ("present") carries the part, B lacks it.
* Detector statistics on homogeneous images (the 2.7 definitions): per-neuron expected deviation zeta of one A
  among Bs (and of one B among As), exclusive-detector counts (zeta > 5).
* Linear achievability: the best deviation a single NEW rectified unit reading the MLP input of a block could
  reach, estimated with a shrinkage LDA direction on patches and an entity max (see ``achievable_zeta``).
* LoRA scopes for the ViT (MLP / attention of chosen blocks, all ViT linear layers incl. the merger) and their
  parameter counts.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_r2 as sr2

# ----------------------------------------------------------------------------- feature pairs

NEW_PAIRS = {
    'sqgap': {'A': 'square with a side gap', 'B': 'closed square', 'levels': {'subtle': 4, 'salient': 12}},
    'trigap': {'A': 'triangle with a side gap', 'B': 'closed triangle', 'levels': {'subtle': 4, 'salient': 12}},
    'sqtail': {'A': 'square with a corner tail', 'B': 'square', 'levels': {'subtle': 6, 'salient': 14}},
    'tritail': {'A': 'triangle with a vertex tail', 'B': 'triangle', 'levels': {'subtle': 6, 'salient': 14}},
    'circdot': {'A': 'circle with an inner dot', 'B': 'circle', 'levels': {'subtle': 4, 'salient': 8}},
    'theta': {'A': 'circle with an inner bar', 'B': 'circle', 'levels': {'subtle': 8, 'salient': 20}},
}
LEVELS = {**{p: v['levels'] for p, v in sa.PAIRS.items()}, **{p: v['levels'] for p, v in sr2.NEW_PAIRS.items()},
          **{p: v['levels'] for p, v in NEW_PAIRS.items()}}
FAMILIES = {
    'gap': ('closure', 'sqgap', 'trigap'),
    'stroke': ('qo', 'term', 'sqtail', 'tritail'),
    'interior': ('dot', 'circdot', 'theta'),
    'rotation': ('plusx',),
}
TRAIN_PAIRS = ('sqgap', 'trigap', 'term', 'sqtail', 'tritail')
HELDOUT_FEATURES = ('closure', 'qo')                  # trained families, untrained features
HELDOUT_FAMILY = ('dot', 'circdot', 'theta', 'plusx')
SYMMETRIC_CONTROLS = ('curve', 'tilt')                # detectors in both directions (2.7)
PROBE_PAIRS = TRAIN_PAIRS + HELDOUT_FEATURES + HELDOUT_FAMILY
LOCATE_PAIRS = PROBE_PAIRS + SYMMETRIC_CONTROLS


def family_of(pair: str) -> str:
    for fam, pairs in FAMILIES.items():
        if pair in pairs:
            return fam
    return 'control'


def split_of(pair: str) -> str:
    if pair in TRAIN_PAIRS:
        return 'train_feature'
    if pair in HELDOUT_FEATURES:
        return 'heldout_feature'
    if pair in HELDOUT_FAMILY:
        return 'heldout_family'
    return 'control'


def _square(cx, cy, h):
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


def _triangle(cx, cy, r, orient):
    """Equilateral triangle with circumradius r, apex rotated by orient x 90 degrees."""
    base = -math.pi / 2 + orient * math.pi / 2
    return [(cx + r * math.cos(base + k * 2 * math.pi / 3), cy + r * math.sin(base + k * 2 * math.pi / 3))
            for k in range(3)]


def _outline(draw, verts, gap_side: int | None = None, gap: float = 0.0):
    """Closed polygon outline; side gap_side (from vertex k to k+1) gets a centred gap of `gap` pixels."""
    n = len(verts)
    for k in range(n):
        p, q = verts[k], verts[(k + 1) % n]
        # extend each side by half a stroke so that corners close
        dx, dy = q[0] - p[0], q[1] - p[1]
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        e = sa.STROKE / 2
        p2, q2 = (p[0] - ux * e, p[1] - uy * e), (q[0] + ux * e, q[1] + uy * e)
        if k == gap_side and gap > 0:
            m = (L - gap) / 2
            sa._line(draw, p2, (p[0] + ux * m, p[1] + uy * m))
            sa._line(draw, (q[0] - ux * m, q[1] - uy * m), q2)
        else:
            sa._line(draw, p2, q2)


def render_entity(pair: str, present: bool, param: float, orient: int, offset: tuple[int, int]) -> np.ndarray:
    """One (64, 64, 3) tile for any 2.7 pair or new pair; present=True draws A (the member with the part)."""
    if pair not in NEW_PAIRS:
        return sr2.render_entity(pair, present, param, orient, offset)
    image, draw = sa._canvas()
    cx, cy = sa.CELL / 2 + offset[0], sa.CELL / 2 + offset[1]
    if pair == 'sqgap':
        _outline(draw, _square(cx, cy, 13.0), orient % 4 if present else None, param)
    elif pair == 'trigap':
        _outline(draw, _triangle(cx, cy + 2, 17.0, orient), orient % 3 if present else None, param)
    elif pair == 'sqtail':
        h = 12.0
        _outline(draw, _square(cx, cy, h))
        if present:
            sx, sy = ((1, 1), (-1, 1), (-1, -1), (1, -1))[orient % 4]
            c0 = (cx + sx * h, cy + sy * h)
            d = param / math.sqrt(2)
            sa._line(draw, c0, (c0[0] + sx * d, c0[1] + sy * d))
    elif pair == 'tritail':
        verts = _triangle(cx, cy + 2, 16.0, orient)
        _outline(draw, verts)
        if present:
            vx, vy = verts[orient % 3]
            ux, uy = vx - cx, vy - (cy + 2)
            n = math.hypot(ux, uy)
            sa._line(draw, (vx, vy), (vx + ux / n * param, vy + uy / n * param))
    elif pair in ('circdot', 'theta'):
        r = 14.0
        sa._polyline(draw, sa._arc_points(cx, cy, r, 0, 2 * math.pi, 128))
        if present and pair == 'circdot':
            q = param / 2
            draw.ellipse([(cx - q) * sa.SS, (cy - q) * sa.SS, (cx + q) * sa.SS, (cy + q) * sa.SS], fill=sa.BLACK)
        elif present:
            q = param / 2
            if orient % 2 == 0:
                sa._line(draw, (cx - q, cy), (cx + q, cy))
            else:
                sa._line(draw, (cx, cy - q), (cx, cy + q))
    return sa._finish(image)


VARIANT_RULES = {'std': lambda rc, t: rc == t, 'swap': lambda rc, t: rc != t,
                 'homA': lambda rc, t: True, 'homB': lambda rc, t: False}


def grid_image(pair: str, param: float, rows: int, cols: int, target: tuple[int, int] | None, orient: int,
               jit: dict, variant: str) -> tuple[Image.Image, dict]:
    """Render one rows x cols image; std = one A among Bs at `target`, swap = one B among As."""
    rule = VARIANT_RULES[variant]
    tiles = {rc: render_entity(pair, rule(rc, target), param, orient, jit[rc]) for rc in jit}
    image, meta = oe.compose(rows, cols, sa.CELL, tiles)
    for inst in meta['instances']:
        inst['has_feature'] = bool(rule((inst['row'], inst['col']), target))
    return image, meta


def patch_index(entity_tokens: list[list[int]]) -> np.ndarray:
    """(n_entities, 16) ViT patch indices: merged token t covers patches 4t .. 4t+3 (the merge-window order)."""
    t = np.asarray(entity_tokens)
    return (t[..., None] * 4 + np.arange(4)).reshape(t.shape[0], -1)


# ----------------------------------------------------------------------------- detector statistics

EPS = 0.01
DETECTOR_ZETA = 5.0


def zeta(mu_t: np.ndarray, mu_d: np.ndarray, sd_d: np.ndarray) -> np.ndarray:
    """Expected deviation of one target among distractors per neuron: (mu_target - mu_distractor) / (sd_d + eps)."""
    return (mu_t - mu_d) / (sd_d + EPS)


def detector_summary(z: np.ndarray) -> dict:
    """Counts and magnitude of exclusive detectors for one direction (z: per-neuron zeta)."""
    s = np.sort(z)[::-1]
    return {'n_detectors': int((z > DETECTOR_ZETA).sum()), 'max_zeta': float(s[0]), 'top10_zeta': float(s[:10].mean()),
            'argmax': int(np.argmax(z))}


def lda_direction(pos: np.ndarray, neg: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Shrinkage LDA direction (Sigma + shrink * tr(Sigma)/d I)^{-1} (mu_pos - mu_neg), unit norm.

    pos, neg: (n, d) patch features.  Sigma is the pooled within-class covariance.
    """
    mp, mn = pos.mean(0), neg.mean(0)
    xc = np.concatenate([pos - mp, neg - mn])
    cov = xc.T @ xc / max(1, len(xc) - 2)
    d = cov.shape[0]
    cov += shrink * np.trace(cov) / d * np.eye(d)
    w = np.linalg.solve(cov, mp - mn)
    return w / (np.linalg.norm(w) + 1e-12)


def achievable_zeta(tgt_fit: np.ndarray, dis_fit: np.ndarray, tgt_test: np.ndarray, dis_test: np.ndarray,
                    shrink: float = 0.1) -> float:
    """Deviation a single new linear-threshold unit on these inputs could reach (held-out images).

    Arguments are (n_entities, 16, d) patch inputs of the MLP of one block: target-class entities (all from
    homogeneous images of the target class) and distractor-class entities, split into fit and test images.
    A direction w is fitted on patches (target patches vs distractor patches, shrinkage LDA); the unit's entity
    response is max_p w^T x_p (a rectified detector fires where its feature is, so the entity maximum is the
    relevant read-out, as in the 2.7 definition); the returned value is
    (mean_target - mean_distractor) / (sd_distractor + eps) of that response on the test entities, i.e. the zeta
    such a unit would have.  It upper-bounds what one new MLP neuron (fc1 row) of that block can do without
    changing the block's input.  The response scale is set by the test distractor spread, so the value is
    comparable with the neuron zeta only through its threshold interpretation (both are > 5 for a clean
    exclusive detector).
    """
    d = tgt_fit.shape[-1]
    w = lda_direction(tgt_fit.reshape(-1, d), dis_fit.reshape(-1, d), shrink)
    st = (tgt_test @ w).max(1)
    sd = (dis_test @ w).max(1)
    return float((st.mean() - sd.mean()) / (sd.std() + 1e-6))


# ----------------------------------------------------------------------------- LoRA scopes

VIT_DIM, VIT_MLP, VIT_BLOCKS = 1152, 4304, 27
MERGER_IN, MERGER_OUT = 4608, 4096


def parse_scope(scope: str) -> dict:
    """Scope grammar: 'vit_mlp:9,10', 'vit_mlp:9-11', 'vit_attn:9-11', 'vit_fc1:9-11', 'vit_all', 'model_all'.

    Returns {'kind': ..., 'blocks': [1-based ViT blocks]}.
    """
    if scope in ('vit_all', 'model_all'):
        return {'kind': scope, 'blocks': list(range(1, VIT_BLOCKS + 1))}
    kind, _, spec = scope.partition(':')
    if kind not in ('vit_mlp', 'vit_attn', 'vit_fc1') or not spec:
        raise ValueError(f'bad scope {scope!r}')
    blocks: list[int] = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        blocks += list(range(int(a), int(b or a) + 1))
    if not blocks or min(blocks) < 1 or max(blocks) > VIT_BLOCKS or len(set(blocks)) != len(blocks):
        raise ValueError(f'bad block list in {scope!r}')
    return {'kind': kind, 'blocks': sorted(blocks)}


def lora_params(scope: str, rank: int, decoder_all_linear: int = 0) -> int:
    """Trainable LoRA parameters r (in + out) summed over the scope's ViT linear layers.

    decoder_all_linear: parameter count of the decoder all-linear LoRA (added for 'model_all'; it depends on the
    decoder shapes and is measured from the model).
    """
    s = parse_scope(scope)
    mlp = rank * (VIT_DIM + VIT_MLP) * 2
    fc1 = rank * (VIT_DIM + VIT_MLP)
    attn = rank * (VIT_DIM + 3 * VIT_DIM) + rank * 2 * VIT_DIM
    merger = rank * 2 * MERGER_IN + rank * (MERGER_IN + MERGER_OUT)
    n = len(s['blocks'])
    if s['kind'] == 'vit_mlp':
        return n * mlp
    if s['kind'] == 'vit_fc1':
        return n * fc1
    if s['kind'] == 'vit_attn':
        return n * attn
    total = VIT_BLOCKS * (mlp + attn) + merger
    return total + (decoder_all_linear if s['kind'] == 'model_all' else 0)


def vit_modules(visual, scope: str) -> dict:
    """{name: nn.Linear} of the ViT part of a scope (visual = model.model.visual)."""
    s = parse_scope(scope)
    out = {}
    for b in s['blocks']:
        blk = visual.blocks[b - 1]
        if s['kind'] in ('vit_mlp', 'vit_all', 'model_all', 'vit_fc1'):
            out[f'V{b}.mlp.linear_fc1'] = blk.mlp.linear_fc1
        if s['kind'] in ('vit_mlp', 'vit_all', 'model_all'):
            out[f'V{b}.mlp.linear_fc2'] = blk.mlp.linear_fc2
        if s['kind'] in ('vit_attn', 'vit_all', 'model_all'):
            out[f'V{b}.attn.qkv'] = blk.attn.qkv
            out[f'V{b}.attn.proj'] = blk.attn.proj
    if s['kind'] in ('vit_all', 'model_all'):
        out['merger.linear_fc1'] = visual.merger.linear_fc1
        out['merger.linear_fc2'] = visual.merger.linear_fc2
    return out
