"""Experiment 3.13 round 2: locating the readout defect and the change of the upstream coordinate code.

Design and results: docs/analysis/position/results_2026-11-03_targeted_finetune_r2.md.

Model-free parts (unit-tested without a GPU):

* splice configurations: which LoRA modules of a trained adapter are active (layer bands, component types) and at
  which token positions (image tokens, question tokens, answer-prefix tokens). Activating the adapter on layers
  1..k only is exactly the residual splice "tuned residual after layer k, base layers k+1..32" at every position;
* geometry of a coordinate code: smooth-plus-periodic regression spectra, the coordinate-code subspace and principal
  angles between subspaces;
* the round-2 training data: GQA scene-graph referring expressions (natural grounding with same-category
  distractors) and prompts for self-distilled instruction samples.

Layer numbers are 1-based; "layer k output" is the residual stream after decoder layer k (layer 0 = input embeddings).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

import numpy as np

N_LAYERS = 32
FULL_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
POSITION_KINDS = ('all', 'image', 'question', 'answer', 'text')


# ----------------------------------------------------------------------------- splice configurations

def _layer(name: str) -> int:
    """'L12.mlp.gate_proj' -> 12."""
    m = re.match(r'L(\d+)\.', name)
    if not m:
        raise ValueError(f'Not an adapter module name: {name}')
    return int(m.group(1))


def _kind(name: str) -> str:
    """Component type of an adapter module: 'mlp', 'fa' (full attention) or 'dn' (Gated DeltaNet)."""
    if '.mlp.' in name:
        return 'mlp'
    if '.self_attn.' in name:
        return 'fa'
    if '.linear_attn.' in name:
        return 'dn'
    raise ValueError(f'Unknown module type: {name}')


@dataclass(frozen=True)
class Splice:
    """One analysis configuration of a trained adapter.

    name:      short identifier used in tables;
    active:    adapter module names whose update is applied (empty = base model);
    positions: 'all' | 'image' | 'question' | 'answer' | 'text' (question + answer) token positions receiving the update;
    group:     table group ('depth_fwd', 'depth_rev', 'band', 'type', 'position', ...).
    """
    name: str
    active: frozenset
    positions: str
    group: str


def splice_configs(names: list[str], full: bool = True) -> list[Splice]:
    """Configurations for an adapter with module names `names` (all-linear adapters get the full set).

    depth_fwd k: adapter active in layers 1..k (tuned residual after layer k fed to base layers k+1..32);
    depth_rev k: adapter active in layers k+1..32 (base residual after layer k fed to tuned layers k+1..32);
    band a-b:    adapter active only in layers a..b;
    type:        only one component type (MLP / full attention / DeltaNet / both token mixers), all depths or split
                 at layer 24 (upstream vs the readout stage 25-32);
    position:    full adapter (or layers 1-24) applied only at image / question / answer-prefix / text positions.
    """
    names = list(names)
    every = frozenset(names)
    out = [Splice('base', frozenset(), 'all', 'ref'), Splice('full', every, 'all', 'ref')]
    if not full:
        for pos in ('image', 'text'):
            out.append(Splice(f'full@{pos}', every, pos, 'position'))
        return out
    sel = lambda f: frozenset(n for n in names if f(n))
    for k in range(4, 32, 4):
        out.append(Splice(f'fwd{k}', sel(lambda n: _layer(n) <= k), 'all', 'depth_fwd'))
        out.append(Splice(f'rev{k}', sel(lambda n: _layer(n) > k), 'all', 'depth_rev'))
    for a in range(1, 33, 4):
        out.append(Splice(f'band{a}_{a + 3}', sel(lambda n: a <= _layer(n) <= a + 3), 'all', 'band'))
    for t in ('mlp', 'fa', 'dn'):
        out.append(Splice(f'{t}', sel(lambda n: _kind(n) == t), 'all', 'type'))
    out.append(Splice('mixer', sel(lambda n: _kind(n) != 'mlp'), 'all', 'type'))
    for t, f in (('mlp', lambda n: _kind(n) == 'mlp'), ('mixer', lambda n: _kind(n) != 'mlp')):
        out.append(Splice(f'{t}_le24', sel(lambda n: f(n) and _layer(n) <= 24), 'all', 'type_split'))
        out.append(Splice(f'{t}_ge25', sel(lambda n: f(n) and _layer(n) >= 25), 'all', 'type_split'))
    for pos in ('image', 'question', 'answer', 'text'):
        out.append(Splice(f'full@{pos}', every, pos, 'position'))
        out.append(Splice(f'fwd24@{pos}', sel(lambda n: _layer(n) <= 24), pos, 'position'))
    return out


def position_mask(kind: str, length: int, image_span: tuple[int, int], prompt_len: int, total: int) -> np.ndarray:
    """(total,) float mask of the positions that receive the adapter update.

    length:     number of real tokens of the row (prompt + answer prefix + forced digit);
    image_span: [start, end) of the image tokens; prompt_len: tokens before the answer prefix;
    total:      padded length of the batch (padding positions get 0 except for kind 'all').
    """
    if kind not in POSITION_KINDS:
        raise ValueError(kind)
    m = np.zeros(total, np.float32)
    s, e = image_span
    if kind == 'all':
        m[:] = 1
    elif kind == 'image':
        m[s:e] = 1
    elif kind == 'question':
        m[:prompt_len] = 1
        m[s:e] = 0
    elif kind == 'answer':
        m[prompt_len:length] = 1
    elif kind == 'text':
        m[:length] = 1
        m[s:e] = 0
    return m


# ----------------------------------------------------------------------------- teacher-forced digit metrics

def digit_metrics(truth: np.ndarray, logp0: np.ndarray, logp1: np.ndarray) -> dict:
    """Readout metrics with the first digit forced to the true digit (3.12 definitions).

    truth (N,) values 100-999 in 0-1000 units; logp0 (N, 10) first-digit log-probabilities at q0; logp1 (N, 10)
    second-digit log-probabilities at q1 after the true first digit.
    p_first: mean P(true first digit) (renormalised over digits); p_second: mean P(true second digit);
    gain: least-squares slope of E[second digit] on the in-bin offset (x - 100 k) / 10 (ideal 1; 3.12 'second-digit
    gain'); value_err: mean |100 k* + 10 E[s] + 5 - x| with k* the arg-max first digit and E[s] taken at the true k
    (a teacher-forced proxy of the coordinate error).
    """
    truth = np.asarray(truth, float)
    p0 = np.exp(logp0 - logp0.max(1, keepdims=True))
    p0 /= p0.sum(1, keepdims=True)
    p1 = np.exp(logp1 - logp1.max(1, keepdims=True))
    p1 /= p1.sum(1, keepdims=True)
    n = np.round(truth).astype(int)
    f, s = n // 100, (n // 10) % 10
    idx = np.arange(len(n))
    E = (p1 * np.arange(10)).sum(1)
    off = (truth - 100 * f) / 10
    gain = float(np.polyfit(off, E, 1)[0]) if len(n) > 2 else float('nan')
    value = 100 * p0.argmax(1) + 10 * E + 5
    return {'p_first': float(p0[idx, f].mean()), 'p_second': float(p1[idx, s].mean()), 'gain': gain,
            'first_acc': float((p0.argmax(1) == f).mean()), 'value_err': float(np.abs(value - truth).mean()),
            'n': int(len(n))}


def recovery(metric_cfg: float, metric_base: float, metric_full: float) -> float:
    """Share of the base -> fully-tuned change reproduced by a configuration (0 = base, 1 = fully tuned)."""
    d = metric_full - metric_base
    return float('nan') if abs(d) < 1e-9 else float((metric_cfg - metric_base) / d)


# ----------------------------------------------------------------------------- geometry of a coordinate code

def smooth_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """(N, degree) Legendre polynomials of x rescaled to [-1, 1] (x in 0-1000 units); the smooth, monotone-like part."""
    u = np.asarray(x, float) / 500.0 - 1.0
    return np.stack([np.polynomial.legendre.Legendre.basis(d)(u) for d in range(1, degree + 1)], 1)


def periodic_basis(x: np.ndarray, period: float) -> np.ndarray:
    """(N, 2) cos / sin of 2 pi x / period."""
    a = 2 * np.pi * np.asarray(x, float) / period
    return np.stack([np.cos(a), np.sin(a)], 1)


def _r2_multi(Y: np.ndarray, X: np.ndarray) -> float:
    """Variance share of the columns of Y (centred) explained by least squares on X (plus intercept)."""
    X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    coef, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    res = Y - X1 @ coef
    Yc = Y - Y.mean(0)
    return float(1 - (res ** 2).sum() / (Yc ** 2).sum())


def pca(H: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k principal components of rows of H: (scores (N, k), components (k, D), explained variance ratios (k,))."""
    Hc = H - H.mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var = S ** 2
    return U[:, :k] * S[:k], Vt[:k], var[:k] / var.sum()


def pca_scores(H: np.ndarray, k: int) -> np.ndarray:
    """Top-k principal component scores (N, k) via the N x N Gram matrix (fast when N << D)."""
    Hc = (H - H.mean(0)).astype(np.float64)
    U, S, _ = np.linalg.svd(Hc @ Hc.T)
    return U[:, :k] * np.sqrt(np.maximum(S[:k], 0))


def periodic_spectrum(H: np.ndarray, x: np.ndarray, periods: tuple[float, ...], n_pc: int = 32,
                      degree: int = 3) -> dict:
    """How much of the coordinate-dependent variation of H is periodic in x.

    H (N, D) residuals of N inputs whose coordinate is x (N,). The rows are reduced to their top n_pc principal
    components Y. smooth = R^2 of Y on a degree-`degree` polynomial of x; for every period T, extra[T] = the
    additional R^2 when cos / sin(2 pi x / T) are appended to the polynomial. A monotone 'place code' has extra ~ 0
    at every T; a number code with a period-T component (e.g. the clock-like Fourier features of text numbers) has a
    peak at T. Least squares on N samples with 2 extra regressors adds about 2 / N by chance (reported as 'chance').
    """
    Y = pca_scores(H, min(n_pc, len(H) - 1))
    S = smooth_basis(x, degree)
    base = _r2_multi(Y, S)
    extra = {float(T): _r2_multi(Y, np.concatenate([S, periodic_basis(x, T)], 1)) - base for T in periods}
    return {'smooth': base, 'extra': extra, 'chance': 2.0 / len(x)}


def code_subspace(H: np.ndarray, x: np.ndarray, k: int = 3, periods: tuple[float, ...] = (100.0,),
                  degree: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis (k, D) of the coordinate code: top-k principal directions of the part of H explained by a
    smooth-plus-periodic function of x, and the (D,) mean derivative direction dH/dx (unit-free, per 0-1000 unit).

    The fitted part removes variation that does not depend on x (content noise, batch noise), so the subspace
    describes where the coordinate itself lives in the residual stream.
    """
    B = smooth_basis(x, degree)
    for T in periods:
        B = np.concatenate([B, periodic_basis(x, T)], 1)
    B1 = np.concatenate([B, np.ones((len(x), 1))], 1)
    coef, *_ = np.linalg.lstsq(B1, H, rcond=None)
    fit = B1 @ coef
    fit -= fit.mean(0)
    _, _, Vt = np.linalg.svd(fit, full_matrices=False)
    order = np.argsort(x)
    xs, fs = np.asarray(x, float)[order], fit[order]
    dx = np.diff(xs)
    ok = dx > 0
    deriv = (np.diff(fs, axis=0)[ok] / dx[ok, None]).mean(0)
    return Vt[:k], deriv


def orthonormal(V: np.ndarray) -> np.ndarray:
    """Orthonormal basis (rank, D) of the row space of V (k, D)."""
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    r = int((S > S.max() * 1e-8).sum()) if len(S) else 0
    return Vt[:r]


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles in degrees (ascending) between the row spaces of A (a, D) and B (b, D).

    With orthonormal bases Qa, Qb the cosines of the angles are the singular values of Qa Qb^T: angle 0 means a
    shared direction, 90 means orthogonal subspaces.
    """
    Qa, Qb = orthonormal(A), orthonormal(B)
    s = np.linalg.svd(Qa @ Qb.T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def readout_subspace(J: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k input directions of a digit readout from Jacobians J (N, 10, D) of the ten digit logits with respect to
    the residual; logits are centred over digits first (a shift common to all digits does not change the digit
    choice). Returns (k, D) right singular vectors and the (k,) share of squared Jacobian norm they capture."""
    Jc = J - J.mean(1, keepdims=True)
    M = Jc.reshape(-1, Jc.shape[-1])
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:k], (S[:k] ** 2) / (S ** 2).sum()


def readout_gain(J: np.ndarray, deriv: np.ndarray) -> np.ndarray:
    """(N, 10) change of the centred digit logits per 100 coordinate units along the code derivative:
    100 * J_c dH/dx. Its spread across digits measures how strongly the readout responds to the coordinate code."""
    Jc = J - J.mean(1, keepdims=True)
    return 100 * Jc @ deriv


# ----------------------------------------------------------------------------- round-2 training data

GQA_SKIP_NAMES = {'air', 'sky', 'ground', 'wall', 'floor', 'grass', 'water', 'background', 'road', 'street', 'field',
                  'sand', 'snow', 'ceiling', 'shadow', 'cloud', 'clouds', 'tree', 'trees', 'leaves', 'building', 'part',
                  'side', 'line', 'lines', 'area', 'distance', 'surface', 'top', 'edge', 'spot', 'letter', 'letters',
                  'writing', 'word', 'words', 'number', 'reflection', 'light', 'lights', 'hair', 'pavement', 'sidewalk',
                  'dirt', 'ocean', 'mountain', 'hill', 'hills', 'forest', 'weeds', 'bushes', 'bush', 'branches'}


def gqa_expressions(graph: dict, min_area: float = 0.01, max_area: float = 0.5) -> list[dict]:
    """Referring expressions with a unique referent from one GQA scene graph.

    Two kinds: 'name' = the only object of its category ("the dog"); 'attribute' = one of several objects of its
    category, singled out by an attribute no other object of that category has ("the white dog" among dogs).
    The 'attribute' kind is the natural-image analogue of the clutter test (same-category distractors). Objects
    smaller than min_area or larger than max_area of the image, and background/stuff categories, are skipped.
    Returns dicts {object_id, expression, kind, box [x0, y0, x1, y1] in source pixels}.
    """
    W, H = graph['width'], graph['height']
    objs = graph['objects']
    by_name: dict = {}
    for oid, o in objs.items():
        by_name.setdefault(o['name'], []).append(oid)
    out = []
    for name, ids in by_name.items():
        if name in GQA_SKIP_NAMES or not re.fullmatch(r'[a-z][a-z ]{1,24}', name):
            continue
        for oid in ids:
            o = objs[oid]
            area = o['w'] * o['h'] / (W * H)
            if not min_area <= area <= max_area:
                continue
            box = [o['x'], o['y'], o['x'] + o['w'], o['y'] + o['h']]
            if len(ids) == 1:
                attr = next((a for a in o['attributes'] if re.fullmatch(r'[a-z]{3,12}', a)), None)
                expr = f'the {attr} {name}' if attr else f'the {name}'
                out.append({'object_id': oid, 'expression': expr, 'kind': 'name', 'box': box})
                continue
            others = Counter(a for j in ids if j != oid for a in objs[j]['attributes'])
            uniq = [a for a in o['attributes'] if others[a] == 0 and re.fullmatch(r'[a-z]{3,12}', a)]
            # Same-category objects whose boxes nearly coincide are duplicate annotations: skip.
            dup = any(_iou(box, [objs[j]['x'], objs[j]['y'], objs[j]['x'] + objs[j]['w'], objs[j]['y'] + objs[j]['h']]) > 0.5
                      for j in ids if j != oid)
            if uniq and not dup:
                out.append({'object_id': oid, 'expression': f'the {uniq[0]} {name}', 'kind': 'attribute', 'box': box})
    return out


def _iou(a, b) -> float:
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.


def scale_box(box: list[float], src: tuple[int, int], dst: tuple[int, int]) -> list[float]:
    """Box in source pixels -> resized-image pixels (independent x / y scale factors)."""
    sx, sy = dst[0] / src[0], dst[1] / src[1]
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def existence_question(names_present: list[str], vocab: list[str], rng: np.random.Generator) -> tuple[str, str]:
    """A POPE-style yes/no existence question (positive from present objects, negative from absent vocabulary)."""
    present = sorted(set(names_present) - GQA_SKIP_NAMES)
    absent = sorted(set(vocab) - set(names_present))
    if present and (rng.random() < 0.5 or not absent):
        return f'Is there a {present[int(rng.integers(len(present)))]} in the image?', 'yes'
    return f'Is there a {absent[int(rng.integers(len(absent)))]} in the image?', 'no'


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of a proportion k / n."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def margin_scatter(truth: np.ndarray, logp0: np.ndarray, lo: float = 200, hi: float = 900) -> dict:
    """3.12 round-3 quantity: the first-digit logit difference m = l0(k+1) - l0(k-1) at q0 (k = true first digit),
    the signal the second digit reads. m is regressed on the in-bin offset (x - 100 k) / 10 (truths lo-hi, so both
    neighbours exist); slope = nats per 10 units, scatter = residual s.d. (nats), resolution = 10 * scatter / slope
    (units of x that the scatter corresponds to; smaller is sharper)."""
    truth = np.asarray(truth, float)
    sel = (truth >= lo) & (truth < hi)
    t, lp = truth[sel], logp0[sel]
    k = (np.round(t).astype(int)) // 100
    idx = np.arange(len(t))
    m = lp[idx, k + 1] - lp[idx, k - 1]
    off = (t - 100 * k) / 10
    if len(t) < 3:
        return {'slope': float('nan'), 'scatter': float('nan'), 'resolution': float('nan'), 'n': int(len(t))}
    coef = np.polyfit(off, m, 1)
    res = m - np.polyval(coef, off)
    sd = float(res.std(ddof=2))
    return {'slope': float(coef[0]), 'scatter': sd, 'resolution': float(10 * sd / coef[0]) if coef[0] > 0 else float('inf'),
            'n': int(len(t))}


def periodic_decodability(H: np.ndarray, x: np.ndarray, period: float = 100.0, n_pc: int = 32, folds: int = 5,
                          alpha: float = 1.0) -> dict:
    """Cross-validated linear decodability of cos / sin(2 pi x / period) and of x itself from H (N, D).

    Rows are reduced to their top n_pc principal components (fitted once on all rows: PCA uses no labels, so it does
    not leak the targets into the held-out folds), standardised, and a ridge
    regression (penalty alpha) predicts the targets; R^2 is computed on the held-out folds. A residual stream that
    holds only a monotone place code supports decoding of x but not of cos / sin at a period much shorter than the
    range of x; a clock code supports both. Used where content variation hides the clock from unsupervised PCA
    (natural images)."""
    x = np.asarray(x, float)
    Y = np.stack([np.cos(2 * np.pi * x / period), np.sin(2 * np.pi * x / period), x / 1000], 1)
    order = np.random.default_rng(0).permutation(len(x))
    pred = np.zeros_like(Y)
    Hc = H - H.mean(0)
    U, S, _ = np.linalg.svd(Hc @ Hc.T)                     # N x N Gram matrix: cheap for N << D
    k = min(n_pc, len(x) - 1)
    Z_all = U[:, :k] * np.sqrt(np.maximum(S[:k], 0))
    for f in range(folds):
        te = np.zeros(len(x), bool)
        te[order[f::folds]] = True
        Z = Z_all
        sd = Z[~te].std(0) + 1e-8
        Z = Z / sd
        Zt = np.concatenate([Z[~te], np.ones((int((~te).sum()), 1))], 1)
        reg = alpha * np.eye(Zt.shape[1])
        reg[-1, -1] = 0
        W = np.linalg.solve(Zt.T @ Zt + reg, Zt.T @ Y[~te])
        pred[te] = np.concatenate([Z[te], np.ones((int(te.sum()), 1))], 1) @ W
    r2 = 1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return {'cos': float(r2[0]), 'sin': float(r2[1]), 'periodic': float((r2[0] + r2[1]) / 2), 'x': float(r2[2])}
