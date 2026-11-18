"""Experiment 3.16: prior localization of the coordinate-readout defect on the base model (no training).

Results: docs/analysis/position/results_2026-11-05_readout_prior_localization.md.

Model-free parts (unit-tested without a GPU):

* block cross-validated decodability of a period-T code ("clock") of a coordinate x from features: the held-out
  folds are whole hundreds of x, so a high score requires the SAME linear readout to give the right phase in a range
  of x it never saw -- the property a digit readout shared by all hundreds needs (interleaved folds only test
  interpolation and are fooled by smooth spline-like features);
* decomposition of the 3.12 first-digit margin m = l0(k+1) - l0(k-1) into a slope on the in-bin offset, a per-hundreds
  bias and a within-bin residual;
* least-squares subspaces of a residual stream that carry the periodic (or the smooth) part of a coordinate code,
  used for subspace transplants;
* summaries of the parameter-gradient leverage map.

Layer numbers are 1-based; "layer k output" is the residual stream after decoder layer k (layer 0 = input embeddings).
"""
from __future__ import annotations

import numpy as np

PERIODS = (100.0, 50.0, 100.0 / 3, 25.0, 20.0, 10.0)


def periodic_targets(x: np.ndarray, period: float = 100.0) -> np.ndarray:
    """(N, 2) cos / sin of 2 pi x / period."""
    a = 2 * np.pi * np.asarray(x, float) / period
    return np.stack([np.cos(a), np.sin(a)], 1)


def block_cv_r2(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, alpha: float = 0.1) -> float:
    """Leave-one-group-out kernel ridge R^2 (pooled over the columns of Y).

    X (N, F) features, Y (N, c) targets, groups (N,) integer group labels. The features are centred on the training
    rows and scaled by their root-mean-square row norm so that one penalty alpha is comparable across feature types
    and layers (linear kernel ridge = ridge regression on the raw features)."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    pred = np.zeros_like(Y)
    for g in np.unique(groups):
        te = groups == g
        if te.all() or not te.any():
            raise ValueError('block_cv_r2 needs at least two groups')
        Xt = X[~te]
        mu = Xt.mean(0)
        Xt = Xt - mu
        sc = np.sqrt((Xt ** 2).sum(1).mean()) + 1e-12
        Xt = Xt / sc
        ym = Y[~te].mean(0)
        K = Xt @ Xt.T
        A = np.linalg.solve(K + alpha * np.eye(len(Xt)), Y[~te] - ym)
        pred[te] = ((X[te] - mu) / sc) @ Xt.T @ A + ym
    return float(1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def clock_decodability(X: np.ndarray, x: np.ndarray, period: float = 100.0, block: float = 100.0,
                       alpha: float = 0.1) -> float:
    """Block cross-validated R^2 of cos / sin(2 pi x / period) from features X (N, F); folds are the bins
    floor(x / block). Values <= 0 mean that no single linear readout recovers the phase across hundreds."""
    x = np.asarray(x, float)
    return block_cv_r2(X, periodic_targets(x, period), np.floor(x / block).astype(int), alpha)


def ruler_decodability(X: np.ndarray, x: np.ndarray, block: float = 100.0, alpha: float = 0.1) -> float:
    """Block cross-validated R^2 of x itself (a monotone 'ruler' readout) with the same folds as the clock."""
    x = np.asarray(x, float)
    return block_cv_r2(X, (x / 1000.0)[:, None], np.floor(x / block).astype(int), alpha)


def margin_decomposition(truth: np.ndarray, logp0: np.ndarray, lo: float = 200, hi: float = 900) -> dict:
    """Decompose the 3.12 first-digit margin m = l0(k+1) - l0(k-1) (k = true first digit, truths lo-hi).

    Model: m_i = a + s * o_i + b_{k_i} + e_i with o = (x - 100 k) / 10 the in-bin offset (tens), b_k a per-hundreds bias
    (sum-to-zero), e a within-bin residual. Returns the slope s of the pooled fit (3.12 round 3 definition), the total
    residual s.d. of the pooled fit, the share of that residual variance explained by the per-hundreds bias, the
    within-bin residual s.d. after removing the bias (with the bias-model slope), and the two resolutions
    10 * sd / slope (units of x): 'resolution' (3.12 definition) and 'resolution_unbiased' (what the m channel would
    achieve if the hundreds readout were calibrated)."""
    truth = np.asarray(truth, float)
    sel = (truth >= lo) & (truth < hi)
    t, lp = truth[sel], np.asarray(logp0)[sel]
    k = np.round(t).astype(int) // 100
    idx = np.arange(len(t))
    m = lp[idx, k + 1] - lp[idx, k - 1]
    o = (t - 100 * k) / 10
    c = np.polyfit(o, m, 1)
    res = m - np.polyval(c, o)
    ks = np.unique(k)
    Xb = np.concatenate([o[:, None], (k[:, None] == ks[None]).astype(float)], 1)
    coef, *_ = np.linalg.lstsq(Xb, m, rcond=None)
    res_b = m - Xb @ coef
    var = float(res.var())
    sd_b = float(res_b.std(ddof=len(ks) + 1)) if len(t) > len(ks) + 1 else float('nan')
    bias = coef[1:] - coef[1:].mean()
    return {'slope': float(c[0]), 'sd': float(res.std(ddof=2)), 'bias_share': float(1 - res_b.var() / var) if var > 0 else 0.0,
            'sd_within': sd_b, 'slope_within': float(coef[0]),
            'resolution': float(10 * res.std(ddof=2) / c[0]) if c[0] > 0 else float('inf'),
            'resolution_unbiased': float(10 * sd_b / coef[0]) if coef[0] > 0 else float('inf'),
            'bias': {int(kk): float(b) for kk, b in zip(ks, bias)}, 'n': int(len(t))}


def legendre(x: np.ndarray, degree: int) -> np.ndarray:
    """(N, degree) Legendre polynomials of x / 500 - 1 (x in 0-1000 units)."""
    u = np.asarray(x, float) / 500.0 - 1.0
    return np.stack([np.polynomial.legendre.Legendre.basis(d)(u) for d in range(1, degree + 1)], 1)


def orthonormal_rows(V: np.ndarray) -> np.ndarray:
    """Orthonormal basis (rank, D) of the row space of V (k, D)."""
    _, S, Vt = np.linalg.svd(np.asarray(V, np.float64), full_matrices=False)
    r = int((S > S.max() * 1e-8).sum()) if len(S) else 0
    return Vt[:r]


def code_subspaces(H: np.ndarray, x: np.ndarray, periods: tuple[float, ...] = PERIODS, degree: int = 5
                   ) -> dict[str, np.ndarray]:
    """Least-squares subspaces of residuals H (N, D) for coordinate x: H ~ [Legendre(x), cos/sin(2 pi x / T)..., 1].

    Returns orthonormal row bases: 'periodic' (span of all cos / sin coefficient vectors, up to 2 len(periods) dims),
    'p100' (the period-100 pair only) and 'smooth' (span of the Legendre coefficient vectors). Transplanting the
    projection of a donor residual onto one of these subspaces moves exactly that part of the code."""
    x = np.asarray(x, float)
    cols = [legendre(x, degree)] + [periodic_targets(x, T) for T in periods] + [np.ones((len(x), 1))]
    X1 = np.concatenate(cols, 1)
    coef, *_ = np.linalg.lstsq(X1, np.asarray(H, np.float64), rcond=None)
    per = coef[degree:degree + 2 * len(periods)]
    i100 = list(periods).index(100.0) if 100.0 in periods else None
    out = {'periodic': orthonormal_rows(per), 'smooth': orthonormal_rows(coef[:degree])}
    if i100 is not None:
        out['p100'] = orthonormal_rows(per[2 * i100:2 * i100 + 2])
    return out


def project_replace(h: np.ndarray, donor: np.ndarray, P: np.ndarray) -> np.ndarray:
    """h with its component in the row space of the orthonormal basis P (k, D) replaced by the donor's:
    h + (donor - h) P^T P. h, donor: (..., D)."""
    return h + ((donor - h) @ P.T) @ P


def random_orthonormal(k: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """(k, D) random orthonormal rows (control subspace)."""
    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    return Q.T


def leverage_table(records: list[dict], key: str = 'fro') -> dict:
    """Average a list of per-batch leverage records {(layer, module, group): value} into a table
    {(layer, module, group): mean value}."""
    acc: dict = {}
    for r in records:
        for k, v in r[key].items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}
