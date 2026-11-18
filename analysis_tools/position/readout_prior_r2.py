"""Experiment 3.16 round 2: model-free helpers (unit-tested without a GPU).

Results: docs/analysis/position/results_2026-11-11_readout_prior_r2.md.

* Fourier spectrum of a coordinate representation: for each period T, the leave-one-hundred-out ridge R^2 of
  [cos(2 pi x / T), sin(2 pi x / T)] from features (readout_prior.clock_decodability with that period). Periods
  T >= 1000 describe a monotone or slowly bending code (less than one turn over the coordinate range); T = 100 is the
  period of the tens digit; T = 10 the period of the units digit.
* Digit decodability: a linear readout followed by arg max over the ten digits (one-hot ridge regression), evaluated
  with whole hundreds held out (the same weights must serve every hundred) or with interleaved folds.
* Simulated digit readouts of three codes (monotone, monotone minus a calibrated hundreds offset, period-100 circle)
  for the derivation figure.
* Donor-shift regression, Jacobian alignment, and the split of a representation change into a constant offset and a
  coordinate-dependent part.
"""
from __future__ import annotations

import numpy as np

from analysis_tools.position import readout_prior as rp

SPECTRUM_PERIODS = (1000.0, 500.0, 200.0, 100.0, 50.0, 100.0 / 3, 25.0, 20.0, 10.0)


def spectrum(X: np.ndarray, x: np.ndarray, periods: tuple[float, ...] = SPECTRUM_PERIODS, alpha: float = 0.1) -> np.ndarray:
    """(len(periods),) leave-one-hundred-out R^2 of cos / sin(2 pi x / T) from X (N, F) for every period T."""
    return np.array([rp.clock_decodability(X, x, period=T, alpha=alpha) for T in periods])


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float = 0.1):
    """Kernel ridge on centred, RMS-row-norm scaled features (the readout_prior convention); returns a predictor."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    mu = X.mean(0)
    Xc = X - mu
    sc = np.sqrt((Xc ** 2).sum(1).mean()) + 1e-12
    Xc = Xc / sc
    ym = Y.mean(0)
    A = np.linalg.solve(Xc @ Xc.T + alpha * np.eye(len(Xc)), Y - ym)
    return lambda Z: ((np.asarray(Z, np.float64) - mu) / sc) @ Xc.T @ A + ym


def r2(P: np.ndarray, Y: np.ndarray) -> float:
    """Pooled coefficient of determination over all columns."""
    return float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def digit_decoding(X: np.ndarray, x: np.ndarray, place: int = 1, block: bool = True, alpha: float = 0.1) -> float:
    """Accuracy of a linear readout + arg max predicting the digit at `place` (0 = hundreds, 1 = tens, 2 = units) of
    round(x). block=True: folds are whole hundreds (one readout for all hundreds); False: 5 interleaved folds."""
    n = np.round(np.asarray(x, float)).astype(int)
    y = {0: n // 100, 1: (n // 10) % 10, 2: n % 10}[place]
    groups = n // 100 if block else np.arange(len(n)) % 5
    Y = np.eye(10)[y]
    hits = np.zeros(len(n), bool)
    for g in np.unique(groups):
        te = groups == g
        f = ridge_fit(X[~te], Y[~te], alpha)
        hits[te] = f(X[te]).argmax(1) == y[te]
    return float(hits.mean())


def cross_decoding(Xa: np.ndarray, xa: np.ndarray, Xb: np.ndarray, xb: np.ndarray, period: float = 100.0,
                   alpha: float = 0.1) -> float:
    """R^2 on set b of a cos / sin(2 pi x / period) ridge readout fitted on set a (same feature space)."""
    f = ridge_fit(Xa, rp.periodic_targets(xa, period), alpha)
    return r2(f(Xb), rp.periodic_targets(xb, period))


# ----------------------------------------------------------------------------- simulated digit readouts

def simulate_codes(x: np.ndarray, noise: float = 0.0, bias_sd: float = 0.0, seed: int = 0) -> dict:
    """Second-digit predictions of three idealised codes read by a linear map + arg max (simulation).

    monotone:     h = x / 1000 (one dimension); the best linear readout of ten digits on a line lets each digit win
                  one interval only, here fitted by least squares on one-hot targets.
    conditional:  h = [o_hat] with o_hat = (x - 100 k) / 10 + c_k + noise, i.e. a monotone code from which the
                  hundreds have been subtracted with a per-hundred miscalibration c_k ~ N(0, bias_sd) (tens units);
                  the readout is floor(o_hat) clipped to 0-9.
    circle:       h = [cos(2 pi x / 100), sin(2 pi x / 100)] + noise; digit d has readout vector at angle
                  2 pi (d + 0.5) / 10, so the arg max is the 36-degree sector containing the angle.
    Returns {name: (N,) predicted second digit}."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = np.floor(x).astype(int)
    k = n // 100
    s = (n // 10) % 10
    out = {}
    h = (x / 1000.0)[:, None]
    f = ridge_fit(h, np.eye(10)[s], 1e-6)
    out['monotone'] = f(h).argmax(1)
    c = rng.normal(0, bias_sd, 10)
    o_hat = (x - 100 * k) / 10 + c[k] + rng.normal(0, noise, len(x))
    out['conditional'] = np.clip(np.floor(o_hat), 0, 9).astype(int)
    th = 2 * np.pi * x / 100 + rng.normal(0, noise * 2 * np.pi / 10, len(x))
    phi = 2 * np.pi * (np.arange(10) + 0.5) / 10
    logits = np.cos(th[:, None] - phi[None, :])
    out['circle'] = logits.argmax(1)
    return out


def argmax_changes(pred: np.ndarray) -> int:
    """Number of changes of the predicted digit along a sorted coordinate sweep."""
    return int((np.diff(pred) != 0).sum())


# ----------------------------------------------------------------------------- donor / Jacobian / change split

def donor_shift(truth: np.ndarray, donor: np.ndarray, logp1: np.ndarray) -> dict:
    """Least-squares fit E[s] = a + b_own o_own + b_donor o_donor (o = in-bin offset in tens) and the share of rows
    whose arg-max second digit equals the donor's tens digit or the own tens digit."""
    p = np.exp(logp1 - logp1.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    E = (p * np.arange(10)).sum(1)
    n, m = np.round(truth).astype(int), np.round(donor).astype(int)
    oo = (truth - 100 * (n // 100)) / 10
    od = (donor - 100 * (m // 100)) / 10
    A = np.stack([np.ones_like(oo), oo, od], 1)
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    am = p.argmax(1)
    return {'b_own': float(coef[1]), 'b_donor': float(coef[2]), 'argmax_donor': float((am == (m // 10) % 10).mean()),
            'argmax_own': float((am == (n // 10) % 10).mean()), 'n': int(len(n))}


def centred_rows(J: np.ndarray) -> np.ndarray:
    """(N, 10, D) digit-logit Jacobians -> (N, 10, D) with the mean over digits removed (softmax-invariant part)."""
    return J - J.mean(1, keepdims=True)


def jacobian_alignment(Ja: np.ndarray, Jb: np.ndarray, k: int = 9, seed: int = 0) -> dict:
    """Agreement of two sets of digit-logit Jacobians (N, 10, D) computed on matched inputs.

    matched_cos: mean over rows of the cosine between the flattened centred Jacobians of the same row;
    shuffled_cos: the same with rows of b permuted (baseline for 'same readout regardless of the coordinate');
    overlap: mean squared cosine of the principal angles between the top-k right singular subspaces of the stacked
    centred Jacobians (1 = identical readout subspace, k / D = random)."""
    A, B = centred_rows(Ja.astype(np.float64)), centred_rows(Jb.astype(np.float64))
    fa, fb = A.reshape(len(A), -1), B.reshape(len(B), -1)
    cos = (fa * fb).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1) + 1e-12)
    perm = np.random.default_rng(seed).permutation(len(B))
    cs = (fa * fb[perm]).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fb[perm], axis=1) + 1e-12)
    Va = np.linalg.svd(A.reshape(-1, A.shape[-1]), full_matrices=False)[2][:k]
    Vb = np.linalg.svd(B.reshape(-1, B.shape[-1]), full_matrices=False)[2][:k]
    sv = np.linalg.svd(Va @ Vb.T, compute_uv=False)
    return {'matched_cos': float(cos.mean()), 'shuffled_cos': float(cs.mean()), 'overlap': float((sv ** 2).mean()),
            'random_overlap': float(k / A.shape[-1])}


def change_split(Hb: np.ndarray, Ht: np.ndarray) -> dict:
    """Split the change Delta = Ht - Hb (N, D) at one site / layer into a constant offset (mean over inputs) and a
    coordinate-dependent remainder. Returns norms relative to the mean base norm: 'const' = |mean Delta| / |h|,
    'vary' = RMS over inputs of |Delta - mean Delta| / |h|."""
    D = Ht.astype(np.float64) - Hb.astype(np.float64)
    hn = np.linalg.norm(Hb.astype(np.float64), axis=1).mean()
    mu = D.mean(0)
    return {'const': float(np.linalg.norm(mu) / hn), 'vary': float(np.sqrt(((D - mu) ** 2).sum(1).mean()) / hn)}


def clock_plane(H: np.ndarray, x: np.ndarray, period: float = 100.0, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted clock coordinates: for every held-out hundred, the ridge readout of [cos, sin](2 pi x / period)
    fitted on the other hundreds is applied to the held-out rows (the readout_prior.block_cv_r2 protocol). Returns the
    held-out predictions P (N, 2) (the position of each input in the two-dimensional readout plane) and the in-plane
    angle atan2(P[:, 1], P[:, 0]). In-sample projections are not used: with N << D they fit any target."""
    x = np.asarray(x, float)
    Y = rp.periodic_targets(x, period)
    groups = np.floor(x / 100).astype(int)
    P = np.zeros_like(Y)
    for g in np.unique(groups):
        te = groups == g
        P[te] = ridge_fit(H[~te], Y[~te], alpha)(H[te])
    return P, np.arctan2(P[:, 1], P[:, 0])


def circular_slope(angle: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Slope (degrees per coordinate unit) and |r| of the unwrapped angle against x on a sorted sweep."""
    o = np.argsort(x)
    un = np.unwrap(angle[o])
    b = np.polyfit(x[o], un, 1)[0]
    r = np.corrcoef(x[o], un)[0, 1]
    return float(np.degrees(b)), float(abs(r))
