"""Experiment 3.12 CPU analysis (run after the GPU stages; no model weights are loaded).

Sections (keys of results.json):
  consistency   factorial arg-max digits vs the round-1 greedy answers of the same sweep images;
  landscape     first-digit and second-digit distributions along the sweeps (arrays for the figures);
  theory        toy readouts of a monotone code (digit_failure_theory);
  variance      decomposition of second-digit logits into image, forced-first-digit and context effects,
                plus Jensen-Shannon "swap" distances;
  clamp         second digit as a function of the offset truth - (first digit) * width, per family;
  evidence      prior / evidence decomposition, Gaussian-evidence fits (effective readout resolution)
                and the digit-sharpness law across families;
  decoding      emulated decoders (greedy, prior-removed, expected value) on the sweeps, RefCOCO and grids;
  dla           direct logit attribution of the second-digit logits (image vs first-digit driven parts)
                and of the first-digit prior;
  oracle        linear probes of the second digit at q1 (accuracy upper bound, low-rank versions);
  grid          column offsets of grid point answers and their decomposition;
  attention     attention mass at q0 / q1 on the image span and on the forced first digit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analysis_tools.position import digit_failure as df
from analysis_tools.position import digit_failure_theory as T
from analysis_tools.position.coordinate_readout_run import read_jsonl
from analysis_tools.position.coordinate_value_analysis import ridge_fit, ridge_predict
from analysis_tools.position.coordinate_value_r2_analysis import boot_ci, mean_ci, prop, sign_flip_p

WIDTH = {'sweep_h': 100, 'sweep_v': 100, 'ctx': 100, 'blank': 100, 'refcoco': 100, 'grid': 100, 'textconv': 100,
         'count': 10, 'textpos': 10}         # width (in the answer's units) of the first studied digit
COORD_FAMILIES = ('sweep_h', 'sweep_v', 'refcoco', 'grid', 'textconv')
# Below about 150 the model often writes two-digit values, which the three-digit decoding emulation cannot
# represent; decoding metrics therefore use truths in [DECODE_MIN, 1000).
DECODE_MIN = 150


# ----------------------------------------------------------------------------- loading

def softmax_digits(logp: np.ndarray) -> np.ndarray:
    """Renormalize digit log-probabilities over the ten digits (conditional on emitting a digit)."""
    p = np.exp(logp - logp.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def load(run: Path) -> dict:
    """Units of the factorial with their q0 / q1 distributions.

    Returns unit id -> {'family', 'axis', 'truth', 'ctx', 'ks', 'logp0' (10,), 'logp1' (K, 10), 'mass1' (K,),
    'rows' list of (shard, index) per k}.
    """
    rows = read_jsonl(run / 'factorial' / 'rows.jsonl')
    U: dict = {}
    for r in rows:
        u = U.setdefault(r['unit'], {'family': r['family'], 'axis': r['axis'], 'truth': r['truth'], 'ctx': r['ctx'],
                                     'ks': [], 'lp0': [], 'lp1': [], 'rows': []})
        u['ks'].append(r['k'])
        u['lp0'].append(r['logp'][0])
        u['lp1'].append(r['logp'][1])
        u['rows'].append((r['shard'], r['index']))
    for u in U.values():
        o = np.argsort(u['ks'])
        u['ks'] = [u['ks'][i] for i in o]
        u['rows'] = [u['rows'][i] for i in o]
        lp1 = np.array(u['lp1'])[o]
        u['logp0'] = np.array(u['lp0']).mean(0)             # q0 precedes k; rows differ only by bf16 batching
        u['lp0_spread'] = float(np.abs(np.array(u['lp0']) - u['logp0']).max())
        u['mass1'] = np.exp(lp1).sum(-1)
        u['logp1'] = np.log(softmax_digits(lp1))
        del u['lp0'], u['lp1']
    return U


def family(U: dict, name: str) -> list[str]:
    ids = [i for i, u in U.items() if u['family'] == name]
    return sorted(ids, key=lambda i: (U[i]['truth'] if np.isfinite(U[i]['truth']) else -1, i))


def load_shard_arrays(feat: Path, keys: tuple[str, ...]) -> dict:
    """shard index -> {key: array} for the factorial feature shards (lazy loading by caller)."""
    out = {}
    for p in sorted((feat / 'factorial').glob('shard_*.npz')):
        z = np.load(p)
        out[int(p.stem.split('_')[1])] = {k: z[k] for k in keys}
    return out


def gather(U: dict, ids: list[str], shards: dict, key: str, ks: tuple[int, ...] | None = None) -> np.ndarray:
    """Stack a per-row feature for the given units; returns (len(ids), K, ...)."""
    out = []
    for i in ids:
        u = U[i]
        sel = [j for j, k in enumerate(u['ks']) if ks is None or k in ks]
        out.append(np.stack([shards[u['rows'][j][0]][key][u['rows'][j][1]] for j in sel]))
    return np.stack(out)


def digits_of(v: float, width: int) -> tuple[int, int]:
    """(first studied digit, second digit) of value v when the first digit has the given width."""
    n = int(round(v))
    return n // width, (n // (width // 10)) % 10


# ----------------------------------------------------------------------------- statistics

def js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits along the last axis."""
    m = 0.5 * (p + q)
    def kl(a, b):
        return (a * (np.log2(a + 1e-12) - np.log2(b + 1e-12))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def entropy_bits(p: np.ndarray) -> np.ndarray:
    return -(p * np.log2(p + 1e-12)).sum(-1)


def anova2(L: np.ndarray) -> dict:
    """Two-way decomposition of logits L (I, K, 10), centered over digits, into image / k / interaction."""
    L = L - L.mean(-1, keepdims=True)
    g = L.mean((0, 1), keepdims=True)
    a = L.mean(1, keepdims=True) - g          # image main effect
    b = L.mean(0, keepdims=True) - g          # first-digit main effect
    ss = lambda x: float((x ** 2).sum())
    tot = ss(L - g)
    I, K = L.shape[:2]
    res = L - g - a - b
    return {'total': tot, 'image': ss(a) * K / tot, 'k': ss(b) * I / tot, 'interaction': ss(res) / tot}


# ----------------------------------------------------------------------------- consistency

def consistency(U: dict, round1: Path) -> dict:
    recs = {r['id']: r for r in read_jsonl(round1 / 'gen' / 'records.jsonl')}
    out = {}
    for fam in ('sweep_h', 'sweep_v'):
        if not family(U, fam):
            continue
        same1 = same2 = n2 = 0
        ids = family(U, fam)
        for i in ids:
            u = U[i]
            v = recs[i]['score']['coords'][u['axis']]
            s = str(int(v))
            k = int(np.argmax(u['logp0'][1:]) + 1)
            same1 += int(s[0]) == k
            if len(s) == 3:
                n2 += 1
                t = int(np.argmax(u['logp1'][u['ks'].index(k)]))
                same2 += (int(s[0]), int(s[1])) == (k, t)
        out[fam] = {'first': prop(same1, len(ids)), 'first_two': prop(same2, n2),
                    'lp0_spread_max': max(U[i]['lp0_spread'] for i in ids)}
    return out


# ----------------------------------------------------------------------------- landscape

def landscape(U: dict) -> tuple[dict, dict]:
    R, A = {}, {}
    for fam in ('sweep_h', 'sweep_v', 'textconv', 'count', 'textpos'):
        if not family(U, fam):
            continue
        ids = family(U, fam)
        w = WIDTH[fam]
        t = np.array([U[i]['truth'] for i in ids])
        P0 = np.exp(np.stack([U[i]['logp0'] for i in ids]))
        P1all = np.exp(np.stack([U[i]['logp1'] for i in ids]))            # (N, K, 10)
        ks = U[ids[0]]['ks']
        first = np.array([digits_of(v, w)[0] for v in t])
        ok = np.isin(first, ks)
        P1true = np.full((len(ids), 10), np.nan)
        P1true[ok] = P1all[ok, [ks.index(f) for f in first[ok]]]
        A[f'{fam}_truth'], A[f'{fam}_P0'], A[f'{fam}_P1true'], A[f'{fam}_P1all'] = t, P0, P1true, P1all
        sec = np.array([digits_of(v, w)[1] for v in t])
        ptrue = P1true[ok, sec[ok]]
        R[fam] = {'n': len(ids), 'n_in_range': int(ok.sum()),
                  'p_true_first': float(P0[np.arange(len(ids)), np.clip(first, 0, 9)].mean()),
                  'p_true_second': float(ptrue.mean()),
                  'second_entropy_bits': float(entropy_bits(P1true[ok]).mean())}
    return R, A


# ----------------------------------------------------------------------------- variance and swaps

def variance(U: dict) -> dict:
    out = {}
    for fam in ('sweep_h', 'sweep_v', 'refcoco', 'grid', 'count', 'textconv', 'textpos'):
        if not family(U, fam):
            continue
        ids = family(U, fam)
        L = np.stack([U[i]['logp1'] for i in ids])
        P = np.exp(L)
        dec = anova2(L)
        I, K = P.shape[:2]
        rng = np.random.default_rng(0)
        a, b = rng.integers(0, I, 4000), rng.integers(0, I, 4000)
        k = rng.integers(0, K, 4000)
        k2 = rng.integers(0, K, 4000)
        keep = a != b
        img_swap = js(P[a[keep], k[keep]], P[b[keep], k[keep]])
        keep2 = k != k2
        k_swap = js(P[a[keep2], k[keep2]], P[a[keep2], k2[keep2]])
        out[fam] = {**dec, 'js_image_swap': mean_ci(img_swap), 'js_k_swap': mean_ci(k_swap), 'n_units': I, 'n_k': K}
    # Context: same image, x value in the prefix 200 / 500 / 800, all k.
    src = defaultdict(dict)
    for i, u in U.items():
        if u['family'] == 'ctx':
            src[i.split('_', 1)[1]][u['ctx']] = u
        elif u['family'] == 'sweep_v':
            src[i][500] = u
    full = [s for s in src.values() if len(s) == 3]
    L = np.stack([[s[c]['logp1'] for c in (200, 500, 800)] for s in full])        # (I, 3, K, 10)
    P = np.exp(L)
    Lc = L - L.mean(-1, keepdims=True)
    g = Lc.mean((0, 1, 2), keepdims=True)
    ss = lambda x: float((x ** 2).sum())
    tot = ss(Lc - g)
    I, C, K = Lc.shape[:3]
    out['ctx'] = {'image': ss(Lc.mean((1, 2), keepdims=True) - g) * C * K / tot,
                  'context': ss(Lc.mean((0, 2), keepdims=True) - g) * I * K / tot,
                  'k': ss(Lc.mean((0, 1), keepdims=True) - g) * I * C / tot,
                  'js_ctx_swap': mean_ci(np.concatenate([js(P[:, 0], P[:, 1]).ravel(), js(P[:, 1], P[:, 2]).ravel()])),
                  'n_images': I}
    return out


# ----------------------------------------------------------------------------- clamp

def clamp(U: dict) -> tuple[dict, dict]:
    """Second digit vs offset o = truth - k * width for every forced k (in-range and out-of-range)."""
    R, A = {}, {}
    for fam in ('sweep_h', 'sweep_v', 'refcoco', 'grid', 'textconv', 'count', 'textpos'):
        if not family(U, fam):
            continue
        ids = family(U, fam)
        w = WIDTH[fam]
        o, E, p9, p0, ideal, same_true = [], [], [], [], [], []
        for i in ids:
            u = U[i]
            sec_true = digits_of(u['truth'], w)[1]
            for j, k in enumerate(u['ks']):
                p = np.exp(u['logp1'][j])
                off = (u['truth'] - k * w) / (w / 10)                     # offset in second-digit units
                o.append(off)
                E.append(float((p * np.arange(10)).sum()))
                p9.append(p[9])
                p0.append(p[0])
                ideal.append(float(np.clip(np.floor(off), 0, 9)))
                same_true.append(p[sec_true])
        o, E, p9, p0, ideal, same_true = map(np.array, (o, E, p9, p0, ideal, same_true))
        A[f'{fam}_offset'], A[f'{fam}_E'], A[f'{fam}_p9'], A[f'{fam}_p0'] = o, E, p9, p0
        below, above = o < -1, o > 11                                      # k too large / too small
        inr = (o >= 0) & (o < 10)
        R[fam] = {'p9_when_k_too_small': float(p9[above].mean()) if above.any() else None,
                  'p0_when_k_too_large': float(p0[below].mean()) if below.any() else None,
                  'p_true_second_when_k_wrong': float(same_true[~inr].mean()) if (~inr).any() else None,
                  'corr_E_offset_in_range': float(np.corrcoef(E[inr], o[inr])[0, 1]) if inr.sum() > 2 else None,
                  # Ideal magnitude-conditional slope is 1 second digit per second-digit unit of offset.
                  'slope_E_offset_in_range': float(np.polyfit(o[inr], E[inr], 1)[0]) if inr.sum() > 2 else None,
                  'E_at_offset_minus5_to_0': float(E[(o >= -5) & (o < 0)].mean()) if ((o >= -5) & (o < 0)).any() else None,
                  'E_at_offset_10_to_15': float(E[(o >= 10) & (o < 15)].mean()) if ((o >= 10) & (o < 15)).any() else None,
                  'E_far_below_minus20': float(E[o < -20].mean()) if (o < -20).any() else None,
                  'E_far_above_30': float(E[o > 30].mean()) if (o > 30).any() else None,
                  'n_rows': int(len(o))}
    return R, A


# ----------------------------------------------------------------------------- evidence and resolution

def gaussian_bin_probs(x: np.ndarray, edges: np.ndarray, sigma: float) -> np.ndarray:
    """P(v in [edges[c], edges[c+1])) for v ~ N(x, sigma), shape (N, C)."""
    import torch
    z = (edges[None, :] - x[:, None]) / (np.sqrt(2) * sigma)
    cdf = 0.5 * (1 + torch.erf(torch.from_numpy(np.asarray(z, np.float64))).numpy())
    return np.clip(np.diff(cdf, axis=1), 1e-12, None)


def fit_evidence(P: np.ndarray, x: np.ndarray, edges: np.ndarray, prior: np.ndarray,
                 sigmas=np.geomspace(2, 3000, 60), betas=np.linspace(0, 1.5, 16), shifts=np.linspace(-60, 60, 25)) -> dict:
    """Fit P(c | x) ∝ prior(c)^beta * P_gauss(c | x + shift, sigma) to observed distributions P (N, C).

    prior (C,) or (N, C). Returns the grid optimum of the mean KL(P || fit) and the KL of the prior-only fit.
    """
    best = None
    lp = np.log(np.clip(prior, 1e-12, None))
    if lp.ndim == 1:
        lp = np.broadcast_to(lp, P.shape)
    for s in sigmas:
        for m in shifts:
            G = np.log(gaussian_bin_probs(x + m, edges, s))
            for b in betas:
                lf = G + b * lp
                lf = lf - np.log(np.exp(lf - lf.max(1, keepdims=True)).sum(1, keepdims=True)) - lf.max(1, keepdims=True)
                kl = float((P * (np.log(P + 1e-12) - lf)).sum(1).mean())
                if best is None or kl < best['kl']:
                    best = {'kl': kl, 'sigma': float(s), 'beta': float(b), 'shift': float(m)}
    flat = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    best['kl_prior_only'] = float((P * (np.log(P + 1e-12) - flat)).sum(1).mean())
    return best


def logistic_fit(x: np.ndarray, y: np.ndarray, centres: np.ndarray, widths=np.geomspace(1, 400, 60)) -> tuple[float, float]:
    """Least-squares fit of y = 1 / (1 + exp(-(x - c) / s)); returns (c, s)."""
    best = (None, None, np.inf)
    for sw in widths:
        z = 1 / (1 + np.exp(-(x[None] - centres[:, None]) / sw))
        err = ((z - y[None]) ** 2).sum(1)
        j = int(err.argmin())
        if err[j] < best[2]:
            best = (float(centres[j]), float(sw), float(err[j]))
    return best[0], best[1]


def boundaries(U: dict) -> tuple[dict, dict]:
    """First-digit band boundaries along the sweeps: location shift and transition width per boundary,
    for the raw distribution and after blank-canvas calibration (prior removed)."""
    R, A = {}, {}
    for fam in ('sweep_h', 'sweep_v'):
        if not family(U, fam):
            continue
        ids = [i for i in family(U, fam) if 100 <= U[i]['truth'] < 1000]
        x = np.array([U[i]['truth'] for i in ids])
        b = U['blank_x' if U[ids[0]]['axis'] == 0 else 'blank_y']
        out = {}
        for name, prior in (('raw', None), ('calibrated', b['logp0'])):
            L = np.stack([U[i]['logp0'] for i in ids])[:, 1:]
            if prior is not None:
                L = L - prior[1:]
            P = softmax_digits(np.concatenate([np.full((len(x), 1), -1e9), L], 1))[:, 1:]   # digits 1..9
            rows = []
            for k in range(2, 10):
                y = P[:, k - 1:].sum(1)                                              # P(first digit >= k)
                c, sw = logistic_fit(x, y, np.linspace(100 * k - 150, 100 * k + 150, 121))
                rows.append({'boundary': 100 * k, 'location': c, 'shift': c - 100 * k, 'width': sw})
            out[name] = rows
            A[f'{fam}_bound_{name}_shift'] = np.array([r['shift'] for r in rows])
            A[f'{fam}_bound_{name}_width'] = np.array([r['width'] for r in rows])
        R[fam] = out
    return R, A


def sharpness_curve(ratio: np.ndarray) -> np.ndarray:
    """Expected P(true digit) under Gaussian evidence of width sigma for a digit of width w (ratio = w / sigma),
    with the true value uniform inside its bin and no prior."""
    from math import erf, sqrt
    u = (np.arange(200) + 0.5) / 200
    out = []
    for r in np.atleast_1d(ratio):
        a = (1 - u) * r / sqrt(2)
        b = -u * r / sqrt(2)
        out.append(np.mean([0.5 * (erf(x) - erf(y)) for x, y in zip(a, b)]))
    return np.array(out)


def evidence(U: dict) -> tuple[dict, dict]:
    R, A = {}, {}
    blank = {0: U.get('blank_x'), 1: U.get('blank_y')}
    for fam in ('sweep_h', 'sweep_v', 'refcoco', 'grid', 'textconv', 'count', 'textpos'):
        if not family(U, fam):
            continue
        ids = family(U, fam)
        w = WIDTH[fam]
        x = np.array([U[i]['truth'] for i in ids])
        lo = w                                                       # restrict to numbers with both digits present
        hi = 10 * w
        keep = (x >= lo) & (x < hi)
        ids = [i for i, k in zip(ids, keep) if k]
        x = x[keep]
        ks = U[ids[0]]['ks']
        P0 = np.exp(np.stack([U[i]['logp0'] for i in ids]))[:, ks]
        P0 = P0 / P0.sum(1, keepdims=True)
        marg0 = P0.mean(0)
        e0 = np.array([k * w for k in ks] + [(ks[-1] + 1) * w], float)
        f0 = fit_evidence(P0, x, e0, marg0)
        first = np.array([digits_of(v, w)[0] for v in x])
        ok = np.isin(first, ks)
        P1 = np.exp(np.stack([U[i]['logp1'][ks.index(f)] for i, f in zip(np.array(ids)[ok], first[ok])]))
        marg1 = np.stack([np.exp(np.stack([U[i]['logp1'][ks.index(f)] for i in ids])).mean(0) for f in first[ok]])
        x1 = x[ok] - first[ok] * w                                   # offset inside the first-digit bin
        e1 = np.arange(11) * (w / 10)
        f1 = fit_evidence(P1, x1, e1.astype(float), marg1, shifts=np.linspace(-6, 6, 7) * w / 100)
        sec = np.array([digits_of(v, w)[1] for v in x[ok]])
        R[fam] = {'first': {**f0, 'width': w, 'p_true': float(P0[np.arange(len(x)), np.searchsorted(ks, first)].mean())
                            if np.isin(first, ks).all() else None},
                  'second': {**f1, 'width': w / 10, 'p_true': float(P1[np.arange(len(x1)), sec].mean())},
                  'n': len(ids)}
        if fam in ('sweep_h', 'sweep_v') and blank[U[ids[0]]['axis']] is not None:
            b = blank[U[ids[0]]['axis']]
            A[f'{fam}_E0'] = np.stack([U[i]['logp0'] for i in ids]) - b['logp0']
            A[f'{fam}_E0_truth'] = x
            A[f'{fam}_prior0'] = b['logp0']
            A[f'{fam}_prior1'] = b['logp1']
    ratio = np.geomspace(0.05, 20, 80)
    A['law_ratio'], A['law_p'] = ratio, sharpness_curve(ratio)
    return R, A


# ----------------------------------------------------------------------------- decoding

def decode(U: dict, ids: list[str], prior0: np.ndarray | None, prior1: np.ndarray | None, alpha: float, mode: str) -> np.ndarray:
    """Emulated three-digit decoding at 10-unit resolution; value = 100 k + 10 t + 5.

    mode 'argmax': greedy on (log p - alpha * log prior) for both digits;
    mode 'mean':   expected value under P(k) P(t | k) (prior-adjusted with the same alpha).
    """
    out = []
    for i in ids:
        u = U[i]
        ks = np.array(u['ks'])
        l0 = u['logp0'][ks] - (alpha * prior0[ks] if prior0 is not None else 0)
        l1 = u['logp1'] - (alpha * prior1 if prior1 is not None else 0)
        if mode == 'argmax':
            j = int(np.argmax(l0))
            t = int(np.argmax(l1[j]))
            out.append(100 * ks[j] + 10 * t + 5)
        else:
            p0 = softmax_digits(l0[None])[0] if len(ks) == 10 else np.exp(l0 - l0.max()) / np.exp(l0 - l0.max()).sum()
            p1 = np.exp(l1 - l1.max(1, keepdims=True))
            p1 = p1 / p1.sum(1, keepdims=True)
            out.append(float((p0[:, None] * p1 * (100 * ks[:, None] + 10 * np.arange(10)[None] + 5)).sum()))
    return np.array(out, float)


def decode_metrics(v: np.ndarray, x: np.ndarray) -> dict:
    err = np.abs(v - x)
    lo = (x >= DECODE_MIN) & (x < 380)
    mid = (x >= 400) & (x < 610)
    return {'mae': mean_ci(err), 'first_acc': float((v // 100 == x // 100).mean()),
            'slope_below_380': float(np.polyfit(x[lo], v[lo], 1)[0]) if lo.sum() > 3 else None,
            'plateau_frac_480_520': float(((v >= 480) & (v <= 520))[mid].mean()) if mid.any() else None,
            'truth_frac_480_520': float(((x >= 480) & (x <= 520))[mid].mean()) if mid.any() else None,
            'tens_0_or_5': float(np.isin((np.round(v) // 10) % 10, (0, 5)).mean())}


def decoding(U: dict) -> tuple[dict, dict]:
    R, A = {}, {}
    for fam in ('sweep_h', 'sweep_v', 'refcoco', 'grid'):
        if not family(U, fam):
            continue
        all_ids = family(U, fam)
        ids = [i for i in all_ids if DECODE_MIN <= U[i]['truth'] < 1000]
        x = np.array([U[i]['truth'] for i in ids])
        axis = U[ids[0]]['axis']
        b = U['blank_x' if axis == 0 else 'blank_y']
        marg0 = np.log(np.exp(np.stack([U[i]['logp0'] for i in ids])).mean(0))
        marg1 = np.log(np.exp(np.stack([U[i]['logp1'] for i in ids])).mean(0))
        variants = {'greedy': (None, None, 0.0, 'argmax'),
                    'blank_calibrated': (b['logp0'], b['logp1'], 1.0, 'argmax'),
                    'blank_calibrated_half': (b['logp0'], b['logp1'], 0.5, 'argmax'),
                    'marginal_calibrated': (marg0, marg1, 1.0, 'argmax'),
                    'expected': (None, None, 0.0, 'mean'),
                    'blank_calibrated_expected': (b['logp0'], b['logp1'], 1.0, 'mean'),
                    'calibrated_first_only': (b['logp0'], None, 1.0, 'argmax')}
        R[fam] = {'n': len(ids), 'n_excluded_below_min': sum(U[i]['truth'] < DECODE_MIN for i in all_ids)}
        base = None
        for name, (p0, p1, a, mode) in variants.items():
            v = decode(U, ids, p0, p1, a, mode)
            m = decode_metrics(v, x)
            if base is None:
                base = np.abs(v - x)
            else:
                d = np.abs(v - x) - base
                m['mae_diff_vs_greedy'] = mean_ci(d)
                m['p_sign_flip'] = sign_flip_p(d)
            R[fam][name] = m
            A[f'{fam}_{name}'] = v
        A[f'{fam}_dec_truth'] = x
    return R, A


# ----------------------------------------------------------------------------- DLA

def dla(U: dict, shards: dict) -> tuple[dict, dict]:
    """Direct attribution of second-digit (q1) and first-digit (q0) logits on the sweeps."""
    R, A = {}, {}
    for fam in ('sweep_h', 'sweep_v'):
        if not family(U, fam):
            continue
        ids = [i for i in family(U, fam) if 100 <= U[i]['truth'] < 1000]
        comp = np.concatenate([gather(U, ids, shards, 'dla_mixer'), gather(U, ids, shards, 'dla_mlp'),
                               gather(U, ids, shards, 'dla_embed')[:, :, :, None]], 3)    # (I, K, 2, 65, 10)
        names = [f'mixer{l}' for l in range(1, 33)] + [f'mlp{l}' for l in range(1, 33)] + ['embed']
        heads = gather(U, ids, shards, 'dla_head')                                          # (I, K, 2, 8, 16, 10)
        x = np.array([U[i]['truth'] for i in ids])
        ks = U[ids[0]]['ks']
        out = {}
        for role, r in (('q0', 0), ('q1', 1)):
            C = comp[:, :, r] - comp[:, :, r].mean(-1, keepdims=True)                      # (I, K, 65, 10)
            if role == 'q0':
                C = C[:, :1]                                                                # q0 is k-independent
            tot = C.sum(2)
            img = C - C.mean(0, keepdims=True)                                              # image-driven part
            kpart = C.mean(0, keepdims=True) - C.mean((0, 1), keepdims=True)                # k-driven part
            const = C.mean((0, 1))                                                          # (65, 10)
            v_img = (img ** 2).sum(-1).mean((0, 1))                                         # per component
            v_k = (kpart ** 2).sum(-1).mean((0, 1))
            v_tot_img = float(((tot - tot.mean(0, keepdims=True)) ** 2).sum(-1).mean())
            out[role] = {'image_var': dict(zip(names, v_img.round(4).tolist())),
                         'k_var': dict(zip(names, v_k.round(4).tolist())),
                         'sum_image_var': v_tot_img, 'sum_of_component_image_var': float(v_img.sum())}
            A[f'{fam}_{role}_image_var'], A[f'{fam}_{role}_k_var'], A[f'{fam}_{role}_const'] = v_img, v_k, const
            # Image-driven part correlated with the true digit: project on the true digit's one-hot.
            w = 100 if role == 'q0' else 10
            true_d = (np.round(x) // w) % 10
            onehot = np.eye(10)[true_d.astype(int)] - 0.1
            if role == 'q1':
                inr = np.array([ks.index(int(v // 100)) if int(v // 100) in ks else -1 for v in x])
                Cin = C[np.arange(len(x)), inr] - C[:, :, :, :].mean((0, 1))[None]
            else:
                Cin = C[:, 0] - C[:, 0].mean(0, keepdims=True)
            tuned = (Cin * onehot[:, None, :]).sum(-1).mean(0)                              # (65,) nats toward truth
            out[role]['tuned_toward_truth'] = dict(zip(names, tuned.round(4).tolist()))
            A[f'{fam}_{role}_tuned'] = tuned
            H = heads[:, :, r] - heads[:, :, r].mean(-1, keepdims=True)                     # (I, K, 8, 16, 10)
            A[f'{fam}_{role}_head_k_var'] = ((H.mean(0, keepdims=True) - H.mean((0, 1), keepdims=True)) ** 2).sum(-1).mean((0, 1))
            A[f'{fam}_{role}_head_image_var'] = ((H - H.mean(0, keepdims=True)) ** 2).sum(-1).mean((0, 1))
        # Who writes the centre prior at q1: contribution to the prior digit after k = 4 (digit 9) and k = 5 (digit 0).
        C1 = comp[:, :, 1] - comp[:, :, 1].mean(-1, keepdims=True)
        H1 = heads[:, :, 1] - heads[:, :, 1].mean(-1, keepdims=True)
        prior_writers = {}
        for k, d in ((4, 9), (5, 0)):
            j = ks.index(k)
            others = [jj for jj in range(len(ks)) if jj != j]
            c = C1[:, j, :, d].mean(0) - C1[:, others, :, d].mean((0, 1))                  # (65,)
            h = H1[:, j, :, :, d].mean(0) - H1[:, others, :, :, d].mean((0, 1))             # (8, 16)
            top = np.argsort(-c)[:8]
            th = np.dstack(np.unravel_index(np.argsort(-h.ravel())[:6], h.shape))[0]
            prior_writers[f'k{k}_d{d}'] = {'components': {names[t]: round(float(c[t]), 3) for t in top},
                                           'heads': {f'L{df.FULL_LAYERS[a]}H{b}': round(float(h[a, b]), 3) for a, b in th},
                                           'total': round(float(c.sum()), 3)}
            A[f'{fam}_prior_k{k}_d{d}'] = c
            A[f'{fam}_prior_heads_k{k}_d{d}'] = h
        out['prior_writers'] = prior_writers
        A['dla_names'] = np.array(names)
        R[fam] = out
    return R, A


# ----------------------------------------------------------------------------- oracle probes

def softmax_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: int = 10, folds: int = 5, l2: float = 1e-2) -> np.ndarray:
    """Cross-validated multinomial logistic predictions (folds over groups)."""
    pred = np.zeros(len(y), int)
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    order = rng.permutation(ug)
    for f in range(folds):
        test_g = order[f::folds]
        te = np.isin(groups, test_g)
        m = T.fit_softmax(X[~te], y[~te], C, l2=l2, iters=200)
        pred[te] = T.predict_logits(m, X[te]).argmax(1)
    return pred


def pca_project(X: np.ndarray, r: int, train: np.ndarray) -> np.ndarray:
    mu = X[train].mean(0)
    _, _, Vt = np.linalg.svd(X[train] - mu, full_matrices=False)
    return (X - mu) @ Vt[:r].T


def oracle(U: dict, shards: dict) -> dict:
    """Linear read-out of the second digit at q1 (rows with the true first digit), per layer and rank."""
    out = {}
    for fam in ('sweep_h', 'sweep_v', 'textconv', 'count'):
        if not family(U, fam):
            continue
        w = WIDTH[fam]
        ids = [i for i in family(U, fam) if w <= U[i]['truth'] < 10 * w]
        x = np.array([U[i]['truth'] for i in ids])
        first = np.array([digits_of(v, w)[0] for v in x])
        sec = np.array([digits_of(v, w)[1] for v in x])
        ks = U[ids[0]]['ks']
        keep = np.isin(first, ks)
        R = np.stack([shards[U[i]['rows'][ks.index(f)][0]]['resid'][U[i]['rows'][ks.index(f)][1], 1]
                      for i, f in zip(np.array(ids)[keep], first[keep])]).astype(np.float32)     # (N, layers, D)
        y = sec[keep]
        groups = np.arange(len(y))
        res = {}
        model_pred = np.array([np.argmax(U[i]['logp1'][ks.index(f)]) for i, f in zip(np.array(ids)[keep], first[keep])])
        res['model'] = {'acc': float((model_pred == y).mean()),
                        'within1': float((np.minimum(np.abs(model_pred - y), 10 - np.abs(model_pred - y)) <= 1).mean())}
        for li, layer in enumerate(df.RESID_LAYERS):
            X = R[:, li]
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
            row = {}
            for r in (2, 8, 32):
                Z = pca_project(Xs, r, np.ones(len(y), bool))
                p = softmax_cv(Z, y, groups)
                d = np.abs(p - y)
                row[f'pc{r}'] = {'acc': float((p == y).mean()), 'within1': float((np.minimum(d, 10 - d) <= 1).mean())}
            # Full-dimensional ridge on the coordinate, converted to the second digit.
            pv = np.zeros(len(y))
            order = np.random.default_rng(0).permutation(len(y))
            for f in range(5):
                te = np.zeros(len(y), bool)
                te[order[f::5]] = True
                m = ridge_fit(X[~te], x[keep][~te, None])
                pv[te] = ridge_predict(m, X[te])[:, 0]
            pd = (np.round(pv).astype(int) // (w // 10)) % 10
            d = np.abs(pd - y)
            row['ridge_coordinate'] = {'acc': float((pd == y).mean()), 'within1': float((np.minimum(d, 10 - d) <= 1).mean()),
                                       'mae': float(np.abs(pv - x[keep]).mean())}
            res[f'L{layer}'] = row
        res['n'] = int(len(y))
        out[fam] = res
    return out


# ----------------------------------------------------------------------------- grids

def grid(run: Path, U: dict, shards: dict) -> tuple[dict, dict]:
    gen = read_jsonl(run / 'grid' / 'gen.jsonl')
    R, A = {}, {}
    for n in df.GRID_SIZES:
        rs = [g for g in gen if g['n'] == n and g['coords']]
        tx = np.array([g['truth']['center_units'][0] for g in rs])
        ox = np.array([g['coords'][0] for g in rs], float)
        edges = np.array(rs[0]['truth']['col_edges_units'])
        cell = edges[1] - edges[0]
        col_true = np.array([g['col'] for g in rs])
        col_out = np.clip(np.searchsorted(edges, ox, side='right') - 1, -1, n)
        d_edge = np.min(np.abs(ox[:, None] - edges[None]), 1) / cell
        centres = (edges[:-1] + edges[1:]) / 2
        d_centre = np.min(np.abs(ox[:, None] - centres[None]), 1) / cell
        dcol = col_out - col_true
        R[n] = {'n': len(rs), 'valid': prop(len(rs), sum(g['n'] == n for g in gen)),
                'col_plus1': prop(int((dcol == 1).sum()), len(rs)), 'col_minus1': prop(int((dcol == -1).sum()), len(rs)),
                'col_exact': prop(int((dcol == 0).sum()), len(rs)), 'mean_err_cells': mean_ci((ox - tx) / cell),
                'median_dist_to_edge_cells': float(np.median(d_edge)), 'median_dist_to_centre_cells': float(np.median(d_centre)),
                'frac_closer_to_edge': float((d_edge < d_centre).mean()), 'cell_units': float(cell)}
        A[f'grid{n}_tx'], A[f'grid{n}_ox'], A[f'grid{n}_edges'] = tx, ox, edges
        # Factorial view of the same images: first digit arg-max with and without calibration.
        ids = [g['id'] for g in rs if g['id'] in U]
        if ids:
            b = U['blank_x']
            k_raw = np.array([np.argmax(U[i]['logp0'][1:]) + 1 for i in ids])
            k_cal = np.array([np.argmax((U[i]['logp0'] - b['logp0'])[1:]) + 1 for i in ids])
            t = np.array([U[i]['truth'] for i in ids])
            R[n]['first_acc_raw'] = float((k_raw == (t // 100)).mean())
            R[n]['first_acc_calibrated'] = float((k_cal == (t // 100)).mean())
    return R, A


# ----------------------------------------------------------------------------- attention

def attention(run: Path) -> tuple[dict, dict]:
    path = run / 'attn' / 'attn.jsonl'
    if not path.exists():
        return {}, {}
    rec = read_jsonl(path)
    R, A = {}, {}
    for fam in sorted({r['family'] for r in rec}):
        rs = [r for r in rec if r['family'] == fam]
        img = np.array([r['image'] for r in rs])          # (N, layers, 2, heads)
        dig = np.array([r['digit'] for r in rs])
        R[fam] = {'n': len(rs), 'image_mass_q0': img[:, :, 0].mean((0, 2)).round(3).tolist(),
                  'image_mass_q1': img[:, :, 1].mean((0, 2)).round(3).tolist(),
                  'digit_mass_q1': dig[:, :, 1].mean((0, 2)).round(3).tolist(),
                  'max_check': float(np.max([r['check'] for r in rs]))}
        A[f'{fam}_img_q0'], A[f'{fam}_img_q1'], A[f'{fam}_dig_q1'] = img[:, :, 0].mean(0), img[:, :, 1].mean(0), dig[:, :, 1].mean(0)
    return R, A


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--round2', required=True)
    p.add_argument('--features', required=True)
    args = p.parse_args()
    run = Path(args.run)
    U = load(run)
    results, arrays = {}, {}

    def put(name, pair):
        R, A = pair
        results[name] = R
        arrays.update(A)
        print(name, 'done', flush=True)

    results['consistency'] = consistency(U, Path(args.round1))
    put('landscape', landscape(U))
    results['theory'] = {f'sigma{s}': T.toy_experiment(s, n=8000) for s in (5.0, 50.0)}
    demo = T.envelope_demo(1001)
    arrays.update({'toy_x': demo['x'], 'toy_hundreds': demo['hundreds'], 'toy_tens': demo['tens']})
    for k in ('hundreds_z', 'tens_z', 'tens_zk'):
        arrays[f'toy_{k}_logits'] = demo[k]['logits']
        results['theory'][f'demo_{k}_acc'] = demo[k]['acc']
    results['variance'] = variance(U)
    put('clamp', clamp(U))
    put('evidence', evidence(U))
    put('boundaries', boundaries(U))
    put('decoding', decoding(U))
    shards = load_shard_arrays(Path(args.features), ('resid', 'dla_mixer', 'dla_mlp', 'dla_head', 'dla_embed'))
    put('dla', dla(U, shards))
    results['oracle'] = oracle(U, shards)
    print('oracle done', flush=True)
    put('grid', grid(run, U, shards))
    put('attention', attention(run))
    (run / 'results.json').write_text(json.dumps(results, indent=1, default=float))
    np.savez(run / 'analysis.npz', **{k: np.asarray(v) for k, v in arrays.items()})
    print('analysis written', flush=True)


if __name__ == '__main__':
    main()
