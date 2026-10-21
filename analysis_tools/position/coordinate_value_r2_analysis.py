"""Experiment 3.6 round 2 analysis (CPU): geometry, probe versus logit lens, digit formats, causal tests, neurons.

Inputs: the round-1 run (records and float16 feature shards) and the round-2 run directory written by
`coordinate_value_r2_run.py`. Outputs in the round-2 run directory: `results.json` (all reported
numbers), `analysis.npz` (figure arrays) and, for the neuron-ablation stage, `neuron_sets.json` and
`neuron_means.npz`.

Conventions (as in round 1): coordinates in 0-1000 units; the truth is the disc's tight-box center;
probes are ridge regressions on z-scored inputs, trained on grid2d and tested on the two sweeps;
layer 0 is the embedding and layer l the output of decoder layer l (1-based).
Each section runs only if its inputs exist, so the analysis can be repeated as stages finish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from analysis_tools.position import coordinate_value as cv
from analysis_tools.position.coordinate_readout_analysis import wilson
from analysis_tools.position.coordinate_readout_run import read_jsonl
from analysis_tools.position.coordinate_value_analysis import r2, ridge_fit, ridge_predict

RNG = np.random.default_rng(20261016)
SIGMA = 15.0                 # Gaussian kernel width (units) of the sweep manifold
CHORD = 20.0                 # half-width (units) of the chord used for tangents and speed
HEAD_SITES = ((20, 11), (20, 10), (16, 11), (20, 8), (32, 10), (32, 8))
NEURON_LAYERS = tuple(range(20, 33))


# ----------------------------------------------------------------------------- generic helpers

def prop(k: int, n: int) -> dict:
    return {'k': int(k), 'n': int(n), 'p': k / n if n else float('nan'), 'ci': wilson(int(k), int(n))}


def boot_ci(x, n: int = 2000) -> list[float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [float('nan')] * 2
    m = RNG.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def mean_ci(x) -> dict:
    x = np.asarray(x, float)
    return {'mean': float(np.nanmean(x)), 'ci': boot_ci(x), 'n': int(np.isfinite(x).sum())}


def sign_flip_p(d, n: int = 20000) -> float:
    """Two-sided paired sign-flip permutation p-value of mean(d) = 0."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float('nan')
    obs = abs(d.mean())
    s = RNG.choice([-1.0, 1.0], (n, len(d)))
    return float(((np.abs((s * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n + 1))


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b and c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def first_digit(v: int) -> int:
    return int(str(int(v))[0])


def entropy_bits(p: np.ndarray) -> np.ndarray:
    p = p / p.sum(-1, keepdims=True)
    return -(p * np.log2(p + 1e-12)).sum(-1)


def circ_diff(a, b, m: int = 10):
    return np.abs((np.asarray(a) - np.asarray(b) + m // 2) % m - m // 2)


# ----------------------------------------------------------------------------- loading

def load_records(run1: Path) -> dict:
    recs = {r['id']: r for r in read_jsonl(run1 / 'gen' / 'records.jsonl')}
    for r in recs.values():
        c = r['score'].get('coords') if r['score'].get('valid') else None
        r['emitted'] = [int(round(v)) for v in c] if c else None
        r['variant'] = tuple(r['ids'][:r['spans'][0][0]]) if r['spans'] else None
    return recs


def load_round1(feat: Path, roles=('x0', 'x1', 'y0', 'y1', 'last')) -> dict:
    index = json.loads((feat / 'index.json').read_text())
    ri = [cv.ROLES.index(r) for r in roles]
    parts = {k: [] for k in ('ids', 'resid', 'attn', 'mlp', 'heads', 'logp')}
    for sh in index['shards']:
        z = np.load(feat / sh['file'])
        parts['ids'] += z['ids'].tolist()
        parts['resid'].append(z['resid'][:, :, ri])
        for k in ('attn', 'mlp', 'heads', 'logp'):
            parts[k].append(z[k])
    out = {k: (np.concatenate(v) if k != 'ids' else v) for k, v in parts.items()}
    out['roles'] = list(roles)
    return out


def readout_params(checkpoint: Path, mlp_layers=()) -> dict:
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']

    def get(name):
        with safe_open(checkpoint / index[name], 'pt') as f:
            return f.get_tensor(name).float().numpy()
    W = get('lm_head.weight')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    P = {'gain': 1.0 + get('model.language_model.norm.weight'), 'W': W[digit_ids], 'digit_ids': digit_ids,
         'o_proj': {l: get(f'model.language_model.layers.{l - 1}.self_attn.o_proj.weight') for l in cv.FULL_LAYERS},
         'embed_digits': get('model.language_model.embed_tokens.weight')[digit_ids]}
    P['down'] = {l: get(f'model.language_model.layers.{l - 1}.mlp.down_proj.weight') for l in mlp_layers}
    return P


def checkpoint_path() -> Path:
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    return resolve_model('Qwen3.5-9B', load_paths()['HF_HUB_CACHE'])


# ----------------------------------------------------------------------------- Q1 geometry

def kernel_curve(t: np.ndarray, H: np.ndarray, sigma: float = SIGMA, loo: bool = False) -> np.ndarray:
    W = np.exp(-0.5 * ((t[:, None] - t[None]) / sigma) ** 2)
    if loo:
        np.fill_diagonal(W, 0)
    return (W / W.sum(1, keepdims=True)) @ H


def curve_metrics(t: np.ndarray, H: np.ndarray) -> dict:
    """Geometry of states H (n, D) along a sorted 1D variable t (n,).

    signal_fraction: share of the state variance explained by the leave-one-out kernel curve of t;
    pr / n90: participation ratio and number of PCs for 90% variance of the (noise-free) curve;
    tortuosity: arc length / chord of the curve (1 = straight line);
    tangent_corr: mean cosine between chord tangents separated by 100 / 250 / 500 units;
    speed: |d mu / dt| profile (state norm units per coordinate unit);
    delta: local discrimination threshold sigma_tangent / speed (units), i.e. the coordinate change that
           moves the curve by one standard deviation of the off-curve scatter projected on the tangent.
    """
    mu = kernel_curve(t, H)
    mu_loo = kernel_curve(t, H, loo=True)
    tot = ((H - H.mean(0)) ** 2).sum()
    res = H - mu_loo
    sig = 1 - (res ** 2).sum() / tot
    C = mu - mu.mean(0)
    ev = np.linalg.svd(C, compute_uv=False) ** 2
    ev = ev / ev.sum()
    pr = 1 / (ev ** 2).sum()
    n90 = int(np.searchsorted(np.cumsum(ev), 0.9) + 1)
    step = np.median(np.diff(t))
    k = max(1, int(round(CHORD / step)))
    j = np.arange(k, len(t) - k)
    chords = mu[j + k] - mu[j - k]
    dt = (t[j + k] - t[j - k])[:, None]
    vel = chords / dt
    speed = np.linalg.norm(vel, axis=1)
    tau = vel / speed[:, None]
    arc = np.linalg.norm(np.diff(mu, axis=0), axis=1).sum()
    chord = np.linalg.norm(mu[-1] - mu[0])
    corr = {}
    for D in (100, 250, 500):
        m = int(round(D / step))
        if m < len(tau):
            corr[str(D)] = float((tau[:-m] * tau[m:]).sum(1).mean())
    # Off-curve scatter projected on the local tangent, in windows of +/- 40 units.
    rt = (res[j] * tau).sum(1)
    sd = np.array([rt[np.abs(t[j] - t[i]) <= 40].std() for i in j])
    return {'signal_fraction': float(sig), 'pr': float(pr), 'n90': n90, 'pc_var': ev[:5].tolist(),
            'tortuosity': float(arc / chord), 'tangent_corr': corr,
            'speed_t': t[j].tolist(), 'speed': speed.tolist(), 'delta': (sd / speed).tolist(),
            'delta_median': float(np.median(sd / speed))}


def site_matrix(F: dict, site: str, layer: int, axis: int, head: tuple | None = None) -> np.ndarray:
    """(N, d) float32 states at the first-digit role of the given axis for one site."""
    role_i = F['roles'].index('x0' if axis == 0 else 'y0')
    if site == 'resid':
        return F['resid'][:, layer, role_i].astype(np.float32)
    if site in ('attn', 'mlp'):
        return F[site][:, layer - 1, axis].astype(np.float32)
    if site == 'head':
        l, h = head
        return F['heads'][:, cv.FULL_LAYERS.index(l), axis, h].astype(np.float32)
    raise ValueError(site)


def sweep_selection(recs: dict, ids: list[str], name: str, axis: int, dominant: bool = True) -> np.ndarray:
    sel = [k for k, i in enumerate(ids) if recs[i]['set'] == name and recs[i]['spans']]
    if dominant:
        vs = [recs[ids[k]]['variant'] for k in sel]
        top = max(set(vs), key=vs.count)
        sel = [k for k in sel if recs[ids[k]]['variant'] == top]
    return np.array(sorted(sel, key=lambda k: recs[ids[k]]['truth_units'][axis]))


def geometry(F: dict, recs: dict) -> tuple[dict, dict]:
    ids = F['ids']
    sets = np.array([recs[i]['set'] for i in ids])
    grid = np.where(sets == 'grid2d')[0]
    out, arrays = {}, {}
    sites = ([('resid', l, None) for l in range(0, 33)] + [('attn', l, None) for l in range(1, 33)]
             + [('mlp', l, None) for l in range(1, 33)] + [('head', h[0], h) for h in HEAD_SITES])
    for axis, name in ((0, 'sweep_h'), (1, 'sweep_v')):
        sel = sweep_selection(recs, ids, name, axis)
        t = np.array([recs[ids[k]]['truth_units'][axis] for k in sel])
        e = np.array([recs[ids[k]]['emitted'][axis] if recs[ids[k]]['emitted'] else np.nan for k in sel])
        tg = np.array([recs[ids[k]]['truth_units'][axis] for k in grid])
        test_all = np.where(sets == name)[0]
        tt = np.array([recs[ids[k]]['truth_units'][axis] for k in test_all])
        arrays[f'geo_{"xy"[axis]}_t'] = t
        arrays[f'geo_{"xy"[axis]}_emitted'] = e
        res = {}
        for site, layer, head in sites:
            key = f'{site}{layer}' + (f'h{head[1]}' if head else '')
            X = site_matrix(F, site, layer, axis, head)
            if site == 'resid' and layer == 0:
                continue                              # one shared token embedding
            m = curve_metrics(t, X[sel])
            fit = ridge_fit(X[grid], tg[:, None])
            pred = ridge_predict(fit, X[test_all])[:, 0]
            m['probe_mae'] = float(np.abs(pred - tt).mean())
            m['probe_r2'] = r2(tt, pred)
            # Compression test: slope of the linear readout against the truth below / above 380 units.
            lo, hi = tt < 380, tt >= 380
            m['probe_slope_lo'] = float(np.polyfit(tt[lo], pred[lo], 1)[0])
            m['probe_slope_hi'] = float(np.polyfit(tt[hi], pred[hi], 1)[0])
            m['norm_mean'] = float(np.linalg.norm(X[sel], axis=1).mean())
            res[key] = m
            if key in ('resid20', 'resid28', 'resid12', 'attn20', 'mlp28', 'mlp24', 'head20h11', 'resid32'):
                C = X[sel] - X[sel].mean(0)
                U, s, Vt = np.linalg.svd(C, full_matrices=False)
                arrays[f'geo_{"xy"[axis]}_{key}_pca3'] = C @ Vt[:3].T
                arrays[f'geo_{"xy"[axis]}_{key}_var3'] = (s ** 2 / (s ** 2).sum())[:3]
                arrays[f'geo_{"xy"[axis]}_{key}_probe'] = pred
                arrays[f'geo_{"xy"[axis]}_{key}_probe_t'] = tt
        out["xy"[axis]] = res
    # Output and model compression for reference.
    for axis, name in ((0, 'sweep_h'), (1, 'sweep_v')):
        k = [i for i in ids if recs[i]['set'] == name and recs[i]['emitted']]
        tt = np.array([recs[i]['truth_units'][axis] for i in k])
        ee = np.array([recs[i]['emitted'][axis] for i in k])
        out[f'model_slope_{"xy"[axis]}'] = {'lo': float(np.polyfit(tt[tt < 380], ee[tt < 380], 1)[0]),
                                           'hi': float(np.polyfit(tt[tt >= 380], ee[tt >= 380], 1)[0])}
    # 2D sheet: encoding directions of x and y at x0 and y0 on grid2d.
    tg = np.array([recs[ids[k]]['truth_units'] for k in grid])
    sheet = {}
    for role, axis in (('x0', 0), ('y0', 1)):
        for l in (12, 16, 20, 24, 28, 32):
            X = site_matrix(F, 'resid', l, axis)[grid]
            A = np.c_[tg - tg.mean(0)]
            B = np.linalg.lstsq(A, X - X.mean(0), rcond=None)[0]        # (2, D)
            cos = float(B[0] @ B[1] / np.linalg.norm(B[0]) / np.linalg.norm(B[1]))
            sheet[f'{role}_L{l}'] = {'cos_beta_x_beta_y': cos, 'norm_beta_x': float(np.linalg.norm(B[0])),
                                     'norm_beta_y': float(np.linalg.norm(B[1]))}
        X = site_matrix(F, 'resid', 20, axis)[grid]
        C = X - X.mean(0)
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        arrays[f'geo_grid_{role}_resid20_pca3'] = C @ Vt[:3].T
        arrays[f'geo_grid_{role}_resid20_var3'] = (s ** 2 / (s ** 2).sum())[:3]
    arrays['geo_grid_t'] = tg
    out['sheet'] = sheet
    # Signal fraction for all 128 heads (x0 / y0): which heads carry the coordinate as a smooth curve.
    for axis, name in ((0, 'sweep_h'), (1, 'sweep_v')):
        sel = sweep_selection(recs, ids, name, axis)
        t = np.array([recs[ids[k]]['truth_units'][axis] for k in sel])
        sf = np.zeros((len(cv.FULL_LAYERS), cv.N_HEADS))
        for k, l in enumerate(cv.FULL_LAYERS):
            for h in range(cv.N_HEADS):
                X = F['heads'][sel, k, axis, h].astype(np.float32)
                mu = kernel_curve(t, X, loo=True)
                sf[k, h] = 1 - ((X - mu) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
        arrays[f'geo_heads_signal_{"xy"[axis]}'] = sf
    return out, arrays


# ----------------------------------------------------------------------------- Q3 probe versus logit lens

def lens_logits(H: np.ndarray, P: dict) -> np.ndarray:
    rms = np.sqrt((H ** 2).mean(-1, keepdims=True) + 1e-6)
    return (H / rms * P['gain']) @ P['W'].T


def reconcile(F: dict, recs: dict, P: dict) -> tuple[dict, dict]:
    """Where is the digit information lost between the residual stream and the digit logits?

    For the first digit (role x0 / y0) and the second digit (role x1 / y1, three-digit answers whose
    first digit equals the true first digit), per layer:
      full_probe:   ridge on the full residual -> coordinate -> digit;
      lens_probe:   linear classifier (one-hot ridge) on the 10 logit-lens digit logits;
      random10:     the same classifier on 10 random directions of the normalized residual;
      lens_argmax:  the logit lens itself;
      beta_in_U:    share of the squared norm of the (normalized-space) coordinate encoding direction that
                    lies in the 9-dimensional span of the centered digit unembedding rows.
    """
    ids = F['ids']
    sets = np.array([recs[i]['set'] for i in ids])
    train = sets == 'grid2d'
    Wc = P['W'] - P['W'].mean(0)
    Uq, _ = np.linalg.qr(Wc.T)                      # (D, 10) orthonormal basis; rank 9 after centering
    Uq = Uq[:, :9]
    R = np.linalg.qr(RNG.standard_normal((P['W'].shape[1], 10)))[0]
    out, arrays = {}, {}
    for role, axis, k in (('x0', 0, 0), ('y0', 1, 0), ('x1', 0, 1), ('y1', 1, 1)):
        ri = F['roles'].index(role)
        t = np.array([recs[i]['truth_units'][axis] for i in ids])
        tr = np.rint(t).astype(int)
        e = np.array([recs[i]['emitted'][axis] if recs[i]['emitted'] else -1 for i in ids])
        ok = np.array([100 <= y <= 999 for y in tr]) & ~np.isnan(F['resid'][:, 1, ri, 0].astype(np.float32))
        if k == 1:
            ok &= np.array([100 <= x <= 999 and str(x)[0] == str(y)[0] for x, y in zip(e, tr)])
        true_d = (tr // 10 ** (2 - k)) % 10
        emit_d = np.where(e >= 100, (e // 10 ** (2 - k)) % 10, -1)
        # Test only on the sweep along this axis: on the other sweep the coordinate is fixed at 500.
        test = sets == ('sweep_h' if axis == 0 else 'sweep_v')
        tri, tei = train & ok, test & ok
        counts = np.bincount(true_d[tei], minlength=10)
        res = {'n_train': int(tri.sum()), 'n_test': int(tei.sum()),
               'majority_baseline': float(counts.max() / counts.sum()),
               'model': prop(int((emit_d[tei] == true_d[tei]).sum()), int(tei.sum())),
               'model_within1': prop(int((circ_diff(emit_d[tei], true_d[tei]) <= 1).sum()), int(tei.sum())), 'layers': {}}
        curves = {key: [] for key in ('full_probe', 'lens_probe', 'random10', 'lens_argmax', 'beta_in_U', 'full_within1',
                                      'lens_within1')}
        for l in range(33):
            H = F['resid'][:, l, ri].astype(np.float32)
            Hn = H / np.sqrt((H ** 2).mean(-1, keepdims=True) + 1e-6) * P['gain']
            fit = ridge_fit(H[tri], t[tri, None])
            pt = np.rint(ridge_predict(fit, H[tei])[:, 0]).astype(int)
            pd = (np.clip(pt, 0, 999) // 10 ** (2 - k)) % 10
            Z = Hn @ P['W'].T
            onehot = np.eye(10)[true_d]
            lens_fit = ridge_fit(Z[tri], onehot[tri])
            lp = ridge_predict(lens_fit, Z[tei]).argmax(1)
            rnd_fit = ridge_fit(Hn[tri] @ R, onehot[tri])
            rp = ridge_predict(rnd_fit, Hn[tei] @ R).argmax(1)
            Xc = Hn[tri] - Hn[tri].mean(0)
            tc = t[tri] - t[tri].mean()
            beta = Xc.T @ tc / (tc @ tc)
            share = float(((Uq.T @ beta) ** 2).sum() / (beta @ beta))
            curves['full_probe'].append(float((pd == true_d[tei]).mean()))
            curves['full_within1'].append(float((circ_diff(pd, true_d[tei]) <= 1).mean()))
            curves['lens_probe'].append(float((lp == true_d[tei]).mean()))
            curves['lens_within1'].append(float((circ_diff(lp, true_d[tei]) <= 1).mean()))
            curves['random10'].append(float((rp == true_d[tei]).mean()))
            curves['lens_argmax'].append(float((Z[tei].argmax(1) == true_d[tei]).mean()))
            curves['beta_in_U'].append(share)
        res['curves'] = curves
        res['beta_in_U_random_expectation'] = 9 / P['W'].shape[1]
        out[role] = res
        for key, v in curves.items():
            arrays[f'rec_{role}_{key}'] = np.array(v)
    return out, arrays


def unembed_geometry(P: dict) -> tuple[dict, dict]:
    """Geometry of the ten digit unembedding rows (and input embeddings) in the normalized space."""
    out, arrays = {}, {}
    for name, M in (('unembed', P['W'] * P['gain']), ('unembed_raw', P['W']), ('embed', P['embed_digits'])):
        C = M - M.mean(0)
        Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
        cos = Cn @ Cn.T
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        d = np.abs(np.arange(10)[:, None] - np.arange(10)[None])
        iu = np.triu_indices(10, 1)
        out[name] = {'pc_var': (s ** 2 / (s ** 2).sum()).tolist(),
                     'corr_cos_vs_numeric_distance': float(np.corrcoef(cos[iu], d[iu])[0, 1]),
                     'mean_cos_adjacent': float(np.mean([cos[i, i + 1] for i in range(9)])),
                     'mean_cos_far': float(cos[iu][d[iu] >= 5].mean()),
                     'norms': np.linalg.norm(M, axis=1).tolist()}
        arrays[f'ug_{name}_cos'] = cos
        arrays[f'ug_{name}_pca3'] = C @ Vt[:3].T
    return out, arrays


# ----------------------------------------------------------------------------- Q2 digit formats

def parse_numbers(text: str, kind: str) -> list[float] | None:
    from analysis_tools.position import coordinate_value_r2 as c2
    if kind == 'count':
        import re
        m = re.search(r'\d+', text)
        return [float(m.group())] if m else None
    spans = c2.point_numbers(text)
    return [float(text[a:b]) for a, b in spans] if len(spans) == 2 else None


SCALE = {'s1000': 1.0, 'text': 1.0, 's100': 0.1, 's10': 0.01, 'pix': 0.512, 'frac': 0.001, 'count': 1.0}


def compliance(cases: list[dict], gen: dict) -> dict:
    """Valid-answer rate, rate of answers on the requested scale, and errors in 0-1000 units."""
    out = {}
    groups: dict = {}
    for c in cases:
        if c['kind'] == 'blank':
            continue
        groups.setdefault((c['kind'], c['fmt']), []).append(c)
    for (kind, fmt), cs in groups.items():
        n, valid, on_scale, err, err_1000 = len(cs), 0, 0, [], []
        for c in cs:
            v = parse_numbers(gen[c['id']]['text'], kind) if c['id'] in gen else None
            if v is None or len(v) != len(c['truth']):
                continue
            valid += 1
            t = np.array(c['truth'], float)
            v = np.array(v)
            s = SCALE[fmt]
            if kind == 'count':
                on_scale += 1
                err.append(float(np.abs(v - t).mean()))
                continue
            # The answer is on the requested scale if it is closer to the scaled truth than to the 0-1000 truth.
            ok = bool(np.all(np.abs(v - t * s) <= np.abs(v - t))) if s != 1.0 else True
            on_scale += ok
            err.append(float(np.abs(v / s - t).mean()))
            err_1000.append(float(np.abs(v - t).mean()))
        out[f'{kind}_{fmt}'] = {'n': n, 'valid': prop(valid, n), 'on_scale': prop(on_scale, n),
                                'mae_requested_scale_in_units': mean_ci(err) if err else None,
                                'mae_if_read_as_0_1000': mean_ci(err_1000) if err_1000 else None}
    return out


def slot_bin(kind: str, fmt: str, slot: dict) -> tuple[str, float]:
    """Slot label counted from the last digit, and the digit bin width in 0-1000 units (count units for counts)."""
    from_end = slot['n_digits'] - 1 - slot['k']
    label = {0: 'last', 1: 'second_last', 2: 'third_last'}.get(from_end, f'{from_end}_from_end')
    if fmt == 'frac':
        return label, 10.0 ** from_end * 10                  # hundredths = 10 units, tenths = 100 units
    return label, 10.0 ** from_end / SCALE[fmt]


def varying_number(case: dict) -> int:
    """Index of the coordinate that varies along a disc sweep (x on sweep_h, y on sweep_v)."""
    return 0 if case['sweep'] == 'sweep_h' else 1


def formats_analysis(run: Path) -> tuple[dict, dict]:
    root = run / 'formats'
    cases = {c['id']: c for c in json.loads((root / 'cases.json').read_text())}
    gen = {r['id']: r for r in read_jsonl(root / 'gen.jsonl')}
    tf = read_jsonl(root / 'tf.jsonl')
    out = {'compliance': compliance(list(cases.values()), gen)}
    arrays = {}
    rows = []
    for r in tf:
        c = cases[r['id']]
        if c['kind'] == 'blank':
            continue
        for j, s in enumerate(r['slots']):
            if c['fmt'] == 'frac' and s['k'] == 0:
                continue                                       # the leading "0" of a fraction
            if c['kind'] == 'disc' and s['number'] != varying_number(c):
                continue                                       # the other coordinate is fixed at 500 on a sweep
            if c['kind'] != 'count' and int(c['true_strings'][s['number']].replace('.', '')) == 1000:
                continue
            lp = np.array(r['digit_logp'][j])
            p = np.exp(lp)
            label, w = slot_bin(c['kind'], c['fmt'], s)
            first = s['k'] == 0 or (c['fmt'] == 'frac' and s['k'] == 1)
            am = int(p.argmax())
            d = s['digit']
            within = abs(am - d) <= 1 if first else circ_diff(am, d) <= 1
            rows.append({'id': r['id'], 'kind': c['kind'], 'fmt': c['fmt'], 'label': label, 'bin': w, 'first': first,
                         'n_digits': s['n_digits'], 'p_true': float(p[d] / p.sum()), 'acc': am == d, 'within1': bool(within),
                         'entropy': float(entropy_bits(p)), 'p_digit': r['p_digit'][j], 'digit': d, 'argmax': am,
                         'p05': float((p[0] + p[5]) / p.sum()), 'number': s['number']})
    groups: dict = {}
    for x in rows:
        groups.setdefault((x['kind'], x['fmt'], x['label'], x['first'], x['n_digits']), []).append(x)
    table = []
    for (kind, fmt, label, first, nd), xs in sorted(groups.items()):
        if len(xs) < 8:
            continue
        table.append({'kind': kind, 'fmt': fmt, 'slot': label, 'first_digit': first, 'n_digits': nd, 'bin_units': xs[0]['bin'],
                      'n': len(xs), 'p_true': mean_ci([x['p_true'] for x in xs]),
                      'acc': prop(sum(x['acc'] for x in xs), len(xs)), 'within1': prop(sum(x['within1'] for x in xs), len(xs)),
                      'entropy_bits': mean_ci([x['entropy'] for x in xs]), 'p_0_or_5': mean_ci([x['p05'] for x in xs])})
    out['digit_table'] = table
    # Probes on the residual at each digit slot (5-fold CV over cases): is the coordinate (count) present?
    out['probes'] = slot_probes(root, cases, tf)
    # Blank-canvas prior of the second digit given the first digit, versus the sweep distribution.
    prior = {}
    for r in tf:
        c = cases[r['id']]
        if c['kind'] != 'blank':
            continue
        for j, s in enumerate(r['slots']):
            if s['k'] == 1:
                p = np.exp(np.array(r['digit_logp'][j]))
                prior.setdefault(c['fmt'], {}).setdefault(int(c['true_strings'][0][0]), []).append(p / p.sum())
    pr_out = {}
    for fmt, by in prior.items():
        M = np.stack([np.mean(by[d], 0) for d in sorted(by)])
        arrays[f'prior_{fmt}'] = M
        pr_out[fmt] = {'p_0_or_5': float((M[:, 0] + M[:, 5]).mean()), 'entropy_bits': float(entropy_bits(M).mean())}
    # Sweep second-digit distributions (three-digit 0-1000 answers) against the prior of their first digit.
    if 's1000' in prior:
        sims, kls, kl_u = [], [], []
        D = []
        for r in tf:
            c = cases[r['id']]
            if c['kind'] != 'disc' or c['fmt'] != 's1000':
                continue
            for j, s in enumerate(r['slots']):
                if s['k'] == 1 and s['n_digits'] == 3:
                    first = int(c['true_strings'][s['number']][0])
                    if first not in prior['s1000']:
                        continue
                    p = np.exp(np.array(r['digit_logp'][j]))
                    p = p / p.sum()
                    q = np.mean(prior['s1000'][first], 0)
                    sims.append(float(p @ q / np.linalg.norm(p) / np.linalg.norm(q)))
                    kls.append(float((p * np.log((p + 1e-9) / (q + 1e-9))).sum()))
                    kl_u.append(float((p * np.log((p + 1e-9) * 10)).sum()))
                    D.append(p)
        pr_out['sweep_vs_prior'] = {'cosine': mean_ci(sims), 'kl_to_prior_nats': mean_ci(kls),
                                    'kl_to_uniform_nats': mean_ci(kl_u)}
        arrays['sweep_second_digit_mean'] = np.mean(D, 0)
    out['prior'] = pr_out
    arrays['fmt_rows_bin'] = np.array([x['bin'] for x in rows])
    return out, arrays, rows


def slot_probes(root: Path, cases: dict, tf: list[dict], layers=(16, 20, 24, 28, 32)) -> dict:
    """Ridge probe (5-fold CV) from the slot residual to the true value; the decoded value gives the slot digit."""
    shards = {}
    for f in sorted(root.glob('tf_shard_*.npz')):
        shards[int(f.stem.split('_')[-1])] = f
    by: dict = {}
    cache: dict = {}
    for r in tf:
        c = cases[r['id']]
        if c['kind'] == 'blank':
            continue
        if r['shard'] not in cache:
            cache = {r['shard']: np.load(shards[r['shard']])['resid']}
        R = cache[r['shard']]
        for j, s in enumerate(r['slots']):
            if c['fmt'] == 'frac' and s['k'] == 0:
                continue
            if c['kind'] == 'disc' and s['number'] != varying_number(c):
                continue
            label, w = slot_bin(c['kind'], c['fmt'], s)
            key = (c['kind'], c['fmt'], s['number'] if c['kind'] != 'count' else 0, label)
            by.setdefault(key, []).append((R[r['row_offset'] + j].astype(np.float32), c['truth'][s['number']], s, c['fmt'],
                                           c['true_strings'][s['number']]))
    out = {}
    for (kind, fmt, number, label), items in by.items():
        if len(items) < 40:
            continue
        X = np.stack([x[0] for x in items])                    # (n, Lr, D)
        t = np.array([x[1] for x in items])
        digits = np.array([x[2]['digit'] for x in items])
        from_end = items[0][2]['n_digits'] - 1 - items[0][2]['k']
        folds = np.arange(len(items)) % 5
        RNG.shuffle(folds)
        res = {'n': len(items)}
        for li, l in enumerate(layers):
            pred = np.zeros(len(items))
            for f in range(5):
                tr, te = folds != f, folds == f
                m = ridge_fit(X[tr, li], t[tr, None])
                pred[te] = ridge_predict(m, X[te, li])[:, 0]
            v = pred * SCALE[fmt]
            if fmt == 'frac':
                v = v * 100
            dig = (np.rint(v).astype(int) // 10 ** from_end) % 10
            res[str(l)] = {'mae_units': float(np.abs(pred - t).mean()), 'digit_acc': float((dig == digits).mean()),
                           'digit_within1': float((circ_diff(dig, digits) <= 1).mean())}
        out[f'{kind}_{fmt}_n{number}_{label}'] = res
    return out


# ----------------------------------------------------------------------------- causal: ablation

def value_of(digits: str) -> float:
    return float(int(digits)) if digits else float('nan')


def ablation_analysis(path: Path) -> tuple[dict, dict]:
    recs = read_jsonl(path)
    names = [c['cond'] for c in recs[0]['conditions']]
    T = np.array([r['truth'] for r in recs])
    tr = np.rint(T).astype(int)
    dstar = np.array([int(str(v)[0]) for v in tr])
    lstar = np.array([len(str(v)) for v in tr])
    V = {n: np.array([value_of(next(c for c in r['conditions'] if c['cond'] == n)['digits']) for r in recs]) for n in names}
    LP = {n: np.array([next(c for c in r['conditions'] if c['cond'] == n)['first_logp'] for r in recs]) for n in names}
    axis = np.array([r['axis'] for r in recs])

    def first_ok(v):
        return np.array([np.isfinite(x) and len(str(int(x))) == l and int(str(int(x))[0]) == d for x, l, d in zip(v, lstar, dstar)])
    clean_ok = first_ok(V['clean'])
    lp_clean = LP['clean']
    e_clean = lp_clean.argmax(1)
    out = {'n': len(recs), 'n_clean_first_wrong': int((~clean_ok).sum()), 'conditions': {}}
    for n in names:
        ok = first_ok(V[n])
        lp = LP[n]
        p_true = np.exp(lp[np.arange(len(recs)), dstar])
        p_true_c = np.exp(lp_clean[np.arange(len(recs)), dstar])
        mae = np.abs(V[n] - T)
        mae_c = np.abs(V['clean'] - T)
        # Margin of the true first digit over the digit the clean model emitted, on clean-wrong samples.
        w = ~clean_ok & (e_clean != dstar)
        marg = lp[np.arange(len(recs)), dstar] - lp[np.arange(len(recs)), e_clean]
        marg_c = lp_clean[np.arange(len(recs)), dstar] - lp_clean[np.arange(len(recs)), e_clean]
        b = int((clean_ok & ~ok).sum())
        c = int((~clean_ok & ok).sum())
        ent = entropy_bits(np.exp(lp))
        out['conditions'][n] = {
            'first_acc': prop(int(ok.sum()), len(ok)), 'mcnemar_p': mcnemar_p(b, c), 'broken': b, 'fixed': c,
            'fixed_rate_among_wrong': prop(c, int((~clean_ok).sum())), 'broken_rate_among_right': prop(b, int(clean_ok.sum())),
            'delta_p_true': mean_ci(p_true - p_true_c), 'delta_p_true_p': sign_flip_p(p_true - p_true_c),
            'mae': mean_ci(mae), 'delta_mae': mean_ci(mae - mae_c), 'delta_mae_p': sign_flip_p(mae - mae_c),
            'delta_margin_true_vs_clean_emitted_on_wrong': mean_ci((marg - marg_c)[w]),
            'delta_margin_p': sign_flip_p((marg - marg_c)[w]),
            'value_unchanged': prop(int((V[n] == V['clean']).sum()), len(recs)),
            'entropy_bits': mean_ci(ent), 'mean_abs_value_shift': mean_ci(np.abs(V[n] - V['clean'])),
            'by_axis_first_acc': {a: float(ok[axis == i].mean()) for i, a in enumerate('xy')}}
    arrays = {'abl_names': np.array(names), 'abl_truth': T, 'abl_axis': axis,
              'abl_values': np.stack([V[n] for n in names]), 'abl_first_logp': np.stack([LP[n] for n in names])}
    return out, arrays


# ----------------------------------------------------------------------------- causal: path patching

def pathpatch_analysis(path: Path) -> tuple[dict, dict]:
    recs = read_jsonl(path)
    names = [k for k in recs[0]['logp'] if k not in ('A', 'B', 'P')]
    rows = []
    for r in recs:
        lp = {k: np.array(v) for k, v in r['logp'].items()}
        dA, dB = int(lp['A'].argmax()), int(lp['B'].argmax())
        if dA == dB:
            continue
        LD = {k: v[dB] - v[dA] for k, v in lp.items()}
        rows.append({'axis': r['axis'], 'LD': LD, 'dP': int(lp['P'].argmax()), 'dA': dA, 'dB': dB})
    out = {'n_pairs': len(recs), 'n_pairs_differing_first_digit': len(rows)}
    arrays = {}
    for axis in (0, 1, None):
        rs = [x for x in rows if axis is None or x['axis'] == axis]
        tag = 'all' if axis is None else 'xy'[axis]
        if not rs:
            continue
        te = np.array([(x['LD']['P'] - x['LD']['A']) / (x['LD']['B'] - x['LD']['A']) for x in rs])
        gap = np.array([x['LD']['P'] - x['LD']['A'] for x in rs])
        eff = np.abs(gap) > 1.0
        res = {'n': len(rs), 'sender_total_effect_normalized': mean_ci(te),
               'P_first_digit_equals_B': prop(sum(x['dP'] == x['dB'] for x in rs), len(rs)),
               'n_sender_effect_gt1': int(eff.sum()), 'mediators': {}}
        for n in names:
            share = np.array([(x['LD'][n] - x['LD']['A']) for x in rs])[eff] / gap[eff]
            norm_b = np.array([(x['LD'][n] - x['LD']['A']) / (x['LD']['B'] - x['LD']['A']) for x in rs])
            res['mediators'][n] = {'share_of_sender_effect': mean_ci(share), 'p': sign_flip_p(share),
                                   'normalized_to_B': mean_ci(norm_b)}
        out[tag] = res
        arrays[f'pp_{tag}_heads'] = np.array([[res['mediators'][f'med_head{l}_{h}']['share_of_sender_effect']['mean']
                                               for h in range(16)] for l in (24, 28, 32)])
        arrays[f'pp_{tag}_mlp'] = np.array([res['mediators'][f'med_mlp{l}']['share_of_sender_effect']['mean'] for l in range(20, 33)])
        arrays[f'pp_{tag}_mlp_ci'] = np.array([res['mediators'][f'med_mlp{l}']['share_of_sender_effect']['ci'] for l in range(20, 33)])
        arrays[f'pp_{tag}_mixer'] = np.array([res['mediators'][f'med_mixer{l}']['share_of_sender_effect']['mean'] for l in range(21, 33)])
        arrays[f'pp_{tag}_mixer_ci'] = np.array([res['mediators'][f'med_mixer{l}']['share_of_sender_effect']['ci'] for l in range(21, 33)])
    return out, arrays


# ----------------------------------------------------------------------------- Q5 neurons

def load_neurons(feat: Path) -> tuple[list[str], np.ndarray]:
    files = sorted((feat / 'neurons').glob('shard_*.npz'))
    ids, acts = [], []
    for f in files:
        z = np.load(f)
        ids += z['ids'].tolist()
        acts.append(z['acts'])
    return ids, np.concatenate(acts)                       # (N, 4 roles, 13 layers, F) float16


def neuron_analysis(run: Path, feat: Path, recs: dict, F: dict, P: dict) -> tuple[dict, dict]:
    """Digit-writing MLP neurons at the first-digit (x0 / y0) and second-digit (x1) positions."""
    nids, A = load_neurons(feat)
    index = {i: k for k, i in enumerate(F['ids'])}
    rowsF = np.array([index[i] for i in nids])
    roles = ('x0', 'x1', 'y0', 'y1')
    sets = np.array([recs[i]['set'] for i in nids])
    out, arrays = {'n_samples': len(nids)}, {}
    WUg = P['W'] * P['gain']                                     # (10, D)
    V = {l: WUg @ P['down'][l] for l in NEURON_LAYERS}           # (10, F) digit write vectors
    Vc = {l: V[l] - V[l].mean(0) for l in NEURON_LAYERS}
    # Reconstruction check: down_proj(acts) equals the captured MLP output at x0 (round-1, float16).
    chk = []
    for l in (24, 28, 32):
        li = NEURON_LAYERS.index(l)
        a = A[:20, 0, li].astype(np.float32)
        m = F['mlp'][rowsF[:20], l - 1, 0].astype(np.float32)
        rec = a @ P['down'][l].T
        chk.append(float(np.linalg.norm(rec - m) / np.linalg.norm(m)))
    out['reconstruction_rel_error_L24_28_32'] = chk
    sel_sets = {}
    for role, axis, k in (('x0', 0, 0), ('y0', 1, 0), ('x1', 0, 1)):
        ri = roles.index(role)
        fi = F['roles'].index(role)
        rho = np.sqrt((F['resid'][rowsF, 32, fi].astype(np.float32) ** 2).mean(-1) + 1e-6)     # final-norm scale
        t = np.array([recs[i]['truth_units'][axis] for i in nids])
        e = np.array([recs[i]['emitted'][axis] if recs[i]['emitted'] else -1 for i in nids])
        ok = (e >= 100) & (np.rint(t) >= 100) & (np.rint(t) <= 999)
        if k == 1:
            ok &= np.array([str(int(x))[0] == str(int(round(y)))[0] for x, y in zip(e, t)])
        sweep = (sets == ('sweep_h' if axis == 0 else 'sweep_v')) & ok
        grid = (sets == 'grid2d') & ok
        true_d = (np.rint(t).astype(int) // 10 ** (2 - k)) % 10
        emit_d = np.where(e >= 100, (e // 10 ** (2 - k)) % 10, -1)
        imp_all, info = [], []
        layer_summary = {}
        recon = {}
        for l in NEURON_LAYERS:
            li = NEURON_LAYERS.index(l)
            a = A[:, ri, li].astype(np.float32)                 # (N, F)
            amean = a[grid].mean(0)
            # Importance: std over grid samples of the neuron's (digit-centered) logit write.
            imp = a[grid].std(0) * np.linalg.norm(Vc[l], axis=0) / rho[grid].mean()
            imp_all.append(imp)
            # Sparsity: digit-logit variance of this MLP explained by its top-k neurons (others at their mean).
            full = ((a[sweep] - amean) @ Vc[l].T) / rho[sweep, None]          # (n, 10)
            tot = ((full - full.mean(0)) ** 2).sum()
            order = np.argsort(-imp)
            fr = {}
            for kk in (10, 30, 100, 300, 1000):
                top = order[:kk]
                part = ((a[sweep][:, top] - amean[top]) @ Vc[l][:, top].T) / rho[sweep, None]
                fr[str(kk)] = float(1 - ((full - part) ** 2).sum() / tot)
            recon[str(l)] = fr
            layer_summary[str(l)] = {'importance_sum': float(imp.sum()), 'top10_importance_share': float(np.sort(imp)[-10:].sum() / imp.sum()),
                                     'mlp_digit_acc_true': float((full.argmax(1) == true_d[sweep]).mean()),
                                     'mlp_digit_acc_emitted': float((full.argmax(1) == emit_d[sweep]).mean())}
        imp_all = np.stack(imp_all)                              # (13, F)
        out[f'{role}_layers'] = layer_summary
        out[f'{role}_topk_variance_explained'] = recon
        # Top neurons across layers 24-32, with tuning curves on the sweep and key->value consistency.
        flat = np.argsort(-imp_all[4:].ravel())[:60]
        top = [(int(NEURON_LAYERS[4 + f // imp_all.shape[1]]), int(f % imp_all.shape[1])) for f in flat]
        sel_sets[role] = top
        order_t = np.argsort(t[sweep])
        ts = t[sweep][order_t]
        curves, writes, cons = [], [], []
        for l, n in top:
            li = NEURON_LAYERS.index(l)
            a = A[sweep, ri, li, n].astype(np.float32)[order_t]
            amean = A[grid, ri, li, n].astype(np.float32).mean()
            contrib = np.outer(a - amean, Vc[l][:, n]) / rho[sweep][order_t, None]   # (n, 10)
            sm = kernel_curve(ts, a[:, None], 10.0)[:, 0]
            curves.append(sm)
            writes.append(Vc[l][:, n])
            strong = np.abs(a - amean) >= np.quantile(np.abs(a - amean), 0.8)
            wd = contrib.argmax(1)
            td = true_d[sweep][order_t]
            ed = emit_d[sweep][order_t]
            rr = np.corrcoef(ts, a)[0, 1]
            cons.append({'layer': l, 'neuron': n, 'importance': float(imp_all[NEURON_LAYERS.index(l), n]),
                         'write_digit_pos': int(Vc[l][:, n].argmax()), 'write_digit_neg': int(Vc[l][:, n].argmin()),
                         'corr_with_t': float(rr), 'strong_matches_true': float((wd[strong] == td[strong]).mean()),
                         'strong_matches_emitted': float((wd[strong] == ed[strong]).mean()),
                         'peak_t': float(ts[np.argmax(np.abs(sm - amean))])})
        out[f'{role}_top_neurons'] = cons
        arrays[f'nr_{role}_t'] = ts
        arrays[f'nr_{role}_curves'] = np.stack(curves)
        arrays[f'nr_{role}_writes'] = np.stack(writes)
        arrays[f'nr_{role}_emitted'] = e[sweep][order_t]
        arrays[f'nr_{role}_importance'] = imp_all
        w = np.array([c['importance'] for c in cons])
        out[f'{role}_key_value'] = {
            'weighted_match_true': float(np.average([c['strong_matches_true'] for c in cons], weights=w)),
            'weighted_match_emitted': float(np.average([c['strong_matches_emitted'] for c in cons], weights=w)),
            'share_monotone_abs_corr_gt_0.8': float(np.mean([abs(c['corr_with_t']) > 0.8 for c in cons]))}
    # Neuron sets for the causal neuron-ablation stage (first-digit roles; importance of x0 and y0 summed).
    imp = arrays['nr_x0_importance'] + arrays['nr_y0_importance']
    layers_abl = list(range(24, 33))
    sub = imp[4:]
    order = np.argsort(-sub.ravel())
    nF = sub.shape[1]
    sets_out = {}
    for kk in (10, 30, 100, 300):
        top = [(layers_abl[f // nF], int(f % nF)) for f in order[:kk]]
        sets_out[f'top{kk}'] = top
        counts = {}
        for l, _ in top:
            counts[l] = counts.get(l, 0) + 1
        for r in range(3):
            pick = []
            for l, c in counts.items():
                pool = np.setdiff1d(np.arange(nF), [n for ll, n in top if ll == l])
                pick += [(l, int(n)) for n in RNG.choice(pool, c, replace=False)]
            sets_out[f'rand{kk}_{r}'] = pick
    l2830 = [i for i in order if layers_abl[i // nF] in (28, 29, 30)][:30]
    sets_out['top30_L28-30'] = [(layers_abl[f // nF], int(f % nF)) for f in l2830]
    (run / 'neuron_sets.json').write_text(json.dumps({'layers': layers_abl, 'sets': sets_out}))
    means = {}
    for l in layers_abl:
        li = NEURON_LAYERS.index(l)
        for axis, role in ((0, 'x0'), (1, 'y0')):
            means[f'L{l}_{"xy"[axis]}'] = A[(sets != 't6'), roles.index(role), li].astype(np.float32).mean(0)
    np.savez(run / 'neuron_means.npz', **means)
    return out, arrays


def key_value_order(neurons: dict, arrays: dict) -> tuple[dict, dict]:
    """Do the most important first-digit neurons form an ordered key->value map?

    For each of the top neurons, take its smoothed activation curve along the sweep, find the coordinate
    where it deviates most from its sweep mean (the key), and the digit it raises there, which is the
    argmax of sign(deviation) times its digit write vector (the value; a negative deviation writes the
    negative pole). Reports the Spearman correlation between key and value with a permutation p-value,
    and the offset of the key from the centre of the value digit's true range (100 d + 50).
    """
    out, arr = {}, {}
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    for role in ('x0', 'y0', 'x1'):
        if f'nr_{role}_curves' not in arrays:
            continue
        t, C, W = arrays[f'nr_{role}_t'], arrays[f'nr_{role}_curves'], arrays[f'nr_{role}_writes']
        dev = C - C.mean(1, keepdims=True)
        k = np.abs(dev).argmax(1)
        sign = np.sign(dev[np.arange(len(C)), k])
        key = t[k]
        val = (sign[:, None] * W).argmax(1).astype(float)
        rho = float(np.corrcoef(rk(key), rk(val))[0, 1])
        perm = np.array([np.corrcoef(rk(key), rk(RNG.permutation(val)))[0, 1] for _ in range(5000)])
        off = key - (100 * val + 50)
        out[role] = {'n': len(C), 'spearman': rho, 'perm_p': float(((np.abs(perm) >= abs(rho)).sum() + 1) / 5001),
                     'median_key_offset_units': float(np.median(off)), 'mean_abs_offset_units': float(np.abs(off).mean()),
                     'share_within_100_units': float((np.abs(off) <= 100).mean())}
        arr[f'kv_{role}_key'], arr[f'kv_{role}_value'] = key, val
    return out, arr


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--round1', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--sections', default='geometry,reconcile,unembed,formats,ablate,pathpatch,neurons,keyvalue,neuron_ablate')
    args = p.parse_args()
    res_path, arr_path = args.run / 'results.json', args.run / 'analysis.npz'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    arrays = dict(np.load(arr_path, allow_pickle=False)) if arr_path.exists() else {}
    secs = args.sections.split(',')
    recs = load_records(args.round1)
    need_F = any(s in secs for s in ('geometry', 'reconcile', 'neurons'))
    F = load_round1(args.round1 / 'features') if need_F else None
    P = readout_params(checkpoint_path(), NEURON_LAYERS if 'neurons' in secs else ())

    def put(name, r):
        # Merge with the files on disk so that sections computed by concurrent processes are kept.
        if res_path.exists():
            results.update({k: v for k, v in json.loads(res_path.read_text()).items() if k != name})
        if arr_path.exists():
            arrays.update({k: v for k, v in np.load(arr_path).items() if k not in r[1]})
        results[name], a = r[0], r[1]
        arrays.update(a)
        res_path.write_text(json.dumps(results, indent=1))
        np.savez(arr_path, **arrays)
        print(f'section {name} done', flush=True)
    if 'unembed' in secs:
        put('unembed', unembed_geometry(P))
    if 'geometry' in secs:
        put('geometry', geometry(F, recs))
    if 'reconcile' in secs:
        put('reconcile', reconcile(F, recs, P))
    if 'formats' in secs and (args.run / 'formats' / 'tf.jsonl').exists():
        o, a, rows = formats_analysis(args.run)
        (args.run / 'format_rows.json').write_text(json.dumps(rows))
        put('formats', (o, a))
    for name, fn in (('ablate', ablation_analysis), ('neuron_ablate', ablation_analysis)):
        if name in secs and (args.run / name / 'records.jsonl').exists():
            put(name, fn(args.run / name / 'records.jsonl'))
    if 'pathpatch' in secs and (args.run / 'pathpatch' / 'records.jsonl').exists():
        put('pathpatch', pathpatch_analysis(args.run / 'pathpatch' / 'records.jsonl'))
    if 'neurons' in secs and (args.features / 'neurons').exists():
        put('neurons', neuron_analysis(args.run, args.features, recs, F, P))
    if 'keyvalue' in secs and 'neurons' in json.loads(res_path.read_text()):
        put('keyvalue', key_value_order(json.loads(res_path.read_text())['neurons'], dict(np.load(arr_path))))


if __name__ == '__main__':
    main()
