"""Experiment 3.6 analysis: behaviour, digit forcing, probes, geometry, logit lens, attribution, steering.

Reads the run directory written by `coordinate_value_run.py` and the float16 feature shards, and
writes `results.json` (all reported numbers) and `analysis.npz` (arrays for the figures).

Conventions:
* coordinates are in 0-1000 units; the truth of a disc is its tight-box center;
* "first digit" is the first written digit of a value (the hundreds digit for 100-999);
* probes are ridge regressions on z-scored residuals, trained on the 29x29 grid (grid2d) and
  evaluated on the two 1D sweeps, which are never used for fitting;
* layer index 0 is the input embedding, index l the output of decoder layer l (1-based).
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

ALPHAS = np.logspace(-1, 5, 13)
LAYERS = np.arange(33)
PCA_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
ROUND_VALUES = (250, 333, 500, 667, 750)
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- generic helpers

def bootstrap_mean(x, n: int = 2000) -> list[float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float('nan')] * 2
    m = RNG.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def prop(k: int, n: int) -> dict:
    return {'k': int(k), 'n': int(n), 'p': k / n if n else float('nan'), 'ci': wilson(int(k), int(n))}


def first_digit(v: int) -> int:
    return int(str(int(v))[0])


def ridge_fit(X: np.ndarray, Y: np.ndarray) -> dict:
    """Ridge on z-scored X with a per-target penalty chosen by exact leave-one-out error.

    X: (n, D); Y: (n, k). Returns the standardization, coefficients (D, k) and intercepts (k,).
    """
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = Y.mean(0)
    Yc = Y - ym
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    UtY = U.T @ Yc
    best_err, best_coef = np.full(Y.shape[1], np.inf), np.zeros((Z.shape[1], Y.shape[1]))
    for a in ALPHAS:
        f = s ** 2 / (s ** 2 + a)
        fit = U @ (f[:, None] * UtY)
        h = (U ** 2) @ f + 1.0 / len(Z)              # leverage including the intercept
        loo = ((Yc - fit) / (1 - h)[:, None]) ** 2
        err = loo.mean(0)
        better = err < best_err
        if better.any():
            coef = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY)
            best_coef[:, better] = coef[:, better]
            best_err[better] = err[better]
    return {'mu': mu, 'sd': sd, 'coef': best_coef, 'icpt': ym}


def ridge_predict(m: dict, X: np.ndarray) -> np.ndarray:
    return ((X - m['mu']) / m['sd']) @ m['coef'] + m['icpt']


def r2(y, p) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ----------------------------------------------------------------------------- data loading

def load_records(run: Path) -> list[dict]:
    recs = read_jsonl(run / 'gen' / 'records.jsonl')
    for r in recs:
        c = r['score'].get('coords') if r['score'].get('valid') else None
        r['emitted'] = [int(round(v)) for v in c] if c else None
    return recs


def load_features(feat: Path) -> dict:
    index = json.loads((feat / 'index.json').read_text())
    parts = {k: [] for k in ('ids', 'resid', 'attn', 'mlp', 'heads', 'logp')}
    for sh in index['shards']:
        z = np.load(feat / sh['file'])
        for k in parts:
            parts[k].append(z[k])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out['ids'] = out['ids'].tolist()
    return out


def readout_params(checkpoint: Path, full_layers) -> dict:
    """Final-norm gain, digit unembedding rows and full-attention o_proj weights (float32)."""
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']

    def get(name):
        with safe_open(checkpoint / index[name], 'pt') as f:
            return f.get_tensor(name).float().numpy()
    return {'gain': 1.0 + get('model.language_model.norm.weight'), 'unembed': get('lm_head.weight'),
            'o_proj': {l: get(f'model.language_model.layers.{l - 1}.self_attn.o_proj.weight') for l in full_layers}}


# ----------------------------------------------------------------------------- behaviour

def behaviour(recs: list[dict]) -> tuple[dict, dict]:
    disc = [r for r in recs if r['set'] != 't6']
    valid = [r for r in disc if r['emitted']]
    out = {'n_disc': len(disc), 'valid': prop(len(valid), len(disc)),
           'hit': prop(sum(r['score']['hit'] for r in valid), len(disc))}
    arrays = {}
    # Diagnostic "hit under either convention" (plan.md section 6 item 9): the same output read as pixels.
    pix = [bool(np.hypot(r['emitted'][0] - r['truth']['center'][0], r['emitted'][1] - r['truth']['center'][1])
                <= r['truth']['diameter'] / 2) for r in valid]
    hit = [bool(r['score']['hit']) for r in valid]
    area = np.pi * (cv.DIAMETER / 2) ** 2
    out['hit_any_convention'] = prop(sum(a or b for a, b in zip(hit, pix)), len(disc))
    out['hit_pixel_only'] = prop(sum(b and not a for a, b in zip(hit, pix)), len(disc))
    out['chance_hit'] = {'normalized': area / cv.CANVAS ** 2, 'pixel': area / 1000 ** 2}
    T = np.array([r['truth_units'] for r in valid])
    E = np.array([r['emitted'] for r in valid], float)
    err = E - T
    out['center_error'] = {'mean': float(np.linalg.norm(err, axis=1).mean()),
                           'ci': bootstrap_mean(np.linalg.norm(err, axis=1))}
    # Signed error as a function of the true coordinate, per axis (bins of 62.5 units = 32 pixels).
    edges = np.linspace(0, 1000, 17)
    bias = {}
    for a, axis in enumerate('xy'):
        idx = np.digitize(T[:, a], edges) - 1
        bias[axis] = [{'bin': [float(edges[i]), float(edges[i + 1])], 'n': int((idx == i).sum()),
                       'mean': float(err[idx == i, a].mean()) if (idx == i).any() else None,
                       'ci': bootstrap_mean(err[idx == i, a])} for i in range(16)]
        slope, icpt = np.polyfit(T[:, a], E[:, a], 1)
        bias[axis + '_fit'] = {'slope': float(slope), 'intercept': float(icpt)}
    out['bias'] = bias
    # Output-value statistics: share of round values and of the most frequent values.
    # Value statistics use the 2D grid only: the sweeps hold one axis at 500 by construction.
    G = np.array([r['set'] == 'grid2d' for r in valid])
    vals = E[G].ravel().astype(int)
    tv = np.rint(T[G].ravel()).astype(int)
    out['values'] = {'n': int(len(vals)), 'n_distinct_emitted': int(len(set(vals))),
                     'n_distinct_truth_rounded': int(len(set(tv))),
                     'top': sorted([[int(v), int(c)] for v, c in zip(*np.unique(vals, return_counts=True))],
                                   key=lambda t: -t[1])[:15],
                     'round_share_emitted': prop(int(np.isin(vals, ROUND_VALUES).sum()), len(vals)),
                     'round_share_truth_within4': prop(int(np.isin(tv, ROUND_VALUES).sum()), len(tv)),
                     'multiple_of_10_share': prop(int((vals % 10 == 0).sum()), len(vals)),
                     'multiple_of_50_share': prop(int((vals % 50 == 0).sum()), len(vals))}
    # First-digit decomposition of the error.
    fd = {}
    for a, axis in enumerate('xy'):
        t_round = np.rint(T[:, a]).astype(int)
        e = E[:, a].astype(int)
        same_len = np.array([len(str(x)) == len(str(y)) for x, y in zip(e, t_round)])
        fd_ok = np.array([first_digit(x) == first_digit(y) for x, y in zip(e, t_round)]) & same_len
        abs_err = np.abs(e - T[:, a])
        fd[axis] = {'first_digit_correct': prop(int(fd_ok.sum()), len(e)),
                    'abs_error_given_first_correct': float(abs_err[fd_ok].mean()),
                    'abs_error_given_first_wrong': float(abs_err[~fd_ok].mean()) if (~fd_ok).any() else None,
                    'share_of_abs_error_from_first_wrong': float(abs_err[~fd_ok].sum() / abs_err.sum()),
                    'first_wrong_within_25_of_boundary': prop(
                        int((np.minimum(T[~fd_ok, a] % 100, 100 - T[~fd_ok, a] % 100) <= 25).sum()), int((~fd_ok).sum()))}
    out['first_digit'] = fd
    for name, a in (('sweep_h', 0), ('sweep_v', 1)):
        rs = sorted([r for r in valid if r['set'] == name], key=lambda r: r['truth_units'][a])
        arrays[f'{name}_truth'] = np.array([r['truth_units'][a] for r in rs])
        arrays[f'{name}_emitted'] = np.array([r['emitted'][a] for r in rs])
        arrays[f'{name}_other'] = np.array([r['emitted'][1 - a] for r in rs])
        # Basin structure: runs of identical outputs along the sweep (one step = 2 pixels = 3.9 units).
        em = arrays[f'{name}_emitted']
        runs, start = [], 0
        for i in range(1, len(em) + 1):
            if i == len(em) or em[i] != em[start]:
                runs.append((int(em[start]), i - start))
                start = i
        out[f'{name}_runs'] = {'n_runs': len(runs), 'n_steps': len(em),
                               'longest': sorted(runs, key=lambda t: -t[1])[:8],
                               'monotone_violations': int((np.diff(em) < 0).sum())}
    arrays['all_truth'], arrays['all_emitted'] = T[G], E[G]
    t6 = [r for r in recs if r['set'] == 't6']
    out['t6'] = {'n': len(t6), 'valid': sum(bool(r['emitted']) for r in t6),
                 'hit': prop(sum(bool(r['score'].get('hit')) for r in t6), len(t6))}
    return out, arrays


# ----------------------------------------------------------------------------- forcing

def forcing(run: Path) -> dict:
    rows = [r for r in read_jsonl(run / 'force' / 'records.jsonl') if r['set'] != 't6']
    out = {}
    for axis in 'xy':
        a = 'xy'.index(axis)
        res = {}
        free_first = {}
        for r in rows:
            for c in r['conditions']:
                if c['axis'] == axis and c['n_forced'] == 0 and c['value'] is not None:
                    tv = c['true_value']
                    free_first[r['id']] = len(str(c['value'])) == len(str(tv)) and first_digit(c['value']) == first_digit(tv)
        for k in (0, 1, 2):
            errs, wrong_first, tens = [], [], []
            n, n_valid = 0, 0
            for r in rows:
                for c in r['conditions']:
                    if c['axis'] != axis or c['n_forced'] != k:
                        continue
                    n += 1
                    if c['value'] is None:
                        continue
                    n_valid += 1
                    e = abs(c['value'] - r['truth_units'][a])
                    errs.append(e)
                    if free_first.get(r['id']) is False:
                        wrong_first.append(e)
                    if c['value'] >= 100:
                        tens.append(c['value'] % 100)
            tens = np.array(tens)
            res[f'forced_{k}'] = {'n': n, 'valid': n_valid, 'mean_abs_error': float(np.mean(errs)),
                                  'ci': bootstrap_mean(errs), 'median_abs_error': float(np.median(errs)),
                                  'mean_abs_error_where_free_first_wrong': float(np.mean(wrong_first)) if wrong_first else None,
                                  'n_free_first_wrong': len(wrong_first),
                                  'last_two_digits_00_or_50': prop(int(np.isin(tens, (0, 50)).sum()), len(tens)),
                                  'last_digit_0': prop(int((tens % 10 == 0).sum()), len(tens))}
        out[axis] = res
    return out


# ----------------------------------------------------------------------------- probes

def probe_targets(recs: dict, ids: list[str], axis: int) -> dict:
    T = np.array([recs[i]['truth_units'][axis] for i in ids])
    E = np.array([recs[i]['emitted'][axis] if recs[i]['emitted'] else np.nan for i in ids], float)
    O = np.array([recs[i]['truth_units'][1 - axis] for i in ids])
    return {'true': T, 'emitted': E, 'loss': T - E, 'other': O,
            'cos100': np.cos(2 * np.pi * E / 100), 'sin100': np.sin(2 * np.pi * E / 100),
            'cos10': np.cos(2 * np.pi * E / 10), 'sin10': np.sin(2 * np.pi * E / 10)}


def probes(F: dict, recs: dict) -> tuple[dict, dict]:
    ids = F['ids']
    sets = np.array([recs[i]['set'] for i in ids])
    train, test = sets == 'grid2d', np.isin(sets, ('sweep_h', 'sweep_v'))
    results, arrays = {}, {}
    names = ('true', 'emitted', 'loss', 'other', 'cos100', 'sin100', 'cos10', 'sin10')
    for role in ('last', 'x0', 'x1', 'x2', 'y0', 'y1', 'y2'):
        axis = 1 if role.startswith('y') else 0
        tg = probe_targets(recs, ids, axis)
        Y = np.stack([tg[n] for n in names], 1)
        ri = cv.ROLES.index(role)
        ok = ~np.isnan(F['resid'][:, 0, ri, 0].astype(np.float32)) & ~np.isnan(Y).any(1)
        tab = np.full((33, len(names), 2), np.nan)          # (layer, target, [R2, MAE])
        preds = {}
        for l in LAYERS:
            X = F['resid'][:, l, ri].astype(np.float32)
            m = ridge_fit(X[train & ok], Y[train & ok])
            P = ridge_predict(m, X[test & ok])
            for j, n in enumerate(names):
                y = Y[test & ok, j]
                tab[l, j] = [r2(y, P[:, j]), float(np.abs(y - P[:, j]).mean())]
            if role in ('x0', 'y0') and l in (8, 12, 14, 16, 20, 24, 28, 32):
                preds[int(l)] = P[:, :2]
        results[role] = {'n_train': int((train & ok).sum()), 'n_test': int((test & ok).sum()), 'targets': names,
                         'r2': tab[..., 0].tolist(), 'mae': tab[..., 1].tolist()}
        arrays[f'probe_{role}'] = tab
        if preds:
            arrays[f'probe_pred_{role}'] = np.stack([preds[k] for k in sorted(preds)])
            arrays[f'probe_pred_{role}_layers'] = np.array(sorted(preds))
            arrays[f'probe_pred_{role}_truth'] = Y[test & ok, :2]
    return results, arrays


# ----------------------------------------------------------------------------- geometry

def pca(X: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, (s ** 2 / (s ** 2).sum())[:k]


def geometry(F: dict, recs: dict) -> tuple[dict, dict]:
    ids = F['ids']
    out, arrays = {}, {}
    for name, role, a in (('sweep_h', 'x0', 0), ('sweep_v', 'y0', 1), ('sweep_h', 'x1', 0)):
        sel = [k for k, i in enumerate(ids) if recs[i]['set'] == name and recs[i]['emitted']]
        sel = sorted(sel, key=lambda k: recs[ids[k]]['truth_units'][a])
        t = np.array([recs[ids[k]]['truth_units'][a] for k in sel])
        e = np.array([recs[ids[k]]['emitted'][a] for k in sel])
        ri = cv.ROLES.index(role)
        H = F['resid'][sel][:, :, ri].astype(np.float32)               # (n, 33, D)
        ok = ~np.isnan(H[:, 0, 0])
        H, t, e = H[ok], t[ok], e[ok]
        key = f'{name}_{role}'
        steps = np.linalg.norm(np.diff(H, axis=0), axis=-1)            # (n-1, 33)
        med = np.median(steps, axis=0, keepdims=True)
        steps = steps / np.where(med > 0, med, np.nan)        # layer 0 at x0/y0 is one shared token embedding
        change = np.diff(e) != 0
        change_first = np.array([first_digit(x) != first_digit(y) or len(str(x)) != len(str(y))
                                 for x, y in zip(e[:-1], e[1:])], dtype=bool)
        auc_v, auc_f = [], []
        for l in LAYERS:
            auc_v.append(auroc(steps[change, l], steps[~change, l]))
            auc_f.append(auroc(steps[change_first, l], steps[~change_first, l]))
        # Continuity: does a 1D smooth curve of true position explain the states? Compare the
        # neighbour structure: correlation between |dt| and state distance for pairs within 60 units.
        out[key] = {'n': int(len(t)), 'n_value_changes': int(change.sum()), 'n_first_digit_changes': int(change_first.sum()),
                    'auroc_step_at_value_change': auc_v, 'auroc_step_at_first_digit_change': auc_f}
        arrays[f'geo_{key}_steps'] = steps
        arrays[f'geo_{key}_truth'] = t
        arrays[f'geo_{key}_emitted'] = e
        pcs, var = [], []
        for l in PCA_LAYERS:
            p, v = pca(H[:, l])
            pcs.append(p)
            var.append(v)
        arrays[f'geo_{key}_pca'] = np.stack(pcs)
        arrays[f'geo_{key}_pcavar'] = np.stack(var)
    # 2D grid sheet at x0: PCA colored by true x and y.
    sel = [k for k, i in enumerate(ids) if recs[i]['set'] == 'grid2d']
    ri = cv.ROLES.index('x0')
    pcs, var = [], []
    for l in PCA_LAYERS:
        p, v = pca(F['resid'][sel, l, ri].astype(np.float32))
        pcs.append(p)
        var.append(v)
    arrays['geo_grid2d_x0_pca'] = np.stack(pcs)
    arrays['geo_grid2d_x0_pcavar'] = np.stack(var)
    arrays['geo_grid2d_truth'] = np.array([recs[ids[k]]['truth_units'] for k in sel])
    arrays['geo_grid2d_emitted'] = np.array([recs[ids[k]]['emitted'] or [np.nan, np.nan] for k in sel], float)
    arrays['pca_layers'] = np.array(PCA_LAYERS)
    return out, arrays


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0 or np.isnan(pos).any() or np.isnan(neg).any():
        return float('nan')
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ----------------------------------------------------------------------------- logit lens and attribution

def digit_logits(H: np.ndarray, P: dict) -> np.ndarray:
    """Logit-lens digit logits (..., 10) of residuals H (..., D) through the final norm and unembedding."""
    rms = np.sqrt((H ** 2).mean(-1, keepdims=True) + 1e-6)
    return (H / rms * P['gain']) @ P['W'].T


def lens_and_attribution(F: dict, recs: dict, P: dict) -> tuple[dict, dict]:
    ids = F['ids']
    out, arrays = {}, {}
    disc = np.array([recs[i]['set'] != 't6' and bool(recs[i]['emitted']) for i in ids])
    for role, axis in (('x0', 0), ('y0', 1)):
        ri = cv.ROLES.index(role)
        e = np.array([first_digit(recs[i]['emitted'][axis]) if recs[i]['emitted'] else -1 for i in ids])
        tv = np.array([int(round(recs[i]['truth_units'][axis])) for i in ids])
        t = np.array([first_digit(v) for v in tv])
        same_len = np.array([bool(recs[i]['emitted']) and len(str(recs[i]['emitted'][axis])) == len(str(v))
                             for i, v in zip(ids, tv)])
        H = F['resid'][disc][:, :, ri].astype(np.float32)
        Z = digit_logits(H, P)                                          # (n, 33, 10)
        arg = Z.argmax(-1)
        out[f'lens_{role}'] = {'agree_emitted': [float((arg[:, l] == e[disc]).mean()) for l in LAYERS],
                               'agree_true': [float((arg[:, l] == t[disc]).mean()) for l in LAYERS]}
        # Direct logit attribution of every attention / MLP block at this role (final-norm scale frozen).
        ci = cv.COMPONENT_ROLES.index(role)
        rms = np.sqrt((F['resid'][disc, 32, ri].astype(np.float32) ** 2).mean(-1) + 1e-6)[:, None, None]
        A = (F['attn'][disc][:, :, ci].astype(np.float32) * P['gain']) @ P['W'].T / rms    # (n, 32, 10)
        M = (F['mlp'][disc][:, :, ci].astype(np.float32) * P['gain']) @ P['W'].T / rms
        n = int(disc.sum())
        ed, td = e[disc], t[disc]
        wrong = (ed != td) | ~same_len[disc]

        def margin(U, a, b):
            return U[np.arange(len(U)), :, a] - U[np.arange(len(U)), :, b] if b is not None else \
                U[np.arange(len(U)), :, a] - (U.sum(-1) - U[np.arange(len(U)), :, a]) / 9

        out[f'dla_{role}'] = {
            'n': n, 'n_first_wrong': int(wrong.sum()),
            'attn_margin_emitted': margin(A, ed, None).mean(0).tolist(),
            'mlp_margin_emitted': margin(M, ed, None).mean(0).tolist(),
            'attn_wrong_minus_true': margin(A[wrong], ed[wrong], td[wrong]).mean(0).tolist() if wrong.any() else None,
            'mlp_wrong_minus_true': margin(M[wrong], ed[wrong], td[wrong]).mean(0).tolist() if wrong.any() else None}
        # Per-head direct attribution and head-level position probes at full-attention layers.
        heads = F['heads'][:, :, ci].astype(np.float32)                    # (N, 8, 16, 256)
        sets = np.array([recs[i]['set'] for i in ids])
        train, test = sets == 'grid2d', np.isin(sets, ('sweep_h', 'sweep_v'))
        tx = np.array([recs[i]['truth_units'][axis] for i in ids])
        h_r2 = np.zeros((len(cv.FULL_LAYERS), cv.N_HEADS))
        h_dla = np.zeros((len(cv.FULL_LAYERS), cv.N_HEADS))
        rms_all = np.sqrt((F['resid'][:, 32, ri].astype(np.float32) ** 2).mean(-1) + 1e-6)
        e_all = np.where(e >= 0, e, 0)
        for k, l in enumerate(cv.FULL_LAYERS):
            Wo = P['o_proj'][l]
            for h in range(cv.N_HEADS):
                z = heads[:, k, h]
                m = ridge_fit(z[train], tx[train, None])
                h_r2[k, h] = r2(tx[test], ridge_predict(m, z[test])[:, 0])
                contrib = (z @ Wo[:, h * cv.HEAD_DIM:(h + 1) * cv.HEAD_DIM].T) * P['gain'] @ P['W'].T / rms_all[:, None]
                c = contrib[disc]
                h_dla[k, h] = float((c[np.arange(n), e_all[disc]] - (c.sum(1) - c[np.arange(n), e_all[disc]]) / 9).mean())
        out[f'heads_{role}'] = {'position_r2': h_r2.tolist(), 'dla_margin_emitted': h_dla.tolist()}
        arrays[f'heads_{role}_r2'], arrays[f'heads_{role}_dla'] = h_r2, h_dla
        arrays[f'dla_{role}_attn'] = margin(A, ed, None).mean(0)
        arrays[f'dla_{role}_mlp'] = margin(M, ed, None).mean(0)
    # Final digit tuning curves along the sweeps (log-probabilities captured from the full model).
    for name, role, a in (('sweep_h', 'x0', 0), ('sweep_h', 'x1', 0), ('sweep_h', 'x2', 0), ('sweep_v', 'y0', 1)):
        sel = sorted([k for k, i in enumerate(ids) if recs[i]['set'] == name],
                     key=lambda k: recs[ids[k]]['truth_units'][a])
        arrays[f'tuning_{name}_{role}'] = F['logp'][sel, cv.ROLES.index(role)]
        arrays[f'tuning_{name}_{role}_truth'] = np.array([recs[ids[k]]['truth_units'][a] for k in sel])
    return out, arrays


# ----------------------------------------------------------------------------- visual-token probe transfer

def visual_transfer(F: dict, recs: dict, feat: Path) -> dict:
    files = sorted(feat.glob('visual_*.npz'))
    if not files:
        return {}
    index = json.loads((feat / 'index.json').read_text())
    layers = index['visual_layers']
    rows = list(range(16))
    yy, xx = np.meshgrid(rows, rows, indexing='ij')
    centers = np.stack([(xx.ravel() + 0.5) / 16 * 1000, (yy.ravel() + 0.5) / 16 * 1000], 1)
    n_train = int(0.8 * len(files))
    ids = F['ids']
    disc = [k for k, i in enumerate(ids) if recs[i]['set'] != 't6']
    T = np.array([recs[ids[k]]['truth_units'] for k in disc])
    out = {}
    for l in layers:
        V = [np.load(f)[f'L{l}'].astype(np.float32) for f in files]
        Xtr, Ytr = np.concatenate(V[:n_train]), np.tile(centers, (n_train, 1))
        sub = RNG.choice(len(Xtr), min(len(Xtr), 3000), replace=False)
        m = ridge_fit(Xtr[sub], Ytr[sub])
        Xte = np.concatenate(V[n_train:])
        vis_mae = float(np.abs(ridge_predict(m, Xte) - np.tile(centers, (len(V) - n_train, 1))).mean())
        res = {'visual_test_mae': vis_mae}
        for role, a in (('x0', 0), ('y0', 1), ('last', 0)):
            X = F['resid'][disc, l, cv.ROLES.index(role)].astype(np.float32)
            p = ridge_predict(m, X)[:, a]
            res[role] = {'r': float(np.corrcoef(p, T[:, a])[0, 1]), 'mae': float(np.abs(p - T[:, a]).mean()),
                         'slope': float(np.polyfit(T[:, a], p, 1)[0])}
        out[str(l)] = res
    return out


# ----------------------------------------------------------------------------- steering

def steering(run: Path) -> tuple[dict, dict]:
    path = run / 'steer' / 'records.jsonl'
    if not path.exists():
        return {}, {}
    rows = read_jsonl(path)
    out, arrays = {}, {}
    layers = sorted({r['layer'] for row in rows for r in row['results']})
    deltas = sorted({r['delta'] for row in rows for r in row['results']})
    for method in ('linear', 'manifold'):
        tab = np.full((len(layers), len(rows), len(deltas)), np.nan)
        for j, row in enumerate(rows):
            base = {r['layer']: r['value'] for r in row['results'] if r['method'] == method and r['delta'] == 0}
            for r in row['results']:
                if r['method'] == method and r['value'] is not None and base.get(r['layer']) is not None:
                    tab[layers.index(r['layer']), j, deltas.index(r['delta'])] = r['value'] - base[r['layer']]
        arrays[f'steer_{method}'] = tab
        res = {}
        for li, l in enumerate(layers):
            d = np.array(deltas, float)
            y = tab[li]
            ok = ~np.isnan(y)
            D = np.broadcast_to(d, y.shape)[ok]
            slope = float(np.polyfit(D, y[ok], 1)[0]) if ok.sum() > 2 else float('nan')
            vals = [r['value'] for row in rows for r in row['results']
                    if r['method'] == method and r['layer'] == l and r['value'] is not None and r['delta'] != 0]
            n_all = sum(1 for row in rows for r in row['results'] if r['method'] == method and r['layer'] == l and r['delta'] != 0)
            res[str(l)] = {'slope': slope, 'n': n_all, 'valid': len(vals),
                           'round_share': prop(int(np.isin(vals, ROUND_VALUES).sum()), len(vals)),
                           'multiple_of_50_share': prop(int((np.array(vals) % 50 == 0).sum()), len(vals))}
        out[method] = res
    arrays['steer_layers'], arrays['steer_deltas'] = np.array(layers), np.array(deltas)
    arrays['steer_truth'] = np.array([row['truth_x'] for row in rows])
    return out, arrays


# ----------------------------------------------------------------------------- T6 transfer

def t6_transfer(F: dict, recs: dict) -> dict:
    """Apply the disc-trained true-position probe (x0 / y0) to the T6 point answers."""
    ids = F['ids']
    sets = np.array([recs[i]['set'] for i in ids])
    train = sets == 'grid2d'
    t6 = (sets == 't6') & np.array([bool(recs[i]['emitted']) for i in ids])
    out = {}
    for role, a in (('x0', 0), ('y0', 1)):
        ri = cv.ROLES.index(role)
        T = np.array([recs[i]['truth_units'][a] for i in ids])
        E = np.array([recs[i]['emitted'][a] if recs[i]['emitted'] else np.nan for i in ids], float)
        res = {}
        for l in (12, 16, 20, 24):
            X = F['resid'][:, l, ri].astype(np.float32)
            m = ridge_fit(X[train], T[train, None])
            p = ridge_predict(m, X[t6])[:, 0]
            res[str(l)] = {'n': int(t6.sum()), 'r_truth': float(np.corrcoef(p, T[t6])[0, 1]),
                           'r_emitted': float(np.corrcoef(p, E[t6])[0, 1]),
                           'mae_truth': float(np.abs(p - T[t6]).mean()), 'mae_emitted_vs_truth': float(np.abs(E[t6] - T[t6]).mean())}
        out[role] = res
    return out


# ----------------------------------------------------------------------------- answer-prefix control

def single_prefix_check(F: dict, recs: dict) -> dict:
    """Probe x0 / y0 using only answers with the dominant prefix (code fence + four-space indent).

    Greedy answers use three prefix variants whose frequency depends on the target position, so the
    tokens before x differ between samples. Restricting training and testing to one variant removes
    that input difference from the probe comparison.
    """
    ids = F['ids']
    fmt = np.array([recs[i]['text'].startswith('```json\n[\n    {') for i in ids])
    sets = np.array([recs[i]['set'] for i in ids])
    train, test = (sets == 'grid2d') & fmt, np.isin(sets, ('sweep_h', 'sweep_v')) & fmt
    out = {'n_train': int(train.sum()), 'n_test': int(test.sum()), 'prefix_counts': {}}
    for r in recs.values():
        if r['set'] != 't6':
            k = r['text'].split('"point_2d"')[0]
            out['prefix_counts'][k] = out['prefix_counts'].get(k, 0) + 1
    for role, a in (('x0', 0), ('y0', 1)):
        ri = cv.ROLES.index(role)
        t = np.array([recs[i]['truth_units'][a] for i in ids])
        e = np.array([recs[i]['emitted'][a] if recs[i]['emitted'] else np.nan for i in ids], float)
        res = {'model_mae': float(np.abs(e[test] - t[test]).mean())}
        for l in (8, 12, 20, 28, 32):
            X = F['resid'][:, l, ri].astype(np.float32)
            m = ridge_fit(X[train], np.stack([t, e], 1)[train])
            P = ridge_predict(m, X[test])
            res[str(l)] = {'mae_true': float(np.abs(P[:, 0] - t[test]).mean()),
                           'mae_emitted': float(np.abs(P[:, 1] - e[test]).mean())}
        out[role] = res
    return out


# ----------------------------------------------------------------------------- interpolation

def interpolation(run: Path) -> tuple[dict, dict]:
    """Summaries of the post-image residual interpolation between a base and a donor disc.

    For each pair and layer, the path of outputs over alpha is summarised by
    * transfer: the alpha = 1 output equals the donor's own unpatched output (within 5 units);
    * max-jump share: the largest change between consecutive alphas divided by |v(1) - v(0)|
      (0.1 for a uniform ramp over 10 steps, 1.0 for a single jump);
    * intermediate share: fraction of interior alphas whose output lies strictly between v(0) and v(1)
      and differs from both by more than 5 units.
    """
    path = run / 'interp' / 'records.jsonl'
    if not path.exists():
        return {}, {}
    rows = read_jsonl(path)
    layers = sorted({r['layer'] for row in rows for r in row['results']})
    alphas = sorted({r['alpha'] for row in rows for r in row['results']})
    out, arrays = {}, {}
    for off in sorted({row['offset'] for row in rows}):
        sub = [row for row in rows if row['offset'] == off]
        V = np.full((len(layers), len(sub), len(alphas)), np.nan)
        for j, row in enumerate(sub):
            for r in row['results']:
                if r['value'] is not None:
                    V[layers.index(r['layer']), j, alphas.index(r['alpha'])] = r['value']
        res = {}
        for li, l in enumerate(layers):
            v0, v1 = V[li, :, 0], V[li, :, -1]
            donor = np.array([row['donor_emitted'] for row in sub])
            span = np.abs(v1 - v0)
            moved = span > 10
            jumps = np.nanmax(np.abs(np.diff(V[li], axis=1)), axis=1)
            lo, hi = np.minimum(v0, v1)[:, None], np.maximum(v0, v1)[:, None]
            inner = V[li, :, 1:-1]
            between = (inner > lo + 5) & (inner < hi - 5)
            res[str(l)] = {'n_pairs': len(sub), 'transfer': prop(int((np.abs(v1 - donor) <= 5).sum()), len(sub)),
                           'n_moved': int(moved.sum()),
                           'max_jump_share': float(np.nanmean(jumps[moved] / span[moved])) if moved.any() else None,
                           'intermediate_share': float(between[moved].mean()) if moved.any() else None}
        out[str(off)] = res
        arrays[f'interp_{off}'] = V
        arrays[f'interp_{off}_truth'] = np.array([[row['base_truth'], row['donor_truth']] for row in sub])
    arrays['interp_layers'], arrays['interp_alphas'] = np.array(layers), np.array(alphas)
    return out, arrays


# ----------------------------------------------------------------------------- tens digit: available vs used

def tens_digit(F: dict, recs: dict) -> dict:
    """Compare the tens digit the model emits with the tens digit linearly decodable from the same state.

    Uses role x1 / y1 (the state that predicts the second digit) of three-digit answers whose first
    digit equals the first digit of the rounded truth. A ridge probe maps the state to
    (cos, sin)(2 pi t / 100) of the true coordinate t (grid2d training, sweep testing); the decoded phase
    gives a tens digit. The model's tens digit is its greedy choice at that state.
    """
    ids = F['ids']
    sets = np.array([recs[i]['set'] for i in ids])
    train, test = sets == 'grid2d', np.isin(sets, ('sweep_h', 'sweep_v'))
    out = {}
    for role, a in (('x1', 0), ('y1', 1)):
        ri = cv.ROLES.index(role)
        t = np.array([recs[i]['truth_units'][a] for i in ids])
        tr = np.rint(t).astype(int)
        e = np.array([recs[i]['emitted'][a] if recs[i]['emitted'] else -1 for i in ids])
        ok = np.array([100 <= x <= 999 and 100 <= y <= 999 and str(x)[0] == str(y)[0] for x, y in zip(e, tr)])
        ok &= ~np.isnan(F['resid'][:, 0, ri, 0].astype(np.float32))
        true_tens = (tr // 10) % 10
        Y = np.stack([np.cos(2 * np.pi * t / 100), np.sin(2 * np.pi * t / 100)], 1)
        logp = F['logp'][:, ri]
        p = np.exp(logp)
        p = p / p.sum(1, keepdims=True)
        sel = test & ok
        model_tens = (e // 10) % 10
        res = {'n_test': int(sel.sum()),
               'model_tens_correct': prop(int((model_tens[sel] == true_tens[sel]).sum()), int(sel.sum())),
               'model_tens_within1': prop(int((np.abs((model_tens[sel] - true_tens[sel] + 5) % 10 - 5) <= 1).sum()), int(sel.sum())),
               'model_p_true_tens': float(p[sel, true_tens[sel]].mean()),
               'model_max_p': float(p[sel].max(1).mean()),
               'model_entropy_bits': float((-(p[sel] * np.log2(p[sel] + 1e-12)).sum(1)).mean()),
               'chance_entropy_bits': float(np.log2(10)), 'layers': {}}
        for l in (8, 12, 16, 20, 24, 28, 32):
            X = F['resid'][:, l, ri].astype(np.float32)
            m = ridge_fit(X[train & ok], Y[train & ok])
            P = ridge_predict(m, X[sel])
            phase = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2 * np.pi)
            probe_tens = np.floor(phase / (2 * np.pi) * 10).astype(int) % 10
            # Linear probe on the raw coordinate, converted to a tens digit, as a second readout.
            m2 = ridge_fit(X[train & ok], t[train & ok, None])
            lin = ridge_predict(m2, X[sel])[:, 0]
            lin_tens = (np.rint(lin).astype(int) // 10) % 10
            res['layers'][str(l)] = {
                'circular_probe_correct': prop(int((probe_tens == true_tens[sel]).sum()), int(sel.sum())),
                'linear_probe_correct': prop(int((lin_tens == true_tens[sel]).sum()), int(sel.sum())),
                'linear_probe_within1': prop(int((np.abs((lin_tens - true_tens[sel] + 5) % 10 - 5) <= 1).sum()), int(sel.sum()))}
        out[role] = res
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--parts', default='behaviour,force,probes,geometry,lens,visual,steer,t6,tens,prefix,interp')
    args = p.parse_args()
    parts = args.parts.split(',')
    res_path = args.run / 'results.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    arr_path = args.run / 'analysis.npz'
    arrays = dict(np.load(arr_path)) if arr_path.exists() else {}
    recs_list = load_records(args.run)
    recs = {r['id']: r for r in recs_list}
    tok = args.run / 'tokenizer_audit.json'
    if tok.exists():
        results['tokenizer'] = json.loads(tok.read_text())
    spot = args.run / 'gen' / 'spotcheck.jsonl'
    if spot.exists():
        s = read_jsonl(spot)
        results['spotcheck'] = {'n': len(s), 'answer_equal': sum(r['answer_equal'] for r in s),
                                'text_equal': sum(r['text_equal'] for r in s)}
    if 'behaviour' in parts:
        results['behaviour'], a = behaviour(recs_list)
        arrays.update(a)
    if 'force' in parts and (args.run / 'force' / 'records.jsonl').exists():
        results['force'] = forcing(args.run)
    need_F = any(x in parts for x in ('probes', 'geometry', 'lens', 'visual', 't6', 'tens', 'prefix'))
    if need_F:
        F = load_features(args.features)
        print('features', F['resid'].shape, flush=True)
    if 'probes' in parts:
        results['probes'], a = probes(F, recs)
        arrays.update(a)
        print('probes done', flush=True)
    if 'geometry' in parts:
        results['geometry'], a = geometry(F, recs)
        arrays.update(a)
    if 'lens' in parts:
        meta = json.loads((args.run / 'experiment.json').read_text())
        from eva.inference import resolve_model
        from utils.project_paths import load_paths
        ck = resolve_model(meta['model']['model'], load_paths()['HF_HUB_CACHE'])
        P = readout_params(Path(ck), cv.FULL_LAYERS)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ck, local_files_only=True)
        digit_ids = [tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]
        P['W'] = P['unembed'][digit_ids]
        del P['unembed']
        results['lens'], a = lens_and_attribution(F, recs, P)
        arrays.update(a)
        print('lens done', flush=True)
    if 'visual' in parts:
        results['visual_transfer'] = visual_transfer(F, recs, args.features)
    if 'steer' in parts:
        results['steer'], a = steering(args.run)
        arrays.update(a)
    if 't6' in parts:
        results['t6_transfer'] = t6_transfer(F, recs)
    if 'interp' in parts:
        results['interp'], a = interpolation(args.run)
        arrays.update(a)
    if 'prefix' in parts:
        results['single_prefix'] = single_prefix_check(F, recs)
    if 'tens' in parts:
        results['tens_digit'] = tens_digit(F, recs)
    res_path.write_text(json.dumps(results, indent=1))
    np.savez(arr_path, **arrays)


if __name__ == '__main__':
    main()
