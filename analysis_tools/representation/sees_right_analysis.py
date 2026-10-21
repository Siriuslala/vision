"""Analysis of experiment 2.5e ("sees right but answers wrong"): attention metrics, probes, causal tests.

Reads the run directory written by `sees_right.py` and writes:
  metrics.npz        per-sample attention metrics (8 full layers x 17 readers x 4 queries)
  probe.npz          held-out probe decodings at the coordinate-digit states
  summary.json       all tables used by the results document
Attention metrics follow the definitions of results_2026-10-16_sees_right.md section 2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from analysis_tools.position import coordinate_readout as cr
from analysis_tools.representation import sees_right as sr

FULL_LAYERS = sr.FULL_LAYERS
RESID_LAYERS = sr.RESID_LAYERS
MEAN = 16                                 # reader index of the head-mean map
Z = 1.959963984540054
READERS = {'L20H11': (20, 11), 'L20mean': (20, MEAN), 'L16mean': (16, MEAN), 'L24mean': (24, MEAN),
           'L12mean': (12, MEAN), 'L16H11': (16, 11), 'L20H10': (20, 10)}
POINT_FAMILY = ('disc_point', 't6g06_point', 't6g10_point', 't6g20_point', 't5g03_point', 't5g06_point',
                't5g10_point', 'rc_point')
ROWCOL_FAMILY = ('t6g06_rowcol', 't6g10_rowcol', 't6g20_rowcol', 't6g20_rowcol_think', 't5g03_rowcol',
                 't5g06_rowcol', 't5g10_rowcol')
ALPHAS = np.logspace(0, 6, 13)


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return [c - h, c + h]


def prop(mask: np.ndarray) -> dict:
    k, n = int(np.sum(mask)), int(np.size(mask))
    return {'k': k, 'n': n, 'p': k / n if n else float('nan'), 'ci': wilson(k, n)}


def boot_mean(x: np.ndarray, rng, n_boot: int = 1000) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {'mean': float('nan'), 'ci': [float('nan')] * 2, 'n': 0}
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return {'mean': float(x.mean()), 'ci': [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))], 'n': int(len(x))}


def axis_of(fmt: str, k: int) -> int:
    """Image axis (0 = x / column, 1 = y / row) of coordinate value k."""
    return k if fmt == 'point' else 1 - k


# ----------------------------------------------------------------------------- geometry

def geometry(case: dict, grid: tuple[int, int]) -> dict:
    """Token geometry of one image: centers (0-1000), target / adjacent masks, token cells (grids)."""
    H, W = grid
    centers = cr.token_centers(grid)
    target = np.zeros(H * W, bool)
    target[cr.tokens_overlapping(case['truth']['box'], grid)] = True
    out = {'centers': centers, 'target': target, 'grid': (H, W)}
    if case['set'] in ('t6', 't5'):
        t = case['truth']
        px = centers * np.asarray(t['canvas'], float) / 1000
        ox, oy = t['grid_origin']
        col = np.floor((px[:, 0] - ox) / t['cell']).astype(int)
        row = np.floor((px[:, 1] - oy) / t['cell']).astype(int)
        inside = (row >= 0) & (row < t['rows']) & (col >= 0) & (col < t['cols'])
        row[~inside], col[~inside] = -99, -99
        tr, tc = t['target_rc']
        cheb = np.maximum(np.abs(row - tr), np.abs(col - tc))
        out.update({'cell_row': row, 'cell_col': col, 'adjacent': (cheb == 1) & inside & ~target})
    else:
        m = target.reshape(H, W)
        dil = m.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                dil |= np.roll(np.roll(m, dr, 0), dc, 1) & _valid_shift(H, W, dr, dc)
        out['adjacent'] = dil.ravel() & ~target
    return out


def _valid_shift(H: int, W: int, dr: int, dc: int) -> np.ndarray:
    v = np.ones((H, W), bool)
    if dr > 0:
        v[:dr] = False
    elif dr < 0:
        v[dr:] = False
    if dc > 0:
        v[:, :dc] = False
    elif dc < 0:
        v[:, dc:] = False
    return v


def attention_metrics(a: np.ndarray, geo: dict, case: dict) -> dict:
    """a: (..., Nv) softmax weights over the visual tokens. Returns metrics with the leading shape.

    tmass   share of the image attention on target tokens; E = tmass / (|T| / Nv)
    rank    1 + number of tokens whose weight exceeds the largest target weight (1: peak on target)
    pcls    peak token class: 0 target, 1 adjacent (cell at Chebyshev distance 1 / token ring), 2 elsewhere
    cen     attention centroid (x, y) in 0-1000; peak: peak-token center (x, y)
    cell    grids: (row, col) of the cell with the largest attention mass (-99 for the margin)
    """
    a = a.astype(np.float32)
    vis = a.sum(-1)
    p = a / np.maximum(vis[..., None], 1e-30)
    T = geo['target']
    tmass = p[..., T].sum(-1)
    tmax = a[..., T].max(-1)
    rank = 1 + (a > tmax[..., None]).sum(-1)
    peak = a.argmax(-1)
    pcls = np.where(T[peak], 0, np.where(geo['adjacent'][peak], 1, 2))
    # Axis sense: the peak token lies in a token column (axis 0) / token row (axis 1) spanned by the target.
    H, W = geo['grid']
    trow, tcol = np.where(T.reshape(H, W))
    in_col = np.isin(peak % W, np.unique(tcol))
    in_row = np.isin(peak // W, np.unique(trow))
    out = {'vis': vis, 'tmass': tmass, 'E': tmass / (T.mean()), 'rank': rank, 'pcls': pcls,
           'cen': p @ geo['centers'], 'peak': geo['centers'][peak], 'paxis': np.stack([in_col, in_row], -1)}
    if 'cell_row' in geo:
        t = case['truth']
        rows, cols = geo['cell_row'], geo['cell_col']
        inside = rows >= 0
        flat = np.where(inside, rows * t['cols'] + cols, t['rows'] * t['cols'])
        n_cells = t['rows'] * t['cols'] + 1
        cm = np.einsum('...j,jc->...c', p, np.eye(n_cells, dtype=np.float32)[flat])
        best = cm[..., :-1].argmax(-1)
        out['cell'] = np.stack([best // t['cols'], best % t['cols']], -1)
        tr, tc = t['target_rc']
        tgt = cm[..., tr * t['cols'] + tc]
        out['cell_rank'] = 1 + (cm[..., :-1] > tgt[..., None]).sum(-1)
    return out


# ----------------------------------------------------------------------------- loading

def load_run(run: Path) -> dict:
    cases = sr.load_cases(run)
    gen = {g['id']: g for g in sr.read_jsonl(run / 'gen.jsonl')}
    recs = {r['id']: r for r in sr.read_jsonl(run / 'readout' / 'records.jsonl')}
    return {'cases': cases, 'gen': gen, 'recs': recs}


def compute_metrics(run: Path, d: dict) -> dict:
    """Per-sample metrics for every (layer, reader, query); reader 16 is the head mean, the L20
    contribution-weighted aggregate is stored separately (key 'c20')."""
    rows = []
    for cid, r in d['recs'].items():
        if r['status'] != 'ok':
            continue
        case, g = d['cases'][cid], d['gen'][cid]
        maps = np.load(run / 'readout' / 'maps' / f'{cid}.npz')
        a = maps['attn'].astype(np.float32)                                    # (8, 16, 4, Nv)
        a = np.concatenate([a, a.mean(1, keepdims=True)], 1)                   # (8, 17, 4, Nv)
        geo = geometry(case, tuple(r['grid']))
        m = attention_metrics(a, geo, case)
        c20 = maps['contrib20'].astype(np.float32).sum(0)                      # (2, Nv) summed over heads
        mc = attention_metrics(c20, geo, case)
        rows.append({'id': cid, 'm': m, 'c20': mc})
    return rows


def flatten(rows: list[dict], d: dict) -> dict:
    """Stack per-sample metrics into arrays; sample metadata as parallel arrays."""
    keys = ['vis', 'tmass', 'E', 'rank', 'pcls', 'cen', 'peak', 'paxis']
    out = {k: np.stack([r['m'][k] for r in rows]) for k in keys}
    out.update({f'c20_{k}': np.stack([r['c20'][k] for r in rows]) for k in keys})
    cell = np.full((len(rows),) + rows[0]['m']['tmass'].shape + (2,), -99, int)
    cell_rank = np.full((len(rows),) + rows[0]['m']['tmass'].shape, -1, int)
    for i, r in enumerate(rows):
        if 'cell' in r['m']:
            cell[i], cell_rank[i] = r['m']['cell'], r['m']['cell_rank']
    out['cell'], out['cell_rank'] = cell, cell_rank
    meta = {'id': [], 'cond': [], 'correct': [], 'truth': [], 'output': [], 'fmt': [], 'set': [],
            'true_rc': [], 'pred_rc': [], 'image_key': [], 'canvas': []}
    for r in rows:
        c, s = d['cases'][r['id']], d['gen'][r['id']]['score']
        meta['id'].append(r['id']); meta['cond'].append(c['cond']); meta['correct'].append(bool(s['correct']))
        meta['truth'].append(s['true_units']); meta['output'].append(s['out_units']); meta['fmt'].append(c['format'])
        meta['set'].append(c['set']); meta['image_key'].append(c['image_key']); meta['canvas'].append(c['truth']['canvas'])
        meta['true_rc'].append(c['truth'].get('target_rc', [-1, -1])); meta['pred_rc'].append(s.get('pred_rc', [-1, -1]))
    out.update({k: np.asarray(v) for k, v in meta.items()})
    return out


# ----------------------------------------------------------------------------- summaries

def cross_fitted_best(F: dict, sel: np.ndarray) -> np.ndarray:
    """Per-sample (layer index, head) of the head with the largest mean target mass on correct answers of
    the other half (halves by sample parity within the condition); pred queries, both values."""
    idx = np.where(sel)[0]
    choice = np.zeros((len(F['id']), 2), int)
    half = np.arange(len(idx)) % 2
    for h in (0, 1):
        train = idx[(half != h) & F['correct'][idx]]
        if len(train) == 0:
            train = idx[half != h]
        score = F['tmass'][train][:, :, :16, :2].mean((0, 3))                 # (8, 16)
        li, hh = np.unravel_index(np.argmax(score), score.shape)
        choice[idx[half == h]] = (li, hh)
    return choice


def reader_values(F: dict, name: str, best: np.ndarray, key: str) -> np.ndarray:
    """(n, 4, ...) metric of one reader for every sample."""
    if name == 'L20contrib':
        v = F[f'c20_{key}']                                                    # (n, 2, ...)
        fill = np.nan if v.dtype.kind == 'f' else (False if v.dtype == bool else -1)
        return np.concatenate([v, np.full_like(v, fill)], 1)
    if name == 'best':
        n = len(F['id'])
        return F[key][np.arange(n), best[:, 0], best[:, 1]]
    l, h = READERS[name]
    return F[key][:, FULL_LAYERS.index(l), h]


def condition_table(F: dict, rng) -> dict:
    out = {}
    readers = list(READERS) + ['L20contrib', 'best']
    for cond in sorted(set(F['cond'])):
        sel = F['cond'] == cond
        best = cross_fitted_best(F, sel)
        fmt = F['fmt'][sel][0]
        entry = {'n': int(sel.sum()), 'n_correct': int(F['correct'][sel].sum()), 'readers': {},
                 'best_heads': sorted({f'L{FULL_LAYERS[l]}H{h}' for l, h in best[sel]})}
        for name in readers:
            vals = {k: reader_values(F, name, best, k)[sel] for k in ('tmass', 'rank', 'pcls', 'cen', 'peak', 'E', 'paxis')}
            cell = reader_values(F, name, best, 'cell')[sel] if name != 'L20contrib' else None
            res = {}
            for corr in (True, False):
                cm = F['correct'][sel] == corr
                if cm.sum() == 0:
                    continue
                per_q = {}
                for q in range(4 if name != 'L20contrib' else 2):
                    k = q % 2
                    ax = axis_of(fmt, k)
                    truth = F['truth'][sel][cm][:, ax]
                    outp = F['output'][sel][cm][:, ax]
                    cen = vals['cen'][cm][:, q, ax]
                    pk = vals['peak'][cm][:, q, ax]
                    rec = {'tmass': boot_mean(vals['tmass'][cm][:, q], rng), 'E_median': float(np.median(vals['E'][cm][:, q])),
                           'rank_median': float(np.median(vals['rank'][cm][:, q])),
                           'peak_target': prop(vals['pcls'][cm][:, q] == 0), 'peak_adjacent': prop(vals['pcls'][cm][:, q] == 1),
                           'peak_elsewhere': prop(vals['pcls'][cm][:, q] == 2),
                           'peak_axis_on_target': prop(vals['paxis'][cm][:, q, ax].astype(bool)),
                           'cen_err_truth': boot_mean(np.abs(cen - truth), rng), 'cen_err_output': boot_mean(np.abs(cen - outp), rng),
                           'peak_err_truth': boot_mean(np.abs(pk - truth), rng), 'peak_err_output': boot_mean(np.abs(pk - outp), rng)}
                    if not corr:
                        rec['peak_closer_truth'] = prop(np.abs(pk - truth) < np.abs(pk - outp))
                        rec['cen_closer_truth'] = prop(np.abs(cen - truth) < np.abs(cen - outp))
                    if cell is not None and F['set'][sel][0] in ('t6', 't5'):
                        ai = 1 - ax                                            # cell array is (row, col)
                        cq = cell[cm][:, q]
                        tr = F['true_rc'][sel][cm]
                        pr = F['pred_rc'][sel][cm]
                        rec['cell_is_target'] = prop((cq[:, 0] == tr[:, 0]) & (cq[:, 1] == tr[:, 1]))
                        rec['cell_axis_is_truth'] = prop(cq[:, ai] == tr[:, ai])
                        if not corr:
                            rec['cell_axis_is_output'] = prop(cq[:, ai] == pr[:, ai])
                            wrong_axis = pr[:, ai] != tr[:, ai]
                            rec['axis_wrong_n'] = int(wrong_axis.sum())
                            if wrong_axis.any():
                                rec['axis_wrong_cell_is_truth'] = prop(cq[wrong_axis, ai] == tr[wrong_axis, ai])
                                rec['axis_wrong_cell_is_output'] = prop(cq[wrong_axis, ai] == pr[wrong_axis, ai])
                    per_q[['pred0', 'pred1', 'tok0', 'tok1'][q]] = rec
                res['correct' if corr else 'wrong'] = per_q
            entry['readers'][name] = res
        # Layer profile per query (pred0, pred1): target share, peak on target, peak in the target's token
        # column / row along the queried axis; heads 0-15 and the head mean (index 16).
        prof = {}
        fmt0 = F['fmt'][sel][0]
        for corr in (True, False):
            cm = sel & (F['correct'] == corr)
            if cm.sum():
                pax = np.stack([F['paxis'][cm][:, :, :, q, axis_of(fmt0, q)] for q in (0, 1)], -1)   # (n, 8, 17, 2)
                prof['correct' if corr else 'wrong'] = {
                    'tmass': F['tmass'][cm][:, :, :, :2].mean(0).tolist(),
                    'peak_target': (F['pcls'][cm][:, :, :, :2] == 0).mean(0).tolist(),
                    'peak_axis': pax.mean(0).tolist()}
        entry['layer_profile'] = prof
        out[cond] = entry
    return out


# ----------------------------------------------------------------------------- probes

def ridge_fit_predict(Xtr: np.ndarray, ytr, Xte: np.ndarray):
    """Standardized ridge with the penalty chosen by exact leave-one-out error over ALPHAS.

    ytr may be one label vector or a list of label vectors sharing the same features (one
    decomposition). The dual form uses the eigendecomposition of the n x n Gram matrix.
    Returns (predictions, alpha) or a list of them.
    """
    single = isinstance(ytr, np.ndarray)
    ys = [ytr] if single else list(ytr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = ((Xtr - mu) / sd).astype(np.float32)
    B = ((Xte - mu) / sd).astype(np.float32)
    lam, U = np.linalg.eigh((A @ A.T).astype(np.float64))
    lam = np.clip(lam, 0, None)
    K_te = (B @ A.T).astype(np.float64)
    out = []
    for y in ys:
        ym = y.mean()
        Uy = U.T @ (y - ym)
        best, best_err = None, np.inf
        for alpha in ALPHAS:
            f = lam / (lam + alpha)
            fit = U @ (f * Uy)
            hdiag = (U ** 2 * f).sum(1)
            loo = ((y - ym - fit) / np.maximum(1 - hdiag, 1e-6)) ** 2
            if loo.mean() < best_err:
                best, best_err = alpha, loo.mean()
        coef = U @ (Uy / (lam + best))                       # dual coefficients
        out.append((K_te @ coef + ym, float(best)))
    return out[0] if single else out


def probe_labels(F: dict, i: np.ndarray, k: int, kind: str, d: dict) -> np.ndarray:
    """Labels for value k: point family in 0-1000 units, row/column family as 1-based indices."""
    fmt = F['fmt'][i[0]]
    ax = axis_of(fmt, k)
    if fmt == 'point':
        return (F['truth'] if kind == 'truth' else F['output'])[i, ax]
    ai = 1 - ax
    return (F['true_rc'] if kind == 'truth' else F['pred_rc'])[i, ai] + 1.0


def run_probes(run: Path, F: dict, d: dict, seed: int, n_folds: int = 5) -> dict:
    """Held-out probe decodings at the pred queries.

    'pos': trained on correct answers only with truth labels (truth = output on those samples), applied
    to every held-out sample. 'truth_all' / 'output_all': trained on all samples with truth / output labels.
    Folds are drawn by image within each family.
    """
    resid = {}
    for i, cid in enumerate(F['id']):
        resid[i] = np.load(run / 'readout' / 'maps' / f'{cid}.npz')['resid'].astype(np.float32)   # (6, 3, D)
    rng = np.random.default_rng(seed)
    dec = {kind: np.full((len(F['id']), len(RESID_LAYERS), 2), np.nan) for kind in ('pos', 'truth_all', 'output_all')}
    alphas = {}
    for fam_name, fam in (('point', POINT_FAMILY), ('rowcol', ROWCOL_FAMILY)):
        idx = np.where(np.isin(F['cond'], fam))[0]
        if len(idx) == 0:
            continue
        keys = sorted(set(F['image_key'][idx]))
        fold_of = dict(zip(keys, rng.permutation(len(keys)) % n_folds))
        folds = np.asarray([fold_of[k] for k in F['image_key'][idx]])
        for li in range(len(RESID_LAYERS)):
            for k in range(2):
                X = np.stack([resid[i][li, k] for i in idx])
                ytruth = probe_labels(F, idx, k, 'truth', d)
                yout = probe_labels(F, idx, k, 'output', d)
                corr = F['correct'][idx]
                for f in range(n_folds):
                    te = folds == f
                    tr_pos = (~te) & corr
                    p, a = ridge_fit_predict(X[tr_pos], ytruth[tr_pos], X[te])
                    dec['pos'][idx[te], li, k] = p
                    alphas[f'{fam_name}_L{RESID_LAYERS[li]}_v{k}_f{f}'] = a
                    (pt, _), (po, _) = ridge_fit_predict(X[~te], [ytruth[~te], yout[~te]], X[te])
                    dec['truth_all'][idx[te], li, k] = pt
                    dec['output_all'][idx[te], li, k] = po
    np.savez_compressed(run / 'probe.npz', **dec, ids=F['id'])
    return {'decoded': dec, 'alphas': alphas}


def target_halfwidth(d: dict, cid: str, k: int) -> float:
    """Half extent of the target along the axis of value k, in the probe's label units."""
    c = d['cases'][cid]
    fmt = c['format']
    ax = axis_of(fmt, k)
    if fmt == 'rowcol':
        return 0.5
    W, H = c['truth']['canvas']
    size = W if ax == 0 else H
    if c['set'] in ('t6', 't5'):
        return 0.5 * c['truth']['cell'] * 1000 / size
    b = c['truth']['box']
    return 0.5 * (b[2 + ax] - b[ax]) * 1000 / size


def probe_table(F: dict, P: dict, d: dict, rng) -> dict:
    out = {}
    dec = P['decoded']
    for cond in sorted(set(F['cond'])):
        sel = np.where(F['cond'] == cond)[0]
        entry = {}
        for li, layer in enumerate(RESID_LAYERS):
            per = {}
            for k in range(2):
                yt = probe_labels(F, sel, k, 'truth', d)
                yo = probe_labels(F, sel, k, 'output', d)
                hw = np.asarray([target_halfwidth(d, F['id'][i], k) for i in sel])
                rec = {}
                for corr in (True, False):
                    m = F['correct'][sel] == corr
                    if not m.any():
                        continue
                    p = dec['pos'][sel, li, k][m]
                    r = {'n': int(m.sum()), 'err_truth': boot_mean(np.abs(p - yt[m]), rng),
                         'within_target': prop(np.abs(p - yt[m]) < hw[m])}
                    if not corr:
                        differs = yo[m] != yt[m]
                        r['err_output'] = boot_mean(np.abs(p - yo[m]), rng)
                        r['axis_wrong_n'] = int(differs.sum())
                        if differs.any():
                            r['closer_truth'] = prop(np.abs(p - yt[m])[differs] < np.abs(p - yo[m])[differs])
                            r['within_target_axis_wrong'] = prop((np.abs(p - yt[m]) < hw[m])[differs])
                            r['err_truth_axis_wrong'] = boot_mean(np.abs(p - yt[m])[differs], rng)
                            r['err_output_axis_wrong'] = boot_mean(np.abs(p - yo[m])[differs], rng)
                            pt = dec['truth_all'][sel, li, k][m][differs]
                            po = dec['output_all'][sel, li, k][m][differs]
                            r['sym_err_truth_probe'] = boot_mean(np.abs(pt - yt[m][differs]), rng)
                            r['sym_err_output_probe'] = boot_mean(np.abs(po - yo[m][differs]), rng)
                    rec['correct' if corr else 'wrong'] = r
                per[f'v{k}'] = rec
            entry[f'L{layer}'] = per
        out[cond] = entry
    return out


# ----------------------------------------------------------------------------- readout bias

def bias_table(F: dict, P: dict, rng) -> dict:
    """Grids: does 'attended position + a fixed readout bias' reproduce the output along each axis?

    For axis index j (1-based row or column) of the truth t, the attended cell a (L20 H11 / layer-20 mean,
    pred query), the probe cell p (correct-only probe, layer 20) and the output o: cross-fitted bias
    b(j) = most frequent o - j among samples of the other half with the same j; prediction j + b(j).
    """
    out = {}
    L20 = RESID_LAYERS.index(20)
    for cond in sorted(set(F['cond'])):
        sel = np.where(F['cond'] == cond)[0]
        if F['set'][sel[0]] not in ('t6', 't5'):
            continue
        fmt = F['fmt'][sel[0]]
        entry = {}
        for k in range(2):
            ax = axis_of(fmt, k)
            ai = 1 - ax
            t = F['true_rc'][sel, ai]
            o = F['pred_rc'][sel, ai]
            sources = {'truth': t}
            for name, (l, h) in (('L20H11', (20, 11)), ('L20mean', (20, MEAN))):
                sources[name] = F['cell'][sel, FULL_LAYERS.index(l), h, k, ai]
            if fmt == 'rowcol':
                sources['probe'] = np.rint(P['decoded']['pos'][sel, L20, k] - 1).astype(int)
            res = {}
            half = np.arange(len(sel)) % 2
            for name, j in sources.items():
                if j is None:
                    continue
                pred = np.zeros_like(o)
                for h in (0, 1):
                    tr, te = half != h, half == h
                    diffs = {}
                    for jj, oo in zip(j[tr], o[tr]):
                        diffs.setdefault(int(jj), []).append(int(oo - jj))
                    allv = np.bincount(np.asarray([x for v in diffs.values() for x in v]) + 50).argmax() - 50 if diffs else 0
                    for i in np.where(te)[0]:
                        v = diffs.get(int(j[i]))
                        b = np.bincount(np.asarray(v) + 50).argmax() - 50 if v else allv
                        pred[i] = j[i] + b
                wrong = F['correct'][sel] == False  # noqa: E712
                res[name] = {'exact_all': prop(pred == o), 'exact_wrong': prop(pred[wrong] == o[wrong]),
                             'source_eq_truth': prop(j == t), 'source_eq_output_wrong': prop(j[wrong] == o[wrong])}
            # Bias curve: mean (o - t) per true index.
            curve = {int(v): float(np.mean(o[t == v] - v)) for v in sorted(set(t.tolist()))}
            entry[f'v{k}'] = {'models': res, 'bias_curve_truth': curve}
        out[cond] = entry
    return out


# ----------------------------------------------------------------------------- causal

def causal_table(run: Path, d: dict, rng) -> dict:
    recs = sr.read_jsonl(run / 'causal' / 'records.jsonl')
    out, per_rows = {}, []
    for r in recs:
        case = d['cases'][r['id']]
        W, H = case['truth']['canvas']
        base = {}
        for row in r['rows']:
            base.setdefault((row['move'], 'patch'), None)
        for row in r['rows']:
            if row['condition'] in ('patch_none', 'kv_none'):
                base[(row['move'], 'patch' if row['condition'] == 'patch_none' else 'kv')] = row['score']
        bfree = {row['move']: row['score'] for row in r['rows'] if row['condition'] == 'B_free'}
        for row in r['rows']:
            group = 'kv' if not row['condition'].startswith(('patch_', 'B_free')) else 'patch'
            b = base.get((row['move'], group))
            s = row['score']
            ax = 0 if row['axis'] == 'h' else 1
            if case['format'] == 'rowcol':
                ai = 1 - ax
                dsh = row['cells']
                val = lambda sc: sc['pred_rc'][ai] if sc.get('valid') else None   # noqa: E731
            else:
                size = W if ax == 0 else H
                shift_tok = row['dc'] if ax == 0 else row['dr']
                dsh = shift_tok * cr.TOKEN_PIXELS * 1000 / size
                val = lambda sc: sc['out_units'][ax] if sc.get('valid') else None  # noqa: E731
            if b is None or val(b) is None or val(s) is None:
                w = np.nan
            else:
                w = (val(s) - val(b)) / dsh
            match_b = np.nan
            if row['move'] in bfree and val(bfree[row['move']]) is not None and val(s) is not None:
                if case['format'] == 'rowcol':
                    match_b = float(s['pred_rc'] == bfree[row['move']]['pred_rc'])
                else:
                    match_b = float(np.max(np.abs(np.asarray(s['out_units']) - bfree[row['move']]['out_units'])) <= 10)
            per_rows.append({'id': r['id'], 'cond': r['cond'], 'baseline_correct': r['baseline_correct'], 'move': row['move'],
                             'axis': row['axis'], 'condition': row['condition'], 'w': w, 'match_B': match_b,
                             'valid': bool(s.get('valid')), 'attn20': row.get('attn20', {})})
    conds = sorted({p['cond'] for p in per_rows})
    for cond in conds:
        entry = {}
        names = sorted({p['condition'] for p in per_rows if p['cond'] == cond})
        for name in names:
            rows = [p for p in per_rows if p['cond'] == cond and p['condition'] == name]
            e = {'n': len(rows), 'valid': prop(np.asarray([p['valid'] for p in rows]))}
            for sub, f in (('all', lambda p: True), ('base_correct', lambda p: p['baseline_correct']),
                           ('base_wrong', lambda p: not p['baseline_correct'])):
                rr = [p for p in rows if f(p)]
                if not rr:
                    continue
                w = np.asarray([p['w'] for p in rr], float)
                mb = np.asarray([p['match_B'] for p in rr], float)
                e[sub] = {'w': boot_mean(w, rng), 'w_h': boot_mean(np.asarray([p['w'] for p in rr if p['axis'] == 'h'], float), rng),
                          'w_v': boot_mean(np.asarray([p['w'] for p in rr if p['axis'] == 'v'], float), rng),
                          'match_B': prop(mb[np.isfinite(mb)] == 1) if np.isfinite(mb).any() else None}
            att = [p['attn20'] for p in rows if p['attn20']]
            if att:
                e['attn20'] = {key: float(np.mean([a[v][key] for a in att for v in a])) for key in ('T_H11', 'S_H11', 'T_mean', 'S_mean')}
            entry[name] = e
        out[cond] = entry
    return {'table': out, 'rows': per_rows}


# ----------------------------------------------------------------------------- main

def baseline_table(d: dict) -> dict:
    out = {}
    for cid, g in d['gen'].items():
        c = d['cases'][cid]
        e = out.setdefault(c['cond'], {'n': 0, 'valid': 0, 'correct': 0, 'truncated': 0, 'readout_ok': 0,
                                       'adjacent_wrong': 0, 'wrong': 0})
        e['n'] += 1
        s = g['score']
        e['valid'] += bool(s.get('valid'))
        e['correct'] += bool(s.get('correct'))
        e['truncated'] += g['finish_reason'] != 'eos'
        e['readout_ok'] += d['recs'].get(cid, {}).get('status') == 'ok'
        if s.get('valid') and not s.get('correct'):
            e['wrong'] += 1
            if 'd_row' in s and max(abs(s['d_row']), abs(s['d_col'])) == 1:
                e['adjacent_wrong'] += 1
    for e in out.values():
        e['acc'] = prop(np.asarray([1] * e['correct'] + [0] * (e['n'] - e['correct'])))
    return out


def spotcheck_table(run: Path) -> dict:
    rows = sr.read_jsonl(run / 'spotcheck.jsonl')
    return {'n': len(rows), 'answer_equal': sum(r['answer_equal'] for r in rows),
            'text_equal': sum(r['text_equal'] for r in rows), 'correct_equal': sum(r['correct_equal'] for r in rows)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reuse', action='store_true', help='reuse metrics.npz and probe.npz (e.g. after new causal records)')
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    d = load_run(args.run)
    if args.reuse and (args.run / 'metrics.npz').exists() and (args.run / 'probe.npz').exists():
        F = dict(np.load(args.run / 'metrics.npz'))
        P = {'decoded': {k: v for k, v in np.load(args.run / 'probe.npz').items() if k != 'ids'}}
        if list(np.load(args.run / 'probe.npz')['ids']) != list(F['id']):
            raise ValueError('probe.npz does not match metrics.npz')
    else:
        rows = compute_metrics(args.run, d)
        F = flatten(rows, d)
        np.savez_compressed(args.run / 'metrics.npz', **{k: v for k, v in F.items()})
        P = run_probes(args.run, F, d, args.seed)
    summary = {'baseline': baseline_table(d), 'spotcheck': spotcheck_table(args.run),
               'attention': condition_table(F, rng), 'probe': probe_table(F, P, d, rng), 'bias': bias_table(F, P, rng)}
    if (args.run / 'causal' / 'records.jsonl').exists():
        C = causal_table(args.run, d, rng)
        summary['causal'] = C['table']
        (args.run / 'causal_rows.json').write_text(json.dumps(C['rows']))
    (args.run / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print('wrote', args.run / 'summary.json')


if __name__ == '__main__':
    main()
