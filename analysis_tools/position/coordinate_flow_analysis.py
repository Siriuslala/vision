"""CPU analysis of experiment 3.11 (information flow of coordinate readout).

Reads the 3.11 run directory (coordinate_flow.py) and the 2.5e run directory (readout maps, answers),
and writes summary.json plus small npz tables used by coordinate_flow_figures.py:

  Q1  head-level patching shares (heads.jsonl); row/column structure of every head's attention at the
      horizontal and vertical queries (2.5e readout maps).
  Q2  answers decoded from attention peaks versus the model's answers (2.5e readout maps).
  Q3  image-KV subset shares (kv.jsonl), move versus decoder-only splice / content swap.
  Q4  per-emitted-value shares under different emission orders and attention mass per answer step.
  Q5  per-layer image attention share, linear readability by position, causal writers (flow/).
  Q6  decoded position fields and their relation to failures (field/).

Recovered fraction (see coordinate_flow.py): r = (LD_e - LD_A) / (LD_B - LD_A) per coordinate value;
values with |LD_B - LD_A| < MIN_GAP logits are excluded, and the per-condition summary is the median
over pairs together with a bootstrap 95% interval of the mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.position import coordinate_readout as cr
from analysis_tools.representation import sees_right as sr
from analysis_tools.representation.sees_right_analysis import axis_of, boot_mean, geometry, prop, wilson

FULL_LAYERS = cr.FULL_LAYERS
MIN_GAP = 2.0
Q2_CONDS = ('disc_point', 't6g06_rowcol', 't6g06_point', 't6g10_rowcol', 't6g10_point', 't6g20_rowcol',
            't6g20_point', 't5g03_rowcol', 't5g03_point', 't5g06_rowcol', 't5g06_point', 't5g10_rowcol',
            't5g10_point', 'rc_point')


def read_jsonl(path: Path) -> list[dict]:
    """Records of a jsonl file; duplicate ids (two processes sharing a work list) keep the first record."""
    out, seen = [], set()
    for r in (sr.read_jsonl(path) if path.exists() else []):
        if r.get('id') in seen:
            continue
        seen.add(r.get('id'))
        out.append(r)
    return out


# ----------------------------------------------------------------------------- shares

def shares(records: list[dict], kind: str | None = None, axis_match: bool = True) -> dict[str, dict]:
    """Recovered fractions per condition name, pooled over pairs and moved coordinate values.

    axis_match: keep only values whose axis equals the displacement axis (the value that moved).
    Returns name -> {'r': array aligned over the kept values (NaN where a record lacks the condition,
    e.g. an empty token group), 'meta': list of (record id, pair tag, value key, emission index, correct)}.
    """
    rows, meta = [], []
    for rec in records:
        for pr in rec['pairs']:
            if kind is not None and pr.get('kind') != kind:
                continue
            for k, v in enumerate(pr['values']):
                if axis_match and v['axis'] != pr['axis']:
                    continue
                if v['dA'] is None or v['dB'] is None or v['dA'] == v['dB']:
                    continue
                base = v['LD']['none']
                gap = v['LD_B'] - base
                if abs(gap) < MIN_GAP:
                    continue
                rows.append({name: (ld - base) / gap for name, ld in v['LD'].items()})
                meta.append((rec['id'], pr['tag'], v['key'], k, rec.get('correct')))
    names = sorted({n for r in rows for n in r})
    return {n: {'r': np.array([r.get(n, np.nan) for r in rows], float), 'meta': meta} for n in names}


def summarize(sh: dict, names, rng) -> dict:
    res = {}
    for n in names:
        if n not in sh:
            continue
        r = sh[n]['r']
        r = r[np.isfinite(r)]
        if len(r) == 0:
            continue
        b = boot_mean(np.clip(r, -3, 3), rng)
        res[n] = {'median': float(np.median(r)), 'mean': b['mean'], 'ci': b['ci'], 'n': int(len(r))}
    return res


# ----------------------------------------------------------------------------- Q1

def q1_heads(run: Path, rng) -> dict:
    recs = read_jsonl(run / 'heads.jsonl')
    out = {}
    conds = sorted({r['cond'] for r in recs})
    heads = [f'h{l}_{h}' for l in FULL_LAYERS for h in range(16)]
    for cond in conds:
        rs = [r for r in recs if r['cond'] == cond]
        out[cond] = {'n_records': len(rs)}
        for axis in ('h', 'v'):
            sub = [{**r, 'pairs': [p for p in r['pairs'] if p['axis'] == axis]} for r in rs]
            sh = shares(sub)
            if not sh:
                continue
            names = ['full_all', 'full_minus_L20'] + [f'L{l}' for l in FULL_LAYERS] + heads + \
                    [f'loo{l}_{h}' for l in (16, 20, 24) for h in range(16)]
            tab = summarize(sh, names, rng)
            single = np.array([tab[h]['mean'] if h in tab else np.nan for h in heads]).reshape(8, 16)
            loo = {k: v for k, v in tab.items() if k.startswith('loo')}
            order = np.argsort(-np.nan_to_num(single.ravel(), nan=-9))
            out[cond][axis] = {'table': {k: v for k, v in tab.items() if not k.startswith('h') and not k.startswith('loo')},
                               'single': single.tolist(),
                               'top': [(heads[i], float(single.ravel()[i])) for i in order[:12]],
                               'loo_drop': {k[3:]: tab['full_all']['mean'] - v['mean'] for k, v in loo.items()},
                               'sum_single': float(np.nansum(single)),
                               'n_heads_over_0.05': int(np.sum(single > 0.05)),
                               'n_values': int(tab['full_all']['n'])}
    return out


def q1_rowcol_attention(source: Path, conds=('t6g10_rowcol', 't6g20_rowcol', 't6g10_point', 'disc_point', 'rc_point'),
                        max_per_cond: int = 160) -> dict:
    """Row/column structure of every head's image attention at the horizontal and vertical queries.

    For each head and query: mass in the target token's row band (excluding the target), in its column
    band (excluding the target) and on the target; the expected values under uniform attention are
    reported for reference. Uses the 2.5e readout maps (pred queries only).
    """
    cases = {c['id']: c for c in json.loads((source / 'cases.json').read_text())}
    gen = {g['id']: g for g in sr.read_jsonl(source / 'gen.jsonl')}
    out = {}
    for cond in conds:
        recs = [r for r in sr.read_jsonl(source / 'readout' / 'records.jsonl')
                if r.get('status') == 'ok' and cases[r['id']]['cond'] == cond][:max_per_cond]
        keys = ('target', 'row', 'col', 'row_left', 'row_right', 'col_above', 'col_below')
        acc = {q: {**{k: [] for k in keys}, **{'u_' + k: [] for k in keys}} for q in ('h', 'v')}
        for r in recs:
            c = cases[r['id']]
            a = np.load(source / 'readout' / 'maps' / f"{r['id']}.npz")['attn'].astype(np.float32)  # (8,16,4,Nv)
            H, W = r['grid']
            T = np.array(r['target_tokens'])
            rows, cols = set(T // W), set(T % W)
            rr, cc = np.divmod(np.arange(H * W), W)
            tm = np.isin(np.arange(H * W), T)
            rowm = np.isin(rr, list(rows)) & ~tm
            colm = np.isin(cc, list(cols)) & ~tm
            p = a / a.sum(-1, keepdims=True)
            masks = {'target': tm, 'row': rowm, 'col': colm,
                     'row_left': rowm & (cc < min(cols)), 'row_right': rowm & (cc > max(cols)),
                     'col_above': colm & (rr < min(rows)), 'col_below': colm & (rr > max(rows))}
            tr0, tc0 = int(np.median(T // W)), int(np.median(T % W))
            for k in range(2):
                q = 'h' if axis_of(c['format'], k) == 0 else 'v'
                pk = p[:, :, k]
                prof = pk.reshape(8, 16, H, W)[:, :, tr0]                       # (8, 16, W) along the target row
                rel = np.full((8, 16, 81), np.nan)
                lo = 40 - tc0
                rel[:, :, lo:lo + W] = prof
                acc[q].setdefault('abs_profile', []).append(prof)
                acc[q].setdefault('rel_profile', []).append(rel)
                acc[q].setdefault('col0_mass', []).append(pk.reshape(8, 16, H, W)[:, :, :, 0].sum(-1))
                for name, m in masks.items():
                    acc[q][name].append(pk[..., m].sum(-1))
                    acc[q]['u_' + name].append(m.mean())
        out[cond] = {q: {k: (np.nanmean(np.stack(v), 0).tolist() if k[:2] != 'u_' else float(np.mean(v)))
                         for k, v in d.items() if k != 'abs_profile' or len({np.shape(x) for x in v}) == 1}
                     for q, d in acc.items()}
        for q in acc:
            out[cond][q]['u_col0'] = float(np.mean([1 / r['grid'][1] for r in recs]))
        out[cond]['n'] = len(recs)
    return out


# ----------------------------------------------------------------------------- Q2

def _decode_peak(p: np.ndarray, geo: dict, axis: int, local: bool) -> float:
    """Coordinate (0-1000) along `axis` from an image-normalized attention map p (Nv,)."""
    H, W = geo['grid']
    j = int(np.argmax(p))
    if not local:
        return float(geo['centers'][j, axis])
    r, c = divmod(j, W)
    idx = [rr * W + cc for rr in range(max(0, r - 1), min(H, r + 2)) for cc in range(max(0, c - 1), min(W, c + 2))]
    w = p[idx]
    return float((w[:, None] * geo['centers'][idx]).sum(0)[axis] / w.sum())


def _value_from_units(case: dict, axis: int, u: float) -> float:
    """Convert a 0-1000 coordinate to the answer's unit: 1-based ordinal for row/column, else unchanged."""
    if case['format'] != 'rowcol':
        return u
    t = case['truth']
    px = u * t['canvas'][axis] / 1000
    o = t['grid_origin'][axis]
    n = t['cols'] if axis == 0 else t['rows']
    return float(np.clip(np.floor((px - o) / t['cell']), 0, n - 1) + 1)


def _correct(case: dict, vals: list[float]) -> tuple[bool, float]:
    """Correctness and center error (0-1000) of an answer given in the case's own format/order."""
    t = case['truth']
    W, H = t['canvas']
    true_u = np.array([1000 * t['center'][0] / W, 1000 * t['center'][1] / H])
    if case['format'] == 'rowcol':
        r, c = int(vals[0]) - 1, int(vals[1]) - 1
        ok = (r, c) == tuple(t['target_rc'])
        return ok, float(np.linalg.norm(np.array(sr.cell_center_units(t, (r, c))) - true_u))
    u = np.array(vals, float)
    err = float(np.linalg.norm(u - true_u))
    px = u * np.array([W, H]) / 1000
    if case['set'] in ('t6', 't5'):
        return sr.point_cell(t, u) == tuple(t['target_rc']), err
    if case['set'] == 'disc':
        return bool(np.linalg.norm(px - np.array(t['center'])) <= t['diameter'] / 2), err
    b = t['box']
    return bool(b[0] <= px[0] <= b[2] and b[1] <= px[1] <= b[3]), err


def q2_attention_answers(source: Path, rng) -> dict:
    cases = {c['id']: c for c in json.loads((source / 'cases.json').read_text())}
    gen = {g['id']: g for g in sr.read_jsonl(source / 'gen.jsonl')}
    recs = [r for r in sr.read_jsonl(source / 'readout' / 'records.jsonl') if r.get('status') == 'ok'
            and cases[r['id']]['cond'] in Q2_CONDS]
    rows = []
    for r in recs:
        c = cases[r['id']]
        g = gen[r['id']]
        a = np.load(source / 'readout' / 'maps' / f"{r['id']}.npz")['attn'].astype(np.float32)[:, :, :2]  # pred queries
        geo = geometry(c, tuple(r['grid']))
        p = a / a.sum(-1, keepdims=True)                                                   # (8,16,2,Nv)
        dec = {}
        maps = {'H11': p[4, 11], 'L20mean': p[4].mean(0), 'L16mean': p[3].mean(0)}
        for name, m in maps.items():
            for local in (False, True):
                vals = [_value_from_units(c, axis_of(c['format'], k), _decode_peak(m[k], geo, axis_of(c['format'], k), local))
                        for k in range(2)]
                ok, err = _correct(c, vals)
                dec[f"{name}{'_local' if local else ''}"] = (ok, err)
        # every head's peak, used for the cross-fitted best head
        allh = np.zeros((8, 16, 2), bool)
        for li in range(8):
            for h in range(16):
                vals = [_value_from_units(c, axis_of(c['format'], k), _decode_peak(p[li, h, k], geo, axis_of(c['format'], k), True))
                        for k in range(2)]
                allh[li, h] = _correct(c, vals)[0], 0
        sc = g['score']
        model_err = sc.get('error_units')
        if model_err is None and sc.get('out_units') is not None:
            model_err = float(np.linalg.norm(np.array(sc['out_units']) - np.array(sc['true_units'])))
        rows.append({'id': r['id'], 'cond': c['cond'], 'model_ok': bool(sc['correct']), 'model_err': model_err,
                     'dec': dec, 'allh': allh[:, :, 0]})
    out = {}
    for cond in Q2_CONDS:
        rs = [x for x in rows if x['cond'] == cond]
        if not rs:
            continue
        idx = np.arange(len(rs))
        H = np.stack([x['allh'] for x in rs])                                              # (n, 8, 16)
        best = np.zeros(len(rs), bool)
        best_heads = []
        for half in (0, 1):
            tr, te = idx % 2 != half, idx % 2 == half
            li, h = np.unravel_index(np.argmax(H[tr].mean(0)), (8, 16))
            best[te] = H[te, li, h]
            best_heads.append(f'L{FULL_LAYERS[li]}H{h}')
        mok = np.array([x['model_ok'] for x in rs])
        tab = {'n': len(rs), 'model': prop(mok), 'best_heads': best_heads}
        for name in list(rs[0]['dec']) + ['best']:
            ok = best if name == 'best' else np.array([x['dec'][name][0] for x in rs])
            ent = {'all': prop(ok), 'model_correct': prop(ok[mok]), 'model_wrong': prop(ok[~mok]),
                   'both': int(np.sum(ok & mok)), 'only_attn': int(np.sum(ok & ~mok)), 'only_model': int(np.sum(~ok & mok))}
            if name != 'best':
                err = np.array([x['dec'][name][1] for x in rs])
                me = np.array([np.nan if x['model_err'] is None else x['model_err'] for x in rs])
                ent['err_attn_median'] = float(np.median(err))
                ent['err_model_median'] = float(np.nanmedian(me))
                ent['err_attn_median_wrong'] = float(np.median(err[~mok])) if (~mok).any() else float('nan')
                ent['err_model_median_wrong'] = float(np.nanmedian(me[~mok])) if (~mok).any() else float('nan')
            b, cc = ent['only_attn'], ent['only_model']
            from math import comb
            nn = b + cc
            ent['mcnemar_p'] = float(min(1.0, 2 * sum(comb(nn, i) for i in range(min(b, cc) + 1)) / 2 ** nn)) if nn else 1.0
            tab[name] = ent
        out[cond] = tab
    return out


# ----------------------------------------------------------------------------- Q3 / Q4

def q3_kv(run: Path, rng) -> dict:
    recs = read_jsonl(run / 'kv.jsonl')
    out = {}
    groups = ['all_img', 'text', 'obj', 'all_but_obj', 'inbox', 'row_left', 'row_right', 'row_same', 'below_cols',
              'below_left', 'below_right', 'below', 'above', 'ring1', 'ring2', 'ring3_4', 'ring5p']
    by_layer = [f'{g}@{l}' for g in ('obj', 'row_same', 'below', 'all_but_obj', 'all_img') for l in FULL_LAYERS]
    for cond in sorted({r['cond'] for r in recs}):
        rs = [r for r in recs if r['cond'] == cond]
        out[cond] = {'n_records': len(rs)}
        for kind in ('move', 'splice', 'swap'):
            for axis in ('h', 'v'):
                sub = [{**r, 'pairs': [p for p in r['pairs'] if p['axis'] == axis]} for r in rs]
                sh = shares(sub, kind)
                if not sh:
                    continue
                out[cond][f'{kind}_{axis}'] = summarize(sh, groups + by_layer, rng)
    return out


def q4_order(run: Path, rng) -> dict:
    """Per emitted value (emission index and key): target-object share, rest-of-image share, layer profile."""
    recs = read_jsonl(run / 'kv.jsonl')
    out = {}
    for cond in sorted({r['cond'] for r in recs}):
        rs = [r for r in recs if r['cond'] == cond]
        for kind in ('move', 'splice', 'swap'):
            sh = shares(rs, kind)
            if not sh:
                continue
            meta = sh['none']['meta']
            keys = sorted({(m[3], m[2]) for m in meta})
            ent = {}
            for k, key in keys:
                sel = np.array([(m[3], m[2]) == (k, key) for m in meta])
                row = {}
                for name in ['all_img', 'obj', 'all_but_obj', 'row_same', 'below'] + \
                        [f'{g}@{l}' for g in ('obj', 'all_but_obj', 'all_img') for l in FULL_LAYERS]:
                    if name in sh:
                        r = sh[name]['r'][sel]
                        r = r[np.isfinite(r)]
                        if len(r):
                            row[name] = {'median': float(np.median(r)), 'mean': float(np.mean(np.clip(r, -3, 3))), 'n': int(len(r))}
                ent[f'{k}:{key}'] = row
            out[f'{cond}|{kind}'] = ent
    # attention mass per answer step (run A): groups target, target_rows, target_cols, other_img, text
    att = {}
    for r in recs:
        f = run / 'kv_attn' / f"{r['id']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        m = z['mass']                                                                       # (8,16,Q,5)
        L = r['prompt_len']
        for k, v in enumerate(r['values_meta']):
            q = v['toks'][0]                      # tail index predicting the first digit
            att.setdefault(r['cond'], {}).setdefault(f"{k}:{v['key']}", []).append(m[:, :, q])
    out['attention'] = {c: {k: np.mean(v, 0).tolist() for k, v in d.items()} for c, d in att.items()}
    return out


# ----------------------------------------------------------------------------- Q5

def ridge_cv(X: np.ndarray, y: np.ndarray, folds: np.ndarray, alphas=np.logspace(0, 5, 11)) -> float:
    """Cross-validated R^2 of ridge regression (features standardized on the training fold).

    Uses the dual (kernel) form, since samples (a few hundred) are far fewer than features (4096); the
    penalty is chosen on each training fold by generalized cross-validation.
    """
    pred = np.zeros_like(y, dtype=float)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = y[tr].mean(0)
        yc = y[tr] - ym
        ev, U = np.linalg.eigh(A @ A.T)
        ev = np.clip(ev, 0, None)
        Uy = U.T @ yc
        best, best_gcv = None, np.inf
        for a in alphas:
            h = ev / (ev + a)
            fit = U @ (h[:, None] * Uy)
            gcv = np.mean((yc - fit) ** 2) / (1 - h.sum() / tr.sum()) ** 2
            if gcv < best_gcv:
                best, best_gcv = a, gcv
        coef = U @ (Uy / (ev + best)[:, None])                 # dual coefficients
        pred[te] = (B @ A.T) @ coef + ym
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return float(1 - (((y - pred) ** 2).sum(0) / ss).mean())


def q5_flow(run: Path) -> dict:
    root = run / 'flow'
    recs = [r for r in read_jsonl(root / 'records.jsonl') if r.get('status') == 'ok']
    out = {}
    for cond in sorted({r['cond'] for r in recs}):
        rs = [r for r in recs if r['cond'] == cond]
        Z = [np.load(root / f"{r['id']}.npz") for r in rs]
        resid = np.stack([z['resid'] for z in Z]).astype(np.float32)        # (n, 33, P, D)
        y = np.array([r['true_units'] for r in rs], float)
        folds = np.arange(len(rs)) % 5
        positions = rs[0]['positions']
        r2 = np.full((resid.shape[1], len(positions)), np.nan)
        for si in range(resid.shape[1]):
            for pi in range(len(positions)):
                r2[si, pi] = ridge_cv(resid[:, si, pi], y, folds)
        # image share of attention: post-image prompt text (mean), last prompt token, x and y queries
        share = {'text': [], 'prompt_last': [], 'q_x': [], 'q_y': [], 'answer_other': []}
        for r, z in zip(rs, Z):
            s = z['img_share'].astype(np.float32)                           # (8, 16, Q) for queries e .. end
            e, L = r['query_start'], r['prompt_len']
            qi = lambda pos: pos - e
            share['text'].append(s[:, :, :qi(L - 1)].mean(-1))
            share['prompt_last'].append(s[:, :, qi(L - 1)])
            share['q_x'].append(s[:, :, qi(r['q_x'])])
            share['q_y'].append(s[:, :, qi(r['q_y'])])
            other = [i for i in range(qi(L), s.shape[-1]) if i not in (qi(r['q_x']), qi(r['q_y']))]
            share['answer_other'].append(s[:, :, other].mean(-1))
        out[cond] = {'n': len(rs), 'positions': positions, 'r2': r2.tolist(),
                     'img_share': {k: np.mean(v, 0).tolist() for k, v in share.items()}}
    return out


# ----------------------------------------------------------------------------- Q6

def field_metrics(dec: np.ndarray, grid, target: list[int]) -> dict:
    """Disorder of a decoded field (Nv, 2) around the target.

    flips: fraction of adjacent token pairs within Chebyshev distance 3 of the target whose decoded
    coordinate does not increase along the corresponding axis; scale: local slope of decoded x (y)
    per token along the target's row (column) relative to the nominal 1000/W (1000/H); bias: decoded
    position of the target tokens minus their true centers (0-1000 units).
    """
    H, W = grid
    F = dec.reshape(H, W, 2)
    T = np.array(target)
    tr, tc = T // W, T % W
    r0, r1, c0, c1 = max(0, tr.min() - 3), min(H, tr.max() + 4), max(0, tc.min() - 3), min(W, tc.max() + 4)
    sub = F[r0:r1, c0:c1]
    fx = np.diff(sub[:, :, 0], axis=1) <= 0
    fy = np.diff(sub[:, :, 1], axis=0) <= 0
    cen = cr.token_centers(grid)
    rowx = F[int(np.median(tr)), c0:c1, 0]
    coly = F[r0:r1, int(np.median(tc)), 1]
    sx = np.polyfit(np.arange(len(rowx)), rowx, 1)[0] / (1000 / W) if len(rowx) > 1 else np.nan
    sy = np.polyfit(np.arange(len(coly)), coly, 1)[0] / (1000 / H) if len(coly) > 1 else np.nan
    bias = (dec[T] - cen[T]).mean(0)
    return {'flip_x': float(fx.mean()), 'flip_y': float(fy.mean()), 'scale_x': float(sx), 'scale_y': float(sy),
            'bias_x': float(bias[0]), 'bias_y': float(bias[1]),
            'mae': float(np.abs(dec - cen).mean())}


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return float((np.sum(pos[:, None] > neg[None]) + 0.5 * np.sum(pos[:, None] == neg[None])) / (len(pos) * len(neg)))


def q6_field(run: Path, source: Path) -> dict:
    cases = {c['id']: c for c in json.loads((source / 'cases.json').read_text())}
    gen = {g['id']: g for g in sr.read_jsonl(source / 'gen.jsonl')}
    out = {}
    for f in sorted((run / 'field').glob('*.npz')):
        z = np.load(f)
        ids = [str(i) for i in z['ids']]
        grids = z['grids']
        # lambda per site chosen by held-out MAE over all tokens
        res = {}
        for site in (0, 4, 8, 12, 16, 20):
            best = None
            for lam in (1e1, 1e2, 1e3, 1e4):
                key = f'{site}_{lam}'
                if key not in z:
                    continue
                D = z[key]
                mae = np.mean([np.abs(D[i] - cr.token_centers(tuple(grids[i]))).mean() for i in range(len(ids))])
                if best is None or mae < best[0]:
                    best = (mae, key)
            D = z[best[1]]
            mets = []
            for i, cid in enumerate(ids):
                c = cases[cid]
                target = cr.tokens_overlapping(c['truth']['box'], tuple(grids[i]))
                m = field_metrics(D[i], tuple(grids[i]), target)
                sc = gen[cid]['score']
                m.update({'correct': bool(sc['correct']), 'd_col': sc.get('d_col'), 'd_row': sc.get('d_row'),
                          'out_units': sc.get('out_units'), 'true_units': sc.get('true_units')})
                mets.append(m)
            ok = np.array([m['correct'] for m in mets])
            tab = {'lambda_key': best[1], 'mae_all_tokens': float(best[0])}
            for k in ('flip_x', 'flip_y', 'scale_x', 'scale_y', 'bias_x', 'bias_y', 'mae'):
                v = np.array([m[k] for m in mets])
                tab[k] = {'correct': float(np.nanmean(v[ok])) if ok.any() else None,
                          'wrong': float(np.nanmean(v[~ok])) if (~ok).any() else None,
                          'auroc_wrong': auroc(v[~np.isnan(v)], (~ok)[~np.isnan(v)])}
            dcol = np.array([m['d_col'] if m['d_col'] is not None else 0 for m in mets])
            if (dcol == 1).any():
                v = np.array([m['bias_x'] for m in mets])
                sel = (dcol == 1) | ok
                tab['colplus1_bias_x_auroc'] = auroc(v[sel], (dcol == 1)[sel])
                tab['colplus1_n'] = int((dcol == 1).sum())
            res[site] = tab
        out[f.stem] = res
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261020)
    p.add_argument('--parts', nargs='+', default=['q1', 'q1att', 'q2', 'q3', 'q4', 'q5', 'q6', 'spot'])
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    path = args.run / 'summary.json'
    summary = json.loads(path.read_text()) if path.exists() else {}
    if 'q1' in args.parts:
        summary['q1_heads'] = q1_heads(args.run, rng)
    if 'q1att' in args.parts:
        summary['q1_rowcol_attention'] = q1_rowcol_attention(args.source)
    if 'q2' in args.parts:
        summary['q2'] = q2_attention_answers(args.source, rng)
    if 'q3' in args.parts:
        summary['q3'] = q3_kv(args.run, rng)
    if 'q4' in args.parts:
        summary['q4'] = q4_order(args.run, rng)
    if 'q5' in args.parts:
        summary['q5'] = q5_flow(args.run)
    if 'q6' in args.parts:
        summary['q6'] = q6_field(args.run, args.source)
    if 'spot' in args.parts:
        sp = read_jsonl(args.run / 'spotcheck.jsonl')
        gen = read_jsonl(args.run / 'gen.jsonl')
        by = {}
        for g in gen:
            d = by.setdefault(g['cond'], [0, 0, 0])
            d[0] += 1
            d[1] += bool(g['score']['valid'])
            d[2] += bool(g['score']['correct'])
        summary['gen'] = {k: {'n': v[0], 'valid': v[1], 'correct': v[2], 'acc_ci': wilson(v[2], v[0])} for k, v in by.items()}
        summary['spotcheck'] = {'n': len(sp), 'answer_equal': sum(s['answer_equal'] for s in sp),
                                'text_equal': sum(s['text_equal'] for s in sp)}
    path.write_text(json.dumps(summary, indent=1, default=float))
    print('wrote', path)


if __name__ == '__main__':
    main()
