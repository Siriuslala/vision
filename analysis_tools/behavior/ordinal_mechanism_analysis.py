"""CPU analysis of experiment 1.5 (ordinal mechanism).

  directions -- encoding directions (coordinate, count) at the long-axis ordinal state, cross-fitted by base parity,
                plus the 3.6 point-prompt coordinate direction and norm-matched random directions -> steer/
  analyze    -- route analysis of the paired variants, reasoning strategies, probes with control tasks, steering
                summary, cheap models of the +1 onset (1.4 data) and row biases (1.3, 1.3 supplement, 1.4, 1.5)

Outputs go to <output>/analysis/ (summary.json, tables.md, figures/).
"""
from __future__ import annotations

import os
# Cap BLAS threads before numpy loads: on the shared 128-core host, unbounded OpenBLAS threading made a
# 960 x 4096 Gram product take minutes instead of a second.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '8')

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from analysis_tools.behavior import ordinal_mechanism as om
from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.behavior.decomposition_analysis import wilson

STEER_LAYERS = (16, 20, 24, 28)
STEER_LABELS = ('none', 'a+3', 'a-3', 'a+10', 'a-10', 'm+1', 'm-1', 'm+3', 'm-3', 'x36n+', 'x36n-', 'rand+', 'rand-')
"""a: linear encoding direction of the model's answer (multiples of one ordinal unit); m: answer-conditional mean
difference mu(a + d) - mu(a) (Gaussian kernel over answers, bandwidth 1); x36n: 3.6 point-prompt x direction with
the norm of m+3 (H only); rand: random direction with the norm of m+3."""
MEAN_BANDWIDTH = 1.0
PROBE_TARGETS = ('unit', 'slot', 'count', 'answer', 'ratio')


def prop(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {'k': int(k), 'n': int(n), 'p': k / n if n else float('nan'), 'ci': [lo, hi]}


def ptxt(s: dict) -> str:
    return f"{s['p']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]" if s['n'] else '–'


# ----------------------------------------------------------------------------- loading

def load_rows(output: Path) -> list[dict]:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    rows = []
    for mode in om.MODES:
        for p in sorted((output / 'gen').glob(f'results_{mode}*.jsonl')):
            for r in read_jsonl(p):
                c = cases[r['id']]
                rows.append({**c, **om.score(c, r), 'response': r['response'], 'finish_reason': r['finish_reason'],
                             'n_tokens': r['n_tokens']})
    return rows


def long_role(axis: str) -> int:
    """Index into ROLES of the state that predicts the long-axis ordinal (col0 for H, row0 for V)."""
    return 2 if axis == 'H' else 1


def load_features(output: Path, conditions: list[str], mode: str, layers=None, roles=None):
    """Concatenate feature files; returns ids, H (N, L, R, D) float32 and the layer list."""
    ids, Hs, lay = [], [], None
    for cond in conditions:
        path = output / 'features' / f'{cond}_{mode}.npz'
        if not path.exists():
            continue
        z = np.load(path)
        lay = [int(v) for v in z['layers']]
        H = z['H']
        li = [lay.index(l) for l in layers] if layers is not None else slice(None)
        H = H[:, li]
        if roles is not None:
            H = H[:, :, roles]
        ids += [str(i) for i in z['ids']]
        Hs.append(H.astype(np.float32))
    if not Hs:
        return [], None, None
    return ids, np.concatenate(Hs), (layers if layers is not None else lay)


# ----------------------------------------------------------------------------- directions

def encoding_directions(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """OLS encoding directions: X (N, D) ~ Z (N, P) + intercept; returns (P, D) partial slopes."""
    Zc = Z - Z.mean(0)
    Xc = X - X.mean(0)
    return np.linalg.lstsq(Zc, Xc, rcond=None)[0]


def stage_directions(output: Path, mode: str, coord_value: Path) -> None:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    rows_by_id = {r['id']: r for r in load_rows(output)}
    beta36 = np.load(coord_value / 'steer' / 'directions.npz')
    l36 = [int(v) for v in beta36['layers']]
    rng = np.random.default_rng(0)
    all_ids, all_vecs, meta = [], [], []
    conds = sorted({c['condition'] for c in cases.values() if c['mode'] == mode})
    for cond in conds:
        ids, H, _ = load_features(output, [cond], mode, layers=list(STEER_LAYERS))
        if not ids:
            continue
        cs = [cases[i] for i in ids]
        axis, cell = om.parse_condition(cond)
        role = long_role(axis)
        unit = np.array([c['truth']['unit'] for c in cs])
        count = np.array([c['truth']['count'] for c in cs], float)
        parity = np.array([c['base'] % 2 for c in cs])
        width = cs[0]['truth']['canvas'][0 if axis == 'H' else 1]
        cell_units = cell / width * 1000
        targets = [j for j, c in enumerate(cs) if c['variant'] == 'base']
        answer = np.array([float(rows_by_id[i]['answer']) if rows_by_id[i]['answer'] is not None else np.nan
                           for i in ids])
        # two folds by base parity: a target's directions come from the bases of the other parity.
        # 'a' is the encoding direction of the model's own answer (one ordinal unit) over all variants;
        # unit and count are nearly collinear across variants, so their partial directions are reported only
        # as diagnostics (cos_x_k) and not used for steering.
        fold = {}
        for p in (0, 1):
            train = (parity != p) & ~np.isnan(answer)
            fold[p] = [(encoding_directions(H[train, li, role], answer[train, None])[0],
                        encoding_directions(H[train, li, role], np.stack([unit[train], count[train]], 1)))
                       for li in range(len(STEER_LAYERS))]
        for j in targets:
            if np.isnan(answer[j]):
                continue
            vecs = np.zeros((len(STEER_LAYERS), len(STEER_LABELS), H.shape[-1]), np.float32)
            other = (parity != parity[j]) & ~np.isnan(answer)
            for li, layer in enumerate(STEER_LAYERS):
                ba, B = fold[parity[j]][li]

                def mean_at(v, li=li):
                    w = np.exp(-0.5 * ((answer[other] - v) / MEAN_BANDWIDTH) ** 2)
                    return (w[:, None] * H[other, li, role]).sum(0) / w.sum()

                here = mean_at(answer[j])
                m = {d: mean_at(answer[j] + d) - here for d in (1, -1, 3, -3)}
                ref = np.linalg.norm(m[3])
                r = rng.standard_normal(H.shape[-1]).astype(np.float32)
                r *= ref / np.linalg.norm(r)
                b36 = beta36['beta'][l36.index(layer), 0]
                b36n = b36 / np.linalg.norm(b36) * ref if axis == 'H' else np.zeros_like(ba)
                table = {'none': 0 * ba, 'a+3': 3 * ba, 'a-3': -3 * ba, 'a+10': 10 * ba, 'a-10': -10 * ba,
                         'm+1': m[1], 'm-1': m[-1], 'm+3': m[3], 'm-3': m[-3], 'x36n+': b36n, 'x36n-': -b36n,
                         'rand+': r, 'rand-': -r}
                vecs[li] = np.stack([table[k] for k in STEER_LABELS])
                if j == targets[0]:
                    bx, bk = B[0] * cell_units, B[1]
                    meta.append({'condition': cond, 'layer': layer, 'norm_a': float(np.linalg.norm(ba)),
                                 'norm_m+1': float(np.linalg.norm(m[1])), 'norm_m+3': float(ref),
                                 'cos_a_m+3': float(ba @ m[3] / np.linalg.norm(ba) / ref),
                                 'cos_m+1_m-1': float(m[1] @ m[-1] / np.linalg.norm(m[1]) / np.linalg.norm(m[-1])),
                                 'cos_m+3_x36': float(m[3] @ b36 / ref / np.linalg.norm(b36)),
                                 'cos_x_k_partial': float(bx @ bk / np.linalg.norm(bx) / np.linalg.norm(bk)),
                                 'state_norm': float(np.linalg.norm(H[:, li, role], axis=1).mean())})
            all_ids.append(ids[j])
            all_vecs.append(vecs.astype(np.float16))
    (output / 'steer').mkdir(exist_ok=True)
    np.savez(output / 'steer' / f'directions_{mode}.npz', ids=np.array(all_ids), vectors=np.stack(all_vecs),
             labels=np.array(STEER_LABELS), layers=np.array(STEER_LAYERS))
    (output / 'steer' / f'directions_{mode}.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'directions {mode}: {len(all_ids)} cases')


# ----------------------------------------------------------------------------- Q1: routes

def route_summary(rows: list[dict]) -> dict:
    """Per condition x mode x variant: match rates of each route; paired change relative to the base answer."""
    out = {}
    by_base = defaultdict(dict)
    for r in rows:
        by_base[(r['condition'], r['mode'], r['base'])][r['variant']] = r
    for (cond, mode) in sorted({(r['condition'], r['mode']) for r in rows}):
        sub = [r for r in rows if r['condition'] == cond and r['mode'] == mode]
        entry = {}
        for v in om.VARIANTS:
            vs = [r for r in sub if r['variant'] == v]
            valid = [r for r in vs if r['valid']]
            e = {'n': len(vs), 'valid': len(valid)}
            for route in ('count', 'slot', 'ratio', 'gap'):
                e[route] = prop(sum(bool(r[f'match_{route}']) for r in valid), len(vs))
            e['gap_only'] = prop(sum(bool(r['match_gap']) and not r['match_count'] for r in valid), len(vs))
            # paired change against the base answer of the same base image
            d = []
            for r in valid:
                b = by_base[(cond, mode, r['base'])].get('base')
                if b is not None and b['valid']:
                    d.append(r['answer'] - b['answer'])
            e['delta_hist'] = dict(sorted(Counter(d).items()))
            e['n_paired'] = len(d)
            e['offset_vs_count'] = dict(sorted(Counter(r['answer'] - r['truth']['count'] for r in valid).items()))
            entry[v] = e
        # follow rates: slot route in shift/insL is +1; count route in delL is -1 (paired to base)
        f = {}
        for v, step in (('shift', 1), ('insL', 1), ('delL', -1), ('delR', 0)):
            h = entry[v]['delta_hist']
            n = entry[v]['n_paired']
            f[v] = {'step': step, **prop(h.get(step, 0), n), 'zero': prop(h.get(0, 0), n)}
        entry['follow'] = f
        out[f'{cond}|{mode}'] = entry
    return out


def follow_by_ordinal(rows: list[dict]) -> dict:
    """Paired change per target ordinal c (for onset-dependent analysis)."""
    by_base = {(r['condition'], r['mode'], r['base'], r['variant']): r for r in rows}
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['variant'] == 'base' or not r['valid']:
            continue
        b = by_base.get((r['condition'], r['mode'], r['base'], 'base'))
        if b is None or not b['valid']:
            continue
        out[f"{r['condition']}|{r['mode']}|{r['variant']}"][r['c']].append(r['answer'] - b['answer'])
    return {k: {int(c): v for c, v in sorted(d.items())} for k, d in out.items()}


def base_offsets(rows: list[dict]) -> dict:
    """Offset of base answers relative to the true ordinal, per condition x mode x c."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['variant'] == 'base' and r['valid']:
            out[f"{r['condition']}|{r['mode']}"][r['c']].append(r['answer'] - r['c'])
    return {k: {int(c): v for c, v in sorted(d.items())} for k, d in out.items()}


def strategy_summary(rows: list[dict]) -> dict:
    out = {}
    think = [r for r in rows if r['mode'] == 'think']
    for cond in sorted({r['condition'] for r in think}):
        sub = [r for r in think if r['condition'] == cond]
        codes = [om.code_strategies(r['response'], r['answer']) for r in sub]
        keys = list(om.STRATEGY_PATTERNS) + ['full_enumeration']
        e = {k: prop(sum(bool(c.get(k)) for c in codes), len(sub)) for k in keys}
        # accuracy (count route) conditional on an explicit walk through the instances
        en = [r for r, c in zip(sub, codes) if c.get('full_enumeration')]
        ne = [r for r, c in zip(sub, codes) if not c.get('full_enumeration')]
        e['count_match_if_enumerate'] = prop(sum(bool(r['match_count']) for r in en), len(en))
        e['count_match_if_not'] = prop(sum(bool(r['match_count']) for r in ne), len(ne))
        e['base_count_match_if_enumerate'] = prop(sum(bool(r['match_count']) for r in en if r['variant'] == 'base'),
                                                  sum(r['variant'] == 'base' for r in en))
        e['truncated'] = prop(sum(r['finish_reason'] != 'eos' for r in sub), len(sub))
        e['mean_tokens'] = float(np.mean([r['n_tokens'] for r in sub]))
        out[cond] = e
    return out


# ----------------------------------------------------------------------------- Q1/Q2: probes

def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


PCS = 128                     # principal components kept for every probe
CLS_LAYERS = (8, 16, 20, 24, 28)


def gram_pca(X: np.ndarray, k: int = PCS) -> np.ndarray:
    """Top-k principal-component scores (N, k) of X (N, D) via the N x N Gram matrix (unsupervised, all samples)."""
    Xc = X - X.mean(0)
    w, U = np.linalg.eigh(Xc @ Xc.T)
    w, U = w[::-1][:k], U[:, ::-1][:, :k]
    return U * np.sqrt(np.clip(w, 0, None))


def fold_ids(groups: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    keys = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(keys))
    fold_of = {k: i % folds for k, i in zip(keys[perm], range(len(keys)))}
    return np.array([fold_of[g] for g in groups])


def ridge_pcs(Z: np.ndarray, Y: np.ndarray, groups: np.ndarray, folds: int = 5,
              alphas=(1e-2, 1e-1, 1, 10, 100)) -> np.ndarray:
    """Cross-validated ridge predictions on PCA scores; alpha per fold and target by closed-form LOO."""
    f = fold_ids(groups, folds)
    pred = np.zeros(Y.shape)
    for k in range(folds):
        tr, te = f != k, f == k
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        A, B = (Z[tr] - mu) / sd, (Z[te] - mu) / sd
        ym = Y[tr].mean(0)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        UtY = U.T @ (Y[tr] - ym)
        best, coef = np.full(Y.shape[1], np.inf), np.zeros((A.shape[1], Y.shape[1]))
        for a in alphas:
            lam = a * len(A)
            g = S / (S ** 2 + lam)
            W = Vt.T @ (g[:, None] * UtY)
            h = ((U ** 2) * (S ** 2 / (S ** 2 + lam))[None]).sum(1)
            err = (((Y[tr] - ym - A @ W) / (1 - h)[:, None]) ** 2).mean(0)
            better = err < best
            best[better], coef[:, better] = err[better], W[:, better]
        pred[te] = B @ coef + ym
    return pred


def classifier_cv(Z: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str, folds: int = 5,
                  epochs: int = 200, seed: int = 0) -> float:
    """Cross-validated accuracy on PCA scores: 'linear' = one-hot ridge (least-squares) classifier,
    'mlp' = one hidden layer of 128 ReLU units trained with Adam (weight decay 1e-3)."""
    import torch
    f = fold_ids(groups, folds, seed)
    classes = np.unique(y)
    yi = np.searchsorted(classes, y)
    correct = 0
    for k in range(folds):
        tr, te = f != k, f == k
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        A, B = (Z[tr] - mu) / sd, (Z[te] - mu) / sd
        if kind == 'linear':
            T = np.eye(len(classes))[yi[tr]]
            W = np.linalg.solve(A.T @ A + 1.0 * len(A) * 0.01 * np.eye(A.shape[1]), A.T @ (T - T.mean(0)))
            pred = (B @ W + T.mean(0)).argmax(1)
        else:
            torch.manual_seed(seed + k)
            net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 128), torch.nn.ReLU(),
                                      torch.nn.Linear(128, len(classes)))
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
            At, t = torch.tensor(A, dtype=torch.float32), torch.tensor(yi[tr])
            for _ in range(epochs):
                opt.zero_grad()
                torch.nn.functional.cross_entropy(net(At), t).backward()
                opt.step()
            with torch.no_grad():
                pred = net(torch.tensor(B, dtype=torch.float32)).argmax(1).numpy()
        correct += int((pred == yi[te]).sum())
    return correct / len(y)


def probe_group(output: Path, cases: dict, rows_by_id: dict, conds: list[str], mode: str) -> dict | None:
    """Ridge probes (all layers and roles) and classifier probes with a control task (CLS_LAYERS) on PCA scores."""
    import torch
    torch.set_num_threads(4)
    ids, H, layers = load_features(output, conds, mode)
    if not ids:
        return None
    cs = [cases[i] for i in ids]
    axis = cs[0]['axis']
    groups = np.array([f"{c['condition']}|{c['base']}" for c in cs])
    ans = [rows_by_id[c['id']]['answer'] for c in cs]
    Y = np.array([[c['truth']['unit'], c['truth']['slot'], c['truth']['count'],
                   np.nan if a is None else a, c['truth']['ratio']] for c, a in zip(cs, ans)], float)
    ok = ~np.isnan(Y).any(1)
    H, Y, groups = H[ok], Y[ok], groups[ok]
    cs = [c for c, o in zip(cs, ok) if o]
    cond = np.array([c['condition'] for c in cs])
    gap_left = (Y[:, 2] - Y[:, 1] == -1).astype(float)       # count - slot = -1 <=> a blank line precedes the target
    error = Y[:, 3] - Y[:, 2]                                 # model answer minus count ordinal
    T = np.column_stack([Y, gap_left, error])
    names = PROBE_TARGETS + ('gap_left', 'error')
    # count beyond position: residual of count after a per-condition linear fit on the coordinate
    resid = np.zeros(len(Y))
    for cn in np.unique(cond):
        m = cond == cn
        Zc = np.column_stack([np.ones(m.sum()), Y[m, 0]])
        resid[m] = Y[m, 2] - Zc @ np.linalg.lstsq(Zc, Y[m, 2], rcond=None)[0]
    Tc = np.column_stack([T, resid])
    for cn in np.unique(cond):                                # targets centred per condition, like the features
        Tc[cond == cn] -= Tc[cond == cn].mean(0)
    res = {'n': int(len(Y)), 'layers': layers, 'roles': ['last', 'row0', 'col0', 'vis'], 'r2': {}, 'partial': {},
           'pcs': PCS, 'classifier': {}}
    count = Y[:, 2].astype(int)
    slot = Y[:, 1].astype(int)
    rng = np.random.default_rng(1)
    control = np.zeros(len(Y), int)
    for cn in np.unique(cond):
        lookup = {s: int(rng.integers(count.min(), count.max() + 1)) for s in np.unique(slot)}
        control[cond == cn] = [lookup[s] for s in slot[cond == cn]]
    for ri, role in enumerate(res['roles']):
        for li, layer in enumerate(layers):
            X = H[:, li, ri].copy()
            for cn in np.unique(cond):
                X[cond == cn] -= X[cond == cn].mean(0)
            if np.abs(X).max() < 1e-6:                        # identical inputs (fixed text token at layer 0)
                for name in names:
                    res['r2'].setdefault(name, {}).setdefault(role, []).append(float('nan'))
                res['partial'].setdefault(role, []).append(float('nan'))
                continue
            Z = gram_pca(X)
            P = ridge_pcs(Z, Tc, groups)
            for ti, name in enumerate(names):
                res['r2'].setdefault(name, {}).setdefault(role, []).append(r2(Tc[:, ti], P[:, ti]))
            res['partial'].setdefault(role, []).append(r2(Tc[:, -1], P[:, -1]))
            if layer in CLS_LAYERS:
                for kind in ('linear', 'mlp'):
                    acc = classifier_cv(Z[:, :64], count, groups, kind)
                    accc = classifier_cv(Z[:, :64], control, groups, kind)
                    res['classifier'].setdefault(role, {}).setdefault(kind, []).append(
                        {'layer': layer, 'count_acc': acc, 'control_acc': accc, 'selectivity': acc - accc})
            if layer == 20 and ri == long_role(axis):
                Pu = P[:, 0] + np.array([Y[cond == cn, 0].mean() for cn in cond])
                onehot = np.eye(len(np.unique(cond)))[np.searchsorted(np.unique(cond), cond)]
                res['count_from_unit_readout'] = {
                    'layer': layer, 'role': role,
                    'state_mlp_acc': res['classifier'][role]['mlp'][-1]['count_acc'],
                    'unit_readout_mlp_acc': classifier_cv(np.column_stack([Pu / 100, onehot]), count, groups, 'mlp'),
                    'true_unit_mlp_acc': classifier_cv(np.column_stack([Y[:, 0] / 100, onehot]), count, groups, 'mlp')}
    # where does the +1 error appear? variants without an interior blank, target = (answer - count == +1)
    from analysis_tools.behavior.decomposition_analysis import auroc
    variant = np.array([c['variant'] for c in cs])
    sub = np.isin(variant, ('base', 'shift', 'delR'))
    plus = (error[sub] == 1).astype(float)
    res['plus1'] = {'n': int(sub.sum()), 'n_plus1': int(plus.sum()), 'auroc': {}}
    for ri, role in enumerate(res['roles']):
        for li, layer in enumerate(layers):
            X = H[sub, li, ri].copy()
            for cn in np.unique(cond[sub]):
                X[cond[sub] == cn] -= X[cond[sub] == cn].mean(0)
            if np.abs(X).max() < 1e-6:
                res['plus1']['auroc'].setdefault(role, []).append(float('nan'))
                continue
            Z = gram_pca(X)
            # the count is a covariate: +1 grows with it, so the probe sees count + state and is compared with count alone
            P = ridge_pcs(np.column_stack([Z, Y[sub, 2]]), plus[:, None], groups[sub])[:, 0]
            res['plus1']['auroc'].setdefault(role, []).append(auroc(list(P[plus == 1]), list(P[plus == 0])))
    Pc = ridge_pcs(np.column_stack([Y[sub, 2], Y[sub, 2] ** 2]), plus[:, None], groups[sub])[:, 0]
    res['plus1']['auroc_count_only'] = auroc(list(Pc[plus == 1]), list(Pc[plus == 0]))
    sd = np.sqrt(np.mean([Y[cond == cn, 2].var() for cn in np.unique(cond)]))
    res['count_sd_within_condition'] = float(sd)
    res['axis'] = axis
    res['conditions'] = conds
    res['chance_count'] = float(max(np.bincount(count)) / len(count))
    # best accuracy for the count ordinal from the exact coordinate and condition: majority count per (condition, slot)
    best = 0
    for key in set(zip(cond, slot)):
        m = (cond == key[0]) & (slot == key[1])
        best += np.bincount(count[m]).max()
    res['count_from_exact_unit_ceiling'] = float(best / len(count))
    res['error_values'] = {int(k): int(v) for k, v in zip(*np.unique(error, return_counts=True))}
    return res


# ----------------------------------------------------------------------------- steering summary

def steer_summary(output: Path, mode: str) -> dict:
    path = output / 'steer' / f'records_{mode}.jsonl'
    if not path.exists():
        return {}
    recs = read_jsonl(path)
    out = {}
    for cond in sorted({r['condition'] for r in recs}):
        sub = [r for r in recs if r['condition'] == cond]
        e = {}
        for layer in STEER_LAYERS:
            for lab in STEER_LABELS[1:]:
                if lab.startswith('x36n') and cond.startswith('V'):
                    continue
                d = []
                for r in sub:
                    runs = {(u['layer'], u['label']): u['value'] for u in r['runs']}
                    a, b = runs.get((layer, lab)), runs.get((layer, 'none'))
                    if a is not None and b is not None:
                        d.append(a - b)
                d = np.array(d)
                e[f'{layer}|{lab}'] = {'n': int(len(d)), 'mean_change': float(d.mean()) if len(d) else float('nan'),
                                       'p_up': float((d > 0).mean()) if len(d) else float('nan'),
                                       'p_down': float((d < 0).mean()) if len(d) else float('nan'),
                                       'hist': dict(sorted(Counter(d.tolist()).items()))}
        out[cond] = e
    out['pooled'] = steer_pooled(recs)
    return out


STEER_C_BINS = ((3, 5), (6, 11), (12, 18))


def _steer_changes(recs: list[dict], layer: int, label: str) -> np.ndarray:
    d = []
    for r in recs:
        runs = {(u['layer'], u['label']): u['value'] for u in r['runs']}
        a, b = runs.get((layer, label)), runs.get((layer, 'none'))
        if a is not None and b is not None:
            d.append(a - b)
    return np.array(d, float)


def steer_pooled(recs: list[dict]) -> dict:
    """Changes pooled over the conditions of one axis: per layer, and per ordinal bin at every layer.
    'p_exact' is the fraction whose change equals the intended shift (m labels only)."""
    intended = {'m+1': 1, 'm-1': -1, 'm+3': 3, 'm-3': -3}
    out = {}
    for axis in ('H', 'V'):
        sub = [r for r in recs if r['condition'].startswith(axis)]
        if not sub:
            continue
        e = {'n': len(sub), 'by_layer': {}, 'by_c': {}}
        for layer in STEER_LAYERS:
            for lab in STEER_LABELS[1:]:
                if lab.startswith('x36n') and axis == 'V':
                    continue
                d = _steer_changes(sub, layer, lab)
                e['by_layer'][f'{layer}|{lab}'] = {
                    'n': int(len(d)), 'mean_change': float(d.mean()), 'p_changed': float((d != 0).mean()),
                    'p_up': float((d > 0).mean()), 'p_down': float((d < 0).mean()),
                    'p_exact': float((d == intended[lab]).mean()) if lab in intended else None}
                for lo, hi in STEER_C_BINS:
                    dc = _steer_changes([r for r in sub if lo <= r['c'] <= hi], layer, lab)
                    e['by_c'][f'{layer}|{lab}|{lo}-{hi}'] = {'n': int(len(dc)),
                                                             'p_changed': float((dc != 0).mean()) if len(dc) else None}
        out[axis] = e
    return out


# ----------------------------------------------------------------------------- Q3: cheap models of the onset

def _cdf(z):
    import torch
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


def disc_normal(mu, sigma, vmax: int):
    """(N, vmax) probabilities of answers 1..vmax under a Gaussian discretized at half-integers (torch)."""
    import torch
    v = torch.arange(1, vmax + 1, dtype=torch.float64)
    P = _cdf((v[None] + 0.5 - mu[:, None]) / sigma[:, None]) - _cdf((v[None] - 0.5 - mu[:, None]) / sigma[:, None])
    return P.clamp_min(1e-12)


def model_probs(name: str, theta, c, n: int, vmax: int, prior=None, K: float | None = None) -> np.ndarray:
    """Answer distribution (len(c), vmax) of one cheap model; accepts numpy or torch inputs."""
    import torch
    as_np = not torch.is_tensor(theta)
    theta = torch.as_tensor(np.asarray(theta, float) if as_np else theta, dtype=torch.float64)
    c = torch.as_tensor(np.asarray(c, float), dtype=torch.float64)
    eps = torch.sigmoid(theta[-1])
    if name == 'subitize':                 # exact up to K, then linear drift and growing noise
        beta, s0, w = theta[0], 0.05 + theta[1].exp(), theta[2].exp()
        over = (c - K).clamp_min(0)        # K is profiled on an integer grid by fit_condition
        mu, sig = c + beta * over, s0 + w * over
    elif name == 'ratio':                  # (c - 0.5) N_hat / N + 0.5 with Weber-like noise
        g, s0, w = theta[0], 0.05 + theta[1].exp(), theta[2].exp()
        mu, sig = (c - 0.5) * (1 + g) + 0.5, (s0 ** 2 + (w * c) ** 2).sqrt()
    elif name == 'anchored':               # proportion judgment anchored at both ends (power law of p = (c - .5)/n)
        a, s0, w = theta[0].exp(), 0.05 + theta[1].exp(), theta[2].exp()
        p = (c - 0.5) / n
        mu = n * p ** a + 0.5
        sig = s0 + w * (p * (1 - p)).sqrt() * n
    elif name == 'digit_prior':            # unbiased Gaussian times a value prior shared across conditions
        s0, w = 0.05 + theta[0].exp(), theta[1].exp()
        mu, sig = c, s0 + w * c
    else:
        raise ValueError(name)
    P = disc_normal(mu, sig, vmax)
    P = P / P.sum(1, keepdim=True)         # answers are truncated to 1..vmax
    if prior is not None:
        pr = torch.as_tensor(np.asarray(prior, float) if not torch.is_tensor(prior) else prior, dtype=torch.float64)
        P = P * pr[:vmax].exp()[None]
        P = P / P.sum(1, keepdim=True)
    P = (1 - eps) * P + eps / vmax
    return P.detach().numpy() if as_np else P


N_PARAMS = {'subitize': 5, 'ratio': 4, 'anchored': 4, 'digit_prior': 3}
INIT = {'subitize': [0.1, -1, -2, -4], 'ratio': [0.05, -1, -3, -4], 'anchored': [0, -1, -3, -4],
        'digit_prior': [-1, -3, -4]}


def _minimize(f, x0):
    """L-BFGS on a torch objective; returns (x, value)."""
    import torch
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([x], lr=0.5, max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        v = f(x)
        v.backward()
        return v

    opt.step(closure)
    with torch.no_grad():
        v = float(f(x))
    return x.detach().numpy(), v


def fit_condition(name: str, c: np.ndarray, a: np.ndarray, n: int, prior=None) -> tuple[np.ndarray, float]:
    """Maximum-likelihood fit; the subitizing range K is profiled over integers 1..n (theta gets K appended first)."""
    import torch
    vmax = n + 3
    idx = torch.as_tensor(np.clip(a, 1, vmax) - 1)
    rows = torch.arange(len(a))
    best = None
    for K in (range(1, n + 1) if name == 'subitize' else [None]):
        def nll(theta, K=K):
            P = model_probs(name, theta, c, n, vmax, prior, K)
            return -P[rows, idx].log().sum()
        x, v = _minimize(nll, INIT[name])
        if K is not None:
            x = np.concatenate([[K], x])
        if np.isfinite(v) and (best is None or v < best[1]):
            best = (x, v)
    return best


def onset_models(token_grid: Path) -> dict:
    """Fit the four cheap models per 1.4 condition (6 rows, phase 0) to the column answers."""
    import torch
    torch.set_num_threads(4)
    from analysis_tools.behavior.ordinal_token_grid import parse_condition
    cases = {c['id']: c for c in read_jsonl(token_grid / 'cases.jsonl')}
    data = defaultdict(lambda: ([], []))
    for p in sorted(token_grid.glob('results*.jsonl')):
        for r in read_jsonl(p):
            c = cases[r['id']]
            if c['task'] != 'T6' or c['format'] != 'row_column' or c['rows'] != 6 or c['phase'] != 0:
                continue
            rc = om.parse_answer(r['response'])
            if rc is None:
                continue
            data[c['condition']][0].append(c['truth']['target_rc'][1] + 1)
            data[c['condition']][1].append(rc[1])
    conds = sorted(data, key=lambda k: parse_condition(k))
    out = {'conditions': {}, 'totals': {}}
    fits = {}
    for name in ('subitize', 'ratio', 'anchored'):
        for cond in conds:
            c, a = map(np.array, data[cond])
            fits[(name, cond)] = fit_condition(name, c, a, parse_condition(cond)[1])
    vmax_all = 30
    prior = np.zeros(vmax_all)
    for _ in range(3):                     # alternate per-condition noise and the shared value prior
        for cond in conds:
            c, a = map(np.array, data[cond])
            fits[('digit_prior', cond)] = fit_condition('digit_prior', c, a, parse_condition(cond)[1], prior)
        prior = fit_prior(conds, data, fits, prior)
    for cond in conds:                     # final noise fit under the final prior
        c, a = map(np.array, data[cond])
        fits[('digit_prior', cond)] = fit_condition('digit_prior', c, a, parse_condition(cond)[1], prior)
    N = sum(len(data[cond][0]) for cond in conds)
    for name in ('subitize', 'ratio', 'anchored', 'digit_prior'):
        tot = sum(fits[(name, cond)][1] for cond in conds)
        k = N_PARAMS[name] * len(conds) + (vmax_all - 1 if name == 'digit_prior' else 0)
        out['totals'][name] = {'nll': tot, 'k': k, 'aic': 2 * tot + 2 * k, 'bic': 2 * tot + k * math.log(N), 'n': N}
    for cond in conds:
        c, a = map(np.array, data[cond])
        n = parse_condition(cond)[1]
        e = {'n_samples': int(len(c)), 'observed_plus1': {}, 'observed_minus1': {}, 'models': {}}
        for cc in range(1, n + 1):
            m = c == cc
            e['observed_plus1'][cc] = float((a[m] == cc + 1).mean()) if m.any() else float('nan')
            e['observed_minus1'][cc] = float((a[m] == cc - 1).mean()) if m.any() else float('nan')
        for name in ('subitize', 'ratio', 'anchored', 'digit_prior'):
            theta, nll = fits[(name, cond)]
            th, K = (theta[1:], theta[0]) if name == 'subitize' else (theta, None)
            P = model_probs(name, th, np.arange(1, n + 1), n, n + 3, prior if name == 'digit_prior' else None, K)
            e['models'][name] = {'theta': theta.tolist(), 'nll': nll,
                                 'pred_plus1': {cc: float(P[cc - 1, cc]) for cc in range(1, n + 1)},
                                 'pred_n_plus_1_at_last': float(P[n - 1, n])}
        e['observed_n_plus_1_at_last'] = float((a[c == n] == n + 1).mean())
        out['conditions'][cond] = e
    out['digit_prior'] = prior.tolist()
    return out


def fit_prior(conds, data, fits, prior):
    import torch
    from analysis_tools.behavior.ordinal_token_grid import parse_condition
    packs = []
    for cond in conds:
        c, a = map(np.array, data[cond])
        n = parse_condition(cond)[1]
        packs.append((c, torch.as_tensor(np.clip(a, 1, n + 3) - 1), n, torch.as_tensor(fits[('digit_prior', cond)][0])))

    def nll(pv):
        full = torch.cat([torch.zeros(1, dtype=torch.float64), pv])
        tot = 0
        for c, idx, n, theta in packs:
            P = model_probs('digit_prior', theta, c, n, n + 3, full)
            tot = tot - P[torch.arange(len(idx)), idx].log().sum()
        return tot + 0.01 * (pv ** 2).sum()

    x, _ = _minimize(nll, prior[1:])
    return np.concatenate([[0], x])


# ----------------------------------------------------------------------------- Q4: row biases

def rowcol_records(cases_path: Path, results: list[Path], select) -> list[dict]:
    cases = {c['id']: c for c in read_jsonl(cases_path)}
    out = []
    for p in results:
        for r in read_jsonl(p):
            c = cases.get(r['id'])
            if c is None or c.get('format') != 'row_column' or not select(c):
                continue
            rc = om.parse_answer(r['response'])
            t = c['truth']
            tr, tc = t['target_rc']
            rec = {'id': r['id'], 'rows': t['rows'], 'cols': t['cols'], 'cell': t['cell'], 'tr': tr + 1, 'tc': tc + 1,
                   'task': c['task'], 'valid': rc is not None}
            if rc is not None:
                rec.update(dr=rc[0] - tr - 1, dc=rc[1] - tc - 1)
            out.append(rec)
    return out


def bias_table(recs: list[dict]) -> dict:
    n = len(recs)
    v = [r for r in recs if r['valid']]
    e = {'n': n}
    for key, name in (('dr', 'row'), ('dc', 'col')):
        for s, lab in ((1, '+1'), (-1, '-1')):
            e[f'{name}{lab}'] = prop(sum(r[key] == s for r in v), n)
    e['both_minus1'] = prop(sum(r['dr'] == -1 and r['dc'] == -1 for r in v), n)
    e['row_error_given_col_error'] = prop(sum(r['dr'] != 0 and r['dc'] != 0 for r in v), max(sum(r['dc'] != 0 for r in v), 1))
    e['row_error_given_col_ok'] = prop(sum(r['dr'] != 0 and r['dc'] == 0 for r in v), max(sum(r['dc'] == 0 for r in v), 1))
    return e


def row_profile(recs: list[dict], key: str = 'dr', index: str = 'tr') -> dict:
    out = defaultdict(lambda: [0, 0, 0])
    for r in recs:
        if not r['valid']:
            continue
        e = out[r[index]]
        e[0] += 1
        e[1] += r[key] == 1
        e[2] += r[key] == -1
    return {int(k): {'n': v[0], 'plus1': v[1] / v[0], 'minus1': v[2] / v[0]} for k, v in sorted(out.items())}


def row_biases(formal: Path, supplement: Path, token_grid: Path, rows15: list[dict]) -> dict:
    sets = {}
    rec = rowcol_records(formal / 'cases.jsonl', [formal / 'results_main.jsonl'],
                         lambda c: c['task'] == 'T6' and c['stage'] == 'main')
    for g in (3, 6, 10, 20):
        sets[f'1.3 T6 {g}x{g} c64'] = [r for r in rec if r['rows'] == g]
    rec = rowcol_records(formal / 'cases.jsonl', [formal / 'results_main.jsonl'],
                         lambda c: c['task'] in ('T4', 'T5') and c['stage'] == 'main' and c['level'] == 4)
    for g in (6, 10, 20):
        sets[f'1.3 T4/T5 L4 {g}x{g} c64'] = [r for r in rec if r['rows'] == g]
    sup_cases = read_jsonl(supplement / 'cases_layouts.jsonl')
    layout_of = {c['id']: c['layout'] for c in sup_cases}
    rec = rowcol_records(supplement / 'cases_layouts.jsonl', sorted(supplement.glob('results_layouts.jsonl')),
                         lambda c: c['task'] == 'T6')
    for lay in sorted({layout_of[r['id']] for r in rec}):
        sub = [r for r in rec if layout_of[r['id']] == lay]
        sets[f"1.3s T6 {lay} ({sub[0]['rows']}x{sub[0]['cols']} c{sub[0]['cell']})"] = sub
    tg_cases = {c['id']: c for c in read_jsonl(token_grid / 'cases.jsonl')}
    rec = rowcol_records(token_grid / 'cases.jsonl', sorted(token_grid.glob('results*.jsonl')),
                         lambda c: c['task'] == 'T6' and c['phase'] == 0)
    for cond in sorted({tg_cases[r['id']]['condition'] for r in rec}):
        sub = [r for r in rec if tg_cases[r['id']]['condition'] == cond]
        sets[f"1.4 {cond} ({sub[0]['rows']}x{sub[0]['cols']})"] = sub
    # 1.5 V conditions: rows on the long axis (base variant only, direct)
    for cond in sorted({r['condition'] for r in rows15}):
        sub = [r for r in rows15 if r['condition'] == cond and r['variant'] == 'base' and r['mode'] == 'direct']
        recs = []
        for r in sub:
            ok = r['answer'] is not None
            long_d = r['answer'] - r['c'] if ok else None
            short_d = r['answer_other'] - r['truth']['other'] if ok else None
            dr, dc = (short_d, long_d) if r['axis'] == 'H' else (long_d, short_d)
            recs.append({'valid': ok, 'dr': dr, 'dc': dc, 'tr': (r['truth']['other'] if r['axis'] == 'H' else r['c']),
                         'tc': (r['c'] if r['axis'] == 'H' else r['truth']['other']), 'rows': 6 if r['axis'] == 'H' else 20,
                         'cols': 20 if r['axis'] == 'H' else 6, 'cell': r['cell']})
        sets[f'1.5 {cond} base ({recs[0]["rows"]}x{recs[0]["cols"]})'] = recs
    out = {'tables': {k: bias_table(v) for k, v in sets.items()},
           'row_profiles': {k: row_profile(v) for k, v in sets.items()},
           'col_profiles': {k: row_profile(v, 'dc', 'tc') for k, v in sets.items()}}
    return out


# ----------------------------------------------------------------------------- main

def stage_analyze(output: Path, token_grid: Path, formal: Path, supplement: Path) -> None:
    from analysis_tools.behavior import ordinal_mechanism_figures as figs
    dest = output / 'analysis'
    (dest / 'figures').mkdir(parents=True, exist_ok=True)
    rows = load_rows(output)
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    rows_by_id = {r['id']: r for r in rows}
    summary = {'n_rows': len(rows), 'routes': route_summary(rows), 'follow_by_c': follow_by_ordinal(rows),
               'base_offsets': base_offsets(rows), 'strategies': strategy_summary(rows)}
    spot = output / 'gen' / 'spotcheck.jsonl'
    if spot.exists():
        sp = read_jsonl(spot)
        summary['spotcheck'] = {m: {'n': sum(r['mode'] == m for r in sp),
                                    'answer_agree': sum(r['mode'] == m and r['answer_batched'] == r['answer_single']
                                                        for r in sp)} for m in om.MODES}
    summary['probes'] = {}
    for mode in om.MODES:
        for axis in ('H', 'V'):
            conds = sorted({c['condition'] for c in cases.values() if c['axis'] == axis and c['mode'] == mode})
            cache = dest / f'probes_{axis}_{mode}.json'
            if cache.exists():
                res = json.loads(cache.read_text())
            else:
                res = probe_group(output, cases, rows_by_id, conds, mode)
                if res is not None:
                    cache.write_text(json.dumps(res, default=float) + '\n')
            if res is not None:
                summary['probes'][f'{axis}|{mode}'] = res
                print('probes', axis, mode, flush=True)
    summary['steer'] = {m: steer_summary(output, m) for m in om.MODES}
    summary['onset_models'] = onset_models(token_grid)
    summary['row_biases'] = row_biases(formal, supplement, token_grid, rows)
    (dest / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    figs.make_all(output, summary, rows, dest / 'figures')
    print('analysis written to', dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('command', choices=['directions', 'analyze'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=om.MODES, default='direct')
    ap.add_argument('--coord-value', type=Path)
    ap.add_argument('--token-grid', type=Path)
    ap.add_argument('--formal', type=Path)
    ap.add_argument('--supplement', type=Path)
    a = ap.parse_args()
    if a.command == 'directions':
        stage_directions(a.output, a.mode, a.coord_value)
    else:
        stage_analyze(a.output, a.token_grid, a.formal, a.supplement)


if __name__ == '__main__':
    main()
