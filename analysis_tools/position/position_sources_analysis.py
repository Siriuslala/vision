"""Experiment 3.15 round 3 analysis: probes, variance partition, DeltaNet read-out weights, raster-order
dissociation, steering and Qwen3-VL flow (CPU).

Results: docs/analysis/position/results_2026-11-11_position_sources.md.
Usage: python -m analysis_tools.position.position_sources_analysis --run-dir RUN --figures FIG [--parts ...]
Parts: probes (capture data -> probes.json, steer_dirs.pt), roll, steer, vlflow, figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis_tools.position import coordinate_mechanism_natural as cmn
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import position_sources as ps

O_LAYERS = (2, 5, 7, 9, 10, 11, 13, 14, 15, 18, 22, 26, 30)
O_LAYERS_NOF = (9, 10, 11, 13, 14, 15, 18, 22)
NOF_LAYERS = (9, 10, 11, 13, 14, 15)
RES_LAYERS = (0, 8, 16, 19)
LIN = tuple(l for l in range(1, 33) if l % 4)
CLASS_NAMES = {0: 'other', 1: 'integrator', 2: 'row_reset', 3: 'onset'}
GATES_38 = 'position/2026-10-16_sequence_order_r2/gates/stats.json'


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


# ----------------------------------------------------------------------------- loading capture data

def load_family(run: Path, fam: str, cond: str) -> dict | None:
    allm = read_jsonl(run / 'capture' / 'meta.jsonl')
    both = {m['case'] for m in allm if m['family'] == fam and m['cond'] == 'clean'} & \
        {m['case'] for m in allm if m['family'] == fam and m['cond'] == 'nof'}
    meta = [m for m in allm if m['family'] == fam and m['cond'] == cond and m['case'] in both]
    if not meta:
        return None
    rows = {'img': [], 'tab': {}, 'o': {}, 'res': {}, 'mix': {}, 'alpha': [], 'beta': [],
            'q_img': [], 'q_tab': {}, 'summ': {}, 'parts': {}, 'cases': [], 'HW': []}
    for k, m in enumerate(meta):
        rec = torch.load(run / 'capture' / f"{m['case']}_{cond}.pt", weights_only=False)
        toks, queries, tab = rec['toks'], rec['queries'], rec['table']
        rows['cases'].append(m['case'])
        rows['HW'].append((m['H'], m['W']))
        rows['img'].append(np.full(len(toks), k))
        for key, v in tab.items():
            rows['tab'].setdefault(key, []).append(np.asarray(v)[toks])
            rows['q_tab'].setdefault(key, []).append(np.asarray(v)[queries])
        rows['tab'].setdefault('Hn', []).append(np.full(len(toks), m['H']))
        rows['tab'].setdefault('Wn', []).append(np.full(len(toks), m['W']))
        rows['q_img'].append(np.full(len(queries), k))
        for l, v in rec['o'].items():
            rows['o'].setdefault(l, []).append(v.reshape(len(toks), -1).float().numpy())
        for l, v in rec['res'].items():
            rows['res'].setdefault(l, []).append(v.float().numpy())
        for l, v in rec['mix'].items():
            rows['mix'].setdefault(l, []).append(v.float().numpy())
        rows['alpha'].append(rec['alpha'].float().numpy())
        rows['beta'].append(rec['beta'].float().numpy())
        for l, d in rec['summ'].items():
            for key, v in d.items():
                rows['summ'].setdefault(l, {}).setdefault(key, []).append(v.float().numpy())
        for l, d in rec['parts'].items():
            for key, v in d.items():
                rows['parts'].setdefault(l, {}).setdefault(key, []).append(v.reshape(len(queries), -1).float().numpy())
    cat = lambda xs: np.concatenate(xs, 0)
    out = {'img': cat(rows['img']), 'q_img': cat(rows['q_img']), 'cases': rows['cases'], 'HW': rows['HW'],
           'tab': {k: cat(v) for k, v in rows['tab'].items()}, 'q_tab': {k: cat(v) for k, v in rows['q_tab'].items()},
           'o': {l: cat(v) for l, v in rows['o'].items()}, 'res': {l: cat(v) for l, v in rows['res'].items()},
           'mix': {l: cat(v) for l, v in rows['mix'].items()},
           'alpha': np.concatenate(rows['alpha'], 1), 'beta': np.concatenate(rows['beta'], 1),
           'summ': {l: {k: cat(v) for k, v in d.items()} for l, d in rows['summ'].items()},
           'parts': {l: {k: cat(v) for k, v in d.items()} for l, d in rows['parts'].items()}}
    return out


# ----------------------------------------------------------------------------- feature groups

def pos_features(tab: dict) -> np.ndarray:
    """Own position: one-hot of 20 bins of the token-centre y and x (0-1000 units)."""
    yb = np.clip((tab['y_tok'] // 50).astype(int), 0, 19)
    xb = np.clip((tab['x_tok'] // 50).astype(int), 0, 19)
    return np.concatenate([ps.one_hot(yb, 20), ps.one_hot(xb, 20)], 1)


def groups_for(tab: dict, fam: str) -> dict[str, np.ndarray]:
    g = {'position': pos_features(tab)}
    content = [tab['tgt'][:, None].astype(np.float32)]
    if 'ent' in tab:
        content.append(tab['ent'][:, None].astype(np.float32))
    g['content'] = np.concatenate(content, 1)
    if fam == 'jgrid':
        g['count'] = np.concatenate([ps.one_hot(np.clip(tab['row_cnt'], 0, 7), 8),
                                     ps.one_hot(np.clip(tab['col_cnt'], 0, 7), 8)], 1)
    aft = tab['after_tgt_row'][:, None].astype(np.float32)
    tyb = ps.one_hot(np.clip((tab['ty'] // 100).astype(int), 0, 9), 10)
    txb = ps.one_hot(np.clip((tab['tx'] // 100).astype(int), 0, 9), 10)
    g['target_coord'] = np.concatenate([aft, aft * tyb, aft * txb], 1)
    return g


def pca(X: np.ndarray, k: int = 64) -> tuple[np.ndarray, float]:
    """Top-k principal-component scores of the column-standardised X and the variance share they cover
    (randomised SVD, torch float32)."""
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Xt = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-6)
    U, S, V = torch.pca_lowrank(Xt, q=min(k + 16, min(Xt.shape)), center=False, niter=3)
    total = float((Xt ** 2).sum())
    scores = (Xt @ V[:, :k]).numpy()
    return scores, float((S[:k] ** 2).sum() / total)


# ----------------------------------------------------------------------------- probes part

def probe_targets(fam: str) -> list[str]:
    return ['y_tok', 'x_tok'] + (['row_cnt', 'col_cnt'] if fam == 'jgrid' else [])


def residualise(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """y minus its least-squares fit on features F (with intercept)."""
    X = np.concatenate([F, np.ones((len(F), 1))], 1)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ w


def run_probes(run: Path) -> dict:
    import time
    t0 = time.time()
    out = {'probe': {}, 'vp': {}, 'heads': {}, 'amount': {}, 'nof': {}, 'transfer': {}, 'heldout': {}, 'summ': {}}
    data = {}
    for fam in ('jgrid', 'single', 'refcoco'):
        for cond in ('clean', 'nof'):
            d = load_family(run, fam, cond)
            if d is not None:
                data[(fam, cond)] = d
                print(f'[{time.time() - t0:.0f}s] loaded {fam} {cond}: {len(d["img"])} tokens', flush=True)
    for (fam, cond), d in data.items():
        if cond != 'clean':
            continue
        tab, img = d['tab'], d['img']
        reps = {f'o{l}': d['o'][l] for l in O_LAYERS if l in d['o']}
        reps.update({f'res{l}': d['res'][l] for l in RES_LAYERS if l in d['res']})
        pr = {}
        for name, X in reps.items():
            Xp, _ = pca(X, 256)
            pr[name] = {t: ps.ridge_cv_r2(Xp, tab[t].astype(float), img)['r2'] for t in probe_targets(fam)}
            if fam == 'jgrid':
                # count beyond own position: residual of the count after the position one-hot
                P = pos_features(tab)
                for t in ('row_cnt', 'col_cnt'):
                    pr[name][t + '|pos'] = ps.ridge_cv_r2(Xp, residualise(tab[t].astype(float), P), img)['r2']
                pr[name]['y_tok|row_cnt'] = ps.ridge_cv_r2(
                    Xp, residualise(tab['y_tok'], ps.one_hot(np.clip(tab['row_cnt'], 0, 7), 8)), img)['r2']
                pr[name]['x_tok|col_cnt'] = ps.ridge_cv_r2(
                    Xp, residualise(tab['x_tok'], ps.one_hot(np.clip(tab['col_cnt'], 0, 7), 8)), img)['r2']
        out['probe'][fam] = pr
        print(f'[{time.time() - t0:.0f}s]', 'probes', fam, {k: {t: round(v, 2) for t, v in vv.items()} for k, vv in pr.items()}, flush=True)
        # variance partition (top-64 PCs)
        vp = {}
        G = groups_for(tab, fam)
        for name, X in reps.items():
            Y, cover = pca(X, 64)
            r = ps.variance_partition(Y, G, img, lam=1.0)
            r['pc_cover'] = cover
            vp[name] = r
        out['vp'][fam] = vp
        print(f'[{time.time() - t0:.0f}s]', 'vp', fam, {k: (round(v['total'], 2), {g: round(u, 3) for g, u in v['unique'].items()})
                          for k, v in vp.items()}, flush=True)
        # per-head probes (layers 9-15)
        hs = {}
        P = pos_features(tab)
        rc = residualise(tab['row_cnt'].astype(float), P) if fam == 'jgrid' else None
        for l in NOF_LAYERS:
            Xl = d['o'][l].reshape(len(img), 32, 128)
            for t in ('y_tok', 'x_tok') + (('row_cnt|pos',) if fam == 'jgrid' else ()):
                y = rc if t == 'row_cnt|pos' else tab[t].astype(float)
                hs.setdefault(t, []).append([ps.ridge_cv_r2(Xl[:, h], y, img, lams=(1e1, 1e3))['r2']
                                             for h in range(32)])
        out['heads'][fam] = hs
        # amount vs content (query tokens, layers 9-15 stacked)
        qt, qi = d['q_tab'], d['q_img']
        am = {}
        for t in probe_targets(fam):
            y = qt[t].astype(float)
            A = np.concatenate([d['summ'][l]['amount'] for l in NOF_LAYERS], 1)
            C = np.concatenate([d['parts'][l]['content_part'] for l in NOF_LAYERS], 1)
            O = np.concatenate([d['parts'][l]['o'] for l in NOF_LAYERS], 1)
            am[t] = {'amount': ps.ridge_cv_r2(A, y, qi)['r2'],
                     'content': ps.ridge_cv_r2(pca(C, 128)[0], y, qi)['r2'],
                     'output': ps.ridge_cv_r2(pca(O, 128)[0], y, qi)['r2']}
        out['amount'][fam] = am
        # read-out weight summaries per head class
        cls = head_classes(run)
        sm = {}
        for l in NOF_LAYERS:
            s = d['summ'][l]
            li = LIN.index(l)
            for key in ('src_i', 'src_j', 'lag', 'same_row', 'ent', 'tgt', 'pre_share', 'recon_err', 'amount'):
                v = s[key]                                              # (Q, 32)
                if key == 'src_i':
                    v = v - qt['i'][:, None]
                if key == 'src_j':
                    v = v - qt['j'][:, None]
                for h in range(32):
                    c = CLASS_NAMES[int(cls[li, h])] if cls is not None else 'all'
                    sm.setdefault(key, {}).setdefault(c, []).append(float(np.median(v[:, h])))
        out['summ'][fam] = {k: {c: [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)),
                                    len(v)] for c, v in d_.items()} for k, d_ in sm.items()}
        # held-out grid size (jgrid): train on R <= 5 rows, test on R >= 6
        if fam == 'jgrid':
            R = np.array([len(set(np.asarray(tab['row_cnt'])[img == k])) - 1 for k in range(img.max() + 1)])
            test = R[img] >= 6
            ho = {}
            for name in ('o9', 'o11', 'o13', 'o15', 'res16', 'res19'):
                if name not in reps:
                    continue
                Xp, _ = pca(reps[name], 256)
                ho[name] = {t: ps.ridge_cv_r2(Xp, tab[t].astype(float), img, test_mask=test)['r2']
                            for t in ('y_tok', 'row_cnt', 'x_tok', 'col_cnt')}
            out['heldout'][fam] = {'n_test_images': int((R >= 6).sum()), 'r2': ho}
    # cross-family transfer of the own-position probe
    for name in ('o9', 'o13', 'o15', 'res16', 'res19'):
        tr = {}
        for fa in ('jgrid', 'single', 'refcoco'):
            for fb in ('jgrid', 'single', 'refcoco'):
                if (fa, 'clean') not in data or (fb, 'clean') not in data or fa == fb:
                    continue
                Xa = rep_of(data[(fa, 'clean')], name)
                Xb = rep_of(data[(fb, 'clean')], name)
                for t in ('y_tok', 'x_tok'):
                    w, b = ps.ridge_fit(Xa, data[(fa, 'clean')]['tab'][t].astype(float), 1e3)
                    yp = (torch.as_tensor(Xb, dtype=torch.float64) @ w + b).numpy()
                    yt = data[(fb, 'clean')]['tab'][t]
                    tr[f'{fa}->{fb}:{t}'] = float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
        out['transfer'][name] = tr
    # no-forgetting: clean-trained probes applied to the nof condition
    for fam in ('jgrid', 'single', 'refcoco'):
        if (fam, 'clean') not in data or (fam, 'nof') not in data:
            continue
        dc, dn = data[(fam, 'clean')], data[(fam, 'nof')]
        if dc['cases'] != dn['cases']:
            raise ValueError('clean / nof capture cases differ')
        res = {}
        for name in [f'o{l}' for l in O_LAYERS_NOF] + ['res16', 'res19']:
            Xc, Xn = rep_of(dc, name), rep_of(dn, name)
            res[name] = {}
            for t in probe_targets(fam):
                y = dc['tab'][t].astype(float)
                w, b = ps.ridge_fit(Xc, y, 1e3)
                pc = (torch.as_tensor(Xc, dtype=torch.float64) @ w + b).numpy()
                pn = (torch.as_tensor(Xn, dtype=torch.float64) @ w + b).numpy()
                dlt = pn - pc
                slope = float(np.polyfit(y, dlt, 1)[0])
                res[name][t] = {'mean_shift': float(dlt.mean()), 'slope_vs_true': slope,
                                'by_bin': binned(y, dlt, t)}
        out['nof'][fam] = res
    return out, data


def binned(y: np.ndarray, d: np.ndarray, t: str) -> list[list[float]]:
    if t in ('y_tok', 'x_tok'):
        edges = np.arange(0, 1001, 100)
    else:
        edges = np.arange(-0.5, 8.5, 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (y >= a) & (y < b)
        if m.sum() >= 5:
            out.append([float((a + b) / 2), float(d[m].mean()), int(m.sum())])
    return out


def rep_of(d: dict, name: str) -> np.ndarray:
    return d['o'][int(name[1:])] if name.startswith('o') else d['res'][int(name[3:])]


def head_classes(run: Path) -> np.ndarray | None:
    p = run.parent.parent / GATES_38
    if not p.exists():
        return None
    return np.array(json.loads(p.read_text())['classes'])


def zfit(X: np.ndarray):
    m, sd = X.mean(0), X.std(0) + 1e-6
    return lambda Z: (Z - m) / sd


def fixed_probes(run: Path, summary: dict) -> None:
    """Per-head probes, cross-family transfer and the no-forgetting probe shift on column-standardised inputs
    (raw DeltaNet head outputs have magnitudes of about 1e-2, so an unstandardised ridge penalty removes them)."""
    pr = summary['probes']
    data = {}
    for fam in ('jgrid', 'single', 'refcoco'):
        for cond in ('clean', 'nof'):
            d = load_family(run, fam, cond)
            if d is not None:
                data[(fam, cond)] = d
    heads = {}
    for fam in ('jgrid', 'single', 'refcoco'):
        d = data[(fam, 'clean')]
        tab, img = d['tab'], d['img']
        P = pos_features(tab)
        rc = residualise(tab['row_cnt'].astype(float), P) if fam == 'jgrid' else None
        hs = {}
        for l in NOF_LAYERS:
            Xl = d['o'][l].reshape(len(img), 32, 128)
            for t in ('y_tok', 'x_tok') + (('row_cnt|pos',) if fam == 'jgrid' else ()):
                y = rc if t == 'row_cnt|pos' else tab[t].astype(float)
                hs.setdefault(t, []).append([ps.ridge_cv_r2(zfit(Xl[:, h])(Xl[:, h]), y, img, lams=(1e0, 1e2, 1e4))['r2']
                                             for h in range(32)])
        heads[fam] = hs
        print('heads', fam, {t: float(np.max(v)) for t, v in hs.items()}, flush=True)
    pr['heads'] = heads
    tr = {}
    for name in ('o9', 'o13', 'o15', 'res16', 'res19'):
        tr[name] = {}
        for fa in ('jgrid', 'single', 'refcoco'):
            Xa = rep_of(data[(fa, 'clean')], name)
            z = zfit(Xa)
            for fb in ('jgrid', 'single', 'refcoco'):
                if fa == fb:
                    continue
                Xb = rep_of(data[(fb, 'clean')], name)
                for t in ('y_tok', 'x_tok'):
                    w, b = ps.ridge_fit(z(Xa), data[(fa, 'clean')]['tab'][t].astype(float), 1e3)
                    yp = (torch.as_tensor(z(Xb), dtype=torch.float64) @ w + b).numpy()
                    yt = data[(fb, 'clean')]['tab'][t]
                    tr[name][f'{fa}->{fb}:{t}'] = float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
    pr['transfer'] = tr
    nof = {}
    for fam in ('jgrid', 'single', 'refcoco'):
        dc, dn = data[(fam, 'clean')], data[(fam, 'nof')]
        res = {}
        for name in [f'o{l}' for l in O_LAYERS_NOF] + ['res16', 'res19']:
            Xc, Xn = rep_of(dc, name), rep_of(dn, name)
            z = zfit(Xc)
            res[name] = {}
            for t in probe_targets(fam):
                y = dc['tab'][t].astype(float)
                w, b = ps.ridge_fit(z(Xc), y, 1e3)
                pc = (torch.as_tensor(z(Xc), dtype=torch.float64) @ w + b).numpy()
                pn = (torch.as_tensor(z(Xn), dtype=torch.float64) @ w + b).numpy()
                dlt = pn - pc
                res[name][t] = {'mean_shift': float(dlt.mean()), 'slope_vs_true': float(np.polyfit(y, dlt, 1)[0]),
                                'fit_r2': float(1 - ((y - pc) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                                'by_bin': binned(y, dlt, t)}
        nof[fam] = res
        print('nof', fam, {n: {t: round(v['mean_shift'], 1) for t, v in r.items()} for n, r in res.items()}, flush=True)
    pr['nof'] = nof


# ----------------------------------------------------------------------------- steering directions

def steer_directions(data: dict, n_fit: int, out_path: Path) -> dict:
    """Unique effects of row/column counts and own position on the DeltaNet mixer outputs (layers 9-15).

    Encoding model per layer: mix (n, 4096) ~ [position one-hots, count one-hots, content flags] by ridge on
    the first n_fit jgrid images; direction 'rowcnt' = mean over k of (coef[row_cnt = k + 1] - coef[k]);
    'ypos' = mean over bins b of (coef[y-bin b + 1] - coef[b]) (one 50-unit bin = one token row on a 20-row
    image); the column analogues 'colcnt' and 'xpos'. Scales are applied as multiples of these steps.
    """
    d = data[('jgrid', 'clean')]
    keep = d['img'] < n_fit
    tab = {k: v[keep] for k, v in d['tab'].items()}
    G = groups_for(tab, 'jgrid')
    X = np.concatenate([G['position'], G['count'], G['content']], 1).astype(np.float64)
    sl = {'ypos': slice(0, 20), 'xpos': slice(20, 40), 'rowcnt': slice(40, 48), 'colcnt': slice(48, 56)}
    dirs = {n: {} for n in sl}
    norms = {n: {} for n in sl}
    for l in NOF_LAYERS:
        Y = d['mix'][l][keep].astype(np.float64)
        w, b = ps.ridge_fit(X, Y, 1.0)
        w = w.numpy()
        for n, s in sl.items():
            block = w[s]
            # use only levels with support
            cnt = X[:, s].sum(0)
            steps = [block[k + 1] - block[k] for k in range(block.shape[0] - 1) if cnt[k] > 50 and cnt[k + 1] > 50]
            v = np.mean(steps, 0)
            dirs[n][l] = torch.tensor(v, dtype=torch.float32)
            norms[n][l] = float(np.linalg.norm(v))
    torch.save({'names': ['rowcnt', 'ypos', 'colcnt', 'xpos'], 'scales': [1.0, -1.0, 2.0, 5.0, -5.0, 10.0, -10.0],
                'dirs': dirs, 'n_fit': n_fit}, out_path)
    return {'norms': norms, 'n_fit_images': int(n_fit),
            'mix_norm_median': {l: float(np.median(np.linalg.norm(d['mix'][l], axis=1))) for l in NOF_LAYERS}}


# ----------------------------------------------------------------------------- answers

def parse_answer(r: dict) -> dict:
    if r['fmt'] == 'rowcol':
        rc = cr.parse_rowcol(r['text'])
        return {'row': rc[0], 'col': rc[1]} if rc else {}
    p = cmn.parse_point(r['text'])
    if p is None:
        return {}
    x, y = p
    t = r['truth']
    if r['family'] == 'single':                                   # pixel-convention diagnostic (plan.md 6.9)
        W, H = t['canvas']
        cx, cy = t['center_units']
        px, py = x * 1000 / W, y * 1000 / H
        if np.hypot(px - cx, py - cy) < 0.5 * np.hypot(x - cx, y - cy):
            x, y = px, py
    return {'x': x, 'y': y}


def capture_answers(run: Path) -> dict:
    recs = read_jsonl(run / 'capture_answers.jsonl')
    by = {}
    for r in recs:
        by.setdefault((r['family'], r['fmt'], r['case']), {})[r['cond']] = (parse_answer(r), r['truth'])
    out = {}
    for (fam, fmt, case), v in by.items():
        if 'clean' not in v or 'nof' not in v:
            continue
        a, t = v['clean']
        b, _ = v['nof']
        e = out.setdefault(f'{fam}:{fmt}', {'dx': [], 'dy': [], 'drow': [], 'dcol': [], 'x': [], 'y': [],
                                            'hit_c': 0, 'hit_n': 0, 'n': 0, 'row_ok_c': 0, 'row_ok_n': 0})
        e['n'] += 1
        if fmt == 'point' and a and b:
            e['dx'].append(b['x'] - a['x'])
            e['dy'].append(b['y'] - a['y'])
            e['x'].append(t['center_units'][0])
            e['y'].append(t['center_units'][1])
            bx = t['box_units']
            e['hit_c'] += int(bx[0] <= a['x'] <= bx[2] and bx[1] <= a['y'] <= bx[3])
            e['hit_n'] += int(bx[0] <= b['x'] <= bx[2] and bx[1] <= b['y'] <= bx[3])
        if fmt == 'rowcol' and a and b:
            e['drow'].append(b['row'] - a['row'])
            e['dcol'].append(b['col'] - a['col'])
            e['row_ok_c'] += int(a['row'] == t['row'])
            e['row_ok_n'] += int(b['row'] == t['row'])
    summ = {}
    for k, e in out.items():
        s = {'n': e['n']}
        for key in ('dx', 'dy', 'drow', 'dcol'):
            if e[key]:
                s[key] = [float(np.mean(e[key])), *ps.boot_mean_ci(np.array(e[key]))]
        if e['dy']:
            s['dy_slope'] = float(np.polyfit(e['y'], e['dy'], 1)[0])
            s['dx_slope'] = float(np.polyfit(e['x'], e['dx'], 1)[0])
            s['hit'] = [e['hit_c'], e['hit_n']]
        if e['drow']:
            s['row_ok'] = [e['row_ok_c'], e['row_ok_n']]
        summ[k] = s
    return summ


# ----------------------------------------------------------------------------- roll

def roll_summary(run: Path, tag: str) -> dict:
    recs = read_jsonl(run / f'roll_{tag}.jsonl')
    if not recs:
        return {}
    by = {}
    for r in recs:
        by.setdefault((r['family'], r['fmt'], r['case']), {})[r['cond']] = r
    out = {}
    for (fam, fmt, case), cs in by.items():
        if 'id' not in cs:
            continue
        a0 = parse_answer(cs['id'])
        if not a0:
            continue
        for name, r in cs.items():
            if name == 'id':
                continue
            a = parse_answer(r)
            e = out.setdefault(f'{fam}:{fmt}:{name}', {'d_ans': [], 'd_seq': [], 'n': 0, 'invalid': 0, 'd_other': [],
                                                        'd_ans_other_axis': []})
            e['n'] += 1
            if not a:
                e['invalid'] += 1
                continue
            t = r['truth']
            if r['axis'] is None:                              # band sweeps: plain shifts
                if fmt == 'point':
                    e['d_ans'].append(a['y'] - a0['y'])
                    e['d_other'].append(a['x'] - a0['x'])
                continue
            ax = r['axis']
            N = r['H'] if ax == 0 else r['W']
            if fmt == 'point':
                key, okey = ('y', 'x') if ax == 0 else ('x', 'y')
                true = t['center_units'][1 if ax == 0 else 0]
                seq = ps.seq_coordinate(true, r['k'], N)
                e['d_ans'].append(a[key] - a0[key])
                e['d_seq'].append(seq - true)
                e['d_ans_other_axis'].append(a[okey] - a0[okey])
            else:
                key = 'row' if ax == 0 else 'col'
                ncell = 10
                true_c = (t[key] - 0.5) / ncell * 1000
                seq = ps.seq_coordinate(true_c, r['k'], N)
                e['d_ans'].append(a[key] - a0[key])
                e['d_seq'].append((seq - true_c) / 1000 * ncell)
    summ = {}
    for k, e in out.items():
        s = {'n': e['n'], 'invalid': e['invalid']}
        if e['d_seq']:
            s['follow'] = ps.following_ratio(np.zeros(len(e['d_ans'])), np.array(e['d_ans']), np.zeros(len(e['d_seq'])),
                                             np.array(e['d_seq']))
            s['other_axis_shift'] = float(np.mean(np.abs(e['d_ans_other_axis']))) if e['d_ans_other_axis'] else None
        elif e['d_ans']:
            s['dy'] = [float(np.mean(e['d_ans'])), *ps.boot_mean_ci(np.array(e['d_ans']))]
            s['dx'] = [float(np.mean(e['d_other'])), *ps.boot_mean_ci(np.array(e['d_other']))]
        summ[k] = s
    return summ


# ----------------------------------------------------------------------------- steer

def steer_summary(run: Path) -> dict:
    recs = read_jsonl(run / 'steer.jsonl')
    if not recs:
        return {}
    by = {}
    for r in recs:
        by.setdefault((r['family'], r['fmt'], r['case']), {})[r['cond']] = r
    out = {}
    for (fam, fmt, case), cs in by.items():
        if 'clean' not in cs:
            continue
        a0 = parse_answer(cs['clean'])
        if not a0:
            continue
        for name, r in cs.items():
            if name == 'clean':
                continue
            a = parse_answer(r)
            e = out.setdefault(f'{fam}:{fmt}:{name}', {'n': 0, 'invalid': 0, 'd': {}})
            e['n'] += 1
            if not a:
                e['invalid'] += 1
                continue
            for k in a:
                e['d'].setdefault(k, []).append(a[k] - a0[k])
    return {k: {'n': e['n'], 'invalid': e['invalid'],
                'shift': {kk: [float(np.mean(v)), *ps.boot_mean_ci(np.array(v))] for kk, v in e['d'].items()}}
            for k, e in out.items()}


# ----------------------------------------------------------------------------- Qwen3-VL flow

def vlflow_summary(run: Path) -> dict:
    recs = read_jsonl(run / 'vlflow.jsonl')
    if not recs:
        return {}
    out = {}
    for fam in sorted({r['family'] for r in recs}):
        rs = [r for r in recs if r['family'] == fam]
        heads = rs[0]['heads']
        fs = {}
        for digit in range(3):
            for kind in ('step', 'win'):
                conds = None
                lA, lB, lX = [], [], {}
                for r in rs:
                    for v in r['values']:
                        if v['digit'] != digit:
                            continue
                        d = v[kind]
                        conds = d['conds']
                        lp = np.array(d['lp'])
                        lA.append(lp[conds.index('A')])
                        lB.append(lp[conds.index('B')])
                        for c in conds:
                            lX.setdefault(c, []).append(lp[conds.index(c)])
                if conds is None:
                    continue
                fs[f'{kind}_d{digit}'] = {c: [cmn.recovery(lA, lB, lX[c]), *cmn.recovery_ci(lA, lB, lX[c])]
                                          for c in conds if c not in ('A', 'B')}
                fs[f'{kind}_d{digit}']['n_values'] = len(lA)
                fs[f'{kind}_d{digit}']['diff_norm'] = float(np.mean(np.linalg.norm(
                    (np.array(lB) - np.array(lB).mean(1, keepdims=True)) - (np.array(lA) - np.array(lA).mean(1, keepdims=True)), axis=1)))
            # B-image visibility scan
            fs[f'tail_d{digit}'] = tail_recovery(rs, digit)
        # attention share per relative step
        att = {}
        for r in rs:
            if not r.get('att'):
                continue
            sh = np.array(r['att']['share'])                  # (L, H, n_steps)
            it = np.array(r['att']['in_target'])
            for vs in r['value_starts']:
                for off in (-3, -2, -1, 0, 1, 2):
                    if vs + off in r['att_steps']:
                        i = r['att_steps'].index(vs + off)
                        att.setdefault(off, {'share': [], 'in_target': []})
                        att[off]['share'].append(sh[:, :, i])
                        att[off]['in_target'].append(it[:, :, i])
        fs['attention'] = {int(o): {'share_layer_mean': np.mean(v['share'], 0).mean(1).round(4).tolist(),
                                    'in_target_layer_mean': np.mean(v['in_target'], 0).mean(1).round(4).tolist(),
                                    'top_in_target': top_heads(np.mean(v['in_target'], 0))}
                           for o, v in att.items()}
        fs['heads'] = heads
        fs['n_pairs'] = len(rs)
        out[fam] = fs
    return out


def tail_recovery(rs: list[dict], digit: int) -> dict:
    by = {}
    for r in rs:
        for v in r['values']:
            if v['digit'] != digit:
                continue
            conds = v['step']['conds']
            lp = np.array(v['step']['lp'])
            a, b = lp[conds.index('A')], lp[conds.index('B')]
            for s_, row in zip(v['tail']['starts'], v['tail']['lp']):
                by.setdefault(s_, ([], [], []))
                by[s_][0].append(a)
                by[s_][1].append(b)
                by[s_][2].append(np.array(row)[2])
    return {int(s_): [cmn.recovery(*v), len(v[0])] for s_, v in sorted(by.items())}


def top_heads(m: np.ndarray, k: int = 8) -> list[list]:
    idx = np.dstack(np.unravel_index(np.argsort(-m.ravel()), m.shape))[0][:k]
    return [[int(l) + 1, int(h), round(float(m[l, h]), 3)] for l, h in idx]


# ----------------------------------------------------------------------------- main

def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--parts', default='probes,roll,steer,vlflow,figures')
    args = p.parse_args()
    torch.set_num_threads(16)
    parts = args.parts.split(',')
    sp = args.run_dir / 'summary.json'
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    if 'probes' in parts:
        pr, data = run_probes(args.run_dir)
        summary['probes'] = pr
        summary['capture_answers'] = capture_answers(args.run_dir)
        if ('jgrid', 'clean') in data:
            n_img = int(data[('jgrid', 'clean')]['img'].max()) + 1
            summary['steer_dirs'] = steer_directions(data, n_img - 48, args.run_dir / 'steer_dirs.pt')
        del data
    if 'fix' in parts:
        fixed_probes(args.run_dir, summary)
    if 'roll' in parts:
        summary['roll'] = {t: roll_summary(args.run_dir, t) for t in ('q35', 'vl')}
    if 'steer' in parts:
        summary['steer'] = steer_summary(args.run_dir)
    if 'vlflow' in parts:
        summary['vlflow'] = vlflow_summary(args.run_dir)
    sp.write_text(json.dumps(to_jsonable(summary), indent=1))
    if 'figures' in parts:
        from analysis_tools.position import position_sources_figures as pf
        pf.make_all(args.run_dir, args.figures, summary)


if __name__ == '__main__':
    main()
