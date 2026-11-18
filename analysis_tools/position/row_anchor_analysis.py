"""Experiment 3.14 round 4: statistics, probes and figures of the row-anchor run (CPU).

Results: docs/analysis/position/results_2026-11-11_row_anchor.md. Inputs are written by
analysis_tools/position/row_anchor.py; the RefCOCO left-edge table re-uses round-3 maps.

Definitions (all at the query predicting a coordinate's first digit, main reading head unless stated):
* content score c[k, j] of token (row k, column j): q~ . R(ref) k / sqrt(d) (row_anchor.measure).
* interior columns J: columns 4 .. W-5 (W // 4 .. for narrow images) without the target's columns.
* left-edge bonus: mean over token rows k without target tokens of c[k, 0] - mean_{j in J} c[k, j].
* first-figure bonus: the same for the first column with drawn pixels in row k, rows whose first drawn column
  is > 0 only (blank left margin).
* anchor share: image attention on the column-0 tokens of the target's token rows (target excluded).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.position import row_anchor as rn
from analysis_tools.position.row_attention_analysis import boot, use_cjk_font

plt.rcParams['axes.unicode_minus'] = False
MODELS = {'vl': 'Qwen3-VL-8B', 'q35': 'Qwen3.5-9B'}
MAIN = {'vl': (24, 30), 'q35': (20, 11)}
XHEAD = {'vl': (25, 13), 'q35': (20, 11)}
# Round-2 single-head recovered fractions of the row/column y reading heads (weights of the weighted share).
YW = {'vl': {(24, 30): 0.16, (22, 11): 0.15, (29, 16): 0.10, (35, 28): 0.10, (22, 8): 0.07},
      'q35': {(20, 11): 0.25, (32, 8): 0.10, (24, 0): 0.07, (32, 14): 0.06, (24, 4): 0.04}}
VLABEL = {'edge': '网格（贴左边缘）', 'margin_left': '网格+左侧留白', 'margin_all': '网格+四周留白',
          'noise': '网格+纹理背景', 'photo': '网格+照片背景', 'table': '数字表格', 'scatter': '散布图形',
          'single': '单个图形', 'photo_single': '照片+单个图形', 'doc': '文本页（左边距 64 px）',
          'doc_edge': '文本页（左边距 4 px）'}
Y_KEYS, X_KEYS = ('y', 'row'), ('x', 'column')


# ----------------------------------------------------------------------------- per-case metrics

def qindex(keys, axis: str):
    ks = [str(k) for k in keys]
    want = Y_KEYS if axis == 'y' else X_KEYS
    return next((i for i, k in enumerate(ks) if k in want), None)


def interior(W: int, tcols: set[int], drop: set[int] = frozenset()) -> list[int]:
    """Columns 4 .. W-5 (W // 4 .. for narrow images) without the target columns and `drop`."""
    m = min(4, W // 4)
    return [j for j in range(m, W - m) if j not in tcols and j not in drop]


def first_ink(ink: np.ndarray, thr: float = 0.02) -> list[int | None]:
    """First column with drawn pixels in every token row of an (H, W) ink map (None for empty rows)."""
    return [next((j for j in range(ink.shape[1]) if ink[k, j] > thr), None) for k in range(ink.shape[0])]


def case_metrics(con: np.ndarray, att: np.ndarray, grid, target, ink: np.ndarray | None) -> dict:
    """Left-edge / first-figure bonus and attention shares of one (head, query); con, att: (Nv,)."""
    H, W = (int(v) for v in grid)
    T = set(int(t) for t in target)
    trows, tcols = {t // W for t in T}, {t % W for t in T}
    c, a = con.astype(float).reshape(H, W), att.astype(float).reshape(H, W)
    J = interior(W, tcols)
    rows = [k for k in range(H) if k not in trows]
    out = {'edge_bonus': float(np.mean([c[k, 0] - c[k, J].mean() for k in rows])),
           'right_bonus': float(np.mean([c[k, W - 1] - c[k, J].mean() for k in rows])),
           'top_bonus': float(np.mean([c[0, j] - c[min(4, H // 4):H - min(4, H // 4), j].mean()
                                        for j in range(W) if j not in tcols])),
           'anchor': float(sum(a[k, 0] for k in trows if k * W not in T)),
           'target': float(sum(a.reshape(-1)[t] for t in T)),
           'col0_all': float(a[:, 0].sum())}
    if ink is not None:
        fi = first_ink(ink.reshape(H, W))
        fb = [c[k, fi[k]] - c[k, interior(W, tcols, {fi[k]})].mean() for k in rows if fi[k] not in (None, 0)]
        out['fig_bonus'] = float(np.mean(fb)) if fb else None
        fa = [a[k, fi[k]] for k in trows if fi[k] not in (None, 0) and k * W + fi[k] not in T]
        out['fig_share'] = float(sum(fa)) if fa else None
    return out


def head_idx(heads, head) -> int | None:
    hs = [tuple(int(v) for v in h) for h in heads]
    return hs.index(tuple(head)) if tuple(head) in hs else None


# ----------------------------------------------------------------------------- factors

def load_dir(path: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(path.glob('*.npz')):
        z = np.load(f, allow_pickle=False)
        out[f.stem] = {k: z[k] for k in z.files}
    return out


def factor_summary(data: dict, tag: str) -> dict:
    res = defaultdict(lambda: defaultdict(list))
    for cid, r in data.items():
        meta = json.loads(str(r['meta']))
        key = f"{meta['variant']}|{meta['format']}"
        for axis, head in (('y', MAIN[tag]), ('x', XHEAD[tag])):
            qi, hi = qindex(r['keys'], axis), head_idx(r['heads'], head)
            if qi is None or hi is None:
                continue
            m = case_metrics(r['con'][hi, qi], r['att'][hi, qi], r['grid'], r['target'], r['ink'])
            for k, v in m.items():
                res[key][f'{axis}_{k}'].append(v)
        qi = qindex(r['keys'], 'y')
        num, den = 0.0, 0.0
        for h, w in YW[tag].items():
            hi = head_idx(r['heads'], h)
            if hi is None:
                continue
            m = case_metrics(r['con'][hi, qi], r['att'][hi, qi], r['grid'], r['target'], r['ink'])
            num += w * m['anchor']
            den += w
        res[key]['y_anchor_weighted'].append(num / den)
    return {k: {m: boot(v) for m, v in d.items()} for k, d in res.items()}


def column_profiles(data: dict, tag: str) -> dict:
    """Mean (c[k, j] - interior mean) over target-free token rows, per variant and format, y query, main head."""
    acc = defaultdict(list)
    for r in data.values():
        meta = json.loads(str(r['meta']))
        qi, hi = qindex(r['keys'], 'y'), head_idx(r['heads'], MAIN[tag])
        H, W = (int(v) for v in r['grid'])
        T = set(int(t) for t in r['target'])
        trows, tcols = {t // W for t in T}, {t % W for t in T}
        c = r['con'][hi, qi].astype(float).reshape(H, W)
        J = interior(W, tcols)
        prof = np.mean([c[k] - c[k, J].mean() for k in range(H) if k not in trows], 0)
        acc[f"{meta['variant']}|{meta['format']}"].append(prof)
    return {k: np.mean(v, 0).tolist() for k, v in acc.items()}


def cross_matrix(data: dict, tag: str, fmt: str) -> dict:
    """Left-edge bonus of variant-A queries (main head, y) against variant-B content keys.

    Every query of A is scored against every case of B (content rotation of B's keys; the positional part of
    the score is negligible, round 3); rows of B that hold its target are excluded."""
    byv = defaultdict(list)
    for r in data.values():
        meta = json.loads(str(r['meta']))
        if meta['format'] == fmt:
            byv[meta['variant']].append(r)
    names = [v for v in rn.VARIANTS if v in byv]
    hi = None
    Q, K = {}, {}
    for v in names:
        Q[v] = []
        K[v] = []
        for r in byv[v]:
            hi = head_idx(r['heads'], MAIN[tag])
            Q[v].append(r['q'][hi, qindex(r['keys'], 'y')].astype(np.float32))
            H, W = (int(x) for x in r['grid'])
            T = set(int(t) for t in r['target'])
            trows, tcols = {t // W for t in T}, {t % W for t in T}
            J = interior(W, tcols)
            kc = r['kc'].astype(np.float32).reshape(H, W, -1)
            rows = [k for k in range(H) if k not in trows]
            # bonus is linear in the key: one difference vector per case
            K[v].append(np.mean([kc[k, 0] - kc[k, J].mean(0) for k in rows], 0))
    d = next(iter(Q.values()))[0].shape[-1]
    mat = [[float(np.mean(np.stack(Q[a]) @ np.stack(K[b]).T) / np.sqrt(d)) for b in names] for a in names]
    return {'variants': names, 'matrix': mat}


# ----------------------------------------------------------------------------- locate

def locate_summary(run: Path, tag: str) -> dict:
    files = sorted((run / f'locate_{tag}').glob('grid_*.npz'))
    acc = defaultdict(lambda: defaultdict(list))
    names = None
    for f in files:
        z = np.load(f)
        names = [str(n) for n in z['conds']]
        H, W = (int(v) for v in z['grid'])
        S = int(z['S'])
        T = set(int(t) for t in z['target'])
        trows, tcols = {t // W for t in T}, {t % W for t in T}
        hi = head_idx(z['heads'], MAIN[tag])
        qi = qindex(z['keys'], 'y')
        rows = [k for k in range(H) if k not in trows]
        J = interior(W, tcols, {S, S + 1})
        for ci, n in enumerate(names):
            con = z['con'][ci, hi, qi].astype(float).reshape(H, W)
            cq = z['cq'][ci, qi].astype(float).reshape(H, W)
            att = z['att'][ci, hi, qi].astype(float).reshape(H, W)
            col = S if n.startswith('imp_') else 0
            acc[n]['bonus_native'].append(np.mean([con[k, col] - con[k, J].mean() for k in rows]))
            acc[n]['bonus_keys'].append(np.mean([cq[k, col] - cq[k, J].mean() for k in rows]))
            acc[n]['anchor'].append(sum(att[k, col] for k in trows))
            num = den = 0.0
            for h, w in YW[tag].items():
                h2 = head_idx(z['heads'], h)
                if h2 is None:
                    continue
                a2 = z['att'][ci, h2, qi].astype(float).reshape(H, W)
                num += w * sum(a2[k, col] for k in trows)
                den += w
            acc[n]['anchor_weighted'].append(num / den)
            acc[n]['target'].append(float(sum(att.reshape(-1)[t] for t in T)))
    out = {n: {m: boot(v) for m, v in acc[n].items()} for n in (names or [])}
    # paired fraction of the clean bonus per image
    if names:
        clean = np.array(acc['clean']['bonus_keys'])
        for n in names:
            out[n]['frac_keys'] = boot(np.array(acc[n]['bonus_keys']) / clean)
    return {'conditions': names or [], 'n': len(files), 'stats': out}


# ----------------------------------------------------------------------------- stage-wise probes

def kernel_ridge_loo_groups(X: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float = 0.1) -> float:
    """Leave-one-group-out R^2 of linear ridge regression (dual form; alpha = lam * mean diagonal of XX^T)."""
    pred = np.zeros_like(y, dtype=float)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        mu = X[tr].mean(0)
        A, B = X[tr] - mu, X[te] - mu
        K = A @ A.T
        alpha = lam * np.trace(K) / len(K)
        ym = y[tr].mean()
        coef = np.linalg.solve(K + alpha * np.eye(len(K)), y[tr] - ym)
        pred[te] = B @ (A.T @ coef) + ym
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def edge_dprime(h0: np.ndarray, hS: np.ndarray, groups: np.ndarray) -> float:
    """Held-out separability of column 0 vs column S along the mean difference direction of the other images."""
    proj0, projS = [], []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        w = (h0[tr] - hS[tr]).mean(0)
        w = w / (np.linalg.norm(w) + 1e-9)
        proj0.append(h0[te] @ w)
        projS.append(hS[te] @ w)
    p0, pS = np.concatenate(proj0), np.concatenate(projS)
    return float((p0.mean() - pS.mean()) / np.sqrt(0.5 * (p0.var() + pS.var()) + 1e-9))


def stage_probes(run: Path, tag: str) -> dict:
    files = sorted((run / f'locate_{tag}').glob('stats_*.npz'))
    if not files:
        return {}
    per = defaultdict(list)
    for gi, f in enumerate(files):
        z = np.load(f)
        H, W = (int(v) for v in z['grid'])
        S = int(z['S'])
        tok = [int(t) for t in z['stat_tokens']]
        cols = np.array([t % W for t in tok])
        rows = np.array([t // W for t in tok])
        for name, arr in (('vit', z['vit']), ('dec', z['dec'])):
            per[name].append((arr.astype(np.float32), cols, rows, S, gi))
    out = {}
    for name, items in per.items():
        n_st = items[0][0].shape[0]
        res = {'norm_ratio': [], 'dprime': [], 'r2_col0': [], 'r2_colS': [], 'r2_col10': []}
        for s in range(n_st):
            h0, hS, h10, y0, yS, y10, g0 = [], [], [], [], [], [], []
            for arr, cols, rows, S, gi in items:
                o0 = np.argsort(rows[cols == 0])
                oS = np.argsort(rows[cols == S])
                h0.append(arr[s][cols == 0][o0])
                hS.append(arr[s][cols == S][oS])
                h10.append(arr[s][cols == 10])
                y0.append(np.sort(rows[cols == 0]))
                yS.append(np.sort(rows[cols == S]))
                y10.append(rows[cols == 10])
                g0.append(np.full((cols == 0).sum(), gi))
            h0, hS, h10 = np.concatenate(h0), np.concatenate(hS), np.concatenate(h10)
            y0, yS, y10, g0 = (np.concatenate(v).astype(float) for v in (y0, yS, y10, g0))
            res['norm_ratio'].append(float(np.linalg.norm(h0, axis=1).mean() / np.linalg.norm(hS, axis=1).mean()))
            res['dprime'].append(edge_dprime(h0, hS, g0))
            res['r2_col0'].append(kernel_ridge_loo_groups(h0, y0, g0))
            res['r2_colS'].append(kernel_ridge_loo_groups(hS, yS, g0))
            res['r2_col10'].append(kernel_ridge_loo_groups(h10, y10, g0))
        out[name] = res
    out['n_images'] = len(files)
    return out


def value_probes(data: dict, tag: str) -> dict:
    """Leave-one-image-out R^2 of the token-row index from the main head's values, per column class."""
    out = {}
    for variant, fmt, classes in (('edge', 'rowcol', {'col0': [0], 'col1': [1], 'col2': [2], 'col10': [10]}),
                                  ('margin_left', 'rowcol', {'col0': [0], 'col2': [2], 'col4': [4], 'col10': [10]}),
                                  ('single', 'point', {'col0': [0], 'col10': [10]}),
                                  ('photo', 'rowcol', {'col0': [0], 'col2': [2], 'col10': [10]})):
        cases = [r for r in data.values() if json.loads(str(r['meta']))['variant'] == variant
                 and json.loads(str(r['meta']))['format'] == fmt]
        if not cases:
            continue
        res = {}
        for cname, cols in classes.items():
            X, y, g = [], [], []
            for gi, r in enumerate(cases):
                H, W = (int(v) for v in r['grid'])
                v = r['v'].astype(np.float32).reshape(H, W, -1)
                T = set(int(t) for t in r['target'])
                for k in range(H):
                    for j in cols:
                        if k * W + j in T:
                            continue
                        X.append(v[k, j])
                        y.append(k)
                        g.append(gi)
            res[cname] = kernel_ridge_loo_groups(np.array(X), np.array(y, float), np.array(g))
        out[f'{variant}|{fmt}'] = res
    return out


# ----------------------------------------------------------------------------- edits and RefCOCO

def edit_summary(run: Path, tag: str) -> dict:
    path = run / f'edits_{tag}.jsonl'
    if not path.exists():
        return {}
    acc = defaultdict(lambda: defaultdict(list))
    n_bad = defaultdict(int)
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        for v in rec['values']:
            if v['axis'] != 'v':
                continue
            ld = v['ld']
            den = ld[0] - ld[1]
            if v['dA'] == v['dB'] or abs(den) < 2:
                n_bad[rec['variant']] += 1
                continue
            for i, name in enumerate(rec['rows'][1:], start=2):
                if name == 'vT' or rec['sizes'].get(name, 0):
                    acc[rec['variant']][name].append((ld[i] - ld[1]) / den)
    return {v: {**{k: boot(x) for k, x in d.items()}, 'undefined': n_bad[v]} for v, d in acc.items()}


def refcoco_edge(r3: Path, tag: str) -> dict:
    """Round-3 RefCOCO maps: left-edge bonus (y query, main head) split by whether the target box reaches the
    first token column (its column match then also favours column 0)."""
    split = defaultdict(list)
    for f in sorted((r3 / f'maps_{tag}').glob('rc*_point.npz')):
        z = np.load(f)
        hi, qi = head_idx(z['heads'], MAIN[tag]), qindex(z['keys'], 'y')
        H, W = (int(v) for v in z['grid'])
        c = z['comp'][hi, qi].astype(float)
        con = c[4:7].sum(0) + c[3]
        m = case_metrics(con, z['att'][hi, qi], z['grid'], z['target'], None)
        touch = any(int(t) % W == 0 for t in z['target'])
        split['touch' if touch else 'inner'].append(m['edge_bonus'])
        split[('touch' if touch else 'inner') + '_anchor'].append(m['col0_all'])
    return {k: boot(v) for k, v in split.items()}


# ----------------------------------------------------------------------------- figures

def fig_examples(run: Path, data: dict[str, dict], path: Path, fmt_pref=('rowcol', 'point')) -> None:
    """One example per variant: image, then the y-query attention of each model's main head (own colour bar)."""
    variants = list(rn.VARIANTS)
    fig, axes = plt.subplots(len(variants), 3, figsize=(10.5, 3.1 * len(variants)))
    for i, v in enumerate(variants):
        fmt = 'rowcol' if 'rowcol' in rn.VARIANTS[v] else 'point'
        cid = f'{v}_000_{fmt}'
        img = Image.open(run / 'factor_images' / f'{v}_000.png').convert('RGB')
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'{VLABEL[v]}（{fmt}）', fontsize=9)
        for j, tag in enumerate(('vl', 'q35'), start=1):
            ax = axes[i, j]
            r = data.get(tag, {}).get(cid)
            if r is None:
                ax.axis('off')
                continue
            H, W = (int(x) for x in r['grid'])
            hi, qi = head_idx(r['heads'], MAIN[tag]), qindex(r['keys'], 'y')
            a = r['att'][hi, qi].astype(float).reshape(H, W)
            a = a / a.sum()
            ax.imshow(img, alpha=0.45, extent=(0, W, H, 0))
            im = ax.imshow(a, cmap='viridis', alpha=0.75, extent=(0, W, H, 0))
            for t in r['target']:
                k, c = divmod(int(t), W)
                ax.add_patch(plt.Rectangle((c, k), 1, 1, fill=False, ec='red', lw=1))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label('图像内 attention 份额', fontsize=7)
            ax.set_title(f'{MODELS[tag]} L{MAIN[tag][0]} H{MAIN[tag][1]} 纵坐标 query', fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_factor_bars(fs: dict, path: Path) -> None:
    variants = list(rn.VARIANTS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for col, tag in enumerate(('vl', 'q35')):
        s = fs.get(tag, {})
        for row, (metric, lab) in enumerate((('y_edge_bonus', '左边缘加分（内容分数差）'),
                                             ('y_anchor', '行首 token 的 attention 份额'))):
            ax = axes[row, col]
            xs = np.arange(len(variants))
            for off, fmt, colr in ((-0.18, 'rowcol', '#1f77b4'), (0.18, 'point', '#ff7f0e')):
                vals = [s.get(f'{v}|{fmt}', {}).get(metric) for v in variants]
                m = [x['mean'] if x and x['mean'] is not None else np.nan for x in vals]
                lo = [x['mean'] - x['lo'] if x and x['mean'] is not None else 0 for x in vals]
                hi = [x['hi'] - x['mean'] if x and x['mean'] is not None else 0 for x in vals]
                ax.bar(xs + off, m, 0.36, yerr=[lo, hi], color=colr, label=f'{fmt} 格式', capsize=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([VLABEL[v] for v in variants], rotation=40, ha='right', fontsize=8)
            ax.set_ylabel(lab)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_title(f'{MODELS[tag]} L{MAIN[tag][0]} H{MAIN[tag][1]}')
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_profiles(prof: dict, path: Path) -> None:
    sel = [('edge', 'rowcol'), ('margin_left', 'rowcol'), ('margin_all', 'rowcol'), ('noise', 'rowcol'),
           ('photo', 'rowcol'), ('table', 'rowcol'), ('scatter', 'point'), ('single', 'point'),
           ('doc', 'point'), ('doc_edge', 'point')]
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, tag in zip(axes, ('vl', 'q35')):
        for i, (v, f) in enumerate(sel):
            p = prof.get(tag, {}).get(f'{v}|{f}')
            if p is None:
                continue
            ax.plot(np.arange(len(p)), p, '-o', ms=2.5, color=cmap(i), label=f'{VLABEL[v]}（{f}）')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('token 列号（0 = 图像最左列）')
        ax.set_ylabel('内容分数 − 行内部均值')
        ax.set_title(f'{MODELS[tag]} L{MAIN[tag][0]} H{MAIN[tag][1]}，纵坐标 query')
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_cross(cross: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, tag in zip(axes, ('vl', 'q35')):
        c = cross.get(tag)
        if not c:
            ax.axis('off')
            continue
        m = np.array(c['matrix'])
        lim = np.abs(m).max()
        im = ax.imshow(m, cmap='RdBu_r', vmin=-lim, vmax=lim)
        labs = [VLABEL[v] for v in c['variants']]
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=8)
        ax.set_xlabel('key 来自的图像类型')
        ax.set_ylabel('query 来自的图像类型')
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(j, i, f'{m[i, j]:.1f}', ha='center', va='center', fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.046).set_label('左边缘加分（内容分数差）')
        ax.set_title(f'{MODELS[tag]} L{MAIN[tag][0]} H{MAIN[tag][1]}（point 格式）')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_locate(loc: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), sharex='col')
    for col, tag in enumerate(('vl', 'q35')):
        L = loc.get(tag)
        if not L or not L['conditions']:
            continue
        names = [n for n in L['conditions'] if not n.startswith('imp_')]
        xs = np.arange(len(names))
        for row, (metric, lab) in enumerate((('bonus_keys', '左边缘加分（干净 query，补丁后 key）'),
                                             ('anchor', '目标行行首 attention 份额'))):
            ax = axes[row, col]
            m = np.array([L['stats'][n][metric]['mean'] for n in names])
            lo = m - np.array([L['stats'][n][metric]['lo'] for n in names])
            hi = np.array([L['stats'][n][metric]['hi'] for n in names]) - m
            colors = ['#7f7f7f' if n == 'clean' else '#2ca02c' if n.startswith('vit') else
                      '#d62728' if n.startswith('emb') else '#1f77b4' for n in names]
            ax.bar(xs, m, yerr=[lo, hi], color=colors, capsize=1.5)
            ax.set_ylabel(lab, fontsize=9)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_title(f'{MODELS[tag]} L{MAIN[tag][0]} H{MAIN[tag][1]}', fontsize=10)
        axes[1, col].set_xticks(xs)
        axes[1, col].set_xticklabels([n.replace('vit', 'ViT').replace('dec', 'D') for n in names],
                                     rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_probes(pr: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for row, tag in enumerate(('vl', 'q35')):
        p = pr.get(tag)
        if not p:
            continue
        nv = len(p['vit']['dprime'])
        nd = len(p['dec']['dprime'])
        x = np.arange(nv + nd)
        lab = [f'ViT{b}' for b in range(nv)] + [f'D{l}' for l in range(1, nd + 1)]
        for ax, (metric, title) in zip(axes[row], (('dprime', '第 0 列与第 S 列的可分性 d′'),
                                                   ('norm_ratio', '范数比 ‖第 0 列‖ / ‖第 S 列‖'),
                                                   ('r2', '行号的留一图像 ridge R²'))):
            if metric == 'r2':
                for k, c, lb in (('r2_col0', '#d62728', '第 0 列'), ('r2_colS', '#1f77b4', '第 S 列'),
                                 ('r2_col10', '#2ca02c', '第 10 列')):
                    ax.plot(x, p['vit'][k] + p['dec'][k], '-o', ms=2, color=c, label=lb)
                ax.legend(fontsize=8)
            else:
                ax.plot(x, p['vit'][metric] + p['dec'][metric], '-o', ms=2, color='#444')
            ax.axvline(nv - 0.5, color='k', ls='--', lw=0.6)
            ax.set_xticks(x[::3])
            ax.set_xticklabels(lab[::3], rotation=90, fontsize=7)
            ax.set_title(f'{MODELS[tag]}：{title}', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_edits(ed: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    groups = [('vT', '目标', '#d62728'), ('vA0', '第 0 列（行首）', '#1f3a93'), ('vF', '首个图形 token', '#9b59b6'),
              ('vRin', '行内其余（目标左侧）', '#5dade2')]
    for ax, tag in zip(axes, ('vl', 'q35')):
        e = ed.get(tag, {})
        for vi, v in enumerate(('edge', 'margin_left')):
            for gi, (g, lab, c) in enumerate(groups):
                s = e.get(v, {}).get(g)
                if not s or s['mean'] is None:
                    continue
                x = vi * 5 + gi
                ax.bar(x, s['mean'], yerr=[[s['mean'] - s['lo']], [s['hi'] - s['mean']]], color=c,
                       label=lab if vi == 1 or g != 'vF' else None, capsize=2)
        ax.set_xticks([1.5, 6.5])
        ax.set_xticklabels([VLABEL['edge'], VLABEL['margin_left']])
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel('恢复比例 r')
        ax.set_title(MODELS[tag])
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def row_match(data: dict, tag: str) -> dict:
    """Table 3 (rowcol, y query, main head; content scores minus the image mean): row match M at column 0 and at
    the first drawn token of each row, the target-row column-0 score and the mean target score."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in data.values():
        meta = json.loads(str(r['meta']))
        if meta['format'] != 'rowcol':
            continue
        H, W = (int(v) for v in r['grid'])
        T = [int(t) for t in r['target']]
        tr = {t // W for t in T}
        hi, qi = head_idx(r['heads'], MAIN[tag]), qindex(r['keys'], 'y')
        c = r['con'][hi, qi].astype(float).reshape(H, W)
        c = c - c.mean()
        fi = first_ink(r['ink'].reshape(H, W))
        others = [k for k in range(H) if k not in tr]
        a = acc[meta['variant']]
        a['M_col0'].append(np.mean([c[k, 0] for k in tr]) - np.mean([c[k, 0] for k in others]))
        a['M_first'].append(np.mean([c[k, fi[k]] for k in tr if fi[k] is not None])
                            - np.mean([c[k, fi[k]] for k in others if fi[k] is not None]))
        a['c_anchor'].append(np.mean([c[k, 0] for k in tr]))
        a['c_target'].append(np.mean([c.reshape(-1)[t] for t in T]))
    return {v: {k: boot(x) for k, x in d.items()} for v, d in acc.items()}


def score_race(data: dict, tag: str) -> dict:
    """Per case (y query, main head): content score of the best row-start token of the target rows minus the
    mean content score of the target tokens, against the anchor share. Softmax gives the anchor more mass than
    the target exactly when this difference is positive (per token), so it should predict the anchor."""
    pts = []
    for r in data.values():
        meta = json.loads(str(r['meta']))
        hi, qi = head_idx(r['heads'], MAIN[tag]), qindex(r['keys'], 'y')
        H, W = (int(v) for v in r['grid'])
        T = [int(t) for t in r['target']]
        trows = sorted({t // W for t in T})
        c = r['con'][hi, qi].astype(float).reshape(-1)
        a = r['att'][hi, qi].astype(float).reshape(-1)
        anchors = [k * W for k in trows if k * W not in T]
        if not anchors:
            continue
        d = max(c[t] for t in anchors) - np.mean([c[t] for t in T])
        pts.append((meta['variant'], meta['format'], float(d), float(sum(a[t] for t in anchors))))
    d = np.array([p[2] for p in pts])
    y = np.array([p[3] for p in pts])
    return {'points': pts, 'corr': float(np.corrcoef(d, y)[0, 1]),
            'share_pos': boot(y[d > 0]), 'share_neg': boot(y[d <= 0])}


def fig_race(race: dict, path: Path) -> None:
    cmap = plt.get_cmap('tab20')
    names = list(rn.VARIANTS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, tag in zip(axes, ('vl', 'q35')):
        r = race.get(tag)
        if not r:
            continue
        for i, v in enumerate(names):
            for fmt, mk in (('rowcol', 'o'), ('point', '^')):
                p = [(x[2], x[3]) for x in r['points'] if x[0] == v and x[1] == fmt]
                if p:
                    p = np.array(p)
                    ax.scatter(p[:, 0], p[:, 1], s=10, marker=mk, color=cmap(i), alpha=0.7,
                               label=VLABEL[v] if fmt == 'point' or 'point' not in rn.VARIANTS[v] else None)
        ax.axvline(0, color='k', lw=0.6, ls='--')
        ax.set_xlabel('目标行行首的内容分数 − 目标的内容分数')
        ax.set_ylabel('目标行行首的 attention 份额')
        ax.set_title(f"{MODELS[tag]} L{MAIN[tag][0]} H{MAIN[tag][1]}（r = {r['corr']:.2f}；圆点 rowcol，三角 point）",
                     fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    axes[1].legend(uniq.values(), uniq.keys(), fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_flow(path: Path) -> None:
    """Schematic of the anchor mechanism (drawn, not measured)."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.axis('off')
    boxes = [
        (0.01, 0.55, 0.17, 0.38, '① ViT + merger\n（含 DeepStack）\n第 0 列 token 的 hidden state 中\n只带有约 2 成左边缘特征'),
        (0.21, 0.55, 0.19, 0.38, '② decoder 前段\nQwen3-VL 第 1–15 层\n（主要在第 14–15 层）\nQwen3.5 第 1–12 层\n依据 token 的位置写入\n“我位于图像最左列”特征'),
        (0.43, 0.55, 0.17, 0.38, '③ 行身份\n第 0 列 token 若含有\n本行的图形或纹理，\n其 key 带有清晰的行号；\n空白边距 token 的行号弱'),
        (0.63, 0.55, 0.17, 0.38, '④ 读取 head 的 key 分数\n行首 = 左边缘加分 + 行匹配\n目标 = 外观匹配 + 行匹配\n两者比较决定 attention'),
        (0.83, 0.55, 0.16, 0.38, '⑤ value 写入\n行首 value 中线性的行号\n经 softmax 加权\n写入纵坐标的首位数字'),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01', fc='#eef3fb', ec='#1f3a93'))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=9)
    for x0, x1 in ((0.18, 0.21), (0.40, 0.43), (0.60, 0.63), (0.80, 0.83)):
        ax.annotate('', xy=(x1, 0.74), xytext=(x0, 0.74), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(0.5, 0.36, '开启锚点的图像性质：大量重复的离散图形或文字（网格、表格、文本）使左边缘加分大；'
            '第 0 列含有本行内容使行匹配成立；行号格式的问题进一步提高行首份额。',
            ha='center', fontsize=10)
    ax.text(0.5, 0.22, '关闭锚点的性质：单个物体或自然照片（左边缘加分小、目标外观分数高）；'
            '左侧空白边距（Qwen3-VL 的 attention 改落在本行第一个图形上）。', ha='center', fontsize=10)
    ax.text(0.5, 0.06, '示意图，由本文测量结果归纳绘制，不是测量数据。', ha='center', fontsize=8, color='#555')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--r3-run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args()
    use_cjk_font()
    args.figures.mkdir(parents=True, exist_ok=True)
    summary = {k: {} for k in ('factors', 'profiles', 'cross', 'cross_rowcol', 'locate', 'probes', 'values',
                               'edits', 'refcoco', 'race', 'rowmatch')}
    data = {}
    for tag in ('vl', 'q35'):
        d = args.run / f'factors_{tag}'
        if d.exists():
            data[tag] = load_dir(d)
            summary['factors'][tag] = factor_summary(data[tag], tag)
            summary['profiles'][tag] = column_profiles(data[tag], tag)
            summary['cross'][tag] = cross_matrix(data[tag], tag, 'point')
            summary['cross_rowcol'][tag] = cross_matrix(data[tag], tag, 'rowcol')
            summary['values'][tag] = value_probes(data[tag], tag)
            summary['race'][tag] = score_race(data[tag], tag)
            summary['rowmatch'][tag] = row_match(data[tag], tag)
        if (args.run / f'locate_{tag}').exists():
            summary['locate'][tag] = locate_summary(args.run, tag)
            # The stage-wise states are scratch files deleted after the first analysis; their probe results
            # are kept in probes_<tag>.json and reused when the scratch files are gone.
            cache = args.run / f'probes_{tag}.json'
            pr = stage_probes(args.run, tag)
            if pr:
                cache.write_text(json.dumps(pr))
            elif cache.exists():
                pr = json.loads(cache.read_text())
            summary['probes'][tag] = pr
        summary['edits'][tag] = edit_summary(args.run, tag)
        summary['refcoco'][tag] = refcoco_edge(args.r3_run, tag)
    (args.run / 'summary.json').write_text(json.dumps(summary, indent=1))
    if data:
        fig_examples(args.run, data, args.figures / 'examples.png')
        fig_factor_bars(summary['factors'], args.figures / 'factor_bars.png')
        fig_profiles(summary['profiles'], args.figures / 'column_profiles.png')
        fig_cross(summary['cross'], args.figures / 'cross_keys_queries.png')
        fig_race(summary['race'], args.figures / 'score_race.png')
    fig_flow(args.figures / 'flow.png')
    if summary['locate']:
        fig_locate(summary['locate'], args.figures / 'locate.png')
        fig_probes(summary['probes'], args.figures / 'stage_probes.png')
    if any(summary['edits'].values()):
        fig_edits(summary['edits'], args.figures / 'value_edits.png')
    print('written', args.run / 'summary.json')


if __name__ == '__main__':
    main()
