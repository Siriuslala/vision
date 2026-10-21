"""CPU analysis of experiment 2.2 (high-norm tokens): reads <run>/<tag>/ stage outputs, writes <run>/<tag>/analysis/.

Outputs results.json with: per-block emergence statistics, high-norm frequency maps, table-coordinate positions,
content/background statistics, massive channels, blue-box effects, decoder norms, source-intervention effects,
attention enrichment, probe readouts, R^2_pos, the 2.1 distance readout and task effects of the ablations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from analysis_tools.position.coordinate_readout_run import read_jsonl

TAU = 3.0
PROBE_SITES = (6, 12, 20, 26)


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def auroc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    r = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (base right, cond wrong) and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


# ----------------------------------------------------------------------------- survey

def table_coords(r: int, c: int, hp: int, wp: int, side: int = 48) -> tuple[float, float]:
    return (r * (side - 1) / max(hp - 1, 1), c * (side - 1) / max(wp - 1, 1))


def survey_stats(sv: dict) -> dict:
    out = {}
    for name, recs in sv.items():
        nb = recs[0]['ratio'].shape[0]
        maxr = np.stack([r['ratio'].reshape(nb, -1).max(1) for r in recs]).astype(float)        # (N, nb)
        n3 = np.stack([(r['ratio'].reshape(nb, -1) > 3).sum(1) for r in recs]).astype(float)
        n5 = np.stack([(r['ratio'].reshape(nb, -1) > 5).sum(1) for r in recs]).astype(float)
        frac = np.stack([(r['ratio'].reshape(nb, -1) > 3).mean(1) for r in recs]).astype(float)
        first = []
        for r in recs:
            hit = np.nonzero((r['ratio'].reshape(nb, -1) > 5).any(1))[0]
            first.append(int(hit[0]) + 1 if len(hit) else None)
        merger_max = [float(r['merger_ratio'].max()) for r in recs]
        st = {'n_images': len(recs), 'max_ratio_median': np.median(maxr, 0).tolist(),
              'max_ratio_q10': np.quantile(maxr, .1, 0).tolist(), 'max_ratio_q90': np.quantile(maxr, .9, 0).tolist(),
              'n_hn3_mean': n3.mean(0).tolist(), 'n_hn5_mean': n5.mean(0).tolist(), 'frac_hn3_mean': frac.mean(0).tolist(),
              'first_block_5x': first, 'first_block_5x_counts': {str(k): first.count(k) for k in sorted(set(first), key=str)},
              'merger_max_ratio_median': float(np.median(merger_max)),
              'merger_max_ratio_q90': float(np.quantile(merger_max, .9))}
        dec = [r['decoder'] for r in recs if 'decoder' in r]
        if dec:
            vmax = np.stack([d['visual'].reshape(d['visual'].shape[0], -1).max(1) /
                             np.median(d['visual'].reshape(d['visual'].shape[0], -1), 1) for d in dec])
            argmax0 = np.stack([d['visual'].reshape(d['visual'].shape[0], -1).argmax(1) == 0 for d in dec])
            vmed = np.stack([np.median(d['visual'].reshape(d['visual'].shape[0], -1), 1) for d in dec])
            pref = np.stack([d['prefix'] for d in dec])                                         # (N, L, 4)
            st['decoder'] = {'n_images': len(dec), 'visual_max_over_median_median': np.median(vmax, 0).tolist(),
                             'visual_max_over_median_max': vmax.max(0).tolist(),
                             'visual_argmax_is_first_frac': argmax0.mean(0).tolist(),
                             'visual_median_norm': np.median(vmed, 0).tolist(),
                             'prefix_norm_median': np.median(pref, 0).tolist(), 'prefix_tokens': dec[0]['prefix_tokens'],
                             'prefix_over_visual_median': (np.median(pref, 0) / np.median(vmed, 0)[:, None]).tolist()}
        out[name] = st
    return out


def frequency_maps(sv: dict, block: int = 12) -> dict:
    """f_p per (set, grid) at one block (tau = 3), raster order."""
    out = {}
    for name, recs in sv.items():
        groups = defaultdict(list)
        for r in recs:
            if name == 'marked' and not r['id'].endswith('|plain'):
                continue
            groups[tuple(r['grid'])].append(r['ratio'][block - 1] > TAU)
        for g, ms in groups.items():
            f = np.mean(ms, 0)
            tot = f.sum()
            top = np.sort(f.flatten())[::-1]
            out[f'{name}|{g[0]}x{g[1]}'] = {'n': len(ms), 'f': f.tolist(), 'max_f': float(top[0]),
                                            'n_positions_f_ge_half': int((f >= .5).sum()),
                                            'top10_share': float(top[:10].sum() / tot) if tot > 0 else None,
                                            'mean_hn_per_image': float(tot)}
    return out


def blank_table_positions(sv: dict, block: int = 12) -> list[dict]:
    rows = []
    for r in sv['blank']:
        hp, wp = r['grid']
        m = r['ratio'][block - 1].astype(float)
        for i, j in zip(*np.nonzero(m > TAU)):
            tr, tc = table_coords(int(i), int(j), hp, wp)
            rows.append({'image': r['id'], 'grid': [hp, wp], 'row': int(i), 'col': int(j), 'table_row': tr, 'table_col': tc,
                         'ratio': float(m[i, j])})
    return rows


def content_background(sv: dict, block: int = 12) -> dict:
    """Share of high-norm patches lying on blank patches (gray std < 0.02) vs the share of blank patches."""
    out = {}
    for name in ('probe', 'natural', 'marked', 'grid'):
        if name not in sv:
            continue
        hn_blank = hn_total = all_blank = all_total = 0
        for r in sv[name]:
            if 'probe' not in r:
                continue
            p = r['probe']
            std = p['content'][:, 17].astype(float)
            is_hn = p[f'is_hn_{block}']
            rand = p['kind'] == 2
            hn_total += int(is_hn.sum())
            hn_blank += int((is_hn & (std < .02)).sum())
            all_total += int(rand.sum())
            all_blank += int((rand & (std < .02)).sum())
        if hn_total:
            out[name] = {'hn_blank': hn_blank, 'hn_total': hn_total, 'hn_blank_frac': hn_blank / hn_total,
                         'hn_blank_ci': wilson(hn_blank, hn_total), 'random_blank': all_blank, 'random_total': all_total,
                         'random_blank_frac': all_blank / max(all_total, 1), 'random_blank_ci': wilson(all_blank, all_total)}
    return out


def massive_channels(sv: dict) -> dict:
    cnt = defaultdict(int)
    n = 0
    for name, recs in sv.items():
        for r in recs:
            if 'hn_top_channels' in r:
                for row in r['hn_top_channels']:
                    n += 1
                    for ch in row.tolist():
                        cnt[ch] += 1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:20]
    return {'n_tokens': n, 'top': [{'channel': int(k), 'freq': v / max(n, 1)} for k, v in top]}


def blue_box(sv: dict, decomposition: Path | None) -> dict:
    """Norm of the tokens on the blue box border, marked vs unmarked twin, per site."""
    recs = {r['id']: r for r in sv.get('marked', [])}
    rows = []
    for rid, r in recs.items():
        if not rid.endswith('|marked'):
            continue
        twin = recs.get(rid.replace('|marked', '|plain'))
        if twin is None:
            raise ValueError(f'Missing unmarked twin of {rid}')
        lab = r['labels']
        mr, mc = lab['marker']
        cell, (ox, oy) = lab['cell'], lab['grid_origin']
        x0, y0 = ox + mc * cell, oy + mr * cell
        hp, wp = r['grid']
        # patches whose 16x16 area intersects the 64x64 cell border band (box drawn on the cell boundary)
        band = np.zeros((hp, wp), bool)
        for i in range(hp):
            for j in range(wp):
                px0, py0 = j * 16, i * 16
                inter = (px0 < x0 + cell + 4) and (px0 + 16 > x0 - 4) and (py0 < y0 + cell + 4) and (py0 + 16 > y0 - 4)
                inner = (px0 >= x0 + 4) and (px0 + 16 <= x0 + cell - 4) and (py0 >= y0 + 4) and (py0 + 16 <= y0 + cell - 4)
                band[i, j] = inter and not inner
        nb = r['ratio'].shape[0]
        # ratios are normalized by each image's own median; convert to absolute norms
        absm = r['ratio'].astype(float) * r['median'][:, None, None]
        absp = twin['ratio'].astype(float) * twin['median'][:, None, None]
        row = {'id': rid, 'rows': lab['rows'], 'vit_band_ratio': (absm[:, band].mean(1) / absp[:, band].mean(1)).tolist(),
               'vit_hn_marked': (r['ratio'].reshape(nb, -1) > TAU).sum(1).tolist(),
               'vit_hn_plain': (twin['ratio'].reshape(nb, -1) > TAU).sum(1).tolist(),
               'vit_hn_on_band_marked': int((r['ratio'][11] > TAU)[band].sum()),
               'vit_hn_on_band_plain': int((twin['ratio'][11] > TAU)[band].sum())}
        mband = band.reshape(hp // 2, 2, wp // 2, 2).any(3).any(1)
        row['merger_band_ratio'] = float(r['merger_ratio'].astype(float)[mband].mean() /
                                         twin['merger_ratio'].astype(float)[mband].mean())
        if 'decoder' in r and 'decoder' in twin:
            dm, dp = r['decoder']['visual'], twin['decoder']['visual']
            row['decoder_band_ratio'] = (dm[:, mband].mean(1) / dp[:, mband].mean(1)).tolist()
            row['decoder_band_over_median_marked'] = (dm[:, mband].mean(1) / np.median(dm.reshape(len(dm), -1), 1)).tolist()
        rows.append(row)
    if not rows:
        return {}
    out = {'n_pairs': len(rows), 'vit_band_ratio_mean': np.mean([r['vit_band_ratio'] for r in rows], 0).tolist(),
           'merger_band_ratio_mean': float(np.mean([r['merger_band_ratio'] for r in rows])),
           'merger_band_ratio_ci': boot_ci([r['merger_band_ratio'] for r in rows]),
           'hn_on_band_marked': int(sum(r['vit_hn_on_band_marked'] for r in rows)),
           'hn_on_band_plain': int(sum(r['vit_hn_on_band_plain'] for r in rows)),
           'hn12_marked_mean': float(np.mean([r['vit_hn_marked'][11] for r in rows])),
           'hn12_plain_mean': float(np.mean([r['vit_hn_plain'][11] for r in rows]))}
    dec = [r for r in rows if 'decoder_band_ratio' in r]
    if dec:
        out['decoder_band_ratio_mean'] = np.mean([r['decoder_band_ratio'] for r in dec], 0).tolist()
        out['decoder_band_over_median_marked_mean'] = np.mean([r['decoder_band_over_median_marked'] for r in dec], 0).tolist()
    out['rows'] = rows
    return out


# ----------------------------------------------------------------------------- probes

class RidgeSVD:
    """Ridge regression for many lambdas and targets from one thin SVD of the standardized training inputs."""

    LAMBDAS = (1e-1, 1, 10, 100, 1e3, 1e4)

    def __init__(self, X: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.U, self.s, self.Vt = np.linalg.svd(Z, full_matrices=False)

    def predict(self, Xn: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
        ym = Y.mean(0)
        f = self.s / (self.s ** 2 + lam)
        W = self.Vt.T @ (f[:, None] * (self.U.T @ (Y - ym)))
        return ((Xn - self.mu) / self.sd) @ W + ym


def _r2(P, Y) -> float:
    den = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ((P - Y) ** 2).sum() / den) if den > 0 else float('nan')


def _bacc(p, y) -> float:
    return float(np.mean([np.mean(p[y == k] == k) for k in np.unique(y)]))


def probe_suite(Xtr, Xte, reg: dict, cls: dict) -> dict:
    """Evaluate ridge regressions (test R^2) and ridge classifiers (one-hot targets, argmax; balanced accuracy).

    Lambda is chosen on a 20% validation split of the training tokens, then refitted on all training tokens.
    reg: name -> (Ytr, Yte); cls: name -> (ytr, yte, n_class).
    """
    n = len(Xtr)
    perm = np.random.default_rng(0).permutation(n)
    nv = max(1, n // 5)
    va, tr = perm[:nv], perm[nv:]
    sub, full = RidgeSVD(Xtr[tr]), RidgeSVD(Xtr)
    out = {}
    for name, (Ytr, Yte) in reg.items():
        best = max(RidgeSVD.LAMBDAS, key=lambda l: np.nan_to_num(_r2(sub.predict(Xtr[va], Ytr[tr], l), Ytr[va]), nan=-1e9))
        out[name] = {'r2': _r2(full.predict(Xte, Ytr, best), Yte), 'lambda': best, 'n_train': int(n), 'n_test': int(len(Xte)),
                     'target_var': float(Yte.var(0).sum())}
    for name, (ytr, yte, k) in cls.items():
        if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
            continue
        Y = np.eye(k)[ytr]
        best = max(RidgeSVD.LAMBDAS, key=lambda l: _bacc(sub.predict(Xtr[va], Y[tr], l).argmax(1), ytr[va]))
        p = full.predict(Xte, Y, best).argmax(1)
        out[name] = {'balanced_acc': _bacc(p, yte), 'acc': float(np.mean(p == yte)), 'lambda': best, 'n_train': int(n),
                     'n_test': int(len(Xte)), 'chance': 1 / len(np.unique(yte))}
    return out


def ridge_fit_eval(Xtr, Ytr, Xte, Yte) -> dict:
    return probe_suite(Xtr, Xte, {'y': (Ytr, Yte)}, {})['y']


def probe_tables(sv: dict) -> dict:
    """Probe readouts of high-norm, position-matched (canonical, not high-norm) and random tokens."""
    recs = [r for r in sv.get('probe', []) if 'probe' in r]
    if not recs:
        return {}
    canvases = sorted({r['labels']['canvas'] for r in recs})
    templates = sorted({r['labels']['template'] for r in recs})
    rng = np.random.default_rng(0)
    img_ids = np.arange(len(recs))
    test_img = set(rng.permutation(len(recs))[:len(recs) // 4].tolist())
    out = {}
    for site in [str(s) for s in PROBE_SITES] + ['merger']:
        hn_key = f'is_hn_{site}' if site != 'merger' else 'is_hn_12'
        groups = {'hn': [], 'canon': [], 'random': []}
        for i, r in enumerate(recs):
            p = r['probe']
            f = p[f'feat_{site}'].astype(np.float32)
            is_hn = p[hn_key]
            hp, wp = r['grid']
            lab = r['labels']
            n_hn = int((is_hn & (p['kind'] == 0)).sum()) if site != 'merger' else int((p['kind'] == 0).sum())
            sel = {'hn': np.nonzero((p['kind'] == 0) & is_hn)[0] if site != 'merger' else np.nonzero(p['kind'] == 0)[0],
                   'canon': np.nonzero((p['kind'] == 1) & ~is_hn)[0],
                   'random': np.nonzero((p['kind'] == 2) & ~is_hn)[0][:max(n_hn, 1)]}
            for g, idx in sel.items():
                for k in idx:
                    groups[g].append({'x': f[k], 'img': i, 'pos': [p['row'][k] / (hp - 1), p['col'][k] / (wp - 1)],
                                      'content': p['content'][k].astype(np.float32),
                                      'template': templates.index(lab['template']), 'canvas': canvases.index(lab['canvas']),
                                      'anomaly': int(lab['anomaly']),
                                      'count': 0 if lab['count'] <= 5 else (1 if lab['count'] <= 10 else 2)})
        site_out = {}
        for g, rows in groups.items():
            if len(rows) < 40:
                continue
            X = np.stack([r['x'] for r in rows])
            te = np.array([r['img'] in test_img for r in rows])
            res = {'n_tokens': len(rows), 'n_images': len({r['img'] for r in rows}),
                   'mean_norm': float(np.linalg.norm(X, axis=1).mean())}
            for norm in ('raw', 'l2'):
                Xn = X / np.linalg.norm(X, axis=1, keepdims=True) if norm == 'l2' else X
                C = np.stack([r['content'] for r in rows])[:, :16]
                P = np.array([r['pos'] for r in rows])
                cls = {}
                for attr, k in (('template', len(templates)), ('canvas', len(canvases)), ('anomaly', 2), ('count', 3)):
                    y = np.array([r[attr] for r in rows])
                    cls[attr] = (y[~te], y[te], k)
                d = probe_suite(Xn[~te], Xn[te], {'content_r2': (C[~te], C[te]), 'position_r2': (P[~te], P[te])}, cls)
                res[norm] = d
            site_out[g] = res
        # image-level global probes on the pooled mean of non-high-norm tokens
        X = np.stack([r['probe'][f'pool_{site}'].astype(np.float32) for r in recs])
        te = np.array([i in test_img for i in img_ids])
        lab = [r['labels'] for r in recs]
        cls = {}
        for attr, k in (('template', len(templates)), ('canvas', len(canvases)), ('anomaly', 2), ('count', 3)):
            y = np.array([templates.index(l['template']) if attr == 'template' else canvases.index(l['canvas'])
                          if attr == 'canvas' else int(l['anomaly']) if attr == 'anomaly' else
                          (0 if l['count'] <= 5 else 1 if l['count'] <= 10 else 2) for l in lab])
            cls[attr] = (y[~te], y[te], k)
        d = probe_suite(X[~te], X[te], {}, cls)
        site_out['pooled'] = {'raw': d, 'n_images': len(recs)}
        out[site] = site_out
        print('probes site', site, {g: v.get('n_tokens') for g, v in site_out.items()}, flush=True)
    return out


def natural_content_probes(sv: dict) -> dict:
    """Own-patch content and position readouts of high-norm vs random tokens on natural images.

    Synthetic high-norm tokens lie only on blank patches (constant content), so the content comparison needs
    natural images, where most high-norm patches carry texture. Only natural images of the modal grid are used.
    """
    recs = [r for r in sv.get('natural', []) if 'probe' in r]
    if not recs:
        return {}
    grids = [tuple(r['grid']) for r in recs]
    rng = np.random.default_rng(1)
    test_img = set(rng.permutation(len(recs))[:len(recs) // 4].tolist())
    out = {}
    for site in [str(s) for s in PROBE_SITES] + ['merger']:
        hn_key = f'is_hn_{site}' if site != 'merger' else 'is_hn_12'
        rows = {'hn': [], 'random': [], 'matched': []}
        for i, r in enumerate(recs):
            p = r['probe']
            hp, wp = r['grid']
            f = p[f'feat_{site}'].astype(np.float32)
            is_hn = p[hn_key]
            hn_idx = np.nonzero((p['kind'] == 0) & is_hn)[0] if site != 'merger' else np.nonzero(p['kind'] == 0)[0]
            pool = np.nonzero((p['kind'] == 2) & ~is_hn)[0]
            rnd = pool[:max(len(hn_idx), 1)]
            # texture-matched control: for each high-norm patch, the unused random patch with the closest gray std
            std = p['content'][:, 17].astype(float)
            matched, free = [], list(pool)
            for k in hn_idx:
                if not free:
                    break
                j = min(free, key=lambda q: abs(std[q] - std[k]))
                matched.append(j)
                free.remove(j)
            for g, idx in (('hn', hn_idx), ('random', rnd), ('matched', np.array(matched, int))):
                for k in idx:
                    rows[g].append((f[k], i, p['content'][k].astype(np.float32), [p['row'][k] / (hp - 1), p['col'][k] / (wp - 1)]))
        so = {}
        for g, rr in rows.items():
            if len(rr) < 40:
                continue
            X = np.stack([a for a, _, _, _ in rr])
            te = np.array([b in test_img for _, b, _, _ in rr])
            C = np.stack([c for _, _, c, _ in rr])
            P = np.array([q for _, _, _, q in rr])
            d = probe_suite(X[~te], X[te], {'content_r2': (C[~te, :16], C[te, :16]), 'position_r2': (P[~te], P[te])}, {})
            d['n_tokens'] = len(rr)
            d['content_std_mean'] = float(C[:, 17].mean())
            so[g] = d
        out[site] = so
    return out


# ----------------------------------------------------------------------------- source / attention / posvar

def source_stats(src: dict, pc: dict | None) -> dict:
    base = {r['id']: r for r in src['base']}
    out = {}
    for cname, rows in src.items():
        jac, nh, nb_, moved = [], [], [], []
        for r in rows:
            b = base[r['id']]
            m = r['ratio'][2] > TAU          # block 12
            mb = b['ratio'][2] > TAU
            # flips of the PE grid are compared with the correspondingly flipped base pattern
            if cname == 'pe_hflip':
                mb = mb[:, ::-1]
            elif cname == 'pe_vflip':
                mb = mb[::-1]
            inter, union = (m & mb).sum(), (m | mb).sum()
            jac.append(float(inter / union) if union else 1.0)
            nh.append(int(m.sum()))
            nb_.append(int(mb.sum()))
        kinds = defaultdict(list)
        for r, j, n in zip(rows, jac, nh):
            kind = 'blank' if r['id'].startswith(('white', 'gray')) else 'noise' if r['id'].startswith('noise') else \
                'synthetic' if r['id'].startswith('p') else 'natural'
            kinds[kind].append((j, n))
        out[cname] = {k: {'jaccard_mean': float(np.mean([a for a, _ in v])), 'n_hn_mean': float(np.mean([b for _, b in v])),
                          'n_images': len(v)} for k, v in kinds.items()}
        out[cname]['n_hn_by_block_mean'] = np.mean([r['n_hn'] for r in rows], 0).tolist()
    res = {'conditions': out}
    if pc is not None:
        full = pc['full640'] > TAU
        win = {}
        for k, m in pc['window'].items():
            r0, c0 = map(int, k.split('_'))
            m = m > TAU
            crop = full[r0:r0 + 32, c0:c0 + 32]
            inter, union = (m & crop).sum(), (m | crop).sum()
            win[k] = {'jaccard_vs_crop_of_640': float(inter / union) if union else 1.0, 'n_hn': int(m.sum()),
                      'n_crop': int(crop.sum())}
        res['window'] = win
        canon = set(pc['canonical_raster'])
        cont = []
        for c in pc['content']:
            m = c['ratio'] > TAU
            x0, y0, x1, y1 = c['box']
            ras = set(np.nonzero(m.flatten())[0].tolist())
            inside = [q for q in ras if x0 <= (q % 32) * 16 < x1 and y0 <= (q // 32) * 16 < y1]
            cont.append({'target': c['target_raster'], 'is_canonical': c['is_canonical'], 'target_still_hn': c['target_raster'] in ras,
                         'n_hn': len(ras), 'n_hn_inside_tile': len(inside), 'canonical_kept': len(ras & canon),
                         'new_positions': sorted(ras - canon)})
        res['content'] = cont
    return res


def attention_stats(vit: list, decoder: list | None) -> dict:
    out = {'vit': {}}
    blocks = vit[0]['blocks']
    for k in blocks:
        e_hn, e_rand, m_hn, m_rand = [], [], [], []
        for r in vit:
            P = len(r['hn_block'])
            n = max(r['n_hn'], 1)
            if r['n_hn'] == 0:
                continue
            mh, mr = r['res'][k]['mass_hn'], r['res'][k]['mass_rand']
            e_hn.append(mh.mean() / (n / P))
            e_rand.append(mr.mean() / (n / P))
            m_hn.append(mh.mean())
            m_rand.append(mr.mean())
        out['vit'][k] = {'enrich_hn_mean': float(np.mean(e_hn)), 'enrich_rand_mean': float(np.mean(e_rand)),
                         'mass_hn_mean': float(np.mean(m_hn)), 'mass_hn_ci': boot_ci(m_hn),
                         'mass_rand_mean': float(np.mean(m_rand)), 'n_images': len(e_hn),
                         'head_max_mass_hn': float(np.max([r['res'][k]['mass_hn'].max() for r in vit if r['n_hn']]))}
    if decoder:
        d = {}
        for layer in decoder[0]['mass']:
            acc = defaultdict(list)
            for r in decoder:
                m = r['mass'][layer]
                for qset, sl in (('last', slice(0, 1)), ('visual', slice(1, None))):
                    acc[f'{qset}_first'].append(m['first'][:, sl].mean())
                    acc[f'{qset}_prefix'].append(m['prefix'][:, sl].mean())
                    acc[f'{qset}_visual_all'].append(m['visual_all'][:, sl].mean())
                    vh = m['visual_hn'][:, sl].mean()
                    acc[f'{qset}_visual_hn'].append(vh)
                    acc[f'{qset}_visual_hn_enrich'].append((vh / max(m['visual_all'][:, sl].mean(), 1e-9)) /
                                                           (m['n_hn'] / m['n_visual']))
                    acc[f'{qset}_visual_00_enrich'].append((m['visual_00'][:, sl].mean() /
                                                            max(m['visual_all'][:, sl].mean(), 1e-9)) * m['n_visual'])
            d[int(layer)] = {k: float(np.mean(v)) for k, v in acc.items()}
        out['decoder'] = d
        out['decoder_check_max'] = float(max(max(r['check'].values()) for r in decoder))
    return out


def readout_stats(readout: dict) -> dict:
    out = {}
    for cname, rows in readout.items():
        d = {}
        for site in ('vit_18', 'merger', 'dec_16'):
            for gap in (4, 8):
                rs = [r for r in rows if r['gap'] == gap and r['target'] is not None]
                hit = sum(int(np.argmax(r[site]) == r['target']) for r in rs)
                tops = [int(np.argmax(r[site])) for r in rs]
                common = max(set(tops), key=tops.count)
                d[f'{site}|gap{gap}'] = {'hit': hit, 'n': len(rs), 'ci': wilson(hit, len(rs)), 'modal_top': common,
                                         'modal_top_frac': tops.count(common) / len(rs)}
            normal = [r for r in rows if r['target'] is None and r['gap'] == 8]
            if normal:
                d[f'{site}|normal_top'] = int(np.argmax(normal[0][site]))
        out[cname] = d
    return out


# ----------------------------------------------------------------------------- ablation

def ablation_stats(root: Path) -> dict:
    cases = {c['id']: c for c in read_jsonl(root / 'cases.jsonl')}
    recs = {}
    for p in sorted(root.glob('records_*.jsonl')):
        recs[p.stem.replace('records_', '')] = {r['id']: r for r in read_jsonl(p)}
    if 'base' not in recs:
        return {}
    base = recs['base']
    tasks = sorted({r['task_tag'] for r in base.values()})
    out = {}
    for cname, rr in recs.items():
        d = {}
        for t in tasks:
            ids = [i for i, r in base.items() if r['task_tag'] == t and i in rr]
            if not ids:
                continue
            corr = np.array([bool(rr[i]['score'].get('correct')) for i in ids])
            corr0 = np.array([bool(base[i]['score'].get('correct')) for i in ids])
            b, c = int((corr0 & ~corr).sum()), int((~corr0 & corr).sum())
            e = {'n': len(ids), 'correct': int(corr.sum()), 'acc': float(corr.mean()), 'ci': wilson(int(corr.sum()), len(ids)),
                 'delta': float(corr.mean() - corr0.mean()), 'base_only': b, 'cond_only': c, 'p_mcnemar': mcnemar_exact(b, c),
                 'n_ablated_mean': float(np.mean([rr[i].get('n_ablated', 0) for i in ids]))}
            if 's' in rr[ids[0]]:
                y = [cases[i]['truth']['answer_yes'] for i in ids]
                e['auroc'] = auroc([rr[i]['s'] for i in ids], y)
                e['auroc_base'] = auroc([base[i]['s'] for i in ids], y)
                ds = np.array([rr[i]['s'] - base[i]['s'] for i in ids])
                e['mean_abs_delta_s'] = float(np.abs(ds).mean())
                e['mean_delta_s'] = float(ds.mean())
                # bootstrap AUROC difference
                rng = np.random.default_rng(0)
                diffs = []
                yy = np.array(y)
                s1 = np.array([rr[i]['s'] for i in ids])
                s0 = np.array([base[i]['s'] for i in ids])
                for _ in range(500):
                    k = rng.integers(0, len(ids), len(ids))
                    diffs.append(auroc(s1[k], yy[k]) - auroc(s0[k], yy[k]))
                e['auroc_delta_ci'] = [float(np.nanquantile(diffs, .025)), float(np.nanquantile(diffs, .975))]
            errs = [(rr[i]['score'].get('error'), base[i]['score'].get('error')) for i in ids]
            errs = [(a, b_) for a, b_ in errs if a is not None and b_ is not None]
            if errs:
                de = np.array([a - b_ for a, b_ in errs])
                e['error_mean'] = float(np.mean([a for a, _ in errs]))
                e['error_delta_mean'] = float(de.mean())
                e['error_delta_ci'] = boot_ci(de)
                e['n_error_pairs'] = len(errs)
            ious = [(rr[i]['score'].get('iou'), base[i]['score'].get('iou')) for i in ids]
            ious = [(a, b_) for a, b_ in ious if a is not None and b_ is not None]
            if ious:
                e['iou_delta_mean'] = float(np.mean([a - b_ for a, b_ in ious]))
            same = [rr[i]['response'].strip() == base[i]['response'].strip() for i in ids]
            e['identical_response_frac'] = float(np.mean(same))
            d[t] = e
        out[cname] = d
    spot = read_jsonl(root / 'spotcheck.jsonl')
    if spot:
        out['_spotcheck'] = {'n': len(spot), 'text_equal': sum(r['text_equal'] for r in spot),
                             'correct_equal': sum(r['correct_equal'] for r in spot)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--tags', default='q35,q3vl')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--reuse', default='', help='comma list of result keys copied from an existing results.json '
                                                '(e.g. probes,probes_natural) instead of recomputing')
    args = p.parse_args()
    torch.set_num_threads(args.threads)       # host CPUs are shared; small problems are slower with many threads
    for tag in args.tags.split(','):
        root = args.run / tag
        if not root.exists():
            continue
        out = root / 'analysis'
        out.mkdir(exist_ok=True)
        res = {}
        old = json.loads((out / 'results.json').read_text()) if (out / 'results.json').exists() else {}
        reuse = {k: old[k] for k in args.reuse.split(',') if k and k in old}
        nj = root / 'neurons' / 'neurons.json'
        if nj.exists():
            n = json.loads(nj.read_text())
            res['neurons'] = {k: n[k] for k in ('register_neurons', 'n_hn_tokens', 'block', 'top_neurons', 'register_contrib_norm',
                                                'fc2_out_channels')}
            imgs = n['images']
            res['neurons']['decomposition'] = {
                part: {'hn': float(np.mean([r['hn'][part] for r in imgs if r['hn'][part] is not None])),
                       'normal': float(np.mean([r['normal'][part] for r in imgs]))} for part in ('in', 'attn', 'mlp', 'out')}
        sdir = root / 'survey'
        if sdir.exists():
            sv = {p_.stem: torch.load(p_, weights_only=False) for p_ in sorted(sdir.glob('*.pt'))}
            res['survey'] = survey_stats(sv)
            res['freq_maps'] = frequency_maps(sv)
            res['blank_table_positions'] = blank_table_positions(sv)
            res['content_background'] = content_background(sv)
            res['massive_channels'] = massive_channels(sv)
            res['blue_box'] = blue_box(sv, None)
            res['probes'] = reuse['probes'] if 'probes' in reuse else probe_tables(sv)
            res['probes_natural'] = reuse['probes_natural'] if 'probes_natural' in reuse else natural_content_probes(sv)
            print(tag, 'survey analysed', flush=True)
        src = root / 'source' / 'records.pt'
        if src.exists():
            pc = root / 'source' / 'phase_content.pt'
            res['source'] = source_stats(torch.load(src, weights_only=False),
                                         torch.load(pc, weights_only=False) if pc.exists() else None)
        at = root / 'attention' / 'vit.pt'
        if at.exists():
            dpath = root / 'attention' / 'decoder.pt'
            res['attention'] = attention_stats(torch.load(at, weights_only=False),
                                               torch.load(dpath, weights_only=False) if dpath.exists() else None)
        pv = root / 'posvar' / 'r2.json'
        if pv.exists():
            res['posvar'] = json.loads(pv.read_text())
        ro = root / 'posvar' / 'readout.json'
        if ro.exists():
            res['readout'] = readout_stats(json.loads(ro.read_text()))
        ab = root / 'ablation'
        if (ab / 'cases.jsonl').exists():
            res['ablation'] = ablation_stats(ab)
        (out / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
        print(tag, 'written', out / 'results.json', flush=True)


if __name__ == '__main__':
    main()
