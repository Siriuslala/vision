"""CPU analysis of experiment 3.11 round 3 (edge-reading heads); writes summary.json and arrays.npz.

Definitions (all from the attention rows recorded by edge_heads.py stage 'gen'):
  step      one emitted coordinate value: pt_x, pt_y (point answer), bb_x1, bb_y1, bb_x2, bb_y2 (box answer
            x_min, y_min, x_max, y_max). The query is the position that predicts the value's first digit.
  attention the head's softmax row restricted to the image tokens and renormalized to sum to 1.
  peak      attention-weighted centre (pixels) of the 3x3 token neighbourhood of the maximum token.
  (u, v)    box-normalized coordinates: u = (x - x_min) / (x_max - x_min), v = (y - y_min) / (y_max - y_min);
            (0, 0) is the top-left corner of the target box, (1, 1) the bottom-right corner.
  edge regression  per head, step and peak coordinate, least squares over images:
            peak_x = b_min * x_min + b_max * x_max + c,   peak_y = b_min * y_min + b_max * y_max + c.
            A centre reader has b_min = b_max = 0.5, a right-edge reader b_max = 1, b_min = 0.
  band      the peak is in the left (right) edge band when u is in [-0.15, 0.2] ([0.8, 1.15]); top/bottom likewise.
  extreme point  the object's rightmost (leftmost, bottom, top) mask pixels: mean v of the mask pixels within
            5% of the box width from the right edge, etc.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from analysis_tools.position import coordinate_readout as cr
from analysis_tools.representation import sees_right as sr

FULL_LAYERS = cr.FULL_LAYERS
LI = {l: i for i, l in enumerate(FULL_LAYERS)}
TP = cr.TOKEN_PIXELS
STEPS = [('point', 0, 'pt_x', 0), ('point', 1, 'pt_y', 1), ('bbox', 0, 'bb_x1', 0), ('bbox', 1, 'bb_y1', 1),
         ('bbox', 2, 'bb_x2', 0), ('bbox', 3, 'bb_y2', 1)]      # (format, value index, name, axis 0=x 1=y)
STEP_NAMES = [s[2] for s in STEPS]
EDGE_MIN = ['20.0', '24.1', '24.2', '24.7', '24.12']             # 3.11 round 2 left/top group
EDGE_MAX = ['20.10', '24.5', '24.6', '24.8', '24.9', '28.0']     # 3.11 round 2 right/bottom group
KEY_HEADS = ['24.7', '24.2', '20.0', '24.12', '24.1', '24.5', '24.6', '24.9', '24.8', '20.10', '28.0', '20.11',
             '24.11', '20.2']
BAND_LO, BAND_HI, BAND_OUT = 0.2, 0.8, 0.15
MIN_SPAN_PX = 3 * TP                  # box extent needed for edge-band statistics along an axis
DENSITY_RANGE, DENSITY_BINS = (-0.3, 1.3), 16
MIN_WILSON_N = 1


def hid(name: str) -> tuple[int, int]:
    l, h = name.split('.')
    return LI[int(l)], int(h)


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float('nan')] * 3
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - w, c + w]


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b and c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


# ----------------------------------------------------------------------------- loading

def token_centres(grid) -> np.ndarray:
    H, W = grid
    j = np.arange(H * W)
    return np.stack([(j % W + 0.5) * TP, (j // W + 0.5) * TP], 1)


def peaks(att: np.ndarray, grid) -> np.ndarray:
    """(..., Nv) attention -> (..., 2) peak (3x3 local attention-weighted centre, pixels)."""
    H, W = grid
    a = att.reshape(-1, H, W)
    j = a.reshape(len(a), -1).argmax(1)
    r, c = j // W, j % W
    out = np.zeros((len(a), 2))
    for n in range(len(a)):
        r0, r1, c0, c1 = max(0, r[n] - 1), min(H, r[n] + 2), max(0, c[n] - 1), min(W, c[n] + 2)
        w = a[n, r0:r1, c0:c1]
        yy, xx = np.mgrid[r0:r1, c0:c1]
        out[n] = [((xx + 0.5) * TP * w).sum() / w.sum(), ((yy + 0.5) * TP * w).sum() / w.sum()]
    return out.reshape(*att.shape[:-1], 2)


def token_coverage(binary: np.ndarray, grid) -> np.ndarray:
    """(Nv,) fraction of each merged token's 32x32 cell covered by a (H, W) bool map."""
    h, w = grid
    a = binary[:h * TP, :w * TP].astype(np.float32)
    return a.reshape(h, TP, w, TP).mean((1, 3)).ravel()


def load_items(run: Path) -> tuple[list[dict], dict]:
    """One item per case with valid point and/or box answers: attention per step (6, 8, 16, Nv) (NaN if absent)."""
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    gen = {g['id']: g for g in sr.read_jsonl(run / 'gen.jsonl')}
    items = []
    for cid, c in cases.items():
        recs = {fmt: gen.get(f'{cid}:{fmt}') for fmt in ('point', 'bbox')}
        if not any(r and r['score']['valid'] for r in recs.values()):
            continue
        grid = None
        att = None
        for k, (fmt, vi, _, _) in enumerate(STEPS):
            r = recs[fmt]
            if not (r and r['score']['valid']):
                continue
            a = np.load(run / 'attn' / f'{cid}_{fmt}.npz')['attn'][vi].astype(np.float32)   # (8, 16, Nv)
            grid = tuple(r['grid'])
            if att is None:
                att = np.full((6, *a.shape), np.nan, np.float32)
            att[k] = a / a.sum(-1, keepdims=True)
        items.append({'id': cid, 'case': c, 'grid': grid, 'att': att, 'box': np.array(c['truth']['box'], float),
                      'canvas': c['truth']['canvas'], 'family': c['family'], 'group': c['group'],
                      'category': c['category'], 'area_frac': c['area_frac'],
                      'ok': {fmt: bool(recs[fmt] and recs[fmt]['score']['valid']) for fmt in ('point', 'bbox')},
                      'correct': {fmt: bool(recs[fmt] and recs[fmt]['score']['correct']) for fmt in ('point', 'bbox')}})
    return items, cases


def mask_of(c: dict) -> np.ndarray | None:
    return np.asarray(Image.open(c['mask_path'])) > 0 if 'mask_path' in c else None


# ----------------------------------------------------------------------------- per-item features

def item_features(it: dict) -> dict:
    """Peaks (6, 8, 16, 2) in pixels and box units, in-box mass, mask extreme points (u/v)."""
    b = it['box']
    pk = peaks(np.nan_to_num(it['att']), it['grid'])
    pk[np.isnan(it['att'][..., 0])] = np.nan
    w, h = b[2] - b[0], b[3] - b[1]
    uv = np.stack([(pk[..., 0] - b[0]) / w, (pk[..., 1] - b[1]) / h], -1)
    cen = token_centres(it['grid'])
    inb = (cen[:, 0] >= b[0]) & (cen[:, 0] <= b[2]) & (cen[:, 1] >= b[1]) & (cen[:, 1] <= b[3])
    f = {'peak': pk, 'uv': uv, 'mass_box': np.nan_to_num(it['att'])[..., inb].sum(-1) if inb.any() else
         np.zeros(it['att'].shape[:-1]), 'w': w, 'h': h}
    m = mask_of(it['case'])
    if m is not None and m.any():
        ys, xs = np.nonzero(m)
        xs, ys = xs + 0.5, ys + 0.5
        tol_x, tol_y = max(0.05 * w, 2), max(0.05 * h, 2)
        ext = {'right': ((ys[xs >= xs.max() - tol_x]).mean() - b[1]) / h,
               'left': ((ys[xs <= xs.min() + tol_x]).mean() - b[1]) / h,
               'bottom': ((xs[ys >= ys.max() - tol_y]).mean() - b[0]) / w,
               'top': ((xs[ys <= ys.min() + tol_y]).mean() - b[0]) / w,
               'centroid_u': (xs.mean() - b[0]) / w, 'centroid_v': (ys.mean() - b[1]) / h}
        f['ext'] = ext
        cov = token_coverage(m, it['grid'])
        pj = np.nan_to_num(it['att'])
        f['mass_mask'] = (pj * cov).sum(-1)
    return f


# ----------------------------------------------------------------------------- edge regression and bands

def regression(items, feats, mask=None) -> dict:
    """beta (6 steps, 2 peak coords, 8, 16, 2) = (b_min, b_max) and R^2 (6, 2, 8, 16), with n per step."""
    beta = np.full((6, 2, 8, 16, 2), np.nan)
    r2 = np.full((6, 2, 8, 16), np.nan)
    ns = []
    for s in range(6):
        sel = [i for i, f in enumerate(feats) if not np.isnan(f['peak'][s, 0, 0, 0]) and (mask is None or mask[i])]
        ns.append(len(sel))
        if len(sel) < 10:
            continue
        for k in range(2):
            X = np.column_stack([[items[i]['box'][k] for i in sel], [items[i]['box'][k + 2] for i in sel], np.ones(len(sel))])
            Y = np.stack([feats[i]['peak'][s, :, :, k] for i in sel]).reshape(len(sel), -1)
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            res = Y - X @ coef
            r2[s, k] = (1 - res.var(0) / np.maximum(Y.var(0), 1e-9)).reshape(8, 16)
            beta[s, k] = coef[:2].T.reshape(8, 16, 2)
    return {'beta': beta, 'r2': r2, 'n': ns}


def bootstrap_beta(items, feats, heads: list[str], rng, n_boot: int = 200) -> dict:
    """95% bootstrap intervals (over images) of (b_min, b_max) for the listed heads, every step and peak coord."""
    out = {}
    for s in range(6):
        sel = [i for i, f in enumerate(feats) if not np.isnan(f['peak'][s, 0, 0, 0])]
        if len(sel) < 10:
            continue
        for k in range(2):
            X = np.column_stack([[items[i]['box'][k] for i in sel], [items[i]['box'][k + 2] for i in sel], np.ones(len(sel))])
            Y = np.stack([[feats[i]['peak'][s, hid(h)[0], hid(h)[1], k] for h in heads] for i in sel])
            bs = []
            for _ in range(n_boot):
                j = rng.integers(0, len(sel), len(sel))
                coef, *_ = np.linalg.lstsq(X[j], Y[j], rcond=None)
                bs.append(coef[:2])
            bs = np.array(bs)                                    # (n_boot, 2, n_heads)
            for hi, h in enumerate(heads):
                out.setdefault(h, {}).setdefault(STEP_NAMES[s], {})['xy'[k]] = {
                    'b_min_ci': np.percentile(bs[:, 0, hi], [2.5, 97.5]).round(3).tolist(),
                    'b_max_ci': np.percentile(bs[:, 1, hi], [2.5, 97.5]).round(3).tolist()}
    return out


def band_of(x: np.ndarray) -> np.ndarray:
    """0 = min-edge band, 1 = interior, 2 = max-edge band, 3 = outside the box margin; NaN-safe."""
    out = np.full(x.shape, 1)
    out[(x >= -BAND_OUT) & (x <= BAND_LO)] = 0
    out[(x >= BAND_HI) & (x <= 1 + BAND_OUT)] = 2
    out[(x < -BAND_OUT) | (x > 1 + BAND_OUT)] = 3
    return out


def band_stats(items, feats, heads: list[str], sel_fn=None) -> dict:
    """Per head and step: fractions of images whose peak lies in each u band and each v band, the joint
    (corner / edge-only) fractions, and the median and IQR of u and v. Only boxes spanning >= 3 tokens on both axes."""
    out = {}
    for h in heads:
        li, hh = hid(h)
        for s in range(6):
            uv = np.array([f['uv'][s, li, hh] for i, f in enumerate(feats)
                           if not np.isnan(f['uv'][s, 0, 0, 0]) and f['w'] >= MIN_SPAN_PX and f['h'] >= MIN_SPAN_PX
                           and (sel_fn is None or sel_fn(items[i]))])
            if len(uv) < 5:
                continue
            bu, bv = band_of(uv[:, 0]), band_of(uv[:, 1])
            d = {'n': int(len(uv)),
                 'u_band': [float((bu == k).mean()) for k in range(4)],
                 'v_band': [float((bv == k).mean()) for k in range(4)],
                 'corner_br': float(((bu == 2) & (bv == 2)).mean()), 'corner_tl': float(((bu == 0) & (bv == 0)).mean()),
                 'right_only': float(((bu == 2) & (bv == 1)).mean()), 'bottom_only': float(((bv == 2) & (bu == 1)).mean()),
                 'left_only': float(((bu == 0) & (bv == 1)).mean()), 'top_only': float(((bv == 0) & (bu == 1)).mean()),
                 'u_med': float(np.median(uv[:, 0])), 'v_med': float(np.median(uv[:, 1])),
                 'u_iqr': float(np.subtract(*np.percentile(uv[:, 0], [75, 25]))),
                 'v_iqr': float(np.subtract(*np.percentile(uv[:, 1], [75, 25])))}
            out.setdefault(h, {})[STEP_NAMES[s]] = d
    return out


def group_peak(it: dict, heads: list[str], s: int, near_box: bool = False) -> np.ndarray:
    """Peak of the mean attention of a head group at step s (pixels). near_box: only tokens whose centre lies
    within the target box enlarged by BAND_OUT box sizes on every side are considered (excludes row-start tokens)."""
    a = np.mean([it['att'][s, hid(h)[0], hid(h)[1]] for h in heads], 0)
    if near_box:
        b = it['box']
        cen = token_centres(it['grid'])
        u = (cen[:, 0] - b[0]) / (b[2] - b[0])
        v = (cen[:, 1] - b[1]) / (b[3] - b[1])
        keep = (u >= -BAND_OUT) & (u <= 1 + BAND_OUT) & (v >= -BAND_OUT) & (v <= 1 + BAND_OUT)
        if keep.any():
            a = np.where(keep, a, 0)
    return peaks(a[None], it['grid'])[0]


def row_start(items, heads: list[str]) -> dict:
    """Attention share on the first image column (row-start tokens) per head and step, for boxes whose left edge
    is at least 2 tokens from the image border (so the first column is not part of the object), and the
    fraction of such images whose peak token lies in the first column. row_v_median: the box-normalized row (v)
    of the first-column attention centroid, for images where that column holds >= 5% of the attention."""
    out = {}
    for h in heads:
        li, hh = hid(h)
        for s in range(6):
            m, pk, vrow = [], [], []
            for it in items:
                if np.isnan(it['att'][s, 0, 0, 0]) or it['box'][0] < 2 * TP:
                    continue
                H, W = it['grid']
                a = it['att'][s, li, hh].reshape(H, W)
                m.append(a[:, 0].sum())
                pk.append(int(a.argmax()) % W == 0)
                b = it['box']
                if a[:, 0].sum() >= 0.05 and b[3] - b[1] >= 3 * TP:      # row of the first-column attention, box units
                    rows = (np.arange(H) + 0.5) * TP
                    vrow.append(((a[:, 0] * rows).sum() / a[:, 0].sum() - b[1]) / (b[3] - b[1]))
            if m:
                out.setdefault(h, {})[STEP_NAMES[s]] = {
                    'n': len(m), 'share': float(np.mean(m)), 'peak_frac': float(np.mean(pk)),
                    'uniform': float(np.mean([1 / it['grid'][1] for it in items])), 'n_row': len(vrow),
                    'row_v_median': float(np.median(vrow)) if vrow else None,
                    'row_v_iqr': float(np.subtract(*np.percentile(vrow, [75, 25]))) if vrow else None}
    return out


# ----------------------------------------------------------------------------- extreme points

def extreme_point_test(items, feats, heads: dict[str, list[str]]) -> dict:
    """Does the off-axis coordinate of an edge reader's peak follow the object's extreme point on that edge?

    For a max-edge reader at an x step (it sits at u ~ 1), compare its v with the v of the rightmost mask
    pixels (extreme point), with the mask centroid v and with the corner value v = 1: Pearson r and mean
    absolute difference. Likewise min-edge readers vs the leftmost pixels, and y steps vs bottom/top pixels.
    Group peaks (mean attention of the group, restricted to tokens near the box so that row-start tokens in
    the first image column do not count) are used; boxes must span >= 3 tokens on both axes.
    """
    plan = [('max', 'pt_x', 'right', 1), ('max', 'bb_x2', 'right', 1), ('min', 'bb_x1', 'left', 1),
            ('min', 'pt_x', 'left', 1), ('max', 'pt_y', 'bottom', 0), ('max', 'bb_y2', 'bottom', 0),
            ('min', 'bb_y1', 'top', 0), ('min', 'pt_y', 'top', 0)]
    out = {}
    for grp, step, edge, comp in plan:
        s = STEP_NAMES.index(step)
        rows = []
        for it, f in zip(items, feats):
            if 'ext' not in f or np.isnan(it['att'][s, 0, 0, 0]) or f['w'] < MIN_SPAN_PX or f['h'] < MIN_SPAN_PX:
                continue
            pk = group_peak(it, heads[grp], s, near_box=True)
            b = it['box']
            uv = [(pk[0] - b[0]) / f['w'], (pk[1] - b[1]) / f['h']]
            cen = f['ext']['centroid_v'] if comp == 1 else f['ext']['centroid_u']
            rows.append([uv[comp], f['ext'][edge], cen, uv[1 - comp]])
        if len(rows) < 10:
            continue
        r = np.array(rows)
        corner = 1.0 if grp == 'max' else 0.0
        out[f'{grp}_{step}'] = {
            'n': len(r), 'edge': edge, 'off_axis': 'v' if comp == 1 else 'u',
            'on_axis_median': float(np.median(r[:, 3])),
            'r_extreme': float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]), 'r_centroid': float(np.corrcoef(r[:, 0], r[:, 2])[0, 1]),
            'mad_extreme': float(np.abs(r[:, 0] - r[:, 1]).mean()), 'mad_centroid': float(np.abs(r[:, 0] - r[:, 2]).mean()),
            'mad_corner': float(np.abs(r[:, 0] - corner).mean()), 'mad_mid': float(np.abs(r[:, 0] - 0.5).mean()),
            'points': r.round(3).tolist()}
    return out


# ----------------------------------------------------------------------------- box-normalized density

def density_maps(items, heads: dict[str, list[str]], sel_fn=None) -> tuple[np.ndarray, list[int]]:
    """(n_sets, 6, B, B) mean attention relative to uniform in box-normalized bins over DENSITY_RANGE.

    For each image the attention of the head (or the mean over a head group) is divided by the uniform value
    1/Nv; each bin takes the mean of this ratio over the tokens whose centre falls in it, and bins are averaged
    over images (empty bins ignored). Only boxes spanning >= 4 tokens on both axes contribute.
    """
    lo, hi = DENSITY_RANGE
    edges = np.linspace(lo, hi, DENSITY_BINS + 1)
    names = list(heads)
    acc = np.zeros((len(names), 6, DENSITY_BINS, DENSITY_BINS))
    cnt = np.zeros((6, DENSITY_BINS, DENSITY_BINS))
    nimg = [0] * 6
    for it in items:
        b = it['box']
        if b[2] - b[0] < 4 * TP or b[3] - b[1] < 4 * TP or (sel_fn is not None and not sel_fn(it)):
            continue
        cen = token_centres(it['grid'])
        u = (cen[:, 0] - b[0]) / (b[2] - b[0])
        v = (cen[:, 1] - b[1]) / (b[3] - b[1])
        iu, iv = np.digitize(u, edges) - 1, np.digitize(v, edges) - 1
        ok = (iu >= 0) & (iu < DENSITY_BINS) & (iv >= 0) & (iv < DENSITY_BINS)
        nv = len(cen)
        for s in range(6):
            if np.isnan(it['att'][s, 0, 0, 0]):
                continue
            nimg[s] += 1
            n_tok = np.zeros((DENSITY_BINS, DENSITY_BINS))
            np.add.at(n_tok, (iv[ok], iu[ok]), 1)
            filled = n_tok > 0
            cnt[s] += filled
            for k, name in enumerate(names):
                a = np.mean([it['att'][s, hid(h)[0], hid(h)[1]] for h in heads[name]], 0) * nv
                sm = np.zeros((DENSITY_BINS, DENSITY_BINS))
                np.add.at(sm, (iv[ok], iu[ok]), a[ok])
                acc[k, s][filled] += sm[filled] / n_tok[filled]
    with np.errstate(invalid='ignore'):
        dens = acc / cnt[None]
    return dens, nimg


# ----------------------------------------------------------------------------- parts (salient part)

def part_enrichment(items, heads: list[str], steps: list[str]) -> dict:
    """Attention share on each annotated part relative to the part's area share, per category and part.

    For an image with a part-label map, token coverage cov[t, k] of part k is computed; restricted to tokens
    covered by any part, share_k = sum_t a_t cov[t, k] / sum_t a_t cov[t, any] and area_k = sum_t cov[t, k] /
    sum_t cov[t, any]. Enrichment = mean share / mean area over the images of a category (1 = no preference).
    The peak statistic counts the images whose peak token has the largest coverage by part k.
    """
    res: dict = {}
    for it in items:
        c = it['case']
        if 'parts_path' not in c:
            continue
        lab = np.asarray(Image.open(c['parts_path']))
        names = c['part_names']
        cov = np.stack([token_coverage(lab == k + 1, it['grid']) for k in range(len(names))], 1)   # (Nv, K)
        anyc = cov.sum(1)
        if anyc.sum() < 1:
            continue
        cat = f"{c['group']}:{c['category']}"
        for h in heads:
            li, hh = hid(h)
            for st in steps:
                s = STEP_NAMES.index(st)
                if np.isnan(it['att'][s, 0, 0, 0]):
                    continue
                a = it['att'][s, li, hh]
                tot = (a * anyc).sum()
                if tot <= 0:
                    continue
                share = (a[:, None] * cov).sum(0) / tot
                area = cov.sum(0) / anyc.sum()
                j = int(np.argmax(a))
                d = res.setdefault(h, {}).setdefault(st, {}).setdefault(cat, {'n': 0, 'family': it['family'], 'parts': {}})
                d['n'] += 1
                for k, nm in enumerate(names):
                    p = d['parts'].setdefault(nm, {'share': 0.0, 'area': 0.0, 'n': 0, 'peak': 0})
                    p['share'] += float(share[k])
                    p['area'] += float(area[k])
                    p['n'] += 1
                    p['peak'] += int(anyc[j] > 0 and np.argmax(cov[j]) == k and cov[j, k] > 0)
    for h in res.values():
        for st in h.values():
            for cat in st.values():
                for p in cat['parts'].values():
                    p['enrichment'] = p['share'] / p['area'] if p['area'] > 0 else float('nan')
                    p['share'] /= p['n']
                    p['area'] /= p['n']
                    p['peak'] /= p['n']
    return res


def part_family_summary(enr: dict, head: str, step: str, part_aliases: dict[str, str]) -> dict:
    """Pool categories into families; per family the attention share, area share and enrichment of each part."""
    out: dict = {}
    for cat, d in enr.get(head, {}).get(step, {}).items():
        fam = d['family']
        for nm, p in d['parts'].items():
            key = part_aliases.get(nm, nm)
            q = out.setdefault(fam, {}).setdefault(key, {'share': 0.0, 'area': 0.0, 'n': 0, 'peak': 0.0})
            q['share'] += p['share'] * p['n']
            q['area'] += p['area'] * p['n']
            q['peak'] += p['peak'] * p['n']
            q['n'] += p['n']
    for fam in out.values():
        for q in fam.values():
            q['enrichment'] = q['share'] / q['area'] if q['area'] > 0 else float('nan')
            q['share'] /= q['n']
            q['area'] /= q['n']
            q['peak'] /= q['n']
    return out


# ----------------------------------------------------------------------------- salient point summary

def salient_summary(items, feats, head: str = '20.11') -> dict:
    """Per family: median (u, v) of the head's peak at pt_x and pt_y, fraction of peaks on the mask, and the
    distance (0-1000 units) from the peak to the box centre and to the mask centroid."""
    li, hh = hid(head)
    out = {}
    for fam in sorted({it['family'] for it in items}) + ['all']:
        rows = []
        for it, f in zip(items, feats):
            if (fam != 'all' and it['family'] != fam) or np.isnan(it['att'][0, 0, 0, 0]) or np.isnan(it['att'][1, 0, 0, 0]):
                continue
            pt = np.array([f['peak'][0, li, hh, 0], f['peak'][1, li, hh, 1]])        # x from the x step, y from the y step
            W, H = it['canvas']
            b = it['box']
            bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            d_box = np.linalg.norm((pt - bc) * 1000 / np.array([W, H]))
            row = [f['uv'][0, li, hh, 0], f['uv'][1, li, hh, 1], d_box, np.nan, np.nan]
            if 'ext' in f:
                mc = np.array([b[0] + f['ext']['centroid_u'] * f['w'], b[1] + f['ext']['centroid_v'] * f['h']])
                row[3] = np.linalg.norm((pt - mc) * 1000 / np.array([W, H]))
                m = mask_of(it['case'])
                row[4] = float(m[min(int(pt[1]), m.shape[0] - 1), min(int(pt[0]), m.shape[1] - 1)])
            rows.append(row)
        if len(rows) < 3:
            continue
        r = np.array(rows, float)
        out[fam] = {'n': len(r), 'u_med': float(np.median(r[:, 0])), 'v_med': float(np.median(r[:, 1])),
                    'v_top_third': float((r[:, 1] < 1 / 3).mean()),
                    'd_box_med': float(np.median(r[:, 2])), 'd_mask_med': float(np.nanmedian(r[:, 3])),
                    'on_mask': float(np.nanmean(r[:, 4]))}
    return out


# ----------------------------------------------------------------------------- stability across families and sizes

def stability(items, feats, heads: list[str]) -> dict:
    """Edge regression coefficients and band fractions per family and per area quartile."""
    out = {'family': {}, 'size': {}}
    fams = sorted({it['family'] for it in items})
    for fam in fams:
        m = np.array([it['family'] == fam for it in items])
        if m.sum() < 15:
            continue
        reg = regression(items, feats, m)
        out['family'][fam] = {'n': int(m.sum()), 'beta': {h: reg['beta'][:, :, hid(h)[0], hid(h)[1]].round(3).tolist()
                                                         for h in heads},
                              'bands': band_stats(items, feats, heads, lambda it, fam=fam: it['family'] == fam)}
    area = np.array([it['area_frac'] for it in items])
    q = np.percentile(area, [25, 50, 75])
    bins = np.digitize(area, q)
    for k in range(4):
        m = bins == k
        reg = regression(items, feats, m)
        lo = float(area[m].min())
        hi = float(area[m].max())
        out['size'][str(k)] = {'n': int(m.sum()), 'area_range': [lo, hi],
                               'beta': {h: reg['beta'][:, :, hid(h)[0], hid(h)[1]].round(3).tolist() for h in heads},
                               'bands': band_stats(items, feats, heads, lambda it, lo=lo, hi=hi: lo <= it['area_frac'] <= hi)}
    return out


# ----------------------------------------------------------------------------- generation, ablation, path

def gen_summary(run: Path, items) -> dict:
    gen = sr.read_jsonl(run / 'gen.jsonl')
    out = {}
    for fmt in ('point', 'bbox'):
        for key in ('group', 'family'):
            cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
            by: dict = {}
            for g in gen:
                if g['format'] != fmt:
                    continue
                k = cases[g['case']][key]
                d = by.setdefault(k, [0, 0, 0, []])
                d[0] += 1
                d[1] += int(g['score']['valid'])
                d[2] += int(g['score']['correct'])
                if fmt == 'bbox' and g['score']['valid']:
                    d[3].append(g['score']['iou'])
            out[f'{fmt}_{key}'] = {k: {'n': v[0], 'valid': v[1], 'hit': wilson(v[2], v[0]),
                                       'iou_median': float(np.median(v[3])) if v[3] else None} for k, v in by.items()}
    sc = sr.read_jsonl(run / 'spotcheck.jsonl')
    out['spotcheck'] = {'n': len(sc), 'answer_equal': sum(r['answer_equal'] for r in sc),
                        'text_equal': sum(r['text_equal'] for r in sc)}
    out['attn_check_max'] = max((g.get('attn_check', 0) for g in gen), default=None)
    return out


def ablate_summary(run: Path) -> dict:
    recs = list({r['id']: r for r in sr.read_jsonl(run / 'ablate.jsonl')}.values())
    if not recs:
        return {}
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    out = {}
    for split in ('rc', 'mc', 'all'):
        rs = [r for r in recs if split == 'all' or (split == 'rc') == (r['group'] == 'rc')]
        if not rs:
            continue
        sets = [x['set'] for x in rs[0]['results']]
        base = {r['id']: r['results'][0]['score']['correct'] for r in rs}
        d = {}
        for k, s in enumerate(sets):
            hit = [r['results'][k]['score']['correct'] for r in rs]
            b = sum(1 for r, h in zip(rs, hit) if base[r['id']] and not h)
            c = sum(1 for r, h in zip(rs, hit) if not base[r['id']] and h)
            shifts = []
            for r in rs:
                v0, v1 = r['results'][0]['values'], r['results'][k]['values']
                if v0 and v1 and len(v0) == 2 and len(v1) == 2:
                    W, H = cases[r['id']]['truth']['canvas']
                    bx = cases[r['id']]['truth']['box']
                    bw, bh = (bx[2] - bx[0]) * 1000 / W, (bx[3] - bx[1]) * 1000 / H
                    shifts.append([(v1[0]['value'] - v0[0]['value']) / bw, (v1[1]['value'] - v0[1]['value']) / bh])
            sh = np.array(shifts) if shifts else np.zeros((0, 2))
            d[s] = {'n': len(rs), 'hit': wilson(sum(hit), len(rs)), 'mcnemar_p': mcnemar_p(b, c),
                    'dx_mean': float(sh[:, 0].mean()) if len(sh) else None, 'dy_mean': float(sh[:, 1].mean()) if len(sh) else None}
        out[split] = d
    fam = {}
    for r in recs:
        f = cases[r['id']]['family'] if r['group'] != 'rc' else 'refcoco_' + cases[r['id']]['family']
        for x in r['results']:
            fam.setdefault(f, {}).setdefault(x['set'], []).append(x['score']['correct'])
    out['family'] = {f: {s: wilson(sum(v), len(v)) for s, v in d.items()} for f, d in fam.items()}
    area = np.array([cases[r['id']]['area_frac'] for r in recs])
    q = np.percentile(area, [25, 50, 75])
    bins = np.digitize(area, q)
    out['size'] = {}
    for k in range(4):
        rs = [r for r, b in zip(recs, bins) if b == k]
        d = {'area_range': [float(area[bins == k].min()), float(area[bins == k].max())]}
        h0 = [r['results'][0]['score']['correct'] for r in rs]
        for j, x in enumerate(rs[0]['results']):
            h1 = [r['results'][j]['score']['correct'] for r in rs]
            b = sum(a and not c for a, c in zip(h0, h1))
            c = sum((not a) and c for a, c in zip(h0, h1))
            d[x['set']] = {'hit': wilson(sum(h1), len(rs)), 'lost': b, 'gained': c, 'mcnemar_p': mcnemar_p(b, c)}
        out['size'][str(k)] = d
    return out


def box_tokens(box, grid) -> np.ndarray:
    cen = token_centres(grid)
    return (cen[:, 0] >= box[0]) & (cen[:, 0] <= box[2]) & (cen[:, 1] >= box[1]) & (cen[:, 1] <= box[3])


def path_summary(run: Path, rng) -> dict:
    """Selection shift of later heads and first-digit recovery under the what-pair conditions.

    For head h at a first-digit query, D = m(B) - m(A) is the attention mass on object B's tokens minus that on
    object A's tokens (tokens inside both boxes excluded). The attention recovery of condition e is
    s_e = mean(D_e - D_none) / mean(D_runB - D_none) over pairs and both queries: 1 means the patch moves the
    head's attention between the two objects as much as replacing the whole expression does.
    The logit recovery r_e is the round-1/2 first-digit measure (d_A != d_B and |LD_B - LD_A| >= 2).
    """
    recs = list({r['id']: r for r in sr.read_jsonl(run / 'path.jsonl')}.values())
    if not recs:
        return {}
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    conds = recs[0]['conds']
    out = {}
    for split in ('rc', 'mc', 'all'):
        rs = [r for r in recs if split == 'all' or (split == 'rc') == (r['group'] == 'rc')]
        if not rs:
            continue
        D, DB, R = [], [], []
        for r in rs:
            c = cases[r['id']]
            z = np.load(run / 'path_attn' / f"{r['id']}.npz")
            A, Bm = z['a'].astype(np.float32), z['b'].astype(np.float32)   # (R, nq, 4, 16, Nv), (nq, 4, 16, Nv)
            ta, tb = box_tokens(c['truth']['box'], r['grid']), box_tokens(c['truth_b']['box'], r['grid'])
            ta, tb = ta & ~tb, tb & ~ta
            norm = lambda x: x / x.sum(-1, keepdims=True)
            A, Bm = norm(A), norm(Bm)
            D.append(A[..., tb].sum(-1) - A[..., ta].sum(-1))                # (R, nq, 4, 16)
            DB.append(Bm[..., tb].sum(-1) - Bm[..., ta].sum(-1))             # (nq, 4, 16)
            for v in r['values']:
                if v['dA'] is None or v['dA'] == v['dB']:
                    continue
                ldA, ldB = v['ld'][0], v['LD_B']
                if abs(ldB - ldA) < 2:
                    continue
                R.append([(x - ldA) / (ldB - ldA) for x in v['ld']])
        D, DB = np.stack(D), np.stack(DB)                                     # (N, R, nq, 4, 16), (N, nq, 4, 16)
        num = (D - D[:, :1]).mean((0, 2))                                     # (R, 4, 16)
        den = (DB - D[:, 0]).mean((0, 1))                                     # (4, 16)
        s = num / den[None]
        boot = []
        for _ in range(500):
            j = rng.integers(0, len(D), len(D))
            boot.append((D[j] - D[j][:, :1]).mean((0, 2)) / (DB[j] - D[j][:, 0]).mean((0, 1))[None])
        boot = np.array(boot)
        Rm = np.array(R)
        rb = np.array([Rm[rng.integers(0, len(Rm), len(Rm))].mean(0) for _ in range(500)]) if len(Rm) else None
        out[split] = {'n_pairs': len(rs), 'conds': conds, 'layers': [20, 24, 28, 32],
                      'den': den.tolist(), 's': s.tolist(),
                      's_lo': np.percentile(boot, 2.5, 0).tolist(), 's_hi': np.percentile(boot, 97.5, 0).tolist(),
                      'n_values': int(len(Rm)),
                      'r': Rm.mean(0).tolist() if len(Rm) else None,
                      'r_ci': [np.percentile(rb, 2.5, 0).tolist(), np.percentile(rb, 97.5, 0).tolist()] if rb is not None else None,
                      'attn_check_max': max(max(r['attn_check'][k] for k in r['attn_check'] if k != '20') for r in rs)}
    return out


def ablation_attention_shift(run: Path) -> dict:
    """Under H11 mean ablation (path condition H11_mean): change of each later head's peak in box units and of
    its attention mass on the target box, relative to the unpatched run A (rc and mc what-pair cases)."""
    recs = list({r['id']: r for r in sr.read_jsonl(run / 'path.jsonl')}.values())
    if not recs:
        return {}
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    conds = recs[0]['conds']
    k = conds.index('H11_mean')
    dm, dp = [], []
    for r in recs:
        c = cases[r['id']]
        z = np.load(run / 'path_attn' / f"{r['id']}.npz")['a'].astype(np.float32)
        z = z / z.sum(-1, keepdims=True)
        ta = box_tokens(c['truth']['box'], r['grid'])
        dm.append(z[k][..., ta].sum(-1) - z[0][..., ta].sum(-1))             # (nq, 4, 16)
        b = c['truth']['box']
        p0, p1 = peaks(z[0], r['grid']), peaks(z[k], r['grid'])               # (nq, 4, 16, 2)
        dp.append(np.stack([(p1[..., 0] - p0[..., 0]) / (b[2] - b[0]), (p1[..., 1] - p0[..., 1]) / (b[3] - b[1])], -1))
    dm, dp = np.stack(dm), np.stack(dp)
    return {'n': len(recs), 'd_mass_target': dm.mean(0).tolist(), 'abs_peak_shift_median': np.median(np.abs(dp), 0).tolist(),
            'frac_peak_moved': (np.abs(dp).max(-1) > 0.25).mean(0).tolist()}


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--round2', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261103)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    items, cases = load_items(args.run)
    feats = [item_features(it) for it in items]
    print(f'{len(items)} items', flush=True)
    reg = regression(items, feats)
    heads_all = [f'{l}.{h}' for l in FULL_LAYERS for h in range(16)]
    groups = {'edge_min': EDGE_MIN, 'edge_max': EDGE_MAX, 'H11': ['20.11']}
    dens_sets = {**{h: [h] for h in KEY_HEADS}, **groups}
    dens, dens_n = density_maps(items, dens_sets)
    dens_rc, _ = density_maps(items, groups, lambda it: it['group'] == 'rc')
    dens_mc, _ = density_maps(items, groups, lambda it: it['group'] != 'rc')
    summary = {
        'n_items': len(items), 'steps': STEP_NAMES, 'key_heads': KEY_HEADS, 'edge_min': EDGE_MIN, 'edge_max': EDGE_MAX,
        'regression_n': reg['n'],
        'beta_key': {h: reg['beta'][:, :, hid(h)[0], hid(h)[1]].round(3).tolist() for h in KEY_HEADS},
        'r2_key': {h: reg['r2'][:, :, hid(h)[0], hid(h)[1]].round(3).tolist() for h in KEY_HEADS},
        'beta_ci': bootstrap_beta(items, feats, KEY_HEADS, rng),
        'bands': band_stats(items, feats, KEY_HEADS),
        'bands_all': band_stats(items, feats, [h for h in heads_all if int(h.split('.')[0]) >= 16]),
        'row_start': row_start(items, KEY_HEADS),
        'extreme': extreme_point_test(items, feats, {'min': EDGE_MIN, 'max': EDGE_MAX}),
        'density_sets': list(dens_sets), 'density_n': dens_n,
        'stability': stability(items, feats, KEY_HEADS),
        'salient': {h: salient_summary(items, feats, h) for h in ('20.11', '24.11', '20.2')},
        'gen': gen_summary(args.run, items),
        'ablate': ablate_summary(args.run),
        'path': path_summary(args.run, rng),
        'path_ablation_shift': ablation_attention_shift(args.run),
    }
    enr = part_enrichment(items, ['20.11', '24.7', '20.10', '24.5', '20.0', '24.11'], ['pt_x', 'pt_y', 'bb_x1', 'bb_x2'])
    summary['parts'] = enr
    summary['parts_family'] = {h: {st: part_family_summary(enr, h, st, {}) for st in ('pt_x', 'pt_y')}
                               for h in ('20.11', '24.7', '20.10')}
    (args.run / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    np.savez_compressed(args.run / 'arrays.npz', beta=reg['beta'], r2=reg['r2'], density=dens,
                        density_rc=dens_rc, density_mc=dens_mc)
    print('summary written', flush=True)


if __name__ == '__main__':
    main()
