"""Experiment 3.15 statistics and figures (CPU only).

Inputs: the run directory of ordinal_vs_coordinate_run.py (answer, cont, gate, refcoco_* jsonl), the
experiment-1.7 generations (grid point answers, direct / think / answer-first) and the experiment-3.9 r2
point answers on the experiment-1.5 slot grids (ordcoord.jsonl). Output: summary.json and figures.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from analysis_tools.behavior import cot_mechanism as cm
from analysis_tools.position import ordinal_vs_coordinate as ov

BLUE, ORANGE, AQUA, YELLOW, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#9a9893', '#0b0b0b'
FULL_LAYERS = tuple(range(4, 33, 4))
L20 = FULL_LAYERS.index(20)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def rate(k: int, n: int) -> dict:
    lo, hi = ov.wilson(k, n)
    return {'k': int(k), 'n': int(n), 'p': k / n if n else math.nan, 'ci': [lo, hi]}


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for discordant counts b, c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(x, n_boot: int = 2000, seed: int = 0) -> list[float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n_boot)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e6e5e1', lw=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- A. 1.7 value structure

def value_structure(work: Path) -> dict:
    """Offsets of 1.7 grid point answers from the column/row ordinal (x/100 - col, y/100 - row)."""
    rs = read_jsonl(work / 'behavior/2026-10-21_cot_mechanism/gen/records.jsonl')
    out = {}
    for mode in ('direct', 'think', 'answer_first'):
        ox, oy, n_all = [], [], 0
        for r in rs:
            if r['task'] != 'point' or r['mode'] != mode:
                continue
            n_all += 1
            a = cm.final_answer('point', r['text'])[0] if mode != 'answer_first' else cm.first_answer('point', r['text'])
            if a is None:
                continue
            ox.append(a[0] / 100 - r['truth']['col'])
            oy.append(a[1] / 100 - r['truth']['row'])
        out[mode] = {'n': n_all, 'n_answer': len(ox), 'x_offsets': ox, 'y_offsets': oy,
                     'x_hist': dict(Counter(f'{round(v * 2) / 2:+.1f}' for v in ox)),
                     'y_hist': dict(Counter(f'{round(v * 2) / 2:+.1f}' for v in oy))}
    # CoT content of the think answers
    think = [r for r in rs if r['task'] == 'point' and r['mode'] == 'think']
    content = Counter()
    for r in think:
        t = r['text']
        rows, cols = ov.claims(t, 'row'), ov.claims(t, 'col')
        content['n'] += 1
        content['states_grid_size'] += bool(re.search(r'10\s*(?:x|×|by)\s*10|10 rows', t))
        content['states_row_and_col'] += bool(rows and cols)
        content['first_claim_correct'] += bool(rows and cols and rows[0][2] == r['truth']['row'] and cols[0][2] == r['truth']['col'])
        content['has_equals'] += '=' in t
        content['half_cell_term'] += bool(re.search(r'0\.5|\+\s*50\b|-\s*50\b|half', t))
        content['self_correction'] += bool(re.search(r'\bwait\b|\bactually\b', t, re.I))
    out['think_content'] = dict(content)
    return out


def fig_values(vs: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    bins = np.arange(-1.75, 1.8, 0.5)
    for ax, axis in zip(axes, ('x', 'y')):
        for k, (mode, col, lab) in enumerate((('direct', ORANGE, 'direct'), ('think', BLUE, 'think (full CoT)'))):
            v = np.asarray(vs[mode][f'{axis}_offsets'])
            h, _ = np.histogram(v, bins=bins)
            ax.bar(bins[:-1] + 0.25 + (k - 0.5) * 0.18, h / max(1, len(v)), width=0.17, color=col, label=f'{lab} (n={len(v)})')
        ax.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        name = 'column' if axis == 'x' else 'row'
        ax.set_xlabel(f'{axis}/100 − {name} ordinal (cells)')
        ax.axvline(-0.5, color=GREY, ls=':', lw=1)
        ax.text(-0.5, 1.02, 'cell centre', ha='center', fontsize=8, color='#52514e', transform=ax.get_xaxis_transform())
        ax.axvline(0, color=GREY, ls='--', lw=1)
        ax.text(0.0, 1.08, 'right/bottom edge', ha='center', fontsize=8, color='#52514e', transform=ax.get_xaxis_transform())
        style(ax)
    axes[0].set_ylabel('share of answers')
    axes[1].legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- B. 3.9 r2 slot-grid re-analysis

def extent_reanalysis(work: Path) -> dict:
    """Condition-mean point shifts on the 1.5 slot grids vs three predictors (cells along the long axis)."""
    cases = {}
    for c in read_jsonl(work / 'behavior/2026-10-16_ordinal_mechanism/cases.jsonl'):
        if c['mode'] == 'direct':
            cases['images:' + c['condition'] + ':' + Path(c['image']).stem] = c
    pts = {}
    for r in read_jsonl(work / 'position/2026-10-20_readout_tool_r2/ordcoord.jsonl'):
        c = cases.get(r['image_id'])
        m = re.search(r'"point_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)', r['text'])
        if c is None or not m:
            continue
        t = c['truth']
        ax = 0 if t['axis'] == 'H' else 1
        size, cell, p, k, n = t['canvas'][ax], t['cell'], t['present'], t['count'], t['n_present']
        unit = 1000 * cell / size
        xf, xl = (p[0] + 0.5) * cell, (p[-1] + 0.5) * cell
        pts[(c['condition'], c['base'], c['variant'])] = {
            'v': float(m.group(1 + ax)) / unit,
            'pixel': (t['target_slot'] + 0.5),
            'count_edge': float(k),
            'extent': (xf + (k - 1) / (n - 1) * (xl - xf)) / cell}
    out = {}
    for var in ('shift', 'insL', 'delL', 'delR'):
        d = defaultdict(list)
        for (cond, b, v), r in pts.items():
            if v != var or (cond, b, 'base') not in pts:
                continue
            r0 = pts[(cond, b, 'base')]
            for key in ('v', 'pixel', 'count_edge', 'extent'):
                d[key].append(r[key] - r0[key])
        out[var] = {'n': len(d['v']), 'observed': float(np.mean(d['v'])), 'observed_ci': boot_ci(d['v']),
                    'observed_median': float(np.median(d['v'])),
                    **{k: float(np.mean(d[k])) for k in ('pixel', 'count_edge', 'extent')}}
    return out


def fig_extent(ex: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    vars_ = ['shift', 'insL', 'delL', 'delR']
    x = np.arange(len(vars_))
    w = 0.2
    series = [('observed', INK, 'model answer (mean, 95% bootstrap CI)'), ('pixel', BLUE, 'pixel position'),
              ('count_edge', ORANGE, 'count × cell (ordinal route)'), ('extent', AQUA, 'count within occupied extent')]
    for k, (key, col, lab) in enumerate(series):
        vals = [ex[v][key] for v in vars_]
        ax.bar(x + (k - 1.5) * w, vals, width=w * 0.9, color=col, label=lab)
        if key == 'observed':
            lo = [ex[v]['observed'] - ex[v]['observed_ci'][0] for v in vars_]
            hi = [ex[v]['observed_ci'][1] - ex[v]['observed'] for v in vars_]
            ax.errorbar(x + (k - 1.5) * w, vals, yerr=[lo, hi], fmt='none', ecolor='#52514e', lw=1)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x, [f'{v}\n(n={ex[v]["n"]})' for v in vars_])
    ax.set_ylabel('shift vs base (cells)')
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='lower left')
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- C. forced answers on grids

def hit(xy, truth) -> bool:
    x0, y0, x1, y1 = truth['cell_box_norm']
    return xy is not None and x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


def centre_err(xy, truth) -> float:
    if xy is None:
        return math.nan
    return float(math.hypot(xy[0] - truth['point'][0], xy[1] - truth['point'][1]))


def digit_stats(rec: dict) -> dict:
    """Sharpness of the x first and second digit and image attention at the x first-digit step."""
    lp = rec['logp_digits']
    toks = rec['tokens']
    out = {}
    if len(lp) >= 2 and toks and toks[0].isdigit():
        p1, p2 = np.exp(lp[0]), np.exp(lp[1])
        out['p_first_max'] = float(p1.max() / p1.sum()) if p1.sum() > 0 else math.nan
        out['p_second_max'] = float(p2.max() / p2.sum()) if p2.sum() > 0 else math.nan
    k = rec['n_cue_steps_kept']
    att = np.asarray(rec['img_att'])                    # (steps, 8 layers, 16 heads)
    if att.shape[0] > k:
        out['img_att_l20'] = float(att[k, L20].mean())
        out['img_att_l20_h11'] = float(att[k, L20, 10])
        out['img_att_layers'] = att[k].mean(1).tolist()
        out['img_att_cue_l20'] = float(att[:k, L20].mean()) if k else math.nan
    return out


def answer_analysis(run: Path) -> dict:
    rs = read_jsonl(run / 'answer.jsonl')
    by = defaultdict(list)
    for r in rs:
        by[r['condition']].append(r)
    out = {}
    for cond, recs in by.items():
        img_truth = [r['truthA'] if r['image'] == 'A' else r['truthB'] for r in recs]
        hits = [hit(r['xy'], t) for r, t in zip(recs, img_truth)]
        errs = [centre_err(r['xy'], t) for r, t in zip(recs, img_truth)]
        ds = [digit_stats(r) for r in recs]
        row = {'n': len(recs), 'hit_image_cell': rate(sum(hits), len(recs)),
               'centre_err_mean': float(np.nanmean(errs)), 'parsed': sum(r['xy'] is not None for r in recs)}
        for key in ('p_first_max', 'p_second_max', 'img_att_l20', 'img_att_l20_h11', 'img_att_cue_l20'):
            v = [d[key] for d in ds if key in d]
            row[key] = float(np.nanmean(v)) if v else math.nan
            row[key + '_ci'] = boot_ci(v)
        lay = [d['img_att_layers'] for d in ds if 'img_att_layers' in d]
        row['img_att_layers'] = np.mean(lay, 0).tolist() if lay else []
        # follow weights: text-implied vs image-implied centre
        wx, wy, exact_text, exact_img_edge = [], [], 0, 0
        for r in recs:
            if r['xy'] is None:
                continue
            idx = written_indices(r)
            t_img = r['truthA'] if r['image'] == 'A' else r['truthB']
            if cond == 'f0_imgB':             # image B without text: weight of B relative to A
                idx = (r['truthB']['row'], r['truthB']['col'])
                t_img = r['truthA']
            if idx is None:
                continue
            tx, ty = ov.cell_centre(idx[1], 10), ov.cell_centre(idx[0], 10)
            ix, iy = ov.cell_centre(t_img['col'], 10), ov.cell_centre(t_img['row'], 10)
            a, b = ov.follow_weight(r['xy'][0], ix, tx), ov.follow_weight(r['xy'][1], iy, ty)
            if a is not None:
                wx.append(a)
            if b is not None:
                wy.append(b)
            exact_text += int(abs(r['xy'][0] - tx) < 1 and abs(r['xy'][1] - ty) < 1)
            exact_img_edge += int(abs(r['xy'][0] - 100 * t_img['col']) < 1)
        if wx or wy:
            row['follow_x'] = {'mean': float(np.mean(wx)), 'median': float(np.median(wx)), 'ci': boot_ci(wx), 'n': len(wx)}
            row['follow_y'] = {'mean': float(np.mean(wy)), 'median': float(np.median(wy)), 'ci': boot_ci(wy), 'n': len(wy)}
            row['exact_text_centre'] = rate(exact_text, row['parsed'])
        out[cond] = row
    # paired mask effects (same item, prefix with vs without answer-time image mask)
    pairs = {}
    idx = {(r['item'], r['condition']): r for r in rs}
    for base in ('f0', 'tplA', 'tplB', 'own', 'full'):
        items = [i for (i, c) in idx if c == base and (i, base + '_mask') in idx]
        same = sum(idx[(i, base)]['xy'] == idx[(i, base + '_mask')]['xy'] for i in items)
        h0 = [hit(idx[(i, base)]['xy'], idx[(i, base)]['truthA']) for i in items]
        h1 = [hit(idx[(i, base + '_mask')]['xy'], idx[(i, base)]['truthA']) for i in items]
        b = sum(x and not y for x, y in zip(h0, h1))
        c = sum(y and not x for x, y in zip(h0, h1))
        pairs[base] = {'n': len(items), 'identical_answer': rate(same, len(items)), 'hit': sum(h0), 'hit_mask': sum(h1),
                       'mcnemar_p': mcnemar(b, c)}
    return {'conditions': out, 'mask_pairs': pairs}


def written_indices(r: dict) -> tuple[int, int] | None:
    c = r['condition']
    A, B = r['truthA'], r['truthB']
    if c in ('tplA', 'tplA_mask', 'tplA_imgB'):
        return A['row'], A['col']
    if c in ('tplB', 'tplB_mask'):
        return B['row'], B['col']
    if c in ('own', 'own_mask', 'own_imgB', 'ownB'):
        rows, cols = ov.claims(r['prefix'], 'row'), ov.claims(r['prefix'], 'col')
        return (rows[0][2], cols[0][2]) if rows and cols else None
    return None


COND_LABEL = {'f0': 'no\nCoT', 'tplA': 'template\nindices', 'own': 'own CoT\nto indices', 'pre': 'own CoT\nbefore indices',
              'full': 'full\nCoT'}


def fig_follow(ans: dict, path: Path) -> None:
    conds = [('f0_imgB', 'no text,\nimage B'), ('tplB', 'template B,\nimage A'), ('tplA_imgB', 'template A,\nimage B'),
             ('ownB', 'own CoT B,\nimage A'), ('own_imgB', 'own CoT A,\nimage B')]
    conds = [(c, l) for c, l in conds if c in ans['conditions'] and 'follow_x' in ans['conditions'][c]]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    x = np.arange(len(conds))
    for k, (axis, col) in enumerate((('x', BLUE), ('y', ORANGE))):
        m = [ans['conditions'][c][f'follow_{axis}']['mean'] for c, _ in conds]
        ci = [ans['conditions'][c][f'follow_{axis}']['ci'] for c, _ in conds]
        ax.bar(x + (k - 0.5) * 0.36, m, width=0.34, color=col, label=f'{axis} axis')
        ax.errorbar(x + (k - 0.5) * 0.36, m, yerr=[[mm - c[0] for mm, c in zip(m, ci)], [c[1] - mm for mm, c in zip(m, ci)]],
                    fmt='none', ecolor='#52514e', lw=1)
    ax.axhline(1, color=GREY, ls='--', lw=1)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.text(len(conds) - 0.5, 1.13, 'w = 1: follows the written index (no-text bar: follows image B)', ha='right', va='bottom', fontsize=8, color='#52514e')
    ax.set_xticks(x, [f'{l}\n(n={ans["conditions"][c]["n"]})' for c, l in conds], fontsize=8)
    ax.set_ylabel('follow weight w')
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, fontsize=8, loc='upper center', ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_mask_digits(ans: dict, rc: dict | None, path: Path) -> None:
    """(a) hit rate with / without answer-time image mask; (b) L20 image attention at the x first digit;
    (c) probability mass of the most likely x second digit."""
    conds = [c for c in ('f0', 'pre', 'tplA', 'own', 'full') if c in ans['conditions']]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    x = np.arange(len(conds))
    labels = [f'{COND_LABEL[c]}\n(n={ans["conditions"][c]["n"]})' for c in conds]
    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.12, 1.02, f'({letter})', transform=ax.transAxes, fontsize=10, weight='bold')
    for k, (suf, col, lab) in enumerate((('', BLUE, 'image visible'), ('_mask', ORANGE, 'image masked at answer'))):
        vals, errs = [], [[], []]
        for c in conds:
            row = ans['conditions'].get(c + suf)
            if row is None:
                vals.append(np.nan); errs[0].append(0); errs[1].append(0)
                continue
            h = row['hit_image_cell']
            vals.append(h['p']); errs[0].append(h['p'] - h['ci'][0]); errs[1].append(h['ci'][1] - h['p'])
        axes[0].bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=col, label=lab)
        axes[0].errorbar(x + (k - 0.5) * 0.36, vals, yerr=errs, fmt='none', ecolor='#52514e', lw=1)
    axes[0].set_ylabel('hit (answer in target cell)')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    for ax, key, ylab in ((axes[1], 'img_att_l20', 'L20 attention on image\nat the x first-digit step'),
                          (axes[2], 'p_second_max', 'max probability of\nthe x second digit')):
        vals = [ans['conditions'][c][key] for c in conds]
        ci = [ans['conditions'][c][key + '_ci'] for c in conds]
        ax.bar(x, vals, width=0.5, color=AQUA if key == 'img_att_l20' else YELLOW)
        ax.errorbar(x, vals, yerr=[[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]],
                    fmt='none', ecolor='#52514e', lw=1)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=7.5)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


OFFSET_TOL = 0.2   # cells (20 units on a 10-cell axis); categories are disjoint because they are 0.5 apart
OFFSET_CATS = (('centre', -0.5, BLUE), ('right/bottom edge', 0.0, ORANGE), ('left/top edge', -1.0, AQUA))


def offset_shares(values: list[float]) -> dict:
    """Share of answers whose value/100 - ordinal is the cell centre (-0.5), the far edge (0), the near edge (-1)."""
    n = len(values)
    out = {name: sum(abs(v - c) <= OFFSET_TOL for v in values) / n if n else math.nan for name, c, _ in OFFSET_CATS}
    out['other'] = 1 - sum(out.values()) if n else math.nan
    out['n'] = n
    return out


def offsets_by_condition(run: Path, vs: dict) -> dict:
    rs = read_jsonl(run / 'answer.jsonl')
    by = defaultdict(lambda: {'x': [], 'y': []})
    for r in rs:
        if r['xy'] is None or r['condition'] not in ('f0', 'tplA', 'tplA_mask', 'own', 'own_mask', 'full'):
            continue
        t = r['truthA']
        by[r['condition']]['x'].append(r['xy'][0] / 100 - t['col'])
        by[r['condition']]['y'].append(r['xy'][1] / 100 - t['row'])
    out = {'1.7 direct': {a: offset_shares(vs['direct'][f'{a}_offsets']) for a in 'xy'},
           '1.7 think': {a: offset_shares(vs['think'][f'{a}_offsets']) for a in 'xy'}}
    for c, d in by.items():
        out[c] = {a: offset_shares(d[a]) for a in 'xy'}
    return out


def fig_offsets(off: dict, path: Path) -> None:
    order = [('1.7 direct', 'direct\n(1.7)'), ('f0', 'no CoT\n(cue)'), ('tplA', 'template\nindices'),
             ('tplA_mask', 'template indices\n+ image mask'), ('own', 'own CoT\nto indices'), ('full', 'full\nCoT')]
    order = [(k, l) for k, l in order if k in off]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, a in zip(axes, 'xy'):
        x = np.arange(len(order))
        bottom = np.zeros(len(order))
        for name, _, col in OFFSET_CATS + (('other', None, '#d9d8d4'),):
            v = np.array([off[k][a][name] for k, _ in order])
            ax.bar(x, v, bottom=bottom, color=col, width=0.6, label=name, edgecolor='white', lw=1)
            bottom += v
        ax.set_xticks(x, [f'{l}\n(n={off[k][a]["n"]})' for k, l in order], fontsize=7.5)
        ax.set_title(f'{a} value relative to the {"column" if a == "x" else "row"} ordinal', fontsize=9)
        style(ax)
    axes[0].set_ylabel('share of answers')
    axes[1].legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_refcoco(rc: dict, path: Path) -> None:
    f = rc['forced']
    conds = [('f0', 'no CoT'), ('f0_mask', 'no CoT\n+ image mask'), ('full', 'full CoT'), ('full_mask', 'full CoT\n+ image mask')]
    conds = [(c, l) for c, l in conds if c in f]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    x = np.arange(len(conds))
    for ax, key, ylab, col in ((axes[0], 'hit', 'hit (answer inside box)', BLUE),
                               (axes[1], 'img_att_l20', 'L20 attention on image\nat the x first-digit step', AQUA),
                               (axes[2], 'p_second_max', 'max probability of\nthe x second digit', YELLOW)):
        if key == 'hit':
            vals = [f[c]['hit']['p'] for c, _ in conds]
            ci = [f[c]['hit']['ci'] for c, _ in conds]
        else:
            vals = [f[c][key] for c, _ in conds]
            ci = [f[c][key + '_ci'] for c, _ in conds]
        ax.bar(x, vals, width=0.55, color=col)
        ax.errorbar(x, vals, yerr=[[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]],
                    fmt='none', ecolor='#52514e', lw=1)
        ax.set_xticks(x, [f'{l}\n(n={f[c]["n"]})' for c, l in conds], fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
        style(ax)
    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.14, 1.02, f'({letter})', transform=ax.transAxes, fontsize=10, weight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- D. continuations

def cont_analysis(run: Path) -> dict:
    rs = read_jsonl(run / 'cont.jsonl')
    by = defaultdict(list)
    for r in rs:
        by[r['condition']].append(r)
    out = {}
    for cond, recs in by.items():
        follow_text, follow_img, other, none, wait, trunc = 0, 0, 0, 0, 0, 0
        wx = []
        for r in recs:
            idx = written_indices(r)
            t_img = r['truthA'] if r['image'] == 'A' else r['truthB']
            a = cm.final_answer('point', r['text'])[0]
            wait += bool(re.search(r'\bwait\b|\bactually\b|double-check|re-?examin', r['text'], re.I))
            trunc += r['finish_reason'] != 'eos'
            if a is None or idx is None:
                none += 1
                continue
            tx, ty = ov.cell_centre(idx[1], 10), ov.cell_centre(idx[0], 10)
            ix, iy = ov.cell_centre(t_img['col'], 10), ov.cell_centre(t_img['row'], 10)
            dt, di = math.hypot(a[0] - tx, a[1] - ty), math.hypot(a[0] - ix, a[1] - iy)
            if dt <= 50 and di <= 50:
                follow_text += 1          # text and image agree (consistent prefix)
                follow_img += 1
            elif dt <= 50 and dt < di:
                follow_text += 1
            elif di <= 50 and di < dt:
                follow_img += 1
            else:
                other += 1
            w = ov.follow_weight(a[0], ix, tx)
            if w is not None:
                wx.append(w)
        n = len(recs)
        out[cond] = {'n': n, 'follow_text': rate(follow_text, n), 'follow_image': rate(follow_img, n),
                     'other': other, 'no_answer': none, 'self_correction_phrase': rate(wait, n), 'truncated': trunc,
                     'follow_x_mean': float(np.mean(wx)) if wx else math.nan}
    return out


# ----------------------------------------------------------------------------- E. graded decay

def gate_analysis(run: Path) -> dict:
    rs = read_jsonl(run / 'gate.jsonl')
    ans = {}
    for r in rs:
        t = r['truth']
        if r['prompt'] == 'rowcol_direct':
            a = cm.final_answer('rowcol', r['text'])[0]
            ans[(r['item'], r['prompt'], r['s'])] = None if a is None else (a[0], a[1])
        else:
            a = cm.final_answer('point', r['text'])[0]
            stated = None
            if r['prompt'] == 'point_think':
                rows = ov.claims(r['text'], 'row')
                stated = rows[-1][2] if rows else None
            ans[(r['item'], r['prompt'], r['s'])] = None if a is None else (a[1] / 100, a[0] / 100, stated)
    items = sorted({k[0] for k in ans})
    s_vals = sorted({k[2] for k in ans}, reverse=True)
    out = {'s': s_vals, 'per_s': {}, 'pairs': []}
    for s in s_vals:
        row = {}
        for p in ('rowcol_direct', 'point_direct', 'point_think'):
            d_row, d_col = [], []
            for i in items:
                a, a0 = ans.get((i, p, s)), ans.get((i, p, 1.0))
                if a is None or a0 is None:
                    continue
                d_row.append(a[0] - a0[0])
                d_col.append(a[1] - a0[1])
            row[p] = {'n': len(d_row), 'd_row': float(np.mean(d_row)) if d_row else math.nan, 'd_row_ci': boot_ci(d_row),
                      'd_col': float(np.mean(d_col)) if d_col else math.nan}
        out['per_s'][str(s)] = row
    # per image pairs (s < 1): ordinal row shift vs point row shift
    for s in s_vals:
        if s == 1.0:
            continue
        for i in items:
            o, o0 = ans.get((i, 'rowcol_direct', s)), ans.get((i, 'rowcol_direct', 1.0))
            pd, pd0 = ans.get((i, 'point_direct', s)), ans.get((i, 'point_direct', 1.0))
            pt, pt0 = ans.get((i, 'point_think', s)), ans.get((i, 'point_think', 1.0))
            if None in (o, o0):
                continue
            out['pairs'].append({'item': i, 's': s, 'd_ord': o[0] - o0[0],
                                 'd_point_direct': None if None in (pd, pd0) else pd[0] - pd0[0],
                                 'd_point_think': None if None in (pt, pt0) else pt[0] - pt0[0],
                                 'd_stated_think': None if None in (pt, pt0) or None in (pt[2], pt0[2]) else pt[2] - pt0[2]})
    for key in ('d_point_direct', 'd_point_think'):
        P = [(p['d_ord'], p[key]) for p in out['pairs'] if p[key] is not None]
        if len(P) > 3:
            a = np.asarray(P, float)
            slope = float(np.polyfit(a[:, 0], a[:, 1], 1)[0]) if a[:, 0].std() > 0 else math.nan
            out[key + '_vs_ord'] = {'n': len(P), 'slope': slope, 'r': float(np.corrcoef(a.T)[0, 1]) if a.std(0).min() > 0 else math.nan}
    for key in ('d_point_direct', 'd_point_think'):
        strat = {}
        for lab, cond in (('ordinal_unchanged', lambda d: d == 0), ('ordinal_minus1', lambda d: d == -1),
                          ('ordinal_minus2_or_more', lambda d: d <= -2)):
            v = [p[key] for p in out['pairs'] if p[key] is not None and cond(p['d_ord'])]
            strat[lab] = {'n': len(v), 'mean': float(np.mean(v)) if v else math.nan, 'ci': boot_ci(v)}
        out[key + '_by_ordinal_shift'] = strat
    P = [(p['d_stated_think'], p['d_point_think']) for p in out['pairs'] if p['d_stated_think'] is not None and p['d_point_think'] is not None]
    if len(P) > 3:
        a = np.asarray(P, float)
        out['think_point_vs_stated'] = {'n': len(P), 'slope': float(np.polyfit(a[:, 0], a[:, 1], 1)[0]) if a[:, 0].std() > 0 else math.nan,
                                        'exact_follow': rate(int(np.sum(np.abs(a[:, 1] - a[:, 0]) < 0.01)), len(P))}
    # think answers: does y equal the arithmetic on the CoT's own stated row and declared row count?
    formula = defaultdict(lambda: [0, 0])
    grid_stated = defaultdict(Counter)
    for r in rs:
        if r['prompt'] != 'point_think' or r['finish_reason'] != 'eos':
            continue
        a = cm.final_answer('point', r['text'])[0]
        rows = ov.claims(r['text'], 'row')
        R = cm.declared_grid(r['text'])[0]
        grid_stated[str(r['s'])][str(R)] += 1
        if a is None or not rows or not R:
            continue
        formula[str(r['s'])][1] += 1
        formula[str(r['s'])][0] += abs(a[1] - ov.cell_centre(rows[-1][2], R)) <= 6
    out['think_y_equals_stated_formula'] = {k: rate(*v) for k, v in formula.items()}
    out['think_declared_rows'] = {k: dict(v) for k, v in grid_stated.items()}
    return out


def fig_gate(g: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    s = g['s']
    labels = {'rowcol_direct': ('row ordinal (direct)', INK), 'point_direct': ('point y/100 (direct)', ORANGE),
              'point_think': ('point y/100 (think)', BLUE)}
    for p, (lab, col) in labels.items():
        m = [g['per_s'][str(v)][p]['d_row'] for v in s]
        ci = [g['per_s'][str(v)][p]['d_row_ci'] for v in s]
        axes[0].errorbar(s, m, yerr=[[a - c[0] if not math.isnan(c[0]) else 0 for a, c in zip(m, ci)],
                                     [c[1] - a if not math.isnan(c[1]) else 0 for a, c in zip(m, ci)]],
                         color=col, marker='o', ms=6, lw=2, capsize=3, label=lab)
    axes[0].invert_xaxis()
    axes[0].set_xlabel('decay exponent s (alpha → alpha^s, layers 9–16, image tokens)')
    axes[0].set_ylabel('mean row shift vs s = 1 (cells)')
    axes[0].legend(frameon=False, fontsize=8)
    style(axes[0])
    strata = [('ordinal_unchanged', 'ordinal\nunchanged'), ('ordinal_minus1', 'ordinal\n−1'), ('ordinal_minus2_or_more', 'ordinal\n≤ −2')]
    x = np.arange(len(strata))
    for k, (key, col, lab) in enumerate((('d_point_direct', ORANGE, 'direct point'), ('d_point_think', BLUE, 'think point'))):
        st = g.get(key + '_by_ordinal_shift')
        if not st:
            continue
        m = [st[s_]['mean'] for s_, _ in strata]
        ci = [st[s_]['ci'] for s_, _ in strata]
        axes[1].bar(x + (k - 0.5) * 0.36, m, width=0.34, color=col, label=lab)
        axes[1].errorbar(x + (k - 0.5) * 0.36, m, yerr=[[a_ - c[0] for a_, c in zip(m, ci)], [c[1] - a_ for a_, c in zip(m, ci)]],
                         fmt='none', ecolor='#52514e', lw=1)
        for xi, s_ in zip(x, strata):
            axes[1].text(xi + (k - 0.5) * 0.36, 0.08, f'n={st[s_[0]]["n"]}', ha='center', fontsize=7, color='#52514e')
    axes[1].set_xticks(x, [l for _, l in strata], fontsize=8)
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].set_xlabel('same image, same s: shift of the direct row ordinal')
    axes[1].set_ylabel('mean point row shift (cells)')
    axes[1].legend(frameon=False, fontsize=8)
    style(axes[1])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- F. RefCOCO

def refcoco_analysis(run: Path, work: Path) -> dict | None:
    gen = read_jsonl(run / 'refcoco_gen.jsonl')
    if not gen:
        return None
    cases = {c['id']: c for c in read_jsonl(work / 'position/2026-10-20_readout_tool_r2/cases.jsonl')}

    def box_norm(c):
        W, H = c['truth']['canvas']
        x0, y0, x1, y1 = c['truth']['box']
        return [1000 * x0 / W, 1000 * y0 / H, 1000 * x1 / W, 1000 * y1 / H]

    def centre(c):
        b = box_norm(c)
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    g = {(r['id'], r['mode']): r for r in gen}
    ids = sorted({r['id'] for r in gen})
    res = {}
    hits = {}
    for mode in ('direct', 'think'):
        h, e, n_ans, trunc = [], [], 0, 0
        for i in ids:
            r = g.get((i, mode))
            a = cm.final_answer('point', r['text'])[0] if r else None
            trunc += bool(r and r['finish_reason'] != 'eos')
            ok = a is not None and ov.in_box(a, box_norm(cases[i]))
            h.append(ok)
            if a is not None:
                n_ans += 1
                cx, cy = centre(cases[i])
                e.append(math.hypot(a[0] - cx, a[1] - cy))
        hits[mode] = h
        res[mode] = {'hit': rate(sum(h), len(ids)), 'centre_err_median': float(np.median(e)) if e else math.nan,
                     'n_answer': n_ans, 'truncated': trunc}
    b = sum(x and not y for x, y in zip(hits['direct'], hits['think']))
    c = sum(y and not x for x, y in zip(hits['direct'], hits['think']))
    res['mcnemar_direct_vs_think'] = {'direct_only': b, 'think_only': c, 'p': mcnemar(b, c)}
    # CoT content
    codes = Counter()
    copy, n_pair = 0, 0
    for i in ids:
        r = g[(i, 'think')]
        body = ov.full_body(r['text']) or r['text']
        cd = ov.code_natural_cot(body)
        for k in ('region', 'numbers', 'pair', 'arithmetic', 'pixel'):
            codes[k] += cd[k]
        a = cm.final_answer('point', r['text'])[0]
        lp = ov.last_pair(body)
        if lp is not None and a is not None:
            n_pair += 1
            copy += math.hypot(a[0] - lp[0], a[1] - lp[1]) < 1
    res['cot_content'] = {k: rate(v, len(ids)) for k, v in codes.items()}
    res['answer_equals_last_written_pair'] = rate(copy, n_pair)
    # forced answers
    fa = read_jsonl(run / 'refcoco_answer.jsonl')
    byc = defaultdict(list)
    for r in fa:
        byc[r['condition']].append(r)
    forced = {}
    for cond, recs in byc.items():
        hh = [r['xy'] is not None and ov.in_box(r['xy'], box_norm(cases[r['id']])) for r in recs]
        ds = [digit_stats(r) for r in recs]
        ee = [math.hypot(r['xy'][0] - centre(cases[r['id']])[0], r['xy'][1] - centre(cases[r['id']])[1])
              for r in recs if r['xy'] is not None]
        row = {'n': len(recs), 'hit': rate(sum(hh), len(recs)), 'centre_err_median': float(np.median(ee)) if ee else math.nan,
               'centre_err_mean': float(np.mean(ee)) if ee else math.nan}
        for key in ('img_att_l20', 'p_second_max'):
            v = [d[key] for d in ds if key in d]
            row[key] = float(np.nanmean(v)) if v else math.nan
            row[key + '_ci'] = boot_ci(v)
        forced[cond] = row
    idx = {(r['id'], r['condition']): r for r in fa}
    for base in ('f0', 'full'):
        items = [i for (i, c) in idx if c == base and (i, base + '_mask') in idx]
        same = sum(idx[(i, base)]['xy'] == idx[(i, base + '_mask')]['xy'] for i in items)
        forced[base]['identical_with_mask'] = rate(same, len(items))
    res['forced'] = forced
    return res


# ----------------------------------------------------------------------------- pipeline figure

def fig_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.4)
    ax.set_ylim(0.1, 7.4)
    ax.axis('off')

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=col, ec='#52514e', lw=1))
        ax.text(x + 0.08, y + h - 0.1, title, fontsize=8.6, weight='bold', va='top', color=INK)
        ax.text(x + 0.08, y + h - 0.38, body, fontsize=7.1, va='top', color='#2b2b2b', linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='#52514e', lw=1.1))

    shared, ordc, coordc, cotc = '#e8f0fb', '#fde9df', '#e3f5ee', '#fdf2d6'
    box(0.1, 4.6, 2.5, 2.4, 'S1 ViT + merger', 'target position written into\ntarget tokens (RoPE; PE on\nuniform canvas)\n[3.4 C2, 3.7 r2]', shared)
    box(2.9, 4.6, 3.0, 2.4, 'S2 decoder L9–16 DeltaNet', 'raster accumulators on image\ntokens: row chain, column count\nreset at local boundaries;\nread from tokens AFTER the\ntarget in its row\n[3.8 r2, 3.10, 3.11, 3.14]', shared)
    box(6.2, 4.6, 3.0, 2.4, 'S3 answer-time look (L16–24)', 'answer queries re-read image KV;\nL20 H11 dominant on grids\n(0.89–0.93 of its mass on image);\nedge readers L20/L24 on photos\n[2.5c, 3.11, 3.11 r2, 3.15]', shared)
    box(9.5, 4.6, 3.4, 2.4, 'Text route', 'a number already written in text\nis copied / converted exactly\n(3.12 text conversion slope 0.99;\n3.15 full CoT: 2nd-digit p = 1.00,\nimage mask changes 0/50 answers)', cotc)
    box(0.1, 1.2, 2.9, 2.9, 'T1 ordinal readout', 'reads counts BEFORE the target;\n+1 added at readout;\nfixed before CoT (63/64);\naccurate on 10×10: 62/64\n[1.5, 3.10, 1.7, 3.15]', ordc)
    box(3.2, 1.2, 3.2, 2.9, 'T2 direct coordinate value', 'grids: count-based position,\n78% of x at 100·col (cell edge);\nshifts with counts before AND\nafter the target; alpha^s edit\nmoves it with the ordinal;\n1st digit from the look, 2nd digit\nfrom 1st-digit margin; 31/64 hit\n[3.9 r2, 3.12, 3.12 r2, 3.15]', coordc)
    box(6.6, 1.2, 3.1, 2.9, 'T3 CoT coordinate (grids)', 'step 1: write indices (T1)\n  hit 0.55 → 0.92, values stay\n  at cell edges; conflicts follow\n  the image (w ≈ 0.2)\nstep 2: write (c − 0.5)·100\n  answer copies the number\n  (centre error 65 → 2)  [1.7, 3.15]', cotc)
    box(9.9, 1.2, 3.0, 2.9, 'Natural images (RefCOCO)', 'no indices; CoT writes box edges\n+ arithmetic (68%), answer =\nlast written pair (107/109);\nwithout CoT the masked answer\nkeeps the region (hit 0.95) but\nloses precision (err 29 → 85)\n[3.11 r2, 3.15]', coordc)
    arrow(2.6, 5.8, 2.9, 5.8)
    arrow(5.9, 5.8, 6.2, 5.8)
    arrow(4.4, 4.6, 1.6, 4.1)
    arrow(4.4, 4.6, 4.6, 4.1)
    arrow(7.7, 4.6, 1.9, 4.1)
    arrow(7.7, 4.6, 5.0, 4.1)
    arrow(7.7, 4.6, 11.2, 4.1)
    arrow(11.2, 4.6, 8.4, 4.1)
    ax.annotate('', xy=(8.1, 1.2), xytext=(1.6, 1.2), arrowprops=dict(arrowstyle='->', color='#52514e', lw=1.1,
                connectionstyle='arc,angleA=-90,angleB=-90,armA=18,armB=18,rad=6'))
    ax.text(4.85, 0.78, 'CoT writes the ordinal as text', ha='center', fontsize=7.5, color='#52514e')
    ax.text(0.1, 7.25, 'Shared substrate (blue) and task-specific readouts (orange: ordinal; green: direct value; yellow: CoT / text arithmetic)',
            fontsize=9, color=INK)
    ax.text(0.1, 0.25, 'Arrows: information flow established by patching / ablation in the cited experiments. '
            'T1 → T3: the CoT writes the ordinal; T3 then uses the text route and bypasses the T2 digit readout.', fontsize=8, color='#52514e')
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def fig_examples(run: Path, path: Path) -> None:
    rs = read_jsonl(run / 'answer.jsonl')
    if not rs:
        return
    r0 = next(r for r in rs if r['condition'] == 'tplB')
    it = next(i for i in cm.build_items('point', 64) if i.id == r0['item'])
    b = cm.pair_item(it)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1, 1, 1.4]})
    for ax, item, name in ((axes[0], it, 'image A'), (axes[1], b, 'image B')):
        ax.imshow(item.render()[0])
        ax.set_title(f'{name}: red figure at row {item.target[0] + 1}, column {item.target[1] + 1}', fontsize=9)
        ax.axis('off')
    axes[2].axis('off')
    A, B = r0['truthA'], r0['truthB']
    txt = ('Prompt (think): "One figure is drawn in red. First describe step by step how you find this figure\n'
           'and estimate the center of its bounding box. Then give the final answer ..."\n\n'
           'Assistant prefix, condition tplB (image A, indices of B):\n"' +
           ov.template_prefix(10, 10, B['row'], B['col']).replace('Scanning', '\nScanning') + '"\n'
           '+ cue "Final answer: [{\\"point_2d\\": ["\n\n'
           f'image-implied centre: ({ov.cell_centre(A["col"], 10):.0f}, {ov.cell_centre(A["row"], 10):.0f})\n'
           f'text-implied centre:  ({ov.cell_centre(B["col"], 10):.0f}, {ov.cell_centre(B["row"], 10):.0f})\n'
           f'model answer: {r0["xy"]}')
    axes[2].text(0, 0.95, txt, fontsize=8.2, va='top', family='monospace', wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--figures', type=Path, required=True)
    args = ap.parse_args()
    work = args.run_dir.parents[1]
    args.figures.mkdir(parents=True, exist_ok=True)
    vs = value_structure(work)
    ex = extent_reanalysis(work)
    ans = answer_analysis(args.run_dir)
    cont = cont_analysis(args.run_dir)
    gate = gate_analysis(args.run_dir)
    rc = refcoco_analysis(args.run_dir, work)
    off = offsets_by_condition(args.run_dir, vs)
    summary = {'offsets': off, 'value_structure': {k: ({kk: vv for kk, vv in v.items() if not kk.endswith('offsets')} if isinstance(v, dict) else v)
                                   for k, v in vs.items()},
               'extent_reanalysis': ex, 'answer': ans, 'cont': cont, 'gate': {k: v for k, v in gate.items() if k != 'pairs'},
               'refcoco': rc}
    (args.run_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    fig_extent(ex, args.figures / 'extent.png')
    fig_pipeline(args.figures / 'pipeline.png')
    fig_offsets(off, args.figures / 'offsets.png')
    if rc and rc.get('forced'):
        fig_refcoco(rc, args.figures / 'refcoco.png')
    if ans['conditions']:
        fig_follow(ans, args.figures / 'follow.png')
        fig_mask_digits(ans, rc, args.figures / 'mask_digits.png')
        fig_examples(args.run_dir, args.figures / 'examples.png')
    if gate['per_s']:
        fig_gate(gate, args.figures / 'gate.png')
    print(json.dumps({'answer': {c: {'hit': v['hit_image_cell']['p'], 'att': v['img_att_l20']} for c, v in ans['conditions'].items()}}, indent=0))


if __name__ == '__main__':
    main()
