"""Summaries and figures for the supplementary conditions of experiment 1.3.

Reads the cases / results / score files written by ``decomposition_conditions.py`` and the
formal run (read only) for matched reference numbers; writes analysis/summary.json,
analysis/tables.md and figures under analysis/figures/.  Location tasks use the diagnostic
(content) scores, as in the formal results document; strict scores are tabulated alongside.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.behavior.decomposition_analysis import INK, MUTED, SERIES, _style, auroc, wilson
from analysis_tools.behavior.decomposition_conditions import LAYOUTS, RINGS, RING_CANVAS, T3_CANDIDATES
from analysis_tools.behavior.decomposition_supplement import mcnemar_exact

KINDS = ('gap', 'color', 'rotate')
MATCHED = {'T4': (('gap', 4), ('rotate', 3)), 'T5': (('gap', 4), ('rotate', 3)), 'T6': (('color', 4),)}


# ----------------------------------------------------------------------------- loading

def load(output: Path, part: str, kind: str = 'diagnostic') -> list[dict]:
    path = output / f'scores_{part}_{kind}.jsonl'
    if not path.exists():
        return []
    cases = {c['id']: c for c in read_jsonl(output / f'cases_{part}.jsonl')}
    return [{**cases[s['id']], 'score': s} for s in read_jsonl(path)]


def load_formal(formal: Path, kind: str = 'diagnostic') -> list[dict]:
    cases = {c['id']: c for c in read_jsonl(formal / 'cases.jsonl') if c['stage'] == 'main'}
    return [{**cases[s['id']], 'score': s} for s in read_jsonl(formal / f'scores_main_{kind}.jsonl')]


def rate(rows, key='correct') -> dict:
    n, k = len(rows), sum(bool(r['score'].get(key)) for r in rows)
    return {'n': n, 'k': k, 'p': k / n if n else float('nan'), 'ci': wilson(k, n)}


def fmt(stat: dict, ci: bool = True) -> str:
    if not stat['n']:
        return '–'
    text = f"{stat['p']:.2f}"
    return text + (f" [{stat['ci'][0]:.2f}, {stat['ci'][1]:.2f}]" if ci else '') + f" (n={stat['n']})"


# ----------------------------------------------------------------------------- T3

def t3_stats(rows: list[dict], s_key: str = 's_same') -> dict:
    """Accuracy, AUROC of the 'same' score, rate of 'same' decisions and raw Yes rate."""
    n = len(rows)
    if not n:
        return {'n': 0}
    pos = [r['score'][s_key] for r in rows if r['truth']['same']]
    neg = [r['score'][s_key] for r in rows if not r['truth']['same']]
    k = sum(r['score']['correct'] for r in rows)
    return {'n': n, 'acc': k / n, 'acc_ci': wilson(k, n), 'auroc': auroc(pos, neg),
            'same_rate': sum(r['score'][s_key] > 0 for r in rows) / n,
            'yes_rate': sum(r['score']['s'] > 0 for r in rows) / n,
            'base_rate': sum(r['truth']['same'] for r in rows) / n}


def formal_t3_rows(formal_rows: list[dict]) -> dict:
    """Formal primary-wording T3 rows keyed by case id, with s_same = s (Yes means same)."""
    out = {}
    for r in formal_rows:
        if r['task'] == 'T3' and r['wording'] == 'primary':
            same = r['truth']['answer_yes']
            out[r['id']] = {**r, 'truth': {**r['truth'], 'same': same},
                            'score': {**r['score'], 's_same': r['score']['s'], 'correct': (r['score']['s'] > 0) == same}}
    return out


def t3dev_summary(rows: list[dict]) -> dict:
    out = {}
    for cand in T3_CANDIDATES:
        rs = [r for r in rows if r['candidate'] == cand]
        if not rs:
            continue
        cells = {}
        for mode in ('box', 'rowcol'):
            for kind in KINDS:
                cells[f'{mode}/{kind}'] = t3_stats([r for r in rs if r['mode'] == mode and r['kind'] == kind])
        per_grid = {f'{mode}/g{g}': t3_stats([r for r in rs if r['mode'] == mode and r['grid'] == g])
                    for mode in ('box', 'rowcol') for g in (3, 6, 10, 20)}
        out[cand] = {'pooled': t3_stats(rs), 'box': t3_stats([r for r in rs if r['mode'] == 'box']),
                     'rowcol': t3_stats([r for r in rs if r['mode'] == 'rowcol']), 'cells': cells, 'grids': per_grid,
                     'bias': float(np.mean([abs(c['same_rate'] - c['base_rate']) for c in cells.values()])),
                     'mean_auroc': float(np.mean([c['auroc'] for c in cells.values()])),
                     'yes_means_same': T3_CANDIDATES[cand][1]}
    if 'C0_original' in out:
        ref = out['C0_original']['mean_auroc']
        eligible = {c: v for c, v in out.items() if v['mean_auroc'] >= ref - 0.02}
        out['_selection'] = {'rule': 'minimum mean |same_rate - base_rate| over mode x kind cells among candidates '
                                     'whose mean cell AUROC is at least that of C0_original minus 0.02',
                             'reference_auroc': ref, 'eligible': sorted(eligible),
                             'selected': min(eligible, key=lambda c: eligible[c]['bias']) if eligible else None}
    return out


def bootstrap_auroc_diff(a_rows: list[dict], b_rows: dict, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """Paired bootstrap (resampling images) of AUROC(revised) - AUROC(original); returns [lo, hi]."""
    by_image = defaultdict(list)
    for r in a_rows:
        by_image[r['record_id']].append((r, b_rows[r['formal_id']]))
    keys = list(by_image)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = [p for k in rng.choice(len(keys), len(keys)) for p in by_image[keys[k]]]
        a = auroc([x['score']['s_same'] for x, _ in pick if x['truth']['same']],
                  [x['score']['s_same'] for x, _ in pick if not x['truth']['same']])
        b = auroc([y['score']['s_same'] for _, y in pick if y['truth']['same']],
                  [y['score']['s_same'] for _, y in pick if not y['truth']['same']])
        diffs.append(a - b)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def t3full_summary(rows: list[dict], formal_t3: dict, dev_reps: int, dev_levels: list[int]) -> dict:
    out = {}
    for label, subset in (('all', rows), ('held_out', [r for r in rows if not (r['rep'] < dev_reps
                                                                               and r['level'] in dev_levels)])):
        block = {}
        for mode in ('box', 'rowcol'):
            for kind in KINDS + ('all',):
                for grid in (3, 6, 10, 20, 'all'):
                    rs = [r for r in subset if r['mode'] == mode and (kind == 'all' or r['kind'] == kind)
                          and (grid == 'all' or r['grid'] == grid)]
                    if not rs:
                        continue
                    orig = [formal_t3[r['formal_id']] for r in rs]
                    b = sum(o['score']['correct'] and not r['score']['correct'] for r, o in zip(rs, orig))
                    c = sum(r['score']['correct'] and not o['score']['correct'] for r, o in zip(rs, orig))
                    entry = {'revised': t3_stats(rs), 'original': t3_stats(orig), 'orig_only': b, 'rev_only': c,
                             'mcnemar_p': mcnemar_exact(b, c)}
                    if grid == 'all' and label == 'all':
                        entry['auroc_diff_ci'] = bootstrap_auroc_diff(rs, formal_t3)
                    block[f'{mode}/{kind}/{grid}'] = entry
        out[label] = block
    return out


# ----------------------------------------------------------------------------- coordinate convention

def pixel_reading(row: dict) -> np.ndarray | None:
    """The answered coordinates read as absolute pixels instead of 0-1000 normalised values."""
    if not row['score'].get('valid') or 'pred_center' not in row['score']:
        return None
    w, h = row['truth']['canvas']
    return np.array(row['score']['pred_center']) * 1000 / np.array([w, h])


def convention(row: dict) -> str | None:
    """'norm' if the normalised reading lies in the target cell, 'pixel' if only the pixel reading does,
    'other' otherwise; None for invalid answers.  A pixel reading in the target cell by chance has
    probability about (cell area / canvas area), below 0.01 for the canvases used here."""
    raw = pixel_reading(row)
    if raw is None:
        return None
    x0, y0, x1, y1 = row['truth']['target_cell_box']
    inside = lambda p: x0 <= p[0] < x1 and y0 <= p[1] < y1
    if inside(row['score']['pred_center']):
        return 'norm'
    return 'pixel' if inside(raw) else 'other'


def convention_block(rows: list[dict]) -> dict:
    pt = [r for r in rows if r['format'] == 'point']
    conv = Counter(convention(r) for r in pt)
    n = len(pt)
    return {'n': n, 'norm_hit': conv['norm'], 'pixel_hit': conv['pixel'], 'other': conv['other'],
            'invalid': conv[None], 'either_hit': rate([{'score': {'correct': convention(r) in ('norm', 'pixel')}}
                                                      for r in pt])}


# ----------------------------------------------------------------------------- grid layouts

def slope(true: np.ndarray, pred: np.ndarray) -> float:
    if len(true) < 3 or np.ptp(true) == 0:
        return float('nan')
    return float(np.polyfit(true, pred, 1)[0])


def locate_block(rows: list[dict]) -> dict:
    """Row/column and point metrics of one task in one layout."""
    rc = [r for r in rows if r['format'] == 'row_column']
    pt = [r for r in rows if r['format'] == 'point']
    wrong = [r for r in rc if r['score'].get('valid') and not r['score']['correct']]
    col_only = [r for r in wrong if r['score']['d_row'] == 0]
    row_only = [r for r in wrong if r['score']['d_col'] == 0]
    d = [r['score']['cell_distance'] for r in pt if r['score'].get('valid')]
    valid_pt = [r for r in pt if r['score'].get('valid')]
    tx = np.array([r['truth']['target_center'][0] / r['truth']['canvas'][0] for r in valid_pt])
    px = np.array([r['score']['pred_center'][0] / r['truth']['canvas'][0] for r in valid_pt])
    ty = np.array([r['truth']['target_center'][1] / r['truth']['canvas'][1] for r in valid_pt])
    py = np.array([r['score']['pred_center'][1] / r['truth']['canvas'][1] for r in valid_pt])
    return {'rowcol': rate(rc), 'rowcol_valid': rate(rc, 'valid'), 'point_hit': rate(pt), 'n_wrong_rc': len(wrong),
            'convention': convention_block(pt),
            'col_only': len(col_only), 'row_only': len(row_only),
            'col_plus1': sum(r['score']['d_col'] == 1 for r in col_only),
            'row_plus1': sum(r['score']['d_row'] == 1 for r in row_only),
            'mean_d_col_wrong': float(np.mean([r['score']['d_col'] for r in wrong])) if wrong else float('nan'),
            'median_cell_distance': float(np.median(d)) if d else float('nan'),
            'frac_within_1cell': float(np.mean(np.array(d) < 1)) if d else float('nan'),
            'slope_x': slope(tx, px), 'slope_y': slope(ty, py),
            'mean_dx_cells': float(np.mean([(r['score']['pred_center'][0] - r['truth']['target_center'][0]) / r['truth']['cell']
                                            for r in valid_pt])) if valid_pt else float('nan'),
            'mean_dy_cells': float(np.mean([(r['score']['pred_center'][1] - r['truth']['target_center'][1]) / r['truth']['cell']
                                            for r in valid_pt])) if valid_pt else float('nan')}


def count_block(rows: list[dict]) -> dict:
    valid = [r for r in rows if r['score'].get('valid')]
    return {'n': len(rows), 'rows_correct': sum(r['score'].get('rows_correct', False) for r in rows),
            'cols_correct': sum(r['score'].get('cols_correct', False) for r in rows),
            'pred_rows': dict(Counter(r['score']['pred_rows'] for r in valid)),
            'pred_cols': dict(Counter(r['score']['pred_cols'] for r in valid))}


def layouts_summary(rows: list[dict], selected: str | None) -> dict:
    out = {}
    for name, (nr, nc, cell) in LAYOUTS.items():
        rs = [r for r in rows if r['layout'] == name]
        if not rs:
            continue
        canvas = next((r['truth']['canvas'] for r in rs if 'canvas' in r['truth']), None)
        if canvas is None:
            raise ValueError(f'Layout {name} has T3 results but no location results yet.')
        entry = {'rows': nr, 'cols': nc, 'cell': cell, 'canvas': canvas,
                 'T0': count_block([r for r in rs if r['task'] == 'T0'])}
        for task in ('T4', 'T5', 'T6'):
            entry[task] = locate_block([r for r in rs if r['task'] == task])
        for cand in ('C0_original', selected):
            if cand:
                t3 = [r for r in rs if r['task'] == 'T3' and r['candidate'] == cand]
                if t3:
                    entry[f'T3_{cand}'] = t3_stats(t3)
                    entry[f'T3_{cand}_by_kind'] = {k: t3_stats([r for r in t3 if r['kind'] == k]) for k in ('gap', 'rotate')}
        out[name] = entry
    return out


def formal_reference(formal_rows: list[dict]) -> dict:
    """Matched formal numbers: T4/T5 at gap 16 px and rotation 45 deg, T6 red, T0, square grids at cell 64."""
    out = {}
    for g in (3, 6, 10, 20):
        rs = [r for r in formal_rows if r['grid'] == g and r['wording'] == 'primary']
        entry = {'rows': g, 'cols': g, 'cell': 64, 'canvas': [max(256, 64 * g)] * 2,
                 'T0': count_block([r for r in rs if r['task'] == 'T0'])}
        for task, changes in MATCHED.items():
            entry[task] = locate_block([r for r in rs if r['task'] == task and (r['kind'], r['level']) in changes])
        out[f'formal_g{g}'] = entry
    return out


def ordinal_accuracy(rows: list[dict], axis: str) -> dict:
    """Accuracy of row/column answers (T4-T6) by the true row or column index."""
    i = 0 if axis == 'row' else 1
    acc = defaultdict(list)
    for r in rows:
        if r['format'] == 'row_column' and r['task'] in ('T4', 'T5', 'T6'):
            ok = r['score'].get('valid') and r['score']['pred_rc'][i] == r['truth']['target_rc'][i]
            acc[r['truth']['target_rc'][i]].append(bool(ok))
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(acc.items())}


# ----------------------------------------------------------------------------- ring

def ring_summary(rows: list[dict]) -> dict:
    out = {}
    for name, (n, radius) in RINGS.items():
        rs = [r for r in rows if r['layout'] == name]
        if not rs:
            continue
        entry = {'n': n, 'radius': radius, 'radius_norm': 1000 * radius / RING_CANVAS}
        t0 = [r for r in rs if r['task'] == 'T0']
        if t0:
            entry['T0'] = {**rate(t0), 'pred': dict(Counter(r['score'].get('pred_count') for r in t0))}
        for task in ('T5', 'T6'):
            pt = [r for r in rs if r['task'] == task and r['format'] == 'point']
            ix = [r for r in rs if r['task'] == task and r['format'] == 'index']
            if not pt:
                continue
            vp = [r for r in pt if r['score'].get('valid')]
            vi = [r for r in ix if r['score'].get('valid')]
            # Radial and angular errors exclude answers identified as absolute-pixel coordinates.
            vn = [r for r in vp if convention(r) != 'pixel']
            ratio = np.array([r['score']['radial_ratio'] for r in vn])
            steps = np.array([r['score']['d_angle_steps'] for r in vn])
            entry[task] = {
                'convention': convention_block(pt), 'n_radial': len(vn),
                'point_hit': rate(pt), 'point_nearest_exact': rate(
                    [{'score': {'correct': r['score'].get('valid', False) and r['score']['d_index_nearest'] == 0}}
                     for r in pt]),
                'mean_dx_cells': float(np.mean([(r['score']['pred_center'][0] - r['truth']['target_center'][0])
                                                / r['truth']['cell'] for r in vn])) if vn else float('nan'),
                'mean_dy_cells': float(np.mean([(r['score']['pred_center'][1] - r['truth']['target_center'][1])
                                                / r['truth']['cell'] for r in vn])) if vn else float('nan'),
                'index_exact': rate(ix), 'index_valid': rate(ix, 'valid'),
                'median_cell_distance': float(np.median([r['score']['cell_distance'] for r in vp])) if vp else float('nan'),
                'radial_ratio_median': float(np.median(ratio)) if len(ratio) else float('nan'),
                'radial_ratio_iqr': [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))] if len(ratio) else None,
                'angle_steps_median_abs': float(np.median(np.abs(steps))) if len(steps) else float('nan'),
                'angle_steps_mean': float(np.mean(steps)) if len(steps) else float('nan'),
                'index_offsets': dict(Counter(r['score']['d_index'] for r in vi)),
                'nearest_offsets': dict(Counter(r['score']['d_index_nearest'] for r in vp)),
            }
            # Paired: point answer near the right instance vs index answer exact, same image.
            by_img = {r['image_id']: r for r in ix}
            both = [(p, by_img[p['image_id']]) for p in pt if p['image_id'] in by_img]
            entry[task]['pair'] = {
                'point_hit_index_wrong': sum(p['score']['correct'] and not q['score']['correct'] for p, q in both),
                'point_miss_index_right': sum(q['score']['correct'] and not p['score']['correct'] for p, q in both),
                'n': len(both)}
        out[name] = entry
    return out


def formal_radial(formal_rows: list[dict]) -> list[dict]:
    """Per-answer radial data of formal T6 point answers: true / predicted distance from the canvas centre."""
    out = []
    for r in formal_rows:
        if (r['task'] == 'T6' and r['format'] == 'point' and r['wording'] == 'primary' and r['score'].get('valid')
                and convention(r) != 'pixel'):
            w, h = r['truth']['canvas']
            c = np.array([w / 2, h / 2])
            t, p = np.array(r['truth']['target_center']), np.array(r['score']['pred_center'])
            out.append({'grid': r['grid'], 'true_norm': 1000 * float(np.linalg.norm(t - c)) / w,
                        'pred_norm': 1000 * float(np.linalg.norm(p - c)) / w})
    return out


# ----------------------------------------------------------------------------- figures

def fig_t3dev(summary: dict, out: Path) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    cands = [c for c in summary if not c.startswith('_')]
    for ax, mode in zip(axes, ('box', 'rowcol')):
        for i, cand in enumerate(cands):
            st = summary[cand][mode]
            ax.scatter(st['same_rate'], st['auroc'], color=SERIES[i % len(SERIES)], s=40, zorder=3,
                       marker='o' if summary[cand]['yes_means_same'] else 's')
            ax.annotate(cand.split('_', 1)[0], (st['same_rate'], st['auroc']), xytext=(4, 3),
                        textcoords='offset points', fontsize=8, color=INK)
        ax.axvline(0.5, color=MUTED, ls='--', lw=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Rate of 'same' decisions (base rate 0.5)")
        ax.set_title({'box': 'Blue-box designation', 'rowcol': 'Row/column designation'}[mode])
    axes[0].set_ylabel('AUROC of the same-score')
    fig.suptitle('T3 prompt candidates on the development subset (circle: Yes = same; square: Yes = different)',
                 fontsize=10)
    fig.savefig(out / 't3dev_candidates.png')
    plt.close(fig)


def fig_t3full(rows: list[dict], formal_t3: dict, selected: str, out: Path) -> None:
    plt = _style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    bins = np.linspace(-12, 12, 49)
    for i, mode in enumerate(('box', 'rowcol')):
        for j, kind in enumerate(KINDS):
            ax = axes[i, j]
            rs = [r for r in rows if r['mode'] == mode and r['kind'] == kind]
            for label, src, ls in (('original', [formal_t3[r['formal_id']] for r in rs], ':'), (selected, rs, '-')):
                for same, color in ((True, SERIES[0]), (False, SERIES[1])):
                    v = np.clip([r['score']['s_same'] for r in src if r['truth']['same'] == same], -12, 12)
                    ax.hist(v, bins=bins, histtype='step', color=color, ls=ls, lw=1.4,
                            label=f"{label}: {'normal' if same else 'target'}")
            ax.axvline(0, color=MUTED, lw=0.8)
            ax.set_title(f'{mode} / {kind}')
            if i == 1:
                ax.set_xlabel('same-score (>0 decides "same")')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('T3 same-score distributions: original (dotted) vs revised prompt (solid)', fontsize=10)
    fig.savefig(out / 't3full_scores.png')
    plt.close(fig)


def fig_t3full_bars(summary: dict, selected: str, out: Path) -> None:
    plt = _style()
    block = summary['all']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    labels, keys = [], []
    for mode in ('box', 'rowcol'):
        for kind in KINDS:
            labels.append(f'{mode}\n{kind}')
            keys.append(f'{mode}/{kind}/all')
    x = np.arange(len(keys))
    for ax, metric, title in zip(axes, ('acc', 'auroc', 'same_rate'), ('Accuracy', 'AUROC', "Rate of 'same' decisions")):
        for k, (src, color) in enumerate((('original', SERIES[1]), ('revised', SERIES[0]))):
            ax.bar(x + (k - 0.5) * 0.38, [block[key][src][metric] for key in keys], 0.38, color=color,
                   label='original' if src == 'original' else selected)
        ax.axhline(0.5, color=MUTED, ls='--', lw=0.8)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.savefig(out / 't3full_bars.png')
    plt.close(fig)


def fig_error_maps(rows: list[dict], names: list[str], task: str, out: Path, fname: str) -> None:
    plt = _style()
    k = len(names)
    cols = min(4, k)
    nrow = math.ceil(k / cols)
    fig, axes = plt.subplots(nrow, cols, figsize=(3.4 * cols, 3.4 * nrow), squeeze=False)
    for ax, name in zip(axes.flat, names):
        rs = [r for r in rows if r['layout'] == name and r['task'] == task and r['format'] == 'point'
              and r['score'].get('valid')]
        if not rs:
            ax.axis('off')
            continue
        t = rs[0]['truth']
        w, h = t['canvas']
        if 'grid_origin' in t:
            ox, oy = t['grid_origin']
            for i in range(t['cols'] + 1):
                ax.plot([ox + i * t['cell']] * 2, [oy, oy + t['rows'] * t['cell']], color='#e4e3df', lw=0.5)
            for i in range(t['rows'] + 1):
                ax.plot([ox, ox + t['cols'] * t['cell']], [oy + i * t['cell']] * 2, color='#e4e3df', lw=0.5)
        else:
            for x0, y0, x1, y1 in t['instance_cell_boxes']:
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='#d9d8d4', lw=0.6))
        for r in rs:
            tx, ty = r['truth']['target_center']
            px, py = r['score']['pred_center']
            conv = convention(r)
            color = SERIES[2] if conv == 'norm' else SERIES[0] if conv == 'pixel' else SERIES[1]
            ax.annotate('', xy=(px, py), xytext=(tx, ty),
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.7, alpha=0.7))
            ax.plot(tx, ty, '.', color=INK, ms=2)
        hit = rate([r for r in rows if r['layout'] == name and r['task'] == task and r['format'] == 'point'])
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{name} ({w}x{h}), hit {hit["p"]:.2f}', fontsize=9)
    for ax in list(axes.flat)[k:]:
        ax.axis('off')
    fig.suptitle(f'{task} point answers: true centre (dot) to prediction (arrow; green = hit, blue = hit only when read as pixels, orange = miss)', fontsize=10)
    fig.savefig(out / fname)
    plt.close(fig)


def fig_examples(output: Path, out: Path) -> None:
    """Input examples: the first T6 image (red target) of every grid layout and ring."""
    from PIL import Image
    plt = _style()
    picks = [(name, f'images/{name}/{name}_color_L4_r000.png') for name in LAYOUTS] + \
            [(name, f'images/ring/ring_{name}_color_L4_r000.png') for name in RINGS]
    picks = [(n, p) for n, p in picks if (output / p).exists()]
    cols = 7
    fig, axes = plt.subplots(2, cols, figsize=(2.6 * cols, 6.2))
    for ax, (name, rel) in zip(axes.flat, picks):
        image = Image.open(output / rel)
        ax.imshow(image)
        ax.set_title(f'{name} ({image.width}x{image.height})', fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in list(axes.flat)[len(picks):]:
        ax.axis('off')
    fig.savefig(out / 'input_examples.png')
    plt.close(fig)


def fig_calibration(rows: list[dict], formal_rows: list[dict], out: Path) -> None:
    """Median predicted vs true normalised coordinate of T6 point answers, by true column (x) and row (y).

    Answers whose pixel reading falls in the target cell are excluded, so the curves describe
    answers given in the requested 0-1000 convention."""
    plt = _style()
    panels = [('formal_g6', 6, 6, 64), ('formal_g10', 10, 10, 64), ('formal_g20', 20, 20, 64)] + \
             [(name, *LAYOUTS[name]) for name in LAYOUTS]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.6))
    for ax, (name, nr, nc, cell) in zip(axes.flat, panels):
        if name.startswith('formal'):
            rs = [r for r in formal_rows if r['grid'] == nr and r['task'] == 'T6' and r['wording'] == 'primary']
        else:
            rs = [r for r in rows if r['layout'] == name and r['task'] == 'T6']
        rs = [r for r in rs if r['format'] == 'point' and r['score'].get('valid') and convention(r) != 'pixel']
        for axis, color, label in ((0, SERIES[1], 'x by column'), (1, SERIES[0], 'y by row')):
            by = defaultdict(list)
            for r in rs:
                size = r['truth']['canvas'][axis]
                by[1000 * r['truth']['target_center'][axis] / size].append(1000 * r['score']['pred_center'][axis] / size)
            xs = sorted(by)
            ax.plot(xs, [np.median(by[x]) for x in xs], '-o', ms=2.5, lw=1.2, color=color, label=label)
        ax.plot([0, 1000], [0, 1000], color=MUTED, ls='--', lw=0.8)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect('equal')
        ax.set_title(f'{name}: {nr}x{nc}, cell {cell} (n={len(rs)})', fontsize=8.5)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis('off')
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].set_ylabel('median predicted coordinate (0-1000)')
    axes[2, 1].set_xlabel('true coordinate of the target centre (0-1000)')
    fig.suptitle('T6 point answers: coordinate calibration per true column (x) and row (y)', fontsize=10)
    fig.savefig(out / 'calibration_T6.png')
    plt.close(fig)


def fig_ordinal(rows: list[dict], formal_rows: list[dict], out: Path) -> None:
    plt = _style()
    panels = [('formal_g20', None), ('g10x20', None), ('g20x10', None), ('g20c32', None), ('f24c32', None),
              ('f12c64', None), ('g4x8', None), ('g8x4', None)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True, gridspec_kw={'hspace': 0.45})
    for ax, (name, _) in zip(axes.flat, panels):
        if name == 'formal_g20':
            rs = [r for r in formal_rows if r['grid'] == 20 and r['wording'] == 'primary'
                  and r['task'] in MATCHED and (r['kind'], r['level']) in MATCHED[r['task']]]
            title = 'formal 20x20, cell 64'
        else:
            rs = [r for r in rows if r['layout'] == name]
            nr, nc, cell = LAYOUTS[name]
            title = f'{name}: {nr} rows x {nc} cols, cell {cell}'
        for axis, color in (('row', SERIES[0]), ('col', SERIES[1])):
            acc = ordinal_accuracy(rs, axis)
            if acc:
                xs = np.array(list(acc)) + 1
                ax.plot(xs, [v[0] for v in acc.values()], '-o', ms=3, color=color, label=axis)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('true index (1-based)')
    axes[0, 0].set_ylabel('accuracy of that coordinate')
    axes[0, 0].legend()
    fig.suptitle('Row and column answers (T4-T6, row/column format) by true row / column index', fontsize=10)
    fig.savefig(out / 'ordinal_accuracy.png')
    plt.close(fig)


def fig_col_offsets(rows: list[dict], formal_rows: list[dict], out: Path) -> None:
    plt = _style()
    names = ['formal_g20', 'g10x20', 'g20x10', 'g20c32', 'f24c32', 'f12c64']
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3), sharey=True)
    for ax, name in zip(axes, names):
        if name == 'formal_g20':
            rs = [r for r in formal_rows if r['grid'] == 20 and r['wording'] == 'primary'
                  and r['task'] in MATCHED and (r['kind'], r['level']) in MATCHED[r['task']]]
        else:
            rs = [r for r in rows if r['layout'] == name]
        rs = [r for r in rs if r['format'] == 'row_column' and r['score'].get('valid')]
        if not rs:
            continue
        for key, color, off in (('d_row', SERIES[0], -0.2), ('d_col', SERIES[1], 0.2)):
            cnt = Counter(int(np.clip(r['score'][key], -3, 3)) for r in rs)
            xs = np.arange(-3, 4)
            ax.bar(xs + off, [cnt.get(x, 0) / len(rs) for x in xs], 0.4, color=color, label=key)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('offset (clipped to ±3)')
    axes[0].set_ylabel('share of answers')
    axes[0].legend()
    fig.savefig(out / 'rowcol_offsets.png')
    plt.close(fig)


def fig_ring_radial(ring_rows: list[dict], formal_rad: list[dict], out: Path) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for g, color in zip((6, 10, 20), SERIES[2:5]):
        d = [x for x in formal_rad if x['grid'] == g and x['true_norm'] > 20]
        bins = np.linspace(0, 700, 15)
        idx = np.digitize([x['true_norm'] for x in d], bins)
        xs, ys = [], []
        for b in sorted(set(idx)):
            sel = [x['pred_norm'] / x['true_norm'] for x, i in zip(d, idx) if i == b]
            if len(sel) >= 3:
                xs.append(np.mean([x['true_norm'] for x, i in zip(d, idx) if i == b]))
                ys.append(np.median(sel))
        ax.plot(xs, ys, '-o', ms=3, color=color, label=f'formal grid {g}x{g} (T6)')
    for task, marker in (('T6', 'D'), ('T5', 'v')):
        for name, (n, radius) in RINGS.items():
            rs = [r for r in ring_rows if r['layout'] == name and r['task'] == task and r['format'] == 'point'
                  and r['score'].get('valid') and convention(r) != 'pixel']
            if rs:
                ratio = [r['score']['radial_ratio'] for r in rs]
                x = 1000 * radius / RING_CANVAS
                ax.errorbar(x + (5 if task == 'T5' else 0), np.median(ratio),
                            yerr=[[np.median(ratio) - np.percentile(ratio, 25)], [np.percentile(ratio, 75) - np.median(ratio)]],
                            fmt=marker, color=SERIES[0] if task == 'T6' else SERIES[1], ms=5, capsize=2)
                ax.annotate(f'{name}', (x, np.median(ratio)), xytext=(6, -10 if task == 'T5' else 4),
                            textcoords='offset points', fontsize=7)
    ax.axhline(1, color=MUTED, ls='--', lw=0.8)
    ax.set_xlabel('true distance from canvas centre (0-1000 units of canvas width)')
    ax.set_ylabel('median predicted / true distance')
    ax.set_title('Radial ratio, pixel-convention answers excluded (ring: T6 diamond, T5 triangle; bars = IQR)', fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    names = [k for k in RINGS if k.startswith('n') and '_' not in k]
    for i, name in enumerate(names):
        rs = [r for r in ring_rows if r['layout'] == name and r['task'] == 'T6' and r['format'] == 'point'
              and r['score'].get('valid') and convention(r) != 'pixel']
        steps = np.clip([r['score']['d_angle_steps'] for r in rs], -3, 3)
        ax.hist(steps, bins=np.linspace(-3, 3, 49), histtype='step', lw=1.4, color=SERIES[i], label=name)
    ax.set_xlabel('angular error in units of the instance spacing (360/N deg), clockwise > 0')
    ax.set_title('T6 point answers: angular error', fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(out / 'ring_radial_angular.png')
    plt.close(fig)


def fig_ring_index(summary: dict, out: Path) -> None:
    plt = _style()
    names = [k for k in summary if k in RINGS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, task in zip(axes, ('T6', 'T5')):
        x = np.arange(len(names))
        for k, (metric, label, color) in enumerate((('point_hit', 'point: hit target cell', SERIES[0]),
                                                    ('point_nearest_exact', 'point: nearest instance = target', SERIES[2]),
                                                    ('index_exact', 'index: exact', SERIES[1]))):
            vals = [summary[n][task][metric] if task in summary[n] else None for n in names]
            ps = [v['p'] if v else np.nan for v in vals]
            err = np.array([[v['p'] - v['ci'][0], v['ci'][1] - v['p']] if v else [0, 0] for v in vals]).T
            ax.bar(x + (k - 1) * 0.27, ps, 0.27, yerr=err, color=color, label=label, capsize=2)
        ax.set_xticks(x, names, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f'{task} on the ring', fontsize=9)
    axes[0].legend(fontsize=7)
    fig.savefig(out / 'ring_accuracy.png')
    plt.close(fig)


def fig_ring_index_confusion(rows: list[dict], out: Path) -> None:
    """Predicted vs true clockwise index (1-based) of T6 index answers, one panel per ring."""
    plt = _style()
    names = ['n8', 'n12', 'n16', 'n24']
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, name in zip(axes, names):
        n = RINGS[name][0]
        mat = np.zeros((n, n))
        for r in rows:
            if r['layout'] == name and r['task'] == 'T6' and r['format'] == 'index' and r['score'].get('valid'):
                mat[r['score']['pred_index'], r['truth']['target_index']] += 1
        mat = mat / np.maximum(mat.sum(0, keepdims=True), 1)
        ax.imshow(mat, origin='lower', cmap='Blues', vmin=0, vmax=1, extent=(0.5, n + 0.5, 0.5, n + 0.5))
        ax.plot([0.5, n + 0.5], [0.5, n + 0.5], color=SERIES[1], lw=0.6, ls='--')
        ax.plot([0.5, n + 0.5], [n + 1.5, 1.5], color=SERIES[2], lw=0.6, ls=':')
        ax.set_title(f'{name}', fontsize=9)
        ax.set_xlabel('true index')
    axes[0].set_ylabel('answered index')
    fig.suptitle('T6 index answers on the ring, column-normalised (dashed: correct; dotted: counter-clockwise numbering)',
                 fontsize=10)
    fig.savefig(out / 'ring_index_confusion.png')
    plt.close(fig)


def fig_ring_offsets(rows: list[dict], out: Path) -> None:
    plt = _style()
    names = ['n8', 'n12', 'n16', 'n24']
    fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
    for ax, name in zip(axes, names):
        for key, fmt_, color, off in (('d_index', 'index', SERIES[1], 0.2), ('d_index_nearest', 'point', SERIES[0], -0.2)):
            rs = [r for r in rows if r['layout'] == name and r['task'] == 'T6' and r['format'] == fmt_
                  and r['score'].get('valid')]
            if not rs:
                continue
            cnt = Counter(int(np.clip(r['score'][key], -4, 4)) for r in rs)
            xs = np.arange(-4, 5)
            ax.bar(xs + off, [cnt.get(x, 0) / len(rs) for x in xs], 0.4, color=color, label=fmt_)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('index offset (clockwise > 0, clipped)')
    axes[0].set_ylabel('share of valid answers')
    axes[0].legend()
    fig.suptitle('T6 on the ring: index offsets of index answers and of the instance nearest to the point answer', fontsize=10)
    fig.savefig(out / 'ring_offsets.png')
    plt.close(fig)


# ----------------------------------------------------------------------------- tables

def write_tables(summary: dict, path: Path) -> None:
    lines = ['# Supplementary conditions: tables', '']
    if 't3dev' in summary:
        lines += ['## T3 development subset', '',
                  '| candidate | Yes=same | pooled acc | pooled AUROC | same rate | box AUROC | box same | rowcol AUROC | rowcol same | mean cell AUROC | mean bias |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for c, v in summary['t3dev'].items():
            if c.startswith('_'):
                continue
            p, b, r = v['pooled'], v['box'], v['rowcol']
            lines.append(f"| {c} | {v['yes_means_same']} | {p['acc']:.2f} | {p['auroc']:.2f} | {p['same_rate']:.2f} | "
                         f"{b['auroc']:.2f} | {b['same_rate']:.2f} | {r['auroc']:.2f} | {r['same_rate']:.2f} | "
                         f"{v['mean_auroc']:.2f} | {v['bias']:.2f} |")
        lines += ['', f"Selection: {json.dumps(summary['t3dev'].get('_selection'))}", '']
        lines += ['### Per mode x kind', '', '| candidate | cell | n | acc | AUROC | same rate |', '| --- | --- | --- | --- | --- | --- |']
        for c, v in summary['t3dev'].items():
            if c.startswith('_'):
                continue
            for cell, st in list(v['cells'].items()) + list(v['grids'].items()):
                lines.append(f"| {c} | {cell} | {st['n']} | {st['acc']:.2f} | {st['auroc']:.2f} | {st['same_rate']:.2f} |")
        lines.append('')
    if 't3full' in summary:
        for label, block in summary['t3full'].items():
            lines += [f'## T3 full set ({label})', '',
                      '| mode/kind/grid | n | acc orig | acc rev | AUROC orig | AUROC rev | same orig | same rev | b | c | McNemar p | AUROC diff 95% CI |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
            for key, e in block.items():
                o, r = e['original'], e['revised']
                ci = e.get('auroc_diff_ci')
                lines.append(f"| {key} | {r['n']} | {o['acc']:.2f} | {r['acc']:.2f} | {o['auroc']:.2f} | {r['auroc']:.2f} | "
                             f"{o['same_rate']:.2f} | {r['same_rate']:.2f} | {e['orig_only']} | {e['rev_only']} | "
                             f"{e['mcnemar_p']:.2g} | {'' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'} |")
            lines.append('')
    if 'layouts' in summary:
        lines += ['## Grid layouts (diagnostic scores)', '',
                  '| layout | rows x cols | cell | canvas | T0 rows | T0 cols | T0 predicted cols | T6 rc | T6 point | T6 d med | T5 rc | T5 point | T4 rc | T4 point |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for name, e in {**summary.get('formal_reference', {}), **summary['layouts']}.items():
            t0 = e['T0']
            lines.append(f"| {name} | {e['rows']}x{e['cols']} | {e['cell']} | {e['canvas'][0]}x{e['canvas'][1]} | "
                         f"{t0['rows_correct']}/{t0['n']} | {t0['cols_correct']}/{t0['n']} | {t0['pred_cols']} | "
                         f"{fmt(e['T6']['rowcol'])} | {fmt(e['T6']['point_hit'])} | {e['T6']['median_cell_distance']:.2f} | "
                         f"{fmt(e['T5']['rowcol'])} | {fmt(e['T5']['point_hit'])} | {fmt(e['T4']['rowcol'])} | "
                         f"{fmt(e['T4']['point_hit'])} |")
        lines += ['', '### Row/column error structure (T4-T6 pooled per task)', '',
                  '| layout | task | wrong | col only | col +1 | row only | row +1 | mean d_col (wrong) | slope x | slope y | mean dx | mean dy | point pixel-reading hits | point hit either reading |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for name, e in {**summary.get('formal_reference', {}), **summary['layouts']}.items():
            for task in ('T4', 'T5', 'T6'):
                b = e[task]
                lines.append(f"| {name} | {task} | {b['n_wrong_rc']} | {b['col_only']} | {b['col_plus1']} | {b['row_only']} | "
                             f"{b['row_plus1']} | {b['mean_d_col_wrong']:.2f} | {b['slope_x']:.2f} | {b['slope_y']:.2f} | "
                             f"{b['mean_dx_cells']:.2f} | {b['mean_dy_cells']:.2f} | {b['convention']['pixel_hit']} | "
                             f"{fmt(b['convention']['either_hit'])} |")
        lines += ['', '### T3 in the layouts (row/column designation)', '',
                  '| layout | prompt | n | acc | AUROC | same rate |', '| --- | --- | --- | --- | --- | --- |']
        for name, e in summary['layouts'].items():
            for key, st in e.items():
                if key.startswith('T3_') and not key.endswith('_by_kind'):
                    lines.append(f"| {name} | {key[3:]} | {st['n']} | {st['acc']:.2f} | {st['auroc']:.2f} | {st['same_rate']:.2f} |")
        lines.append('')
    if 'ring' in summary:
        lines += ['## Ring', '',
                  '| ring | N | R (px) | T0 count | T0 predicted | task | point hit | nearest = target | index exact | index valid | d med | radial ratio med [IQR] | |angle| med (steps) | mean angle (steps) | mean dx (cells) | mean dy (cells) | pixel-reading hits | hit under either reading |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for name, e in summary['ring'].items():
            t0 = e.get('T0')
            for task in ('T6', 'T5'):
                if task not in e:
                    continue
                b = e[task]
                iqr = b['radial_ratio_iqr']
                lines.append(f"| {name} | {e['n']} | {e['radius']} | {fmt(t0, False) if t0 else '–'} | "
                             f"{t0['pred'] if t0 else '–'} | {task} | {fmt(b['point_hit'])} | {fmt(b['point_nearest_exact'])} | "
                             f"{fmt(b['index_exact'])} | {fmt(b['index_valid'], False)} | {b['median_cell_distance']:.2f} | "
                             f"{b['radial_ratio_median']:.2f} [{iqr[0]:.2f}, {iqr[1]:.2f}] | {b['angle_steps_median_abs']:.2f} | "
                             f"{b['angle_steps_mean']:.2f} | {b['mean_dx_cells']:.2f} | {b['mean_dy_cells']:.2f} | "
                             f"{b['convention']['pixel_hit']} | {fmt(b['convention']['either_hit'])} |")
        lines += ['', '### Index offsets', '']
        for name, e in summary['ring'].items():
            for task in ('T6', 'T5'):
                if task in e:
                    lines.append(f"- {name} {task}: index answers {dict(sorted(e[task]['index_offsets'].items()))}; "
                                 f"nearest to point {dict(sorted(e[task]['nearest_offsets'].items()))}; pair {e[task]['pair']}")
        lines.append('')
    lines += ['## Batching spot-check', '', json.dumps(summary.get('spotcheck')), '']
    path.write_text('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------- batching spot-check

def spotcheck_summary(output: Path, part: str) -> dict | None:
    """Answer-level agreement between batched decoding and batch size 1 on the spot-check cases."""
    from analysis_tools.behavior.decomposition import score_case
    from analysis_tools.behavior.decomposition_conditions import score_ring
    path = output / f'spotcheck_{part}.jsonl'
    if not path.exists():
        return None
    cases = {c['id']: c for c in read_jsonl(output / f'cases_{part}.jsonl')}
    rows = []
    for r in read_jsonl(path):
        case = cases[r['id']]
        scorer = (lambda rec: score_ring(case, rec, True)) if part == 'ring' else (lambda rec: score_case(case, rec, True))
        a = scorer({'response': r['batched'], 'finish_reason': 'eos'})
        b = scorer({'response': r['single'], 'finish_reason': r['single_finish_reason']})
        answer = lambda x: (x.get('pred_rc'), x.get('pred_index'), x.get('pred_count'), x.get('pred_rows'),
                            x.get('pred_cols'), x.get('hit'))
        rows.append({'same_text': r['batched'].strip() == r['single'].strip(), 'same_correct': a['correct'] == b['correct'],
                     'same_answer': answer(a) == answer(b)})
    n = len(rows)
    return {'n': n, **{k: sum(x[k] for x in rows) for k in ('same_text', 'same_correct', 'same_answer')}}


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--formal', type=Path, required=True)
    p.add_argument('--dev-reps', type=int, default=4)
    p.add_argument('--dev-levels', type=int, nargs='+', default=[2, 4])
    args = p.parse_args()
    out = args.output / 'analysis'
    figs = out / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    formal_rows = load_formal(args.formal)
    formal_t3 = formal_t3_rows(formal_rows)
    summary: dict = {}
    selected = None
    dev = load(args.output, 't3dev')
    if dev:
        summary['t3dev'] = t3dev_summary(dev)
        selected = summary['t3dev']['_selection']['selected']
        fig_t3dev(summary['t3dev'], figs)
    full = load(args.output, 't3full')
    if full:
        selected = full[0]['candidate']
        summary['t3full'] = t3full_summary(full, formal_t3, args.dev_reps, args.dev_levels)
        fig_t3full(full, formal_t3, selected, figs)
        fig_t3full_bars(summary['t3full'], selected, figs)
    lay = load(args.output, 'layouts')
    if lay:
        summary['layouts'] = layouts_summary(lay, selected)
        summary['formal_reference'] = formal_reference(formal_rows)
        summary['ordinal'] = {name: {'row': ordinal_accuracy([r for r in lay if r['layout'] == name], 'row'),
                                     'col': ordinal_accuracy([r for r in lay if r['layout'] == name], 'col')}
                              for name in LAYOUTS}
        fig_error_maps(lay, list(LAYOUTS), 'T6', figs, 'layouts_error_map_T6.png')
        fig_error_maps(lay, list(LAYOUTS), 'T5', figs, 'layouts_error_map_T5.png')
        fig_ordinal(lay, formal_rows, figs)
        fig_calibration(lay, formal_rows, figs)
        fig_examples(args.output, figs)
        fig_col_offsets(lay, formal_rows, figs)
        strict = load(args.output, 'layouts', 'strict')
        summary['layouts_strict'] = {name: {f'{t}/{f}': rate([r for r in strict if r['layout'] == name
                                                               and r['task'] == t and r['format'] == f])
                                            for t in ('T4', 'T5', 'T6') for f in ('row_column', 'point')}
                                     for name in LAYOUTS}
    ring = load(args.output, 'ring')
    if ring:
        summary['ring'] = ring_summary(ring)
        fig_error_maps(ring, list(RINGS), 'T6', figs, 'ring_error_map_T6.png')
        fig_error_maps(ring, [k for k in RINGS if '_' not in k], 'T5', figs, 'ring_error_map_T5.png')
        fig_ring_radial(ring, formal_radial(formal_rows), figs)
        fig_ring_index(summary['ring'], figs)
        fig_ring_offsets(ring, figs)
        fig_ring_index_confusion(ring, figs)
    summary['spotcheck'] = {part: spotcheck_summary(args.output, part) for part in ('layouts', 'ring')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    write_tables(summary, out / 'tables.md')
    print('wrote', out)


if __name__ == '__main__':
    main()
