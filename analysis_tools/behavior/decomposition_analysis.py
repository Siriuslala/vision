"""Summaries and figures for experiment 1.3 (design_decomposition.md, sections 4-5).

Reads cases.jsonl, results_main.jsonl and the strict / diagnostic score files written by
``decomposition.py score``; writes analysis/summary.json, analysis/tables.md and figures.
Binary tasks use the sign of s = z_Yes - z_No (accuracy) and AUROC of s; location tasks
use the strict format scores unless a table states that it uses the coordinate diagnostic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from analysis_tools.behavior.decomposition import GRIDS, KINDS, LEVELS, magnitude_label, read_jsonl
from analysis_tools.common.stimuli import stroke_color, Change

# Ordinal blue ramp for magnitude levels and categorical slots for sub-tasks (dataviz reference palette).
LEVEL_COLORS = ('#86b6ef', '#3987e5', '#1c5cab', '#0d366b')
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, MUTED, GRID_INK = '#0b0b0b', '#52514e', '#d9d8d4'
KIND_NAMES = {'gap': 'Gap', 'color': 'Color', 'rotate': 'Rotation'}

# ----------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def auroc(pos: list[float], neg: list[float]) -> float:
    """Probability that a random positive outranks a random negative (ties count one half)."""
    if not pos or not neg:
        return float('nan')
    p, q = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > q).sum() + 0.5 * (p == q).sum()) / (p.size * q.size))


def red_delta_e() -> float:
    """CIELAB Delta E76 between black and pure red, used as the magnitude of colour level 4."""
    def lin(c):
        c = c / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in stroke_color(Change('color', 4)))
    x, y, z = (np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
               @ np.array([r, g, b])) / np.array([0.95047, 1.0, 1.08883])
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    L, a, bb = 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))
    return float(math.sqrt(L * L + a * a + bb * bb))


def numeric_magnitude(kind: str, level: int) -> float:
    value = LEVELS[kind][level - 1]
    return red_delta_e() if value == 'red' else float(value)


def fit_psychometric(x: np.ndarray, correct: np.ndarray) -> dict:
    """Maximum-likelihood fit of p(x) = 0.5 + 0.5 / (1 + exp(-(x - m) / w)) on log magnitude x.

    The 75% threshold is exp(m).  A grid search keeps the fit deterministic; thresholds outside
    the tested range are reported as bounds instead of extrapolated values.
    """
    ms = np.linspace(x.min() - 2, x.max() + 2, 241)
    ws = np.geomspace(0.05, 5, 60)
    best = (-np.inf, None, None)
    for m in ms:
        for w in ws:
            p = 0.5 + 0.5 / (1 + np.exp(-(x - m) / w))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            ll = float(np.sum(correct * np.log(p) + (1 - correct) * np.log(1 - p)))
            if ll > best[0]:
                best = (ll, m, w)
    _, m, w = best
    lo, hi = x.min(), x.max()
    status = 'below_range' if m < lo else 'above_range' if m > hi else 'in_range'
    return {'m': float(m), 'w': float(w), 'delta75': float(math.exp(m)), 'status': status,
            'range': [float(math.exp(lo)), float(math.exp(hi))]}


def auroc_threshold(levels: list[float], aucs: list[float], criterion: float = 0.75) -> dict:
    """Smallest magnitude at which AUROC reaches the criterion (log-linear interpolation).

    AUROC is independent of the Yes/No decision threshold, so this threshold separates
    discrimination from response bias; the accuracy-based fit mixes both.
    """
    x = np.log(np.asarray(levels, float))
    y = np.asarray(aucs, float)
    if y[0] >= criterion:
        return {'delta75': float(levels[0]), 'status': 'below_range'}
    for i in range(1, len(y)):
        if y[i] >= criterion:
            t = (criterion - y[i - 1]) / (y[i] - y[i - 1])
            return {'delta75': float(math.exp(x[i - 1] + t * (x[i] - x[i - 1]))), 'status': 'in_range'}
    return {'delta75': float(levels[-1]), 'status': 'above_range'}


# ----------------------------------------------------------------------------- data


def load(output: Path, stage: str = 'main', kind: str = 'strict') -> list[dict]:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl') if c['stage'] == stage}
    results = {r['id']: r for r in read_jsonl(output / f'results_{stage}.jsonl')}
    rows = []
    for s in read_jsonl(output / f'scores_{stage}_{kind}.jsonl'):
        case = cases[s['id']]
        if case['format'] == 'bbox' and s.get('valid'):
            # Box-centre hit: same criterion as the point hit, applied to the centre of the predicted box.
            s = {**s, 'center_hit': s['pred_rc'] == list(case['truth']['target_rc'])}
        rows.append({**case, 'score': s, 'result': results[s['id']]})
    return rows


def select(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if (r[k] in v if isinstance(v, (tuple, list, set)) else r[k] == v)]
    return out


def binary_stats(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r['score']['correct'] for r in rows)
    pos = [r['score']['s'] for r in rows if r['truth']['answer_yes']]
    neg = [r['score']['s'] for r in rows if not r['truth']['answer_yes']]
    return {'n': n, 'acc': k / n if n else float('nan'), 'ci': wilson(k, n), 'auroc': auroc(pos, neg),
            'yes_rate': sum(r['score']['s'] > 0 for r in rows) / n if n else float('nan'),
            'format_valid': sum(r['score']['format_valid'] for r in rows) / n if n else float('nan')}


def rate_stats(rows: list[dict], key: str = 'correct') -> dict:
    n = len(rows)
    k = sum(bool(r['score'].get(key)) for r in rows)
    return {'n': n, 'acc': k / n if n else float('nan'), 'ci': wilson(k, n)}


def chance(task: str, fmt: str, grid: int, canvas: int | None = None) -> float:
    if fmt == 'yes_no':
        return 0.5
    if fmt == 'row_column':
        return 1 / grid ** 2
    side = max(256, grid * 64) if canvas is None else canvas
    return (64 / side) ** 2 if fmt == 'point' else float('nan')


# ----------------------------------------------------------------------------- tables


def fmt_ci(stat: dict) -> str:
    if not stat['n']:
        return '–'
    lo, hi = stat['ci']
    return f"{stat['acc']:.2f} [{lo:.2f}, {hi:.2f}]"


SUBTASKS = [  # (label, task, variant filter, format, metric key)
    ('T1', 'T1', None, 'yes_no', 'binary'),
    ('T2', 'T2', None, 'yes_no', 'binary'),
    ('T3 box', 'T3', ('box_target', 'box_normal'), 'yes_no', 'binary'),
    ('T3 row/col', 'T3', ('rowcol_target', 'rowcol_normal'), 'yes_no', 'binary'),
    ('T4 row/col', 'T4', None, 'row_column', 'correct'),
    ('T4 point', 'T4', None, 'point', 'hit'),
    ('T4 bbox', 'T4', None, 'bbox', 'correct'),
    ('T5 row/col', 'T5', None, 'row_column', 'correct'),
    ('T5 point', 'T5', None, 'point', 'hit'),
    ('T5 bbox', 'T5', None, 'bbox', 'correct'),
]


def subtask_rows(rows, task, variants, fmt, **kw):
    out = select(rows, task=task, format=fmt, wording='primary', **kw)
    if variants:
        out = select(out, variant=variants)
    return out


def stat_for(rows, metric):
    return binary_stats(rows) if metric == 'binary' else rate_stats(rows, metric)


def summarize(strict: list[dict], diag: list[dict]) -> dict:
    out: dict = {'subtask': {}, 'format': {}, 't6': {}, 't0': {}, 'wording': {}, 'psychometric': {},
                 'conditional': {}, 'compression': {}, 'distance': {}}
    # Main sub-task grid.
    for kind in KINDS:
        for level in range(1, 5):
            for grid in GRIDS:
                for label, task, variants, fmt, metric in SUBTASKS:
                    rs = subtask_rows(strict, task, variants, fmt, kind=kind, level=level, grid=grid)
                    rd = subtask_rows(diag, task, variants, fmt, kind=kind, level=level, grid=grid)
                    key = f'{kind}|{level}|{grid}|{label}'
                    out['subtask'][key] = {'strict': stat_for(rs, metric), 'diagnostic': stat_for(rd, metric)}
                    if fmt == 'point':
                        d = [r['score']['cell_distance'] for r in rd if r['score']['valid']]
                        out['distance'][key] = {'n_valid': len(d), 'median': float(np.median(d)) if d else None,
                                                'frac_lt_0.5': float(np.mean(np.array(d) < 0.5)) if d else None}
    # Format compliance and truncation per task and format (primary and alternative wording).
    groups = defaultdict(list)
    for r in strict:
        groups[(r['task'], r['format'], r['wording'])].append(r)
    dg = {r['id']: r for r in diag}
    for (task, fmt, wording), rs in sorted(groups.items()):
        out['format'][f'{task}|{fmt}|{wording}'] = {
            'n': len(rs), 'truncated': sum(r['result']['finish_reason'] != 'eos' for r in rs),
            'strict_valid': sum(r['score'].get('format_valid', r['score']['valid']) for r in rs),
            'diagnostic_valid': sum(dg[r['id']]['score'].get('format_valid', dg[r['id']]['score']['valid']) for r in rs),
            'mean_tokens': float(np.mean([r['result']['n_tokens'] for r in rs]))}
    # T6 and T0 per grid.
    for grid in GRIDS:
        for fmt, metric in (('row_column', 'correct'), ('point', 'hit'), ('bbox', 'correct')):
            rs = select(strict, task='T6', format=fmt, wording='primary', grid=grid)
            rd = select(diag, task='T6', format=fmt, wording='primary', grid=grid)
            out['t6'][f'{grid}|{fmt}'] = {'strict': rate_stats(rs, metric), 'diagnostic': rate_stats(rd, metric)}
        rs = select(strict, task='T0', wording='primary', grid=grid)
        out['t0'][str(grid)] = {'both': rate_stats(rs), 'rows': rate_stats(rs, 'rows_correct'),
                                'cols': rate_stats(rs, 'cols_correct'),
                                'answers': [[r['score'].get('pred_rows'), r['score'].get('pred_cols')] for r in rs]}
    # Wording robustness: same images, primary vs alternative wording.
    for task in ('T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'):
        for grid in GRIDS:
            alt = select(strict, task=task, wording='alt', grid=grid)
            if not alt:
                continue
            keys = {(r['image_id'], r['variant'], r['format']) for r in alt}
            prim = [r for r in select(strict, task=task, wording='primary', grid=grid)
                    if (r['image_id'], r['variant'], r['format']) in keys]
            metric = 'binary' if task in ('T1', 'T2', 'T3') else 'correct'
            out['wording'][f'{task}|{grid}'] = {'primary': stat_for(prim, metric), 'alt': stat_for(alt, metric)}
    # Psychometric thresholds on log magnitude.
    for kind in KINDS:
        for label, task, variants, grids in (('T1', 'T1', None, GRIDS), ('T2', 'T2', None, GRIDS)) + tuple(
                (f'T3 box g{g}', 'T3', ('box_target', 'box_normal'), (g,)) for g in GRIDS) + tuple(
                (f'T3 row/col g{g}', 'T3', ('rowcol_target', 'rowcol_normal'), (g,)) for g in GRIDS):
            rs = subtask_rows(strict, task, variants, 'yes_no', kind=kind, grid=grids)
            x = np.log([numeric_magnitude(kind, r['level']) for r in rs])
            c = np.array([r['score']['correct'] for r in rs], float)
            if len(rs):
                out['psychometric'][f'{kind}|{label}'] = fit_psychometric(x, c)
    # Bias-free thresholds from AUROC per magnitude level, and the Yes rate that explains accuracy near 0.5.
    out['psychometric_auroc'], out['binary_bias'] = {}, {}
    binary_labels = [('T1', 'T1', None), ('T2', 'T2', None), ('T3 box', 'T3', ('box_target', 'box_normal')),
                     ('T3 row/col', 'T3', ('rowcol_target', 'rowcol_normal'))]
    for kind in KINDS:
        mags = [numeric_magnitude(kind, l) for l in range(1, 5)]
        for label, task, variants in binary_labels:
            for grids in ((GRIDS,) if task != 'T3' else tuple((g,) for g in GRIDS)):
                rs = subtask_rows(strict, task, variants, 'yes_no', kind=kind, grid=grids)
                name = label if task != 'T3' else f'{label} g{grids[0]}'
                aucs = [binary_stats([r for r in rs if r['level'] == l])['auroc'] for l in range(1, 5)]
                out['psychometric_auroc'][f'{kind}|{name}'] = {'auroc': aucs, **auroc_threshold(mags, aucs)}
                out['binary_bias'][f'{kind}|{name}'] = binary_stats(rs)
    # Location tasks: conditional records and error structure use the coordinate diagnostic,
    # so that a list-wrapped but otherwise correct answer is not counted as a location failure.
    out['conditional'] = conditional(diag)
    out['compression'] = compression(diag)
    out['rowcol_errors'] = rowcol_error_structure(diag)
    for task in ('T4', 'T5', 'T6'):
        for grid in GRIDS:
            rd = select(diag, task=task, format='bbox', wording='primary', grid=grid)
            out['distance'][f'{task}|{grid}|bbox centre'] = rate_stats(rd, 'center_hit')
    return out


def rowcol_error_structure(diag: list[dict]) -> dict:
    """Composition of wrong row/col answers: off by one in one axis, row only, column only, both, far."""
    out = {}
    for task in ('T4', 'T5', 'T6'):
        for grid in GRIDS:
            rs = [r for r in select(diag, task=task, format='row_column', wording='primary', grid=grid)
                  if r['score']['valid'] and not r['score']['correct']]
            n = len(rs)
            dr = np.array([r['score']['d_row'] for r in rs])
            dc = np.array([r['score']['d_col'] for r in rs])
            if not n:
                out[f'{task}|{grid}'] = {'n_wrong': 0}
                continue
            out[f'{task}|{grid}'] = {
                # Answer (1, 1) is the typical answer when no target is found (mostly invisible colour changes).
                'pred_11': float(np.mean([r['score']['pred_rc'] == [0, 0] for r in rs])),
                'n_wrong': n, 'row_only': float(np.mean((dr != 0) & (dc == 0))),
                'col_only': float(np.mean((dr == 0) & (dc != 0))), 'both': float(np.mean((dr != 0) & (dc != 0))),
                'adjacent': float(np.mean(np.abs(dr) + np.abs(dc) == 1)),
                'chebyshev_le1': float(np.mean(np.maximum(np.abs(dr), np.abs(dc)) <= 1)),
                'median_abs_dr': float(np.median(np.abs(dr))), 'median_abs_dc': float(np.median(np.abs(dc))),
                'mean_dr': float(dr.mean()), 'mean_dc': float(dc.mean())}
    return out


def conditional(strict: list[dict]) -> dict:
    """Per-image success record across sub-tasks (primary wording).

    Binary tasks count as solved for an image only if both the target case and its paired
    control case are answered correctly (sign of s).
    """
    per = defaultdict(dict)
    pairs = {'T1': ('neg', 'pos'), 'T2': ('neg', 'pos'), 'T3 box': ('box_target', 'box_normal'),
             'T3 row/col': ('rowcol_target', 'rowcol_normal')}
    binary = defaultdict(dict)
    for r in select(strict, wording='primary'):
        rid = r['record_id']
        if r['format'] == 'yes_no':
            label = r['task'] if r['task'] != 'T3' else ('T3 box' if r['variant'].startswith('box') else 'T3 row/col')
            binary[(rid, label)][r['variant']] = r['score']['correct']
        elif r['task'] in ('T4', 'T5', 'T6') and r['format'] in ('row_column', 'point'):
            key = 'correct' if r['format'] == 'row_column' else 'hit'
            per[rid][f"{r['task']} {'row/col' if r['format'] == 'row_column' else 'point'}"] = bool(r['score'].get(key))
        per[rid]['_grid'] = r['grid']
        per[rid]['_kind'] = r['kind']
        per[rid]['_level'] = r['level']
    for (rid, label), v in binary.items():
        a, b = pairs[label]
        per[rid][label] = bool(v.get(a) and v.get(b))
        per[rid][label + ' target'] = bool(v.get(a))
    tasks = ['T1', 'T2', 'T3 box', 'T3 row/col', 'T4 row/col', 'T5 row/col', 'T5 point']
    out = {'records': {k: v for k, v in per.items()}, 'tables': {}}
    for grid in GRIDS:
        recs = [v for v in per.values() if v['_grid'] == grid and v['_kind'] != 'none']
        table = {}
        for a in tasks:
            for b in tasks:
                given = [v for v in recs if v.get(a) is True]
                table[f'{b}|{a}'] = {'n': len(given), 'p': (sum(bool(v.get(b)) for v in given) / len(given)) if given else None}
        fail5 = [v for v in recs if v.get('T5 row/col') is False]
        table['given_T5_fail'] = {t: (sum(bool(v.get(t)) for v in fail5) / len(fail5) if fail5 else None) for t in tasks}
        table['n_T5_fail'] = len(fail5)
        # T5 success split by the outcome of each other sub-task on the same image (section 5.2).
        table['T5_given'] = {}
        for a in ['T1', 'T2', 'T3 box', 'T3 row/col', 'T4 row/col']:
            ok = [v for v in recs if v.get(a) is True and 'T5 row/col' in v]
            bad = [v for v in recs if v.get(a) is False and 'T5 row/col' in v]
            table['T5_given'][a] = {
                'n_ok': len(ok), 'p_ok': sum(v['T5 row/col'] for v in ok) / len(ok) if ok else None,
                'n_fail': len(bad), 'p_fail': sum(v['T5 row/col'] for v in bad) / len(bad) if bad else None}
        out['tables'][str(grid)] = table
    return out


def compression(diag: list[dict]) -> dict:
    """Least-squares slope of predicted on true normalized coordinates (point answers, diagnostic)."""
    out = {}
    for task in ('T4', 'T5', 'T6'):
        for grid in GRIDS:
            rs = [r for r in select(diag, task=task, format='point', wording='primary', grid=grid) if r['score']['valid']]
            if len(rs) < 3:
                continue
            res = {}
            for axis, name in ((0, 'x'), (1, 'y')):
                t = np.array([1000 * r['truth']['target_center'][axis] / r['truth']['canvas'][axis] for r in rs])
                p = np.array([1000 * r['score']['pred_center'][axis] / r['truth']['canvas'][axis] for r in rs])
                slope, intercept = np.polyfit(t, p, 1)
                res[name] = {'slope': float(slope), 'intercept': float(intercept),
                             'r': float(np.corrcoef(t, p)[0, 1]) if t.std() > 0 and p.std() > 0 else None}
            out[f'{task}|{grid}'] = {'n': len(rs), **res}
    return out


def write_tables(summary: dict, path: Path) -> None:
    lines = ['# Experiment 1.3 tables (generated)', '']
    lines += ['## Format compliance', '', '| task | format | wording | n | truncated | strict valid | diagnostic valid | mean tokens |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for key, v in summary['format'].items():
        t, f, w = key.split('|')
        lines.append(f"| {t} | {f} | {w} | {v['n']} | {v['truncated']} | {v['strict_valid']} | {v['diagnostic_valid']} | {v['mean_tokens']:.1f} |")
    labels = [s[0] for s in SUBTASKS]
    for which in ('strict', 'diagnostic'):
        lines += ['', f'## Sub-task accuracy ({which}); binary = sign accuracy, location = row/col exact, point hit, bbox IoU>=0.5', '']
        lines += ['| kind | level | grid | ' + ' | '.join(labels) + ' |', '| ' + ' | '.join(['---'] * (3 + len(labels))) + ' |']
        for kind in KINDS:
            for level in range(1, 5):
                for grid in GRIDS:
                    cells = [summary['subtask'][f'{kind}|{level}|{grid}|{l}'][which] for l in labels]
                    lines.append(f'| {kind} | {magnitude_label(kind, level)} | {grid} | ' + ' | '.join(
                        f"{c['acc']:.2f}" if c['n'] else '–' for c in cells) + ' |')
    lines += ['', '## Binary AUROC of s', '', '| kind | level | grid | T1 | T2 | T3 box | T3 row/col |', '| --- | --- | --- | --- | --- | --- | --- |']
    for kind in KINDS:
        for level in range(1, 5):
            for grid in GRIDS:
                cells = [summary['subtask'][f'{kind}|{level}|{grid}|{l}']['strict'] for l in labels[:4]]
                lines.append(f'| {kind} | {magnitude_label(kind, level)} | {grid} | ' + ' | '.join(f"{c['auroc']:.2f}" for c in cells) + ' |')
    lines += ['', '## T6 and T0', '', '| grid | T6 row/col | T6 row/col (diag) | T6 point hit | T6 bbox IoU>=0.5 | T6 point hit (diag) | T6 bbox (diag) | T0 both | T0 rows | T0 cols |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for grid in GRIDS:
        t6 = summary['t6']
        t0 = summary['t0'][str(grid)]
        lines.append(f"| {grid} | {fmt_ci(t6[f'{grid}|row_column']['strict'])} | {fmt_ci(t6[f'{grid}|row_column']['diagnostic'])} | {fmt_ci(t6[f'{grid}|point']['strict'])} | "
                     f"{fmt_ci(t6[f'{grid}|bbox']['strict'])} | {fmt_ci(t6[f'{grid}|point']['diagnostic'])} | "
                     f"{fmt_ci(t6[f'{grid}|bbox']['diagnostic'])} | {fmt_ci(t0['both'])} | {fmt_ci(t0['rows'])} | {fmt_ci(t0['cols'])} |")
    lines += ['', '## Binary bias (pooled over magnitudes; T1/T2 pooled over grids)', '',
              '| kind | sub-task | n | accuracy | AUROC | Yes rate |', '| --- | --- | --- | --- | --- | --- |']
    for key, v in summary['binary_bias'].items():
        kind, label = key.split('|')
        lines.append(f"| {kind} | {label} | {v['n']} | {v['acc']:.2f} | {v['auroc']:.2f} | {v['yes_rate']:.2f} |")
    lines += ['', '## AUROC thresholds (magnitude at which AUROC reaches 0.75, log-linear interpolation)', '',
              '| kind | sub-task | AUROC per level | delta75 | status |', '| --- | --- | --- | --- | --- |']
    for key, v in summary['psychometric_auroc'].items():
        kind, label = key.split('|')
        lines.append(f"| {kind} | {label} | {' / '.join(f'{a:.2f}' for a in v['auroc'])} | {v['delta75']:.3g} | {v['status']} |")
    lines += ['', '## Wrong row/col answers (diagnostic): composition', '',
              '| task | grid | n wrong | answer (1,1) | row only | column only | both | adjacent | Chebyshev<=1 | median abs dr | median abs dc |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for key, v in summary['rowcol_errors'].items():
        t, g = key.split('|')
        if v['n_wrong']:
            lines.append(f"| {t} | {g} | {v['n_wrong']} | {v['pred_11']:.2f} | {v['row_only']:.2f} | {v['col_only']:.2f} | {v['both']:.2f} | "
                         f"{v['adjacent']:.2f} | {v['chebyshev_le1']:.2f} | {v['median_abs_dr']:.0f} | {v['median_abs_dc']:.0f} |")
    lines += ['', '## Box-centre hit (diagnostic)', '', '| task | grid | box-centre hit |', '| --- | --- | --- |']
    for key, v in summary['distance'].items():
        if key.endswith('bbox centre'):
            t, g, _ = key.split('|')
            lines.append(f'| {t} | {g} | {fmt_ci(v)} |')
    lines += ['', '## Psychometric thresholds (delta75, accuracy fit on log magnitude)', '', '| kind | sub-task | delta75 | status |', '| --- | --- | --- | --- |']
    for key, v in summary['psychometric'].items():
        kind, label = key.split('|')
        lines.append(f"| {kind} | {label} | {v['delta75']:.3g} | {v['status']} |")
    lines += ['', '## Wording check (same images)', '', '| task | grid | primary | alternative |', '| --- | --- | --- | --- |']
    for key, v in summary['wording'].items():
        t, g = key.split('|')
        lines.append(f"| {t} | {g} | {fmt_ci(v['primary'])} (n={v['primary']['n']}) | {fmt_ci(v['alt'])} |")
    lines += ['', '## Coordinate compression (point, diagnostic)', '', '| task | grid | n | slope x | slope y |', '| --- | --- | --- | --- | --- |']
    for key, v in summary['compression'].items():
        t, g = key.split('|')
        lines.append(f"| {t} | {g} | {v['n']} | {v['x']['slope']:.2f} | {v['y']['slope']:.2f} |")
    lines += ['', '## Success of each sub-task among images whose T5 row/col answer failed (location: diagnostic)', '',
              '| grid | n | ' + ' | '.join(['T1', 'T2', 'T3 box', 'T3 row/col', 'T4 row/col', 'T5 point']) + ' |', '| --- ' * 8 + '|']
    for grid in GRIDS:
        t = summary['conditional']['tables'][str(grid)]
        g = t['given_T5_fail']
        lines.append(f"| {grid} | {t['n_T5_fail']} | " + ' | '.join(
            f'{g[k]:.2f}' if g[k] is not None else '–' for k in ['T1', 'T2', 'T3 box', 'T3 row/col', 'T4 row/col', 'T5 point']) + ' |')
    lines += ['', '## T5 row/col success split by the outcome of another sub-task on the same image', '',
              '| grid | sub-task | n solved | P(T5) given solved | n failed | P(T5) given failed |', '| --- | --- | --- | --- | --- | --- |']
    for grid in GRIDS:
        for a, v in summary['conditional']['tables'][str(grid)]['T5_given'].items():
            f = lambda x: f'{x:.2f}' if x is not None else '–'
            lines.append(f"| {grid} | {a} | {v['n_ok']} | {f(v['p_ok'])} | {v['n_fail']} | {f(v['p_fail'])} |")
    path.write_text('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------- figures


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': GRID_INK, 'grid.linewidth': 0.6, 'axes.titlesize': 10,
                         'savefig.dpi': 160, 'savefig.bbox': 'tight', 'legend.frameon': False})
    return plt


def plot_curves(summary: dict, out: Path, metric: str = 'acc') -> None:
    """Accuracy (or AUROC) of each sub-task against grid size; one line per magnitude level."""
    plt = _style()
    labels = [s for s in SUBTASKS] if metric == 'acc' else [s for s in SUBTASKS if s[4] == 'binary']
    xs = np.arange(len(GRIDS))
    fig, axes = plt.subplots(len(KINDS), len(labels), figsize=(1.75 * len(labels) + 0.6, 2.0 * len(KINDS) + 0.5),
                             sharey=True, squeeze=False)
    for i, kind in enumerate(KINDS):
        for j, (label, task, variants, fmt, m) in enumerate(labels):
            ax = axes[i, j]
            for level in range(1, 5):
                stats = [summary['subtask'][f'{kind}|{level}|{g}|{label}']['diagnostic'] for g in GRIDS]
                if metric == 'acc':
                    y = np.array([s['acc'] for s in stats])
                    lo = np.array([s['ci'][0] for s in stats])
                    hi = np.array([s['ci'][1] for s in stats])
                    ax.errorbar(xs + (level - 2.5) * 0.06, y, yerr=[y - lo, hi - y], color=LEVEL_COLORS[level - 1],
                                lw=1.6, marker='o', ms=3.5, capsize=0, elinewidth=0.8,
                                label=magnitude_label(kind, level))
                else:
                    ax.plot(xs, [s['auroc'] for s in stats], color=LEVEL_COLORS[level - 1], lw=1.6, marker='o', ms=3.5,
                            label=magnitude_label(kind, level))
            if metric == 'acc' and fmt != 'yes_no':
                ch = [chance(task, fmt, g) for g in GRIDS]
                if not np.isnan(ch).all():
                    ax.plot(xs, ch, ls='--', color=MUTED, lw=0.9)
            else:
                ax.axhline(0.5, ls='--', color=MUTED, lw=0.9)
            ax.set_xticks(xs, [f'{g}×{g}' for g in GRIDS])
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(f"{KIND_NAMES[kind]}\n{'accuracy' if metric == 'acc' else 'AUROC'}")
            if j == len(labels) - 1:
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, title='magnitude', title_fontsize=8)
    fig.supxlabel('grid size', fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_t6_t0(summary: dict, out: Path) -> None:
    plt = _style()
    xs = np.arange(len(GRIDS))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    series = [('T6 row/col', [summary['t6'][f'{g}|row_column']['diagnostic'] for g in GRIDS], SERIES[0]),
              ('T6 point hit', [summary['t6'][f'{g}|point']['diagnostic'] for g in GRIDS], SERIES[1]),
              ('T6 bbox IoU≥0.5', [summary['t6'][f'{g}|bbox']['diagnostic'] for g in GRIDS], SERIES[2])]
    for ax, group, title in ((axes[0], series, 'T6 red target'),
                             (axes[1], [('T0 rows and columns', [summary['t0'][str(g)]['both'] for g in GRIDS], SERIES[0]),
                                        ('T0 rows', [summary['t0'][str(g)]['rows'] for g in GRIDS], SERIES[1]),
                                        ('T0 columns', [summary['t0'][str(g)]['cols'] for g in GRIDS], SERIES[2])],
                              'T0 row and column count')):
        for k, (name, stats, color) in enumerate(group):
            y = np.array([s['acc'] for s in stats])
            lo, hi = np.array([s['ci'][0] for s in stats]), np.array([s['ci'][1] for s in stats])
            ax.errorbar(xs + (k - 1) * 0.07, y, yerr=[y - lo, hi - y], color=color, lw=1.6, marker='o', ms=3.5,
                        elinewidth=0.8, label=name)
        ax.set_xticks(xs, [f'{g}×{g}' for g in GRIDS])
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8, loc='lower left')
    axes[0].plot(xs, [1 / g ** 2 for g in GRIDS], ls='--', color=MUTED, lw=0.9)
    axes[0].set_ylabel('accuracy')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_psychometric(strict: list[dict], summary: dict, out: Path) -> None:
    """T1, T2 and T3 (box) accuracy against magnitude with fitted curves, one panel per change type."""
    plt = _style()
    fig, axes = plt.subplots(1, len(KINDS), figsize=(10.5, 3.0), sharey=True)
    curves = [('T1', 'T1', None, GRIDS, SERIES[0]), ('T2', 'T2', None, GRIDS, SERIES[1])] + [
        (f'T3 box g{g}', 'T3', ('box_target', 'box_normal'), (g,), c)
        for g, c in zip(GRIDS, (SERIES[2], SERIES[3], SERIES[4], SERIES[6]))]
    for ax, kind in zip(axes, KINDS):
        mags = [numeric_magnitude(kind, l) for l in range(1, 5)]
        for label, task, variants, grids, color in curves:
            rs = subtask_rows(strict, task, variants, 'yes_no', kind=kind, grid=grids)
            acc = [np.mean([r['score']['correct'] for r in rs if r['level'] == l]) for l in range(1, 5)]
            fit = summary['psychometric'][f'{kind}|{label}']
            grid_x = np.geomspace(mags[0], mags[-1], 100)
            p = 0.5 + 0.5 / (1 + np.exp(-(np.log(grid_x) - fit['m']) / fit['w']))
            ax.plot(mags, acc, 'o', color=color, ms=4)
            name = label if len(grids) > 1 else f'T3 box, {grids[0]}×{grids[0]}'
            ax.plot(grid_x, p, color=color, lw=1.4, label=name)
        ax.axhline(0.75, color=MUTED, ls=':', lw=0.9)
        ax.axhline(0.5, color=MUTED, ls='--', lw=0.9)
        ax.set_xscale('log')
        ax.set_xticks(mags, [magnitude_label(kind, l) for l in range(1, 5)])
        ax.minorticks_off()
        ax.set_title(KIND_NAMES[kind])
        ax.set_xlabel('change magnitude')
    axes[0].set_ylabel('accuracy (sign of s)')
    axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_conditional(summary: dict, out: Path) -> None:
    plt = _style()
    tasks = ['T1', 'T2', 'T3 box', 'T3 row/col', 'T4 row/col', 'T5 point']
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    data = np.array([[summary['conditional']['tables'][str(g)]['given_T5_fail'][t] or np.nan for t in tasks] for g in GRIDS])
    im = ax.imshow(data, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(len(GRIDS)):
        for j in range(len(tasks)):
            v = data[i, j]
            ax.text(j, i, '–' if np.isnan(v) else f'{v:.2f}', ha='center', va='center',
                    color='white' if v > 0.6 else INK, fontsize=8)
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_yticks(range(len(GRIDS)), [f"{g}×{g} (n={summary['conditional']['tables'][str(g)]['n_T5_fail']})" for g in GRIDS])
    ax.grid(False)
    ax.set_title('Success rate among images whose T5 row/col answer is wrong')
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_rowcol_errors(strict: list[dict], out: Path) -> None:
    """Row/column offset histograms of wrong row/col answers (T4-T6) for the two largest grids."""
    plt = _style()
    tasks = ('T4', 'T5', 'T6')
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.4))
    span = 5
    for i, grid in enumerate((10, 20)):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            rs = [r for r in select(strict, task=task, format='row_column', wording='primary', grid=grid) if r['score']['valid']]
            h = np.zeros((2 * span + 1, 2 * span + 1))
            far = 0
            for r in rs:
                dr, dc = r['score']['d_row'], r['score']['d_col']
                if abs(dr) <= span and abs(dc) <= span:
                    h[dr + span, dc + span] += 1
                else:
                    far += 1
            n = len(rs)
            im = ax.imshow(h / max(n, 1), cmap='Blues', vmin=0, extent=(-span - .5, span + .5, span + .5, -span - .5))
            ax.grid(False)
            ax.set_title(f'{task}, {grid}×{grid}: n={n}, exact={h[span, span] / max(n, 1):.2f}, |Δ|>{span}: {far / max(n, 1):.2f}', fontsize=8)
            ax.set_xlabel('column offset (pred − true)')
            ax.set_ylabel('row offset')
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_ordinal_accuracy(strict: list[dict], out: Path) -> None:
    """Row/col exact rate as a function of the true target row and column (pooled T4-T6, 10 and 20 grids)."""
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8), sharey=True)
    for ax, grid in zip(axes, (10, 20)):
        rs = select(strict, task=('T4', 'T5', 'T6'), format='row_column', wording='primary', grid=grid)
        bins = np.array_split(np.arange(grid), 5)
        for k, (axis, name) in enumerate(((0, 'target row'), (1, 'target column'))):
            row_ok = [[(r['score'].get('pred_rc') or [None, None])[axis] == r['truth']['target_rc'][axis]
                       for r in rs if r['truth']['target_rc'][axis] in b] for b in bins]
            y = [np.mean(v) if v else np.nan for v in row_ok]
            ci = [wilson(sum(v), len(v)) for v in row_ok]
            xs = np.arange(len(bins)) + (k - 0.5) * 0.08
            ax.errorbar(xs, y, yerr=[[a - c[0] for a, c in zip(y, ci)], [c[1] - a for a, c in zip(y, ci)]],
                        color=SERIES[k], marker='o', ms=3.5, lw=1.5, elinewidth=0.8,
                        label=f'{name.split()[1]} ordinal correct')
        ax.set_xticks(range(len(bins)), [f'{b[0] + 1}–{b[-1] + 1}' for b in bins])
        ax.set_xlabel('true ordinal (1-based)')
        ax.set_title(f'{grid}×{grid}, T4–T6 row/col answers')
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel('fraction correct')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_vector_field(diag: list[dict], out: Path) -> None:
    """Displacement from true target centre to predicted point (normalized 0-1000 coordinates)."""
    plt = _style()
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4))
    for i, grid in enumerate((10, 20)):
        for j, task in enumerate(('T4', 'T5', 'T6')):
            ax = axes[i, j]
            rs = [r for r in select(diag, task=task, format='point', wording='primary', grid=grid) if r['score']['valid']]
            for r in rs:
                cw, ch = r['truth']['canvas']
                t = np.array(r['truth']['target_center']) / [cw, ch] * 1000
                p = np.array(r['score']['pred_center']) / [cw, ch] * 1000
                color = SERIES[2] if r['score']['hit'] else SERIES[7]
                ax.annotate('', xy=p, xytext=t, arrowprops=dict(arrowstyle='->', color=color, lw=0.8, alpha=0.8))
                ax.plot(*t, 'o', color=INK, ms=1.8)
            ax.set_xlim(0, 1000)
            ax.set_ylim(1000, 0)
            ax.set_aspect('equal')
            hits = sum(r['score']['hit'] for r in rs)
            ax.set_title(f'{task}, {grid}×{grid}: hit {hits}/{len(rs)}', fontsize=9)
    fig.text(0.5, 0.005, 'black dot: true target centre; arrow head: predicted point (green: inside target cell, red: outside)',
             ha='center', fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_compression(diag: list[dict], out: Path) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.8), sharey=True)
    for ax, grid in zip(axes, GRIDS):
        for k, task in enumerate(('T4', 'T5', 'T6')):
            rs = [r for r in select(diag, task=task, format='point', wording='primary', grid=grid) if r['score']['valid']]
            t = [1000 * r['truth']['target_center'][0] / r['truth']['canvas'][0] for r in rs]
            p = [1000 * r['score']['pred_center'][0] / r['truth']['canvas'][0] for r in rs]
            ax.scatter(t, p, s=9, color=SERIES[k], alpha=0.6, label=task, linewidths=0)
        ax.plot([0, 1000], [0, 1000], color=MUTED, ls='--', lw=0.8)
        ax.set_title(f'{grid}×{grid}')
        ax.set_xlabel('true x (0–1000)')
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
    axes[0].set_ylabel('predicted x (0–1000)')
    axes[-1].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    strict, diag = load(args.output, kind='strict'), load(args.output, kind='diagnostic')
    if not strict:
        raise ValueError('No scored results.')
    target = args.output / 'analysis'
    (target / 'figures').mkdir(parents=True, exist_ok=True)
    summary = summarize(strict, diag)
    records = summary['conditional'].pop('records')
    (target / 'per_image_records.json').write_text(json.dumps(records, indent=1) + '\n')
    (target / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    write_tables(summary, target / 'tables.md')
    fig = target / 'figures'
    plot_curves(summary, fig / 'subtask_curves.png')
    plot_curves(summary, fig / 'binary_auroc.png', metric='auroc')
    plot_t6_t0(summary, fig / 't6_t0.png')
    plot_psychometric(strict, summary, fig / 'psychometric.png')
    plot_conditional(summary, fig / 'conditional_t5_fail.png')
    plot_rowcol_errors(diag, fig / 'rowcol_offsets.png')
    plot_ordinal_accuracy(diag, fig / 'ordinal_accuracy.png')
    plot_vector_field(diag, fig / 'vector_field.png')
    plot_compression(diag, fig / 'compression_x.png')
    print((target / 'tables.md').read_text()[:4000])


if __name__ == '__main__':
    main()
