"""Statistics and figures for experiment 4.3 (light): DeepStack ablation inside Qwen3-VL.

Reads runs/<model>/<condition>.scores.jsonl written by deepstack_ablation.score and writes
analysis/summary.json, analysis/tables.md and the figures used in
docs/analysis/deepstack/results_2026-10-15_ablation_light.md.

Primary family (Holm-corrected): conditions off, blur, mean against full on the metrics in
``PRIMARY`` for Qwen3-VL-8B-Instruct.  Everything else (single-layer conditions, per-grid splits,
the 4B model) is reported with uncorrected exact McNemar p values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.behavior.decomposition_analysis import GRID_INK, INK, MUTED, _style
from analysis_tools.deepstack.deepstack_ablation import BLUR_SIGMA, CONDITIONS, blurred
from analysis_tools.deepstack.deepstack_comparison import slug
from analysis_tools.deepstack.deepstack_comparison_analysis import (
    auroc_diff, fast_auroc, fmt_p, holm, mcnemar, paired_bootstrap, rate,
)

MODELS = ('Qwen3-VL-8B-Instruct', 'Qwen3-VL-4B-Instruct')
CONTRASTS = ('off', 'blur', 'mean')
SINGLE = ('off1', 'off2', 'off3')
# Validated categorical palette (dataviz validator, light surface); full uses neutral ink.
COLORS = {'full': '#52514e', 'off': '#2a78d6', 'blur': '#eb6834', 'mean': '#1baf7a',
          'off1': '#6250d6', 'off2': '#6250d6', 'off3': '#6250d6'}
LABELS = {'full': 'Full', 'off': 'Off', 'blur': 'Blur', 'mean': 'Mean', 'off1': 'Off-1', 'off2': 'Off-2',
          'off3': 'Off-3'}
# metric key -> (task, format, grids or None, score field, area strata or None)
METRICS = {
    'T1 colour': ('T1', 'yes_no', None, 'correct', None),
    'T2 colour': ('T2', 'yes_no', None, 'correct', None),
    'T5 point': ('T5', 'point', (10, 20), 'hit', None),
    'T5 bbox': ('T5', 'bbox', (10, 20), 'correct', None),
    'T5 row/col': ('T5', 'row_column', (10, 20), 'correct', None),
    'T6 point': ('T6', 'point', (10, 20), 'hit', None),
    'T6 bbox': ('T6', 'bbox', (10, 20), 'correct', None),
    'T6 row/col': ('T6', 'row_column', (10, 20), 'correct', None),
    'T5 point (either convention)': ('T5', 'point', (10, 20), 'hit_any', None),
    'T6 point (either convention)': ('T6', 'point', (10, 20), 'hit_any', None),
    'RefCOCO a>=5%': ('RefCOCO', 'bbox', None, 'correct', ('5to10', 'ge10')),
    'RefCOCO 1-5%': ('RefCOCO', 'bbox', None, 'correct', ('1to5',)),
}
PRIMARY = tuple(k for k in METRICS if k != 'RefCOCO 1-5%' and 'either' not in k)
SPLITS = [(f'{t} {f} {g}x{g}', (t, f, (g,), 'hit' if f == 'point' else 'correct', None))
          for t in ('T5', 'T6') for f in ('point', 'bbox', 'row_column') for g in (10, 20)]
SPLITS += [(f'{t} colour dE{m}', (t, 'yes_no', None, 'correct', None, lv)) for t in ('T1', 'T2')
           for lv, m in ((2, 10), (3, 20))]


def load_runs(output: Path, model: str) -> dict[str, dict[str, dict]]:
    """condition -> case id -> score row merged with its case."""
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    runs = {}
    for cond in CONDITIONS:
        path = output / 'runs' / slug(model) / f'{cond}.scores.jsonl'
        if path.exists():
            runs[cond] = {r['id']: with_pixel_reading({**cases[r['id']], **r}) for r in read_jsonl(path)}
    return runs


def with_pixel_reading(row: dict) -> dict:
    """Diagnostic 'hit under either convention' for point answers (docs/analysis/plan.md section 6, item 9).

    The strict score reads the output (x, y) as 0-1000 normalized coordinates.  The pixel reading takes
    the same numbers as absolute pixels and requires x < W, y < H and (x, y) inside the target cell B.
    chance_any is the hit rate of a uniform random output in [0, 1000]^2 under either reading
    (sum of the two areas; their overlap is ignored, so it is a slight upper bound).
    """
    if row.get('suite') != 'decomp' or row.get('format') != 'point':
        return row
    W, H = row['truth']['canvas']
    x0, y0, x1, y1 = row['truth']['target_cell_box']
    hit_pix = False
    if row.get('valid'):
        x, y = row['pred_center'][0] * 1000 / W, row['pred_center'][1] * 1000 / H
        hit_pix = x < W and y < H and x0 <= x < x1 and y0 <= y < y1
    area = (x1 - x0) * (y1 - y0)
    clipped = max(0, min(x1, 1000) - x0) * max(0, min(y1, 1000) - y0)
    row.update(hit_pix=bool(hit_pix), hit_any=bool(row.get('hit')) or bool(hit_pix),
               chance_any=area / (W * H) + clipped / 1e6)
    return row


def select(rows: dict, spec) -> list[str]:
    task, fmt, grids, _, strata = spec[:5]
    level = spec[5] if len(spec) > 5 else None
    out = []
    for cid, r in rows.items():
        if r['task'] != task or r['format'] != fmt:
            continue
        if grids and r['grid'] not in grids:
            continue
        if strata and r['stratum'] not in strata:
            continue
        if level and r['level'] != level:
            continue
        out.append(cid)
    return sorted(out)


def contrast(runs: dict, spec, cond: str, n_boot: int, seed: int) -> dict | None:
    if cond not in runs or 'full' not in runs:
        return None
    ids = [i for i in select(runs['full'], spec) if i in runs[cond]]
    if not ids or len(ids) != len(select(runs['full'], spec)):
        return None                    # incomplete run: never compare unequal item sets
    field = spec[3]
    a = [bool(runs[cond][i].get(field)) for i in ids]
    b = [bool(runs['full'][i].get(field)) for i in ids]
    test = mcnemar(a, b)
    ci = paired_bootstrap(np.array(a, float), np.array(b, float), n_boot, seed)
    return {'n': len(ids), 'full': rate(b), 'cond': rate(a), 'delta': float(np.mean(a) - np.mean(b)),
            'delta_ci': ci, 'only_cond': test['only_first'], 'only_full': test['only_second'], 'p': test['p'],
            'invalid_cond': sum(not runs[cond][i].get('valid', True) for i in ids),
            'invalid_full': sum(not runs['full'][i].get('valid', True) for i in ids)}


def binary_auroc(runs: dict, task: str, cond: str, n_boot: int, seed: int, level: int | None = None) -> dict | None:
    if cond not in runs:
        return None
    spec = (task, 'yes_no', None, 'correct', None, level) if level else (task, 'yes_no', None, 'correct', None)
    ids = select(runs['full'], spec)
    if not ids or any(i not in runs[cond] for i in ids):
        return None
    labels = np.array([runs['full'][i]['answer_yes'] for i in ids])
    s_full = np.array([runs['full'][i]['s'] for i in ids])
    s_cond = np.array([runs[cond][i]['s'] for i in ids])
    groups = np.array([runs['full'][i]['record_id'] for i in ids])
    out = {'auroc_full': fast_auroc(s_full, labels), 'auroc_cond': fast_auroc(s_cond, labels),
           'mean_s_full': float(s_full.mean()), 'mean_s_cond': float(s_cond.mean())}
    if cond != 'full':
        out['diff'] = auroc_diff(s_cond, s_full, labels, groups, n_boot, seed)
    return out


def rowcol_offsets(runs: dict, cond: str, grid: int) -> dict:
    """Histogram of (d_row, d_col) of valid T5+T6 row/column answers, clipped to [-2, 2]."""
    hist = np.zeros((5, 5), int)
    n, invalid, d_rows, d_cols = 0, 0, [], []
    for r in runs.get(cond, {}).values():
        if r['task'] in ('T5', 'T6') and r['format'] == 'row_column' and r['grid'] == grid:
            n += 1
            if not r.get('valid'):
                invalid += 1
                continue
            hist[int(np.clip(r['d_row'], -2, 2)) + 2, int(np.clip(r['d_col'], -2, 2)) + 2] += 1
            d_rows.append(r['d_row'])
            d_cols.append(r['d_col'])
    return {'hist': hist.tolist(), 'n': n, 'invalid': invalid,
            'mean_d_row': float(np.mean(d_rows)) if d_rows else None,
            'mean_d_col': float(np.mean(d_cols)) if d_cols else None,
            'exact': int(hist[2, 2]), 'one_under': int(hist[1, 2] + hist[2, 1] + hist[1, 1])}


def point_distance(runs: dict, cond: str) -> dict:
    """Median centre distance (in cells) of valid T5/T6 point answers per grid."""
    out = {}
    for g in (10, 20):
        d = [r['cell_distance'] for r in runs.get(cond, {}).values()
             if r['task'] in ('T5', 'T6') and r['format'] == 'point' and r['grid'] == g and r.get('valid')]
        out[g] = {'median': float(np.median(d)) if d else None, 'n_valid': len(d)}
    return out


def failures(runs: dict, cond: str) -> dict:
    """Outcome counts of T5/T6 location answers (grids 10 and 20) and RefCOCO answers."""
    out: dict = {}
    for r in runs.get(cond, {}).values():
        if r['task'] in ('T5', 'T6'):
            key = f"{r['task']} {r['format']}"
        elif r['task'] == 'RefCOCO':
            key = 'RefCOCO'
        else:
            continue
        outcome = 'valid' if r.get('valid') else r.get('failure', 'invalid')
        out.setdefault(key, {}).setdefault(outcome, 0)
        out[key][outcome] += 1
    return out


def point_conventions(runs: dict, cond: str) -> list[dict]:
    out = []
    for task in ('T5', 'T6'):
        for g in (10, 20):
            rs = [r for r in runs.get(cond, {}).values() if r['task'] == task and r['format'] == 'point'
                  and r['grid'] == g]
            if rs:
                out.append({'task': task, 'grid': g, 'n': len(rs), 'strict': float(np.mean([bool(r.get('hit')) for r in rs])),
                            'pixel_only': float(np.mean([r['hit_pix'] and not r.get('hit') for r in rs])),
                            'any': float(np.mean([r['hit_any'] for r in rs])),
                            'chance': float(np.mean([r['chance_any'] for r in rs]))})
    return out


def coordinate_points(runs: dict, cond: str, task: str, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """True and predicted centres of valid point answers, as fractions of the canvas, shape (n, 2)."""
    rows = [r for r in runs.get(cond, {}).values() if r['task'] == task and r['format'] == 'point'
            and r['grid'] == grid and r.get('valid')]
    if not rows:
        return np.zeros((0, 2)), np.zeros((0, 2))
    size = np.array([r['truth']['canvas'] for r in rows], float)
    true = np.array([r['truth']['target_center'] for r in rows], float) / size
    pred = np.array([r['pred_center'] for r in rows], float) / size
    return true, pred


def coordinate_calibration(runs: dict, cond: str) -> list[dict]:
    """Least-squares fit pred = slope * true + intercept per axis (canvas fractions) for valid point answers."""
    out = []
    for task in ('T5', 'T6'):
        for g in (10, 20):
            true, pred = coordinate_points(runs, cond, task, g)
            if len(true) < 3:
                continue
            row = {'task': task, 'grid': g, 'n_valid': len(true)}
            for i, axis in enumerate('xy'):
                slope, intercept = np.polyfit(true[:, i], pred[:, i], 1)
                row[axis] = {'slope': float(slope), 'intercept': float(intercept),
                             'r': float(np.corrcoef(true[:, i], pred[:, i])[0, 1])}
            out.append(row)
    # RefCOCO: centre of the predicted box against the centre of the ground-truth box, as image fractions.
    rows = [r for r in runs.get(cond, {}).values() if r['task'] == 'RefCOCO' and r.get('valid')]
    if len(rows) >= 3:
        size = np.array([[r['width'], r['height']] for r in rows], float)
        true = np.array([[(r['bbox'][0] + r['bbox'][2]) / 2, (r['bbox'][1] + r['bbox'][3]) / 2] for r in rows]) / size
        pred = np.array([[(r['pred_box'][0] + r['pred_box'][2]) / 2, (r['pred_box'][1] + r['pred_box'][3]) / 2]
                         for r in rows]) / size
        row = {'task': 'RefCOCO', 'grid': '-', 'n_valid': len(rows)}
        for i, axis in enumerate('xy'):
            slope, intercept = np.polyfit(true[:, i], pred[:, i], 1)
            row[axis] = {'slope': float(slope), 'intercept': float(intercept),
                         'r': float(np.corrcoef(true[:, i], pred[:, i])[0, 1])}
        out.append(row)
    return out


def agreement_with_end_to_end(runs: dict, end_to_end: Path, model: str) -> dict | None:
    """Item-level agreement of this run's Full condition with the 4.2 scores of the same model."""
    path = end_to_end / 'decomposition' / slug(model) / 'scores_main_diagnostic.jsonl'
    ref_path = end_to_end / 'refcoco' / f'refcoco_scores_{slug(model)}.jsonl'
    if 'full' not in runs or not path.exists():
        return None
    ref = {r['id']: r for r in read_jsonl(path)}
    if ref_path.exists():
        ref.update({r['id']: r for r in read_jsonl(ref_path)})
    same, total = 0, 0
    for cid, r in runs['full'].items():
        if cid not in ref:
            continue
        key = 'hit' if r['format'] == 'point' else 'correct'
        total += 1
        same += bool(r.get(key)) == bool(ref[cid].get(key))
    return {'n': total, 'agree': same}


def analyze(output: Path, end_to_end: Path, n_boot: int, seed: int) -> dict:
    summary = {'blur_sigma': BLUR_SIGMA, 'models': {}}
    for model in MODELS:
        runs = load_runs(output, model)
        if 'full' not in runs:
            continue
        m = {'conditions': sorted(runs), 'metrics': {}, 'splits': {}, 'auroc': {}, 'rowcol': {}, 'distance': {},
             'agreement_4_2': agreement_with_end_to_end(runs, end_to_end, model)}
        for name, spec in METRICS.items():
            m['metrics'][name] = {c: contrast(runs, spec, c, n_boot, seed) for c in CONTRASTS + SINGLE}
            ids = select(runs['full'], spec)
            if ids:
                m['metrics'][name]['full_rate'] = rate([bool(runs['full'][i].get(spec[3])) for i in ids])
        for name, spec in SPLITS:
            m['splits'][name] = {c: contrast(runs, spec, c, n_boot, seed) for c in CONTRASTS + SINGLE}
        for task in ('T1', 'T2'):
            for lv in (None, 2, 3):
                key = f'{task}' + (f' L{lv}' if lv else '')
                m['auroc'][key] = {c: binary_auroc(runs, task, c, n_boot, seed, lv) for c in ('full',) + CONTRASTS + SINGLE}
        for c in runs:
            m['rowcol'][c] = {g: rowcol_offsets(runs, c, g) for g in (10, 20)}
            m['distance'][c] = point_distance(runs, c)
            m.setdefault('point_conventions', {})[c] = point_conventions(runs, c)
            m.setdefault('failures', {})[c] = failures(runs, c)
            m.setdefault('calibration', {})[c] = coordinate_calibration(runs, c)
            m.setdefault('coords', {})[c] = {f'{t}|{g}': [x.tolist() for x in coordinate_points(runs, c, t, g)]
                                            for t in ('T5', 'T6') for g in (10, 20)}
        # Holm correction over the primary family of the 8B model.
        family = [(name, c) for name in PRIMARY for c in CONTRASTS if m['metrics'][name].get(c)]
        adjusted = holm([m['metrics'][n][c]['p'] for n, c in family])
        for (n, c), p in zip(family, adjusted):
            m['metrics'][n][c]['p_holm'] = p
        m['holm_family_size'] = len(family)
        summary['models'][model] = m
    return summary


# ----------------------------------------------------------------------------- tables


def _cell(st: dict | None, holm_key: bool = True) -> str:
    if not st:
        return '–'
    p = st.get('p_holm', st['p']) if holm_key else st['p']
    star = '*' if p < 0.05 else ''
    return (f"{st['cond']['acc']:.2f} ({st['delta']:+.2f} [{st['delta_ci'][0]:+.2f}, {st['delta_ci'][1]:+.2f}]; "
            f"{st['only_cond']}/{st['only_full']}, p={fmt_p(p)}{star})")


def write_tables(summary: dict, path: Path) -> None:
    lines = ['# Experiment 4.3 (light) tables', '',
             'Cell: accuracy under the condition (difference to Full [paired bootstrap 95% CI]; '
             'McNemar b/c = only condition correct / only Full correct, p; Holm-adjusted for the 8B primary family).', '']
    for model, m in summary['models'].items():
        lines += [f'## {model}', '', f"Conditions run: {', '.join(m['conditions'])}; agreement of Full with 4.2: "
                  f"{m['agreement_4_2']}", '']
        conds = [c for c in CONTRASTS + SINGLE if c in m['conditions']]
        lines += ['| metric | n | Full | ' + ' | '.join(LABELS[c] for c in conds) + ' |',
                  '|' + ' --- |' * (3 + len(conds))]
        for name, row in m['metrics'].items():
            if 'full_rate' not in row:
                continue
            fr = row['full_rate']
            lines.append(f"| {name} | {fr['n']} | {fr['acc']:.2f} [{fr['ci'][0]:.2f}, {fr['ci'][1]:.2f}] | "
                         + ' | '.join(_cell(row.get(c)) for c in conds) + ' |')
        lines += ['', '### Splits (uncorrected p)', '', '| split | n | Full | ' + ' | '.join(LABELS[c] for c in conds) + ' |',
                  '|' + ' --- |' * (3 + len(conds))]
        for name, row in m['splits'].items():
            first = next((v for v in row.values() if v), None)
            if not first:
                continue
            lines.append(f"| {name} | {first['n']} | {first['full']['acc']:.2f} | "
                         + ' | '.join(_cell(row.get(c), False) for c in conds) + ' |')
        lines += ['', '### AUROC of s = z_Yes - z_No (difference to Full, bootstrap over records)', '',
                  '| set | Full | ' + ' | '.join(LABELS[c] for c in conds) + ' |', '|' + ' --- |' * (2 + len(conds))]
        for key, row in m['auroc'].items():
            if not row.get('full'):
                continue
            cells = []
            for c in conds:
                v = row.get(c)
                cells.append('–' if not v else f"{v['auroc_cond']:.3f} ({v['diff']['diff']:+.3f} "
                             f"[{v['diff']['ci'][0]:+.3f}, {v['diff']['ci'][1]:+.3f}]; mean s {v['mean_s_cond']:+.2f})")
            lines.append(f"| {key} | {row['full']['auroc_full']:.3f} (mean s {row['full']['mean_s_full']:+.2f}) | "
                         + ' | '.join(cells) + ' |')
        lines += ['', '### Point answers: pixel-only hits and chance of the either-convention metric', '',
                  '| condition | task | grid | n | strict hit | pixel-only hit | either | chance (either) |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for c, rows in m['point_conventions'].items():
            for r in rows:
                lines.append(f"| {LABELS[c]} | {r['task']} | {r['grid']} | {r['n']} | {r['strict']:.2f} | "
                             f"{r['pixel_only']:.2f} | {r['any']:.2f} | {r['chance']:.3f} |")
        lines += ['', '### Answer validity (location tasks at 10x10 and 20x20, RefCOCO)', '',
                  '| condition | answers | valid | schema | truncated | other |', '| --- | --- | --- | --- | --- | --- |']
        for c, rows in m['failures'].items():
            for key, cnt in sorted(rows.items()):
                other = sum(v for k, v in cnt.items() if k not in ('valid', 'schema', 'truncated'))
                lines.append(f"| {LABELS[c]} | {key} | {cnt.get('valid', 0)} | {cnt.get('schema', 0)} | "
                             f"{cnt.get('truncated', 0)} | {other} |")
        lines += ['', '### Coordinate calibration of valid point answers (pred = slope * true + intercept, canvas fractions)',
                  '', '| condition | task | grid | n valid | x slope | x intercept | x r | y slope | y intercept | y r |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for c, rows in m['calibration'].items():
            for r in rows:
                lines.append(f"| {LABELS[c]} | {r['task']} | {r['grid']} | {r['n_valid']} | "
                             + ' | '.join(f"{r[a][k]:.2f}" for a in 'xy' for k in ('slope', 'intercept', 'r')) + ' |')
        lines += ['', '### Row/column answers of T5 and T6 (signed errors of valid answers)', '',
                  '| condition | grid | n | invalid | exact | one under (row, column or both) | mean row error | mean column error |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for c, grids in m['rowcol'].items():
            for g, rc in grids.items():
                if rc['n']:
                    lines.append(f"| {LABELS[c]} | {g} | {rc['n']} | {rc['invalid']} | {rc['exact']} | {rc['one_under']} | "
                                 f"{rc['mean_d_row']:+.2f} | {rc['mean_d_col']:+.2f} |")
        lines += ['', '### Median centre distance of point answers (cells)', '',
                  '| condition | 10x10 | 20x20 |', '| --- | --- | --- |']
        for c, d in m['distance'].items():
            lines.append(f"| {LABELS[c]} | {d[10]['median']} (n={d[10]['n_valid']}) | {d[20]['median']} "
                         f"(n={d[20]['n_valid']}) |")
        lines.append('')
    path.write_text('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------- figures


def plot_rates(summary: dict, model: str, out: Path) -> None:
    """Accuracy per condition and metric, Wilson 95% CI, conditions as dodged dots."""
    plt = _style()
    m = summary['models'][model]
    names = [n for n in METRICS if 'full_rate' in m['metrics'][n] and 'either' not in n]   # either = strict here, see tables
    conds = [c for c in ('full',) + CONTRASTS if c in m['conditions']]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 1.0))
    ys = np.arange(len(names))[::-1]
    for j, c in enumerate(conds):
        off = (j - (len(conds) - 1) / 2) * 0.17
        for y, n in zip(ys, names):
            st = m['metrics'][n]['full_rate'] if c == 'full' else (m['metrics'][n].get(c) or {}).get('cond')
            if not st:
                continue
            ax.plot([st['ci'][0], st['ci'][1]], [y + off] * 2, color=COLORS[c], lw=1.2)
            ax.plot(st['acc'], y + off, 'o', ms=5, color=COLORS[c], mec='white', mew=0.8,
                    label=LABELS[c] if n == names[0] else None)
    ax.set_yticks(ys, [f"{n} (n={m['metrics'][n]['full_rate']['n']})" for n in names])
    for y in ys[:-1]:
        ax.axhline(y - 0.5, color=GRID_INK, lw=0.6)
    if 'RefCOCO a>=5%' in names:      # separates fine-grained metrics from the general-capability controls
        ax.axhline(ys[names.index('RefCOCO a>=5%')] + 0.5, color=MUTED, lw=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('accuracy (Wilson 95% CI)')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0 + 0.9 / (0.42 * len(names) + 1.0)), ncol=len(conds),
              fontsize=8, handletextpad=0.2)
    ax.set_title(model, loc='left', pad=22)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_deltas(summary: dict, model: str, out: Path, conds: tuple[str, ...], title: str) -> None:
    """Difference to Full with paired bootstrap 95% CI; filled markers = Holm/uncorrected p < 0.05."""
    plt = _style()
    m = summary['models'][model]
    names = [n for n in METRICS if 'full_rate' in m['metrics'][n] and 'either' not in n]   # either = strict here, see tables
    conds = [c for c in conds if c in m['conditions']]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 1.0))
    ys = np.arange(len(names))[::-1]
    markers = {'off': 'o', 'blur': 's', 'mean': 'D', 'off1': 'v', 'off2': 's', 'off3': '^'}
    shades = {'off1': '#a79df0', 'off2': '#6250d6', 'off3': '#3b2c9c'}
    for j, c in enumerate(conds):
        off = (j - (len(conds) - 1) / 2) * 0.2
        color = shades.get(c, COLORS[c])
        for y, n in zip(ys, names):
            st = m['metrics'][n].get(c)
            if not st:
                continue
            p = st.get('p_holm', st['p'])
            ax.plot(st['delta_ci'], [y + off] * 2, color=color, lw=1.2)
            ax.plot(st['delta'], y + off, markers[c], ms=6, color=color if p < 0.05 else 'white', mec=color, mew=1.2,
                    label=LABELS[c] if n == names[0] else None)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(ys, names)
    for y in ys[:-1]:
        ax.axhline(y - 0.5, color=GRID_INK, lw=0.6)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('accuracy difference to Full (paired bootstrap 95% CI); filled = p < 0.05')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0 + 0.9 / (0.42 * len(names) + 1.0)), ncol=len(conds),
              fontsize=8, handletextpad=0.2)
    ax.set_title(title, loc='left', pad=22)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_auroc_levels(summary: dict, model: str, out: Path) -> None:
    plt = _style()
    m = summary['models'][model]
    conds = [c for c in ('full',) + CONTRASTS if c in m['conditions']]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, task in zip(axes, ('T1', 'T2')):
        for c in conds:
            vals = [(m['auroc'].get(f'{task} L{lv}') or {}).get(c) for lv in (2, 3)]
            if any(v is None for v in vals):
                continue
            ys = [v['auroc_full'] if c == 'full' else v['auroc_cond'] for v in vals]
            ax.plot([10, 20], ys, '-o', color=COLORS[c], lw=2, ms=6, label=LABELS[c])
            ax.annotate(LABELS[c], (20, ys[-1]), xytext=(5, 0), textcoords='offset points', va='center',
                        fontsize=8, color=INK)
        ax.set_xticks([10, 20], ['ΔE 10', 'ΔE 20'])
        ax.set_xlim(7, 25)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_title(f'{task} colour', loc='left')
    axes[0].set_ylabel('AUROC of s')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_rowcol(summary: dict, model: str, out: Path) -> None:
    plt = _style()
    m = summary['models'][model]
    conds = [c for c in ('full',) + CONTRASTS if c in m['conditions']]
    fig, axes = plt.subplots(2, len(conds), figsize=(1.9 * len(conds) + 0.6, 4.2))
    for i, g in enumerate((10, 20)):
        for j, c in enumerate(conds):
            ax = axes[i, j]
            rc = m['rowcol'][c][g]
            h = np.array(rc['hist'], float)
            frac = h / max(rc['n'], 1)
            ax.imshow(frac, cmap='Blues', vmin=0, vmax=0.6)
            for (r, k), v in np.ndenumerate(frac):
                if h[r, k]:
                    ax.text(k, r, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                            color='white' if v > 0.35 else INK)
            ax.set_xticks(range(5), ['≤-2', '-1', '0', '1', '≥2'], fontsize=6.5)
            ax.set_yticks(range(5), ['≤-2', '-1', '0', '1', '≥2'], fontsize=6.5)
            ax.grid(False)
            ax.set_title(f"{LABELS[c]} {g}x{g}\nn={rc['n']}, invalid {rc['invalid']}", fontsize=8)
            if j == 0:
                ax.set_ylabel('row error')
            if i == 1:
                ax.set_xlabel('column error')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_coord_scatter(summary: dict, model: str, out: Path, task: str = 'T6') -> None:
    """Predicted against true target centre (x and y pooled, canvas fractions) per condition and grid."""
    plt = _style()
    m = summary['models'][model]
    conds = [c for c in ('full',) + CONTRASTS if c in m['conditions']]
    fig, axes = plt.subplots(2, len(conds), figsize=(1.85 * len(conds) + 0.5, 4.0), sharex=True, sharey=True)
    for i, g in enumerate((10, 20)):
        for j, c in enumerate(conds):
            ax = axes[i, j]
            true, pred = (np.array(v) for v in m['coords'][c][f'{task}|{g}'])
            ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=':')
            if len(true):
                ax.scatter(true[:, 0], pred[:, 0], s=9, color=COLORS[c], alpha=0.75, lw=0, label='x')
                ax.scatter(true[:, 1], pred[:, 1], s=9, facecolor='none', edgecolor=COLORS[c], lw=0.7, label='y')
            cal = next((r for r in m['calibration'][c] if r['task'] == task and r['grid'] == g), None)
            text = (f"slope x {cal['x']['slope']:.2f}, y {cal['y']['slope']:.2f}\nn valid {cal['n_valid']}"
                    if cal else 'n valid < 3')
            ax.text(0.03, 0.97, text, transform=ax.transAxes, va='top', fontsize=6.5, color=INK)
            ax.set_xlim(0, 1), ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f'{LABELS[c]}, {g}x{g}', fontsize=8)
            if i == 1:
                ax.set_xlabel('true centre', fontsize=8)
            if j == 0:
                ax.set_ylabel('predicted centre', fontsize=8)
    axes[0, 0].legend(fontsize=7, loc='lower right', handletextpad=0.1)
    fig.suptitle(f'{model}: {task} point answers (filled = x, open = y)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_inputs(output: Path, out: Path) -> None:
    """Example inputs and their blurred copies (the blur condition recomputes DeepStack from the latter)."""
    plt = _style()
    cases = read_jsonl(output / 'cases.jsonl')
    pick = [next(c for c in cases if c['task'] == 'T1' and c['level'] == 3 and not c['truth']['answer_yes']),
            next(c for c in cases if c['task'] == 'T2' and c['level'] == 3 and c['variant'] == 'neg'),
            next(c for c in cases if c['task'] == 'T6' and c['grid'] == 20 and c['format'] == 'point')]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0))
    for j, c in enumerate(pick):
        im = Image.open(output / c['image']).convert('RGB')
        crop = None
        if c['task'] == 'T6':
            x, y = c['truth']['target_center']
            crop = (int(x) - 96, int(y) - 96, int(x) + 96, int(y) + 96)
        for i, shown in enumerate((im, blurred(im))):
            ax = axes[i, j]
            ax.imshow(shown.crop(crop) if crop else shown)
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            label = f"{c['task']} {'colour ΔE 20' if c['task'] != 'T6' else '20x20 (crop around target)'}"
            ax.set_title(('original' if i == 0 else f'blurred, sigma={BLUR_SIGMA:g} px') + f'\n{label}', fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--end-to-end', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    out = a.output / 'analysis'
    out.mkdir(exist_ok=True)
    summary = analyze(a.output, a.end_to_end, a.n_boot, a.seed)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    write_tables(summary, out / 'tables.md')
    figs = {'inputs.png': lambda f: plot_inputs(a.output, f)}
    for model in summary['models']:
        s = slug(model)
        figs[f'rates_{s}.png'] = lambda f, model=model: plot_rates(summary, model, f)
        figs[f'deltas_{s}.png'] = lambda f, model=model: plot_deltas(summary, model, f, CONTRASTS, model)
        figs[f'auroc_{s}.png'] = lambda f, model=model: plot_auroc_levels(summary, model, f)
        figs[f'rowcol_{s}.png'] = lambda f, model=model: plot_rowcol(summary, model, f)
        figs[f'coords_{s}.png'] = lambda f, model=model: plot_coord_scatter(summary, model, f)
        if any(c in summary['models'][model]['conditions'] for c in SINGLE):
            figs[f'deltas_single_{s}.png'] = lambda f, model=model: plot_deltas(summary, model, f, SINGLE,
                                                                               f'{model}: single-layer Off')
    a.figures.mkdir(parents=True, exist_ok=True)
    for name, fn in figs.items():
        fn(out / name)
        shutil.copy(out / name, a.figures / name)
    print((out / 'tables.md').read_text()[:4000])


if __name__ == '__main__':
    main()
