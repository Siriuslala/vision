"""Summaries and figures of experiment 1.4 (ordinal offset and the visual token grid).

Reads the files written by ``ordinal_token_grid.py`` (cases, scores, mech records, convention
columns) and writes analysis/summary.json, analysis/tables.md and analysis/figures/*.png under the
run directory.  Location answers use the content-diagnostic scores (as in experiment 1.3); strict
accuracies are tabulated alongside.  Earlier runs are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from analysis_tools.behavior.decomposition import answer_part, parse_json, read_jsonl
from analysis_tools.behavior.decomposition_analysis import INK, MUTED, SERIES, _style, wilson
from analysis_tools.behavior.ordinal_token_grid import (
    GridSpec, POINTING_HEADS, ROWS, parse_condition, render,
)

TOKEN = 32
CELLS = (32, 48, 64, 80, 96)
COUNTS = (10, 16, 20, 24)


# ----------------------------------------------------------------------------- loading

def ordinal_offsets(case: dict, score: dict, response: str) -> tuple[int | None, int | None]:
    """(d_row, d_col) of a row/column answer, including answers outside the grid (e.g. column n+1).

    The diagnostic score marks out-of-range answers as invalid; for the offset analysis the parsed
    ordinals are kept, because column n+1 for a target in column n is exactly the +1 error.
    """
    if score.get('valid'):
        return score['d_row'], score['d_col']
    if score.get('failure') != 'out_of_range':
        return None, None
    value = parse_json(answer_part(response))
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    tr, tc = case['truth']['target_rc']
    return value['row'] - 1 - tr, value['column'] - 1 - tc


def load(output: Path) -> list[dict]:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    strict = {s['id']: s for s in read_jsonl(output / 'scores_strict.jsonl')}
    responses = {r['id']: r['response'] for p in sorted(output.glob('results*.jsonl')) for r in read_jsonl(p)}
    rows = []
    for s in read_jsonl(output / 'scores_diagnostic.jsonl'):
        case = cases[s['id']]
        dr, dc = ordinal_offsets(case, s, responses[s['id']]) if case['format'] == 'row_column' else (None, None)
        rows.append({**case, 'score': s, 'strict': strict[s['id']], 'dr': dr, 'dc': dc})
    missing = len(cases) - len(rows)
    if missing:
        print(f'WARNING: {missing} cases have no result yet')
    return rows


def prop(k: int, n: int) -> dict:
    return {'k': int(k), 'n': int(n), 'p': k / n if n else float('nan'), 'ci': list(wilson(k, n))}


def ptxt(s: dict, ci: bool = True) -> str:
    if not s['n']:
        return '–'
    return f"{s['p']:.2f}" + (f" [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]" if ci else '')




# ----------------------------------------------------------------------------- per-condition summary

def condition_summary(rows: list[dict]) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r['condition']][(r['task'], r['format'])].append(r)
    out = {}
    for name, g in by.items():
        cell, n, phase = parse_condition(name)
        rc, pt, t0 = g[('T6', 'row_column')], g[('T6', 'point')], g[('T0', 'count')]
        valid = [r for r in rc if r['score'].get('valid')]
        parsed = [r for r in rc if r['dc'] is not None]
        dc = Counter(r['dc'] for r in parsed)
        dr = Counter(r['dr'] for r in parsed)
        col_only = [r for r in parsed if r['dr'] == 0 and r['dc'] != 0]
        onset = None
        pc = per_column_rate(rc)
        for j in sorted(pc):
            if pc[j] >= 0.2:
                onset = j
                break
        cols_ans = Counter(r['score'].get('pred_cols') for r in t0)
        rows_ans = Counter(r['score'].get('pred_rows') for r in t0)
        pt_valid = [r for r in pt if r['score'].get('valid')]
        pt_dx = [(r['score']['pred_center'][0] - r['truth']['target_center'][0]) / cell for r in pt_valid]
        pt_col = [r['score']['pred_rc'][1] - r['truth']['target_rc'][1] for r in pt_valid if r['score'].get('pred_rc')]
        conv = [r['score']['convention'] for r in pt]
        out[name] = {
            'cell': cell, 'n': n, 'phase': phase, 'rows': rc[0]['rows'] if rc else ROWS, 'axis_pixels': n * cell, 'tokens_per_instance': cell / TOKEN,
            'canvas': rc[0]['truth']['canvas'] if rc else None,
            'T6_rowcol_acc': prop(sum(r['score']['correct'] for r in rc), len(rc)),
            'T6_rowcol_acc_strict': prop(sum(r['strict']['correct'] for r in rc), len(rc)),
            'T6_col_acc': prop(dc[0], len(rc)), 'T6_row_acc': prop(dr[0], len(rc)),
            'T6_col_plus1': prop(dc[1], len(rc)), 'T6_col_minus1': prop(dc[-1], len(rc)),
            'T6_col_other_err': prop(sum(v for k, v in dc.items() if k not in (-1, 0, 1)), len(rc)),
            'T6_rowcol_invalid': len(rc) - len(valid), 'T6_rowcol_unparsed': len(rc) - len(parsed),
            'T6_col_out_of_range': sum(r['score'].get('failure') == 'out_of_range' for r in rc),
            'T6_row_plus1': prop(dr[1], len(rc)), 'T6_row_minus1': prop(dr[-1], len(rc)),
            'T6_col_only_errors': len(col_only), 'T6_col_only_plus1': sum(r['dc'] == 1 for r in col_only),
            'T6_mean_dcol': float(np.mean([r['dc'] for r in parsed])) if parsed else float('nan'),
            'onset_col': onset + 1 if onset is not None else None,
            'onset_token': (rc[0]['truth']['grid_origin'][0] + (onset + 0.5) * cell) / TOKEN if onset is not None else None,
            'T6_dcol_hist': {str(k): v for k, v in sorted(dc.items())},
            'T6_point_hit': prop(sum(bool(r['score'].get('hit')) for r in pt), len(pt)),
            'T6_point_hit_strict': prop(sum(bool(r['strict'].get('hit')) for r in pt), len(pt)),
            'T6_point_col_plus1': prop(sum(d == 1 for d in pt_col), len(pt)),
            'T6_point_col_acc': prop(sum(d == 0 for d in pt_col), len(pt)),
            'T6_point_dx_median_cells': float(np.median(pt_dx)) if pt_dx else float('nan'),
            'T6_point_pixel_only': prop(sum(c['pixel_only'] for c in conv), len(conv)),
            'T6_point_any': prop(sum(c['h_any'] for c in conv), len(conv)),
            'T0_cols_correct': prop(cols_ans[n], len(t0)), 'T0_rows_correct': prop(rows_ans[t0[0]['rows']] if t0 else 0, len(t0)),
            'T0_cols_answers': {str(k): v for k, v in sorted(cols_ans.items(), key=lambda kv: str(kv[0]))},
            'T0_rows_answers': {str(k): v for k, v in sorted(rows_ans.items(), key=lambda kv: str(kv[0]))},
        }
    return out


def per_column_rate(rc: list[dict]) -> dict:
    """Zero-based target column -> P(column answer = true + 1) for row/column cases."""
    by = defaultdict(list)
    for r in rc:
        by[r['truth']['target_rc'][1]].append(r['dc'] == 1)
    return {j: float(np.mean(v)) for j, v in by.items()}


def per_column(rows: list[dict]) -> dict:
    """condition -> per target column: n, +1 rate, column accuracy (row/column format), point column +1."""
    by = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'plus1': 0, 'col_ok': 0, 'pt_n': 0, 'pt_plus1': 0,
                                                    'pt_ok': 0, 'pt_dx': []}))
    for r in rows:
        if r['task'] != 'T6':
            continue
        j = r['truth']['target_rc'][1]
        e = by[r['condition']][j]
        s = r['score']
        if r['format'] == 'row_column':
            e['n'] += 1
            e['plus1'] += int(r['dc'] == 1)
            e['col_ok'] += int(r['dc'] == 0)
            e['minus1'] = e.get('minus1', 0) + int(r['dc'] == -1)
        else:
            e['pt_n'] += 1
            pr = s.get('pred_rc') if s.get('valid') else None
            e['pt_plus1'] += int(pr is not None and pr[1] - j == 1)
            e['pt_ok'] += int(pr is not None and pr[1] == j)
            if s.get('valid'):
                e['pt_dx'].append((s['pred_center'][0] - r['truth']['target_center'][0]) / r['cell'])
    out = {}
    for name, cols in by.items():
        out[name] = {}
        for j, e in sorted(cols.items()):
            out[name][j] = {'n': e['n'], 'plus1': e['plus1'] / max(e['n'], 1), 'col_acc': e['col_ok'] / max(e['n'], 1),
                            'minus1': e.get('minus1', 0) / max(e['n'], 1),
                            'pt_n': e['pt_n'], 'pt_plus1': e['pt_plus1'] / max(e['pt_n'], 1),
                            'pt_col_acc': e['pt_ok'] / max(e['pt_n'], 1),
                            'pt_dx_median': float(np.median(e['pt_dx'])) if e['pt_dx'] else float('nan')}
    return out


def origin_x(rows: list[dict]) -> dict:
    return {r['condition']: r['truth']['grid_origin'][0] for r in rows}


# ----------------------------------------------------------------------------- logistic model

def logistic(X: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood logistic regression (IRLS); returns coefficients and standard errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        w = p * (1 - p) + 1e-9
        H = X.T @ (X * w[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    return beta, np.sqrt(np.diag(cov))


def factor_model(rows: list[dict]) -> dict:
    """Case-level +1 indicator ~ log2 n + log2 cell on the phase-0 factorial.

    Axis length L = n * cell, so log L = log n + log cell: H2 (axis length) predicts equal slopes,
    H1 (tokens per instance) a zero slope for n, H3 (instance count) a zero slope for cell.
    """
    data = [r for r in rows if r['task'] == 'T6' and r['format'] == 'row_column' and r['phase'] == 0 and r['rows'] == ROWS]
    if len({r['n'] for r in data}) < 2 or len({r['cell'] for r in data}) < 2:
        return {'n_cases': len(data), 'note': 'factorial incomplete'}
    y = np.array([float(r['dc'] == 1) for r in data])
    X = np.array([[1.0, math.log2(r['n']), math.log2(r['cell'])] for r in data])
    beta, se = logistic(X, y)
    diff = beta[1] - beta[2]
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    se_diff = math.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
    return {'n_cases': len(data), 'coef': {'intercept': beta[0], 'log2_n': beta[1], 'log2_cell': beta[2]},
            'se': {'intercept': se[0], 'log2_n': se[1], 'log2_cell': se[2]},
            'slope_difference_n_minus_cell': diff, 'se_difference': se_diff}


# ----------------------------------------------------------------------------- mechanism

def mech_summary(output: Path, rows: list[dict]) -> dict:
    recs = [r for p in sorted((output / 'mech').glob('records*.jsonl')) for r in read_jsonl(p)]
    if not recs:
        return {}
    cases = {r['id']: r for r in rows}
    out = {'n_records': len(recs), 'n_unparsed': sum(r['status'] != 'ok' for r in recs), 'conditions': {}}
    by = defaultdict(list)
    for r in recs:
        if r['status'] == 'ok':
            by[r['condition']].append(r)
    heads = [f'L{l}H{h}' for l, h in POINTING_HEADS]
    for name, rs in sorted(by.items()):
        cell, n, phase = parse_condition(name)
        groups = {'correct': [], 'plus1': [], 'other': []}
        margin_by_col = defaultdict(list)
        for r in rs:
            tc, pc = r['true_col'], r['pred_col']
            g = 'correct' if pc == tc else 'plus1' if pc == tc + 1 else 'other'
            lp = {int(k): v for k, v in r['logp'].items()}
            margin = lp.get(tc + 1, float('nan')) - lp[tc]
            margin_by_col[tc].append(margin)
            case = cases[r['id']]
            ox = case['truth']['grid_origin'][0]
            # Attention mass per merged-token column, averaged over the pointing heads, re-indexed
            # relative to the token column that holds the target instance's centre.
            prof = np.mean([r['heads'][h]['col_mass'] for h in heads], 0)
            prof = prof / max(prof.sum(), 1e-12)
            centre_tok = int((ox + (tc - 0.5) * cell) // TOKEN)
            peak = int(np.argmax(prof))
            # Fractional instance index at the attention peak: which instance column the peak lies in.
            peak_inst = ((peak + 0.5) * TOKEN - ox) / cell
            left_tok = int((ox + (tc - 1) * cell) // TOKEN)
            groups[g].append({'margin': margin, 'profile': prof.tolist(), 'centre_tok': centre_tok, 'left_tok': left_tok,
                              'peak_offset_tokens': peak - centre_tok, 'peak_instance_offset': peak_inst - (tc - 0.5),
                              'mass_target_cols': float(prof[max(centre_tok - 1, 0):centre_tok + 2].sum())})
        summ = {}
        for g, items in groups.items():
            if not items:
                summ[g] = {'n': 0}
                continue
            m = np.array([i['margin'] for i in items])
            summ[g] = {'n': len(items), 'margin_mean': float(np.nanmean(m)),
                       'peak_offset_tokens_median': float(np.median([i['peak_offset_tokens'] for i in items])),
                       'peak_instance_offset_mean': float(np.mean([i['peak_instance_offset'] for i in items])),
                       'peak_on_target_instance': float(np.mean([abs(i['peak_instance_offset']) <= 0.5 for i in items]))}
        out['conditions'][name] = {'groups': summ, 'items': groups,
                                   'margin_by_col': {int(k): float(np.nanmean(v)) for k, v in sorted(margin_by_col.items())}}
    return out


# ----------------------------------------------------------------------------- convention diagnostic

def convention_table(rows: list[dict], key) -> list[dict]:
    """Rows of (group, n, h_norm, h_pix, pixel_only, h_any, chance, shuffled baseline)."""
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    out = []
    for g, rs in sorted(by.items(), key=lambda kv: str(kv[0])):
        n = len(rs)
        stat = {k: prop(sum(r[k] for r in rs), n) for k in ('h_norm', 'h_pix', 'pixel_only', 'h_any')}
        chance = {k: float(np.mean([r[k] for r in rs])) for k in ('p_norm', 'p_pix', 'p_any')}
        # Shuffled baseline: every valid prediction paired with the targets of the other images in the group.
        pts = [(r['point'], r['canvas'], r['box']) for r in rs if r['point'] is not None]
        hits, pairs = 0, 0
        for i, (p, canvas, _) in enumerate(pts):
            for j, (_, canvas_j, box) in enumerate(pts):
                if i == j or canvas_j != canvas:
                    continue
                W, H = canvas
                x, y = p
                norm = 0 <= x <= 1000 and 0 <= y <= 1000 and box[0] <= x * W / 1000 < box[2] and box[1] <= y * H / 1000 < box[3]
                pix = box[0] <= x < box[2] and box[1] <= y < box[3]
                hits += norm or pix
                pairs += 1
        canvases = sorted({tuple(r['canvas']) for r in rs})
        out.append({'group': g, 'n': n, **stat, 'chance': chance,
                    'shuffled_any': hits / pairs if pairs else float('nan'),
                    'canvas': [list(c) for c in canvases]})
    return out


def convention_all(output: Path, formal: Path, supplement: Path, rows: list[dict]) -> dict:
    dest = output / 'convention'
    res = {}
    def attach(conv_rows, cases):
        out = []
        for c in conv_rows:
            case = cases[c['id']]
            out.append({**c, 'case': case, 'canvas': case['truth']['canvas'], 'box': case['truth']['target_cell_box']})
        return out
    fcases = {c['id']: c for c in read_jsonl(formal / 'cases.jsonl')}
    f = attach(read_jsonl(dest / 'formal_main.jsonl'), fcases)
    res['formal'] = convention_table(f, lambda r: (r['case']['task'], f"{r['case']['grid']:02d}x{r['case']['grid']:02d}"))
    for part in ('layouts', 'ring'):
        cases = {c['id']: c for c in read_jsonl(supplement / f'cases_{part}.jsonl')}
        s = attach(read_jsonl(dest / f'supplement_{part}.jsonl'), cases)
        res[f'supplement_{part}'] = convention_table(s, lambda r: (r['case']['task'], r['case']['layout']))
    new = [{**r['score']['convention'], 'case': r, 'canvas': r['truth']['canvas'], 'box': r['truth']['target_cell_box']}
           for r in rows if r['format'] == 'point']
    res['ordinal'] = convention_table(new, lambda r: ('T6', r['case']['condition']))
    return res


# ----------------------------------------------------------------------------- figures

def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_token_grid(out: Path) -> None:
    """Example crops with the 16-pixel patch grid and the 32-pixel merged-token grid."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    examples = [(32, 0), (48, 0), (64, 0), (80, 0), (96, 0), (64, 8), (64, 16), (32, 16)]
    width, height = 384, 192
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 8.6))
    axes = axes.T.ravel()
    for ax, (cell, phase) in zip(axes, examples):
        spec = GridSpec('person', ROWS, 20, cell, phase, (0, 1))
        img, meta = render(spec)
        ox, oy = meta['grid_origin']
        x0, y0 = (ox // TOKEN) * TOKEN, (oy // TOKEN) * TOKEN
        crop = np.asarray(img)[y0:y0 + height, x0:x0 + width]
        ax.imshow(crop, extent=(0, crop.shape[1], crop.shape[0], 0), interpolation='nearest')
        for x in range(0, width + 1, 16):
            major = x % TOKEN == 0
            ax.axvline(x, color='#2a78d6' if major else '#86b6ef', lw=1.1 if major else 0.5,
                       ls='-' if major else ':', zorder=3)
        for y in range(0, height + 1, 16):
            major = y % TOKEN == 0
            ax.axhline(y, color='#2a78d6' if major else '#86b6ef', lw=1.1 if major else 0.5,
                       ls='-' if major else ':', zorder=3)
        for c in range(20):
            for r in range(ROWS):
                bx, by = ox + c * cell - x0, oy + r * cell - y0
                if bx < width and by < height:
                    ax.add_patch(Rectangle((bx, by), cell, cell, fill=False, ec='#eb6834', lw=1.3, ls='--', zorder=4))
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xticks(range(0, width + 1, TOKEN))
        ax.set_xticklabels([str(i) for i in range(0, width // TOKEN + 1)], fontsize=7)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f'cell {cell} px = {cell / TOKEN:g} merged tokens, phase {phase} px', loc='left', fontsize=9)
    fig.supxlabel('merged-token column counted from the first token of the grid (32 px each); solid blue = merged-token '
                  'borders, dotted blue = 16-px patch borders, orange dashed = instance cells', fontsize=9)
    fig.tight_layout()
    save(fig, out / 'token_grid_examples.png')


def fig_factorial(summary: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    panels = [('T6_col_plus1', 'T6 row/column: column ordinal +1'), ('T6_col_acc', 'T6 row/column: column correct'),
              ('T0_cols_correct', 'T0: column count correct'), ('T6_point_hit', 'T6 point: hit (0-1000)')]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (key, title) in zip(axes, panels):
        M = np.full((len(CELLS), len(COUNTS)), np.nan)
        for i, c in enumerate(CELLS):
            for j, n in enumerate(COUNTS):
                s = summary.get(f'c{c}n{n}p0')
                if s:
                    M[i, j] = s[key]['p']
        ax.imshow(M, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for i in range(len(CELLS)):
            for j in range(len(COUNTS)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:.2f}\n{CELLS[i] * COUNTS[j]}px', ha='center', va='center', fontsize=7.5,
                            color='white' if M[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(COUNTS)), [str(n) for n in COUNTS])
        ax.set_yticks(range(len(CELLS)), [f'{c} ({c / TOKEN:g} tok)' for c in CELLS])
        ax.set_xlabel('instances on the long axis (n)')
        ax.set_title(title, fontsize=9)
        ax.grid(False)
    axes[0].set_ylabel('cell size, px')
    fig.tight_layout()
    save(fig, out / 'factorial_heatmaps.png')


def fig_columns_aligned(percol: dict, origins: dict, summary: dict, out: Path, key: str = 'plus1',
                        name: str = 'column_error_token_aligned.png', label: str = 'P(column answer = true + 1)') -> None:
    """Per-column rates drawn over each column's pixel extent, x in merged-token units."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import cm, colors
    conds = [f'c{c}n{n}p0' for n in COUNTS for c in CELLS] + \
            ['c64n16p8', 'c64n16p16', 'c64n20p8', 'c64n20p16', 'c64n24p8', 'c64n24p16', 'c32n20p16', 'c32n24p16',
             'c32n20p0r10', 'c32n20p0r20', 'c64n20p0r10', 'c64n20p0r20']
    conds = [c for c in conds if c in percol]
    cmap, norm = plt.get_cmap('Blues'), colors.Normalize(0, 1)
    max_tok = max(summary[c]['canvas'][0] for c in conds) / TOKEN
    fig, ax = plt.subplots(figsize=(13, 0.32 * len(conds) + 1.2))
    for i, c in enumerate(conds):
        cell, n, phase = parse_condition(c)
        ox = origins[c]
        for j in range(n):
            v = percol[c].get(j, {}).get(key, np.nan)
            ax.add_patch(Rectangle(((ox + j * cell) / TOKEN, i - 0.42), cell / TOKEN, 0.84,
                                   fc=cmap(norm(v)) if not np.isnan(v) else '#eeeeee', ec='white', lw=0.6))
        ax.add_patch(Rectangle((0, i - 0.45), summary[c]['canvas'][0] / TOKEN, 0.9, fill=False, ec=MUTED, lw=0.4))
    for x in range(int(max_tok) + 1):
        ax.axvline(x, color='#d9d8d4', lw=0.4, zorder=0)
    ax.set_yticks(range(len(conds)), [f"{c}  ({parse_condition(c)[0] / TOKEN:g} tok/inst)" for c in conds], fontsize=7)
    ax.set_ylim(len(conds) - 0.5, -0.5)
    ax.set_xlim(0, max_tok)
    ax.set_xlabel('horizontal position, merged-token columns (32 px each); each box is one instance column')
    ax.grid(False)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label=label)
    fig.tight_layout()
    save(fig, out / name)


def fig_column_profiles(percol: dict, origins: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for k, n in enumerate(COUNTS):
        for row, xmode in enumerate(('index', 'token')):
            ax = axes[row, k]
            for i, c in enumerate(CELLS):
                name = f'c{c}n{n}p0'
                if name not in percol:
                    continue
                js = sorted(percol[name])
                y = [percol[name][j]['plus1'] for j in js]
                x = [j + 1 for j in js] if xmode == 'index' else [(origins[name] + (j + 0.5) * c) / TOKEN for j in js]
                ax.plot(x, y, '-o', ms=3, lw=1.5, color=SERIES[i], label=f'{c} px')
            ax.set_title(f'n = {n}', fontsize=9)
            ax.set_xlabel('target column (1-based)' if xmode == 'index' else 'target centre, merged-token column')
        axes[0, 0].set_ylabel('P(column +1), row/column answer')
        axes[1, 0].set_ylabel('P(column +1), row/column answer')
    axes[0, -1].legend(title='cell', fontsize=7)
    fig.tight_layout()
    save(fig, out / 'column_profiles.png')


def fig_phase(summary: dict, percol: dict, origins: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    for i, phase in enumerate((0, 8, 16)):
        xs, ys, lo, hi = [], [], [], []
        for n in (16, 20, 24):
            s = summary.get(f'c64n{n}p{phase}')
            if s:
                xs.append(n + (i - 1) * 0.6)
                ys.append(s['T6_col_plus1']['p'])
                lo.append(s['T6_col_plus1']['p'] - s['T6_col_plus1']['ci'][0])
                hi.append(s['T6_col_plus1']['ci'][1] - s['T6_col_plus1']['p'])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt='o', color=SERIES[i], label=f'64 px, phase {phase}', capsize=2)
    for i, n in enumerate((20, 24)):
        for phase, mk in ((0, 's'), (16, 'D')):
            s = summary.get(f'c32n{n}p{phase}')
            if s:
                ax.plot(n + 1.4 + 0.3 * (phase > 0), s['T6_col_plus1']['p'], mk, color=SERIES[3 + (phase > 0)],
                        label=f'32 px, phase {phase}' if i == 0 else None)
    ax.set_xticks((16, 20, 24))
    ax.set_xlabel('instances on the long axis (n)')
    ax.set_ylabel('P(column +1)')
    ax.legend(fontsize=7)
    ax = axes[1]
    for i, phase in enumerate((0, 8, 16)):
        name = f'c64n20p{phase}'
        if name in percol:
            js = sorted(percol[name])
            ax.plot([(origins[name] + (j + 0.5) * 64) / TOKEN for j in js], [percol[name][j]['plus1'] for j in js],
                    '-o', ms=3, color=SERIES[i], label=f'phase {phase}')
    ax.set_xlabel('target centre, merged-token column')
    ax.set_ylabel('P(column +1)')
    ax.set_title('c64, n = 20', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, out / 'phase_offsets.png')


def fig_mech(mech: dict, rows: list[dict], out: Path) -> None:
    """Left: teacher-forced log p(c+1) - log p(c) per true column.  Right: pointing-head attention
    per token column around the target, correct versus +1 answers, one panel per condition."""
    import matplotlib.pyplot as plt
    conds = list(mech.get('conditions', {}))
    if not conds:
        return
    main = [c for c in (f'c{x}n20p0' for x in CELLS) if c in conds]
    extra = [c for c in ('c64n20p16', 'c32n20p16', 'c64n20p0r20', 'c32n20p0r20') if c in conds]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, group, title in ((axes[0], main, '6 rows, phase 0'), (axes[1], extra, 'phase 16 and 20-row grids')):
        for i, c in enumerate(group):
            m = mech['conditions'][c]['margin_by_col']
            ax.plot(list(m), list(m.values()), '-o', ms=3, lw=1.6, color=SERIES[i], label=c)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel('true column (1-based)')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('mean log p(c+1) - log p(c)')
    fig.tight_layout()
    save(fig, out / 'mech_margin_by_column.png')

    origin = {r['condition']: r['truth']['grid_origin'][0] for r in rows}
    panels = main + extra
    ncol = 3
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.7 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, panels):
        cell = parse_condition(c)[0]
        ox = origin[c]
        for g, color in (('correct', SERIES[0]), ('plus1', SERIES[1])):
            items = mech['conditions'][c]['items'][g]
            if len(items) < 5:
                continue
            offs = np.arange(-8, 9)
            prof = []
            for o in offs:
                vals = [it['profile'][it['left_tok'] + o] for it in items if 0 <= it['left_tok'] + o < len(it['profile'])]
                prof.append(np.mean(vals) if vals else np.nan)
            ax.plot(offs + 0.5, prof, '-o', ms=2.5, lw=1.5, color=color, label=f'{g} (n={len(items)})')
        # Target instance extent and right neighbour, in token units relative to the target's first token.
        for k, shade in ((0, '#d9d8d4'), (1, '#f0efec')):
            x0 = (ox + 0) % TOKEN / TOKEN + k * cell / TOKEN
            ax.axvspan(x0, x0 + cell / TOKEN, color=shade, zorder=0)
        ax.set_title(f'{c} ({cell / TOKEN:g} tokens per instance)', fontsize=9)
        ax.set_xlim(-8, 9)
        ax.legend(fontsize=7)
    for ax in axes[len(panels):]:
        ax.axis('off')
    for ax in axes[-ncol:]:
        ax.set_xlabel('token offset from the first target token')
    for ax in axes[::ncol]:
        ax.set_ylabel('attention mass')
    fig.tight_layout()
    save(fig, out / 'mech_attention_profiles.png')


# ----------------------------------------------------------------------------- tables

def tables(summary: dict, model: dict, mech: dict, conv: dict, spot: dict) -> str:
    L = ['# Experiment 1.4 tables', '']
    L += ['## Per-condition summary (row/column and point: content-diagnostic scores)', '',
          '| condition | cell | n | rows | axis px | canvas | T0 cols ok | T0 col answers | T6 rc acc | strict | col +1 | col −1 | '
          'col other | col n+1 (out of range) | row +1 | row −1 | col-only err (+1) | onset col (token) | point hit | '
          'point col +1 | point dx median (cells) |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    order = sorted(summary, key=lambda c: (summary[c]['phase'] > 0, summary[c]['n'], summary[c]['cell'], summary[c]['phase']))
    for c in order:
        s = summary[c]
        onset = f"{s['onset_col']} ({s['onset_token']:.1f})" if s['onset_col'] else '–'
        L.append(f"| {c} | {s['cell']} | {s['n']} | {s['rows']} | {s['axis_pixels']} | {s['canvas'][0]}×{s['canvas'][1]} | "
                 f"{s['T0_cols_correct']['k']}/{s['T0_cols_correct']['n']} | {s['T0_cols_answers']} | "
                 f"{ptxt(s['T6_rowcol_acc'])} | {s['T6_rowcol_acc_strict']['p']:.2f} | {ptxt(s['T6_col_plus1'])} | "
                 f"{s['T6_col_minus1']['p']:.2f} | {s['T6_col_other_err']['p']:.2f} | {s['T6_col_out_of_range']} | "
                 f"{s['T6_row_plus1']['p']:.2f} | {s['T6_row_minus1']['p']:.2f} | "
                 f"{s['T6_col_only_errors']} ({s['T6_col_only_plus1']}) | {onset} | {ptxt(s['T6_point_hit'])} | "
                 f"{s['T6_point_col_plus1']['p']:.2f} | {s['T6_point_dx_median_cells']:+.2f} |")
    L += ['', '## Logistic model of the column +1 indicator (phase-0 factorial)', '', '```', json.dumps(model, indent=2), '```', '']
    if mech:
        L += ['## Mechanism (teacher forcing and pointing-head attention)', '',
              '| condition | group | n | mean log p(c+1) − log p(c) | median peak offset (tokens) | '
              'mean peak offset (instances) | peak on target instance |', '| --- | --- | --- | --- | --- | --- | --- |']
        for c, d in mech['conditions'].items():
            for g, s in d['groups'].items():
                if s['n']:
                    L.append(f"| {c} | {g} | {s['n']} | {s['margin_mean']:+.2f} | {s['peak_offset_tokens_median']:+.1f} | "
                             f"{s['peak_instance_offset_mean']:+.2f} | {s['peak_on_target_instance']:.2f} |")
        L.append('')
    L += ['## Coordinate convention diagnostic', '']
    for key, rows in conv.items():
        L += [f'### {key}', '', '| group | canvas | n | h_norm | pixel only | h_any | chance norm | chance pix | chance any | '
              'shuffled any |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in rows:
            canv = ', '.join(f'{w}×{h}' for w, h in r['canvas'])
            L.append(f"| {r['group']} | {canv} | {r['n']} | {r['h_norm']['p']:.3f} | {r['pixel_only']['p']:.3f} "
                     f"[{max(r['pixel_only']['ci'][0], 0.):.3f}, {r['pixel_only']['ci'][1]:.3f}] | {r['h_any']['p']:.3f} | "
                     f"{r['chance']['p_norm']:.4f} | {r['chance']['p_pix']:.4f} | {r['chance']['p_any']:.4f} | "
                     f"{r['shuffled_any']:.4f} |")
        L.append('')
    if spot:
        L += ['## Batching spot-check', '', json.dumps(spot), '']
    return '\n'.join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--formal', type=Path, required=True)
    p.add_argument('--supplement', type=Path, required=True)
    args = p.parse_args()
    _style()
    rows = load(args.output)
    summary = condition_summary(rows)
    percol = per_column(rows)
    origins = origin_x(rows)
    model = factor_model(rows)
    mech = mech_summary(args.output, rows)
    conv = convention_all(args.output, args.formal, args.supplement, rows)
    spot = {}
    sp = args.output / 'spotcheck.jsonl'
    if sp.exists():
        s = read_jsonl(sp)
        spot = {'n': len(s), 'answer_agree': sum(r['answer_batched'] == r['answer_single'] for r in s),
                'text_identical': sum(r['batched'].strip() == r['single'].strip() for r in s)}
    dest = args.output / 'analysis'
    figs = dest / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    mech_light = {k: v for k, v in mech.items() if k != 'conditions'}
    mech_light['conditions'] = {c: {'groups': d['groups'], 'margin_by_col': d['margin_by_col']}
                                for c, d in mech.get('conditions', {}).items()}
    (dest / 'summary.json').write_text(json.dumps({'conditions': summary, 'per_column': percol, 'model': model,
                                                   'mech': mech_light, 'convention': conv, 'spotcheck': spot},
                                                  indent=1, default=float) + '\n')
    (dest / 'tables.md').write_text(tables(summary, model, mech, conv, spot))
    fig_token_grid(figs)
    fig_factorial(summary, figs)
    fig_columns_aligned(percol, origins, summary, figs)
    fig_columns_aligned(percol, origins, summary, figs, key='col_acc', name='column_accuracy_token_aligned.png',
                        label='P(column answer correct)')
    fig_columns_aligned(percol, origins, summary, figs, key='pt_plus1', name='point_column_plus1_token_aligned.png',
                        label='P(point falls one column right)')
    fig_columns_aligned(percol, origins, summary, figs, key='minus1', name='column_minus1_token_aligned.png',
                        label='P(column answer = true - 1)')
    fig_column_profiles(percol, origins, figs)
    fig_phase(summary, percol, origins, figs)
    fig_mech(mech, rows, figs)
    print('analysis written to', dest)


if __name__ == '__main__':
    main()
