"""Scoring, paired statistics and figures of experiment 1.6 (PE scaling and T0 variation).

Scores per stage:
  grid / grid_explicit  strict and content-diagnostic scores of decomposition.py, the tolerant level of
                        tolerant_parse.py, and the "hit under either convention" diagnostic for point answers
  t0 / t0_explicit      exact row and column count (strict JSON; tolerant: one "rows" and one "columns" integer)
  refcoco               IoU >= 0.5 for bbox prompts (experiment 4.2 scorer), centre-in-box for point prompts
  babyvision            eva.benchmark.score (strict) and eva.diagnostic_scoring (format tolerant), plus
                        descriptive features of the reasoning text
Every PE condition is compared with x1 on the same cases by an exact McNemar test; rates carry Wilson 95% CIs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

import numpy as np

from analysis_tools.behavior import coordinate_convention as cc
from analysis_tools.behavior.decomposition import answer_part, last_json_object, parse_json, read_jsonl, score_case
from analysis_tools.behavior.decomposition_supplement import mcnemar_exact
from analysis_tools.behavior.pe_scaling_t0 import PE_CONDITIONS, t0_cases
from analysis_tools.behavior.tolerant_parse import score_tolerant

CONDS = list(PE_CONDITIONS)
COLORS = {'x1': '#2a78d6', 'x2': '#eb6834', 'x2.76': '#1baf7a'}
LABELS = {'x1': 'PE ×1 (default)', 'x2': 'PE ×2', 'x2.76': 'PE ×2.76'}

# ----------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(flags: list[bool]) -> dict:
    k, n = int(sum(flags)), len(flags)
    return {'k': k, 'n': n, 'acc': k / n if n else float('nan'), 'ci': wilson(k, n)}


def mcnemar(base: dict, other: dict) -> dict:
    """Paired comparison on shared keys; gained = wrong at x1 and right in the other condition."""
    keys = sorted(set(base) & set(other))
    if len(keys) != len(base) or len(keys) != len(other):
        raise ValueError(f'Unpaired records: {len(base)} vs {len(other)} with {len(keys)} shared.')
    gained = sum(not base[k] and other[k] for k in keys)
    lost = sum(base[k] and not other[k] for k in keys)
    return {'n': len(keys), 'gained': gained, 'lost': lost, 'p': mcnemar_exact(gained, lost)}


def fisher_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact test of k1/n1 vs k2/n2 (sum of tables at most as likely as the observed one)."""
    total, kt = n1 + n2, k1 + k2
    def prob(a):
        return math.comb(n1, a) * math.comb(n2, kt - a) / math.comb(total, kt)
    p_obs = prob(k1)
    lo, hi = max(0, kt - n2), min(kt, n1)
    return min(1.0, sum(prob(a) for a in range(lo, hi + 1) if prob(a) <= p_obs * (1 + 1e-9)))


def fmt_rate(r: dict) -> str:
    return f"{r['k']}/{r['n']} = {r['acc']:.3f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"


def fmt_p(p: float) -> str:
    return f'{p:.2g}' if p >= 1e-4 else f'{p:.1e}'


# ----------------------------------------------------------------------------- grid


def score_grid_record(case: dict, rec: dict) -> dict:
    strict = score_case(case, rec, False)
    diag = score_case(case, rec, True)
    tol = score_tolerant(case, rec, {'id': rec['id'], 'finish_reason': rec['finish_reason'], **diag},
                         lambda c, r: score_case(c, r, True))
    row = {'case_id': case['id'], 'condition': rec['condition'], 'stage': case['stage'], 'task': case['task'],
           'format': case['format'], 'grid': case['grid'], 'kind': case['kind'], 'level': case['level'],
           'strict': bool(strict['correct']), 'tolerant': bool(tol.get('correct')), 'parse': tol.get('parse'),
           'finish_reason': rec['finish_reason'], 'n_tokens': rec['n_tokens']}
    if case['format'] == 'row_column':
        row.update(pred_rc=tol.get('pred_rc'), d_row=tol.get('d_row'), d_col=tol.get('d_col'))
    if case['format'] == 'point':
        conv = cc.score_point(rec['response'], rec['finish_reason'], case['truth']['canvas'],
                              case['truth']['target_cell_box'])
        row.update(conv_any=conv['h_any'], conv_norm=conv['h_norm'], p_any=conv['p_any'],
                   cell_distance=tol.get('cell_distance'), pred_center=tol.get('pred_center'),
                   pred_rc=tol.get('pred_rc'))
    if case['format'] == 'yes_no':
        row.update(s=rec.get('s'), answer_yes=case['truth']['answer_yes'])
    return row


def grid_rows(out: Path, formal: Path) -> tuple[list[dict], dict]:
    cases = {c['id']: c for c in read_jsonl(out / 'grid' / 'cases.jsonl')}
    rows = []
    for stage in ('grid', 'grid_explicit'):
        path = out / 'grid' / f'results_{stage}.jsonl'
        if not path.exists():
            continue
        for rec in read_jsonl(path):
            rows.append(score_grid_record(cases[rec['case_id']], rec))
    # Agreement of the batched x1 answers with the 1.3 formal answers (unbatched).
    formal_rec = {r['id']: r for f in ('results_main.jsonl', 'results_explicit.jsonl') for r in read_jsonl(formal / f)}
    agree = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r['condition'] != 'x1':
            continue
        c = cases[r['case_id']]
        f = score_grid_record(c, {**formal_rec[c['id']], 'id': c['id'], 'condition': 'formal'})
        same_answer = {
            'row_column': lambda: r.get('pred_rc') == f.get('pred_rc'),
            'point': lambda: r.get('pred_rc') == f.get('pred_rc'),      # same grid cell
            'yes_no': lambda: (r['s'] > 0) == (f['s'] > 0),
        }[c['format']]()
        a = agree[r['stage']]
        a[0] += 1
        a[1] += r['tolerant'] == f['tolerant']
        a[2] += bool(same_answer)
    return rows, {k: {'n': v[0], 'same_correctness': v[1], 'same_answer': v[2]} for k, v in agree.items()}


def summarize_grid(rows: list[dict]) -> dict:
    groups = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r['stage'], r['task'], r['format'], r['grid'] if r['format'] != 'yes_no' else 'all')
        groups[key][r['condition']][r['case_id']] = r
    out = []
    for key in sorted(groups, key=str):
        by = groups[key]
        entry = {'stage': key[0], 'task': key[1], 'format': key[2], 'grid': key[3], 'conditions': {}, 'paired': {}}
        metrics = ['strict', 'tolerant'] + (['conv_any'] if key[2] == 'point' else [])
        for c in CONDS:
            if c not in by:
                continue
            rs = list(by[c].values())
            e = {m: rate([bool(r[m]) for r in rs]) for m in metrics}
            if key[2] == 'row_column':
                wrong = [r for r in rs if r['d_col'] is not None and not r['tolerant']]
                e['col_plus1'] = rate([r['d_col'] == 1 and r['d_row'] == 0 for r in rs])
                e['row_plus1'] = rate([r['d_row'] == 1 and r['d_col'] == 0 for r in rs])
                e['n_wrong_valid'] = len(wrong)
            if key[2] == 'point':
                d = [r['cell_distance'] for r in rs if r['cell_distance'] is not None]
                e['median_cell_distance'] = float(np.median(d)) if d else None
                e['chance_any'] = float(np.mean([r['p_any'] for r in rs]))
            e['truncated'] = sum(r['finish_reason'] != 'eos' for r in rs)
            entry['conditions'][c] = e
        for c in CONDS[1:]:
            if c in by and 'x1' in by:
                entry['paired'][c] = {m: mcnemar({k: bool(v[m]) for k, v in by['x1'].items()},
                                                 {k: bool(v[m]) for k, v in by[c].items()}) for m in metrics}
        out.append(entry)
    return out


def pooled_grid(rows: list[dict]) -> list[dict]:
    """Pooled over grids per (stage, task, format) for the headline paired tests."""
    groups = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        groups[(r['stage'], r['task'], r['format'])][r['condition']][r['case_id']] = r
    out = []
    for key in sorted(groups, key=str):
        by = groups[key]
        metrics = ['strict', 'tolerant'] + (['conv_any'] if key[2] == 'point' else [])
        e = {'stage': key[0], 'task': key[1], 'format': key[2],
             'rates': {c: {m: rate([bool(r[m]) for r in by[c].values()]) for m in metrics} for c in CONDS if c in by},
             'paired': {c: {m: mcnemar({k: bool(v[m]) for k, v in by['x1'].items()},
                                       {k: bool(v[m]) for k, v in by[c].items()}) for m in metrics}
                        for c in CONDS[1:] if c in by and 'x1' in by}}
        out.append(e)
    return out


# ----------------------------------------------------------------------------- T0

_ROWS = re.compile(r'"rows"\s*:\s*(\d+)')
_COLS = re.compile(r'"columns"\s*:\s*(\d+)')


def score_t0(case: dict, rec: dict) -> dict:
    t = case['truth']
    out = {'case_id': case['id'], 'condition': rec['condition'], 'stage': case['stage'], 'layout': case['layout'],
           'variation': case['variation'], 'template': case['template'], 'rows': t['rows'], 'cols': t['cols'],
           'finish_reason': rec['finish_reason'], 'n_tokens': rec['n_tokens'], 'strict': False, 'tolerant': False,
           'pred_rows': None, 'pred_cols': None}
    if rec['finish_reason'] != 'eos':
        return out
    text = answer_part(rec['response'])
    value = None
    try:
        value = last_json_object(text) if case['stage'] == 'explicit' else parse_json(text)
    except ValueError:
        pass
    ok = isinstance(value, dict) and type(value.get('rows')) is int and type(value.get('columns')) is int
    if ok:
        pr, pc = value['rows'], value['columns']
        out['strict'] = (pr, pc) == (t['rows'], t['cols'])
    else:
        tail = text.strip().splitlines()[-1] if case['stage'] == 'explicit' and text.strip() else text
        r, c = _ROWS.findall(tail), _COLS.findall(tail)
        if len(r) != 1 or len(c) != 1:
            return out
        pr, pc = int(r[0]), int(c[0])
    out.update(pred_rows=pr, pred_cols=pc, tolerant=(pr, pc) == (t['rows'], t['cols']),
               rows_ok=pr == t['rows'], cols_ok=pc == t['cols'], d_rows=pr - t['rows'], d_cols=pc - t['cols'])
    return out


def t0_rows(out: Path, dataset: Path, per_layout: int) -> list[dict]:
    cases = {c['id']: c for c in t0_cases(dataset, per_layout)}
    rows = []
    for stage in ('t0', 't0_explicit'):
        path = out / 't0' / f'results_{stage}.jsonl'
        if not path.exists():
            continue
        for rec in read_jsonl(path):
            rows.append(score_t0(cases[rec['case_id']], rec))
    return rows


def summarize_t0(rows: list[dict]) -> list[dict]:
    groups = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        groups[(r['stage'], r['layout'], r['variation'])][r['condition']][r['case_id']] = r
    out = []
    for key in sorted(groups, key=str):
        by = groups[key]
        e = {'stage': key[0], 'layout': key[1], 'variation': key[2], 'conditions': {}, 'paired': {}}
        for c in CONDS:
            if c not in by:
                continue
            rs = list(by[c].values())
            answers = defaultdict(int)
            for r in rs:
                answers[f"{r['pred_rows']}x{r['pred_cols']}"] += 1
            e['conditions'][c] = {'tolerant': rate([r['tolerant'] for r in rs]),
                                  'strict': rate([r['strict'] for r in rs]),
                                  'rows_ok': rate([bool(r.get('rows_ok')) for r in rs]),
                                  'cols_ok': rate([bool(r.get('cols_ok')) for r in rs]),
                                  'n_distinct_images': len({r['case_id'] for r in rs}),
                                  'answers': dict(sorted(answers.items(), key=lambda kv: -kv[1]))}
        for c in CONDS[1:]:
            if c in by and 'x1' in by:
                e['paired'][c] = mcnemar({k: v['tolerant'] for k, v in by['x1'].items()},
                                         {k: v['tolerant'] for k, v in by[c].items()})
        out.append(e)
    return out


def t0_variation_tests(summary: list[dict]) -> list[dict]:
    """Unpaired Fisher tests of V2/V3 against V1 (different images) at each layout, x1 and x2.76."""
    idx = {(e['stage'], e['layout'], e['variation']): e for e in summary}
    out = []
    for (stage, layout, var), e in sorted(idx.items()):
        if stage != 't0' or var in ('V0_fixed', 'V0s_shifted', 'V1_jitter'):
            continue
        ref = idx[(stage, layout, 'V1_jitter')]
        for c in ('x1', 'x2.76'):
            a, b = e['conditions'][c]['tolerant'], ref['conditions'][c]['tolerant']
            out.append({'layout': layout, 'variation': var, 'condition': c, 'acc': a['acc'], 'acc_V1': b['acc'],
                        'p_fisher': fisher_two_sided(a['k'], a['n'], b['k'], b['n'])})
    return out


# ----------------------------------------------------------------------------- RefCOCO


def refcoco_rows(out: Path) -> list[dict]:
    from analysis_tools.deepstack.deepstack_comparison import score_refcoco_case
    cases = {c['id']: c for c in read_jsonl(out / 'refcoco' / 'cases.jsonl')}
    rows = []
    for rec in read_jsonl(out / 'refcoco' / 'results.jsonl'):
        c = cases[rec['case_id']]
        r = {'case_id': c['id'], 'condition': rec['condition'], 'format': c['format'], 'stratum': c['stratum']}
        if c['format'] == 'bbox':
            s = score_refcoco_case(c, rec)
            r.update(correct=bool(s['correct']), iou=s['iou'], center_in=bool(s['center_in']), valid=s['valid'])
        else:
            pt = cc.parse_point(rec['response']) if rec['finish_reason'] == 'eos' else None
            valid = pt is not None and 0 <= pt[0] <= 1000 and 0 <= pt[1] <= 1000
            if valid:
                x, y = pt[0] / 1000 * c['width'], pt[1] / 1000 * c['height']
                g = c['bbox']
                inside = g[0] <= x <= g[2] and g[1] <= y <= g[3]
            else:
                inside = False
            r.update(correct=inside, valid=valid)
        rows.append(r)
    return rows


def summarize_refcoco(rows: list[dict], seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for fmt in ('bbox', 'point'):
        by = defaultdict(dict)
        for r in rows:
            if r['format'] == fmt:
                by[r['condition']][r['case_id']] = r
        e = {'format': fmt, 'rates': {c: rate([r['correct'] for r in by[c].values()]) for c in CONDS if c in by},
             'paired': {c: mcnemar({k: v['correct'] for k, v in by['x1'].items()},
                                   {k: v['correct'] for k, v in by[c].items()}) for c in CONDS[1:] if c in by}}
        if fmt == 'bbox':
            keys = sorted(by['x1'])
            base = np.array([by['x1'][k]['iou'] for k in keys])
            e['mean_iou'] = {c: float(np.mean([by[c][k]['iou'] for k in keys])) for c in CONDS if c in by}
            e['iou_diff'] = {}
            for c in CONDS[1:]:
                if c not in by:
                    continue
                d = np.array([by[c][k]['iou'] for k in keys]) - base
                boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                e['iou_diff'][c] = {'mean': float(d.mean()), 'ci': [float(np.percentile(boot, 2.5)),
                                                                    float(np.percentile(boot, 97.5))]}
        e['by_stratum'] = {s: {c: rate([r['correct'] for r in by[c].values() if r['stratum'] == s])
                               for c in CONDS if c in by} for s in sorted({r['stratum'] for r in rows})}
        out.append(e)
    return out


# ----------------------------------------------------------------------------- BabyVision

HEDGE = re.compile(r"\b(assum\w*|without (?:seeing|the image|access)|can(?:not|'t) (?:see|view|observe)|"
                   r"as described|typical(?:ly)?|common (?:type|puzzle|design)|standard (?:puzzle|design))\b", re.I)
WAIT = re.compile(r'\bwait\b', re.I)
DIRECTION = re.compile(r'\b(up|down|left|right|upward|downward|leftward|rightward)\b', re.I)
COORD = re.compile(r'\(\s*\d+\s*,\s*\d+\s*\)')
GRID_DECL = re.compile(r'\b(\d+)\s*(?:x|×|by)\s*(\d+)\b', re.I)


def text_features(text: str, n_tokens: int) -> dict:
    per_k = 1000 / max(n_tokens, 1)
    return {'n_tokens': n_tokens, 'hedge': bool(HEDGE.search(text)), 'n_wait': len(WAIT.findall(text)),
            'direction_per_k': len(DIRECTION.findall(text)) * per_k, 'coord_per_k': len(COORD.findall(text)) * per_k,
            'grid_decls': sorted({f'{a}x{b}' for a, b in GRID_DECL.findall(text)})}


def babyvision_rows(out: Path, data_root: Path) -> list[dict]:
    from eva.benchmark import load_cases, score
    from eva.diagnostic_scoring import diagnostic_score
    manifest = {r['taskId']: r for r in load_cases(data_root, 'babyvision', 'train')[0]}
    rows = []
    for rec in read_jsonl(out / 'babyvision' / 'results.jsonl'):
        m = manifest[rec['taskId']]
        strict = score(m, rec['response'], 'babyvision')
        diag = diagnostic_score(m, rec['response'], rec['finish_reason'])
        rows.append({'case_id': rec['case_id'], 'taskId': rec['taskId'], 'subtype': m['subtype'],
                     'condition': rec['condition'], 'seed': rec['seed'], 'strict': bool(strict['correct']),
                     'diagnostic': bool(diag['correct']), 'prediction': strict['prediction'],
                     'reference': strict['reference'], 'finish_reason': rec['finish_reason'],
                     **text_features(rec['response'], rec['n_tokens'])})
    return rows


def summarize_babyvision(rows: list[dict], reference_run: Path | None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    by = defaultdict(dict)
    for r in rows:
        by[r['condition']][(r['taskId'], r['seed'])] = r
    # Conditions still running are reported by count only and left out of the statistics.
    incomplete = {c: len(by[c]) for c in list(by) if set(by[c]) != set(by['x1'])}
    for c in incomplete:
        del by[c]
    rows = [r for r in rows if r['condition'] in by]
    subtypes = sorted({r['subtype'] for r in rows})
    out = {'n_items': len({r['taskId'] for r in rows}), 'seeds': sorted({r['seed'] for r in rows}),
           'incomplete_conditions': incomplete, 'overall': {},
           'by_subtype': {}, 'paired': {}, 'features': {}}
    for m in ('strict', 'diagnostic'):
        out['overall'][m] = {c: rate([r[m] for r in by[c].values()]) for c in CONDS if c in by}
        out['paired'][m] = {c: mcnemar({k: v[m] for k, v in by['x1'].items()}, {k: v[m] for k, v in by[c].items()})
                            for c in CONDS[1:] if c in by and 'x1' in by}
        out['by_subtype'][m] = {s: {c: rate([r[m] for r in by[c].values() if r['subtype'] == s])
                                    for c in CONDS if c in by} for s in subtypes}
    for f in ('n_tokens', 'n_wait', 'direction_per_k', 'coord_per_k', 'hedge'):
        e = {c: float(np.mean([float(r[f]) for r in by[c].values()])) for c in CONDS if c in by}
        diffs = {}
        for c in CONDS[1:]:
            if c not in by or 'x1' not in by:
                continue
            keys = sorted(by['x1'])
            d = np.array([float(by[c][k][f]) - float(by['x1'][k][f]) for k in keys])
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            diffs[c] = {'mean': float(d.mean()), 'ci': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        out['features'][f] = {'mean': e, 'diff_vs_x1': diffs}
    out['truncated'] = {c: sum(r['finish_reason'] != 'eos' for r in by[c].values()) for c in CONDS if c in by}
    if reference_run is not None and reference_run.exists():
        ids = {r['taskId'] for r in rows}
        ref = [r for r in read_jsonl(reference_run) if r['taskId'] in ids]
        out['reference_run'] = rate([bool(r['score']['correct']) for r in ref])
    return out


# ----------------------------------------------------------------------------- figures


def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e4e3df', lw=0.6)
    ax.set_axisbelow(True)


def _err(r: dict) -> tuple[float, list[float]]:
    return r['acc'], [r['acc'] - r['ci'][0], r['ci'][1] - r['acc']]


def fig_grid(summary: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    panels = [(t, f, m) for t in ('T4', 'T5', 'T6') for f, m in (('row_column', 'tolerant'), ('point', 'tolerant'),
                                                                 ('point', 'conv_any'))]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)
    titles = {('row_column', 'tolerant'): 'row/column, tolerant', ('point', 'tolerant'): 'point, tolerant (0–1000)',
              ('point', 'conv_any'): 'point, hit under either convention'}
    for ax, (task, fmt, metric) in zip(axes.flat, panels):
        es = sorted([e for e in summary if e['stage'] == 'main' and e['task'] == task and e['format'] == fmt],
                    key=lambda e: e['grid'])
        grids = [e['grid'] for e in es]
        for j, c in enumerate(CONDS):
            ys = [e['conditions'][c][metric] for e in es]
            x = np.arange(len(grids)) + (j - 1) * 0.08
            ax.errorbar(x, [y['acc'] for y in ys], yerr=np.array([_err(y)[1] for y in ys]).T, color=COLORS[c],
                        marker='o', ms=6, lw=2, capsize=3, label=LABELS[c])
            if c != 'x1':
                for xi, e in zip(x, es):
                    if e['paired'][c][metric]['p'] < 0.05:
                        ax.annotate('*', (xi, min(0.98, e['conditions'][c][metric]['ci'][1] + 0.02)), color=COLORS[c],
                                    ha='center', fontsize=13)
        ax.set_xticks(range(len(grids)), [f'{g}×{g}' for g in grids])
        ax.set_title(f'{task} — {titles[(fmt, metric)]}', fontsize=10)
        ax.set_ylim(0, 1.02)
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel('accuracy')
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('Qwen3.5-9B, 1.3 grid subset: accuracy under ViT PE scaling (95% Wilson CI; * McNemar p<0.05 vs ×1)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_grid_explicit(summary: list[dict], control: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [3, 1.3]})
    es = [e for e in summary if e['stage'] == 'explicit']
    es.sort(key=lambda e: (e['task'], e['grid']))
    labels = [f"{e['task']} {e['grid']}×{e['grid']}" for e in es]
    for j, c in enumerate(CONDS):
        ys = [e['conditions'][c]['tolerant'] for e in es]
        axes[0].bar(np.arange(len(es)) + (j - 1) * 0.26, [y['acc'] for y in ys], 0.24, color=COLORS[c],
                    yerr=np.array([_err(y)[1] for y in ys]).T, capsize=2, label=LABELS[c])
    axes[0].set_xticks(range(len(es)), labels)
    axes[0].set_ylabel('accuracy (tolerant)')
    axes[0].set_title('Think-then-answer row/column (1.3 explicit cases, thinking off)', fontsize=10)
    axes[0].legend(frameon=False, fontsize=9)
    cs = [e for e in control if e['stage'] == 'main' and e['format'] == 'yes_no']
    for j, c in enumerate(CONDS):
        ys = [e['rates'][c]['strict'] for e in cs]
        axes[1].bar(np.arange(len(cs)) + (j - 1) * 0.26, [y['acc'] for y in ys], 0.24, color=COLORS[c],
                    yerr=np.array([_err(y)[1] for y in ys]).T, capsize=2)
    axes[1].set_xticks(range(len(cs)), [e['task'] for e in cs])
    axes[1].set_title('Recognition controls T1/T2 (sign of s)', fontsize=10)
    for ax in axes:
        ax.set_ylim(0, 1.02)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_col_offset(rows: list[dict], path: Path) -> None:
    """Distribution of the column error d_col of T6 row/column answers at 20x20 (direct answers)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, axis in zip(axes, ('d_col', 'd_row')):
        bins = np.arange(-4, 5)
        for j, c in enumerate(CONDS):
            vals = [r[axis] for r in rows if r['stage'] == 'main' and r['task'] in ('T4', 'T5', 'T6')
                    and r['format'] == 'row_column' and r['grid'] == 20 and r['condition'] == c and r[axis] is not None]
            cnt = np.array([sum(v == b for v in vals) for b in bins]) / max(len(vals), 1)
            ax.bar(bins + (j - 1) * 0.27, cnt, 0.25, color=COLORS[c], label=f'{LABELS[c]} (n={len(vals)})')
        ax.set_xticks(bins)
        ax.set_xlabel(f'{"column" if axis == "d_col" else "row"} error (answer − truth, cells)')
        _style(ax)
    axes[0].set_ylabel('fraction of valid answers')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('T4–T6 row/column answers on 20×20 grids: ordinal error distribution', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


T0_ORDER = ['3x3_c64', '6x6_c64', '10x10_c64', '20x20_c64', '4x8_c64', '8x4_c64', '6x20_c64', '20x6_c64',
            '10x20_c64', '20x10_c64', '24x24_c32']
VAR_ORDER = ['V0_fixed', 'V0s_shifted', 'V1_jitter', 'V2_jitter_color', 'V3_jitter_color_mixed']


def fig_t0_heatmap(summary: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    idx = {(e['layout'], e['variation']): e for e in summary if e['stage'] == 't0'}
    names = {'V0_fixed': 'V0 fixed', 'V0s_shifted': 'V0s shifted layout', 'V1_jitter': 'V1 jitter',
             'V2_jitter_color': 'V2 +colour', 'V3_jitter_color_mixed': 'V3 +mixed templates'}
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for ax, c in zip(axes, CONDS):
        vars_ = [v for v in VAR_ORDER if all((l, v) in idx and c in idx[(l, v)]['conditions'] for l in T0_ORDER)]
        mat = np.array([[idx[(l, v)]['conditions'][c]['tolerant']['acc'] for l in T0_ORDER] for v in vars_])
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        for i, v in enumerate(vars_):
            for j, l in enumerate(T0_ORDER):
                r = idx[(l, v)]['conditions'][c]['tolerant']
                ax.text(j, i, f"{r['k']}/{r['n']}", ha='center', va='center', fontsize=8,
                        color='white' if r['acc'] < 0.6 else 'black')
        ax.set_xticks(range(len(T0_ORDER)), [l.replace('x', '×').replace('_c', '\ncell ') for l in T0_ORDER], fontsize=9)
        ax.set_title(LABELS[c], fontsize=10)
        ax.set_yticks(range(len(vars_)), [names[v] for v in vars_])
    cb = fig.colorbar(im, ax=axes, fraction=0.02)
    cb.set_label('T0 accuracy (rows and columns both correct)')
    fig.suptitle('T0 counting, direct answers: accuracy per layout (rows×cols, cell px) and variation')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_t0_errors(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    layouts = ['20x20_c64', '6x20_c64', '20x6_c64', '10x20_c64', '20x10_c64', '24x24_c32']
    fig, axes = plt.subplots(2, len(layouts), figsize=(16, 5.5), sharey='row')
    bins = np.arange(-3, 4)
    for col, l in enumerate(layouts):
        for row_i, axis in enumerate(('d_rows', 'd_cols')):
            ax = axes[row_i, col]
            for j, v in enumerate(['V0s_shifted', 'V1_jitter', 'V3_jitter_color_mixed']):
                vals = [r.get(axis) for r in rows if r['stage'] == 't0' and r['layout'] == l and r['variation'] == v
                        and r['condition'] == 'x1']
                n = len(vals)
                cnt = np.array([sum(x == b for x in vals if x is not None) for b in bins]) / max(n, 1)
                ax.bar(bins + (j - 1) * 0.27, cnt, 0.25, color=['#6250d6', '#2a78d6', '#eda100'][j],
                       label=['V0s shifted layout (4 images)', 'V1 jitter', 'V3 jitter+colour+mixed'][j])
            ax.set_xticks(bins)
            _style(ax)
            if row_i == 0:
                ax.set_title(l.replace('_c', ', cell '), fontsize=9)
            ax.set_xlabel(f'{"row" if axis == "d_rows" else "column"} count error')
    axes[0, 0].set_ylabel('fraction of images')
    axes[1, 0].set_ylabel('fraction of images')
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle('T0 count errors (answer − truth) on layouts with ≥20 instances on an axis, PE ×1', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_refcoco(summary: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for j, c in enumerate(CONDS):
        ys = [e['rates'][c] for e in summary]
        ax.bar(np.arange(len(summary)) + (j - 1) * 0.26, [y['acc'] for y in ys], 0.24, color=COLORS[c],
               yerr=np.array([_err(y)[1] for y in ys]).T, capsize=3, label=LABELS[c])
    ax.set_xticks(range(len(summary)), ['bbox: IoU ≥ 0.5', 'point: centre in GT box'])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('accuracy')
    ax.set_title(f"RefCOCO test subset (n={summary[0]['rates']['x1']['n']} expressions)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_babyvision(summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    subs = list(summary['by_subtype']['strict'])
    conds = [c for c in CONDS if c in summary['overall']['strict']]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={'width_ratios': [3, 1.4]})
    for j, c in enumerate(conds):
        ys = [summary['by_subtype']['diagnostic'][s][c] for s in subs]
        axes[0].bar(np.arange(len(subs)) + (j - 1) * 0.26, [y['acc'] for y in ys], 0.24, color=COLORS[c],
                    yerr=np.array([_err(y)[1] for y in ys]).T, capsize=2, label=LABELS[c])
    axes[0].set_xticks(range(len(subs)), [s.replace(' ', '\n', 1) for s in subs], fontsize=8)
    axes[0].set_ylabel('accuracy (diagnostic scoring)')
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title('BabyVision subtypes (items × seeds)', fontsize=10)
    feats = ['n_tokens', 'n_wait', 'direction_per_k', 'coord_per_k']
    names = ['tokens / 1000', '"wait" count', 'direction words / 1k tok', 'coordinates / 1k tok']
    for j, c in enumerate(conds[1:]):
        d = [summary['features'][f]['diff_vs_x1'][c] for f in feats]
        scale = [1000 if f == 'n_tokens' else 1 for f in feats]
        mean = [x['mean'] / s for x, s in zip(d, scale)]
        err = np.array([[(x['mean'] - x['ci'][0]) / s, (x['ci'][1] - x['mean']) / s] for x, s in zip(d, scale)]).T
        axes[1].errorbar(mean, np.arange(len(feats)) + (j - 0.5) * 0.25, xerr=err, fmt='o', color=COLORS[c],
                         capsize=3, label=f'{LABELS[c]} − ×1')
    axes[1].axvline(0, color='#52514e', lw=0.8)
    axes[1].set_yticks(range(len(feats)), names, fontsize=8)
    axes[1].set_title('Reasoning-text features: paired difference (95% bootstrap CI)', fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_t0_examples(dataset: Path, path: Path) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image
    meta = read_jsonl(dataset / 'metadata.jsonl')
    picks = []
    for layout in ('6x6_c64', '20x10_c64'):
        for v in ('V0_fixed', 'V1_jitter', 'V2_jitter_color', 'V3_jitter_color_mixed'):
            picks.append(next(m for m in meta if m['layout'] == layout and m['variation'] == v))
    fig, axes = plt.subplots(2, 4, figsize=(14, 9))
    for ax, m in zip(axes.flat, picks):
        ax.imshow(Image.open(dataset / m['image']))
        ax.set_title(f"{m['layout'].replace('_c', ', cell ')} — {m['variation']}\ncanvas {m['canvas'][0]}×{m['canvas'][1]}",
                     fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--formal', type=Path, required=True)
    p.add_argument('--t0-dataset', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--t0-explicit-per-layout', type=int, default=12)
    p.add_argument('--babyvision-reference', type=Path)
    p.add_argument('--parts', nargs='+', default=['grid', 't0', 'refcoco', 'babyvision'])
    args = p.parse_args()
    out, figs = args.output, args.figures
    figs.mkdir(parents=True, exist_ok=True)
    (out / 'analysis').mkdir(exist_ok=True)
    summary_path = out / 'analysis' / 'summary.json'
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if 'grid' in args.parts:
        rows, agree = grid_rows(out, args.formal)
        summary, pooled = summarize_grid(rows), pooled_grid(rows)
        result['grid'] = {'summary': summary, 'pooled': pooled, 'agreement_with_formal': agree}
        (out / 'analysis' / 'grid_rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        fig_grid(summary, figs / 'grid_accuracy.png')
        if any(e['stage'] == 'explicit' and len(e['conditions']) == len(CONDS) for e in summary):
            fig_grid_explicit(summary, pooled, figs / 'grid_explicit_controls.png')
        fig_col_offset(rows, figs / 'grid_ordinal_error.png')
    if 't0' in args.parts:
        rows = t0_rows(out, args.t0_dataset, args.t0_explicit_per_layout)
        summary = summarize_t0(rows)
        result['t0'] = {'summary': summary, 'variation_tests': t0_variation_tests(summary)}
        (out / 'analysis' / 't0_rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        fig_t0_heatmap(summary, figs / 't0_accuracy.png')
        fig_t0_errors(rows, figs / 't0_errors.png')
        fig_t0_examples(args.t0_dataset, figs / 't0_examples.png')
    if 'refcoco' in args.parts:
        summary = summarize_refcoco(refcoco_rows(out))
        result['refcoco'] = summary
        fig_refcoco(summary, figs / 'refcoco.png')
    if 'babyvision' in args.parts:
        rows = babyvision_rows(out, args.data_root)
        summary = summarize_babyvision(rows, args.babyvision_reference)
        result['babyvision'] = summary
        (out / 'analysis' / 'babyvision_rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        fig_babyvision(summary, figs / 'babyvision.png')
    summary_path.write_text(json.dumps(result, indent=1) + '\n')
    print('wrote', summary_path)


if __name__ == '__main__':
    main()
