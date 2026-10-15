"""Statistics and figures for experiment 4.2 (docs/analysis/deepstack/plan.md, section 5.2).

Inputs (all under --output, the 4.2 artifact directory):
  babyvision/*.jsonl               eva.evaluation outputs of the new runs; the Qwen3.5-9B thinking-off
                                   run is read from --legacy-babyvision
  decomposition/<model>/scores_main_{strict,diagnostic}.jsonl   written by decomposition.score
  refcoco/refcoco_scores_<model>.jsonl                           written by deepstack_comparison refcoco_score

Outputs: analysis/summary.json, analysis/tables.md and analysis/figures/*.png.
Paired comparisons always use the same items in both models of a pair (McNemar, exact binomial
test on the discordant pairs; Holm correction within each family of cells).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from analysis_tools.behavior.decomposition import GRIDS, KINDS, magnitude_label, read_jsonl
from analysis_tools.behavior.decomposition_analysis import (
    INK, MUTED, GRID_INK, KIND_NAMES, auroc, auroc_threshold, binary_stats, fit_psychometric, load,
    numeric_magnitude, wilson, _style,
)
from analysis_tools.deepstack.deepstack_comparison import MODELS, STRATA, slug

PAIRS = (('Qwen3-VL-8B-Instruct', 'Qwen3.5-9B'), ('Qwen3-VL-4B-Instruct', 'Qwen3.5-4B'))
SHORT = {'Qwen3-VL-8B-Instruct': 'VL-8B', 'Qwen3.5-9B': '3.5-9B', 'Qwen3-VL-4B-Instruct': 'VL-4B',
         'Qwen3.5-4B': '3.5-4B'}
# Qwen3-VL = blue family, Qwen3.5 = orange family; the 4B models use the lighter step of the same hue.
COLORS = {'Qwen3-VL-8B-Instruct': '#1c5cab', 'Qwen3-VL-4B-Instruct': '#86b6ef',
          'Qwen3.5-9B': '#c4501f', 'Qwen3.5-4B': '#f2a37f'}
DIVERGING = ('#c4501f', '#f0efec', '#1c5cab')     # Qwen3.5 better <- neutral -> Qwen3-VL better
FOCUS_SUBTYPES = ('Find the different', 'Fine-grained Discrimination', 'Visual Tracking')

# (label, task, variant filter, format, metric key in the score record)
SUBTASKS = (
    ('T0', 'T0', None, 'count', 'correct'),
    ('T1', 'T1', None, 'yes_no', 'correct'),
    ('T2', 'T2', None, 'yes_no', 'correct'),
    ('T3 box', 'T3', ('box_target', 'box_normal'), 'yes_no', 'correct'),
    ('T3 row/col', 'T3', ('rowcol_target', 'rowcol_normal'), 'yes_no', 'correct'),
    ('T4 row/col', 'T4', None, 'row_column', 'correct'),
    ('T4 point', 'T4', None, 'point', 'hit'),
    ('T5 row/col', 'T5', None, 'row_column', 'correct'),
    ('T5 point', 'T5', None, 'point', 'hit'),
    ('T6 row/col', 'T6', None, 'row_column', 'correct'),
    ('T6 point', 'T6', None, 'point', 'hit'),
    ('T4 bbox', 'T4', None, 'bbox', 'correct'),
    ('T5 bbox', 'T5', None, 'bbox', 'correct'),
    ('T6 bbox', 'T6', None, 'bbox', 'correct'),
)
BINARY = ('T1', 'T2', 'T3 box', 'T3 row/col')


# ----------------------------------------------------------------------------- statistics


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test; a and b are paired correctness vectors."""
    a, b = [bool(x) for x in a], [bool(y) for y in b]
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    if n == 0:
        p = 1.0
    else:
        k = min(n10, n01)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {'n_pairs': len(a), 'only_first': n10, 'only_second': n01, 'p': p}


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def rate(flags: list[bool]) -> dict:
    n, k = len(flags), int(sum(flags))
    # Clamp to [0, 1]: the shared Wilson helper can return -1e-17 at k = 0 through rounding.
    return {'n': n, 'k': k, 'acc': k / n if n else float('nan'), 'ci': [min(max(c, 0.0), 1.0) for c in wilson(k, n)]}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> list[float]:
    """95% percentile interval of mean(a) - mean(b), resampling items (one item per image)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def fast_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties (equals decomposition_analysis.auroc)."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    ranks = (upper - (counts - 1) / 2.0)[inverse]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc_diff(s_a: np.ndarray, s_b: np.ndarray, labels: np.ndarray, groups: np.ndarray, n_boot: int,
               seed: int) -> dict:
    """AUROC(a) - AUROC(b) on the same items, with a paired bootstrap that resamples source images."""
    value = fast_auroc(s_a, labels) - fast_auroc(s_b, labels)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        boot.append(fast_auroc(s_a[idx], labels[idx]) - fast_auroc(s_b[idx], labels[idx]))
    boot = np.array(boot)
    boot = boot[~np.isnan(boot)]
    return {'diff': float(value), 'ci': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def fmt(stat: dict | None) -> str:
    if not stat or not stat['n']:
        return '–'
    return f"{stat['acc']:.2f} [{stat['ci'][0]:.2f}, {stat['ci'][1]:.2f}]"


def fmt_p(p: float) -> str:
    return '<0.001' if p < 1e-3 else f'{p:.3f}'


# ----------------------------------------------------------------------------- BabyVision


CAP = 4096   # Qwen3-VL output budget on BabyVision; used for the matched-budget comparison


def capped_correct(record: dict) -> bool:
    return bool(record['score']['correct']) and record['generated_tokens'] <= CAP


def is_repetitive(text: str, window: int = 1500, min_len: int = 30, min_count: int = 3) -> bool:
    """True if some line of at least min_len characters recurs min_count times in the response tail.

    Used to tell repetition loops apart from long but progressing reasoning among truncated answers.
    """
    tail = text[-window:]
    lines = [l.strip() for l in tail.splitlines() if len(l.strip()) >= min_len]
    counts: dict = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    if counts and max(counts.values()) >= min_count:
        return True
    # Loops inside one long line: a 60-character chunk that recurs.
    chunk = tail[-60:]
    return len(chunk) == 60 and tail.count(chunk) >= min_count


def babyvision_files(output: Path, legacy: Path) -> dict[str, Path]:
    files = {}
    for model in MODELS:
        if model == 'Qwen3.5-9B':
            found = sorted(legacy.glob('qwen35_9b_babyvision-thinkingoff-*.jsonl'))
        else:
            found = sorted((output / 'babyvision').glob(f'{slug(model)}_babyvision-thinkingoff-*.jsonl'))
        if len(found) > 1:
            raise ValueError(f'Ambiguous BabyVision runs for {model}: {found}')
        if found:
            files[model] = found[0]
    return files


def babyvision(output: Path, legacy: Path) -> dict:
    files = babyvision_files(output, legacy)
    data = {}
    for model, path in files.items():
        records = read_jsonl(path)
        ids = [r['taskId'] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate taskIds in {path}')
        meta = json.loads(path.with_suffix(path.suffix + '.meta.json').read_text())['config']
        data[model] = {'path': path, 'records': {r['taskId']: r for r in records}, 'meta': meta}
    out = {'files': {m: str(d['path']) for m, d in data.items()}, 'models': {}, 'pairs': {}}
    for model, d in data.items():
        recs = list(d['records'].values())
        groups = {'overall': recs}
        for key in ('type', 'subtype'):
            for r in recs:
                groups.setdefault(f'{key}:{r[key]}', []).append(r)
        cfg = d['meta']
        out['models'][model] = {
            'complete': len(recs) == 388,
            'decoding': {k: cfg.get(k) for k in ('thinking', 'do_sample', 'temperature', 'top_p', 'top_k', 'min_p',
                                                 'repetition_penalty', 'max_new_tokens', 'batchsize', 'seed',
                                                 'attn_implementation')},
            'groups': {g: {**rate([bool(r['score']['correct']) for r in rs]),
                           'unparsed': sum(r['score']['prediction'] is None for r in rs),
                           'truncated': sum(r['finish_reason'] == 'length' for r in rs),
                           'truncated_repetitive': sum(r['finish_reason'] == 'length' and is_repetitive(r['response'])
                                                       for r in rs),
                           'median_tokens': float(np.median([r['generated_tokens'] for r in rs]))}
                       for g, rs in groups.items()},
            # Matched output budget: a response longer than CAP tokens counts as wrong for every model.
            'groups_cap': {g: rate([capped_correct(r) for r in rs]) for g, rs in groups.items()}}
    for first, second in PAIRS:
        if first not in data or second not in data:
            continue
        common = sorted(set(data[first]['records']) & set(data[second]['records']))
        res = {}
        for group in out['models'][first]['groups']:
            if group == 'overall':
                ids = common
            else:
                key, value = group.split(':', 1)
                ids = [i for i in common if data[first]['records'][i][key] == value]
            a = [bool(data[first]['records'][i]['score']['correct']) for i in ids]
            b = [bool(data[second]['records'][i]['score']['correct']) for i in ids]
            res[group] = {**mcnemar(a, b), 'diff': (np.mean(a) - np.mean(b)) if ids else float('nan')}
            ac = [capped_correct(data[first]['records'][i]) for i in ids]
            bc = [capped_correct(data[second]['records'][i]) for i in ids]
            res[group]['cap'] = {**mcnemar(ac, bc), 'diff': (np.mean(ac) - np.mean(bc)) if ids else float('nan')}
        subtype_groups = [g for g in res if g.startswith('subtype:')]
        for g, p in zip(subtype_groups, holm([res[g]['p'] for g in subtype_groups])):
            res[g]['p_holm'] = p
        out['pairs'][f'{first}|{second}'] = res
    return out


# ----------------------------------------------------------------------------- experiment 1.3 sub-tasks


def decomp_rows(output: Path) -> dict[str, dict[str, dict]]:
    """model -> case id -> merged case/score row (binary tasks: strict; location tasks: diagnostic)."""
    out = {}
    for model in MODELS:
        d = output / 'decomposition' / slug(model)
        if not (d / 'scores_main_diagnostic.jsonl').exists():
            continue
        strict, diag = load(d, kind='strict'), load(d, kind='diagnostic')
        if len(strict) != 37752:
            print(f'warning: {model} has {len(strict)}/37752 scored cases; statistics use the scored subset')
        dg = {r['id']: r for r in diag}
        out[model] = {r['id']: (r if r['format'] == 'yes_no' else dg[r['id']]) for r in strict}
    return out


def _subtask_filter(rows, task, variants, fmt, **kw):
    out = [r for r in rows if r['task'] == task and r['format'] == fmt and r['wording'] == 'primary']
    if variants:
        out = [r for r in out if r['variant'] in variants]
    for k, v in kw.items():
        out = [r for r in out if (r[k] in v if isinstance(v, (tuple, list, set)) else r[k] == v)]
    return out


def _flag(r, metric):
    return bool(r['score'].get(metric))


def decomposition(rows: dict[str, dict[str, dict]], n_boot_auroc: int = 500) -> dict:
    out = {'cells': {}, 'kind_cells': {}, 'level_cells': {}, 'auroc': {}, 'delta75_auroc': {},
           'delta75_fit': {}, 'pairs': {}, 'n_scored': {m: len(v) for m, v in rows.items()}}
    lists = {m: list(v.values()) for m, v in rows.items()}
    for model, rs in lists.items():
        for label, task, variants, fmt_, metric in SUBTASKS:
            out['cells'][f'{model}|{label}|all'] = rate([_flag(r, metric) for r in
                                                        _subtask_filter(rs, task, variants, fmt_)])
            for grid in GRIDS:
                sel = _subtask_filter(rs, task, variants, fmt_, grid=grid)
                out['cells'][f'{model}|{label}|{grid}'] = rate([_flag(r, metric) for r in sel])
                for kind in KINDS:
                    ks = [r for r in sel if r['kind'] == kind]
                    out['kind_cells'][f'{model}|{label}|{grid}|{kind}'] = rate([_flag(r, metric) for r in ks])
                    for level in range(1, 5):
                        ls = [r for r in ks if r['level'] == level]
                        out['level_cells'][f'{model}|{label}|{grid}|{kind}|{level}'] = rate(
                            [_flag(r, metric) for r in ls])
        # Original T3 wording: accuracy, AUROC of s and Yes rate per designation mode and grid.
        for label in ('T3 box', 'T3 row/col'):
            variants = next(v for l, _, v, *_ in SUBTASKS if l == label)
            for grid in list(GRIDS) + ['all']:
                sel = _subtask_filter(rs, 'T3', variants, 'yes_no', **({} if grid == 'all' else {'grid': grid}))
                out.setdefault('t3_original', {})[f'{model}|{label}|{grid}'] = binary_stats(sel)
        # Thresholds: AUROC-based (bias free) and accuracy psychometric fit, as in experiment 1.3.
        for kind in KINDS:
            mags = [numeric_magnitude(kind, l) for l in range(1, 5)]
            for label, task, variants, fmt_, _ in SUBTASKS:
                if label not in BINARY:
                    continue
                for grids in ((GRIDS,) if task != 'T3' else tuple((g,) for g in GRIDS)):
                    name = label if task != 'T3' else f'{label} g{grids[0]}'
                    sel = _subtask_filter(rs, task, variants, fmt_, kind=kind, grid=grids)
                    aucs = [binary_stats([r for r in sel if r['level'] == l])['auroc'] for l in range(1, 5)]
                    out['auroc'][f'{model}|{kind}|{name}'] = aucs
                    out['delta75_auroc'][f'{model}|{kind}|{name}'] = auroc_threshold(mags, aucs)
                    if not sel:
                        continue
                    x = np.log([numeric_magnitude(kind, r['level']) for r in sel])
                    c = np.array([r['score']['correct'] for r in sel], float)
                    out['delta75_fit'][f'{model}|{kind}|{name}'] = fit_psychometric(x, c)
    for first, second in PAIRS:
        if first not in rows or second not in rows:
            continue
        common = sorted(set(rows[first]) & set(rows[second]))
        a_rows = [rows[first][i] for i in common]
        res = {}
        for label, task, variants, fmt_, metric in SUBTASKS:
            for grid in list(GRIDS) + ['all']:
                sel = _subtask_filter(a_rows, task, variants, fmt_, **({} if grid == 'all' else {'grid': grid}))
                a = [_flag(rows[first][r['id']], metric) for r in sel]
                b = [_flag(rows[second][r['id']], metric) for r in sel]
                res[f'{label}|{grid}'] = {**mcnemar(a, b), 'diff': float(np.mean(a) - np.mean(b)) if sel else 0.0}
        keys = [k for k in res if not k.endswith('|all')]
        for k, p in zip(keys, holm([res[k]['p'] for k in keys])):
            res[k]['p_holm'] = p
        # Discrimination (AUROC) differences per binary sub-task, kind and level, pooled over grids.
        auc = {}
        for label, task, variants, fmt_, _ in SUBTASKS:
            if label not in BINARY:
                continue
            for kind in KINDS:
                for level in range(1, 5):
                    vals = []
                    for model in (first, second):
                        sel = _subtask_filter(lists[model], task, variants, fmt_, kind=kind, level=level)
                        vals.append(binary_stats(sel)['auroc'])
                    auc[f'{label}|{kind}|{level}'] = vals
        res['auroc_pooled'] = auc
        adiff = {}
        for label, task, variants, fmt_, _ in SUBTASKS:
            if label not in BINARY:
                continue
            for kind in KINDS:
                for grid in list(GRIDS) + ['all']:
                    sel = _subtask_filter(a_rows, task, variants, fmt_, kind=kind,
                                          **({} if grid == 'all' else {'grid': grid}))
                    sa = np.array([rows[first][r['id']]['score']['s'] for r in sel])
                    sb = np.array([rows[second][r['id']]['score']['s'] for r in sel])
                    y = np.array([r['truth']['answer_yes'] for r in sel], bool)
                    g = np.array([r['record_id'] for r in sel])
                    adiff[f'{label}|{kind}|{grid}'] = {
                        **auroc_diff(sa, sb, y, g, n_boot_auroc, 7), 'auroc_first': fast_auroc(sa, y),
                        'auroc_second': fast_auroc(sb, y), 'n': len(sel)}
        res['auroc_diff'] = adiff
        out['pairs'][f'{first}|{second}'] = res
    return out


# ----------------------------------------------------------------------------- T3 revised wording (C5)


def t3c5(output: Path, supplement: Path) -> dict:
    """Revised T3 wording of the 1.3 supplement; Qwen3.5-9B scores come from the supplement run."""
    cases = {c['id']: c for c in read_jsonl(supplement / 'cases_t3full.jsonl')}
    data = {}
    for model in MODELS:
        if model == 'Qwen3.5-9B':
            path = supplement / 'scores_t3full_strict.jsonl'
        else:
            path = output / 'decomposition' / slug(model) / 'results_t3c5.jsonl'
        if not path.exists():
            continue
        rows = {r['id']: r for r in read_jsonl(path)}
        if len(rows) != len(cases):
            print(f'warning: C5 T3 for {model} has {len(rows)}/{len(cases)} cases')
        data[model] = rows
    out = {'models': {}, 'pairs': {}}
    for model, rows in data.items():
        for mode in ('box', 'rowcol'):
            for grid in list(GRIDS) + ['all']:
                ids = [i for i in rows if cases[i]['mode'] == mode and (grid == 'all' or cases[i]['grid'] == grid)]
                pos = [rows[i]['s_same'] for i in ids if cases[i]['truth']['same']]
                neg = [rows[i]['s_same'] for i in ids if not cases[i]['truth']['same']]
                out['models'][f'{model}|{mode}|{grid}'] = {
                    **rate([bool(rows[i]['correct']) for i in ids]), 'auroc': auroc(pos, neg),
                    'same_rate': float(np.mean([rows[i]['s_same'] > 0 for i in ids])) if ids else float('nan')}
    for first, second in PAIRS:
        if first not in data or second not in data:
            continue
        res = {}
        for mode in ('box', 'rowcol'):
            for grid in list(GRIDS) + ['all']:
                ids = sorted(i for i in set(data[first]) & set(data[second])
                             if cases[i]['mode'] == mode and (grid == 'all' or cases[i]['grid'] == grid))
                a = [bool(data[first][i]['correct']) for i in ids]
                b = [bool(data[second][i]['correct']) for i in ids]
                res[f'{mode}|{grid}'] = {**mcnemar(a, b), 'diff': float(np.mean(a) - np.mean(b)) if ids else 0.0}
                sa = np.array([data[first][i]['s_same'] for i in ids])
                sb = np.array([data[second][i]['s_same'] for i in ids])
                y = np.array([cases[i]['truth']['same'] for i in ids], bool)
                g = np.array([cases[i]['record_id'] for i in ids])
                res[f'{mode}|{grid}']['auroc_diff'] = auroc_diff(sa, sb, y, g, 1000, 11)
        out['pairs'][f'{first}|{second}'] = res
    return out


# ----------------------------------------------------------------------------- RefCOCO


def refcoco(output: Path, n_boot: int, seed: int) -> dict:
    d = output / 'refcoco'
    if not (d / 'refcoco_cases.jsonl').exists():
        return {}
    cases = {c['id']: c for c in read_jsonl(d / 'refcoco_cases.jsonl')}
    build = json.loads((d / 'refcoco_build.json').read_text())
    scores = {}
    for model in MODELS:
        p = d / f'refcoco_scores_{slug(model)}.jsonl'
        if p.exists():
            scores[model] = {s['id']: s for s in read_jsonl(p)}
    out = {'build': build, 'models': {}, 'pairs': {}}
    names = [n for n, *_ in STRATA]
    for model, sc in scores.items():
        out['models'][model] = {}
        for name in names + ['all']:
            ids = [i for i in sc if name == 'all' or cases[i]['stratum'] == name]
            out['models'][model][name] = {
                **rate([sc[i]['correct'] for i in ids]),
                'valid': sum(sc[i]['valid'] for i in ids),
                'center_in': rate([sc[i]['center_in'] for i in ids])['acc'] if ids else float('nan'),
                'mean_iou': float(np.mean([sc[i]['iou'] for i in ids])) if ids else float('nan')}
    for k, (first, second) in enumerate(PAIRS):
        if first not in scores or second not in scores:
            continue
        res = {}
        for j, name in enumerate(names + ['all']):
            ids = sorted(i for i in set(scores[first]) & set(scores[second]) if name == 'all' or cases[i]['stratum'] == name)
            if not ids:
                res[name] = {'n': 0}
                continue
            a = np.array([scores[first][i]['correct'] for i in ids], float)
            b = np.array([scores[second][i]['correct'] for i in ids], float)
            res[name] = {'n': len(ids), 'delta': float(a.mean() - b.mean()),
                         'ci': paired_bootstrap(a, b, n_boot, seed + 10 * k + j),
                         **mcnemar(list(a.astype(bool)), list(b.astype(bool)))}
        # Trend: difference of Delta between the smallest and the largest populated stratum.
        populated = [n for n in names if res.get(n, {}).get('n')]
        lo, hi = populated[0], populated[-1]
        a_lo = np.array([scores[first][i]['correct'] for i in sorted(scores[first]) if cases[i]['stratum'] == lo], float)
        b_lo = np.array([scores[second][i]['correct'] for i in sorted(scores[first]) if cases[i]['stratum'] == lo], float)
        a_hi = np.array([scores[first][i]['correct'] for i in sorted(scores[first]) if cases[i]['stratum'] == hi], float)
        b_hi = np.array([scores[second][i]['correct'] for i in sorted(scores[first]) if cases[i]['stratum'] == hi], float)
        rng = np.random.default_rng(seed + 100 + k)
        boot = []
        for _ in range(n_boot):
            il, ih = rng.integers(0, len(a_lo), len(a_lo)), rng.integers(0, len(a_hi), len(a_hi))
            boot.append((a_lo[il].mean() - b_lo[il].mean()) - (a_hi[ih].mean() - b_hi[ih].mean()))
        res['trend'] = {'small': lo, 'large': hi,
                        'value': float((a_lo.mean() - b_lo.mean()) - (a_hi.mean() - b_hi.mean())),
                        'ci': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        out['pairs'][f'{first}|{second}'] = res
    return out


# ----------------------------------------------------------------------------- batching checks


def budget_extension(output: Path) -> dict:
    """Re-answers of a random sample of budget-truncated Qwen3-VL BabyVision items with a larger budget.

    The re-answer is a fresh sample (new batch seed), so it measures what a larger budget buys for these
    items, not a continuation of the truncated text.
    """
    out = {}
    for path in sorted((output / 'babyvision' / 'extension').glob('*.jsonl')):
        model = path.name.split('_babyvision')[0]
        out.setdefault(model, [])
        out[model] += read_jsonl(path)
    res = {}
    for model, rs in out.items():
        finished = [r for r in rs if r['finish_reason'] != 'length']
        res[model] = {'n': len(rs), 'finished': len(finished), 'correct': sum(bool(r['score']['correct']) for r in rs),
                      'repetitive_truncated': sum(is_repetitive(r['response']) for r in rs if r['finish_reason'] == 'length'),
                      'max_new_tokens': sorted({r.get('max_new_tokens') for r in rs} - {None}),
                      'median_tokens_finished': float(np.median([r['generated_tokens'] for r in finished])) if finished else None,
                      'correct_finished': rate([bool(r['score']['correct']) for r in finished])}
    return res


def batching_checks(output: Path, decomp_source: Path) -> dict:
    """Answer-level agreement of batched and batch-size-1 decoding.

    A pair agrees when both texts receive the same score (binary: same sign of s; location and
    count: same diagnostic correctness; RefCOCO: same IoU>=0.5 outcome).
    """
    from analysis_tools.behavior.decomposition import score_case
    from analysis_tools.deepstack.deepstack_comparison import score_refcoco_case

    cases = {c['id']: c for c in read_jsonl(decomp_source / 'cases.jsonl') if c['stage'] == 'main'}
    files = sorted((output / 'checks').glob('batch_validation_*.jsonl'))
    files += sorted((output / 'decomposition').glob('*/spotcheck_decomp_*.jsonl'))
    out = {}
    for path in files:
        rows = read_jsonl(path)
        agree = []
        for r in rows:
            c = cases[r['id']]
            if 's_batched' in r:
                agree.append((r['s_batched'] > 0) == (r['s_single'] > 0))
                continue
            a = score_case(c, {'response': r['batched'], 'finish_reason': 'eos'}, True)
            b = score_case(c, {'response': r['single'], 'finish_reason': 'eos'}, True)
            key = 'hit' if c['format'] == 'point' else 'correct'
            agree.append(bool(a.get(key)) == bool(b.get(key)))
        out[path.stem] = {'n': len(rows), 'text_equal': sum(r['text_equal'] for r in rows), 'answer_equal': sum(agree)}
    rc_dir = output / 'refcoco'
    if (rc_dir / 'refcoco_cases.jsonl').exists():
        rc_cases = {c['id']: c for c in read_jsonl(rc_dir / 'refcoco_cases.jsonl')}
        for path in sorted(rc_dir.glob('spotcheck_refcoco_*.jsonl')):
            rows = read_jsonl(path)
            agree = [score_refcoco_case(rc_cases[r['id']], {'response': r['batched'], 'finish_reason': 'eos'})['correct']
                     == score_refcoco_case(rc_cases[r['id']], {'response': r['single'], 'finish_reason': 'eos'})['correct']
                     for r in rows]
            out[path.stem] = {'n': len(rows), 'text_equal': sum(r['text_equal'] for r in rows),
                              'answer_equal': sum(agree)}
    return out


# ----------------------------------------------------------------------------- figures


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('pair', DIVERGING)


def plot_babyvision(bv: dict, out: Path) -> None:
    plt = _style()
    models = [m for m in MODELS if m in bv['models'] and bv['models'][m]['complete']]
    any_model = bv['models'][models[0]]['groups']
    by_type = defaultdict(list)
    files = {m: read_jsonl(Path(bv['files'][m])) for m in models}
    for r in files[models[0]]:
        if r['subtype'] not in by_type[r['type']]:
            by_type[r['type']].append(r['subtype'])
    labels = [('overall', 'Overall')]
    for t in sorted(by_type):
        labels.append((f'type:{t}', f'{t} (all)'))
        labels += [(f'subtype:{s}', f'   {s}') for s in sorted(by_type[t])]
    fig, ax = plt.subplots(figsize=(7.4, 0.3 * len(labels) + 0.9))
    ys = np.arange(len(labels))[::-1]
    offsets = {m: (i - (len(models) - 1) / 2) * 0.17 for i, m in enumerate(models)}
    for m in models:
        g = bv['models'][m]['groups']
        acc = np.array([g[k]['acc'] for k, _ in labels])
        lo = np.array([g[k]['ci'][0] for k, _ in labels])
        hi = np.array([g[k]['ci'][1] for k, _ in labels])
        ax.errorbar(acc, ys + offsets[m], xerr=[acc - lo, hi - acc], fmt='o', ms=4, color=COLORS[m], elinewidth=0.8,
                    capsize=0, label=m)
    ax.set_yticks(ys, [f'{name} (n={any_model[k]["n"]})' for k, name in labels], fontsize=8)
    for y, (k, _) in zip(ys, labels):
        if k == 'overall' or k.startswith('type:'):
            ax.axhline(y + 0.5, color=GRID_INK, lw=0.8)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel('accuracy (Wilson 95% CI)')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_decomp_heatmap(dec: dict, out: Path) -> None:
    """Accuracy difference (Qwen3-VL minus Qwen3.5) per sub-task and grid, both pairs side by side."""
    plt = _style()
    pairs = [p for p in PAIRS if f'{p[0]}|{p[1]}' in dec['pairs']]
    labels = [s[0] for s in SUBTASKS]
    cols = [str(g) for g in GRIDS] + ['all']
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.4 * len(pairs) + 1.6, 0.38 * len(labels) + 1.4), squeeze=False,
                             gridspec_kw={'wspace': 0.08})
    for k, (ax, (first, second)) in enumerate(zip(axes[0], pairs)):
        res = dec['pairs'][f'{first}|{second}']
        mat = np.array([[res[f'{l}|{c if c == "all" else int(c)}']['diff'] for c in cols] for l in labels])
        im = ax.imshow(mat, cmap=_diverging(), vmin=-0.6, vmax=0.6, aspect='auto')
        for i, l in enumerate(labels):
            for j, c in enumerate(cols):
                cell = res[f'{l}|{c if c == "all" else int(c)}']
                star = '*' if cell.get('p_holm', cell['p']) < 0.05 else ''
                ax.text(j, i, f'{cell["diff"]:+.2f}{star}', ha='center', va='center', fontsize=7.5, color=INK)
        ax.set_xticks(range(len(cols)), [f'{c}×{c}' if c != 'all' else 'all' for c in cols])
        ax.set_yticks(range(len(labels)), labels if k == 0 else [])
        ax.set_title(f'{SHORT[first]} − {SHORT[second]}')
        ax.grid(False)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label='accuracy difference (Qwen3-VL − Qwen3.5)')
    fig.savefig(out)
    plt.close(fig)


def plot_auroc_heatmap(dec: dict, out: Path) -> None:
    """AUROC difference (Qwen3-VL minus Qwen3.5) of the binary sub-tasks per change type and grid."""
    plt = _style()
    pairs = [p for p in PAIRS if f'{p[0]}|{p[1]}' in dec['pairs']]
    rows = [(l, k) for l in BINARY for k in KINDS]
    cols = [str(g) for g in GRIDS] + ['all']
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.4 * len(pairs) + 1.8, 0.38 * len(rows) + 1.4), squeeze=False,
                             gridspec_kw={'wspace': 0.08})
    for k, (ax, (first, second)) in enumerate(zip(axes[0], pairs)):
        res = dec['pairs'][f'{first}|{second}']['auroc_diff']
        mat = np.array([[res[f'{l}|{kind}|{c if c == "all" else int(c)}']['diff'] for c in cols] for l, kind in rows])
        im = ax.imshow(mat, cmap=_diverging(), vmin=-0.5, vmax=0.5, aspect='auto')
        for i, (l, kind) in enumerate(rows):
            for j, c in enumerate(cols):
                cell = res[f'{l}|{kind}|{c if c == "all" else int(c)}']
                star = '*' if cell['ci'][0] > 0 or cell['ci'][1] < 0 else ''
                ax.text(j, i, f'{cell["diff"]:+.2f}{star}', ha='center', va='center', fontsize=7.5, color=INK)
        ax.set_xticks(range(len(cols)), [f'{c}×{c}' if c != 'all' else 'all' for c in cols])
        ax.set_yticks(range(len(rows)), [f'{l} · {KIND_NAMES[kind]}' for l, kind in rows] if k == 0 else [])
        for y in (2.5, 5.5, 8.5):
            ax.axhline(y, color='white', lw=2)
        ax.set_title(f'{SHORT[first]} − {SHORT[second]}')
        ax.grid(False)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label='AUROC difference (Qwen3-VL − Qwen3.5)')
    fig.savefig(out)
    plt.close(fig)


def plot_decomp_curves(dec: dict, out: Path, labels: tuple[str, ...]) -> None:
    """Accuracy vs grid size per change type; one line per model, magnitude levels pooled."""
    plt = _style()
    models = [m for m in MODELS if any(k.startswith(m + '|') for k in dec['kind_cells'])]
    xs = np.arange(len(GRIDS))
    fig, axes = plt.subplots(len(KINDS), len(labels), figsize=(1.9 * len(labels) + 1.4, 1.9 * len(KINDS) + 0.6),
                             sharey=True, squeeze=False)
    for i, kind in enumerate(KINDS):
        for j, label in enumerate(labels):
            ax = axes[i, j]
            for m in models:
                stats = [dec['kind_cells'][f'{m}|{label}|{g}|{kind}'] for g in GRIDS]
                y = np.array([s['acc'] for s in stats])
                lo = np.array([s['ci'][0] for s in stats])
                hi = np.array([s['ci'][1] for s in stats])
                style = '-' if 'VL' in m else '--'
                ax.errorbar(xs, y, yerr=[y - lo, hi - y], color=COLORS[m], ls=style, lw=1.6, marker='o', ms=3.5,
                            elinewidth=0.8, capsize=0, label=m)
            ax.set_xticks(xs, [f'{g}' for g in GRIDS])
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(f'{KIND_NAMES[kind]}\naccuracy')
    axes[0, -1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.supxlabel('grid size (instances per side); magnitude levels pooled', fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_auroc_levels(dec: dict, out: Path) -> None:
    """AUROC of s per magnitude level for T1, T2 and T3 (row/col, 20x20), all four models."""
    plt = _style()
    models = [m for m in MODELS if any(k.startswith(m + '|') for k in dec['auroc'])]
    names = ('T1', 'T2', 'T3 box g6', 'T3 row/col g6', 'T3 row/col g20')
    fig, axes = plt.subplots(len(KINDS), len(names), figsize=(1.9 * len(names) + 1.4, 1.9 * len(KINDS) + 0.6),
                             sharey=True, squeeze=False)
    for i, kind in enumerate(KINDS):
        for j, name in enumerate(names):
            ax = axes[i, j]
            for m in models:
                ax.plot(range(1, 5), dec['auroc'][f'{m}|{kind}|{name}'], color=COLORS[m],
                        ls='-' if 'VL' in m else '--', lw=1.6, marker='o', ms=3.5, label=m)
            ax.axhline(0.5, ls=':', color=MUTED, lw=0.9)
            ax.axhline(0.75, ls='-', color=GRID_INK, lw=0.9)
            ax.set_xticks(range(1, 5), [magnitude_label(kind, l).replace(' ', '') for l in range(1, 5)], fontsize=7)
            ax.set_ylim(0.3, 1.02)
            if i == 0:
                ax.set_title(name)
            if j == 0:
                ax.set_ylabel(f'{KIND_NAMES[kind]}\nAUROC')
    axes[0, -1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.supxlabel('change magnitude', fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_refcoco(rc: dict, out: Path) -> None:
    plt = _style()
    names = [n for n, *_ in STRATA if rc['build']['sampled'][n]]
    tick = {'lt1': 'a<1%', '1to5': '1–5%', '5to10': '5–10%', 'ge10': '≥10%'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    xs = np.arange(len(names))
    for i, m in enumerate([m for m in MODELS if m in rc['models']]):
        st = [rc['models'][m][n] for n in names]
        y = np.array([s['acc'] for s in st])
        lo, hi = np.array([s['ci'][0] for s in st]), np.array([s['ci'][1] for s in st])
        ax1.errorbar(xs + (i - 1.5) * 0.07, y, yerr=[y - lo, hi - y], color=COLORS[m], ls='-' if 'VL' in m else '--',
                     marker='o', ms=4, lw=1.6, elinewidth=0.8, capsize=0, label=m)
    ax1.set_xticks(xs, [f'{tick[n]}\n(n={rc["build"]["sampled"][n]})' for n in names])
    ax1.set_ylabel('IoU ≥ 0.5 rate')
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7, loc='lower right')
    for k, (first, second) in enumerate(PAIRS):
        res = rc['pairs'].get(f'{first}|{second}')
        if not res:
            continue
        y = np.array([res[n]['delta'] for n in names])
        lo, hi = np.array([res[n]['ci'][0] for n in names]), np.array([res[n]['ci'][1] for n in names])
        ax2.errorbar(xs + (k - 0.5) * 0.1, y, yerr=[y - lo, hi - y], color=COLORS[first], marker='o', ms=4, lw=1.6,
                     elinewidth=0.8, capsize=0, label=f'{SHORT[first]} − {SHORT[second]}')
    ax2.axhline(0, color=MUTED, lw=0.9)
    ax2.set_xticks(xs, [tick[n] for n in names])
    ax2.set_ylabel(r'$\Delta_s$ (paired bootstrap 95% CI)')
    ax2.legend(fontsize=7)
    fig.supxlabel('target box area / image area', fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_refcoco_examples(output: Path, out: Path, n: int = 4, seed: int = 3) -> None:
    """Small-object RefCOCO inputs with the ground-truth box and the four models' predicted boxes."""
    from PIL import Image
    plt = _style()
    d = output / 'refcoco'
    cases = [c for c in read_jsonl(d / 'refcoco_cases.jsonl') if c['stratum'] == '1to5']
    scores = {m: {s['id']: s for s in read_jsonl(d / f'refcoco_scores_{slug(m)}.jsonl')} for m in MODELS
              if (d / f'refcoco_scores_{slug(m)}.jsonl').exists()}
    rng = np.random.default_rng(seed)
    picked = [cases[i] for i in rng.choice(len(cases), size=n, replace=False)]
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.4))
    for ax, c in zip(axes, picked):
        ax.imshow(Image.open(d / 'refcoco_images' / c['image']))
        x0, y0, x1, y1 = c['bbox']
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='white', lw=3.2))
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK, lw=1.6, ls='--'))
        for m, sc in scores.items():
            box = sc[c['id']].get('pred_box')
            if box:
                ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False,
                                           ec=COLORS[m], lw=1.6))
        ax.set_title(f'"{c["expression"]}"\n(a = {100 * c["area"]:.1f}%)', fontsize=8)
        ax.axis('off')
    handles = [plt.Line2D([], [], color=INK, ls='--', label='ground truth')] + [
        plt.Line2D([], [], color=COLORS[m], label=m) for m in scores]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables


def write_tables(summary: dict, path: Path) -> None:
    L = []
    bv = summary.get('babyvision')
    if bv:
        models = [m for m in MODELS if m in bv['models']]
        L += ['## BabyVision (local strict scoring)', '', '| group | ' + ' | '.join(models) + ' |',
              '| --- |' + ' --- |' * len(models)]
        groups = sorted({g for m in models for g in bv['models'][m]['groups']}, key=lambda g: (g != 'overall', g))
        for g in groups:
            L.append(f'| {g} | ' + ' | '.join(fmt(bv['models'][m]['groups'].get(g)) for m in models) + ' |')
        L += ['', f'### Matched output budget ({CAP} tokens; longer responses count as wrong)', '',
              '| group | ' + ' | '.join(models) + ' |', '| --- |' + ' --- |' * len(models)]
        for g in groups:
            L.append(f'| {g} | ' + ' | '.join(fmt(bv['models'][m]['groups_cap'].get(g)) for m in models) + ' |')
        for pair, res in bv['pairs'].items():
            L += ['', f'### McNemar {pair} (matched budget)', '', '| group | only VL | only 3.5 | diff | p |',
                  '| --- | --- | --- | --- | --- |']
            for g, r in res.items():
                c = r['cap']
                L.append(f"| {g} | {c['only_first']} | {c['only_second']} | {c['diff']:+.3f} | {fmt_p(c['p'])} |")
        for pair, res in bv['pairs'].items():
            L += ['', f'### McNemar {pair}', '', '| group | n | only VL | only 3.5 | diff | p | p (Holm, subtypes) |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
            for g, r in res.items():
                L.append(f"| {g} | {r['n_pairs']} | {r['only_first']} | {r['only_second']} | {r['diff']:+.3f} | "
                         f"{fmt_p(r['p'])} | {fmt_p(r['p_holm']) if 'p_holm' in r else ''} |")
        L += ['', '| model | unparsed | truncated | truncated with repetition | median tokens | budget |',
              '| --- | --- | --- | --- | --- | --- |']
        for m in models:
            g = bv['models'][m]['groups']['overall']
            L.append(f"| {m} | {g['unparsed']} | {g['truncated']} | {g['truncated_repetitive']} | "
                     f"{g['median_tokens']:.0f} | {bv['models'][m]['decoding']['max_new_tokens']} |")
    dec = summary.get('decomposition')
    if dec and dec['pairs']:
        for pair, res in dec['pairs'].items():
            first, second = pair.split('|')
            L += ['', f'## Sub-tasks {pair} (levels and change types pooled; location tasks: coordinate diagnostic)', '',
                  '| sub-task | grid | ' + f'{first} | {second} | diff | only VL | only 3.5 | p (Holm) |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
            for label, *_ in SUBTASKS:
                for grid in list(GRIDS) + ['all']:
                    r = res[f'{label}|{grid}']
                    a = dec['cells'].get(f'{first}|{label}|{grid}')
                    b = dec['cells'].get(f'{second}|{label}|{grid}')
                    L.append(f"| {label} | {grid} | {fmt(a) if a else ''} | {fmt(b) if b else ''} | {r['diff']:+.3f} | "
                             f"{r['only_first']} | {r['only_second']} | {fmt_p(r.get('p_holm', r['p']))} |")
        for pair, res in dec['pairs'].items():
            L += ['', f'## Binary sub-tasks: AUROC of s, {pair} (paired image bootstrap)', '',
                  '| sub-task | change | grid | AUROC VL | AUROC 3.5 | diff [95% CI] |', '| --- | --- | --- | --- | --- | --- |']
            for key, r in res['auroc_diff'].items():
                label, kind, grid = key.split('|')
                L.append(f"| {label} | {kind} | {grid} | {r['auroc_first']:.3f} | {r['auroc_second']:.3f} | "
                         f"{r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] |")
        models = [m for m in MODELS if m in dec['n_scored']]
        L += ['', '## delta_75 (AUROC criterion 0.75; log-linear interpolation)', '',
              '| change | sub-task | ' + ' | '.join(models) + ' |', '| --- | --- |' + ' --- |' * len(models)]
        def order(name):
            kind, label = name.split('|')
            grid = int(label.rsplit(' g', 1)[1]) if ' g' in label else 0
            return KINDS.index(kind), label.rsplit(' g', 1)[0], grid
        names = sorted({k.split('|', 1)[1] for k in dec['delta75_auroc']}, key=order)
        for name in names:
            cells = []
            for m in models:
                t = dec['delta75_auroc'][f'{m}|{name}']
                cells.append({'below_range': f'<{t["delta75"]:.3g}', 'above_range': f'>{t["delta75"]:.3g}'}.get(
                    t['status'], f'{t["delta75"]:.3g}'))
            kind, label = name.split('|')
            L.append(f'| {kind} | {label} | ' + ' | '.join(cells) + ' |')
        L += ['', '## delta_75 (accuracy psychometric fit, design 5.1)', '',
              '| change | sub-task | ' + ' | '.join(models) + ' |', '| --- | --- |' + ' --- |' * len(models)]
        for name in names:
            cells = []
            for m in models:
                t = dec['delta75_fit'].get(f'{m}|{name}')
                if t is None:
                    cells.append('–')
                    continue
                cells.append({'below_range': f'<{t["range"][0]:.3g}', 'above_range': f'>{t["range"][1]:.3g}'}.get(
                    t['status'], f'{t["delta75"]:.3g}'))
            kind, label = name.split('|')
            L.append(f'| {kind} | {label} | ' + ' | '.join(cells) + ' |')
    t3 = summary.get('t3c5')
    dec = summary.get('decomposition') or {}
    if t3 and t3['models']:
        models = [m for m in MODELS if any(k.startswith(m + '|') for k in t3['models'])]
        L += ['', '## T3: original wording vs revised wording C5 (acc / AUROC / same-rate)', '',
              '| mode | grid | ' + ' | '.join(models) + ' |', '| --- | --- |' + ' --- |' * len(models)]
        for mode, label in (('box', 'T3 box'), ('rowcol', 'T3 row/col')):
            for grid in list(GRIDS) + ['all']:
                cells = []
                for m in models:
                    o = dec.get('t3_original', {}).get(f'{m}|{label}|{grid}')
                    c = t3['models'].get(f'{m}|{mode}|{grid}')
                    orig = f"{o['acc']:.2f}/{o['auroc']:.2f}/{o['yes_rate']:.2f}" if o else '–'
                    new = f"{c['acc']:.2f}/{c['auroc']:.2f}/{c['same_rate']:.2f}" if c and c['n'] else '–'
                    cells.append(f'orig {orig}; C5 {new}')
                L.append(f'| {mode} | {grid} | ' + ' | '.join(cells) + ' |')
        for pair, res in t3['pairs'].items():
            L += ['', f'### C5 McNemar and AUROC difference {pair}', '',
                  '| mode | grid | acc diff | only VL | only 3.5 | p | AUROC diff [95% CI] |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
            for k, r in res.items():
                mode, grid = k.split('|')
                a = r['auroc_diff']
                L.append(f"| {mode} | {grid} | {r['diff']:+.3f} | {r['only_first']} | {r['only_second']} | {fmt_p(r['p'])} | "
                         f"{a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] |")
    rc = summary.get('refcoco')
    if rc:
        models = [m for m in MODELS if m in rc['models']]
        L += ['', '## RefCOCO val (IoU >= 0.5)', '', '| stratum | ' + ' | '.join(models) + ' |',
              '| --- |' + ' --- |' * len(models)]
        for n in [s[0] for s in STRATA] + ['all']:
            L.append(f'| {n} | ' + ' | '.join(fmt(rc['models'][m][n]) for m in models) + ' |')
        for pair, res in rc['pairs'].items():
            L += ['', f'### Delta_s {pair}', '', '| stratum | n | Delta | 95% CI | McNemar p |', '| --- | --- | --- | --- | --- |']
            for n in [s[0] for s in STRATA] + ['all']:
                r = res[n]
                if r['n']:
                    L.append(f"| {n} | {r['n']} | {r['delta']:+.3f} | [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {fmt_p(r['p'])} |")
            t = res['trend']
            L.append(f"\nTrend Delta({t['small']}) - Delta({t['large']}) = {t['value']:+.3f} "
                     f"[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]")
    if summary.get('batching_checks'):
        L += ['', '## Batched vs batch-size-1 decoding', '', '| check | n | identical text | identical answer score |',
              '| --- | --- | --- | --- |']
        for name, r in summary['batching_checks'].items():
            L.append(f"| {name} | {r['n']} | {r['text_equal']} | {r['answer_equal']} |")
    if summary.get('budget_extension'):
        L += ['', '## BabyVision budget extension (truncated Qwen3-VL items re-answered with a larger budget)', '',
              '| model | n | finished | correct | still truncated, repetitive | median tokens (finished) |',
              '| --- | --- | --- | --- | --- | --- |']
        for model, r in summary['budget_extension'].items():
            L.append(f"| {model} | {r['n']} | {r['finished']} | {r['correct']} | {r['repetitive_truncated']} | "
                     f"{r['median_tokens_finished']} |")
    path.write_text('\n'.join(L) + '\n')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--decomp-source', type=Path, required=True)
    p.add_argument('--legacy-babyvision', type=Path, required=True)
    p.add_argument('--t3-supplement', type=Path, required=True)
    p.add_argument('--n-boot', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    target = args.output / 'analysis'
    (target / 'figures').mkdir(parents=True, exist_ok=True)
    summary = {'babyvision': babyvision(args.output, args.legacy_babyvision)}
    rows = decomp_rows(args.output)
    summary['decomposition'] = decomposition(rows) if rows else {}
    summary['refcoco'] = refcoco(args.output, args.n_boot, args.seed)
    summary['t3c5'] = t3c5(args.output, args.t3_supplement)
    summary['batching_checks'] = batching_checks(args.output, args.decomp_source)
    summary['budget_extension'] = budget_extension(args.output)
    (target / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    write_tables(summary, target / 'tables.md')
    fig = target / 'figures'
    if summary['babyvision']['models']:
        plot_babyvision(summary['babyvision'], fig / 'babyvision_subtypes.png')
    if summary['decomposition'].get('pairs'):
        plot_decomp_heatmap(summary['decomposition'], fig / 'decomp_diff_heatmap.png')
        plot_auroc_heatmap(summary['decomposition'], fig / 'decomp_auroc_diff_heatmap.png')
        plot_decomp_curves(summary['decomposition'], fig / 'decomp_curves_binary.png', ('T1', 'T2', 'T3 box', 'T3 row/col'))
        plot_decomp_curves(summary['decomposition'], fig / 'decomp_curves_locate.png',
                           ('T4 row/col', 'T5 row/col', 'T5 point', 'T6 row/col', 'T6 point'))
        plot_auroc_levels(summary['decomposition'], fig / 'decomp_auroc_levels.png')
    if summary['refcoco'].get('pairs'):
        plot_refcoco(summary['refcoco'], fig / 'refcoco_strata.png')
        plot_refcoco_examples(args.output, fig / 'refcoco_examples.png')
    print((target / 'tables.md').read_text()[:3000])


if __name__ == '__main__':
    main()
