"""Experiment 1.7 analysis: taxonomy tables, faithfulness statistics, information-flow statistics, figures.

All numbers are computed from the artifacts of cot_mechanism_taxonomy.py and cot_mechanism_run.py;
nothing is fitted except where a figure caption in the results doc states otherwise.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.behavior import cot_mechanism as cm
from analysis_tools.behavior import cot_mechanism_engine as ce

PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GREY = '#9a9994'
INK = '#0b0b0b'
INK2 = '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def prop(k: int, n: int) -> dict:
    return {'k': k, 'n': n, 'p': k / n if n else float('nan'), 'ci': wilson(k, n)}


# ----------------------------------------------------------------------------- taxonomy

GROUPS = {
    # display name: predicate on a coded record
    'red target, 8x8 (3.10)': lambda r: r['source'] == 'rc310_think' and r['extra']['prefill_edit'] == 'none'
                                         and r['mode'] == 'think_off',
    'red target, ordinal layouts (1.5)': lambda r: r['source'] == 'ord15_think',
    'find-the-odd and T4-T6 (1.3, 1.6)': lambda r: r['source'] in ('dec13_explicit', 'pe16_explicit')
                                                   and r['task'] in ('rowcol', 'odd'),
    'find-the-odd (2.6)': lambda r: r['source'] == 'odd26_think',
    'fresh rowcol (1.7)': lambda r: r['source'] == 'fresh17' and r['task'] == 'rowcol' and r['mode'] == 'think_off',
    'fresh odd (1.7)': lambda r: r['source'] == 'fresh17' and r['task'] == 'odd' and r['mode'] == 'think_off',
    'fresh point (1.7)': lambda r: r['source'] == 'fresh17' and r['task'] == 'point' and r['mode'] == 'think_off',
    'fresh count (1.7)': lambda r: r['source'] == 'fresh17' and r['task'] == 'count' and r['mode'] == 'think_off',
    'prefill-edited 8x8/V64 (3.10)': lambda r: r['source'] == 'rc310_think' and r['extra']['prefill_edit'] != 'none'
                                               and r['mode'] == 'think_off',
    'thinking on (1.3, 3.10, 1.7)': lambda r: r['mode'] == 'thinking_on' and r['task'] in ('rowcol', 'odd', 'point')
                                              and not (r['source'] == 'rc310_think' and r['extra']['prefill_edit'] != 'none'),
}
MODE_ORDER = ['clean', 'grid_wrong_but_right', 'disclaimer_but_right', 'cot_wrong_answer_right',
              'enum_mislabel', 'assert_wrong', 'grid_wrong', 'arith_slip', 'disclaimer', 'inconsistent',
              'no_claim', 'truncated']
MODE_LABEL = {
    'clean': 'correct, CoT correct', 'grid_wrong_but_right': 'correct, grid size wrong',
    'disclaimer_but_right': 'correct, "cannot see"', 'cot_wrong_answer_right': 'correct, CoT location wrong',
    'enum_mislabel': 'wrong: enumerated, label off', 'assert_wrong': 'wrong: asserted location',
    'grid_wrong': 'wrong: grid size wrong', 'arith_slip': 'wrong: arithmetic slip',
    'disclaimer': 'wrong: "cannot see"', 'inconsistent': 'wrong: answer != CoT', 'no_claim': 'wrong: no location stated',
    'truncated': 'truncated',
}
MODE_COLOR = {'clean': '#1baf7a', 'grid_wrong_but_right': '#7fd3b3', 'disclaimer_but_right': '#b9e6d4',
              'cot_wrong_answer_right': '#dff3ea', 'enum_mislabel': '#2a78d6', 'assert_wrong': '#86b6ef',
              'grid_wrong': '#6250d6', 'arith_slip': '#e87ba4', 'disclaimer': '#eb6834', 'inconsistent': '#e34948',
              'no_claim': '#eda100', 'truncated': GREY}


def point_mode(code: dict) -> str:
    """Failure label for point answers, where 'correct' means the point lies in the target cell."""
    return cm.primary_mode(code)


def count_mode(code: dict) -> str:
    if code.get('correct'):
        return 'clean'
    if code.get('truncated') or not code.get('has_answer'):
        return 'truncated'
    if code.get('disclaimer'):
        return 'disclaimer'
    if code.get('consistent') is False:
        return 'inconsistent'
    if code.get('row_enum') or code.get('col_enum'):
        return 'enum_mislabel'
    return 'assert_wrong'


def _label(r: dict) -> str:
    if r['task'] == 'count':
        return count_mode(r['code'])
    if r['task'] == 'point':
        # stated rows/columns in point CoTs mix 0-based and 1-based indices, so CoT-location coding is not
        # used; the label is correct / truncated / other wrong
        c = r['code']
        return 'clean' if c.get('correct') else ('truncated' if c.get('truncated') or not c.get('has_answer') else 'assert_wrong')
    return r['code']['mode_label']


def taxonomy(coded: list[dict]) -> dict:
    out = {}
    for name, pred in GROUPS.items():
        rows = [r for r in coded if pred(r)]
        if not rows:
            continue
        labels = Counter(_label(r) for r in rows)
        n = len(rows)
        flags = {}
        for f in ('disclaimer', 'assume', 'truncated'):
            flags[f] = prop(sum(bool(r['code'].get(f)) for r in rows), n)
        decl = [r for r in rows if r['code'].get('grid_rows_ok') is not None or r['code'].get('grid_cols_ok') is not None]
        flags['grid_declared_wrong'] = prop(sum((r['code'].get('grid_rows_ok') is False) or (r['code'].get('grid_cols_ok') is False)
                                                for r in decl), len(decl))
        cons = [r for r in rows if r['code'].get('consistent') is not None and r['task'] != 'point']
        flags['answer_follows_cot'] = prop(sum(r['code']['consistent'] for r in cons), len(cons))
        flags['row_enumeration'] = prop(sum(bool(r['code'].get('row_enum')) for r in rows), n)
        flags['self_correction'] = prop(sum(r['code'].get('n_self_correct', 0) > 0 for r in rows), n)
        acc = prop(sum(bool(r['code'].get('correct')) for r in rows), n)
        out[name] = {'n': n, 'accuracy': acc, 'modes': dict(labels), 'flags': flags}
    # disclaimer vs accuracy (thinking off, unedited)
    dis = {}
    for task in ('rowcol', 'odd', 'binary', 'babyvision', 'point', 'count'):
        rows = [r for r in coded if r['task'] == task and r['mode'] == 'think_off'
                and not (r['source'] == 'rc310_think' and r['extra']['prefill_edit'] != 'none')]
        for d in (False, True):
            sub = [r for r in rows if r['code']['disclaimer'] == d]
            dis[f'{task}|{"disclaimer" if d else "none"}'] = prop(sum(bool(r['code'].get('correct')) for r in sub), len(sub))
    out['_disclaimer_accuracy'] = dis
    # disclaimer rate by perceptual difficulty in find-the-odd (2.6)
    dd = {}
    for r in coded:
        if r['source'] == 'odd26_think' and r['extra']['group'] == 'core':
            dd.setdefault(f"{r['extra']['kind']}_{r['extra']['variant']}", []).append(r)
    out['_odd26_disclaimer_by_condition'] = {k: {'disclaimer': prop(sum(x['code']['disclaimer'] for x in v), len(v)),
                                                  'accuracy': prop(sum(bool(x['code'].get('correct')) for x in v), len(v))}
                                             for k, v in sorted(dd.items())}
    # prefill-edited corpus: enumerated red-row label vs truth and answer
    ed = [r for r in coded if r['source'] == 'rc310_think' and r['extra']['prefill_edit'] != 'none'
          and r['mode'] == 'think_off' and r['code'].get('red_row_labels')]
    out['_edited_enumeration'] = {
        'n_with_red_label': len(ed),
        'label_eq_truth': prop(sum(r['code']['enum_red_row_ok'] for r in ed), len(ed)),
        'label_eq_answer': prop(sum(bool(r['code'].get('enum_red_row_eq_answer')) for r in ed), len(ed)),
        'label_wrong_and_eq_answer': prop(sum((not r['code']['enum_red_row_ok']) and bool(r['code'].get('enum_red_row_eq_answer'))
                                              for r in ed), len(ed)),
    }
    cl = [r for r in coded if GROUPS['red target, 8x8 (3.10)'](r) and r['code'].get('red_row_labels')]
    out['_clean_enumeration'] = {'n_with_red_label': len(cl),
                                 'label_eq_truth': prop(sum(r['code']['enum_red_row_ok'] for r in cl), len(cl)),
                                 'label_eq_answer': prop(sum(bool(r['code'].get('enum_red_row_eq_answer')) for r in cl), len(cl))}
    # BabyVision flags by subtype
    bv = defaultdict(list)
    for r in coded:
        if r['source'] == 'bv16':
            bv[r['condition']].append(r)
    out['_babyvision'] = {k: {'n': len(v), 'correct': prop(sum(x['code']['correct'] for x in v), len(v)),
                              'disclaimer': prop(sum(x['code']['disclaimer'] for x in v), len(v)),
                              'assume': prop(sum(x['code']['assume'] for x in v), len(v)),
                              'self_correction': prop(sum(x['code']['n_self_correct'] > 0 for x in v), len(v))}
                          for k, v in sorted(bv.items())}
    return out


def fig_taxonomy(tax: dict, path: Path) -> None:
    names = [k for k in GROUPS if k in tax]
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(names) + 1.6))
    y = np.arange(len(names))[::-1]
    for yi, name in zip(y, names):
        left = 0.0
        n = tax[name]['n']
        for m in MODE_ORDER:
            v = tax[name]['modes'].get(m, 0) / n
            if v > 0:
                ax.barh(yi, v, left=left, color=MODE_COLOR[m], edgecolor='white', linewidth=1.0, height=0.7)
                if v >= 0.06:
                    ax.text(left + v / 2, yi, f'{100 * v:.0f}', ha='center', va='center', fontsize=7,
                            color='white' if m in ('enum_mislabel', 'grid_wrong', 'disclaimer', 'inconsistent', 'clean') else INK)
            left += v
        ax.text(1.01, yi, f'n={n}', va='center', fontsize=7, color=INK2)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1)
    ax.set_xlabel('share of responses (numbers in bars: percent)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLOR[m]) for m in MODE_ORDER]
    ax.legend(handles, [MODE_LABEL[m] for m in MODE_ORDER], ncol=3, fontsize=7, frameon=False,
              loc='upper center', bbox_to_anchor=(0.45, -0.18 if len(names) > 6 else -0.25))
    ax.set_title('Primary CoT failure mode per response (rule coding)', loc='left', fontsize=10)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- faithfulness

def _answer(task: str, text: str, forced_prefix: str | None = None):
    if forced_prefix is not None:
        text = forced_prefix + text
    a, _ = cm.final_answer(task, text)
    return a


def _same(task: str, a, b) -> bool | None:
    if a is None or b is None:
        return None
    if task == 'point':
        return math.hypot(a[0] - b[0], a[1] - b[1]) <= 25        # within a quarter cell of a 10x10 grid
    return tuple(a) == tuple(b)


def _correct(task: str, a, truth: dict) -> bool | None:
    if a is None:
        return None
    if task in ('rowcol', 'odd'):
        return a[0] == truth['row'] and a[1] == truth['col']
    if task == 'count':
        return a[0] == truth['rows'] and a[1] == truth['cols']
    x0, y0, x1, y1 = truth['cell_box_norm']
    return x0 <= a[0] <= x1 and y0 <= a[1] <= y1


def faithfulness(run: Path) -> dict:
    gen = {r['id']: r for r in read(run / 'gen/records.jsonl')}
    fa = [r for f in sorted((run / 'faith').glob('records*.jsonl')) for r in read(f)]
    out = {'truncate': {}, 'edit': {}, 'answer_first': {}}
    trunc = defaultdict(lambda: defaultdict(list))
    for r in fa:
        src = gen[r['src']]
        task = r['task']
        full = cm.final_answer(task, src['text'])[0]
        if r['test'] == 'truncate':
            cue = cm.FORCE_CUE[task].split('Final answer: ')[1]
            a = _answer(task, r['text'], cue)
            trunc[task][r['fraction']].append((_same(task, a, full), _correct(task, a, src['truth']), _correct(task, full, src['truth'])))
    for task, d in trunc.items():
        out['truncate'][task] = {}
        for f, v in sorted(d.items()):
            n = len(v)
            out['truncate'][task][f'{f:.1f}'] = {'agree_full': prop(sum(bool(x[0]) for x in v), n),
                                                'accuracy': prop(sum(bool(x[1]) for x in v), n),
                                                'accuracy_full': prop(sum(bool(x[2]) for x in v), n)}
    # edits
    ed = defaultdict(list)
    for r in fa:
        if not r['test'].startswith(('edit_', 'keep_')):
            continue
        src = gen[r['src']]
        task = r['task']
        a = cm.final_answer(task, r['text'])[0]
        full = cm.final_answer(task, src['text'])[0]
        if task in ('rowcol', 'odd'):
            val = None if a is None else a[0]
        elif task == 'count':
            val = None if a is None else a[0]
        else:   # point: row of the cell that contains the answer point
            rows = src['truth']['rows']
            val = None if a is None else int(min(rows - 1, max(0, a[1] / 1000 * rows))) + 1
        ed[(task, r['test'])].append({'follows_new': val == r['new_value'], 'keeps_orig': val == r['orig_value'],
                                      'valid': a is not None, 'same_as_full': _same(task, a, full)})
    for (task, test), v in sorted(ed.items()):
        n = len(v)
        out['edit'][f'{task}|{test}'] = {'n': n, 'follows_new': prop(sum(x['follows_new'] for x in v), n),
                                        'keeps_orig': prop(sum(x['keeps_orig'] for x in v), n),
                                        'invalid': prop(sum(not x['valid'] for x in v), n)}
    # answer-first vs direct vs think
    by_item = defaultdict(dict)
    for r in gen.values():
        if r['mode'] in ('direct', 'think', 'answer_first'):
            a = cm.first_answer(r['task'], r['text']) if r['mode'] == 'answer_first' else cm.final_answer(r['task'], r['text'])[0]
            by_item[(r['task'], r['item'])][r['mode']] = (a, r)
    for task in ('rowcol', 'odd', 'point', 'count'):
        its = [v for (t, _), v in by_item.items() if t == task and len(v) == 3]
        if not its:
            continue
        res = {}
        for m in ('direct', 'think', 'answer_first'):
            res[f'acc_{m}'] = prop(sum(bool(_correct(task, v[m][0], v[m][1]['truth'])) for v in its), len(its))
        for a, b in (('direct', 'think'), ('direct', 'answer_first'), ('think', 'answer_first')):
            res[f'agree_{a}_{b}'] = prop(sum(bool(_same(task, v[a][0], v[b][0])) for v in its), len(its))
            bb = sum(bool(_correct(task, v[a][0], v[a][1]['truth'])) and not _correct(task, v[b][0], v[b][1]['truth']) for v in its)
            cc = sum((not _correct(task, v[a][0], v[a][1]['truth'])) and bool(_correct(task, v[b][0], v[b][1]['truth'])) for v in its)
            res[f'mcnemar_{a}_{b}'] = {'b': bb, 'c': cc, 'p': mcnemar(bb, cc)}
        # does the answer-first explanation restate a different answer at the end?
        changed = 0
        for v in its:
            a0 = v['answer_first'][0]
            alast = cm.final_answer(task, v['answer_first'][1]['text'])[0]
            changed += bool(a0 is not None and alast is not None and not _same(task, a0, alast))
        res['answer_first_revised_later'] = prop(changed, len(its))
        out['answer_first'][task] = res
    return out


def fig_faith(fa: dict, path: Path) -> None:
    tasks = [t for t in ('rowcol', 'odd', 'point', 'count') if t in fa['truncate']]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for i, t in enumerate(tasks):
        d = fa['truncate'][t]
        fs = sorted(float(f) for f in d)
        ag = [d[f'{f:.1f}']['agree_full']['p'] for f in fs]
        lo = [d[f'{f:.1f}']['agree_full']['ci'][0] for f in fs]
        hi = [d[f'{f:.1f}']['agree_full']['ci'][1] for f in fs]
        axes[0].plot(fs, ag, '-o', color=PALETTE[i], lw=2, ms=4, label=t)
        axes[0].fill_between(fs, lo, hi, color=PALETTE[i], alpha=0.12, lw=0)
        ac = [d[f'{f:.1f}']['accuracy']['p'] for f in fs]
        axes[1].plot(fs, ac, '-o', color=PALETTE[i], lw=2, ms=4, label=t)
        axes[1].axhline(d['1.0']['accuracy_full']['p'], color=PALETTE[i], ls=':', lw=1)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[0].set_xlabel('fraction of CoT kept before the forced answer')
    axes[1].set_xlabel('fraction of CoT kept before the forced answer')
    axes[0].set_ylabel('forced answer = full-CoT answer')
    axes[1].set_ylabel('forced-answer accuracy (dotted: full CoT)')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title('(a) agreement with the full-CoT answer', loc='left', fontsize=9)
    axes[1].set_title('(b) accuracy', loc='left', fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def fig_edit(fa: dict, path: Path) -> None:
    keys = [k for k in fa['edit']]
    tasks = [t for t in ('rowcol', 'odd', 'point', 'count') if any(k.startswith(t + '|') for k in keys)]
    tests = ['edit_last', 'keep_last', 'edit_first', 'keep_first']
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    w = 0.2
    for j, test in enumerate(tests):
        xs, ys, los, his = [], [], [], []
        for i, t in enumerate(tasks):
            d = fa['edit'].get(f'{t}|{test}')
            if not d:
                continue
            xs.append(i + (j - 1.5) * w)
            ys.append(d['follows_new']['p'] if test.startswith('edit') else d['keeps_orig']['p'])
            los.append(ys[-1] - d['follows_new' if test.startswith('edit') else 'keeps_orig']['ci'][0])
            his.append(d['follows_new' if test.startswith('edit') else 'keeps_orig']['ci'][1] - ys[-1])
        color = PALETTE[0] if 'last' in test else PALETTE[1]
        ax.bar(xs, ys, width=w * 0.9, color=color, alpha=1.0 if test.startswith('edit') else 0.35,
               yerr=[los, his], error_kw={'lw': 1, 'ecolor': INK2}, label={
                   'edit_last': 'last stated row edited: answer = edited value',
                   'keep_last': 'last stated row, no edit: answer = stated value',
                   'edit_first': 'first stated row edited: answer = edited value',
                   'keep_first': 'first stated row, no edit: answer = stated value'}[test])
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('share of continuations')
    ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.set_title('Answer after editing a stated row (count: stated total of rows)', loc='left', fontsize=9)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- mechanism

def _follow(task: str, ans, ta: dict, tb: dict) -> str:
    if ans is None:
        return 'invalid'
    if task == 'point':
        da = math.hypot(ans[0] - ta['point'][0], ans[1] - ta['point'][1])
        db = math.hypot(ans[0] - tb['point'][0], ans[1] - tb['point'][1])
        return 'A' if da < db else 'B'
    if ans[0] == ta['row'] and ans[1] == ta['col']:
        return 'A'
    if ans[0] == tb['row'] and ans[1] == tb['col']:
        return 'B'
    near_a = abs(ans[0] - ta['row']) + abs(ans[1] - ta['col'])
    near_b = abs(ans[0] - tb['row']) + abs(ans[1] - tb['col'])
    return 'nearA' if near_a < near_b else ('nearB' if near_b < near_a else 'other')


def _first_claim(task: str, text: str):
    code = cm.code_cot(task if task != 'point' else 'point', text, 'eos',
                       {'row': 0, 'col': 0, 'rows': 99, 'cols': 99, 'point': [0, 0]})
    return code.get('first_row_claim'), code.get('first_col_claim')


def mechanism(run: Path) -> dict:
    recs = read(run / 'mech/records.jsonl')
    out = {'swap': {}, 'claims': {}}
    by = defaultdict(list)
    for r in recs:
        by[(r['task'], r['condition'])].append(r)
    for (task, cond), rows in sorted(by.items()):
        c = Counter()
        cc = Counter()
        for r in rows:
            ans = cm.final_answer(task, r['text'])[0] if r['finish_reason'] == 'eos' else None
            f = _follow(task, ans, r['truth_a'], r['truth_b'])
            c[f] += 1
            fr = _first_claim(task, r['text'])
            if fr[0] is None:
                cc['none'] += 1
            elif fr[0] == r['truth_a']['row']:
                cc['A'] += 1
            elif fr[0] == r['truth_b']['row']:
                cc['B'] += 1
            else:
                cc['other'] += 1
        n = len(rows)
        out['swap'][f'{task}|{cond}'] = {'n': n, **{k: prop(c[k], n) for k in ('A', 'nearA', 'B', 'nearB', 'other', 'invalid')}}
        out['claims'][f'{task}|{cond}'] = {'n': n, **{k: prop(cc[k], n) for k in ('A', 'B', 'other', 'none')}}
    # paired agreement with clean_A / clean_B texts
    texts = {(r['task'], r['condition'], r['item_a']): r for r in recs}
    agree = {}
    for (task, cond, item), r in texts.items():
        for ref in ('clean_A', 'clean_B'):
            o = texts.get((task, ref, item))
            if o is None:
                continue
            key = f'{task}|{cond}|text_eq_{ref}'
            agree.setdefault(key, [0, 0])
            agree[key][0] += r['text'] == o['text']
            agree[key][1] += 1
    out['text_equal'] = {k: prop(*v) for k, v in sorted(agree.items())}
    # link to 3.10: image KV from a prefill with DeltaNet alpha=1 at image tokens (layers 9-16)
    ed = defaultdict(list)
    for r in read(run / 'mech_edit/records.jsonl'):
        a = cm.final_answer('rowcol', r['text'])[0] if r['finish_reason'] == 'eos' else None
        ed[r['condition']].append((a, r['truth_a']))
    out['prefill_edit'] = {}
    for cond, v in ed.items():
        n = len(v)
        valid = [(a, t) for a, t in v if a is not None]
        out['prefill_edit'][cond] = {
            'n': n, 'row_correct': prop(sum(a[0] == t['row'] for a, t in valid), n),
            'row_under': prop(sum(a[0] < t['row'] for a, t in valid), n),
            'invalid': prop(n - len(valid), n),
            'mean_row_error_valid': float(np.mean([a[0] - t['row'] for a, t in valid])) if valid else None}
    return out


TOKEN_CLASSES = ('digit', 'colour', 'row/column', 'figure noun', 'other word', 'format')


def token_class(tok: str) -> str:
    t = tok.strip().lower()
    if not t or all(ch in '{}[]":,.*-#()→\n' for ch in t):
        return 'format'
    if any(ch.isdigit() for ch in t):
        return 'digit'
    if t in ('red', 'black', 'colored', 'coloured', 'color', 'colour', 'gray', 'grey', 'blue', 'white'):
        return 'colour'
    if t in ('row', 'rows', 'column', 'columns', 'col'):
        return 'row/column'
    if t in ('figure', 'figures', 'circle', 'circles', 'ring', 'rings', 'person', 'stick', 'arrow', 'arrows', 'gap',
             'square', 'squares'):
        return 'figure noun'
    return 'other word'


def attention(run: Path, tokenizer) -> dict:
    """Image-attention statistics from the probe files of clean_A and img_from_B runs."""
    recs = read(run / 'mech/records.jsonl')
    by_batch = defaultdict(list)
    for r in recs:
        by_batch[r['batch_id']].append(r)
    out = {}
    for task in ('rowcol', 'odd', 'point'):
        res = {'by_class': defaultdict(lambda: defaultdict(list)), 'curve': [], 'target_share': [], 'h11_target': [],
               'pre_img_txt_gen': []}
        for f in sorted((run / 'mech/probes').glob(f'{task}_b*_clean_A.npz')):
            z = np.load(f)
            bid = f.stem.replace(f'{task}_b', '').replace('_clean_A', '')
            rows = sorted(by_batch[f'{task}|b{bid}|clean_A'], key=lambda r: r['row_in_batch'])
            img = z['img'].astype(np.float32)         # (steps, B, L, H)
            maps = z['map'].astype(np.float32)        # (steps, B, L, n_img)
            h11 = z['map_h11'].astype(np.float32)     # (steps, B, n_img)
            for b, r in enumerate(rows):
                n = r['n_tokens']
                ids = r['ids']
                m = img[:n, b].mean(-1)              # (n, L) head-mean image mass
                res['curve'].append(m)
                parts = np.stack([z[k][:n, b].astype(np.float32).mean(-1) for k in ('pre', 'img', 'txt', 'gen')], -1)
                res['pre_img_txt_gen'].append(parts.mean(0))    # (L, 4)
                for t in range(n):
                    cls = token_class(tokenizer.decode([ids[t]]))
                    for li, l in enumerate(ce.FULL_LAYERS):
                        res['by_class'][cls][l].append(float(m[t, li]))
                # target-cell share of image attention
                sp = r['sp']
                gh, gw = r.get('grid_tokens') or _grid_tokens(r)
                tmask = _cell_mask(r['truth_a'], gh, gw)
                share = (maps[:n, b] * tmask[None, None]).sum(-1) / np.maximum(maps[:n, b].sum(-1), 1e-9)   # (n, L)
                res['target_share'].append(share)
                res['h11_target'].append((h11[:n, b] * tmask[None]).sum(-1) / np.maximum(h11[:n, b].sum(-1), 1e-9))
                res.setdefault('uniform_share', []).append(float(tmask.mean()))
                # row tracking: while a line "Row k: ..." is being written, where does image attention go?
                rows_n = r['truth_a']['rows']
                row_of_tok = _token_rows(gh, gw, rows_n)                  # (n_img,) 1-based grid row
                labels = _current_row_label(tokenizer, ids[:n])
                for t, k in enumerate(labels):
                    if k is None or not 1 <= k <= rows_n:
                        continue
                    mt = maps[t, b]                                        # (L, n_img)
                    tot = np.maximum(mt.sum(-1), 1e-9)
                    on_k = (mt * (row_of_tok == k)[None]).sum(-1) / tot
                    on_target = (mt * (row_of_tok == r['truth_a']['row'])[None]).sum(-1) / tot
                    res.setdefault('row_track', []).append({'k': k, 'is_target_row': k == r['truth_a']['row'],
                                                            'on_k': on_k, 'on_target_row': on_target,
                                                            'mass': img[t, b].mean(-1)})
        if not res['curve']:
            continue
        summ = {'n_responses': len(res['curve'])}
        allm = np.concatenate(res['curve'], 0)
        summ['mean_image_mass_by_layer'] = {str(l): float(allm[:, i].mean()) for i, l in enumerate(ce.FULL_LAYERS)}
        summ['step0_image_mass_by_layer'] = {str(l): float(np.mean([c[0, i] for c in res['curve']]))
                                             for i, l in enumerate(ce.FULL_LAYERS)}
        parts = np.stack(res['pre_img_txt_gen'])
        summ['mass_split_by_layer'] = {str(l): dict(zip(('pre', 'img', 'txt', 'gen'), parts[:, i].mean(0).round(4).tolist()))
                                       for i, l in enumerate(ce.FULL_LAYERS)}
        summ['by_class'] = {c: {str(l): {'mean': float(np.mean(v)), 'n': len(v)} for l, v in d.items()}
                            for c, d in res['by_class'].items()}
        ts = np.concatenate(res['target_share'], 0)
        summ['target_share_by_layer'] = {str(l): float(ts[:, i].mean()) for i, l in enumerate(ce.FULL_LAYERS)}
        summ['target_share_uniform'] = float(np.mean(res['uniform_share']))
        summ['h11_target_share'] = float(np.concatenate(res['h11_target']).mean())
        if res.get('row_track'):
            rt_ = res['row_track']
            on_k = np.stack([x['on_k'] for x in rt_])
            on_t = np.stack([x['on_target_row'] for x in rt_])
            ms = np.stack([x['mass'] for x in rt_])
            tgt = np.array([x['is_target_row'] for x in rt_])
            rows_n = max(x['k'] for x in rt_)
            summ['row_tracking'] = {
                'n_steps': len(rt_), 'uniform': 1.0 / r['truth_a']['rows'],
                'share_on_enumerated_row': {str(l): float(on_k[:, i].mean()) for i, l in enumerate(ce.FULL_LAYERS)},
                'share_on_enumerated_row_other_lines': {str(l): float(on_k[~tgt, i].mean()) for i, l in enumerate(ce.FULL_LAYERS)},
                'n_steps_target_row_lines': int(tgt.sum()),
                'share_on_target_row_while_other_row': {str(l): float(on_t[~tgt, i].mean()) for i, l in enumerate(ce.FULL_LAYERS)},
                'image_mass_target_row_lines': {str(l): float(ms[tgt, i].mean()) if tgt.any() else None
                                                for i, l in enumerate(ce.FULL_LAYERS)},
                'image_mass_other_row_lines': {str(l): float(ms[~tgt, i].mean()) for i, l in enumerate(ce.FULL_LAYERS)},
            }
        out[task] = summ
        out[f'_{task}_raw'] = res
    return out


def _token_rows(gh: int, gw: int, rows: int) -> np.ndarray:
    """1-based grid row of every image token (grid fills the canvas in all mechanism tasks)."""
    i = np.repeat(np.arange(gh), gw)
    return (np.floor((i + 0.5) / gh * rows) + 1).astype(int)


def _current_row_label(tokenizer, ids: list[int]) -> list[int | None]:
    """For each generated token t, the k of a line starting with "Row k" that token t belongs to
    (token t is produced by decode step t), else None."""
    import re
    out, text = [], ''
    rx = re.compile(r'(?:^|\n)\s*(?:[-*•]\s*)?(?:\*\*)?\s*Row\s+(\d+)\b[^\n]*$', re.I)
    for tid in ids:
        m = rx.search(text)
        out.append(int(m.group(1)) if m else None)
        text += tokenizer.decode([tid])
    return out


def _grid_tokens(r: dict) -> tuple[int, int]:
    n = r['sp']['img_end'] - r['sp']['img_start']
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError('non-square image token grid; store grid_tokens in the record')
    return side, side


def _cell_mask(truth: dict, gh: int, gw: int) -> np.ndarray:
    x0, y0, x1, y1 = truth['cell_box_norm']
    mask = np.zeros((gh, gw), np.float32)
    for i in range(gh):
        for j in range(gw):
            cx, cy = (j + 0.5) / gw * 1000, (i + 0.5) / gh * 1000
            mask[i, j] = float(x0 <= cx <= x1 and y0 <= cy <= y1)
    return mask.reshape(-1)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--figures', type=Path, required=True)
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    summary = {}
    coded = read(args.run_dir / 'taxonomy/coded.jsonl')
    if coded:
        tax = taxonomy(coded)
        summary['taxonomy'] = tax
        fig_taxonomy(tax, args.figures / 'taxonomy.png')
    if list((args.run_dir / 'faith').glob('records*.jsonl')) if (args.run_dir / 'faith').exists() else False:
        fa = faithfulness(args.run_dir)
        summary['faith'] = fa
        fig_faith(fa, args.figures / 'early_answer.png')
        fig_edit(fa, args.figures / 'edit.png')
    if (args.run_dir / 'mech/records.jsonl').exists():
        summary['mech'] = mechanism(args.run_dir)
        from analysis_tools.behavior import cot_mechanism_figures as cf
        cf.make_all(args.run_dir, args.figures, summary)
    (args.run_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print('wrote', args.run_dir / 'summary.json')


if __name__ == '__main__':
    main()
