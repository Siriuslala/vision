"""Supplementary analyses for experiment 1.3 (design_decomposition.md, sections 3.1, 6.4 and 7).

1. T3 answer bias: Yes rate, mean discriminant score s on target / normal designations and
   AUROC of s, pooled per grid size, so that near-chance T3 accuracy can be attributed either
   to a response bias (AUROC high, Yes rate extreme) or to missing discrimination (AUROC ~0.5).
2. Reasoning conditions vs the main evaluation on the same cases (explicit reasoning with
   thinking off, thinking on): accuracy per task x grid and an exact McNemar test on the
   paired outcomes.  Binary reasoning answers are scored from the parsed final answer, since
   the first generated token of a reasoning response is not the answer.
3. BabyVision correspondence questions: per-task accuracy and a per-item table.

Reads the score files written by ``decomposition.py score``; writes analysis/supplement.json,
analysis/supplement.md and two figures.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path

import numpy as np

from analysis_tools.behavior.decomposition import GRIDS, KINDS, read_jsonl
from analysis_tools.behavior.decomposition_analysis import (
    INK, MUTED, SERIES, _style, auroc, fmt_ci, load, wilson,
)


def rate(flags: list[bool]) -> dict:
    n, k = len(flags), int(sum(flags))
    return {'n': n, 'k': k, 'acc': k / n if n else float('nan'), 'ci': wilson(k, n)}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def outcome(row: dict) -> bool:
    """Correctness used for comparison; reasoning stages use the parsed final answer for yes/no."""
    if row['format'] == 'yes_no' and row['stage'] in ('explicit', 'thinking'):
        return bool(row['score']['parsed_correct'])
    return bool(row['score']['correct'])


# ----------------------------------------------------------------------------- T3 bias

def t3_bias(strict: list[dict]) -> dict:
    out = {}
    for grid in GRIDS:
        for mode in ('box', 'rowcol'):
            for kind in KINDS + ('all',):
                rs = [r for r in strict if r['task'] == 'T3' and r['grid'] == grid and r['wording'] == 'primary'
                      and r['variant'].startswith(mode) and (kind == 'all' or r['kind'] == kind)]
                if not rs:
                    continue
                tgt = [r['score']['s'] for r in rs if not r['truth']['answer_yes']]
                nrm = [r['score']['s'] for r in rs if r['truth']['answer_yes']]
                out[f'{grid}|{mode}|{kind}'] = {
                    'n': len(rs), 'acc': float(np.mean([r['score']['correct'] for r in rs])),
                    'yes_rate': float(np.mean([r['score']['s'] > 0 for r in rs])),
                    'yes_rate_target': float(np.mean([s > 0 for s in tgt])),
                    'yes_rate_normal': float(np.mean([s > 0 for s in nrm])),
                    'mean_s_target': float(np.mean(tgt)), 'mean_s_normal': float(np.mean(nrm)),
                    'auroc': auroc(nrm, tgt)}
    return out


def plot_t3_scores(strict: list[dict], out: Path) -> None:
    """Distribution of s for designated target (answer No) and normal (answer Yes) instances."""
    plt = _style()
    fig, axes = plt.subplots(2, len(GRIDS), figsize=(2.6 * len(GRIDS), 4.4), sharey='row')
    for i, mode in enumerate(('box', 'rowcol')):
        for j, grid in enumerate(GRIDS):
            ax = axes[i, j]
            rs = [r for r in strict if r['task'] == 'T3' and r['grid'] == grid and r['wording'] == 'primary'
                  and r['variant'].startswith(mode)]
            tgt = [r['score']['s'] for r in rs if not r['truth']['answer_yes']]
            nrm = [r['score']['s'] for r in rs if r['truth']['answer_yes']]
            lo, hi = min(tgt + nrm), max(tgt + nrm)
            bins = np.linspace(lo, hi, 25)
            ax.hist(nrm, bins=bins, color=SERIES[0], alpha=0.6, label='normal designated (Yes)')
            ax.hist(tgt, bins=bins, color=SERIES[1], alpha=0.6, label='target designated (No)')
            ax.axvline(0, color=INK, lw=0.8)
            ax.set_title(f"{'blue box' if mode == 'box' else 'row/col'}, {grid}x{grid}  AUROC {auroc(nrm, tgt):.2f}")
            if j == 0:
                ax.set_ylabel('count')
            if i == 1:
                ax.set_xlabel('s = z_Yes - z_No')
    axes[0, 0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- reasoning vs main

def reasoning_comparison(output: Path, main_diag: list[dict]) -> dict:
    """Location answers use the coordinate diagnostic in both conditions, so that the comparison
    measures the answer content; a list-wrapped JSON answer is a format error, not a location error."""
    main = {r['id'].split('|', 1)[1]: r for r in main_diag}
    out = {}
    for stage in ('explicit', 'thinking'):
        if not (output / f'scores_{stage}_strict.jsonl').exists():
            continue
        rows = load(output, stage=stage, kind='diagnostic')
        groups = defaultdict(list)
        for r in rows:
            m = main[r['id'].split('|', 1)[1]]
            if r['truth'].get('designated_rc') != m['truth'].get('designated_rc'):
                raise ValueError(f'Reasoning case {r["id"]} designates a different instance from the main case.')
            pair = (outcome(m), outcome(r), r['result']['finish_reason'] != 'eos', r['result']['n_tokens'])
            groups[(r['task'], r['grid'])].append(pair)
            groups[(r['task'], 'all')].append(pair)
        for (task, grid), pairs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            b = sum(m and not s for m, s, _, _ in pairs)
            c = sum(s and not m for m, s, _, _ in pairs)
            out[f'{stage}|{task}|{grid}'] = {
                'main': rate([m for m, _, _, _ in pairs]), 'reasoning': rate([s for _, s, _, _ in pairs]),
                'main_only': b, 'reasoning_only': c, 'p_mcnemar': mcnemar_exact(b, c),
                'truncated': sum(t for _, _, t, _ in pairs), 'mean_tokens': float(np.mean([n for *_, n in pairs]))}
    return out


# ----------------------------------------------------------------------------- BabyVision

def babyvision(output: Path) -> dict:
    path = output / 'scores_babyvision_strict.jsonl'
    if not path.exists():
        return {}
    cases = {c['id']: c for c in read_jsonl(output / 'babyvision_cases.jsonl')}
    rows = [{**cases[s['id']], 'score': s} for s in read_jsonl(path)]
    out = {'task': {}, 'items': {}}
    groups = defaultdict(list)
    for r in rows:
        v = r['variant']
        if r['task'] == 'T3':
            v = v.split('_')[0] + ('_target' if 'target' in v else '_normal')
        groups[(r['task'], v)].append(r)
    for (task, v), rs in sorted(groups.items()):
        out['task'][f'{task}|{v}'] = rate([bool(r['score']['correct']) for r in rs])
    for mode in ('box', 'rowcol'):
        rs = [r for r in rows if r['task'] == 'T3' and r['variant'].startswith(mode)]
        tgt = [r['score']['s'] for r in rs if not r['truth']['answer_yes']]
        nrm = [r['score']['s'] for r in rs if r['truth']['answer_yes']]
        out['task'][f'T3|{mode}_auroc'] = auroc(nrm, tgt)
        out['task'][f'T3|{mode}_yes_rate'] = float(np.mean([r['score']['s'] > 0 for r in rs]))
    diag = [{**cases[s['id']], 'score': s} for s in read_jsonl(output / 'scores_babyvision_diagnostic.jsonl')]
    for task in ('T5', 'T6'):
        out['task'][f'{task}|target_diagnostic'] = rate([bool(r['score']['correct']) for r in diag if r['task'] == task])
    t2 = [r for r in rows if r['task'] == 'T2']
    out['task']['T2|auroc'] = auroc([r['score']['s'] for r in t2 if r['truth']['answer_yes']],
                                    [r['score']['s'] for r in t2 if not r['truth']['answer_yes']])
    per = defaultdict(dict)
    for r in rows:
        item = r['id'].split('|')[2]
        per[item]['grid'] = f"{r['truth']['rows']}x{r['truth']['cols']}"
        if r['task'] in ('T5', 'T6'):
            per[item][r['task']] = bool(r['score']['correct'])
            per[item][r['task'] + '_pred'] = r['score'].get('pred_rc')
        elif r['variant'] in ('neg', 'pos', 'box_target', 'rowcol_target'):
            per[item][f"{r['task']} {r['variant']}"] = bool(r['score']['correct'])
    out['items'] = dict(per)
    return out


# ----------------------------------------------------------------------------- report

def write_report(sup: dict, path: Path) -> None:
    L = ['# Experiment 1.3 supplementary tables (generated)', '',
         '## T3 response bias (primary wording, pooled over levels unless kind given)', '',
         '| grid | mode | kind | n | acc | Yes rate | Yes rate (target) | Yes rate (normal) | mean s target | mean s normal | AUROC |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for key, v in sup['t3_bias'].items():
        g, m, k = key.split('|')
        L.append(f"| {g} | {m} | {k} | {v['n']} | {v['acc']:.2f} | {v['yes_rate']:.2f} | {v['yes_rate_target']:.2f} | "
                 f"{v['yes_rate_normal']:.2f} | {v['mean_s_target']:.2f} | {v['mean_s_normal']:.2f} | {v['auroc']:.2f} |")
    L += ['', '## Reasoning conditions vs main evaluation (same cases)', '',
          '| stage | task | grid | n | main | reasoning | main only | reasoning only | McNemar p | truncated | mean tokens |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for key, v in sup['reasoning'].items():
        s, t, g = key.split('|')
        L.append(f"| {s} | {t} | {g} | {v['main']['n']} | {fmt_ci(v['main'])} | {fmt_ci(v['reasoning'])} | "
                 f"{v['main_only']} | {v['reasoning_only']} | {v['p_mcnemar']:.3g} | {v['truncated']} | {v['mean_tokens']:.0f} |")
    bv = sup.get('babyvision') or {}
    if bv:
        L += ['', '## BabyVision correspondence', '', '| task | variant | value |', '| --- | --- | --- |']
        for key, v in bv['task'].items():
            t, var = key.split('|')
            L.append(f"| {t} | {var} | {fmt_ci(v) + ' (n=' + str(v['n']) + ')' if isinstance(v, dict) else f'{v:.2f}'} |")
        cols = ['T2 neg', 'T2 pos', 'T3 box_target', 'T3 rowcol_target', 'T5', 'T6']
        L += ['', '| item | grid | ' + ' | '.join(cols) + ' | T5 pred | T6 pred |', '| --- ' * (len(cols) + 4) + '|']
        for item, v in sorted(bv['items'].items(), key=lambda kv: int(kv[0])):
            L.append(f"| {item} | {v['grid']} | " + ' | '.join('1' if v.get(c) else '0' for c in cols) +
                     f" | {v.get('T5_pred')} | {v.get('T6_pred')} |")
    path.write_text('\n'.join(L) + '\n')


def plot_reasoning(sup: dict, out: Path) -> None:
    plt = _style()
    tasks = ['T3', 'T4', 'T5', 'T6']
    grids = sorted({k.split('|')[2] for k in sup['reasoning'] if k.split('|')[2] != 'all'}, key=int)
    fig, axes = plt.subplots(1, len(grids), figsize=(3.4 * len(grids), 2.8), sharey=True)
    width = 0.27
    for ax, grid in zip(np.atleast_1d(axes), grids):
        x = np.arange(len(tasks))
        for i, (stage, color) in enumerate((('main', MUTED), ('explicit', SERIES[0]), ('thinking', SERIES[1]))):
            vals, errs = [], [[], []]
            for t in tasks:
                src = sup['reasoning'].get(f"{'explicit' if stage == 'main' else stage}|{t}|{grid}")
                if src is None:
                    vals.append(np.nan); errs[0].append(0); errs[1].append(0)
                    continue
                st = src['main'] if stage == 'main' else src['reasoning']
                vals.append(st['acc']); errs[0].append(st['acc'] - st['ci'][0]); errs[1].append(st['ci'][1] - st['acc'])
            ax.bar(x + (i - 1) * width, vals, width, color=color, yerr=errs, capsize=2, ecolor=INK,
                   label={'main': 'direct answer (explicit subset)', 'explicit': 'explicit reasoning',
                          'thinking': 'thinking on'}[stage])
        ax.set_xticks(x, tasks)
        ax.set_ylim(0, 1)
        ax.set_title(f'{grid}x{grid} grid')
    np.atleast_1d(axes)[0].set_ylabel('accuracy (row/col exact; T3 yes/no)')
    np.atleast_1d(axes)[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    strict = load(args.output, kind='strict')
    sup = {'t3_bias': t3_bias(strict), 'reasoning': reasoning_comparison(args.output, load(args.output, kind='diagnostic')),
           'babyvision': babyvision(args.output)}
    target = args.output / 'analysis'
    (target / 'figures').mkdir(parents=True, exist_ok=True)
    (target / 'supplement.json').write_text(json.dumps(sup, indent=1) + '\n')
    write_report(sup, target / 'supplement.md')
    plot_t3_scores(strict, target / 'figures' / 't3_score_distribution.png')
    if sup['reasoning']:
        plot_reasoning(sup, target / 'figures' / 'reasoning_accuracy.png')
    print(f"wrote {target / 'supplement.md'}")


if __name__ == '__main__':
    main()
