"""Reasoning-text coding of BabyVision Fine-grained Discrimination responses for experiment 4.2.

Coding table: docs/analysis/behavior/design_decomposition.md, section 6.3.  Commands:

  code       -- rule-based codes for every Fine-grained Discrimination response of every model
  export     -- blinded, stratified sample for manual coding (model identity hidden behind item ids)
  summarize  -- merge manual codes (manual_codes.json) with the rule codes and write tables / figures;
                inter-coder agreement is computed when a second coding (manual_codes_second.json) exists

Rule codes (declared grid, declared row/column, answer consistency) follow
``decomposition_reasoning.extract_codes``; they are computed for Find the different, whose
answers are grid positions with an annotated ground-truth grid.  For all subtypes the number of
explicit self-revisions ("wait", "actually", ...) and the response length are recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

import numpy as np

from analysis_tools.behavior.decomposition import read_jsonl, write_jsonl
from analysis_tools.behavior.decomposition_analysis import wilson, _style, INK, MUTED
from analysis_tools.behavior.decomposition_reasoning import extract_codes
from analysis_tools.deepstack.deepstack_comparison import MODELS
from analysis_tools.deepstack.deepstack_comparison_analysis import COLORS, SHORT, babyvision_files

TYPE = 'Fine-grained Discrimination'
FTD = 'Find the different'
REVISION = re.compile(r"\b(wait|actually|hmm|let me (?:re-?|double[- ]?)check|on (?:closer|second) (?:look|inspection)|"
                      r"looking (?:again|more carefully)|correction|i made a mistake)\b", re.I)
RC_ANSWER = re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)')
# Manual code values (design_decomposition.md 6.3); failure stage is coded for wrong answers only.
MANUAL_VALUES = {
    'attribute': ('accurate', 'hallucinated', 'vague', 'mixed'),
    'search': ('row_scan', 'pairwise', 'direct_claim', 'none'),
    'failure': ('perception', 'comparison', 'counting', 'inconsistency', 'other'),
}


def load_records(output: Path, legacy: Path) -> dict[str, list[dict]]:
    files = babyvision_files(output, legacy)
    return {m: [r for r in read_jsonl(p) if r['type'] == TYPE] for m, p in files.items()}


def code(output: Path, legacy: Path, annotation: Path) -> list[dict]:
    grid = {it['taskId']: it for it in json.loads(annotation.read_text())['items']}
    rows = []
    for model, records in load_records(output, legacy).items():
        for r in records:
            text = r['response']
            row = {'model': model, 'taskId': r['taskId'], 'subtype': r['subtype'], 'correct': bool(r['score']['correct']),
                   'parsed': r['score']['prediction'] is not None, 'finish_reason': r['finish_reason'],
                   'n_tokens': r['generated_tokens'], 'n_revisions': len(REVISION.findall(text))}
            if r['subtype'] == FTD:
                ann = grid[r['taskId']]
                m = RC_ANSWER.search(r['score']['prediction'] or '')
                final_rc = (int(m.group(1)), int(m.group(2))) if m else None
                reasoning = text.rsplit('Final answer', 1)[0]
                codes = extract_codes(reasoning, final_rc)
                truth_rc = [ann['answer_rc'][0] + 1, ann['answer_rc'][1] + 1]
                codes.update(truth_grid=[ann['rows'], ann['cols']], truth_rc=truth_rc,
                             grid_correct=None if codes['declared_grid'] is None
                             else codes['declared_grid'] == [ann['rows'], ann['cols']],
                             declared_rc_correct=None if codes['declared_rc'] is None
                             else codes['declared_rc'] == truth_rc)
                row.update(codes)
            rows.append(row)
    write_jsonl(output / 'reasoning' / 'rule_codes.jsonl', rows)
    return rows


def export(output: Path, legacy: Path, data_root: Path, seed: int, wrong_per_subtype: int, right_per_subtype: int) -> None:
    """All Find the different responses plus, per model and other subtype, a few wrong and right ones."""
    rng = random.Random(seed)
    picked = []
    for model, records in load_records(output, legacy).items():
        by = defaultdict(list)
        for r in records:
            by[(r['subtype'], bool(r['score']['correct']))].append(r)
        for (subtype, correct), rs in sorted(by.items()):
            rs = sorted(rs, key=lambda r: r['taskId'])
            k = len(rs) if subtype == FTD else (right_per_subtype if correct else wrong_per_subtype)
            picked += [(model, r) for r in rng.sample(rs, min(k, len(rs)))]
    rng.shuffle(picked)
    items, key = [], {}
    for i, (model, r) in enumerate(picked):
        item_id = f'item{i:04d}'
        key[item_id] = {'model': model, 'taskId': r['taskId']}
        items.append({'item_id': item_id, 'taskId': r['taskId'], 'subtype': r['subtype'],
                      'image': str(Path(r['image']).relative_to(data_root)), 'prompt': r['prompt'],
                      'reference': r['score']['reference'], 'prediction': r['score']['prediction'],
                      'correct': bool(r['score']['correct']), 'finish_reason': r['finish_reason'],
                      'response': r['response']})
    d = output / 'reasoning'
    d.mkdir(parents=True, exist_ok=True)
    write_jsonl(d / 'manual_sample.jsonl', items)
    (d / 'manual_key.json').write_text(json.dumps(key, indent=1) + '\n')
    print(f'exported {len(items)} responses; ' + json.dumps(Counter(v['model'] for v in key.values())))


def _share(values: list, levels: tuple) -> dict:
    n = len(values)
    return {'n': n, **{lv: {'k': values.count(lv), 'p': values.count(lv) / n if n else float('nan'),
                            'ci': list(wilson(values.count(lv), n))} for lv in levels}}


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa of two coders' labels on the same items."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return float('nan') if pe == 1 else (po - pe) / (1 - pe)


def agreement(first: dict, second: dict, sample: dict) -> dict:
    """Per-field agreement rate and Cohen's kappa on the doubly coded items (failure: wrong answers only)."""
    out = {}
    for field in MANUAL_VALUES:
        ids = [i for i in sorted(second) if field != 'failure' or not sample[i]['correct']]
        a, b = [first[i][field] for i in ids], [second[i][field] for i in ids]
        out[field] = {'n': len(ids), 'agreement': sum(x == y for x, y in zip(a, b)) / len(ids),
                      'kappa': cohen_kappa(a, b)}
    return out


def summarize(output: Path) -> dict:
    d = output / 'reasoning'
    rules = read_jsonl(d / 'rule_codes.jsonl')
    out = {'rules': {}, 'manual': {}}
    for model in MODELS:
        rs = [r for r in rules if r['model'] == model]
        if not rs:
            continue
        ftd = [r for r in rs if r['subtype'] == FTD]
        out['rules'][model] = {
            'n': len(rs), 'median_tokens': float(np.median([r['n_tokens'] for r in rs])),
            'mean_revisions': float(np.mean([r['n_revisions'] for r in rs])),
            'revisions_wrong': float(np.mean([r['n_revisions'] for r in rs if not r['correct']] or [np.nan])),
            'revisions_right': float(np.mean([r['n_revisions'] for r in rs if r['correct']] or [np.nan])),
            'ftd_n': len(ftd), 'ftd_correct': sum(r['correct'] for r in ftd),
            'ftd_grid_declared': sum(r['declared_grid'] is not None for r in ftd),
            'ftd_grid_correct': sum(bool(r['grid_correct']) for r in ftd),
            'ftd_rc_declared': sum(r['declared_rc'] is not None for r in ftd),
            'ftd_rc_correct': sum(bool(r['declared_rc_correct']) for r in ftd),
            'ftd_consistent': sum(r['consistent'] is True for r in ftd),
            'ftd_inconsistent': sum(r['consistent'] is False for r in ftd)}
    manual_path = d / 'manual_codes.json'
    if manual_path.exists():
        manual = json.loads(manual_path.read_text())
        key = json.loads((d / 'manual_key.json').read_text())
        sample = {r['item_id']: r for r in read_jsonl(d / 'manual_sample.jsonl')}
        missing = sorted(set(sample) - set(manual))
        if missing:
            raise ValueError(f'{len(missing)} sampled items lack manual codes, e.g. {missing[:5]}')
        for item_id, c in manual.items():
            for field, levels in MANUAL_VALUES.items():
                value = c.get(field)
                if field == 'failure' and sample[item_id]['correct']:
                    continue
                if value not in levels:
                    raise ValueError(f'{item_id}: invalid {field}={value!r}')
        for model in MODELS:
            ids = [i for i, k in key.items() if k['model'] == model]
            if not ids:
                continue
            res = {}
            for scope, sel in (('ftd', [i for i in ids if sample[i]['subtype'] == FTD]),
                               ('other', [i for i in ids if sample[i]['subtype'] != FTD])):
                res[scope] = {
                    'attribute': _share([manual[i]['attribute'] for i in sel], MANUAL_VALUES['attribute']),
                    'search': _share([manual[i]['search'] for i in sel], MANUAL_VALUES['search']),
                    'failure': _share([manual[i]['failure'] for i in sel if not sample[i]['correct']],
                                      MANUAL_VALUES['failure'])}
            out['manual'][model] = res
        second_path = d / 'manual_codes_second.json'
        if second_path.exists():
            out['agreement'] = agreement(manual, json.loads(second_path.read_text()), sample)
            print('agreement', json.dumps(out['agreement']))
    (d / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')
    if out['manual']:
        plot_failure(out, d / 'failure_stages.png')
    print(json.dumps(out['rules'], indent=1)[:2500])
    return out


def plot_failure(out: dict, path: Path) -> None:
    """Stacked shares of the first failing stage for wrong answers, per model and scope."""
    plt = _style()
    stage_colors = {'perception': '#2a78d6', 'comparison': '#1baf7a', 'counting': '#eda100',
                    'inconsistency': '#e87ba4', 'other': '#a3a29e'}
    models = [m for m in MODELS if m in out['manual']]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.6), sharey=True)
    for ax, (scope, title) in zip(axes, (('ftd', 'Find the different (all responses)'),
                                         ('other', 'Other Fine-grained subtypes (sample)'))):
        for y, m in enumerate(models[::-1]):
            share = out['manual'][m][scope]['failure']
            left = 0.0
            for stage, color in stage_colors.items():
                p = share[stage]['p'] if share['n'] else 0
                ax.barh(y, p, left=left, color=color, edgecolor='white', linewidth=2, height=0.6,
                        label=stage if y == 0 else None)
                left += p
            ax.text(1.01, y, f"n={share['n']}", va='center', fontsize=8, color=MUTED)
        ax.set_yticks(range(len(models)), models[::-1])
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.set_xlabel('share of wrong answers')
        ax.grid(False)
    axes[1].legend(loc='center left', bbox_to_anchor=(1.12, 0.5), fontsize=8, title='first failing stage',
                   title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', nargs='?', default='all', choices=('code', 'export', 'summarize', 'all'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--legacy-babyvision', type=Path, required=True)
    p.add_argument('--annotation', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wrong-per-subtype', type=int, default=2)
    p.add_argument('--right-per-subtype', type=int, default=1)
    args = p.parse_args()
    (args.output / 'reasoning').mkdir(parents=True, exist_ok=True)
    if args.command in ('code', 'all'):
        code(args.output, args.legacy_babyvision, args.annotation)
    if args.command in ('export', 'all') and not (args.output / 'reasoning' / 'manual_sample.jsonl').exists():
        export(args.output, args.legacy_babyvision, args.data_root, args.seed, args.wrong_per_subtype,
               args.right_per_subtype)
    if args.command in ('summarize', 'all'):
        summarize(args.output)


if __name__ == '__main__':
    main()
