"""Rescore existing BabyVision runs with the format-tolerant diagnostic scorer (experiment 4.2).

Strict scores stay untouched. For every run file ``<run>.jsonl`` the tool writes
``<run>.jsonl.diagnostic.json`` next to it (the suffix keeps the run globs ``*.jsonl`` of the
4.2 analysis unambiguous) and a combined report under ``--output``:

  summary.json   strict and diagnostic counts per model, overall and per subtype
  flips.jsonl    every item whose diagnostic score differs from its strict score
  tables.md      the per-model / per-subtype table used in the results document

Before rescoring, the stored strict score of every record is recomputed with
``eva.benchmark.score`` and must match, so the strict numbers in the report are exactly those
of the original runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.deepstack.deepstack_comparison import MODELS
from analysis_tools.deepstack.deepstack_comparison_analysis import babyvision_files, mcnemar, rate
from eva.benchmark import load_cases, score
from eva.diagnostic_scoring import METRIC, diagnostic_score

SHORT = {'Qwen3-VL-8B-Instruct': 'VL-8B', 'Qwen3.5-9B': '3.5-9B', 'Qwen3-VL-4B-Instruct': 'VL-4B',
         'Qwen3.5-4B': '3.5-4B'}
NOTE = ('Diagnostic only: strict local scoring (eva.benchmark.score) remains the primary metric. '
        'Diagnostic correct = strict correct plus answers whose content matches the reference '
        'under the conservative format rules of eva.diagnostic_scoring.')


def rescore_run(path: Path, data_root: Path) -> list[dict]:
    records = read_jsonl(path)
    ids = [r['taskId'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate taskIds in {path}')
    cases, _ = load_cases(data_root, 'babyvision', 'train', ids)
    rows = []
    for record, case in zip(records, cases):
        strict = score(case, record['response'], 'babyvision')
        if strict != record['score']:
            raise ValueError(f'Stored strict score differs from eva.benchmark.score: {path} taskId {case["taskId"]}')
        diag = diagnostic_score(case, record['response'], record.get('finish_reason'))
        rows.append({'taskId': case['taskId'], 'type': case['type'], 'subtype': case['subtype'],
                     'ansType': case['ansType'], 'finish_reason': record.get('finish_reason'),
                     'strict': strict, 'diagnostic': diag})
    return rows


def group(rows: list[dict]) -> dict:
    strict = [r['strict']['correct'] for r in rows]
    diag = [r['diagnostic']['correct'] for r in rows]
    if any(s and not d for s, d in zip(strict, diag)):
        raise AssertionError('Diagnostic scoring must contain the strict correct set')
    return {'strict': rate(strict), 'diagnostic': rate(diag), 'flips': sum(d and not s for s, d in zip(strict, diag)),
            'mcnemar': mcnemar(diag, strict)}


def summarize(rows: list[dict]) -> dict:
    result = {'all': group(rows), 'by_subtype': {}}
    for subtype in sorted({r['subtype'] for r in rows}):
        result['by_subtype'][subtype] = group([r for r in rows if r['subtype'] == subtype])
    return result


def tables(summary: dict, flips: list[dict]) -> str:
    models = [m for m in MODELS if m in summary]
    lines = ['| 范围 | $n$ | ' + ' | '.join(f'{SHORT[m]} 严格／诊断' for m in models) + ' |',
             '| --- | --- | ' + ' | '.join('---' for _ in models) + ' |']
    first = summary[models[0]]
    scopes = [('全集', 'all', None)] + [(s, 'by_subtype', s) for s in first['by_subtype']]
    for label, key, sub in scopes:
        cells, n = [], None
        for m in models:
            g = summary[m][key] if sub is None else summary[m][key][sub]
            n = g['strict']['n']
            mark = f"{g['strict']['k']}／{g['diagnostic']['k']}"
            cells.append(f'**{mark}**' if g['flips'] else mark)
        lines.append(f'| {label} | {n} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| 模型 | taskId | 子类型 | 参考答案 | 模型答案 | 规则 | 抽取 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for f in flips:
        lines.append(f"| {SHORT[f['model']]} | {f['taskId']} | {f['subtype']} | {f['reference']} | "
                     f"{f['prediction']} | {f['rule']} | {f['extraction']} |")
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-dir', type=Path, required=True, help='4.2 output directory (contains babyvision/)')
    parser.add_argument('--legacy-babyvision', type=Path, required=True, help='directory of the Qwen3.5-9B run')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = babyvision_files(args.runs_dir, args.legacy_babyvision)
    missing = [m for m in MODELS if m not in files]
    if missing:
        raise FileNotFoundError(f'Missing BabyVision runs for {missing}')
    args.output.mkdir(parents=True, exist_ok=True)
    summary, flips = {}, []
    for model, path in files.items():
        rows = rescore_run(path, args.data_root)
        summary[model] = summarize(rows)
        summary[model]['run'] = path.name
        side = path.with_name(path.name + '.diagnostic.json')
        side.write_text(json.dumps({'metric': METRIC, 'metric_note': NOTE, 'run': path.name,
                                    'summary': summary[model], 'records': rows}, ensure_ascii=False, indent=1))
        for r in rows:
            if r['diagnostic']['correct'] != r['strict']['correct']:
                flips.append({'model': model, 'taskId': r['taskId'], 'subtype': r['subtype'],
                              'reference': r['diagnostic']['reference'], 'strict_prediction': r['strict']['prediction'],
                              'prediction': r['diagnostic']['prediction'], 'rule': r['diagnostic']['rule'],
                              'extraction': r['diagnostic']['extraction']})
        g = summary[model]['all']
        print(f"{model}: strict {g['strict']['k']}/{g['strict']['n']}, diagnostic {g['diagnostic']['k']}, "
              f"flips {g['flips']} -> {side.name}")
    (args.output / 'summary.json').write_text(json.dumps({'metric_note': NOTE, 'models': summary},
                                                         ensure_ascii=False, indent=1))
    with (args.output / 'flips.jsonl').open('w') as fh:
        for f in flips:
            fh.write(json.dumps(f, ensure_ascii=False) + '\n')
    (args.output / 'tables.md').write_text(tables(summary, flips))


if __name__ == '__main__':
    main()
