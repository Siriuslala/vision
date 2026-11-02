"""Experiment 1.7, part 1: LLM-free rule coding of existing and fresh CoT corpora.

Every corpus is converted into unified records
    {source, task, mode, condition, id, text, finish_reason, truth, extra}
and coded with cot_mechanism.code_cot. Corpora (all Qwen3.5-9B, greedy unless noted):

* dec13_explicit  experiment 1.3 explicit reasoning (T3 binary, T4/T5/T6 row-column), thinking off
* dec13_thinking  experiment 1.3 thinking on (same tasks, small subset)
* ord15_think     experiment 1.5 think prompt on H32/H64/V64 ordinal layouts
* pe16_explicit   experiment 1.6 explicit reasoning, PE x1 condition only
* rc310_think     experiment 3.10 think prompt on 8x8 and V64 grids; clean prefill ('none') and
                  prefill-edited runs ('nodecay|mid', 'nowrite|mid'); plus the thinking-on subset
* odd26_think     experiment 2.6 explicit reasoning on find-the-odd grids
* bv16            experiment 1.6 BabyVision, PE x1, thinking off (sampling decode of the reference run)
* fresh17         this experiment's fresh think generations (rowcol, odd, point, count)

Paths are relative to WORK_DIR/analysis.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from analysis_tools.behavior import cot_mechanism as cm


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def _split_thinking(text: str) -> str:
    """For thinking-on responses keep the whole text (thinking + answer) so the body includes the reasoning."""
    return text


def load_dec13(root: Path) -> list[dict]:
    base = root / 'behavior/2026-10-13_decomposition'
    cases = {c['id']: c for c in _jsonl(base / 'cases.jsonl')}
    out = []
    for fname, mode in (('results_explicit.jsonl', 'think_off'), ('results_thinking.jsonl', 'thinking_on')):
        for r in _jsonl(base / fname):
            c = cases[r['id']]
            t = _lit(c['truth'])
            if c['task'] == 'T3':
                task, truth = 'binary', {'answer_yes': t['answer_yes'], 'rows': c['grid'], 'cols': c['grid']}
            else:
                if c['format'] != 'row_column':
                    continue
                task = 'rowcol' if c['task'] == 'T6' else 'odd'
                truth = {'row': t['target_rc'][0] + 1, 'col': t['target_rc'][1] + 1, 'rows': t['rows'], 'cols': t['cols']}
            out.append({'source': 'dec13_explicit' if mode == 'think_off' else 'dec13_thinking', 'task': task,
                        'mode': mode, 'condition': f"{c['task']}_g{c['grid']}", 'id': r['id'],
                        'text': r['response'], 'finish_reason': r['finish_reason'], 'truth': truth,
                        'extra': {'kind': c.get('kind'), 'level': c.get('level'), 'grid': c['grid'], 'subtask': c['task']}})
    return out


def load_ord15(root: Path) -> list[dict]:
    base = root / 'behavior/2026-10-16_ordinal_mechanism'
    cases = {c['id']: c for c in _jsonl(base / 'cases.jsonl')}
    out = []
    for r in _jsonl(base / 'gen/results_think.jsonl'):
        c = cases[r['id']]
        t = _lit(c['truth'])
        row, col = (t['other'], t['count']) if t['axis'] == 'H' else (t['count'], t['other'])
        out.append({'source': 'ord15_think', 'task': 'rowcol', 'mode': 'think_off',
                    'condition': f"{c['condition']}_{c['variant']}", 'id': r['id'], 'text': r['response'],
                    'finish_reason': r['finish_reason'], 'truth': {'row': row, 'col': col},
                    'extra': {'axis': t['axis'], 'variant': c['variant'], 'cell': t['cell']}})
    return out


def load_pe16(root: Path) -> list[dict]:
    base = root / 'behavior/2026-10-16_pe_scaling_t0/grid'
    cases = {c['id']: c for c in _jsonl(base / 'cases.jsonl')}
    out = []
    for r in _jsonl(base / 'results_grid_explicit.jsonl'):
        if r['condition'] != 'x1':
            continue
        c = cases[r['case_id']]
        t = _lit(c['truth'])
        task = 'rowcol' if c['task'] == 'T6' else 'odd'
        out.append({'source': 'pe16_explicit', 'task': task, 'mode': 'think_off',
                    'condition': f"{c['task']}_g{c['grid']}", 'id': r['id'], 'text': r['response'],
                    'finish_reason': r['finish_reason'],
                    'truth': {'row': t['target_rc'][0] + 1, 'col': t['target_rc'][1] + 1, 'rows': t['rows'], 'cols': t['cols']},
                    'extra': {'kind': c.get('kind'), 'level': c.get('level'), 'grid': c['grid'], 'subtask': c['task']}})
    return out


def load_rc310(root: Path) -> list[dict]:
    base = root / 'position/2026-10-20_rowcol_mechanism/implicit/records.jsonl'
    out = []
    for r in _jsonl(base):
        if r['mode'] == 'direct':
            continue
        grid = (8, 8) if r['set'] == 'g8x8' else (None, None)
        for run, v in r['runs'].items():
            mode = 'thinking_on' if r['mode'] == 'thinking_on' else 'think_off'
            out.append({'source': 'rc310_think', 'task': 'rowcol', 'mode': mode,
                        'condition': f"{r['set']}|{run}", 'id': f"{r['id']}|{run}", 'text': v['text'],
                        'finish_reason': v['finish_reason'],
                        'truth': {'row': r['truth_row'], 'col': r['truth_col'], 'rows': grid[0], 'cols': grid[1]},
                        'extra': {'set': r['set'], 'prefill_edit': run}})
    return out


def load_odd26(root: Path) -> list[dict]:
    base = root / 'representation/2026-10-20_odd_entity'
    items = {i['id']: i for i in _jsonl(base / 'items.jsonl')}
    out = []
    for r in _jsonl(base / 'results_behave.jsonl'):
        if r['mode'] != 'think':
            continue
        it = items[r['item']]
        out.append({'source': 'odd26_think', 'task': 'odd', 'mode': 'think_off',
                    'condition': f"{it['kind']}_{it['variant']}", 'id': r['id'], 'text': r['text'],
                    'finish_reason': r['finish_reason'],
                    'truth': {'row': it['target'][0] + 1, 'col': it['target'][1] + 1, 'rows': it['grid'], 'cols': it['grid']},
                    'extra': {'kind': it['kind'], 'variant': it['variant'], 'group': it['group']}})
    return out


def load_bv16(root: Path) -> list[dict]:
    base = root / 'behavior/2026-10-16_pe_scaling_t0'
    scores = {(s['case_id'], s['condition']): s for s in _jsonl(base / 'analysis/babyvision_rows.jsonl')}
    out = []
    for r in _jsonl(base / 'babyvision/results.jsonl'):
        if r['condition'] != 'x1':
            continue
        s = scores[(r['case_id'], 'x1')]
        out.append({'source': 'bv16', 'task': 'babyvision', 'mode': 'think_off', 'condition': s['subtype'],
                    'id': r['id'], 'text': r['response'], 'finish_reason': r['finish_reason'], 'truth': {},
                    'extra': {'strict': s['strict'], 'diagnostic': s['diagnostic'], 'subtype': s['subtype'],
                              'grid_decls': s.get('grid_decls')}})
    return out


def load_fresh(run_dir: Path) -> list[dict]:
    path = run_dir / 'gen/records.jsonl'
    if not path.exists():
        return []
    out = []
    for r in _jsonl(path):
        if r['mode'] not in ('think', 'thinking_on'):
            continue
        out.append({'source': 'fresh17', 'task': r['task'], 'mode': 'think_off' if r['mode'] == 'think' else 'thinking_on',
                    'condition': r['task'], 'id': r['id'], 'text': r['text'], 'finish_reason': r['finish_reason'],
                    'truth': r['truth'], 'extra': {'template': r.get('template')}})
    return out


LOADERS = {'dec13': load_dec13, 'ord15': load_ord15, 'pe16': load_pe16, 'rc310': load_rc310,
           'odd26': load_odd26, 'bv16': load_bv16}


def code_all(root: Path, run_dir: Path) -> list[dict]:
    records = []
    for name, fn in LOADERS.items():
        records += fn(root)
    records += load_fresh(run_dir)
    out = []
    for r in records:
        code = cm.code_cot(r['task'], r['text'], r['finish_reason'], r['truth'])
        if r['task'] == 'babyvision':
            code['correct'] = bool(r['extra']['strict'])
            code['has_answer'] = 'final answer' in r['text'].lower()
        code['mode_label'] = cm.primary_mode(code) if r['task'] not in ('babyvision', 'binary') else None
        out.append({k: r[k] for k in ('source', 'task', 'mode', 'condition', 'id', 'finish_reason', 'truth', 'extra')}
                   | {'code': code})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis-root', type=Path, required=True, help='WORK_DIR/analysis')
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    coded = code_all(args.analysis_root, args.run_dir)
    out = args.run_dir / 'taxonomy'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'coded.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in coded))
    print(f'coded {len(coded)} responses -> {out / "coded.jsonl"}')


if __name__ == '__main__':
    main()
