"""Format-tolerant diagnostic score for location answers of experiments 1.3 and 1.4.

The strict and the content-diagnostic scores of ``decomposition.py`` require the answer to be valid
JSON.  Many row/column answers contain the correct content in malformed JSON, for example
``[{"row": 4, "column": 4]``.  This module adds a third, separate score level ("tolerant") and leaves
the strict and diagnostic score files unchanged:

* every answer that the diagnostic score accepts, or rejects for a reason other than a JSON parse
  error (truncation, wrong schema of a parsable value, value out of range, degenerate box), keeps
  its diagnostic score;
* an answer that the diagnostic score rejects as unparsable is accepted only when the final answer
  region contains exactly one numeric candidate for the requested keys (one ``"row": int`` adjacent
  to one ``"column": int``; one ``"point_2d": [x, y]``; one ``"bbox_2d": [x0, y0, x1, y1]``),
  regardless of bracket balance.  The extracted value is then scored by the unchanged diagnostic
  scorer, so range checks and correctness criteria are identical to the diagnostic level.

Commands:
  score     -- write scores_*_tolerant.jsonl next to the existing score files of the three runs
  analyze   -- invalid-answer classification and the key 1.3 / supplement / 1.4 tables under the
               diagnostic and the tolerant level, written to the analysis output directory
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Callable

import numpy as np

from analysis_tools.behavior.decomposition import answer_part, read_jsonl, score_case, write_jsonl

LOCATE_FORMATS = ('row_column', 'point', 'bbox')

# ----------------------------------------------------------------------------- tolerant extraction

_INT = r'(-?\d+)(?![\d.eE])'
_NUM = r'(-?\d+(?:\.\d+)?)(?![\d.eE])'
_ROW = re.compile(r'"row"\s*:\s*' + _INT)
_COL = re.compile(r'"column"\s*:\s*' + _INT)
_ROW_COL = re.compile(r'"row"\s*:\s*' + _INT + r'\s*,\s*"column"\s*:\s*' + _INT)
_COL_ROW = re.compile(r'"column"\s*:\s*' + _INT + r'\s*,\s*"row"\s*:\s*' + _INT)
_LIST_KEY = {'point': ('point_2d', 2), 'bbox': ('bbox_2d', 4)}


def _list_patterns(key: str, length: int) -> tuple[re.Pattern, re.Pattern]:
    """(numeric list start, complete numeric list of the required length) for a coordinate key."""
    start = re.compile(rf'"{key}"\s*:\s*\[\s*-?\d')
    full = re.compile(rf'"{key}"\s*:\s*\[\s*' + r'\s*,\s*'.join([_NUM] * length) + r'\s*\]')
    return start, full


def extract_tolerant(text: str, fmt: str) -> tuple[object | None, str]:
    """Unambiguous coordinate content of an answer text, ignoring bracket balance.

    Returns (value, reason).  ``value`` is the canonical JSON value of the requested schema
    ({"row": r, "column": c} or [{"point_2d"/"bbox_2d": [...]}]) or None; ``reason`` is 'ok',
    'no_candidate' (no numeric candidate for the requested keys) or 'ambiguous' (more than one
    candidate, or keys that do not form one adjacent pair / one complete list).
    """
    if fmt == 'row_column':
        rows, cols = _ROW.findall(text), _COL.findall(text)
        if not rows and not cols:
            return None, 'no_candidate'
        pairs = [(int(r), int(c)) for r, c in _ROW_COL.findall(text)] + \
                [(int(r), int(c)) for c, r in _COL_ROW.findall(text)]
        if len(rows) != 1 or len(cols) != 1 or len(pairs) != 1:
            return None, 'ambiguous'
        row, col = pairs[0]
        return {'row': row, 'column': col}, 'ok'
    if fmt not in _LIST_KEY:
        raise ValueError(f'No tolerant parse for format {fmt!r}')
    key, length = _LIST_KEY[fmt]
    start, full = _list_patterns(key, length)
    n_start, found = len(start.findall(text)), full.findall(text)
    if not n_start:
        return None, 'no_candidate'
    if n_start != 1 or len(found) != 1:
        return None, 'ambiguous'
    values = [float(v) if '.' in v else int(v) for v in found[0]]
    return [{key: values}], 'ok'


# ----------------------------------------------------------------------------- invalid-answer classification

INVALID_TYPES = ('truncated', 'bracket_mismatch', 'extra_text', 'different_schema', 'no_coordinate', 'value_invalid')


def strip_fence(text: str) -> str:
    return re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())


def brackets_balanced(text: str) -> bool:
    """True when {} and [] outside JSON strings nest correctly and every string is closed."""
    stack, in_str, esc = [], False, False
    pairs = {'}': '{', ']': '['}
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in '{[':
            stack.append(ch)
        elif ch in '}]':
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack and not in_str


def classify_invalid(case: dict, record: dict, diag: dict) -> str | None:
    """Type of an answer that the content-diagnostic score marks invalid; None for valid answers.

    truncated         -- generation stopped at the token budget (finish_reason = length)
    bracket_mismatch  -- not parsable as JSON; brackets outside strings do not nest or balance
    extra_text        -- not parsable as JSON with balanced brackets: text around or inside the
                         JSON (prose, a stray backtick, an unescaped echo of the prompt in a label)
    different_schema  -- parsable JSON with a structure other than the requested one (e.g. a bare
                         [r, c] list or several answer objects)
    no_coordinate     -- not parsable and no numeric candidate for the requested keys (prose only)
    value_invalid     -- requested schema, but the value fails the range check (ordinal outside the
                         grid, coordinate above 1000, degenerate box)
    """
    if diag.get('valid'):
        return None
    failure = diag.get('failure')
    if failure == 'truncated' or record['finish_reason'] != 'eos':
        return 'truncated'
    if failure in ('out_of_range', 'degenerate_box'):
        return 'value_invalid'
    text = answer_part(record['response'])
    if failure == 'unparsable':
        _, reason = extract_tolerant(text, case['format'])
        if reason == 'no_candidate':
            return 'no_coordinate'
        return 'extra_text' if brackets_balanced(strip_fence(text)) else 'bracket_mismatch'
    if failure == 'schema':
        value, reason = extract_tolerant(text, case['format'])
        if reason == 'ok' and case['format'] in _LIST_KEY:
            nums = value[0][_LIST_KEY[case['format']][0]]
            if any(not 0 <= v <= 1000 for v in nums):
                return 'value_invalid'
        return 'different_schema'
    raise ValueError(f'Unknown failure {failure!r} for {case["id"]}')


# ----------------------------------------------------------------------------- tolerant score

def canonical_response(value) -> str:
    return json.dumps(value)


def score_tolerant(case: dict, record: dict, diag: dict, scorer: Callable[[dict, dict], dict]) -> dict:
    """Tolerant score of one answer given its stored diagnostic score.

    ``scorer(case, record)`` must be the diagnostic scorer that produced ``diag``.  Only answers that
    the diagnostic rejected as unparsable and that finished with EOS are re-examined.
    """
    out = {**diag, 'parse': 'json' if diag.get('valid') else 'rejected'}
    if case.get('format') in LOCATE_FORMATS:
        kind = classify_invalid(case, record, diag) if 'valid' in diag else None
        if kind:
            out['invalid_type'] = kind
    if diag.get('failure') != 'unparsable' or record['finish_reason'] != 'eos' \
            or case.get('format') not in LOCATE_FORMATS:
        return out
    value, reason = extract_tolerant(answer_part(record['response']), case['format'])
    if value is None:
        return {**out, 'tolerant_reason': reason}
    rescored = scorer(case, {**record, 'response': canonical_response(value)})
    keep = {k: diag[k] for k in ('id', 'finish_reason') if k in diag}
    return {**keep, **rescored, 'parse': 'tolerant', 'invalid_type': out['invalid_type'], 'tolerant_value': value}


# ----------------------------------------------------------------------------- run I/O

def _strip(s: dict) -> dict:
    return {k: v for k, v in s.items() if k not in ('id', 'finish_reason', 'convention')}


def _rescore_file(cases: dict, records: dict, diag_path: Path, out_path: Path,
                  scorer: Callable[[dict, dict], dict], post: Callable[[dict, dict, dict], dict] | None = None) -> Counter:
    """Verify the stored diagnostic file against ``scorer`` and write the tolerant file."""
    rows, stats = [], Counter()
    for d in read_jsonl(diag_path):
        case, record = cases[d['id']], records[d['id']]
        if case.get('format') in LOCATE_FORMATS:
            fresh = scorer(case, record)
            if fresh != _strip(d):
                raise ValueError(f'Stored diagnostic score of {d["id"]} differs from the current scorer.')
        t = score_tolerant(case, record, d, scorer)
        if post is not None:
            t = post(case, record, t)
        stats[t['parse']] += 1
        if t['parse'] == 'tolerant':
            stats['rescued_correct'] += bool(t.get('correct'))
        rows.append(t)
    write_jsonl(out_path, rows)
    return stats


def score_formal(formal: Path) -> None:
    for stage, cases_file in (('main', 'cases.jsonl'), ('explicit', 'cases.jsonl'), ('thinking', 'cases.jsonl'),
                              ('babyvision', 'babyvision_cases.jsonl')):
        cases = {c['id']: c for c in read_jsonl(formal / cases_file)}
        records = {r['id']: r for r in read_jsonl(formal / f'results_{stage}.jsonl')}
        stats = _rescore_file(cases, records, formal / f'scores_{stage}_diagnostic.jsonl',
                              formal / f'scores_{stage}_tolerant.jsonl', lambda c, r: score_case(c, r, True))
        print(f'formal {stage}: {dict(stats)}')


def score_supplement(supplement: Path) -> None:
    from analysis_tools.behavior.decomposition_conditions import results_files, score_ring
    for part in ('layouts', 'ring'):
        cases = {c['id']: c for c in read_jsonl(supplement / f'cases_{part}.jsonl')}
        records = {r['id']: r for p in results_files(supplement, part) for r in read_jsonl(p)}
        scorer = (lambda c, r: score_ring(c, r, True)) if part == 'ring' else (lambda c, r: score_case(c, r, True))
        stats = _rescore_file(cases, records, supplement / f'scores_{part}_diagnostic.jsonl',
                              supplement / f'scores_{part}_tolerant.jsonl', scorer)
        print(f'supplement {part}: {dict(stats)}')


def score_ordinal(ordinal: Path) -> None:
    from analysis_tools.behavior import coordinate_convention as cc
    from analysis_tools.behavior.ordinal_token_grid import result_files
    cases = {c['id']: c for c in read_jsonl(ordinal / 'cases.jsonl')}
    records = {r['id']: r for p in result_files(ordinal) for r in read_jsonl(p)}

    def post(case, record, t):
        # The convention column re-reads the raw text; rescued point answers are re-read from the
        # canonical value so that both columns describe the same answer.
        if case['format'] == 'point' and t['parse'] == 'tolerant':
            t['convention'] = cc.score_point(canonical_response(t['tolerant_value']), 'eos', case['truth']['canvas'],
                                             case['truth']['target_cell_box'])
        return t
    stats = _rescore_file(cases, records, ordinal / 'scores_diagnostic.jsonl', ordinal / 'scores_tolerant.jsonl',
                          lambda c, r: score_case(c, r, True), post)
    print(f'ordinal: {dict(stats)}')


# ----------------------------------------------------------------------------- analysis

def _wilson(k: int, n: int) -> tuple[float, float]:
    from analysis_tools.behavior.decomposition_analysis import wilson
    return wilson(k, n)


def _rate(flags: list[bool]) -> dict:
    n, k = len(flags), int(sum(flags))
    return {'k': k, 'n': n, 'p': k / n if n else float('nan'), 'ci': list(_wilson(k, n))}


def _fmt(s: dict, ci: bool = True) -> str:
    if not s['n']:
        return '–'
    return f"{s['p']:.2f}" + (f" [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]" if ci else '')


def _mcnemar(b: int, c: int) -> float:
    from analysis_tools.behavior.decomposition_supplement import mcnemar_exact
    return mcnemar_exact(b, c)


def invalid_table(rows: list[dict], group: Callable[[dict], tuple]) -> list[dict]:
    """Counts of invalid-answer types per group; ``rows`` carry 'case' and the tolerant 'score'."""
    by = defaultdict(list)
    for r in rows:
        if r['format'] in LOCATE_FORMATS:
            by[group(r)].append(r)
    out = []
    natural = lambda k: tuple((0, x, '') if isinstance(x, int) else (1, 0, str(x)) for x in k)
    for key in sorted(by, key=natural):
        rs = by[key]
        types = Counter(r['score'].get('invalid_type') for r in rs)
        out.append({'group': list(key), 'n': len(rs), **{t: types[t] for t in INVALID_TYPES},
                    'n_invalid': sum(types[t] for t in INVALID_TYPES),
                    'rescued': sum(r['score']['parse'] == 'tolerant' for r in rs),
                    'rescued_correct': sum(r['score']['parse'] == 'tolerant' and bool(r['score'].get('correct')) for r in rs)})
    return out


def _md_invalid(table: list[dict], header: list[str]) -> list[str]:
    cols = ['n', 'n_invalid', *INVALID_TYPES, 'rescued', 'rescued_correct']
    lines = ['| ' + ' | '.join(header + cols) + ' |', '| ' + ' | '.join(['---'] * (len(header) + len(cols))) + ' |']
    for t in table:
        if t['n_invalid']:
            lines.append('| ' + ' | '.join([str(g) for g in t['group']] + [str(t[c]) for c in cols]) + ' |')
    return lines


def formal_comparison(formal: Path) -> dict:
    """Key 1.3 tables under the diagnostic and the tolerant level (primary wording, main stage)."""
    from analysis_tools.behavior.decomposition import GRIDS, KINDS
    from analysis_tools.behavior.decomposition_analysis import conditional, load, rowcol_error_structure
    out: dict = {}
    levels = {}
    for kind in ('diagnostic', 'tolerant'):
        rows = [r for r in load(formal, kind=kind) if r['wording'] == 'primary']
        levels[kind] = rows
        acc = {}
        for task in ('T4', 'T5', 'T6'):
            for grid in GRIDS:
                rc = [r for r in rows if r['task'] == task and r['format'] == 'row_column' and r['grid'] == grid]
                acc[f'{task}|{grid}|all'] = _rate([bool(r['score'].get('correct')) for r in rc])
                if task == 'T6':
                    continue
                for k in KINDS:
                    acc[f'{task}|{grid}|{k}'] = _rate([bool(r['score'].get('correct')) for r in rc if r['kind'] == k])
                # Discriminable changes: all but colour differences of 5 and 10 (levels 1 and 2).
                acc[f'{task}|{grid}|discriminable'] = _rate([bool(r['score'].get('correct')) for r in rc
                                                             if not (r['kind'] == 'color' and r['level'] <= 2)])
            for fmt, key in (('point', 'hit'), ('bbox', 'center_hit')):
                for grid in GRIDS:
                    rs = [r for r in rows if r['task'] == task and r['format'] == fmt and r['grid'] == grid]
                    acc[f'{task}|{grid}|{fmt}'] = _rate([bool(r['score'].get(key)) for r in rs])
        cond = conditional(rows)
        records = cond.pop('records')
        # Paired T4 / T5 row/col outcome on the same image.
        paired = {}
        for grid in GRIDS:
            recs = [v for v in records.values() if v['_grid'] == grid and 'T4 row/col' in v and 'T5 row/col' in v]
            b = sum(v['T4 row/col'] and not v['T5 row/col'] for v in recs)
            c = sum(v['T5 row/col'] and not v['T4 row/col'] for v in recs)
            paired[str(grid)] = {'n': len(recs), 'T4': _rate([v['T4 row/col'] for v in recs]),
                                 'T5': _rate([v['T5 row/col'] for v in recs]), 'T4_only': b, 'T5_only': c,
                                 'mcnemar_p': _mcnemar(b, c)}
        t2_t5 = {str(g): sum(v.get('T2') is True and v.get('T5 row/col') is False for v in records.values() if v['_grid'] == g)
                 for g in GRIDS}
        out[kind] = {'acc': acc, 'conditional': cond['tables'], 'paired': paired, 'T2_ok_T5_fail': t2_t5,
                     'errors': rowcol_error_structure(rows)}
    return out


def supplement_comparison(supplement: Path) -> dict:
    from analysis_tools.behavior.decomposition_conditions_analysis import layouts_summary, load
    out = {}
    for kind in ('diagnostic', 'tolerant'):
        rows = load(supplement, 'layouts', kind)
        summ = layouts_summary(rows, None)
        out[kind] = {name: {task: {'rowcol': e[task]['rowcol'], 'point_hit': e[task]['point_hit'],
                                   'col_only': e[task]['col_only'], 'col_plus1': e[task]['col_plus1'],
                                   'row_only': e[task]['row_only'], 'row_plus1': e[task]['row_plus1'],
                                   'n_wrong': e[task]['n_wrong_rc']} for task in ('T4', 'T5', 'T6')}
                     for name, e in summ.items()}
    return out


def ordinal_comparison(ordinal: Path) -> dict:
    """Changed 1.4 condition entries: T6 row/col accuracy and the column +1 rate."""
    from analysis_tools.behavior.ordinal_token_grid_analysis import condition_summary, ordinal_offsets
    cases = {c['id']: c for c in read_jsonl(ordinal / 'cases.jsonl')}
    strict = {s['id']: s for s in read_jsonl(ordinal / 'scores_strict.jsonl')}
    responses = {r['id']: r['response'] for p in sorted(ordinal.glob('results*.jsonl')) for r in read_jsonl(p)}
    out = {}
    for kind, fname in (('diagnostic', 'scores_diagnostic.jsonl'), ('tolerant', 'scores_tolerant.jsonl')):
        rows = []
        for s in read_jsonl(ordinal / fname):
            case = cases[s['id']]
            dr = dc = None
            if case['format'] == 'row_column':
                if s.get('parse') == 'tolerant':
                    tr, tc = case['truth']['target_rc']
                    v = s['tolerant_value']
                    dr, dc = v['row'] - 1 - tr, v['column'] - 1 - tc
                else:
                    dr, dc = ordinal_offsets(case, s, responses[s['id']])
            rows.append({**case, 'score': s, 'strict': strict[s['id']], 'dr': dr, 'dc': dc})
        summ = condition_summary(rows)
        out[kind] = {k: {m: v[m] for m in ('T6_rowcol_acc', 'T6_col_plus1', 'T6_row_plus1', 'T6_point_hit')}
                     for k, v in summ.items()}
    changed = {k: {kind: out[kind][k] for kind in out} for k in out['diagnostic']
               if out['diagnostic'][k] != out['tolerant'][k]}
    return {'changed': changed, 'n_conditions': len(out['diagnostic'])}


def analyze(formal: Path, supplement: Path, ordinal: Path, target: Path) -> None:
    from analysis_tools.behavior.decomposition import GRIDS
    target.mkdir(parents=True, exist_ok=True)
    # Invalid-answer classification on all three runs (both wordings of the formal run).
    def rows_of(cases_path: Path, score_path: Path, stage: str | None = None) -> list[dict]:
        cases = {c['id']: c for c in read_jsonl(cases_path)}
        return [{**cases[s['id']], 'score': s} for s in read_jsonl(score_path)
                if stage is None or cases[s['id']].get('stage') == stage]
    formal_rows = rows_of(formal / 'cases.jsonl', formal / 'scores_main_tolerant.jsonl')
    layout_rows = rows_of(supplement / 'cases_layouts.jsonl', supplement / 'scores_layouts_tolerant.jsonl')
    ring_rows = rows_of(supplement / 'cases_ring.jsonl', supplement / 'scores_ring_tolerant.jsonl')
    ordinal_rows = rows_of(ordinal / 'cases.jsonl', ordinal / 'scores_tolerant.jsonl')
    other_stages = {st: rows_of(formal / ('babyvision_cases.jsonl' if st == 'babyvision' else 'cases.jsonl'),
                                formal / f'scores_{st}_tolerant.jsonl') for st in ('explicit', 'thinking', 'babyvision')}
    summary = {
        'invalid': {
            'formal_main': invalid_table(formal_rows, lambda r: (r['task'], r['format'], r['grid'])),
            **{f'formal_{st}': invalid_table(rs, lambda r: (r['task'], r['format'], r.get('grid')))
               for st, rs in other_stages.items()},
            'supplement_layouts': invalid_table(layout_rows, lambda r: (r['task'], r['format'], r['layout'])),
            'supplement_ring': invalid_table(ring_rows, lambda r: (r['task'], r['format'])),
            'ordinal': invalid_table(ordinal_rows, lambda r: (r['task'], r['format'], r['condition']))},
        'formal': formal_comparison(formal),
        'supplement': supplement_comparison(supplement),
        'ordinal': ordinal_comparison(ordinal)}
    (target / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')

    L = ['# Tolerant parse of location answers (generated)', '']
    for name, header in (('formal_main', ['task', 'format', 'grid']), ('formal_explicit', ['task', 'format', 'grid']),
                         ('formal_thinking', ['task', 'format', 'grid']), ('formal_babyvision', ['task', 'format', 'grid']),
                         ('supplement_layouts', ['task', 'format', 'layout']), ('supplement_ring', ['task', 'format']),
                         ('ordinal', ['task', 'format', 'condition'])):
        L += [f'## Invalid answers: {name}', ''] + _md_invalid(summary['invalid'][name], header) + ['']
    f = summary['formal']
    L += ['## Formal run: row/col accuracy (primary wording), diagnostic -> tolerant', '',
          '| task | grid | subset | diagnostic | tolerant | delta |', '| --- | --- | --- | --- | --- | --- |']
    for key, d in f['diagnostic']['acc'].items():
        t = f['tolerant']['acc'][key]
        L.append(f"| {' | '.join(key.split('|'))} | {_fmt(d)} | {_fmt(t)} | {t['p'] - d['p']:+.3f} |")
    L += ['', '## Formal run: paired T4 vs T5 row/col (same image)', '',
          '| grid | level | n | T4 | T5 | T4 only | T5 only | McNemar p |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for g in GRIDS:
        for kind in ('diagnostic', 'tolerant'):
            p = f[kind]['paired'][str(g)]
            L.append(f"| {g} | {kind} | {p['n']} | {_fmt(p['T4'])} | {_fmt(p['T5'])} | {p['T4_only']} | {p['T5_only']} | {p['mcnemar_p']:.3g} |")
    L += ['', '## Formal run: T5 row/col success split by another sub-task', '',
          '| grid | sub-task | level | n ok | P(T5 | ok) | n fail | P(T5 | fail) |', '| --- | --- | --- | --- | --- | --- | --- |']
    for g in GRIDS:
        for a in ('T1', 'T2', 'T4 row/col'):
            for kind in ('diagnostic', 'tolerant'):
                v = f[kind]['conditional'][str(g)]['T5_given'][a]
                L.append(f"| {g} | {a} | {kind} | {v['n_ok']} | {v['p_ok']:.3f} | {v['n_fail']} | {v['p_fail']:.3f} |")
    L += ['', '## Formal run: images with paired T2 success and T5 row/col failure', '',
          '| level | ' + ' | '.join(str(g) for g in GRIDS) + ' |', '| --- ' * (len(GRIDS) + 1) + '|']
    for kind in ('diagnostic', 'tolerant'):
        L.append(f'| {kind} | ' + ' | '.join(str(f[kind]['T2_ok_T5_fail'][str(g)]) for g in GRIDS) + ' |')
    L += ['', '## Formal run: wrong row/col answers', '',
          '| task | grid | level | n wrong | answer (1,1) | column only | adjacent |', '| --- | --- | --- | --- | --- | --- | --- |']
    for key in f['diagnostic']['errors']:
        for kind in ('diagnostic', 'tolerant'):
            e = f[kind]['errors'][key]
            if e['n_wrong']:
                L.append(f"| {' | '.join(key.split('|'))} | {kind} | {e['n_wrong']} | {e['pred_11']:.2f} | {e['col_only']:.2f} | {e['adjacent']:.2f} |")
    L += ['', '## Supplement layouts: row/col accuracy, diagnostic -> tolerant', '',
          '| layout | task | diagnostic | tolerant | wrong (diag -> tol) | col +1 among column-only (diag -> tol) |',
          '| --- | --- | --- | --- | --- | --- |']
    s = summary['supplement']
    for name in s['diagnostic']:
        for task in ('T4', 'T5', 'T6'):
            d, t = s['diagnostic'][name][task], s['tolerant'][name][task]
            L.append(f"| {name} | {task} | {_fmt(d['rowcol'])} | {_fmt(t['rowcol'])} | {d['n_wrong']} -> {t['n_wrong']} | "
                     f"{d['col_plus1']}/{d['col_only']} -> {t['col_plus1']}/{t['col_only']} |")
    o = summary['ordinal']
    L += ['', f"## Ordinal token grid (1.4): conditions whose T6 entries change ({len(o['changed'])}/{o['n_conditions']})", '',
          '| condition | metric | diagnostic | tolerant |', '| --- | --- | --- | --- |']
    for name, v in o['changed'].items():
        for m in v['diagnostic']:
            if v['diagnostic'][m] != v['tolerant'][m]:
                L.append(f"| {name} | {m} | {_fmt(v['diagnostic'][m])} | {_fmt(v['tolerant'][m])} |")
    (target / 'tables.md').write_text('\n'.join(L) + '\n')
    plot_invalid(summary['invalid']['formal_main'], summary['invalid']['supplement_layouts'], target / 'figures' / 'invalid_types.png')
    plot_conditional(f, target / 'figures' / 't5_given_t2.png')
    print(f"wrote {target / 'tables.md'}")


# ----------------------------------------------------------------------------- figures

TYPE_COLORS = {'bracket_mismatch': '#2a78d6', 'extra_text': '#1baf7a', 'different_schema': '#eda100',
               'no_coordinate': '#e87ba4', 'truncated': '#6250d6', 'value_invalid': '#b8b7b2'}
TYPE_LABELS = {'bracket_mismatch': 'bracket mismatch', 'extra_text': 'extra text', 'different_schema': 'different schema',
               'no_coordinate': 'no coordinate', 'truncated': 'truncated', 'value_invalid': 'value invalid'}


def plot_invalid(formal: list[dict], layouts: list[dict], path: Path) -> None:
    """Stacked counts of invalid row/col answers per task x grid (formal) and per task x layout."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis_tools.behavior.decomposition_analysis import INK, MUTED, _style
    _style()
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [('1.3 formal run (row/col)', [t for t in formal if t['group'][1] == 'row_column'], lambda g: f'{g[0]} {g[2]}×{g[2]}'),
              ('1.3 supplement layouts (row/col)', [t for t in layouts if t['group'][1] == 'row_column' and t['n_invalid']],
               lambda g: f'{g[0]} {g[2]}')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [len(p[1]) or 1 for p in panels]})
    for ax, (title, table, label) in zip(axes, panels):
        x = np.arange(len(table))
        bottom = np.zeros(len(table))
        for typ in ('bracket_mismatch', 'extra_text', 'different_schema', 'no_coordinate', 'truncated', 'value_invalid'):
            h = np.array([t[typ] for t in table], float)
            ax.bar(x, h, bottom=bottom, color=TYPE_COLORS[typ], width=0.72, label=TYPE_LABELS[typ], edgecolor='white', linewidth=0.5)
            bottom += h
        for xi, t in zip(x, table):
            ax.text(xi, bottom[xi] + 1, f"{t['n_invalid']}/{t['n']}", ha='center', va='bottom', fontsize=7, color=MUTED)
        ax.set_ylim(0, max(bottom.max(initial=1), 1) * 1.1)
        ax.set_xticks(x, [label(t['group']) for t in table], rotation=60, ha='right', fontsize=8)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_ylabel('invalid answers (diagnostic score)')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_conditional(f: dict, path: Path) -> None:
    """P(T5 row/col correct | T2 paired correct / failed) and T4/T5 accuracy per grid, both levels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis_tools.behavior.decomposition import GRIDS
    from analysis_tools.behavior.decomposition_analysis import INK, _style
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(GRIDS))
    styles = {'diagnostic': dict(ls='--', marker='o', mfc='white'), 'tolerant': dict(ls='-', marker='o')}
    for kind, st in styles.items():
        given = [f[kind]['conditional'][str(g)]['T5_given']['T2'] for g in GRIDS]
        axes[0].plot(x, [v['p_ok'] for v in given], color='#2a78d6', label=f'T2 solved ({kind})', **st)
        axes[0].plot(x, [v['p_fail'] for v in given], color='#eb6834', label=f'T2 failed ({kind})', **st)
        for task, color in (('T4', '#1baf7a'), ('T5', '#6250d6')):
            acc = [f[kind]['acc'][f'{task}|{g}|all'] for g in GRIDS]
            p = np.array([a['p'] for a in acc])
            lo = p - np.array([a['ci'][0] for a in acc])
            hi = np.array([a['ci'][1] for a in acc]) - p
            axes[1].errorbar(x + (0.06 if kind == 'tolerant' else -0.06), p, yerr=[lo, hi], color=color, capsize=3,
                             label=f'{task} row/col ({kind})', **st)
    axes[0].set_title('P(T5 row/col correct | T2 outcome on the same image)', color=INK, fontsize=10)
    axes[1].set_title('Row/col accuracy, primary wording (95% Wilson)', color=INK, fontsize=10)
    for ax in axes:
        ax.set_xticks(x, [f'{g}×{g}' for g in GRIDS])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('grid')
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('score', 'analyze'))
    p.add_argument('--formal', type=Path, required=True, help='Run directory of the formal 1.3 evaluation.')
    p.add_argument('--supplement', type=Path, required=True, help='Run directory of the 1.3 supplement.')
    p.add_argument('--ordinal', type=Path, required=True, help='Run directory of experiment 1.4.')
    p.add_argument('--output', type=Path, required=True, help='Directory for the classification and comparison tables.')
    args = p.parse_args()
    for path in (args.formal, args.supplement, args.ordinal):
        if not path.is_dir():
            p.error(f'Missing run directory {path}')
    if args.command == 'score':
        score_formal(args.formal)
        score_supplement(args.supplement)
        score_ordinal(args.ordinal)
    else:
        analyze(args.formal, args.supplement, args.ordinal, args.output)


if __name__ == '__main__':
    main()
