"""Reasoning-text observation for experiment 1.3 (design_decomposition.md, section 6).

Rule-based codes (declared grid size, declared target position, answer consistency) are
extracted from every explicit-reasoning and thinking-on response.  A stratified sample is
exported for manual coding; manual codes, if present in manual_codes.json, are summarised
together with the rule-based codes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from analysis_tools.behavior.decomposition import read_jsonl, write_jsonl

WORDS = {w: i for i, w in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen '
    'seventeen eighteen nineteen twenty'.split())}
NUM = r'(\d+|' + '|'.join(sorted(WORDS, key=len, reverse=True)) + r')'
ORDINALS = {w: i + 1 for i, w in enumerate(
    'first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth '
    'fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split())}
ORD = r'(\d+(?:st|nd|rd|th)|' + '|'.join(sorted(ORDINALS, key=len, reverse=True)) + r')'


def _num(token: str) -> int:
    token = token.lower()
    if token in ORDINALS:
        return ORDINALS[token]
    digits = re.match(r'\d+', token)
    return int(digits.group(0)) if digits else WORDS[token]


# Words before "N rows" / "N columns" that mark a movement or a subset ("count down 13 rows",
# "the first 10 rows") rather than the grid size.
MOVE_WORDS = re.compile(r'(?:down|across|over|move|moving|go|skip|count|counting|up|left|right|back|'
                        r'first|last|top|bottom|next|other|remaining)\W*$', re.I)


def _grid_count(text: str, noun: str) -> list[tuple[int, int]]:
    """(position, value) of statements of the number of rows or columns of the grid."""
    out = []
    for m in re.finditer(rf'\b{NUM}\s+{noun}s\b', text, re.I):
        if not MOVE_WORDS.search(text[max(0, m.start() - 20):m.start()]):
            out.append((m.start(), _num(m.group(1))))
    # "Rows: 6", "Total rows = 6", "Rows: 1 to 18" (range end); enumerations ("Rows: 1, 2, 3") and lists
    # starting on the next line are ignored.
    for m in re.finditer(rf'\b(?:total\s+)?{noun}s\**[ \t]*[:=][ \t]*\**[ \t]*(?:1[ \t]*(?:to|-|–)[ \t]*)?{NUM}\b(?![ \t]*,)',
                         text, re.I):
        out.append((m.start(), _num(m.group(1))))
    return out


def extract_codes(text: str, final_rc: tuple[int, int] | None) -> dict:
    """Declared grid (rows, cols), last declared 1-based (row, col), and consistency with the final answer.

    Grid size: the first "R x C" / "R by C" statement or the first pair of "N rows" and "N columns"
    (including "Rows: N" lists); counts that follow movement words ("count down 13 rows") are ignored.
    Position: the last "row r ... column c" statement (at most 40 non-digit characters between the two) (cardinal, "Row: r / Column: c",
    "row = r", bold markup) or "r-th row ... c-th column".
    """
    flags = re.I
    candidates = []
    m = re.search(rf'\b{NUM}\s*(?:x|×|by)\s*{NUM}\b', text, flags)
    if m:
        candidates.append((m.start(), [_num(m.group(1)), _num(m.group(2))]))
    rows, cols = _grid_count(text, 'row'), _grid_count(text, 'column')
    if rows and cols:
        candidates.append((min(rows[0][0], cols[0][0]), [rows[0][1], cols[0][1]]))
    declared_grid = min(candidates)[1] if candidates else None
    sep = r'[^\d]{0,40}?'  # up to 40 characters without digits between row and column
    cardinal = rf'\brow\**\s*(?:[:=]\s*)?\**\s*{NUM}\b{sep}\b(?:column|col\.?)\**\s*(?:[:=]\s*)?\**\s*{NUM}\b'
    ordinal = rf'\b{ORD}\s+row\b{sep}\b{ORD}\s+column\b'
    found = [(m.start(), m.group(1), m.group(2)) for m in re.finditer(cardinal, text, flags)]
    found += [(m.start(), m.group(1), m.group(2)) for m in re.finditer(ordinal, text, flags)]
    found.sort()
    declared_rc = [_num(found[-1][1]), _num(found[-1][2])] if found else None
    consistent = None
    if declared_rc is not None and final_rc is not None:
        consistent = tuple(declared_rc) == tuple(final_rc)
    return {'declared_grid': declared_grid, 'declared_rc': declared_rc, 'consistent': consistent}


def split_reasoning(response: str, stage: str) -> tuple[str, str]:
    """(reasoning text, final answer text)."""
    if stage == 'thinking':
        if '</think>' not in response:
            return response, ''
        head, tail = response.split('</think>', 1)
        return head, tail.strip()
    lines = response.rstrip().splitlines()
    return '\n'.join(lines[:-1]), (lines[-1] if lines else '')


def code_all(output: Path) -> list[dict]:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    rows = []
    for stage in ('explicit', 'thinking'):
        rpath, spath = output / f'results_{stage}.jsonl', output / f'scores_{stage}_diagnostic.jsonl'
        if not rpath.exists() or not spath.exists():
            continue
        scores = {s['id']: s for s in read_jsonl(spath)}
        for r in read_jsonl(rpath):
            c, s = cases[r['id']], scores[r['id']]
            reasoning, final = split_reasoning(r['response'], stage)
            final_rc = None
            if c['format'] == 'row_column' and s.get('pred_rc') is not None:
                final_rc = (s['pred_rc'][0] + 1, s['pred_rc'][1] + 1)
            codes = extract_codes(reasoning, final_rc)
            truth = c['truth']
            codes['grid_correct'] = None if codes['declared_grid'] is None else codes['declared_grid'] == [truth['rows'], truth['cols']]
            if c['format'] == 'row_column':
                codes['declared_rc_correct'] = None if codes['declared_rc'] is None else \
                    codes['declared_rc'] == [truth['target_rc'][0] + 1, truth['target_rc'][1] + 1]
            rows.append({'id': r['id'], 'stage': stage, 'task': c['task'], 'kind': c['kind'], 'level': c['level'],
                         'grid': c['grid'], 'variant': c['variant'], 'correct': bool(s['correct']),
                         'valid': bool(s['valid']), 'finish_reason': r['finish_reason'], 'n_tokens': r['n_tokens'],
                         **codes})
    return rows


def sample_for_manual(rows: list[dict], per_stratum: int, seed: int) -> list[dict]:
    """Stratify by stage x task x correctness and draw up to ``per_stratum`` responses per stratum."""
    strata = defaultdict(list)
    for r in rows:
        strata[(r['stage'], r['task'], r['correct'])].append(r)
    rng = random.Random(seed)
    picked = []
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda r: r['id'])
        picked += rng.sample(items, min(per_stratum, len(items)))
    return picked


def summarise(rows: list[dict], manual: dict | None) -> dict:
    out = {'by_stage_task': {}, 'manual': {}}
    groups = defaultdict(list)
    for r in rows:
        groups[(r['stage'], r['task'], r['grid'])].append(r)
    for (stage, task, grid), rs in sorted(groups.items()):
        n = len(rs)
        decl = [r for r in rs if r['declared_grid'] is not None]
        pos = [r for r in rs if r.get('declared_rc') is not None]
        cons = [r for r in rs if r['consistent'] is not None]
        out['by_stage_task'][f'{stage}|{task}|{grid}'] = {
            'n': n, 'accuracy': sum(r['correct'] for r in rs) / n, 'truncated': sum(r['finish_reason'] != 'eos' for r in rs),
            'mean_tokens': sum(r['n_tokens'] for r in rs) / n,
            'grid_declared': len(decl), 'grid_correct': sum(bool(r['grid_correct']) for r in decl),
            'declared_grids': Counter(str(r['declared_grid']) for r in decl).most_common(5),
            'position_declared': len(pos), 'position_correct': sum(bool(r.get('declared_rc_correct')) for r in pos),
            'consistent': sum(bool(r['consistent']) for r in cons), 'consistency_checked': len(cons)}
    if manual:
        coded = manual['codes']
        for field in ('attribute_description', 'search_strategy', 'failure_stage'):
            tally = defaultdict(Counter)
            for item in coded:
                if item.get(field) is not None:
                    tally[f"{item['stage']}|{item['task']}|{'correct' if item['correct'] else 'wrong'}"][item[field]] += 1
            out['manual'][field] = {k: dict(v) for k, v in sorted(tally.items())}
        # Pooled over tasks, split by stage and grid size.
        grid_of = {r['id']: r['grid'] for r in rows}
        for field in ('attribute_description', 'search_strategy', 'failure_stage'):
            tally = defaultdict(Counter)
            for item in coded:
                if item.get(field) is not None:
                    tally[f"{item['stage']}|{grid_of[item['id']]}"][item[field]] += 1
            out['manual'][field + '_by_grid'] = {k: dict(v) for k, v in sorted(tally.items())}
        checked = [i['rule_extraction_ok'] for i in coded if i.get('rule_extraction_ok') is not None]
        out['manual']['rule_extraction_ok'] = {'n': len(checked), 'ok': sum(checked)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-stratum', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    rows = code_all(args.output)
    if not rows:
        raise ValueError('No scored reasoning results.')
    target = args.output / 'reasoning'
    target.mkdir(exist_ok=True)
    write_jsonl(target / 'rule_codes.jsonl', rows)
    sheet = target / 'manual_sample.jsonl'
    if not sheet.exists():
        # The sheet carries everything the manual coder needs: input image, prompt, truth and response.
        cases = {c['id']: c for c in read_jsonl(args.output / 'cases.jsonl')}
        responses = {}
        for stage in ('explicit', 'thinking'):
            if (args.output / f'results_{stage}.jsonl').exists():
                responses.update({r['id']: r['response'] for r in read_jsonl(args.output / f'results_{stage}.jsonl')})
        write_jsonl(sheet, [{'id': r['id'], 'stage': r['stage'], 'task': r['task'], 'correct': r['correct'],
                             'image': str(args.output / cases[r['id']]['image']), 'prompt': cases[r['id']]['prompt'],
                             'truth': cases[r['id']]['truth'], 'kind': r['kind'], 'level': r['level'],
                             'template': cases[r['id']].get('template'), 'response': responses[r['id']],
                             'rule_codes': {k: r.get(k) for k in ('declared_grid', 'declared_rc', 'consistent')}}
                            for r in sample_for_manual(rows, args.per_stratum, args.seed)])
    manual_path = target / 'manual_codes.json'
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else None
    summary = summarise(rows, manual)
    (target / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary['by_stage_task'], indent=1)[:5000])


if __name__ == '__main__':
    main()
