"""Experiment 1.7: behaviour and mechanism of chain-of-thought (CoT) reasoning in Qwen3.5-9B.

This module holds the pure (model-free) parts of the experiment:

* task prompts for the four fresh task sets (row/column of a red figure, find-the-odd,
  point coordinates, row/column counting), in direct, think-then-answer and answer-first form;
* stimulus construction for those sets on top of stimuli-v1, including matched image pairs
  (A, B) with identical token layout and different target cells for the channel-swap tests;
* LLM-free rule coding of CoT texts (final answer, stated location, declared grid size,
  enumeration, disclaimers, self-correction, arithmetic slips) and the failure-mode taxonomy;
* helpers that locate and edit a stated location inside a CoT for the perturbation test.

Rows and columns are 1-based in every coded field; stimulus targets are 0-based (row, col)
as in stimuli-v1. GPU code lives in cot_mechanism_engine.py and cot_mechanism_run.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re

from analysis_tools.common import stimuli as st

# ----------------------------------------------------------------------------- prompts

ORDINAL_NOTE = 'counting rows from top to bottom and columns from left to right, starting at 1'
COORD_NOTE = ' All coordinates use 0-1000 normalization; x increases rightward and y increases downward.'

TASK_QUERY = {
    'rowcol': 'One figure is drawn in red.',
    'odd': 'One figure differs from all the others.',
    'point': 'One figure is drawn in red.',
    'count': 'How many rows and how many columns of figures are in this image?',
}
ANSWER_FORMAT = {
    'rowcol': '{"row": r, "column": c}',
    'odd': '{"row": r, "column": c}',
    'point': '[{"point_2d": [x, y]}]',
    'count': '{"rows": R, "columns": C}',
}
_TAIL = {
    'rowcol': f', {ORDINAL_NOTE}.',
    'odd': f', {ORDINAL_NOTE}.',
    'point': '.' + COORD_NOTE,
    'count': '.',
}
_THINK_HOW = {
    'rowcol': 'how you determine its row and its column',
    'odd': 'what you observe and how you find this figure, including how many rows and columns of figures there are',
    'point': 'how you find this figure and estimate the center of its bounding box',
    'count': 'how you count the rows and the columns',
}
_DIRECT_ASK = {
    'rowcol': ' Give its position. Return only {"row": r, "column": c}',
    'odd': ' Give its position. Return only {"row": r, "column": c}',
    'point': ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}]',
    'count': ' Return only {"rows": R, "columns": C}',
}
JSON_ONLY = ' Output only the JSON, with no explanation.'


def prompt(task: str, mode: str) -> str:
    """Prompt text for a task in mode 'direct', 'think' (CoT first) or 'answer_first'.

    The think prompt for 'rowcol' equals the experiment-1.5 / 3.10 think prompt and the one for
    'odd' equals the experiment-1.3 explicit-reasoning prompt, so earlier CoT corpora are comparable.
    """
    q, fmt, tail = TASK_QUERY[task], ANSWER_FORMAT[task], _TAIL[task]
    if mode == 'direct':
        return q + _DIRECT_ASK[task] + tail + JSON_ONLY
    if mode == 'think':
        return (f'{q} First describe step by step {_THINK_HOW[task]}. Then give the final answer on the last line '
                f'as {fmt}{tail}')
    if mode == 'answer_first':
        return (f'{q} First give the final answer on the first line as {fmt}{tail} Then explain step by step '
                f'{_THINK_HOW[task]}.')
    raise ValueError(f'unknown prompt mode {mode!r}')


# Answer cue appended after a (possibly truncated) CoT to force an answer (early-answering test).
FORCE_CUE = {
    'rowcol': '\n\nFinal answer: {"row": ',
    'odd': '\n\nFinal answer: {"row": ',
    'point': '\n\nFinal answer: [{"point_2d": [',
    'count': '\n\nFinal answer: {"rows": ',
}

# ----------------------------------------------------------------------------- stimuli

@dataclass(frozen=True)
class TaskItem:
    """One fresh stimulus. target is 0-based (row, col); rows/cols give the grid size."""
    id: str
    task: str
    template: str
    rows: int
    cols: int
    target: tuple[int, int] | None
    change: tuple[str, int, int] | None      # (kind, level, site) of stimuli-v1, None for count
    cell: int = 64

    def spec(self) -> st.StimulusSpec:
        change = st.NONE if self.change is None else st.Change(*self.change)
        return st.StimulusSpec(template=self.template, rows=self.rows, cols=self.cols, change=change,
                               target=self.target, cell=self.cell)

    def render(self):
        return st.generate(self.spec())


def unit_seed(*parts) -> int:
    text = '|'.join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


def _rng_cells(rows: int, cols: int, n: int, seed: int) -> list[tuple[int, int]]:
    import random
    rng = random.Random(seed)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    return cells[:n]


def partner_cell(target: tuple[int, int], rows: int, cols: int, seed: int, min_dist: int = 2) -> tuple[int, int]:
    """A second target cell differing from `target` by at least min_dist in both row and column."""
    import random
    rng = random.Random(seed)
    options = [(r, c) for r in range(rows) for c in range(cols)
               if abs(r - target[0]) >= min_dist and abs(c - target[1]) >= min_dist]
    if not options:
        raise ValueError(f'no partner cell for {target} in a {rows}x{cols} grid')
    return rng.choice(options)


TASK_SETS = {
    # task: (grid rows, grid cols, templates, change)
    'rowcol': (8, 8, ('person', 'ring'), ('color', 4, 0)),
    'odd': (6, 6, ('person', 'arrow'), ('gap', 4, 0)),
    'point': (10, 10, ('person', 'ring'), ('color', 4, 0)),
}
COUNT_RANGE = (3, 12)      # rows and columns of the counting grids are drawn uniformly from this range


def build_items(task: str, n: int, seed: int = 17) -> list[TaskItem]:
    """n fresh items of a task, balanced over templates (and grid sizes for 'count')."""
    if task == 'count':
        import random
        rng = random.Random(unit_seed(seed, 'count'))
        out = []
        for i in range(n):
            rows, cols = rng.randint(*COUNT_RANGE), rng.randint(*COUNT_RANGE)
            template = ('person', 'ring')[i % 2]
            out.append(TaskItem(f'count_{template}_{rows}x{cols}_{i:03d}', 'count', template, rows, cols, None, None))
        return out
    rows, cols, templates, change = TASK_SETS[task]
    out = []
    per = math.ceil(n / len(templates))
    for t_i, template in enumerate(templates):
        for k, cell in enumerate(_rng_cells(rows, cols, per, unit_seed(seed, task, template))):
            if len(out) >= n:
                break
            out.append(TaskItem(f'{task}_{template}_r{cell[0]}c{cell[1]}', task, template, rows, cols, cell, change))
    return out[:n]


def pair_item(item: TaskItem, seed: int = 23) -> TaskItem:
    """Matched partner B of item A: same template, grid and change; target moved by >=2 rows and columns."""
    if item.target is None:
        raise ValueError('count items have no target and cannot be paired')
    cell = partner_cell(item.target, item.rows, item.cols, unit_seed(seed, item.id))
    return TaskItem(f'{item.id}__B_r{cell[0]}c{cell[1]}', item.task, item.template, item.rows, item.cols, cell,
                    item.change, item.cell)


def truth_of(item: TaskItem, meta: dict | None = None) -> dict:
    """Ground truth in coded units: 1-based row/col, grid size, and the 0-1000 target centre
    (tight-box centre from the stimuli-v1 metadata)."""
    if item.target is None:
        return {'rows': item.rows, 'cols': item.cols}
    meta = meta if meta is not None else item.render()[1]
    inst = next(i for i in meta['instances'] if i['is_target'])
    r, c = item.target
    return {'row': r + 1, 'col': c + 1, 'rows': item.rows, 'cols': item.cols, 'point': list(inst['center_norm']),
            'cell_box_norm': [1000 * inst['cell_box'][0] / meta['canvas'][0], 1000 * inst['cell_box'][1] / meta['canvas'][1],
                              1000 * inst['cell_box'][2] / meta['canvas'][0], 1000 * inst['cell_box'][3] / meta['canvas'][1]]}


# ----------------------------------------------------------------------------- answer parsing

ROWCOL_RE = re.compile(r'"row"\s*:\s*(\d+)\s*,\s*"column"\s*:\s*(\d+)|"column"\s*:\s*(\d+)\s*,\s*"row"\s*:\s*(\d+)')
POINT_RE = re.compile(r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')
COUNT_RE = re.compile(r'"rows"\s*:\s*(\d+)\s*,\s*"columns"\s*:\s*(\d+)')
YESNO_RE = re.compile(r'Answer:\s*\**\s*(Yes|No)\b', re.I)


def _rowcol_matches(text: str) -> list[tuple[int, int, int, int]]:
    """(row, col, start, end) for every JSON row/column object in the text."""
    out = []
    for m in ROWCOL_RE.finditer(text):
        if m.group(1) is not None:
            out.append((int(m.group(1)), int(m.group(2)), m.start(), m.end()))
        else:
            out.append((int(m.group(4)), int(m.group(3)), m.start(), m.end()))
    return out


def final_answer(task: str, text: str) -> tuple[tuple | str | None, int]:
    """Last answer object of the response and its start offset (len(text) if absent).

    For think prompts the answer is the last object; for answer-first prompts callers use first_answer.
    """
    if task in ('rowcol', 'odd'):
        found = _rowcol_matches(text)
        return ((found[-1][0], found[-1][1]), found[-1][2]) if found else (None, len(text))
    if task == 'point':
        found = list(POINT_RE.finditer(text))
        return ((float(found[-1].group(1)), float(found[-1].group(2))), found[-1].start()) if found else (None, len(text))
    if task == 'count':
        found = list(COUNT_RE.finditer(text))
        return ((int(found[-1].group(1)), int(found[-1].group(2))), found[-1].start()) if found else (None, len(text))
    if task == 'binary':
        found = list(YESNO_RE.finditer(text))
        if found:
            return found[-1].group(1).capitalize(), found[-1].start()
        bare = re.findall(r'\b(Yes|No)\b\W*$', text.strip())
        return (bare[-1], len(text)) if bare else (None, len(text))
    raise ValueError(task)


def first_answer(task: str, text: str):
    """First answer object (answer-first prompts)."""
    if task in ('rowcol', 'odd'):
        found = _rowcol_matches(text)
        return (found[0][0], found[0][1]) if found else None
    if task == 'point':
        m = POINT_RE.search(text)
        return (float(m.group(1)), float(m.group(2))) if m else None
    if task == 'count':
        m = COUNT_RE.search(text)
        return (int(m.group(1)), int(m.group(2))) if m else None
    raise ValueError(task)


# ----------------------------------------------------------------------------- CoT coding

_ORD_WORDS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
              'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12, 'last': None}
_NUMW = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
         'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
         'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20}
_N = r'(\d+|' + '|'.join(_NUMW) + r')'
_ORD = r'(\d+)(?:st|nd|rd|th)|(' + '|'.join(k for k in _ORD_WORDS if k != 'last') + r')'


def _num(tok: str) -> int:
    tok = tok.lower()
    return int(tok) if tok.isdigit() else _NUMW[tok]


def _ord(m: re.Match, g1: int, g2: int) -> int:
    return int(m.group(g1)) if m.group(g1) else _ORD_WORDS[m.group(g2).lower()]


DISCLAIMER_RE = re.compile(
    r"can(?:no|')t (?:actually |directly )?see|cannot (?:view|access)|unable to (?:see|view)|don't have (?:access|the ability)"
    r"|do not have (?:access|the ability)|as an ai\b|text-based|simulat(?:e|ed|ing)|without (?:seeing|access to|being able to see)"
    r" the (?:actual )?image|without (?:the )?image\b|not (?:able|possible) to (?:see|view)|no image (?:is |was )?(?:provided|attached)"
    r"|since i (?:can't|cannot)", re.I)
# Unsupported assumptions in place of observation ("let's assume", "as described", "standard puzzle").
ASSUME_RE = re.compile(r"let's assume|let us assume|i'll assume|i will assume|as described|assuming (?:a |the )?(?:standard|typical)"
                       r"|hypothetical|(?:standard|typical|common|well-known) (?:puzzle|problem|version)", re.I)
SELF_CORRECT_RE = re.compile(r"\bwait\b|\bactually\b|\bre-?examin|\bre-?count|\brecheck|\bdouble[- ]check|\blet me "
                             r"(?:re|count again|look again|check)|\bcorrection\b|\bmistake\b|\bi miscounted\b", re.I)

GRID_RE = [
    re.compile(r'\b' + _N + r'\s*(?:rows?)\s*(?:and|,|×|x|by)\s*' + _N + r'\s*columns?\b', re.I),
    re.compile(r'\b(\d+)\s*[x×]\s*(\d+)\s*(?:grid|arrangement|layout|array|matrix)\b', re.I),
    re.compile(r'\bgrid\s+(?:of|with|is)\s+(\d+)\s*[x×]\s*(\d+)\b', re.I),
]
ROWS_DECL_RE = re.compile(r'\b(?:total of|there are|has|contains|with|are|of)\s+' + _N + r'\s+rows\b'
                          r'|\b' + _N + r'\s+rows\s+(?:in total|total|of figures|and)\b'
                          r'|(?:total|number of) rows\s*(?:=|:|is)\s*' + _N + r'\b', re.I)
COLS_DECL_RE = re.compile(r'\b(?:total of|there are|has|contains|with|are|of)\s+' + _N + r'\s+(?:columns|figures (?:per|in each) row)\b'
                          r'|\b' + _N + r'\s+columns\s+(?:in total|total|of figures)\b'
                          r'|\b' + _N + r'\s+figures (?:per|in each) row\b'
                          r'|(?:total|number of) columns\s*(?:=|:|is)\s*' + _N + r'\b', re.I)
ENUM_LINE_RE = re.compile(r'^\s*(?:[-*•]|\d+\.)?\s*(?:\*\*)?\s*(row|column|col)\s*(\d+)\b(.*)$', re.I | re.M)
PRECEDING_RE = re.compile(r'\b' + _N + r'\s+(?:black\s+)?(?:figures?|circles?|columns?|rows?|stick figures?|ones?)\s+'
                          r'(?:to (?:the|its) left|above|before|preceding|on (?:the|its) left)', re.I)
ORDINAL_AFTER_RE = re.compile(r'\b(?:' + _ORD + r')\b|\b(?:position|column|row)\s*(\d+)\b', re.I)


def _value(tok: str) -> int | None:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return _NUMW.get(tok, _ORD_WORDS.get(tok))


def _first_group(m: re.Match) -> int | None:
    for g in m.groups():
        if g:
            return _value(g)
    return None


_MENTION = {
    'row': re.compile(r'\brow\s*(?:index\s*|number\s*)?(?:[=:]\s*)?(\d+)\b|\b(?:' + _ORD + r')\s+row\b', re.I),
    'col': re.compile(r'\bcolumn\s*(?:index\s*|number\s*)?(?:[=:]\s*)?(\d+)\b|\b(?:' + _ORD + r')\s+(?:column|figure|circle|position|one)\b', re.I),
}
_EXCLUDE_BEFORE = re.compile(r'(?:\bfrom|\bto|\bthrough|\bthru|[–-]|\bstart(?:ing|s)?\s+(?:at|from|with)|\brows|\bcolumns|\bbetween|\band|\bor|\(|\bas)\s*(?:the\s+)?$|\bstart\w*\b[^.\n]{0,30}$|\bcounting from [a-z ]+:[^.\n]{0,12}$', re.I)
_EXCLUDE_AFTER = re.compile(r'^\s*(?:is|=)\s+(?:the\s+)?(?:top|bottom|left|right|next|first|last|far)|^\s*(?:to|through|–|-)\s*(?:row|column)?\s*\d|^\s*,\s*\d|^\s+from the (?:bottom|right)|^\s*(?:is|was|being) the (?:leftmost|rightmost|topmost|bottommost)'
                            r'|^\s+(?:of|in)\s+(?:the\s+|that\s+|this\s+)?(?:\w+\s+)?(?:block|group|section|segment|cluster|half|part)', re.I)


def _claims(axis: str, body: str) -> list[tuple[int, int]]:
    """Statements of the target's row ('row') or column ('col') in the CoT body, in order.

    A mention counts as a claim unless it is an enumeration label at the start of a line ("Row 3: ..."),
    part of a range ("rows 1-4", "from row 1 to row 6"), a description of the coordinate system
    ("Row 1 is the topmost"), or an ordinal inside a sub-block ("second row of the lower block").
    """
    out = []
    for m in _MENTION[axis].finditer(body):
        v = int(m.group(1)) if m.group(1) else (int(m.group(2)) if m.group(2) else _ORD_WORDS.get(m.group(3).lower()))
        if v is None:
            continue
        line_start = body.rfind('\n', 0, m.start()) + 1
        prefix = body[line_start:m.start()]
        if prefix.strip(' -*•\t#>') in ('', '**') and re.match(r'\s*(?:\*\*)?\s*[:(]', body[m.end():m.end() + 4]):
            continue
        if _EXCLUDE_BEFORE.search(body[max(0, m.start() - 24):m.start()]) or _EXCLUDE_AFTER.search(body[m.end():m.end() + 40]):
            continue
        out.append((v, m.start()))
    return out


def declared_grid(body: str) -> tuple[int | None, int | None]:
    """First declared total of rows and columns in the CoT body."""
    for rx in GRID_RE:
        m = rx.search(body)
        if m:
            return _num(m.group(1)), _num(m.group(2))
    r = ROWS_DECL_RE.search(body)
    c = COLS_DECL_RE.search(body)
    rows = next((_num(g) for g in r.groups() if g), None) if r else None
    cols = next((_num(g) for g in c.groups() if g), None) if c else None
    return rows, cols


def enumeration(body: str) -> dict:
    """Row/column labels listed one per line ("Row 1: ...", "- Column 3"), in order of appearance."""
    rows, cols, red_rows, red_cols = [], [], [], []
    for m in ENUM_LINE_RE.finditer(body):
        axis, k, rest = m.group(1).lower(), int(m.group(2)), m.group(3)
        target = rows if axis == 'row' else cols
        target.append(k)
        if re.search(r'\bred\b', rest, re.I) and not re.search(r'\b(?:no|not)\s+red\b', rest, re.I):
            (red_rows if axis == 'row' else red_cols).append(k)
    return {'row_labels': rows, 'col_labels': cols, 'red_row_labels': red_rows, 'red_col_labels': red_cols,
            'row_enum': _is_run(rows), 'col_enum': _is_run(cols),
            'ellipsis': bool(re.search(r'\b(?:row|column)\s*\d+\s*(?:,\s*\d+\s*)*,?\s*(?:\.\.\.|…)', body, re.I))
            or bool(re.search(r'\b1\s*,\s*2\s*,\s*3\s*,?\s*(?:\.\.\.|…)', body))}


def _is_run(labels: list[int]) -> bool:
    """At least three consecutive labels 1, 2, 3 appear in order."""
    return any(labels[i:i + 3] == [1, 2, 3] for i in range(len(labels) - 2))


def arithmetic_slips(body: str) -> list[dict]:
    """'N figures before/left/above it' followed within 120 characters by an ordinal that is not N+1."""
    slips = []
    for m in PRECEDING_RE.finditer(body):
        n = _num(m.group(1))
        window = body[m.end():m.end() + 120]
        o = ORDINAL_AFTER_RE.search(window)
        if not o:
            continue
        val = int(o.group(1)) if o.group(1) else (_ORD_WORDS.get(o.group(2).lower()) if o.group(2) else int(o.group(3)))
        if val is not None and val != n + 1:
            slips.append({'preceding': n, 'stated': val, 'text': body[m.start():m.end() + 60]})
    return slips


def code_cot(task: str, text: str, finish_reason: str, truth: dict, answer_pos: str = 'last') -> dict:
    """Rule codes of one CoT response.

    task: 'rowcol', 'odd', 'point', 'count', 'binary' or 'babyvision'.
    truth: 1-based 'row'/'col', 'rows'/'cols', optional 'point' [x, y] (0-1000) or 'answer_yes'.
    Returned flags are None when the corresponding statement is absent from the text.
    """
    code: dict = {'n_chars': len(text), 'finish_reason': finish_reason}
    code['truncated'] = finish_reason != 'eos'
    code['disclaimer'] = bool(DISCLAIMER_RE.search(text))
    code['assume'] = bool(ASSUME_RE.search(text))
    code['n_self_correct'] = len(SELF_CORRECT_RE.findall(text))
    if task == 'babyvision':
        return code
    ans, pos = final_answer(task, text) if answer_pos == 'last' else (first_answer(task, text), len(text))
    body = text[:pos] if answer_pos == 'last' else text
    code['answer'] = list(ans) if isinstance(ans, tuple) else ans
    code['has_answer'] = ans is not None and not code['truncated']
    rows_d, cols_d = declared_grid(body)
    code['declared_rows'], code['declared_cols'] = rows_d, cols_d
    if truth.get('rows') is not None:
        code['grid_rows_ok'] = None if rows_d is None else rows_d == truth['rows']
        code['grid_cols_ok'] = None if cols_d is None else cols_d == truth['cols']
    en = enumeration(body)
    code.update({k: en[k] for k in ('row_enum', 'col_enum', 'ellipsis')})
    code['max_row_label'] = max(en['row_labels'], default=None)
    code['max_col_label'] = max(en['col_labels'], default=None)
    code['red_row_labels'], code['red_col_labels'] = en['red_row_labels'], en['red_col_labels']
    slips = arithmetic_slips(body)
    code['arith_slips'] = len(slips)
    if task == 'binary':
        code['correct'] = None if ans is None else (ans == 'Yes') == truth['answer_yes']
        return code
    if task in ('rowcol', 'odd', 'point'):
        rows_c = _claims('row', body)
        cols_c = _claims('col', body)
        code['row_claim'] = rows_c[-1][0] if rows_c else None
        code['col_claim'] = cols_c[-1][0] if cols_c else None
        code['first_row_claim'] = rows_c[0][0] if rows_c else None
        code['first_col_claim'] = cols_c[0][0] if cols_c else None
        code['n_row_claims_distinct'] = len({v for v, _ in rows_c})
        code['n_col_claims_distinct'] = len({v for v, _ in cols_c})
        # stated location exceeding the declared grid size is an internal inconsistency
        code['claim_exceeds_grid'] = bool((rows_d and code['row_claim'] and code['row_claim'] > rows_d)
                                          or (cols_d and code['col_claim'] and code['col_claim'] > cols_d))
    if task in ('rowcol', 'odd'):
        tr, tc = truth['row'], truth['col']
        code['correct'] = None if ans is None else (ans[0] == tr and ans[1] == tc)
        code['row_correct'] = None if ans is None else ans[0] == tr
        code['col_correct'] = None if ans is None else ans[1] == tc
        code['cot_row_ok'] = None if code['row_claim'] is None else code['row_claim'] == tr
        code['cot_col_ok'] = None if code['col_claim'] is None else code['col_claim'] == tc
        cons = []
        if ans is not None and code['row_claim'] is not None:
            cons.append(ans[0] == code['row_claim'])
        if ans is not None and code['col_claim'] is not None:
            cons.append(ans[1] == code['col_claim'])
        code['consistent'] = None if not cons else all(cons)
        if en['red_row_labels']:
            code['enum_red_row_ok'] = en['red_row_labels'][-1] == tr
            code['enum_red_row_eq_answer'] = None if ans is None else en['red_row_labels'][-1] == ans[0]
    elif task == 'point':
        rows, cols = truth['rows'], truth['cols']
        code['correct'] = None
        if ans is not None:
            ax, ay = ans
            code['err'] = math.hypot(ax - truth['point'][0], ay - truth['point'][1])
            # hit = the point falls inside the target cell (both 0-1000 conventions are equal here: square canvas)
            cell_c = int(min(cols - 1, max(0, ax / 1000 * cols))) + 1
            cell_r = int(min(rows - 1, max(0, ay / 1000 * rows))) + 1
            code['answer_cell'] = [cell_r, cell_c]
            code['correct'] = cell_r == truth['row'] and cell_c == truth['col']
            cons = []
            if code['row_claim'] is not None:
                cons.append(cell_r == code['row_claim'])
            if code['col_claim'] is not None:
                cons.append(cell_c == code['col_claim'])
            code['consistent'] = None if not cons else all(cons)
        code['cot_row_ok'] = None if code['row_claim'] is None else code['row_claim'] == truth['row']
        code['cot_col_ok'] = None if code['col_claim'] is None else code['col_claim'] == truth['col']
        code['mentions_grid_arith'] = bool(re.search(r'1000\s*/\s*\d+|\d+\s*/\s*\d+\s*[×x*]\s*1000|\(\s*\d+\s*-\s*0\.5\s*\)'
                                                     r'|\d+(?:\.\d+)?\s*[×x*]\s*\d+(?:\.\d+)?\s*=', body))
    elif task == 'count':
        code['correct'] = None if ans is None else (ans[0] == truth['rows'] and ans[1] == truth['cols'])
        code['row_count_correct'] = None if ans is None else ans[0] == truth['rows']
        code['col_count_correct'] = None if ans is None else ans[1] == truth['cols']
        cons = []
        if ans is not None and rows_d is not None:
            cons.append(ans[0] == rows_d)
        if ans is not None and cols_d is not None:
            cons.append(ans[1] == cols_d)
        code['consistent'] = None if not cons else all(cons)
        # enumerated label count vs the answer
        if ans is not None and code['max_row_label']:
            code['enum_rows_eq_answer'] = code['max_row_label'] == ans[0]
        if ans is not None and code['max_col_label']:
            code['enum_cols_eq_answer'] = code['max_col_label'] == ans[1]
    return code


FAILURE_MODES = ('truncated', 'disclaimer', 'inconsistent', 'grid_wrong', 'arith_slip', 'enum_mislabel',
                 'assert_wrong', 'no_claim')
REVERSE_MODES = ('cot_wrong_answer_right', 'disclaimer_but_right', 'grid_wrong_but_right', 'clean')


def primary_mode(code: dict) -> str:
    """Primary failure mode of a coded response (precedence follows the order of FAILURE_MODES).

    For correct responses the label is one of REVERSE_MODES.
    """
    correct = code.get('correct')
    grid_wrong = code.get('grid_rows_ok') is False or code.get('grid_cols_ok') is False
    cot_wrong = code.get('cot_row_ok') is False or code.get('cot_col_ok') is False
    if correct:
        if cot_wrong:
            return 'cot_wrong_answer_right'
        if code.get('disclaimer'):
            return 'disclaimer_but_right'
        if grid_wrong:
            return 'grid_wrong_but_right'
        return 'clean'
    if code.get('truncated') or not code.get('has_answer'):
        return 'truncated'
    if code.get('disclaimer'):
        return 'disclaimer'
    if code.get('consistent') is False:
        return 'inconsistent'
    if grid_wrong or code.get('claim_exceeds_grid'):
        return 'grid_wrong'
    if code.get('arith_slips'):
        return 'arith_slip'
    if cot_wrong:
        return 'enum_mislabel' if (code.get('row_enum') or code.get('col_enum')) else 'assert_wrong'
    return 'no_claim'


# ----------------------------------------------------------------------------- CoT editing (perturbation test)

def _claim_spans(task: str, text: str, axis: str) -> list[tuple[int, int, int]]:
    """Editable digit spans (start, end, value) of stated rows/columns (or declared totals for 'count')
    in the CoT body, in order; enumeration labels at line start ("Row 3: ...") are skipped."""
    ans, pos = final_answer(task, text)
    body = text[:pos]
    if task == 'count':
        rx = re.compile(r'\b(\d+)\s+rows\b' if axis == 'row' else r'\b(\d+)\s+columns\b', re.I)
    elif axis == 'row':
        rx = re.compile(r'\brow\s*(\d+)\b|\b(\d+)(?:st|nd|rd|th)\s+row\b', re.I)
    else:
        rx = re.compile(r'\bcolumn\s*(\d+)\b|\b(\d+)(?:st|nd|rd|th)\s+(?:column|figure|position)\b', re.I)
    out = []
    for m in rx.finditer(body):
        g = 1 if m.group(1) else 2
        line_start = body.rfind('\n', 0, m.start()) + 1
        prefix = body[line_start:m.start()].strip(' -*•\t')
        if task != 'count' and prefix in ('', '**') and re.match(r'\s*(?:\*\*)?\s*[:(]', body[m.end():m.end() + 4]):
            continue
        if task != 'count' and (_EXCLUDE_BEFORE.search(body[max(0, m.start() - 24):m.start()])
                                or _EXCLUDE_AFTER.search(body[m.end():m.end() + 40])):
            continue
        out.append((m.start(g), m.end(g), int(m.group(g))))
    return out


def locate_claim(task: str, text: str, axis: str) -> tuple[int, int, int] | None:
    """Span of the LAST stated row/column (or declared total for 'count') in the CoT body."""
    found = _claim_spans(task, text, axis)
    return found[-1] if found else None


def locate_first_claim(task: str, text: str, axis: str) -> tuple[int, int, int] | None:
    """Span of the FIRST stated row/column, if it precedes the last one (otherwise None)."""
    found = _claim_spans(task, text, axis)
    return found[0] if len(found) >= 2 else None


def edit_claim(text: str, span: tuple[int, int, int], delta: int) -> tuple[str, int]:
    """CoT prefix truncated right after the edited number, and the new value."""
    start, end, value = span
    new = value + delta
    return text[:start] + str(new), new
