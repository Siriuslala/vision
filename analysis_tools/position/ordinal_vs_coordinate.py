"""Experiment 3.15: row/column ordinals vs 0-1000 coordinate values -- what is shared and what differs.

Pure (model-free) helpers. GPU stages live in ordinal_vs_coordinate_run.py, statistics and figures in
ordinal_vs_coordinate_analysis.py.

Three tests are supported here:

* index prefix (grid point CoT): the CoT of experiment 1.7 is cut right after the line in which it first
  states the target's row AND column, before any arithmetic; the stated indices can be edited, and a forced
  answer cue is appended. Whether the coordinate follows the written indices or the image is measured with a
  per-axis follow weight (see `follow_weight`);
* graded count-state damage (direct and CoT answers): the Gated DeltaNet decay alpha = exp(g) at image tokens
  is weakened to alpha^s (s = 1 natural, s = 0 no decay) by editing the in_proj_a output (see `graded_a`);
* CoT content coding on natural images (RefCOCO): which kinds of statements a CoT for a point answer contains.

Rows and columns are 1-based; grid point answers are 0-1000 normalised (x, y).
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F

from analysis_tools.behavior import cot_mechanism as cm

# ----------------------------------------------------------------------------- index prefix

_ORD_SUFFIX = {1: 'st', 2: 'nd', 3: 'rd'}


def ordinal(n: int) -> str:
    """English ordinal of a positive integer (1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st ...)."""
    if 10 <= n % 100 <= 20:
        return f'{n}th'
    return f'{n}{_ORD_SUFFIX.get(n % 10, "th")}'


_WORDS = ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth')
_WORD_VALUE = {w: i + 1 for i, w in enumerate(_WORDS)}
_OBJ = r'(?:figures?|circles?|stick figures?|positions?|ones?|items?)'
_CLAIM = {
    'row': re.compile(r'\brow\s+(\d+)\b|\b(\d+)(?:st|nd|rd|th)\s+row\b|\b(' + '|'.join(_WORDS) + r')\s+row\b', re.I),
    'col': re.compile(r'\bcolumn\s+(\d+)\b|\b(\d+)(?:st|nd|rd|th)\s+(?:column|' + _OBJ + r')\b|\b('
                      + '|'.join(_WORDS) + r')\s+(?:column|' + _OBJ + r')\b', re.I),
}
# Text that makes a prefix unusable: arithmetic, a computed coordinate, or 0-based indices that an edit
# of the 1-based claims would leave inconsistent.
_REJECT = re.compile(r'=|%|≈|~\s*\d|\b[2-9]\d{2}\b|\b1000\b|0-based|zero-based|\brow 0\b|\bcolumn 0\b', re.I)
_ZERO_INDEX = re.compile(r'\bindex\s+(\d+)\b', re.I)


def claims(text: str, axis: str) -> list[tuple[int, int, int, str]]:
    """(start, end, value, kind) of target row ('row') or column ('col') statements; kind is 'num', 'ord'
    (4th) or 'word' (fourth). Enumeration labels ('Row 3: ...'), ranges and coordinate-system descriptions
    are skipped with the experiment-1.7 exclusion rules."""
    out = []
    for m in _CLAIM[axis].finditer(text):
        line_start = text.rfind('\n', 0, m.start()) + 1
        lead = text[line_start:m.start()].strip(' -*•\t')
        if lead in ('', '**') and re.match(r'\s*(?:\*\*)?\s*[:(]', text[m.end():m.end() + 4]):
            continue
        if cm._EXCLUDE_BEFORE.search(text[max(0, m.start() - 24):m.start()]) or cm._EXCLUDE_AFTER.search(text[m.end():m.end() + 40]):
            continue
        if m.group(1):
            out.append((m.start(1), m.end(1), int(m.group(1)), 'num'))
        elif m.group(2):
            out.append((m.start(2), m.end(2) + 2, int(m.group(2)), 'ord'))
        else:
            out.append((m.start(3), m.end(3), _WORD_VALUE[m.group(3).lower()], 'word'))
    return out


def index_prefix(text: str) -> dict | None:
    """Cut a point CoT right after the line that completes the first statement of both row and column.

    Returns {'prefix', 'row', 'col', 'spans'} where spans are (start, end, axis, kind) of all row/column
    claims inside the prefix, or None when the CoT never states both, or the prefix already contains
    arithmetic, a coordinate value or 0-based indices (the prefix must hold the indices and nothing else
    computed from them).
    """
    rows, cols = claims(text, 'row'), claims(text, 'col')
    if not rows or not cols:
        return None
    end = max(rows[0][1], cols[0][1])
    nl = text.find('\n', end)
    cut = len(text) if nl < 0 else nl
    prefix = text[:cut]
    if _REJECT.search(prefix):
        return None
    spans = [(s, e, 'row', k) for s, e, _, k in rows if e <= cut] + [(s, e, 'col', k) for s, e, _, k in cols if e <= cut]
    values = {(s, axis): v for (s, e, axis, k), v in zip(spans, [v for _, e, v, _ in rows if e <= cut]
                                                                   + [v for _, e, v, _ in cols if e <= cut])}
    # 0-based restatements ('5th row (index 4 if starting from 0)') must equal the preceding claim minus 1;
    # they are edited along with it ('zero' kind), otherwise the prefix is rejected.
    for m in _ZERO_INDEX.finditer(prefix):
        before = [(s, axis) for s, e, axis, k in spans if e <= m.start() and k != 'zero']
        if not before or int(m.group(1)) != values[max(before)] - 1:
            return None
        spans.append((m.start(1), m.end(1), max(before)[1], 'zero'))
    if 'from 0' in prefix.lower() and not any(k == 'zero' for *_, k in spans):
        return None
    spans = sorted(spans)
    return {'prefix': prefix, 'row': rows[0][2], 'col': cols[0][2], 'spans': spans}


def pre_index_prefix(text: str) -> str | None:
    """CoT cut at the start of the line holding the first row or column claim (no index written yet)."""
    rows, cols = claims(text, 'row'), claims(text, 'col')
    if not rows or not cols:
        return None
    first = min(rows[0][0], cols[0][0])
    return text[:text.rfind('\n', 0, first) + 1]


def edit_indices(info: dict, row: int, col: int) -> str:
    """Prefix with every stated row replaced by `row` and every stated column by `col`, keeping the
    surface form of each claim (digit, '4th' or 'fourth')."""
    text = info['prefix']
    for s, e, axis, kind in sorted(info['spans'], reverse=True):
        value = row if axis == 'row' else col
        if kind == 'num':
            new = str(value)
        elif kind == 'zero':
            new = str(value - 1)
        elif kind == 'ord':
            new = ordinal(value)
        else:
            if value > len(_WORDS):
                raise ValueError(f'no ordinal word for {value}')
            word = _WORDS[value - 1]
            new = word.capitalize() if text[s].isupper() else word
        text = text[:s] + new + text[e:]
    return text


TEMPLATE_PREFIX = ('The image is a grid of {rows} rows and {cols} columns. Scanning it row by row, the red figure '
                   'is in row {row} and column {col}, counting from 1 at the top-left.')


def template_prefix(rows: int, cols: int, row: int, col: int) -> str:
    """A fixed assistant sentence that states the grid size and the target's 1-based row and column."""
    return TEMPLATE_PREFIX.format(rows=rows, cols=cols, row=row, col=col)


def full_body(text: str) -> str | None:
    """CoT body before its final point answer (None if the CoT never gives one)."""
    ans, pos = cm.final_answer('point', text)
    if ans is None:
        return None
    body = text[:pos]
    # drop a trailing opening of the answer object, e.g. '[{' or '```json\n['
    return re.sub(r'[\s`\[{]*(?:json)?[\s`\[{]*$', '', body)


def cell_centre(index: int, n: int) -> float:
    """0-1000 centre of 1-based cell `index` on an axis with n equal cells."""
    return 1000.0 * (index - 0.5) / n


def follow_weight(answer: float, image_value: float, text_value: float) -> float | None:
    """Projection weight of the answer on the image -> text displacement along one axis.

    w = (answer - image) / (text - image): 0 means the answer equals the image-implied value, 1 means it
    equals the value implied by the written index. None when the two hypotheses coincide (< 50 units apart).
    """
    d = text_value - image_value
    if abs(d) < 50:
        return None
    return (answer - image_value) / d


# ----------------------------------------------------------------------------- graded decay edit

def graded_a(a: torch.Tensor, dt_bias: torch.Tensor, s: float) -> torch.Tensor:
    """in_proj_a output a' such that the decay becomes alpha' = alpha^s.

    Gated DeltaNet uses g = -exp(A_log) * softplus(a + dt_bias) and alpha = exp(g). Scaling softplus by s
    scales g by s, hence alpha' = alpha^s; s = 0 gives alpha' = 1 (no decay), s = 1 leaves a unchanged.
    a: (..., heads); dt_bias: (heads,).
    """
    if s == 1.0:
        return a
    sp = F.softplus(a.float() + dt_bias.float())
    if s == 0.0:
        return torch.full_like(a, -1e4)
    target = (sp * s).clamp_min(1e-12)
    inv = torch.where(target > 20, target, torch.log(torch.expm1(target)))
    return (inv - dt_bias.float()).to(a.dtype)


# ----------------------------------------------------------------------------- CoT content coding (natural images)

_NUM = re.compile(r'(?<![\w.])(\d{1,4})(?:\.\d+)?(?![\w.])')
_REGION = re.compile(r'\b(left|right|top|bottom|upper|lower|middle|cent(?:er|re)|corner|half|third|quarter|edge)\b', re.I)
_PIXEL = re.compile(r'\bpixels?\b|\bpx\b', re.I)
_PAIR = re.compile(r'[\[(]\s*\d{1,4}\s*,\s*\d{1,4}\s*(?:,\s*\d{1,4}\s*,\s*\d{1,4}\s*)?[\])]')


def code_natural_cot(body: str) -> dict:
    """Statement types in a natural-image point CoT body (before the final answer).

    region: spatial words (left, top, centre ...); numbers: integers 10-1000 written in the body;
    pair: an explicit coordinate pair or box; arithmetic: an '=' sign or a binary operator between numbers;
    pixel: pixel units mentioned.
    """
    nums = [int(m.group(1)) for m in _NUM.finditer(body) if 10 <= int(m.group(1)) <= 1000]
    return {'region': bool(_REGION.search(body)), 'n_numbers': len(nums), 'numbers': bool(nums),
            'pair': bool(_PAIR.search(body)), 'arithmetic': bool(re.search(r'=|\d\s*[*×/+−-]\s*\d', body)),
            'pixel': bool(_PIXEL.search(body))}


def last_pair(body: str) -> tuple[float, float] | None:
    """Last explicit (x, y) pair in a CoT body; for a 4-number box its centre."""
    found = list(_PAIR.finditer(body))
    if not found:
        return None
    vals = [float(v) for v in re.findall(r'\d{1,4}', found[-1].group(0))]
    if len(vals) == 4:
        return (vals[0] + vals[2]) / 2, (vals[1] + vals[3]) / 2
    return vals[0], vals[1]


def in_box(point, box) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= point[0] <= x1 and y0 <= point[1] <= y1


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
