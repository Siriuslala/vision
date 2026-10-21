"""Format-tolerant diagnostic scoring for BabyVision.

This module never replaces ``eva.benchmark.score``: strict scoring stays the primary metric and
its outputs are not modified. The diagnostic scorer answers one question only, namely how many
strict errors are pure format mismatches, e.g. "(10,5)" against the reference "10,5".

Every rule is conservative: both the reference and the prediction must parse into the same
typed answer (position, integer list, pair mapping, short item list or canonical string), and
the typed contents must be identical. No rule tolerates a different number, letter, pairing
or ordering of an ordered answer. Rules that change answer semantics (e.g. swapping (x, y))
are deliberately absent.

An item counts as diagnostically correct when it is strictly correct or when one of the rules
below accepts it, so the diagnostic correct set always contains the strict correct set.

Extraction rules (applied before matching; truncated responses are never rescued):
  E1 the last "Final answer: X" / "final answer is X" anywhere inside a line, not only at
     line start; an empty marker line takes the next nonempty line;
  E2 without any marker, a standalone last line of the form \\boxed{X};
  E3 wrappers around X are removed: markdown emphasis, quotes, $, \\boxed{}, a repeated
     "Final answer:" prefix and a trailing period.
"""
from __future__ import annotations

from collections import Counter
import re
import unicodedata

from eva.benchmark import normalize, score as strict_score

METRIC = 'diagnostic_format_tolerant'

# Unicode hyphens, dashes and minus signs that render like "-".
_DASHES = dict.fromkeys(map(ord, '\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe63\uff0d'), '-')
_ORDINALS = {w: i for i, w in enumerate(
    'first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth '
    'fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split(), start=1)}
_MARKER = re.compile(r'final\s+answer\s*(?::|：|\bis\b)\s*', re.I)
_BOXED = re.compile(r'\\boxed\{(.*)\}', re.S)
_ORD = r'(?:\d+(?:st|nd|rd|th)?|' + '|'.join(_ORDINALS) + r')'


def canonical(text: str) -> str:
    """NFKC plus dash canonicalization; whitespace is kept so that list separators survive."""
    return unicodedata.normalize('NFKC', str(text)).translate(_DASHES).strip()


def clean(text: str) -> str:
    """Remove presentation wrappers (E3) until a fixed point is reached."""
    text = canonical(text)
    while True:
        before = text
        text = text.strip().strip('*_`$').strip()
        text = re.sub(r'^["\'\u201c\u201d\u2018\u2019]+|["\'\u201c\u201d\u2018\u2019]+$', '', text).strip()
        text = _MARKER.sub('', text, count=1) if _MARKER.match(text) else text
        boxed = _BOXED.fullmatch(text)
        if boxed:
            text = boxed[1].replace('\\ ', ' ').replace('\\,', ' ')
        text = re.sub(r'^\\text\{(.*)\}$', r'\1', text)
        text = text.rstrip('.').strip()
        if text == before:
            return text


def extract_final(response: str, finish_reason: str | None = None) -> tuple[str | None, str]:
    """Return (candidate, extraction rule); candidate is None when nothing reliable exists."""
    if finish_reason == 'length':
        return None, 'truncated'
    final = response.split('</think>')[-1]
    lines = [line for line in final.splitlines() if line.strip()]
    for index in range(len(lines) - 1, -1, -1):
        found = list(_MARKER.finditer(lines[index]))
        if not found:
            continue
        candidate = clean(lines[index][found[-1].end():])
        if candidate:
            return candidate, 'marker'
        if index + 1 < len(lines):
            candidate = clean(lines[index + 1])
            return (candidate or None), 'marker_next_line'
        return None, 'marker_empty'
    if lines and _BOXED.fullmatch(lines[-1].strip().strip('*$').strip()):
        return clean(lines[-1]) or None, 'boxed'
    if len(lines) == 1:  # the strict scorer also accepts a single standalone line
        return clean(lines[0]) or None, 'single_line'
    return None, 'none'


# ----------------------------------------------------------------------------- typed parsers
def parse_int_list(text: str) -> tuple[int, ...] | None:
    """Integers separated by commas, semicolons, whitespace or "and"; optional outer brackets."""
    text = text.strip()
    if re.fullmatch(r'[(\[].*[)\]]', text):
        text = text[1:-1].strip()
    if not re.fullmatch(r'\d+(?:\s*(?:[,;]|\band\b|\s)\s*\d+)*', text, flags=re.I):
        return None
    return tuple(int(x) for x in re.findall(r'\d+', text))


def _ordinal(token: str) -> int:
    token = token.casefold()
    return _ORDINALS[token] if token in _ORDINALS else int(re.match(r'\d+', token)[0])


def parse_position(text: str, numeric: bool) -> tuple[int, int] | None:
    """(row, column) from a labelled phrase, or from a bare integer pair when ``numeric``.

    Labelled forms: "Row 1, Column 2", "Third row first column", "row 2 col 3". A bare pair
    "(a,b)", "a,b", "a-b" or "a b" is read as (row, column) and is only allowed when the
    question itself fixes the row-before-column convention (``numeric=True``).
    """
    text = text.strip().casefold()
    if numeric:
        pair = re.fullmatch(r'[(\[]?\s*(\d+)\s*(?:,|-|\s)\s*(\d+)\s*[)\]]?', text)
        if pair:
            return int(pair[1]), int(pair[2])
    found = {}
    rest = text
    for key, label in (('row', r'rows?'), ('col', r'(?:columns?|cols?)')):
        hits = list(re.finditer(rf'\b{label}\s*(?:no\.?|number)?\s*({_ORD})\b|\b({_ORD})\s+{label}\b', rest))
        if len(hits) != 1:
            return None
        found[key] = _ordinal(hits[0][1] or hits[0][2])
        rest = rest[:hits[0].start()] + ' ' + rest[hits[0].end():]
    if re.sub(r'\b(?:the|and|in|is|it)\b|[,;:()\s]', '', rest):
        return None  # leftover words could change the meaning
    return found['row'], found['col']


def _pair_element(text: str) -> str | None:
    text = re.sub(r'\s+', ' ', text.strip().casefold())
    return text if re.fullmatch(r'[\w#][\w# ]*', text) else None


def parse_pairs(text: str) -> Counter | None:
    """A mapping written as "X-Y" items; the listing order of the items carries no meaning.

    Items are separated by commas or semicolons, or by whitespace when every item is a compact
    "X-Y" token. Each pair keeps its orientation, except that a (letters, digits) pair is
    stored as (digits, letters): the element types already identify which side is which.
    """
    text = text.strip()
    items = re.split(r'\s*[,;]\s*', text)
    if len(items) == 1 and re.fullmatch(r'[^\s-]+-[^\s-]+(?:\s+[^\s-]+-[^\s-]+)+', text):
        items = text.split()
    pairs = []
    for item in items:
        if item.count('-') != 1:
            return None
        left, right = (_pair_element(x) for x in item.split('-'))
        if left is None or right is None:
            return None
        if re.fullmatch(r'[a-z]+', left) and re.fullmatch(r'\d+', right):
            left, right = right, left
        pairs.append((left, right))
    return Counter(pairs)


def parse_items(text: str) -> tuple[str, ...] | None:
    """Ordered single-character or integer items separated by commas, semicolons or spaces."""
    items = [x for x in re.split(r'[\s,;]+', text.strip().casefold()) if x]
    if len(items) < 2 or not all(re.fullmatch(r'[a-z0-9]|\d+', x) for x in items):
        return None
    return tuple(items)


def question_fixes_row_column(question: str) -> bool:
    """True when the question names rows before columns, which fixes how a bare pair is read."""
    return re.search(r'\brows?\b.*\bcolumns?\b', question, flags=re.I | re.S) is not None


def match_blank(reference: str, prediction: str, question: str) -> str | None:
    """Name of the first rule under which the two answers are identical, else None."""
    ref, pred = clean(reference), clean(prediction)
    if normalize(ref) == normalize(pred):
        return 'canonical'
    numeric = question_fixes_row_column(question)
    ref_pos = parse_position(ref, numeric)
    if ref_pos is not None and ref_pos == parse_position(pred, numeric):
        return 'position'
    ref_ints = parse_int_list(ref)
    if ref_ints is not None and ref_ints == parse_int_list(pred):
        return 'int_list'
    ref_pairs = parse_pairs(ref)
    if ref_pairs is not None and ref_pairs == parse_pairs(pred):
        return 'pair_set'
    ref_items = parse_items(ref)
    if ref_items is not None and ref_items == parse_items(pred):
        return 'item_list'
    return None


def parse_choice(candidate: str, n_options: int) -> str | None:
    match = re.fullmatch(r'(?:option\s*)?[(\[]?([A-Z])[)\]]?[.)]?', candidate.strip(), flags=re.I)
    if not match or ord(match[1].upper()) - 65 >= n_options:
        return None
    return match[1].upper()


def diagnostic_score(row: dict, response: str, finish_reason: str | None = None) -> dict:
    """Format-tolerant diagnostic score of one BabyVision (non-generation) answer.

    The diagnostic correct set is the strict correct set plus the answers accepted by the rules
    above, so an item that is strictly correct is never marked wrong here.
    """
    strict = strict_score(row, response, 'babyvision')
    candidate, extraction = extract_final(response, finish_reason)
    if row['ansType'] == 'choice':
        reference = chr(65 + row['choiceAns'])
        prediction = parse_choice(candidate, len(row['options'])) if candidate else None
        rule = 'choice' if prediction == reference else None
    else:
        reference = row['blankAns']
        prediction = candidate if candidate and '\n' not in candidate else None
        rule = match_blank(reference, prediction, row['question']) if prediction else None
    if strict['correct']:
        rule = 'strict'
    return {'prediction': prediction, 'reference': reference, 'correct': rule is not None,
            'rule': rule, 'extraction': extraction, 'strict_correct': strict['correct'],
            'metric': METRIC}
