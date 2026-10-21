"""Experiment 1.5: the mechanism of row/column ordinal answers (Qwen3.5-9B, thinking off).

Three candidate routes produce an ordinal o for the red target on an axis of instances:

  (a) coordinate route -- ground the target to a continuous coordinate, then convert it with an
      estimate of the instance count (ratio estimate over the occupied extent);
  (b) slot route       -- read the target's slot index on the regular grid (pixel position / cell);
  (c) count route      -- count the instances (columns or rows) that precede the target.

Stimuli are 6 x 21 (axis H, ordinal = column) or 21 x 6 (axis V, ordinal = row) slot grids. One
slot line (a whole column for H, a whole row for V) is left blank, which lets five paired variants
of every base image change one route's prediction while keeping the others:

  base   -- lines 1..20 present, slot 21 blank           count c, slot c,   pixel x
  shift  -- lines 2..21 present (blank at the near edge)  count c, slot c+1, pixel x + s
  insL   -- blank line b in 2..c, target slot c+1         count c, slot c+1, pixel x + s
  delL   -- lines 1..20 present except b in 2..c-1         count c-1, slot c, pixel x
  delR   -- lines 1..20 present except b in c+1..19        count c, slot c, pixel x

This module holds the stimuli, the cases, the answer parser, the route predictions and the
reasoning-strategy coder (no GPU). GPU stages live in `ordinal_mechanism_run.py`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import (
    JSON_ONLY, LOCATE_SUFFIX, T6_QUERY, ImageStore, read_jsonl, unit_seed, write_jsonl,
)
from analysis_tools.behavior.ordinal_token_grid import GridSpec, layout
from analysis_tools.common.stimuli import DEVELOPMENT_TEMPLATES, GENERATOR_VERSION, NONE, WHITE, Change, render_instance

GENERATOR = GENERATOR_VERSION + '+slot_lines'
RED = Change('color', 4)
SHORT = 6                     # instances on the short axis
SLOTS = 21                    # slot lines on the long axis (20 present in every variant except insL)
PRESENT = 20
VARIANTS = ('base', 'shift', 'insL', 'delL', 'delR')
C_RANGE = range(3, 19)        # target count ordinals that admit every variant (1-based)
MODES = ('direct', 'think')

DIRECT_PROMPT = T6_QUERY[0] + LOCATE_SUFFIX['row_column'] + JSON_ONLY
THINK_SUFFIX = (' First describe step by step how you determine its row and its column. Then give the final answer '
                'on the last line as {"row": r, "column": c}, counting rows from top to bottom and columns from left '
                'to right, starting at 1.')
THINK_PROMPT = T6_QUERY[0] + THINK_SUFFIX


# ----------------------------------------------------------------------------- stimuli

@dataclass(frozen=True)
class SlotGrid:
    """Grid of SLOTS x SHORT slots along the long axis with some slot lines left blank.

    axis:    'H' -> the long axis is horizontal (21 columns, 6 rows), the probed ordinal is the column;
             'V' -> the long axis is vertical (21 rows, 6 columns), the probed ordinal is the row.
    present: sorted 0-based indices of the present slot lines on the long axis.
    target:  (long-axis slot, short-axis index), both 0-based; the target slot must be present.
    """
    template: str
    axis: str
    cell: int
    present: tuple[int, ...]
    target: tuple[int, int]

    def __post_init__(self):
        if self.axis not in ('H', 'V'):
            raise ValueError(f'axis must be H or V, got {self.axis!r}')
        if self.target[0] not in self.present or not 0 <= self.target[1] < SHORT:
            raise ValueError(f'target {self.target} is not on a present slot line')
        if any(not 0 <= p < SLOTS for p in self.present) or list(self.present) != sorted(set(self.present)):
            raise ValueError('present must be sorted unique slot indices in [0, SLOTS)')

    @property
    def shape(self) -> tuple[int, int]:
        return (SHORT, SLOTS) if self.axis == 'H' else (SLOTS, SHORT)

    @property
    def target_rc(self) -> tuple[int, int]:
        """0-based (row, col) slot of the target."""
        g, o = self.target
        return (o, g) if self.axis == 'H' else (g, o)


def render(spec: SlotGrid) -> tuple[Image.Image, dict]:
    rows, cols = spec.shape
    width, height, ox, oy = layout(GridSpec(spec.template, rows, cols, spec.cell, 0, spec.target_rc))
    canvas = np.full((height, width, 3), WHITE, np.uint8)
    tiles = {c: render_instance(spec.template, c, spec.cell) for c in (NONE, RED)}
    present = set(spec.present)
    boxes = []
    for r in range(rows):
        for c in range(cols):
            if (c if spec.axis == 'H' else r) not in present:
                continue
            is_target = spec.target_rc == (r, c)
            x0, y0 = ox + c * spec.cell, oy + r * spec.cell
            region = canvas[y0:y0 + spec.cell, x0:x0 + spec.cell]
            np.minimum(region, tiles[RED if is_target else NONE], out=region)
            if is_target:
                boxes = [x0, y0, x0 + spec.cell, y0 + spec.cell]
    image = Image.fromarray(canvas)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    meta = {'generator': GENERATOR, 'canvas': [width, height], 'grid_origin': [ox, oy],
            'merged_grid': [height // 32, width // 32], 'target_cell_box': boxes,
            'sha256': hashlib.sha256(buf.getvalue()).hexdigest()}
    return image, meta


def variant_layout(variant: str, c: int, rng: np.random.Generator) -> tuple[tuple[int, ...], int, int | None]:
    """Present slot lines, the 0-based target slot and the blank interior slot for count ordinal c (1-based)."""
    if variant == 'base':
        return tuple(range(PRESENT)), c - 1, None
    if variant == 'shift':
        return tuple(range(1, SLOTS)), c, None
    if variant == 'insL':
        b = int(rng.integers(1, c))                    # 0-based blank in [1, c-1] -> 1-based 2..c
        return tuple(i for i in range(SLOTS) if i != b), c, b
    if variant == 'delL':
        b = int(rng.integers(1, c - 1))                # 1-based 2..c-1
        return tuple(i for i in range(PRESENT) if i != b), c - 1, b
    if variant == 'delR':
        b = int(rng.integers(c, PRESENT - 1))          # 1-based c+1..19
        return tuple(i for i in range(PRESENT) if i != b), c - 1, b
    raise ValueError(f'unknown variant {variant!r}')


def route_predictions(present: tuple[int, ...], target_slot: int) -> dict:
    """Ordinal (1-based) predicted by each route for one image.

    count: number of present lines up to and including the target.
    slot:  1-based slot index on the full 21-slot grid (pixel position divided by the cell size).
    ratio: position inside the occupied extent rescaled by the present count,
           1 + (N - 1) (g - g_first) / (g_last - g_first), kept continuous.
    gap:   count restarted after the nearest interior blank line that precedes the target
           (equals count when no interior blank precedes the target).
    """
    p = list(present)
    n = len(p)
    ratio = 1 + (n - 1) * (target_slot - p[0]) / (p[-1] - p[0])
    holes = [i for i in range(p[0], target_slot) if i not in present]
    gap = sum(1 for i in p if holes[-1] < i <= target_slot) if holes else p.index(target_slot) + 1
    return {'count': p.index(target_slot) + 1, 'slot': target_slot + 1, 'ratio': ratio, 'gap': gap, 'n_present': n}


def axis_unit(meta: dict, spec: SlotGrid) -> float:
    """Target centre on the long axis in 0-1000 normalized units."""
    x0, y0, x1, y1 = meta['target_cell_box']
    w, h = meta['canvas']
    return ((x0 + x1) / 2 / w if spec.axis == 'H' else (y0 + y1) / 2 / h) * 1000


# ----------------------------------------------------------------------------- cases

CONDITION = re.compile(r'([HV])(\d+)')


def parse_condition(name: str) -> tuple[str, int]:
    m = CONDITION.fullmatch(name)
    if not m:
        raise ValueError(f'Bad condition {name!r}; expected H<cell> or V<cell>')
    return m.group(1), int(m.group(2))


def build(output: Path, cfg: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    store = ImageStore(output)
    cases = []
    for name in cfg['conditions']:
        axis, cell = parse_condition(name)
        rng = np.random.default_rng(unit_seed(cfg['seed'], 'ordinal_mechanism', name))
        bases = [c for c in C_RANGE for _ in range(cfg['bases_per_ordinal'])]
        think_left = {c: cfg['think_per_ordinal'] if name in cfg['think_conditions'] else 0 for c in C_RANGE}
        for b, c in enumerate(bases):
            other = int(rng.integers(0, SHORT))
            template = DEVELOPMENT_TEMPLATES[b % len(DEVELOPMENT_TEMPLATES)]
            think = think_left[c] > 0
            think_left[c] -= 1
            for variant in VARIANTS:
                present, slot, blank = variant_layout(variant, c, rng)
                spec = SlotGrid(template, axis, cell, present, (slot, other))
                img, meta = render(spec)
                stem = f'{name}_b{b:03d}_{variant}'
                rec = store.save(img, f'images/{name}/{stem}.png', role='T6', spec={
                    'template': template, 'axis': axis, 'cell': cell, 'present': list(present),
                    'target': [slot, other]})
                routes = route_predictions(present, slot)
                truth = {'canvas': meta['canvas'], 'merged_grid': meta['merged_grid'], 'cell': cell, 'axis': axis,
                         'present': list(present), 'blank': blank, 'target_slot': slot, 'other': other + 1,
                         'target_cell_box': meta['target_cell_box'], 'unit': axis_unit(meta, spec), **routes}
                for mode in MODES if think else ('direct',):
                    cases.append({'id': f'{stem}|{mode}', 'condition': name, 'axis': axis, 'cell': cell, 'base': b,
                                  'c': c, 'variant': variant, 'mode': mode, 'template': template,
                                  'prompt': DIRECT_PROMPT if mode == 'direct' else THINK_PROMPT,
                                  'image': rec['path'], 'image_sha256': rec['sha256'], 'truth': truth})
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case identifiers.')
    path = output / 'cases.jsonl'
    if path.exists() and read_jsonl(path) != cases:
        raise ValueError('Existing cases.jsonl differs from the regenerated cases.')
    write_jsonl(path, cases)
    write_jsonl(output / 'images.jsonl', store.records)
    (output / 'config.json').write_text(json.dumps({'generator': GENERATOR, 'build': cfg, 'n_cases': len(cases)},
                                                   indent=2) + '\n')
    print(len(cases), 'cases', len(store.records), 'images')


# ----------------------------------------------------------------------------- answers

ROWCOL = re.compile(r'"row"\s*:\s*(\d+)\s*,\s*"column"\s*:\s*(\d+)')


def parse_answer(text: str) -> tuple[int, int] | None:
    """Last {"row": r, "column": c} pair in the response (1-based), or None."""
    found = ROWCOL.findall(text)
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def probed_answer(case: dict, rc: tuple[int, int] | None) -> tuple[int | None, int | None]:
    """(long-axis ordinal, short-axis ordinal) of an answer for the case's axis."""
    if rc is None:
        return None, None
    return (rc[1], rc[0]) if case['axis'] == 'H' else (rc[0], rc[1])


def score(case: dict, record: dict) -> dict:
    rc = parse_answer(record['response']) if record['finish_reason'] == 'eos' else None
    long_, short = probed_answer(case, rc)
    t = case['truth']
    out = {'id': case['id'], 'valid': rc is not None, 'answer': long_, 'answer_other': short,
           'other_correct': None if short is None else short == t['other']}
    for route in ('count', 'slot', 'gap'):
        out[f'match_{route}'] = None if long_ is None else long_ == t[route]
    out['match_ratio'] = None if long_ is None else long_ == int(np.floor(t['ratio'] + 0.5))
    return out


# ----------------------------------------------------------------------------- reasoning strategies

_NUM = r'(?:\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)'
STRATEGY_PATTERNS = {
    # explicit enumeration of consecutive positions: "1, 2, 3", "column 1 ... column 2 ... column 3"
    'enumerate': re.compile(r'\b1\s*,\s*2\s*,\s*3\b|(?:column|row)\s*1\b.*?(?:column|row)\s*2\b.*?(?:column|row)\s*3\b',
                            re.I | re.S),
    # counting from the far end and subtracting, e.g. "3rd from the right"
    'from_far_end': re.compile(r'from the (?:right|bottom|end)|counting (?:from the )?(?:right|bottom)', re.I),
    # an estimate of position as a fraction or a coordinate
    'fraction_or_coordinate': re.compile(r'\b(?:halfway|half of|percent|fraction|proportion|pixels?)\b|%|'
                                         r'\b[xy]-?coordinate|\b[xy]\s*[=≈]', re.I),
    # a stated total number of rows/columns
    'total': re.compile(r'\b\d+\s+(?:columns|rows)\b|\b(?:columns|rows)\s*:\s*\d+', re.I),
    # counting the instances that precede the target ("there are 7 columns to the left")
    'preceding': re.compile(r'\b\d+\s+(?:columns?|rows?|figures?)\s+(?:to the left|to its left|above|before)', re.I),
    # reference to the blank gap
    'gap': re.compile(r'\b(?:gap|empty|blank|missing|space)\b', re.I),
}


def code_strategies(text: str, answer: int | None = None) -> dict[str, bool]:
    """Rule-based strategy flags of a think-then-answer response (reasoning part only).

    full_enumeration: every integer 1..answer (the long-axis ordinal) appears in the reasoning and no ellipsis
    abbreviates the list, i.e. the response actually walks through the instances up to the target. A templated
    list such as "Column 1, 2, 3, ..., 20" sets `enumerate` but not `full_enumeration`.
    """
    body = text[:ROWCOL.search(text).start()] if ROWCOL.search(text) else text
    out = {k: bool(p.search(body)) for k, p in STRATEGY_PATTERNS.items()}
    if answer is not None:
        numbers = {int(x) for x in re.findall(r'\b(\d+)(?:st|nd|rd|th)?\b', body)}
        out['full_enumeration'] = answer >= 2 and set(range(1, answer + 1)) <= numbers \
            and '...' not in body and '…' not in body
    return out


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('command', choices=['build'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--conditions', nargs='+', required=True)
    ap.add_argument('--think-conditions', nargs='*', default=[])
    ap.add_argument('--bases-per-ordinal', type=int, default=4)
    ap.add_argument('--think-per-ordinal', type=int, default=2)
    a = ap.parse_args()
    build(a.output, {'seed': a.seed, 'conditions': a.conditions, 'think_conditions': a.think_conditions,
                     'bases_per_ordinal': a.bases_per_ordinal, 'think_per_ordinal': a.think_per_ordinal})


if __name__ == '__main__':
    main()
