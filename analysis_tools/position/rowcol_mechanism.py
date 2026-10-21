"""Experiment 3.10: one mechanism for row and column ordinals in Qwen3.5-9B (thinking off by default).

Engine (no model loading here; GPU stages live in `rowcol_mechanism_run.py`):

* slot-grid geometry of the experiment-1.5 stimuli (6 x 21 'H' grids whose long axis is horizontal and
  21 x 6 'V' grids whose long axis is vertical): which merged tokens a slot cell covers, the running count
  of instance lines up to every cell, the count restarted after the nearest interior blank line, and the
  token positions that open a new run of figures after a blank line ("boundary tokens");
* prompts: the experiment-1.5 direct and think prompts and a direct prompt with the output order swapped
  (column first);
* answer parsing that does not depend on the order of the "row" and "column" keys;
* head sets of the Gated DeltaNet layers taken from the experiment-3.8 gate classes.

Layers are 1-based. Token indices are relative to the image span (raster order of the merged grid) unless
a name says otherwise. The shared modules and the 1.5 / 3.8 engines are used unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from analysis_tools.behavior import ordinal_mechanism as om
from analysis_tools.behavior.decomposition import JSON_ONLY, T6_QUERY
from analysis_tools.behavior.ordinal_token_grid import GridSpec, layout

TOKEN = 32                                   # pixels per merged token
DIRECT_PROMPT = om.DIRECT_PROMPT             # {"row": r, "column": c}
THINK_PROMPT = om.THINK_PROMPT
SWAPPED_PROMPT = (T6_QUERY[0] + ' Give its position. Return only {"column": c, "row": r}, counting columns from left '
                  'to right and rows from top to bottom, starting at 1.' + JSON_ONLY)
PROMPTS = {'direct': DIRECT_PROMPT, 'think': THINK_PROMPT, 'swapped': SWAPPED_PROMPT}
ROW_KEY = re.compile(r'"row"\s*:\s*(\d+)')
COL_KEY = re.compile(r'"column"\s*:\s*(\d+)')
LINEAR_LAYERS = tuple(i for i in range(1, 33) if i % 4)          # the 24 Gated DeltaNet layers
MID_LINEAR = (9, 10, 11, 13, 14, 15)
CLASS_NAMES = {0: 'other', 1: 'integrator', 2: 'row_reset', 3: 'onset'}


# ----------------------------------------------------------------------------- answers

def parse_rowcol_any(text: str) -> tuple[int, int] | None:
    """(row, column), 1-based, from the last JSON object holding both keys in either order."""
    objs = re.findall(r'\{[^{}]*\}', text)
    for obj in reversed(objs):
        r, c = ROW_KEY.search(obj), COL_KEY.search(obj)
        if r and c:
            return int(r.group(1)), int(c.group(1))
    return None


def long_short(axis: str, rc: tuple[int, int] | None) -> tuple[int | None, int | None]:
    """(long-axis ordinal, short-axis ordinal) of an answer: H probes the column, V the row."""
    if rc is None:
        return None, None
    return (rc[1], rc[0]) if axis == 'H' else (rc[0], rc[1])


# ----------------------------------------------------------------------------- slot-grid geometry

@dataclass(frozen=True)
class SlotGeometry:
    """Merged-token geometry of one experiment-1.5 slot grid.

    axis:    'H' (21 columns on the long axis) or 'V' (21 rows on the long axis).
    cell:    cell size in pixels (32 or 64 here; a cell covers (cell/32)^2 merged tokens).
    present: sorted 0-based long-axis slot lines that hold instances.
    target:  (long-axis slot, short-axis index) of the red instance, 0-based.
    grid:    merged token grid (rows, cols); origin: (row, col) of the first grid token.
    """
    axis: str
    cell: int
    present: tuple[int, ...]
    target: tuple[int, int]
    grid: tuple[int, int]
    origin: tuple[int, int]

    @property
    def k(self) -> int:
        """Tokens per cell side."""
        return self.cell // TOKEN

    def cell_tokens(self, slot: int, other: int) -> list[int]:
        """Token indices covered by the cell at long-axis slot `slot` and short-axis index `other`."""
        r, c = (other, slot) if self.axis == 'H' else (slot, other)
        r0, c0 = self.origin[0] + r * self.k, self.origin[1] + c * self.k
        return [(r0 + i) * self.grid[1] + c0 + j for i in range(self.k) for j in range(self.k)]

    def running_count(self, slot: int) -> int:
        """Number of present lines with index <= slot (the count a left-to-right / top-to-bottom reader holds)."""
        return int(sum(1 for p in self.present if p <= slot))

    def count_since_gap(self, slot: int) -> int:
        """Running count restarted after the nearest interior blank line before `slot` (1.5's k_gap rule)."""
        p = list(self.present)
        holes = [i for i in range(p[0], slot) if i not in self.present]
        if not holes:
            return self.running_count(slot)
        return int(sum(1 for i in p if holes[-1] < i <= slot))

    def interior_blanks(self) -> list[int]:
        p = list(self.present)
        return [i for i in range(p[0], p[-1]) if i not in self.present]

    def boundary_tokens(self) -> np.ndarray:
        """Tokens that open a new run of figures after an interior blank line, in raster order.

        H: in every token row of the grid, the first token column of the line after the blank (the first
        figure token met after the blank inside each token row). V: the whole first token row of the line
        after the blank (raster order meets it as a new token row whose predecessor rows are blank).
        """
        out = []
        H, W = self.grid
        for b in self.interior_blanks():
            if self.axis == 'H':
                col = self.origin[1] + (b + 1) * self.k
                rows = range(self.origin[0], self.origin[0] + om.SHORT * self.k)
                out += [r * W + col for r in rows]
            else:
                row = self.origin[0] + (b + 1) * self.k
                cols = range(self.origin[1], self.origin[1] + om.SHORT * self.k)
                out += [row * W + c for c in cols]
        return np.array(sorted(out), dtype=int)

    def matched_control_tokens(self, shift: int = 3) -> np.ndarray:
        """Same layout as boundary_tokens, moved `shift` present lines further along the long axis (no blank)."""
        out = []
        H, W = self.grid
        for b in self.interior_blanks():
            line = b + 1 + shift
            if line >= om.SLOTS or line not in self.present:
                continue
            if self.axis == 'H':
                col = self.origin[1] + line * self.k
                out += [r * W + col for r in range(self.origin[0], self.origin[0] + om.SHORT * self.k)]
            else:
                row = self.origin[0] + line * self.k
                out += [row * W + c for c in range(self.origin[1], self.origin[1] + om.SHORT * self.k)]
        return np.array(sorted(out), dtype=int)


def geometry(truth: dict, axis: str) -> SlotGeometry:
    """SlotGeometry from a 1.5 case's truth dict (canvas, merged grid, present lines, target)."""
    rows, cols = (om.SHORT, om.SLOTS) if axis == 'H' else (om.SLOTS, om.SHORT)
    width, height, ox, oy = layout(GridSpec('person', rows, cols, truth['cell'], 0, (0, 0)))
    if [width, height] != list(truth['canvas']):
        raise ValueError(f"canvas {truth['canvas']} does not match the layout rule ({width}, {height})")
    if ox % TOKEN or oy % TOKEN:
        raise ValueError('grid origin is not token-aligned')
    box = truth['target_cell_box']
    other = (box[1] - oy) // truth['cell'] if axis == 'H' else (box[0] - ox) // truth['cell']
    return SlotGeometry(axis, truth['cell'], tuple(truth['present']), (truth['target_slot'], int(other)),
                        tuple(truth['merged_grid']), (oy // TOKEN, ox // TOKEN))


# ----------------------------------------------------------------------------- 1.5 cases

def load_ordinal_cases(run: Path, conditions: tuple[str, ...], mode: str = 'direct') -> list[dict]:
    """Experiment-1.5 cases of the given conditions and prompt mode, with the model's direct answer attached."""
    cases = [c for c in om.read_jsonl(run / 'cases.jsonl') if c['mode'] == mode and c['condition'] in conditions]
    answers = {r['id']: r for r in om.read_jsonl(run / 'gen' / f'results_{mode}.jsonl')}
    for c in cases:
        r = answers.get(c['id'])
        rc = parse_rowcol_any(r['response']) if r is not None and r['finish_reason'] == 'eos' else None
        c['answer_long'], c['answer_short'] = long_short(c['axis'], rc)
    return cases


def head_sets(gate_stats_path: Path) -> dict[str, dict[int, list[int]]]:
    """Per experiment-3.8 class: {1-based DeltaNet layer: [heads]} over all 24 DeltaNet layers."""
    g = json.loads(gate_stats_path.read_text())
    lab = np.array(g['classes'])
    out = {}
    for code, name in CLASS_NAMES.items():
        out[name] = {int(l): [int(h) for h in np.where(lab[i] == code)[0]] for i, l in enumerate(g['layers'])}
    return out


def restrict(sets: dict[int, list[int]], layers) -> dict[int, list[int]]:
    return {l: h for l, h in sets.items() if l in layers and h}


# ----------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))
