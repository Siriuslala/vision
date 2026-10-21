"""CPU tests of the experiment-3.10 engine (geometry, counts, boundaries, parsing, statistics)."""
import unittest

import numpy as np

from analysis_tools.behavior import ordinal_mechanism as om
from analysis_tools.position import rowcol_mechanism as rm
from analysis_tools.position import rowcol_mechanism_analysis as ra


def h_geo(present, target=(5, 2), cell=64):
    return rm.SlotGeometry('H', cell, tuple(present), target, (12, 42), (0, 0))


def test_parse_rowcol_any_order_free():
    assert rm.parse_rowcol_any('{"row": 3, "column": 7}') == (3, 7)
    assert rm.parse_rowcol_any('```json\n{"column": 7, "row": 3}\n```') == (3, 7)
    assert rm.parse_rowcol_any('first {"row": 1, "column": 1} then {"column": 9, "row": 4}') == (4, 9)
    assert rm.parse_rowcol_any('row 3') is None


def test_long_short():
    assert rm.long_short('H', (2, 9)) == (9, 2)
    assert rm.long_short('V', (2, 9)) == (2, 9)
    assert rm.long_short('V', None) == (None, None)


def test_counts_match_route_predictions():
    rng = np.random.default_rng(0)
    for variant in om.VARIANTS:
        for c in range(3, 19):
            present, slot, _ = om.variant_layout(variant, c, rng)
            g = h_geo(present, (slot, 0))
            rp = om.route_predictions(present, slot)
            assert g.running_count(slot) == rp['count']
            assert g.count_since_gap(slot) == rp['gap']


def test_cell_tokens_and_boundaries_h():
    present = tuple(i for i in range(21) if i != 4)          # blank line 4 (insL-like)
    g = h_geo(present, (6, 1))
    assert g.cell_tokens(6, 1) == [2 * 42 + 12, 2 * 42 + 13, 3 * 42 + 12, 3 * 42 + 13]
    b = g.boundary_tokens()
    assert len(b) == 12 and set(b % 42) == {10} and list(b // 42) == list(range(12))
    ctrl = g.matched_control_tokens(shift=3)
    assert set(ctrl % 42) == {16}


def test_boundaries_v():
    present = tuple(i for i in range(20) if i != 7)
    g = rm.SlotGeometry('V', 64, present, (10, 2), (42, 12), (0, 0))
    b = g.boundary_tokens()
    assert len(b) == 12 and set(b // 12) == {16}
    assert g.cell_tokens(10, 2) == [20 * 12 + 4, 20 * 12 + 5, 21 * 12 + 4, 21 * 12 + 5]


def test_no_boundary_at_edge_blank():
    g = h_geo(tuple(range(1, 21)), (5, 0))                    # shift variant: blank at the near edge only
    assert g.interior_blanks() == [] and len(g.boundary_tokens()) == 0


def test_geometry_from_truth_rejects_mismatch():
    truth = {'cell': 64, 'canvas': [1344, 384], 'merged_grid': [12, 42], 'present': list(range(20)),
             'target_slot': 4, 'target_cell_box': [256, 128, 320, 192]}
    g = rm.geometry(truth, 'H')
    assert g.target == (4, 2) and g.origin == (0, 0)
    try:
        rm.geometry({**truth, 'canvas': [1000, 384]}, 'H')
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched canvas accepted')


def test_statistics():
    lo, hi = rm.wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert rm.mcnemar_exact(0, 0) == 1.0
    assert rm.mcnemar_exact(0, 10) < 0.01


def test_select_heads_fallback():
    a_c = np.full((24, 32), 0.9)
    a_bh = a_c.copy()
    a_bh[6, 4] = 0.3
    st = {'H64': {'a_b': a_bh, 'a_c': a_c}, 'V64': {'a_b': a_c.copy(), 'a_c': a_c}}
    sel = ra.select_heads(st)
    assert sel['H']['heads'][rm.LINEAR_LAYERS[6]] == [4] and sel['H']['n'] == 1
    assert sel['V']['fallback'] and sel['V']['n'] == 1


class RowColMechanismTest(unittest.TestCase):
    def test_parse_rowcol_any_order_free(self):
        test_parse_rowcol_any_order_free()

    def test_long_short(self):
        test_long_short()

    def test_counts_match_route_predictions(self):
        test_counts_match_route_predictions()

    def test_cell_tokens_and_boundaries_h(self):
        test_cell_tokens_and_boundaries_h()

    def test_boundaries_v(self):
        test_boundaries_v()

    def test_no_boundary_at_edge_blank(self):
        test_no_boundary_at_edge_blank()

    def test_geometry_from_truth_rejects_mismatch(self):
        test_geometry_from_truth_rejects_mismatch()

    def test_statistics(self):
        test_statistics()

    def test_select_heads_fallback(self):
        test_select_heads_fallback()


if __name__ == '__main__':
    unittest.main()
