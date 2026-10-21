"""CPU tests for the slot-order constructors and statistics of experiment 3.8."""
import unittest

import numpy as np

from analysis_tools.position import sequence_order as so
from analysis_tools.position import sequence_order_analysis as soa
from analysis_tools.position import sequence_order_run as sor

H = W = 16


class SlotOrderTest(unittest.TestCase):
    def assert_perm(self, order):
        self.assertEqual(sorted(order.tolist()), list(range(H * W)))

    def test_orders_are_permutations(self):
        T = so.cell_tokens((3, 2), (H, W))
        for order in (so.order_roll_rows(H, W, 2), so.order_roll_within_rows(H, W, -2), so.order_transpose(H, W),
                      so.order_reverse_rows(H, W), so.order_mirror_rows(H, W), so.order_shuffle_others(H, W, T, 0),
                      so.order_carry(H, W, (3, 2), 1, 4, 0), so.order_segment_shift(H, W, (3, 2), 1, 4)):
            self.assert_perm(order)

    def test_slot_cells(self):
        T = so.cell_tokens((3, 2), (H, W))
        self.assertEqual(so.slot_cell(so.order_identity(H, W), T, W), (3.0, 2.0))
        down = so.order_block_move(H, W, T, so.cell_tokens((4, 2), (H, W)))
        self.assertEqual(so.slot_cell(down, T, W), (4.0, 2.0))
        self.assertEqual(so.slot_cell(so.order_roll_rows(H, W, 2), T, W), (4.0, 2.0))
        self.assertEqual(so.slot_cell(so.order_roll_within_rows(H, W, 2), T, W), (3.0, 3.0))
        self.assertEqual(so.slot_cell(so.order_segment_shift(H, W, (3, 2), 1, 4), T, W), (3.0, 3.0))
        self.assertEqual(so.slot_cell(so.order_segment_shift(H, W, (3, 2), -1, 2), T, W), (3.0, 1.0))
        self.assertEqual(so.slot_cell(so.order_transpose(H, W), T, W), (2.0, 3.0))

    def test_carry_whole_rows_swaps_cell_rows(self):
        order = so.order_carry(H, W, (3, 2), 1, W, W).reshape(H, W)
        self.assertTrue(np.array_equal(order[6:8], np.arange(8 * W, 10 * W).reshape(2, W)))
        self.assertTrue(np.array_equal(order[8:10], np.arange(6 * W, 8 * W).reshape(2, W)))

    def test_shuffle_others_keeps_target(self):
        T = so.cell_tokens((5, 5), (H, W))
        order = so.order_shuffle_others(H, W, T, 3)
        self.assertEqual(order[T].tolist(), T)

    def test_edit_conditions(self):
        case = {'target': [3, 2], 'rows': 8, 'cols': 8}
        conds = {c['name']: c for c in sor.edit_conditions(case, H, W)}
        self.assertEqual(len(conds['dup_row_before']['index']), H * W + 2 * W)
        self.assertEqual(len(conds['delete_row_before']['index']), H * W - 2 * W)
        self.assertEqual(len(conds['dup_col_before']['index']), H * W + 4)
        self.assertEqual(len(conds['delete_col_before']['index']), H * W - 4)
        # The duplicated row lands right before the target's first token row.
        idx = conds['dup_row_before']['index']
        self.assertEqual(idx[6 * W:8 * W].tolist(), list(range(4 * W, 6 * W)))


class StatsTest(unittest.TestCase):
    def test_wilson(self):
        lo, hi = soa.wilson(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_follow(self):
        base = {'slot_cell': [3.0, 2.0], 'e_row': 3.0, 'e_col': 2.0}
        cond = {'slot_cell': [4.0, 2.0], 'e_row': 3.5, 'e_col': 2.0, 'axis': 'v'}
        self.assertAlmostEqual(soa.follow({}, cond, base)['f'], 0.5)


if __name__ == '__main__':
    unittest.main()
