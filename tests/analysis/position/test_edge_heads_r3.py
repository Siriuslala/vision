import unittest

import numpy as np

from analysis_tools.position import edge_heads_r3 as e3
from analysis_tools.position import edge_heads_r3_analysis as an


class TokenRegionTest(unittest.TestCase):
    def test_partition(self):
        grid = (16, 16)
        reg = e3.token_regions([96, 160, 192, 256], [192, 64, 288, 160], grid)
        parts = [set(reg[k].tolist()) for k in e3.REGIONS[:-1]]
        self.assertEqual(sum(len(p) for p in parts), 256)          # disjoint and covering
        self.assertEqual(set().union(*parts), set(range(256)))
        # target = tokens overlapping either box (columns 3..8, rows 2..7 for the union bounding rows/cols)
        self.assertIn(5 * 16 + 4, reg['target'])
        self.assertIn(2 * 16 + 7, reg['target'])
        # ring is adjacent to the target, row_right lies right of the target columns in target rows
        self.assertIn(5 * 16 + 2, reg['ring'])
        self.assertTrue(all(i % 16 > 8 for i in reg['row_right']))
        self.assertTrue(all(i % 16 < 3 for i in reg['row_left']))

    def test_edge_box_stays_in_grid(self):
        reg = e3.token_regions([0, 0, 64, 64], [0, 0, 64, 64], (8, 8))
        self.assertEqual(sorted(reg['target'].tolist()), [0, 1, 8, 9])
        self.assertEqual(len(reg['ring']), 5)


class RecoveryTest(unittest.TestCase):
    def test_r_of_and_filters(self):
        ld = {'dA': 3, 'dB': 4, 'LD_B': 6.0, 'ld': [-4.0, 1.0, 6.0]}
        self.assertAlmostEqual(an.r_of(ld, 1), 0.5)
        self.assertAlmostEqual(an.r_of(ld, 2), 1.0)
        self.assertIsNone(an.r_of({'dA': 3, 'dB': 3, 'LD_B': 6.0, 'ld': [0, 1]}, 1))
        self.assertIsNone(an.r_of({'dA': 3, 'dB': 4, 'LD_B': 1.0, 'ld': [0.0, 1]}, 1))
        self.assertAlmostEqual(an.r_diff(ld, 2, 1), 0.5)

    def test_moved_slopes_norm_convention(self):
        rec = {'side': 'right', 'format': 'point', 'canvas': [500, 500], 'box_a': [100, 100, 300, 300],
               'box_b': [100, 100, 400, 300],
               'gen': [{'cond': 'A', 'values': {'x': 400, 'y': 400}}, {'cond': 'B', 'values': {'x': 500, 'y': 400}},
                       {'cond': 'grp_moved', 'values': None}]}
        s = an.moved_slopes(rec)
        self.assertAlmostEqual(s['B'], 0.5)                          # 100 units = 50 px for a 100 px edge move
        self.assertIsNone(s['grp_moved'])
        self.assertEqual(s['_conv'], 'norm')


if __name__ == '__main__':
    unittest.main()
