"""CPU tests of experiment 3.14 round 2 (reading heads, target-centred maps, accumulator simulation)."""
import unittest

import numpy as np

from analysis_tools.position import causal_vs_linear_heads_analysis as ha


def value(key, axis, ld_none, ld_b, lds, da=1, db=2):
    return {'key': key, 'axis': axis, 'dA': da, 'dB': db, 'LD_B': ld_b, 'LD': {'none': ld_none, **lds}}


class TestHeadRecovery(unittest.TestCase):
    def test_recovery_and_axis_filter(self):
        rec = {'cond': 'c', 'values_meta': [{'key': 'row'}, {'key': 'column'}],
               'pairs': [{'axis': 'v', 'values': [value('row', 'v', 0.0, 10.0, {'h20_11': 4.0}),
                                                  value('column', 'h', 0.0, 10.0, {'h20_11': 9.0})]}]}
        av = ha.axis_values(rec)
        self.assertEqual([(a, k) for a, k, _ in av], [('v', 0)])     # the column value of a vertical move is dropped
        self.assertAlmostEqual(ha.recovery(av[0][2], 'h20_11'), 0.4)
        hs = ha.head_summary([rec])
        self.assertAlmostEqual(hs['c|v']['h20_11']['mean'], 0.4)

    def test_recovery_exclusions(self):
        self.assertIsNone(ha.recovery(value('x', 'h', 0.0, 1.5, {'a': 1.0}), 'a'))       # |LD_B - LD_A| < 2
        self.assertIsNone(ha.recovery(value('x', 'h', 0.0, 9.0, {'a': 1.0}, 3, 3), 'a'))  # same first digit

    def test_reading_heads_ranked_singles_only(self):
        hs = {'k': {'h4_1': {'mean': 0.1}, 'h20_11': {'mean': 0.5}, 'L20': {'mean': 0.9}, 'full_all': {'mean': 1.0}}}
        self.assertEqual(ha.reading_heads(hs, 'k', 5), [('h20_11', 0.5), ('h4_1', 0.1)])


class TestMaps(unittest.TestCase):
    def test_centred_map_places_target_at_origin(self):
        grid = (4, 5)
        a = np.zeros(20)
        a[7] = 3.0                                                   # row 1, col 2
        a[0] = 1.0
        m = ha.centred_map(a, grid, [7, 8, 12, 13])
        self.assertAlmostEqual(m[ha.HALF, ha.HALF], 0.75)            # normalised to image mass 1
        self.assertAlmostEqual(m[ha.HALF - 1, ha.HALF - 2], 0.25)
        self.assertTrue(np.isnan(m[0, 0]))                           # outside the image


class TestSimulation(unittest.TestCase):
    def test_flag_only_after_target_in_its_row(self):
        sim = ha.simulate_accumulators(H=4, W=5, target=(2, 2), beta=0.5, alpha_reset=0.0)
        flag = sim['target-in-row flag']
        self.assertTrue(np.all(flag[2, 2:] > 0))
        self.assertTrue(np.all(flag[2, :2] == 0) and np.all(flag[3] == 0) and np.all(flag[:2] == 0))
        self.assertTrue(np.all(sim['row accumulator'][:, 0] == np.arange(1, 5)))
        col = sim['column accumulator']
        self.assertTrue(np.allclose(col[:, 0], 0.5) and np.all(np.diff(col, axis=1) > 0))


if __name__ == '__main__':
    unittest.main()
