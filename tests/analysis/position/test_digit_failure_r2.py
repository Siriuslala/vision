"""Model-free checks of experiment 3.12 round 2: cache editing helpers, pair selection, analysis statistics."""
import unittest

import numpy as np
import torch

from analysis_tools.position import digit_failure as df
from analysis_tools.position import digit_failure_r2 as R2
from analysis_tools.position import digit_failure_r2_analysis as M


class FakeLinearLayer:
    def __init__(self, rows: int):
        self.recurrent_states = {0: torch.arange(rows, dtype=torch.float32).view(rows, 1, 1).repeat(1, 2, 2)}
        self.conv_states = {0: torch.arange(rows, dtype=torch.float32).view(rows, 1, 1).repeat(1, 3, 4)}
        self.is_recurrent_states_initialized = {0: True}
        self.is_conv_states_initialized = {0: True}


class FakeFullLayer:
    def __init__(self, rows: int):
        self.keys = torch.arange(rows, dtype=torch.float32).view(rows, 1, 1, 1).repeat(1, 2, 5, 3)
        self.values = self.keys.clone()


class CacheEditTests(unittest.TestCase):
    def test_copy_kv_positions_only(self):
        full = {4: FakeFullLayer(3)}
        R2.copy_kv(full, 2, 1, [1, 3], (4,))
        k = full[4].keys
        self.assertTrue(torch.all(k[2, :, [1, 3]] == 1))
        self.assertTrue(torch.all(k[2, :, [0, 2, 4]] == 2))
        self.assertTrue(torch.all(k[0] == 0))

    def test_copy_delta_last_slot_and_layers(self):
        lin = {25: FakeLinearLayer(3), 1: FakeLinearLayer(3)}
        R2.copy_delta(lin, 2, 1, recurrent=False, last_only=True, layers=range(25, 33))
        c = lin[25].conv_states[0]
        self.assertTrue(torch.all(c[2, :, -1] == 1))
        self.assertTrue(torch.all(c[2, :, :-1] == 2))
        self.assertTrue(torch.all(lin[25].recurrent_states[0][2] == 2))
        self.assertTrue(torch.all(lin[1].conv_states[0][2] == 2))          # layer outside the range untouched
        R2.copy_delta(lin, 0, 1)
        self.assertTrue(torch.all(lin[1].recurrent_states[0][0] == 1))


class PairTests(unittest.TestCase):
    def test_patch_pairs(self):
        units = [df.Unit(f'sweep_h_{i}', 'sweep_h', 0, float(v), '', '') for i, v in enumerate(range(100, 1000, 8))]
        pairs = R2.patch_pairs(units)
        by_id = {u.id: u for u in units}
        q1 = [p for p in pairs if p['role'] == 'q1']
        q0 = [p for p in pairs if p['role'] == 'q0']
        self.assertTrue(q1 and q0)
        for p in q1:
            a, b = by_id[p['A']].truth, by_id[p['B']].truth
            self.assertEqual(int(a // 100), p['k'])
            self.assertEqual(int(b // 100), p['k'])
        for p in q0:
            self.assertEqual(abs(int(by_id[p['A']].truth // 100) - int(by_id[p['B']].truth // 100)), 1)
            self.assertEqual(int(by_id[p['A']].truth // 100), p['k'])

    def test_conditions_known(self):
        for role in ('q0', 'q1'):
            conds = R2.patch_conditions(role)
            self.assertEqual(len(conds), len(set(conds)))
            self.assertIn('img@20', conds)
            self.assertEqual(conds[:2], ['A', 'B'])
        self.assertIn('q0', R2.patch_conditions('q1'))
        self.assertNotIn('q0', R2.patch_conditions('q0'))

    def test_ablation_conditions(self):
        order = np.array([[27, 5], [31, 2], [28, 7], [27, 1]])
        plan = {'order': order, 'n_values': np.array([2, 4]), 'mean_L32': np.zeros((16, 256), np.float32)}
        for l in M.PRIOR_MLP_LAYERS:
            plan[f'mean_mlp{l}'] = np.arange(12288, dtype=np.float32)
            plan[f'rand_mlp{l}'] = np.array([3], dtype=np.int64) if l == 29 else np.array([], dtype=np.int64)
        conds = R2.ablation_conditions(plan)
        names = [c[0] for c in conds]
        self.assertEqual(names[:3], ['none', 'H8', 'H10'])
        n2 = dict((c[0], c) for c in conds)['N2']
        self.assertEqual(sorted(n2[1]), [27, 31])
        self.assertEqual(n2[1][27][0].tolist(), [5])
        self.assertEqual(n2[1][27][1].tolist(), [5.0])


class AnalysisTests(unittest.TestCase):
    def test_recovery(self):
        a, b = np.zeros(10), np.arange(10.0)
        self.assertAlmostEqual(np.divide(*M.recovery(a, b, b)), 1.0)
        self.assertAlmostEqual(np.divide(*M.recovery(a, b, 0.5 * b)), 0.5)
        self.assertAlmostEqual(M.recovery(a, b, a + 3.0)[0], 0.0)           # a digit-constant shift is no change

    def test_ratio_ci(self):
        est, lo, hi = M.ratio_ci(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
        self.assertAlmostEqual(est, 0.5)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)

    def test_trajectory_rank_of_arc(self):
        th = np.linspace(0, np.pi, 200)
        phase = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        L = 5 * np.cos(th[:, None] - phase[None])
        tr = M.trajectory(L)
        self.assertGreater(tr['ev'][:2].sum(), 0.999)
        self.assertLess(tr['ev'][0], 0.9)
        self.assertGreater(tr['agree'][2], 0.99)

    def test_runs_per_digit(self):
        self.assertEqual(M.runs_per_digit(np.array([1, 1, 2, 2, 3])), 1.0)
        self.assertEqual(M.runs_per_digit(np.array([0, 1, 0, 1])), 2.0)

    def test_lin_r2(self):
        x = np.arange(20.0)
        self.assertAlmostEqual(M.lin_r2(x[:, None], 2 * x + 1), 1.0)


if __name__ == '__main__':
    unittest.main()
