"""Tests of the model-free parts of experiment 3.15 round 3 (position sources)."""
import unittest

import numpy as np
import torch

from analysis_tools.position import position_sources as ps


class TestStimuli(unittest.TestCase):
    def test_jgrid_layout_and_table(self):
        cases = ps.jgrid_cases(8)
        self.assertEqual(len({c['id'] for c in cases}), 8)
        img, truth = ps.jgrid_image(cases[0])
        self.assertEqual(img.size, (ps.CANVAS, ps.CANVAS))
        self.assertEqual(len(truth['entities']), cases[0]['rows'] * cases[0]['cols'])
        tab = ps.token_table(truth, 20, 20, 'jgrid')
        self.assertEqual(len(tab['i']), 400)
        self.assertGreater(tab['tgt'].sum(), 0)
        self.assertEqual(tab['row_cnt'].max(), cases[0]['rows'])
        self.assertLessEqual(tab['col_cnt'].max(), cases[0]['cols'])
        # count is monotone along raster order within a column
        g = tab['row_cnt'].reshape(20, 20)
        self.assertTrue((np.diff(g, axis=0) >= 0).all())


class TestDeltaWeights(unittest.TestCase):
    def test_matches_recurrence(self):
        torch.manual_seed(0)
        T, H, d = 25, 2, 6
        q = torch.nn.functional.normalize(torch.randn(T, H, d), dim=-1) / d ** 0.5
        k = torch.nn.functional.normalize(torch.randn(T, H, d), dim=-1)
        v, g, beta = torch.randn(T, H, d), -torch.rand(T, H), torch.rand(T, H)
        S, o = torch.zeros(H, d, d), torch.zeros(T, H, d)
        for t in range(T):
            S = S * g[t].exp()[:, None, None]
            delta = (v[t] - (S * k[t][:, :, None]).sum(1)) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            o[t] = (S * q[t][:, :, None]).sum(1)
        qs = np.array([0, 7, 24])
        A = ps.delta_weights(q, k, g, beta, qs)
        self.assertTrue(torch.allclose(torch.einsum('qhm,mhd->qhd', A, v), o[qs], atol=1e-5))
        self.assertTrue((A[1, :, 8:] == 0).all())


class TestRollAndStats(unittest.TestCase):
    def test_roll_shift_keeps_target_contiguous(self):
        rng = np.random.default_rng(0)
        box = [100.0, 300.0, 160.0, 350.0]
        k = ps.roll_shift(box, 20, 20, 0, rng)
        self.assertIsNotNone(k)
        lo, hi = int(300 // 32), int((350 - 1e-6) // 32)
        self.assertLessEqual((lo + k) % 20, (hi + k) % 20)
        self.assertAlmostEqual(ps.seq_coordinate(500.0, 5, 20), 750.0)

    def test_following_ratio(self):
        d_seq = np.array([100.0, -200.0, 300.0, 150.0])
        f = ps.following_ratio(np.zeros(4), 0.5 * d_seq, np.zeros(4), d_seq)
        self.assertAlmostEqual(f['f'], 0.5)

    def test_variance_partition_unique(self):
        rng = np.random.default_rng(1)
        n = 400
        a, b = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
        Y = np.concatenate([a @ rng.normal(size=(3, 4)), 0.01 * rng.normal(size=(n, 4))], 1)
        r = ps.variance_partition(Y, {'a': a, 'b': b}, np.arange(n) % 20)
        self.assertGreater(r['unique']['a'], 0.9)
        self.assertLess(abs(r['unique']['b']), 0.02)


if __name__ == '__main__':
    unittest.main()
