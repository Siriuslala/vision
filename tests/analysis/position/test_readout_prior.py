"""Tests of the model-free parts of experiment 3.16 (analysis_tools/position/readout_prior.py)."""
import unittest

import numpy as np

from analysis_tools.position import readout_prior as rp


class ClockDecodabilityTest(unittest.TestCase):
    def test_clock_code_is_decodable_across_hundreds(self):
        x = np.linspace(100, 999, 300)
        X = np.concatenate([rp.periodic_targets(x), x[:, None] / 1000], 1)
        X = X + 0.01 * np.random.default_rng(0).standard_normal(X.shape)
        self.assertGreater(rp.clock_decodability(X, x), 0.95)

    def test_ruler_code_is_not(self):
        x = np.linspace(100, 999, 300)
        X = np.stack([x, (x / 1000) ** 2, (x / 1000) ** 3], 1)
        self.assertLess(rp.clock_decodability(X, x), 0.1)
        self.assertGreater(rp.ruler_decodability(X, x), 0.9)


class MarginDecompositionTest(unittest.TestCase):
    def test_bias_share(self):
        rng = np.random.default_rng(1)
        t = rng.uniform(200, 899, 400)
        k = t.astype(int) // 100
        o = (t - 100 * k) / 10
        bias = np.array([0, 0, 3, -2, 1, 0, 2, -3, 1, 0])
        m = 0.8 * o + bias[k] + 0.1 * rng.standard_normal(len(t))
        lp = np.zeros((len(t), 10))
        lp[np.arange(len(t)), k + 1] = m / 2
        lp[np.arange(len(t)), k - 1] = -m / 2
        d = rp.margin_decomposition(t, lp)
        self.assertGreater(d['bias_share'], 0.9)
        self.assertAlmostEqual(d['slope_within'], 0.8, delta=0.02)
        self.assertLess(d['resolution_unbiased'], d['resolution'])


class SubspaceTest(unittest.TestCase):
    def test_transplant_moves_only_the_subspace(self):
        rng = np.random.default_rng(2)
        x = np.linspace(100, 999, 200)
        D = 64
        a, b, u = rng.standard_normal((3, D))
        H = np.outer(np.cos(2 * np.pi * x / 100), a) + np.outer(np.sin(2 * np.pi * x / 100), b) + np.outer(x / 1000, u)
        subs = rp.code_subspaces(H + 1e-3 * rng.standard_normal(H.shape), x)
        P = subs['p100']
        self.assertEqual(P.shape[0], 2)
        h, donor = rng.standard_normal(D), rng.standard_normal(D)
        new = rp.project_replace(h, donor, P)
        np.testing.assert_allclose(new @ P.T, donor @ P.T, atol=1e-8)
        Q = rp.random_orthonormal(3, D, rng)
        np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
