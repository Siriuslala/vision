"""Model-free checks of experiment 3.12: stimuli, digit bookkeeping, toy theory and analysis statistics."""
import unittest

import numpy as np

from analysis_tools.position import digit_failure as df
from analysis_tools.position import digit_failure_analysis as A
from analysis_tools.position import digit_failure_theory as T


class StimulusTests(unittest.TestCase):
    def test_text_position_row(self):
        row = df.text_position_row(3, 5)
        self.assertEqual(row, 'o o x o o')
        with self.assertRaises(ValueError):
            df.text_position_row(0, 5)
        self.assertIn('x', df.text_position_prompt(42))

    def test_point_prefix(self):
        self.assertTrue(df.point_prefix(0, None).endswith('['))
        self.assertTrue(df.point_prefix(1, 500).endswith('[500, '))
        with self.assertRaises(ValueError):
            df.point_prefix(1, None)

    def test_grid_cases_and_truth(self):
        cases = df.grid_cases()
        self.assertEqual(len(cases), 2 * (6 + 10 + 20))
        img, truth = df.grid_input(cases[0])
        self.assertEqual(img.size, tuple(truth['canvas']))
        edges = truth['col_edges_units']
        self.assertEqual(len(edges), truth['n'] + 1)
        self.assertTrue(edges[cases[0]['col']] <= truth['center_units'][0] <= edges[cases[0]['col'] + 1])


class DigitTests(unittest.TestCase):
    def test_digits(self):
        self.assertEqual(df.first_and_second(357.4), (3, 5, 3))
        self.assertEqual(df.first_and_second(62.5), (6, 2, 2))
        self.assertEqual(A.digits_of(357, 100), (3, 5))
        self.assertEqual(A.digits_of(23, 10), (2, 3))

    def test_softmax_digits(self):
        p = A.softmax_digits(np.log(np.array([[0.1, 0.3] + [0.0001] * 8])))
        self.assertAlmostEqual(p.sum(), 1.0, places=6)
        self.assertGreater(p[0, 1], p[0, 0])


class StatisticsTests(unittest.TestCase):
    def test_anova_pure_effects(self):
        rng = np.random.default_rng(0)
        img = rng.normal(size=(20, 1, 10))
        L = np.broadcast_to(img, (20, 9, 10)).copy()
        dec = A.anova2(L)
        self.assertAlmostEqual(dec['image'], 1.0, places=6)
        kk = rng.normal(size=(1, 9, 10))
        dec = A.anova2(np.broadcast_to(kk, (20, 9, 10)).copy())
        self.assertAlmostEqual(dec['k'], 1.0, places=6)

    def test_js(self):
        p = np.array([0.5, 0.5])
        self.assertAlmostEqual(float(A.js(p, p)), 0.0, places=6)
        self.assertAlmostEqual(float(A.js(np.array([1.0, 0.0]), np.array([0.0, 1.0]))), 1.0, places=3)

    def test_sharpness_curve_limits(self):
        lo, hi = A.sharpness_curve(np.array([1e-3, 100.0]))
        self.assertLess(lo, 0.01)
        self.assertGreater(hi, 0.98)

    def test_fit_evidence_recovers_sigma(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(100, 1000, 200)
        edges = np.arange(1, 11) * 100.0
        P = A.gaussian_bin_probs(x, edges, 40.0)
        P = P / P.sum(1, keepdims=True)
        fit = A.fit_evidence(P, x, edges, np.full(9, 1 / 9), sigmas=np.array([10.0, 40.0, 160.0]),
                             betas=np.array([0.0, 1.0]), shifts=np.array([0.0]))
        self.assertEqual(fit['sigma'], 40.0)

    def test_decode_metrics(self):
        x = np.array([150.0, 250.0, 450.0, 550.0])
        m = A.decode_metrics(x.copy(), x)
        self.assertEqual(m['mae']['mean'], 0.0)
        self.assertEqual(m['first_acc'], 1.0)


class TheoryTests(unittest.TestCase):
    def test_envelope_bands(self):
        demo = T.envelope_demo(401)
        self.assertGreater(demo['hundreds_z']['acc'], 0.95)
        self.assertLess(demo['tens_z']['acc'], 0.3)
        self.assertGreater(demo['tens_zk']['acc'], demo['tens_z']['acc'])

    def test_periodic_features_read_tens(self):
        r = T.toy_experiment(sigma=2.0, n=4000)
        self.assertGreater(r['tens|periodic']['acc'], 0.6)
        self.assertLess(r['tens|z']['acc'], 0.25)


if __name__ == '__main__':
    unittest.main()
