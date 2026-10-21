"""Model-free checks of experiment 3.6: cases, query positions, ridge probe, digit helpers and metrics."""
import unittest

import numpy as np

from analysis_tools.position import coordinate_value as cv
from analysis_tools.position.coordinate_value_analysis import auroc, first_digit, ridge_fit, ridge_predict, r2
from analysis_tools.position.coordinate_value_run import manifold_mean


class CaseTests(unittest.TestCase):
    def test_case_counts_and_bounds(self):
        cases = cv.disc_cases()
        self.assertEqual(sum(c['set'] == 'grid2d' for c in cases), 29 * 29)
        self.assertEqual(sum(c['set'] == 'sweep_h' for c in cases), 225)
        for c in cases:
            for v in c['center']:
                self.assertTrue(cv.DIAMETER / 2 <= v <= cv.CANVAS - cv.DIAMETER / 2)

    def test_truth_units(self):
        _, truth, _ = cv.disc_input({'center': [256.0, 128.0], 'diameter': 64, 'canvas': 512})
        self.assertEqual(cv.truth_units(truth), (500.0, 250.0))


class PositionTests(unittest.TestCase):
    def test_query_positions_point_to_previous_token(self):
        roles = cv.query_positions(100, [[5, 6, 7], [10, 11]])
        self.assertEqual(roles, {'last': 99, 'x0': 104, 'x1': 105, 'x2': 106, 'y0': 109, 'y1': 110})

    def test_digits(self):
        self.assertEqual(cv.leading_digits(357, 2), [3, 5])
        self.assertEqual(cv.leading_digits(42, 1), [4])
        self.assertEqual(cv.coarse_digit(99.6), 1)
        self.assertEqual(first_digit(62), 6)


class StatisticsTests(unittest.TestCase):
    def test_ridge_recovers_linear_map(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 30))
        y = X @ rng.normal(size=30) + 5
        m = ridge_fit(X[:150], y[:150, None])
        self.assertGreater(r2(y[150:], ridge_predict(m, X[150:])[:, 0]), 0.95)

    def test_auroc(self):
        self.assertEqual(auroc(np.array([3., 4.]), np.array([1., 2.])), 1.0)
        self.assertEqual(auroc(np.array([1.]), np.array([2.])), 0.0)

    def test_manifold_mean_excludes_self(self):
        x = np.array([0., 10., 20.])
        H = np.stack([np.full((1, 3, 2), v, np.float32) for v in (0., 1., 2.)])
        m = manifold_mean(x, H, at=10., exclude=1)
        self.assertAlmostEqual(float(m[0, 0, 0]), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
