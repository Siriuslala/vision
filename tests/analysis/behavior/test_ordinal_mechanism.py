"""Stimuli, route predictions, answer parsing and CPU analysis helpers of experiment 1.5 (no GPU)."""
import unittest

import numpy as np

from analysis_tools.behavior import ordinal_mechanism as om
from analysis_tools.behavior import ordinal_mechanism_analysis as oma


class VariantTests(unittest.TestCase):
    def test_route_predictions_per_variant(self):
        rng = np.random.default_rng(0)
        for c in om.C_RANGE:
            expect = {'base': (c, c), 'shift': (c, c + 1), 'insL': (c, c + 1), 'delL': (c - 1, c), 'delR': (c, c)}
            for v, (count, slot) in expect.items():
                present, target, blank = om.variant_layout(v, c, rng)
                r = om.route_predictions(present, target)
                self.assertEqual((r['count'], r['slot']), (count, slot), (v, c))
                self.assertEqual(r['n_present'], 20)  if v in ('base', 'shift', 'insL') else self.assertEqual(r['n_present'], 19)
                if blank is not None:
                    self.assertTrue(0 < blank < om.SLOTS - 1)
                    self.assertNotIn(blank, present)
                if v in ('insL', 'delL'):
                    self.assertEqual(r['gap'], sum(1 for p in present if blank < p <= target))
                else:
                    self.assertEqual(r['gap'], r['count'])

    def test_render_blank_line_and_single_target(self):
        for axis in ('H', 'V'):
            present, target, blank = om.variant_layout('delL', 9, np.random.default_rng(1))
            spec = om.SlotGrid('square', axis, 64, present, (target, 2))
            img, meta = om.render(spec)
            a = np.asarray(img)
            w, h = meta['canvas']
            self.assertEqual((w % 32, h % 32), (0, 0))
            ox, oy = meta['grid_origin']
            if axis == 'H':
                band = a[oy:oy + 6 * 64, ox + blank * 64: ox + (blank + 1) * 64]
            else:
                band = a[oy + blank * 64: oy + (blank + 1) * 64, ox:ox + 6 * 64]
            self.assertTrue((band == 255).all())
            red = (a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] < 60)
            x0, y0, x1, y1 = meta['target_cell_box']
            self.assertGreater(red[y0:y1, x0:x1].sum(), 0)
            self.assertEqual(red.sum(), red[y0:y1, x0:x1].sum())

    def test_invalid_target(self):
        with self.assertRaises(ValueError):
            om.SlotGrid('square', 'H', 64, (0, 1, 2), (5, 0))


class AnswerTests(unittest.TestCase):
    def test_last_pair_is_used(self):
        text = 'Row 3 ... {"row": 2, "column": 4} is wrong; final:\n{"row": 3, "column": 7}'
        self.assertEqual(om.parse_answer(text), (3, 7))
        self.assertIsNone(om.parse_answer('row 3, column 7'))

    def test_probed_axis(self):
        self.assertEqual(om.probed_answer({'axis': 'H'}, (3, 7)), (7, 3))
        self.assertEqual(om.probed_answer({'axis': 'V'}, (3, 7)), (3, 7))

    def test_strategy_codes_ignore_answer(self):
        codes = om.code_strategies('Counting columns 1, 2, 3, 4 from the left.\n{"row": 1, "column": 4}')
        self.assertTrue(codes['enumerate'])
        self.assertFalse(codes['from_far_end'])


class AnalysisTests(unittest.TestCase):
    def test_ridge_cv_recovers_linear_target(self):
        rng = np.random.default_rng(0)
        L = rng.standard_normal((200, 5))
        X = L @ rng.standard_normal((5, 50)) + 0.05 * rng.standard_normal((200, 50))
        y = L[:, :3] @ np.array([1.0, -2.0, 0.5])
        P = oma.ridge_pcs(oma.gram_pca(X, 20), y[:, None], np.arange(200))
        self.assertGreater(oma.r2(y, P[:, 0]), 0.95)
        self.assertGreater(oma.classifier_cv(L[:, :3], (y > 0).astype(int), np.arange(200), 'linear'), 0.9)

    def test_model_probabilities_normalised(self):
        c = np.arange(1, 21)
        for name, theta in oma.INIT.items():
            P = oma.model_probs(name, np.array(theta, float), c, 20, 23, np.zeros(30) if name == 'digit_prior' else None, 5)
            self.assertTrue(np.allclose(P.sum(1), 1, atol=1e-6), name)

    def test_encoding_directions(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((300, 2))
        B = rng.standard_normal((2, 10))
        X = Z @ B + 0.01 * rng.standard_normal((300, 10))
        self.assertTrue(np.allclose(oma.encoding_directions(X, Z), B, atol=0.01))


if __name__ == '__main__':
    unittest.main()
