"""Model-free checks of experiment 3.6 round 2: formats, digit slots, padding, interventions and metrics."""
import unittest

import numpy as np
import torch

from analysis_tools.position import coordinate_value_r2 as c2
from analysis_tools.position import coordinate_value_r2_analysis as A


class CharTokenizer:
    """Tokenizer stub: one token per character, with offsets (digits are single tokens, as in Qwen3.5)."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class FormatTests(unittest.TestCase):
    def test_true_strings(self):
        self.assertEqual(c2.true_string('s1000', 357.4), '357')
        self.assertEqual(c2.true_string('s100', 357.4), '36')
        self.assertEqual(c2.true_string('s10', 357.4), '4')
        self.assertEqual(c2.true_string('pix', 500.0), '256')
        self.assertEqual(c2.true_string('frac', 62.5), '0.06')
        with self.assertRaises(ValueError):
            c2.true_string('hex', 1.0)

    def test_baseline_prompt_matches_round1(self):
        from analysis_tools.position import coordinate_readout as cr
        self.assertEqual(c2.format_prompt('s1000'), cr.DISC_QUERY + cr.SUFFIX['point'])

    def test_ideal_point_text(self):
        text, ok = c2.ideal_point_text('```json\n[{"point_2d": [33, 41], "label": "x"}]\n```', ['62', '0.50'])
        self.assertTrue(ok)
        self.assertIn('[62, 0.50]', text)
        text, ok = c2.ideal_point_text('no answer', ['1', '2'])
        self.assertFalse(ok)
        self.assertIn('[1, 2]', text)

    def test_digit_slots(self):
        ans = '[{"point_2d": [0.37, 512]}]'
        ids, slots = c2.digit_slots(CharTokenizer(), 100, ans, c2.point_numbers(ans))
        self.assertEqual([(s.number, s.k, s.digit, s.after_point) for s in slots],
                         [(0, 0, 0, False), (0, 1, 3, True), (0, 2, 7, True), (1, 0, 5, False), (1, 1, 1, False), (1, 2, 2, False)])
        first = ans.index('0.37')
        self.assertEqual(slots[0].pos, 100 + first - 1)

    def test_count_image(self):
        img, centers = c2.count_image(12, 3)
        self.assertEqual(len(centers), 12)
        self.assertEqual(img.size, (512, 512))


class TensorTests(unittest.TestCase):
    def test_pad_batch(self):
        xs = [torch.ones(1, 3, 4), torch.ones(1, 5, 4)]
        ps = [torch.arange(3).view(1, 1, 3).repeat(4, 1, 1), torch.arange(5).view(1, 1, 5).repeat(4, 1, 1)]
        X, P = c2.pad_batch(xs, ps)
        self.assertEqual(X.shape, (2, 5, 4))
        self.assertEqual(P.shape, (4, 2, 5))
        self.assertEqual(P[0, 0].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(float(X[0, 3:].abs().sum()), 0.0)

    def test_apply_copy_and_set(self):
        h = torch.arange(2 * 4 * 6, dtype=torch.float32).view(2, 4, 6)
        ops = [c2.Op(0, 1, 'mlp', (1, 3), 1), c2.Op(1, 1, 'mlp', 0, torch.zeros(6))]
        c2.Interventions._apply(h, ops, None)
        self.assertTrue(torch.equal(h[0, 1:3], torch.arange(48, dtype=torch.float32).view(2, 4, 6)[1, 1:3]))
        self.assertEqual(float(h[1, 0].abs().sum()), 0.0)
        g = torch.zeros(1, 2, 8)
        c2.Interventions._apply(g, [c2.Op(0, 1, 'head', 1, torch.ones(4), 1)], 4)
        self.assertEqual(g[0, 1].tolist(), [0, 0, 0, 0, 1, 1, 1, 1])


class MetricTests(unittest.TestCase):
    def test_curve_metrics_line_and_circle(self):
        t = np.linspace(0, 1000, 201)
        line = np.c_[t, 2 * t, np.zeros_like(t)] / 100
        m = A.curve_metrics(t, line + np.random.default_rng(0).normal(0, 0.01, line.shape))
        self.assertLess(m['tortuosity'], 1.05)
        self.assertGreater(m['tangent_corr']['500'], 0.95)
        circ = np.c_[np.cos(2 * np.pi * t / 1000), np.sin(2 * np.pi * t / 1000)] * 10
        mc = A.curve_metrics(t, circ)
        self.assertLess(mc['tangent_corr']['500'], -0.9)
        self.assertAlmostEqual(mc['pr'], 2.0, delta=0.1)

    def test_mcnemar_and_sign_flip(self):
        self.assertEqual(A.mcnemar_p(0, 0), 1.0)
        self.assertLess(A.mcnemar_p(0, 12), 0.001)
        self.assertLess(A.sign_flip_p(np.ones(30)), 0.001)

    def test_slot_bin(self):
        self.assertEqual(A.slot_bin('disc', 's1000', {'n_digits': 3, 'k': 1}), ('second_last', 10.0))
        self.assertEqual(A.slot_bin('disc', 's100', {'n_digits': 2, 'k': 0}), ('second_last', 100.0))
        self.assertEqual(A.slot_bin('disc', 'frac', {'n_digits': 3, 'k': 2}), ('last', 10.0))
        self.assertEqual(A.slot_bin('disc', 'frac', {'n_digits': 3, 'k': 1}), ('second_last', 100.0))

    def test_circ_diff(self):
        self.assertEqual(int(A.circ_diff(9, 0)), 1)
        self.assertEqual(int(A.circ_diff(2, 7)), 5)


if __name__ == '__main__':
    unittest.main()
