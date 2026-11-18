"""CPU tests of experiment 7.3 Part C (counting-gate fine-tuning): scopes, stimuli, layer choice, token classes."""
import unittest

import numpy as np

from analysis_tools.application import counting_gate as cg
from analysis_tools.application import counting_gate_run as cgr


class TestScopes(unittest.TestCase):
    def test_parse_and_params(self):
        self.assertEqual(cg.parse_scope('gate:13,9')['layers'], [9, 13])
        with self.assertRaises(ValueError):
            cg.parse_scope('gate:12')           # full-attention layer
        self.assertEqual(cg.lora_params('gate:9', 8), 2 * 8 * (4096 + 32))
        self.assertEqual(len(cg.parse_scope('dnall')['layers']), 24)


class TestStimuli(unittest.TestCase):
    def test_random_grid_answer_counts_present_lines(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            spec, count, other = cg.random_slot_grid(rng)
            self.assertEqual(count, sum(1 for p in spec.present if p <= spec.target[0]))
            r, c = cg.rowcol_answer(spec, count, other)
            self.assertEqual((r, c), (other, count) if spec.axis == 'H' else (count, other))
            self.assertIn(spec.template, cg.TRAIN_TEMPLATES)


class TestChoice(unittest.TestCase):
    def test_rule(self):
        base = {'H64': 0.7, 'V64': 0.6, 'g8': 0.9}
        rows = [{'layer': 0, 'edit': 'none', 'acc': base}]
        drops = {9: 0.3, 10: 0.05, 13: 0.2, 1: 0.0}
        for l, dr in drops.items():
            for e in cgr.SWEEP_EDITS:
                rows.append({'layer': l, 'edit': e, 'acc': {k: v - (dr if e == 'nowrite' else 0) for k, v in base.items()}})
        ch = cgr.choose_layers(rows)
        self.assertEqual(ch['chosen'], [9, 13])
        self.assertEqual(ch['wrong'], [1, 10])


class TestTokenClasses(unittest.TestCase):
    def test_onsets(self):
        case = {'axis': 'H', 'truth': {'cell': 64, 'target_cell_box': [128, 0, 192, 64], 'target_slot': 2, 'other': 1,
                                       'present': [0, 2]}}
        cls = cgr.token_classes(case, 2, 6).reshape(2, 6)
        self.assertEqual(cls[0].tolist(), [1, 2, 0, 0, 1, 2])


if __name__ == '__main__':
    unittest.main()
