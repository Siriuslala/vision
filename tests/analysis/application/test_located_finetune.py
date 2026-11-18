"""CPU tests of experiment 7.3 (located ViT fine-tuning): stimuli, scopes, detector statistics, block choice."""
import unittest

import numpy as np

from analysis_tools.application import located_finetune as lf
from analysis_tools.application import located_finetune_analysis as lfa


class TestStimuli(unittest.TestCase):
    def test_render_pairs_differ_only_when_present(self):
        for pair in lf.LOCATE_PAIRS:
            p = lf.LEVELS[pair]['salient']
            a = lf.render_entity(pair, True, p, 1, (0, 0))
            b = lf.render_entity(pair, False, p, 1, (0, 0))
            self.assertEqual(a.shape, (64, 64, 3))
            self.assertGreater(np.abs(a.astype(int) - b.astype(int)).max(), 100, pair)

    def test_grid_image_variants(self):
        jit = {(r, c): (0, 0) for r in range(3) for c in range(3)}
        _, meta = lf.grid_image('sqgap', 4, 3, 3, (1, 2), 0, jit, 'swap')
        flags = [i['has_feature'] for i in meta['instances']]
        self.assertEqual(flags.count(False), 1)
        self.assertFalse(flags[1 * 3 + 2])
        self.assertEqual(lf.patch_index([[0, 1], [2, 3]]).tolist(),
                         [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])

    def test_splits_are_disjoint(self):
        self.assertFalse(set(lf.TRAIN_PAIRS) & (set(lf.HELDOUT_FEATURES) | set(lf.HELDOUT_FAMILY)))
        self.assertFalse(set(lf.FAMILIES['interior']) & set(lf.TRAIN_PAIRS))


class TestScopes(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(lf.parse_scope('vit_mlp:9-11')['blocks'], [9, 10, 11])
        self.assertEqual(lf.parse_scope('vit_attn:1,3')['blocks'], [1, 3])
        with self.assertRaises(ValueError):
            lf.parse_scope('vit_mlp:0-2')
        with self.assertRaises(ValueError):
            lf.parse_scope('decoder:1')

    def test_params(self):
        self.assertEqual(lf.lora_params('vit_mlp:9', 8), 8 * (1152 + 4304) * 2)
        self.assertEqual(lf.lora_params('model_all', 8, 100), lf.lora_params('vit_all', 8) + 100)


class TestStatistics(unittest.TestCase):
    def test_achievable_zeta_detects_linear_signal(self):
        rng = np.random.default_rng(0)
        d = 16
        dis = rng.normal(size=(40, 16, d))
        tgt = rng.normal(size=(40, 16, d))
        tgt[:, 3, 0] += 8.0              # one patch of every target entity carries a strong linear feature
        self.assertGreater(lf.achievable_zeta(tgt[:30], dis[:30], tgt[30:], dis[30:]), 5)
        self.assertLess(abs(lf.achievable_zeta(dis[:20], dis[20:30], dis[30:35], dis[35:])), 3)

    def test_zeta_and_summary(self):
        z = lf.zeta(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]))
        self.assertAlmostEqual(z[0], 100.0)
        self.assertEqual(z[1], 0)
        s = lf.detector_summary(np.array([6.0, 1.0, 7.0]))
        self.assertEqual((s['n_detectors'], s['argmax']), (2, 2))

    def test_choose_window_respects_eligibility(self):
        prof = {'transplant_dhit': np.array([[0.0, 0.9, 0.9, 0.9, 0.1, 0.1]]),
                'eligible_fraction': np.array([1, 0.5, 1, 1, 1, 1.0])}
        self.assertEqual(lfa.choose_window(prof)['blocks'], [3, 4, 5])


if __name__ == '__main__':
    unittest.main()
