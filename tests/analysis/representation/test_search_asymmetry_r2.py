"""Model-free tests of experiment 2.7 round 2: new stimuli, layouts and metrics."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_r2 as r2
from analysis_tools.representation import search_asymmetry_r2_analysis as A
from analysis_tools.representation import search_asymmetry_r2_run as rr


class StimulusTests(unittest.TestCase):
    def test_new_pairs_differ_and_fit_the_cell(self):
        for pair, spec in r2.NEW_PAIRS.items():
            for param in spec['levels'].values():
                for orient in range(4):
                    a = r2.render_entity(pair, True, param, orient, (sa.JITTER, -sa.JITTER))
                    b = r2.render_entity(pair, False, param, orient, (sa.JITTER, -sa.JITTER))
                    self.assertEqual(a.shape, (64, 64, 3))
                    self.assertGreater(np.abs(a.astype(int) - b.astype(int)).sum(), 0)
                    for t in (a, b):
                        self.assertTrue((t[:2] == 255).all() and (t[-2:] == 255).all()
                                        and (t[:, :2] == 255).all() and (t[:, -2:] == 255).all())

    def test_subset_pairs_are_pixel_subsets(self):
        # in a subset pair every dark pixel of B is also dark in A (A = B plus a part)
        for pair in ('term', 'plus', 'dot'):
            p = r2.NEW_PAIRS[pair]['levels']['salient']
            a = r2.render_entity(pair, True, p, 1, (0, 0)).mean(-1) < 128
            b = r2.render_entity(pair, False, p, 1, (0, 0)).mean(-1) < 128
            self.assertLess((b & ~a).sum(), 0.02 * b.sum())
            self.assertGreater((a & ~b).sum(), 0)

    def test_layout_grid_geometry(self):
        rng = np.random.default_rng(0)
        jit = sa.jitter_map(3, 12, rng)
        out = r2.grid_images('qo', 6, 3, 12, (2, 11), 0, jit, ('swap',))
        image, meta = out['swap']
        tok = oe.entity_tokens(meta)
        self.assertEqual(len(tok), 36)
        self.assertEqual(meta['rows'], 3)
        self.assertEqual(sum(not i['has_feature'] for i in meta['instances']), 1)
        self.assertEqual(image.size, tuple(meta['canvas']))


class MetricTests(unittest.TestCase):
    def test_causal_surprise(self):
        x = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 3.0]])
        s = r2.causal_surprise(x)
        self.assertTrue(np.isnan(s[0]))
        np.testing.assert_allclose(s[1:], [2.0, 0.0, 3.0])
        g = r2.global_surprise(x)
        np.testing.assert_allclose(g[3], np.linalg.norm(x[3] - x[:3].mean(0)))

    def test_oddness_direction_cancels_feature(self):
        odd, feat = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        u = r2.oddness_direction((odd + feat)[None].repeat(3, 0), (odd - feat)[None].repeat(3, 0))
        np.testing.assert_allclose(u, odd)

    def test_rotation_to_canonical(self):
        m = np.zeros((4, 4))
        m[3, 0] = 1                      # orient 1 points down-left
        self.assertEqual(A.rotate_to_canonical(m, 1)[3, 3], 1)    # canonical orient 0 points down-right

    def test_entity_matrix(self):
        probs = torch.tensor([[[0.5, 0.5, 0.0], [0.2, 0.2, 0.6], [0.0, 0.0, 1.0]]])
        a = rr.assignment([[0, 1], [2]], 3, 'cpu')
        m = rr.entity_matrix(probs, a, a)
        np.testing.assert_allclose(m[0].numpy(), [[0.7, 0.3], [0.0, 1.0]], atol=1e-6)

    def test_local_salience_detects_a_unique_unit(self):
        rng = np.random.default_rng(0)
        act = rng.normal(0, 0.05, (36, 50))
        act[5, 7] += 3.0
        self.assertGreater(A.local_salience(act, 5), 3 * A.local_salience(act, 6))


if __name__ == '__main__':
    unittest.main()
