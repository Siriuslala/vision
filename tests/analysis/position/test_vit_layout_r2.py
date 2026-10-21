"""Model-free checks of experiment 3.7 round 2: stimuli, regions, permutations, gathers and metrics."""
import unittest

import numpy as np
import torch
from PIL import Image

from analysis_tools.position import vit_layout_r2 as v2
from analysis_tools.position.vit_layout_r2_analysis import overlap, rope_frequencies, spearman


def fake_case(W=128, H=96):
    return {'id': 'x', 'canvas': [W, H], 'box': [10, 20, 50, 70], 'expression': 'the man',
            'polygon': [[12, 22], [48, 22], [48, 68], [12, 68]]}


class StimulusTests(unittest.TestCase):
    def test_block_shuffle_keeps_blocks(self):
        a = np.random.default_rng(0).integers(0, 255, (64, 96, 3), dtype=np.uint8)
        b = np.asarray(v2.block_shuffle(Image.fromarray(a), 1))
        blocks_a = {a[r:r + 32, c:c + 32].tobytes() for r in (0, 32) for c in (0, 32, 64)}
        blocks_b = {b[r:r + 32, c:c + 32].tobytes() for r in (0, 32) for c in (0, 32, 64)}
        self.assertEqual(blocks_a, blocks_b)

    def test_checker_patches_identical(self):
        c = v2.checker(64, 32)
        p = [c[r:r + 16, k:k + 16].tobytes() for r in (0, 16) for k in range(0, 64, 16)]
        self.assertEqual(len(set(p)), 1)

    def test_target_box_flip(self):
        case = fake_case()
        self.assertEqual(v2.target_box(case, 'natural_vflip'), [10, 26, 50, 76])

    def test_score_point_lenient(self):
        truth = {'canvas': [100, 100], 'box': [0, 0, 50, 50], 'center': [25, 25]}
        s = v2.score_point(truth, '[{"point_2d": [250, 250]},{"point_2d": [250, 250]}]')
        self.assertTrue(s['lenient_valid'] and not s['valid'] and s['hit'])
        self.assertFalse(v2.score_point(truth, 'nothing')['lenient_valid'])


class RegionTests(unittest.TestCase):
    def test_primary_region_disjoint_and_shifted(self):
        grid = (4, 6)
        T = [0, 1, 6, 7]
        reg = v2.primary_region(T, grid)
        self.assertEqual(reg['axis'], 'h')
        self.assertFalse(set(reg['region']) & set(T))
        self.assertEqual(reg['region'], [t + reg['dc'] for t in T])

    def test_patch_perm_is_involution(self):
        perm = v2.region_patch_perm(4 * 6 * 4, [0, 1], [4, 5])
        self.assertTrue(np.array_equal(perm[perm], np.arange(len(perm))))
        self.assertEqual(perm[:4].tolist(), [16, 17, 18, 19])

    def test_homogeneity_regions_order(self):
        tex = np.arange(24, dtype=float)
        out = v2.homogeneity_regions([0], (4, 6), tex, 'h')
        self.assertLess(out['homog']['texture'], out['textured']['texture'])

    def test_w_shift(self):
        self.assertAlmostEqual(v2.w_shift([150, 0], [100, 0], [100, 0]), 0.5)


class MetricTests(unittest.TestCase):
    def test_overlap(self):
        q = torch.linalg.qr(torch.randn(50, 50))[0].T
        self.assertAlmostEqual(overlap(q, q, 5), 1.0, places=5)
        self.assertLess(overlap(q[:5], q[5:10], 5), 1e-6)

    def test_rope_frequencies(self):
        f = rope_frequencies(72)
        self.assertEqual(len(f), 18)
        self.assertAlmostEqual(float(f[0]), 1.0)

    def test_spearman(self):
        a = np.arange(30, dtype=float)
        rho, p = spearman(a, a ** 2, n_perm=200)
        self.assertAlmostEqual(rho, 1.0)
        self.assertLess(p, 0.01)


if __name__ == '__main__':
    unittest.main()
