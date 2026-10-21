"""GPU-free checks of experiment 2.2: high-norm detection, ablation replacement, PosVar, PE flips and metrics."""
import unittest

import numpy as np
import torch

from analysis_tools.position import vit_layout as vl
from analysis_tools.representation import high_norm as hn
from analysis_tools.representation.high_norm_analysis import auroc, mcnemar_exact, ridge_fit_eval, table_coords
from analysis_tools.representation.high_norm_run import probe_case, probe_render


class DetectionTest(unittest.TestCase):
    def test_mask_is_per_image(self):
        x = torch.ones(10, 4)
        x[2] *= 10           # image 0 outlier
        x[7] *= 2            # image 1: below 3x its own median
        m = hn.hn_mask(x, [5, 5])
        self.assertEqual(m.nonzero().flatten().tolist(), [2])

    def test_replace_modes(self):
        x = torch.ones(8, 3)
        x[1] = 50
        log = []
        out = hn._replace(x, [8], hn.Ablation('hn_mean'), log, 1, 10)
        self.assertTrue(torch.allclose(out[1], torch.ones(3)))
        out = hn._replace(x, [8], hn.Ablation('hn_zero'), log, 1, 10)
        self.assertTrue(torch.equal(out[1], torch.zeros(3)))
        out = hn._replace(x, [8], hn.Ablation('rand_mean'), log, 1, 10)
        self.assertTrue(torch.equal(out[1], x[1]))                 # high-norm token untouched
        self.assertEqual(int((out != x).any(1).sum()), 0)          # mean of ones replaces a one: unchanged values
        self.assertEqual([e['n_hn'] for e in log], [1, 1, 1])


class PosVarTest(unittest.TestCase):
    def test_pure_position_and_pure_noise(self):
        g = torch.Generator().manual_seed(0)
        P, D = 6, 5
        mu = torch.randn(P, D, generator=g) * 5
        pos, noise = hn.PosVar(P, D), hn.PosVar(P, D)
        for n in range(40):
            pos.add(mu + 0.01 * torch.randn(P, D, generator=g), n)
            noise.add(torch.randn(P, D, generator=g), n)
        self.assertGreater(pos.r2()['r2'], 0.99)
        self.assertLess(noise.r2()['r2'], 0.05)
        ex = torch.zeros(P, dtype=torch.bool)
        ex[0] = True
        self.assertEqual(pos.r2(ex)['positions'], P - 1)


class GeometryTest(unittest.TestCase):
    def test_pe_flip_is_raster_mirror(self):
        hp, wp = 4, 6
        pe = vl.to_block(torch.arange(hp * wp, dtype=torch.float32)[:, None], hp, wp)
        out = vl.to_raster(hn.pe_flip('h')(pe, hp, wp), hp, wp).view(hp, wp)
        self.assertTrue(torch.equal(out, torch.arange(hp * wp, dtype=torch.float32).view(hp, wp).flip(1)))

    def test_table_coords(self):
        self.assertEqual(table_coords(0, 31, 32, 32), (0.0, 47.0))

    def test_patch_content_shape(self):
        im = probe_render(probe_case(0, 1, 384))
        c = hn.patch_content(im, 24, 24)
        self.assertEqual(c.shape, (576, 18))
        self.assertTrue((c[:, :16] <= 1).all())

    def test_outlier_table_fn(self):
        t = torch.ones(10, 3)
        t[0] *= 10
        out = hn.table_outlier_fn(3.0, 'shrink')(t.clone())
        self.assertAlmostEqual(float(out[0].norm()), float(t[1].norm()), places=5)


class MetricTest(unittest.TestCase):
    def test_auroc_and_mcnemar(self):
        self.assertEqual(auroc([3, 2, 1, 0], [1, 1, 0, 0]), 1.0)
        self.assertEqual(auroc([1, 1], [1, 0]), 0.5)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 2 / 64)

    def test_ridge_recovers_linear_map(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5))
        Y = X @ rng.normal(size=(5, 2))
        self.assertGreater(ridge_fit_eval(X[:200], Y[:200], X[200:], Y[200:])['r2'], 0.99)


if __name__ == '__main__':
    unittest.main()
