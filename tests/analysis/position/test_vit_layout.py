"""Model-free checks of experiment 3.7: grid orders, PE edits, stimuli, LEACE and metrics."""
import unittest

import numpy as np
import torch
from transformers.vision_utils import get_vision_interpolation_indices_and_weights

from analysis_tools.position import vit_layout as vl
from analysis_tools.position.vit_layout_analysis import auroc, reread_metrics
from analysis_tools.position.vit_layout_run import reread_cases


class GridTests(unittest.TestCase):
    def test_block_raster_roundtrip(self):
        t = torch.arange(6 * 8)
        self.assertTrue(torch.equal(vl.to_raster(vl.to_block(t, 6, 8), 6, 8), t))
        # the first 2x2 block holds raster patches 0, 1, 8, 9
        self.assertEqual(vl.to_block(t, 6, 8)[:4].tolist(), [0, 1, 8, 9])

    def test_window_of_full_canvas_equals_default_taps(self):
        idx, w = vl.interp_taps(20, 24)
        widx, ww = vl.window_taps(20, 24, 20, 24, 0, 0)
        table = torch.randn(48 * 48, 3)
        a = (table[idx] * w[..., None]).sum(1)
        b = (table[widx] * ww[..., None]).sum(1)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_patch_xy_block_order(self):
        xy = vl.patch_xy(4, 4)
        self.assertEqual(xy[:4].tolist(), [[125, 125], [375, 125], [125, 375], [375, 375]])


class EditTests(unittest.TestCase):
    def setUp(self):
        self.table = torch.randn(48 * 48, 5)

    def test_zero_mean_shuffle(self):
        base = vl.edited_pe(self.table, vl.ViTEdit(), 8, 8)
        self.assertEqual(float(vl.edited_pe(self.table, vl.ViTEdit(pe='zero'), 8, 8).abs().max()), 0.0)
        mean = vl.edited_pe(self.table, vl.ViTEdit(pe='mean'), 8, 8)
        self.assertTrue(torch.allclose(mean[0], base.mean(0), atol=1e-6))
        shuf = vl.edited_pe(self.table, vl.ViTEdit(pe='shuffle'), 8, 8)
        self.assertTrue(torch.allclose(shuf.sort(0).values, base.sort(0).values))
        self.assertTrue(torch.allclose(vl.edited_pe(self.table, vl.ViTEdit(pe_scale=4.0), 8, 8), 4 * base))

    def test_native_uses_table_entries(self):
        pe = vl.to_raster(vl.edited_pe(self.table, vl.ViTEdit(pe='native'), 4, 6), 4, 6)
        self.assertTrue(torch.equal(pe[0], self.table[22 * 48 + 21]))
        with self.assertRaises(ValueError):
            vl.edited_pe(self.table, vl.ViTEdit(pe='native'), 50, 50)

    def test_native_768_equals_interpolation(self):
        a = vl.edited_pe(self.table, vl.ViTEdit(), 48, 48)
        b = vl.edited_pe(self.table, vl.ViTEdit(pe='native'), 48, 48)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_rope_edits(self):
        ids = vl.edited_rope(vl.ViTEdit(), 4, 4)
        self.assertEqual(float(vl.edited_rope(vl.ViTEdit(rope='zero'), 4, 4).abs().max()), 0.0)
        off = vl.edited_rope(vl.ViTEdit(rope='offset', rope_dr=3, rope_dc=5), 4, 4)
        self.assertTrue(torch.equal(off - ids, torch.tensor([3, 5]).expand_as(ids)))


class StimulusTests(unittest.TestCase):
    def test_disc_cases_inside_canvas(self):
        for canvas in (256, 1536):
            for c in vl.disc_grid_cases(canvas, 0):
                r = c['diameter'] / 2
                self.assertTrue(all(r <= v <= canvas - r for v in c['center']))

    def test_t2_balanced(self):
        cases = vl.t2_cases(512, 0)
        self.assertEqual(sum(c['same'] for c in cases), 24)
        self.assertEqual(vl.t2_render(cases[0]).size, (512, 512))

    def test_layout_masks_and_labels(self):
        c = vl.layout_case(3, 0)
        masks = vl.object_patch_masks(c, 32, 32)
        self.assertTrue(masks['red'].any() and masks['blue'].any() and masks['background'].any())
        self.assertFalse((masks['background'] & (masks['red'] | masks['blue'])).any())
        lab = vl.layout_labels(c)
        self.assertEqual(lab['red_left'], float(c['red_center'][0] < c['blue_center'][0]))

    def test_reread_crop_inside(self):
        for c in reread_cases(0, 20):
            self.assertTrue(all(0 <= o <= 256 for o in c['origin']))
            self.assertTrue(all(24 <= v <= 232 for v in c['local_center']))


class LeaceTests(unittest.TestCase):
    def test_erasure_removes_linear_information(self):
        g = torch.Generator().manual_seed(0)
        n, d = 4000, 12
        z = torch.randn(n, 2, generator=g, dtype=torch.float64)
        x = torch.randn(n, d, generator=g, dtype=torch.float64) + z @ torch.randn(2, d, generator=g, dtype=torch.float64)
        e = vl.LeaceEraser.fit(n, x.sum(0), x.T @ x, z.sum(0), x.T @ z)
        self.assertEqual(e.rank, 2)
        xe = e(x)
        cov = ((xe - xe.mean(0)).T @ (z - z.mean(0))) / n
        self.assertLess(float(cov.abs().max()), 1e-8)
        er = vl.LeaceEraser.fit(n, x.sum(0), x.T @ x, z.sum(0), x.T @ z, random_rank=2)
        self.assertEqual(er.rank, 2)


class MetricTests(unittest.TestCase):
    def test_auroc(self):
        self.assertEqual(auroc(np.array([2., 3.]), np.array([0., 1.])), 1.0)
        self.assertEqual(auroc(np.array([1.]), np.array([1.])), 0.5)

    def test_reread_weight(self):
        rec = {'truth_local': [100, 100], 'truth_full': [300, 100],
               'conditions': [{'condition': 'a', 'center': [200, 100]}, {'condition': 'b', 'center': None}]}
        m = reread_metrics([rec])
        self.assertAlmostEqual(m['a']['w']['mean'], 0.5)
        self.assertEqual(m['b']['valid']['k'], 0)


if __name__ == '__main__':
    unittest.main()
