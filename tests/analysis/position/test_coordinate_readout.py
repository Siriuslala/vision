"""Model-free checks of experiment 3.5: stimuli, parsing, position-source edits and metrics."""
import unittest

import numpy as np
import torch
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position.coordinate_readout_analysis import displacement_weight, wilson
from analysis_tools.position.coordinate_readout_run import disc_cases, swap_regions


def fake_prepared(H: int = 4, W: int = 4, before: int = 3, after: int = 5) -> cr.Prepared:
    """A Prepared prompt with text-image-text MRoPE ids built as in get_rope_index."""
    L = before + H * W + after
    t = torch.arange(before).repeat(3, 1)
    r, c = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    img = torch.stack([torch.full((H * W,), before), before + r.flatten(), before + c.flatten()])
    tail = (before + max(H, W) + torch.arange(after)).repeat(3, 1)
    mrope = torch.cat([t, img, tail], 1)[:, None]
    pos = torch.cat([torch.arange(L).view(1, 1, L), mrope], 0)
    grid = torch.tensor([[1, 2 * H, 2 * W]])
    interp = get_vision_interpolation_indices_and_weights(grid, 48, mode='bilinear', align_corners=True,
                                                          spatial_merge_size=2)
    return cr.Prepared({}, torch.zeros(L, dtype=torch.long), {'start': before, 'end': before + H * W, 'grid': (H, W)},
                       get_vision_position_ids(grid, 2), interp, pos)


class StimulusTests(unittest.TestCase):
    def test_disc_truth_box_is_centered(self):
        _, truth = cr.disc_image((200, 300), 64)
        self.assertEqual(truth['box'], [168.0, 268.0, 232.0, 332.0])
        self.assertEqual(truth['center'], [200.0, 300.0])

    def test_tokens_overlapping_half_open_box(self):
        self.assertEqual(cr.tokens_overlapping([32, 32, 64, 64], (16, 16)), [17])
        self.assertEqual(cr.tokens_overlapping([30, 0, 34, 10], (16, 16)), [0, 1])

    def test_case_counts(self):
        cases = disc_cases(0)
        sets = {s: sum(c['set'] == s for c in cases) for s in ('grid', 'trans', 'size')}
        self.assertEqual(sets, {'grid': 216, 'trans': 86, 'size': 42})
        for c in cases:
            if c['set'] == 'grid':
                continue                     # outer grid positions may clip large discs; truth uses the visible box
            r = c['diameter'] / 2
            self.assertTrue(r <= c['center'][0] <= c['canvas'] - r and r <= c['center'][1] <= c['canvas'] - r)


class ParseTests(unittest.TestCase):
    def test_point_and_bbox_wrappers(self):
        self.assertEqual(cr.parse_coords('```json\n[{"point_2d": [1, 2], "label": "x"}]\n```', 'point'), [1.0, 2.0])
        self.assertEqual(cr.parse_coords('{"bbox_2d": [1, 2, 3, 4]}', 'bbox'), [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(cr.parse_coords('[{"point_2d": [1]}]', 'point'))
        self.assertEqual(cr.parse_rowcol('{"row": 3, "column": 12}'), (3, 12))

    def test_disc_score(self):
        truth = {'canvas': [512, 512], 'center': [256.0, 256.0], 'diameter': 64, 'box': [224, 224, 288, 288]}
        s = cr.score_disc(truth, 'point', '[{"point_2d": [500, 560]}]')
        self.assertTrue(s['hit'])
        self.assertAlmostEqual(s['error'], 60.0)


class PositionEditTests(unittest.TestCase):
    def test_gap_and_shift_keep_text_ids_aligned(self):
        p = fake_prepared()
        gap = cr.gap_pos(p, 8)
        s, e = p.span['start'], p.span['end']
        self.assertTrue(torch.equal(gap[:, :, :s], p.pos[:, :, :s]))
        self.assertTrue(torch.equal(gap[1:, :, s:e], p.pos[1:, :, s:e] + 8))
        self.assertTrue(torch.equal(gap[1:, :, e:], p.pos[1:, :, e:] + 16))
        sh = cr.shift_image_pos(p, gap, 2, -3)
        self.assertTrue(torch.equal(sh[2, :, s:e], gap[2, :, s:e] + 2))
        self.assertTrue(torch.equal(sh[3, :, s:e], gap[3, :, s:e] - 3))
        self.assertTrue(torch.equal(sh[:, :, e:], gap[:, :, e:]))

    def test_swap_and_permutation(self):
        p = fake_prepared()
        a, b = [0, 1], [2, 3]
        sw = cr.swap_image_pos(p, p.pos, a, b)
        s = p.span['start']
        self.assertTrue(torch.equal(sw[2:, :, s + 0], p.pos[2:, :, s + 2]))
        order = cr.swap_index(16, a, b)
        self.assertEqual(order[:4].tolist(), [2, 3, 0, 1])
        perm = cr.permute_pos(p, p.pos, order)
        self.assertTrue(torch.equal(perm[1:, :, s], p.pos[1:, :, s + 2]))
        with self.assertRaises(ValueError):
            cr.swap_index(16, [0, 1], [1, 2])

    def test_scale_about_image_start(self):
        p = fake_prepared()
        sc = cr.scale_image_pos(p, p.pos, 1.0, 0.5)
        s, e = p.span['start'], p.span['end']
        self.assertTrue(torch.allclose(sc[3, :, s:e], s + 0.5 * (p.pos[3, :, s:e].float() - s)))

    def test_vit_swaps_and_centered_pe_window(self):
        p = fake_prepared()
        vp, (idx, w) = cr.swap_vit(p, [0], [5], pe=False, rope=True)
        self.assertTrue(torch.equal(vp[0:4], p.vit_pos[20:24]))
        self.assertTrue(torch.equal(idx, p.interp[0]))
        i0, w0 = cr.shifted_pe(p, 0, 0, 0)
        self.assertTrue(torch.equal(i0, p.interp[0]))
        self.assertTrue(torch.allclose(w0, p.interp[1]))
        with self.assertRaises(ValueError):
            cr.shifted_pe(p, 2, 3, 0)

    def test_swap_regions_stay_inside_and_disjoint(self):
        target = cr.tokens_overlapping([200, 200, 264, 264], (16, 16))
        for reg in swap_regions(target, (16, 16), (232, 232), 512):
            if reg['valid']:
                self.assertFalse(set(reg['region']) & set(target))
                self.assertTrue(all(0 <= t < 256 for t in reg['region']))


class MetricTests(unittest.TestCase):
    def test_displacement_weight(self):
        w, e = displacement_weight(np.array([50., 0.]), np.array([100., 0.]))
        self.assertAlmostEqual(w, 0.5)
        self.assertAlmostEqual(e, 0.0)
        w, e = displacement_weight(np.array([0., 100.]), np.array([100., 0.]))
        self.assertAlmostEqual(w, 0.0)
        self.assertAlmostEqual(e, 1.0)

    def test_wilson_bounds(self):
        lo, hi = wilson(5, 10)
        self.assertTrue(0 < lo < 0.5 < hi < 1)

    def test_token_centers(self):
        c = cr.token_centers((2, 4))
        np.testing.assert_allclose(c[0], [125, 250])
        np.testing.assert_allclose(c[-1], [875, 750])


if __name__ == '__main__':
    unittest.main()


class SupplementTests(unittest.TestCase):
    """Model-free checks of the C4 patching hooks, RefCOCO scoring and C5 insertion helpers."""

    def test_insert_band_moves_content(self):
        from PIL import Image
        im = Image.new('RGB', (64, 64), (255, 255, 255))
        im.putpixel((10, 40), (255, 0, 0))
        out = cr.insert_band(im, 'v', 32, 32)
        self.assertEqual(out.size, (64, 96))
        self.assertEqual(out.getpixel((10, 72)), (255, 0, 0))
        out = cr.insert_band(im, 'h', 0, 32)
        self.assertEqual(out.getpixel((42, 40)), (255, 0, 0))

    def test_insert_ids_shifts_rows_and_tail(self):
        prep = fake_prepared(4, 4, before=3, after=2)
        pos = cr.insert_ids(prep, 'v', 2, 2)
        s, e = 3, 19
        h = pos[2, 0, s:e].view(4, 4)[:, 0].tolist()
        self.assertEqual(h, [3, 4, 7, 8])
        self.assertTrue(torch.equal(pos[2:, 0, s:e].view(2, 4, 4)[1], prep.pos[3, 0, s:e].view(4, 4)))
        self.assertEqual(pos[1:, 0, e].tolist(), [(prep.pos[1, 0, e] + 2).item()] * 3)

    def test_point_conventions(self):
        from analysis_tools.position.coordinate_readout_supp import point_conventions
        box = [100, 100, 200, 200]
        r = point_conventions([300, 300], box, (500, 500))      # norm -> (150, 150): hit; pixel (300, 300): miss
        self.assertTrue(r['h_norm'] and not r['h_pix'] and r['h_any'])
        r = point_conventions([150, 150], box, (500, 500))      # norm -> (75, 75): miss; pixel: hit
        self.assertTrue(r['h_pix'] and not r['h_norm'])
        self.assertAlmostEqual(r['chance_norm'], 0.04)
        self.assertAlmostEqual(r['chance_pix'], 0.01)
        self.assertAlmostEqual(r['chance_any'], 0.05)          # regions [200,400]^2 and [100,200]^2 are disjoint

    def test_refcoco_swaps_disjoint(self):
        from analysis_tools.position.coordinate_readout_supp import refcoco_swaps
        target = [r * 10 + c for r in (1, 2) for c in (2, 3, 4)]
        regs = refcoco_swaps(target, (8, 10))
        for g in regs:
            self.assertTrue(g['valid'])
            self.assertFalse(set(g['region']) & set(target))
        self.assertEqual(regs[0]['dc'], 4)
        self.assertFalse(refcoco_swaps(list(range(10)) + list(range(10, 20)), (3, 10))[1]['valid'])

    def test_mixer_edit_copies_rows(self):
        class Proj(torch.nn.Module):
            def forward(self, x):
                return x
        class Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj, self.head_dim = Proj(), 2
        class Block(torch.nn.Module):
            block_type = 'full_attention'
            def __init__(self):
                super().__init__()
                self.self_attn = Attn()
        class RT:
            layers = [Block()]
        rt = RT()
        x = torch.arange(2 * 3 * 4, dtype=torch.float).view(2, 3, 4)
        with cr.MixerEdit(rt, [(1, 1, 1, 0)], start=1):
            y = rt.layers[0].self_attn.o_proj(x)
        self.assertTrue(torch.equal(y[1, 1:, 2:], x[0, 1:, 2:]))
        self.assertTrue(torch.equal(y[1, :1], x[1, :1]))
        self.assertTrue(torch.equal(y[1, :, :2], x[1, :, :2]))
