"""Unit tests for experiment 4.5 helpers (no model weights needed)."""
from types import SimpleNamespace
import random
import unittest

import numpy as np
import torch
from PIL import Image

from analysis_tools.deepstack import deepstack_natural as dn


def fake_rt(n_layers=3):
    visual = SimpleNamespace(deepstack_visual_indexes=list(range(n_layers)))
    return SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(visual=visual)))


class GeometryTest(unittest.TestCase):
    def test_resize32_multiples_and_budget(self):
        im, sx, sy = dn.resize32(Image.new('RGB', (1700, 2000)), 1024 * 1024)
        self.assertEqual(im.width % 32, 0)
        self.assertEqual(im.height % 32, 0)
        self.assertLessEqual(im.width * im.height, 1024 * 1024)
        self.assertAlmostEqual(sx, im.width / 1700)
        small, sx, _ = dn.resize32(Image.new('RGB', (330, 260)), 1024 * 1024)
        self.assertEqual(small.size, (320, 256))
        tiny, _, _ = dn.resize32(Image.new('RGB', (224, 224)), 1024 * 1024)
        self.assertEqual(tiny.size, (256, 256))
        self.assertGreaterEqual(dn.resize32(Image.new('RGB', (128, 128)), 1024 * 1024)[0].size[0] ** 2, dn.MIN_PIXELS)

    def test_token_mask_covers_overlapping_cells(self):
        m = dn.token_mask([[40, 10, 70, 20]], 128, 64)       # columns 1-2 of row 0
        self.assertEqual(m.shape, (2, 4))
        self.assertEqual(np.argwhere(m).tolist(), [[0, 1], [0, 2]])
        tiny = dn.token_mask([[33, 33, 33.5, 33.5]], 128, 64)
        self.assertEqual(int(tiny.sum()), 1)

    def test_control_mask_same_shape_no_overlap(self):
        m = np.zeros((6, 8), bool)
        m[1:3, 2:5] = True
        c = dn.control_mask(m, random.Random(0))
        self.assertEqual(int(c.sum()), int(m.sum()))
        self.assertFalse((c & m).any())
        full = np.ones((2, 2), bool)
        self.assertIsNone(dn.control_mask(full, random.Random(0)))


class OverrideTest(unittest.TestCase):
    def test_combine_replaces_masked_tokens_only(self):
        full = [torch.zeros(6, 2)] * 3
        blank = [torch.ones(6, 2)] * 3
        masks = [np.array([True, False, False, False]), np.array([False, True])]
        out = dn.NaturalOverride.combine(full, blank, masks)
        self.assertEqual(out[0][:, 0].tolist(), [1, 0, 0, 0, 0, 1])
        with self.assertRaises(ValueError):
            dn.NaturalOverride.combine(full, blank, [np.array([True])])

    def test_region_masks_follow_condition(self):
        case = {'id': 'x', 'width': 64, 'height': 64, 'region': [0], 'control': [3]}
        self.assertEqual(dn.NaturalOverride(fake_rt(), 'blank_in').region_masks([case])[0].tolist(),
                         [True, False, False, False])
        self.assertEqual(dn.NaturalOverride(fake_rt(), 'blank_out').region_masks([case])[0].tolist(),
                         [False, True, True, True])
        self.assertEqual(dn.NaturalOverride(fake_rt(), 'blank_ctrl').region_masks([case])[0].tolist(),
                         [False, False, False, True])
        with self.assertRaises(ValueError):
            dn.NaturalOverride(fake_rt(), 'blank_ctrl').region_masks([dict(case, control=None)])

    def test_unknown_condition_and_consumption(self):
        with self.assertRaises(ValueError):
            dn.NaturalOverride(fake_rt(), 'blur3')
        ov = dn.NaturalOverride(fake_rt(), 'blank_in')
        embeds = [torch.ones(4, 2)] * 3
        with self.assertRaises(ValueError):
            ov.replace(embeds)
        ov.prepared = [torch.zeros(4, 2)] * 3
        self.assertEqual(float(ov.replace(embeds)[0].sum()), 0.0)
        with self.assertRaises(ValueError):
            ov.replace(embeds)


class PositionOnlyTest(unittest.TestCase):
    def test_position_only_centres_per_image_and_adds_mean(self):
        grids = [[1, 2, 4], [1, 2, 2]]                    # 2 tokens and 1 token
        blank = torch.tensor([[1.0], [3.0], [5.0]])
        out = dn.position_only(blank, grids, torch.tensor([10.0]))
        self.assertEqual(out[:, 0].tolist(), [9.0, 11.0, 10.0])
        with self.assertRaises(ValueError):
            dn.position_only(torch.zeros(4, 1), grids, torch.tensor([0.0]))


class ScoringTest(unittest.TestCase):
    def test_norm_and_anls(self):
        self.assertEqual(dn.norm_answer('The Nokia.\nextra'), 'nokia')
        self.assertEqual(dn.anls('P. Carter', ['P. Carter']), 1.0)
        self.assertGreater(dn.anls('P. Cartr', ['P. Carter']), 0.8)
        self.assertEqual(dn.anls('xyz', ['P. Carter']), 0.0)

    def test_parse_letter(self):
        self.assertEqual(dn.parse_letter('(b) Closed'), 'b')
        self.assertEqual(dn.parse_letter('a'), 'a')
        self.assertIsNone(dn.parse_letter('Closed'))

    def test_score_case(self):
        cub = {'id': 'c', 'task': 'cub', 'answers': ['No']}
        self.assertTrue(dn.score_case(cub, {'response': 'No', 's': -1.0, 'finish_reason': 'eos'})['correct'])
        tv = {'id': 't', 'task': 'textvqa', 'answers': ['nokia']}
        row = dn.score_case(tv, {'response': 'Nokia', 'finish_reason': 'eos', 'gold_logp': -0.1})
        self.assertTrue(row['correct'])
        self.assertEqual(row['gold_logp'], -0.1)


if __name__ == '__main__':
    unittest.main()
