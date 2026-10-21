"""Unit tests for experiment 4.4 helpers (no model weights needed)."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from PIL import Image

from analysis_tools.deepstack import deepstack_todo as dt
from analysis_tools.deepstack import deepstack_todo_analysis as dta


def fake_rt(n_layers=3):
    visual = SimpleNamespace(deepstack_visual_indexes=list(range(n_layers)))
    return SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(visual=visual)))


class ConditionTest(unittest.TestCase):
    def test_condition_names(self):
        self.assertEqual(dt.blur_sigma('blur8'), 8.0)
        self.assertIsNone(dt.blur_sigma('blank'))
        with self.assertRaises(ValueError):
            dt.TodoOverride(fake_rt(), 'blur3')
        with self.assertRaises(ValueError):
            dt.TodoOverride(fake_rt(), 'mean')

    def test_flip_tokens_mirrors_each_image(self):
        # Two images: merged grids 2x3 and 1x2 (image_grid_thw is in patches, i.e. twice the token grid).
        grids = [[1, 4, 6], [1, 2, 4]]
        feats = torch.arange(8.0)[:, None]
        out = dt.flip_tokens(feats, grids)[:, 0].tolist()
        self.assertEqual(out, [2, 1, 0, 5, 4, 3, 7, 6])
        with self.assertRaises(ValueError):
            dt.flip_tokens(torch.zeros(7, 1), grids)

    def test_blank_like_keeps_size_and_mean(self):
        im = Image.new('RGB', (64, 32), (0, 0, 0))
        im.paste((255, 255, 255), (0, 0, 32, 32))
        b = dt.blank_like(im)
        self.assertEqual(b.size, (64, 32))
        self.assertEqual(b.getpixel((10, 10)), (128, 128, 128))

    def test_prepared_features_are_consumed(self):
        ov = dt.TodoOverride(fake_rt(), 'blank')
        embeds = [torch.ones(4, 2)] * 3
        with self.assertRaises(ValueError):
            ov.replace(embeds)
        ov.prepared = [torch.zeros(4, 2)] * 3
        out = ov.replace(embeds)
        self.assertEqual(float(out[0].abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            ov.replace(embeds)
        self.assertTrue(torch.equal(dt.TodoOverride(fake_rt(), 'full').replace(embeds)[1], embeds[1]))
        self.assertEqual(float(dt.TodoOverride(fake_rt(), 'off').replace(embeds)[2].sum()), 0.0)


class TargetTest(unittest.TestCase):
    def test_token_targets_shapes_and_order(self):
        im = Image.new('RGB', (96, 64), (255, 255, 255))
        im.paste((0, 0, 0), (64, 32, 96, 64))       # bottom-right token is black
        tg = dt.token_targets(im)
        self.assertEqual(tg['grid'], (2, 3))
        self.assertEqual(tg['highpass'].shape, (6, 1024))
        self.assertEqual(tg['coarse'].shape, (6, 48))
        self.assertAlmostEqual(float(tg['grey'][5].mean()), 0.0, places=5)
        np.testing.assert_allclose(tg['xy'][5], [5 / 6, 3 / 4])

    def test_detail_targets(self):
        im = Image.new('RGB', (64, 32), (255, 255, 255))
        im.paste((0, 0, 0), (0, 0, 16, 32))           # left half of token 0 is black
        t = dt.detail_targets(im)
        self.assertEqual(t['structure'].shape, (2, 64))
        self.assertEqual(t['hfenergy'].shape, (2, 16))
        self.assertAlmostEqual(float(t['structure'][0].mean()), 0.0, places=5)
        self.assertLess(t['structure'][0, 0], 0)      # dark cell below the token mean
        np.testing.assert_allclose(t['structure'][1], 0, atol=1e-6)
        self.assertGreater(t['hfenergy'][0].max(), t['hfenergy'][1].max())   # the edge carries high-pass energy

    def test_select_tokens_prefers_content(self):
        content = np.zeros(200, bool)
        content[:30] = True
        idx = dt.select_tokens(content, np.random.default_rng(0), n=64)
        self.assertEqual(len(idx), 64)
        self.assertEqual(int(content[idx].sum()), 30)

    def test_vqa_and_pope_scoring(self):
        case = {'id': 'x', 'task': 'TextVQA', 'truth': {'answers': ['Dakota'] * 3 + ['other'] * 7}}
        self.assertTrue(dt.score_general(case, {'response': 'dakota.'})['correct'])
        pope = {'id': 'p', 'task': 'POPE', 'truth': {'answer_yes': False}}
        self.assertTrue(dt.score_general(pope, {'s': -1.0})['correct'])


class RidgeTest(unittest.TestCase):
    def test_ridge_recovers_linear_map(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 20)).astype(np.float32)
        Y = X[:, :2] * 3 + 1
        f = dta.ridge_fit(X, Y, [1e-6])[1e-6]
        self.assertGreater(dta.r2(Y, f(X), Y.mean(0)), 0.999)


if __name__ == '__main__':
    unittest.main()
