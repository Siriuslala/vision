"""CPU tests of experiment 5.4 (learned 2x module, condition geometry, DocVQA scoring)."""
import unittest

import torch
from PIL import Image

from analysis_tools.reread import cache_only as co
from analysis_tools.reread import cached_reread as cr
from analysis_tools.reread import reread_pilot as rp
from analysis_tools.reread import reread_pilots as P


class ConcatMerger(torch.nn.Module):
    """Stand-in merger: concatenates each group of 4 patch states, so slot contents are observable."""
    def forward(self, x):
        return x.reshape(-1, 4 * x.shape[-1])


def fake_feats(H, W, d=3, D=4):
    P27 = torch.randn(2 * H, 2 * W, d)
    return {'grid': (H, W), 'vis': torch.randn(H * W, D), 'patch_27': P27, 'patch_0': torch.randn(2 * H, 2 * W, d)}


class ModuleTest(unittest.TestCase):
    def test_zero_init_reproduces_nearest_reader(self):
        f = fake_feats(3, 5, d=8)
        m = P.Upsampler2x(2, d=8, c=16, n_lo=1, n_hi=1)
        out = P.learned_tokens(m, ConcatMerger(), f, (27, 0))
        ref, grid, step = co.read_window(ConcatMerger(), f['patch_27'], rp.Window(0, 0, 3, 5), 'nearest')
        self.assertEqual(grid, (6, 10))
        self.assertEqual(step, 0.5)
        self.assertTrue(torch.equal(out, ref))

    def test_output_shape_and_locality(self):
        m = P.Upsampler2x(1, d=8, c=16, n_lo=1, n_hi=1)
        torch.nn.init.normal_(m.head.weight)
        x = torch.randn(10, 12, 8)
        y = m([x])
        self.assertEqual(tuple(y.shape), (20, 24, 8))
        x2 = x.clone()
        x2[0, 0] += 10.0                         # a corner change must not reach the far corner
        self.assertTrue(torch.allclose(m([x2])[-1, -1], y[-1, -1], atol=1e-5))

    def test_param_count_is_small(self):
        self.assertLess(P.n_params(P.Upsampler2x(2)), 12_000_000)


class ConditionTest(unittest.TestCase):
    def test_ratio(self):
        self.assertEqual(P.ratio('lo', 'hi'), 2)
        self.assertEqual(P.ratio('hi', 'hi'), 1)
        with self.assertRaises(ValueError):
            P.ratio('hi', 'lo')

    def test_tool_tokens_and_cap(self):
        win = rp.Window(0, 0, 10, 10)
        self.assertEqual(P.tool_tokens(P.OPT2['lo_c27'], win), 100)
        self.assertEqual(P.tool_tokens(P.OPT2['lo_cache_hi'], win), 400)
        self.assertEqual(P.tool_tokens(P.OPT2['lo_crop_hi'], win), 400)
        self.assertEqual(P.tool_tokens(P.OPT2['hi_crop1'], win), 100)
        self.assertEqual(P.tool_tokens(P.OPT1['l27'], win), 400)
        self.assertFalse(P.capped(P.OPT2['lo_cache_hi'], win))
        big = rp.Window(0, 0, 21, 20)              # 420 tokens -> 1680 > 1600
        self.assertTrue(P.capped(P.OPT2['lo_cache_hi'], big))
        self.assertFalse(P.capped(P.OPT2['hi_c27'], big))

    def test_cache_hi_slice_matches_hi_tokens(self):
        Hl, Wl = 4, 6
        lo, hi = fake_feats(Hl, Wl), fake_feats(2 * Hl, 2 * Wl)
        win = rp.Window(1, 2, 2, 3)
        seg, cost = P.window_tokens(P.OPT2['lo_cache_hi'], win, {}, {'lo': lo, 'hi': hi}, {}, None, None, None)
        self.assertEqual(seg.grid, (4, 6))
        self.assertEqual(seg.step, 0.5)
        self.assertEqual(seg.origin, (1.0, 2.0))
        hiv = hi['vis'].view(2 * Hl, 2 * Wl, -1)
        self.assertTrue(torch.equal(seg.emb.view(4, 6, -1), hiv[2:6, 4:10]))
        self.assertEqual(cost['vit_patches'], 0)

    def test_learned_slice(self):
        lo = fake_feats(4, 6)
        T = torch.randn(8 * 12, 4)
        win = rp.Window(1, 1, 2, 2)
        seg, _ = P.window_tokens(P.OPT1['l27'], win, {}, {'lo': lo}, {'l27': T}, None, None, None)
        self.assertTrue(torch.equal(seg.emb.view(4, 4, -1), T.view(8, 12, -1)[2:6, 2:6]))

    def test_passes(self):
        self.assertEqual(P.OPT2['lo_crop_hi'].passes(), {'lo'})
        self.assertEqual(P.OPT2['lo_cache_hi'].passes(), {'lo', 'hi'})
        self.assertEqual(P.OPT2['hi_c27'].passes(), {'hi'})
        self.assertEqual(P.OPT1['h27'].passes(), {'lo', 'up'})


class DocTest(unittest.TestCase):
    def test_anls(self):
        self.assertEqual(P.anls('P. Carter', ['P. Carter']), 1.0)
        self.assertAlmostEqual(P.anls('P. Carte', ['P. Carter']), 1 - 1 / 9)
        self.assertEqual(P.anls('completely wrong', ['P. Carter']), 0.0)
        self.assertEqual(P.anls('p.  carter', ['x', 'P. Carter']), 1.0)

    def test_doc_images_double(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'a.png')
            Image.new('RGB', (1700, 2200), (255, 255, 255)).save(path)
            lo, hi = P.doc_images(path, 1024 * 1024)
            self.assertEqual((hi.width, hi.height), (2 * lo.width, 2 * lo.height))
            self.assertEqual(lo.width % 32, 0)
            self.assertLessEqual(lo.width * lo.height, 1024 * 1024)

    def test_score_doc(self):
        c = cr.Case('x', 'docvqa', 'q', ['T.F. Riehl'], P.DOC_FMT)
        s = P.score_doc(c, 'Final answer: T.F. Riehl')
        self.assertEqual(s['correct'], 1.0)
        self.assertEqual(s['em'], 1.0)


if __name__ == '__main__':
    unittest.main()
