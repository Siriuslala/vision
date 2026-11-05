"""CPU tests of experiment 7.1 (image-block mask, contrast adapter, prior-contrast processor, data helpers)."""
import math
import unittest

import torch
from PIL import Image

from analysis_tools.application import interp_opt as io_

IMG = 9


class TestMask(unittest.TestCase):
    def setUp(self):
        # row 0: text, 3 image tokens, text, text; row 1: left pad, text, 2 image tokens, text, text
        self.ids = torch.tensor([[1, IMG, IMG, IMG, 2, 3], [0, 1, IMG, IMG, 2, 3]])
        self.am = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])

    def test_segments(self):
        seg = io_.image_segments(torch.tensor([[IMG, IMG, 1, IMG, 2]]), IMG)
        self.assertEqual(seg.tolist(), [[1, 1, 0, 2, 0]])

    def test_causal_mode_equals_tril_with_padding(self):
        m = io_.prefill_mask(self.ids, self.am, IMG, bidirectional=False)[:, 0]
        self.assertTrue(torch.equal(m[0], torch.ones(6, 6, dtype=torch.bool).tril()))
        self.assertFalse(m[1, 3, 0])            # padding key is never attended
        self.assertTrue(m[1, 0, 0])             # padding query keeps its diagonal

    def test_bidirectional_only_inside_image(self):
        m = io_.prefill_mask(self.ids, self.am, IMG)[:, 0]
        self.assertTrue(m[0, 1, 3])             # image -> later image token allowed
        self.assertFalse(m[0, 1, 4])            # image -> later text still masked
        self.assertFalse(m[0, 0, 1])            # text -> later image still masked
        self.assertTrue(m[0, 5, 1])             # text -> earlier image allowed
        self.assertTrue(m[1, 2, 3] and not m[1, 2, 4])

    def test_two_images_do_not_see_each_other_forward(self):
        ids = torch.tensor([[IMG, IMG, 1, IMG, IMG]])
        m = io_.prefill_mask(ids, None, IMG)[0, 0]
        self.assertTrue(m[0, 1])
        self.assertFalse(m[0, 3])               # first image cannot read the later image
        self.assertTrue(m[3, 0])                # causal reading of the earlier image remains


class TestContrastAdapter(unittest.TestCase):
    def test_zero_init_identity_and_local_mean(self):
        torch.manual_seed(0)
        v = torch.randn(12, 8)
        ad = io_.ContrastAdapter(8, rank=4, k=3)
        self.assertTrue(torch.allclose(ad(v, (3, 4)), v))
        const = torch.ones(12, 8)
        self.assertTrue(torch.allclose(io_.local_mean(const, (3, 4), 3), const))

    def test_signed_deviation_reaches_output(self):
        torch.manual_seed(0)
        v = torch.zeros(16, 4)
        v[5] = -1.0                              # an "absent" token: negative deviation
        ad = io_.ContrastAdapter(4, rank=2, k=3)
        with torch.no_grad():
            ad.B.fill_(1.0)
        e = ad(v, (4, 4)) - v
        self.assertGreater(float(e[5].norm()), float(e[0].norm()))


class TestPriorContrast(unittest.TestCase):
    def test_subtracts_prior_and_respects_plausibility(self):
        real = torch.log(torch.tensor([[0.50, 0.45, 0.05]]))
        blank = torch.log(torch.tensor([[0.90, 0.05, 0.05]]))
        p = io_.PriorContrastProcessor(alpha=1.0, beta=0.1)
        s = p.combine(real, blank)
        self.assertEqual(int(s.argmax()), 1)    # the prior-favoured token 0 loses
        p2 = io_.PriorContrastProcessor(alpha=1.0, beta=0.2)
        self.assertTrue(math.isinf(float(p2.combine(real, blank)[0, 2])))

    def test_token_subset_and_pairing(self):
        real = torch.log(torch.tensor([[0.50, 0.45, 0.05]]))
        blank = torch.log(torch.tensor([[0.90, 0.05, 0.05]]))
        p = io_.PriorContrastProcessor(alpha=1.0, beta=0.1, token_ids=[2])
        s = p.combine(real, blank)
        self.assertAlmostEqual(float(s[0, 0]), math.log(0.5), places=5)
        ids = torch.tensor([[1, 2], [1, 2]])
        out = p(ids, torch.cat([real, blank]))
        self.assertTrue(torch.equal(out[0], out[1]))


class TestHelpers(unittest.TestCase):
    def test_fit_area(self):
        im = io_.fit_area(Image.new('RGB', (1024, 730)))
        self.assertEqual((im.width % 32, im.height % 32), (0, 0))
        self.assertLessEqual(im.width * im.height, 512 * 512)

    def test_blank_like(self):
        im = Image.new('RGB', (64, 32), (10, 20, 30))
        self.assertEqual(io_.blank_like(im).getpixel((5, 5)), (10, 20, 30))

    def test_score_count_and_stats(self):
        r = io_.score_count('```json\n{"rows": 5, "columns": 7}\n```', {'rows': 5, 'cols': 8})
        self.assertTrue(r['row_ok'] and not r['correct'] and r['d_cols'] == -1)
        self.assertAlmostEqual(io_.mcnemar_exact(0, 6), 0.03125)
        p, lo, hi = io_.wilson(5, 10)
        self.assertTrue(lo < p < hi)


if __name__ == '__main__':
    unittest.main()
