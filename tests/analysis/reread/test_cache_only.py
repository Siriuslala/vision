"""CPU tests of the experiment 5.3 cache-only readers (patch-grid order, merge variants, ids, FLOPs)."""
import types
import unittest

import torch

from analysis_tools.reread import cache_only as co
from analysis_tools.reread import cached_reread as cr
from analysis_tools.reread import reread_pilot as rp


class SumMerger(torch.nn.Module):
    """Stand-in merger: concatenates each group of 4 patches (so slot order is observable)."""
    def forward(self, x):
        return x.view(-1, 4 * x.shape[-1])


def coded_grid(H, W):
    """Patch states in processor group order whose value encodes (row, col) of the patch."""
    x = torch.zeros(4 * H * W, 2)
    for r in range(H):
        for c in range(W):
            for a in range(2):
                for b in range(2):
                    x[4 * (r * W + c) + 2 * a + b] = torch.tensor([2 * r + a, 2 * c + b], dtype=torch.float)
    return x


class GridTest(unittest.TestCase):
    def test_group_to_grid_positions(self):
        P = co.group_to_grid(coded_grid(3, 4), (3, 4))
        self.assertEqual(tuple(P.shape), (6, 8, 2))
        for i in range(6):
            for j in range(8):
                self.assertEqual(P[i, j].tolist(), [i, j])

    def test_grid_merge_reproduces_groups(self):
        x = coded_grid(3, 4)
        P = co.group_to_grid(x, (3, 4))
        tok, gr, step = co.read_window(SumMerger(), P, rp.Window(0, 0, 3, 4), 'grid')
        self.assertEqual(gr, (3, 4))
        self.assertEqual(step, 1.0)
        self.assertTrue(torch.equal(tok, x.view(12, 8)))

    def test_shift_contains_grid_tokens(self):
        P = co.group_to_grid(coded_grid(4, 4), (4, 4))
        win = rp.Window(1, 1, 2, 3)
        sh, sg, step = co.read_window(SumMerger(), P, win, 'shift')
        self.assertEqual(sg, (3, 5))
        self.assertEqual(step, 0.5)
        gt, _, _ = co.read_window(SumMerger(), P, win, 'grid')
        self.assertTrue(torch.equal(sh.view(3, 5, -1)[::2, ::2].reshape(-1, 8), gt))
        # the token at shift offset (1, 1) covers patches rows 3-4, cols 3-4
        self.assertEqual(sh.view(3, 5, -1)[1, 1].tolist(), [3, 3, 3, 4, 4, 3, 4, 4])

    def test_nearest_one_patch_per_token(self):
        P = co.group_to_grid(coded_grid(2, 2), (2, 2))
        tok, gr, _ = co.read_window(SumMerger(), P, rp.Window(0, 0, 2, 2), 'nearest')
        self.assertEqual(gr, (4, 4))
        self.assertEqual(tok[5].tolist(), [1, 1] * 4)

    def test_bilinear_matches_whole_image_upsampling(self):
        torch.manual_seed(0)
        P = torch.randn(8, 10, 3)
        win = rp.Window(1, 1, 2, 3)
        tok, gr, _ = co.read_window(SumMerger(), P, win, 'bilinear')
        self.assertEqual(gr, (4, 6))
        up = torch.nn.functional.interpolate(P.permute(2, 0, 1)[None], scale_factor=2, mode='bilinear',
                                             align_corners=False)[0].permute(1, 2, 0)
        ref, _ = co.groups(up[4:12, 4:16], 2)
        self.assertTrue(torch.allclose(tok, ref.reshape(ref.shape[0], -1), atol=1e-6))

    def test_orig_ids_of_shift_segment(self):
        segs = [cr.VSeg(torch.zeros(16, 2), (4, 4)),
                cr.VSeg(torch.zeros(9, 2), (3, 3), 'orig', (1.0, 2.0), 0.5)]
        pos = cr.mrope_float([-1] + [0] * 16 + [-1] + [1] * 9 + [-1], segs)
        self.assertEqual(pos[1, 18:21].tolist(), [2.0, 2.0, 2.0])          # first + r0 + i * 0.5, i = 0
        self.assertEqual(pos[2, 18:21].tolist(), [3.0, 3.5, 4.0])          # first + c0 + j * 0.5
        self.assertEqual(pos[1, 24].item(), 3.0)                           # i = 2


class CharTok:
    """Character-level stand-in tokenizer (one id per character)."""
    def decode(self, ids):
        return ''.join(chr(i) for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(ch) for ch in text]}


class CloseCallTest(unittest.TestCase):
    def test_close_call_appends_tail(self):
        rt = types.SimpleNamespace(tokenizer=CharTok())
        gen = [ord(c) for c in '10, 20, 300, 400]']
        ids, text = co.close_call(rt, gen)
        self.assertEqual(text, '10, 20, 300, 400]' + co.CALL_CLOSE)
        self.assertEqual(cr.parse_call(co.CALL_PREFIX + text), [10.0, 20.0, 300.0, 400.0])

    def test_close_call_skips_tail_already_decoded(self):
        rt = types.SimpleNamespace(tokenizer=CharTok())
        ids, text = co.close_call(rt, [ord(c) for c in '1, 2, 3, 4]\n'])
        self.assertEqual(text, '1, 2, 3, 4]' + co.CALL_CLOSE)


class CostTest(unittest.TestCase):
    def test_flops_scale(self):
        v = types.SimpleNamespace(hidden_size=1152, intermediate_size=4304, depth=27, in_channels=3,
                                  temporal_patch_size=2, patch_size=16, out_hidden_size=4096)
        f1, f4 = co.vit_flops(1024, v), co.vit_flops(4096, v)
        self.assertGreater(f4, 4 * f1)                                     # attention is quadratic
        self.assertLess(co.merger_flops(256, v), 0.05 * f1)

    def test_conditions_compatible_with_5_2(self):
        for c in co.CONDITIONS.values():
            for k in ('name', 'kind', 'scale', 'site', 'ids', 'oracle', 'order'):
                self.assertTrue(hasattr(c, k))
        self.assertEqual(co.cache_blocks([co.CONDITIONS['c27'], co.CONDITIONS['s25'], co.CONDITIONS['u27']]), (25, 27))
        self.assertFalse(co.CONDITIONS['s27'].extra_vit)
        self.assertTrue(co.CONDITIONS['crop1'].extra_vit)


if __name__ == '__main__':
    unittest.main()
