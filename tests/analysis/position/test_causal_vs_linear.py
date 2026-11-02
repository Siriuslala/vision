"""CPU tests of experiment 3.14 (masks, profile groups, recovery, toy models, probe statistics)."""
import unittest

import numpy as np
import torch

from analysis_tools.position import causal_vs_linear as cv
from analysis_tools.position import causal_vs_linear_analysis as ca
from analysis_tools.position import causal_vs_linear_toy as toy


class TestMasks(unittest.TestCase):
    def test_knockout_img_keeps_self_text_and_causality(self):
        L, span, grid = 12, (2, 8), (2, 3)
        m = cv.knockout_mask(L, span, 'img', grid, 'cpu')[0, 0]
        self.assertFalse(m[0, 1])                                  # causal
        self.assertTrue(m[5, 5])                                   # self
        self.assertFalse(m[5, 3])                                  # image -> earlier image removed
        self.assertTrue(m[5, 0] and m[5, 1])                       # image -> text kept
        self.assertTrue(m[10, 4])                                  # text query still reads the image

    def test_knockout_local_same_row_only(self):
        L, span, grid = 12, (0, 10), (2, 5)
        m = cv.knockout_mask(L, span, 'local', grid, 'cpu')[0, 0]
        self.assertFalse(m[3, 2] or m[3, 1] or m[3, 0])            # left neighbours within 3 in row 0
        self.assertTrue(m[5, 4])                                   # token 5 starts row 1: token 4 is the previous row
        self.assertTrue(m[9, 5])                                   # distance 4 in row 1 is kept

    def test_profile_masks_partition(self):
        m = cv.profile_masks((4, 5), [7])                          # row 1, col 2
        self.assertEqual(int(m['row_after'].sum()), 2)
        self.assertEqual(int(m['row_before'].sum()), 2)
        self.assertEqual(int(m['col'].sum()), 3)
        self.assertEqual(int(m['above'].sum()), 5)
        self.assertEqual(int(m['below'].sum()), 10)
        self.assertEqual(int(m['col0'].sum()), 4)


class TestRecovery(unittest.TestCase):
    def test_recovery_definition_and_exclusions(self):
        v = {'LD': {'none': -2.0, 'obj': 1.0}, 'LD_B': 4.0, 'dA': 3, 'dB': 5}
        self.assertAlmostEqual(ca.recovery(v, 'obj'), 0.5)
        self.assertIsNone(ca.recovery({**v, 'dB': 3}, 'obj'))
        self.assertIsNone(ca.recovery({**v, 'LD_B': -1.0}, 'obj'))

    def test_boot_handles_none(self):
        b = ca.boot([None, 1.0, 1.0])
        self.assertEqual(b['n'], 2)
        self.assertAlmostEqual(b['mean'], 1.0)


class TestToy(unittest.TestCase):
    def test_ordinals_count_figure_lines(self):
        rng = np.random.default_rng(0)
        content, tgt, ords = toy.sample_grids(50, 4, 5, rng)
        for c, (r, k), (orow, ocol) in zip(content, tgt, ords):
            self.assertEqual(c[r, k], 2)
            self.assertEqual(orow, int((c[:r + 1] > 0).any(1).sum()))
            self.assertEqual(ocol, int((c[:, :k + 1] > 0).any(0).sum()))

    def test_causal_mixers_do_not_see_the_future(self):
        torch.manual_seed(0)
        for arch in ('causal_softmax', 'causal_delta', 'hybrid'):
            m = toy.ToyModel(arch, 4, 5, 6).eval()
            tok = torch.randint(0, 3, (1, 20))
            tok2 = tok.clone()
            tok2[0, 12:] = (tok2[0, 12:] + 1) % 3
            with torch.no_grad():
                a, b = m.image_states(tok), m.image_states(tok2)
            self.assertTrue(torch.allclose(a[0, :12], b[0, :12], atol=1e-5), arch)

    def test_bidirectional_mixers_see_the_future(self):
        torch.manual_seed(0)
        for arch in ('bidir_softmax', 'bidir_delta'):
            m = toy.ToyModel(arch, 4, 5, 6).eval()
            tok = torch.zeros(1, 20, dtype=torch.long)
            tok2 = tok.clone()
            tok2[0, 19] = 2
            with torch.no_grad():
                a, b = m.image_states(tok), m.image_states(tok2)
            self.assertFalse(torch.allclose(a[0, 0], b[0, 0], atol=1e-6), arch)

    def test_parallel_scan_matches_recurrence(self):
        torch.manual_seed(0)
        scan = toy.DeltaScan(16, 2)
        x = torch.randn(2, 7, 16)
        with torch.no_grad():
            o, alpha, beta = scan(x, return_gates=True)
            u = scan.conv(scan.qkv(x).transpose(1, 2))[..., :7].transpose(1, 2)
            q, k, v = torch.nn.functional.silu(u).chunk(3, -1)
            q = torch.nn.functional.normalize(q.view(2, 7, 2, 8), dim=-1)
            k = torch.nn.functional.normalize(k.view(2, 7, 2, 8), dim=-1)
            v = v.view(2, 7, 2, 8)
            S = torch.zeros(2, 2, 8, 8)
            outs = []
            for t in range(7):
                S = alpha[:, t, :, None, None] * S + beta[:, t, :, None, None] * torch.einsum('bhi,bhj->bhij', v[:, t], k[:, t])
                outs.append(torch.einsum('bhij,bhj->bhi', S, q[:, t]))
            ref = scan.out(torch.stack(outs, 1).reshape(2, 7, 16))
        self.assertTrue(torch.allclose(o, ref, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
