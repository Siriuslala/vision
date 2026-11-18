"""Tests of the pure helpers of experiment 3.14 round 3 (row attention)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from analysis_tools.position import row_attention as ra


class SectionTests(unittest.TestCase):
    def test_labels_qwen3_vl(self):
        lab = ra.section_labels(128, 128, [24, 20, 20])
        self.assertEqual(lab.shape, (128,))
        half = lab[:64]
        self.assertEqual((half == 1).sum(), 20)
        self.assertEqual((half == 2).sum(), 20)
        self.assertEqual((half == 0).sum(), 24)
        self.assertEqual(list(half[:6]), [0, 1, 2, 0, 1, 2])
        self.assertTrue((lab[64:] == half).all())

    def test_labels_qwen3_5_partial(self):
        lab = ra.section_labels(64, 256, [11, 11, 10])
        self.assertEqual((lab == 3).sum(), 192)
        self.assertEqual((lab[:32] == 1).sum(), 11)
        self.assertEqual((lab[:32] == 2).sum(), 10)

    def test_rotate_matches_transformers_and_sections_sum(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        g = torch.Generator().manual_seed(0)
        hd, r, n = 16, 8, 5
        q = torch.randn(1, 1, n, hd, generator=g)
        k = torch.randn(1, 1, n, hd, generator=g)
        ang = torch.randn(1, n, r // 2, generator=g)
        cos = torch.cat([ang.cos(), ang.cos()], -1)
        sin = torch.cat([ang.sin(), ang.sin()], -1)
        q_ref, k_ref = apply_rotary_pos_emb(q, k, cos, sin)
        self.assertTrue(torch.allclose(ra.rotate(q[0, 0], cos[0], sin[0]), q_ref[0, 0], atol=1e-6))
        self.assertTrue(torch.allclose(ra.rotate(k[0, 0], cos[0], sin[0]), k_ref[0, 0], atol=1e-6))
        labels = torch.as_tensor(ra.section_labels(r, hd, [2, 1, 1]))
        parts = ra.section_scores(q_ref[0, 0, :1], k_ref[0, 0], labels, 0.5)    # (1, 4, n)
        full = (q_ref[0, 0, :1] @ k_ref[0, 0].T) * 0.5
        self.assertTrue(torch.allclose(parts.sum(-2), full, atol=1e-5))

    def test_relative_rotation(self):
        """Rotating q and k by the same angle leaves each section score unchanged (RoPE is relative)."""
        g = torch.Generator().manual_seed(1)
        hd = 8
        q, k = torch.randn(hd, generator=g), torch.randn(hd, generator=g)
        a = torch.randn(hd // 2, generator=g)
        cos, sin = torch.cat([a.cos(), a.cos()]), torch.cat([a.sin(), a.sin()])
        s1 = (ra.rotate(q, cos, sin) * ra.rotate(k, cos, sin)).sum()
        self.assertAlmostEqual(float(s1), float((q * k).sum()), places=5)


class GroupTests(unittest.TestCase):
    def test_groups(self):
        grid = (6, 6)
        target = [2 * 6 + 3, 2 * 6 + 4, 3 * 6 + 3, 3 * 6 + 4]
        g = ra.token_groups(grid, target)
        self.assertEqual(g['Ra'], [12, 18])
        self.assertEqual(len(g['R']), 8)
        self.assertEqual(sorted(g['Rl'] + g['Rr'] + g['Ra']), g['R'])
        self.assertEqual(g['Rr'], [17, 23])
        self.assertEqual(len(g['C']), 8)
        self.assertFalse(set(g['C']) & set(g['R']))

    def test_anchor_inside_target(self):
        g = ra.token_groups((4, 4), [0, 1, 4, 5])
        self.assertEqual(g['Ra'], [])

    def test_shifted(self):
        dst, src = ra.shifted([0, 5, 15], (4, 4), 2, 0)
        self.assertEqual(dst, [0, 5])
        self.assertEqual(src, [8, 13])

    def test_edit_plan(self):
        grid = (8, 8)
        target = [2 * 8 + 2, 2 * 8 + 3]
        moved = [6 * 8 + 2, 6 * 8 + 3]
        plan, off = ra.edit_plan(grid, target, moved)
        self.assertEqual(off, (4, 0))
        self.assertEqual(set(plan), set(ra.EDIT_ROWS))
        kind, dst, src = plan['vRa']
        self.assertEqual((kind, dst, src), ('self', [16], [48]))
        self.assertEqual(plan['vT'], ('B', target, moved))


if __name__ == '__main__':
    unittest.main()
