"""Tests of the pure helpers of experiment 3.14 round 4 (row anchor)."""
from __future__ import annotations

import unittest

import numpy as np

from analysis_tools.position import row_anchor as rn


class StimulusTests(unittest.TestCase):
    def test_compose_truth_and_size(self):
        for variant in rn.VARIANTS:
            im, t = rn.compose(variant, 'ring', (4, 3), 7, [])  if 'photo' not in variant else (None, None)
            if im is None:
                continue
            W, H = im.size
            self.assertEqual(t['canvas'], [W, H])
            self.assertEqual(W % rn.TOKEN, 0)
            self.assertEqual(H % rn.TOKEN, 0)
            self.assertEqual(len(t['ink']), (W // rn.TOKEN) * (H // rn.TOKEN))
            x0, y0, x1, y1 = t['box']
            self.assertTrue(0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H, variant)
            # the target box contains red pixels
            a = np.asarray(im)[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))].astype(int)
            red = (a[..., 0] > 150) & (a[..., 1] < 120) & (a[..., 2] < 120)
            self.assertGreater(red.sum(), 5, variant)

    def test_margin_shifts_grid(self):
        _, e = rn.compose('edge', 'ring', (4, 3), 7, [])
        _, m = rn.compose('margin_left', 'ring', (4, 3), 7, [])
        self.assertEqual(m['box'][0] - e['box'][0], 128)
        ink = np.array(m['ink']).reshape(20, 24)
        self.assertEqual(ink[:, :4].sum(), 0)          # left margin is blank
        self.assertGreater(ink[:, 4].sum(), 0)        # first figure column starts at token column 4

    def test_scatter_no_overlap(self):
        pos = rn.scatter_positions(40, 640, 64, np.random.default_rng(0))
        for i, (a, b) in enumerate(pos):
            for c, d in pos[i + 1:]:
                self.assertTrue(abs(a - c) >= 64 or abs(b - d) >= 64)

    def test_column_zero_matches_source_column(self):
        im, _ = rn.compose('edge', 'person', (5, 6), 3, [])
        a = np.asarray(im)
        for S in (2, 4):
            col0 = a[:, :rn.TOKEN]
            colS = a[:, S * rn.TOKEN:(S + 1) * rn.TOKEN]
            rows = [r for r in range(20) if r // 2 != 5]
            for r in rows:
                self.assertTrue((col0[r * 32:(r + 1) * 32] == colS[r * 32:(r + 1) * 32]).all())


class IndexTests(unittest.TestCase):
    def test_patch_rows_and_index(self):
        d, s = rn.patch_rows((3, 5), 0, 2)
        self.assertEqual(d, [0, 5, 10])
        self.assertEqual(s, [2, 7, 12])
        self.assertEqual(rn.patch_index([0, 2]), [0, 1, 2, 3, 8, 9, 10, 11])

    def test_vit_patch_order_is_block_major(self):
        import torch
        from transformers.vision_utils import get_vision_position_ids
        pos = get_vision_position_ids(torch.tensor([[1, 4, 6]]), 2)
        # merged token t = (row, col) in a 2 x 3 merged grid covers patches 4t .. 4t + 3
        for t in range(6):
            r, c = divmod(t, 3)
            p = pos[4 * t:4 * t + 4]
            self.assertTrue(((p[:, 0] // 2) == r).all() and ((p[:, 1] // 2) == c).all())

    def test_source_column(self):
        self.assertEqual(rn.source_column(1), 4)
        self.assertEqual(rn.source_column(5), 2)

    def test_edit_groups(self):
        grid = (20, 24)
        ink = np.zeros(480)
        ink[[i * 24 + 4 for i in range(20)]] = 0.3
        target = [8 * 24 + 10, 8 * 24 + 11, 9 * 24 + 10, 9 * 24 + 11]
        g = rn.edit_groups(grid, target, ink, 4)
        self.assertEqual(g['A0'][0], [8 * 24, 9 * 24])
        self.assertEqual(g['A0'][1], [12 * 24, 13 * 24])
        self.assertEqual(g['F'][0], [8 * 24 + 4, 9 * 24 + 4])
        self.assertEqual(len(g['Rin'][0]), 2 * 8)


if __name__ == '__main__':
    unittest.main()
