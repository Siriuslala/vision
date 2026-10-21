"""CPU tests of the experiment 5.1 engine (windows, MRoPE ids, prompts, scoring)."""
import unittest

import numpy as np
import torch

from analysis_tools.reread import reread_pilot as rp


class WindowTest(unittest.TestCase):
    def test_boxes_and_indices(self):
        w = rp.Window(2, 4, 6, 6)
        self.assertEqual(w.pixel_box(), (128, 64, 320, 256))
        self.assertEqual(w.norm_box((20, 20)), [200, 100, 500, 400])
        self.assertEqual(w.indices(20)[:2], [44, 45])
        self.assertEqual(len(w.indices(20)), 36)

    def test_centered_cells_clamp(self):
        self.assertEqual(rp.centered_cells((0, 9), 3, 10, 10), (0, 7))
        self.assertEqual(rp.centered_cells((5, 5), 3, 10, 10), (4, 4))

    def test_random_window_avoids(self):
        rng = np.random.default_rng(0)
        oracle = rp.Window(0, 0, 10, 10)
        for _ in range(50):
            r = rp.random_window((20, 20), (10, 10), [oracle], rng, step=2)
            self.assertFalse(r.overlaps(oracle))
            self.assertEqual((r.r0 % 2, r.c0 % 2), (0, 0))

    def test_random_cells_contain_target(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            r0, c0 = rp.random_cells_containing((7, 2), 5, 10, 10, rng)
            self.assertTrue(r0 <= 7 < r0 + 5 and c0 <= 2 < c0 + 5)


class MropeTest(unittest.TestCase):
    def test_default_rule(self):
        """Two 'new' segments reproduce the Qwen3.5 get_rope_index rule: text +1, image t=s, h=s+r, w=s+c,
        resume at s + max(h, w)."""
        segs = [rp.Segment(torch.zeros(6, 2), (2, 3)), rp.Segment(torch.zeros(4, 2), (2, 2))]
        kinds = [-1, -1] + [0] * 6 + [-1] + [1] * 4 + [-1]
        ids = rp.mrope_ids(kinds, segs)
        self.assertEqual(ids[:, 0].tolist(), [0, 0, 0])
        self.assertEqual(ids[:, 2].tolist(), [2, 2, 2])          # first image token
        self.assertEqual(ids[:, 7].tolist(), [2, 3, 4])          # row 1, col 2
        self.assertEqual(ids[:, 8].tolist(), [5, 5, 5])          # text resumes at 2 + 3
        self.assertEqual(ids[:, 9].tolist(), [6, 6, 6])          # second image start
        self.assertEqual(ids[:, 12].tolist(), [6, 7, 7])
        self.assertEqual(ids[:, 13].tolist(), [8, 8, 8])

    def test_orig_mode(self):
        win = rp.Window(1, 2, 2, 2)
        segs = [rp.Segment(torch.zeros(16, 2), (4, 4)), rp.Segment(torch.zeros(4, 2), (2, 2), 'orig', win)]
        kinds = [-1] + [0] * 16 + [-1] + [1] * 4 + [-1]
        ids = rp.mrope_ids(kinds, segs)
        s1, s2 = 1, 1 + 4 + 1
        self.assertEqual(ids[:, 18].tolist(), [s2, s1 + 1, s1 + 2])
        self.assertEqual(ids[:, 21].tolist(), [s2, s1 + 2, s1 + 3])
        self.assertEqual(ids[:, 22].tolist(), [s2 + 2] * 3)       # same text resume as the 'new' layout

    def test_segment_checks(self):
        with self.assertRaises(ValueError):
            rp.Segment(torch.zeros(5, 2), (2, 3))
        with self.assertRaises(ValueError):
            rp.Segment(torch.zeros(4, 2), (2, 2), 'orig')


class PromptAndScoreTest(unittest.TestCase):
    def test_hint(self):
        self.assertTrue(rp.hint_text(1, [1, 2, 3, 4]).startswith('The second image shows region (1, 2, 3, 4)'))
        self.assertIn('second and third images all show', rp.hint_text(2, [1, 2, 3, 4]))

    def test_point_frame_error(self):
        grid, n = (20, 20), 10
        win = rp.cell_window(5, 5, 5)                     # cells 5..9 -> norm box 500..1000
        target = (6, 7)
        hit = rp.score_point('[{"point_2d": [750, 650]}]', 'eos', n, target, win, grid)
        self.assertTrue(hit['hit'] and hit['in_window'] and not hit['frame_error'])
        # local coords of the target cell centre: ((7-5+.5)/5, (6-5+.5)/5) * 1000 = (500, 300)
        loc = rp.score_point('[{"point_2d": [500, 300]}]', 'eos', n, target, win, grid)
        self.assertFalse(loc['hit'])
        self.assertTrue(loc['frame_error'])
        self.assertEqual(rp.score_point('x', 'length', n, target, win, grid)['failure'], 'truncated')

    def test_rowcol(self):
        grid, n = (20, 20), 10
        win = rp.cell_window(5, 5, 5)
        ok = rp.score_rowcol('{"row": 7, "column": 8}', 'eos', n, (6, 7), win, grid)
        self.assertTrue(ok['correct'])
        fe = rp.score_rowcol('{"row": 2, "column": 3}', 'eos', n, (6, 7), win, grid)
        self.assertTrue(fe['frame_error'])
        last = rp.score_rowcol('{"row": 1, "column": 1} ... {"row": 7, "column": 8}', 'eos', n, (6, 7), win, grid, last=True)
        self.assertTrue(last['correct'])

    def test_bbox(self):
        win = rp.Window(0, 0, 10, 10)
        s = rp.score_bbox('```json\n[{"bbox_2d": [100, 100, 200, 200], "label": "x"}]\n```', 'eos',
                          [100, 100, 200, 200], win, (20, 20))
        self.assertTrue(s['correct'] and s['center_in'])
        self.assertAlmostEqual(rp.box_iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7)

    def test_wilson(self):
        lo, hi = rp.wilson(5, 10)
        self.assertTrue(0.2 < lo < 0.5 < hi < 0.8)


if __name__ == '__main__':
    unittest.main()
