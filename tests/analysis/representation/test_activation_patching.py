"""GPU-free checks of experiment 2.5a: pair construction, token groups, and ratios."""
import unittest

import numpy as np
import torch

from analysis_tools.common.stimuli import TOKEN_PIXELS, generate
from analysis_tools.representation.activation_patching import (
    CONDITIONS, bootstrap_summary, cell_tokens, damage, high_norm_tokens, make_pairs, neighbor_tokens,
    qualify, random_tokens, recovery, render_pair, text_groups,
)


class PairTests(unittest.TestCase):
    def test_pixel_difference_is_confined_to_target_cell(self):
        for condition in CONDITIONS:
            for pair in make_pairs(condition, 8, seed=0):
                clean, meta, corrupt, _ = render_pair(pair)
                diff = (np.asarray(clean) != np.asarray(corrupt)).any(-1)
                ys, xs = np.nonzero(diff)
                x0, y0, x1, y1 = meta['instances'][pair.target[0] * 6 + pair.target[1]]['cell_box']
                self.assertTrue(len(xs) > 0)
                self.assertTrue(((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).all(), pair.pair_id)

    def test_conditions_share_targets_and_marker_follows_designation(self):
        red, gap = make_pairs('red_box', 16, 0), make_pairs('gap_rowcol', 16, 0)
        self.assertEqual([p.target for p in red], [p.target for p in gap])
        self.assertTrue(all(p.clean.marker == p.target == p.corrupt.marker for p in red))
        self.assertTrue(all(p.clean.marker is None for p in gap))
        self.assertTrue(all(f'row {p.target[0] + 1}, column {p.target[1] + 1}' in p.question for p in gap))

    def test_clean_sign_follows_question(self):
        self.assertEqual(make_pairs('gap_box', 1, 0)[0].clean_sign, -1)
        self.assertEqual(make_pairs('gap_box_attr', 1, 0)[0].clean_sign, 1)


class TokenGroupTests(unittest.TestCase):
    def setUp(self):
        self.pair = make_pairs('red_rowcol', 1, 0)[0]
        _, self.meta = generate(self.pair.clean)

    def test_cell_tokens_cover_the_cell_pixels(self):
        width = self.meta['merged_grid'][1]
        for rc in ((0, 0), (2, 3), (5, 5)):
            tokens = cell_tokens(self.meta, rc)
            self.assertEqual(len(tokens), 4)
            x0, y0, x1, y1 = self.meta['instances'][rc[0] * 6 + rc[1]]['cell_box']
            for t in tokens:
                r, c = divmod(t, width)
                self.assertTrue(x0 <= c * TOKEN_PIXELS < x1 and y0 <= r * TOKEN_PIXELS < y1)
        self.assertEqual(cell_tokens(self.meta, (0, 0)), [0, 1, 12, 13])
        self.assertEqual(cell_tokens(self.meta, (5, 5)), [130, 131, 142, 143])

    def test_neighbors_exclude_target_and_respect_edges(self):
        self.assertEqual(len(neighbor_tokens(self.meta, (0, 0))), 8)
        self.assertEqual(len(neighbor_tokens(self.meta, (2, 3))), 16)
        self.assertFalse(set(neighbor_tokens(self.meta, (2, 3))) & set(cell_tokens(self.meta, (2, 3))))

    def test_random_and_high_norm_groups(self):
        target = cell_tokens(self.meta, (2, 3))
        picked = random_tokens(144, target, 4, seed=3)
        self.assertEqual(len(set(picked)), 4)
        self.assertFalse(set(picked) & set(target))
        self.assertEqual(picked, random_tokens(144, target, 4, seed=3))
        norms = torch.arange(144, dtype=torch.float)
        self.assertEqual(high_norm_tokens(norms), list(range(136, 144)))   # ceil(0.05 * 144) = 8

    def test_text_groups_partition_post_image_tokens(self):
        # 0..3 prefix, 4..7 image, 8 <|vision_end|>, 9..11 question, 12 <|im_end|>, 13..15 header
        ids = [1, 2, 3, 4, 50, 50, 50, 50, 60, 7, 8, 9, 99, 5, 6, 7]
        groups = text_groups(ids, 8, im_end_id=99)
        self.assertEqual(groups['question'], [9, 10, 11])
        self.assertEqual(groups['template'], [8, 12, 13, 14])
        self.assertEqual(groups['last'], [15])
        self.assertEqual(sorted(groups['question'] + groups['template'] + groups['last']), groups['text_all'])


class RatioTests(unittest.TestCase):
    def test_recovery_and_damage_bounds(self):
        s_clean, s_corrupt = -6.0, 2.0
        self.assertEqual(recovery(s_corrupt, s_corrupt, s_clean), 0.0)
        self.assertEqual(recovery(s_clean, s_corrupt, s_clean), 1.0)
        self.assertAlmostEqual(float(recovery(-2.0, s_corrupt, s_clean)), 0.5)
        self.assertEqual(damage(s_clean, s_corrupt, s_clean), 0.0)
        self.assertEqual(damage(s_corrupt, s_corrupt, s_clean), 1.0)
        self.assertLess(float(recovery(4.0, s_corrupt, s_clean)), 0.0)   # moving away from clean is negative
        with self.assertRaises(ValueError):
            recovery(1.0, 2.0, 2.0)

    def test_qualification_reasons(self):
        self.assertEqual(qualify(-3.0, 2.0, -1), 'ok')
        self.assertEqual(qualify(1.0, 2.0, -1), 'clean_wrong')
        self.assertEqual(qualify(-3.0, -1.0, -1), 'corrupt_wrong')
        self.assertEqual(qualify(1.0, -1.0, -1), 'both_wrong')
        self.assertEqual(qualify(-0.4, 0.4, -1), 'small_gap')
        self.assertEqual(qualify(3.0, -2.0, 1), 'ok')

    def test_bootstrap_summary(self):
        nums, dens = np.array([1.0, 2.0, 3.0, 8.0]), np.array([2.0, 2.0, 4.0, 8.0])
        out = bootstrap_summary(nums, dens, n_boot=500, seed=0)
        self.assertAlmostEqual(out['median'], np.median(nums / dens))
        self.assertAlmostEqual(out['ratio_of_means'], 14 / 16)
        self.assertLessEqual(out['median_lo'], out['median'])
        self.assertGreaterEqual(out['median_hi'], out['median'])


if __name__ == '__main__':
    unittest.main()
