"""Unit tests of the GPU-free helpers of experiment 7.2 round 2 (prompt variants, token groups, donor choice)."""
import unittest

import numpy as np

from analysis_tools.application import located_r2_run as lr
from analysis_tools.position import readout_prior_r2_run as r2r

PROMPT = ('There is one red circle in the image. Give the center of its bounding box. Return only [{"point_2d": [x, y]}]. '
          'All coordinates use 0-1000 normalization; x increases rightward and y increases downward. Output only the JSON, '
          'with no explanation.')


class LocatedR2Test(unittest.TestCase):
    def test_variant_prompts_keep_description_and_json_format(self):
        for v in ('para', 'zh'):
            p = lr.variant_prompt(PROMPT, v)
            self.assertTrue(p.startswith('There is one red circle in the image.'))
            self.assertIn('point_2d', p)
            self.assertNotIn('Give the center', p)
        self.assertEqual(lr.variant_prompt(PROMPT, 'qfirst'), PROMPT)
        with self.assertRaises(ValueError):
            lr.variant_prompt('Where is it?', 'para')

    def test_site_positions(self):
        r = {'span': (10, 14), 'question': [15, 16, 17], 'target': [12, 13]}
        self.assertEqual(lr.site_positions('first', r), [10])
        self.assertEqual(lr.site_positions('image_nofirst', r), [11, 12, 13])
        self.assertEqual(lr.site_positions('question', r), [15, 16, 17])
        self.assertEqual(lr.site_positions('target', r), [12, 13])

    def test_donor_same_hundred_shifted_tens(self):
        truth = np.arange(100, 1000, 4.0)
        d = r2r.donor_index(truth)
        self.assertTrue(np.all(np.round(truth[d]) // 100 == np.round(truth) // 100))
        o, od = (truth % 100) / 10, (truth[d] % 100) / 10
        gap = np.abs(((od - o) % 10) - 5)
        self.assertLess(np.median(gap), 0.5)


if __name__ == '__main__':
    unittest.main()
