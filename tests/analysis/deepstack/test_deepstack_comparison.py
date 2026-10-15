"""Scoring and statistics checks for experiment 4.2 (DeepStack end-to-end comparison)."""
import unittest

import numpy as np

from analysis_tools.deepstack.deepstack_comparison import parse_bbox, score_refcoco_case, slug, stratum_of
from analysis_tools.behavior.decomposition_analysis import auroc
from analysis_tools.deepstack.deepstack_comparison_analysis import fast_auroc, holm, mcnemar, paired_bootstrap
from analysis_tools.deepstack.deepstack_reasoning import REVISION, cohen_kappa


class DeepStackComparisonTests(unittest.TestCase):
    def test_slugs_match_the_bash_entry(self):
        self.assertEqual(slug('Qwen3-VL-8B-Instruct'), 'qwen3_vl_8b')
        self.assertEqual(slug('Qwen3.5-4B'), 'qwen35_4b')

    def test_strata_are_half_open(self):
        self.assertEqual(stratum_of(0.0099), 'lt1')
        self.assertEqual(stratum_of(0.01), '1to5')
        self.assertEqual(stratum_of(0.05), '5to10')
        self.assertEqual(stratum_of(0.5), 'ge10')

    def test_bbox_parsing_and_normalised_scoring(self):
        self.assertEqual(parse_bbox('[{"bbox_2d": [1, 2, 3, 4]}]'), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(parse_bbox('{"bbox_2d": [1, 2, 3, 4]}'), [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(parse_bbox('the box is somewhere'))
        self.assertEqual(parse_bbox('[10, 20.5, 30, 40]'), [10.0, 20.5, 30.0, 40.0])
        case = {'id': 'x', 'width': 200, 'height': 100, 'bbox': [20, 10, 100, 50]}
        hit = score_refcoco_case(case, {'response': '[{"bbox_2d": [100, 100, 500, 500]}]', 'finish_reason': 'eos'})
        self.assertAlmostEqual(hit['iou'], 1.0)
        self.assertTrue(hit['correct'] and hit['center_in'])
        cut = score_refcoco_case(case, {'response': '[{"bbox_2d": [100, 100, 500, 500]}]', 'finish_reason': 'length'})
        self.assertFalse(cut['valid'] or cut['correct'])
        flipped = score_refcoco_case(case, {'response': '[{"bbox_2d": [500, 100, 100, 500]}]', 'finish_reason': 'eos'})
        self.assertFalse(flipped['valid'])

    def test_exact_mcnemar(self):
        r = mcnemar([True] * 10 + [False] * 5, [False] * 10 + [False] * 5)
        self.assertEqual((r['only_first'], r['only_second']), (10, 0))
        self.assertAlmostEqual(r['p'], 2 / 2 ** 10)
        self.assertEqual(mcnemar([True, False], [True, False])['p'], 1.0)

    def test_holm_is_monotone_and_capped(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm([0.9, 0.8]), [1.0, 1.0])

    def test_paired_bootstrap_brackets_the_difference(self):
        a = np.array([1, 1, 1, 0] * 50, float)
        b = np.array([1, 0, 0, 0] * 50, float)
        lo, hi = paired_bootstrap(a, b, 2000, 0)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_fast_auroc_matches_pairwise_definition(self):
        rng = np.random.default_rng(0)
        s = np.round(rng.normal(size=200), 1)
        y = rng.random(200) < 0.4
        self.assertAlmostEqual(fast_auroc(s, y), auroc(list(s[y]), list(s[~y])))

    def test_revision_markers(self):
        self.assertEqual(len(REVISION.findall('Wait, actually the third one. Let me double-check.')), 3)

    def test_cohen_kappa(self):
        self.assertEqual(cohen_kappa(['a', 'b', 'a', 'b'], ['a', 'b', 'a', 'b']), 1.0)
        # Observed agreement 0.5 equals chance agreement 0.5, so kappa is 0.
        self.assertAlmostEqual(cohen_kappa(['a', 'a', 'b', 'b'], ['a', 'b', 'a', 'b']), 0.0)


if __name__ == '__main__':
    unittest.main()
