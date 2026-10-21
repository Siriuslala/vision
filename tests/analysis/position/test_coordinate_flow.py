"""Model-free checks of experiment 3.11: answer parsing, token groups, recovered fractions, field metrics."""
import unittest

import numpy as np

from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import coordinate_flow_analysis as cfa
from analysis_tools.position import coordinate_readout as cr


class ParsingTests(unittest.TestCase):
    def test_list_format(self):
        text = '```json\n[{"point_2d": [123, 456], "label": "x"}]\n```'
        spans = cf.value_spans(text, 'point')
        self.assertEqual([k for k, _ in spans], ['x', 'y'])
        self.assertEqual([text[a:b] for _, (a, b) in spans], ['123', '456'])

    def test_key_format_orders_by_emission(self):
        text = '{"x": 7, "y": 81}'                     # model ignored the requested y-first order
        spans = cf.value_spans(text, 'point_yxk')
        self.assertEqual([k for k, _ in spans], ['x', 'y'])
        self.assertIsNone(cf.value_spans('{"y": 5}', 'point_yxk'))

    def test_bbox_keys(self):
        text = '{"y_min": 1, "x_min": 2, "y_max": 3, "x_max": 4}'
        self.assertEqual([k for k, _ in cf.value_spans(text, 'bbox_yxk')], ['y_min', 'x_min', 'y_max', 'x_max'])

    def test_score_values(self):
        case = {'format': 'colrow', 'set': 't6', 'truth': {'canvas': [640, 640], 'target_rc': [2, 3]}}
        vals = [{'key': 'column', 'value': 4}, {'key': 'row', 'value': 3}]
        self.assertTrue(cf.score_values(case, vals)['correct'])
        disc = {'format': 'point_yxk', 'set': 'disc',
                'truth': {'canvas': [512, 512], 'center': [256, 256], 'diameter': 32, 'box': [240, 240, 272, 272]}}
        self.assertTrue(cf.score_values(disc, [{'key': 'y', 'value': 500}, {'key': 'x', 'value': 505}])['correct'])
        self.assertFalse(cf.score_values(disc, [{'key': 'y', 'value': 600}, {'key': 'x', 'value': 505}])['correct'])


class GroupTests(unittest.TestCase):
    def test_groups_partition(self):
        grid = (8, 10)
        target, moved = [23, 24], [27, 28]
        g = cf.token_groups(grid, target, moved)
        parts = ['obj', 'inbox', 'row_left', 'row_right', 'below_cols', 'below_left', 'below_right', 'above']
        allv = sorted(sum((g[p] for p in parts), []))
        self.assertEqual(allv, list(range(80)))                  # disjoint cover of the image
        self.assertEqual(g['above'], list(range(20)))
        self.assertEqual(sorted(g['obj']), [23, 24, 27, 28])
        self.assertEqual(sorted(g['ring1'] + g['ring2'] + g['ring3_4'] + g['ring5p'] + g['obj']), list(range(80)))

    def test_kv_conditions_positions(self):
        conds = {n: (p, l) for n, p, l in cf.kv_conditions((4, 4), [5], [7], s=10, e=26, L=40)}
        self.assertEqual(conds['all_img'][0], list(range(10, 26)))
        self.assertEqual(conds['text'][0], list(range(26, 39)))
        self.assertEqual(conds['obj@20'], ([15, 17], (20,)))
        self.assertNotIn('above', {n for n, _, _ in cf.kv_conditions((4, 4), [1], [3], 10, 26, 40)})


class ShareTests(unittest.TestCase):
    def test_recovered_fraction(self):
        rec = {'id': 'a', 'correct': True, 'pairs': [{'tag': 'h+1', 'kind': 'move', 'axis': 'h', 'values': [
            {'key': 'x', 'axis': 'h', 'dA': 3, 'dB': 5, 'LD_B': 6.0, 'LD': {'none': -4.0, 'obj': 1.0, 'all_img': 6.0}},
            {'key': 'y', 'axis': 'v', 'dA': 3, 'dB': 5, 'LD_B': 6.0, 'LD': {'none': -4.0, 'obj': 6.0, 'all_img': 6.0}},
            {'key': 'x', 'axis': 'h', 'dA': 3, 'dB': 3, 'LD_B': 0.0, 'LD': {'none': 0.0, 'obj': 0.0, 'all_img': 0.0}}]}]}
        sh = cfa.shares([rec], 'move')
        np.testing.assert_allclose(sh['obj']['r'], [0.5])        # only the moved, distinct-digit value counts
        np.testing.assert_allclose(sh['all_img']['r'], [1.0])


class FieldTests(unittest.TestCase):
    def test_perfect_field(self):
        grid = (6, 8)
        dec = cr.token_centers(grid)
        m = cfa.field_metrics(dec, grid, [19])
        self.assertEqual(m['flip_x'], 0.0)
        self.assertAlmostEqual(m['scale_x'], 1.0)
        self.assertAlmostEqual(m['scale_y'], 1.0)
        self.assertAlmostEqual(m['bias_x'], 0.0)

    def test_auroc(self):
        self.assertEqual(cfa.auroc(np.array([1., 2., 3., 4.]), np.array([False, False, True, True])), 1.0)


if __name__ == '__main__':
    unittest.main()
