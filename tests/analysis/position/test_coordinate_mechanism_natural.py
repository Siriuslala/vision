"""Model-free tests for experiment 3.15 round 2 (coordinate mechanism on natural vs synthetic images)."""
import unittest

import numpy as np

from analysis_tools.position import coordinate_mechanism_natural as cmn


class TestStimuli(unittest.TestCase):
    def test_grid_truth_and_erase(self):
        img, t = cmn.synth_grid(10, 64, (3, 5), 'person')
        self.assertEqual(img.size, (640, 640))
        cb = t['cell_box_units']
        self.assertAlmostEqual(cb[0], 500.0)
        self.assertAlmostEqual(cb[2], 600.0)
        self.assertTrue(cb[0] <= t['center_units'][0] <= cb[2])
        img2, _ = cmn.synth_grid(10, 64, (3, 5), 'person', erase=(3, 4))
        a, b = np.asarray(img), np.asarray(img2)
        self.assertTrue((b[3 * 64:4 * 64, 4 * 64:5 * 64] == 255).all())
        self.assertTrue((a[3 * 64:4 * 64, 5 * 64:6 * 64] == b[3 * 64:4 * 64, 5 * 64:6 * 64]).all())

    def test_scatter_delL_keeps_target(self):
        case = cmn.scatter_cases(2)[0]
        a, ta = cmn.scatter_image(case)
        b, tb = cmn.scatter_image(case, erase_prev=True)
        self.assertEqual(ta['box'], tb['box'])
        self.assertGreater(np.abs(np.asarray(a, int) - np.asarray(b, int)).sum(), 0)
        self.assertGreaterEqual(ta['raster_rank'], 4)

    def test_raster_order(self):
        pts = [(100, 40), (10, 70), (50, 5)]
        self.assertEqual(cmn.raster_order(pts), [2, 0, 1])   # token rows 1, 2, 0 -> by row then x


class TestCropPair(unittest.TestCase):
    def test_horizontal(self):
        w = cmn.crop_pair((512, 384), [200, 100, 300, 200], 0, 28)
        self.assertIsNotNone(w)
        a, b = w
        self.assertEqual(a[2] - a[0], 448)
        self.assertEqual(b[0] - a[0], 28)
        ta, tb = cmn.crop_truth([200, 100, 300, 200], a), cmn.crop_truth([200, 100, 300, 200], b)
        self.assertAlmostEqual(ta['center_units'][0] - tb['center_units'][0], 1000 * 28 / 448, places=4)
        self.assertAlmostEqual(ta['center_units'][1], tb['center_units'][1])

    def test_diagonal_and_unfit(self):
        w = cmn.crop_pair((512, 384), [200, 100, 300, 200], 2, 28)
        self.assertEqual(w[1][0] - w[0][0], 28)
        self.assertEqual(w[1][1] - w[0][1], 28)
        self.assertIsNone(cmn.crop_pair((512, 384), [0, 100, 500, 200], 0, 28))


class TestParsing(unittest.TestCase):
    def test_answer_values(self):
        toks = ['```', 'json', '\n', '[', '{"', 'bbox', '_', '2', 'd', '":', ' [', '1', '2', ',', ' ', '3', '4', '5',
                ',', ' ', '6', '7', '8', ',', ' ', '9', '0', '0', ']']
        v = cmn.answer_values(toks, 'bbox')
        self.assertEqual([x['value'] for x in v], [12, 345, 678, 900])
        self.assertEqual([x['role'] for x in v], ['x_min', 'y_min', 'x_max', 'y_max'])
        self.assertEqual(v[1]['start'], 15)
        self.assertEqual(cmn.answer_values(['no', 'json'], 'point'), [])

    def test_parse(self):
        self.assertEqual(cmn.parse_point('[{"point_2d": [552, 271], "label": "a"}]'), [552.0, 271.0])
        self.assertEqual(cmn.parse_bbox('{"bbox_2d": [1, 2, 3, 4]}'), [1.0, 2.0, 3.0, 4.0])


class TestStats(unittest.TestCase):
    def test_recovery(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(5, 10)), rng.normal(size=(5, 10))
        self.assertAlmostEqual(cmn.recovery(a, b, b), 1.0)
        self.assertAlmostEqual(cmn.recovery(a, b, a), 0.0)
        self.assertAlmostEqual(cmn.recovery(a, b, (a + b) / 2), 0.5)
        self.assertAlmostEqual(cmn.recovery(a, b, a + 3.0), 0.0)        # uniform shifts are removed by centring

    def test_position_class(self):
        self.assertEqual(cmn.position_class(cmn.relative_position(650, 600, 700)), 'centre')
        self.assertEqual(cmn.position_class(cmn.relative_position(700, 600, 700)), 'upper_edge')
        self.assertEqual(cmn.position_class(cmn.relative_position(605, 600, 700)), 'lower_edge')
        lo, hi = cmn.wilson(5, 10)
        self.assertTrue(lo < 0.5 < hi)


if __name__ == '__main__':
    unittest.main()
