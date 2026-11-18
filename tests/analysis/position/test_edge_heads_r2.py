import unittest

import numpy as np
from PIL import Image

from analysis_tools.position import edge_heads_r2 as e2
from analysis_tools.position import edge_heads_r2_analysis as an


class RegionProtocolTest(unittest.TestCase):
    def test_labels_bands_and_corners(self):
        grid = (10, 10)
        box = [64, 64, 288, 256]                      # columns 2..8, rows 2..7
        lab = e2.region_labels(box, grid).reshape(grid)
        R = e2.REGIONS.index
        self.assertEqual(lab[2, 2], R('top_left'))
        self.assertEqual(lab[3, 3], R('top_left'))    # two-token bands
        self.assertEqual(lab[7, 8], R('bottom_right'))
        self.assertEqual(lab[4, 2], R('left'))
        self.assertEqual(lab[4, 8], R('right'))
        self.assertEqual(lab[2, 5], R('top'))
        self.assertEqual(lab[7, 5], R('bottom'))
        self.assertEqual(lab[4, 5], R('interior'))
        self.assertEqual(lab[1, 5], R('out_ring'))
        self.assertEqual(lab[5, 0], R('out_col0'))
        self.assertEqual(lab[9, 9], R('out_far'))

    def test_small_box_rejected(self):
        with self.assertRaises(ValueError):
            e2.region_labels([64, 64, 160, 256], (10, 10))
        self.assertFalse(e2.box_span_ok([64, 64, 160, 256], (10, 10)))

    def test_region_mass_sums_to_one(self):
        rng = np.random.default_rng(0)
        p = rng.random(100)
        p /= p.sum()
        lab = e2.region_labels([64, 64, 288, 256], (10, 10))
        self.assertAlmostEqual(float(e2.region_mass(p, lab).sum()), 1.0, places=5)

    def test_profiles_right_band(self):
        grid = (10, 10)
        box = [64, 64, 288, 256]
        p = np.zeros(100)
        p[4 * 10 + 8] = 1.0                           # right band, middle row
        pr = an.profiles(p, box, grid, 'x')
        np.testing.assert_allclose(pr['across'], [0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(pr['along'][2], [0, 1, 0], atol=1e-6)


class ImageOpsTest(unittest.TestCase):
    def test_stretch_moves_one_edge(self):
        arr = np.zeros((64, 200, 3), np.uint8)
        arr[:, 50:100] = 255                           # object columns 50..99
        img = Image.fromarray(arr)
        out = np.asarray(e2.stretch_image(img, [50, 0, 100, 64], 'right', 40))[0, :, 0]
        on = np.nonzero(out > 127)[0]
        self.assertEqual(on.min(), 50)
        self.assertTrue(138 <= on.max() <= 140)
        out = np.asarray(e2.stretch_image(img, [50, 0, 100, 64], 'left', 30))[0, :, 0]
        on = np.nonzero(out > 127)[0]
        self.assertTrue(19 <= on.min() <= 21)
        self.assertEqual(on.max(), 99)
        self.assertEqual(e2.stretched_box([50, 0, 100, 64], 'left', 30), [20, 0, 100, 64])

    def test_translate(self):
        arr = np.zeros((64, 64, 3), np.uint8)
        arr[10, 20] = 255
        out = np.asarray(e2.translate_image(Image.fromarray(arr), 5, -3))
        self.assertEqual(out[7, 25, 0], 255)

    def test_translation_shift_towards_centre(self):
        self.assertEqual(e2.translation_shift([10, 10, 100, 100], (512, 512)), [96, 96])
        self.assertEqual(e2.translation_shift([400, 300, 500, 500], (512, 512)), [-96, -96])


class AnalysisHelpersTest(unittest.TestCase):
    def test_r_values_excludes_small_changes(self):
        rec = {'lds': [{'key': 'x', 'dA': 3, 'dB': 5, 'LD_B': 4.0, 'ld': [-2.0, 1.0, 4.0]},
                       {'key': 'y', 'dA': 3, 'dB': 3, 'LD_B': 0.0, 'ld': [0.0, 0.0, 0.0]}]}
        r = an.r_values(rec)
        self.assertEqual(list(r), ['x'])
        np.testing.assert_allclose(r['x'], [0, 0.5, 1])

    def test_convention(self):
        self.assertEqual(an.convention({'x': 250, 'y': 200}, [160, 105, 331, 300], (512, 512), 'point'), 'pixel')
        self.assertEqual(an.convention({'x': 480, 'y': 395}, [160, 105, 331, 300], (512, 512), 'point'), 'norm')


if __name__ == '__main__':
    unittest.main()
