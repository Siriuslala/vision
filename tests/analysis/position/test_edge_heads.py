"""Model-free checks of experiment 3.11 round 3: masks, case selection, scoring, peaks, edge regression, bands."""
import unittest

import numpy as np

from analysis_tools.position import edge_heads as eh
from analysis_tools.position import edge_heads_analysis as ea


def rle_encode_string(mask: np.ndarray) -> str:
    """COCO compressed RLE string of a bool mask (port of pycocotools rleToString, for testing the decoder)."""
    flat = mask.T.ravel()
    counts, val, n = [], False, 0
    for x in flat:
        if bool(x) != val:
            counts.append(n)
            n, val = 0, not val
        n += 1
    counts.append(n)
    out = []
    for i, x in enumerate(counts):
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            c = x & 0x1f
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return ''.join(out)


class MaskTests(unittest.TestCase):
    def test_rle_roundtrip(self):
        rng = np.random.default_rng(0)
        m = rng.random((23, 17)) > 0.6
        m[5:15, 3:12] = True
        self.assertTrue(np.array_equal(eh.rle_decode(rle_encode_string(m), 23, 17), m))
        self.assertTrue(np.array_equal(eh.rle_decode([0, 3, 1], 2, 2), np.array([[True, True], [True, False]])))

    def test_components_and_stuff_instance(self):
        m = np.zeros((40, 40), bool)
        m[2:10, 2:10] = True
        m[20:30, 20:22] = True
        lab = eh.components(m)
        self.assertEqual(len(np.unique(lab)), 3)
        self.assertEqual(len(np.unique(lab[2:10, 2:10])), 1)
        one = np.zeros((200, 200), np.uint8)
        one[40:120, 40:120] = 5
        self.assertIsNotNone(eh.stuff_instance(one, 5))
        two = one.copy()
        two[150:190, 150:190] = 5                    # a second blob holding 20% of the pixels
        self.assertIsNone(eh.stuff_instance(two, 5))

    def test_polygon_and_box(self):
        m = eh.polygons_mask([[10, 10, 30, 10, 30, 20, 10, 20]], (50, 40))
        self.assertEqual(eh.mask_box(m), [10.0, 10.0, 31.0, 21.0])
        self.assertEqual(eh.scale_box([0, 0, 100, 50], (200, 100), (100, 50)), [0, 0, 50, 25])


class ScoringTests(unittest.TestCase):
    def test_score_point_and_box(self):
        case = {'truth': {'canvas': [500, 500], 'box': [100, 100, 200, 200]}}
        pt = [{'key': 'x', 'value': 300}, {'key': 'y', 'value': 300}]
        self.assertTrue(eh.score(case, pt, 'point')['correct'])
        bb = [{'key': k, 'value': v} for k, v in zip(('x_min', 'y_min', 'x_max', 'y_max'), (200, 200, 400, 400))]
        s = eh.score(case, bb, 'bbox')
        self.assertAlmostEqual(s['iou'], 1.0, places=3)
        self.assertFalse(eh.score(case, pt, 'bbox')['valid'])
        self.assertTrue(eh.prompt_for('the dog', 'bbox').startswith('Locate the object described as: the dog. Give its tight'))

    def test_path_conditions(self):
        names = [c[0] for c in eh.path_conditions()]
        self.assertEqual(names[0], 'none')
        self.assertEqual(len(names), len(set(names)))
        mixed = [c for c in eh.path_conditions() if c[2] == 'mixed'][0]
        self.assertIn((20, 11), mixed[1])
        self.assertTrue(all(l < 20 for l, h in mixed[1] if h < 0))


class AnalysisTests(unittest.TestCase):
    def test_peaks(self):
        a = np.zeros((4, 6))
        a[2, 4] = 1.0
        p = ea.peaks(a.ravel()[None], (4, 6))[0]
        self.assertEqual(p.tolist(), [4.5 * ea.TP, 2.5 * ea.TP])

    def test_regression_recovers_edge_reader(self):
        rng = np.random.default_rng(1)
        items, feats = [], []
        for _ in range(60):
            x0, y0 = rng.uniform(0, 200, 2)
            w, h = rng.uniform(60, 250, 2)
            box = np.array([x0, y0, x0 + w, y0 + h])
            pk = np.zeros((6, 8, 16, 2))
            pk[..., 0] = box[2]                       # every head at the right edge ...
            pk[..., 1] = (box[1] + box[3]) / 2        # ... and halfway down
            items.append({'box': box})
            feats.append({'peak': pk})
        reg = ea.regression(items, feats)
        np.testing.assert_allclose(reg['beta'][0, 0, 0, 0], [0, 1], atol=1e-6)
        np.testing.assert_allclose(reg['beta'][0, 1, 0, 0], [0.5, 0.5], atol=1e-6)

    def test_bands(self):
        b = ea.band_of(np.array([-0.5, -0.1, 0.5, 0.95, 1.4]))
        self.assertEqual(b.tolist(), [3, 0, 1, 2, 3])
        self.assertEqual(ea.hid('24.7'), (ea.LI[24], 7))

    def test_wilson_and_mcnemar(self):
        p, lo, hi = ea.wilson(5, 10)
        self.assertTrue(lo < p < hi)
        self.assertEqual(ea.mcnemar_p(0, 0), 1.0)
        self.assertLess(ea.mcnemar_p(0, 12), 0.001)


if __name__ == '__main__':
    unittest.main()
