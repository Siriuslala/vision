import unittest

import numpy as np

from analysis_tools.position import readout_tool_r3 as r3
from analysis_tools.position.readout_tool_r3_analysis import CONFIGS, config_channels, flip_negate, peak_points_all_heads
from analysis_tools.position.readout_tool_r3_eval import cost
from analysis_tools.position.readout_tool_r3_run import ridge_fit, ridge_predict


class HeadIndexTest(unittest.TestCase):
    def test_roundtrip(self):
        self.assertEqual(r3.head_index(20, 11), 75)
        self.assertEqual(r3.head_name(75), 'L20H11')
        self.assertEqual(r3.parse_heads('L20H11,L16H0'), (75, 48))
        self.assertEqual(len(r3.parse_heads('L16,L20')), 32)
        self.assertEqual(len(r3.parse_heads('all')), 128)
        self.assertEqual(r3.parse_heads(''), ())
        with self.assertRaises(ValueError):
            r3.parse_heads('L20H11,L20')

    def test_configs_valid(self):
        for name in CONFIGS:
            d = config_channels(name)
            self.assertGreater(sum(d.values()), 0)
            if d['approx']:
                self.assertEqual(flip_negate(name), (sum(d.values()) - 3,))


class DecoderTest(unittest.TestCase):
    def setUp(self):
        self.grid = (4, 5)
        self.m = np.full(20, 1e-3)
        self.m[2 * 5 + 3] = 1.0           # peak at row 2, col 3

    def test_peak_centroid_single_token(self):
        p = r3.peak_centroid(self.m, self.grid)
        np.testing.assert_allclose(p, [700.0, 625.0], atol=2.0)

    def test_vectorized_matches_scalar(self):
        rng = np.random.default_rng(0)
        maps = rng.random((6, 20))
        peak, mean = peak_points_all_heads(maps, self.grid)
        for h in range(6):
            np.testing.assert_allclose(peak[h], r3.peak_centroid(maps[h], self.grid), atol=1e-6)
            np.testing.assert_allclose(mean[h], r3.map_centroid(maps[h], self.grid), atol=1e-6)

    def test_extent_regions(self):
        m = self.m.copy()
        m[2 * 5 + 2] = 0.5                 # neighbour above threshold
        m[0] = 0.9                         # disconnected token above threshold
        reg = r3.attention_extent(m, self.grid, 0.4)
        self.assertTrue(reg[2, 3] and reg[2, 2] and not reg[0, 0])
        np.testing.assert_allclose(r3.region_box(reg, self.grid), [400, 500, 800, 750])

    def test_content_extent_follows_similar_tokens(self):
        content = np.zeros((20, 3))
        content[:, 0] = 1.0                # background direction
        obj = [2 * 5 + 3, 2 * 5 + 4, 3 * 5 + 3]
        content[obj] = [0.0, 1.0, 0.0]
        reg = r3.content_extent(self.m, content, self.grid, 0.9, top=1)
        self.assertEqual(set(np.flatnonzero(reg.reshape(-1))), set(obj))


class EdgeGroupTest(unittest.TestCase):
    def test_edge_group_box_corners(self):
        grid = (4, 5)
        maps = np.full((128, 20), 1e-3)
        tl, br = r3.parse_heads('L24H7,L20H0'), r3.parse_heads('L24H5')
        maps[list(tl), 0 * 5 + 1] = 1.0    # top-left readers peak at row 0, col 1
        maps[list(br), 3 * 5 + 4] = 1.0    # bottom-right reader peaks at row 3, col 4
        b = r3.edge_group_box(maps, grid, tl, br)
        np.testing.assert_allclose(b, [300.0, 125.0, 900.0, 875.0], atol=3.0)
        self.assertEqual(len(r3.EDGE_TL) + len(r3.EDGE_BR), 11)


class FailureHelpersTest(unittest.TestCase):
    def test_selection_and_occlusion(self):
        t = np.array([0, 0, 100, 100.])
        o = [np.array([50, 50, 200, 200.])]
        self.assertEqual(r3.selection_outcome(np.array([10, 10.]), t, o), 'target')
        self.assertEqual(r3.selection_outcome(np.array([150, 150.]), t, o), 'distractor')
        self.assertEqual(r3.selection_outcome(np.array([500, 500.]), t, o), 'background')
        self.assertAlmostEqual(r3.occlusion_fraction(t, o), 0.25)

    def test_polygon_mask(self):
        m = r3.polygon_mask([0, 0, 10, 0, 10, 10, 0, 10], (20, 20), (40, 40))
        self.assertEqual(m.shape, (40, 40))
        self.assertTrue(m[5, 5] and not m[30, 30])


class CostAndRidgeTest(unittest.TestCase):
    def test_cost_scales_with_heads(self):
        c1 = cost(CONFIGS['k1_H11_content_approx'], 1000)
        c4 = cost(CONFIGS['k4_content_approx'], 1000)
        self.assertEqual((c1['heads'], c4['heads']), (1, 4))
        self.assertAlmostEqual(c4['attn_mflops'], 4 * c1['attn_mflops'])

    def test_ridge_recovers_linear_map(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 5))
        W = rng.normal(size=(5, 2))
        Y = X @ W + 3.0
        m = ridge_fit(X, Y, 1e-6)
        np.testing.assert_allclose(ridge_predict(m, X), Y, atol=1e-3)


class CompareHelpersTest(unittest.TestCase):
    def test_coco_image_of(self):
        self.assertEqual(r3.coco_image_of('COCO_train2014_000000296747_0.jpg'), 'COCO_train2014_000000296747')

    def test_answer_texts_parse_back(self):
        from analysis_tools.position import targeted_finetune as tf
        from analysis_tools.position.readout_tool_r3_compare import answer_texts
        t = answer_texts(np.array([100.4, 200.6, 50, 60, 150, 340]))
        self.assertEqual(tf.parse_answer(t['point'], 'point'), [100.0, 201.0])
        self.assertEqual(tf.parse_answer(t['bbox'], 'bbox'), [50.0, 60.0, 150.0, 340.0])
        self.assertNotIn('bbox', answer_texts(np.array([1, 2, np.nan, np.nan, np.nan, np.nan])))

    def test_tf_calls_merge_formats(self):
        import json
        import tempfile
        from pathlib import Path
        from analysis_tools.position.readout_tool_r3_run import tf_calls
        rows = [{'id': f'a_{f}', 'image': '/x/img1.png', 'label': 'red circle', 'family': 'flat', 'fmt': f,
                 'truth': {'canvas': [64, 64], 'box': [0, 0, 10, 10], 'center': [5, 5]}} for f in ('point', 'bbox')]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'test.jsonl'
            p.write_text('\n'.join(json.dumps(r) for r in rows))
            calls = tf_calls(p)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['items'], ['a_point', 'a_bbox'])


if __name__ == '__main__':
    unittest.main()
