import unittest

import numpy as np
import torch

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position import readout_tool_r2 as r2
from analysis_tools.position.readout_tool_r2_dissoc import cell_tokens, neighbour


class GradedSceneTest(unittest.TestCase):
    def test_levels_deterministic_and_valid(self):
        bg = None
        for level in ('flat', 'texture', 'clutter', 'chart_ui'):
            for seed in range(3):
                a, ta, ga, ma = r2.graded_scene(level, seed, bg)
                b, tb, gb, mb = r2.graded_scene(level, seed, bg)
                self.assertEqual((ta, ga), (tb, gb))
                self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
                W, H = ta['canvas']
                self.assertEqual((W % 32, H % 32, a.size), (0, 0, (W, H)))
                x0, y0, x1, y1 = ta['box']
                self.assertTrue(0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H)
                self.assertTrue(ga.startswith('the '))

    def test_photo_bg_requires_backgrounds(self):
        with self.assertRaises(ValueError):
            r2.graded_scene('photo_bg', 0, [])


class ApproxParseTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(r2.parse_approx('225, 552]\n</parameter>'), [225, 552])
        self.assertEqual(r2.parse_approx('7,1000]'), [7, 1000])
        self.assertIsNone(r2.parse_approx('about 200'))

    def test_approx_features(self):
        f = r2.approx_features([500, 500], (2, 2))
        self.assertEqual(f.shape, (4, 3))
        np.testing.assert_allclose(f[0, :2], [-2.5, -2.5])
        self.assertTrue((r2.approx_features(None, (2, 2)) == 0).all())


class GridPointerTest(unittest.TestCase):
    def test_decode_geometry(self):
        """A pointer concentrated on one token with zero offsets returns that token's centre and a box
        of the regressed extent."""
        out = torch.zeros(1, 7, 4, 5)
        out[0, 0] = -50
        out[0, 0, 2, 3] = 50
        out[0, 3:7] = torch.log(torch.expm1(torch.tensor(0.5)))   # softplus^-1(0.5): half a token each side
        valid = torch.ones(1, 4, 5, dtype=torch.bool)
        alpha, point, box = r2.grid_decode(out, valid, [(4, 5)])
        np.testing.assert_allclose(point[0].numpy(), [700.0, 625.0], atol=1e-3)
        np.testing.assert_allclose(box[0].numpy(), [600.0, 500.0, 800.0, 750.0], atol=1e-2)
        self.assertAlmostEqual(float(alpha.sum()), 1.0, places=5)

    def test_padding_masked(self):
        items = [{'F': np.random.randn(6, 3).astype(np.float16), 'grid': (2, 3), 'q': np.ones(6), 'y': np.zeros(6)},
                 {'F': np.random.randn(12, 3).astype(np.float16), 'grid': (3, 4), 'q': np.ones(12), 'y': np.zeros(6)}]
        X, V, Q, Y, grids = r2.make_grid_batch(items, np.zeros(3), np.ones(3), 'cpu')
        self.assertEqual(tuple(X.shape), (2, 3, 3, 4))
        self.assertEqual(int(V[0].sum()), 6)
        m = r2.GridPointer(3, 16)
        alpha, _, _ = r2.grid_decode(m(X), V, grids)
        self.assertAlmostEqual(float(alpha[0][~V[0].reshape(-1)].sum()), 0.0)

    def test_fit_learns_toy_task(self):
        """The reader must learn to point at the token whose first feature is high."""
        rng = np.random.default_rng(0)
        items = []
        for _ in range(160):
            gh, gw = int(rng.integers(4, 8)), int(rng.integers(4, 8))
            F = rng.normal(0, 1, (gh * gw, 2)).astype(np.float32)
            j = int(rng.integers(gh * gw))
            F[j, 0] += 6
            r, c = divmod(j, gw)
            box = [c * 1000 / gw, r * 1000 / gh, (c + 1) * 1000 / gw, (r + 1) * 1000 / gh]
            y = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] + box)
            items.append({'F': F.astype(np.float16), 'grid': (gh, gw), 'q': rt_.token_overlap(box, (gh, gw)), 'y': y})
        params, hist = r2.fit_grid_pointer(items[:128], items[128:], epochs=20, width=16, batch=8, log=lambda s: None)
        m = r2.load_grid_pointer(params)
        P = r2.predict_grid_pointer(m, params['mean'], params['std'], items[128:])
        err = np.linalg.norm(P[:, :2] - np.stack([it['y'][:2] for it in items[128:]]), axis=1).mean()
        self.assertLess(err, 60.0)


class DissocGeometryTest(unittest.TestCase):
    truth = {'canvas': [256, 256], 'box': [70.0, 70.0, 110.0, 110.0], 'center': [90.0, 90.0],
             'hit': {'kind': 'box', 'box': [64.0, 64.0, 128.0, 128.0]},
             'grid': {'rows': 4, 'cols': 4, 'origin': [0, 0], 'cell': 64, 'target_rc': [1, 1]}}

    def test_neighbour_and_tokens(self):
        box, d = neighbour(self.truth, 'x')
        self.assertEqual(d, (0, 1))
        self.assertEqual(box, [134.0, 70.0, 174.0, 110.0])
        t = dict(self.truth, grid=dict(self.truth['grid'], target_rc=[3, 3]))
        self.assertEqual(neighbour(t, 'y')[1], (-1, 0))
        toks = cell_tokens([64, 64, 128, 128], [256, 256])
        self.assertEqual(sorted(toks.tolist()), [18, 19, 26, 27])


if __name__ == '__main__':
    unittest.main()
