import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position.readout_tool_analysis import mcnemar_exact, wilson
from analysis_tools.position.readout_tool_run import batches


class GenericSceneTest(unittest.TestCase):
    def test_deterministic_and_valid(self):
        for seed in range(6):
            a, ta = rt_.generic_scene(seed)
            b, tb = rt_.generic_scene(seed)
            self.assertEqual(ta, tb)
            self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
            W, H = ta['canvas']
            self.assertEqual((W % 32, H % 32), (0, 0))
            x0, y0, x1, y1 = ta['box']
            self.assertTrue(0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H)
            self.assertIn(ta['color'], ta['target'])
            self.assertIn(ta['shape'], ta['target'])


class ScoringTest(unittest.TestCase):
    truth = {'canvas': [500, 250], 'box': [100.0, 50.0, 200.0, 150.0], 'center': [150.0, 100.0],
             'hit': {'kind': 'box', 'box': [100.0, 50.0, 200.0, 150.0]}}

    def test_point(self):
        s = rt_.score_point(self.truth, [300, 400])        # (150, 100) px
        self.assertTrue(s['hit'])
        self.assertAlmostEqual(s['error'], 0.0)
        miss = rt_.score_point(self.truth, [150, 100])     # pixel convention: inside only as pixels
        self.assertFalse(miss['hit'])
        self.assertTrue(miss['hit_pix_only'])
        self.assertFalse(rt_.score_point(self.truth, None)['valid'])

    def test_box(self):
        s = rt_.score_box(self.truth, [200, 200, 400, 600])
        self.assertAlmostEqual(s['iou'], 1.0)
        self.assertFalse(rt_.score_box(self.truth, [400, 200, 200, 600])['valid'])

    def test_disc_hit(self):
        t = {'canvas': [512, 512], 'box': [0, 0, 64, 64], 'center': [32.0, 32.0],
             'hit': {'kind': 'disc', 'center': [32.0, 32.0], 'radius': 32.0}}
        self.assertTrue(rt_.score_point(t, [62.5, 62.5])['hit'])
        self.assertFalse(rt_.score_point(t, [120, 120])['hit'])


class ParseTest(unittest.TestCase):
    def test_call(self):
        self.assertEqual(rt_.parse_call(rt_.call_text('the man on the left')), 'the man on the left')
        self.assertIsNone(rt_.parse_call('[{"point_2d": [1, 2]}]'))

    def test_rowcol_free(self):
        self.assertEqual(rt_.parse_rowcol_free('Rows: 6.\nFinal: {"row": 3, "column": 4}'), (3, 4))
        self.assertEqual(rt_.parse_rowcol_free('```json\n{"row": 2, "column": 5}\n```'), (2, 5))
        self.assertIsNone(rt_.parse_rowcol_free('{"row": 2}'))

    def test_answer(self):
        self.assertEqual(rt_.parse_answer('```json\n[{"point_2d": [10, 20]}]\n```', 'point'), [10.0, 20.0])
        self.assertIsNone(rt_.parse_answer('[{"point_2d": [10]}]', 'point'))


class RegressorTest(unittest.TestCase):
    def test_ridge_recovers_linear_map(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 20))
        W = rng.normal(size=(20, 6))
        Y = X @ W * 50 + 500
        model, curve = rt_.fit_ridge(X[:300], Y[:300], (1e-3, 1.0, 1e3), X[300:], Y[300:])
        self.assertLess(np.abs(model.predict(X[300:]) - Y[300:]).max(), 1.0)
        self.assertEqual(set(curve), {1e-3, 1.0, 1e3})

    def test_tool_format_and_roundtrip(self):
        out = rt_.ReadoutTool.format(np.asarray([1003.2, -4.0, 600, 700, 500, 300]))
        self.assertEqual(out, {'point_2d': [1000, 0], 'bbox_2d': [500, 300, 600, 700]})
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 8))
        Y = rng.normal(size=(50, 6))
        m, _ = rt_.fit_ridge(X, Y, (1.0,), X, Y)
        tool = rt_.ReadoutTool('ridge', 'call_end', (20,), {k: getattr(m, k) for k in ('mean', 'scale', 'weight', 'bias', 'alpha')})
        with tempfile.TemporaryDirectory() as d:
            tool.save(Path(d) / 't.pt', {})
            back = rt_.ReadoutTool.load(Path(d) / 't.pt')
        self.assertTrue(np.allclose(back.predict_raw(X), tool.predict_raw(X)))


class PointerTest(unittest.TestCase):
    def test_overlap_and_features(self):
        q = rt_.token_overlap([0, 0, 500, 250], (4, 4))          # top-left half width, quarter height
        self.assertAlmostEqual(q.reshape(4, 4)[0, :2].sum(), 2.0)
        self.assertAlmostEqual(q.sum(), 2.0)
        maps = np.full((3, 6), 1 / 6, dtype=np.float32)
        f = rt_.token_features(maps, (2, 3))
        self.assertEqual(f.shape, (6, 6))
        self.assertTrue(np.allclose(f, np.log(1 + 1e-3), atol=1e-5))
        with self.assertRaises(ValueError):
            rt_.token_features(maps, (3, 3))

    def test_pointer_learns_attended_token(self):
        rng = np.random.default_rng(0)
        items = []
        for _ in range(64):
            g = (4, 4)
            t = int(rng.integers(16))
            maps = rng.dirichlet(np.ones(16), size=4).astype(np.float32)
            maps[0] = 0.01
            maps[0, t] = 0.85
            r, c = divmod(t, 4)
            y = np.asarray([(c + .5) * 250, (r + .5) * 250, c * 250, r * 250, (c + 1) * 250, (r + 1) * 250])
            items.append({'F': rt_.token_features(maps, g), 'centers': rt_.token_centers((1, 8, 8)).numpy(),
                          'q': rt_.token_overlap(y[2:], g), 'y': y, 'grid': g})
        params, _ = rt_.fit_pointer(items[:48], items[48:], epochs=40, batch=16)
        tool = rt_.ReadoutTool('pointer', 'target_end', (4,), params)
        P = rt_.pointer_predict_items(tool.pointer, tool.pmean, tool.pstd, items[48:], tool.tau)
        err = np.linalg.norm(P[:, :2] - np.stack([it['y'][:2] for it in items[48:]]), axis=1)
        self.assertLess(err.mean(), 60)


class HelpersTest(unittest.TestCase):
    def test_batches_respect_budget(self):
        cases = [{'truth': {'canvas': [s, s]}} for s in (256, 512, 512, 1344, 1344, 1344)]
        groups = list(batches(cases, max_tokens=4000, max_batch=8))
        self.assertEqual(sum(len(g) for g in groups), len(cases))
        for g in groups:
            if len(g) > 1:
                n = max((c['truth']['canvas'][0] // 32) ** 2 for c in g)
                self.assertLessEqual(len(g) * n + 200 * len(g), 4000)

    def test_stats(self):
        lo, hi = wilson(5, 10)
        self.assertTrue(lo < 0.5 < hi)
        self.assertAlmostEqual(mcnemar_exact(0, 0), 1.0)
        self.assertLess(mcnemar_exact(20, 2), 0.01)


if __name__ == '__main__':
    unittest.main()
