"""Model-free checks of experiment 2.5e: scoring, displacements, attention metrics and the ridge probe."""
import unittest

import numpy as np

from analysis_tools.common.stimuli import Change, StimulusSpec, generate
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.representation import sees_right as sr
from analysis_tools.representation import sees_right_analysis as sa


def grid_case(g=6, rc=(2, 3), fmt='rowcol', style='direct'):
    spec = StimulusSpec(template='ring', rows=g, cols=g, change=Change('color', 4), target=rc, seed=1)
    _, meta = generate(spec)
    return {'set': 't6', 'format': fmt, 'style': style, 'grid': g, 'truth': sr.grid_truth(meta, rc),
            'spec': sr.spec_dict(spec)}


class ScoreTests(unittest.TestCase):
    def test_rowcol_direct_and_think(self):
        c = grid_case()
        s = sr.score(c, '{"row": 3, "column": 4}')
        self.assertTrue(s['correct'])
        self.assertEqual(s['pred_rc'], [2, 3])
        c['style'] = 'think'
        s = sr.score(c, 'Row 1 has {"row": 1, "column": 1} ... final {"row": 3, "column": 5}')
        self.assertFalse(s['correct'])
        self.assertEqual((s['d_row'], s['d_col']), (0, 1))

    def test_point_grid_cell(self):
        c = grid_case(fmt='point')
        t = c['truth']
        u = [1000 * t['center'][0] / t['canvas'][0], 1000 * t['center'][1] / t['canvas'][1]]
        s = sr.score(c, '[{"point_2d": [%d, %d]}]' % tuple(u))
        self.assertTrue(s['correct'])
        self.assertEqual(s['pred_rc'], [2, 3])

    def test_cell_center_round_trip(self):
        c = grid_case()
        u = sr.cell_center_units(c['truth'], (4, 1))
        self.assertEqual(sr.point_cell(c['truth'], np.asarray(u)), (4, 1))

    def test_disc_and_invalid(self):
        image, truth = cr.disc_image((100.0, 300.0), 64)
        case = {'set': 'disc', 'format': 'point', 'style': 'direct', 'truth': truth}
        x, y = 1000 * truth['center'][0] / 512, 1000 * truth['center'][1] / 512
        self.assertTrue(sr.score(case, '[{"point_2d": [%d, %d]}]' % (x, y))['correct'])
        self.assertFalse(sr.score(case, 'no json')['valid'])


class MoveTests(unittest.TestCase):
    def test_grid_moves_stay_inside(self):
        c = grid_case(g=6, rc=(5, 5))
        grid = (12, 12)
        target = cr.tokens_overlapping(c['truth']['box'], grid)
        mv = sr.moves(c, grid, target, 2, 4)
        self.assertEqual({m['axis'] for m in mv}, {'h', 'v'})
        self.assertTrue(all(m['sign'] == -1 for m in mv))
        self.assertTrue(all(not m['overlap'] for m in mv))

    def test_shift_out_of_bounds(self):
        self.assertIsNone(sr.shift_tokens([0, 1], (4, 4), 0, -1))
        self.assertEqual(sr.shift_tokens([0, 1], (4, 4), 1, 1), [5, 6])


class MetricTests(unittest.TestCase):
    def test_peak_classes(self):
        c = grid_case(g=6, rc=(2, 3))
        grid = (12, 12)
        geo = sa.geometry(c, grid)
        T = np.where(geo['target'])[0]
        A = np.where(geo['adjacent'])[0]
        a = np.full((3, 144), 1e-4)
        a[0, T[0]] = 1.0
        a[1, A[0]] = 1.0
        a[2, 0] = 1.0
        m = sa.attention_metrics(a, geo, c)
        self.assertEqual(m['pcls'].tolist(), [0, 1, 2])
        self.assertEqual(m['rank'][0], 1)
        self.assertGreater(m['rank'][1], 1)
        self.assertEqual(m['cell'][0].tolist(), [2, 3])
        self.assertTrue(m['paxis'][0].all())

    def test_ridge_recovers_linear_map(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 50))
        y = X @ rng.normal(size=50) + 3
        pred, _ = sa.ridge_fit_predict(X[:150], y[:150], X[150:])
        self.assertLess(np.abs(pred - y[150:]).mean(), 0.2 * np.abs(y[150:] - y.mean()).mean())


if __name__ == '__main__':
    unittest.main()
