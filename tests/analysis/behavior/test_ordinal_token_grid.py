"""Stimulus geometry, case construction and the coordinate-convention metric of experiment 1.4 (no GPU)."""
from pathlib import Path
import tempfile
import unittest

from analysis_tools.behavior import coordinate_convention as cc
from analysis_tools.behavior import ordinal_token_grid as og
from analysis_tools.behavior.decomposition import read_jsonl, score_case
from analysis_tools.common.stimuli import Change, StimulusSpec, generate


class StimulusTests(unittest.TestCase):
    def test_matches_stimuli_v1_for_token_multiple_cells(self):
        for cell, cols, phase in ((32, 20, 0), (64, 20, 0), (96, 10, 0), (64, 16, 8), (64, 20, 16), (32, 24, 16)):
            spec = og.GridSpec('person', og.ROWS, cols, cell, phase, (2, cols - 1))
            _, meta = og.render(spec)
            ref = StimulusSpec('person', og.ROWS, cols, Change('color', 4), target=(2, cols - 1), cell=cell,
                               phase=phase)
            _, ref_meta = generate(ref)
            self.assertEqual(meta['sha256'], ref_meta['sha256'], (cell, cols, phase))
            self.assertEqual(meta['grid_origin'], ref_meta['grid_origin'])

    def test_free_cells_keep_canvas_on_token_grid(self):
        for cell in (48, 80):
            for n in (10, 16, 20, 24):
                _, meta = og.render(og.GridSpec('ring', og.ROWS, n, cell, 0, (0, 0)))
                w, h = meta['canvas']
                self.assertEqual((w % 32, h % 32), (0, 0))
                self.assertEqual(meta['grid_origin'][0] % 32, 0)
                self.assertGreaterEqual(w, n * cell)
        with self.assertRaises(ValueError):
            og.GridSpec('person', og.ROWS, 15, 48)        # 720 pixels is not a multiple of 32

    def test_red_target_is_unique(self):
        img, meta = og.render(og.GridSpec('square', og.ROWS, 16, 48, 0, (3, 7)))
        red = [i for i in meta['instances'] if i['is_target']]
        self.assertEqual(len(red), 1)
        x0, y0, x1, y1 = red[0]['cell_box']
        patch = img.crop((x0, y0, x1, y1))
        self.assertIn((255, 0, 0), {c for _, c in patch.getcolors(10000)})

    def test_stratified_targets_balance_columns(self):
        t = og.stratified_targets(6, 20, 10, seed=3)
        self.assertEqual(len(t), 200)
        self.assertEqual({c: sum(1 for _, cc_ in t if cc_ == c) for c in range(20)}, {c: 10 for c in range(20)})

    def test_condition_names(self):
        self.assertEqual(og.parse_condition(og.condition_name(80, 24, 8)), (80, 24, 8))
        with self.assertRaises(ValueError):
            og.parse_condition('g20x10')


class BuildTests(unittest.TestCase):
    def test_build_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg = dict(seed=0, conditions=['c48n10p0', 'c64n16p16'], t6_per_condition=20, t0_per_condition=2)
            og.build(out, cfg)
            cases = read_jsonl(out / 'cases.jsonl')
            self.assertEqual(sum(c['task'] == 'T6' for c in cases), 2 * (20 + 32))
            og.build(out, cfg)                               # rebuild is idempotent
            case = next(c for c in cases if c['format'] == 'row_column' and c['condition'] == 'c48n10p0')
            r, c = case['truth']['target_rc']
            rec = {'response': f'{{"row": {r + 1}, "column": {c + 2}}}', 'finish_reason': 'eos'}
            s = score_case(case, rec, diagnostic=False)
            self.assertEqual((s['d_row'], s['d_col']), (0, 1))
            pt = next(x for x in cases if x['format'] == 'point' and x['condition'] == 'c48n10p0')
            cx, cy = pt['truth']['target_center']
            w, h = pt['truth']['canvas']
            rec = {'response': f'[{{"point_2d": [{1000 * cx / w:.0f}, {1000 * cy / h:.0f}]}}]', 'finish_reason': 'eos'}
            self.assertTrue(score_case(pt, rec, diagnostic=False)['hit'])

    def test_column_prefix(self):
        self.assertEqual(og.column_prefix('{"row": 3, "column": 12}'), ('{"row": 3, "column": ', 12, '}'))
        self.assertEqual(og.column_prefix('```json\n{"row": 5, "column": 20}\n```\n'),
                         ('```json\n{"row": 5, "column": ', 20, '}'))
        self.assertIsNone(og.column_prefix('[{"row": 3, "column": 12}]'))


class ConventionTests(unittest.TestCase):
    canvas, box = (1280, 640), (640, 320, 704, 384)

    def test_parse_wrappers_and_large_values(self):
        self.assertEqual(cc.parse_point('[{"point_2d": [10, 20]}]'), (10., 20.))
        self.assertEqual(cc.parse_point('{"point_2d": [1200, 20]}'), (1200., 20.))
        self.assertEqual(cc.parse_point('[3, 4]'), (3., 4.))
        self.assertIsNone(cc.parse_point('[{"bbox_2d": [1, 2, 3, 4]}]'))
        self.assertIsNone(cc.parse_point('not json'))

    def test_hits_under_each_reading(self):
        norm = cc.convention_hits((520., 550.), self.canvas, self.box)      # 665.6, 352 px
        self.assertEqual((norm['h_norm'], norm['h_pix'], norm['pixel_only']), (True, False, False))
        pix = cc.convention_hits((650., 350.), self.canvas, self.box)
        self.assertEqual((pix['h_norm'], pix['h_pix'], pix['h_any'], pix['pixel_only']), (False, True, True, True))
        out = cc.convention_hits((1300., 350.), self.canvas, self.box)    # x beyond the canvas width
        self.assertFalse(out['h_any'])
        self.assertFalse(cc.convention_hits(None, self.canvas, self.box)['point_valid'])

    def test_chance_rates(self):
        p = cc.chance_rates(self.canvas, self.box)
        self.assertAlmostEqual(p['p_norm'], 64 * 64 / (1280 * 640))
        self.assertAlmostEqual(p['p_pix'], 64 * 64 / 1e6)
        self.assertLessEqual(p['p_any'], p['p_norm'] + p['p_pix'] + 1e-12)
        # At a 1000x1000 canvas both readings coincide and the union adds nothing.
        q = cc.chance_rates((1000, 1000), (100, 100, 200, 200))
        self.assertAlmostEqual(q['p_any'], q['p_norm'])
        # A box beyond 1000 pixels cannot be hit under the pixel reading of a 0-1000 output.
        self.assertEqual(cc.chance_rates((1280, 1280), (1100, 0, 1164, 64))['p_pix'], 0.)

    def test_monte_carlo_chance(self):
        import numpy as np
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 1000, (200000, 2))
        hits = np.mean([cc.convention_hits(tuple(p), self.canvas, self.box)['h_any'] for p in pts[:20000]])
        self.assertAlmostEqual(hits, cc.chance_rates(self.canvas, self.box)['p_any'], delta=0.003)


if __name__ == '__main__':
    unittest.main()
