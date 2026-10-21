"""Tests of experiment 1.6 (PE scaling and T0 variation): case selection, T0 rendering, scoring, statistics."""
from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from analysis_tools.behavior import pe_scaling_t0 as pt
from analysis_tools.behavior import pe_scaling_t0_analysis as pa
from analysis_tools.common.stimuli import TOKEN_PIXELS, WHITE


def _case(task, fmt, grid, kind, level, i, stage='main', wording='primary', variant='target'):
    return {'id': f'{stage}|{task}|{fmt}|{grid}|{kind}|{level}|{variant}|{i}', 'task': task, 'format': fmt,
            'grid': grid, 'kind': kind, 'level': level, 'stage': stage, 'wording': wording, 'variant': variant,
            'thinking': False}


class GridSelectionTest(unittest.TestCase):
    def test_strata_sizes(self):
        cases = []
        for i in range(20):
            for kind in ('gap', 'color'):
                for level in (1, 2):
                    cases.append(_case('T4', 'row_column', 6, kind, level, i))
                    cases.append(_case('T4', 'bbox', 6, kind, level, i))         # format not selected
                    cases.append(_case('T4', 'row_column', 3, kind, level, i))   # grid not selected
                    cases.append(_case('T1', 'yes_no', 3, kind, level, i, variant='neg'))
                    cases.append(_case('T4', 'row_column', 6, kind, level, i, wording='alt'))
        cases += [_case('T6', 'point', 10, 'color', 4, i) for i in range(5)]
        cases += [_case('T5', 'row_column', 20, 'gap', 1, i, stage='explicit') for i in range(3)]
        sel = pt.select_grid_cases(cases, per_combo=3, control_per_combo=2, seed=0)
        count = lambda f: sum(f(c) for c in sel)
        self.assertEqual(count(lambda c: c['task'] == 'T4'), 4 * 3)
        self.assertEqual(count(lambda c: c['task'] == 'T1'), 4 * 2)
        self.assertEqual(count(lambda c: c['task'] == 'T6'), 5)
        self.assertEqual(count(lambda c: c['subset'] == 'explicit'), 3)
        self.assertTrue(all(c['wording'] == 'primary' for c in sel))
        self.assertEqual(sel, pt.select_grid_cases(cases, 3, 2, 0))       # deterministic

    def test_too_small_stratum_raises(self):
        cases = [_case('T4', 'point', 6, 'gap', 1, i) for i in range(2)]
        with self.assertRaises(ValueError):
            pt.select_grid_cases(cases, per_combo=3, control_per_combo=1, seed=0)


class T0RenderTest(unittest.TestCase):
    def test_render_jitter_colour_mixed(self):
        rng = np.random.default_rng(0)
        color = pt.sample_stroke_color(rng)
        templates = ['person', 'ring', 'square', 'arrow'] * 6
        image, meta = pt.render_t0(4, 6, 64, 8, color, templates, seed=3)
        a = np.asarray(image)
        self.assertEqual(a.shape[0] % TOKEN_PIXELS, 0)
        self.assertEqual(a.shape[1] % TOKEN_PIXELS, 0)
        colours = {tuple(p) for p in a.reshape(-1, 3)}
        self.assertEqual(colours, {(WHITE,) * 3, color})
        self.assertEqual(len(meta['instances']), 24)
        jit = np.array([i['jitter'] for i in meta['instances']])
        self.assertLessEqual(np.abs(jit).max(), 8)
        self.assertGreater(len({tuple(j) for j in jit.tolist()}), 1)
        self.assertEqual([i['template'] for i in meta['instances']], templates)
        for inst in meta['instances']:
            x0, y0, x1, y1 = inst['tight_box']
            self.assertTrue(0 <= x0 < x1 <= a.shape[1] and 0 <= y0 < y1 <= a.shape[0])

    def test_colour_is_dark(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            rgb = pt.sample_stroke_color(rng)
            self.assertLess(sum(rgb) / 3, 160)

    def test_build_and_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / 'ds'
            old = pt.T0_LAYOUTS
            pt.T0_LAYOUTS = ((3, 3, 64), (2, 4, 64))
            try:
                pt.build_t0(ds, n_per_variation=2, seed=5)
                with self.assertRaises(FileExistsError):
                    pt.build_t0(ds, n_per_variation=2, seed=5)
            finally:
                pt.T0_LAYOUTS = old
            meta = pt.read_jsonl(ds / 'metadata.jsonl')
            self.assertEqual(len(meta), 2 * (4 + 4 + 3 * 2))   # V0 and V0s: one image per template
            self.assertEqual(len({m['sha256'] for m in meta if m['variation'] not in ('V0_fixed', 'V0s_shifted')}), 2 * 3 * 2)
            cases = pt.t0_cases(ds, explicit_per_layout=1)
            self.assertEqual(sum(c['stage'] == 't0' for c in cases), len(meta))
            self.assertEqual(sum(c['stage'] == 'explicit' for c in cases), 2)


class ScoringTest(unittest.TestCase):
    def _t0(self, stage='t0'):
        return {'id': 'x', 'stage': stage, 'layout': 'l', 'variation': 'v', 'template': 't',
                'truth': {'rows': 20, 'cols': 10, 'canvas': [672, 1312]}}

    def test_t0_strict_and_tolerant(self):
        rec = {'condition': 'x1', 'finish_reason': 'eos', 'n_tokens': 9, 'response': '{"rows": 20, "columns": 10}'}
        s = pa.score_t0(self._t0(), rec)
        self.assertTrue(s['strict'] and s['tolerant'])
        s = pa.score_t0(self._t0(), {**rec, 'response': '{"rows": 21, "columns": 10'})
        self.assertFalse(s['strict'])
        self.assertFalse(s['tolerant'])
        self.assertEqual((s['d_rows'], s['d_cols']), (1, 0))
        s = pa.score_t0(self._t0(), {**rec, 'response': '{"rows": 20, "columns": 10', 'finish_reason': 'length'})
        self.assertFalse(s['tolerant'])
        s = pa.score_t0(self._t0('explicit'), {**rec, 'response': 'I count 20 rows.\n{"rows": 20, "columns": 10}'})
        self.assertTrue(s['strict'])

    def test_text_features(self):
        f = pa.text_features('Wait, assuming a 7x7 grid, go left then up to (2,3). Wait.', 100)
        self.assertTrue(f['hedge'])
        self.assertEqual(f['n_wait'], 2)
        self.assertEqual(f['grid_decls'], ['7x7'])
        self.assertAlmostEqual(f['direction_per_k'], 20.0)
        self.assertAlmostEqual(f['coord_per_k'], 10.0)


class StatsTest(unittest.TestCase):
    def test_mcnemar_and_unpaired(self):
        base = {1: True, 2: False, 3: False, 4: True}
        other = {1: True, 2: True, 3: True, 4: False}
        m = pa.mcnemar(base, other)
        self.assertEqual((m['gained'], m['lost'], m['n']), (2, 1, 4))
        with self.assertRaises(ValueError):
            pa.mcnemar(base, {1: True})

    def test_fisher_and_wilson(self):
        self.assertAlmostEqual(pa.fisher_two_sided(3, 3, 0, 3), 0.1, places=6)
        self.assertAlmostEqual(pa.fisher_two_sided(2, 4, 2, 4), 1.0)
        lo, hi = pa.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_pe_ratio_matches_experiment_37(self):
        src = (Path(__file__).resolve().parents[3] / 'analysis_tools/position/vit_layout_run.py').read_text()
        self.assertEqual(float(re.search(r'^PE_RATIO = ([\d.]+)', src, re.M).group(1)), pt.PE_RATIO)


if __name__ == '__main__':
    unittest.main()
