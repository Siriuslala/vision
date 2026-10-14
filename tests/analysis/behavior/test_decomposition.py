"""Case construction, parsing and scoring checks for experiment 1.3 (no GPU required)."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from analysis_tools.behavior import decomposition as d
from analysis_tools.behavior.decomposition_analysis import (
    auroc, auroc_threshold, fit_psychometric, rowcol_error_structure, wilson,
)
from analysis_tools.behavior.decomposition_reasoning import extract_codes
from analysis_tools.behavior.decomposition_supplement import mcnemar_exact

CFG = dict(seed=0, images_per_unit=4, count_images_per_grid=2, alt_period=2, reasoning_grids=[6, 20],
           explicit_levels=[1, 3], explicit_per_cell=1, thinking_levels=[3], thinking_per_cell=1)


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        d.build(cls.out, CFG)
        cls.cases = d.read_jsonl(cls.out / 'cases.jsonl')
        cls.images = {r['path']: r for r in d.read_jsonl(cls.out / 'images.jsonl')}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def main(self, **kw):
        return [c for c in self.cases if c['stage'] == 'main' and all(c[k] == v for k, v in kw.items())]

    def test_binary_tasks_are_balanced_per_image(self):
        for task in ('T1', 'T2', 'T3'):
            rows = self.main(task=task, wording='primary')
            counts = Counter(c['truth']['answer_yes'] for c in rows)
            self.assertEqual(counts[True], counts[False], task)
        # Four T3 queries per image: box/rowcol x target/normal.
        per = Counter(c['record_id'] for c in self.main(task='T3', wording='primary'))
        self.assertEqual(set(per.values()), {4})

    def test_unit_counts_and_template_balance(self):
        for kind in d.KINDS:
            for level in range(1, 5):
                for grid in d.GRIDS:
                    rows = self.main(task='T5', format='row_column', wording='primary', kind=kind, level=level, grid=grid)
                    self.assertEqual(len(rows), CFG['images_per_unit'])
        rot = Counter(c['template'] for c in self.main(task='T5', format='row_column', wording='primary', kind='rotate'))
        self.assertEqual(set(rot), {'person', 'arrow'})
        gap = Counter(c['template'] for c in self.main(task='T5', format='row_column', wording='primary', kind='gap'))
        self.assertEqual(len(set(gap.values())), 1)

    def test_t6_only_on_red_images(self):
        t6 = self.main(task='T6')
        self.assertTrue(t6)
        self.assertTrue(all(c['kind'] == 'color' and c['magnitude'] == 'red' for c in t6))

    def test_target_truth_matches_pixels(self):
        for c in self.main(task='T5', format='row_column', wording='primary')[:40]:
            image = np.asarray(Image.open(self.out / c['image']))
            t = c['truth']
            x0, y0, x1, y1 = (int(v) for v in t['target_cell_box'])
            r, col = t['target_rc']
            self.assertEqual([x0, y0], [t['grid_origin'][0] + col * 64, t['grid_origin'][1] + r * 64])
            tile = image[y0:y1, x0:x1]
            other_rc = ((r + 1) % t['rows'], col)
            ox0, oy0 = t['grid_origin'][0] + other_rc[1] * 64, t['grid_origin'][1] + other_rc[0] * 64
            other = image[oy0:oy0 + 64, ox0:ox0 + 64]
            self.assertFalse((tile == other).all(), 'target instance must differ from a normal instance')

    def test_marker_designates_the_stated_instance(self):
        for c in self.main(task='T3', wording='primary')[:48]:
            if not c['variant'].startswith('box'):
                continue
            rec = self.images[c['image']]
            self.assertEqual(rec['marker_rc'], c['truth']['designated_rc'])
            image = np.asarray(Image.open(self.out / c['image']))
            r, col = c['truth']['designated_rc']
            x0, y0 = rec['grid_origin'][0] + col * 64, rec['grid_origin'][1] + r * 64
            self.assertTrue((image[y0 + 1, x0 + 10] == [0, 0, 255]).all())
            expected_yes = c['truth']['designated_rc'] != c['truth'].get('target_rc', rec['target_rc'])
            self.assertEqual(c['truth']['answer_yes'], expected_yes)

    def test_rowcol_prompt_uses_one_based_ordinals(self):
        c = self.main(task='T3', variant='rowcol_target', wording='primary')[0]
        r, col = c['truth']['designated_rc']
        self.assertIn(f'row {r + 1}, column {col + 1}', c['prompt'])

    def test_single_and_pair_images_vary_in_position_and_keep_size(self):
        offsets = set()
        for c in self.main(task='T1', variant='neg', wording='primary'):
            rec = self.images[c['image']]
            self.assertEqual(rec['canvas'], [256, 256])
            offsets.add(tuple(rec['boxes'][0][:2]))
        self.assertGreater(len(offsets), 10)
        for c in self.main(task='T2', variant='neg', wording='primary')[:20]:
            image = np.asarray(Image.open(self.out / c['image']))
            boxes = self.images[c['image']]['boxes']
            self.assertEqual(len(boxes), 2)
            crops = []
            for b in boxes:
                x = int(b[0]) // 32 * 32
                crops.append(image[:, x:x + 64])
            self.assertEqual(image.shape, (256, 256, 3))

    def test_pair_negative_has_exactly_one_changed_instance(self):
        for c in self.main(task='T2', wording='primary')[:30]:
            image = np.asarray(Image.open(self.out / c['image']))
            boxes = self.images[c['image']]['boxes']
            y0 = int(boxes[0][1]) // 32 * 32
            x0 = int(boxes[0][0]) // 32 * 32
            left, right = image[y0:y0 + 64, x0:x0 + 64], image[y0:y0 + 64, x0 + 64:x0 + 128]
            self.assertEqual(bool((left == right).all()), c['truth']['answer_yes'])

    def test_alternative_wording_subset(self):
        alt = {c['record_id'] for c in self.main(wording='alt') if c['task'] != 'T0'}
        self.assertTrue(alt)
        self.assertTrue(all(int(rid[-2:]) % CFG['alt_period'] == 0 for rid in alt))

    def test_reasoning_cases_reuse_main_images(self):
        main_images = {c['image'] for c in self.main()}
        for stage in ('explicit', 'thinking'):
            rows = [c for c in self.cases if c['stage'] == stage]
            self.assertTrue(rows)
            self.assertTrue(all(c['image'] in main_images for c in rows))
            self.assertEqual({c['grid'] for c in rows}, {6, 20})
        self.assertTrue(all(c['thinking'] for c in self.cases if c['stage'] == 'thinking'))

    def test_rebuild_is_identical_and_config_change_raises(self):
        d.build(self.out, CFG)
        with self.assertRaises(ValueError):
            d.build(self.out, {**CFG, 'seed': 1})


def _case(fmt, stage='main', rows=10, cols=10, target=(2, 3)):
    origin, cell = [0, 0], 64
    tr, tc = target
    box = [tc * 64 + 16, tr * 64 + 8, tc * 64 + 49, tr * 64 + 59]
    truth = dict(canvas=[640, 640], grid_origin=origin, cell=cell, rows=rows, cols=cols, target_rc=[tr, tc],
                 target_cell_box=[tc * 64, tr * 64, tc * 64 + 64, tr * 64 + 64], target_tight_box=box,
                 target_center=[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return dict(format=fmt, stage=stage, truth=truth)


def _rec(text, finish='eos'):
    return dict(response=text, finish_reason=finish)


class ScoringTests(unittest.TestCase):
    def test_row_column(self):
        c = _case('row_column')
        self.assertTrue(d.score_locate(c, _rec('{"row": 3, "column": 4}'))['correct'])
        r = d.score_locate(c, _rec('```json\n{"row": 4, "column": 4}\n```'))
        self.assertTrue(r['valid'])
        self.assertEqual((r['d_row'], r['d_col']), (1, 0))
        self.assertFalse(d.score_locate(c, _rec('{"row": 11, "column": 4}'))['valid'])
        self.assertFalse(d.score_locate(c, _rec('{"row": "3", "column": 4}'))['valid'])
        self.assertFalse(d.score_locate(c, _rec('The answer is {"row": 3, "column": 4}'))['valid'])
        self.assertFalse(d.score_locate(c, _rec('{"row": 3, "column": 4}', 'length'))['valid'])

    def test_explicit_reasoning_takes_last_json(self):
        c = _case('row_column', stage='explicit')
        text = 'Row 1 looks fine {"row": 1, "column": 1} ... final:\n{"row": 3, "column": 4}'
        self.assertTrue(d.score_locate(c, _rec(text))['correct'])

    def test_point_hit_and_distance(self):
        c = _case('point')
        # Target centre (224.5, 161.5) px -> normalized (350.8, 252.3).
        r = d.score_locate(c, _rec('[{"point_2d": [351, 252]}]'))
        self.assertTrue(r['hit'])
        self.assertLess(r['cell_distance'], 0.05)
        r = d.score_locate(c, _rec('[{"point_2d": [451, 252]}]'))   # one cell to the right
        self.assertFalse(r['hit'])
        self.assertAlmostEqual(r['cell_distance'], 1.0, delta=0.02)
        self.assertEqual(r['pred_rc'], [2, 4])

    def test_bare_point_only_in_diagnostic(self):
        c = _case('point')
        self.assertFalse(d.score_locate(c, _rec('[351, 252]'))['valid'])
        self.assertTrue(d.score_locate(c, _rec('[351, 252]'), diagnostic=True)['hit'])
        self.assertFalse(d.score_locate(c, _rec('[1, 2, 3, 4]'), diagnostic=True)['valid'])
        self.assertFalse(d.score_locate(c, _rec('{"point_2d": [351, 252]}'))['valid'])
        self.assertTrue(d.score_locate(c, _rec('{"point_2d": [351, 252]}'), diagnostic=True)['hit'])

    def test_listed_row_column_only_in_diagnostic(self):
        c = _case('row_column')
        text = '```json\n[\n    {"row": 3, "column": 4}\n]\n```'
        self.assertFalse(d.score_locate(c, _rec(text))['valid'])
        self.assertTrue(d.score_locate(c, _rec(text), diagnostic=True)['correct'])

    def test_bbox_iou_and_degenerate(self):
        c = _case('bbox')
        b = c['truth']['target_tight_box']
        norm = [round(v / 640 * 1000, 3) for v in b]
        r = d.score_locate(c, _rec('[{"bbox_2d": %s}]' % norm))
        self.assertTrue(r['correct'])
        self.assertGreater(r['iou'], 0.99)
        self.assertFalse(d.score_locate(c, _rec('[{"bbox_2d": [500, 500, 400, 600]}]'))['valid'])
        self.assertFalse(d.score_locate(c, _rec('[{"bbox_2d": [0, 0, 1200, 10]}]'))['valid'])

    def test_count(self):
        c = dict(format='count', stage='main', truth=dict(rows=6, cols=6))
        self.assertTrue(d.score_count(c, _rec('{"rows": 6, "columns": 6}'))['correct'])
        r = d.score_count(c, _rec('{"rows": 6, "columns": 7}'))
        self.assertTrue(r['rows_correct'])
        self.assertFalse(r['correct'])

    def test_binary_sign_and_format(self):
        c = dict(format='yes_no', stage='main', truth=dict(answer_yes=False))
        r = d.score_binary(c, dict(response='No.', finish_reason='eos', s=-1.0))
        self.assertTrue(r['correct'] and r['format_valid'])
        r = d.score_binary(c, dict(response='Yes, because', finish_reason='eos', s=0.5))
        self.assertFalse(r['correct'] or r['format_valid'])
        e = dict(format='yes_no', stage='explicit', truth=dict(answer_yes=True))
        r = d.score_binary(e, dict(response='The figure ... \nAnswer: Yes', finish_reason='eos'))
        self.assertTrue(r['correct'])
        r = d.score_binary(e, dict(response='I think yes', finish_reason='eos'))
        self.assertFalse(r['valid'])
        # The runner also stores s for reasoning stages; the parsed final answer must decide.
        r = d.score_binary(e, dict(response='Step 1 ...\nAnswer: Yes', finish_reason='eos', s=-3.0))
        self.assertTrue(r['correct'] and r['valid'])

    def test_thinking_answer_part(self):
        c = dict(format='yes_no', stage='thinking', truth=dict(answer_yes=True))
        r = d.score_binary(c, dict(response='hmm, no wait...\n</think>\n\nYes', finish_reason='eos'))
        self.assertTrue(r['correct'])


class StatisticsTests(unittest.TestCase):
    def test_auroc_and_wilson(self):
        self.assertEqual(auroc([2, 3], [0, 1]), 1.0)
        self.assertEqual(auroc([1], [1]), 0.5)
        lo, hi = wilson(24, 48)
        self.assertAlmostEqual(hi - 0.5, 0.137, delta=0.005)

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 5), 2 / 32)
        self.assertAlmostEqual(mcnemar_exact(3, 3), 1.0)

    def test_psychometric_threshold(self):
        x = np.log(np.repeat([2., 4., 8., 16.], 200))
        p = 0.5 + 0.5 / (1 + np.exp(-(x - np.log(5.0)) / 0.3))
        rng = np.random.default_rng(0)
        fit = fit_psychometric(x, (rng.random(x.size) < p).astype(float))
        self.assertAlmostEqual(fit['delta75'], 5.0, delta=0.8)
        self.assertEqual(fit['status'], 'in_range')

    def test_auroc_threshold_interpolates_on_log_magnitude(self):
        th = auroc_threshold([2., 4., 8., 16.], [0.5, 0.7, 0.8, 0.9])
        self.assertEqual(th['status'], 'in_range')
        self.assertAlmostEqual(th['delta75'], 4 * 2 ** 0.5, places=6)  # halfway between 4 and 8 in log
        self.assertEqual(auroc_threshold([2., 4.], [0.9, 1.0])['status'], 'below_range')
        self.assertEqual(auroc_threshold([2., 4.], [0.5, 0.6])['status'], 'above_range')

    def test_rowcol_error_structure(self):
        def row(dr, dc, pred):
            return {'task': 'T5', 'format': 'row_column', 'wording': 'primary', 'grid': 20,
                    'score': {'valid': True, 'correct': dr == dc == 0, 'd_row': dr, 'd_col': dc, 'pred_rc': pred}}
        rows = [row(0, 1, [3, 4]), row(0, 1, [5, 6]), row(2, 3, [0, 0]), row(0, 0, [1, 1])]
        v = rowcol_error_structure(rows)['T5|20']
        self.assertEqual(v['n_wrong'], 3)
        self.assertAlmostEqual(v['col_only'], 2 / 3)
        self.assertAlmostEqual(v['adjacent'], 2 / 3)
        self.assertAlmostEqual(v['pred_11'], 1 / 3)


class ReasoningCodeTests(unittest.TestCase):
    def test_extracts_grid_and_position(self):
        text = ('The image shows a 6x6 grid of figures. Scanning each row, the figure in row 2, column 5 has a gap.\n'
                '{"row": 2, "column": 5}')
        codes = extract_codes(text, final_rc=(2, 5))
        self.assertEqual(codes['declared_grid'], [6, 6])
        self.assertEqual(codes['declared_rc'], [2, 5])
        self.assertTrue(codes['consistent'])
        codes = extract_codes('There are six rows and six columns. Row 3, Column 1 differs.', final_rc=(3, 2))
        self.assertEqual(codes['declared_grid'], [6, 6])
        self.assertFalse(codes['consistent'])
        codes = extract_codes('It is a 10x10 grid; the figure in the **third row, fifth column** is red.', final_rc=(3, 5))
        self.assertEqual(codes['declared_rc'], [3, 5])
        self.assertTrue(codes['consistent'])

    def test_extracts_list_formats_and_ignores_movements(self):
        codes = extract_codes('Rows: 6 / Columns: 6.\nAnswer:\n- Row: 6\n- Column: 5', final_rc=(6, 5))
        self.assertEqual((codes['declared_grid'], codes['declared_rc']), ([6, 6], [6, 5]))
        codes = extract_codes('The red one is in the **row 1**, **column 3**; the 4th row and 6th column is normal.', None)
        self.assertEqual(codes['declared_rc'], [4, 6])
        codes = extract_codes('We count down 13 rows, then across 15 columns. Rows: 1, 2, 3. Columns: 1, 2.', None)
        self.assertIsNone(codes['declared_grid'])
        codes = extract_codes('Rows: 1 to 18. Columns: 1 to 20. The first 10 rows are black.', None)
        self.assertEqual(codes['declared_grid'], [18, 20])


if __name__ == '__main__':
    unittest.main()
