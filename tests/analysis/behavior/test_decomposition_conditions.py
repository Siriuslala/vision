"""Case construction and scoring checks for the supplementary conditions of experiment 1.3 (no GPU)."""
import json
from pathlib import Path
import tempfile
import unittest

from analysis_tools.behavior import decomposition as d
from analysis_tools.behavior import decomposition_conditions as c

CFG = dict(seed=0, dev_reps=4, dev_levels=[2, 4], t3_candidates=list(c.T3_CANDIDATES), t3_selected='',
           layouts=['g4x8', 'f24c32'], layout_t3_candidates=['C0_original', 'C7_odd'], search_per_change=2,
           red_per_layout=2, count_per_layout=1, rings=['n8', 'n8_r128'], ring_full=['n8'])


class PromptTests(unittest.TestCase):
    def test_original_candidate_reproduces_formal_wording(self):
        rc = (2, 4)
        formal_rowcol = d.binary_prompt(d.T3_ROWCOL[0].format(r=3, c=5), 'direct')
        formal_box = d.binary_prompt(d.T3_BOX[0], 'direct')
        self.assertEqual(c.t3_prompt('C0_original', 'rowcol', rc)[0], formal_rowcol)
        self.assertEqual(c.t3_prompt('C0_original', 'box', rc)[0], formal_box)

    def test_polarity_and_reference(self):
        text, yes_same = c.t3_prompt('C7_odd', 'rowcol', (0, 0))
        self.assertFalse(yes_same)
        self.assertIn('row 1, column 1', text)
        with self.assertRaises(ValueError):
            c.t3_prompt('C4_reference', 'box', (0, 0))
        text, _ = c.t3_prompt('C4_reference', 'box', (0, 0), (2, 3))
        self.assertIn('row 3, column 4', text)
        self.assertIn(d.ORDINAL_NOTE, text)

    def test_reference_avoids_designated_and_target(self):
        for seed in range(50):
            self.assertNotIn(c.reference_rc(3, 3, [(0, 0), (1, 1)], seed), [(0, 0), (1, 1)])


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        for part in ('layouts', 'ring'):
            c.build(cls.out, cls.out, CFG, part)
        cls.layouts = d.read_jsonl(cls.out / 'cases_layouts.jsonl')
        cls.ring = d.read_jsonl(cls.out / 'cases_ring.jsonl')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_layout_truth_matches_spec(self):
        for case in self.layouts:
            rows, cols, cell = c.LAYOUTS[case['layout']]
            if case['task'] == 'T3':
                self.assertEqual(case['truth']['same'], case['variant'].endswith('normal'))
                self.assertEqual(case['truth']['answer_yes'], case['truth']['same'] == case['yes_means_same'])
                continue
            self.assertEqual((case['truth']['rows'], case['truth']['cols'], case['truth']['cell']), (rows, cols, cell))
        canvases = {cs['layout']: tuple(cs['truth']['canvas']) for cs in self.layouts if 'canvas' in cs['truth']}
        self.assertEqual(canvases, {'g4x8': (512, 256), 'f24c32': (768, 768)})

    def test_rebuild_accepts_added_cases_only(self):
        c.build(self.out, self.out, CFG, 'layouts')
        bigger = {**CFG, 'layout_t3_candidates': ['C0_original', 'C7_odd', 'C3_copy']}
        c.build(self.out, self.out, bigger, 'layouts')
        smaller_changed = {**CFG, 'seed': 1}
        with self.assertRaises(ValueError):
            c.build(self.out, self.out, smaller_changed, 'layouts')

    def test_ring_cases(self):
        tasks = {(cs['layout'], cs['task'], cs['format']) for cs in self.ring}
        self.assertIn(('n8', 'T5', 'index'), tasks)
        self.assertIn(('n8', 'T0', 'ring_count'), tasks)
        self.assertNotIn(('n8_r128', 'T5', 'point'), tasks)
        for cs in self.ring:
            if 'target_index' in cs['truth']:
                k = cs['truth']['target_index']
                self.assertEqual(cs['truth']['target_cell_box'], cs['truth']['instance_cell_boxes'][k])


def _record(text):
    return {'response': text, 'finish_reason': 'eos'}


class RingScoringTests(unittest.TestCase):
    def setUp(self):
        from analysis_tools.common.stimuli import Change, RingSpec, generate_ring
        _, meta = generate_ring(RingSpec('person', 8, 352, 768, Change('color', 4), target=2))
        self.truth = c.ring_truth(meta)    # target at 3 o'clock

    def case(self, fmt):
        return {'format': fmt, 'truth': self.truth}

    def test_point_hit_radial_and_angle(self):
        x, y = self.truth['target_center']
        s = c.score_ring(self.case('point'), _record(json.dumps([{'point_2d': [1000 * x / 768, 1000 * y / 768]}])), False)
        self.assertTrue(s['correct'])
        self.assertAlmostEqual(s['radial_ratio'], 1.0, places=6)
        self.assertAlmostEqual(s['d_angle_deg'], 0.0, places=6)
        # A point halfway between the canvas centre and the target has radial ratio 0.5.
        s = c.score_ring(self.case('point'), _record(json.dumps([{'point_2d': [500 + (1000 * x / 768 - 500) / 2, 500]}])), False)
        self.assertFalse(s['correct'])
        self.assertAlmostEqual(s['radial_ratio'], 0.5, delta=0.02)

    def test_index_offsets_are_circular(self):
        s = c.score_ring(self.case('index'), _record('{"index": 3}'), False)
        self.assertTrue(s['correct'])
        s = c.score_ring(self.case('index'), _record('{"index": 4}'), False)
        self.assertEqual(s['d_index'], 1)
        self.assertEqual(c._circular(7 - 0, 8), -1)
        s = c.score_ring(self.case('index'), _record('{"index": 9}'), False)
        self.assertEqual(s['failure'], 'out_of_range')
        s = c.score_ring(self.case('index'), _record('[{"index": 3}]'), False)
        self.assertEqual(s['failure'], 'schema')
        s = c.score_ring(self.case('index'), _record('[{"index": 3}]'), True)
        self.assertTrue(s['correct'])

    def test_count(self):
        s = c.score_ring(self.case('ring_count'), _record('{"count": 8}'), False)
        self.assertTrue(s['correct'])


if __name__ == '__main__':
    unittest.main()
