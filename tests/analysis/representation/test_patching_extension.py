"""GPU-free checks of experiments 2.5b / 2.5c: pairs, answer parsing, contrasts, jobs and probes."""
import unittest

import numpy as np
import torch

from analysis_tools.representation.patching_extension import (
    AXIS_INDEX, GEN_CONDITIONS, Job, YESNO_CONDITIONS, answer_cell, build_jobs, choose_contrast,
    coordinate_digits, coordinate_spans, edits_for, make_gen_pairs, make_yesno_pairs, paired_probe,
    parse_rowcol_loose, recovery, render_pair, t3_prompt, wilson,
)


class CharTokenizer:
    """Stand-in tokenizer: one token per character; digit tokens are their own code points."""

    def convert_tokens_to_ids(self, token):
        return ord(token)


class FakeRuntime:
    tokenizer = CharTokenizer()


def encode(text):
    return [ord(ch) for ch in text]


def decode(ids):
    return ''.join(chr(i) for i in ids)


class PairTests(unittest.TestCase):
    def test_move_pairs_keep_the_other_coordinate(self):
        for cond, cfg in GEN_CONDITIONS.items():
            for pair in make_gen_pairs(cond, 12, seed=0):
                if cfg['pairing'] == 'remove':
                    self.assertIsNone(pair.corrupt_target)
                    self.assertEqual(pair.corrupt.change.kind, 'none')
                    continue
                a, b = pair.clean_target, pair.corrupt_target
                fixed = 0 if cfg['axis'] == 'col' else 1
                self.assertEqual(a[fixed], b[fixed])
                self.assertNotEqual(a[1 - fixed], b[1 - fixed])

    def test_pixel_difference_is_confined_to_anomaly_cells(self):
        for cond in ('t6_rowcol_col', 't5_rowcol_row', 't5_rowcol_remove'):
            for pair in make_gen_pairs(cond, 4, seed=0):
                render_pair(pair)           # raises on any difference outside the cells
        for cond in YESNO_CONDITIONS:
            if YESNO_CONDITIONS[cond]['grid'] <= 10:
                for pair in make_yesno_pairs(cond, 3, seed=0):
                    render_pair(pair)

    def test_revised_t3_prompt(self):
        text = t3_prompt((1, 4))
        self.assertIn('row 2, column 5 that differs from the majority of the figures?', text)
        self.assertTrue(text.endswith('Answer with only Yes or No.'))


class AnswerTests(unittest.TestCase):
    def test_loose_rowcol_accepts_bad_bracket(self):
        self.assertEqual(parse_rowcol_loose('```json\n[\n {"row": 4, "column": 4]\n```'), (4, 4))

    def test_point_cell(self):
        meta = {'canvas': [384, 384], 'grid_origin': [0, 0], 'spec': {'cell': 64, 'rows': 6, 'cols': 6}}
        self.assertEqual(answer_cell('[{"point_2d": [583, 250]}]', 'point', meta), (1, 3))
        self.assertIsNone(answer_cell('[{"point_2d": [583]}]', 'point', meta))

    def test_coordinate_digits_and_contrast(self):
        clean = encode('```json\n{"row": 3, "column": 5}\n```')
        digits = coordinate_digits(decode, clean, 'rowcol')
        self.assertEqual([v for _, v in digits], ['3', '5'])
        row = choose_contrast({}, clean, digits, ['3', '2'], 'corrupt_answer', FakeRuntime(), AXIS_INDEX['rowcol']['col'])
        self.assertEqual(row['status'], 'scan')
        self.assertEqual(decode(row['prefix']), '```json\n{"row": 3, "column": ')
        self.assertEqual((chr(row['token_a']), chr(row['token_b'])), ('5', '2'))

    def test_failure_contrast_scores_truth_against_output(self):
        clean = encode('{"row": 12, "column": 7}')
        digits = coordinate_digits(decode, clean, 'rowcol')
        row = choose_contrast({}, clean, digits, ['13', '7'], 'truth', FakeRuntime())
        self.assertEqual((chr(row['token_a']), chr(row['token_b'])), ('3', '2'))
        self.assertEqual(decode(row['prefix']), '{"row": 1')
        row = choose_contrast({}, encode('{"row": 1, "column": 7}'),
                              coordinate_digits(decode, encode('{"row": 1, "column": 7}'), 'rowcol'),
                              ['10', '7'], 'truth', FakeRuntime())
        self.assertEqual(row['status'], 'digit_count_differs')

    def test_point_spans_follow_xy_order(self):
        text = '[{"point_2d": [125, 749]}]'
        (a, b), (c, d) = coordinate_spans(text, 'point')
        self.assertEqual((text[a:b], text[c:d]), ('125', '749'))
        self.assertEqual(AXIS_INDEX['point']['col'], 0)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.groups = {'target_clean': [3, 4], 'targets': [3, 4, 5], 'random': [6, 7], 'nontarget': [5, 6, 7, 8],
                       'all_visual': [3, 4, 5, 6, 7, 8], 'question': [10, 11], 'q_end': [11], 'think_end': [13],
                       'last_prompt': [14], 'prompt_text': [9, 10, 11, 12, 13, 14], 'query': [16],
                       'answer_prefix': [15]}

    def test_path_modes_only_for_generation(self):
        gen = build_jobs(self.groups, 'gen', 'restore', (0, 1), (1,))
        yesno = build_jobs(self.groups, 'yesno', 'restore', (0, 1), (1,))
        self.assertTrue(any(j.mode == 'via' for j in gen))
        self.assertFalse(any(j.mode in ('via', 'direct') for j in yesno))
        self.assertFalse(any(j.site == 0 and j.group == 'prompt_text' for j in gen))
        self.assertTrue(any(j.mode == 'via' for j in build_jobs(self.groups, 'gen', 'damage', (1,), (1,))))

    def test_direct_and_via_edits_cover_the_right_positions(self):
        L, P = 17, 15
        src = [torch.full((L, 2), 1.0) for _ in range(33)]
        rec = [torch.zeros(L, 2) for _ in range(33)]
        direct = edits_for(Job('direct', 30, 'all_visual'), self.groups, L, P, src, rec, None)
        self.assertEqual({e[0] for e in direct}, {30, 31, 32})
        for site, pos, source in direct:
            self.assertTrue(int(pos.max()) < P)          # answer positions are never fixed
        text = torch.tensor(self.groups['prompt_text'])
        capture = {k: torch.full((len(text), 2), 7.0) for k in range(31, 33)}
        via = edits_for(Job('via', 30, 'all_visual'), self.groups, L, P, src, rec, capture)
        self.assertEqual({e[0] for e in via}, {31, 32})
        text_edit = [e for e in via if torch.equal(e[1], text)][0]
        self.assertTrue(torch.all(text_edit[2][text] == 7.0))
        d1 = edits_for(Job('direct1', 31, 'targets'), self.groups, L, P, src, rec, None)
        self.assertEqual(sorted({e[0] for e in d1}), [31, 32])


class MetricTests(unittest.TestCase):
    def test_recovery_and_wilson(self):
        self.assertAlmostEqual(float(recovery(1.0, -1.0, 3.0)), 0.5)
        lo, hi = wilson(5, 10)
        self.assertTrue(lo < 0.5 < hi)

    def test_paired_probe_detects_consistent_direction(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(size=(40, 16)) + 2.0
        noise = rng.normal(size=(40, 16)) * np.where(rng.random((40, 1)) < 0.5, 1, -1)
        self.assertGreater(paired_probe(signal, 8, 0)['acc'], 0.9)
        self.assertLess(abs(paired_probe(noise, 8, 0)['acc'] - 0.5), 0.3)


if __name__ == '__main__':
    unittest.main()
