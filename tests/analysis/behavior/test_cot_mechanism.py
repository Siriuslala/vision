"""CPU tests of the experiment-1.7 prompt, stimulus and CoT-coding helpers."""
import unittest

from analysis_tools.behavior import cot_mechanism as cm
from analysis_tools.behavior import cot_mechanism_analysis as ca

COT = ('To determine the row and column of the red figure:\n'
       '1. There are 8 rows and 8 columns.\n'
       '   - Row 1: all black\n'
       '   - Row 2: all black\n'
       '   - Row 3: one red figure\n'
       '   → So, the red figure is in **row 3**.\n'
       '2. In row 3, the red figure is the 5th figure from the left, so it is in column 5.\n'
       'Thus, the red figure is at row 3, column 5.\n\n{"row": 3, "column": 5}\n')
TRUTH = {'row': 3, 'col': 5, 'rows': 8, 'cols': 8}


class CotMechanismTest(unittest.TestCase):
    def test_prompts_match_earlier_experiments(self):
        assert cm.prompt('rowcol', 'think').startswith('One figure is drawn in red. First describe step by step how you '
                                                       'determine its row and its column.')
        assert cm.prompt('rowcol', 'direct').endswith('Output only the JSON, with no explanation.')
        assert 'first line' in cm.prompt('point', 'answer_first')

    def test_pairs_differ_in_row_and_column(self):
        for task in ('rowcol', 'odd', 'point'):
            for a in cm.build_items(task, 16):
                b = cm.pair_item(a)
                assert abs(a.target[0] - b.target[0]) >= 2 and abs(a.target[1] - b.target[1]) >= 2
                assert a.render()[0].size == b.render()[0].size

    def test_truth_point_inside_cell(self):
        a = cm.build_items('point', 4)[0]
        t = cm.truth_of(a)
        x0, y0, x1, y1 = t['cell_box_norm']
        assert x0 <= t['point'][0] <= x1 and y0 <= t['point'][1] <= y1

    def test_code_cot_consistent_and_correct(self):
        code = cm.code_cot('rowcol', COT, 'eos', TRUTH)
        assert code['answer'] == [3, 5] and code['correct']
        assert code['row_claim'] == 3 and code['col_claim'] == 5 and code['consistent']
        assert code['declared_rows'] == 8 and code['grid_rows_ok']
        assert code['red_row_labels'] == [3] and code['row_enum']
        assert cm.primary_mode(code) == 'clean'

    def test_failure_modes(self):
        wrong = COT.replace('{"row": 3, "column": 5}', '{"row": 4, "column": 5}')
        assert cm.primary_mode(cm.code_cot('rowcol', wrong, 'eos', TRUTH)) == 'inconsistent'
        shifted = cm.code_cot('rowcol', COT, 'eos', {**TRUTH, 'row': 4})
        assert cm.primary_mode(shifted) == 'enum_mislabel'
        trunc = cm.code_cot('rowcol', COT[:80], 'length', TRUTH)
        assert cm.primary_mode(trunc) == 'truncated'
        disc = cm.code_cot('rowcol', "Since I can't see the image, " + COT, 'eos', {**TRUTH, 'row': 4})
        assert cm.primary_mode(disc) == 'disclaimer'

    def test_range_and_label_mentions_are_not_claims(self):
        text = 'Rows 1-4 are black. From row 1 to row 6 I count. Row 1 is the topmost row.\nRow 2: black\nIt is in row 5.'
        assert [v for v, _ in cm._claims('row', text)] == [5]

    def test_edit_last_claim(self):
        span = cm.locate_claim('rowcol', COT, 'row')
        assert COT[span[0]:span[1]] == '3' and span[2] == 3
        prefix, new = cm.edit_claim(COT, span, 1)
        assert new == 4 and prefix.endswith('row 4')
        first = cm.locate_first_claim('rowcol', COT, 'row')
        assert first is not None and first[0] < span[0]

    def test_count_and_point_parsing(self):
        assert cm.final_answer('count', 'so {"rows": 7, "columns": 9}')[0] == (7, 9)
        assert cm.final_answer('point', '[{"point_2d": [455, 610]}]')[0] == (455.0, 610.0)
        code = cm.code_cot('count', 'There are 7 rows and 9 columns.\n{"rows": 7, "columns": 9}', 'eos', {'rows': 7, 'cols': 8})
        assert code['consistent'] and code['row_count_correct'] and not code['col_count_correct']


    def test_follow_and_cell_mask(self):
        ta = {'row': 2, 'col': 3, 'point': [300.0, 200.0]}
        tb = {'row': 6, 'col': 7, 'point': [700.0, 600.0]}
        self.assertEqual(ca._follow('rowcol', (2, 3), ta, tb), 'A')
        self.assertEqual(ca._follow('rowcol', (6, 6), ta, tb), 'nearB')
        self.assertEqual(ca._follow('point', (650.0, 610.0), ta, tb), 'B')
        mask = ca._cell_mask({'cell_box_norm': [250.0, 125.0, 375.0, 250.0]}, 16, 16)
        self.assertEqual(int(mask.sum()), 4)
        rows = ca._token_rows(16, 16, 8)
        self.assertEqual(rows[0], 1)
        self.assertEqual(rows[-1], 8)

    def test_token_class(self):
        self.assertEqual(ca.token_class(' red'), 'colour')
        self.assertEqual(ca.token_class('5'), 'digit')
        self.assertEqual(ca.token_class('":'), 'format')


if __name__ == '__main__':
    unittest.main()
