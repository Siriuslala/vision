import unittest

from analysis_tools.deepstack.babyvision_diagnostic import group
from eva.benchmark import score
from eva.diagnostic_scoring import (clean, diagnostic_score, extract_final, match_blank, parse_int_list,
                                    parse_pairs, parse_position)

ROWCOL_Q = ('Find the position of the different item. The answer format is (a,b), which means the '
            'a-th row from the top and the b-th column from the left.')
PLAIN_Q = 'Connect the shapes with their top views.'


def blank(answer, question=ROWCOL_Q):
    return {'ansType': 'blank', 'blankAns': answer, 'question': question, 'options': []}


def choice(index, n=4):
    return {'ansType': 'choice', 'choiceAns': index, 'question': 'Pick one.', 'options': [str(i) for i in range(n)]}


class ExtractionTest(unittest.TestCase):
    def test_marker_inside_line_and_wrappers(self):
        self.assertEqual(extract_final('...\nSo Final answer: B\n'), ('B', 'marker'))
        self.assertEqual(extract_final('So "Final answer: 9"\n')[0], '9')
        self.assertEqual(extract_final('**Final answer: C**')[0], 'C')
        self.assertEqual(extract_final('Final Answer: \\boxed{2,3,4}**')[0], '2,3,4')
        self.assertEqual(extract_final('The final answer is \\boxed{B}')[0], 'B')
        self.assertEqual(extract_final('Final answer: Final answer: 7.')[0], '7')

    def test_last_marker_wins(self):
        self.assertEqual(extract_final('Final answer: A\nwait\nFinal answer: D')[0], 'D')

    def test_marker_on_its_own_line_takes_next_line(self):
        self.assertEqual(extract_final('Final Answer:\n\n\\boxed{C}\n'), ('C', 'marker_next_line'))

    def test_boxed_last_line_without_marker(self):
        self.assertEqual(extract_final('reasoning\nSo it is 5.\n\\boxed{5}\n'), ('5', 'boxed'))
        # A boxed value inside reasoning is not a final answer.
        self.assertEqual(extract_final('maybe \\boxed{5}\nthen more text')[0], None)

    def test_truncated_and_markerless_reasoning_are_not_rescued(self):
        self.assertEqual(extract_final('Final answer: A', 'length'), (None, 'truncated'))
        self.assertEqual(extract_final('B looks plausible.\nBut A is correct.')[0], None)

    def test_clean_does_not_unwrap_partial_boxed(self):
        self.assertEqual(clean('\\boxed{8,7} but sum 41'), '\\boxed{8,7} but sum 41')


class PositionTest(unittest.TestCase):
    def test_bare_pairs_with_row_column_question(self):
        for pred in ('(10,5)', '10-5', '10 5', '[10, 5]', 'Row 10, Column 5', 'tenth row fifth column'):
            self.assertEqual(match_blank('10,5', pred, ROWCOL_Q), 'position', pred)

    def test_position_near_misses(self):
        for pred in ('(5,10)', '(10,6)', '(10,5,1)', '10', 'Row 10', 'row 10, column 5, maybe 6', '(10.5)'):
            self.assertIsNone(match_blank('10,5', pred, ROWCOL_Q), pred)

    def test_verbal_reference(self):
        q = 'Find the matching shadow. (Answer which row and which column)'
        self.assertEqual(match_blank('The third row and first column', 'Row 3, Column 1', q), 'position')
        self.assertEqual(match_blank('Row 1, Column 2', '1,2', q), 'position')
        self.assertEqual(match_blank('Second row third column', 'column 3, row 2', q), 'position')
        self.assertIsNone(match_blank('Second row third column', '3,2', q))
        self.assertIsNone(match_blank('Second row third column', 'Row 2, Column 2', q))

    def test_bare_pair_needs_row_column_question(self):
        # Without a question that names rows before columns, a bare pair has no defined order.
        self.assertIsNone(match_blank('Row 1, Column 2', '1,2', 'Which shadow matches?'))
        self.assertEqual(match_blank('Row 1, Column 2', 'row 1 column 2', 'Which shadow matches?'), 'position')
        self.assertIsNone(parse_position('1,2', numeric=False))

    def test_underspecified_phrases_do_not_parse(self):
        self.assertIsNone(parse_position('Second row second', numeric=True))
        self.assertIsNone(parse_position('Second column, first one', numeric=True))
        self.assertIsNone(parse_position('row 2 and row 3, column 1', numeric=True))


class ListAndPairTest(unittest.TestCase):
    def test_int_list(self):
        self.assertEqual(parse_int_list('3 and 11'), (3, 11))
        self.assertEqual(match_blank('8, 3, 7, 4', '8,3,7,4', PLAIN_Q), 'canonical')
        self.assertEqual(match_blank('1, 3', '1 3', PLAIN_Q), 'int_list')
        # Order, count and digits all carry content.
        for pred in ('3,8,7,4', '8,3,7', '8,3,7,4,0', '83,7,4', '8-3-7-4'):
            self.assertIsNone(match_blank('8, 3, 7, 4', pred, PLAIN_Q), pred)

    def test_pair_mapping_order_free(self):
        self.assertEqual(match_blank('1-C, 2-D, 3-A, 4-B.', '1-C 2-D 3-A 4-B', PLAIN_Q), 'pair_set')
        self.assertEqual(match_blank('2-8, 1-3, 5-6, 4-7', '1-3,2-8,4-7,5-6', PLAIN_Q), 'pair_set')
        self.assertEqual(match_blank('A-6, B-3', '6-A; 3-B', PLAIN_Q), 'pair_set')
        self.assertEqual(match_blank('A - Blue, B - Yellow', 'b-yellow, a-blue', PLAIN_Q), 'pair_set')

    def test_pair_mapping_near_misses(self):
        for pred in ('1-C, 2-D, 3-B, 4-A', '1-C, 2-D, 3-A', '1-C, 2-D, 3-A, 4-B, 5-E', 'C-1 2-D 3-A 4-B extra',
                     '1-C,2-D,3-A,4-B,4-B', '1C 2D 3A 4B'):
            self.assertIsNone(match_blank('1-C, 2-D, 3-A, 4-B.', pred, PLAIN_Q), pred)
        # Same-type elements keep their orientation: "8-2" is not "2-8".
        self.assertIsNone(match_blank('2-8, 1-3', '8-2, 1-3', PLAIN_Q))
        self.assertIsNone(parse_pairs('8-5=3'))

    def test_item_list(self):
        self.assertEqual(match_blank('G,4,D,6', 'G 4 D 6', PLAIN_Q), 'item_list')
        self.assertIsNone(match_blank('G,4,D,6', 'G,4,6,D', PLAIN_Q))
        self.assertIsNone(match_blank('BG,CE,DF', 'BG CE FD', PLAIN_Q))

    def test_unicode_dashes(self):
        self.assertEqual(match_blank('8\u22125=3', '8-5=3', PLAIN_Q), 'canonical')
        self.assertIsNone(match_blank('8\u22125=3', '8-3=5', PLAIN_Q))
        self.assertIsNone(match_blank('7\u22121=6', '7-6=1', PLAIN_Q))

    def test_numbers_and_symbols(self):
        self.assertIsNone(match_blank('5', '15', PLAIN_Q))
        self.assertIsNone(match_blank('<', '>', PLAIN_Q))
        self.assertIsNone(match_blank('9', '9 cubes', PLAIN_Q))


class DiagnosticScoreTest(unittest.TestCase):
    def test_contains_strict_and_reports_rule(self):
        row = blank('10,5')
        self.assertFalse(score(row, 'Final answer: (10,5)', 'babyvision')['correct'])
        result = diagnostic_score(row, 'Final answer: (10,5)')
        self.assertEqual((result['correct'], result['rule'], result['strict_correct']), (True, 'position', False))
        self.assertEqual(diagnostic_score(row, 'Final answer: 10,5')['rule'], 'strict')

    def test_choice(self):
        row = choice(1, n=3)
        self.assertTrue(diagnostic_score(row, 'I guess.\nSo Final answer: B\n')['correct'])
        self.assertTrue(diagnostic_score(row, 'Final answer: (B)')['correct'])
        self.assertTrue(diagnostic_score(row, 'Final answer: option B')['correct'])
        for response in ('So Final answer: C', 'Final answer: D', 'Final answer: B or C', 'B is wrong, A.'):
            self.assertFalse(diagnostic_score(row, response)['correct'], response)

    def test_truncated_strictly_wrong_stays_wrong(self):
        self.assertFalse(diagnostic_score(choice(1), 'So Final answer: B and then loops', 'length')['correct'])

    def test_group_rejects_lost_strict_item(self):
        rows = [{'strict': {'correct': True}, 'diagnostic': {'correct': False}}]
        with self.assertRaises(AssertionError):
            group(rows)


if __name__ == '__main__':
    unittest.main()
