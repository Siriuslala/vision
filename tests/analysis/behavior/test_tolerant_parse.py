"""Tolerant parse of location answers (experiments 1.3 / 1.4); no GPU required."""
import unittest

from analysis_tools.behavior import tolerant_parse as tp
from analysis_tools.behavior.decomposition import score_case

TRUTH = {'rows': 6, 'cols': 6, 'target_rc': [3, 3], 'grid_origin': [0, 0], 'cell': 64, 'canvas': [384, 384],
         'target_center': [224.0, 224.0], 'target_tight_box': [208, 200, 240, 248]}


def case(fmt: str) -> dict:
    return {'id': f'x|{fmt}', 'format': fmt, 'stage': 'main', 'truth': TRUTH}


def record(text: str, finish: str = 'eos') -> dict:
    return {'id': 'x', 'response': text, 'finish_reason': finish}


def tolerant(fmt: str, text: str, finish: str = 'eos') -> dict:
    c, r = case(fmt), record(text, finish)
    scorer = lambda cc, rr: score_case(cc, rr, True)
    return tp.score_tolerant(c, r, {'id': 'x', **scorer(c, r)}, scorer)


class ExtractTests(unittest.TestCase):
    def test_bracket_mismatch_variants_are_accepted(self):
        for text in ('```json\n[\n\t{"row": 4, "column": 4]\n```', '[{"row": 4, "column": 4\n]',
                     '[{"row": 4, "column": 4]\n]', '[{"row": 4, "column": 4"]]'):
            self.assertEqual(tp.extract_tolerant(text, 'row_column'), ({'row': 4, 'column': 4}, 'ok'), text)

    def test_reverse_key_order(self):
        self.assertEqual(tp.extract_tolerant('{"column": 2, "row": 5]', 'row_column')[0], {'row': 5, 'column': 2})

    def test_multiple_pairs_are_rejected(self):
        text = '[{"row": 1, "column": 2}, {"row": 3, "column": 4]'
        self.assertEqual(tp.extract_tolerant(text, 'row_column'), (None, 'ambiguous'))

    def test_unpaired_or_non_integer_keys_are_rejected(self):
        self.assertEqual(tp.extract_tolerant('{"row": 4, "label": "x", "column": 4]', 'row_column')[1], 'ambiguous')
        self.assertEqual(tp.extract_tolerant('{"row": 4.5, "column": 4]', 'row_column')[1], 'ambiguous')
        self.assertEqual(tp.extract_tolerant('{"row": 4, "column": 4, "row": 5]', 'row_column')[1], 'ambiguous')
        self.assertEqual(tp.extract_tolerant('{"row": "4", "column": "4"]', 'row_column')[1], 'no_candidate')

    def test_prose_without_candidate(self):
        self.assertEqual(tp.extract_tolerant('There is no different figure.', 'row_column'), (None, 'no_candidate'))

    def test_point_with_placeholder_echo_is_accepted(self):
        text = '[{"point_2d": [498, 125], "label": "Return only [{"point_2d": [x, y]}]."}]'
        self.assertEqual(tp.extract_tolerant(text, 'point'), ([{'point_2d': [498, 125]}], 'ok'))

    def test_two_points_or_wrong_length_are_rejected(self):
        self.assertEqual(tp.extract_tolerant('[{"point_2d": [1, 2]}, {"point_2d": [3, 4]]', 'point')[1], 'ambiguous')
        self.assertEqual(tp.extract_tolerant('[{"point_2d": [1, 2, 3]]', 'point')[1], 'ambiguous')
        self.assertEqual(tp.extract_tolerant('[{"bbox_2d": [1, 2, 3, 4]}`]', 'bbox')[0], [{'bbox_2d': [1, 2, 3, 4]}])


class ScoreTests(unittest.TestCase):
    def test_rescued_answer_is_scored_like_the_diagnostic(self):
        s = tolerant('row_column', '```json\n[\n\t{"row": 4, "column": 4]\n```')
        self.assertEqual((s['parse'], s['valid'], s['correct'], s['invalid_type']), ('tolerant', True, True, 'bracket_mismatch'))
        self.assertEqual(s['id'], 'x')

    def test_rescued_value_still_range_checked(self):
        s = tolerant('row_column', '[{"row": 4, "column": 9]')
        self.assertEqual((s['valid'], s['failure']), (False, 'out_of_range'))

    def test_truncated_answer_is_not_rescued(self):
        s = tolerant('row_column', '[{"row": 4, "column": 4]', finish='length')
        self.assertEqual((s['valid'], s['parse'], s['invalid_type']), (False, 'rejected', 'truncated'))

    def test_valid_and_schema_answers_keep_the_diagnostic_score(self):
        s = tolerant('row_column', '{"row": 4, "column": 4}')
        self.assertEqual((s['parse'], s['correct']), ('json', True))
        self.assertNotIn('invalid_type', s)
        s = tolerant('row_column', '[4, 4]')
        self.assertEqual((s['valid'], s['invalid_type']), (False, 'different_schema'))

    def test_classification(self):
        self.assertTrue(tp.brackets_balanced('[{"a": "]"}]'))
        self.assertFalse(tp.brackets_balanced('[{"row": 1, "column": 2]'))
        self.assertEqual(tolerant('bbox', '[{"bbox_2d": [1, 2, 300, 400]}`]')['invalid_type'], 'extra_text')
        self.assertEqual(tolerant('point', '[{"point_2d": [1200, 30]}]')['invalid_type'], 'value_invalid')
        self.assertEqual(tolerant('row_column', 'No figure differs.')['invalid_type'], 'no_coordinate')


if __name__ == '__main__':
    unittest.main()
