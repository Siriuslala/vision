import json
from pathlib import Path
import tempfile
import unittest

from eva.benchmark import load_cases, parse_ids, prompt_for, score, summarize


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.row = {'taskId': 462, 'question': 'Choose the same pattern.', 'options': ['left', 'right'],
                    'ansType': 'choice', 'choiceAns': 1, 'blankAns': None, 'coT': 'SECRET_REASONING',
                    'answerImage': 'SECRET_IMAGE', 'type': 'test', 'subtype': 'same'}

    def test_ids(self):
        for values in (['[445, 462]'], ['445,462'], ['445', '462']):
            self.assertEqual(parse_ids(values), [445, 462])
        for values in (['[]'], ['445,445'], ['[1.2]'], ['[true]'], ['[-1]']):
            with self.assertRaises(ValueError):
                parse_ids(values)

    def test_prompt_excludes_labels(self):
        prompt = prompt_for(self.row, 'babyvision')
        self.assertNotIn('SECRET', prompt)
        self.assertIn('A. left\nB. right', prompt)
        self.assertEqual(prompt, prompt_for({**self.row, 'choiceAns': 0, 'coT': 'OTHER'}, 'babyvision'))

    def test_scoring(self):
        self.assertTrue(score(self.row, 'reasoning A\nFinal answer: B', 'babyvision')['correct'])
        self.assertFalse(score(self.row, 'B looks plausible, but A is correct.', 'babyvision')['correct'])
        self.assertIsNone(score(self.row, 'Final answer: C', 'babyvision')['prediction'])
        blank = {**self.row, 'ansType': 'blank', 'blankAns': '(4,7)'}
        self.assertTrue(score(blank, 'Final answer: (4, 7)', 'babyvision')['correct'])
        self.assertFalse(score(blank, 'Final answer: (7,4)', 'babyvision')['correct'])
        self.assertIsNone(score(self.row, 'anything', 'babyvision-gen')['correct'])

    def test_selection_order_and_missing_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'BabyVision'
            root.mkdir()
            (root / 'image.png').write_bytes(b'placeholder')
            rows = [{**self.row, 'taskId': i, 'image': 'image.png'} for i in (445, 462)]
            (root / 'train.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            selected, _ = load_cases(folder, 'babyvision', 'train', [462, 445])
            self.assertEqual([r['taskId'] for r in selected], [462, 445])
            with self.assertRaises(ValueError):
                load_cases(folder, 'babyvision', 'train', [0])

    def test_unparsed_counts_as_wrong(self):
        rows = [{'score': {'correct': True, 'prediction': 'B'}},
                {'score': {'correct': False, 'prediction': None}}]
        result = summarize(rows)
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['unparsed'], 1)


if __name__ == '__main__':
    unittest.main()
