import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from eva.inputs import load_image_prompts
from eva.inference import main, parser


class ImageInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'image with spaces.png'
        Image.new('RGB', (16, 16), 'red').save(self.image)
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        self.output = self.root / 'results.jsonl'
        self.path_patch = patch('eva.inference.load_paths', return_value={
            'DATA_DIR': str(self.root), 'HF_HUB_CACHE': str(self.root)})
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def run_dry(self, prompts, extra=(), full=False):
        argv = ['test', '--model', str(self.model), '--output', str(self.output),
                '--dry-run', '--batchsize', '2', '--gpu_ids', 'cpu']
        if not full:
            argv += ['--image_path', *([str(self.image)] * len(prompts)), '--prompt', *prompts]
        with patch('sys.argv', argv + list(extra)), contextlib.redirect_stdout(io.StringIO()):
            main(full=full)

    def test_pairing_preserves_text_and_duplicate_images(self):
        prompts = ['  Describe the color.\nKeep this line.  ', 'Count the objects.']
        rows = load_image_prompts([str(self.image)] * 2, prompts)
        self.assertEqual([row['prompt'] for row in rows], prompts)
        self.assertEqual([row['sample_id'] for row in rows], [0, 1])
        self.assertEqual(rows[0]['image_sha256'], rows[1]['image_sha256'])

    def test_invalid_inputs(self):
        for paths, prompts in (([], []), ([str(self.image)], []), ([str(self.image)], ['  '])):
            with self.assertRaises(ValueError):
                load_image_prompts(paths, prompts)
        with self.assertRaises(FileNotFoundError):
            load_image_prompts([str(self.root / 'missing.png')], ['Describe.'])
        broken = self.root / 'broken.png'
        broken.write_text('invalid image bytes')
        with self.assertRaisesRegex(ValueError, 'readable image'):
            load_image_prompts([str(broken)], ['Describe.'])

    def test_batch_order_and_resume_provenance(self):
        prompts = ['First question.\nSecond line.', 'Another question.', 'Third question.']
        self.run_dry(prompts)
        records = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([r['sample_id'] for r in records], [0, 1, 2])
        self.assertEqual([r['prompt'] for r in records], prompts)
        self.assertEqual([r['batch_size'] for r in records], [2, 2, 1])
        self.assertTrue(all('score' not in r and 'benchmark' not in r for r in records))
        original = self.output.read_bytes()
        self.run_dry(prompts, ['--resume'])
        self.assertEqual(self.output.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            self.run_dry(['Changed prompt.', *prompts[1:]], ['--resume'])
        Image.new('RGB', (16, 16), 'blue').save(self.image)
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            self.run_dry(prompts, ['--resume'])

    def test_benchmark_evaluation_still_scores(self):
        dataset = self.root / 'BabyVision'
        dataset.mkdir()
        Image.new('RGB', (16, 16)).save(dataset / 'input.png')
        row = {'taskId': 445, 'image': 'input.png', 'question': 'Choose.',
               'type': 'test', 'subtype': 'same', 'ansType': 'choice',
               'options': ['left', 'right'], 'choiceAns': 1}
        (dataset / 'train.jsonl').write_text(json.dumps(row) + '\n')
        self.run_dry([], full=True)
        record = json.loads(self.output.read_text())
        self.assertEqual(record['taskId'], 445)
        self.assertEqual(record['benchmark'], 'babyvision')
        self.assertIn('A. left', record['prompt'])
        self.assertEqual(record['score']['metric'], 'dry_run')

    def test_cli_interfaces(self):
        direct_help = parser().format_help()
        self.assertIn('--image_path', direct_help)
        self.assertNotIn('--bench', direct_help)
        evaluation_help = parser(full=True).format_help()
        self.assertIn('--case-ids', evaluation_help)
        self.assertNotIn('--image_path', evaluation_help)
