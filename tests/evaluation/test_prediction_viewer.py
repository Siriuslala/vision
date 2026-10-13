"""Check reference joining, safe rendering, live reload, and HTTP image access."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler

urlopen = build_opener(ProxyHandler({})).open
from http.server import ThreadingHTTPServer
from PIL import Image

from eva.viewer.server import Results, handler_for


class PredictionViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new('RGB', (16, 16), 'red').save(self.root / 'test.png')
        self.path = self.root / 'predictions.jsonl'
        self.manifest = self.root / 'train.jsonl'
        self.manifest.write_text(json.dumps({'taskId':445, 'image':'test.png', 'question':'Choose.',
                                            'ansType':'choice', 'choiceAns':1, 'options':['one','two'],
                                            'coT':'Reference\nreasoning.'})+'\n')
        self.row = {'taskId':445, 'image':str(self.root/'test.png'), 'prompt':'Exact prompt.',
                    'response':'**Answer**\nSecond line.\n\n<script>alert(1)</script>\n\n```python\na = 1\nb = 2\n```',
                    'score':{'reference':'B', 'correct':True}, 'finish_reason':'eos'}
        self.save([self.row])

    def save(self, rows):
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def test_http_rendering_images_and_reload(self):
        results = Results([self.path], self.manifest)
        server = ThreadingHTTPServer(('127.0.0.1',0), handler_for(results))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        with urlopen(base+'/api/sample?run=0&index=0') as response:
            data = json.load(response)
        self.assertEqual(data['reference_reasoning'], 'Reference\nreasoning.')
        self.assertEqual(data['reference'], 'B')
        self.assertIn('B. two', data['question'])
        self.assertIn('<strong>Answer</strong><br', data['html']['response'])
        self.assertNotIn('<script>', data['html']['response'])
        self.assertIn('a = 1\nb = 2', data['html']['response'])
        self.assertEqual(data['response'], self.row['response'])
        with urlopen(base+'/api/image?run=0&index=0') as response:
            self.assertEqual(response.headers['Content-Type'], 'image/png')
            self.assertEqual(response.read(), (self.root/'test.png').read_bytes())
        self.save([self.row, {**self.row, 'taskId':446}])
        with urlopen(base+'/api/runs') as response:
            self.assertEqual(len(json.load(response)[0]['samples']), 2)
        for route in ['/api/image?run=0&index=-1', '/api/sample?run=99&index=0', '/../../.env']:
            with self.assertRaises(HTTPError):
                urlopen(base+route)
        with self.path.open('a') as stream:
            stream.write('{"unfinished":')
        with self.assertRaises(HTTPError):
            urlopen(base+'/api/runs')

    def test_direct_inputs_and_empty_file(self):
        self.save([{'sample_id':0, 'image_path':'test.png', 'prompt':'Question\nline two',
                    'response':'Answer', 'reasoning_content':'Model reasoning'}])
        row = Results([self.path]).record(0,0)
        self.assertEqual(row['question'], 'Question\nline two')
        self.assertEqual(row['reference'], '')
        self.assertEqual(row['reasoning'], 'Model reasoning')
        self.save([])
        self.assertEqual(Results([self.path]).catalog()[0]['samples'], [])

    def test_embedded_references_take_priority_over_manifest(self):
        self.save([{**self.row, 'question':'Embedded question', 'options':['red','blue'],
                    'answer':'A', 'coT':'Embedded reference reasoning',
                    'reasoning_content':'Separate model reasoning'}])
        for manifest in (None, self.manifest):
            row = Results([self.path], manifest).record(0,0)
            self.assertEqual(row['question'], 'Embedded question\n\nA. red\nB. blue')
            self.assertEqual(row['reference'], 'A')
            self.assertEqual(row['reference_reasoning'], 'Embedded reference reasoning')
            self.assertEqual(row['reasoning'], 'Separate model reasoning')
        self.save([{**self.row, 'reference_answer':0, 'reference_reasoning':'Proof'}])
        row = Results([self.path]).record(0,0)
        self.assertEqual(row['reference'], '0')
        self.assertEqual(row['reference_reasoning'], 'Proof')

    def test_rejects_duplicate_ids_and_missing_images(self):
        self.save([self.row, self.row])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            Results([self.path])
        self.save([{**self.row, 'image':'missing.png'}])
        with self.assertRaisesRegex(ValueError,'missing image'):
            Results([self.path])

    def test_absolute_and_relative_paths_resolve_per_result_file(self):
        nested = self.root / 'nested'
        nested.mkdir()
        path = nested / 'results.jsonl'
        path.write_text(json.dumps({**self.row, 'image':'../test.png'})+'\n')
        self.assertEqual(Results([path]).record(0,0)['image_path'], self.root / 'test.png')
        # An absolute image_path takes precedence over a stale legacy image field.
        path.write_text(json.dumps({**self.row, 'image':'missing.png',
                                   'image_path':str(self.root/'test.png')})+'\n')
        self.assertEqual(Results([path]).record(0,0)['image_path'], self.root / 'test.png')
