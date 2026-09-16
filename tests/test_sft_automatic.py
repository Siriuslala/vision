import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from data.sft.core.common import validate_record
from data.sft.tools.automatic import parse_generation, process_task, process_with_retry, verify_verdict, validate_crop_evidence, rewrite_text_events
from data.sft.tools.window_catalog import candidate_windows
from data.sft.tools.local_teacher import json_object


class FakeTeacher:
    info = {'model': 'fixture'}

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, images, prompt, max_new_tokens):
        self.prompts.append((images, prompt))
        return {'finish_reason': 'eos', 'response': json.dumps(next(self.responses))}


class AutomaticTests(unittest.TestCase):
    def setUp(self):
        raw = io.BytesIO()
        Image.new('RGB', (100, 50)).save(raw, format='PNG')
        self.base = {'source_id': 'fixture', 'image': raw.getvalue(), 'query': 'Which color?',
                     'answer': {'type': 'text', 'value': 'red'}, 'task_type': 'general_vqa'}
        self.events = [{'type': 'visual_read', 'box': [100, 200, 300, 600]},
                       {'type': 'reasoning_text', 'text': 'The relevant object is red.'}]
        self.verdict = {k: True for k in ('answer_correct', 'observations_supported',
                        'reasoning_complete', 'reads_useful', 'visual_only_safe')}
        self.verdict.update(self_correction=False, reason='The complete target is visible in the crop.')

    def test_independent_answer_and_actual_crop_then_modes(self):
        teacher = FakeTeacher([{'answerable': True, 'answer': 'red', 'events': self.events}, self.verdict])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = list(process_task(self.base, 'llava', teacher, root/'run', root, 1000, io.StringIO(), crop_check=False))
            self.assertEqual(len(records), 2)
            for record in records:
                validate_record(record, root)
                self.assertEqual(record['metadata']['curriculum']['read_count'], 1)
                self.assertEqual(record['quality']['status'], 'pending')
            self.assertNotRegex(teacher.prompts[0][1], r'\bred\b')
            self.assertEqual(teacher.prompts[1][0][1].size, (20, 20))
            self.assertEqual(records[0]['trajectory'][0]['region']['center'], [.2, .2])
            self.assertTrue(all(e['type'] != 'reasoning_text' for e in records[1]['trajectory']))

    def test_answer_disagreement_cannot_reuse_source_correction(self):
        base = {**self.base, 'source_events': [{'type': 'visual_read', 'box': [0, 0, 20, 20]}]}
        teacher = FakeTeacher([{'answerable': True, 'answer': 'blue', 'events': self.events}, {'equivalent':False,'reason':'Different colors.'}])
        with tempfile.TemporaryDirectory() as directory:
            records = list(process_task(base, 'thyme', teacher, Path(directory)/'run', Path(directory), 1000, io.StringIO(), crop_check=False))
        self.assertEqual(len(records), 2)
        self.assertTrue(all('reject_reason' in r for r in records))
        self.assertEqual(len(teacher.prompts), 2)

    def test_direct_answer_is_explicit_and_valid(self):
        teacher = FakeTeacher([{'answerable': True, 'answer': 'red', 'events': []}, self.verdict])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = list(process_task(self.base, 'llava', teacher, root/'run', root, 1000, io.StringIO(), crop_check=False))
            for record in records:
                validate_record(record, root)
                self.assertEqual(record['metadata']['curriculum']['read_group'], 'zero_read')
                self.assertEqual([e['type'] for e in record['trajectory']], ['finish'])

    def test_bad_geometry_and_false_or_incomplete_critic_rejected(self):
        for box in ([0, 0, 1001, 1000], [0, 0, float('nan'), 2], [20, 20, 10, 10]):
            with self.assertRaises(ValueError):
                parse_generation({'answerable': True, 'answer': 'red', 'events': [{'type': 'visual_read', 'box': box}]}, 100, 50)
        for field in ('answer_correct', 'observations_supported', 'reasoning_complete', 'reads_useful'):
            verdict = {**self.verdict, field: False}
            with self.assertRaises(ValueError):
                verify_verdict(verdict, self.base['answer'], 'red', self.events, 'normal')
        with self.assertRaises(ValueError):
            verify_verdict({**self.verdict, 'self_correction': True}, self.base['answer'], 'red', self.events, 'normal')
        with self.assertRaises(ValueError):
            verify_verdict(self.verdict, self.base['answer'], 'red', self.events, 'self_correction')

    def test_normal_and_correction_have_separate_viewer_identity(self):
        base = {**self.base, 'source_events': [{'type': 'visual_read', 'box': [0, 0, 20, 20]},
                    {'type': 'reasoning_text', 'text': 'I initially chose the wrong object; this is red.'}]}
        teacher = FakeTeacher([{'answerable': True, 'answer': 'red', 'events': self.events},
                               self.verdict, {'texts':[{'event_index':1,'text':'I initially chose the wrong object; this is red.'}]}, {**self.verdict, 'self_correction': True, 'error_type':'wrong_region'}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = list(process_task(base, 'thyme', teacher, root/'run', root, 1000, io.StringIO(), crop_check=False))
            self.assertEqual(len(records), 3)
            self.assertNotEqual(records[0]['provenance']['source_sample_id'], records[2]['provenance']['source_sample_id'])
            self.assertEqual(records[2]['reasoning_mode'], 'interleaved')
            self.assertEqual(records[0]['provenance']['group_id'], records[2]['provenance']['group_id'])

    def test_crop_only_rejection_precedes_full_image_critic(self):
        teacher = FakeTeacher([{'answerable': True, 'answer': 'red', 'events': self.events},
            {'answerable': False, 'answer': '', 'observations': [{'window': 1, 'usable': False, 'evidence': 'Only background.'}]}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = list(process_task(self.base, 'llava', teacher, root/'run', root, 1000, io.StringIO()))
        self.assertEqual(len(results), 1)
        self.assertIn('Crop-only evidence rejected', results[0]['reject_reason'])
        self.assertEqual(len(teacher.prompts), 2)
        self.assertEqual(len(teacher.prompts[1][0]), 1)
        self.assertEqual(teacher.prompts[1][0][0].size, (20,20))
        self.assertNotRegex(teacher.prompts[1][1], r'\bred\b')
        with self.assertRaises(ValueError):
            validate_crop_evidence({'answerable':True,'answer':'red','observations':[]},1,'red')

    def test_catalog_selection_preserves_nonsquare_pixel_geometry(self):
        windows, _ = candidate_windows(Image.new('RGB',(100,50)))
        events = parse_generation({'answerable':True,'answer':'red','events':[
            {'type':'visual_read','window_id':1}, {'type':'reasoning_text','text':'Visible evidence.'}]},100,50,windows=windows)
        self.assertEqual(events[0]['box'],[0,0,50,25])
        with self.assertRaises(ValueError):
            parse_generation({'answerable':True,'answer':'red','events':[{'type':'visual_read','window_id':999}]},100,50,windows=windows)

    def test_retry_uses_failed_visual_evidence_without_leaking_source_answer(self):
        teacher = FakeTeacher([
            {'answerable':True,'answer':'blue','events':[self.events[0],{'type':'reasoning_text','text':'The target is blue.'}]},
            {'equivalent':False,'reason':'The answers name different colors.'},
            {'answerable':True,'answer':'red','events':self.events},
            {'answerable':True,'answer':'red','observations':[{'window':1,'usable':True,'adds_new_evidence':True,'evidence':'A red target.'}]},
            self.verdict])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = list(process_with_retry(self.base,'llava',teacher,root/'run',root,1000,io.StringIO(),'coordinates',1))
        self.assertEqual(len(results),3)
        self.assertIn('reject_reason',results[0])
        self.assertTrue(results[1]['metadata']['generation']['retried'])
        self.assertNotRegex(teacher.prompts[2][1],r'\bred\b')

    def test_rewrite_cannot_change_observation_geometry_or_drop_reasoning(self):
        events = [{'type':'visual_read','box':[1,2,3,4]},{'type':'reasoning_text','text':'Source wording.'}]
        result = rewrite_text_events(events,{'texts':[{'event_index':1,'text':'Rewritten wording.'}]})
        self.assertEqual(result[0],events[0])
        self.assertEqual(events[1]['text'],'Source wording.')
        with self.assertRaises(ValueError):rewrite_text_events(events,{'texts':[]})
        with self.assertRaises(ValueError):rewrite_text_events(events,{'texts':[{'event_index':0,'text':'Wrong index.'}]})

    def test_model_wrappers_do_not_repair_malformed_json_content(self):
        for text in ('A preface\n```json\n{"answer":"yes"}\n```','{"answer":"yes"}`\n','{"answer":"yes"}}'):
            self.assertEqual(json_object(text),{'answer':'yes'})
        with self.assertRaises(ValueError):json_object('{"type":"reasoning_text":"invalid"}')

    def test_error_classification_must_agree_with_trajectory_type(self):
        with self.assertRaises(ValueError):
            verify_verdict({**self.verdict,'error_type':'wrong_region'},self.base['answer'],'red',self.events,'normal')
        source = {**self.verdict,'self_correction':True,'error_type':'wrong_region'}
        source.pop('visual_only_safe')
        verify_verdict(source,self.base['answer'],'red',self.events,'self_correction')
        with self.assertRaises(ValueError):
            verify_verdict({**source,'error_type':'none'},self.base['answer'],'red',self.events,'self_correction')

    def test_proposal_labels_cannot_become_incorrect_event_references(self):
        windows=[[0,0,20,20],[20,0,40,20],[40,0,60,20]]
        with self.assertRaises(ValueError):
            parse_generation({'answerable':True,'answer':'red','steps':[{'window_id':3,'finding':'Window 3 has the target.'}],
                              'conclusion':'The target is red.'},100,50,windows=windows)
        result=parse_generation({'answerable':True,'answer':'red','steps':[
            {'window_id':1,'finding':'First property.'},{'window_id':1,'finding':'Another property in the same view.'},
            {'window_id':2,'finding':'A different object.'},{'window_id':1,'finding':'Revisit the first object.'}],
            'conclusion':'The evidence supports the answer.'},100,50,windows=windows)
        self.assertEqual([e['box'] for e in result if e['type']=='visual_read'],[windows[0],windows[1],windows[0]])
