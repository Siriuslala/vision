"""Regression checks for source semantics and evidence requirements."""
import unittest
from copy import deepcopy

from data.sft.core.annotations import apply_annotation, source_key
from data.sft.datasets.llava import convert_task, extract_answer


class LlavaTests(unittest.TestCase):
    def setUp(self):
        self.row = {'id':'one','data_source':'Geometry3K(MathV360K)',
                    'image':{'bytes':b'image'},'conversations':[
                        {'from':'human','value':'<image>\nQuestion: Find x.\nChoices:\n(A) 1\n(B) 2'},
                        {'from':'gpt','value':'The answer is B'}]}

    def test_choice_and_source_identity(self):
        item = convert_task(self.row,'Geometry3K(MathV360K)/train-00000.parquet','commit')
        self.assertEqual(item['answer'],{'type':'choice','value':'B'})
        self.assertEqual(item['source_id'],'Geometry3K(MathV360K)/one/turn0')
        self.assertNotIn('<image>',item['query'])
        self.assertNotIn('events',item)
        self.assertEqual(apply_annotation(item,{})['reject_reason'],'missing_region_annotation')

    def test_multiturn_not_silently_flattened(self):
        row=deepcopy(self.row)
        row['conversations'] += [{'from':'human','value':'Why?'},{'from':'gpt','value':'Because...'}]
        with self.assertRaisesRegex(ValueError,'context_conversion'):
            convert_task(row,'train-0.parquet','commit')

    def test_ambiguous_reasoning_not_used_as_final(self):
        with self.assertRaisesRegex(ValueError,'ambiguous_final_answer'):
            extract_answer('Let us inspect the triangle. Because two sides agree, the answer is B.','Find x')
        self.assertEqual(extract_answer('<think>Examine the shape.</think><answer>7</answer>','How many?')[0]['value'],'7')

    def test_external_split_and_missing_image(self):
        with self.assertRaisesRegex(ValueError,'non_training'):
            convert_task(self.row,'test-0.parquet','commit')
        self.row['image']=None
        with self.assertRaisesRegex(ValueError,'missing_embedded'):
            convert_task(self.row,'train-0.parquet','commit')

    def test_counting_task_boundaries(self):
        self.row['data_source']='CLEVR-Math(MathV360K)'
        self.row['conversations'][0]['value']='<image>\nAdd 4 brown spheres. How many brown spheres exist?'
        self.row['conversations'][1]['value']='The answer is 5'
        self.assertEqual(convert_task(self.row,'train-0.parquet','commit')['task_type'],'spatial_reasoning')
        self.row['data_source']='IconQA(MathV360K)'
        with self.assertRaisesRegex(ValueError,'counting_requires_reviewed'):
            convert_task(self.row,'train-0.parquet','commit')

    def test_direct_counting_cannot_bypass_instance_metadata_through_another_subset(self):
        self.row['conversations'][1]['value'] = '2'
        for subset in ('Super-CLEVR(MathV360K)', 'VisualWebInstruct(filtered)',
                       'CLEVR-Math(MathV360K)', 'IconQA(MathV360K)'):
            self.row['data_source'] = subset
            for query in ('How many wheels are visible?', 'Count the red objects.',
                          'What is the number of spheres in the image?'):
                self.row['conversations'][0]['value'] = '<image>\n' + query
                with self.subTest(subset=subset, query=query), self.assertRaisesRegex(
                        ValueError, 'counting_requires_reviewed'):
                    convert_task(self.row, 'train-0.parquet', 'commit')

    def test_quantities_and_geometric_measures_are_not_instance_counts(self):
        self.row['conversations'][1]['value'] = '20'
        for subset, query in (
                ('TabMWP(MathV360K)', 'How many tickets were sold in 2020?'),
                ('FigureQA(MathV360K)', 'How many units does the red bar represent?'),
                ('Geometry3K(MathV360K)', 'How many degrees is angle x?')):
            self.row['data_source'] = subset
            self.row['conversations'][0]['value'] = '<image>\n' + query
            with self.subTest(subset=subset):
                self.assertEqual(convert_task(self.row, 'train-0.parquet', 'commit')['answer']['value'], '20')

    def test_annotation_bound_to_answer_and_image(self):
        base=convert_task(self.row,'train-0.parquet','commit')
        key=source_key(base['source_id'],base['query'],base['answer'],base['image'])
        annotation={'source_key':key,'status':'reviewed','reviewer':'tester',
                    'events':[{'type':'visual_read','box':[0,0,10,10],'reference_region_id':0}]}
        self.assertIn('events',apply_annotation(base,{key:annotation}))
        base['image']=b'changed'
        self.assertEqual(apply_annotation(base,{key:annotation})['reject_reason'],'missing_region_annotation')


if __name__=='__main__':
    unittest.main()
