"""Independent-count and per-instance evidence must both pass."""
import unittest
from data.sft.tools.automatic_counting import check_independent_count, check_instances


class AutomaticCountingTests(unittest.TestCase):
    def test_independent_answer_cannot_be_overridden(self):
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            check_independent_count({'answerable':True,'count':2,'evidence':'two people'},3)
        with self.assertRaises(ValueError):
            check_independent_count({'answerable':True,'count':True,'evidence':'one person'},1)

    def test_global_approval_does_not_replace_instance_evidence(self):
        verdict={'all_instances_correct':True,'complete':True,'one_instance_per_box':True,'instances':[]}
        with self.assertRaisesRegex(ValueError,'every instance'):check_instances(verdict,1)
        verdict['instances']=[{'id':'point_0','label_correct':True,'one_instance':True,'visible_extent_complete':False,'evidence':'head only'}]
        with self.assertRaisesRegex(ValueError,'Individual'):check_instances(verdict,1)
        verdict['instances'][0]['visible_extent_complete']=True
        check_instances(verdict,1)

    def test_explicit_image_revision_retains_original_identity(self):
        import hashlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from PIL import Image
        from data.sft.tools.automatic_counting import process
        replies = [
            {'answerable':True,'count':1,'evidence':'one square at the left'},
            {'label_correct':True,'one_instance':True,'visible_extent_complete':True,'evidence':'one complete square'},
            {'all_instances_correct':True,'complete':True,'one_instance_per_box':True,
             'instances':[{'id':'point_0','label_correct':True,'one_instance':True,
                           'visible_extent_complete':True,'evidence':'the square is fully visible'}]},
            {'answer_correct':True,'observations_supported':True,'reasoning_complete':True,
             'reads_useful':True,'self_correction':False,'visual_only_safe':True,'reason':'one full square and correct count'}]
        class Teacher:
            info = {'model':'fixture'}
            def generate(self, images, prompt, tokens):
                return {'response':json.dumps(replies.pop(0)),'finish_reason':'eos'}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); run=root/'run';run.mkdir()
            image=root/'image.png';Image.new('RGB',(30,20),'white').save(image)
            actual=hashlib.sha256(image.read_bytes()).hexdigest()
            row={'image_sha256':'0'*64,'image_url':'fixture','label':'squares','count':1,
                 'points':{'x':[5.],'y':[5.]}}
            proposal={'source_id':'fixture:0','image_path':str(image),'image_sha256':actual,
                      'source_image_sha256':'0'*64,'points':[[5.,5.]],'count':1,'label':'squares',
                      'candidate_instances':[{'point_index':0,'rounds':[{'candidates':[
                          {'status':'pending','box':[2,2,8,8],'score':0.7}]}]}]}
            records=process(proposal,row,'original-revision',Teacher(),run,root,io.StringIO(),256)
            self.assertEqual(len(records),2)
            for record in records:
                self.assertEqual(record['provenance']['annotation_origin'],'mixed')
                self.assertIn(':reannotation:',record['provenance']['source_revision'])
                self.assertEqual(record['metadata']['image_reannotation']['original_sha256'],'0'*64)
                self.assertEqual(record['quality']['status'],'pending')
            self.assertEqual(row['image_sha256'],'0'*64)
