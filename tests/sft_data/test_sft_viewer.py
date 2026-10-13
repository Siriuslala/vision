import json
import tempfile
import unittest
from pathlib import Path
from data.sft.viewer.server import Dataset

class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.base={'schema_version':'1.1','coordinate_system':'image_long_side_normalized_xy','sample_id':'sample_i','reasoning_mode':'interleaved','query':'Question','answer':{'type':'text','value':'A'},'image':{'width':1000,'height':500,'path':'image.png','sha256':'a'*64},'trajectory':[],'provenance':{'source_name':'source','source_sample_id':'1','source_revision':'revision'}}
    def dataset(self,rows):
        path=self.root/'test.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return Dataset({'name':'demo','jsonl_path':str(path),'data_root':str(self.root)})
    def test_modes_pair_and_search_alias(self):
        other={**self.base,'sample_id':'sample_v','reasoning_mode':'visual_only'}
        ds=self.dataset([self.base,other]);self.assertEqual(len(ds.groups),1)
        self.assertEqual(ds.ids['sample_i'],ds.ids['sample_v']);self.assertEqual(set(ds.record(0)),{'interleaved','visual_only'})
    def test_distinct_questions_not_paired(self):
        other={**self.base,'sample_id':'another','query':'Different','reasoning_mode':'visual_only'}
        self.assertEqual(len(self.dataset([self.base,other]).groups),2)
    def test_missing_mode_and_modified_input(self):
        ds=self.dataset([self.base]);self.assertNotIn('visual_only',ds.record(0))
        with (self.root/'test.jsonl').open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'changed'):ds.record(0)
    def test_duplicate_id_and_unsupported_coordinates(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):self.dataset([self.base,self.base])
        with self.assertRaisesRegex(ValueError,'schema'):self.dataset([{**self.base,'coordinate_system':'image_normalized_xy'}])
    def test_image_path_cannot_escape_root(self):
        ds=self.dataset([{**self.base,'image':{**self.base['image'],'path':'../secret.png'}}])
        with self.assertRaisesRegex(ValueError,'escapes'):ds.image_path(0)
