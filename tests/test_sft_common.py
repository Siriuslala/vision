import copy
import io
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from data.sft.core.common import build_records, pixel_box, validate_record
from data.sft.tools.teacher import parse_proposal


class CommonTests(unittest.TestCase):
    def test_non_square_round_trip_and_modes(self):
        raw=io.BytesIO();Image.new('RGB',(1000,500)).save(raw,format='PNG')
        candidate={'source_id':'a','task_type':'grounding','query':'Where?','answer':{'type':'text','value':'red'},'image':raw.getvalue(),'visual_only_safe':True,'events':[{'type':'visual_read','box':[500,150,700,250]},{'type':'reasoning_text','text':'The target is red.'}]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            records=build_records(candidate,'cosyn',root/'images',root)
            self.assertEqual(records[0]['initial_region']['size'],[1,.5])
            self.assertEqual(records[0]['trajectory'][0]['region'],{'center':[.6,.2],'size':[.2,.1]})
            self.assertEqual(len(records[1]['trajectory']),2)
            self.assertEqual(records[0]['provenance']['group_id'],records[1]['provenance']['group_id'])
            for record in records:validate_record(record,root)
            bad=copy.deepcopy(records[0]);bad['trajectory'][0]['reference_region_id']=1
            with self.assertRaises(ValueError):validate_record(bad)
    def test_geometry_never_clamps_invalid_labels(self):
        with self.assertRaises(ValueError):pixel_box([0,0,10,501],1000,500)
        with self.assertRaises(ValueError):pixel_box([0,0,float('nan'),5],10,10)
    def test_teacher_coordinate_conversion(self):
        events=parse_proposal('{"events":[{"type":"visual_read","box":[100,200,400,600]},{"type":"reasoning_text","text":"Visible evidence."}]}',1000,500)
        self.assertEqual(events[0]['box'],[100,100,400,300])
        with self.assertRaises(ValueError):parse_proposal('{"events":[{"type":"visual_read","box":[0,0,1001,1000]}]}',1000,500)

    def test_nested_schema_and_persistent_split_conflicts(self):
        from data.sft.core.splits import register
        raw=io.BytesIO();Image.new('RGB',(20,10)).save(raw,format='PNG')
        c={'source_id':'x','task_type':'grounding','query':'Q','answer':{'type':'text','value':'a'},'image':raw.getvalue(),'events':[{'type':'visual_read','box':[0,0,10,10]}]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);r=build_records(c,'cosyn',root/'images',root)[0]
            for field,value in [('provenance',{}),('verification',{'garbage':True})]:
                bad=copy.deepcopy(r);bad[field]=value
                with self.assertRaises(ValueError):validate_record(bad)
            bad=copy.deepcopy(r);bad['metadata']['counting']={'groups':[{'read_event_id':999}]}
            with self.assertRaises(ValueError):validate_record(bad)
            register([r],root/'splits.sqlite')
            bad=copy.deepcopy(r);bad['provenance']['split']='test' if r['provenance']['split']!='test' else 'train'
            with self.assertRaises(ValueError):register([bad],root/'splits.sqlite')

    def test_counting_mode_metadata_remap(self):
        from data.sft.datasets.pixmo import counting_candidate
        import hashlib
        raw=io.BytesIO();Image.new('RGB',(100,50)).save(raw,format='PNG')
        sha=hashlib.sha256(raw.getvalue()).hexdigest()
        row={'image_sha256':sha,'points':{'x':[10,25],'y':[10,10]},'count':2,'label':'circles','image_url':'fixture'}
        audit={'source_id':'fixture','image_sha256':sha,'instance_boxes_reviewed':True,'complete_instance_set_reviewed':True,'method':'geometry','config_id':'fixture-v1','instances':[{'instance_id':'point_0','box':[5,5,15,15]},{'instance_id':'point_1','box':[20,5,30,15]}]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'source.png';path.write_bytes(raw.getvalue())
            c=counting_candidate(row,'fixture','fixture',path,audit,'audit/fixture.json')
            records=build_records(c,'pixmo',root/'images',root)
            self.assertIsNone(records[1]['metadata']['counting']['sum_event_id'])
            self.assertIsNone(records[1]['metadata']['counting']['groups'][0]['count_event_id'])
            bad=copy.deepcopy(records[0]);bad['metadata']['counting']['groups'][0]['cumulative_count']=3
            with self.assertRaises(ValueError):validate_record(bad)
