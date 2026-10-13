"""Ground-truth and scoring checks for comparison and geometry experiments."""
import unittest
import numpy as np
from analysis_tools.comparison_grounding import majority_image,figure_boxes,score_answer,box_iou


class ComparisonGroundingTests(unittest.TestCase):
    def test_reversing_majority_reverses_only_the_gap_property(self):
        for target in (8,21,28):
            for minority in ('gap','intact'):
                image=np.array(majority_image(target,minority))
                gaps=[]
                for i in range(36):
                    x,y=i%6*64,i//6*64
                    gaps.append(bool(np.all(image[y+32:y+34,x+39:x+47]==255)))
                self.assertEqual(sum(gaps),1 if minority=='gap' else 35)
                self.assertEqual(gaps[target],minority=='gap')

    def test_boxes_remain_identical_after_reversing_majority(self):
        self.assertEqual(figure_boxes(majority_image(8,'gap')),figure_boxes(majority_image(8,'intact')))
        self.assertEqual(figure_boxes(majority_image(8,'gap'))[0],[16.,8.,49.,59.])

    def test_bbox_geometry_and_rejection(self):
        case=dict(format='bbox',size=[1000,1000],bbox=[100,100,200,200],source='refcoco')
        valid=score_answer(case,dict(response='[{"bbox_2d":[100,100,200,200]}]',finish_reason='eos'))
        invalid=score_answer(case,dict(response='[{"bbox_2d":[200,100,100,200]}]',finish_reason='eos'))
        truncated=score_answer(case,dict(response='[{"bbox_2d":[100,100,200,200]}]',finish_reason='length'))
        self.assertTrue(valid['correct']);self.assertFalse(invalid['valid']);self.assertFalse(truncated['valid'])
        self.assertEqual(box_iou([0,0,10,10],[10,10,20,20]),0)

    def test_point_precision_differs_from_point_inside_bbox(self):
        case=dict(format='point',size=[1000,1000],bbox=[100,100,200,200],source='refcoco')
        r=score_answer(case,dict(response='[{"point_2d":[110,110]}]',finish_reason='eos'))
        self.assertTrue(r['point_in_box']);self.assertFalse(r['correct'])

    def test_bare_coordinate_diagnostic_preserves_task_type(self):
        case=dict(format='point',size=[1000,1000],bbox=[100,100,200,200],source='refcoco')
        raw=dict(response='[150,150]',finish_reason='eos')
        self.assertFalse(score_answer(case,raw)['valid'])
        self.assertTrue(score_answer(case,raw,allow_bare_coordinates=True)['correct'])
        wrong_type=dict(response='[100,100,200,200]',finish_reason='eos')
        self.assertFalse(score_answer(case,wrong_type,allow_bare_coordinates=True)['valid'])


if __name__=='__main__':
    unittest.main()
