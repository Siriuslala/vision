"""Model-free checks of experiment 3.11 round 2: stimuli, distractor choice, flips, edge regression, scoring."""
import unittest

import numpy as np

from analysis_tools.position import coordinate_flow_natural as cfn
from analysis_tools.position import coordinate_flow_natural_analysis as cfa


class StimulusTests(unittest.TestCase):
    def test_multishape_unique_names_and_no_overlap(self):
        img, objs = cfn.multishape_image(np.random.default_rng(0), 160, n_obj=3)
        self.assertEqual(img.size, (512, 512))
        self.assertEqual(len({o['color'] for o in objs}), 3)
        self.assertEqual(len({o['shape'] for o in objs}), 3)
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                A, B = a['box'], b['box']
                self.assertTrue(A[2] <= B[0] or B[2] <= A[0] or A[3] <= B[1] or B[3] <= A[1])
            px = img.getpixel((int(a['center'][0]), int(a['center'][1])))
            self.assertEqual(px, cfn.COLORS[a['color']])

    def test_prompt_matches_refcoco_template(self):
        p = cfn.point_prompt('blue square')
        self.assertTrue(p.startswith('Locate the object described as: blue square. Give the center of its bounding box.'))

    def test_distractor_first_digit_guarantee(self):
        tgt = [0, 0, 50, 50]
        near = {'box': [40, 0, 90, 50]}       # 78 units apart: first digit not guaranteed
        far = {'box': [400, 400, 450, 450]}
        self.assertIs(cfn.pick_distractor(tgt, [near, far], (512, 512)), far)
        self.assertIsNone(cfn.pick_distractor(tgt, [near], (512, 512)))
        self.assertEqual(cfn.coco_image_id('COCO_train2014_000000384745_7.jpg'), '000000384745')

    def test_flipped_case(self):
        c = {'truth': {'canvas': [100, 200], 'box': [10, 20, 30, 60], 'center': [20, 40]}, 'others': [[0, 0, 5, 5]],
             'mask': [[10, 20], [30, 60]]}
        f = cfn.flipped_case(c)
        self.assertEqual(f['truth']['box'], [10, 140, 30, 180])
        self.assertEqual(f['truth']['center'], [20, 160])
        self.assertEqual(f['others'], [[0, 195, 5, 200]])
        self.assertEqual(f['mask'], [[10, 180], [30, 140]])


class ScoringTests(unittest.TestCase):
    def test_wrong_object(self):
        case = {'set': 'ms', 'truth': {'canvas': [512, 512], 'box': [0, 0, 100, 100]}, 'others': [[300, 300, 400, 400]]}
        sc = cfn.score_point(case, [{'key': 'x', 'value': 680}, {'key': 'y', 'value': 680}])
        self.assertFalse(sc['correct'])
        self.assertTrue(sc['wrong_object'])
        sc = cfn.score_point(case, [{'key': 'x', 'value': 100}, {'key': 'y', 'value': 100}])
        self.assertTrue(sc['correct'])
        self.assertFalse(cfn.score_point(case, None)['valid'])

    def test_parse_sets(self):
        s = cfn.parse_sets('H11=20.11;e=24.7,28.0')
        self.assertEqual(s, {'none': [], 'H11': [(20, 11)], 'e': [(24, 7), (28, 0)]})
        with self.assertRaises(ValueError):
            cfn.parse_sets('bad=21.3')

    def test_patch_conditions(self):
        conds = dict(cfn.patch_conditions())
        self.assertEqual(len([k for k in conds if k.startswith('h')]), 128)
        self.assertEqual(len(conds['all_mixers']), 128 + 24)
        self.assertEqual(sorted({l for l, _ in conds['dn_all']}), list(cfn.DN_LAYERS))


class EdgeTests(unittest.TestCase):
    def _items(self, where: str, n=40):
        rng = np.random.default_rng(1)
        items = []
        H = W = 16
        for _ in range(n):
            x0, y0 = rng.integers(0, 6, 2)
            w, h = rng.integers(4, 9, 2)
            box = [x0 * 32.0, y0 * 32.0, (x0 + w) * 32.0, (y0 + h) * 32.0]
            a = np.full((8, 16, 2, H * W), 1e-4)
            for k in range(2):
                if where == 'min':
                    r, c = y0, x0
                elif where == 'max':
                    r, c = y0 + h - 1, x0 + w - 1
                else:
                    r, c = y0 + h // 2, x0 + w // 2
                a[:, :, k, r * W + c] = 1.0
            items.append({'attn': a, 'grid': (H, W), 'box': box})
        return items

    def test_edge_regression_recovers_edges(self):
        b = np.array(cfa.edge_profile(self._items('min'))['beta'])
        self.assertGreater(b[5, 0, 0, 0], 0.85)
        self.assertLess(abs(b[5, 0, 0, 1]), 0.15)
        b = np.array(cfa.edge_profile(self._items('max'))['beta'])
        self.assertGreater(b[5, 0, 1, 1], 0.85)
        uv = np.array(cfa.edge_profile(self._items('max'))['uv_median'])
        self.assertGreater(uv[5, 0, 0], 0.8)

    def test_category(self):
        self.assertEqual(cfa.category('man in blue'), 'person')
        self.assertEqual(cfa.category('right front sheep'), 'animal')
        self.assertEqual(cfa.category('top doughnut'), 'other')


if __name__ == '__main__':
    unittest.main()
