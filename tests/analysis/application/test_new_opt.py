"""CPU tests of experiment 7.5 (new optimisation pilots): masks, scopes, answer formats, parsing."""
import unittest

import torch

from analysis_tools.application import new_opt as no
from analysis_tools.application import new_opt_run as nor


class TestMask(unittest.TestCase):
    def test_own_entity_mask(self):
        groups = [[2, 3], [4, 5]]
        m = no.own_entity_mask(7, groups)
        self.assertTrue(m[3, 1])                 # earlier text token stays visible
        self.assertFalse(m[3, 2])                # own entity (other token)
        self.assertFalse(m[3, 3])                # own entity (itself)
        self.assertTrue(m[4, 3])                 # earlier entity stays visible
        self.assertFalse(m[2, 4])                # causal
        self.assertTrue(m[6, 6])                 # text token after the image keeps itself
        self.assertTrue(bool(m.any(-1).all()))   # every query keeps at least one key

    def test_conditions(self):
        self.assertEqual(nor.parse_condition('own_l3-5_all', 32), [(3, None), (4, None), (5, None)])
        self.assertEqual(nor.parse_condition('own_l8_h691', 32), [(8, (6, 9, 1))])
        with self.assertRaises(ValueError):
            nor.parse_condition('own_l40_all', 36)

    def test_bins(self):
        self.assertEqual(no.raster_bin([0, 0]), 'c1')
        self.assertEqual(no.raster_bin([0, 5]), 'c2-6')
        self.assertEqual(no.raster_bin([1, 5]), 'c7-12')
        self.assertEqual(no.raster_bin([5, 5]), 'c13-36')


class TestDecomp(unittest.TestCase):
    def test_restrict_rows(self):
        from analysis_tools.position import targeted_finetune as tf
        hd, H = 2, 3
        mods = {'L8.self_attn.q_proj': torch.nn.Linear(4, H * 2 * hd), 'L8.self_attn.k_proj': torch.nn.Linear(4, hd)}
        for part in ('gate', 'query_key'):
            lora = tf.LoRA(mods, 2, 8)
            with torch.no_grad():
                for p in lora.B.values():
                    p.fill_(1.0)
            nor._restrict(lora, part, hd)
            Bq = lora.B[lora._key('L8.self_attn.q_proj')]
            Bk = lora.B[lora._key('L8.self_attn.k_proj')]
            gate_rows = (torch.arange(H * 2 * hd) % (2 * hd)) >= hd
            if part == 'gate':
                self.assertTrue(bool((Bq[gate_rows] == 1).all() and (Bq[~gate_rows] == 0).all()))
                self.assertEqual(float(Bk.abs().sum()), 0.0)
            else:
                self.assertTrue(bool((Bq[gate_rows] == 0).all() and (Bq[~gate_rows] == 1).all()))
                self.assertTrue(bool((Bk == 1).all()))


class TestScopes(unittest.TestCase):
    def test_params(self):
        self.assertEqual(no.scope_params('fa8qk'), 16 * (4096 + 8192 + 4096 + 1024))
        self.assertEqual(no.scope_params('fa8qk'), no.scope_params('fa12qk'))
        self.assertEqual(no.scope_params('mlp8_15'), 8 * 3 * (4096 + 12288))


class TestFormats(unittest.TestCase):
    P = ('Locate the object described as: top cow. Give the center of its bounding box. Return only '
         '[{"point_2d": [x, y]}]. All coordinates use 0-1000 normalization; x increases rightward and y increases '
         'downward. Output only the JSON, with no explanation.')
    B = P.replace('Give the center of its bounding box. Return only [{"point_2d": [x, y]}].',
                  'Give its tight bounding box. Return only [{"bbox_2d": [x_min, y_min, x_max, y_max]}].')

    def test_prompts(self):
        self.assertEqual(no.format_prompt(self.P, 'point', 'list'), self.P)
        k = no.format_prompt(self.P, 'point', 'key')
        self.assertIn('{"x": x, "y": y}', k)
        self.assertIn('Output only the answer,', k)
        self.assertIn('x_min,y_min', no.format_prompt(self.B, 'bbox', 'bare'))
        with self.assertRaises(ValueError):
            no.format_prompt('no format here', 'point', 'key')

    def test_parse(self):
        self.assertEqual(no.parse_values('[{"point_2d": [512, 87]}]', 'point', 'list'), [512.0, 87.0])
        self.assertEqual(no.parse_values('{"x": 512, "y": 87}', 'point', 'key'), [512.0, 87.0])
        self.assertEqual(no.parse_values('512,87\n', 'point', 'bare'), [512.0, 87.0])
        self.assertEqual(no.parse_values('```\n512,87\n```', 'point', 'bare'), [512.0, 87.0])
        self.assertEqual(no.parse_values('```json\n[\n\t{"point_2d": [550, 160], "label": "cow"}\n]\n```', 'point',
                                         'list'), [550.0, 160.0])
        self.assertEqual(no.parse_values('{"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}', 'bbox', 'key'),
                         [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(no.parse_values('512', 'point', 'bare'))
        self.assertIsNone(no.parse_values('{"x": 512}', 'point', 'key'))

    def test_wilson(self):
        p, lo, hi = no.wilson(5, 10)
        self.assertAlmostEqual(p, 0.5)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


if __name__ == '__main__':
    unittest.main()
