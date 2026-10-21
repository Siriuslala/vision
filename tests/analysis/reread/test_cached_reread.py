"""CPU tests of the experiment 5.2 engine (windows, float MRoPE ids, tool-call parsing, scoring, statistics)."""
import types
import unittest

import numpy as np
import torch

from analysis_tools.reread import cached_reread as cr
from analysis_tools.reread import cached_reread_analysis as ca
from analysis_tools.reread import reread_pilot as rp


class WindowTest(unittest.TestCase):
    def test_box_to_window_snaps_outward(self):
        w = cr.box_to_window([100, 200, 400, 600], (10, 20))
        self.assertEqual((w.r0, w.c0, w.h, w.w), (2, 2, 4, 6))

    def test_min_side_and_border(self):
        w = cr.box_to_window([990, 990, 1000, 1000], (10, 10))
        self.assertEqual((w.h, w.w), (4, 4))
        self.assertEqual((w.r0 + w.h, w.c0 + w.w), (10, 10))

    def test_reversed_and_out_of_range_boxes(self):
        w = cr.box_to_window([600, 700, -50, 1200], (10, 10))
        self.assertEqual((w.r0, w.c0, w.h, w.w), (6, 0, 4, 6))

    def test_crop_scale_cap(self):
        self.assertEqual(cr.crop_scale(rp.Window(0, 0, 10, 10), 4), 4)
        self.assertEqual(cr.crop_scale(rp.Window(0, 0, 40, 40), 2), 1)

    def test_resize_budget(self):
        from PIL import Image
        im = cr.resize_budget(Image.new('RGB', (1024, 664)), 512 * 512)
        self.assertEqual((im.width % 32, im.height % 32), (0, 0))
        self.assertLessEqual(im.width * im.height, 512 * 512)


class MropeTest(unittest.TestCase):
    def test_new_mode_matches_pilot(self):
        segs = [cr.VSeg(torch.zeros(6, 2), (2, 3)), cr.VSeg(torch.zeros(4, 2), (2, 2))]
        kinds = [-1, -1] + [0] * 6 + [-1] + [1] * 4 + [-1, -1]
        ref = rp.mrope_ids(kinds, [rp.Segment(torch.zeros(6, 2), (2, 3)), rp.Segment(torch.zeros(4, 2), (2, 2))])
        self.assertTrue(torch.equal(cr.mrope_float(kinds, segs), ref.float()))

    def test_orig_mode_with_half_step(self):
        segs = [cr.VSeg(torch.zeros(16, 2), (4, 4)), cr.VSeg(torch.zeros(4, 2), (2, 2), 'orig', (1.0, 2.0), 0.5)]
        kinds = [-1] + [0] * 16 + [-1] + [1] * 4 + [-1]
        ids = cr.mrope_float(kinds, segs)
        seg = ids[:, 18:22]
        self.assertTrue(torch.allclose(seg[1], torch.tensor([2.0, 2.0, 2.5, 2.5])))   # first_start 1 + 1 + r/2
        self.assertTrue(torch.allclose(seg[2], torch.tensor([3.0, 3.5, 3.0, 3.5])))
        self.assertEqual(float(seg[0, 0]), 6.0)          # temporal id at the segment's own position
        self.assertEqual(float(ids[0, -1]), 8.0)         # text resumes at start + max(h, w)

    def test_reverse_order_ids_follow_tokens(self):
        segs = [cr.VSeg(torch.zeros(4, 2), (2, 2)),
                cr.VSeg(torch.zeros(4, 2), (2, 2), 'orig', (0.0, 0.0), 1.0, torch.arange(3, -1, -1))]
        ids = cr.mrope_float([0] * 4 + [1] * 4, segs)
        self.assertTrue(torch.equal(ids[1:, 4:], ids[1:, :4].flip(-1)))

    def test_first_segment_must_be_default(self):
        segs = [cr.VSeg(torch.zeros(1, 2), (1, 1), 'orig')]
        with self.assertRaises(ValueError):
            cr.mrope_float([0], segs)


class ParseTest(unittest.TestCase):
    def test_parse_call(self):
        t = ('I will zoom.\n\n<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n[10, 20.5, 300, 400]\n'
             '</parameter>\n<parameter=label>\nsign\n</parameter>\n</function>\n</tool_call>')
        self.assertEqual(cr.parse_call(t), [10.0, 20.5, 300.0, 400.0])
        self.assertIsNone(cr.parse_call('The answer is 3.'))
        self.assertIsNone(cr.parse_call('<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n[1, 2]\n'
                                        '</parameter>'))

    def test_answers(self):
        self.assertEqual(cr.vqa_accuracy('The Oscars.', ['oscars'] * 3 + ['x'] * 7), 1.0)
        self.assertAlmostEqual(cr.vqa_accuracy('two', ['2', 'two', 'three']), 2 / 3)
        self.assertEqual(cr.option_letter('(b) Closed'), 'b')
        self.assertEqual(cr.option_letter('A.'), 'a')
        self.assertIsNone(cr.option_letter('Based on the image'))
        self.assertEqual(cr.yes_no_answer('Yes.'), 'yes')
        self.assertIsNone(cr.yes_no_answer('Based on'))

    def test_final_answer(self):
        self.assertEqual(cr.final_answer('The sign says stop.\nFinal answer: **Stop**.'), ('Stop', True))
        self.assertEqual(cr.final_answer('Final answer: no\nFinal answer: yes'), ('yes', True))
        self.assertEqual(cr.final_answer('giraffe'), ('giraffe', False))

    def test_kv_bytes(self):
        t = types.SimpleNamespace(layer_types=['linear_attention'] * 3 + ['full_attention'], num_key_value_heads=4,
                                  head_dim=256, linear_num_value_heads=32, linear_key_head_dim=128,
                                  linear_value_head_dim=128, linear_num_key_heads=16, linear_conv_kernel_dim=4)
        k = cr.kv_bytes(types.SimpleNamespace(text_config=t), 10)
        self.assertEqual(k['per_token_bytes'], 2 * 4 * 256 * 2)
        self.assertEqual(k['kv_bytes'], 10 * 4096)


class StatsTest(unittest.TestCase):
    def test_holm_monotone(self):
        self.assertEqual(ca.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_paired_mcnemar(self):
        a = np.array([1, 1, 1, 0, 1.0])
        b = np.array([0, 0, 1, 0, 1.0])
        r = ca.paired(a, b, np.random.default_rng(0), B=200)
        self.assertEqual((r['a_only'], r['b_only']), (2, 0))
        self.assertAlmostEqual(r['p'], 0.5)

    def test_auroc(self):
        self.assertEqual(ca.auroc(np.array([2.0, 1.0, 0.0]), np.array([1, 0, 0])), 1.0)


if __name__ == '__main__':
    unittest.main()
