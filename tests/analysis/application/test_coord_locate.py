"""Tests of the model-free parts of experiment 7.2 (analysis_tools/application/coord_locate.py)."""
import unittest

import numpy as np
import torch

from analysis_tools.application import coord_locate as cl


class MaskTest(unittest.TestCase):
    def test_ids_mask(self):
        ids = torch.tensor([[1, 9, 9, 9, 2, 3]])
        self.assertEqual(cl.ids_mask(ids, 9, 'image').tolist(), [[0, 1, 1, 1, 0, 0]])
        self.assertEqual(cl.ids_mask(ids, 9, 'image_nofirst').tolist(), [[0, 0, 1, 1, 0, 0]])
        self.assertEqual(cl.ids_mask(ids, 9, 'first').tolist(), [[0, 1, 0, 0, 0, 0]])

    def test_span_mask_matches_ids_mask(self):
        for kind in cl.POSITION_KINDS:
            ids = torch.tensor([[1, 9, 9, 9, 2, 3, 0]])
            np.testing.assert_array_equal(cl.span_mask(kind, (1, 4), 7), cl.ids_mask(ids, 9, kind)[0].numpy())


class SubsetTest(unittest.TestCase):
    def test_nested_and_stratified(self):
        fam = ['a'] * 60 + ['b'] * 40
        s = cl.nested_subsets(fam, (10, 50), 0)
        self.assertTrue(set(s[10]) <= set(s[50]))
        self.assertEqual(sum(fam[i] == 'a' for i in s[50]), 30)


class ScoreTest(unittest.TestCase):
    def test_point_hit(self):
        r = cl.point_hit([250, 500], [100, 200, 160, 300], (512, 512))
        self.assertTrue(r['hit'])
        self.assertFalse(cl.point_hit(None, [0, 0, 1, 1], (1, 1))['valid'])

    def test_lenient_rowcol(self):
        self.assertEqual(cl.lenient_rowcol('reasoning\nFinal answer: (4, 7)'), (4, 7))
        self.assertEqual(cl.lenient_rowcol('Row 4 Column 17'), (4, 17))
        self.assertIsNone(cl.lenient_rowcol('no numbers'))

    def test_ordinal_error_kind(self):
        self.assertEqual(cl.ordinal_error_kind(14, 14, 675), 'correct')
        self.assertEqual(cl.ordinal_error_kind(15, 14, 675), 'off_by_1')
        self.assertEqual(cl.ordinal_error_kind(67, 14, 675), 'coord_prefix')
        self.assertEqual(cl.ordinal_error_kind(71, 15, 725), 'coord_prefix')
        self.assertEqual(cl.ordinal_error_kind(22, 15, 725), 'other')
        self.assertEqual(cl.ordinal_error_kind(None, 1, 25), 'unparsed')


if __name__ == '__main__':
    unittest.main()
