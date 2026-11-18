"""Tests of the model-free parts of experiment 7.4 (analysis_tools/application/bv_rowcol.py)."""
import unittest

import numpy as np

from analysis_tools.application import bv_rowcol as bv


class ReplicaTest(unittest.TestCase):
    def test_render_geometry_and_marking(self):
        for i, fam in enumerate(bv.FAMILIES):
            rep, img, marked = bv.make_replica(i, 'eval', fam, 100 + i)
            self.assertEqual(img.size, rep.canvas)
            self.assertEqual(img.size[0] % 32, 0)
            self.assertEqual(len(rep.boxes), rep.rows * rep.cols)
            a, b = np.asarray(img).astype(int), np.asarray(marked).astype(int)
            changed = np.argwhere((a != b).any(-1))
            x0, y0, x1, y1 = rep.boxes[rep.target[0] * rep.cols + rep.target[1]]
            # Every recoloured pixel lies inside the target cell (tight box plus the cell padding).
            self.assertTrue(len(changed) > 0)
            self.assertTrue((changed[:, 1] >= x0 - rep.cell[0]).all() and (changed[:, 1] <= x1 + rep.cell[0]).all())
            self.assertTrue((changed[:, 0] >= y0 - rep.cell[1]).all() and (changed[:, 0] <= y1 + rep.cell[1]).all())

    def test_train_eval_content_disjoint(self):
        self.assertFalse(set(bv.LETTER_PAIRS['train']) & set(bv.LETTER_PAIRS['eval']))
        self.assertFalse(set(bv.WORDS['train']) & set(bv.WORDS['eval']))
        self.assertFalse(set(bv.TEMPLATES['train']) & set(bv.TEMPLATES['eval']))
        self.assertNotIn('number', bv.TRAIN_FAMILIES)

    def test_mediated_answer_gives_truth(self):
        for i in range(20):
            fam = bv.TRAIN_FAMILIES[i % 4]
            rep, _, _ = bv.make_replica(i, 'train', fam, 500 + i)
            text, rc = bv.mediated_answer(rep)
            self.assertEqual(rc, (rep.target[0] + 1, rep.target[1] + 1))
            self.assertEqual(bv.final_rowcol(text), rc)


class ParseTest(unittest.TestCase):
    def test_final_rowcol(self):
        self.assertEqual(bv.final_rowcol('blah (3,4)\nFinal answer: (5, 7)'), (5, 7))
        self.assertEqual(bv.final_rowcol('Final answer: **(12,3)**'), (12, 3))
        self.assertEqual(bv.final_rowcol('row 2 column 9'), (2, 9))
        self.assertIsNone(bv.final_rowcol('no numbers'))

    def test_format_class(self):
        self.assertEqual(bv.format_class('x', 'length', (1, 2)), 'truncated')
        self.assertEqual(bv.format_class("I can't see the image clearly", 'eos', (1, 2)), 'cannot_see')
        self.assertEqual(bv.format_class('hello', 'eos', None), 'unparsed')
        self.assertIsNone(bv.format_class('Final answer: (1,2)', 'eos', (1, 2)))


class DiagnoseTest(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(bv.diagnose((3, 4), None, None, (3, 4), None), 'correct')
        self.assertEqual(bv.diagnose(None, 'truncated', None, (3, 4), None), 'format')
        self.assertEqual(bv.diagnose((3, 5), None, None, (3, 4), None), 'ordinal')
        self.assertEqual(bv.diagnose((9, 9), None, (9, 9), (3, 4), None), 'ordinal')
        self.assertEqual(bv.diagnose((9, 9), None, (3, 4), (3, 4), (2, 3)), 'ordinal')     # point in target cell
        self.assertEqual(bv.diagnose((9, 9), None, (3, 4), (3, 4), (8, 8)), 'detection')


class StatsTest(unittest.TestCase):
    def test_wilson_and_mcnemar(self):
        lo, hi = bv.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertEqual(bv.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(bv.mcnemar_exact(0, 6), 0.03125)


if __name__ == '__main__':
    unittest.main()
