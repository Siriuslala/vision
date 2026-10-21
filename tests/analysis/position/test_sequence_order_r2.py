"""CPU tests for the layout builder, gate formulas and probe helpers of experiment 3.8 round 2."""
import unittest

import numpy as np
import torch
from PIL import Image, ImageDraw

from analysis_tools.position import sequence_order as so
from analysis_tools.position import sequence_order_r2 as s2

W = 16


class LayoutTest(unittest.TestCase):
    def test_identity_layout_is_raster(self):
        L = s2.make_layout(s2.grid_rows(8, 8), W)
        self.assertEqual(L.index.tolist(), list(range(256)))
        for cell in L.cells:
            self.assertEqual(cell['ord_row'], cell['vit_row'])
            self.assertEqual(cell['ord_col'], cell['vit_col'])
            self.assertEqual(cell['slot_row'], cell['vit_row'])
        self.assertEqual(sorted(L.index[L.cell_of_slot == 10].tolist()), so.cell_tokens((1, 2), (16, 16)))

    def test_dissociating_layouts(self):
        lay = s2.layout_conditions()
        blank = s2.make_layout(lay['insert_blank_start'], W)
        self.assertEqual((blank.index < 0).sum(), 2 * W)
        self.assertTrue(all(c['ord_row'] == c['vit_row'] and c['slot_row'] == c['vit_row'] + 1 for c in blank.cells))
        dele = s2.make_layout(lay['delete_row_mid'], W)
        below = [c for c in dele.cells if c['vit_row'] >= 4]
        self.assertTrue(all(c['ord_row'] == c['vit_row'] - 1 for c in below))
        dup = s2.make_layout(lay['dup_col_mid'], W)
        self.assertEqual(len(dup.index), 8 * 9 * 4)
        self.assertTrue(all(c['ord_col'] == c['vit_col'] + 1 for c in dup.cells if c['ord_col'] >= 5))
        roll = s2.make_layout(lay['roll_rows+1'], W)
        self.assertEqual(roll.cells[0]['vit_row'], 7)
        self.assertEqual(roll.cells[0]['ord_row'], 0)


class JobTest(unittest.TestCase):
    def test_intervention_positions(self):
        from analysis_tools.position import sequence_order_r2_run as r2
        jobs = r2.intervention_jobs({'id': 'x', 'template': 'person', 'target': [3, 5]}, 16, 16, 100)
        byname = {}
        for j in jobs:
            byname.setdefault((j['base'], j['name']), []).append(j)
        rs = byname[('identity', 'reset_rowstart|mid')][0]['ov']
        self.assertEqual(rs['reset'].tolist(), list(range(100, 356, 16)))
        top = byname[('identity', 'nowrite_top|mid')][0]['ov']['nowrite']
        self.assertEqual(len(top), 128)
        self.assertTrue(all(((p - 100) // 16) % 2 == 0 for p in top))
        ks = sorted(j['k'] for j in byname[('identity', 'reset_at_k|all')])
        self.assertEqual(ks, [2, 3])
        k3 = [j for j in byname[('identity', 'reset_at_k|all')] if j['k'] == 3][0]['ov']['reset']
        self.assertEqual(k3.tolist(), [100 + 3 * 32])
        tgt = byname[('block_down', 'nowrite_target|mid')][0]['ov']['nowrite']
        self.assertEqual(sorted((tgt - 100).tolist()), so.cell_tokens((4, 5), (16, 16)))
        jobs2 = r2.intervention_jobs2({'id': 'x', 'template': 'ring', 'target': [3, 5]}, 16, 16, 0)
        self.assertEqual({j['base'] for j in jobs2}, {'identity', 'delete_row_before', 'roll_rows-1', 'block_down'})
        dele = [j for j in jobs2 if j['base'] == 'delete_row_before' and j['name'] == 'reset_rowstart|all'][0]
        self.assertEqual(len(dele['order']), 224)
        self.assertEqual(dele['ov']['reset'].tolist(), list(range(0, 224, 16)))


class HelperTest(unittest.TestCase):
    def test_gate_values(self):
        mod = torch.nn.Module()
        mod.A_log = torch.nn.Parameter(torch.log(torch.tensor([0.5, 2.0])))
        mod.dt_bias = torch.nn.Parameter(torch.zeros(2))
        a = torch.tensor([[[0.0, 0.0], [s2.BIG, -s2.BIG]]])
        b = torch.tensor([[[0.0, -s2.BIG], [0.0, 0.0]]])
        al, be = s2.gate_values(mod, a, b)
        self.assertAlmostEqual(float(al[0, 0, 0]), float(torch.exp(-0.5 * torch.log(torch.tensor(2.0)))), places=5)
        self.assertAlmostEqual(float(al[0, 1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(al[0, 1, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(be[0, 0, 1]), 0.0, places=6)

    def test_group_mean_and_ridge(self):
        h = torch.arange(12.0).view(6, 2)
        g = torch.tensor([0, 0, 1, -1, 1, 2])
        m = s2.group_mean(h, g, 3)
        self.assertTrue(torch.allclose(m, torch.tensor([[1.0, 2.0], [6.0, 7.0], [10.0, 11.0]])))
        X = torch.randn(400, 5, dtype=torch.float64)
        y = X @ torch.tensor([1.0, -2.0, 0.0, 0.5, 3.0], dtype=torch.float64)
        model, _ = s2.ridge_select(X, y, np.arange(400) // 10, alphas=(1e-3,))
        self.assertGreater(s2.r2_score(s2.ridge_predict(model, X), y), 0.999)

    def test_figure_mask_and_parse(self):
        im = Image.new('RGB', (64, 64), (255, 255, 255))
        ImageDraw.Draw(im).rectangle([0, 0, 20, 20], fill=(0, 0, 0))
        self.assertEqual(s2.figure_mask(im, (2, 2)).tolist(), [True, False, False, False])
        self.assertEqual(s2.parse_last_rowcol('first {"row": 2, "column": 3} then {"row": 4, "column": 5}'), (4, 5))
        self.assertIsNone(s2.parse_last_rowcol('no answer'))


if __name__ == '__main__':
    unittest.main()
