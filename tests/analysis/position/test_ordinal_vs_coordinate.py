"""CPU tests of the experiment-3.15 helpers (index prefixes, edits, follow weights, graded decay, coding)."""
import math
import unittest

import torch
import torch.nn.functional as F

from analysis_tools.position import ordinal_vs_coordinate as ov
from analysis_tools.position import ordinal_vs_coordinate_analysis as oa

COT = ('To find the red figure:\n\n1. **Scan the grid**: The image contains a 10x10 grid of circles (100 total).\n\n'
       '2. **Identify the red circle**: In row 5 (counting from 1 at the top), column 7 (counting from 1 at the left), '
       'there is a red circle.\n\n3. Each cell is 100 units, so x = (7 - 0.5) * 100 = 650.\n\n[{"point_2d": [650, 450]}]\n')
COT_ORD = ('Steps:\n2. The red figure appears in the **9th row** (counting from the top) and **4th column** '
           '(index 3 if starting from 0).\n3. x = 350\n')


class OrdinalVsCoordinateTest(unittest.TestCase):
    def test_index_prefix_stops_before_arithmetic(self):
        info = ov.index_prefix(COT)
        assert info is not None and (info['row'], info['col']) == (5, 7)
        assert '=' not in info['prefix'] and info['prefix'].endswith('there is a red circle.')

    def test_edit_keeps_surface_form(self):
        info = ov.index_prefix(COT)
        edited = ov.edit_indices(info, 2, 9)
        assert 'row 2 (counting' in edited and 'column 9 (counting' in edited
        info = ov.index_prefix(COT_ORD)
        edited = ov.edit_indices(info, 2, 1)
        assert '**2nd row**' in edited and '**1st column**' in edited and 'index 0 if' in edited

    def test_inconsistent_zero_index_is_rejected(self):
        assert ov.index_prefix(COT_ORD.replace('index 3', 'index 5')) is None

    def test_pre_index_prefix_has_no_claim(self):
        pre = ov.pre_index_prefix(COT)
        assert pre is not None and 'row 5' not in pre and pre.endswith('\n')

    def test_full_body_strips_answer(self):
        body = ov.full_body(COT)
        assert body.endswith('= 650.') and 'point_2d' not in body

    def test_follow_weight(self):
        assert ov.follow_weight(550, 350, 550) == 1.0
        assert ov.follow_weight(350, 350, 550) == 0.0
        assert ov.follow_weight(400, 400, 420) is None
        assert ov.cell_centre(7, 10) == 650.0

    def test_graded_a_powers_alpha(self):
        a = torch.randn(3, 4)
        dt = torch.rand(4)
        A_log = torch.rand(4)

        def alpha(x):
            return torch.exp(-A_log.exp() * F.softplus(x.float() + dt))
        for s in (0.5, 0.25):
            assert torch.allclose(alpha(ov.graded_a(a, dt, s)), alpha(a) ** s, atol=1e-5)
        assert torch.allclose(alpha(ov.graded_a(a, dt, 0.0)), torch.ones(3, 4))
        assert torch.equal(ov.graded_a(a, dt, 1.0), a)

    def test_template_and_ordinals(self):
        assert ov.ordinal(1) == '1st' and ov.ordinal(12) == '12th' and ov.ordinal(23) == '23rd'
        assert 'row 3 and column 8' in ov.template_prefix(10, 10, 3, 8)

    def test_natural_coding(self):
        cd = ov.code_natural_cot('The man is on the left side, around (210, 480) in normalized units.')
        assert cd['region'] and cd['pair'] and cd['numbers'] and not cd['arithmetic']
        assert ov.last_pair('box [100, 200, 300, 400] then') == (200.0, 300.0)

    def test_stats(self):
        assert oa.mcnemar(0, 0) == 1.0 and oa.mcnemar(0, 10) < 0.01
        lo, hi = ov.wilson(5, 10)
        assert lo < 0.5 < hi and math.isnan(ov.wilson(0, 0)[0])


if __name__ == '__main__':
    unittest.main()
