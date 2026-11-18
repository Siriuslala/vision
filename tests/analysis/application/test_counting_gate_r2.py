"""CPU tests of experiment 7.3 Part C round 2: gate-modification specs and the alpha / beta transform."""
import math
import unittest

import torch
from torch import nn

from analysis_tools.application import counting_gate_r2 as cg2


class _Lin(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_a = nn.Linear(cg2.cg.HIDDEN, cg2.cg.N_HEADS, bias=False)
        self.in_proj_b = nn.Linear(cg2.cg.HIDDEN, cg2.cg.N_HEADS, bias=False)
        self.dt_bias = nn.Parameter(torch.randn(cg2.cg.N_HEADS))
        self.A_log = nn.Parameter(torch.randn(cg2.cg.N_HEADS))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = _Lin()


def _log_alpha(mod, a):
    return -mod.A_log.exp() * nn.functional.softplus(a + mod.dt_bias)


class TestSpec(unittest.TestCase):
    def test_parse(self):
        s = cg2.parse_spec('full:6;fix:13-15:nonint:0.5@img')
        self.assertEqual(s['parts'][0]['layers'], [6])
        self.assertEqual(s['parts'][1]['layers'], [13, 14, 15])
        self.assertEqual(cg2.parse_layers('1-8'), [1, 2, 3, 5, 6, 7])
        with self.assertRaises(ValueError):
            cg2.parse_spec('full:8@img')
        with self.assertRaises(ValueError):
            cg2.parse_spec('full:6')
        self.assertEqual(cg2.n_trainable('full:6@img'), 2 * 32 * 4096 + 96)
        self.assertEqual(cg2.n_trainable('lora8:6,7@all'), 2 * 2 * 8 * (4096 + 32))
        self.assertEqual(cg2.n_trainable('fix:15:all:1@img'), 0)


class TestTransform(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.layers = nn.ModuleList([_Layer() for _ in range(15)])
        self.x = torch.randn(1, 5, cg2.cg.HIDDEN)

    def test_fixed_scale_and_mask(self):
        lin = self.layers[14].linear_attn
        a0 = lin.in_proj_a(self.x)
        mod = cg2.GateMod(self.layers, 'fix:15:3/4:0.5;fix:15:7:1@img')
        mod.attach()
        mod.pos_mask = torch.tensor([[0., 1., 1., 1., 0.]])
        a1 = lin.in_proj_a(self.x)
        mod.detach()
        la0, la1 = _log_alpha(lin, a0), _log_alpha(lin, a1)
        torch.testing.assert_close(la1[0, 1:4, 3], 0.5 * la0[0, 1:4, 3], rtol=1e-3, atol=1e-5)
        self.assertTrue(torch.all(la1[0, 1:4, 7].abs() < 1e-6))                  # alpha = 1
        torch.testing.assert_close(la1[0, 1:4, 0], la0[0, 1:4, 0])                 # other heads untouched
        torch.testing.assert_close(a1[0, [0, 4]], a0[0, [0, 4]])                   # text positions untouched

    def test_zero_trainable_is_identity_and_dA_scales(self):
        lin = self.layers[5].linear_attn
        a0, b0 = lin.in_proj_a(self.x), lin.in_proj_b(self.x)
        mod = cg2.GateMod(self.layers, 'full:6@all')
        mod.attach()
        torch.testing.assert_close(lin.in_proj_a(self.x), a0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lin.in_proj_b(self.x), b0)
        with torch.no_grad():
            mod.P['L6_dA'][2] = math.log(2.0) / cg2.SCALAR_MULT
            mod.P['L6_db'][1] = 1.0 / cg2.SCALAR_MULT
        a1, b1 = lin.in_proj_a(self.x), lin.in_proj_b(self.x)
        mod.detach()
        torch.testing.assert_close(_log_alpha(lin, a1)[..., 2], 2 * _log_alpha(lin, a0)[..., 2], rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(b1[..., 1], b0[..., 1] + 1.0)
        st = mod.state()
        mod2 = cg2.GateMod(self.layers, 'full:6@all')
        mod2.load(st)
        torch.testing.assert_close(mod2.P['L6_dA'], mod.P['L6_dA'])


if __name__ == '__main__':
    unittest.main()
