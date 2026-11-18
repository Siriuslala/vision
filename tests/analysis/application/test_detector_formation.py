"""CPU tests of experiment 7.3 round 2 (ViT detector formation): trainable updates and readout measures."""
import unittest

import numpy as np
import torch
from torch import nn

from analysis_tools.application import detector_formation as dfm


class _MLP(nn.Module):
    def __init__(self, d=6, h=10):
        super().__init__()
        self.linear_fc1 = nn.Linear(d, h)
        self.linear_fc2 = nn.Linear(h, d)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class TestUpdates(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mlp = _MLP()
        self.x = torch.randn(4, 6)

    def test_dense_delta_starts_at_identity_and_matches_weight_edit(self):
        dd = dfm.DenseDelta({'fc1': self.mlp.linear_fc1})
        y0 = self.mlp(self.x)
        dd.attach()
        self.assertTrue(torch.allclose(self.mlp(self.x), y0))
        with torch.no_grad():
            dd.W['fc1'].normal_()
            dd.b['fc1'].normal_()
        y1 = self.mlp(self.x)
        dd.detach()
        ref = _MLP()
        ref.load_state_dict(self.mlp.state_dict())
        with torch.no_grad():
            ref.linear_fc1.weight += dd.W['fc1']
            ref.linear_fc1.bias += dd.b['fc1']
        self.assertTrue(torch.allclose(y1, ref(self.x), atol=1e-5))
        self.assertEqual(dd.n_params(), dfm.full_params(1, 6, 10) - (6 * 10 + 6))   # fc1 only here

    def test_width_extension_starts_at_identity_and_roundtrips(self):
        we = dfm.WidthExtension({'m': self.mlp}, m=3, seed=1)
        y0 = self.mlp(self.x)
        we.attach()
        self.assertTrue(torch.allclose(self.mlp(self.x), y0))
        with torch.no_grad():
            we.W_out['m'].normal_()
        we.record = True
        y1 = self.mlp(self.x)
        a = we.acts['m']
        self.assertEqual(tuple(a.shape), (4, 3))
        self.assertTrue(torch.allclose(y1 - y0, a @ we.W_out['m'].t(), atol=1e-5))
        state = we.state()
        we.detach()
        we2 = dfm.WidthExtension({'m': self.mlp}, m=3, seed=5)
        we2.load(state)
        self.assertTrue(torch.equal(we2.W_in['m'], we.W_in['m']))
        self.assertEqual(we.n_params(), dfm.width_params(3, 1, 6))
        # new rows start with the median fc1 row norm and the median fc1 bias
        self.assertAlmostEqual(float(we.W_in['m'].norm(dim=1)[0]),
                               float(self.mlp.linear_fc1.weight.norm(dim=1).median()), places=5)


class TestMeasures(unittest.TestCase):
    def test_contributions_sum_to_projection(self):
        rng = np.random.default_rng(0)
        W = rng.normal(size=(5, 8))
        v = rng.normal(size=5)
        a_t, a_d = rng.normal(size=8), rng.normal(size=8)
        c = dfm.neuron_contributions(v @ W, a_t, a_d)
        self.assertAlmostEqual(c.sum(), v @ (W @ a_t - W @ a_d))

    def test_effective_count(self):
        self.assertAlmostEqual(dfm.effective_count(np.array([1.0, 0, 0, 0])), 1.0)
        self.assertAlmostEqual(dfm.effective_count(np.ones(50)), 50.0)
        self.assertAlmostEqual(dfm.top_share(np.array([3.0, 1.0, 0.0]), 1), 0.75)

    def test_windows_and_spearman(self):
        self.assertEqual(dfm.WINDOWS[0], (1, 2, 3))
        self.assertEqual(dfm.WINDOWS[-1], (25, 26, 27))
        ws = dfm.window_scores({b: float(b) for b in range(1, 28)}, dfm.WINDOWS)
        self.assertEqual(ws[(4, 5, 6)], 5.0)
        self.assertAlmostEqual(dfm.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(dfm.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


if __name__ == '__main__':
    unittest.main()
