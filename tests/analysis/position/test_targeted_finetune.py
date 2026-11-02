"""Model-free checks of experiment 3.13: answer format, parsing, metrics and the hook-based LoRA."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from analysis_tools.position import targeted_finetune as tf


def _sample(fmt='point', diameter=None):
    truth = {'canvas': [500, 250], 'box': [100.0, 50.0, 200.0, 150.0], 'center': [150.0, 100.0]}
    if diameter:
        truth['diameter'] = diameter
    return tf.Sample('s0', 'flat', 'test', '', fmt, truth, tf.question('the red square', fmt), 'the red square')


class FormatTests(unittest.TestCase):
    def test_answer_text_units(self):
        t = _sample().truth
        a = tf.answer_text('point', t, 'x "y"')
        self.assertIn('"point_2d": [300, 400]', a)
        self.assertIn("x 'y'", a)
        b = tf.answer_text('bbox', t, 'z')
        self.assertIn('"bbox_2d": [200, 200, 400, 600]', b)
        self.assertEqual(tf.parse_answer(a, 'point'), [300.0, 400.0])
        self.assertEqual(tf.parse_answer(b, 'bbox'), [200.0, 200.0, 400.0, 600.0])
        with self.assertRaises(ValueError):
            tf.answer_text('row', t, 'z')

    def test_to_units_clips(self):
        self.assertEqual(tf.to_units(-3, 100), 0)
        self.assertEqual(tf.to_units(130, 100), 1000)

    def test_parse_rejects(self):
        self.assertIsNone(tf.parse_answer('no json', 'point'))
        self.assertIsNone(tf.parse_answer('{"point_2d": [1, 2, 3]}', 'point'))

    def test_score_point_and_bbox(self):
        s = _sample()
        r = tf.score(s, '[{"point_2d": [300, 400]}]')
        self.assertTrue(r['hit'])
        self.assertAlmostEqual(r['error'], 0.0)
        r = tf.score(s, '[{"point_2d": [900, 900]}]')
        self.assertFalse(r['hit'])
        b = tf.score(_sample('bbox'), '[{"bbox_2d": [200, 200, 400, 600]}]')
        self.assertAlmostEqual(b['iou'], 1.0)
        self.assertFalse(tf.score(s, 'garbage')['valid'])
        d = tf.score(_sample(diameter=10), '[{"point_2d": [300, 400]}]')
        self.assertTrue(d['hit'])


class MetricTests(unittest.TestCase):
    def test_round_and_plateau(self):
        self.assertAlmostEqual(tf.round_share(np.array([10, 20, 33, 47])), 0.5)
        share, n = tf.plateau_share(np.array([498, 500, 450, 700]), np.array([450, 550, 450, 700]))
        self.assertEqual(n, 3)
        self.assertAlmostEqual(share, 2 / 3)

    def test_slope_and_edges(self):
        x = np.linspace(0, 1000, 21)
        self.assertAlmostEqual(tf.slope(0.5 * x + 3, x), 0.5, places=6)
        edges = [np.linspace(0, 1000, 11)] * 2
        closer, dist = tf.edge_share(np.array([100.0, 150.0]), edges)
        self.assertAlmostEqual(closer, 0.5)
        self.assertAlmostEqual(dist, 0.25)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj = nn.Linear(4, 6), nn.Linear(4, 6), nn.Linear(6, 4)


class LoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.layers = nn.ModuleList([_Layer() for _ in range(32)])

    def test_target_modules(self):
        mods = tf.target_modules(self.layers, 'mlp28_32')
        self.assertEqual(len(mods), 15)
        self.assertIn('L28.mlp.gate_proj', mods)
        with self.assertRaises(ValueError):
            tf.target_modules(self.layers, 'mixer16_24')          # toy layers have no mixers

    def test_zero_init_and_update(self):
        mods = tf.target_modules(self.layers, 'mlp28_32')
        lin = mods['L28.mlp.up_proj']
        x = torch.randn(3, 4)
        ref = lin(x)
        lora = tf.LoRA(mods, rank=2, alpha=4)
        lora.attach()
        self.assertTrue(torch.allclose(lin(x), ref))
        with torch.no_grad():
            lora.B['L28__mlp__up_proj'].fill_(0.1)
        A = lora.A['L28__mlp__up_proj']
        expect = ref + 2.0 * (x @ A.t()) @ torch.full((6, 2), 0.1).t()
        self.assertTrue(torch.allclose(lin(x), expect, atol=1e-5))
        dense = 2.0 * torch.full((6, 2), 0.1) @ A
        self.assertAlmostEqual(lora.delta_norms()['L28.mlp.up_proj'], float(dense.norm()), places=4)
        lora.detach()
        self.assertTrue(torch.allclose(lin(x), ref))

    def test_state_round_trip(self):
        mods = tf.target_modules(self.layers, 'mlp28_32')
        lora = tf.LoRA(mods, 2, 4)
        with torch.no_grad():
            for p in lora.B.values():
                p.normal_()
        state = lora.state()
        state['condition'] = 'mlp28_32'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'a.pt'
            torch.save(state, path)
            loaded = tf.load_adapter(self.layers, path, 'mlp28_32')
            for n in lora.names:
                self.assertTrue(torch.equal(loaded.B[lora._key(n)], lora.B[lora._key(n)]))
            loaded.detach()
            with self.assertRaises(ValueError):
                tf.load_adapter(self.layers, path, 'mlp10_14')


if __name__ == '__main__':
    unittest.main()


class _DeltaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Layer().mlp
        self.linear_attn = nn.Module()
        self.linear_attn.in_proj_qkv = nn.Linear(4, 6)
        self.linear_attn.conv1d = nn.Conv1d(6, 6, 4, groups=6, bias=False)


class ConvDeltaTests(unittest.TestCase):
    def test_conv_delta_attach_detach(self):
        torch.manual_seed(0)
        layers = nn.ModuleList([_DeltaLayer() for _ in range(32)])
        mods = tf.target_modules(layers, 'late25_32')
        self.assertIn('L25.linear_attn.conv1d', mods)
        conv = mods['L25.linear_attn.conv1d']
        w0 = conv.weight.detach().clone()
        lora = tf.LoRA(mods, 2, 4)
        lora.attach()
        self.assertTrue(torch.equal(conv.weight, w0))
        with torch.no_grad():
            lora.D['L25__linear_attn__conv1d'].fill_(0.5)
        self.assertTrue(torch.allclose(conv.weight, w0 + 0.5))
        x = torch.randn(1, 6, 10)
        conv(x).sum().backward()
        self.assertIsNotNone(lora.D['L25__linear_attn__conv1d'].grad)
        state = lora.state()
        self.assertIn('L25.linear_attn.conv1d', state['D'])
        lora.detach()
        self.assertTrue(torch.equal(conv.weight, w0))
