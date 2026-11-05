"""Model-free checks of experiment 3.13 round 2: splice configurations, position masks, LoRA controls, digit metrics,
code geometry and the GQA expression builder."""
import unittest

import numpy as np
import torch
from torch import nn

from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_r2 as t2


def _names():
    names = []
    for l in range(1, 33):
        names += [f'L{l}.mlp.{m}' for m in ('gate_proj', 'up_proj', 'down_proj')]
        if l in t2.FULL_LAYERS:
            names += [f'L{l}.self_attn.{m}' for m in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
        else:
            names += [f'L{l}.linear_attn.{m}' for m in ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj')]
    return names


class SpliceTests(unittest.TestCase):
    def test_depth_configs_partition(self):
        cfg = {c.name: c for c in t2.splice_configs(_names())}
        every = cfg['full'].active
        for k in range(4, 32, 4):
            fwd, rev = cfg[f'fwd{k}'].active, cfg[f'rev{k}'].active
            self.assertEqual(fwd | rev, every)
            self.assertFalse(fwd & rev)
            self.assertTrue(all(int(n.split('.')[0][1:]) <= k for n in fwd))
        self.assertEqual(len(cfg['base'].active), 0)
        self.assertTrue(all('.mlp.' in n for n in cfg['mlp'].active))
        self.assertEqual(cfg['mixer'].active | cfg['mlp'].active, every)
        self.assertEqual(cfg['fa'].active | cfg['dn'].active, cfg['mixer'].active)
        bands = set().union(*(cfg[f'band{a}_{a + 3}'].active for a in range(1, 33, 4)))
        self.assertEqual(bands, every)
        self.assertEqual(cfg['fwd24@image'].positions, 'image')

    def test_position_masks(self):
        span, L, P, T = (3, 7), 12, 10, 14
        img = t2.position_mask('image', L, span, P, T)
        q = t2.position_mask('question', L, span, P, T)
        a = t2.position_mask('answer', L, span, P, T)
        txt = t2.position_mask('text', L, span, P, T)
        self.assertEqual(img.sum(), 4)
        self.assertEqual(q.sum(), 6)
        self.assertEqual(a.sum(), 2)
        np.testing.assert_array_equal(txt, q + a)
        np.testing.assert_array_equal(img + txt, np.r_[np.ones(L), np.zeros(T - L)])
        with self.assertRaises(ValueError):
            t2.position_mask('foo', L, span, P, T)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj = nn.Linear(4, 6), nn.Linear(4, 6), nn.Linear(6, 4)


class LoRAControlTests(unittest.TestCase):
    def test_active_and_mask(self):
        torch.manual_seed(0)
        layer = _Layer()
        mods = {'L1.mlp.up_proj': layer.mlp.up_proj, 'L1.mlp.gate_proj': layer.mlp.gate_proj}
        lora = tf.LoRA(mods, 2, 4.0)
        with torch.no_grad():
            for p in lora.B.values():
                p.normal_()
        x = torch.randn(2, 5, 4)
        base = layer.mlp.up_proj(x)
        lora.attach()
        full = layer.mlp.up_proj(x)
        self.assertGreater(float((full - base).abs().max()), 1e-4)
        lora.active = {'L1.mlp.gate_proj'}
        torch.testing.assert_close(layer.mlp.up_proj(x), base)
        lora.active = None
        mask = torch.zeros(2, 5)
        mask[:, 2] = 1
        lora.pos_mask = mask
        out = layer.mlp.up_proj(x)
        torch.testing.assert_close(out[:, [0, 1, 3, 4]], base[:, [0, 1, 3, 4]])
        torch.testing.assert_close(out[:, 2], full[:, 2])
        lora.pos_mask = torch.zeros(2, 4)
        with self.assertRaises(ValueError):
            layer.mlp.up_proj(x)
        lora.pos_mask = None
        lora.detach()

    def test_image_mask_from_position_ids(self):
        class _Back(nn.Module):
            def forward(self, *args, **kwargs):
                return None

        class _MM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Back()
        mm = _MM()
        lora = tf.LoRA({'L1.mlp.up_proj': _Layer().mlp.up_proj}, 2, 4.0)
        tf.attach_image_mask(mm, lora, 99)
        t = torch.tensor([2, 3, 4, 4, 4, 4, 6, 7])
        h = torch.tensor([2, 3, 4, 4, 5, 5, 6, 7])
        w = torch.tensor([2, 3, 4, 5, 4, 5, 6, 7])
        pos = torch.stack([torch.arange(8), t, h, w])[:, None]
        mm.model(inputs_embeds=torch.zeros(1), position_ids=pos)
        self.assertEqual(lora.pos_mask[0].tolist(), [0, 0, 1, 1, 1, 1, 0, 0])
        mm.model(input_ids=torch.tensor([[1, 99, 99, 2]]))
        self.assertEqual(lora.pos_mask[0].tolist(), [0, 1, 1, 0])
        lora.detach()

    def test_new_conditions(self):
        c = tf.CONDITIONS
        self.assertEqual(c['all_but_25_32']['layers'], tuple(range(1, 25)))
        self.assertEqual(c['mlp28_32_lr1e3']['lr'], 1e-3)
        ex = {(l, n) for ls, ns in c['all_but_mlp28_32']['exclude'] for l in ls for n in ns}
        self.assertIn((30, 'mlp.up_proj'), ex)
        self.assertNotIn((27, 'mlp.up_proj'), ex)

    def test_exclude_in_target_modules(self):
        layers = nn.ModuleList([_Layer() for _ in range(32)])
        mods = tf.target_modules(layers, 'all_but_mlp28_32')
        self.assertIn('L27.mlp.up_proj', mods)
        self.assertNotIn('L28.mlp.up_proj', mods)
        self.assertNotIn('L32.mlp.down_proj', mods)


class MetricTests(unittest.TestCase):
    def test_digit_metrics_ideal(self):
        x = np.arange(100, 1000, 7).astype(float)
        f, s = (x.astype(int) // 100), (x.astype(int) // 10) % 10
        lp0 = np.full((len(x), 10), -30.0)
        lp0[np.arange(len(x)), f] = 0
        lp1 = np.full((len(x), 10), -30.0)
        lp1[np.arange(len(x)), s] = 0
        m = t2.digit_metrics(x, lp0, lp1)
        self.assertAlmostEqual(m['p_first'], 1.0, places=6)
        self.assertAlmostEqual(m['p_second'], 1.0, places=6)
        self.assertAlmostEqual(m['gain'], 1.0, delta=0.02)
        self.assertLess(m['value_err'], 5.1)
        flat = t2.digit_metrics(x, lp0, np.zeros((len(x), 10)))
        self.assertAlmostEqual(flat['gain'], 0.0, places=6)
        self.assertAlmostEqual(t2.recovery(0.5, 0.0, 1.0), 0.5)

    def test_margin_scatter(self):
        x = np.arange(200, 900, 3).astype(float)
        k = x.astype(int) // 100
        off = (x - 100 * k) / 10
        lp = np.zeros((len(x), 10))
        lp[np.arange(len(x)), k + 1] = 0.8 * off / 2
        lp[np.arange(len(x)), k - 1] = -0.8 * off / 2
        m = t2.margin_scatter(x, lp)
        self.assertAlmostEqual(m['slope'], 0.8, places=6)
        self.assertLess(m['scatter'], 1e-6)


class GeometryTests(unittest.TestCase):
    def test_periodic_spectrum_detects_period(self):
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(100, 900, 200))
        D = 64
        a, b, c = (rng.normal(size=D) for _ in range(3))
        place = np.outer(x / 1000, a) + 0.01 * rng.normal(size=(len(x), D))
        clock = place + np.outer(np.cos(2 * np.pi * x / 100), b) + np.outer(np.sin(2 * np.pi * x / 100), c)
        sp = t2.periodic_spectrum(place, x, (50.0, 100.0, 200.0))
        sc = t2.periodic_spectrum(clock, x, (50.0, 100.0, 200.0))
        self.assertLess(sp['extra'][100.0], 0.05)
        self.assertGreater(sc['extra'][100.0], 0.5)
        self.assertGreater(sc['extra'][100.0], sc['extra'][50.0])

    def test_periodic_decodability(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(100, 900, 300)
        D = 40
        a, b, c = (rng.normal(size=D) for _ in range(3))
        noise = 0.05 * rng.normal(size=(len(x), D))
        place = np.outer(x / 1000, a) + noise
        clock = place + np.outer(np.cos(2 * np.pi * x / 100), b) + np.outer(np.sin(2 * np.pi * x / 100), c)
        dp, dc = t2.periodic_decodability(place, x, n_pc=10), t2.periodic_decodability(clock, x, n_pc=10)
        self.assertGreater(dp['x'], 0.9)
        self.assertLess(dp['periodic'], 0.2)
        self.assertGreater(dc['periodic'], 0.9)

    def test_principal_angles_and_code(self):
        e = np.eye(5)
        np.testing.assert_allclose(t2.principal_angles(e[:2], e[:2]), [0, 0], atol=1e-5)
        np.testing.assert_allclose(t2.principal_angles(e[:1], e[1:2]), [90], atol=1e-5)
        ang = t2.principal_angles(np.array([[1.0, 1, 0, 0, 0]]), e[:1])
        self.assertAlmostEqual(float(ang[0]), 45.0, places=4)
        x = np.linspace(100, 900, 50)
        H = np.outer(x, e[0]) + 3.0
        V, d = t2.code_subspace(H, x, k=1, periods=())
        self.assertAlmostEqual(abs(float(V[0, 0])), 1.0, places=5)
        np.testing.assert_allclose(d, e[0], atol=1e-6)
        J = np.zeros((4, 10, 5))
        J[:, :, 1] = np.arange(10)
        R, share = t2.readout_subspace(J, 1)
        self.assertAlmostEqual(abs(float(R[0, 1])), 1.0, places=5)
        self.assertAlmostEqual(float(share[0]), 1.0, places=5)
        g = t2.readout_gain(J, e[1])
        np.testing.assert_allclose(g[0], 100 * (np.arange(10) - 4.5))


class DataTests(unittest.TestCase):
    def test_gqa_expressions(self):
        graph = {'width': 100, 'height': 100, 'objects': {
            '1': {'name': 'dog', 'x': 10, 'y': 10, 'w': 20, 'h': 20, 'attributes': ['white']},
            '2': {'name': 'dog', 'x': 60, 'y': 60, 'w': 20, 'h': 20, 'attributes': ['black']},
            '3': {'name': 'cat', 'x': 40, 'y': 10, 'w': 15, 'h': 15, 'attributes': []},
            '4': {'name': 'sky', 'x': 0, 'y': 0, 'w': 100, 'h': 30, 'attributes': ['blue']},
            '5': {'name': 'cup', 'x': 0, 'y': 0, 'w': 2, 'h': 2, 'attributes': []}}}
        ex = {e['expression']: e for e in t2.gqa_expressions(graph)}
        self.assertEqual(set(ex), {'the white dog', 'the black dog', 'the cat'})
        self.assertEqual(ex['the white dog']['kind'], 'attribute')
        self.assertEqual(ex['the cat']['kind'], 'name')
        self.assertEqual(ex['the cat']['box'], [40, 10, 55, 25])
        self.assertEqual(t2.scale_box([10, 20, 30, 40], (100, 200), (50, 100)), [5, 10, 15, 20])
        q, a = t2.existence_question(['dog'], ['dog', 'cat'], np.random.default_rng(1))
        self.assertIn(a, ('yes', 'no'))
        lo, hi = t2.wilson(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


class FigureStyleTests(unittest.TestCase):
    def test_scaled_styles_distinct(self):
        from analysis_tools.position import targeted_finetune_r2_figures as F
        scaled = ('scaled_all_linear', 'scaled_early1_4_img', 'scaled_all_but_25_32')
        styles = [F.SCALED_STYLE[c] for c in scaled]
        self.assertEqual(len(set(styles)), len(styles))       # the three scaled conditions are distinguishable
        self.assertTrue(all(c in F.COND for c in F.SCALED_STYLE))


if __name__ == '__main__':
    unittest.main()
