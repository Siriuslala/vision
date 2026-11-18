"""Model-free tests of experiment 2.7 round 3: counterfactual stimuli and delta-rule bookkeeping."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_circuit as sc
from analysis_tools.representation import search_circuit_run as rr


class StimulusTests(unittest.TestCase):
    def test_odd_and_corrupted_differ_only_at_target(self):
        rng = np.random.default_rng(0)
        jit = sa.jitter_map(6, 6, rng)
        for variant in ('std', 'swap'):
            imgs = sc.render_pair('qo', 6, variant, (2, 3), 1, jit, set())
            a = np.asarray(imgs['odd'][0]).astype(int)
            b = np.asarray(imgs['corrupted'][0]).astype(int)
            diff = np.abs(a - b).sum(-1) > 0
            ys, xs = np.nonzero(diff)
            self.assertTrue(len(ys) > 0)
            self.assertTrue((ys // 64 == 2).all() and (xs // 64 == 3).all())

    def test_blank_cells_and_prev_count(self):
        rng = np.random.default_rng(1)
        for n in sc.PREV_COUNTS:
            blank = sc.blank_cells(n, sc.PREV_TARGET, rng)
            jit = sa.jitter_map(6, 6, rng)
            imgs = sc.render_pair('closure', 4, 'swap', sc.PREV_TARGET, 0, jit, blank)
            meta = imgs['odd'][1]
            self.assertEqual(sc.n_prev_entities(meta, sc.PREV_TARGET), n)
            img = np.asarray(imgs['odd'][0])
            for r, c in blank:
                self.assertTrue((img[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] == 255).all())
            self.assertFalse(any(sc.PREV_TARGET == rc for rc in blank))


class DeltaRuleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        T, d = 12, 8
        k = rng.normal(size=(T, d)); k /= np.linalg.norm(k, axis=1, keepdims=True)
        q = rng.normal(size=(T, d)); q /= np.linalg.norm(q, axis=1, keepdims=True) * np.sqrt(d)
        self.k, self.q = k, q
        self.v = rng.normal(size=(T, d))
        self.g = -rng.uniform(0, 0.5, T)
        self.beta = rng.uniform(0.1, 0.9, T)

    def test_unrolled_prediction_matches_recurrence(self):
        out, err, pred = sc.delta_rule_reference(self.q, self.k, self.v, self.g, self.beta)
        for t in range(1, len(self.k)):
            w = sc.prediction_sources(self.k, self.g, self.beta, err, t)
            np.testing.assert_allclose(w @ err[:t], pred[t], atol=1e-10)

    def test_repeated_token_error_decays(self):
        # one key/value pair repeated: the delta rule is online least squares, so the error shrinks geometrically
        T = 10
        k = np.tile(self.k[:1], (T, 1)); v = np.tile(self.v[:1], (T, 1)); q = np.tile(self.q[:1], (T, 1))
        _, err, _ = sc.delta_rule_reference(q, k, v, np.zeros(T), np.full(T, 0.5))
        norms = np.linalg.norm(err, axis=1)
        np.testing.assert_allclose(norms[1:] / norms[:-1], 0.5, atol=1e-9)

    def test_gpu_tracer_matches_reference(self):
        ctrl = rr.DNControl()
        ctrl.s, ctrl.e = 2, 10
        T, d = self.k.shape
        to = lambda x: torch.tensor(x, dtype=torch.float32)
        q = to(self.q * np.sqrt(d))[None, :, None]          # tracer re-normalises and rescales q
        k, v = to(self.k)[None, :, None], to(self.v)[None, :, None]
        g, beta = to(self.g)[None, :, None], to(self.beta)[None, :, None]
        ctrl.target_tok = [torch.tensor([7])]
        out = rr._trace_recurrence(ctrl, 1, q, k, v, g, beta)
        ref, err, pred = sc.delta_rule_reference(self.q, self.k, self.v, self.g, self.beta)
        np.testing.assert_allclose(out[0, :, 0].numpy(), ref, atol=1e-4)
        rec = ctrl.records[1]
        np.testing.assert_allclose(rec['err_norm'][0, 0].numpy(), np.linalg.norm(err[2:10], axis=1), atol=1e-4)
        self.assertAlmostEqual(float(rec['contrib'][0, 0].sum()), 1.0, places=4)


if __name__ == '__main__':
    unittest.main()


class CircuitHelperTests(unittest.TestCase):
    def test_inverse_rope_is_transpose(self):
        from analysis_tools.representation import search_circuit_heads as H
        torch.manual_seed(0)
        x, y = torch.randn(3, 16), torch.randn(3, 16)
        ang = torch.randn(3, 2)
        cos = torch.cat([ang.cos(), ang.cos()], -1)
        sin = torch.cat([ang.sin(), ang.sin()], -1)
        # <R y, x> = <y, R^T x> with partial rotary (first 4 dims rotated)
        lhs = (rr._rope(y, cos, sin) * x).sum(-1)
        rhs = (y * H.rope_inv(x, cos, sin)).sum(-1)
        torch.testing.assert_close(lhs, rhs)
