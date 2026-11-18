"""Model-free tests of experiment 2.7 round 4: same-type image replay, geometry helpers and statistics."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from analysis_tools.behavior.decomposition import unit_seed
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_circuit as sc
from analysis_tools.representation import search_pipeline as sp
from analysis_tools.representation import search_pipeline_run as spr


def _fake_item(cfg, pair, level, variant, p):
    """Odd item as written by search_circuit.build (only the fields used by the replay)."""
    rng = np.random.default_rng(unit_seed(cfg['seed'], 'search_circuit', 'atlas', pair, level, variant))
    targets = oe.balanced_targets(sc.GRID, sc.GRID, cfg['atlas_items'], rng)
    for q, target in enumerate(targets):
        orient = int(rng.integers(0, 4))
        jit = sa.jitter_map(sc.GRID, sc.GRID, rng)
        if q == p:
            imgs = sc.render_pair(pair, sc.level_param(pair, level), variant, target, orient, jit, set())
            return {'id': f'images:atlas:{pair}:{level}_{variant}_p{p:03d}_odd', 'set': 'atlas', 'pair_name': pair,
                    'level': level, 'param': sc.level_param(pair, level), 'odd_variant': variant,
                    'target': list(target), 'sha256': sp._png_sha(imgs['odd'][0])}, imgs
    raise AssertionError


class SameTypeTests(unittest.TestCase):
    cfg = {'seed': 0, 'atlas_items': 24}

    def test_replay_reproduces_odd_and_same_type_matches_target(self):
        for variant in ('std', 'swap'):
            it, imgs = _fake_item(self.cfg, 'qo', 'subtle', variant, 5)
            same, meta, ok = sp.render_same_type(self.cfg, it)
            self.assertTrue(ok)
            a = np.asarray(imgs['odd'][0]).astype(int)
            b = np.asarray(same).astype(int)
            r, c = it['target']
            # the target cell is identical; every other cell now shows the target type, so it differs from the odd image
            self.assertTrue((a[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] == b[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64]).all())
            self.assertTrue((np.abs(a - b).sum(-1) > 0).any())
            self.assertTrue(all(inst['has_feature'] == (variant == 'std') for inst in meta['instances']))

    def test_replay_detects_wrong_target(self):
        it, _ = _fake_item(self.cfg, 'closure', 'subtle', 'swap', 3)
        it['target'] = [(it['target'][0] + 1) % 6, it['target'][1]]
        with self.assertRaises(ValueError):
            sp.replay_item(self.cfg, it)


class HelperTests(unittest.TestCase):
    def test_vit_patches_and_redirect(self):
        self.assertEqual(sp.vit_patches_of_tokens([0, 13]), [0, 1, 2, 3, 52, 53, 54, 55])
        for t in range(36):
            e = sp.redirect_cell(t)
            self.assertNotEqual(e // 6, t // 6)
            self.assertNotEqual(e % 6, t % 6)

    def test_zeta_and_participation(self):
        z = sp.zeta(np.array([1.0, 0.1]), np.array([0.0, 0.0]), np.array([0.05, 0.05]))
        self.assertGreater(z[0], 5)
        self.assertLess(z[1], 5)
        x = np.random.default_rng(0).normal(size=(500, 3)) * np.array([1.0, 1.0, 1e-3])
        self.assertAlmostEqual(sp.participation_ratio(x), 2.0, delta=0.15)

    def test_redirect_algebra(self):
        """sigma(g) * sum_i a_i (v_dst_i - v_src_i) added to the head output equals recomputing the head output with the
        source value vectors replaced by the destination ones (attention pattern fixed)."""
        g = torch.Generator().manual_seed(0)
        L, hd = 12, 8
        a = torch.softmax(torch.randn(L, generator=g), 0)
        v = torch.randn(L, hd, generator=g)
        gate = torch.sigmoid(torch.randn(hd, generator=g))
        src, dst = [2, 3], [7, 9]
        o = (a[:, None] * v).sum(0) * gate
        v2 = v.clone()
        v2[src] = v[dst]
        o2 = (a[:, None] * v2).sum(0) * gate
        corr = (a[src][:, None] * (v[dst] - v[src])).sum(0) * gate
        self.assertTrue(torch.allclose(o + corr, o2, atol=1e-6))

    def test_cond_key(self):
        it = {'set': 'mag', 'pair_name': 'qo', 'level': '10.0', 'param': 10.0, 'odd_variant': 'swap'}
        self.assertEqual(spr.cond_key(it), 'mag|qo|10.0|10.0|swap')


if __name__ == '__main__':
    unittest.main()
