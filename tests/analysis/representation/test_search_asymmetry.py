"""Model-free tests of experiment 2.7 (search asymmetry): stimuli, scoring and metrics."""
from __future__ import annotations

import unittest

import numpy as np

from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_run as sr


class SearchAsymmetryTests(unittest.TestCase):
    def test_a_and_b_differ_and_fit_the_cell(self):
        for pair in sa.PAIRS:
            for param in sa.PAIRS[pair]['levels'].values():
                a = sa.render_entity(pair, True, param, 1, (sa.JITTER, -sa.JITTER))
                b = sa.render_entity(pair, False, param, 1, (sa.JITTER, -sa.JITTER))
                self.assertEqual(a.shape, (64, 64, 3))
                self.assertGreater(np.abs(a.astype(int) - b.astype(int)).sum(), 0)
                # the shape stays inside the cell with a white border (entities never touch neighbours)
                for t in (a, b):
                    self.assertTrue((t[:2] == 255).all() and (t[-2:] == 255).all()
                                    and (t[:, :2] == 255).all() and (t[:, -2:] == 255).all())

    def test_salient_differs_more_than_subtle(self):
        for pair, spec in sa.PAIRS.items():
            b = sa.render_entity(pair, False, spec['levels']['subtle'], 0, (0, 0)).astype(int)
            d_sub = np.abs(sa.render_entity(pair, True, spec['levels']['subtle'], 0, (0, 0)).astype(int) - b).sum()
            b2 = sa.render_entity(pair, False, spec['levels']['salient'], 0, (0, 0)).astype(int)
            d_sal = np.abs(sa.render_entity(pair, True, spec['levels']['salient'], 0, (0, 0)).astype(int) - b2).sum()
            assert d_sal > d_sub, pair


    def test_variants_share_everything_but_the_feature(self):
        rng = np.random.default_rng(0)
        jit = sa.jitter_map(3, 3, rng)
        imgs = sa.pair_images('qo', 'subtle', 3, (1, 2), 0, jit)
        arr = {v: np.asarray(im) for v, (im, _) in imgs.items()}
        meta = imgs['std'][1]
        x0, y0, x1, y1 = meta['instances'][1 * 3 + 2]['cell_box']
        mask = np.zeros(arr['std'].shape[:2], bool)
        mask[y0:y1, x0:x1] = True
        # std equals homB outside the target cell and homA inside it; swap is the mirror image
        assert (arr['std'][~mask] == arr['homB'][~mask]).all() and (arr['std'][mask] == arr['homA'][mask]).all()
        assert (arr['swap'][~mask] == arr['homA'][~mask]).all() and (arr['swap'][mask] == arr['homB'][mask]).all()
        flags = [i['has_feature'] for i in imgs['swap'][1]['instances']]
        assert sum(flags) == 8 and not flags[5]


    def test_conjunction_distractors(self):
        rng = np.random.default_rng(1)
        jit = sa.jitter_map(6, 6, rng)
        _, meta = sa.conj_image('conj', 6, (2, 3), jit, rng)
        kinds = [(i['color'], i['shape']) for i in meta['instances']]
        assert kinds.count(('red', 'circle')) == 1
        assert kinds.count(('green', 'circle')) == kinds.count(('red', 'square')) == 17 or \
            abs(kinds.count(('green', 'circle')) - kinds.count(('red', 'square'))) == 1


    def test_score_near_flag(self):
        item = {'target': [4, 4]}
        rec = {'text': '{"row": 4, "column": 6}', 'finish_reason': 'eos'}
        out = sa.score(item, rec)
        assert not out['correct'] and out['near']
        rec = {'text': '{"row": 5, "column": 5}', 'finish_reason': 'length'}
        assert not sa.score(item, rec)['valid']


    def test_mcnemar(self):
        assert sa.mcnemar_exact(0, 0) == 1.0
        assert sa.mcnemar_exact(10, 0) == 2 / 2 ** 10


    def test_mahalanobis_salience_tracks_distractor_spread(self):
        rng = np.random.default_rng(0)
        tight = rng.normal(0, 0.1, (36, 20))
        broad = rng.normal(0, 1.0, (36, 20))
        tight[5] += 1.0
        broad[5] += 1.0
        assert sa.mahalanobis_salience(tight, 5) > 3 * sa.mahalanobis_salience(broad, 5)


    def test_selectivity_and_target_dprime(self):
        a = np.ones((10, 3)) + np.random.default_rng(0).normal(0, 0.1, (10, 3))
        b = np.zeros((10, 3)) + np.random.default_rng(1).normal(0, 0.1, (10, 3))
        assert (sa.selectivity(a, b) > 5).all()
        act = np.zeros((36, 2))
        act[3] = [1.0, -1.0]
        dev = sa.target_dprime(act, 3)
        assert dev[0] > 0 > dev[1]


    def test_mech_conditions_cover_layer_types(self):
        tok = [[i] for i in range(36)]
        conds = {c['name']: c for c in sr.mech_conditions(6, (2, 3), tok, 100)}
        assert all(f'mix_tgt_L{l}' in conds for l in range(1, 17))
        dn = [site for site, _, _ in conds['mix_tgt_dn1_8']['mix']]
        assert dn == ['dec1', 'dec2', 'dec3', 'dec5', 'dec6', 'dec7']
        assert [site for site, _, _ in conds['mix_tgt_fa4_8']['mix']] == ['dec4', 'dec8']
        reset = conds['reset_tgt_dn1_16']['gates']['reset']
        assert sorted(reset) == [l for l in range(1, 17) if l % 4] and list(reset[1][0]) == [100 + 15]
        before = conds['nowrite_before_dn1_8']['gates']['nowrite'][1][0]
        assert list(before) == [100 + i for i in range(15)]


    def test_vit_patch_index(self):
        assert sr.vit_patch_index([0, 2]) == [0, 1, 2, 3, 8, 9, 10, 11]


if __name__ == '__main__':
    unittest.main()
