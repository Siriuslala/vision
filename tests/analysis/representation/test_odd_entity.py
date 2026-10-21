"""Model-free checks of experiment 2.6: stimuli, token groups, scoring and the oddness/feature split."""
import unittest

import numpy as np

from analysis_tools.common.stimuli import Change, StimulusSpec, generate
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import odd_entity_analysis as oa


class StimulusTests(unittest.TestCase):
    def test_quadruple_matches_stimuli_v1(self):
        quad = oe.grid_quadruple('gap16', 'person', 6, 6, (2, 4), site=1)
        ref, _ = generate(StimulusSpec('person', 6, 6, Change('gap', 4, 1), target=(2, 4)))
        self.assertTrue(np.array_equal(np.asarray(quad['std'][0]), np.asarray(ref)))
        ref_swap, _ = generate(StimulusSpec('person', 6, 6, Change('gap', 4, 1), target=(2, 4), swap=True))
        self.assertTrue(np.array_equal(np.asarray(quad['swap'][0]), np.asarray(ref_swap)))

    def test_quadruple_differences_are_local(self):
        quad = oe.grid_quadruple('red', 'ring', 6, 6, (0, 0), site=0)
        a, x = np.asarray(quad['std'][0]), np.asarray(quad['homX'][0])
        t = np.asarray(quad['homT'][0])
        diff_std_t = np.argwhere((a != t).any(-1))
        self.assertTrue((diff_std_t < 64).all())          # std differs from homT only inside the target cell
        diff_std_x = np.argwhere((a != x).any(-1))
        self.assertFalse(((diff_std_x[:, 0] < 64) & (diff_std_x[:, 1] < 64)).any())  # target identical to homX

    def test_small_grid_is_centred_and_token_aligned(self):
        image, meta = oe.grid_quadruple('red', 'arrow', 3, 3, (1, 1), 0, variants=('std',))['std']
        self.assertEqual(image.size, (256, 256))
        toks = oe.entity_tokens(meta)
        self.assertEqual(len(toks), 9)
        self.assertTrue(all(len(t) == 4 for t in toks))
        self.assertEqual(toks[0], [1 * 8 + 1, 1 * 8 + 2, 2 * 8 + 1, 2 * 8 + 2])

    def test_boundary_halves(self):
        _, meta = oe.boundary_image('red', 'person', 'TB', True, 0)
        changed = [i['changed'] for i in meta['instances']]
        self.assertEqual(sum(changed), 18)
        self.assertTrue(all(changed[:18]))


class GroupTests(unittest.TestCase):
    def test_context_groups(self):
        g = oe.context_groups(6, 6, (0, 0))
        self.assertEqual(sorted(g['nbr']), [1, 6, 7])
        self.assertEqual(len(g['all']), 35)
        self.assertEqual(sorted(g['nbr'] + g['far']), g['all'])
        self.assertEqual(g['before'], [])
        g = oe.context_groups(6, 6, (2, 3))
        self.assertEqual(len(g['nbr']), 8)
        self.assertEqual(len(g['before']), 15)


class ScoreTests(unittest.TestCase):
    def test_score_and_spans(self):
        item = {'target': [1, 4]}
        text = '```json\n{"row": 2, "column": 5}\n```'
        s = oe.score(item, {'text': text, 'finish_reason': 'eos'})
        self.assertTrue(s['correct'])
        s = oe.score(item, {'text': '{"row": 2, "column": 4}', 'finish_reason': 'eos'})
        self.assertFalse(s['correct'])
        self.assertTrue(s['adjacent'])
        self.assertFalse(oe.score(item, {'text': text, 'finish_reason': 'length'})['valid'])
        (a, b), (c, d) = oe.digit_spans(text)
        self.assertEqual(text[a:b], '2')
        self.assertEqual(text[c:d], '5')

    def test_keyword_category(self):
        self.assertEqual(oe.keyword_category(['the black dog', 'dog on left']), 'dog')
        self.assertIsNone(oe.keyword_category(['dog next to cat']))
        self.assertIsNone(oe.keyword_category(['left one']))

    def test_mcnemar(self):
        self.assertAlmostEqual(oa.mcnemar_exact(0, 0), 1.0)
        self.assertLess(oa.mcnemar_exact(20, 1), 1e-4)


class SplitTests(unittest.TestCase):
    def _images(self, odd_weight, feat_weight, n=40, seed=0):
        rng = np.random.default_rng(seed)
        d, k = 32, 9
        u_feat, u_odd, u_ctx = rng.normal(size=(3, d))
        Hs, Hw, targets = [], [], []
        for _ in range(n):
            t = int(rng.integers(0, k))
            for variant, store in (('std', Hs), ('swap', Hw)):
                feat = np.zeros(k)
                if variant == 'std':
                    feat[t] = 1
                else:
                    feat[:] = 1
                    feat[t] = 0
                odd = np.zeros(k)
                odd[t] = 1
                ctx = feat.mean()
                H = (feat[:, None] * feat_weight * u_feat + odd[:, None] * odd_weight * u_odd
                     + ctx * u_ctx + 0.05 * rng.normal(size=(k, d)))
                store.append(H)
            targets.append(t)
        return np.array(Hs), np.array(Hw), np.array(targets)

    def test_pure_feature_code(self):
        Hs, Hw, t = self._images(odd_weight=0.0, feat_weight=1.0)
        m, _ = oa.split_metrics(Hs, Hw, t, n_boot=20)
        self.assertLess(m['cos'], -0.9)
        self.assertLess(m['rho'], 0.05)
        self.assertLess(m['top1_odd_swap'], 0.5)
        self.assertGreater(m['top1_feat_std'], 0.9)

    def test_pure_oddness_code(self):
        Hs, Hw, t = self._images(odd_weight=1.0, feat_weight=0.0)
        m, _ = oa.split_metrics(Hs, Hw, t, n_boot=20)
        self.assertGreater(m['cos'], 0.9)
        self.assertGreater(m['top1_odd_std'], 0.9)
        self.assertGreater(m['top1_odd_swap'], 0.9)

    def test_retained(self):
        self.assertAlmostEqual(oe.retained(-1.0, -5.0, -1.0), 1.0)
        self.assertAlmostEqual(oe.retained(-5.0, -5.0, -1.0), 0.0)
        self.assertTrue(np.isnan(oe.retained(0, 1, 1)))


if __name__ == '__main__':
    unittest.main()
