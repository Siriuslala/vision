"""Unit tests for the experiment-4.3 DeepStack override and case selection (no model weights needed)."""
from types import SimpleNamespace
import unittest

import torch

from analysis_tools.deepstack import deepstack_ablation as da
from analysis_tools.deepstack.deepstack_ablation_analysis import select


def fake_rt(n_layers=3):
    visual = SimpleNamespace(deepstack_visual_indexes=list(range(n_layers)))
    return SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(visual=visual)))


class OverrideTest(unittest.TestCase):
    def setUp(self):
        self.embeds = [torch.full((5, 4), float(k + 1)) for k in range(3)]

    def test_full_is_identity(self):
        out = da.DeepstackOverride(fake_rt(), 'full').replace(self.embeds)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(out, self.embeds)))

    def test_off_and_single_layer(self):
        out = da.DeepstackOverride(fake_rt(), 'off').replace(self.embeds)
        self.assertTrue(all(float(o.abs().sum()) == 0 for o in out))
        out = da.DeepstackOverride(fake_rt(), 'off2').replace(self.embeds)
        self.assertEqual([float(o.sum()) for o in out], [20.0, 0.0, 60.0])

    def test_mean_broadcasts(self):
        means = [torch.arange(4.0) + k for k in range(3)]
        out = da.DeepstackOverride(fake_rt(), 'mean', means).replace(self.embeds)
        self.assertEqual(out[2].shape, (5, 4))
        self.assertTrue(torch.equal(out[2][3], means[2]))

    def test_blur_requires_prepared_features_once(self):
        ov = da.DeepstackOverride(fake_rt(), 'blur')
        with self.assertRaises(ValueError):
            ov.replace(self.embeds)
        ov.blur_features = [torch.zeros(5, 4)] * 3
        ov.replace(self.embeds)
        with self.assertRaises(ValueError):        # consumed by the first prefill
            ov.replace(self.embeds)
        ov.blur_features = [torch.zeros(6, 4)] * 3
        with self.assertRaises(ValueError):        # token count mismatch
            ov.replace(self.embeds)

    def test_layer_count_and_unknown_condition(self):
        with self.assertRaises(ValueError):
            da.DeepstackOverride(fake_rt(2), 'off').replace(self.embeds)
        with self.assertRaises(ValueError):
            da.DeepstackOverride(fake_rt(), 'zero')
        with self.assertRaises(ValueError):
            da.DeepstackOverride(fake_rt(), 'mean')

    def test_wrapper_only_touches_prefill(self):
        seen = []

        class LM:
            def forward(self, **kw):
                seen.append(kw.get('deepstack_visual_embeds'))
        rt = fake_rt()
        rt.model.model.language_model = LM()
        ov = da.DeepstackOverride(rt, 'off')
        with da.deepstack_override(rt, ov):
            rt.model.model.language_model.forward(deepstack_visual_embeds=self.embeds)
            rt.model.model.language_model.forward(deepstack_visual_embeds=None)
        self.assertEqual(ov.calls, 1)
        self.assertEqual(float(seen[0][0].sum()), 0.0)
        self.assertIsNone(seen[1])
        self.assertNotIn('forward', vars(rt.model.model.language_model))   # restored


class SelectionTest(unittest.TestCase):
    def test_select_decomp(self):
        cases = []
        for task in ('T1', 'T2'):
            for level in (1, 2, 3, 4):
                cases.append({'id': f'{task}{level}', 'stage': 'main', 'wording': 'primary', 'task': task,
                              'kind': 'color', 'level': level, 'grid': 3, 'format': 'yes_no'})
        for grid in (6, 10, 20):
            for fmt in ('point', 'bbox', 'row_column'):
                cases.append({'id': f'T6{grid}{fmt}', 'stage': 'main', 'wording': 'primary', 'task': 'T6',
                              'kind': 'color', 'level': 4, 'grid': grid, 'format': fmt})
        for r in range(da.T5_RECORDS_PER_CELL + 2):
            for fmt in ('point', 'bbox', 'row_column'):
                cases.append({'id': f'T5{r}{fmt}', 'stage': 'main', 'wording': 'primary', 'task': 'T5',
                              'kind': 'gap', 'level': 1, 'grid': 10, 'format': fmt, 'record_id': f'r{r}'})
        chosen = da._select_decomp(cases, seed=0)
        ids = {c['id'] for c in chosen}
        self.assertEqual({i for i in ids if i.startswith(('T1', 'T2'))}, {'T12', 'T13', 'T22', 'T23'})
        self.assertEqual(len([i for i in ids if i.startswith('T6')]), 6)
        t5 = [c for c in chosen if c['task'] == 'T5']
        self.assertEqual(len(t5), 3 * da.T5_RECORDS_PER_CELL)
        self.assertEqual(len({c['record_id'] for c in t5}), da.T5_RECORDS_PER_CELL)   # same images in all formats

    def test_select_metric_filters(self):
        rows = {'a': {'task': 'RefCOCO', 'format': 'bbox', 'stratum': 'ge10'},
                'b': {'task': 'RefCOCO', 'format': 'bbox', 'stratum': '1to5'},
                'c': {'task': 'T6', 'format': 'point', 'grid': 10}}
        self.assertEqual(select(rows, ('RefCOCO', 'bbox', None, 'correct', ('5to10', 'ge10'))), ['a'])
        self.assertEqual(select(rows, ('T6', 'point', (10, 20), 'hit', None)), ['c'])


if __name__ == '__main__':
    unittest.main()
