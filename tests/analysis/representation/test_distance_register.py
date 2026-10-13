"""Numerical checks for distance controls and edit construction."""
import unittest
from copy import deepcopy
import numpy as np
import torch
from analysis_tools.distance_register import paired_maps, single_scores, synthetic, unique_top, validate_checkpoint, external_peak_record


class DistanceRegisterTests(unittest.TestCase):
    def test_metrics_separate_norm_and_direction(self):
        a = torch.tensor([[[1., 2., 3.], [2., -1., 1.]]])
        maps = paired_maps(a, a*2)
        self.assertTrue(torch.allclose(maps['l2'], maps['rms']*np.sqrt(3)))
        self.assertTrue(torch.all(maps['cosine'] < 1e-6))
        self.assertTrue(torch.all(maps['l2'] > 0))

    def test_repeated_cells_tie_and_changed_cell_selected(self):
        grid = torch.ones(12, 12, 3)
        self.assertIsNone(unique_top(single_scores(grid)['rms_median']))
        grid[6:8, 6:8, 0] = -1
        scores = single_scores(grid)
        for name in ['rms_median', 'cosine_median', 'unit_token_rms', 'cosine_pairwise_mean']:
            self.assertEqual(unique_top(scores[name]),21)

    def test_local_edit_stays_in_target_cell(self):
        for family in ['person','ring','square']:
            for shift in [0,4]:
                baseline = np.asarray(synthetic(family,shift))
                for target in [0,21,35]:
                    changed = np.any(baseline != np.asarray(synthetic(family,shift,target)),axis=-1)
                    y,x = np.where(changed)
                    self.assertGreater(len(x),0)
                    self.assertTrue(np.all(x//64 == target%6))
                    self.assertTrue(np.all(y//64 == target//6))

    def test_checkpoint_rejects_legacy_and_incompatible_metadata(self):
        expected = {'schema_version': 2, 'model': 'test-model', 'snapshot': 'snapshot-a',
                    'dtype': 'float32', 'source_sha256': 'source-a',
                    'families': ['person'], 'shifts': [0], 'stages': ['patch_embed'], 'targets': [0]}
        checkpoint = {'metadata': expected, 'normals': [{'case': 'person_shift0', 'stage': 'patch_embed'}],
                      'records': [{'case': 'person_shift0', 'stage': 'patch_embed', 'target': 0}]}
        validate_checkpoint(checkpoint, expected)
        legacy = deepcopy(checkpoint)
        del legacy['metadata']
        with self.assertRaisesRegex(ValueError, 'metadata'):
            validate_checkpoint(legacy, expected)
        for field in ['model', 'snapshot', 'dtype', 'source_sha256']:
            altered = deepcopy(checkpoint)
            altered['metadata'][field] = 'incompatible'
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                validate_checkpoint(altered, expected)
        incomplete = deepcopy(checkpoint)
        incomplete['records'] = []
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_checkpoint(incomplete, expected)

    def test_external_peak_ties_are_failures(self):
        tied = external_peak_record(torch.tensor([[3., 3.], [0., 0.]]), (0, 0))
        self.assertFalse(tied['unique_peak'])
        self.assertFalse(tied['correct'])
        unique = external_peak_record(torch.tensor([[3., 4.], [0., 0.]]), (0, 1))
        self.assertTrue(unique['unique_peak'])
        self.assertTrue(unique['correct'])
