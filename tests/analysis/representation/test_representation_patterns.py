"""Test support matching and the geometry of controlled layout changes."""
import unittest
import numpy as np
import torch
from analysis_tools.representation.representation_patterns import layouts, render, region_indices, pairwise_scores, readouts, chosen_targets


class RepresentationPatternTests(unittest.TestCase):
    def test_edits_are_local_and_windows_align(self):
        for layout in layouts():
            for family in ['person', 'ring', 'square']:
                normal = np.asarray(render(layout, family))
                target = len(layout['boxes'])//2
                changed = np.any(normal != np.asarray(render(layout,family,target)),axis=-1)
                y,x = np.where(changed)
                self.assertGreater(len(x),0)
                a,b,c,d = layout['boxes'][target]
                self.assertTrue(((x>=a)&(x<c)&(y>=b)&(y<d)).all())
            for stride in [16,32]:
                ids = region_indices((layout['size']//stride,)*2,layout['size'],layout['boxes'])
                self.assertEqual(ids.shape[0],len(layout['boxes']))
                self.assertEqual(len(torch.unique(ids)),ids.numel())

    def test_pair_distance_uses_common_support(self):
        x = torch.tensor([[[1.,0.],[100.,0.]], [[1.,0.],[-100.,0.]], [[3.,0.],[9.,0.]]])
        keep = torch.tensor([[True,False],[True,True],[True,False]])
        scores = pairwise_scores(x,keep)
        self.assertTrue(torch.allclose(scores['rms'],torch.tensor([2**.5/2,2**.5/2,2**.5])))
        self.assertTrue(torch.allclose(scores['cosine'],torch.zeros(3)))

    def test_empty_objects_are_not_selected(self):
        x = torch.ones(3,2,4)
        keep = torch.tensor([[False,False],[True,True],[True,True]])
        self.assertTrue(torch.isneginf(pairwise_scores(x,keep)['rms'][0]))

    def test_exclusion_has_matched_count_and_uses_no_target(self):
        grid = torch.randn(4,4,6)
        ids = torch.arange(16).reshape(4,4)
        scores,support = readouts(grid,ids)
        self.assertEqual(support['exclude_high5']['excluded_image_tokens'],1)
        for seed in range(3):
            self.assertEqual(support[f'exclude_random5_seed{seed}']['excluded_image_tokens'],1)
        self.assertEqual(len(scores),10)

    def test_target_sampling_covers_off_diagonal_positions(self):
        for side, expected in [(6,12),(20,24)]:
            targets = chosen_targets(side*side)
            self.assertEqual(len(targets),expected)
            self.assertGreater(sum(t//side != t%side for t in targets),expected//2)
            self.assertTrue({0,side-1,side*(side-1),side*side-1}.issubset(targets))
            self.assertEqual(targets,chosen_targets(side*side))
        self.assertTrue(set(chosen_targets(400,12)).issubset(chosen_targets(400)))
