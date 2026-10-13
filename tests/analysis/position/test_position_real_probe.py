import numpy as np
import unittest
from analysis_tools.position_real_probe import patch_grid, labels


class PositionRealProbeTests(unittest.TestCase):
  def test_patch_grid_matches_spatial_merge_groups(self):
    got=patch_grid(4,4)
    np.testing.assert_array_equal(got[:8],[[0,0],[1,0],[0,1],[1,1],[2,0],[3,0],[2,1],[3,1]])
    assert len({tuple(x) for x in got})==16
    with self.assertRaises(ValueError): patch_grid(3,4)


  def test_patch_and_merged_labels_use_correct_center_and_units(self):
    grid,image=labels((4,8),np.array([0,3,4,7]),False)
    np.testing.assert_array_equal(grid,[[0,0],[3,0],[0,1],[3,1]])
    np.testing.assert_allclose(image,[[125,250],[875,250],[125,750],[875,750]])
    grid,image=labels((4,8),np.array([0,3,4]),True)
    np.testing.assert_array_equal(grid,[[0,0],[1,1],[2,0]])
    np.testing.assert_allclose(image,[[62.5,125],[187.5,375],[312.5,125]])

  def test_patch_labels_agree_with_model_rope_grid(self):
    import torch
    from transformers.vision_utils import get_vision_position_ids
    for height,width in [(4,8),(16,32),(32,18)]:
        rope=get_vision_position_ids(torch.tensor([[1,height,width]]),2).numpy()
        np.testing.assert_array_equal(patch_grid(height,width),rope[:,[1,0]])
