"""Check coordinate labels and the numerical probe, without model downloads."""
import unittest
import numpy as np
from analysis_tools.position.position_probe import coordinates, fit_ridge, metrics, make_image


class PositionProbeTests(unittest.TestCase):
    def test_rectangular_coordinates_are_xy_row_major(self):
        values = coordinates(2, 3)
        np.testing.assert_allclose(values[[0,2,3,5]], [[1/6,1/4],[5/6,1/4],[1/6,3/4],[5/6,3/4]])

    def test_intercept_and_generalization_of_linear_probe(self):
        rng = np.random.default_rng(14)
        x = rng.normal(size=(100, 8))
        weights = rng.normal(size=(8, 2))
        target = x @ weights + np.array([3., -2.])
        w,b = fit_ridge(x[:70], target[:70], 1e-8)
        self.assertLess(metrics(x[70:]@w+b,target[70:])['mean_coordinate_mae'], 1e-7)

    def test_shift_is_reproducible_and_changes_pixels(self):
        a = np.asarray(make_image((256,256),'circle',17))
        b = np.asarray(make_image((256,256),'circle',17,True))
        c = np.asarray(make_image((256,256),'circle',17,True))
        self.assertTrue(np.any(a != b))
        np.testing.assert_array_equal(b,c)


if __name__ == '__main__':
    unittest.main()
