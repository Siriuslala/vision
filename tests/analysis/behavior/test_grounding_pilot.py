"""Check geometry and conservative scoring used in the grounding pilot."""
import json
import unittest

import numpy as np

from analysis_tools.behavior.grounding_pilot import make_grid, score_point


class GroundingPilotTests(unittest.TestCase):
    def test_spacing_changes_location_without_changing_canvas(self):
        regular, centers = make_grid(20, False, (4, 5), 'normal')
        uneven, moved = make_grid(20, True, (4, 5), 'normal')
        self.assertEqual(regular.size, uneven.size)
        self.assertEqual(centers[85], [352., 288.5])
        self.assertEqual(moved[85], [264., 360.5])

    def test_gap_changes_only_sixteen_pixels(self):
        for n in (6, 20):
            for uneven in (False, True):
                normal, _ = make_grid(n, uneven, (4, 5), 'normal')
                changed, _ = make_grid(n, uneven, (4, 5), 'gap')
                self.assertEqual(np.any(np.array(normal) != np.array(changed), axis=-1).sum(), 16)

    def test_point_identity_and_precision_are_distinct(self):
        centers = [[100., 100.], [200., 100.]]
        exact = score_point('[{"point_2d":[100,100]}]', centers, 0, (1000, 1000))
        near = score_point('[{"point_2d":[120,100]}]', centers, 0, (1000, 1000))
        self.assertTrue(exact['precise'])
        self.assertTrue(near['identity_correct'])
        self.assertFalse(near['precise'])

    def test_tie_is_not_correct_identity(self):
        result = score_point('[{"point_2d":[150,100]}]', [[100.,100.],[200.,100.]], 0, (1000,1000))
        self.assertFalse(result['identity_correct'])

    def test_malformed_or_ambiguous_points_are_not_silently_accepted(self):
        for value in [[], [{'point_2d':[1,2]}, {'point_2d':[3,4]}],
                      [{'point_2d':[True,2]}], [{'point_2d':[1001,2]}],
                      [{'point_2d':[float('nan'),2]}], {'point':[1,2]}]:
            self.assertFalse(score_point(json.dumps(value), [[1.,2.]], 0, (1000,1000))['valid'])


if __name__ == '__main__':
    unittest.main()
