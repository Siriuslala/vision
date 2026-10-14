"""Invariants of the shared synthetic stimulus generator."""
import unittest

import numpy as np

from analysis_tools.common.stimuli import (
    Change, NONE, StimulusSpec, TEMPLATES, changed_pixels, generate, render_instance, valid_changes,
)
from analysis_tools.representation.visual_detail import figure_image


def dark(a):
    return (a < 255).any(-1)


class StimuliTests(unittest.TestCase):
    def test_person_matches_historical_generator(self):
        historical = np.asarray(figure_image(None, 0))[:64, :64]
        self.assertTrue((historical == render_instance('person')).all())

    def test_every_valid_change_alters_the_template(self):
        for template in TEMPLATES:
            for change in valid_changes(template):
                self.assertGreater(changed_pixels(template, change), 0, (template, change))

    def test_gaps_only_remove_pixels_and_spurs_only_add_pixels(self):
        for template in TEMPLATES:
            base = dark(render_instance(template))
            for site in (0, 1, 2):
                for level in (1, 2, 3, 4):
                    gap = dark(render_instance(template, Change('gap', level, site)))
                    spur = dark(render_instance(template, Change('spur', level, site)))
                    self.assertFalse((gap & ~base).any())
                    self.assertFalse((base & ~spur).any())

    def test_stroke_changes_grow_with_level(self):
        for template in TEMPLATES:
            for kind in ('gap', 'spur'):
                for site in (0, 1, 2):
                    counts = [changed_pixels(template, Change(kind, level, site)) for level in (1, 2, 3, 4)]
                    self.assertEqual(counts, sorted(counts), (template, kind, site))

    def test_canvas_is_token_aligned_and_never_below_processor_minimum(self):
        for rows, cols, cell in [(1, 1, 64), (1, 2, 64), (3, 3, 64), (6, 6, 64), (20, 20, 64), (24, 24, 32), (4, 8, 64)]:
            for phase in (0, 16):
                _, meta = generate(StimulusSpec('person', rows, cols, cell=cell, phase=phase))
                width, height = meta['canvas']
                self.assertEqual((width % 32, height % 32), (0, 0))
                self.assertGreaterEqual(width * height, 65536)
                for inst in meta['instances']:
                    x0, y0, x1, y1 = inst['cell_box']
                    self.assertTrue(0 <= x0 and 0 <= y0 and x1 <= width and y1 <= height)
                    if not phase:
                        self.assertEqual((x0 % 32, y0 % 32), (0, 0))

    def test_swap_inverts_which_instances_carry_the_change(self):
        change = Change('gap', 3, 0)
        _, plain = generate(StimulusSpec('ring', 6, 6, change, target=(2, 4)))
        _, swapped = generate(StimulusSpec('ring', 6, 6, change, target=(2, 4), swap=True))
        changed = lambda meta: [i['index'] for i in meta['instances'] if i['change']['kind'] != 'none']
        self.assertEqual(changed(plain), [16])
        self.assertEqual(len(changed(swapped)), 35)
        self.assertNotIn(16, changed(swapped))
        self.assertEqual([i['cell_box'] for i in plain['instances']], [i['cell_box'] for i in swapped['instances']])

    def test_marker_does_not_touch_instance_pixels(self):
        spec = StimulusSpec('person', 6, 6, Change('gap', 4, 0), target=(3, 3), marker=(3, 3))
        marked, _ = generate(spec)
        plain, _ = generate(StimulusSpec('person', 6, 6, Change('gap', 4, 0), target=(3, 3)))
        diff = (np.asarray(marked) != np.asarray(plain)).any(-1)
        self.assertTrue(diff.any())
        self.assertTrue((np.asarray(marked)[diff] == [0, 0, 255]).all())
        self.assertTrue((np.asarray(plain)[diff] == 255).all())

    def test_generation_is_deterministic(self):
        spec = StimulusSpec('arrow', 6, 6, Change('color', 4), target=(1, 1), jitter=8, seed=5)
        self.assertEqual(generate(spec)[1]['sha256'], generate(spec)[1]['sha256'])

    def test_invalid_specs_are_rejected(self):
        with self.assertRaises(ValueError):
            StimulusSpec('ring', 6, 6, Change('rotate', 1), target=(0, 0))
        with self.assertRaises(ValueError):
            StimulusSpec('person', 6, 6, Change('gap', 1), target=None)
        with self.assertRaises(ValueError):
            StimulusSpec('person', 6, 6, cell=48)
        with self.assertRaises(ValueError):
            Change('gap', 5)
        with self.assertRaises(ValueError):
            StimulusSpec('person', 6, 6, NONE, marker=(0, 0), jitter=4)


if __name__ == '__main__':
    unittest.main()
