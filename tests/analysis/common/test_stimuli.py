"""Invariants of the shared synthetic stimulus generator."""
import unittest

import numpy as np

from analysis_tools.common.stimuli import (
    Change, NONE, RingSpec, StimulusSpec, TEMPLATES, changed_pixels, generate, generate_ring, min_ring_radius,
    render_instance, valid_changes,
)

# PNG hashes recorded with stimuli-v1 before the ring layout was added; grid specs must keep them.
PINNED_GRID_HASHES = [
    (StimulusSpec('person', 3, 3, Change('gap', 4, 1), target=(1, 2), seed=3),
     '3caeb45a8a570cdd842718d85c121dde358b1a160579b0386a3ed5bd1d395c52'),
    (StimulusSpec('arrow', 20, 20, Change('rotate', 2), target=(5, 17), marker=(2, 2), seed=7),
     'ea0b02094a36b75d2901b0eeb073c936ed06a6cfe745245b55abbdbed6633a9b'),
    (StimulusSpec('ring', 6, 6, Change('color', 3), target=(0, 0), cell=128, seed=1),
     '2c4d73586ca2659e0ea7fe11575633a7cdb73c84135f63f94a75bcdb9ac8640e'),
    (StimulusSpec('square', 24, 24, Change('gap', 1, 2), target=(23, 4), cell=32, seed=2),
     'bff479ba51b8742c8c1afc561142ec147530e95e754bbe51effe623d2e12f1e3'),
    (StimulusSpec('person', 4, 8, Change('color', 4), target=(3, 7), jitter=8, seed=5),
     'b934b06a7eec7e12250c99219e8a9b3681198a1c35fc4594c716e11934095386'),
    (StimulusSpec('house', 2, 2, Change('spur', 3, 0), target=(0, 1), phase=16, swap=True, seed=9),
     'fa4a232f02a1d08427f465bc929708e159827842ba212eef7b4521ea382da066'),
]
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

    def test_grid_images_are_unchanged(self):
        for spec, digest in PINNED_GRID_HASHES:
            self.assertEqual(generate(spec)[1]['sha256'], digest, spec)

    def test_ring_geometry(self):
        for n, radius in [(8, 128), (8, 352), (12, 352), (16, 352), (24, 352)]:
            image, meta = generate_ring(RingSpec('person', n, radius, 768, Change('gap', 4, 1), target=n // 3))
            self.assertEqual(image.size, (768, 768))
            boxes = [i['cell_box'] for i in meta['instances']]
            for a in range(n):
                self.assertTrue(0 <= boxes[a][0] and boxes[a][2] <= 768 and 0 <= boxes[a][1] and boxes[a][3] <= 768)
                for b in range(a + 1, n):   # cells are pairwise disjoint
                    self.assertTrue(boxes[a][2] <= boxes[b][0] or boxes[b][2] <= boxes[a][0]
                                    or boxes[a][3] <= boxes[b][1] or boxes[b][3] <= boxes[a][1])
            for inst in meta['instances']:   # cell centres lie on the circle up to rounding
                dx, dy = inst['cell_center'][0] - 384, inst['cell_center'][1] - 384
                self.assertLess(abs((dx * dx + dy * dy) ** 0.5 - radius), 1.0)
            top, right = meta['instances'][0], meta['instances'][n // 4]
            self.assertEqual(top['cell_center'], [384.0, 384.0 - radius])      # index 0 at 12 o'clock
            self.assertEqual(right['cell_center'], [384.0 + radius, 384.0])    # clockwise order
            self.assertEqual([i['is_target'] for i in meta['instances']].index(True), n // 3)

    def test_ring_target_is_the_only_changed_instance(self):
        spec = RingSpec('arrow', 12, 352, 768, Change('rotate', 3), target=5)
        changed, _ = generate_ring(spec)
        plain, meta = generate_ring(RingSpec('arrow', 12, 352, 768))
        diff = (np.asarray(changed) != np.asarray(plain)).any(-1)
        x0, y0, x1, y1 = meta['instances'][5]['cell_box']
        inside = np.zeros_like(diff)
        inside[y0:y1, x0:x1] = True
        self.assertTrue(diff.any())
        self.assertFalse((diff & ~inside).any())

    def test_invalid_rings_are_rejected(self):
        self.assertEqual(min_ring_radius(12), 192)
        with self.assertRaises(ValueError):
            RingSpec('person', 12, 160, 768)          # overlapping cells
        with self.assertRaises(ValueError):
            RingSpec('person', 8, 352, 704)           # cells beyond the canvas
        with self.assertRaises(ValueError):
            RingSpec('person', 8, 100, 768)           # radius not token-aligned
        with self.assertRaises(ValueError):
            RingSpec('ring', 8, 128, 320, Change('rotate', 1), target=0)

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
