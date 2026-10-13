"""Point proposal tests use synthetic, regular objects, not natural-image claims."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw
from data.sft.tools.point_windows import Sam3LocalBackend, file_sha256, geometry_candidates, mask_candidate, propose, select_mask_candidate


class PointWindowTests(unittest.TestCase):
    def fixture(self):
        image = Image.new("RGB", (180, 80), "white")
        draw = ImageDraw.Draw(image)
        points = [[30., 40.], [90., 40.], [150., 40.]]
        for x, y in points:
            draw.rectangle((x-8, y-8, x+8, y+8), fill="black")
        return image, points

    def test_geometry_refinement_and_half_open_boundaries(self):
        image, points = self.fixture()
        proposals = geometry_candidates(image, points, regular_compact=True)
        self.assertEqual(proposals[1]["local_scale"], 60)
        candidate = proposals[1]["candidates"][1]
        self.assertEqual(candidate["refinement"]["box"], [82, 32, 99, 49])
        self.assertTrue(candidate["stable_across_thresholds"])
        self.assertEqual(candidate["other_point_indices_in_box"], [])
        with self.assertRaisesRegex(ValueError, "regular_compact"):
            geometry_candidates(image, points)
        isolated = geometry_candidates(image, [points[0]], regular_compact=True)
        self.assertEqual(isolated[0]["status"], "isolated_requires_segmentation")

    def test_proposal_files_remain_unreviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audits, cache = root / "audits", root / "images"
            audits.mkdir(); cache.mkdir()
            image, points = self.fixture()
            image_path = root / "fixture.png"
            image.save(image_path)
            digest = file_sha256(image_path)
            image_path.rename(cache / digest)
            audit = {"source_id": "fixture", "image_sha256": digest,
                     "points": [{"instance_id": f"point_{i}", "point": point} for i, point in enumerate(points)]}
            (audits / "fixture.json").write_text(json.dumps(audit))
            propose(audits, cache, root / "output", method="geometry", regular_compact=True)
            result = json.loads((root / "output/fixture.json").read_text())
            self.assertFalse(result["instance_boxes_reviewed"])
            self.assertFalse(result["complete_instance_set_reviewed"])
            self.assertEqual(result["instances"], [])
            self.assertEqual(len(result["candidate_instances"]), 3)
            self.assertEqual(json.loads((audits / "fixture.json").read_text()), audit)

    def test_automatic_selection_requires_consistent_prompts(self):
        def record(box, score=0.9):
            return {"candidates": [{"status": "pending", "box": box, "score": score}]}
        self.assertEqual(select_mask_candidate([record([0, 0, 10, 10])])["status"], "requires_semantic_verification")
        self.assertEqual(select_mask_candidate([record([0, 0, 10, 10]), record([8, 8, 18, 18])])["status"], "rejected")
        self.assertEqual(select_mask_candidate([record([0, 0, 10, 10], 0.3)])["status"], "rejected")

    def test_sam_missing_checkpoint_is_explicit(self):
        with self.assertRaises(FileNotFoundError):
            Sam3LocalBackend(Path("/__missing_sam3_checkpoint__"))

    def test_mask_point_constraints_and_single_pixel_box(self):
        import torch
        mask = torch.zeros((10, 10), dtype=torch.bool)
        mask[2, 3] = True
        self.assertEqual(mask_candidate(mask, [3., 2.], [], 0.9)["box"], [3, 2, 4, 3])
        self.assertEqual(mask_candidate(mask, [3., 2.], [[3., 2.]], 0.9)["status"], "includes_other_instance_point")
        self.assertEqual(mask_candidate(mask, [1., 1.], [], 0.9)["status"], "misses_positive")


if __name__ == "__main__":
    unittest.main()
