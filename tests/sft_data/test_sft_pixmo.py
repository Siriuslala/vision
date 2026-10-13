"""Counting geometry and source-integrity regression checks."""
import hashlib
import io
import json
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from data.sft.datasets.pixmo import counting_candidate, group_instances, points_of, prepare


class PixmoTests(unittest.TestCase):
    def test_exclusive_groups_and_count_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (200, 100), "white").save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            row = {"image_sha256": digest, "image_url": "fixture", "label": "squares", "count": 3,
                   "points": {"x": [15., 35., 175.], "y": [15., 15., 75.]}}
            audit = {"source_id": "test:0", "image_sha256": digest,
                     "instance_boxes_reviewed": True, "complete_instance_set_reviewed": True,
                     "method": "geometry", "config_id": "fixture-v1",
                     "instances": [{"instance_id": "point_2", "box": [170., 70., 180., 80.]},
                                   {"instance_id": "point_0", "box": [10., 10., 20., 20.]},
                                   {"instance_id": "point_1", "box": [30., 10., 40., 20.]}]}
            candidate = counting_candidate(row, "test:0", "test", path, audit, "audits/test.json")
            counting = candidate["metadata"]["counting"]
            self.assertEqual(counting["instance_order"], ["point_0", "point_1", "point_2"])
            self.assertEqual([g["local_count"] for g in counting["groups"]], [2, 1])
            self.assertEqual(counting["groups"][-1]["cumulative_count"], 3)
            self.assertEqual(candidate["events"][-1]["text"], "Adding the counts gives 2 + 1 = 3.")
            row["image_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                counting_candidate(row, "test:0", "test", path, audit, "audits/test.json")

    def test_changed_image_requires_explicit_reannotation_queue(self):
        buffer = io.BytesIO()
        Image.new("RGB", (20, 10), "white").save(buffer, format="PNG")
        raw = buffer.getvalue()
        row = {"image_sha256": "0"*64, "image_url": "fixture", "label": "squares", "count": 1,
               "points": {"x": [5.], "y": [5.]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("data.sft.datasets.pixmo.rows", return_value=iter([("train:0", row, "revision")])), patch("data.sft.datasets.pixmo.urlopen", return_value=io.BytesIO(raw)):
                with self.assertRaises(ValueError):
                    prepare(root, root/"strict", 1, True)
            with patch("data.sft.datasets.pixmo.rows", return_value=iter([("train:0", row, "revision")])), patch("data.sft.datasets.pixmo.urlopen", return_value=io.BytesIO(raw)):
                prepare(root, root/"reannotate", 1, True, allow_image_reannotation=True)
            audit = json.loads((root/"reannotate/audits/train_0.json").read_text())
            self.assertEqual(audit["source_image_sha256"], "0"*64)
            self.assertEqual(audit["image_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertTrue(audit["image_reannotation_required"])
            self.assertFalse(audit["complete_instance_set_reviewed"])
            self.assertEqual(row["image_sha256"], "0"*64)

    def test_conflicting_points_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            points_of({"points": {"x": [1., 1.], "y": [2., 2.]}, "count": 2})

    def test_intersecting_independent_groups_rejected(self):
        instances = [{"instance_id": "a", "box": [0., 0., 20., 20.]},
                     {"instance_id": "b", "box": [10., 10., 30., 30.]}]
        with self.assertRaisesRegex(ValueError, "intersect"):
            group_instances(instances, max_group=1)


if __name__ == "__main__":
    unittest.main()
