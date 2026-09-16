"""Checks for source coordinates and language/geometry separation."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from data.sft.datasets.visual_cot import candidate_from_record, iter_candidates, iter_tasks, _relative


class VisualCotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image_dir = self.root / "cot_image_data" / "gqa"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (500, 375)).save(image_dir / "test.jpg")
        self.record = {"dataset": "gqa", "split": "train", "question": "What is on the table?",
                       "answer": "cup", "image": "test.jpg", "width": 500, "height": 375,
                       "bboxs": [[10, 20, 100, 90], [200, 40, 250, 80]],
                       "reasoning": [{"argument": "cup (12345)"}],
                       "thought": "Inspect the object on the table."}

    def test_pixel_union_and_source_thought_order(self):
        result = candidate_from_record(self.record, self.root, "fixture.jsonl", 1)
        self.assertEqual(result["events"][0]["box"], [10, 20, 250, 90])
        self.assertEqual(result["events"][1]["text"], self.record["thought"])
        self.assertNotIn("12345", json.dumps(result["events"]))
        self.assertFalse(result["visual_only_safe"])
        from data.sft.core.common import build_records, validate_record
        record = build_records(result, "visual_cot", self.root / "exported", self.root)[0]
        validate_record(record, self.root)
        self.assertEqual(record["provenance"]["annotation_origin"], "derived")

    def test_teacher_task_preserves_individual_boxes_without_assuming_a_trajectory(self):
        meta = self.root / "metadata"
        meta.mkdir()
        (meta / "gqa_cot_train.jsonl").write_text(json.dumps(self.record) + "\n")
        task = list(iter_tasks(self.root, 1))[0]
        self.assertEqual(task["source_boxes"], self.record["bboxs"])
        self.assertNotIn("events", task)
        self.assertNotIn("visual_only_safe", task)
        self.assertEqual(task["metadata"]["source_thought"], self.record["thought"])
        self.assertEqual(task["metadata"]["source_box_order"], "unverified")

    def test_source_out_of_bounds_rejected_without_clipping(self):
        self.record["bboxs"] = [[0, 100, 500, 376]]
        with self.assertRaisesRegex(ValueError, "out_of_bounds"):
            candidate_from_record(self.record, self.root, "fixture.jsonl", 1)

    def test_no_thought_and_rejection_limit(self):
        self.record.pop("thought")
        result = candidate_from_record(self.record, self.root, "fixture.jsonl", 1)
        self.assertEqual(len(result["events"]), 1)
        self.assertTrue(result["visual_only_safe"])
        meta = self.root / "metadata"
        meta.mkdir()
        (meta / "gqa_cot_train.jsonl").write_text("{invalid\n" + json.dumps(self.record) + "\n")
        records = list(iter_candidates(self.root, 1))
        self.assertEqual(len(records), 1)
        self.assertIn("reject_reason", records[0])

    def test_held_out_and_path_traversal_rejected(self):
        self.record["split"] = "val"
        with self.assertRaisesRegex(ValueError, "held_out"):
            candidate_from_record(self.record, self.root, "fixture.jsonl", 1)
        for path in ["../../bad.jpg", "/tmp/bad.jpg", "foo\\bar"]:
            with self.assertRaises(ValueError):
                _relative(path)


if __name__ == "__main__":
    unittest.main()
