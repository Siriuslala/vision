"""Geometry, chronology, and source-code rejection checks for Thyme conversion."""
import base64
import io
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from PIL import Image

from data.sft.datasets.thyme import CropParser, convert_row, task_from_row, source_family, iter_tasks


def encoded(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class ThymeTest(unittest.TestCase):
    def test_per_shard_sampling_counts_rejections_and_preserves_global_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            for name in ("2round-00003.parquet", "2round-00012.parquet"):
                pq.write_table(pa.table({"valid": [False, True, True]}), root / "data" / name)
            def convert(row, source_id, revision, split):
                if not row["valid"]:
                    raise ValueError("fixture rejected row")
                return {"source_id": source_id}
            with patch("data.sft.datasets.thyme.task_from_row", side_effect=convert):
                sampled = list(iter_tasks(root, 0, per_shard_limit=2))
                self.assertEqual([x["source_id"] for x in sampled], [
                    "2round-00003.parquet:0", "2round-00003.parquet:1",
                    "2round-00012.parquet:0", "2round-00012.parquet:1"])
                self.assertIn("reject_reason", sampled[0])
                self.assertEqual(len(list(iter_tasks(root, 3, per_shard_limit=2))), 3)
                self.assertEqual(len(list(iter_tasks(root, 0))), 6)
            with self.assertRaisesRegex(ValueError, "per_shard_limit"):
                list(iter_tasks(root, 0, per_shard_limit=-1))

    def test_source_image_namespace_preserves_training_boundary(self):
        self.assertEqual(source_family("/source/coco2017/train2017/image.jpg"), "coco2017_train")
        with self.assertRaises(ValueError):
            source_family("/source/coco2017/val2017/image.jpg")

    def test_nested_crop_uses_original_coordinates_and_reference(self):
        parser = CropParser("original.jpg", (100, 50))
        first = parser.parse('p="original.jpg"\na=Image.open(p)\nb=a.crop((10,10,90,40))\nb.save("out.jpg")')
        second = parser.parse('c=b.resize((160,60))\nd=c.crop((20,10,100,50))\nd.save("out2.jpg")')
        self.assertEqual(first.box, (10, 10, 90, 40))
        self.assertEqual(second.box, (20, 15, 60, 35))
        self.assertEqual(second.reference, 1)

    def test_arbitrary_execution_and_augmentation_rejected(self):
        for code in ['import subprocess\nsubprocess.run("x")', 'x=__import__("os")',
                     'a=Image.open("other.jpg")', 'a=Image.open("original.jpg")\nb=a.rotate(90)',
                     'a=Image.open("original.jpg")\nb=a.crop((-1,0,10,10))']:
            with self.subTest(code=code), self.assertRaises(ValueError):
                CropParser("original.jpg", (100, 50)).parse(code)

    def test_true_interleaving_and_answer_boundary(self):
        original = Image.new("RGB", (100, 50), "red")
        row = {"image": [encoded(original), encoded(original.crop((10, 5, 60, 30)))],
               "question": '<image>\nWhat color is the object?\n### User Image Path:** "/data/gqa/images/1.jpg"\n### User Image Size:** "100x50"',
               "response": '<think>Inspect the object.\n<code>\n```python\np="/data/gqa/images/1.jpg"\na=Image.open(p)\nb=a.crop((10,5,60,30))\nb.save("out.jpg")\n```\n</code><sandbox_output><image></sandbox_output>It is red.</think><answer>red</answer>'}
        result = convert_row(row, "test:1", "revision", "2round")
        self.assertEqual([x["type"] for x in result["events"]], ["reasoning_text", "visual_read", "reasoning_text"])
        self.assertEqual(result["events"][1]["box"], [10, 5, 60, 30])
        self.assertEqual(result["query"], "What color is the object?")
        self.assertEqual(result["answer"], {"type": "text", "value": "red"})
        self.assertFalse(result["visual_only_safe"])
        task = task_from_row(row, "test:1", "revision", "2round")
        self.assertNotIn("events", task)
        self.assertEqual(task["source_events"], result["events"])
        self.assertEqual(task["metadata"]["source_correction"], "unverified")
        self.assertEqual(task["metadata"]["source_read_count"], 1)
        row["image"][1] = encoded(Image.new("RGB", (50, 25), "blue"))
        task = task_from_row(row, "test:1", "revision", "2round")
        self.assertEqual(task["answer"], result["answer"])
        self.assertNotIn("source_events", task)
        self.assertEqual(task["metadata"]["source_trajectory_status"], "crop_conversion_failed")
        with self.assertRaisesRegex(ValueError, "derived image mismatch"):
            convert_row(row, "test:1", "revision", "2round")


if __name__ == "__main__":
    unittest.main()
