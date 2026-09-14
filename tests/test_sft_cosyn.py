"""Protect CoSyn QA identity and the separation of source hints from evidence."""
import io
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from data.sft.core.annotations import source_key
from data.sft.core.common import build_records
from data.sft.datasets.cosyn import iter_candidates, iter_tasks, qa_items


class CoSynTest(unittest.TestCase):
    def test_misaligned_parallel_fields_fail(self):
        with self.assertRaisesRegex(ValueError, "lengths disagree"):
            qa_items({"question": ["Q"], "answer": [], "explanation": ["E"]})

    def test_bounded_tasks_annotation_binding_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "chart").mkdir(parents=True)
            image = io.BytesIO()
            Image.new("RGB", (100, 50), "white").save(image, format="PNG")
            row = {"id": "figure-1", "image": {"bytes": image.getvalue(), "path": "unused"},
                   "qa_pairs": {"question": ["What value?", "What title?"],
                                "answer": ["42", "A"], "explanation": ["Secret CSV gives 42", "At top"]},
                   "data": "value\n42", "code": "raise RuntimeError('must not execute')", "metadata": {"figure_type": "Bar"}}
            pq.write_table(pa.Table.from_pylist([row]), source / "chart/train-00000.parquet")
            tasks = list(iter_tasks(source, 1))
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertEqual(task["source_id"], "chart/figure-1/qa/0")
            self.assertNotIn("events", task)
            self.assertEqual(task["metadata"]["source_explanation"], "Secret CSV gives 42")
            self.assertEqual(list(iter_candidates(source, 1))[0]["reject_reason"], "missing_region_annotation")
            annotation = {"source_key": source_key(task["source_id"], task["query"], task["answer"], task["image"]),
                          "status": "reviewed", "reviewer": "test-reviewer", "visual_only_safe": True,
                          "visual_only_basis": "All required evidence remains in the same observed region.",
                          "events": [{"type": "visual_read", "box": [20, 10, 80, 40], "reference_region_id": 0},
                                     {"type": "reasoning_text", "text": "The visible label reads 42."}]}
            annotations = root / "annotations.jsonl"
            annotations.write_text(json.dumps(annotation) + "\n")
            candidate = next(iter_candidates(source, 1, annotations))
            records = build_records(candidate, "cosyn", root / "images", root)
            self.assertEqual(len(records), 2)
            interleaved = next(record for record in records if record["reasoning_mode"] == "interleaved")
            self.assertEqual(interleaved["trajectory"][0]["region"], {"center": [0.5, 0.25], "size": [0.6, 0.3]})
            self.assertNotIn("Secret CSV", json.dumps(interleaved["trajectory"]))
            annotation["source_key"] = "changed"
            annotations.write_text(json.dumps(annotation) + "\n")
            self.assertEqual(next(iter_candidates(source, 1, annotations))["reject_reason"], "missing_region_annotation")


if __name__ == "__main__":
    unittest.main()
