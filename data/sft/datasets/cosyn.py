"""Read CoSyn QA tasks and attach separately reviewed visual evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

from data.sft.core.annotations import apply_annotation, load_annotations
from data.sft.core.common import digest
from data.sft.core.parquet import iter_rows


TASK_TYPES = {"chart": "chart_qa", "table": "table_qa", "document": "document_qa",
              "chemical": "general_vqa", "circuit": "general_vqa",
              "diagram": "general_vqa", "math": "general_vqa", "music": "general_vqa",
              "graphic": "general_vqa", "nutrition": "document_qa"}


def qa_items(value: dict | list) -> list[dict]:
    """Accept Arrow's columnar sequence representation and equivalent row lists."""
    fields = ("question", "explanation", "answer")
    if isinstance(value, dict):
        if any(not isinstance(value.get(key), list) for key in fields):
            raise ValueError("qa_pairs must contain question/explanation/answer lists")
        if len({len(value[key]) for key in fields}) != 1:
            raise ValueError("QA parallel list lengths disagree")
        return [dict(zip(fields, values)) for values in zip(*(value[key] for key in fields))]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError("Unsupported qa_pairs representation")


def iter_tasks(source_root: Path, limit: int) -> Iterator[dict]:
    """Limit counts inspected QA candidates, including rejects; zero means all."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    inspected = 0
    # The bounded shared reader caps source rows as well; one row can have many QAs.
    for shard, row in iter_rows(source_root, limit):
        parts = Path(shard).parts
        subset = parts[0] if len(parts) > 1 else source_root.name
        source_id = f"{subset}/{quote(str(row.get('id', 'missing')), safe='')}"
        try:
            if subset not in TASK_TYPES:
                raise ValueError(f"Unregistered CoSyn subset: {subset}")
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError("Missing source record id")
            pairs = qa_items(row["qa_pairs"])
            if not pairs:
                raise ValueError("No QA pairs")
            image = row["image"]["bytes"]
            if not isinstance(image, bytes) or not image:
                raise ValueError("Missing embedded image bytes; external paths are not followed")
        except (ValueError, KeyError, TypeError) as exc:
            inspected += 1
            yield {"source_id": source_id, "reject_reason": str(exc)}
            if limit and inspected >= limit:
                return
            continue
        info = source_root / ".cache/huggingface/download" / (shard + ".metadata")
        version = info.read_text().splitlines() if info.is_file() else []
        # Content identity remains available for copies without HF cache metadata.
        row_hash = digest({key: val for key, val in row.items() if key != "image"})
        revision = version[0] if version else "row-sha256:" + row_hash
        split = Path(shard).name.split("-")[0]
        for index, pair in enumerate(pairs):
            if limit and inspected >= limit:
                return
            inspected += 1
            qa_id = f"{source_id}/qa/{index}"
            if any(not isinstance(pair.get(key), str) or not pair[key].strip() for key in ("question", "answer")):
                yield {"source_id": qa_id, "reject_reason": "Missing/nontext question or answer"}
                continue
            if not isinstance(pair.get("explanation"), str):
                yield {"source_id": qa_id, "reject_reason": "Missing/nontext source explanation"}
                continue
            yield {"source_id": qa_id, "source_split": split, "source_revision": revision,
                   "task_type": TASK_TYPES[subset], "query": pair["question"],
                   "answer": {"type": "text", "value": pair["answer"]}, "image": image,
                   "group_id": f"cosyn/{subset}/{row['id']}", "annotation_origin": "derived",
                   "metadata": {"source_subset": subset, "source_shard": shard,
                                "source_row_sha256": row_hash, "source_qa_index": index,
                                "source_explanation": pair["explanation"],
                                "source_metadata": row.get("metadata"),
                                "source_data_sha256": hashlib.sha256(str(row.get("data", "")).encode()).hexdigest(),
                                "source_code_sha256": hashlib.sha256(str(row.get("code", "")).encode()).hexdigest(),
                                "source_explanation_status": "unverified_offline_annotation",
                                "image_answerability_review": "pending",
                                "semantic_trajectory_review": "pending"}}


def iter_candidates(source_root: Path, limit: int, annotations: Path | None = None) -> Iterator[dict]:
    evidence = load_annotations(annotations)
    for base in iter_tasks(source_root, limit):
        yield base if "reject_reason" in base else apply_annotation(base, evidence)
