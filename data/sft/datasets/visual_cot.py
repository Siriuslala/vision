"""Convert Visual-CoT pixel metadata, preserving source natural-language thoughts."""

from __future__ import annotations

import hashlib
import json
import math
import argparse
import io
import shutil
import tarfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


class _PartsReader(io.RawIOBase):
    """Expose consecutive completed split files as one forward-only stream."""

    def __init__(self, paths: list[Path]):
        self.paths = iter(paths)
        self.current = None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise ValueError("bounded reads required")
        chunks = []
        remaining = size
        while remaining:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open("rb")
            chunk = self.current.read(remaining)
            if not chunk:
                self.current.close()
                self.current = None
                continue
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
        super().close()


def extract_images(source_root: Path, output_root: Path, limit: int = 10) -> int:
    """Extract a bounded prefix of completed official tar parts into a new directory.

    Full extraction (limit=0) requires all 13 exact-size parts. Prefix extraction
    may use fewer complete parts and fails if the requested prefix is unavailable.
    """
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if output_root.exists():
        raise FileExistsError("extraction output must be a new independent directory")
    paths = []
    for index in range(13):
        path = source_root / "cot_images_tar_split" / f"cot_images_{index:02d}"
        expected = 10403440640 if index == 12 else 10737418240
        if not path.is_file():
            if limit == 0:
                raise FileNotFoundError(f"missing archive part: {path.name}")
            break
        if path.stat().st_size != expected:
            raise ValueError(f"incomplete archive part {path.name}: expected {expected} bytes")
        paths.append(path)
    if not paths:
        raise FileNotFoundError("no completed cot_images_00 archive part")
    output_root.mkdir(parents=True)
    count = 0
    with _PartsReader(paths) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            relative = _relative(member.name)
            if not relative.parts or relative.parts[0] != "cot_image_data":
                raise ValueError(f"unexpected archive namespace: {member.name}")
            if member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"unsupported archive member: {member.name}")
            if relative.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                raise ValueError(f"unexpected non-image archive member: {member.name}")
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"unreadable member: {member.name}")
            with extracted, destination.open("xb") as output:
                shutil.copyfileobj(extracted, output)
            count += 1
            if limit and count >= limit:
                return count
    if limit and count < limit:
        raise ValueError(f"available archive prefix yielded only {count}/{limit} images")
    return count


TASKS = {
    "cub": "recognition", "docvqa": "document_qa", "dude": "document_qa",
    "flickr30k": "general_vqa", "gqa": "general_vqa",
    "infographicsvqa": "document_qa", "openimages": "general_vqa",
    "sroie": "document_qa", "textcap": "captioning", "textvqa": "ocr",
    "visual7w": "general_vqa", "vsr": "spatial_reasoning",
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("source image must be a safe relative POSIX path")
    return Path(*path.parts)


def resolve_image(source_root: Path, dataset: str, image_name: str) -> Path:
    """Support the released tar layout and the original author's training layout."""
    relative = _relative(image_name)
    image_dataset = "textvqa" if dataset == "textcap" else dataset
    choices = [
        source_root / "cot_image_data" / image_dataset / relative,
        source_root / "cot" / image_dataset / relative,
        source_root / image_dataset / relative,
        source_root / relative,
    ]
    found = [p for p in choices if p.is_file()]
    if not found:
        raise FileNotFoundError(f"missing_image:{image_dataset}/{image_name}; extract completed image archive first")
    return found[0]


def candidate_from_record(record: dict[str, Any], source_root: Path, source_file: str,
                          line_number: int) -> dict[str, Any]:
    """Build one union-window observation from original pixel xyxy annotations.

    Functional programs contain privileged object identifiers and stay offline.
    The source provides no alignment of individual thoughts to individual boxes;
    the complete thought therefore follows the author's single union ROI.
    """
    dataset = _text(record.get("dataset"), "dataset")
    if dataset not in TASKS:
        raise ValueError(f"unsupported dataset: {dataset}")
    split = _text(record.get("split"), "split")
    if split != "train":
        raise ValueError(f"held_out_source_split:{split}")
    question = _text(record.get("question"), "question")
    answer = _text(record.get("answer"), "answer")
    name = _text(record.get("image"), "image")
    width, height = record.get("width"), record.get("height")
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("width and height must be positive integers")
    boxes = record.get("bboxs")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("missing nonempty pixel bboxs")
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in box)):
            raise ValueError("bbox must contain four finite pixel coordinates")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"out_of_bounds_source_bbox:{box}; image={width}x{height}")
    union = [float(min(b[0] for b in boxes)), float(min(b[1] for b in boxes)),
             float(max(b[2] for b in boxes)), float(max(b[3] for b in boxes))]
    image = resolve_image(source_root, dataset, name)
    with Image.open(image) as decoded:
        if decoded.size != (width, height):
            raise ValueError(f"source_image_size_mismatch:{decoded.size}!={(width, height)}")
    events: list[dict[str, Any]] = [
        {"type": "visual_read", "box": union, "reference_region_id": 0}
    ]
    thought = record.get("thought")
    if thought is not None:
        thought = _text(thought, "thought")
        # Never serialize the functional program or full_answer as a thought.
        if "<image>" in thought or "```" in thought:
            raise ValueError("unsupported tool/image content in source thought")
        events.append({"type": "reasoning_text", "text": thought})
    identity = hashlib.sha256((dataset + "\0" + name + "\0" + question).encode()).hexdigest()
    return {
        "source_id": f"{dataset}:{identity}", "source_split": split,
        "task_type": TASKS[dataset],
        "query": question, "answer": {"type": "text", "value": answer},
        "image": image, "events": events,
        "visual_only_safe": thought is None,
        "annotation_origin": "derived",
        "source_revision": "row-sha256:" + hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "metadata": {
            "source_file": source_file, "source_line": line_number,
            "source_dataset": dataset, "source_image": name,
            "source_coordinate_system": "original_image_pixel_xyxy",
            "source_boxes_pixel_xyxy": boxes,
            "source_bbox_count": len(boxes), "window_method": "source_bbox_union_v1",
            "source_functional_program": record.get("reasoning"),
            "source_full_answer": record.get("full_answer"),
            "source_thought": thought,
            "language_origin": "source_thought" if thought else "none",
            "review_requirements": ["window_evidence_coverage", "cached_feature_legibility",
                                    "source_thought_image_consistency"],
            "visual_only_policy": "no_language_removal_without_independent_review",
        },
    }


def iter_candidates(source_root: Path, limit: int,
                    image_cache: Path | None = None) -> Iterator[dict[str, Any]]:
    """Read only completed JSONL files, preferring detailed GQA on exact duplicates.

    limit bounds inspected source rows, including rejection rows. Zero means all.
    No private download cache, incomplete file, or source image is modified.
    """
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    detailed = sorted((source_root / "cot_with_detailed_reasoning_steps").glob("*_train.jsonl"))
    metadata = sorted((source_root / "metadata").glob("*_train.jsonl"))
    if not metadata:
        metadata = sorted((source_root / "viscot_dataset").glob("*_train.jsonl"))
    paths = detailed + metadata
    if not paths:
        raise FileNotFoundError("Visual-CoT requires completed metadata/*_train.jsonl or "
                                "cot_with_detailed_reasoning_steps/*_train.jsonl. "
                                "viscot_363k.json uses padded coordinates; download original metadata.")
    seen: set[str] = set()
    count = 0
    for path in paths:
        relative = path.relative_to(source_root).as_posix()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if limit and count >= limit:
                    return
                count += 1
                source_id = f"{relative}:{line_number}"
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record must be an object")
                    candidate = candidate_from_record(record, image_cache or source_root, relative, line_number)
                    source_id = candidate["source_id"]
                    if source_id in seen:
                        raise ValueError("duplicate_image_question_prefer_first_detailed_record")
                    seen.add(source_id)
                    yield candidate
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    yield {"source_id": source_id, "reject_reason": str(exc),
                           "metadata": {"source_file": relative, "source_line": line_number}}


def iter_tasks(source_root: Path, limit: int,
               image_cache: Path | None = None) -> Iterator[dict[str, Any]]:
    """Expose separate source boxes as unverified teacher hints, without imposing order.

    The legacy candidate converter retains its union-window behavior. A teacher
    task intentionally has no events: the source does not align each box to a
    thought, so a new sequence needs evidence and semantic checks.
    """
    for candidate in iter_candidates(source_root, limit, image_cache):
        if "reject_reason" in candidate:
            yield candidate
            continue
        candidate.pop("events")
        candidate.pop("visual_only_safe")
        metadata = candidate["metadata"]
        metadata.pop("window_method")
        metadata["source_box_order"] = "unverified"
        metadata["source_correction"] = "unverified"
        metadata["trajectory_origin"] = "local_teacher_regeneration"
        candidate["source_boxes"] = metadata["source_boxes_pixel_xyxy"]
        yield candidate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safely extract completed Visual-CoT image tar parts")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps({"extracted_images": extract_images(args.source_root, args.output_root, args.limit)}))
