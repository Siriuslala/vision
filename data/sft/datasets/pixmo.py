"""PixMo point audit preparation and reviewed-box counting conversion.

Image acquisition and annotation are separate from conversion. Natural-image
points never become instance boxes without an explicit annotation sidecar.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Iterator
from urllib.request import urlopen

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from data.sft.tools.point_windows import file_sha256

VERSION = "pixmo-count-v1"


def rows(source_root: Path, limit: int) -> Iterator[tuple[str, dict, str]]:
    """Stream completed train parquet files; zero means all available rows."""
    files = sorted((source_root / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError("No completed PixMo training parquet found")
    seen = 0
    for path in files:
        revision = file_sha256(path)
        parquet = pq.ParquetFile(path)
        offset = 0
        for batch in parquet.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                if limit and seen >= limit:
                    return
                yield f"{path.stem}:{offset}", row, revision
                offset += 1
                seen += 1


def points_of(row: dict) -> list[tuple[float, float]]:
    points = row["points"]
    if not isinstance(points, dict) or len(points["x"]) != len(points["y"]):
        raise ValueError("Missing or inconsistent training points")
    result = list(zip(points["x"], points["y"], strict=True))
    if len(result) != row["count"] or len(set(result)) != len(result):
        raise ValueError("Point count mismatch or duplicate points require source review")
    if any(not math.isfinite(v) for p in result for v in p):
        raise ValueError("Non-finite point")
    return result


def audit_key(source_id: str) -> str:
    return source_id.replace(":", "_") + ".json"


def checked_image(path: Path, row: dict) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Image missing: {path}; run pixmo prepare with downloads")
    actual = file_sha256(path)
    if actual != row["image_sha256"]:
        raise ValueError(f"Source image SHA256 mismatch: expected {row['image_sha256']}, actual {actual}")
    with Image.open(path) as image:
        image.load()
        if image.getexif().get(274, 1) != 1:
            raise ValueError("EXIF orientation requires reviewed point coordinate mapping")
        return image.size


def box_union(boxes: list[list[float]]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def intersects(a: list[float], b: list[float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def group_instances(instances: list[dict], max_group: int = 6) -> list[list[dict]]:
    """Use row spans, horizontal order and conservative exclusive box checks.

    This proposes geometry only. Actual feature discernibility remains a review
    requirement; geometry thresholds cannot stand in for VLM feature validation.
    """
    if max_group < 1:
        raise ValueError("max_group must be positive")
    def cy(obj):
        return (obj["box"][1] + obj["box"][3]) / 2
    ordered = sorted(instances, key=lambda o: (cy(o), o["instance_id"]))
    row_groups: list[list[dict]] = []
    for obj in ordered:
        proposal = (row_groups[-1] if row_groups else []) + [obj]
        heights = [o["box"][3] - o["box"][1] for o in proposal]
        span = max(map(cy, proposal)) - min(map(cy, proposal))
        if row_groups and span <= median(heights) * 0.5:
            row_groups[-1].append(obj)
        else:
            row_groups.append([obj])
    order = [o for row in row_groups for o in sorted(row, key=lambda o: (o["box"][0], cy(o), o["box"][2], o["instance_id"]))]
    groups: list[list[dict]] = []
    for obj in order:
        proposal = (groups[-1] if groups else []) + [obj]
        merged = box_union([o["box"] for o in proposal])
        area = (merged[2]-merged[0]) * (merged[3]-merged[1])
        sum_area = sum((o["box"][2]-o["box"][0])*(o["box"][3]-o["box"][1]) for o in proposal)
        if groups and len(proposal) <= max_group and area <= 2 * sum_area:
            groups[-1].append(obj)
        else:
            groups.append([obj])
    # Overlapping natural objects often cannot support exclusive rectangles.
    # Reject instead of inventing identity language for new_instances mode.
    for group in groups:
        ids = {o["instance_id"] for o in group}
        window = box_union([o["box"] for o in group])
        if any(intersects(window, o["box"]) for o in instances if o["instance_id"] not in ids):
            raise ValueError("Exclusive windows intersect other instances; regroup or review new_instances")
    return groups


def counting_candidate(row: dict, source_id: str, revision: str, image_path: Path,
                       audit: dict, audit_path: str, max_group: int = 6) -> dict:
    width, height = checked_image(image_path, row)
    points = points_of(row)
    if audit.get("image_sha256") != row["image_sha256"] or audit.get("source_id") != source_id:
        raise ValueError("Audit identity or image digest mismatch")
    if audit.get("instance_boxes_reviewed") is not True or audit.get("complete_instance_set_reviewed") is not True:
        raise ValueError("Instance boxes and target completeness require review")
    if audit.get("method") not in {"geometry", "sam3", "geometry_sam3"} or not audit.get("config_id"):
        raise ValueError("Audit must identify the point-to-box method and immutable config")
    instances = audit["instances"]
    expected = {f"point_{i}" for i in range(len(points))}
    if len(instances) != len(points) or {o["instance_id"] for o in instances} != expected:
        raise ValueError("Instance IDs must match the distinct original points")
    for obj in instances:
        box = obj["box"]
        if len(box) != 4 or any(not math.isfinite(x) for x in box):
            raise ValueError("Invalid instance box")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Instance box outside image")
        point = points[int(obj["instance_id"].split("_")[1])]
        if not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1):
            raise ValueError("Instance box does not contain its point")
    groups = group_instances(instances, max_group) if instances else [[]]
    events, records, derivations = [], [], []
    cumulative = 0
    for group in groups:
        ids = [o["instance_id"] for o in group]
        box = box_union([o["box"] for o in group]) if group else [0, 0, width, height]
        read_id = len(events) + 1
        events.append({"type": "visual_read", "box": box, "reference_region_id": 0})
        cumulative += len(ids)
        events.append({"type": "reasoning_text", "text": f"This region contains {len(ids)} {row['label']}. The running total is {cumulative}."})
        records.append({"read_event_id": read_id, "count_event_id": read_id + 1,
                        "visible_instance_ids": ids, "new_instance_ids": ids,
                        "local_count": len(ids), "new_count": len(ids), "cumulative_count": cumulative})
        if ids:
            derivations.append({"event_id": read_id, "method": audit["method"],
                                "config_id": audit["config_id"], "source_point_ids": ids,
                                "audit_record_path": audit_path})
    sum_id = len(events) + 1
    expression = " + ".join(str(len(g)) for g in groups)
    events.append({"type": "reasoning_text", "text": f"Adding the counts gives {expression} = {row['count']}."})
    return {"source_id": source_id, "source_split": "train", "source_revision": revision,
            "task_type": "counting", "query": f"How many {row['label']} are in the image?",
            "answer": {"type": "number", "value": row["count"]}, "image": image_path,
            "events": events, "visual_only_safe": True, "annotation_origin": "derived",
            "metadata": {"window_derivations": derivations,
                         "counting": {"config_id": f"{VERSION}-group-{max_group}", "mode": "exclusive",
                                      "instance_order": [o["instance_id"] for g in groups for o in g],
                                      "audit_record_path": audit_path, "groups": records, "sum_event_id": sum_id},
                         "feature_and_window_review": "pending", "source_point_units": "pixels",
                         "source_image_url": row["image_url"]}}


def iter_candidates(source_root: Path, limit: int = 10, *, image_cache: Path | None = None,
                    audit_dir: Path | None = None, data_root: Path | None = None) -> Iterator[dict]:
    image_cache = Path(image_cache or os.environ.get("PIXMO_IMAGE_CACHE", source_root / "images"))
    audit_dir = Path(audit_dir or os.environ.get("PIXMO_AUDIT_DIR", source_root / "sft_audits"))
    data_root = Path(data_root or os.environ.get("DATA_DIR", source_root.parent)).resolve()
    for source_id, row, revision in rows(source_root, limit):
        try:
            image_path = image_cache / row["image_sha256"]
            checked_image(image_path, row)
            path = audit_dir / audit_key(source_id)
            if not path.is_file():
                raise ValueError("Reviewed point-to-box audit missing; run pixmo prepare")
            audit = json.loads(path.read_text())
            relative = path.resolve().relative_to(data_root).as_posix()
            yield counting_candidate(row, source_id, revision, image_path, audit, relative)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            yield {"source_id": source_id, "reject_reason": str(exc)}


def prepare(source_root: Path, output_root: Path, limit: int, download: bool) -> None:
    """Create an immutable point audit queue and optional image downloads."""
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Preparation output must be empty; existing annotations are never overwritten")
    image_dir, audit_dir = output_root / "images", output_root / "audits"
    image_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    with (output_root / "report.jsonl").open("w") as report:
        for source_id, row, revision in rows(source_root, limit):
            result = {"source_id": source_id, "source_revision": revision, "image_url": row["image_url"]}
            try:
                points = points_of(row)
                path = image_dir / row["image_sha256"]
                if download:
                    with urlopen(row["image_url"], timeout=40) as response:
                        raw = response.read(32 * 1024 * 1024 + 1)
                    if len(raw) > 32 * 1024 * 1024:
                        raise ValueError("Image exceeds 32 MiB limit")
                    path.write_bytes(raw)
                width, height = checked_image(path, row)
                if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
                    raise ValueError("Point outside image")
                audit = {"source_id": source_id, "image_sha256": row["image_sha256"],
                         "label": row["label"], "count": row["count"], "point_units": "pixels",
                         "points": [{"instance_id": f"point_{i}", "point": p} for i, p in enumerate(points)],
                         "method": None, "config_id": None, "instance_boxes_reviewed": False,
                         "complete_instance_set_reviewed": False, "instances": []}
                (audit_dir / audit_key(source_id)).write_text(json.dumps(audit, indent=2) + "\n")
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    draw = ImageDraw.Draw(image)
                    for i, (x, y) in enumerate(points):
                        draw.ellipse((x-5, y-5, x+5, y+5), outline="red", width=2)
                        draw.text((x+7, y), str(i), fill="red")
                    image.save(audit_dir / (audit_key(source_id) + ".jpg"))
                result["status"] = "point_audit_required"
                success += 1
            except (ValueError, OSError, KeyError, TypeError) as exc:
                result.update(status="rejected", reason=str(exc))
            report.write(json.dumps(result) + "\n")
            report.flush()
    if not success:
        raise ValueError("No verified point audit records produced; inspect report.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--download-images", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    prepare(args.source_root, args.output_root, args.limit, args.download_images)


if __name__ == "__main__":
    main()
