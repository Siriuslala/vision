"""Offline point-to-box candidates; every result still requires review."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from statistics import median

from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_points(points: list[list[float]], width: int, height: int) -> None:
    if len({tuple(p) for p in points}) != len(points):
        raise ValueError("Duplicate points require instance identity review")
    if any(len(p) != 2 or any(not math.isfinite(v) for v in p)
           or not (0 <= p[0] < width and 0 <= p[1] < height) for p in points):
        raise ValueError("Points must lie inside the image pixel grid")


def refine_component(image: Image.Image, point: list[float], box: list[float],
                     threshold: float = 30) -> dict:
    """Flood the point's foreground using the crop border's median background.

    Restricted to solid, compact foreground on a locally uniform background.
    Touching the crop boundary and other-instance merges require review.
    """
    x0, y0, x1, y1 = math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])
    crop = image.crop((x0, y0, x1, y1)).convert("RGB")
    width, height = crop.size
    pixels = crop.load()
    border = [pixels[x, y] for x in range(width) for y in (0, height-1)]
    border += [pixels[x, y] for y in range(height) for x in (0, width-1)]
    background = [median(c[channel] for c in border) for channel in range(3)]
    def foreground(x, y):
        return max(abs(pixels[x, y][c] - background[c]) for c in range(3)) > threshold
    start = (math.floor(point[0])-x0, math.floor(point[1])-y0)
    if not foreground(*start):
        return {"status": "point_not_on_solid_foreground", "box": None}
    queue, visited = deque([start]), {start}
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited and foreground(nx, ny):
                visited.add((nx, ny))
                queue.append((nx, ny))
    xs, ys = zip(*visited)
    touching = min(xs) == 0 or min(ys) == 0 or max(xs) == width-1 or max(ys) == height-1
    return {"status": "touches_candidate_boundary" if touching else "refined_pending_review",
            "box": [min(xs)+x0, min(ys)+y0, max(xs)+x0+1, max(ys)+y0+1],
            "foreground_pixels": len(visited)}


def geometry_candidates(image: Image.Image, points: list[list[float]], *,
                        regular_compact: bool = False) -> list[dict]:
    if not regular_compact:
        raise ValueError("Geometry requires explicit regular_compact assumption; use SAM3 for natural objects")
    validate_points(points, *image.size)
    result = []
    for index, point in enumerate(points):
        distances = sorted(math.dist(point, other) for j, other in enumerate(points) if j != index)[:3]
        if not distances:
            result.append({"point_index": index, "status": "isolated_requires_segmentation", "candidates": []})
            continue
        scale = median(distances)
        candidates = []
        for ratio in (0.6, 0.8, 1.0):
            half = scale * ratio / 2
            box = [max(0., point[0]-half), max(0., point[1]-half),
                   min(float(image.width), point[0]+half), min(float(image.height), point[1]+half)]
            refinements = [refine_component(image, point, box, threshold=t) for t in (25, 30, 35)]
            refined = refinements[1]
            selected = refined["box"]
            stable = selected is not None and all(r["box"] == selected for r in refinements)
            other_points = [] if selected is None else [j for j, p in enumerate(points) if j != index
                and selected[0] <= p[0] < selected[2] and selected[1] <= p[1] < selected[3]]
            candidates.append({"ratio": ratio, "box": box, "refinement": refined,
                               "stable_across_thresholds": stable, "other_point_indices_in_box": other_points})
        result.append({"point_index": index, "status": "pending", "neighbor_count": len(distances),
                       "local_scale": scale, "candidates": candidates})
    return result


def mask_candidate(mask, positive: list[float], negatives: list[list[float]], score: float) -> dict:
    """Inspect a 2D bool tensor using floor pixel lookup and half-open boxes."""
    if not math.isfinite(score):
        return {"status": "nonfinite_score", "box": None}
    coords = mask.nonzero(as_tuple=False)
    if not len(coords):
        return {"status": "empty_mask", "box": None, "score": score}
    if not bool(mask[math.floor(positive[1]), math.floor(positive[0])]):
        return {"status": "misses_positive", "box": None, "score": score}
    if any(bool(mask[math.floor(p[1]), math.floor(p[0])]) for p in negatives):
        return {"status": "includes_other_instance_point", "box": None, "score": score}
    ymin, xmin = coords.min(dim=0).values.tolist()
    ymax, xmax = coords.max(dim=0).values.tolist()
    return {"status": "pending", "box": [xmin, ymin, xmax+1, ymax+1], "score": score}


def select_mask_candidate(rounds: list[dict], minimum_score: float = 0.8,
                          minimum_box_iou: float = 0.85) -> dict:
    """Choose geometrically consistent masks for a separate semantic teacher check."""
    winners = []
    for record in rounds:
        valid = [c for c in record["candidates"] if c["status"] == "pending"
                 and c.get("score", 0) >= minimum_score and c.get("box")]
        if not valid:
            return {"status": "rejected", "reason": "no_mask_passes_point_and_score_checks"}
        winners.append(max(valid, key=lambda c: c["score"]))
    if not winners:
        return {"status": "rejected", "reason": "no_prompt_rounds"}
    a = winners[0]["box"]
    for winner in winners[1:]:
        b = winner["box"]
        intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
        if not union or intersection / union < minimum_box_iou:
            return {"status": "rejected", "reason": "box_changes_with_negative_prompt"}
    return {"status": "requires_semantic_verification", "box": winners[-1]["box"],
            "mask_file": winners[-1].get("mask_file"), "score": winners[-1]["score"],
            "minimum_score": minimum_score, "minimum_box_iou": minimum_box_iou}


class Sam3LocalBackend:
    """Use the installed Transformers SAM3 Tracker interface, local files only."""
    def __init__(self, checkpoint: str | Path = "facebook/sam3", device: str = "cpu"):
        from huggingface_hub import snapshot_download
        if Path(checkpoint).is_absolute() and not Path(checkpoint).is_dir():
            raise FileNotFoundError("SAM3 local checkpoint directory does not exist")
        checkpoint = Path(checkpoint) if Path(checkpoint).is_dir() else Path(
            snapshot_download(str(checkpoint), local_files_only=True))
        self.checkpoint = checkpoint
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor, AutoConfig
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        if config.model_type not in {"sam3_tracker", "sam3_video"}:
            raise ValueError("Checkpoint must contain SAM3 tracker or video weights")
        # Transformers explicitly extracts tracker_config and tracker_model weights
        # from the official video checkpoint; loading diagnostics remain strict.
        self.processor = Sam3TrackerProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model, loading = Sam3TrackerModel.from_pretrained(checkpoint, local_files_only=True, output_loading_info=True)
        if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("SAM3 checkpoint has missing or incompatible weights")
        self.model = self.model.to(device).eval()
        self.device = device

    def propose(self, image: Image.Image, points: list[list[float]], mask_dir: Path) -> list[dict]:
        import torch
        validate_points(points, *image.size)
        mask_dir.mkdir(parents=True, exist_ok=True)
        result = []
        # Encode once, then decode each point group. At most one additional
        # decode adds the remaining nearby negative to check prompt consistency.
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            embeddings = self.model.get_image_embeddings(image_inputs["pixel_values"])
            for index, point in enumerate(points):
                neighbor_ids = sorted((j for j in range(len(points)) if j != index),
                                      key=lambda j: math.dist(point, points[j]))[:4]
                initial = neighbor_ids[:3]
                rounds = [initial] + ([neighbor_ids] if len(neighbor_ids) > len(initial) else [])
                round_records = []
                for round_index, negative_ids in enumerate(rounds):
                    inputs = self.processor(original_sizes=image_inputs["original_sizes"], input_points=[[[point] + [points[j] for j in negative_ids]]],
                                            input_labels=[[[1] + [0] * len(negative_ids)]], return_tensors="pt").to(self.device)
                    outputs = self.model(image_embeddings=embeddings, input_points=inputs["input_points"],
                                         input_labels=inputs["input_labels"], multimask_output=True)
                    if not torch.isfinite(outputs.pred_masks).all() or not torch.isfinite(outputs.iou_scores).all():
                        raise ValueError("SAM3 produced non-finite mask logits or scores")
                    masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][0]
                    scores = outputs.iou_scores[0, 0].detach().cpu().tolist()
                    candidates = []
                    for mask_index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
                        record = mask_candidate(mask, point, [p for j, p in enumerate(points) if j != index], score)
                        filename = f"point_{index}_round_{round_index}_mask_{mask_index}.png"
                        Image.fromarray((mask.to(torch.uint8)*255).numpy()).save(mask_dir / filename)
                        record["mask_file"] = filename
                        candidates.append(record)
                    round_records.append({"positive": point, "negative_point_indices": negative_ids, "candidates": candidates})
                selection = select_mask_candidate(round_records)
                result.append({"point_index": index, "status": "pending", "rounds": round_records,
                               "automatic_selection": selection,
                               "semantic_and_prompt_consistency_review": "required"})
        return result


def propose(audit_dir: Path, image_cache: Path, output_root: Path, *, method: str,
            regular_compact: bool = False, checkpoint: str | Path | None = None, device: str = "cpu", limit: int = 3) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Proposal output must be empty; source audits are never overwritten")
    if method not in {"geometry", "sam3"}:
        raise ValueError("Unknown proposal method")
    backend = Sam3LocalBackend(checkpoint or "facebook/sam3", device) if method == "sam3" else None
    config = {"version": "point-windows-v1", "method": method, "regular_compact": regular_compact,
              "ratios": [0.6, 0.8, 1.0], "neighbors": 3, "thresholds": [25, 30, 35],
              "point_rasterization": "floor", "checkpoint_files": {}}
    if method == "sam3":
        import transformers
        config["sam3"] = {"transformers": transformers.__version__, "backend": "Sam3TrackerModel", "initial_negatives": 3, "review_negatives": 4, "multimask_output": True, "mask_threshold": 0.0, "binarize": True, "selection_minimum_score": 0.8, "selection_minimum_box_iou": 0.85}
    if backend:
        checkpoint = backend.checkpoint
        config["checkpoint_files"] = {p.relative_to(checkpoint).as_posix(): file_sha256(p)
            for p in sorted(checkpoint.rglob("*")) if p.is_file() and not p.name.startswith(".")}
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    files = sorted(audit_dir.glob("*.json"))
    if limit:
        files = files[:limit]
    success = 0
    with (output_root / "report.jsonl").open("w") as report:
        for path in files:
            record = {"audit_file": path.name}
            try:
                audit = json.loads(path.read_text())
                image_path = image_cache / audit["image_sha256"]
                if file_sha256(image_path) != audit["image_sha256"]:
                    raise ValueError("Image SHA256 mismatch")
                points = [entry["point"] for entry in audit["points"]]
                with Image.open(image_path) as image:
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError("EXIF orientation requires point mapping")
                    image = image.convert("RGB")
                    candidates = backend.propose(image, points, output_root / (path.stem + "_masks")) if backend else geometry_candidates(image, points, regular_compact=regular_compact)
                proposal = copy.deepcopy(audit)
                proposal.update(method=method, config_id=config_id, instance_boxes_reviewed=False,
                                complete_instance_set_reviewed=False, instances=[], candidate_instances=candidates)
                # Candidate boxes remain selectable evidence; no automatic
                # promotion into an audited instance set occurs here.
                (output_root / path.name).write_text(json.dumps(proposal, indent=2) + "\n")
                record["status"] = "pending_review"
                success += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record.update(status="rejected", reason=str(exc))
            report.write(json.dumps(record) + "\n")
    if not success:
        raise ValueError("No point proposal records produced; inspect report.jsonl and input audit directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("audit-dir", "image-cache", "output-root"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--method", choices=("geometry", "sam3"), default="geometry")
    parser.add_argument("--regular-compact", action="store_true")
    parser.add_argument("--checkpoint", default="facebook/sam3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    propose(**vars(args))


if __name__ == "__main__":
    main()
