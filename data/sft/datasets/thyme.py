"""Convert a strictly interpreted crop-only subset of Thyme-SFT; never run source code."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageChops, ImageStat

CONVERSION_VERSION = "thyme-crop-v1"
# This heuristic detects gross image/crop mismatches, not semantic correctness.
MAX_PREVIEW_MAE = 20.0


@dataclass(frozen=True)
class View:
    box: tuple[float, float, float, float]
    size: tuple[int, int]
    reference: int = 0
    cropped: bool = False


class CropParser:
    """Interpret only image geometry and inert filename bookkeeping using an AST."""

    def __init__(self, source_path: str, size: tuple[int, int]):
        self.source_path = source_path
        self.root = View((0, 0, *size), size)
        self.env: dict[str, Any] = {}
        self.saved: list[View] = []
        self.region_count = 0

    @staticmethod
    def name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return CropParser.name(node.value) + "." + node.attr
        return ""

    def value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise ValueError(f"unresolved variable: {node.id}")
            return self.env[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.value(x) for x in node.elts)
        if isinstance(node, ast.JoinedStr):
            # Filenames are inert and never opened. Evaluate inserts to reject hidden calls.
            return "".join(str(self.value(x.value)) if isinstance(x, ast.FormattedValue) else str(self.value(x)) for x in node.values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            val = self.value(node.operand)
            return -val if isinstance(node.op, ast.USub) else val
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)):
            a, b = self.value(node.left), self.value(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise ValueError("non-numeric arithmetic")
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
            return a // b
        if isinstance(node, ast.Attribute):
            if node.attr == "hex" and isinstance(node.value, ast.Call) and self.name(node.value.func) == "uuid4":
                return "offline"
            if self.name(node).startswith(("cv2.INTER_", "Image.Resampling.", "Image.LANCZOS", "Image.BICUBIC")):
                return 0
            raise ValueError("unsupported attribute")
        if isinstance(node, ast.Subscript):
            obj = self.value(node.value)
            if isinstance(obj, View):
                if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) != 2:
                    raise ValueError("crop requires two slices")
                y, x = node.slice.elts
                if not isinstance(x, ast.Slice) or not isinstance(y, ast.Slice) or x.step or y.step:
                    raise ValueError("unsupported crop slice")
                bounds = (self.value(x.lower) if x.lower else 0, self.value(y.lower) if y.lower else 0,
                          self.value(x.upper) if x.upper else obj.size[0], self.value(y.upper) if y.upper else obj.size[1])
                if any(type(v) is not int for v in bounds):
                    raise ValueError("numpy slice coordinates must be integers")
                return self.crop(obj, bounds)
            if isinstance(node.slice, ast.Slice):
                return obj[slice(self.value(node.slice.lower) if node.slice.lower else None,
                                 self.value(node.slice.upper) if node.slice.upper else None)]
            return obj[self.value(node.slice)]
        if isinstance(node, ast.Call):
            name = self.name(node.func)
            if name in {"cv2.imread", "Image.open"}:
                if len(node.args) != 1 or node.keywords or self.value(node.args[0]) != self.source_path:
                    raise ValueError("image loading must refer to the declared original")
                return self.root
            if name == "int" and len(node.args) == 1 and not node.keywords:
                return int(self.value(node.args[0]))
            if name in {"uuid4", "random.randint"}:
                for arg in node.args: self.value(arg)
                return "offline" if name == "uuid4" else 1234
            if name in {"os.path.join", "os.path.basename", "os.path.splitext"}:
                args = [self.value(x) for x in node.args]
                if name.endswith("join"): return "/".join(args)
                if name.endswith("basename"): return args[0].rsplit("/", 1)[-1]
                p = Path(args[0]); return str(p.with_suffix("")), p.suffix
            if name == "cv2.resize":
                obj = self.value(node.args[0]); target = self.value(node.args[1])
                kw = {x.arg: self.value(x.value) for x in node.keywords}
                if not isinstance(obj, View) or set(kw) - {"fx", "fy", "interpolation"}:
                    raise ValueError("unsupported resize")
                size = target if target not in (None, (0, 0)) else (round(obj.size[0] * kw["fx"]), round(obj.size[1] * kw["fy"]))
                return self.resize(obj, size)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"crop", "resize"}:
                obj = self.value(node.func.value)
                if not isinstance(obj, View): raise ValueError("method needs an image")
                args = [self.value(x) for x in node.args]
                if node.func.attr == "crop":
                    if len(args) != 1 or node.keywords: raise ValueError("unsupported PIL crop")
                    return self.crop(obj, tuple(round(v) for v in args[0]))
                for kw in node.keywords:
                    if kw.arg != "resample": raise ValueError("unsupported PIL resize option")
                    self.value(kw.value)
                return self.resize(obj, args[0])
            raise ValueError(f"unsupported call: {name or type(node.func).__name__}")
        raise ValueError(f"unsupported expression: {type(node).__name__}")

    def crop(self, obj: View, bounds: tuple) -> View:
        if len(bounds) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bounds):
            raise ValueError("invalid crop coordinates")
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= obj.size[0] and 0 <= y0 < y1 <= obj.size[1]):
            raise ValueError("crop outside source view")
        left, top, right, bottom = obj.box
        sx, sy = (right - left) / obj.size[0], (bottom - top) / obj.size[1]
        return View((left + x0 * sx, top + y0 * sy, left + x1 * sx, top + y1 * sy),
                    (int(x1 - x0), int(y1 - y0)), obj.reference, True)

    @staticmethod
    def resize(obj: View, size: tuple) -> View:
        if len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
            raise ValueError("invalid resized dimensions")
        return View(obj.box, tuple(size), obj.reference, obj.cropped)

    def bind(self, target: ast.AST, value: Any) -> None:
        if isinstance(target, ast.Name): self.env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == len(value):
            for key, val in zip(target.elts, value): self.bind(key, val)
        else: raise ValueError("unsupported assignment target")

    def parse(self, code: str) -> View:
        self.saved = []
        for node in ast.parse(code).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [node.module] if isinstance(node, ast.ImportFrom) else [x.name for x in node.names]
                if any(x not in {"cv2", "os", "numpy", "uuid", "random", "PIL"} for x in modules):
                    raise ValueError("unsupported import")
            elif isinstance(node, ast.Assign):
                val = self.value(node.value)
                for target in node.targets: self.bind(target, val)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value; name = self.name(call.func)
                if name == "cv2.imwrite" and len(call.args) == 2 and not call.keywords:
                    self.value(call.args[0]); self.saved.append(self.value(call.args[1]))
                elif isinstance(call.func, ast.Attribute) and call.func.attr == "save" and len(call.args) == 1 and not call.keywords:
                    self.value(call.args[0]); self.saved.append(self.value(call.func.value))
                elif name == "print" and len(call.args) == 1 and not call.keywords:
                    if not isinstance(self.value(call.args[0]), str): raise ValueError("non-path program output")
                else: raise ValueError(f"unsupported statement call: {name}")
            else: raise ValueError(f"unsupported statement: {type(node).__name__}")
        if len(self.saved) != 1 or not isinstance(self.saved[0], View) or not self.saved[0].cropped:
            raise ValueError("each code block must save exactly one crop")
        view = self.saved[0]
        self.region_count += 1
        # A later crop of a displayed view references that view; root stays region zero.
        for key, val in list(self.env.items()):
            if isinstance(val, View) and val.box == view.box:
                self.env[key] = View(val.box, val.size, self.region_count, val.cropped)
        return view


def convert_row(row: dict, source_id: str, revision: str, source_split: str) -> dict:
    question, response = row["question"], row["response"]
    path_match = re.search(r'### User Image Path:\*\*\s*"([^"\n]+)"', question)
    if not path_match: raise ValueError("missing original image identity")
    source_path = path_match.group(1)
    if "/gqa/" not in source_path:
        raise ValueError("unregistered source task family; requires explicit mapping")
    query = question[:path_match.start()].replace("<image>", "").strip()
    if question[:path_match.start()].count("<image>") != 1 or not query:
        raise ValueError("requires one original input image and a nonempty query")
    match = re.fullmatch(r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*", response, re.S)
    if not match or not match[2].strip(): raise ValueError("missing complete think/answer structure")
    images = [base64.b64decode(x, validate=True) for x in row["image"]]
    if not images: raise ValueError("missing embedded original image")
    with Image.open(io.BytesIO(images[0])) as src:
        if src.getexif().get(274, 1) != 1: raise ValueError("nonidentity EXIF requires coordinate transformation")
        original = src.convert("RGB")
    declared = re.search(r'User Image Size:\*\*\s*"(\d+)x(\d+)"', question)
    if not declared or original.size != (int(declared[1]), int(declared[2])):
        raise ValueError("declared and embedded original dimensions disagree")
    parser = CropParser(source_path, original.size)
    body = match[1]
    pattern = re.compile(r"<code>\s*```python\s*\n(.*?)```\s*</code>\s*<sandbox_output>\s*<image>\s*</sandbox_output>", re.S)
    blocks = list(pattern.finditer(body))
    if not blocks or len(images) != len(blocks) + 1:
        raise ValueError("tool outputs and embedded image count do not match")
    events: list[dict] = []; checks = []; cursor = 0
    for index, block in enumerate(blocks, 1):
        text = body[cursor:block.start()].strip()
        if "<" in text or "```" in text: raise ValueError("unparsed tool markup")
        if text: events.append({"type": "reasoning_text", "text": text})
        view = parser.parse(block[1])
        with Image.open(io.BytesIO(images[index])) as target:
            preview = target.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        replay = original.crop(view.box).resize((32, 32), Image.Resampling.BILINEAR)
        mae = sum(ImageStat.Stat(ImageChops.difference(preview, replay)).mean) / 3
        if mae > MAX_PREVIEW_MAE:
            raise ValueError(f"derived image mismatch at read {index}: RGB MAE {mae:.3f}")
        checks.append({"region_id": index, "preview_rgb_mae": mae})
        events.append({"type": "visual_read", "box": list(view.box), "reference_region_id": view.reference})
        cursor = block.end()
    text = body[cursor:].strip()
    if "<" in text or "```" in text: raise ValueError("unparsed final reasoning markup")
    if text: events.append({"type": "reasoning_text", "text": text})
    return {"source_id": source_id, "source_revision": revision, "source_split": source_split,
            "task_type": "general_vqa", "query": query, "answer": {"type": "text", "value": match[2].strip()},
            "image": images[0], "events": events, "visual_only_safe": False, "annotation_origin": "derived",
            "metadata": {"conversion_version": CONVERSION_VERSION, "source_family": "gqa", "source_image_id": Path(source_path).name,
                         "trajectory_origin": "source_teacher_with_static_crop_conversion", "preview_checks": checks,
                         "preview_mae_threshold": MAX_PREVIEW_MAE, "feature_compatibility": "pending",
                         "semantic_trajectory_review": "pending", "visual_only_review": "pending",
                         "source_row_sha256": hashlib.sha256((question + response).encode()).hexdigest()}}


def iter_candidates(source_root: Path, limit: int) -> Iterator[dict]:
    """Scan a stable list of completed parquet files; limit counts inspected source rows."""
    if limit < 0: raise ValueError("limit must be nonnegative (zero means all)")
    paths = sorted((source_root / "data").glob("*.parquet"))
    if not paths: raise FileNotFoundError("no completed Thyme parquet shards in source_root/data")
    inspected = 0
    for path in paths:
        stat = path.stat()
        # HF download metadata stores immutable upstream commit and content ETag.
        info = source_root / ".cache/huggingface/download/data" / (path.name + ".metadata")
        lines = info.read_text().splitlines() if info.is_file() else []
        revision = lines[0] if lines else f"local-{path.name}-{stat.st_size}-{stat.st_mtime_ns}"
        try:
            parquet = pq.ParquetFile(path)
            row_index = 0
            for batch in parquet.iter_batches(batch_size=1):
                for row in batch.to_pylist():
                    if limit and inspected >= limit: return
                    source_id = f"{path.name}:{row_index}"
                    inspected += 1; row_index += 1
                    try: yield convert_row(row, source_id, revision, path.name.split("-000")[0])
                    except (ValueError, TypeError, KeyError, IndexError, OSError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
                        yield {"source_id": source_id, "reject_reason": str(exc)}
            if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
                raise RuntimeError(f"source shard changed during reading: {path.name}")
        except (OSError, pa.ArrowInvalid) as exc:
            yield {"source_id": path.name, "reject_reason": f"unreadable parquet: {exc}"}
