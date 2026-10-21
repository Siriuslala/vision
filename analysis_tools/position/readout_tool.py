"""Experiment 3.9: a coordinate readout tool for Qwen3.5-9B.

Design: docs/analysis/position/design_readout_tool.md.

The tool is exposed to the frozen model through the native Qwen3.5 function-calling template as
``locate(target)``.  When the model emits a call, the runtime reads the decoder residual stream at
one token of that call (``call_end`` = the ``</tool_call>`` token by default), applies a regressor
trained on diverse data, and returns ``{"point_2d": [x, y], "bbox_2d": [x0, y0, x1, y1]}`` in 0-1000
normalized units as the tool response.  The model weights are never modified.

This module holds the reusable engine:

* the tool schema, prompts, canonical call text and readout sites;
* a generic synthetic scene generator for training (random shapes, colours, sizes, backgrounds,
  canvases; no grids, rings or single red discs on white, which are evaluation tasks);
* case builders for RefCOCO (image-disjoint train / validation / test pools) and for the held-out
  synthetic evaluation sets (experiment-3.6 discs, experiment-1.3 T6 grids, 1.3-supplement layouts
  and rings);
* scoring (point hit, centre error, bbox IoU, nearest instance, coordinate conventions);
* batched teacher-forced capture of residual states at the readout sites;
* ridge and MLP regressors and the ``ReadoutTool`` wrapper used online.

Layer numbers are 1-based (output of decoder layer l), as in the analysis documents.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from analysis_tools.common import qwen_runtime as qr

# ----------------------------------------------------------------------------- runtime

def load_runtime(name: str, max_memory_gib: list[float] | None = None) -> qr.Runtime:
    """`qwen_runtime.load` on one GPU, or split across the visible GPUs when per-GPU budgets are given.

    The shared GPUs are often too full for one 19 GB copy of the weights; accelerate's layer-wise
    placement leaves the computation unchanged (inputs enter on cuda:0, hidden states move between
    devices at layer boundaries).
    """
    if not max_memory_gib:
        return qr.load(name, 'cuda:0')
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    checkpoint = resolve_model(name, load_paths()['HF_HUB_CACHE'])
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa',
        max_memory={i: f'{m}GiB' for i, m in enumerate(max_memory_gib)}).eval()
    model.requires_grad_(False)
    return qr.Runtime(name, checkpoint, model, processor, 'cuda:0')


# ----------------------------------------------------------------------------- tool protocol

TOOL_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'locate',
        'description': ('Return the location of one object in the current image as 0-1000 normalized '
                        'coordinates (point_2d is the center of its bounding box, bbox_2d is '
                        '[x_min, y_min, x_max, y_max]).'),
        'parameters': {'type': 'object',
                       'properties': {'target': {'type': 'string', 'description': 'Description of the object to locate.'}},
                       'required': ['target']},
    },
}
QUERY = 'Locate the object described as: {target}.'
TOOL_INSTRUCTION = ' Call the locate tool to find it.'
CALL_TEMPLATE = '<tool_call>\n<function=locate>\n<parameter=target>\n{target}\n</parameter>\n</function>\n</tool_call>'
SITES = ('prompt_end', 'target_end', 'call_end')
CAPTURE_LAYERS = (8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32)
COORD_NOTE = ' All coordinates use 0-1000 normalization; x increases rightward and y increases downward.'
JSON_ONLY = ' Output only the JSON, with no explanation.'
OWN_SUFFIX = {
    'point': ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].' + COORD_NOTE + JSON_ONLY,
    'bbox': ' Give its tight bounding box. Return only [{"bbox_2d": [x_min, y_min, x_max, y_max]}].' + COORD_NOTE + JSON_ONLY,
}


def chat_text(rt: qr.Runtime, messages: list[dict], tools: bool, generation_prompt: bool) -> str:
    return rt.processor.apply_chat_template(messages, tools=[TOOL_SCHEMA] if tools else None, tokenize=False,
                                            add_generation_prompt=generation_prompt, enable_thinking=False)


def user_message(text: str) -> dict:
    return {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}


def tool_prompt(rt: qr.Runtime, user_text: str) -> str:
    """Single-image tool-enabled prompt ending with the (thinking-off) assistant generation prompt."""
    return chat_text(rt, [user_message(user_text)], tools=True, generation_prompt=True)


def call_text(target: str) -> str:
    return CALL_TEMPLATE.format(target=target)


def call_sites(tokenizer, target: str) -> tuple[int, dict[str, int]]:
    """Token count of the canonical call and each site's offset from the END of prompt + call.

    The call starts with the special token <tool_call>, so its tokenization does not depend on the
    preceding prompt (verified by `verify_call_tokenization`).  Offsets are negative indices, which
    stay valid under left padding.
    """
    text = call_text(target)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc['input_ids'], enc['offset_mapping']
    n = len(ids)
    t_start = text.index('<parameter=target>\n') + len('<parameter=target>\n')
    t_last = t_start + len(target) - 1
    k_target = next(i for i, (a, b) in enumerate(offsets) if a <= t_last < b)
    k_end = max(i for i, t in enumerate(ids) if tokenizer.decode([t]) == '</tool_call>')
    return n, {'prompt_end': -n - 1, 'target_end': k_target - n, 'call_end': k_end - n}


def verify_call_tokenization(rt: qr.Runtime, user_text: str, target: str) -> None:
    prompt = tool_prompt(rt, user_text)
    whole = rt.tokenizer(prompt + call_text(target), add_special_tokens=False)['input_ids']
    parts = (rt.tokenizer(prompt, add_special_tokens=False)['input_ids']
             + rt.tokenizer(call_text(target), add_special_tokens=False)['input_ids'])
    if whole != parts:
        raise ValueError('Canonical call tokenization depends on the prompt; site offsets would be wrong.')


def tool_response_text(pred: dict) -> str:
    return json.dumps({'point_2d': pred['point_2d'], 'bbox_2d': pred['bbox_2d']})


def tool_loop_text(rt: qr.Runtime, user_text: str, target: str, response: str) -> str:
    """Prompt for the answer after one tool round: user -> assistant call -> tool response -> assistant."""
    messages = [user_message(user_text),
                {'role': 'assistant', 'content': '',
                 'tool_calls': [{'type': 'function', 'function': {'name': 'locate', 'arguments': {'target': target}}}]},
                {'role': 'tool', 'content': response}]
    return chat_text(rt, messages, tools=True, generation_prompt=True)


def parse_call(text: str) -> str | None:
    """Target argument of a generated locate call, or None if the text holds no well-formed call."""
    m = re.search(r'<tool_call>\s*<function=locate>\s*<parameter=target>\n(.*?)\n</parameter>\s*</function>\s*</tool_call>',
                  text, re.S)
    return m.group(1) if m else None


# ----------------------------------------------------------------------------- generic synthetic scenes

PALETTE = {
    'red': (220, 30, 30), 'green': (30, 160, 50), 'blue': (30, 70, 220), 'yellow': (240, 210, 20),
    'orange': (245, 130, 20), 'purple': (130, 40, 170), 'pink': (240, 120, 180), 'brown': (130, 80, 30),
    'black': (15, 15, 15), 'white': (250, 250, 250), 'gray': (128, 128, 128), 'cyan': (20, 200, 220),
}
SHAPES = ('circle', 'square', 'triangle', 'star', 'cross', 'diamond', 'hexagon', 'ring', 'ellipse', 'rectangle')
EXPRESSIONS = ('the {color} {shape}', 'the {shape} drawn in {color}', 'the {color} {shape} in the image')


def _shape_polygon(shape: str, cx: float, cy: float, w: float, h: float, rot: float) -> list[tuple[float, float]]:
    def pts(radii, n, offset=0.0):
        out = []
        for k in range(n):
            a = offset + rot + 2 * math.pi * k / n
            r = radii[k % len(radii)]
            out.append((cx + r * w / 2 * math.cos(a), cy + r * h / 2 * math.sin(a)))
        return out
    if shape == 'triangle':
        return pts([1.0], 3, -math.pi / 2)
    if shape == 'star':
        return pts([1.0, 0.42], 10, -math.pi / 2)
    if shape == 'diamond':
        return pts([1.0], 4, -math.pi / 2)
    if shape == 'hexagon':
        return pts([1.0], 6)
    if shape == 'cross':
        a, b = w / 2, w / 6
        c, d = h / 2, h / 6
        return [(cx - b, cy - c), (cx + b, cy - c), (cx + b, cy - d), (cx + a, cy - d), (cx + a, cy + d), (cx + b, cy + d),
                (cx + b, cy + c), (cx - b, cy + c), (cx - b, cy + d), (cx - a, cy + d), (cx - a, cy - d), (cx - b, cy - d)]
    raise ValueError(shape)


def draw_shape(size: tuple[int, int], shape: str, box: tuple[float, float, float, float], rot: float,
               outline: bool, stroke: int) -> np.ndarray:
    """Boolean mask (H, W) of one shape inscribed in a pixel box."""
    W, H = size
    mask = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(mask)
    x0, y0, x1, y1 = box
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    kw = {'outline': 255, 'width': stroke} if outline else {'fill': 255}
    if shape in ('circle', 'ellipse'):
        d.ellipse(box, **kw)
    elif shape in ('square', 'rectangle'):
        d.rectangle(box, **kw)
    elif shape == 'ring':
        d.ellipse(box, outline=255, width=max(stroke, int(min(w, h) * 0.18)))
    else:
        poly = _shape_polygon(shape, cx, cy, w, h, rot if shape != 'cross' else 0.0)
        if outline:
            d.line(poly + [poly[0]], fill=255, width=stroke, joint='curve')
        else:
            d.polygon(poly, fill=255)
    return np.asarray(mask) > 127


def _background(rng: np.random.Generator, W: int, H: int) -> np.ndarray:
    kind = rng.choice(['solid', 'gradient', 'noise', 'stripes'], p=[0.4, 0.25, 0.2, 0.15])
    c1 = rng.integers(0, 256, 3)
    if kind == 'solid':
        return np.broadcast_to(c1, (H, W, 3)).astype(np.float32)
    if kind == 'gradient':
        c2 = rng.integers(0, 256, 3)
        a = rng.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:H, 0:W]
        t = (xx * math.cos(a) + yy * math.sin(a))
        t = (t - t.min()) / max(np.ptp(t), 1)
        return (c1[None, None] * (1 - t[..., None]) + c2[None, None] * t[..., None]).astype(np.float32)
    if kind == 'noise':
        small = rng.normal(0, 1, (max(2, H // 64), max(2, W // 64), 3))
        img = Image.fromarray(np.uint8(np.clip(128 + 40 * small, 0, 255))).resize((W, H), Image.Resampling.BICUBIC)
        return np.clip(np.asarray(img, np.float32) * 0.6 + c1 * 0.4, 0, 255)
    c2 = rng.integers(0, 256, 3)
    period = int(rng.integers(12, 80))
    yy, xx = np.mgrid[0:H, 0:W]
    band = ((xx + (yy if rng.random() < 0.5 else 0)) // period) % 2
    col = np.where(band[..., None] == 0, c1, (0.7 * c1 + 0.3 * c2))
    return col.astype(np.float32)


def generic_scene(seed: int) -> tuple[Image.Image, dict]:
    """One random scene with a uniquely described target object.

    Canvas sides are multiples of 32 in [256, 1344] with aspect ratio <= 2.5; 1-14 objects of random
    shape, colour, size (16 px up to 40% of the short side), fill or outline, placed at random
    non-overlapping positions.  The target has a (colour, shape) pair unique in the scene.  Scenes
    that would reproduce the experiment-3.6 disc task (a lone red circle on a near-white canvas) are
    rejected by construction; grids and rings are never produced because positions are random.
    """
    rng = np.random.default_rng(seed)
    while True:
        W, H = (int(rng.integers(8, 43)) * 32 for _ in range(2))
        if max(W, H) / min(W, H) <= 2.5:
            break
    bg = _background(rng, W, H)
    n = int(rng.integers(1, 15))
    short = min(W, H)
    objects, boxes = [], []
    for _ in range(n * 20):
        if len(objects) == n:
            break
        shape = SHAPES[int(rng.integers(len(SHAPES)))]
        s = float(np.exp(rng.uniform(np.log(16), np.log(0.4 * short))))
        aspect = float(np.exp(rng.uniform(-0.6, 0.6))) if shape in ('ellipse', 'rectangle', 'cross', 'diamond') else 1.0
        w, h = s * math.sqrt(aspect), s / math.sqrt(aspect)
        if w >= W - 4 or h >= H - 4:
            continue
        x0, y0 = rng.uniform(2, W - w - 2), rng.uniform(2, H - h - 2)
        box = (x0, y0, x0 + w, y0 + h)
        if any(not (box[2] + 4 < b[0] or b[2] + 4 < box[0] or box[3] + 4 < b[1] or b[3] + 4 < box[1]) for b in boxes):
            continue
        color = list(PALETTE)[int(rng.integers(len(PALETTE)))]
        objects.append({'shape': shape, 'color': color, 'box': box, 'rot': float(rng.uniform(0, 2 * math.pi)),
                        'outline': bool(rng.random() < 0.2), 'stroke': int(rng.integers(2, 6))})
        boxes.append(box)
    pairs = [(o['color'], o['shape']) for o in objects]
    unique = [i for i, p in enumerate(pairs) if pairs.count(p) == 1]
    target = int(unique[int(rng.integers(len(unique)))]) if unique else None
    img = bg.copy()
    masks = []
    for o in objects:
        m = draw_shape((W, H), o['shape'], o['box'], o['rot'], o['outline'], o['stroke'])
        img[m] = PALETTE[o['color']]
        masks.append(m)
    o = objects[target] if target is not None else None
    lone_red_disc = (o is not None and len(objects) == 1 and o['shape'] == 'circle' and o['color'] == 'red'
                     and bg.mean() > 230)
    tmask = masks[target] if target is not None else None
    # Objects have disjoint boxes, so the target mask is fully visible; require colour contrast with
    # the background under the target so that the description refers to a visible object.
    low_contrast = target is not None and float(np.abs(bg[tmask] - np.asarray(PALETTE[o['color']])).mean()) < 40
    if target is None or lone_red_disc or tmask.sum() < 30 or low_contrast:
        return generic_scene(seed + 7919 * 1000003)
    image = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
    if rng.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.5))))
    ys, xs = np.nonzero(tmask)
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    expr = EXPRESSIONS[int(rng.integers(len(EXPRESSIONS)))].format(color=o['color'], shape=o['shape'])
    return image, {'canvas': [W, H], 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                   'target': expr, 'n_objects': len(objects), 'shape': o['shape'], 'color': o['color'],
                   'hit': {'kind': 'box', 'box': box}}


# ----------------------------------------------------------------------------- RefCOCO

def resize_long(image: Image.Image, longest: int) -> Image.Image:
    """Same rule as experiment 3.4: long side to `longest`, both sides rounded to multiples of 32."""
    ratio = longest / max(image.size)
    size = tuple(max(32, int(round(v * ratio / 32)) * 32) for v in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def refcoco_index(refcoco_dir: Path) -> tuple[dict, list[dict]]:
    """All RefCOCO rows grouped by image digest, deduplicated by (digest, rounded bbox).

    Returns (digest -> (shard, row) of one row holding the image bytes, list of reference records).
    """
    import pyarrow.parquet as pq
    shards = sorted((refcoco_dir / 'data').glob('*.parquet'))
    if not shards:
        raise FileNotFoundError(f'No RefCOCO parquet shards under {refcoco_dir}/data')
    image_of, refs, seen = {}, [], set()
    for shard in shards:
        table = pq.read_table(shard, columns=['image', 'question_id', 'answer', 'bbox', 'file_name'])
        imgs = table.column('image').to_pylist()
        for idx, (img, qid, ans, bbox, name) in enumerate(zip(imgs, *(table.column(k).to_pylist() for k in
                                                                     ('question_id', 'answer', 'bbox', 'file_name')))):
            digest = hashlib.sha256(img['bytes']).hexdigest()
            image_of.setdefault(digest, (shard.name, idx))
            key = (digest, tuple(round(v, 1) for v in bbox))
            if key in seen or not ans:
                continue
            seen.add(key)
            refs.append({'digest': digest, 'shard': shard.name, 'row': idx, 'question_id': qid,
                         'expression': ans[0], 'bbox_xywh': [float(v) for v in bbox], 'file_name': name})
    return image_of, refs


def load_refcoco_image(refcoco_dir: Path, shard: str, row: int, cache: dict) -> Image.Image:
    import pyarrow.parquet as pq
    if shard not in cache:
        cache.clear()       # keep one shard's image column in memory at a time
        cache[shard] = pq.read_table(refcoco_dir / 'data' / shard, columns=['image']).column('image')
    return Image.open(io.BytesIO(cache[shard][row].as_py()['bytes'])).convert('RGB')


def refcoco_truth(ref: dict, source_size: tuple[int, int], input_size: tuple[int, int]) -> dict:
    sw, sh = source_size
    iw, ih = input_size
    x, y, w, h = ref['bbox_xywh']
    box = [x * iw / sw, y * ih / sh, (x + w) * iw / sw, (y + h) * ih / sh]
    return {'canvas': [iw, ih], 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
            'hit': {'kind': 'box', 'box': box}}


# ----------------------------------------------------------------------------- held-out synthetic sets

T6_TARGET = 'the figure drawn in red'
DISC_TARGET = 'the red circle'


def grid_truth(t: dict) -> dict:
    """Experiment-1.3 / supplement grid truth -> common truth (tight box, cell hit box, geometry)."""
    box = [float(v) for v in t['target_tight_box']]
    out = {'canvas': list(t['canvas']), 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
           'hit': {'kind': 'box', 'box': [float(v) for v in t['target_cell_box']]}}
    if 'rows' in t:
        out['grid'] = {'rows': t['rows'], 'cols': t['cols'], 'origin': list(t['grid_origin']), 'cell': t['cell'],
                       'target_rc': list(t['target_rc'])}
    if 'instance_centers' in t:
        out['instances'] = [list(c) for c in t['instance_centers']]
        out['target_index'] = t['target_index']
    return out


def disc_truth(t: dict) -> dict:
    box = list(t['box'])
    return {'canvas': list(t['canvas']), 'box': box, 'center': list(t['center']),
            'hit': {'kind': 'disc', 'center': list(t['center']), 'radius': t['diameter'] / 2}}


# ----------------------------------------------------------------------------- scoring

def to_units(xy, canvas) -> np.ndarray:
    W, H = canvas
    return 1000 * np.asarray(xy, float) / np.asarray([W, H] * (len(xy) // 2), float)


def to_pixels(units, canvas) -> np.ndarray:
    W, H = canvas
    return np.asarray(units, float) * np.asarray([W, H] * (len(units) // 2), float) / 1000


def inside(px: np.ndarray, hit: dict) -> bool:
    if hit['kind'] == 'disc':
        return float(np.linalg.norm(px - np.asarray(hit['center']))) <= hit['radius']
    x0, y0, x1, y1 = hit['box']
    return bool(x0 <= px[0] < x1 and y0 <= px[1] < y1)


def box_iou(a, b) -> float:
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.


def score_point(truth: dict, point_units) -> dict:
    """Point answer in 0-1000 units -> strict hit, pixel-convention hit, centre error, nearest instance."""
    if point_units is None:
        return {'valid': False, 'hit': False, 'hit_any': False}
    p = np.asarray(point_units, float)
    px = to_pixels(p, truth['canvas'])
    hit = inside(px, truth['hit'])
    W, H = truth['canvas']
    hit_pix = bool(p[0] < W and p[1] < H and inside(p, truth['hit']))
    out = {'valid': True, 'point': p.tolist(), 'hit': hit, 'hit_pix_only': bool(hit_pix and not hit),
           'hit_any': bool(hit or hit_pix), 'error': float(np.linalg.norm(p - to_units(truth['center'], truth['canvas']))),
           'dx': float(p[0] - to_units(truth['center'], truth['canvas'])[0]),
           'dy': float(p[1] - to_units(truth['center'], truth['canvas'])[1])}
    if 'instances' in truth:
        d = np.linalg.norm(np.asarray(truth['instances']) - px, axis=1)
        out['nearest_correct'] = bool(int(d.argmin()) == truth['target_index'])
    if 'grid' in truth:
        g = truth['grid']
        col = int((px[0] - g['origin'][0]) // g['cell'])
        row = int((px[1] - g['origin'][1]) // g['cell'])
        out['pred_rc'] = [row, col]
    return out


def answer_center(coords) -> list[float]:
    """Point answer, or the centre of a bbox answer, in the same units."""
    c = np.asarray(coords, float)
    return (c if len(c) == 2 else (c[:2] + c[2:]) / 2).tolist()


def score_box(truth: dict, box_units) -> dict:
    if box_units is None:
        return {'valid': False, 'iou': 0.0, 'iou50': False}
    b = np.asarray(box_units, float)
    if not (b[2] > b[0] and b[3] > b[1]):
        return {'valid': False, 'iou': 0.0, 'iou50': False}
    iou = box_iou(to_pixels(b, truth['canvas']).tolist(), truth['box'])
    return {'valid': True, 'box': b.tolist(), 'iou': iou, 'iou50': iou >= 0.5}


def parse_answer(text: str, fmt: str) -> list[float] | None:
    """Strict parse of the model's own point / bbox JSON answer (same acceptance as experiment 3.5)."""
    key, n = ('point_2d', 2) if fmt == 'point' else ('bbox_2d', 4)
    body = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.split('</think>')[-1].strip())
    try:
        value = json.loads(body)
    except ValueError:
        return None
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]
    if isinstance(value, dict):
        value = value.get(key)
    if not (isinstance(value, list) and len(value) == n and all(type(v) in (int, float) for v in value)):
        return None
    return [float(v) for v in value]


def parse_rowcol_free(text: str) -> tuple[int, int] | None:
    """Row/column from a direct JSON answer or, for think-then-answer text, the last JSON object."""
    body = text.split('</think>')[-1]
    for match in reversed(list(re.finditer(r'\{[^{}]*\}', body))):
        try:
            v = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(v, dict) and type(v.get('row')) is int and type(v.get('column')) is int:
            return v['row'], v['column']
    return None


# ----------------------------------------------------------------------------- batched capture

ATTN_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)           # the full-attention decoder layers
ATTN_FEATURES = ('mass', 'cx', 'cy', 'sx', 'sy', 'px', 'py')


def _attention_features(mod, h: torch.Tensor, pe, idx: torch.Tensor, valid: torch.Tensor, vis: torch.Tensor,
                        centers: list[torch.Tensor], map_site: int | None = None):
    """Softmax attention of full-attention heads from the site queries, summarized over visual tokens.

    h: (B, L, D) normalized input of self_attn; pe: rotary (cos, sin) of shape (B, L, r); idx: (B, S)
    query positions; valid: (B, L) attention mask; vis: (B, L) visual-token mask; centers[b]: (Nv_b, 2)
    token centres in 0-1000 units in raster order.  Returns (B, S, heads, 7): visual mass, attention
    centroid (cx, cy), spread (sx, sy) and the centre of the most attended visual token (px, py); and,
    when map_site is given, the list of raw visual attention maps (heads, Nv_b) of that site.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    B, L, _ = h.shape
    hd = mod.head_dim
    cos, sin = pe
    hq = torch.gather(h, 1, idx[..., None].expand(-1, -1, h.shape[-1]))                    # (B, S, D)
    q, _ = torch.chunk(mod.q_proj(hq).view(B, idx.shape[1], -1, hd * 2), 2, dim=-1)
    q = mod.q_norm(q).transpose(1, 2)                                                       # (B, H, S, hd)
    cq = torch.gather(cos, 1, idx[..., None].expand(-1, -1, cos.shape[-1]))
    sq = torch.gather(sin, 1, idx[..., None].expand(-1, -1, sin.shape[-1]))
    q, _ = apply_rotary_pos_emb(q, q, cq, sq)
    k = mod.k_norm(mod.k_proj(h).view(B, L, -1, hd)).transpose(1, 2)
    _, k = apply_rotary_pos_emb(k, k, cos, sin)
    k = repeat_kv(k, mod.num_key_value_groups)
    scores = (q.float() @ k.float().transpose(2, 3)) * mod.scaling                          # (B, H, S, L)
    pos = torch.arange(L, device=h.device)
    allowed = valid[:, None, None, :].bool() & (pos[None, None, None, :] <= idx[:, None, :, None])
    a = scores.masked_fill(~allowed, float('-inf')).softmax(-1)
    out, maps = [], []
    for b in range(B):
        av = a[b][..., vis[b]]                                                              # (H, S, Nv)
        if map_site is not None:
            maps.append(av[:, map_site].half().cpu())
        c = centers[b].to(av.device)
        mass = av.sum(-1, keepdim=True)
        w = av / mass.clamp_min(1e-12)
        mu = w @ c                                                                          # (H, S, 2)
        var = w @ (c ** 2) - mu ** 2
        peak = c[av.argmax(-1)]                                                             # (H, S, 2)
        f = torch.cat([mass, mu, var.clamp_min(0).sqrt(), peak], -1)                        # (H, S, 7)
        out.append(f.transpose(0, 1))
    return torch.stack(out).cpu(), maps                                                     # (B, S, H, 7)


def token_centers(grid_thw: list[int]) -> torch.Tensor:
    _, h, w = grid_thw
    gh, gw = h // 2, w // 2
    r, c = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing='ij')
    return torch.stack([(c.flatten() + 0.5) * 1000 / gw, (r.flatten() + 0.5) * 1000 / gh], -1).float()


@torch.inference_mode()
def capture_sites(rt: qr.Runtime, texts: list[str], images: list[Image.Image], offsets: list[dict[str, int]],
                  layers=CAPTURE_LAYERS, sites=SITES, attn_layers=(), map_site: str | None = None):
    """Teacher-forced left-padded batch; returns float16 array (B, len(sites), len(layers), D).

    offsets[b][site] is a negative index from the end of sample b's token sequence.  With attn_layers,
    also returns float32 attention features (B, len(sites), len(attn_layers), heads, 7) and, for
    map_site, a list of float16 visual attention maps of shape (len(attn_layers) * heads, Nv_b).
    """
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    grids = batch['image_grid_thw'].tolist()
    for image, (_, h, w) in zip(images, grids):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError(f'Processor resized a {image.width}x{image.height} image.')
    L = batch['input_ids'].shape[1]
    idx = torch.tensor([[L + o[s] for s in sites] for o in offsets], device=rt.device)       # (B, S)
    vis = batch['input_ids'] == rt.model.config.image_token_id
    centers = [token_centers(g) for g in grids]
    if any(int(v.sum()) != len(c) for v, c in zip(vis, centers)):
        raise ValueError('Visual token count does not match image_grid_thw.')
    store, attn, handles = {}, {}, []
    for layer in layers:
        def hook(module, inputs, output, layer=layer):
            h = output[0] if isinstance(output, tuple) else output
            store[layer] = torch.gather(h, 1, idx.to(h.device)[..., None].expand(-1, -1, h.shape[-1])).half().cpu()
        handles.append(rt.layers[layer - 1].register_forward_hook(hook))
    for layer in attn_layers:
        mod = rt.layers[layer - 1].self_attn
        def pre(module, args, kwargs, layer=layer):
            h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            pe = kwargs['position_embeddings']
            d = h.device
            attn[layer] = _attention_features(module, h, pe, idx.to(d), batch['attention_mask'].to(d), vis.to(d), centers,
                                              None if map_site is None else sites.index(map_site))
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        rt.model(**batch, use_cache=False, logits_to_keep=1)
    finally:
        for hd in handles:
            hd.remove()
    hidden = torch.stack([store[l] for l in layers], 2).numpy()                              # (B, S, L, D)
    if not attn_layers:
        return hidden
    feats = torch.stack([attn[l][0] for l in attn_layers], 2).numpy()                      # (B, S, La, H, 7)
    if map_site is None:
        return hidden, feats
    maps = [torch.cat([attn[l][1][b] for l in attn_layers], 0).numpy() for b in range(len(texts))]
    return hidden, feats, maps


# ----------------------------------------------------------------------------- regressors

TARGET_NAMES = ('cx', 'cy', 'x0', 'y0', 'x1', 'y1')


def targets_units(truth: dict) -> np.ndarray:
    """Regression target (6,) in 0-1000 units: box centre then box corners."""
    c = to_units(truth['center'], truth['canvas'])
    b = to_units(truth['box'], truth['canvas'])
    return np.concatenate([c, b])


@dataclass
class Ridge:
    mean: np.ndarray        # (D,)
    scale: np.ndarray       # (D,)
    weight: np.ndarray      # (D, 6)
    bias: np.ndarray        # (6,)
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.scale) @ self.weight + self.bias


def fit_ridge(X: np.ndarray, Y: np.ndarray, alphas, Xv: np.ndarray, Yv: np.ndarray) -> tuple[Ridge, dict]:
    """Standardized ridge; alpha chosen by validation mean centre error.  Solved in float64 via eigh."""
    mean = X.mean(0)
    scale = X.std(0) + 1e-3
    Z = ((X - mean) / scale).astype(np.float64)
    ym = Y.mean(0)
    G = Z.T @ Z
    evals, evecs = np.linalg.eigh(G)
    ZtY = Z.T @ (Y - ym)
    proj = evecs.T @ ZtY
    Zv = ((Xv - mean) / scale).astype(np.float64)
    curve, best = {}, None
    for a in alphas:
        W = evecs @ (proj / (evals + a)[:, None])
        P = Zv @ W + ym
        err = float(np.linalg.norm(P[:, :2] - Yv[:, :2], axis=1).mean())
        curve[float(a)] = err
        if best is None or err < best[0]:
            best = (err, a, W)
    return Ridge(mean, scale, best[2], ym, float(best[1])), curve


class MLP(torch.nn.Module):
    def __init__(self, dim: int, hidden: int = 1024, out: int = 6, dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                       torch.nn.Dropout(dropout), torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x) * 500.0 + 500.0


def fit_mlp(X: np.ndarray, Y: np.ndarray, Xv: np.ndarray, Yv: np.ndarray, seed: int, epochs: int = 60,
            hidden: int = 1024, lr: float = 1e-3, wd: float = 0.05, batch: int = 256, device: str = 'cpu') -> tuple[MLP, dict]:
    """Huber-loss MLP; keeps the epoch with the lowest validation centre error."""
    torch.manual_seed(seed)
    model = MLP(X.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    Xvt = torch.tensor(Xv, dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    history, best = [], (float('inf'), None, -1)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for k in range(0, len(perm), batch):
            j = perm[k:k + batch]
            loss = torch.nn.functional.huber_loss(model(Xt[j].to(device)), Yt[j].to(device), delta=20.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            P = model(Xvt).cpu().numpy()
        err = float(np.linalg.norm(P[:, :2] - Yv[:, :2], axis=1).mean())
        history.append(err)
        if err < best[0]:
            best = (err, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    model.cpu().eval()
    return model, {'val_error_by_epoch': history, 'best_epoch': best[2]}


# ----------------------------------------------------------------------------- pointer readout

def token_features(maps: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Attention maps (C, Nv) of C full-attention heads -> per-token features (Nv, 2C).

    The first C features are log(Nv * a + 1e-3), the attention relative to a uniform distribution on a
    log scale (independent of the number of visual tokens); the next C are their 3x3 neighbourhood
    means on the token grid (edge padded), which give each token the context of its neighbours.
    """
    gh, gw = grid
    C, nv = maps.shape
    if nv != gh * gw:
        raise ValueError(f'map length {nv} does not match grid {grid}')
    f = np.log(maps.astype(np.float32) * nv + 1e-3).reshape(C, gh, gw)
    pad = np.pad(f, ((0, 0), (1, 1), (1, 1)), mode='edge')
    nb = sum(pad[:, dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3)) / 9
    return np.concatenate([f, nb], 0).reshape(2 * C, nv).T


def token_overlap(box_units, grid: tuple[int, int]) -> np.ndarray:
    """Fraction (Nv,) of each merged token's cell covered by a box given in 0-1000 units."""
    gh, gw = grid
    x0, y0, x1, y1 = box_units
    ex, ey = np.linspace(0, 1000, gw + 1), np.linspace(0, 1000, gh + 1)
    ox = np.clip(np.minimum(ex[1:], x1) - np.maximum(ex[:-1], x0), 0, None) / (1000 / gw)
    oy = np.clip(np.minimum(ey[1:], y1) - np.maximum(ey[:-1], y0), 0, None) / (1000 / gh)
    return (oy[:, None] * ox[None, :]).reshape(-1)


def grid_of(canvas) -> tuple[int, int]:
    W, H = canvas
    return H // 32, W // 32


class PointerScorer(torch.nn.Module):
    """Per-token MLP on attention features -> (pointer logit, mask logit)."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.GELU(), torch.nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def _pad_batch(items, mean, std):
    n = max(len(it['F']) for it in items)
    B, D = len(items), items[0]['F'].shape[1]
    F = torch.zeros(B, n, D)
    C = torch.zeros(B, n, 2)
    Q = torch.zeros(B, n)
    M = torch.zeros(B, n, dtype=torch.bool)
    for b, it in enumerate(items):
        k = len(it['F'])
        F[b, :k] = (torch.tensor(it['F'], dtype=torch.float32) - mean) / std
        C[b, :k] = torch.tensor(it['centers'])
        if 'q' in it:
            Q[b, :k] = torch.tensor(it['q'])
        M[b, :k] = True
    return F, C, Q, M


def pointer_forward(model, F, C, M):
    out = model(F)
    s = out[..., 0].masked_fill(~M, float('-inf'))
    alpha = s.softmax(-1)
    point = (alpha[..., None] * C).sum(1)
    return alpha, point, out[..., 1]


def pointer_box(mask_logit: np.ndarray, alpha: np.ndarray, grid: tuple[int, int], tau: float) -> list[float]:
    """Extent of the token cells whose mask probability is >= tau (argmax pointer token if none)."""
    gh, gw = grid
    p = 1 / (1 + np.exp(-mask_logit))
    sel = np.nonzero(p >= tau)[0]
    if len(sel) == 0:
        sel = np.asarray([int(alpha.argmax())])
    r, c = sel // gw, sel % gw
    return [c.min() * 1000 / gw, r.min() * 1000 / gh, (c.max() + 1) * 1000 / gw, (r.max() + 1) * 1000 / gh]


def fit_pointer(train: list[dict], val: list[dict], seed: int = 0, epochs: int = 30, hidden: int = 64,
                lr: float = 2e-3, lam: float = 1.0, batch: int = 64, taus=(0.3, 0.4, 0.5, 0.6, 0.7)) -> tuple[dict, dict]:
    """Train the pointer scorer.  Items: F (Nv, D), centers (Nv, 2), q (Nv,) target overlap, y (6,), grid.

    Loss = listwise cross-entropy between softmax(pointer logits) and q / sum(q), plus lam times the
    L1 error of the pointer centroid (in units of 100), plus binary cross-entropy of the mask logits
    against [q >= 0.5].  The epoch with the lowest validation centre error is kept; the mask threshold
    tau is then chosen by validation mean IoU.
    """
    torch.manual_seed(seed)
    allF = np.concatenate([it['F'][::4] for it in train]).astype(np.float32)
    mean, std = torch.tensor(allF.mean(0)), torch.tensor(allF.std(0) + 1e-3)
    model = PointerScorer(train[0]['F'].shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    order = sorted(range(len(train)), key=lambda i: len(train[i]['F']))
    groups = [order[k:k + batch] for k in range(0, len(order), batch)]
    rng = np.random.default_rng(seed)
    best, history = (float('inf'), None, -1), []
    for ep in range(epochs):
        model.train()
        for gi in rng.permutation(len(groups)):
            F, C, Q, M = _pad_batch([train[i] for i in groups[gi]], mean, std)
            y = torch.tensor(np.stack([train[i]['y'][:2] for i in groups[gi]]), dtype=torch.float32)
            alpha, point, mlog = pointer_forward(model, F, C, M)
            qn = Q / Q.sum(1, keepdim=True).clamp_min(1e-6)
            ce = -(qn * torch.log(alpha.clamp_min(1e-9))).sum(1).mean()
            l1 = (point - y).abs().sum(1).mean() / 100
            bce = torch.nn.functional.binary_cross_entropy_with_logits(mlog[M], (Q[M] >= 0.5).float())
            loss = ce + lam * l1 + bce
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        P = pointer_predict_items(model, mean, std, val, tau=0.5)
        err = float(np.mean([np.linalg.norm(p[:2] - it['y'][:2]) for p, it in zip(P, val)]))
        history.append(err)
        if err < best[0]:
            best = (err, {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    model.eval()
    ious = {}
    for tau in taus:
        P = pointer_predict_items(model, mean, std, val, tau=tau)
        ious[tau] = float(np.mean([box_iou(p[2:], it['y'][2:]) for p, it in zip(P, val)]))
    tau = max(ious, key=ious.get)
    params = {'dim': train[0]['F'].shape[1], 'hidden': hidden, 'state': model.state_dict(), 'mean': mean, 'std': std, 'tau': tau}
    return params, {'val_error_by_epoch': history, 'best_epoch': best[2], 'val_iou_by_tau': ious, 'tau': tau}


@torch.no_grad()
def pointer_predict_items(model, mean, std, items: list[dict], tau: float, batch: int = 64) -> np.ndarray:
    """(N, 6) predictions: pointer centroid then the thresholded token-cell box."""
    out = np.zeros((len(items), 6))
    order = sorted(range(len(items)), key=lambda i: len(items[i]['F']))
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        F, C, _, M = _pad_batch([items[i] for i in idx], mean, std)
        alpha, point, mlog = pointer_forward(model, F, C, M)
        for j, i in enumerate(idx):
            n = len(items[i]['F'])
            out[i, :2] = point[j].numpy()
            out[i, 2:] = pointer_box(mlog[j, :n].numpy(), alpha[j, :n].numpy(), items[i]['grid'], tau)
    return out


class ReadoutTool:
    """Online tool: model internals at the readout site -> point_2d / bbox_2d text in 0-1000 units.

    kind: 'ridge' or 'mlp' (hidden states of `layers` concatenated, 1-based), or 'pointer' (attention
    maps of the full-attention layers at the target_end site, scored per visual token).
    Predictions are clipped to [0, 1000] and rounded to integers, the output format of the model.
    """

    def __init__(self, kind: str, site: str, layers: tuple[int, ...], params: dict):
        self.kind, self.site, self.layers = kind, site, tuple(layers)
        if kind == 'ridge':
            self.ridge = Ridge(**{k: np.asarray(v) if k != 'alpha' else v for k, v in params.items()})
        elif kind == 'pointer':
            self.pointer = PointerScorer(params['dim'], params['hidden'])
            self.pointer.load_state_dict(params['state'])
            self.pointer.eval()
            self.pmean, self.pstd, self.tau = params['mean'], params['std'], params['tau']
        else:
            self.mlp = MLP(params['dim'], params['hidden'])
            self.mlp.load_state_dict(params['state'])
            self.mlp.eval()

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        if self.kind == 'ridge':
            return self.ridge.predict(X.astype(np.float64))
        with torch.no_grad():
            return self.mlp(torch.tensor(X, dtype=torch.float32)).numpy()

    def predict_maps(self, maps: list[np.ndarray], grids: list[tuple[int, int]]) -> np.ndarray:
        """Pointer tool: attention maps (C, Nv) at the target_end site -> (N, 6) raw predictions."""
        items = [{'F': token_features(m, g), 'centers': token_centers((1, 2 * g[0], 2 * g[1])).numpy(), 'grid': g}
                 for m, g in zip(maps, grids)]
        return pointer_predict_items(self.pointer, self.pmean, self.pstd, items, self.tau)

    @staticmethod
    def format(raw: np.ndarray) -> dict:
        v = np.clip(np.rint(raw), 0, 1000).astype(int)
        box = [int(min(v[2], v[4])), int(min(v[3], v[5])), int(max(v[2], v[4])), int(max(v[3], v[5]))]
        return {'point_2d': [int(v[0]), int(v[1])], 'bbox_2d': box}

    def save(self, path: Path, meta: dict) -> None:
        if self.kind == 'ridge':
            params = {k: getattr(self.ridge, k) for k in ('mean', 'scale', 'weight', 'bias', 'alpha')}
        elif self.kind == 'pointer':
            params = {'dim': self.pointer.net[0].in_features, 'hidden': self.pointer.net[0].out_features,
                      'state': self.pointer.state_dict(), 'mean': self.pmean, 'std': self.pstd, 'tau': self.tau}
        else:
            params = {'dim': self.mlp.net[1].in_features, 'hidden': self.mlp.net[1].out_features,
                      'state': self.mlp.state_dict()}
        torch.save({'kind': self.kind, 'site': self.site, 'layers': self.layers, 'params': params, 'meta': meta}, path)

    @classmethod
    def load(cls, path: Path) -> 'ReadoutTool':
        d = torch.load(path, weights_only=False)
        return cls(d['kind'], d['site'], d['layers'], d['params'])
