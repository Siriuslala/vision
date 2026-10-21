"""Experiment 3.9 round 2: coordinate readout tool, revised.

Design: docs/analysis/position/design_readout_tool.md (section "第二轮").

Round 2 changes the tool along the three axes asked for by the PI review of round 1:

* interface (variant B with a model-filled argument): the ``locate`` schema gains a required
  ``approx_point`` argument.  The model writes its own rough [x, y] inside the call; the tool reads
  the attention of the token that predicts the first digit of that argument (the ``slot`` site, the
  call-time analogue of the experiment-3.6 answer slot x0), and may also use the rough point itself;
* reader: a small convolutional pointer over the visual-token grid (receptive field of ~15 tokens)
  with sub-token centre offsets and per-token box-edge regression, replacing the per-token MLP and
  the thresholded token-cell box of round 1;
* data: phrasing variants of the user question, alternative RefCOCO expressions, and a graded
  synthetic family (flat shapes -> textured shapes -> clutter with occlusion -> charts and UI
  panels -> shapes composited onto natural photographs), also used to delimit where the tool helps.

Cheaper inputs that need no attention weights (hidden states at the slot, merger / decoder visual
token features under a learned query, residual correction of the model's rough point) are fitted on
the same capture for the cost comparison.

Layer numbers are 1-based (output of decoder layer l); layer 0 is the decoder input (merger output
for visual tokens).
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import readout_tool as rt_

# ----------------------------------------------------------------------------- protocol

TOOL_SCHEMA_R2 = copy.deepcopy(rt_.TOOL_SCHEMA)
TOOL_SCHEMA_R2['function']['parameters']['properties']['approx_point'] = {
    'type': 'array', 'items': {'type': 'integer'},
    'description': ('Your rough estimate [x, y] of the object center in 0-1000 normalized coordinates '
                    '(x rightward, y downward).')}
TOOL_SCHEMA_R2['function']['parameters']['required'] = ['target', 'approx_point']

CALL_PREFIX = '<tool_call>\n<function=locate>\n<parameter=target>\n{target}\n</parameter>\n<parameter=approx_point>\n['
CALL_SUFFIX = ']\n</parameter>\n</function>\n</tool_call>'
SITES_R2 = ('target_end', 'slot')
HIDDEN_LAYERS = (16, 20, 24, 28)
CONTENT_LAYERS = (0, 16)            # visual-token features projected to PCA_DIM (0 = merger output)
PCA_DIM = 64

# Question phrasings.  Index 0 is the canonical round-1 question; 1-5 are mixed into training;
# HELDOUT is only used at test time to measure phrasing robustness.
PHRASINGS = (
    'Locate the object described as: {target}. Call the locate tool to find it.',
    'Where is {target}? Use the locate tool.',
    'Find {target} in the image. Call the locate tool to find it.',
    'I need the position of {target}. Please call the locate tool.',
    'Point to {target} using the locate tool.',
    'Use the locate tool to get the location of {target}.',
)
HELDOUT_PHRASING = 'Can you tell me where {target} is? Call the locate tool first.'


def tool_prompt_r2(rt: qr.Runtime, user_text: str) -> str:
    msgs = [rt_.user_message(user_text)]
    return rt.processor.apply_chat_template(msgs, tools=[TOOL_SCHEMA_R2], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)


def call_prefix(target: str) -> str:
    return CALL_PREFIX.format(target=target)


def prefix_sites(tokenizer, target: str) -> dict[str, int]:
    """Offsets (negative, from the end of prompt + call prefix) of the target_end and slot tokens.

    The slot is the last prefix token, whose next-token prediction is the first digit of approx_point.
    The prefix starts with the special token <tool_call>, so its tokens do not depend on the prompt.
    """
    text = call_prefix(target)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    n = len(ids)
    t_start = text.index('<parameter=target>\n') + len('<parameter=target>\n')
    t_last = t_start + len(target) - 1
    k_target = next(i for i, (a, b) in enumerate(offs) if a <= t_last < b)
    return {'target_end': k_target - n, 'slot': -1}


def verify_prefix_tokenization(rt: qr.Runtime, user_text: str, target: str, approx=(123, 456)) -> None:
    """The prefix tokens must be a prefix of the tokens of a complete call with digits filled in, so
    that the teacher-forced slot equals the state the model reaches when it writes the call itself."""
    prompt = tool_prompt_r2(rt, user_text)
    full = rt.tokenizer(prompt + call_prefix(target) + f'{approx[0]}, {approx[1]}' + CALL_SUFFIX, add_special_tokens=False)['input_ids']
    pre = rt.tokenizer(prompt + call_prefix(target), add_special_tokens=False)['input_ids']
    parts = rt.tokenizer(prompt, add_special_tokens=False)['input_ids'] + rt.tokenizer(call_prefix(target), add_special_tokens=False)['input_ids']
    if full[:len(pre)] != pre or pre != parts:
        raise ValueError('Call-prefix tokenization is not stable; the slot site would not match free generation.')


APPROX_RE = re.compile(r'^\s*(-?\d+)\s*,\s*(-?\d+)\s*\]')


def parse_approx(continuation: str) -> list[int] | None:
    """Rough point written by the model after the slot ('225, 552]...'), or None."""
    m = APPROX_RE.match(continuation)
    return [int(m.group(1)), int(m.group(2))] if m else None


# ----------------------------------------------------------------------------- graded synthetic family

LEVELS = ('flat', 'texture', 'clutter', 'chart_ui', 'photo_bg')
UI_WORDS = ('Save', 'Open', 'Cancel', 'Print', 'Share', 'Delete', 'Edit', 'Help', 'Search', 'Upload', 'Settings',
            'Export', 'Login', 'Next', 'Back', 'Close', 'Send', 'Copy', 'Paste', 'Undo', 'Refresh', 'Apply')
BAR_LABELS = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L')
MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


def _canvas(rng, lo=12, hi=33, max_ratio=2.0) -> tuple[int, int]:
    while True:
        W, H = (int(rng.integers(lo, hi)) * 32 for _ in range(2))
        if max(W, H) / min(W, H) <= max_ratio:
            return W, H


def _font(size: int):
    return ImageFont.load_default(size=size)


def _texture_fill(rng, mask: np.ndarray, color, box) -> np.ndarray:
    """RGB values (n_pixels, 3) for the pixels of `mask`: a texture whose dominant hue is `color`."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    c = np.asarray(color, np.float32)
    x0, y0, x1, y1 = box
    u = (xs - x0) / max(x1 - x0, 1)
    v = (ys - y0) / max(y1 - y0, 1)
    kinds = ['stripes', 'checker', 'noise', 'gradient']
    if float(np.ptp(c)) > 60:                    # shading and highlights would change achromatic colour names
        kinds.append('sphere')
    kind = rng.choice(kinds)
    if kind == 'stripes':
        period = rng.uniform(0.12, 0.3)
        a = rng.uniform(0, math.pi)
        t = ((u * math.cos(a) + v * math.sin(a)) / period) % 1.0 < 0.5
        f = np.where(t, 1.0, 0.62)
    elif kind == 'checker':
        k = int(rng.integers(3, 7))
        f = np.where(((u * k).astype(int) + (v * k).astype(int)) % 2 == 0, 1.0, 0.65)
    elif kind == 'noise':
        f = np.clip(0.85 + 0.15 * rng.normal(0, 1, len(xs)), 0.55, 1.15)
    elif kind == 'sphere':
        d = np.hypot(u - 0.35, v - 0.35)
        f = np.clip(1.25 - 0.9 * d, 0.45, 1.2)
    else:
        f = 0.6 + 0.45 * u
    rgb = c[None] * f[:, None]
    if kind == 'sphere':                         # specular highlight towards white
        hl = np.clip(1 - np.hypot(u - 0.3, v - 0.3) / 0.18, 0, 1)[:, None]
        rgb = rgb * (1 - 0.6 * hl) + 255 * 0.6 * hl
    return np.clip(rgb, 0, 255)


def _textured_background(rng, W: int, H: int) -> np.ndarray:
    kind = rng.choice(['fine_noise', 'wood', 'checker', 'paper'])
    c1 = rng.integers(40, 230, 3).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    if kind == 'fine_noise':
        img = c1 + rng.normal(0, 22, (H, W, 3))
    elif kind == 'wood':
        period = rng.uniform(10, 40)
        warp = 6 * np.sin(yy / rng.uniform(20, 60))
        t = 0.5 + 0.5 * np.sin((xx + warp) / period * 2 * math.pi)
        img = c1[None, None] * (0.75 + 0.25 * t[..., None]) + rng.normal(0, 6, (H, W, 3))
    elif kind == 'checker':
        k = int(rng.integers(16, 64))
        c2 = rng.integers(40, 230, 3).astype(np.float32)
        b = ((xx // k + yy // k) % 2)[..., None]
        img = np.where(b == 0, c1, 0.6 * c1 + 0.4 * c2)
    else:
        small = rng.normal(0, 1, (max(2, H // 16), max(2, W // 16), 3))
        lo = np.asarray(Image.fromarray(np.uint8(np.clip(128 + 50 * small, 0, 255))).resize((W, H), Image.Resampling.BICUBIC), np.float32)
        img = 0.55 * c1 + 0.45 * lo + rng.normal(0, 8, (H, W, 3))
    return np.clip(img, 0, 255).astype(np.float32)


def _place(rng, W, H, short, n, overlap: bool, min_size=20, max_frac=0.3):
    objs, boxes = [], []
    for _ in range(n * 30):
        if len(objs) == n:
            break
        shape = rt_.SHAPES[int(rng.integers(len(rt_.SHAPES)))]
        s = float(np.exp(rng.uniform(np.log(min_size), np.log(max_frac * short))))
        aspect = float(np.exp(rng.uniform(-0.5, 0.5))) if shape in ('ellipse', 'rectangle', 'cross', 'diamond') else 1.0
        w, h = s * math.sqrt(aspect), s / math.sqrt(aspect)
        if w >= W - 4 or h >= H - 4:
            continue
        x0, y0 = rng.uniform(2, W - w - 2), rng.uniform(2, H - h - 2)
        box = (x0, y0, x0 + w, y0 + h)
        if not overlap and any(not (box[2] + 4 < b[0] or b[2] + 4 < box[0] or box[3] + 4 < b[1] or b[3] + 4 < box[1]) for b in boxes):
            continue
        color = list(rt_.PALETTE)[int(rng.integers(len(rt_.PALETTE)))]
        objs.append({'shape': shape, 'color': color, 'box': box, 'rot': float(rng.uniform(0, 2 * math.pi))})
        boxes.append(box)
    return objs


def _truth_from_mask(mask: np.ndarray, W: int, H: int) -> dict:
    ys, xs = np.nonzero(mask)
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    return {'canvas': [W, H], 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
            'hit': {'kind': 'box', 'box': box}}


def _shape_scene(rng, level: str, background: Image.Image | None):
    """texture / clutter / photo_bg scenes: returns (image, truth, target, meta) or None to resample."""
    if level == 'photo_bg':
        W, H = background.size
        bg = np.asarray(background, np.float32).copy()
        n = int(rng.integers(1, 7))
    else:
        W, H = _canvas(rng)
        bg = _textured_background(rng, W, H)
        n = int(rng.integers(3, 12)) if level == 'texture' else int(rng.integers(15, 36))
    short = min(W, H)
    objs = _place(rng, W, H, short, n, overlap=(level == 'clutter'), max_frac=0.3 if level != 'clutter' else 0.22)
    pairs = [(o['color'], o['shape']) for o in objs]
    unique = [i for i, p in enumerate(pairs) if pairs.count(p) == 1]
    if not unique:
        return None
    target = int(unique[int(rng.integers(len(unique)))])
    img = bg.copy()
    owner = -np.ones((H, W), np.int32)
    for k, o in enumerate(objs):
        m = rt_.draw_shape((W, H), o['shape'], o['box'], o['rot'], False, 3)
        if level == 'photo_bg':            # soft drop shadow before the object
            sh = Image.fromarray(np.uint8(m) * 255).filter(ImageFilter.GaussianBlur(4))
            sh = np.roll(np.roll(np.asarray(sh, np.float32) / 255, 4, 0), 4, 1)[..., None]
            img = img * (1 - 0.45 * sh)
        use_tex = level != 'clutter' or rng.random() < 0.5
        img[m] = _texture_fill(rng, m, rt_.PALETTE[o['color']], o['box']) if use_tex else rt_.PALETTE[o['color']]
        owner[m] = k
    if level == 'clutter':                 # thin random strokes over everything
        canvas = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
        d = ImageDraw.Draw(canvas)
        for _ in range(int(rng.integers(10, 30))):
            pts = [(float(rng.uniform(0, W)), float(rng.uniform(0, H))) for _ in range(int(rng.integers(2, 5)))]
            d.line(pts, fill=tuple(int(v) for v in rng.integers(0, 256, 3)), width=int(rng.integers(1, 3)))
        img = np.asarray(canvas, np.float32)
    tmask = owner == target
    full = rt_.draw_shape((W, H), objs[target]['shape'], objs[target]['box'], objs[target]['rot'], False, 3)
    if tmask.sum() < 150 or tmask.sum() < 0.6 * full.sum():
        return None
    ring = np.asarray(Image.fromarray(np.uint8(full) * 255).filter(ImageFilter.MaxFilter(9)), bool) & ~full
    if ring.any() and float(np.abs(bg[ring].mean(0) - np.asarray(rt_.PALETTE[objs[target]['color']])).mean()) < 40:
        return None
    o = objs[target]
    expr = rt_.EXPRESSIONS[int(rng.integers(len(rt_.EXPRESSIONS)))].format(color=o['color'], shape=o['shape'])
    image = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
    meta = {'n_objects': len(objs), 'shape': o['shape'], 'color': o['color'],
            'visible_frac': float(tmask.sum() / max(full.sum(), 1))}
    return image, _truth_from_mask(tmask, W, H), expr, meta


def _bar_chart(rng):
    W, H = _canvas(rng, 14, 33, 1.8)
    img = Image.new('RGB', (W, H), tuple(int(v) for v in rng.integers(235, 256, 3)))
    d = ImageDraw.Draw(img)
    n = int(rng.integers(4, 11))
    labels = list(BAR_LABELS[:n]) if rng.random() < 0.5 else list(MONTHS[int(rng.integers(0, 12 - n + 1)):][:n])
    fs = max(11, int(min(W, H) * 0.035))
    font = _font(fs)
    left, right, top, bottom = int(W * 0.1), int(W * 0.95), int(H * 0.12), int(H * 0.85)
    d.text((left, int(H * 0.03)), 'Quarterly values', fill=(30, 30, 30), font=_font(int(fs * 1.2)))
    d.line([(left, top), (left, bottom), (right, bottom)], fill=(40, 40, 40), width=2)
    for k in range(5):                       # gridlines and tick labels
        y = bottom - (bottom - top) * k / 4
        d.line([(left, y), (right, y)], fill=(210, 210, 210), width=1)
        d.text((max(2, left - fs * 2.2), y - fs / 2), str(25 * k), fill=(60, 60, 60), font=font)
    slot = (right - left) / n
    single = rng.random() < 0.5
    base = tuple(int(v) for v in rng.integers(20, 200, 3))
    boxes = []
    for k in range(n):
        h = rng.uniform(0.1, 1.0) * (bottom - top)
        x0, x1 = left + slot * (k + 0.2), left + slot * (k + 0.8)
        col = base if single else tuple(int(v) for v in rng.integers(20, 220, 3))
        d.rectangle([x0, bottom - h, x1, bottom - 1], fill=col)
        tw = d.textlength(labels[k], font=font)
        d.text(((x0 + x1) / 2 - tw / 2, bottom + 4), labels[k], fill=(20, 20, 20), font=font)
        boxes.append([float(x0), float(bottom - h), float(x1 + 1), float(bottom)])
    t = int(rng.integers(n))
    box = boxes[t]
    truth = {'canvas': [W, H], 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], 'hit': {'kind': 'box', 'box': box}}
    return img, truth, f'the bar labeled {labels[t]}', {'kind': 'bar', 'n_objects': n}


def _ui_panel(rng):
    W, H = _canvas(rng, 14, 33, 1.8)
    img = Image.new('RGB', (W, H), tuple(int(v) for v in rng.integers(200, 250, 3)))
    d = ImageDraw.Draw(img)
    fs = max(11, int(min(W, H) * 0.03))
    font = _font(fs)
    d.rectangle([0, 0, W, int(fs * 2.2)], fill=tuple(int(v) for v in rng.integers(30, 120, 3)))
    d.text((fs, int(fs * 0.5)), 'Document Editor', fill=(250, 250, 250), font=font)
    n = int(rng.integers(4, 13))
    words = [UI_WORDS[i] for i in rng.choice(len(UI_WORDS), n, replace=False)]
    boxes, y, x = [], fs * 3.2, fs
    bw_pad, bh = fs * 0.9, fs * 2.0
    accent = tuple(int(v) for v in rng.integers(40, 200, 3))
    for k, w in enumerate(words):
        bw = d.textlength(w, font=font) + 2 * bw_pad
        if x + bw > W - fs:
            x, y = fs, y + bh + fs * rng.uniform(0.6, 2.5)
        if y + bh > H - fs:
            break
        fill = accent if rng.random() < 0.3 else (245, 245, 245)
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=int(fs * 0.4), fill=fill, outline=(120, 120, 120), width=1)
        d.text((x + bw_pad, y + bh / 2 - fs / 2), w, fill=(255, 255, 255) if fill == accent else (20, 20, 20), font=font)
        boxes.append((w, [float(x), float(y), float(x + bw + 1), float(y + bh + 1)]))
        x += bw + fs * rng.uniform(0.5, 3.0)
    for _ in range(int(rng.integers(1, 4))):  # distractor text fields
        yy = rng.uniform(y + bh + fs, H - fs * 2) if y + bh + fs < H - fs * 2 else None
        if yy is None:
            break
        xx = rng.uniform(fs, W * 0.5)
        d.rectangle([xx, yy, xx + W * 0.35, yy + fs * 1.6], fill=(255, 255, 255), outline=(150, 150, 150))
        d.text((xx + 4, yy + 2), 'Type here', fill=(160, 160, 160), font=font)
    if len(boxes) < 3:
        return None
    w, box = boxes[int(rng.integers(len(boxes)))]
    truth = {'canvas': [W, H], 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], 'hit': {'kind': 'box', 'box': box}}
    return img, truth, f'the {w} button', {'kind': 'ui', 'n_objects': len(boxes)}


def graded_scene(level: str, seed: int, backgrounds: list[Path] | None = None) -> tuple[Image.Image, dict, str, dict]:
    """One scene of the graded synthetic family; deterministic in (level, seed).

    flat      the round-1 generic scene (flat colours, disjoint shapes);
    texture   textured shapes (stripes, checker, noise, shaded spheres, gradients) on textured backgrounds;
    clutter   15-35 overlapping shapes with partial occlusion (target >= 60% visible) and stroke clutter;
    chart_ui  bar charts ("the bar labeled C") and UI panels ("the Save button");
    photo_bg  shaded shapes with drop shadows composited onto natural photographs (RefCOCO images).
    Truth boxes are the extent of the target's visible pixels.
    """
    if level not in LEVELS:
        raise ValueError(f'unknown level {level}')
    for attempt in range(200):
        rng = np.random.default_rng([seed, attempt, LEVELS.index(level)])
        if level == 'flat':
            image, t = rt_.generic_scene(int(seed))
            target = t.pop('target')
            truth = {k: t[k] for k in ('canvas', 'box', 'center', 'hit')}
            return image, truth, target, {'n_objects': t['n_objects']}
        if level == 'chart_ui':
            out = _bar_chart(rng) if rng.random() < 0.5 else _ui_panel(rng)
        else:
            bg = None
            if level == 'photo_bg':
                if not backgrounds:
                    raise ValueError('photo_bg needs background images')
                bg = Image.open(backgrounds[int(rng.integers(len(backgrounds)))]).convert('RGB')
            out = _shape_scene(rng, level, bg)
        if out is not None:
            return out
    raise RuntimeError(f'could not generate a {level} scene for seed {seed}')


# ----------------------------------------------------------------------------- capture

def token_centers(grid: tuple[int, int]) -> np.ndarray:
    return rt_.token_centers((1, 2 * grid[0], 2 * grid[1])).numpy()


@torch.inference_mode()
def capture_r2(rt: qr.Runtime, texts: list[str], images: list[Image.Image], offsets: list[dict[str, int]],
               pca: dict | None, max_new_tokens: int = 12) -> list[dict]:
    """Prefill (inside greedy generation) of prompt + call prefix; per sample returns

    hidden   (len(HIDDEN_LAYERS), 4096) float16 residual at the slot;
    maps     {site: (8 * 16, Nv) float16} softmax attention of the full-attention heads over visual tokens;
    content  {layer: (Nv, PCA_DIM) float16} PCA projection of visual-token states (if pca is given);
    digit_logp (10,) log-probabilities of the first digit at the slot;
    cont     decoded continuation (the model's rough point, e.g. '225, 552]...').
    Attention probabilities are recomputed from the layer input for the two site queries only, which
    is what a deployed tool would do from the KV cache (no full attention matrix is materialized).
    """
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    grids = batch['image_grid_thw'].tolist()
    for image, (_, h, w) in zip(images, grids):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError(f'Processor resized a {image.width}x{image.height} image.')
    L = batch['input_ids'].shape[1]
    idx = torch.tensor([[L + o[s] for s in SITES_R2] for o in offsets], device=rt.device)
    vis = batch['input_ids'] == rt.model.config.image_token_id
    centers = [rt_.token_centers(g) for g in grids]
    store, attn, content, handles = {}, {}, {}, []

    def is_prefill(h):
        return h.shape[1] == L

    for layer in HIDDEN_LAYERS:
        def hook(module, inputs, output, layer=layer):
            h = output[0] if isinstance(output, tuple) else output
            if is_prefill(h):
                store[layer] = h[:, -1].half().cpu()
        handles.append(rt.layers[layer - 1].register_forward_hook(hook))
    if pca is not None:
        for layer in CONTENT_LAYERS:
            mu, U = pca[layer]['mean'].to(rt.device), pca[layer]['basis'].to(rt.device)

            def take(h, layer=layer, mu=mu, U=U):
                v, mu_, U_ = vis.to(h.device), mu.to(h.device), U.to(h.device)
                content[layer] = [((h[b][v[b]].float() - mu_) @ U_).half().cpu() for b in range(h.shape[0])]
            if layer == 0:
                def pre0(module, args, kwargs, take=take):
                    h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
                    if is_prefill(h):
                        take(h)
                handles.append(rt.layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
            else:
                def hk(module, inputs, output, take=take):
                    h = output[0] if isinstance(output, tuple) else output
                    if is_prefill(h):
                        take(h)
                handles.append(rt.layers[layer - 1].register_forward_hook(hk))
    for layer in rt_.ATTN_LAYERS:
        mod = rt.layers[layer - 1].self_attn

        def pre(module, args, kwargs, layer=layer):
            h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            if not is_prefill(h):
                return
            d = h.device
            maps = []
            for si in range(len(SITES_R2)):
                maps.append(rt_._attention_features(module, h, kwargs['position_embeddings'], idx[:, si:si + 1].to(d),
                                                    batch['attention_mask'].to(d), vis.to(d), centers, 0)[1])
            attn[layer] = maps
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        out = rt.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                top_k=None, output_scores=True, return_dict_in_generate=True)
    finally:
        for hd in handles:
            hd.remove()
    digits = [rt.tokenizer.convert_tokens_to_ids(str(k)) for k in range(10)]
    logp = out.scores[0].float().log_softmax(-1)[:, digits].cpu().numpy()
    seqs = out.sequences[:, L:]
    res = []
    for b in range(len(texts)):
        maps = {s: torch.cat([attn[l][si][b] for l in rt_.ATTN_LAYERS], 0).numpy() for si, s in enumerate(SITES_R2)}
        row = {'hidden': torch.stack([store[l][b] for l in HIDDEN_LAYERS]).numpy(), 'maps': maps,
               'digit_logp': logp[b], 'cont': rt.tokenizer.decode(seqs[b], skip_special_tokens=True),
               'grid': (grids[b][1] // 2, grids[b][2] // 2)}
        if pca is not None:
            row['content'] = {l: content[l][b].numpy() for l in CONTENT_LAYERS}
        res.append(row)
    return res


@torch.inference_mode()
def visual_states(rt: qr.Runtime, texts: list[str], images: list[Image.Image], layers=CONTENT_LAYERS) -> dict[int, list[torch.Tensor]]:
    """Visual-token states (float32, on CPU) at the given layers from one prefill; used to fit the PCA."""
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    vis = batch['input_ids'] == rt.model.config.image_token_id
    out, handles = {l: [] for l in layers}, []
    for layer in layers:
        if layer == 0:
            def pre0(module, args, kwargs):
                h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
                v = vis.to(h.device)
                out[0].extend(h[b][v[b]].float().cpu() for b in range(h.shape[0]))
            handles.append(rt.layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
        else:
            def hk(module, inputs, output, layer=layer):
                h = output[0] if isinstance(output, tuple) else output
                v = vis.to(h.device)
                out[layer].extend(h[b][v[b]].float().cpu() for b in range(h.shape[0]))
            handles.append(rt.layers[layer - 1].register_forward_hook(hk))
    try:
        rt.model(**batch, use_cache=False, logits_to_keep=1)
    finally:
        for hd in handles:
            hd.remove()
    return out


def fit_pca(states: list[torch.Tensor], dim: int = PCA_DIM) -> dict:
    X = torch.cat(states).double()
    mu = X.mean(0)
    C = ((X - mu).T @ (X - mu)) / (len(X) - 1)
    evals, evecs = torch.linalg.eigh(C)
    order = evals.argsort(descending=True)[:dim]
    return {'mean': mu.float(), 'basis': evecs[:, order].float(), 'explained': float(evals[order].sum() / evals.sum())}


# ----------------------------------------------------------------------------- convolutional pointer

class GridPointer(torch.nn.Module):
    """Convolutional pointer over the visual-token grid.

    Input (B, C, gh, gw) per-token features.  Output per token: a pointer logit s, a sub-token centre
    offset (dx, dy) in token units, and distances (l, t, r, b) from the token centre to the four box
    edges in token units.  The tool's point and box are pointer-weighted averages:
        point = sum_j alpha_j (c_j + delta_j),  box = sum_j alpha_j (c_j - (l_j, t_j), c_j + (r_j, b_j)).
    Four residual 3x3 blocks with dilations 1, 2, 4, 1 give a receptive field of 17 tokens, enough to
    see a target and its neighbours; no absolute position is input, so the reader is translation
    equivariant and independent of the canvas size.
    """

    def __init__(self, cin: int, width: int = 96, dilations=(1, 2, 4, 1)):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Conv2d(cin, width, 1), torch.nn.GELU())
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Conv2d(width, width, 3, padding=d, dilation=d), torch.nn.GroupNorm(8, width), torch.nn.GELU())
            for d in dilations)
        self.head = torch.nn.Conv2d(width, 7, 1)
        self.cin, self.width, self.dilations = cin, width, tuple(dilations)

    def forward(self, x):
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h)                                           # (B, 7, gh, gw)


def grid_decode(out: torch.Tensor, valid: torch.Tensor, grids: list[tuple[int, int]]):
    """out (B, 7, Gh, Gw), valid (B, Gh, Gw) -> alpha (B, Gh*Gw), point (B, 2), box (B, 4) in 0-1000 units."""
    B, _, Gh, Gw = out.shape
    s = out[:, 0].masked_fill(~valid, float('-inf')).reshape(B, -1)
    alpha = s.softmax(-1)
    gh = torch.tensor([g[0] for g in grids], dtype=out.dtype, device=out.device)[:, None, None]
    gw = torch.tensor([g[1] for g in grids], dtype=out.dtype, device=out.device)[:, None, None]
    r = torch.arange(Gh, device=out.device, dtype=out.dtype)[None, :, None].expand(B, Gh, Gw)
    c = torch.arange(Gw, device=out.device, dtype=out.dtype)[None, None, :].expand(B, Gh, Gw)
    cw, ch = 1000.0 / gw, 1000.0 / gh                                 # token cell size in units
    cx, cy = (c + 0.5) * cw, (r + 0.5) * ch
    dx, dy = 1.5 * torch.tanh(out[:, 1]) * cw, 1.5 * torch.tanh(out[:, 2]) * ch
    ext = torch.nn.functional.softplus(out[:, 3:7])
    x0, y0 = cx - ext[:, 0] * cw, cy - ext[:, 1] * ch
    x1, y1 = cx + ext[:, 2] * cw, cy + ext[:, 3] * ch
    flat = lambda t: t.reshape(B, -1)
    point = torch.stack([(alpha * flat(cx + dx)).sum(1), (alpha * flat(cy + dy)).sum(1)], 1)
    box = torch.stack([(alpha * flat(v)).sum(1) for v in (x0, y0, x1, y1)], 1)
    return alpha, point, box


def make_grid_batch(items: list[dict], mean: np.ndarray, std: np.ndarray, device: str):
    """Items hold F (Nv, C) float16, grid (gh, gw), q (Nv,), y (6,).  Returns padded tensors."""
    Gh = max(it['grid'][0] for it in items)
    Gw = max(it['grid'][1] for it in items)
    C = items[0]['F'].shape[1]
    X = np.zeros((len(items), C, Gh, Gw), np.float32)
    V = np.zeros((len(items), Gh, Gw), bool)
    Q = np.zeros((len(items), Gh, Gw), np.float32)
    for b, it in enumerate(items):
        gh, gw = it['grid']
        X[b, :, :gh, :gw] = ((it['F'].astype(np.float32) - mean) / std).T.reshape(C, gh, gw)
        V[b, :gh, :gw] = True
        if 'q' in it:
            Q[b, :gh, :gw] = it['q'].reshape(gh, gw)
    Y = np.stack([it['y'] for it in items]).astype(np.float32) if 'y' in items[0] else None
    t = lambda a: torch.from_numpy(a).to(device)
    return t(X), t(V), t(Q).reshape(len(items), -1), (t(Y) if Y is not None else None), [tuple(it['grid']) for it in items]


def fit_grid_pointer(train: list[dict], val: list[dict], seed: int = 0, epochs: int = 20, width: int = 96,
                     lr: float = 2e-3, batch: int = 32, device: str = 'cpu', max_tokens: int = 24000,
                     flip: bool = True, flip_negate: tuple[int, ...] = (), log=print) -> tuple[dict, dict]:
    """Loss = CE(alpha, q / sum q) + |point - centre|_1 / 100 + |box - box*|_1 / 100.

    The cross-entropy places the pointer on the tokens the target covers; the two L1 terms train the
    sub-token offsets and the box-edge distances.  Keeps the epoch with the lowest validation
    centre error.  Batches are formed from items of similar grid size under a padded-token budget.
    With `flip`, half of the batches are mirrored horizontally; feature channels listed in
    `flip_negate` (signed x offsets) change sign under the mirror.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sample = np.concatenate([it['F'][::7] for it in train[::3]]).astype(np.float32)
    mean, std = sample.mean(0), sample.std(0) + 1e-3
    model = GridPointer(train[0]['F'].shape[1], width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    order = sorted(range(len(train)), key=lambda i: (train[i]['grid'][0] * train[i]['grid'][1]))
    groups, cur = [], []
    for i in order:
        n = train[i]['grid'][0] * train[i]['grid'][1]
        if cur and (len(cur) + 1) * n > max_tokens or len(cur) == batch:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    steps = epochs * len(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    best, history = (float('inf'), None, -1), []
    for ep in range(epochs):
        model.train()
        for gi in rng.permutation(len(groups)):
            X, V, Q, Y, grids = make_grid_batch([train[i] for i in groups[gi]], mean, std, device)
            if flip and rng.random() < 0.5:                # horizontal flip augmentation
                X, V = X.flip(-1), V.flip(-1)
                if flip_negate:
                    X[:, list(flip_negate)] = -X[:, list(flip_negate)] - 2 * torch.as_tensor(mean[list(flip_negate)] / std[list(flip_negate)], device=X.device)[None, :, None, None]
                Q = Q.reshape(V.shape).flip(-1).reshape(len(V), -1)
                X, V, Q = _shift_left(X, V, Q, grids)
                Y = Y.clone()
                Y[:, [0, 2, 4]] = 1000 - Y[:, [0, 4, 2]]
            alpha, point, box = grid_decode(model(X), V, grids)
            qn = Q / Q.sum(1, keepdim=True).clamp_min(1e-6)
            ce = -(qn * torch.log(alpha.clamp_min(1e-9))).sum(1).mean()
            loss = ce + (point - Y[:, :2]).abs().sum(1).mean() / 100 + (box - Y[:, 2:]).abs().sum(1).mean() / 100
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        P = predict_grid_pointer(model, mean, std, val, device)
        err = float(np.mean(np.linalg.norm(P[:, :2] - np.stack([it['y'][:2] for it in val]), axis=1)))
        history.append(err)
        log(f'  epoch {ep}: val centre error {err:.1f}')
        if err < best[0]:
            best = (err, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    model.cpu().eval()
    params = {'cin': model.cin, 'width': model.width, 'dilations': model.dilations, 'state': model.state_dict(),
              'mean': mean, 'std': std}
    return params, {'val_error_by_epoch': history, 'best_epoch': best[2]}


def _shift_left(X, V, Q, grids):
    """After a flip of the padded grid, move each sample's valid region back to the top-left corner."""
    Gw = X.shape[-1]
    Qg = Q.reshape(V.shape)
    for b, (gh, gw) in enumerate(grids):
        k = Gw - gw
        if k:
            X[b] = torch.roll(X[b], -k, -1)
            V[b] = torch.roll(V[b], -k, -1)
            Qg[b] = torch.roll(Qg[b], -k, -1)
    return X, V, Qg.reshape(len(V), -1)


@torch.no_grad()
def predict_grid_pointer(model, mean, std, items: list[dict], device: str = 'cpu', batch: int = 32,
                         return_alpha: bool = False):
    model.eval()
    out = np.zeros((len(items), 6))
    alphas = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: items[i]['grid'][0] * items[i]['grid'][1])
    dev = next(model.parameters()).device
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        X, V, _, _, grids = make_grid_batch([items[i] for i in idx], mean, std, str(dev))
        alpha, point, box = grid_decode(model(X), V, grids)
        out[idx, :2] = point.cpu().numpy()
        out[idx, 2:] = box.cpu().numpy()
        if return_alpha:
            Gh, Gw = V.shape[1:]
            for j, i in enumerate(idx):
                gh, gw = grids[j]
                alphas[i] = alpha[j].reshape(Gh, Gw)[:gh, :gw].cpu().numpy()
    return (out, alphas) if return_alpha else out


def load_grid_pointer(params: dict) -> GridPointer:
    m = GridPointer(params['cin'], params['width'], params['dilations'])
    m.load_state_dict(params['state'])
    return m.eval()


# ----------------------------------------------------------------------------- per-token features

def attention_features(maps: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """(C, Nv) attention -> (Nv, C) log-ratio to uniform; the CNN supplies the neighbourhood."""
    nv = maps.shape[1]
    if nv != grid[0] * grid[1]:
        raise ValueError(f'map length {nv} does not match grid {grid}')
    return np.log(maps.astype(np.float32) * nv + 1e-3).T


def approx_features(approx: list[int] | None, grid: tuple[int, int]) -> np.ndarray:
    """(Nv, 3): token-centre offset from the model's rough point (in units of 100) and a validity flag."""
    c = token_centers(grid)
    if approx is None:
        return np.zeros((len(c), 3), np.float32)
    a = np.clip(np.asarray(approx, np.float32), 0, 1000)
    return np.concatenate([(c - a) / 100.0, np.ones((len(c), 1))], 1).astype(np.float32)


# ----------------------------------------------------------------------------- learned query over visual tokens

class QueryPointer(torch.nn.Module):
    """Attention-free pointer: a learned query from the slot hidden state scores the visual tokens.

    s_j = (W_q h) . (W_k v_j) / sqrt(d), with v_j the PCA-projected visual-token states; the output is
    decoded like the grid pointer after a small 3x3 convolution on the scores and features.  It uses
    only states that exist without attention probabilities (flash-attention compatible).
    """

    def __init__(self, dh: int, dv: int, d: int = 64, width: int = 48):
        super().__init__()
        self.q = torch.nn.Sequential(torch.nn.LayerNorm(dh), torch.nn.Linear(dh, d))
        self.k = torch.nn.Linear(dv, d)
        self.conv = GridPointer(1 + dv, width, (1, 2, 1))
        self.d = d

    def forward(self, h, X):
        q = self.q(h)                                                  # (B, d)
        k = self.k(X.permute(0, 2, 3, 1))                              # (B, Gh, Gw, d)
        s = (k * q[:, None, None]).sum(-1) / math.sqrt(self.d)
        return self.conv(torch.cat([s[:, None], X], 1))
