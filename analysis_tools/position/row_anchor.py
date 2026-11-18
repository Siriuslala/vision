"""Experiment 3.14 round 4: origin of the row-start anchor of the y-reading heads (GPU stages).

Results: docs/analysis/position/results_2026-11-11_row_anchor.md. Builds on round 3
(analysis_tools/position/row_attention.py), whose helpers (AttnInputs, head_qk, rotate, truth_answer) are reused.

Every measurement is taken at the query that predicts the first digit of a coordinate value, teacher-forced on
the true answer. For a reading head with rotated query q~ (at the query position) and pre-RoPE key k_m of image
token m, the content score is c_m = q~ . R(ref) k_m / sqrt(d), i.e. every key rotated as if it sat at the
target's first token (round 3 showed that the positional part of the score is negligible, <= 0.22% of the
variance). The left-edge bonus of a token row is c at column 0 minus the mean c over interior columns.

Stages:
* `factors`: synthetic images that interpolate between the grid stimulus and natural-image properties
  (left margin, textured / photo background, scattered layout, single object, text document, ruled table).
  Stores, per case, the image attention and content scores of the reading heads, the main head's values and
  content-rotated keys, and the ink fraction of every token.
* `locate`: content-matched activation patching on 10x10 grids. The states of the column-0 tokens of every
  token row are replaced by those of column S of the same row (S = 2 or 4: the left half of another figure,
  pixel-identical to the left half drawn in column 0) at one site: the input of a ViT block, the ViT merger
  output and/or the DeepStack features, or the residual stream entering a decoder layer. The reverse implant
  (column S receives column 0's states) tests sufficiency. Also stores, for the clean run, the pooled hidden
  states of columns 0, S and 10 at every ViT block and decoder layer (for the CPU probes).
* `edits`: answer-time value edits (round-3 protocol) on edge vs left-margin grids: values of the column-0
  tokens, of the first figure-bearing token of the target rows, or of the target are replaced by the values
  two cells lower; the recovered fraction r is the 3.11 definition.
Layers are 1-based; ViT blocks are 0-based as in the checkpoint (block b's input is the output of block b-1).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.common.stimuli import Change, StimulusSpec, generate
from analysis_tools.position import causal_vs_linear as cv
from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import row_attention as ra

sr = cf.sr
read_jsonl, append_jsonl = sr.read_jsonl, sr.append_jsonl

TOKEN = 32                                      # pixels per merged visual token (patch 16, merge 2)
CELL = 64
TEMPLATES = ('person', 'ring', 'arrow', 'square')
RED = Change('color', 4, 0)
# variant -> answer formats; grid-type variants also get the row/column format.
VARIANTS = {
    'edge': ('rowcol', 'point'), 'margin_left': ('rowcol', 'point'), 'margin_all': ('rowcol', 'point'),
    'noise': ('rowcol', 'point'), 'photo': ('rowcol', 'point'), 'table': ('rowcol', 'point'),
    'scatter': ('point',), 'single': ('point',), 'photo_single': ('point',),
    'doc': ('point',), 'doc_edge': ('point',),
}
QUERY = {'table': 'One number is written in red.', 'doc': 'One word is written in red.',
         'doc_edge': 'One word is written in red.'}
# Main reading heads (round 2): y heads of the row/column and point formats, plus the x head.
HEADS = {'qwen3_vl': [(22, 8), (22, 11), (24, 30), (25, 13), (27, 20), (29, 16), (35, 28)],
         'qwen3_5': [(20, 11), (24, 0), (24, 4), (32, 8), (32, 14)]}
MAIN = {'qwen3_vl': (24, 30), 'qwen3_5': (20, 11)}
VIT_SITES = (0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27)      # 27 = merger input
DEC_SITES = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)                # residual entering layer l


# ----------------------------------------------------------------------------- stimuli (pure, tested)

def ink_mask(img: np.ndarray) -> np.ndarray:
    """(H, W) bool: pixels of a white-background rendering that carry drawing (any channel < 250)."""
    return (img < 250).any(-1)


def token_ink(mask: np.ndarray) -> np.ndarray:
    """(H/32 * W/32,) fraction of drawn pixels per merged token, raster order."""
    H, W = mask.shape
    return mask.reshape(H // TOKEN, TOKEN, W // TOKEN, TOKEN).mean((1, 3)).reshape(-1)


def background(kind: str, size: tuple[int, int], rng: np.random.Generator, photos: list[Path]) -> np.ndarray:
    """(H, W, 3) uint8 background: 'white', 'noise' (blurred grey texture) or 'photo' (lightened RefCOCO photo)."""
    W, H = size
    if kind == 'white':
        return np.full((H, W, 3), 255, np.uint8)
    if kind == 'noise':
        z = rng.normal(0, 1, (H, W, 3))
        im = Image.fromarray(np.uint8(np.clip(128 + 60 * z, 0, 255))).filter(ImageFilter.GaussianBlur(3))
        a = np.asarray(im, dtype=np.float64)
        a = (a - a.mean()) / (a.std() + 1e-6)
        return np.uint8(np.clip(195 + 22 * a, 140, 245))
    if kind == 'photo':
        p = Image.open(photos[int(rng.integers(len(photos)))]).convert('RGB')
        s = max(W / p.width, H / p.height)
        p = p.resize((int(np.ceil(p.width * s)), int(np.ceil(p.height * s))), Image.BICUBIC)
        x0, y0 = (p.width - W) // 2, (p.height - H) // 2
        a = np.asarray(p.crop((x0, y0, x0 + W, y0 + H)), dtype=np.float64)
        return np.uint8(0.55 * a + 0.45 * 255)
    raise ValueError(f'Unknown background {kind!r}')


def composite(fg: np.ndarray, mask: np.ndarray, bg: np.ndarray) -> np.ndarray:
    return np.where(mask[..., None], fg, bg)


def scatter_positions(n: int, size: int, cell: int, rng: np.random.Generator, tries: int = 20000) -> list[tuple]:
    """n non-overlapping cell x cell boxes at continuous (1-pixel) positions inside a size x size canvas."""
    out = []
    for _ in range(tries):
        x, y = (int(v) for v in rng.integers(0, size - cell + 1, 2))
        if all(abs(x - a) >= cell or abs(y - b) >= cell for a, b in out):
            out.append((x, y))
            if len(out) == n:
                return out
    raise RuntimeError(f'Could not place {n} cells')


def font(size: int) -> ImageFont.FreeTypeFont:
    import matplotlib
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'), size)


def text_page(margin: int, rng: np.random.Generator, size: int = 640) -> tuple[Image.Image, np.ndarray, list[float]]:
    """Left-aligned pseudo-word text, one line per token row (line height 32 px), one word in red.

    Returns (image, ink mask, tight box of the red word). Words are random lowercase strings (3-8 letters)."""
    im = Image.new('RGB', (size, size), 'white')
    d = ImageDraw.Draw(im)
    f = font(19)
    lines = size // TOKEN
    tline = int(rng.integers(1, lines - 1))
    boxes = []
    for i in range(lines):
        x, y = margin, i * TOKEN + 5
        words = []
        while True:
            w = ''.join(chr(97 + int(c)) for c in rng.integers(0, 26, int(rng.integers(3, 9))))
            bb = d.textbbox((x, y), w, font=f)
            if bb[2] > size - 24:
                break
            words.append((w, x, bb))
            x = bb[2] + 9
        red = int(rng.integers(1, max(2, len(words) - 1))) if i == tline else -1
        for k, (w, wx, bb) in enumerate(words):
            d.text((wx, y), w, fill=(220, 30, 30) if k == red else (0, 0, 0), font=f)
            if k == red:
                boxes.append([float(v) for v in bb])
    a = np.asarray(im)
    return im, ink_mask(a), boxes[0]


def table_page(rng: np.random.Generator, target: tuple[int, int], rows: int = 10, cols: int = 10
               ) -> tuple[Image.Image, np.ndarray, list[float]]:
    """Ruled 10x10 table (64 px cells, grey 2 px rules incl. the outer border), a 2-3 digit number per cell."""
    size = rows * CELL
    im = Image.new('RGB', (cols * CELL, size), 'white')
    d = ImageDraw.Draw(im)
    for k in range(rows + 1):
        y = min(k * CELL, size - 2)
        d.rectangle((0, y, cols * CELL - 1, y + 1), fill=(120, 120, 120))
    for k in range(cols + 1):
        x = min(k * CELL, cols * CELL - 2)
        d.rectangle((x, 0, x + 1, size - 1), fill=(120, 120, 120))
    f = font(20)
    box = None
    for r in range(rows):
        for c in range(cols):
            s = str(int(rng.integers(10, 1000)))
            bb = d.textbbox((0, 0), s, font=f)
            x = c * CELL + (CELL - (bb[2] - bb[0])) / 2 - bb[0]
            y = r * CELL + (CELL - (bb[3] - bb[1])) / 2 - bb[1]
            red = (r, c) == tuple(target)
            d.text((x, y), s, fill=(220, 30, 30) if red else (0, 0, 0), font=f)
            if red:
                box = [float(v) for v in d.textbbox((x, y), s, font=f)]
    a = np.asarray(im)
    return im, ink_mask(a), box


def compose(variant: str, template: str, target: tuple[int, int], seed: int, photos: list[Path]
            ) -> tuple[Image.Image, dict]:
    """Image and ground truth of one factor stimulus.

    truth: canvas [W, H], box (tight, pixels), center, target_rc (grid variants), grid_origin, and ink
    (per-token drawn-pixel fraction, raster order; for photo backgrounds the drawn figures only)."""
    rng = np.random.default_rng(seed)
    if variant in ('doc', 'doc_edge'):
        im, mask, box = text_page(64 if variant == 'doc' else 4, rng)
        return im, _truth(im.size, box, None, (0, 0), mask)
    if variant == 'table':
        im, mask, box = table_page(rng, target)
        return im, _truth(im.size, box, target, (0, 0), mask)
    spec = StimulusSpec(template=template, rows=10, cols=10, change=RED, target=tuple(target), seed=seed)
    base, meta = generate(spec)
    if list(meta['grid_origin']) != [0, 0] or base.size != (10 * CELL, 10 * CELL):
        raise ValueError(f"Unexpected grid layout {meta['grid_origin']} {base.size}")
    a = np.asarray(base)
    m = ink_mask(a)
    inst = next(i for i in meta['instances'] if (i['row'], i['col']) == tuple(target))
    box = list(inst['tight_box'])
    if variant in ('edge', 'noise', 'photo'):
        bg = background({'edge': 'white', 'noise': 'noise', 'photo': 'photo'}[variant], base.size, rng, photos)
        return Image.fromarray(composite(a, m, bg)), _truth(base.size, box, target, (0, 0), m)
    if variant in ('margin_left', 'margin_all'):
        ox, oy = (128, 0) if variant == 'margin_left' else (64, 64)
        W, H = base.width + ox * (2 if variant == 'margin_all' else 1), base.height + 2 * oy
        out = np.full((H, W, 3), 255, np.uint8)
        out[oy:oy + base.height, ox:ox + base.width] = a
        mm = np.zeros((H, W), bool)
        mm[oy:oy + base.height, ox:ox + base.width] = m
        return Image.fromarray(out), _truth((W, H), [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy],
                                            target, (ox, oy), mm)
    if variant in ('single', 'photo_single'):
        cb = inst['cell_box']
        keep = np.zeros_like(m)
        keep[cb[1]:cb[3], cb[0]:cb[2]] = m[cb[1]:cb[3], cb[0]:cb[2]]
        bg = background('white' if variant == 'single' else 'photo', base.size, rng, photos)
        return Image.fromarray(composite(a, keep, bg)), _truth(base.size, box, None, (0, 0), keep)
    if variant == 'scatter':
        n = 40
        pos = scatter_positions(n, base.width, CELL, rng)
        cells = [(r, c) for r in range(10) for c in range(10) if (r, c) != tuple(target)]
        pick = [cells[i] for i in rng.choice(len(cells), n - 1, replace=False)] + [tuple(target)]
        out = np.full_like(a, 255)
        mm = np.zeros_like(m)
        for (x, y), (r, c) in zip(pos, pick):
            tile = a[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
            out[y:y + CELL, x:x + CELL] = np.minimum(out[y:y + CELL, x:x + CELL], tile)
            mm[y:y + CELL, x:x + CELL] |= m[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
        x, y = pos[-1]
        bx = [box[0] - target[1] * CELL + x, box[1] - target[0] * CELL + y,
              box[2] - target[1] * CELL + x, box[3] - target[0] * CELL + y]
        return Image.fromarray(out), _truth(base.size, bx, None, (0, 0), mm)
    raise ValueError(f'Unknown variant {variant!r}')


def _truth(size, box, target, origin, mask) -> dict:
    W, H = size
    return {'canvas': [W, H], 'box': [float(v) for v in box], 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
            'target_rc': None if target is None else [int(target[0]), int(target[1])],
            'grid_origin': list(origin), 'ink': token_ink(mask).astype(np.float32).tolist()}


def factor_cases(n: int, seed: int) -> list[dict]:
    """n cases per (variant, format); targets uniform over cells with cell column >= 1 (row-start anchor exists)."""
    rng = np.random.default_rng(seed)
    out = []
    for variant, fmts in VARIANTS.items():
        for i in range(n):
            r, c = int(rng.integers(0, 10)), int(rng.integers(1, 10))
            s = int(seed * 1000 + len(out))
            for fmt in fmts:
                out.append({'id': f'{variant}_{i:03d}_{fmt}', 'variant': variant, 'format': fmt,
                            'template': TEMPLATES[i % 4], 'target': [r, c], 'seed': s,
                            'prompt': QUERY.get(variant, cf.T6_QUERY) + cf.FORMAT_SUFFIX[fmt]})
    return out


def answer_text(fmt: str, truth: dict) -> str:
    if fmt == 'rowcol':
        r, c = truth['target_rc']
        return f'{{"row": {r + 1}, "column": {c + 1}}}'
    x, y = truth['center']
    W, H = truth['canvas']
    return f'[{{"point_2d": [{round(1000 * x / W)}, {round(1000 * y / H)}]}}]'


def source_column(target_col_cell: int) -> int:
    """Token column S whose figure halves match column 0: the left half of cell 1, or of cell 2 if cell 1 holds
    the target."""
    return 4 if target_col_cell == 1 else 2


def patch_rows(grid: tuple[int, int], dst_col: int, src_col: int) -> tuple[list[int], list[int]]:
    """Merged-token indices (raster) of column dst_col and src_col in every token row."""
    H, W = grid
    return [i * W + dst_col for i in range(H)], [i * W + src_col for i in range(H)]


def patch_index(tokens: list[int]) -> list[int]:
    """ViT patch indices of merged tokens: the ViT orders patches block-major, 4 per merged token."""
    return [4 * t + u for t in tokens for u in range(4)]


# ----------------------------------------------------------------------------- model-side helpers

class Stop(Exception):
    pass


@contextmanager
def vit_patch(rt, block: int | None, dst: list[int], src: list[int]):
    """Replace ViT hidden states of patches dst by those of patches src at the input of block `block`
    (block == depth means the merger input)."""
    if block is None:
        yield
        return
    vis = rt.model.model.visual
    mod = vis.merger if block == len(vis.blocks) else vis.blocks[block]
    d, s = torch.tensor(dst), torch.tensor(src)

    def pre(module, args):
        h = args[0]
        h[d.to(h.device)] = h[s.to(h.device)]
        return (h,) + tuple(args[1:])
    hd = mod.register_forward_pre_hook(pre)
    try:
        yield
    finally:
        hd.remove()


@contextmanager
def decoder_patch(rt, layer: int | None, span: tuple[int, int], dst: list[int], src: list[int]):
    """Replace the residual stream entering decoder layer `layer` (1-based) at image tokens dst by src."""
    if layer is None:
        yield
        return
    s0 = span[0]
    d, s = torch.tensor(dst) + s0, torch.tensor(src) + s0

    def pre(module, args, kwargs):
        h = args[0] if args else kwargs['hidden_states']
        h[:, d.to(h.device)] = h[:, s.to(h.device)]
    hd = rt.layers[layer - 1].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        hd.remove()


@contextmanager
def stage_capture(rt, span, tokens: list[int], last_layer: int):
    """Pooled hidden states of chosen merged tokens at every ViT block output and decoder layer input.

    store['vit'][b]: (n_tok, 1152) mean of the 4 patches after block b; store['dec'][l]: (n_tok, D) residual
    entering decoder layer l (l = 1 is the embedding; DeepStack additions after layers 1-3 are included in
    the inputs of layers 2-4)."""
    store = {'vit': {}, 'dec': {}}
    hs = []
    vis = rt.model.model.visual
    pidx = torch.tensor(patch_index(tokens))
    for b, blk in enumerate(vis.blocks):
        def post(module, args, out, b=b):
            h = out[0] if isinstance(out, tuple) else out
            store['vit'][b] = h[pidx.to(h.device)].view(len(tokens), 4, -1).float().mean(1).cpu().half()
        hs.append(blk.register_forward_hook(post))
    tid = torch.tensor(tokens) + span[0]
    for layer in range(1, last_layer + 1):
        def pre(module, args, kwargs, layer=layer):
            h = args[0] if args else kwargs['hidden_states']
            store['dec'][layer] = h[0, tid.to(h.device)].float().cpu().half()
        hs.append(rt.layers[layer - 1].register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


@torch.inference_mode()
def vision(rt, prep):
    """(merged visual embeddings (Nv, D), DeepStack maps or None) of the prepared image under active hooks."""
    if rt.family == 'qwen3_vl':
        out = rt.model.model.get_image_features(prep.batch['pixel_values'], prep.batch['image_grid_thw'],
                                                return_dict=True)
        return out.pooler_output[0], [d for d in out.deepstack_features]
    return cr.vision_embeds(rt, prep), None


@torch.inference_mode()
def measure(rt, prep, x, pos, queries, heads, ds, ref: int, main=None, stop: bool = True) -> dict:
    """Image attention and content scores of `heads` at `queries`; values / content keys of the main head.

    ref: sequence index of the target's first token (content rotation). Returns numpy arrays:
    att, con (n_heads, Q, Nv); q (n_heads, Q, hd) rotated queries; for the main head v, kc (Nv, hd)."""
    s, e = prep.span['start'], prep.span['end']
    layers = sorted({l for l, _ in heads})
    last = max(layers)
    hds = []
    if stop:
        def halt(module, args, kwargs):
            raise Stop
        hds.append(rt.layers[last - 1].mlp.register_forward_pre_hook(halt, with_kwargs=True))
    try:
        with ra.AttnInputs(rt, layers) as cap, cv.deepstack_rows(rt, None if ds is None else [ds], (s, e)):
            try:
                rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
            except Stop:
                pass
    finally:
        for h in hds:
            h.remove()
    out = {'att': [], 'con': [], 'q': []}
    qi = torch.tensor(queries)
    for layer in layers:
        hidden, (cos, sin) = cap.store[layer]
        q, k, mod = ra.head_qk(rt, layer, hidden, queries)
        cos, sin = cos[0].float(), sin[0].float()
        qd = qi.to(q.device)
        qr_ = ra.rotate(q, cos[qd], sin[qd])
        kr = ra.rotate(k, cos, sin)
        kc = ra.rotate(k[:, s:e], cos[ref].expand(e - s, -1), sin[ref].expand(e - s, -1))
        for l2, h in heads:
            if l2 != layer:
                continue
            g = h // mod.num_key_value_groups
            sc = (qr_[h] @ kr[g].T) * mod.scaling
            sc = sc.masked_fill(torch.arange(sc.shape[1], device=sc.device)[None] > qd[:, None], float('-inf'))
            out['att'].append(sc.softmax(-1)[:, s:e].cpu().numpy())
            out['con'].append(((qr_[h] @ kc[g].T) * mod.scaling).cpu().numpy())
            out['q'].append(qr_[h].cpu().numpy())
            if (l2, h) == main:
                v = mod.v_proj(hidden[:, s:e]).view(e - s, -1, mod.head_dim)[:, g].float()
                out['v'] = v.cpu().numpy().astype(np.float16)
                out['kc'] = kc[g].cpu().numpy().astype(np.float16)
    res = {k: np.stack(out[k]).astype(np.float16) for k in ('att', 'con', 'q')}
    for k in ('v', 'kc'):
        if k in out:
            res[k] = out[k]
    return res


def tf_inputs(rt, prep, fmt: str, truth: dict, vis=None):
    ids = rt.tokenizer(answer_text(fmt, truth), add_special_tokens=False)['input_ids']
    vals = cf.answer_values(rt, ids, fmt)
    if vals is None:
        raise ValueError(f'Cannot parse the teacher-forced answer {answer_text(fmt, truth)!r}')
    x, pos = cr.teacher_forced(rt, prep, ids, x=cr.embeds(rt, prep, vis=vis))
    return ids, vals, x, pos, cf.first_digit_queries(vals, len(prep.ids))


def photos_list(args) -> list[Path]:
    return sorted(Path(args.photo_dir).glob('*.png'))


# ----------------------------------------------------------------------------- stage factors

def stage_factors(rt, out: Path, args) -> None:
    mdir = out / f'factors_{args.tag}'
    mdir.mkdir(exist_ok=True)
    idir = out / 'factor_images'
    idir.mkdir(exist_ok=True)
    photos = photos_list(args)
    heads = HEADS[rt.family]
    for c in factor_cases(args.n_factor, args.seed):
        f = mdir / f"{c['id']}.npz"
        if f.exists():
            continue
        t0 = time.time()
        image, truth = compose(c['variant'], c['template'], c['target'], c['seed'], photos)
        stem = c['id'].rsplit('_', 1)[0]
        if not (idir / f'{stem}.png').exists():
            image.save(idir / f'{stem}.png')
        prep = ra.prepare(rt, image, c['prompt'])
        H, W = prep.grid
        if H * TOKEN != truth['canvas'][1] or W * TOKEN != truth['canvas'][0]:
            raise ValueError(f"Image was resized for {c['id']}: grid {prep.grid}, canvas {truth['canvas']}")
        target = cr.tokens_overlapping(truth['box'], prep.grid)
        ids, vals, x, pos, queries = tf_inputs(rt, prep, c['format'], truth)
        rec = measure(rt, prep, x, pos, queries, heads, ra.ds_of(rt, prep), prep.span['start'] + min(target),
                      main=MAIN[rt.family])
        np.savez_compressed(f, **rec, heads=np.array(heads), keys=np.array([v['key'] for v in vals]),
                            target=np.array(target), grid=np.array(prep.grid), ink=np.array(truth['ink'], np.float32),
                            meta=json.dumps({**c, 'truth': {k: v for k, v in truth.items() if k != 'ink'}}))
        print(f"factors {c['id']} {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- stage locate

def locate_conditions(rt) -> list[dict]:
    """Patch conditions: site kind, site index and direction (remove: column 0 <- S; implant: S <- column 0)."""
    conds = [{'name': 'clean'}]
    depth = len(rt.model.model.visual.blocks)
    for b in VIT_SITES:
        if b <= depth:
            conds.append({'name': f'vit{b}', 'vit': b})
    conds += [{'name': 'emb', 'emb': True}, {'name': 'emb_vis', 'emb': True, 'ds_keep': True}]
    if rt.family == 'qwen3_vl':
        conds.append({'name': 'emb_ds', 'ds_only': True})
    for l in DEC_SITES:
        conds.append({'name': f'dec{l}', 'dec': l})
    conds += [{'name': 'imp_vit0', 'vit': 0, 'implant': True}, {'name': 'imp_emb', 'emb': True, 'implant': True},
              {'name': 'imp_dec12', 'dec': 12, 'implant': True}]
    return conds


def stage_locate(rt, out: Path, args) -> None:
    mdir = out / f'locate_{args.tag}'
    mdir.mkdir(exist_ok=True)
    heads = [h for h in HEADS[rt.family] if h[0] <= max(MAIN[rt.family][0], 29 if rt.family == 'qwen3_vl' else 24)]
    main = MAIN[rt.family]
    conds = locate_conditions(rt)
    rng = np.random.default_rng(args.seed + 7)
    for i in range(args.n_locate):
        f = mdir / f'grid_{i:03d}.npz'
        if f.exists():
            continue
        t0 = time.time()
        r, c = int(rng.integers(0, 10)), int(rng.integers(3, 10))
        tpl = TEMPLATES[i % 4]
        seed = int(args.seed * 1000 + 500 + i)
        image, truth = compose('edge', tpl, (r, c), seed, [])
        prep = ra.prepare(rt, image, cf.T6_QUERY + cf.FORMAT_SUFFIX['rowcol'])
        grid = prep.grid
        S = source_column(c)
        d0, sS = patch_rows(grid, 0, S)
        target = cr.tokens_overlapping(truth['box'], grid)
        ref = prep.span['start'] + min(target)
        span = (prep.span['start'], prep.span['end'])
        recs = {}
        stats = None
        for cond in conds:
            dst, src = (sS, d0) if cond.get('implant') else (d0, sS)
            vb = cond.get('vit')
            if cond['name'] == 'clean':
                with stage_capture(rt, span, sorted(set(d0 + sS + [i2 * grid[1] + 10 for i2 in range(grid[0])])),
                                   main[0]) as st:
                    vis, ds = vision(rt, prep)
                    ids, vals, x, pos, queries = tf_inputs(rt, prep, 'rowcol', truth, vis=vis)
                    rec = measure(rt, prep, x, pos, queries, heads, ds, ref, main=main)
                stats = st
            else:
                with vit_patch(rt, vb, patch_index(dst), patch_index(src)):
                    vis, ds = vision(rt, prep)
                vis = vis.clone()
                dt, st_ = torch.tensor(dst, device=vis.device), torch.tensor(src, device=vis.device)
                if cond.get('emb') and not cond.get('ds_only'):
                    vis[dt] = vis[st_]
                if ds is not None and (cond.get('ds_only') or (cond.get('emb') and not cond.get('ds_keep'))):
                    ds = [d.clone() for d in ds]
                    for d in ds:
                        d[dt] = d[st_]
                ids, vals, x, pos, queries = tf_inputs(rt, prep, 'rowcol', truth, vis=vis)
                with decoder_patch(rt, cond.get('dec'), span, dst, src):
                    rec = measure(rt, prep, x, pos, queries, heads, ds, ref, main=main)
            recs[cond['name']] = rec
        names = list(recs)
        # Key-side content scores: the clean run's main-head query against every condition's content keys.
        mi = [tuple(h) for h in heads].index(main)
        q0 = recs['clean']['q'][mi].astype(np.float32)                              # (Q, hd)
        cq = np.stack([recs[n]['kc'].astype(np.float32) @ q0.T / np.sqrt(q0.shape[-1]) for n in names])
        tok = sorted(set(d0 + sS + [i2 * grid[1] + 10 for i2 in range(grid[0])]))
        np.savez_compressed(
            f, conds=np.array(names), heads=np.array(heads), keys=np.array([v['key'] for v in vals]),
            att=np.stack([recs[n]['att'] for n in names]), con=np.stack([recs[n]['con'] for n in names]),
            cq=cq.transpose(0, 2, 1).astype(np.float16),
            q=recs['clean']['q'], kc=recs['clean']['kc'], v=recs['clean']['v'],
            target=np.array(target), grid=np.array(grid), S=S, stat_tokens=np.array(tok),
            meta=json.dumps({'template': tpl, 'target': [r, c], 'seed': seed}))
        if i < args.n_stats:
            # Pooled stage-wise states for the CPU probes (scratch; deleted once the summary is final).
            np.savez_compressed(mdir / f'stats_{i:03d}.npz', stat_tokens=np.array(tok), grid=np.array(grid), S=S,
                                vit=np.stack([stats['vit'][b].numpy() for b in sorted(stats['vit'])]),
                                dec=np.stack([stats['dec'][l].numpy() for l in sorted(stats['dec'])]))
        if i == 0:
            image.save(out / f'locate_example_{args.tag}.png')
        print(f'locate {i} {time.time() - t0:.1f}s', flush=True)
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- stage edits

def edit_groups(grid, target: list[int], ink: np.ndarray, dr: int) -> dict[str, tuple[list[int], list[int]]]:
    """(destinations, sources) of the shifted-value edits: A0 = column 0 of the target rows; F = first token of
    the target rows with drawn pixels left of the target (the first figure of the row); Rin = other target-row
    tokens left of the target, excluding A0 and F."""
    H, W = grid
    T = set(target)
    rows = sorted({t // W for t in target})
    c0 = min(t % W for t in target)
    groups = {'A0': [], 'F': [], 'Rin': []}
    for i in rows:
        line = [i * W + j for j in range(c0)]
        first = next((t for t in line if ink[t] > 0.02), None)
        for t in line:
            if t in T:
                continue
            key = 'A0' if t % W == 0 else ('F' if t == first else 'Rin')
            groups[key].append(t)
    out = {}
    for k, toks in groups.items():
        out[k] = ra.shifted(toks, grid, dr, 0)
    return out


@torch.inference_mode()
def edit_run(rt, xa, xb, pos, L, span, plans: list[tuple[str, str, list[int], list[int]]], ds_a, ds_b):
    """Teacher-forced tail logits: row 0 = run B, row 1 = run A, row 1 + i = run A with plans[i] applied to the
    cached values of every full-attention layer (kind 'self': run-A values of the sources; 'B': run-B values)."""
    s, e = span
    R = 2 + len(plans)
    x = torch.cat([xb] + [xa] * (R - 1), 0)
    p = pos.expand(-1, R, -1).contiguous()
    ds = None if ds_a is None else [ds_b] + [ds_a] * (R - 1)
    with cv.deepstack_rows(rt, ds, span):
        o = rt.model(inputs_embeds=x[:, :L - 1], position_ids=p[:, :, :L - 1], use_cache=True, logits_to_keep=1)
    cache = o.past_key_values
    for _, ci in cv.kv_index(rt).items():
        _, values = sr.cache_kv(cache, ci)
        ref = values[1].clone()
        for b, (_, kind, dst, src) in enumerate(plans, start=2):
            if not dst:
                continue
            d = torch.tensor(dst, device=values.device) + s
            sidx = torch.tensor(src, device=values.device) + s
            values[b, :, d] = (values[0] if kind == 'B' else ref)[:, sidx]
    o = rt.model(inputs_embeds=x[:, L - 1:], position_ids=p[:, :, L - 1:], past_key_values=cache, use_cache=True)
    return o.logits


def stage_edits(rt, out: Path, args) -> None:
    path = out / f'edits_{args.tag}.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    digits = cf.digit_ids(rt)
    rng = np.random.default_rng(args.seed + 11)
    for variant in ('edge', 'margin_left'):
        for i in range(args.n_edits):
            r, c = int(rng.integers(0, 10)), int(rng.integers(3, 10))
            cid = f'{variant}_{i:03d}'
            if cid in done:
                continue
            t0 = time.time()
            tpl, seed = TEMPLATES[i % 4], int(args.seed * 1000 + 800 + i)
            dr_cells = 2 if r <= 7 else -2
            ia, ta = compose(variant, tpl, (r, c), seed, [])
            ib, tb = compose(variant, tpl, (r + dr_cells, c), seed, [])
            prompt = cf.T6_QUERY + cf.FORMAT_SUFFIX['rowcol']
            pa, pb = ra.prepare(rt, ia, prompt), ra.prepare(rt, ib, prompt)
            L = len(pa.ids)
            if len(pb.ids) != L:
                raise ValueError(f'Run B length differs for {cid}')
            span = (pa.span['start'], pa.span['end'])
            target = cr.tokens_overlapping(ta['box'], pa.grid)
            moved = cr.tokens_overlapping(tb['box'], pa.grid)
            ids, vals, xa, pos, _ = tf_inputs(rt, pa, 'rowcol', ta)
            xb, _ = cr.teacher_forced(rt, pb, ids, x=cr.embeds(rt, pb))
            dr = 2 * dr_cells
            g = edit_groups(pa.grid, target, np.array(ta['ink']), dr)
            plans = [('vT', 'B', sorted(target), sorted(moved))]
            plans += [(f'v{k}', 'self', d, s_) for k, (d, s_) in g.items()]
            lg = edit_run(rt, xa, xb, pos, L, span, plans, ra.ds_of(rt, pa), ra.ds_of(rt, pb))
            tab = cf.ld_table(lg, ids + [0], vals, digits, b_row=0)
            append_jsonl(path, [{'id': cid, 'variant': variant, 'target_rc': [r, c], 'dr_cells': dr_cells,
                                 'rows': ['none'] + [p[0] for p in plans],
                                 'sizes': {p[0]: len(p[2]) for p in plans}, 'values': tab,
                                 'groups': {k: v[0] for k, v in g.items()}, 'seconds': round(time.time() - t0, 1)}])
            print(f'edits {cid} {time.time() - t0:.1f}s', flush=True)
            torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--stages', nargs='+', default=['factors', 'locate', 'edits'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--photo-dir', type=Path, required=True, help='RefCOCO photos used as backgrounds')
    p.add_argument('--n-factor', type=int, default=24)
    p.add_argument('--n-locate', type=int, default=32)
    p.add_argument('--n-edits', type=int, default=32)
    p.add_argument('--n-stats', type=int, default=16, help='locate images whose stage-wise states are stored')
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--torch-deltanet', action='store_true', help='Qwen3.5 on A40: torch DeltaNet kernel')
    args = p.parse_args()
    if args.torch_deltanet:
        from analysis_tools.position import causal_vs_linear_run as cvr
        cvr.use_torch_deltanet()
    args.output.mkdir(parents=True, exist_ok=True)
    rt = ra.load_runtime(args.model)
    args.tag = 'vl' if rt.family == 'qwen3_vl' else 'q35'
    append_jsonl(args.output / 'run_log.jsonl', [{
        'stages': args.stages, 'tag': args.tag, 'heads': HEADS[rt.family], 'main': MAIN[rt.family],
        'args': {k: str(v) for k, v in vars(args).items()}, 'runtime': qr.describe(rt),
        'device': torch.cuda.get_device_name(0), 'started': datetime.now(timezone.utc).isoformat()}])
    for st in args.stages:
        {'factors': stage_factors, 'locate': stage_locate, 'edits': stage_edits}[st](rt, args.output, args)


if __name__ == '__main__':
    main()
