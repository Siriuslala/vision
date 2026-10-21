"""Experiment 3.7 round 2: ViT position on synthetic vs natural images (engine).

Reusable pieces (stage drivers live in `vit_layout_r2_run.py`):

* Stimulus ladder: every stimulus is derived from one experiment-3.4 RefCOCO test image (same canvas, so
  the token grid is shared across the ladder): a red disc on white / patch-periodic checker / pixel noise /
  block-shuffled photo / the photo itself; the referent (segmentation mask) on white / on its block-shuffled
  photo; the photo; the vertically flipped photo.
* `RowSpec` and `run_rows(...)`: batched greedy generation in which each row (one image + prompt) carries
  (a) a replacement of the ViT position inputs (PE mode / scale / patch permutation, RoPE mode / patch
  permutation) and (b) a gather on the vision-tower output (merger and DeepStack features) that may take
  tokens from another row of the same batch. Content swaps and token-group mixing are both gathers. The
  same code path serves Qwen3.5 and Qwen3-VL (DeepStack features are gathered with the merger output).
* Region helpers: target tokens, displaced regions T' (3.5 rule plus homogeneity-selected variants).

Patch-level tensors are in the processor's spatial-merge-block order; merged tokens are in raster order,
which equals the block order of the 2x2 blocks (see vit_layout.py).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import vit_layout as vl

TOKEN = 32                      # pixels per merged token side
RED = (255, 0, 0)
REFCOCO_QUERY = 'Locate the object described as: {expr}.'
DISC_PROMPT = cr.DISC_QUERY + cr.SUFFIX['point']
STIMULI = ('disc_white', 'disc_checker', 'disc_noise', 'disc_shuffled', 'disc_scene',
           'obj_white', 'obj_shuffled', 'natural', 'natural_vflip')
MOVABLE = {'disc_white', 'disc_checker', 'disc_noise', 'disc_shuffled', 'disc_scene', 'obj_white', 'obj_shuffled'}
VERTICAL_WORDS = re.compile(r'\b(top|bottom|above|below|upper|lower|up|down|under|over|high|higher|highest|low|'
                            r'lowest|underneath|beneath|front|back|behind|far|near|closest|nearest)\b', re.I)


# ----------------------------------------------------------------------------- cases

def load_cases(coord_cases: Path, probe_cases: Path, refcoco_dir: Path) -> list[dict]:
    """Point cases of the 3.5 RefCOCO set (3.4 test images) with the referent's segmentation polygon.

    coord_cases: experiment-3.5 refcoco/cases.json; probe_cases: experiment-3.4 cases.json (shard, row);
    refcoco_dir: RefCOCO parquet root. The polygon is rescaled to the 512-pixel input image.
    """
    import pyarrow.parquet as pq
    meta = {c['id']: c for c in json.loads(probe_cases.read_text())['natural']}
    tables, out = {}, []
    for c in json.loads(coord_cases.read_text()):
        if c['format'] != 'point':
            continue
        src = meta[c['source_case']]
        if src['shard'] not in tables:
            tables[src['shard']] = pq.read_table(refcoco_dir / 'data' / src['shard'],
                                                 columns=['segmentation', 'file_name', 'question_id'])
        row = tables[src['shard']].slice(src['row'], 1).to_pylist()[0]
        if row['file_name'] != src['file_name'] or str(row['question_id']) != str(c['question_id']):
            raise ValueError(f"RefCOCO row mismatch for case {c['id']}")
        sw, sh = src['source_size']
        iw, ih = src['input_size']
        seg = np.asarray(row['segmentation'], float).reshape(-1, 2) * [iw / sw, ih / sh]
        out.append({'id': c['image_key'], 'image_path': c['image_path'], 'image_sha256': c['image_sha256'],
                    'expression': c['expression'], 'box': c['truth']['box'], 'canvas': c['truth']['canvas'],
                    'polygon': seg.round(2).tolist(), 'source_case': c['source_case']})
    return out


def load_image(case: dict) -> Image.Image:
    p = Path(case['image_path'])
    if hashlib.sha256(p.read_bytes()).hexdigest() != case['image_sha256']:
        raise ValueError(f'Image hash mismatch for {p}')
    im = Image.open(p).convert('RGB')
    if list(im.size) != list(case['canvas']):
        raise ValueError(f"{p}: size {im.size} != {case['canvas']}")
    return im


def object_mask(case: dict) -> np.ndarray:
    """(H, W) bool mask of the referent polygon."""
    W, H = case['canvas']
    m = Image.new('L', (W, H), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in case['polygon']], fill=1)
    return np.asarray(m, bool)


def block_shuffle(image: Image.Image, seed: int, block: int = TOKEN) -> Image.Image:
    """Permute the token-aligned blocks of an image (each block keeps its pixels, context is destroyed)."""
    a = np.asarray(image)
    H, W = a.shape[0] // block, a.shape[1] // block
    if (H * block, W * block) != a.shape[:2]:
        raise ValueError('Image size is not a multiple of the block size.')
    tiles = a.reshape(H, block, W, block, 3).transpose(0, 2, 1, 3, 4).reshape(H * W, block, block, 3)
    perm = np.random.default_rng(seed).permutation(H * W)
    return Image.fromarray(tiles[perm].reshape(H, W, block, block, 3).transpose(0, 2, 1, 3, 4).reshape(a.shape))


def checker(W: int, H: int, period: int = 16, lo: int = 200, hi: int = 240) -> np.ndarray:
    """Gray checkerboard whose period equals one ViT patch: every patch has identical pixels."""
    y, x = np.mgrid[0:H, 0:W]
    v = np.where(((x // (period // 2)) + (y // (period // 2))) % 2 == 0, lo, hi).astype(np.uint8)
    return np.repeat(v[..., None], 3, 2)


def disc_geometry(case: dict) -> tuple[list[float], int]:
    """Disc centered on the referent box; diameter 0.8 x the short box side, clipped to [32, 96] pixels."""
    x0, y0, x1, y1 = case['box']
    d = int(np.clip(0.8 * min(x1 - x0, y1 - y0), 32, 96))
    W, H = case['canvas']
    cx = float(np.clip((x0 + x1) / 2, d / 2, W - d / 2))
    cy = float(np.clip((y0 + y1) / 2, d / 2, H - d / 2))
    return [cx, cy], d


def target_box(case: dict, stimulus: str) -> list[float]:
    """Pixel box of the target of a stimulus (disc bbox or referent bbox; flipped for natural_vflip)."""
    if stimulus.startswith('disc'):
        (cx, cy), d = disc_geometry(case)
        return [cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2]
    x0, y0, x1, y1 = case['box']
    if stimulus == 'natural_vflip':
        H = case['canvas'][1]
        return [x0, H - y1, x1, H - y0]
    return [x0, y0, x1, y1]


def render(case: dict, stimulus: str, shift: tuple[int, int] = (0, 0), seed: int = 0) -> tuple[Image.Image, dict, str]:
    """Image, truth and prompt of one ladder stimulus; shift = (dx, dy) pixels moves the target (real move).

    Truth: {'canvas': [W, H], 'box': pixel box, 'center': [x, y], 'diameter' (discs only)}.
    """
    W, H = case['canvas']
    dx, dy = shift
    if (dx or dy) and stimulus not in MOVABLE:
        raise ValueError(f'{stimulus} has no real-move version.')
    photo = load_image(case)
    if stimulus.startswith('disc'):
        if stimulus == 'disc_white':
            bg = np.full((H, W, 3), 255, np.uint8)
        elif stimulus == 'disc_checker':
            bg = checker(W, H)
        elif stimulus == 'disc_noise':
            rng = np.random.default_rng(seed)
            bg = np.repeat(rng.integers(170, 256, (H, W, 1), dtype=np.uint8), 3, 2)
        elif stimulus == 'disc_shuffled':
            bg = np.asarray(block_shuffle(photo, seed)).copy()
        elif stimulus == 'disc_scene':
            bg = np.asarray(photo).copy()
        else:
            raise ValueError(stimulus)
        (cx, cy), d = disc_geometry(case)
        cx, cy = cx + dx, cy + dy
        img = Image.fromarray(bg)
        ImageDraw.Draw(img).ellipse((cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2), fill=RED)
        truth = {'canvas': [W, H], 'box': [cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2], 'center': [cx, cy],
                 'diameter': d}
        return img, truth, DISC_PROMPT
    prompt = REFCOCO_QUERY.format(expr=case['expression']) + cr.SUFFIX['point']
    if stimulus in ('natural', 'natural_vflip'):
        img = photo if stimulus == 'natural' else photo.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        box = target_box(case, stimulus)
    else:
        mask = object_mask(case)
        bg = np.full((H, W, 3), 255, np.uint8) if stimulus == 'obj_white' else np.asarray(block_shuffle(photo, seed)).copy()
        src = np.asarray(photo)
        ys, xs = np.nonzero(mask)
        yt, xt = ys + dy, xs + dx
        ok = (yt >= 0) & (yt < H) & (xt >= 0) & (xt < W)
        if not ok.all():
            raise ValueError(f"{case['id']}: moved object leaves the canvas")
        bg[yt, xt] = src[ys, xs]
        img = Image.fromarray(bg)
        x0, y0, x1, y1 = case['box']
        box = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
    return img, {'canvas': [W, H], 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]}, prompt


def score_point(truth: dict, text: str) -> dict:
    """Strict point score (0-1000 convention): hit inside the target box (disc: inside the disc) and
    center error; plus the diagnostic lenient parse for repeated identical points (Qwen3-VL)."""
    coords = cr.parse_coords(text, 'point')
    lenient = False
    if coords is None:
        pts = re.findall(r'"point_2d":\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', text)
        if pts and len(set(pts)) == 1:
            coords, lenient = [float(v) for v in pts[0]], True
    if coords is None:
        return {'valid': False, 'lenient_valid': False, 'hit': False}
    W, H = truth['canvas']
    c = np.asarray(coords, float)
    px = c * [W / 1000, H / 1000]
    if 'diameter' in truth:
        hit = float(np.linalg.norm(px - np.asarray(truth['center']))) <= truth['diameter'] / 2
    else:
        x0, y0, x1, y1 = truth['box']
        hit = x0 <= px[0] <= x1 and y0 <= px[1] <= y1
    t = 1000 * np.asarray(truth['center']) / [W, H]
    return {'valid': not lenient, 'lenient_valid': True, 'center': c.tolist(), 'hit': bool(hit),
            'error': float(np.linalg.norm(c - t))}


# ----------------------------------------------------------------------------- regions

def displaced(target: list[int], grid: tuple[int, int], dr: int, dc: int) -> list[int] | None:
    H, W = grid
    out = []
    for t in target:
        r, c = divmod(t, W)
        if not (0 <= r + dr < H and 0 <= c + dc < W):
            return None
        out.append((r + dr) * W + c + dc)
    return None if set(out) & set(target) else out


def primary_region(target: list[int], grid: tuple[int, int]) -> dict | None:
    """3.5 rule: translate T by its own extent plus one token toward the larger free side, horizontal
    first, vertical if the horizontal copy does not fit."""
    H, W = grid
    rows = sorted({t // W for t in target})
    cols = sorted({t % W for t in target})
    for axis in ('h', 'v'):
        lo, hi, n = (cols[0], cols[-1], W) if axis == 'h' else (rows[0], rows[-1], H)
        k = hi - lo + 2
        toward = 1 if (n - 1 - hi) >= lo else -1
        dr, dc = (0, toward * k) if axis == 'h' else (toward * k, 0)
        moved = displaced(target, grid, dr, dc)
        if moved is not None:
            return {'axis': axis, 'dr': dr, 'dc': dc, 'region': moved, 'd': [1000 * dc / W, 1000 * dr / H]}
    return None


def token_texture(image: Image.Image) -> np.ndarray:
    """(H*W,) gray-level standard deviation inside each 32x32 token cell (raster order)."""
    g = np.asarray(image.convert('L'), np.float32)
    H, W = g.shape[0] // TOKEN, g.shape[1] // TOKEN
    return g.reshape(H, TOKEN, W, TOKEN).std((1, 3)).ravel()


def homogeneity_regions(target: list[int], grid: tuple[int, int], texture: np.ndarray, axis: str) -> dict:
    """Most homogeneous and most textured displaced copies of T along one axis (|shift| >= extent + 1)."""
    H, W = grid
    rows = sorted({t // W for t in target})
    cols = sorted({t % W for t in target})
    extent = (cols[-1] - cols[0] + 1) if axis == 'h' else (rows[-1] - rows[0] + 1)
    cands = []
    n = W if axis == 'h' else H
    for s in range(-n, n + 1):
        if abs(s) < extent + 1:
            continue
        dr, dc = (0, s) if axis == 'h' else (s, 0)
        moved = displaced(target, grid, dr, dc)
        if moved is not None:
            cands.append((float(texture[moved].mean()), dr, dc, moved))
    if len(cands) < 2:
        return {}
    cands.sort(key=lambda c: c[0])
    out = {}
    for name, c in (('homog', cands[0]), ('textured', cands[-1])):
        out[name] = {'axis': axis, 'dr': c[1], 'dc': c[2], 'region': c[3], 'd': [1000 * c[2] / W, 1000 * c[1] / H],
                     'texture': c[0]}
    return out


# ----------------------------------------------------------------------------- ViT edits and output gathers

@dataclass
class RowSpec:
    """One generation row.

    pe:        'interp' | 'zero' | 'mean' | 'shuffle'   (vit_layout.ViTEdit semantics), then pe_scale.
    pe_perm:   optional (Np,) patch permutation of the PE (patch i receives the PE of patch pe_perm[i]).
    rope:      'grid' | 'zero' | 'shuffle'.
    rope_perm: optional (Np,) patch permutation of the RoPE ids.
    gather:    optional (Nv, 2) int array: output token j of this row is token gather[j, 1] of row
               (this row's batch index + gather[j, 0]); None keeps the row's own output.
    """
    image: Image.Image
    prompt: str
    pe: str = 'interp'
    pe_scale: float = 1.0
    pe_perm: np.ndarray | None = None
    rope: str = 'grid'
    rope_perm: np.ndarray | None = None
    gather: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def region_patch_perm(n_patches: int, a: list[int], b: list[int]) -> np.ndarray:
    """Patch permutation (block order) that exchanges the patches of merged-token regions a and b."""
    perm = np.arange(n_patches)
    pa, pb = cr.patch_rows(a).numpy(), cr.patch_rows(b).numpy()
    perm[pa], perm[pb] = pb, pa
    return perm


def own_gather(n: int, order: np.ndarray | None = None, offset: int = 0) -> np.ndarray:
    order = np.arange(n) if order is None else order
    return np.stack([np.full(n, offset), order], 1)


def token_swap_order(n: int, a: list[int], b: list[int]) -> np.ndarray:
    return cr.swap_index(n, a, b).numpy()


def _pe_rope(table: torch.Tensor, spec: RowSpec, hp: int, wp: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    edit = vl.ViTEdit(pe=spec.pe, rope=spec.rope, pe_scale=spec.pe_scale, seed=int(spec.meta.get('seed', 0)))
    pe = vl.edited_pe(table, edit, hp, wp)
    ids = vl.edited_rope(edit, hp, wp, device)
    if spec.pe_perm is not None:
        pe = pe[torch.as_tensor(spec.pe_perm, device=pe.device)]
    if spec.rope_perm is not None:
        ids = ids[torch.as_tensor(spec.rope_perm, device=ids.device)]
    return pe, ids


@contextmanager
def row_hooks(rt: qr.Runtime, specs: list[RowSpec], keep: list[dict] | None = None):
    """Apply per-row ViT position inputs and output gathers in the next vision forward.

    keep: if given, receives one dict per row with the final merger output ('merger', (Nv, D) float16 CPU).
    """
    visual = rt.model.model.visual
    table = visual.pos_embed.weight
    state = {}

    def pre(module, args, kwargs):
        grid = kwargs['grid_thw']
        if len(grid) != len(specs):
            raise ValueError(f'{len(specs)} row specs for {len(grid)} images.')
        pes, ids, sizes = [], [], []
        for s, (t, hp, wp) in zip(specs, grid.tolist()):
            if t != 1:
                raise ValueError('Only single images are supported.')
            pe, rid = _pe_rope(table, s, hp, wp, grid.device)
            pes.append(pe)
            ids.append(rid)
            sizes.append(hp * wp // 4)
        state['pe'] = torch.cat(pes)
        state['sizes'] = sizes
        n = len(state['pe'])
        kwargs['interp_indices'] = torch.zeros(n, 1, device=grid.device, dtype=torch.long)
        kwargs['interp_weights'] = torch.ones(n, 1, device=grid.device, dtype=torch.float32)
        kwargs['position_ids'] = torch.cat(ids)
        return args, kwargs

    def emb(module, inputs, output):
        if 'pe' not in state:
            return output
        pe = state.pop('pe')
        if output.shape[:2] != (len(pe), 1):
            raise RuntimeError('pos_embed lookup does not match the injected identity taps.')
        return pe[:, None].to(output.dtype)

    def gather_rows(t: torch.Tensor) -> torch.Tensor:
        parts = list(torch.split(t, state['sizes']))
        out = []
        for i, s in enumerate(specs):
            if s.gather is None:
                out.append(parts[i])
                continue
            g = np.asarray(s.gather)
            if len(g) != len(parts[i]):
                raise ValueError('Gather length does not match the row token count.')
            src = [parts[i + int(o)][int(j)] for o, j in g]
            out.append(torch.stack(src))
        return torch.cat(out)

    def post(module, inputs, output):
        output.pooler_output = gather_rows(output.pooler_output)
        ds = getattr(output, 'deepstack_features', None)
        if ds:
            output.deepstack_features = [gather_rows(d) for d in ds]
        if keep is not None:
            for p in torch.split(output.pooler_output, state['sizes']):
                keep.append({'merger': p.detach().to('cpu', torch.float16)})
        return output

    hs = [visual.register_forward_pre_hook(pre, with_kwargs=True), visual.pos_embed.register_forward_hook(emb),
          visual.register_forward_hook(post)]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def run_rows(rt: qr.Runtime, specs: list[RowSpec], max_new_tokens: int, keep_merger: bool = False) -> list[dict]:
    """Greedy-decode all rows in one left-padded batch; gathers may only reference rows of this batch."""
    for i, s in enumerate(specs):
        if s.gather is not None:
            offs = np.asarray(s.gather)[:, 0]
            if ((i + offs) < 0).any() or ((i + offs) >= len(specs)).any():
                raise ValueError('Gather references a row outside the batch.')
    keep = [] if keep_merger else None
    with row_hooks(rt, specs, keep):
        res = qr.generate_batch(rt, [([s.image], s.prompt) for s in specs], max_new_tokens)
    if keep_merger:
        for r, k in zip(res, keep):
            r.update(k)
    return res


def w_shift(center: list[float] | None, base: list[float] | None, d: list[float]) -> float | None:
    """Displacement weight w = (y - y0) . d / |d|^2 (0-1000 units)."""
    if center is None or base is None:
        return None
    d = np.asarray(d, float)
    return float((np.asarray(center) - np.asarray(base)) @ d / (d @ d))
