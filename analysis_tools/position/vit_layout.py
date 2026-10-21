"""Experiment 3.7: position and layout representations inside the ViT of Qwen3.5 / Qwen3-VL.

Reusable engine (stage drivers live in `vit_layout_run.py`):

* `ViTEdit` and `vit_edit(...)`: a context manager that replaces the two ViT position inputs of every
  image passing through the vision tower -- the learned absolute position embedding (PE) added once after
  the patch embedding, and the 2D RoPE ids used by all 27 attention blocks. Edits are computed per image
  from its patch grid, so the same edit works inside batched `generate` calls and for Qwen3-VL, whose
  DeepStack features are produced inside the same vision forward.
* `vit_capture(...)`: records patch content, ViT input (content + PE), every block output and the merger
  output of one vision forward.
* `merger_edit(...)`: applies a function to the merger output (used for linear position erasure).
* stimuli: red-disc localization on canvases of any size, two-figure same/different pairs (T2) on
  canvases of any size, and two-object layout images.
* `LeaceEraser`: closed-form linear concept erasure (LEACE, Belrose et al. 2023) fitted from moments.

All patch-level tensors are in the processor's spatial-merge-block order (2x2 blocks, raster over blocks),
the order in which the vision tower and the merger consume patches.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, asdict
import math

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.common.stimuli import NONE, Change, render_instance
from analysis_tools.position.coordinate_readout import disc_image as cr_disc_image

PATCH = 16
MERGE = 2
TABLE_SIDE = 48                 # 48 x 48 learned PE table; native size = 768 pixels


# ----------------------------------------------------------------------------- grid helpers

def to_block(t: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
    """Raster (hp*wp, ...) patch tensor -> spatial-merge-block order."""
    rest = t.shape[1:]
    return t.reshape(hp // MERGE, MERGE, wp // MERGE, MERGE, *rest).transpose(1, 2).reshape(hp * wp, *rest)


def to_raster(t: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
    """Spatial-merge-block order -> raster (hp*wp, ...)."""
    rest = t.shape[1:]
    return t.reshape(hp // MERGE, wp // MERGE, MERGE, MERGE, *rest).transpose(1, 2).reshape(hp * wp, *rest)


def patch_xy(hp: int, wp: int) -> np.ndarray:
    """(hp*wp, 2) normalized 0-1000 centers (x, y) of patches in block order."""
    r, c = np.meshgrid(np.arange(hp), np.arange(wp), indexing='ij')
    xy = np.stack([(c.ravel() + .5) / wp * 1000, (r.ravel() + .5) / hp * 1000], 1)
    return to_block(torch.from_numpy(xy), hp, wp).numpy()


def token_xy(h: int, w: int) -> np.ndarray:
    """(h*w, 2) normalized 0-1000 centers (x, y) of merged tokens in raster order."""
    r, c = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    return np.stack([(c.ravel() + .5) / w * 1000, (r.ravel() + .5) / h * 1000], 1)


def interp_taps(hp: int, wp: int, device=None, side: int = TABLE_SIDE) -> tuple[torch.Tensor, torch.Tensor]:
    """Default bilinear taps of a (hp, wp) patch grid into the PE table, block order."""
    grid = torch.tensor([[1, hp, wp]], device=device)
    return get_vision_interpolation_indices_and_weights(grid, side, mode='bilinear', align_corners=True,
                                                        spatial_merge_size=MERGE)


def window_taps(hp: int, wp: int, vh: int, vw: int, r0: int, c0: int, device=None,
                side: int = TABLE_SIDE) -> tuple[torch.Tensor, torch.Tensor]:
    """Taps of the (hp, wp) window at patch offset (r0, c0) of a virtual (vh, vw) canvas, block order.

    The window receives exactly the PE values its patches would have in the full virtual canvas.
    """
    if r0 < 0 or c0 < 0 or r0 + hp > vh or c0 + wp > vw:
        raise ValueError('PE window exceeds the virtual canvas.')
    grid = torch.tensor([[1, vh, vw]], device=device)
    idx, w = get_vision_interpolation_indices_and_weights(grid, side, mode='bilinear', align_corners=True,
                                                          spatial_merge_size=1)
    idx = idx.view(vh, vw, -1)[r0:r0 + hp, c0:c0 + wp].reshape(hp * wp, -1)
    w = w.view(vh, vw, -1)[r0:r0 + hp, c0:c0 + wp].reshape(hp * wp, -1)
    return to_block(idx, hp, wp), to_block(w, hp, wp)


def rope_ids(hp: int, wp: int, device=None) -> torch.Tensor:
    return get_vision_position_ids(torch.tensor([[1, hp, wp]], device=device), MERGE)


# ----------------------------------------------------------------------------- ViT position edits

@dataclass(frozen=True)
class ViTEdit:
    """Replacement of the ViT position inputs of each image.

    pe:        'interp' (model default: bilinear resampling of the 48x48 table), 'zero', 'mean' (the image's
               mean interpolated PE at every patch), 'shuffle' (interpolated PE randomly permuted across
               patches), 'native' (1:1 table entries of a centered crop, only for grids <= 48 patches),
               'window' (taps of a window of a virtual canvas, see window_*).
    pe_scale:  multiplies the PE after the edit.
    rope:      'grid' (model default), 'zero' (all ids 0: attention carries no position), 'shuffle'
               (ids permuted across patches), 'offset' (ids + (rope_dr, rope_dc)).
    seed:      permutation seed for the shuffles.
    """
    pe: str = 'interp'
    pe_scale: float = 1.0
    rope: str = 'grid'
    rope_dr: int = 0
    rope_dc: int = 0
    window_vh: int = 0
    window_vw: int = 0
    window_r0: int = 0
    window_c0: int = 0
    seed: int = 0

    @property
    def is_default(self) -> bool:
        return self.pe == 'interp' and self.pe_scale == 1.0 and self.rope == 'grid'

    def as_dict(self) -> dict:
        return asdict(self)


def edited_pe(table: torch.Tensor, edit: ViTEdit, hp: int, wp: int) -> torch.Tensor:
    """(hp*wp, D) float32 PE of one image after the edit, block order."""
    device = table.device
    if edit.pe == 'window':
        idx, w = window_taps(hp, wp, edit.window_vh, edit.window_vw, edit.window_r0, edit.window_c0, device)
    else:
        idx, w = interp_taps(hp, wp, device)
    pe = (table[idx].float() * w[..., None].float()).sum(1)
    if edit.pe in ('interp', 'window'):
        pass
    elif edit.pe == 'zero':
        pe = torch.zeros_like(pe)
    elif edit.pe == 'mean':
        pe = pe.mean(0, keepdim=True).expand_as(pe).clone()
    elif edit.pe == 'shuffle':
        g = torch.Generator().manual_seed(edit.seed)
        pe = pe[torch.randperm(hp * wp, generator=g).to(device)]
    elif edit.pe == 'native':
        if hp > TABLE_SIDE or wp > TABLE_SIDE:
            raise ValueError(f'Native 1:1 PE needs a grid of at most {TABLE_SIDE} patches, got {hp}x{wp}.')
        r0, c0 = (TABLE_SIDE - hp) // 2, (TABLE_SIDE - wp) // 2
        r, c = torch.meshgrid(torch.arange(hp, device=device), torch.arange(wp, device=device), indexing='ij')
        pe = to_block(table[((r + r0) * TABLE_SIDE + c + c0).flatten()].float(), hp, wp)
    else:
        raise ValueError(f'Unknown PE edit {edit.pe!r}')
    return pe * edit.pe_scale


def edited_rope(edit: ViTEdit, hp: int, wp: int, device=None) -> torch.Tensor:
    ids = rope_ids(hp, wp, device)
    if edit.rope == 'grid':
        return ids
    if edit.rope == 'zero':
        return torch.zeros_like(ids)
    if edit.rope == 'shuffle':
        g = torch.Generator().manual_seed(edit.seed + 1)
        return ids[torch.randperm(len(ids), generator=g).to(ids.device)]
    if edit.rope == 'offset':
        return ids + torch.tensor([edit.rope_dr, edit.rope_dc], device=ids.device)
    raise ValueError(f'Unknown RoPE edit {edit.rope!r}')


@contextmanager
def vit_edit(rt: qr.Runtime, edit: ViTEdit | list[ViTEdit]):
    """Replace PE and RoPE ids of every image in the next vision forwards.

    edit: one ViTEdit for all images, or a list with one edit per image of the batch (in batch order).
    The PE is injected through the pos_embed module: interp taps become (Np, 1) dummy taps with weight 1,
    and a forward hook replaces their embedding lookup by the edited PE, so pos_embeds = edited PE
    (cast to the model dtype).
    """
    visual = rt.model.model.visual
    table = visual.pos_embed.weight
    state = {}

    def pre(module, args, kwargs):
        grid = kwargs['grid_thw']
        edits = edit if isinstance(edit, list) else [edit] * len(grid)
        if len(edits) != len(grid):
            raise ValueError(f'{len(edits)} edits for {len(grid)} images.')
        pes, ids = [], []
        for e, (t, hp, wp) in zip(edits, grid.tolist()):
            if t != 1:
                raise ValueError('Only single images are supported.')
            pes.append(edited_pe(table, e, hp, wp))
            ids.append(edited_rope(e, hp, wp, grid.device))
        pe = torch.cat(pes)
        state['pe'] = pe
        n = len(pe)
        kwargs['interp_indices'] = torch.zeros(n, 1, device=grid.device, dtype=torch.long)   # lookup replaced below
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

    h1 = visual.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = visual.pos_embed.register_forward_hook(emb)
    try:
        yield
    finally:
        h1.remove()
        h2.remove()


@contextmanager
def merger_edit(rt: qr.Runtime, fn):
    """Replace the merger output m (N, D) of a vision forward by fn(m, grid_thw)."""
    visual = rt.model.model.visual
    state = {}

    def pre(module, args, kwargs):
        state['grid'] = kwargs['grid_thw']

    def post(module, inputs, output):
        return fn(output, state['grid'])

    h1 = visual.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = visual.merger.register_forward_hook(post)
    try:
        yield
    finally:
        h1.remove()
        h2.remove()


# ----------------------------------------------------------------------------- capture

def pixel_inputs(rt: qr.Runtime, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
    """pixel_values and image_grid_thw of a list of images, with the no-resize check of qwen_runtime."""
    out = rt.processor.image_processor(images=images, return_tensors='pt')
    grid = out['image_grid_thw']
    for image, (_, h, w) in zip(images, grid.tolist()):
        if (h * PATCH, w * PATCH) != (image.height, image.width):
            raise ValueError(f'Processor resized a {image.width}x{image.height} image.')
    return out['pixel_values'].to(rt.device), grid.to(rt.device)


@torch.inference_mode()
def vit_capture(rt: qr.Runtime, images: list[Image.Image], blocks: list[int] | None = None,
                edit: ViTEdit | None = None) -> dict:
    """One vision forward; returns per-image dicts of float32 GPU tensors.

    Keys: 'content' (patch embedding, Np x 1152), 'pe' (added PE), 'input' (content + PE),
    'block_k' for k in blocks (1-based block outputs), 'merger' (Nv x 4096, raster order),
    'grid' (hp, wp). Patch-level tensors are in block order.
    """
    visual = rt.model.model.visual
    blocks = list(range(1, len(visual.blocks) + 1)) if blocks is None else blocks
    edited = edit is not None and not edit.is_default
    store, handles = {}, []
    pixel, grid = pixel_inputs(rt, images)
    with ExitStack() as stack:
        if edited:
            stack.enter_context(vit_edit(rt, edit))      # registered first so the capture sees the edited PE
        else:
            store['weights'] = get_vision_interpolation_indices_and_weights(
                grid, visual.num_grid_per_side, mode='bilinear', align_corners=True, spatial_merge_size=MERGE)[1]
        handles.append(visual.patch_embed.register_forward_hook(lambda m, i, o: store.__setitem__('content', o)))
        handles.append(visual.pos_embed.register_forward_hook(lambda m, i, o: store.__setitem__('pe_taps', o)))
        for k in blocks:
            handles.append(visual.blocks[k - 1].register_forward_hook(
                lambda m, i, o, k=k: store.__setitem__(f'block_{k}', o[0] if isinstance(o, tuple) else o)))
        handles.append(visual.merger.register_forward_hook(lambda m, i, o: store.__setitem__('merger', o)))
        try:
            visual(pixel, grid_thw=grid)
        finally:
            for h in handles:
                h.remove()
    taps = store.pop('pe_taps').float()
    w = store.pop('weights', None)
    pe = taps[:, 0] if w is None else (taps * w[..., None].float()).sum(1)
    sizes = [int(h * w_) for _, h, w_ in grid.tolist()]
    out = [dict(grid=(h, w_)) for _, h, w_ in grid.tolist()]
    def split(t, unit):
        return torch.split(t.float(), [s // unit for s in sizes])
    for key, t in [('content', store['content']), ('pe', pe)] + [(f'block_{k}', store[f'block_{k}']) for k in blocks]:
        for o, part in zip(out, split(t, 1)):
            o[key] = part
    for o in out:
        o['input'] = o['content'] + o['pe']
    for o, part in zip(out, split(store['merger'], MERGE * MERGE)):
        o['merger'] = part
    return out


# ----------------------------------------------------------------------------- stimuli

RED, BLUE = (255, 0, 0), (0, 90, 255)


def disc_case(canvas: int, u: float, v: float, diameter: int) -> dict:
    return {'canvas': canvas, 'center': [u * canvas, v * canvas], 'diameter': diameter}


def disc_grid_cases(canvas: int, seed: int, n: int = 6) -> list[dict]:
    """6x6 jittered positions, diameter canvas/8 (>= 32 pixels); jitter +-canvas/32."""
    rng = np.random.default_rng(seed + canvas)
    step, jit = canvas / n, canvas // 32
    d = max(32, canvas // 8)
    out = []
    for i in range(n * n):
        r, c = divmod(i, n)
        jx, jy = rng.integers(-jit, jit + 1, 2).tolist()
        clamp = lambda v: float(min(max(v, d / 2), canvas - d / 2))       # keep the disc inside the canvas
        out.append({'id': f's{canvas}_p{i:02d}', 'canvas': canvas, 'diameter': d,
                    'center': [clamp((c + .5) * step + jx), clamp((r + .5) * step + jy)]})
    return out


def disc_render(case: dict) -> Image.Image:
    """Red disc of a disc case on a white canvas, or on per-pixel gray noise (values 170-255) when
    case['background'] == 'noise'. The truth geometry is that of the white-canvas disc."""
    image, _ = cr_disc_image(tuple(case['center']), case['diameter'], case['canvas'])
    if case.get('background', 'white') == 'white':
        return image
    rng = np.random.default_rng(case.get('noise_seed', 0))
    n = case['canvas']
    arr = np.repeat(rng.integers(170, 256, (n, n, 1), dtype=np.uint8), 3, axis=2)
    disc = (np.asarray(image) != 255).any(-1)
    arr[disc] = np.asarray(image)[disc]
    return Image.fromarray(arr)


def t2_image(canvas: int, tiles: list[np.ndarray], rng: np.random.Generator) -> tuple[Image.Image, list]:
    """Two horizontally adjacent tiles of side canvas/4 at a random 32-aligned offset."""
    cell = tiles[0].shape[0]
    xs = np.arange(0, canvas - 2 * cell + 1, 32)
    ys = np.arange(0, canvas - cell + 1, 32)
    x0, y0 = int(rng.choice(xs)), int(rng.choice(ys))
    img = np.full((canvas, canvas, 3), 255, np.uint8)
    boxes = []
    for i, tile in enumerate(tiles):
        img[y0:y0 + cell, x0 + i * cell:x0 + (i + 1) * cell] = tile
        boxes.append([x0 + i * cell, y0, x0 + (i + 1) * cell, y0 + cell])
    return Image.fromarray(img), boxes


T2_TEMPLATES = ('person', 'ring', 'square', 'arrow')


def t2_cases(canvas: int, seed: int, n_pairs: int = 24) -> list[dict]:
    """n_pairs different pairs (gap level 3 or rotation level 3) and n_pairs identical pairs."""
    rng = np.random.default_rng(seed + 7 * canvas)
    out = []
    for i in range(n_pairs):
        template = T2_TEMPLATES[i % 4]
        if i % 2 == 0 or template in ('ring', 'square'):
            change = Change('gap', 3, int(rng.integers(0, 3)))
        else:
            change = Change('rotate', 3)
        for same in (False, True):
            out.append({'id': f's{canvas}_t2_{i:02d}_{"same" if same else "diff"}', 'canvas': canvas,
                        'template': template, 'change': [change.kind, change.level, change.site], 'same': same,
                        'changed_left': bool(rng.integers(0, 2)), 'seed': int(rng.integers(0, 2 ** 31))})
    return out


def t2_render(case: dict) -> Image.Image:
    cell = case['canvas'] // 4
    plain = render_instance(case['template'], NONE, cell)
    changed = render_instance(case['template'], Change(*case['change']), cell)
    tiles = [plain, plain] if case['same'] else ([changed, plain] if case['changed_left'] else [plain, changed])
    return t2_image(case['canvas'], tiles, np.random.default_rng(case['seed']))[0]


T2_PROMPT = 'Are the two figures exactly identical? Answer with only Yes or No.'


def layout_case(i: int, seed: int, canvas: int = 512) -> dict:
    """Red disc and blue square at random non-overlapping positions with random sizes (pixels)."""
    rng = np.random.default_rng(seed * 100003 + i)
    while True:
        s = rng.integers(40, 161, 2)
        c = [rng.uniform(s[k] / 2 + 8, canvas - s[k] / 2 - 8, 2) for k in range(2)]
        gap = np.abs(c[0] - c[1]) - (s[0] + s[1]) / 2
        if gap.max() > 16 and abs(int(s[0]) - int(s[1])) >= 12 and abs(c[0][0] - c[1][0]) > 24 \
                and abs(c[0][1] - c[1][1]) > 24:
            break
    return {'id': f'layout_{i:05d}', 'canvas': canvas, 'red_center': c[0].tolist(), 'red_size': int(s[0]),
            'blue_center': c[1].tolist(), 'blue_size': int(s[1])}


def layout_render(case: dict) -> Image.Image:
    n = case['canvas']
    img = Image.new('RGB', (n, n), (255, 255, 255))
    d = ImageDraw.Draw(img)
    (rx, ry), r = case['red_center'], case['red_size'] / 2
    d.ellipse((rx - r, ry - r, rx + r, ry + r), fill=RED)
    (bx, by), b = case['blue_center'], case['blue_size'] / 2
    d.rectangle((bx - b, by - b, bx + b, by + b), fill=BLUE)
    return img


def object_patch_masks(case: dict, hp: int, wp: int, margin: int = 1) -> dict:
    """Block-order boolean masks over patches: red object, blue object, background far from both.

    Background patches are at least `margin` patches away from either object's bounding box.
    """
    r, c = np.meshgrid(np.arange(hp), np.arange(wp), indexing='ij')
    def box_mask(center, size, pad):
        x0, x1 = center[0] - size / 2, center[0] + size / 2
        y0, y1 = center[1] - size / 2, center[1] + size / 2
        return ((c + 1) * PATCH > x0 - pad * PATCH) & (c * PATCH < x1 + pad * PATCH) & \
               ((r + 1) * PATCH > y0 - pad * PATCH) & (r * PATCH < y1 + pad * PATCH)
    red = box_mask(case['red_center'], case['red_size'], 0)
    blue = box_mask(case['blue_center'], case['blue_size'], 0)
    near = box_mask(case['red_center'], case['red_size'], margin) | box_mask(case['blue_center'], case['blue_size'], margin)
    blk = lambda m: to_block(torch.from_numpy(m.ravel()), hp, wp).numpy()
    return {'red': blk(red), 'blue': blk(blue), 'background': blk(~near)}


def layout_labels(case: dict) -> dict:
    (rx, ry), (bx, by) = case['red_center'], case['blue_center']
    n = case['canvas']
    return {'red_x': 1000 * rx / n, 'red_y': 1000 * ry / n, 'blue_x': 1000 * bx / n, 'blue_y': 1000 * by / n,
            'red_left': float(rx < bx), 'red_above': float(ry < by), 'red_bigger': float(case['red_size'] > case['blue_size'])}


# ----------------------------------------------------------------------------- linear erasure

class LeaceEraser:
    """LEACE eraser r(x) = x - P (x - mu) with P = W^+ U U^T W (Belrose et al., 2023).

    W = Sigma_xx^{-1/2} whitens x; U is an orthonormal basis of the column space of W Sigma_xz.
    After erasure no linear function of x has nonzero covariance with z, and among all such affine
    maps the edit has minimal expected squared change in the whitened norm. rank(P) <= dim(z).
    """

    def __init__(self, mean: torch.Tensor, proj: torch.Tensor, rank: int, singular: torch.Tensor):
        self.mean, self.proj, self.rank, self.singular = mean, proj, rank, singular

    @classmethod
    def fit(cls, n, sx, sxx, sz, sxz, eig_floor: float = 1e-5, rank_tol: float = 1e-6,
            random_rank: int | None = None, seed: int = 0) -> 'LeaceEraser':
        """Fit from raw float64 moments: n, sum x (D,), sum x x^T (D,D), sum z (k,), sum x z^T (D,k).

        random_rank: if given, U is a random orthonormal basis of that rank in the whitened space
        (a control eraser of equal rank that carries no concept information).
        """
        mu = sx / n
        cov = sxx / n - torch.outer(mu, mu)
        cxz = sxz / n - torch.outer(mu, sz / n)
        evals, q = torch.linalg.eigh((cov + cov.T) / 2)
        evals = evals.clamp_min(eig_floor * float(evals.max()))
        w = (q / evals.sqrt()) @ q.T
        w_inv = (q * evals.sqrt()) @ q.T
        m = w @ cxz
        u, s, _ = torch.linalg.svd(m, full_matrices=False)
        if random_rank is None:
            keep = s > rank_tol * float(s.max())
            u = u[:, keep]
        else:
            g = torch.Generator().manual_seed(seed)
            u, _ = torch.linalg.qr(torch.randn(m.shape[0], random_rank, generator=g, dtype=m.dtype).to(m.device))
        proj = w_inv @ u @ u.T @ w
        return cls(mu, proj, u.shape[1], s)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xd = x.double()
        return (xd - (xd - self.mean) @ self.proj.T).to(x.dtype)


def onehot(n: int, k: int) -> torch.Tensor:
    z = torch.zeros(n, k, dtype=torch.float64)
    z[torch.arange(n), torch.arange(n) % k] = 1
    return z


def position_features(h: int, w: int, kind: str) -> torch.Tensor:
    """Concept matrix z of the merged-token positions (raster order) for LEACE.

    'onehot': (h*w, h*w) indicator of the token slot; 'xy': (h*w, 2) normalized coordinates.
    """
    if kind == 'onehot':
        return torch.eye(h * w, dtype=torch.float64)
    if kind == 'xy':
        return torch.from_numpy(token_xy(h, w)).double()
    raise ValueError(kind)


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]
