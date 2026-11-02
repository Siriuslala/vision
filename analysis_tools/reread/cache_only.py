"""Experiment 5.3: cache-only re-read without re-encoding.

Design and results: docs/analysis/reread/results_2026-10-21_cache_only.md (builds on experiment 5.2,
docs/analysis/reread/design_cached_reread.md).

Every cache-only reader takes its features from the single ViT pass that encodes the input image for the
LLM; no reader runs the ViT again. What varies is how the window's cached ViT patch states (16x16-pixel
patches, 1152-d, before the 2x2 patch merger) are turned into LLM tokens through the frozen merger:

* 'grid'     the standard non-overlapping 2x2 merge (for block 27 these are exactly the original visual
             tokens of the window); h x w tokens for an h x w window of merged tokens;
* 'shift'    overlapping 2x2 merges at stride 1 on the patch grid: the standard groups plus the groups
             shifted by one patch horizontally, vertically and diagonally; (2h - 1) x (2w - 1) tokens;
* 'nearest'  each patch fills all four merger slots (equivalently: nearest-neighbour 2x upsampling of the
             patch grid followed by the standard merge); 2h x 2w tokens, one per 16-pixel patch;
* 'bilinear' bilinear 2x upsampling of the patch grid followed by the standard merge; 2h x 2w tokens.

The merger input can be the output of ViT block 27 (last), 26 or 25 (1-based; 0-based 26 / 25 / 24).
Conditions are duck-type compatible with experiment 5.2's Condition, so the native multi-round protocol of
cached_reread_run can execute them as well. Layer and block numbers are 1-based.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.reread import cached_reread as cr
from analysis_tools.reread import reread_pilot as rp

PATCH = rp.PATCH                       # pixels per ViT patch side (16)
MERGES = ('grid', 'shift', 'nearest', 'bilinear')
LAST_BLOCK = 27

# Lean protocol: the first assistant turn is prefilled up to the bbox value, and the answer turn is prefilled
# with the answer marker, so neither turn contains free-form reasoning. Applied to every condition.
CALL_PREFIX = '<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n['
ANSWER_PREFIX = 'Final answer:'


@dataclass(frozen=True)
class Condition:
    """One way of answering. kind: 'base' | 'full2x' | 'crop' | 'cache'.

    The fields name ... order mirror experiment 5.2's Condition (cached_reread.Condition); block and merge
    are new. A cache condition with scale 2 reads a cache built by an extra ViT pass on the 2x image (5.2's
    h27, kept only as a reference); every other cache condition uses no extra ViT computation.
    """
    name: str
    kind: str
    scale: int = 1
    site: str = 'vis'
    ids: str = 'orig'
    oracle: bool = False
    order: str = 'raster'
    block: int = LAST_BLOCK
    merge: str = 'grid'

    @property
    def extra_vit(self) -> bool:
        return self.kind == 'crop' or (self.kind == 'cache' and self.scale == 2) or self.kind == 'full2x'


def _c(name, block=LAST_BLOCK, merge='grid', ids='orig', oracle=False):
    return Condition(name, 'cache', 1, 'vis' if block == LAST_BLOCK else f'block_{block}', ids, oracle,
                     block=block, merge=merge)


CONDITIONS = {c.name: c for c in [
    Condition('base', 'base'),
    Condition('full2x', 'full2x', 2),
    Condition('crop1', 'crop', 1),
    Condition('crop2', 'crop', 2),
    _c('c27'), _c('c26', 26), _c('c25', 25),
    _c('s27', merge='shift'), _c('s26', 26, 'shift'), _c('s25', 25, 'shift'),
    _c('u27', merge='nearest'), _c('b27', merge='bilinear'),
    _c('c27n', ids='new'), _c('s27n', merge='shift', ids='new'),
    Condition('h27', 'cache', 2, 'vis'),
    Condition('o_crop2', 'crop', 2, oracle=True),
    _c('o_c27', oracle=True), _c('o_s27', merge='shift', oracle=True),
]}


# ----------------------------------------------------------------------------- patch grids and merges

def group_to_grid(x: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """ViT patch states (4HW, C) in the processor's merge-group order -> spatial patch grid (2H, 2W, C).

    The Qwen image processor orders patches so that the four patches of each 2x2 merge group are contiguous:
    index 4 (r W + c) + 2 a + b holds the patch at row 2r + a, column 2c + b.
    """
    H, W = grid
    if x.shape[0] != 4 * H * W:
        raise ValueError(f'{x.shape[0]} patch states for a {H}x{W} merged grid.')
    return x.view(H, W, 2, 2, -1).permute(0, 2, 1, 3, 4).reshape(2 * H, 2 * W, -1)


def groups(P: torch.Tensor, stride: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """2x2 groups of a patch grid P (h, w, C) at the given stride -> (n, 4, C) in merger slot order, grid."""
    a, b, c, d = P[:-1:stride, :-1:stride], P[:-1:stride, 1::stride], P[1::stride, :-1:stride], P[1::stride, 1::stride]
    n0, n1 = min(t.shape[0] for t in (a, b, c, d)), min(t.shape[1] for t in (a, b, c, d))
    g = torch.stack([t[:n0, :n1] for t in (a, b, c, d)], 2)          # (n0, n1, 4, C)
    return g.reshape(n0 * n1, 4, -1), (n0, n1)


def merge(merger: torch.nn.Module, g: torch.Tensor) -> torch.Tensor:
    """Apply the frozen patch merger (per-patch LayerNorm, concat 4, MLP) to groups (n, 4, C) -> (n, D)."""
    return merger(g.reshape(-1, g.shape[-1]))


def window_patches(P: torch.Tensor, win: rp.Window, margin: int = 0) -> tuple[torch.Tensor, tuple[int, int]]:
    """Patch rows / columns of a merged-token window, with up to `margin` extra patches on each side.

    Returns the slice and the offset (rows, columns) of the window's first patch inside the slice.
    """
    r0, c0 = 2 * win.r0, 2 * win.c0
    r1, c1 = r0 + 2 * win.h, c0 + 2 * win.w
    lo_r, lo_c = max(0, r0 - margin), max(0, c0 - margin)
    hi_r, hi_c = min(P.shape[0], r1 + margin), min(P.shape[1], c1 + margin)
    return P[lo_r:hi_r, lo_c:hi_c], (r0 - lo_r, c0 - lo_c)


def read_window(merger: torch.nn.Module, P: torch.Tensor, win: rp.Window, how: str
                ) -> tuple[torch.Tensor, tuple[int, int], float]:
    """Cache-only tokens of a window from a spatial patch grid P (2H, 2W, C).

    Returns (tokens (n, D) in raster order, token grid (rows, cols), step), where step is the row / column
    spacing of consecutive tokens in units of merged tokens (1 for 'grid', 1/2 otherwise). The first token
    sits at the window origin (r0, c0) in every mode, which defines the 'orig' MRoPE ids.
    """
    if how == 'grid':
        Pw, _ = window_patches(P, win)
        g, gr = groups(Pw, 2)
        return merge(merger, g), gr, 1.0
    if how == 'shift':
        Pw, _ = window_patches(P, win)
        g, gr = groups(Pw, 1)
        return merge(merger, g), gr, 0.5
    if how == 'nearest':
        Pw, _ = window_patches(P, win)
        h, w, C = Pw.shape
        return merge(merger, Pw.reshape(h * w, 1, C).expand(-1, 4, -1)), (h, w), 0.5
    if how == 'bilinear':
        # one patch of context on each side keeps the border tokens identical to a whole-image upsampling
        Pm, (orr, occ) = window_patches(P, win, margin=1)
        up = F.interpolate(Pm.permute(2, 0, 1)[None].float(), scale_factor=2, mode='bilinear',
                           align_corners=False)[0].permute(1, 2, 0).to(P.dtype)
        up = up[2 * orr:2 * orr + 4 * win.h, 2 * occ:2 * occ + 4 * win.w]
        g, gr = groups(up, 2)
        return merge(merger, g), gr, 0.5
    raise ValueError(f'Unknown merge {how!r}.')


# ----------------------------------------------------------------------------- encoding

@torch.inference_mode()
def encode(rt: qr.Runtime, image: Image.Image, blocks: tuple[int, ...] = ()) -> dict:
    """One ViT pass. Returns {'vis': (HW, D), 'grid': (H, W), 'patch_<k>': (2H, 2W, 1152)} for each k in blocks.

    The patch grids are the pre-merger outputs of ViT block k, captured by forward hooks during the same
    pass that produces the LLM's visual tokens; capturing them adds no computation.
    """
    if image.width % rp.TOKEN or image.height % rp.TOKEN:
        raise ValueError(f'Image {image.size} is not a multiple of {rp.TOKEN} pixels.')
    out = rt.processor.image_processor(images=[image], do_resize=False, return_tensors='pt')
    grid_thw = out['image_grid_thw']
    _, h, w = grid_thw[0].tolist()
    if (h * PATCH, w * PATCH) != (image.height, image.width):
        raise ValueError('Image processor changed the image size.')
    visual = rt.model.model.visual
    store, handles = {}, []
    for k in blocks:
        handles.append(visual.blocks[k - 1].register_forward_hook(
            lambda m, i, o, k=k: store.__setitem__(k, o[0] if isinstance(o, tuple) else o)))
    try:
        feats = rt.model.model.get_image_features(out['pixel_values'].to(rt.device), grid_thw.to(rt.device)).pooler_output
    finally:
        for hd in handles:
            hd.remove()
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(list(feats), 0)
    grid = (h // 2, w // 2)
    res = {'vis': feats, 'grid': grid}
    for k in blocks:
        res[f'patch_{k}'] = group_to_grid(store[k], grid)
    return res


def cache_blocks(conds: list[Condition]) -> tuple[int, ...]:
    """ViT blocks whose pre-merger states a set of conditions reads (block 27 'grid' reads the visual tokens)."""
    need = {c.block for c in conds if c.kind == 'cache' and c.scale == 1 and not (c.block == LAST_BLOCK and c.merge == 'grid')}
    return tuple(sorted(need))


# ----------------------------------------------------------------------------- tool output

def cache_segment(rt: qr.Runtime, cond: Condition, f1: dict, win: rp.Window) -> cr.VSeg:
    """The tool's visual output of a scale-1 cache condition for one window (no ViT computation)."""
    if cond.block == LAST_BLOCK and cond.merge == 'grid':
        idx = torch.tensor(win.indices(f1['grid'][1]), device=f1['vis'].device)
        emb, gr, step = f1['vis'][idx], (win.h, win.w), 1.0
    else:
        emb, gr, step = read_window(rt.model.model.visual.merger, f1[f'patch_{cond.block}'], win, cond.merge)
    return cr.VSeg(emb, gr, cond.ids, (float(win.r0), float(win.c0)), step)


# ----------------------------------------------------------------------------- cost accounting

def vit_flops(n_patches: int, vcfg) -> float:
    """Forward FLOPs (2 per multiply-add) of the Qwen3.5 ViT on one image of n_patches 16x16 patches.

    Counts the patch embedding (Conv3d over 2 x 16 x 16 x 3 inputs), per block the QKV and output projections,
    the full (non-windowed) attention scores and weighted sum, and the two-layer MLP, and the final merger.
    LayerNorms, RoPE and the position-embedding interpolation are ignored (< 1%).
    """
    d, dff, L = vcfg.hidden_size, vcfg.intermediate_size, vcfg.depth
    k_in = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2
    N = n_patches
    embed = 2 * N * k_in * d
    per_block = 2 * N * d * 3 * d + 2 * N * d * d + 2 * 2 * N * N * d + 2 * 2 * N * d * dff
    return float(embed + L * per_block + merger_flops(N // 4, vcfg))


def merger_flops(n_tokens: int, vcfg) -> float:
    """FLOPs of the patch merger producing n_tokens LLM tokens (fc1 4C -> 4C, fc2 4C -> D)."""
    c4 = 4 * vcfg.hidden_size
    return float(2 * n_tokens * (c4 * c4 + c4 * vcfg.out_hidden_size))


def lean_call_prefix_ids(rt: qr.Runtime) -> list[int]:
    return rt.tokenizer(CALL_PREFIX, add_special_tokens=False)['input_ids']


def answer_prefix_ids(rt: qr.Runtime) -> list[int]:
    return rt.tokenizer(ANSWER_PREFIX, add_special_tokens=False)['input_ids']


# ----------------------------------------------------------------------------- closed-call variant (lean2)

CALL_CLOSE = '\n</parameter>\n</function>\n</tool_call>'
_BRACKET_STOPS: dict[int, set[int]] = {}


def bracket_stop_ids(rt: qr.Runtime) -> set[int]:
    """Vocabulary ids whose text contains ']' : decoding the bbox value stops at its closing bracket."""
    key = id(rt.tokenizer)
    if key not in _BRACKET_STOPS:
        tok = rt.tokenizer
        _BRACKET_STOPS[key] = {i for i in range(len(tok)) if ']' in tok.decode([i])}
    return _BRACKET_STOPS[key]


def close_call(rt: qr.Runtime, gen: list[int]) -> tuple[list[int], str]:
    """Close a bbox call decoded up to its ']' token with the fixed tail of the Qwen3.5 XML call format.

    Returns (generated ids followed by the tail ids, call text after CALL_PREFIX). The tail omits whatever
    part of it the stop token already contains (e.g. a ']\n' token).
    """
    text = rt.tokenizer.decode(gen)
    after = text[text.index(']') + 1:] if ']' in text else ''
    tail = CALL_CLOSE
    k = 0
    while k < len(after) and k < len(tail) and after[k] == tail[k]:
        k += 1
    tail_ids = rt.tokenizer(tail[k:], add_special_tokens=False)['input_ids']
    return gen + tail_ids, text + tail[k:]
