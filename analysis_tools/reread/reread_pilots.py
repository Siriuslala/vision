"""Experiment 5.4: two agile tests of cached re-reading (Qwen3.5-9B).

Design: docs/analysis/reread/design_cached_reread.md (section 0, the full-pipeline specification);
results: docs/analysis/reread/results_2026-11-03_reread_pilots.md.

Option 1 (learned 2x cache). A lightweight convolutional module maps the cached pre-merger ViT patch states of
the single 1x pass (block 27, optionally plus block 0 = patch embedding + absolute position embedding, or block 9)
to a 2x-denser grid of pre-merger states. The frozen patch merger turns that grid into 2H x 2W LLM tokens. The
module is trained to reproduce the visual tokens of a ViT pass on the bicubic 2x-upscaled image (the "2x cache"
h27 of experiment 5.2, whose accuracy matched the 2x crop re-encode). The output is initialised to the nearest
upsampling of block 27 (reader u27 of experiment 5.3), so the module only has to learn a residual.

Option 2 (high-resolution documents). Two resolutions of one document: the "lo" image under a pixel budget and
the "hi" image with exactly twice the lo side lengths, resized from the original pixels. Conditions differ in
which image the LLM sees globally and where the tool's window comes from (crop re-encode or cache slice).

Block numbers are 1-based; block 0 denotes the input of block 1 (patch embedding plus position embedding).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from analysis_tools.reread import cache_only as co
from analysis_tools.reread import cached_reread as cr
from analysis_tools.reread import reread_pilot as rp

VARIANTS = {'l27': (27,), 'l27_0': (27, 0), 'l27_9': (27, 9)}


# ----------------------------------------------------------------------------- vision tower only

def load_vision(checkpoint: Path, device: str, dtype=torch.bfloat16):
    """Qwen3.5 ViT + merger without the 9B decoder (weights read from the checkpoint shards that hold them)."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    cfg = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    vcfg = cfg.vision_config
    vcfg._attn_implementation = 'sdpa'
    visual = Qwen3_5VisionModel(vcfg)
    index = json.loads((Path(checkpoint) / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.visual.'
    state = {}
    for shard in sorted({f for k, f in index.items() if k.startswith(prefix)}):
        with safe_open(str(Path(checkpoint) / shard), framework='pt') as f:
            for k in f.keys():
                if k.startswith(prefix):
                    state[k[len(prefix):]] = f.get_tensor(k)
    missing, unexpected = visual.load_state_dict(state, strict=False)
    missing = [m for m in missing if 'inv_freq' not in m]
    if missing or unexpected:
        raise RuntimeError(f'Vision weights mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}')
    visual = visual.to(device, dtype).eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True).image_processor
    return visual, processor


def pixel_inputs(image_processor, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
    for im in images:
        if im.width % rp.TOKEN or im.height % rp.TOKEN:
            raise ValueError(f'Image {im.size} is not a multiple of {rp.TOKEN} pixels.')
    out = image_processor(images=images, do_resize=False, return_tensors='pt')
    for im, (_, h, w) in zip(images, out['image_grid_thw'].tolist()):
        if (h * rp.PATCH, w * rp.PATCH) != (im.height, im.width):
            raise ValueError('Image processor changed the image size.')
    return out['pixel_values'], out['image_grid_thw']


@torch.no_grad()
def vit_states(visual, pixel_values: torch.Tensor, grid_thw: torch.Tensor, blocks: tuple[int, ...] = ()
               ) -> list[dict]:
    """One ViT call on several images. Per image: 'vis' (HW, D) visual tokens in raster order, 'grid' (H, W)
    merged grid, and 'patch_k' (2H, 2W, 1152) pre-merger states of block k (k = 0: input of block 1)."""
    dev = next(visual.parameters()).device
    store, handles = {}, []
    for k in blocks:
        if k == 0:
            handles.append(visual.blocks[0].register_forward_pre_hook(
                lambda m, a, kw: store.__setitem__(0, (a[0] if a else kw['hidden_states']).clone()),
                with_kwargs=True))
        elif k != co.LAST_BLOCK:
            handles.append(visual.blocks[k - 1].register_forward_hook(
                lambda m, i, o, k=k: store.__setitem__(k, o[0] if isinstance(o, tuple) else o)))
    try:
        out = visual(pixel_values.to(dev, next(visual.parameters()).dtype), grid_thw=grid_thw.to(dev))
    finally:
        for h in handles:
            h.remove()
    if co.LAST_BLOCK in blocks:
        store[co.LAST_BLOCK] = out.last_hidden_state
    grids = [(h // 2, w // 2) for _, h, w in grid_thw.tolist()]
    vis = torch.split(out.pooler_output, [H * W for H, W in grids])
    res = [{'vis': v, 'grid': g} for v, g in zip(vis, grids)]
    for k in blocks:
        for r, part in zip(res, torch.split(store[k], [4 * H * W for H, W in grids])):
            r[f'patch_{k}'] = co.group_to_grid(part, r['grid'])
    return res


def merge_grid(merger: nn.Module, P: torch.Tensor) -> torch.Tensor:
    """Standard non-overlapping 2x2 merge of a patch grid P (2a, 2b, C) -> tokens (a*b, D) in raster order."""
    g, _ = co.groups(P, 2)
    return co.merge(merger, g)


# ----------------------------------------------------------------------------- learned 2x module

class ConvNeXtBlock(nn.Module):
    """Depthwise 7x7 convolution, LayerNorm, pointwise MLP (4x), residual; operates on (B, C, h, w)."""

    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.norm = nn.LayerNorm(c)
        self.fc1, self.fc2 = nn.Linear(c, 4 * c), nn.Linear(4 * c, c)

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.fc2(F.gelu(self.fc1(self.norm(y))))
        return x + y.permute(0, 3, 1, 2)


class Upsampler2x(nn.Module):
    """Cached 1x pre-merger states (2H, 2W, d) of one or more ViT blocks -> 2x pre-merger grid (4H, 4W, d).

    out = nearest_up(X27) + Head(Conv_hi(PixelShuffle(Conv_lo(Proj([LN(X_k)]_k)))))  with Head zero-initialised,
    so the untrained module reproduces reader u27 exactly. All operations are local (receptive field about
    13 patches), so the module applies to any image size and any window.
    """

    def __init__(self, n_in: int, d: int = 1152, c: int = 384, n_lo: int = 4, n_hi: int = 2):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_in))
        self.proj = nn.Linear(n_in * d, c)
        self.lo = nn.Sequential(*[ConvNeXtBlock(c) for _ in range(n_lo)])
        self.up = nn.Conv2d(c, 4 * c, 1)
        self.hi = nn.Sequential(*[ConvNeXtBlock(c) for _ in range(n_hi)])
        self.head = nn.Linear(c, d)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """xs[0] must be block 27. Each (2H, 2W, d). Returns (4H, 4W, d) in the dtype of xs[0]."""
        base = xs[0].repeat_interleave(2, 0).repeat_interleave(2, 1)
        z = self.proj(torch.cat([n(x.float()) for n, x in zip(self.norms, xs)], -1))
        z = self.lo(z.permute(2, 0, 1)[None])
        z = F.pixel_shuffle(self.up(z), 2)
        z = self.hi(z)[0].permute(1, 2, 0)
        return base + self.head(z).to(base.dtype)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def learned_tokens(module: Upsampler2x, merger: nn.Module, f: dict, blocks: tuple[int, ...]) -> torch.Tensor:
    """Full-image learned 2x tokens (2H*2W, D) in raster order from one image's cached states."""
    P2 = module([f[f'patch_{k}'] for k in blocks])
    return merge_grid(merger, P2.to(f['patch_27'].dtype))


def load_modules(ckpt_dir: Path, names: list[str], device) -> dict:
    out = {}
    for n in names:
        path = ckpt_dir / f'{n}.pt'
        if not path.exists():
            raise FileNotFoundError(f'Missing learned module checkpoint {path}.')
        sd = torch.load(path, map_location='cpu')
        m = Upsampler2x(len(VARIANTS[n]), **sd['hparams'])
        m.load_state_dict(sd['state'])
        out[n] = m.to(device).eval().requires_grad_(False)
    return out


# ----------------------------------------------------------------------------- training data

def training_files(data_dir: Path, n_textvqa: int, n_coco: int, seed: int) -> list[str]:
    """TextVQA train images (OpenImages) and COCO val2014 images; disjoint from the TextVQA val evaluation."""
    tv = sorted((data_dir / 'TextVQA/data/images').glob('*.jpg'))
    coco = sorted((data_dir / 'VQA_v2/images/val2014').glob('*.jpg'))
    if len(tv) < n_textvqa or len(coco) < n_coco:
        raise ValueError(f'Not enough training images: {len(tv)} TextVQA, {len(coco)} COCO.')
    rng = random.Random(seed)
    files = rng.sample(tv, n_textvqa) + rng.sample(coco, n_coco)
    rng.shuffle(files)
    return [str(f) for f in files]


class PairDataset(torch.utils.data.Dataset):
    """Image under the pixel budget and its bicubic 2x upscale, as processor pixel values (CPU workers)."""

    def __init__(self, files: list[str], image_processor, max_pixels: int):
        self.files, self.ip, self.max_pixels = files, image_processor, max_pixels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = cr.resize_budget(Image.open(self.files[i]).convert('RGB'), self.max_pixels)
        p1, g1 = pixel_inputs(self.ip, [im])
        p2, g2 = pixel_inputs(self.ip, [cr.upscale(im, 2)])
        return p1, g1, p2, g2


def collate_pairs(batch):
    return (torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch]),
            torch.cat([b[2] for b in batch]), torch.cat([b[3] for b in batch]))


def token_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-token squared error summed over the D dimensions, (n,)."""
    return (pred.float() - target.float()).pow(2).sum(-1)


# ----------------------------------------------------------------------------- option 2: documents

DOC_FMT = 'Answer the question using a single word or phrase. ' + 'Write the final answer on its own last line as "Final answer: <answer>".'


def doc_cases(meta_path: Path, image_dir: Path, n: int, seed: int, max_words: int = 6) -> list[cr.Case]:
    """DocVQA questions (Visual-CoT DocVQA train metadata with evidence boxes), one per available image."""
    rows = [json.loads(l) for l in meta_path.open()]
    avail = {p.name for p in image_dir.glob('*.png')}
    rows = [r for r in rows if r['image'] in avail and len(str(r['answer']).split()) <= max_words]
    rng = random.Random(seed)
    rng.shuffle(rows)
    seen, out = set(), []
    for r in rows:
        if r['image'] in seen:
            continue
        seen.add(r['image'])
        b = r['bboxs']
        ub = [min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)]
        box = [1000 * ub[0] / r['width'], 1000 * ub[1] / r['height'], 1000 * ub[2] / r['width'],
               1000 * ub[3] / r['height']]
        area = (ub[2] - ub[0]) * (ub[3] - ub[1]) / (r['width'] * r['height'])
        out.append(cr.Case(f"doc_{len(out):04d}_{r['image'][:-4]}", 'docvqa', r['question'] + '\n' + DOC_FMT,
                           r.get('possible_answers') or [r['answer']], DOC_FMT, str(image_dir / r['image']), box,
                           {'area_ratio': area, 'evidence_box': box}))
        if len(out) == n:
            break
    if len(out) < n:
        raise ValueError(f'Only {len(out)} DocVQA cases available, {n} requested.')
    return out


def doc_images(path: str, max_pixels: int) -> tuple[Image.Image, Image.Image]:
    """(lo, hi): lo under the pixel budget (sides multiples of 32); hi has exactly twice lo's sides and is resized
    from the original pixels (bicubic), so hi carries the document's own detail rather than an upscale of lo."""
    raw = Image.open(path).convert('RGB')
    lo = cr.resize_budget(raw, max_pixels)
    hi = raw.resize((2 * lo.width, 2 * lo.height), Image.Resampling.BICUBIC)
    return lo, hi


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, answers: list[str], tau: float = 0.5) -> float:
    """Average normalized Levenshtein similarity (DocVQA metric) for one question: max over answers of
    1 - NL if NL < tau else 0, with NL = edit distance / max length on lower-cased, whitespace-stripped strings."""
    p = ' '.join(cr.short_answer(pred).lower().split())
    best = 0.0
    for a in answers:
        a = ' '.join(str(a).lower().split())
        if not a and not p:
            return 1.0
        nl = levenshtein(p, a) / max(len(p), len(a), 1)
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best


def score_doc(case: cr.Case, text: str) -> dict:
    ans, parsed = cr.final_answer(text)
    em = float(any(cr.vqa_normalize(cr.short_answer(ans)) == cr.vqa_normalize(a) for a in case.answer))
    return {'correct': anls(ans, case.answer), 'em': em, 'pred': ans, 'parsed': parsed}


# ----------------------------------------------------------------------------- generic conditions

@dataclass(frozen=True)
class PCond:
    """A condition of the pilots.

    view: image the LLM sees in the first turn ('lo' | 'hi').
    tool: None (no tool), 'crop' (re-encode the window's pixels of `src`, upscaled by k), 'cache' (slice of the
          visual tokens of the ViT pass on `src`, or a merge variant of its block-27 patch states), 'learned'
          (slice of the learned 2x tokens computed from the lo cache, module `variant`).
    src: 'lo' | 'hi' | 'up' (bicubic 2x upscale of lo) for crop / cache.
    """
    name: str
    view: str = 'lo'
    tool: str | None = None
    src: str = 'lo'
    k: int = 1
    merge: str = 'grid'
    variant: str = ''

    def passes(self) -> set[str]:
        """Window-independent ViT passes this condition needs ('lo' carries the learned-module hooks)."""
        need = {self.view}
        if self.tool == 'cache':
            need.add(self.src)
        if self.tool == 'learned':
            need.add('lo')
        return need


OPT1 = {c.name: c for c in [
    PCond('base'), PCond('crop1', tool='crop'), PCond('crop2', tool='crop', k=2), PCond('c27', tool='cache'),
    PCond('u27', tool='cache', merge='nearest'), PCond('h27', tool='cache', src='up'),
    *[PCond(v, tool='learned', variant=v) for v in VARIANTS]]}

OPT2 = {c.name: c for c in [
    PCond('base_lo'), PCond('base_hi', view='hi'),
    PCond('lo_c27', tool='cache'), PCond('lo_crop2', tool='crop', k=2),
    PCond('lo_crop_hi', tool='crop', src='hi'), PCond('lo_cache_hi', tool='cache', src='hi'),
    PCond('hi_crop1', view='hi', tool='crop', src='hi'), PCond('hi_c27', view='hi', tool='cache', src='hi'),
    *[PCond('lo_' + v, tool='learned', variant=v) for v in VARIANTS]]}


def ratio(view: str, src: str) -> int:
    """Linear size of src relative to the view image (1 or 2)."""
    r = {'lo': 1, 'hi': 2, 'up': 2}
    if r[src] < r[view]:
        raise ValueError(f'Source {src!r} is coarser than view {view!r}.')
    return r[src] // r[view]


def tool_tokens(cond: PCond, win: rp.Window) -> int:
    """Number of visual tokens the tool would return for a window (before the crop cap)."""
    s = ratio(cond.view, cond.src) if cond.tool in ('crop', 'cache') else 2
    if cond.tool == 'crop':
        s *= cond.k
    if cond.tool == 'cache' and cond.merge != 'grid':
        s *= 2
    return s * s * win.h * win.w


def capped(cond: PCond, win: rp.Window) -> bool:
    """Extra-token readers obey the native tool's area cap (CROP_MAX_PIXELS of re-encoded pixels, i.e. 1600
    tokens); a capped condition returns the view's own 1x tokens for that window."""
    return tool_tokens(cond, win) > win.h * win.w and tool_tokens(cond, win) * rp.TOKEN ** 2 > cr.CROP_MAX_PIXELS


def window_tokens(cond: PCond, win: rp.Window, images: dict, feats: dict, learned: dict, rt_visual, merger,
                  image_processor) -> tuple[cr.VSeg, dict]:
    """The tool's visual segment for window `win` (merged-token units of the view grid) and its cost."""
    view = cond.view
    if capped(cond, win):
        f = feats[view]
        idx = torch.tensor(win.indices(f['grid'][1]), device=f['vis'].device)
        return cr.VSeg(f['vis'][idx], (win.h, win.w), 'orig', (float(win.r0), float(win.c0)), 1.0), \
            {'vit_patches': 0, 'capped': True}
    if cond.tool == 'crop':
        s = ratio(view, cond.src)
        sw = rp.Window(win.r0 * s, win.c0 * s, win.h * s, win.w * s)
        crop = cr.upscale(rp.crop(images[cond.src], sw), cond.k)
        pv, g = pixel_inputs(image_processor, [crop])
        e = vit_states(rt_visual, pv, g)[0]
        return cr.VSeg(e['vis'], e['grid'], 'new'), {'vit_patches': 4 * e['grid'][0] * e['grid'][1], 'capped': False}
    if cond.tool == 'learned':
        T, (H2, W2) = learned[cond.variant], (2 * feats['lo']['grid'][0], 2 * feats['lo']['grid'][1])
        sw = rp.Window(2 * win.r0, 2 * win.c0, 2 * win.h, 2 * win.w)
        idx = torch.tensor(sw.indices(W2), device=T.device)
        return cr.VSeg(T[idx], (sw.h, sw.w), 'orig', (float(win.r0), float(win.c0)), 0.5), \
            {'vit_patches': 0, 'capped': False}
    # cache
    s = ratio(view, cond.src)
    f = feats[cond.src]
    sw = rp.Window(win.r0 * s, win.c0 * s, win.h * s, win.w * s)
    if cond.merge == 'grid':
        idx = torch.tensor(sw.indices(f['grid'][1]), device=f['vis'].device)
        emb, gr, step = f['vis'][idx], (sw.h, sw.w), 1.0 / s
    else:
        emb, gr, st = co.read_window(merger, f['patch_27'], sw, cond.merge)
        step = st / s
    return cr.VSeg(emb, gr, 'orig', (float(win.r0), float(win.c0)), step), {'vit_patches': 0, 'capped': False}


def cache_bytes(grid: tuple[int, int], blocks: tuple[int, ...], d: int = 1152) -> int:
    """bf16 bytes of the cached pre-merger patch states of the given blocks for an H x W merged grid."""
    return 4 * grid[0] * grid[1] * d * 2 * len(blocks)


def window_area(win: rp.Window, grid: tuple[int, int]) -> float:
    return win.h * win.w / (grid[0] * grid[1])


def check_finite(t: torch.Tensor, what: str) -> None:
    if not torch.isfinite(t).all():
        raise FloatingPointError(f'Non-finite values in {what}.')


def flops_conv_module(m: Upsampler2x, grid: tuple[int, int]) -> float:
    """Forward FLOPs (2 per multiply-add) of the learned module on an H x W merged grid (linear and conv layers)."""
    H, W = grid
    n_lo, n_hi = 4 * H * W, 16 * H * W
    c = m.proj.out_features
    d = m.head.out_features
    fl = 2 * n_lo * m.proj.in_features * c
    blk = lambda n: 2 * n * (49 * c + 8 * c * c)
    fl += len(m.lo) * blk(n_lo) + 2 * n_lo * c * 4 * c + len(m.hi) * blk(n_hi) + 2 * n_hi * c * d
    return float(fl)

