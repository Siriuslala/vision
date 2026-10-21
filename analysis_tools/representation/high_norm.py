"""Experiment 2.2: source, content and role of high-norm tokens in the ViT of Qwen3.5 / Qwen3-VL.

Reusable engine (stage drivers live in `high_norm_run.py`, CPU analysis in `high_norm_analysis.py`):

* `GridTracker` / `segments`: per-image token ranges inside the packed vision sequence, so that every
  hook works for batched vision forwards (including the single prefill vision call of `generate`).
* `hn_mask`: high-norm detection, ||x|| > tau * median_q ||x_q|| within each image.
* `vit_ablation(...)`: in-network ablations of high-norm tokens at a ViT block output (mean / zero /
  random-token control) and the test-time "register neuron" intervention on the block MLP.
* `pe_override(...)`: replaces the learned absolute position embedding (PE) by an arbitrary function of
  the 48 x 48 table or of the interpolated per-image PE (flip, outlier removal, ...).
* `vit_attention_mass(...)`: softmax attention of every ViT block recomputed from pre-hooks, reduced to
  the mass each key receives.
* `yes_no_batch(...)`: first-token margins z_Yes - z_No for a left-padded batch (prefill only).
* `PosVar`: streaming split-half estimate of the position-explained variance R^2_pos.

Patch-level tensors inside the ViT are in spatial-merge-block order (2 x 2 blocks, raster over blocks).
Block numbers are 1-based, as in the analysis documents.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, asdict
import math

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import vit_layout as vl

REGISTER_BLOCK = 10          # first ViT block whose output contains high-norm tokens (both models)
TAU = 3.0                    # detection threshold relative to the per-image median norm


# ----------------------------------------------------------------------------- packed-sequence helpers

def segments(sizes: list[int]) -> list[slice]:
    out, start = [], 0
    for s in sizes:
        out.append(slice(start, start + s))
        start += s
    return out


@contextmanager
def grid_tracker(rt: qr.Runtime):
    """Record the per-image patch counts of every vision forward (state['sizes'], state['grids'])."""
    visual = rt.model.model.visual
    state = {}

    def pre(module, args, kwargs):
        grid = kwargs['grid_thw'] if 'grid_thw' in kwargs else args[1]
        state['grids'] = [(int(t), int(h), int(w)) for t, h, w in grid.tolist()]
        state['sizes'] = [t * h * w for t, h, w in state['grids']]

    handle = visual.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield state
    finally:
        handle.remove()


def hn_mask(x: torch.Tensor, sizes: list[int], tau: float = TAU) -> torch.Tensor:
    """(N,) bool: token norm above tau times the median norm of its own image."""
    n = x.float().norm(dim=-1)
    mask = torch.zeros_like(n, dtype=torch.bool)
    for seg in segments(sizes):
        v = n[seg]
        mask[seg] = v > tau * v.median()
    return mask


def ratio_map(x: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
    """(hp, wp) norm / median norm in raster order from a block-order (hp*wp, D) tensor."""
    n = x.float().norm(dim=-1)
    return vl.to_raster((n / n.median())[:, None], hp, wp)[:, 0].view(hp, wp)


# ----------------------------------------------------------------------------- ablations

@dataclass(frozen=True)
class Ablation:
    """In-network intervention on the ViT.

    mode:     'none' | 'hn_mean' (replace detected high-norm tokens by the mean of the image's other tokens)
              | 'hn_zero' (set them to zero) | 'rand_mean' (replace the same number of randomly chosen
              non-high-norm tokens by the same mean; control) | 'neuron_zero' (zero the activations of the
              register neurons in the MLP of `block`, for all tokens).
    block:    1-based ViT block whose output is edited (or whose MLP neurons are zeroed).
    persist:  for hn_* / rand_mean, repeat detection-and-replacement at every later block except the last.
    tau:      detection threshold.
    neurons:  register-neuron indices of the block MLP (fc2 input channels) for 'neuron_zero'.
    seed:     seed of the random-token control.
    """
    mode: str = 'none'
    block: int = REGISTER_BLOCK
    persist: bool = False
    tau: float = TAU
    neurons: tuple[int, ...] = ()
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _replace(h: torch.Tensor, sizes: list[int], ab: Ablation, log: list, call: int, block: int) -> torch.Tensor:
    mask = hn_mask(h, sizes, ab.tau)
    h = h.clone()
    for i, seg in enumerate(segments(sizes)):
        m = mask[seg]
        k = int(m.sum())
        log.append({'call': call, 'block': block, 'image': i, 'n_hn': k})
        if k == 0:
            continue
        x = h[seg]
        normal = x[~m]
        fill = normal.float().mean(0).to(x.dtype)
        if ab.mode == 'hn_mean':
            x[m] = fill
        elif ab.mode == 'hn_zero':
            x[m] = 0
        elif ab.mode == 'rand_mean':
            g = torch.Generator().manual_seed(ab.seed * 1000003 + call * 1009 + i * 31 + block)
            cand = (~m).nonzero().flatten().cpu()
            pick = cand[torch.randperm(len(cand), generator=g)[:k]].to(x.device)
            x[pick] = fill
        else:
            raise ValueError(f'Unknown ablation mode {ab.mode!r}')
        h[seg] = x
    return h


@contextmanager
def vit_ablation(rt: qr.Runtime, ab: Ablation):
    """Apply `ab` to every vision forward inside the context; yields a log of detected counts."""
    visual = rt.model.model.visual
    log: list[dict] = []
    if ab.mode == 'none':
        yield log
        return
    nb = len(visual.blocks)
    with ExitStack() as stack:
        state = stack.enter_context(grid_tracker(rt))
        calls = {'n': 0}
        handles = []

        def count_call(module, args, kwargs):
            calls['n'] += 1
        handles.append(visual.register_forward_pre_hook(count_call, with_kwargs=True))
        if ab.mode == 'neuron_zero':
            if not ab.neurons:
                raise ValueError('neuron_zero needs register-neuron indices.')
            idx = torch.tensor(ab.neurons)

            def pre_fc2(module, args):
                a = args[0].clone()
                a[..., idx.to(a.device)] = 0
                return (a,)
            handles.append(visual.blocks[ab.block - 1].mlp.linear_fc2.register_forward_pre_hook(pre_fc2))
        elif ab.mode in ('hn_mean', 'hn_zero', 'rand_mean'):
            blocks = range(ab.block, nb) if ab.persist else [ab.block]
            for b in blocks:
                def post(module, args, output, b=b):
                    h = output[0] if isinstance(output, tuple) else output
                    new = _replace(h, state['sizes'], ab, log, calls['n'], b)
                    return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
                handles.append(visual.blocks[b - 1].register_forward_hook(post))
        else:
            raise ValueError(f'Unknown ablation mode {ab.mode!r}')
        try:
            yield log
        finally:
            for h in handles:
                h.remove()


# ----------------------------------------------------------------------------- PE override

@contextmanager
def pe_override(rt: qr.Runtime, table_fn=None, pe_fn=None):
    """Replace the PE of every image in the next vision forwards.

    table_fn(table (48*48, D) float32) -> edited table, resampled with the model's default bilinear taps;
    pe_fn(pe (hp*wp, D) float32 block order, hp, wp) -> edited interpolated PE.
    The 2D RoPE ids are left unchanged. Implementation mirrors `vit_layout.vit_edit`: identity taps plus a
    hook on the pos_embed lookup.
    """
    visual = rt.model.model.visual
    table = visual.pos_embed.weight.float()
    edited_table = table_fn(table.clone()) if table_fn is not None else table
    state = {}

    def pre(module, args, kwargs):
        grid = kwargs['grid_thw']
        pes = []
        for t, hp, wp in grid.tolist():
            if t != 1:
                raise ValueError('Only single images are supported.')
            idx, w = vl.interp_taps(hp, wp, grid.device)
            pe = (edited_table[idx] * w[..., None].float()).sum(1)
            if pe_fn is not None:
                pe = pe_fn(pe, hp, wp)
            pes.append(pe)
        pe = torch.cat(pes)
        state['pe'] = pe
        kwargs['interp_indices'] = torch.zeros(len(pe), 1, device=grid.device, dtype=torch.long)
        kwargs['interp_weights'] = torch.ones(len(pe), 1, device=grid.device, dtype=torch.float32)
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


def pe_flip(axis: str):
    """pe_fn mirroring the interpolated PE grid horizontally ('h') or vertically ('v')."""
    def fn(pe, hp, wp):
        r = vl.to_raster(pe, hp, wp).view(hp, wp, -1)
        r = r.flip(1) if axis == 'h' else r.flip(0)
        return vl.to_block(r.reshape(hp * wp, -1), hp, wp)
    return fn


def table_outlier_mask(table: torch.Tensor, ratio: float = 3.0) -> torch.Tensor:
    """(48*48,) bool: table entries whose norm exceeds `ratio` times the median entry norm."""
    n = table.float().norm(dim=1)
    return n > ratio * n.median()


def table_outlier_fn(ratio: float = 3.0, mode: str = 'shrink'):
    """table_fn removing the PE outlier entries: rescale them to the median norm ('shrink') or zero them."""
    def fn(table):
        m = table_outlier_mask(table, ratio)
        n = table.norm(dim=1, keepdim=True)
        if mode == 'shrink':
            table[m] = table[m] / n[m] * n.median()
        elif mode == 'zero':
            table[m] = 0
        else:
            raise ValueError(mode)
        return table
    return fn


# ----------------------------------------------------------------------------- capture helpers

@torch.inference_mode()
def vit_forward_record(rt: qr.Runtime, images: list[Image.Image], blocks: list[int], keep: list[int] = (),
                       mlp_blocks: list[int] = ()) -> list[dict]:
    """One vision forward under whatever hooks are active.

    Returns per image: 'grid' (hp, wp); 'ratio' (len(blocks), hp, wp) float16 norm/median maps (raster);
    'median' (len(blocks),) median norms; 'feat_k' (hp*wp, D) block-order float32 GPU features for k in keep;
    'merger' (Nv, Dm) float32 merger output (raster); 'mlp_k'/'attn_k' (hp*wp, D) sublayer outputs for
    k in mlp_blocks, and 'act_k' (hp*wp, I) MLP activations (fc2 inputs).
    """
    visual = rt.model.model.visual
    pixel, grid = vl.pixel_inputs(rt, images)
    store, handles = {}, []
    for k in sorted(set(blocks) | set(keep)):
        handles.append(visual.blocks[k - 1].register_forward_hook(
            lambda m, i, o, k=k: store.__setitem__(f'block_{k}', o[0] if isinstance(o, tuple) else o)))
    for k in mlp_blocks:
        blk = visual.blocks[k - 1]
        handles.append(blk.mlp.register_forward_hook(lambda m, i, o, k=k: store.__setitem__(f'mlp_{k}', o)))
        handles.append(blk.attn.register_forward_hook(lambda m, i, o, k=k: store.__setitem__(
            f'attn_{k}', o[0] if isinstance(o, tuple) else o)))
        handles.append(blk.mlp.linear_fc2.register_forward_pre_hook(
            lambda m, a, k=k: store.__setitem__(f'act_{k}', a[0])))
    handles.append(visual.merger.register_forward_hook(lambda m, i, o: store.__setitem__('merger', o)))
    try:
        visual(pixel, grid_thw=grid)
    finally:
        for h in handles:
            h.remove()
    grids = [(int(h), int(w)) for _, h, w in grid.tolist()]
    sizes = [h * w for h, w in grids]
    out = []
    for i, seg in enumerate(segments(sizes)):
        hp, wp = grids[i]
        rec = {'grid': (hp, wp)}
        ratios, meds = [], []
        for k in blocks:
            x = store[f'block_{k}'][seg]
            n = x.float().norm(dim=-1)
            meds.append(float(n.median()))
            ratios.append(vl.to_raster((n / n.median())[:, None], hp, wp)[:, 0].view(hp, wp))
        rec['ratio'] = torch.stack(ratios).half().cpu().numpy() if ratios else None
        rec['median'] = np.array(meds, np.float32)
        for k in keep:
            rec[f'feat_{k}'] = store[f'block_{k}'][seg].float()
        for k in mlp_blocks:
            for key in ('mlp', 'attn', 'act'):
                rec[f'{key}_{k}'] = store[f'{key}_{k}'][seg].float()
        out.append(rec)
    mseg = segments([s // 4 for s in sizes])
    for rec, seg in zip(out, mseg):
        rec['merger'] = store['merger'][seg].float()
    return out


@torch.inference_mode()
def decoder_norms(rt: qr.Runtime, image: Image.Image, text: str = 'Describe the image.') -> dict:
    """Prefill norms of every decoder layer: 'prefix' (layers, n_prefix), 'visual' (layers, h, w) raster,
    'suffix' (layers, n_suffix), and the input embedding norms at layer 0."""
    batch = qr.build(rt, [image], text)
    sp = qr.image_spans(rt, batch)[0]
    nl = len(rt.layers)
    norms = {}
    handles = []
    for l in range(1, nl + 1):
        def hook(module, inputs, output, l=l):
            h = output[0] if isinstance(output, tuple) else output
            if l not in norms:
                norms[l] = h[0].float().norm(dim=-1).cpu()
        handles.append(rt.layers[l - 1].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            emb = rt.model.model.get_input_embeddings()(batch['input_ids'])[0].float().norm(dim=-1).cpu()
            rt.model(**batch, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    s, e = sp['start'], sp['end']
    gh, gw = sp['grid']
    mat = torch.stack([norms[l] for l in range(1, nl + 1)])
    return {'prefix': mat[:, :s].numpy(), 'visual': mat[:, s:e].reshape(nl, gh, gw).numpy(),
            'suffix': mat[:, e:].numpy(), 'emb_prefix': emb[:s].numpy(), 'grid': (gh, gw),
            'prefix_tokens': [rt.tokenizer.decode([t]) for t in batch['input_ids'][0, :s].tolist()]}


# ----------------------------------------------------------------------------- ViT attention

@torch.inference_mode()
def vit_attention_mass(rt: qr.Runtime, image: Image.Image, blocks: list[int], sets: dict[str, torch.Tensor]) -> dict:
    """Recompute ViT softmax attention for one image at the given blocks.

    sets: name -> (P,) bool masks in block order (e.g. high-norm tokens, random control tokens).
    Returns per block: 'received' (heads, P) mean attention each key receives over all queries (block
    order), 'mass_<name>' (heads,) mean attention mass of all queries on the set, and 'self_<name>'
    (heads,) mass of the set's own queries on the set.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision
    visual = rt.model.model.visual
    pixel, grid = vl.pixel_inputs(rt, [image])
    store, handles = {}, []
    for k in blocks:
        def pre(module, args, kwargs, k=k):
            store[k] = (args[0] if args else kwargs['hidden_states'], kwargs['position_embeddings'])
        handles.append(visual.blocks[k - 1].attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        visual(pixel, grid_thw=grid)
    finally:
        for h in handles:
            h.remove()
    out = {}
    for k in blocks:
        mod = visual.blocks[k - 1].attn
        h, (cos, sin) = store[k]
        P = h.shape[0]
        q, kk, _ = mod.qkv(h).reshape(P, 3, mod.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, kk = apply_rotary_pos_emb_vision(q, kk, cos, sin)
        q, kk = q.float().transpose(0, 1), kk.float().transpose(0, 1)          # (heads, P, hd)
        rec = {'received': torch.zeros(mod.num_heads, P)}
        for name in sets:
            rec[f'mass_{name}'] = torch.zeros(mod.num_heads)
            rec[f'self_{name}'] = torch.zeros(mod.num_heads)
        for hd in range(mod.num_heads):
            a = ((q[hd] @ kk[hd].T) * mod.scaling).softmax(-1)                  # (P, P)
            rec['received'][hd] = a.mean(0).cpu()
            for name, m in sets.items():
                m = m.to(a.device)
                to_set = a[:, m].sum(-1)
                rec[f'mass_{name}'][hd] = float(to_set.mean())
                rec[f'self_{name}'][hd] = float(to_set[m].mean()) if m.any() else float('nan')
        out[k] = rec
    return out


# ----------------------------------------------------------------------------- batched yes/no margins

@torch.inference_mode()
def yes_no_batch(rt: qr.Runtime, items: list[tuple[list[Image.Image], str]]) -> list[dict]:
    """First-answer-token margins s = z_Yes - z_No for a left-padded batch (prefill only)."""
    texts = [qr.prompt_text(rt, len(imgs), text) for imgs, text in items]
    images = [im for imgs, _ in items for im in imgs]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images or None, padding=True, return_tensors='pt').to(rt.device)
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist()):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError('Processor resized an image.')
    logits = rt.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    y, n = qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No')
    out = []
    for row in logits:
        top = int(row.argmax())
        out.append({'s': float(row[y] - row[n]), 'z_yes': float(row[y]), 'z_no': float(row[n]),
                    'response': rt.tokenizer.decode([top]), 'finish_reason': 'eos'})
    return out


# ----------------------------------------------------------------------------- position-explained variance

class PosVar:
    """Streaming split-half estimate of R^2_pos (design 4.2) for one site and one canvas size.

    Images are assigned to half A or B by their index parity. For each half the per-position sums S_p,
    per-position sums of squared norms Q_p and counts are kept, so that
    sum_{n in B} ||x_{n,p} - mu^A_p||^2 = Q^B_p - 2 <S^B_p, mu^A_p> + N_B ||mu^A_p||^2.
    """

    def __init__(self, P: int, D: int, device='cpu'):
        self.S = torch.zeros(2, P, D, dtype=torch.float64, device=device)
        self.Q = torch.zeros(2, P, dtype=torch.float64, device=device)
        self.N = [0, 0]

    def add(self, x: torch.Tensor, index: int) -> None:
        h = index % 2
        x = x.to(self.S.device, torch.float64)
        self.S[h] += x
        self.Q[h] += x.square().sum(-1)
        self.N[h] += 1

    def r2(self, exclude: torch.Tensor | None = None) -> dict:
        """R^2_pos averaged over the two directions; exclude: (P,) bool positions left out of both sums."""
        keep = torch.ones(self.S.shape[1], dtype=torch.bool) if exclude is None else ~exclude.cpu()
        keep = keep.to(self.S.device)
        vals = []
        for a, b in ((0, 1), (1, 0)):
            mu = self.S[a, keep] / self.N[a]
            gbar = mu.mean(0)
            Sb, Qb, Nb = self.S[b, keep], self.Q[b, keep], self.N[b]
            num = (Qb - 2 * (Sb * mu).sum(-1) + Nb * mu.square().sum(-1)).sum()
            den = (Qb - 2 * (Sb @ gbar) + Nb * gbar.square().sum()).sum()
            vals.append(float(1 - num / den))
        return {'r2': float(np.mean(vals)), 'r2_ab': vals, 'n': list(self.N), 'positions': int(keep.sum())}


# ----------------------------------------------------------------------------- probe inputs

def patch_content(image: Image.Image, hp: int, wp: int) -> np.ndarray:
    """(hp*wp, 18) raster per-patch content: 4x4 mean-pooled gray (16), mean gray and gray std."""
    g = np.asarray(image.convert('L'), np.float32) / 255.
    t = g[:hp * 16, :wp * 16].reshape(hp, 16, wp, 16).transpose(0, 2, 1, 3)
    pooled = t.reshape(hp, wp, 4, 4, 4, 4).mean((3, 5)).reshape(hp * wp, 16)
    return np.concatenate([pooled, t.reshape(hp * wp, -1).mean(1, keepdims=True),
                           t.reshape(hp * wp, -1).std(1, keepdims=True)], 1)


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    return vl.wilson(k, n, z)
