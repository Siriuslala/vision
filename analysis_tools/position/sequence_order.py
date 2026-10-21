"""Experiment 3.8: the positional role of sequence order in Qwen3.5-9B.

Engine of the experiment (stage drivers live in `sequence_order_run.py`):

* grid stimuli with one red target (stimuli-v1 generator) and the experiment-1.3 row/column prompt;
* a sequence builder that writes an arbitrary list of visual embeddings (a permutation of the image's
  own merged tokens, optionally with inserted extra tokens) into the image span of the prompt;
* slot orders that dissociate the four candidate mechanisms of the order effect (token-index
  counting, row-start counting, sequence-neighbour borrowing, ViT position);
* a readout of the row and column answers as expected values of the digit distribution;
* mixer-output restoration hooks: at chosen layers the token-mixer (Gated DeltaNet or full
  attention) output at visual slots is replaced by the output the same visual token received
  in the unpermuted run, which removes the order dependence that the layer writes into visual tokens.

Layers are 1-based. Token coordinates are in merged-token units (32 pixels).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.common import stimuli as st
from analysis_tools.position import coordinate_readout as cr

TOKEN = st.TOKEN_PIXELS
CELL = 64                                    # grid cell size in pixels (2x2 merged tokens)
CELL_TOKENS = CELL // TOKEN
RED = st.Change('color', 4)                  # level 4 of 'color' is pure red, as in experiment 1.3 T6
T6_QUERY = 'One figure is drawn in red.'
ROWCOL_SUFFIX = (' Give its position. Return only {"row": r, "column": c}, counting rows from top to bottom '
                 'and columns from left to right, starting at 1.')
PROMPT = T6_QUERY + ROWCOL_SUFFIX + cr.JSON_ONLY
LINEAR_LAYERS = tuple(i for i in range(1, 33) if i % 4)
FULL_LAYERS = tuple(range(4, 33, 4))


# ----------------------------------------------------------------------------- stimuli

def grid_image(template: str, rows: int, cols: int, target: tuple[int, int], seed: int = 0) -> tuple[Image.Image, dict]:
    """rows x cols grid of 64-pixel instances with one red target; canvas fits the grid exactly when >= 256 px."""
    spec = st.StimulusSpec(template, rows, cols, change=RED, target=target, cell=CELL, seed=seed)
    image, meta = st.generate(spec)
    if meta['grid_origin'] != [0, 0]:
        raise ValueError('Grid must fill the canvas so that cell (r, c) covers tokens (2r..2r+1, 2c..2c+1).')
    return image, meta


def cell_tokens(rc: tuple[int, int], token_grid: tuple[int, int]) -> list[int]:
    """Raster indices of the 2x2 merged tokens of a grid cell (grid origin at the canvas corner)."""
    r, c = rc
    W = token_grid[1]
    return [(CELL_TOKENS * r + i) * W + CELL_TOKENS * c + j for i in range(CELL_TOKENS) for j in range(CELL_TOKENS)]


# ----------------------------------------------------------------------------- slot orders

def order_identity(H: int, W: int) -> np.ndarray:
    return np.arange(H * W)


def order_roll_rows(H: int, W: int, k: int) -> np.ndarray:
    """Slot row R holds token row (R - k) mod H: the image is rolled down by k token rows, rows stay intact."""
    g = np.arange(H * W).reshape(H, W)
    return np.roll(g, k, axis=0).ravel()


def order_roll_within_rows(H: int, W: int, k: int) -> np.ndarray:
    """Slot (R, C) holds token (R, (C - k) mod W): every row is rolled right by k tokens."""
    g = np.arange(H * W).reshape(H, W)
    return np.roll(g, k, axis=1).ravel()


def order_transpose(H: int, W: int) -> np.ndarray:
    """Column-major reading: slot k holds token (k mod H, k // H); the slot grid keeps shape (H, W)."""
    g = np.arange(H * W).reshape(H, W)
    return g.T.ravel()


def order_reverse_rows(H: int, W: int) -> np.ndarray:
    """Rows read bottom to top, each row left to right."""
    return np.arange(H * W).reshape(H, W)[::-1].ravel()


def order_mirror_rows(H: int, W: int) -> np.ndarray:
    """Rows read top to bottom, each row right to left."""
    return np.arange(H * W).reshape(H, W)[:, ::-1].ravel()


def order_block_move(H: int, W: int, src: list[int], dst: list[int]) -> np.ndarray:
    """Exchange the slots of two equal-size disjoint token sets (experiment 3.5 S4 permutation)."""
    return cr.swap_index(H * W, src, dst).numpy()


def order_shuffle_others(H: int, W: int, keep: list[int], seed: int, move_to: list[int] | None = None) -> np.ndarray:
    """Randomly permute every token outside `keep`; `keep` stays in place or is placed at slots `move_to`."""
    rng = np.random.default_rng(seed)
    n = H * W
    dest = list(keep) if move_to is None else list(move_to)
    others = [t for t in range(n) if t not in set(keep)]
    free = [s for s in range(n) if s not in set(dest)]
    order = np.empty(n, int)
    order[dest] = keep
    order[free] = rng.permutation(others)
    return order


def order_carry(H: int, W: int, rc: tuple[int, int], dr: int, left: int, right: int) -> np.ndarray:
    """Move the target cell dr cells vertically together with `left` / `right` tokens of row context.

    The source is the target's two token rows restricted to token columns [2c - left, 2c + 2 + right)
    (clipped to the image); it exchanges slots with the same columns dr cells away. left = right = 0
    is the block move of experiment 3.5; left = right = W exchanges two complete cell rows.
    """
    r, c = rc
    c0, c1 = max(0, CELL_TOKENS * c - left), min(W, CELL_TOKENS * (c + 1) + right)
    rows = [CELL_TOKENS * r + i for i in range(CELL_TOKENS)]
    src = [R * W + C for R in rows for C in range(c0, c1)]
    dst = [(R + CELL_TOKENS * dr) * W + C for R in rows for C in range(c0, c1)]
    return order_block_move(H, W, src, dst)


def order_segment_shift(H: int, W: int, rc: tuple[int, int], dc: int, carry: int) -> np.ndarray:
    """Shift the target cell dc = +1 / -1 cells horizontally together with `carry` tokens of trailing context.

    For dc = +1 the token segment [2c - carry, 2c + 4) of the target's token rows is rolled right by
    2 slots (the target and its `carry` left neighbours move right; the two displaced tokens wrap to the
    segment start); dc = -1 mirrors this with right-hand context. carry = 0 equals the block move.
    """
    r, c = rc
    g = np.arange(H * W).reshape(H, W)
    if dc == 1:
        c0, c1 = CELL_TOKENS * c - carry, CELL_TOKENS * (c + 2)
    else:
        c0, c1 = CELL_TOKENS * (c - 1), CELL_TOKENS * (c + 1) + carry
    if c0 < 0 or c1 > W:
        raise ValueError('Segment exceeds the image.')
    for R in range(CELL_TOKENS * r, CELL_TOKENS * (r + 1)):
        g[R, c0:c1] = np.roll(g[R, c0:c1], CELL_TOKENS * dc)
    return g.ravel()


def slot_cell(order: np.ndarray, target_tokens: list[int], W: int, offset: int = 0) -> tuple[float, float]:
    """Cell coordinates (row, col) implied by the raster position of the slots holding the target tokens.

    offset: number of extra slots inserted before the image tokens (shifts the raster index).
    """
    inv = np.argsort(order)
    slots = inv[target_tokens] + offset
    rows, cols = slots // W, slots % W
    return float((rows.mean() - 0.5) / CELL_TOKENS), float((cols.mean() - 0.5) / CELL_TOKENS)


# ----------------------------------------------------------------------------- sequence builder

@dataclass
class Sequence:
    """One visual-span content: (Nv', D) embeddings and (3, Nv') MRoPE ids written into the image span."""
    vis: torch.Tensor
    mrope: torch.Tensor
    order: np.ndarray | None = None        # permutation of the original tokens, when the span is a permutation


def permuted(prep: cr.Prepared, order: np.ndarray) -> Sequence:
    """Slot i holds token order[i] with its own embedding and MRoPE ids (experiment 3.5 S4).

    `order` may also be any index list (duplicated or omitted tokens) for sequence-editing conditions.
    """
    s, e = prep.span['start'], prep.span['end']
    o = torch.as_tensor(order, device=prep.vis.device)
    return Sequence(prep.vis[o], prep.pos[1:, 0, s:e][:, o.to(prep.pos.device)], np.asarray(order))


def with_inserted(prep: cr.Prepared, base: Sequence, extra: torch.Tensor, at: int,
                  extra_mrope: torch.Tensor | None = None) -> Sequence:
    """Insert (k, D) extra embeddings before slot `at`; they reuse the MRoPE ids of the slot they precede."""
    k = extra.shape[0]
    if extra_mrope is None:
        ref = base.mrope[:, min(at, base.mrope.shape[1] - 1)]
        extra_mrope = ref[:, None].expand(3, k)
    vis = torch.cat([base.vis[:at], extra.to(base.vis), base.vis[at:]], 0)
    mrope = torch.cat([base.mrope[:, :at], extra_mrope.to(base.mrope), base.mrope[:, at:]], 1)
    return Sequence(vis, mrope, None)


def build(rt: qr.Runtime, prep: cr.Prepared, seq: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
    """inputs_embeds (1, L', D) and position ids (4, 1, L') for a prompt whose image span holds `seq`.

    Row 0 of the position ids (sequence index, used for the causal mask) is re-numbered 0..L'-1; the text
    MRoPE ids before and after the image are left unchanged (experiment 3.5: MRoPE ids do not affect answers).
    """
    s, e = prep.span['start'], prep.span['end']
    emb = rt.model.model.get_input_embeddings()
    pre, post = emb(prep.ids[None, :s]), emb(prep.ids[None, e:])
    x = torch.cat([pre, seq.vis[None].to(pre.dtype), post], 1)
    L = x.shape[1]
    mrope = torch.cat([prep.pos[1:, :, :s], seq.mrope[:, None].to(prep.pos), prep.pos[1:, :, e:]], 2)
    pos = torch.cat([torch.arange(L, device=mrope.device, dtype=mrope.dtype).view(1, 1, L), mrope], 0)
    return x, pos


# ----------------------------------------------------------------------------- answer readout

def digit_ids(rt: qr.Runtime) -> list[int]:
    return [qr.token_id(rt, str(d)) for d in range(10)]


def read_rowcol(rt: qr.Runtime, result: dict, n_rows: int, n_cols: int) -> dict:
    """Parse a row/column answer and the expected row/column from the digit logits at the value tokens.

    Expected values are taken over the digits 1..n (zero-based output: value - 1) and require
    single-digit answers; answers outside that form keep only their parsed values.
    """
    out = {'text': result['text'], 'finish_reason': result['finish_reason']}
    rc = cr.parse_rowcol(result['text'])
    out['valid'] = rc is not None
    if rc is None:
        return out
    out['row'], out['col'] = rc[0] - 1, rc[1] - 1
    spans = cr.coordinate_tokens(rt, result['ids'], 'rowcol')
    if len(spans) == 2 and all(len(s) == 1 for s in spans):
        for key, (step,), n in (('row', spans[0], n_rows), ('col', spans[1], n_cols)):
            z = torch.tensor(result['tracked'][step][1:n + 1])
            p = z.softmax(0)
            out[f'e_{key}'] = float((p * torch.arange(n)).sum())
            out[f'p_{key}'] = [round(float(v), 5) for v in p]
    return out


def decode(rt: qr.Runtime, xs: list[torch.Tensor], poss: list[torch.Tensor], batch_size: int,
           max_new_tokens: int = 24, hooks=None) -> list[dict]:
    """Greedy decoding of equal-length conditions in unpadded batches, tracking the ten digit logits.

    hooks: optional callable (start, stop) -> context manager installed around each batch (prefill only).
    """
    ids = digit_ids(rt)
    results = []
    for i in range(0, len(xs), batch_size):
        x = torch.cat(xs[i:i + batch_size], 0)
        p = torch.cat(poss[i:i + batch_size], 1)
        if hooks is None:
            results.extend(cr.greedy(rt, x, p, max_new_tokens, track_ids=ids))
        else:
            with hooks(i, min(i + batch_size, len(xs))):
                results.extend(cr.greedy(rt, x, p, max_new_tokens, track_ids=ids))
    return results


# ----------------------------------------------------------------------------- mixer capture and restoration

def mixer(rt: qr.Runtime, layer: int) -> torch.nn.Module:
    block = rt.layers[layer - 1]
    return block.linear_attn if block.block_type == 'linear_attention' else block.self_attn


def _mix_out(output):
    return output[0] if isinstance(output, tuple) else output


@torch.inference_mode()
def capture_mixers(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, span: tuple[int, int]) -> dict:
    """Token-mixer outputs (Nv, D) at the visual span for every layer of one unpermuted prefill pass."""
    s, e = span
    store, handles = {}, []
    for layer in range(1, 33):
        def hook(module, args, output, layer=layer):
            store[layer] = _mix_out(output)[0, s:e].detach().clone()
        handles.append(mixer(rt, layer).register_forward_hook(hook))
    try:
        rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return store


@contextmanager
def restore_mixers(rt: qr.Runtime, clean: dict, span: tuple[int, int], rows: list[tuple], prompt_len: int):
    """Replace the mixer output at visual slots with the clean output of the token occupying each slot.

    rows: per batch row, (order, layers, slots): slot i receives clean[layer][order[i]] at every layer in
    `layers`, for the slots listed in `slots` (None means every visual slot). Rows with no layers are untouched.
    Fires only on the prefill pass (sequence length equal to `prompt_len`).
    """
    s, e = span
    handles = []
    for layer in range(1, 33):
        sel = []
        for b, (o, ls, slots) in enumerate(rows):
            if layer not in ls:
                continue
            idx = torch.arange(e - s) if slots is None else torch.as_tensor(slots)
            sel.append((b, idx.to(clean[layer].device), torch.as_tensor(o)[idx].to(clean[layer].device)))
        if not sel:
            continue
        def hook(module, args, output, layer=layer, sel=sel):
            h = _mix_out(output)
            if h.shape[1] != prompt_len:
                return output
            for b, idx, src in sel:
                h[b, s + idx] = clean[layer][src].to(h.dtype)
            return output
        handles.append(mixer(rt, layer).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.inference_mode()
def capture_residuals(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, span: tuple[int, int],
                      layers: list[int]) -> dict:
    """Residual stream at the visual span: layer 0 is the decoder input, layer l the output of layer l (float16, CPU)."""
    s, e = span
    store, handles = {}, []
    if 0 in layers:
        def pre(module, args, kwargs):
            hs = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            store[0] = hs[0, s:e].detach().to(torch.float16).cpu()
        handles.append(rt.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
    for layer in [l for l in layers if l > 0]:
        def hook(module, args, output, layer=layer):
            store[layer] = _mix_out(output)[0, s:e].detach().to(torch.float16).cpu()
        handles.append(rt.layers[layer - 1].register_forward_hook(hook))
    try:
        rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return store


# ----------------------------------------------------------------------------- probes

def ridge_r2(X: torch.Tensor, y: torch.Tensor, train: torch.Tensor, test: torch.Tensor,
             alphas=(1e0, 1e1, 1e2, 1e3, 1e4)) -> float:
    """Held-out R^2 of a ridge probe; alpha chosen on the last fifth of the training rows.

    X: (n, d) features, y: (n,) target, train/test: index tensors. Features are standardized on train.
    """
    X = X.double()
    y = y.double()
    mu, sd = X[train].mean(0), X[train].std(0).clamp_min(1e-6)
    Z = (X - mu) / sd
    def fit(idx, alpha):
        A = Z[idx]
        ym = y[idx].mean()
        if A.shape[0] < A.shape[1]:
            K = A @ A.T
            w = A.T @ torch.linalg.solve(K + alpha * torch.eye(len(idx), dtype=K.dtype, device=K.device), y[idx] - ym)
        else:
            G = A.T @ A
            w = torch.linalg.solve(G + alpha * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), A.T @ (y[idx] - ym))
        return w, ym
    def r2(idx, w, ym):
        pred = Z[idx] @ w + ym
        return float(1 - ((pred - y[idx]) ** 2).sum() / ((y[idx] - y[idx].mean()) ** 2).sum().clamp_min(1e-12))
    cut = int(len(train) * 0.8)
    inner, val = train[:cut], train[cut:]
    best = max(alphas, key=lambda a: r2(val, *fit(inner, a)))
    return r2(test, *fit(train, best))
