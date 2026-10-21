"""Experiment 3.8 round 2: how the Gated DeltaNet layers of Qwen3.5-9B turn sequence order into row ordinals.

Engine (stage drivers live in `sequence_order_r2_run.py`):

* cell-level sequence layouts: a visual span described as a list of rows of items (grid cells or blank
  tokens), with the per-cell variables that dissociate the candidate position sources
  (ViT cell row / column, figure-row ordinal counted along the sequence, slot row);
* gate recording: the decay alpha_t = exp(g_t) and write strength beta_t of every Gated DeltaNet head at
  the visual span, computed from the hooked `in_proj_a` / `in_proj_b` outputs exactly as the model does;
* gate overrides at chosen layers and sequence positions (reset alpha = 0, no decay alpha = 1,
  no write beta = 0), prefill only;
* grouped feature capture: residual-stream outputs and per-head DeltaNet read-outs (the core attention
  output before the gated RMSNorm) averaged over token groups (grid cells).

Layers are 1-based; token coordinates are in merged-token units (32 pixels). The shared modules
(`qwen_runtime`, `stimuli`) and the round-1 engine are used unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import sequence_order as so

CT = so.CELL_TOKENS
ROWCOL = re.compile(r'\{\s*"row"\s*:\s*(\d+)\s*,\s*"column"\s*:\s*(\d+)\s*\}')
THINK_SUFFIX = (' First describe step by step how you determine its row and its column. Then give the final answer '
                'on the last line as {"row": r, "column": c}, counting rows from top to bottom and columns from left '
                'to right, starting at 1.')
THINK_PROMPT = so.T6_QUERY + THINK_SUFFIX
MID_LINEAR = (9, 10, 11, 13, 14, 15)          # DeltaNet layers inside layers 9-16
EARLY_LINEAR = (1, 2, 3, 5, 6, 7)
LATE_LINEAR = (17, 18, 19, 21, 22, 23)


def parse_last_rowcol(text: str) -> tuple[int, int] | None:
    """Last {"row": r, "column": c} pair of a free-form response (1-based), or None."""
    found = ROWCOL.findall(text)
    return (int(found[-1][0]), int(found[-1][1])) if found else None


# ----------------------------------------------------------------------------- figure masks

def figure_mask(image: Image.Image, grid: tuple[int, int], dark: int = 200, frac: float = 0.03) -> np.ndarray:
    """(H*W,) bool: merged tokens whose 32x32 patch has more than `frac` non-background pixels.

    A pixel counts as figure when any channel is below `dark` (the canvas is white).
    """
    H, W = grid
    a = np.asarray(image.convert('RGB'), dtype=np.uint8)
    ink = (a.min(-1) < dark).astype(np.float32)
    ink = ink[:H * so.TOKEN, :W * so.TOKEN].reshape(H, so.TOKEN, W, so.TOKEN).mean((1, 3))
    return (ink > frac).ravel()


# ----------------------------------------------------------------------------- cell layouts

@dataclass
class Layout:
    """A visual span built from grid cells.

    rows:  list of cell rows; each cell row is a list of items, an item is (r, c) for grid cell (r, c)
           of the image or None for a blank cell (2x2 white-canvas tokens). Every cell row spans two
           token rows; the token order is raster order within the cell row (top token row of every
           item, then the bottom token row).
    index: (Nv',) token indices into the image (-1 for blank tokens), in sequence order.
    cell_of_slot: (Nv',) running id of the cell instance occupying each slot (-1 for blank tokens).
    cells: per cell instance, a dict of the dissociating variables (all 0-based, in cells):
           vit_row / vit_col  the cell's position in the image (carried by its ViT embedding);
           ord_row            number of figure cell rows earlier in the sequence (counted from the start);
           ord_col            number of figure cells earlier in the same cell row;
           slot_row           cell-row index implied by the token index (raster index // (2 W)).
    """
    rows: list
    index: np.ndarray
    cell_of_slot: np.ndarray
    cells: list = field(default_factory=list)


def make_layout(rows: list, W: int) -> Layout:
    index, cell_of, cells = [], [], []
    ord_row = 0
    for items in rows:
        has_fig = any(it is not None for it in items)
        ids = []
        oc = 0
        for it in items:
            if it is None:
                ids.append(-1)
                continue
            ids.append(len(cells))
            cells.append({'vit_row': it[0], 'vit_col': it[1], 'ord_row': ord_row, 'ord_col': oc})
            oc += 1
        for i in range(CT):
            for it, cid in zip(items, ids):
                for j in range(CT):
                    index.append(-1 if it is None else (CT * it[0] + i) * W + CT * it[1] + j)
                    cell_of.append(cid)
        ord_row += int(has_fig)
    index, cell_of = np.array(index), np.array(cell_of)
    for cid, cell in enumerate(cells):
        slots = np.where(cell_of == cid)[0]
        cell['slot_row'] = float(slots.mean() // (CT * W))
        cell['first_slot'] = int(slots.min())
    return Layout(rows, index, cell_of, cells)


def grid_rows(R: int, C: int) -> list:
    return [[(r, c) for c in range(C)] for r in range(R)]


def layout_conditions(R: int = 8, C: int = 8) -> dict[str, list]:
    """Cell-row layouts that dissociate ViT row / column, figure ordinal and slot row (results doc §2.3)."""
    g = grid_rows(R, C)
    blank_row = [None] * C
    return {
        'identity': g,
        'insert_blank_start': [blank_row] + g,
        'insert_blank_mid': g[:4] + [blank_row] + g[4:],
        'dup_row_mid': g[:4] + [g[3]] + g[4:],
        'delete_row_mid': g[:3] + g[4:],
        'delete_row0': g[1:],
        'roll_rows+1': g[-1:] + g[:-1],
        'roll_rows-1': g[1:] + g[:1],
        'swap_rows_34': g[:3] + [g[4], g[3]] + g[5:],
        # Column analogues edit every cell row in the same way.
        'dup_col_mid': [row[:4] + [row[3]] + row[4:] for row in g],
        'delete_col_mid': [row[:3] + row[4:] for row in g],
        'roll_within+1': [row[-1:] + row[:-1] for row in g],
    }


def layout_sequence(prep, layout: Layout, blank_token: torch.Tensor) -> so.Sequence:
    """Visual embeddings for a layout; blank tokens take `blank_token` (D,) and the MRoPE id of the next slot."""
    s, e = prep.span['start'], prep.span['end']
    idx = torch.as_tensor(layout.index, device=prep.vis.device)
    vis = prep.vis[idx.clamp_min(0)].clone()
    vis[idx < 0] = blank_token.to(vis)
    mrope_all = prep.pos[1:, 0, s:e]
    mi = torch.as_tensor(layout.index).clone()
    for k in range(len(mi) - 1, -1, -1):              # blanks copy the MRoPE id of the following slot
        if mi[k] < 0:
            mi[k] = mi[k + 1] if k + 1 < len(mi) else mrope_all.shape[1] - 1
    return so.Sequence(vis, mrope_all[:, mi.to(mrope_all.device)], None)


# ----------------------------------------------------------------------------- gate recording and overrides

def linear_block(rt: qr.Runtime, layer: int):
    block = rt.layers[layer - 1]
    if block.block_type != 'linear_attention':
        raise ValueError(f'Layer {layer} is not a Gated DeltaNet layer.')
    return block.linear_attn


def gate_values(mod, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """alpha = exp(g) with g = -exp(A_log) softplus(a + dt_bias) and beta = sigmoid(b), as in the model forward."""
    g = -mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias)
    return g.exp(), b.float().sigmoid()


@contextmanager
def record_gates(rt: qr.Runtime, layers, span: tuple[int, int], store: dict, prompt_len: int):
    """store[layer] = (alpha, beta), each (B, span_len, 32) float32 on CPU, from the prefill pass."""
    s, e = span
    handles, tmp = [], {}
    for layer in layers:
        mod = linear_block(rt, layer)

        def hook(m, args, out, layer=layer, mod=mod, key='a'):
            if out.shape[1] != prompt_len:
                return
            tmp.setdefault(layer, {})[key] = out[:, s:e].detach()
            if len(tmp[layer]) == 2:                  # in_proj_b runs before in_proj_a in the model forward
                d = tmp.pop(layer)
                al, be = gate_values(mod, d['a'], d['b'])
                store[layer] = (al.cpu(), be.cpu())
        handles.append(mod.in_proj_a.register_forward_hook(hook))
        handles.append(mod.in_proj_b.register_forward_hook(lambda m, a, o, h=hook: h(m, a, o, key='b')))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


BIG = 1e4   # a = +BIG gives g <= -0.01 * BIG (alpha = 0); a = -BIG gives softplus = 0 (alpha = 1); b = -BIG gives beta = 0


@contextmanager
def override_gates(rt: qr.Runtime, rows: list[dict], prompt_len: int):
    """Per batch row, a dict {'layers': iterable, 'reset': positions, 'nodecay': positions, 'nowrite': positions}.

    Positions are absolute sequence indices. reset sets alpha_t = 0 (state erased before the write at t),
    nodecay sets alpha_t = 1, nowrite sets beta_t = 0. Heads: all 32 heads, or row['heads'] if given.
    Only the prefill pass (length `prompt_len`) is modified.
    """
    layers = sorted({l for row in rows for l in row.get('layers', ())})
    handles = []

    def plan(layer, key):
        out = []
        for b, row in enumerate(rows):
            if layer in row.get('layers', ()) and len(row.get(key, ())):
                heads = row.get('heads')
                out.append((b, torch.as_tensor(np.asarray(row[key]), dtype=torch.long),
                            None if heads is None else torch.as_tensor(heads, dtype=torch.long)))
        return out
    for layer in layers:
        mod = linear_block(rt, layer)
        a_plan = [(p, BIG) for p in plan(layer, 'reset')] + [(p, -BIG) for p in plan(layer, 'nodecay')]
        b_plan = plan(layer, 'nowrite')

        def edit(out, items, value=None):
            if out.shape[1] != prompt_len:
                return out
            out = out.clone()
            for item in items:
                (b, pos, heads), v = item if value is None else (item, value)
                pos = pos.to(out.device)
                if heads is None:
                    out[b, pos] = v
                else:
                    sub = out[b, pos]
                    sub[:, heads.to(out.device)] = v
                    out[b, pos] = sub
            return out
        if a_plan:
            handles.append(mod.in_proj_a.register_forward_hook(lambda m, a, o, p=a_plan: edit(o, p)))
        if b_plan:
            handles.append(mod.in_proj_b.register_forward_hook(lambda m, a, o, p=b_plan: edit(o, p, -BIG)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ----------------------------------------------------------------------------- grouped features

def group_mean(h: torch.Tensor, groups: torch.Tensor, n: int) -> torch.Tensor:
    """h: (T, ...) features, groups: (T,) ids in [0, n) or -1; returns (n, ...) group means (float32)."""
    keep = groups >= 0
    h, g = h[keep].float(), groups[keep].to(h.device)
    out = torch.zeros((n,) + h.shape[1:], device=h.device)
    out.index_add_(0, g, h)
    cnt = torch.bincount(g, minlength=n).clamp_min(1).view(-1, *([1] * (h.dim() - 1)))
    return out / cnt


@torch.inference_mode()
def capture_grouped(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, span_start: int, groups: np.ndarray,
                    res_layers, head_layers) -> dict:
    """One prefill pass (batch 1); returns {'res': {layer: (n, D)}, 'head': {layer: (n, 32, 128)}} float16 CPU.

    Residual layer 0 is the decoder input. Head features are the DeltaNet core outputs o_t = S_t^T q_t
    (before the gated RMSNorm), i.e. what each head reads from its recurrent state at the token.
    """
    g = torch.as_tensor(groups, dtype=torch.long)
    n = int(g.max()) + 1
    s, e = span_start, span_start + len(groups)
    L = x.shape[1]
    store = {'res': {}, 'head': {}}
    handles = []
    if 0 in res_layers:
        def pre(module, args, kwargs):
            hs = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            store['res'][0] = group_mean(hs[0, s:e], g, n).half().cpu()
        handles.append(rt.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
    for layer in [l for l in res_layers if l > 0]:
        def hook(module, args, output, layer=layer):
            h = output[0] if isinstance(output, tuple) else output
            store['res'][layer] = group_mean(h[0, s:e], g, n).half().cpu()
        handles.append(rt.layers[layer - 1].register_forward_hook(hook))
    for layer in head_layers:
        mod = linear_block(rt, layer)

        def pre_norm(module, args, layer=layer):
            o = args[0].view(1, L, -1, args[0].shape[-1])
            store['head'][layer] = group_mean(o[0, s:e], g, n).half().cpu()
        handles.append(mod.norm.register_forward_pre_hook(pre_norm))
    try:
        rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return store


# ----------------------------------------------------------------------------- ridge

def ridge_fit(X: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardized ridge; returns (mu, sd, w, intercept). X: (n, d), y: (n,) or (n, k)."""
    X, y = X.double(), y.double()
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Z = (X - mu) / sd
    ym = y.mean(0)
    if Z.shape[0] < Z.shape[1]:
        K = Z @ Z.T
        w = Z.T @ torch.linalg.solve(K + alpha * torch.eye(len(Z), dtype=K.dtype, device=K.device), y - ym)
    else:
        G = Z.T @ Z
        w = torch.linalg.solve(G + alpha * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), Z.T @ (y - ym))
    return mu, sd, w, ym


def ridge_predict(model, X: torch.Tensor) -> torch.Tensor:
    mu, sd, w, ym = model
    return ((X.double() - mu) / sd) @ w + ym


def r2_score(pred: torch.Tensor, y: torch.Tensor) -> float:
    y = y.double()
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-12))


def ridge_select(X: torch.Tensor, y: torch.Tensor, groups: np.ndarray, alphas=(1e1, 1e2, 1e3, 1e4)):
    """Fit ridge with alpha chosen by a held-out fifth of the groups (images); returns (model, alpha)."""
    ug = np.unique(groups)
    val = torch.as_tensor(np.isin(groups, ug[::5])).to(X.device)
    best, best_r2 = None, -np.inf
    for a in alphas:
        m = ridge_fit(X[~val], y[~val], a)
        sc = r2_score(ridge_predict(m, X[val]), y[val])
        if sc > best_r2:
            best, best_r2 = a, sc
    return ridge_fit(X, y, best), best
