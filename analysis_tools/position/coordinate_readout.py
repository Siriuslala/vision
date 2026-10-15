"""Experiment 3.5: coordinate readout path of Qwen3.5-9B.

This module holds the reusable engine of the experiment:

* single-target stimuli (red discs on a white 512x512 canvas) and answer parsing;
* a prepared prompt (`Prepared`) that exposes the three position sources the design
  intervenes on: ViT inputs (learned absolute position embedding taps and 2D RoPE ids),
  decoder MRoPE position ids, and the sequence order of visual tokens;
* a manual greedy decoder that accepts arbitrary inputs_embeds and (4, B, L) position ids,
  so that every intervention is a pure change of tensors fed to the frozen model;
* attention readout at full-attention layers (softmax weights, gated value contributions)
  and gradient x activation attribution at every layer.

Stage drivers that produce the artifacts live in `coordinate_readout_run.py`.
Layer numbers are 1-based throughout, as in the analysis documents.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import re

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

from analysis_tools.common import qwen_runtime as qr

CANVAS = 512
TOKEN_PIXELS = 32           # one merged visual token covers 32x32 input pixels
MERGE = 2                   # ViT patches per merged token along each axis
RED = (255, 0, 0)
COORD_NOTE = ' All coordinates use 0-1000 normalization; x increases rightward and y increases downward.'
JSON_ONLY = ' Output only the JSON, with no explanation.'
DISC_QUERY = 'There is one red circle in the image.'
SUFFIX = {
    'point': ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].' + COORD_NOTE + JSON_ONLY,
    'bbox': ' Give its tight bounding box. Return only [{"bbox_2d": [x_min, y_min, x_max, y_max]}].'
            + COORD_NOTE + JSON_ONLY,
}
RECOGNITION_QUESTION = 'Is there a red circle in the image? Answer with only Yes or No.'


# ----------------------------------------------------------------------------- stimuli

def disc_image(center: tuple[float, float], diameter: int, size: int = CANVAS) -> tuple[Image.Image, dict]:
    """White canvas with one filled red disc; the truth box is the half-open tight pixel box."""
    image = Image.new('RGB', (size, size), (255, 255, 255))
    cx, cy = center
    r = diameter / 2
    ImageDraw.Draw(image).ellipse((cx - r, cy - r, cx + r - 1, cy + r - 1), fill=RED)
    dark = np.argwhere((np.asarray(image) != 255).any(-1))
    (y0, x0), (y1, x1) = dark.min(0), dark.max(0) + 1
    box = [float(x0), float(y0), float(x1), float(y1)]
    return image, {'canvas': [size, size], 'diameter': diameter, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                   'box': box}


def grid_centers(n: int = 6, size: int = CANVAS) -> list[tuple[float, float]]:
    step = size / n
    return [((c + 0.5) * step, (r + 0.5) * step) for r in range(n) for c in range(n)]


def tokens_overlapping(box: list[float], grid: tuple[int, int]) -> list[int]:
    """Raster indices of merged visual tokens whose 32x32 cell overlaps a half-open pixel box."""
    H, W = grid
    r0, r1 = int(box[1] // TOKEN_PIXELS), int(math.ceil(box[3] / TOKEN_PIXELS))
    c0, c1 = int(box[0] // TOKEN_PIXELS), int(math.ceil(box[2] / TOKEN_PIXELS))
    return [r * W + c for r in range(max(r0, 0), min(r1, H)) for c in range(max(c0, 0), min(c1, W))]


# ----------------------------------------------------------------------------- answers

def parse_coords(text: str, fmt: str) -> list[float] | None:
    """Strict-content parse of point/bbox answers: first JSON value carrying the requested key.

    Accepts the requested list-of-dict schema, a bare dict, or a bare number list of the right length;
    returns None for anything else (counted as a failure by callers).
    """
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


def parse_rowcol(text: str) -> tuple[int, int] | None:
    body = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.split('</think>')[-1].strip())
    try:
        value = json.loads(body)
    except ValueError:
        return None
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]
    if not isinstance(value, dict) or type(value.get('row')) is not int or type(value.get('column')) is not int:
        return None
    return value['row'], value['column']


def answer_center(coords: list[float]) -> np.ndarray:
    """Point answer, or the center of a bbox answer, in 0-1000 units."""
    c = np.asarray(coords, float)
    return c if len(c) == 2 else (c[:2] + c[2:]) / 2


def box_iou(a, b) -> float:
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.


def score_disc(truth: dict, fmt: str, text: str) -> dict:
    """Point hit (inside the disc), center error (0-1000) and bbox IoU against the tight box."""
    coords = parse_coords(text, fmt)
    if coords is None:
        return {'valid': False, 'hit': False, 'correct': False}
    size = np.asarray(truth['canvas'], float)
    center = answer_center(coords)
    true_center = 1000 * np.asarray(truth['center']) / size
    pixel = center * size / 1000
    inside = float(np.linalg.norm(pixel - np.asarray(truth['center']))) <= truth['diameter'] / 2
    out = {'valid': True, 'coords': coords, 'center': center.tolist(), 'hit': bool(inside),
           'error': float(np.linalg.norm(center - true_center))}
    if fmt == 'bbox':
        box = (np.asarray(coords) * np.tile(size, 2) / 1000).tolist()
        out['iou'] = box_iou(box, truth['box'])
        out['correct'] = out['iou'] >= 0.5
    else:
        out['correct'] = out['hit']
    return out


# ----------------------------------------------------------------------------- prepared prompts

@dataclass
class Prepared:
    """One tokenized prompt with every position source exposed as an editable tensor.

    ids:      (L,) input ids.
    span:     image token span {'start', 'end', 'grid': (H, W)} in merged units.
    vit_pos:  (Np, 2) ViT 2D RoPE (row, col) ids in spatial-merge-block patch order.
    interp:   ((Np, 4) indices, (Np, 4) weights) bilinear taps into the 48x48 learned table.
    pos:      (4, 1, L) decoder position ids; row 0 is the sequence index used for the mask,
              rows 1-3 are the MRoPE (t, h, w) ids.
    """
    batch: dict
    ids: torch.Tensor
    span: dict
    vit_pos: torch.Tensor
    interp: tuple[torch.Tensor, torch.Tensor]
    pos: torch.Tensor
    vis: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)

    @property
    def grid(self) -> tuple[int, int]:
        return tuple(self.span['grid'])

    @property
    def visual_index(self) -> torch.Tensor:
        return torch.arange(self.span['start'], self.span['end'])


def prepare(rt: qr.Runtime, image: Image.Image, text: str, thinking: bool = False) -> Prepared:
    batch = qr.build(rt, [image], text, thinking)
    span = qr.image_spans(rt, batch)[0]
    grid_thw = batch['image_grid_thw']
    num_side = rt.model.model.visual.num_grid_per_side
    vit_pos = get_vision_position_ids(grid_thw, MERGE)
    interp = get_vision_interpolation_indices_and_weights(grid_thw, num_side, mode='bilinear', align_corners=True,
                                                          spatial_merge_size=MERGE)
    mrope, _ = rt.model.model.get_rope_index(batch['input_ids'], mm_token_type_ids=batch['mm_token_type_ids'],
                                             image_grid_thw=grid_thw)
    L = batch['input_ids'].shape[1]
    seq = torch.arange(L, device=mrope.device).view(1, 1, L)
    pos = torch.cat([seq, mrope], 0)
    prep = Prepared(batch, batch['input_ids'][0], span, vit_pos, interp, pos)
    prep.vis = vision_embeds(rt, prep)
    return prep


@torch.inference_mode()
def vision_embeds(rt: qr.Runtime, prep: Prepared, vit_pos: torch.Tensor | None = None,
                  interp: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """Merged visual embeddings (Nv, D) with optionally replaced ViT position inputs."""
    vit_pos = prep.vit_pos if vit_pos is None else vit_pos
    interp = prep.interp if interp is None else interp
    out = rt.model.model.get_image_features(prep.batch['pixel_values'], prep.batch['image_grid_thw'],
                                            position_ids=vit_pos, interp_indices=interp[0], interp_weights=interp[1])
    return out.pooler_output[0]


def embeds(rt: qr.Runtime, prep: Prepared, vis: torch.Tensor | None = None,
           order: torch.Tensor | None = None) -> torch.Tensor:
    """inputs_embeds (1, L, D) with the visual rows written at the image span.

    order: optional (Nv,) permutation; slot i of the image span receives visual token order[i].
    """
    vis = prep.vis if vis is None else vis
    if order is not None:
        vis = vis[order.to(vis.device)]
    x = rt.model.model.get_input_embeddings()(prep.ids[None]).clone()
    x[0, prep.span['start']:prep.span['end']] = vis.to(x.dtype)
    return x


def permute_pos(prep: Prepared, pos: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Apply the same slot permutation to the MRoPE ids of the image span (row 0 stays the slot index)."""
    out = pos.clone()
    s, e = prep.span['start'], prep.span['end']
    out[1:, :, s:e] = pos[1:, :, s:e][:, :, order.to(pos.device)]
    return out


# ----------------------------------------------------------------------------- decoding

def eos_ids(rt: qr.Runtime) -> set[int]:
    e = rt.model.generation_config.eos_token_id
    return set(e if isinstance(e, list) else [e])


@torch.inference_mode()
def greedy(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, max_new_tokens: int = 64,
           keep_logits: int = 0, track_ids: list[int] | None = None) -> list[dict]:
    """Greedy decoding from explicit inputs_embeds (B, L, D) and position ids (4, B, L).

    All rows share the same length, so no padding is involved. New tokens receive position
    ids that continue from the last prompt token, as in the reference generate() path.
    keep_logits: number of leading decoding steps whose float32 logits are returned.
    track_ids: token ids whose logits are returned at every step (e.g. the ten digit tokens).
    """
    B, L, _ = x.shape
    eos = eos_ids(rt)
    out = rt.model(inputs_embeds=x, position_ids=pos, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    logits = out.logits[:, -1].float()
    nxt_mrope = pos[1:, :, -1] + 1                     # (3, B)
    tokens = [[] for _ in range(B)]
    kept = [[] for _ in range(B)]
    tracked = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=x.device)
    embed = rt.model.model.get_input_embeddings()
    for step in range(max_new_tokens):
        tok = logits.argmax(-1)
        for b in range(B):
            if not done[b]:
                tokens[b].append(int(tok[b]))
                if step < keep_logits:
                    kept[b].append(logits[b].cpu())
                if track_ids is not None:
                    tracked[b].append(logits[b, track_ids].cpu().tolist())
        done |= torch.tensor([int(t) in eos for t in tok.tolist()], device=x.device)
        if bool(done.all()) or step == max_new_tokens - 1:
            break
        p = torch.cat([torch.full((1, B), L + step, device=x.device, dtype=pos.dtype),
                       nxt_mrope + step], 0)[:, :, None]
        out = rt.model(inputs_embeds=embed(tok[:, None]), position_ids=p, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1].float()
    results = []
    for b in range(B):
        ids = tokens[b]
        finished = bool(ids) and ids[-1] in eos
        results.append({'text': rt.tokenizer.decode(ids, skip_special_tokens=True), 'ids': ids,
                        'finish_reason': 'eos' if finished else 'length',
                        **({'logits': torch.stack(kept[b])} if keep_logits else {}),
                        **({'tracked': tracked[b]} if track_ids is not None else {})})
    return results


def run_conditions(rt: qr.Runtime, xs: list[torch.Tensor], poss: list[torch.Tensor], max_new_tokens: int = 64,
                   batch_size: int = 8, keep_logits: int = 0, track_ids: list[int] | None = None) -> list[dict]:
    """Decode several conditions of the same prompt (equal length) in batches."""
    results = []
    for i in range(0, len(xs), batch_size):
        x = torch.cat(xs[i:i + batch_size], 0)
        p = torch.cat(poss[i:i + batch_size], 1)
        results.extend(greedy(rt, x, p, max_new_tokens, keep_logits, track_ids))
    return results


# ----------------------------------------------------------------------------- position-source edits

def scale_image_pos(prep: Prepared, base: torch.Tensor, ah: float, aw: float) -> torch.Tensor:
    """Scale the h / w MRoPE ids of the image tokens about the image's first id (float ids)."""
    pos = base.clone().float()
    s, e = prep.span['start'], prep.span['end']
    origin = pos[1, :, s:s + 1]                       # temporal id of the image equals its start id
    pos[2, :, s:e] = origin + ah * (pos[2, :, s:e] - origin)
    pos[3, :, s:e] = origin + aw * (pos[3, :, s:e] - origin)
    return pos


def gap_pos(prep: Prepared, gap: int) -> torch.Tensor:
    """Gap baseline: image MRoPE ids +gap; text after the image starts gap later (2*gap in total)."""
    pos = prep.pos.clone()
    s, e = prep.span['start'], prep.span['end']
    pos[1:, :, s:e] += gap
    pos[1:, :, e:] += 2 * gap
    return pos


def shift_image_pos(prep: Prepared, base: torch.Tensor, dh: float, dw: float) -> torch.Tensor:
    """Shift the h and w MRoPE ids of the image tokens only; text ids stay fixed."""
    pos = base.clone()
    s, e = prep.span['start'], prep.span['end']
    pos[2, :, s:e] += dh
    pos[3, :, s:e] += dw
    return pos


def swap_index(n: int, a: list[int], b: list[int]) -> torch.Tensor:
    """Permutation of range(n) that exchanges the paired entries of a and b."""
    if len(a) != len(b) or set(a) & set(b):
        raise ValueError('Swap regions must have equal size and be disjoint.')
    order = torch.arange(n)
    order[torch.tensor(a)] = torch.tensor(b)
    order[torch.tensor(b)] = torch.tensor(a)
    return order


def swap_image_pos(prep: Prepared, base: torch.Tensor, a: list[int], b: list[int]) -> torch.Tensor:
    """Exchange the (h, w) MRoPE ids of two equal-size token regions; embeddings and order stay."""
    pos = base.clone()
    s = prep.span['start']
    ia, ib = torch.tensor(a) + s, torch.tensor(b) + s
    pos[2:, :, ia], pos[2:, :, ib] = base[2:, :, ib], base[2:, :, ia]
    return pos


def patch_rows(tokens: list[int]) -> torch.Tensor:
    """ViT patch rows (block order) belonging to merged tokens given in raster order."""
    t = torch.tensor(tokens)
    return (t[:, None] * MERGE * MERGE + torch.arange(MERGE * MERGE)).flatten()


def swap_vit(prep: Prepared, a: list[int], b: list[int], pe: bool, rope: bool):
    """Exchange learned-PE taps and/or ViT RoPE ids between the patches of two token regions."""
    pa, pb = patch_rows(a), patch_rows(b)
    idx, w = prep.interp[0].clone(), prep.interp[1].clone()
    vp = prep.vit_pos.clone()
    if pe:
        idx[pa], idx[pb] = prep.interp[0][pb], prep.interp[0][pa]
        w[pa], w[pb] = prep.interp[1][pb], prep.interp[1][pa]
    if rope:
        vp[pa], vp[pb] = prep.vit_pos[pb], prep.vit_pos[pa]
    return vp, (idx, w)


def shifted_pe(prep: Prepared, margin: int, dr: int, dc: int, num_side: int = 48):
    """Learned-PE taps of a (Hp, Wp) window at offset (margin+dr, margin+dc) of a virtual canvas.

    The virtual canvas has (Hp + 2*margin, Wp + 2*margin) patches; the centered window (dr = dc = 0)
    is the matching baseline. Returns (indices, weights) in spatial-merge-block order.
    """
    H, W = prep.grid
    Hp, Wp = H * MERGE, W * MERGE
    if abs(dr) > margin or abs(dc) > margin:
        raise ValueError('Shift exceeds the virtual-canvas margin.')
    virt = torch.tensor([[1, Hp + 2 * margin, Wp + 2 * margin]], device=prep.interp[0].device)
    idx, w = get_vision_interpolation_indices_and_weights(virt, num_side, mode='bilinear', align_corners=True,
                                                          spatial_merge_size=1)
    idx = idx.view(Hp + 2 * margin, Wp + 2 * margin, -1)[margin + dr:margin + dr + Hp, margin + dc:margin + dc + Wp]
    w = w.view(Hp + 2 * margin, Wp + 2 * margin, -1)[margin + dr:margin + dr + Hp, margin + dc:margin + dc + Wp]
    def block(t):
        return t.reshape(H, MERGE, W, MERGE, -1).permute(0, 2, 1, 3, 4).reshape(Hp * Wp, -1)
    return block(idx), block(w).to(prep.interp[1].dtype)


# ----------------------------------------------------------------------------- attention readout

FULL_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


@dataclass
class Readout:
    """Per full-attention layer readout at selected query positions.

    attn:   layer -> (heads, Q, L) float32 softmax weights.
    contrib:layer -> (heads, Q, Nv) float32 effective contributions c (gated, projected value norms).
    check:  layer -> max abs difference between the recomputed and the actual attention output.
    """
    attn: dict
    contrib: dict
    check: dict


@torch.inference_mode()
def attention_readout(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, queries: list[int],
                      vis_slice: tuple[int, int], layers=FULL_LAYERS) -> Readout:
    """Recompute softmax attention rows of full-attention layers for the given query positions.

    The forward pass runs with the model's own attention implementation; pre-hooks capture the
    normalized hidden states and rotary tables entering each self_attn module, and q, k, v, gate are
    recomputed from them. The recomputed gated output is compared against the module output.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    store, handles = {}, []
    for layer in layers:
        attn_mod = rt.layers[layer - 1].self_attn
        def pre(module, args, kwargs, layer=layer):
            store.setdefault(layer, {})['h'] = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            store[layer]['pe'] = kwargs['position_embeddings']
        def post(module, args, kwargs, output, layer=layer):
            store[layer]['out'] = output[0]
        handles.append(attn_mod.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(attn_mod.register_forward_hook(post, with_kwargs=True))
    try:
        rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    q_idx = torch.tensor(queries, device=x.device)
    s, e = vis_slice
    attn_all, contrib_all, check = {}, {}, {}
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn
        h, (cos, sin), out = store[layer]['h'], store[layer]['pe'], store[layer]['out']
        L, hd = h.shape[1], mod.head_dim
        qg = mod.q_proj(h[:, q_idx]).view(1, len(queries), -1, hd * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)                                  # (1, Q, heads, hd)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(1, L, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(1, L, -1, hd).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, q_idx], sin[:, q_idx])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        k, v = repeat_kv(k, mod.num_key_value_groups), repeat_kv(v, mod.num_key_value_groups)
        scores = (q.float() @ k.float().transpose(2, 3)) * mod.scaling        # (1, heads, Q, L)
        mask = torch.arange(L, device=x.device)[None, :] > q_idx[:, None]
        scores = scores.masked_fill(mask, float('-inf'))
        a = scores.softmax(-1)
        g = torch.sigmoid(gate.float()).transpose(1, 2)                        # (1, heads, Q, hd)
        o = ((a @ v.float()) * g).transpose(1, 2).reshape(1, len(queries), -1)
        recon = mod.o_proj(o.to(h.dtype))
        check[layer] = float((recon[0] - out[0, q_idx]).abs().max() / out[0, q_idx].abs().max())
        # Effective contribution of visual token j through head k: a * ||W_O^k (gamma ⊙ v_j)||.
        wo = mod.o_proj.weight.float().view(-1, a.shape[1], hd)                 # (D, heads, hd)
        gram = torch.einsum('dkh,dkg->khg', wo, wo)                             # (heads, hd, hd)
        vv = v[0, :, s:e].float()                                               # (heads, Nv, hd)
        contrib = []
        for qi in range(len(queries)):
            gv = vv * g[0, :, qi, None, :]                                      # (heads, Nv, hd)
            norm = torch.einsum('knh,khg,kng->kn', gv, gram, gv).clamp_min(0).sqrt()
            contrib.append(a[0, :, qi, s:e] * norm)
        attn_all[layer] = a[0].cpu()
        contrib_all[layer] = torch.stack(contrib, 1).cpu()
    return Readout(attn_all, contrib_all, check)


def grad_attribution(rt: qr.Runtime, x: torch.Tensor, pos: torch.Tensor, targets: list[tuple[int, int]],
                     contrast_ids: list[int]) -> dict:
    """Gradient x activation at every decoder-layer input for each (query position, target token).

    z = logit(target) - mean(logit(contrast tokens other than target)) at the query position.
    Returns {'A': (n_targets, layers, L) float32} with A[t, l, j] = grad_z(x^l_j) . x^l_j.
    """
    stash, handles = [], []
    def pre(module, args, kwargs):
        hs = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
        hs.retain_grad()
        stash.append(hs)
    for layer in rt.layers:
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    out_rows = []
    try:
        with torch.enable_grad():
            xg = x.detach().clone().requires_grad_(True)
            logits = rt.model(inputs_embeds=xg, position_ids=pos, use_cache=False).logits[0]
            for t, (qpos, tok) in enumerate(targets):
                others = [c for c in contrast_ids if c != tok]
                z = logits[qpos, tok].float() - logits[qpos, others].float().mean()
                for hs in stash:
                    hs.grad = None
                z.backward(retain_graph=t < len(targets) - 1)
                out_rows.append(torch.stack([(hs.grad[0].float() * hs[0].detach().float()).sum(-1)
                                             for hs in stash]).cpu())
    finally:
        for h in handles:
            h.remove()
    return {'A': torch.stack(out_rows)}


# ----------------------------------------------------------------------------- answer token positions

def coordinate_tokens(rt: qr.Runtime, ids: list[int], fmt: str) -> list[list[int]]:
    """Indices (into the generated ids) of the digit tokens of each coordinate value, in answer order.

    point -> [x, y]; bbox -> [x_min, y_min, x_max, y_max]; rowcol -> [row, column].
    Returns an empty list when the answer does not contain the requested structure.
    """
    text = rt.tokenizer.decode(ids, skip_special_tokens=False)
    ends = [len(rt.tokenizer.decode(ids[:i + 1], skip_special_tokens=False)) for i in range(len(ids))]
    starts = [0] + ends[:-1]
    if fmt == 'rowcol':
        spans = [m.span(1) for key in ('row', 'column') for m in [re.search(rf'"{key}":\s*(\d+)', text)] if m]
        if len(spans) != 2:
            return []
    else:
        m = re.search(r'"(?:point_2d|bbox_2d)":\s*\[([^\]]*)\]', text)
        if not m:
            return []
        spans = [(m.start(1) + n.start(), m.start(1) + n.end()) for n in re.finditer(r'\d+', m.group(1))]
        if len(spans) != (2 if fmt == 'point' else 4):
            return []
    out = []
    for a, b in spans:
        toks = [i for i in range(len(ids)) if starts[i] >= a and ends[i] <= b]
        if ''.join(rt.tokenizer.decode([ids[i]]) for i in toks) != text[a:b]:
            return []
        out.append(toks)
    return out


def teacher_forced(rt: qr.Runtime, prep: Prepared, answer_ids: list[int], x: torch.Tensor | None = None,
                   pos: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """inputs_embeds and position ids of prompt + answer (the answer's last token is not needed as input)."""
    x = embeds(rt, prep) if x is None else x
    pos = prep.pos if pos is None else pos
    tail = torch.tensor(answer_ids, device=x.device)
    xa = rt.model.model.get_input_embeddings()(tail)[None]
    L, n = x.shape[1], len(answer_ids)
    step = torch.arange(1, n + 1, device=x.device, dtype=pos.dtype)
    pa = torch.cat([(L - 1 + step).view(1, 1, n), pos[1:, :, -1:] + step.view(1, 1, n)], 0)
    return torch.cat([x, xa], 1), torch.cat([pos, pa], 2)


def token_centers(grid: tuple[int, int]) -> np.ndarray:
    """(Nv, 2) centers (x, y) of merged tokens in 0-1000 units, raster order."""
    H, W = grid
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return np.stack([(c.ravel() + 0.5) / W * 1000, (r.ravel() + 0.5) / H * 1000], 1)
