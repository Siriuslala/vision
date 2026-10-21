"""Experiment 3.9 round 2: coordinate / ordinal dissociation on grid images.

Question: when Qwen3.5-9B answers a row/column (ordinal) question, does it compute a coordinate of
the target and convert it, or are coordinate output and ordinal output separate computations?

Two interventions, each applied to both a point question and a row/column question on the same image
(direct answers, thinking off, batch size 1):

* pointer steering (acts on the read at answer time): in the full-attention layers STEER_LAYERS, the
  attention logits of every query position after the image (question text and generated tokens) get
  +STEER_BIAS on the visual tokens of a neighbouring instance and -STEER_BIAS on the target's tokens.
  Visual-token processing during prefill is untouched.  Variants: neighbour to the right (or left in
  the last column) and neighbour below (or above in the last row);
* counting ablation (acts on the visual-token recurrence): Gated DeltaNet alpha = 1 at every image
  token in the DeltaNet layers among 9-16, the intervention of experiment 3.8 round 2 (E6) that
  lowers row accuracy from 0.99 to 0.39 on its 8x8 grids.

If ordinals were converted from the coordinate, both interventions would move the two answers
together.  Shifts are measured in cells relative to the unperturbed answer on the same image.
"""
from __future__ import annotations

from contextlib import contextmanager
import json

import numpy as np
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import readout_tool as rt_
from analysis_tools.position import readout_tool_run as r1run
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

STEER_LAYERS = (12, 16, 20, 24)
STEER_BIAS = 8.0
GATE_LAYERS = tuple(range(9, 17))
SUBSETS = {'t6': ('g06', 'g10'), 'layout': ('g4x8', 'g8x4', 'g20x10')}
N_PER_SUBSET = 40
SEED = 20261021


def cell_tokens(cell_box_px, canvas) -> np.ndarray:
    """Raster indices of merged visual tokens (32 px) whose cell overlaps the pixel box by >= 25%."""
    W, H = canvas
    gw, gh = W // 32, H // 32
    x0, y0, x1, y1 = cell_box_px
    out = []
    for r in range(gh):
        for c in range(gw):
            ox = max(0.0, min(x1, (c + 1) * 32) - max(x0, c * 32)) / 32
            oy = max(0.0, min(y1, (r + 1) * 32) - max(y0, r * 32)) / 32
            if ox * oy >= 0.25:
                out.append(r * gw + c)
    return np.asarray(out, int)


def neighbour(truth: dict, axis: str) -> tuple[list[float], tuple[int, int]]:
    """Pixel box of the instance one cell to the right (x) or below (y); left / above at the border."""
    g = truth['grid']
    r, c = g['target_rc']                       # 0-based
    dr, dc = (0, 1 if c < g['cols'] - 1 else -1) if axis == 'x' else (1 if r < g['rows'] - 1 else -1, 0)
    x0, y0, x1, y1 = truth['box']
    cw, ch = (g['cell'], g['cell']) if np.isscalar(g['cell']) else g['cell']
    return [x0 + dc * cw, y0 + dr * ch, x1 + dc * cw, y1 + dr * ch], (dr, dc)


@contextmanager
def steer(rt: qr.Runtime, prompt_len: int, query_start: int, plus: np.ndarray, minus: np.ndarray, vis_start: int):
    """Additive attention-logit bias in STEER_LAYERS for queries at positions >= query_start."""
    handles, count = [], {}
    for layer in STEER_LAYERS:
        mod = rt.layers[layer - 1].self_attn
        count[layer] = 0

        def pre(module, args, kwargs, layer=layer):
            h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            q = h.shape[1]
            if q == 1:                                  # decode step t (1-based) sees prompt_len + t keys
                count[layer] += 1
            k = prompt_len + count[layer] if q == 1 else q
            dt = h.dtype
            neg = torch.finfo(dt).min
            m = torch.zeros(q, k, device=h.device, dtype=torch.float32)
            if q > 1:
                m = m.masked_fill(torch.ones(q, k, dtype=torch.bool, device=h.device).triu(1), neg)
            rows = torch.arange(k - q, k, device=h.device) >= query_start
            bias = torch.zeros(k, device=h.device)
            bias[torch.as_tensor(vis_start + plus, device=h.device)] += STEER_BIAS
            bias[torch.as_tensor(vis_start + minus, device=h.device)] -= STEER_BIAS
            m[rows] = m[rows] + bias
            kwargs['attention_mask'] = m.to(dt)[None, None]
            return args, kwargs
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def generate_one(rt, batch, max_new: int) -> str:
    out = rt.model.generate(**batch, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
    return rt.tokenizer.decode(out[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)


def dissoc_cases(cases: list[dict]) -> list[dict]:
    rng = np.random.default_rng(SEED)
    out = []
    for s, subs in SUBSETS.items():
        for sub in subs:
            g = [c for c in cases if c['set'] == s and c['subset'] == sub and c['phrasing'] == 0 and 'grid' in c['truth']]
            out += [g[i] for i in sorted(rng.choice(len(g), min(N_PER_SUBSET, len(g)), replace=False))]
    return out


@torch.inference_mode()
def stage_dissoc(rt, args) -> None:
    from analysis_tools.position import readout_tool_r2_run as run
    from analysis_tools.position.sequence_order_r2 import override_gates
    path = args.output / 'dissoc.jsonl'
    done = {(r['id'], r['task'], r['condition']) for r in read_jsonl(path)}
    cases = dissoc_cases(run.load_cases(args))
    prompts = {'point': lambda c: r1run.own_prompt(dict(c, target=rt_.T6_TARGET), 'point'),
               'rowcol': lambda c: r1run.RC_QUESTION + r1run.RC_DIRECT}
    gate_layers = [l for l in GATE_LAYERS if rt.layers[l - 1].block_type == 'linear_attention']
    img_id = rt.model.config.image_token_id
    for n, c in enumerate(cases):
        image = run.load_image(c, args)
        t = c['truth']
        for task, make in prompts.items():
            batch = qr.build(rt, [image], make(c))
            ids = batch['input_ids'][0]
            vis = torch.nonzero(ids == img_id).flatten()
            vs, ve = int(vis[0]), int(vis[-1]) + 1
            L = len(ids)
            rows = []
            for cond in ('base', 'steer_x', 'steer_y', 'gate'):
                if (c['id'], task, cond) in done:
                    continue
                extra = {}
                if cond == 'base':
                    text = generate_one(rt, batch, 24)
                elif cond.startswith('steer'):
                    nb_box, d = neighbour(t, cond[-1])
                    nb = cell_tokens(nb_box, t['canvas'])
                    tgt = cell_tokens(t['box'], t['canvas'])
                    with steer(rt, L, ve, nb, tgt, vs):
                        text = generate_one(rt, batch, 24)
                    extra = {'steer_dir': list(d), 'n_plus': int(len(nb)), 'n_minus': int(len(tgt))}
                else:
                    plan = [{'layers': gate_layers, 'nodecay': np.arange(vs, ve)}]
                    with override_gates(rt, plan, L):
                        text = generate_one(rt, batch, 24)
                    extra = {'gate_layers': gate_layers}
                rows.append(dict({'id': c['id'], 'set': c['set'], 'subset': c['subset'], 'task': task, 'condition': cond,
                                  'text': text}, **extra))
            append_jsonl(path, rows)
        if n % 20 == 0:
            print(f'dissoc {n}/{len(cases)}', flush=True)


# ----------------------------------------------------------------------------- stimulus double dissociation

ORDINAL_RUN = 'behavior/2026-10-16_ordinal_mechanism'      # experiment 1.5 images and direct row/col answers


@torch.inference_mode()
def stage_ordcoord(rt, args) -> None:
    """Point answers on the experiment-1.5 grids (base, shift, insL, delL, delR; 5 conditions x 64 bases).

    shift moves the whole line of instances by one cell (pixel position +1 cell, ordinal unchanged);
    delL deletes an instance before the target (pixel position unchanged, ordinal -1); delR deletes
    one after it (neither changes).  The 1.5 run already holds the direct row/column answers.
    """
    import hashlib
    from PIL import Image
    root = args.behavior_root.parent / ORDINAL_RUN
    path = args.output / 'ordcoord.jsonl'
    done = {r['image_id'] for r in read_jsonl(path)}
    images = [r for r in read_jsonl(root / 'images.jsonl') if r['image_id'] not in done]
    prompt = r1run.own_prompt({'target': rt_.T6_TARGET}, 'point')
    for k in range(0, len(images), 12):
        b = images[k:k + 12]
        ims = []
        for r in b:
            p = root / r['path']
            if hashlib.sha256(p.read_bytes()).hexdigest() != r['sha256']:
                raise ValueError(f'hash mismatch {p}')
            ims.append(Image.open(p).convert('RGB'))
        res = qr.generate_batch(rt, [([im], prompt) for im in ims], 48)
        append_jsonl(path, [{'image_id': r['image_id'], 'text': x['text'], 'batch_size': len(b)} for r, x in zip(b, res)])
    print('ordcoord done', flush=True)
