"""Experiment 3.16 round 2 stage drivers (results: docs/analysis/position/results_2026-11-11_readout_prior_r2.md).

All stages run teacher-forced prefill passes (first digit forced to the true digit) on the 3.6 disc sweeps, their
text-given twins (3.12 'textconv') and RefCOCO points, re-using the row builders of 3.13 round 2 / 3.16.

Stages (comma list via --stages):
  capture     residual stream after every layer (0-32) at q0, q1, the target-token mean, the first image token, the
              question-token mean and the last question token; per-question-token residuals at QTOK_LAYERS (sweeps);
              attention mass of q0 / q1 per head of the full-attention layers ATTN_LAYERS on six token groups; for the
              base model and the cited 7.2 adapters (--conditions);
  mismatch    validity of the text-given reference: (a) the text-given q0 residual of a donor coordinate whose tens
              digit differs by 5 (same hundreds digit) is transplanted after layer 20 / 24 into the image input; (b) the
              Jacobian of the second-digit logits at q1 (and first-digit logits at q0) w.r.t. the q0 residual after
              layer 20 on text-given inputs, image inputs and image inputs with the mlp1_4_img adapter;
  layers      per-layer / per-module / per-token decomposition of the trained 7.2 adapters (lora.active subsets);
  steer       per-layer minimal intervention on the base model: one learned vector v in R^4096 added to the residual
              after layer l at the first image token, at every image token, or at every question token; fitted on even
              horizontal-sweep rows, evaluated on held-out rows, the vertical sweep and RefCOCO;
  grad2       parameter- and residual-gradient norms of the second-digit cross-entropy by token group (system, first
              image token, other image tokens, question, answer template) for every layer;
  vlcapture   (with --model Qwen3-VL-8B-Instruct) residual stream at q0 and the target-token mean of every layer on the
              sweeps and text-given inputs, with the digit distributions (mechanism comparison only).
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.application import coord_locate as cl
from analysis_tools.application import coord_locate_run as clr
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import coordinate_value as cv
from analysis_tools.position import coordinate_value_r2 as c2
from analysis_tools.position import digit_failure as df
from analysis_tools.position import readout_prior_run as rpr
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_r2 as t2
from analysis_tools.position import targeted_finetune_r2_run as tr2
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

SEED = 20261111
ALL_LAYERS = tuple(range(0, 33))
SITES = ('q0', 'q1', 'target', 'first', 'qmean', 'qlast')
QTOK_LAYERS = (4, 8, 12, 16, 19, 20, 24)
ATTN_LAYERS = (16, 20, 24)
ATTN_GROUPS = ('system', 'first', 'target', 'image_other', 'question', 'template')
STEER_LAYERS = {'first': (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19), 'image': (1, 2, 4, 6, 8, 12, 16, 19),
                'question': (4, 8, 12, 16, 19)}
ADAPTER_DIR_DEFAULT = 'analysis/application/2026-11-05_located_finetune'


# ----------------------------------------------------------------------------- rows and token groups

def unit_args(args) -> Namespace:
    return Namespace(round1=args.round1, output=args.finetune_run)


def add_groups(rt: qr.Runtime, rows: list[dict]) -> list[dict]:
    """Add the question span [e + 1, im_end) to rows built by rpr.build_rows_ids (e = image end, the vision-end
    token; im_end = the end-of-turn token closing the user message)."""
    im_end = rt.tokenizer.convert_tokens_to_ids('<|im_end|>')
    for r in rows:
        s, e = r['span']
        ids = r['ids']
        end = next(i for i in range(e, len(ids)) if ids[i] == im_end)
        r['question'] = list(range(e + 1, end))
        r['qlast'] = end - 1
    return rows


def build(rt: qr.Runtime, units) -> list[dict]:
    return add_groups(rt, rpr.build_rows_ids(rt, units))


def group_index(r: dict) -> dict[str, list[int]]:
    s, e = r['span']
    tgt = set(r['target'])
    return {'system': list(range(0, s)), 'first': [s], 'target': sorted(tgt),
            'image_other': [i for i in range(s + 1, e) if i not in tgt], 'question': r['question'],
            'template': list(range(r['qlast'] + 1, r['q0']))}


def load_adapter(rt: qr.Runtime, args, cond: str):
    """7.2 adapter (or None for 'base') attached without the input_ids hook; returns (lora, position kind)."""
    if cond == 'base':
        return None, None
    lora = clr.attach_trained(rt, Path(args.adapter_run), cond, with_hook=False)
    return lora, cl.condition_positions(cond)


TEXT_SUBSETS = ('system', 'question', 'template')   # text-token subsets of group_index (analysis masks only)


def set_mask(lora, kind, chunk: list[dict]) -> None:
    """pos_mask for a 7.2 position kind, or for one text-token subset ('system': tokens before the image incl. the
    vision-start token; 'question': between the image and the end of the user turn; 'template': end of turn,
    assistant header and answer prefix)."""
    if lora is None:
        return
    Ltot = max(r['length'] for r in chunk)
    if kind in TEXT_SUBSETS:
        m = np.zeros((len(chunk), Ltot), np.float32)
        for i, r in enumerate(chunk):
            g = group_index(r)
            idx = g[kind] + ([r['q0'], r['q1']] if kind == 'template' else [])
            m[i, idx] = 1
        lora.pos_mask = torch.tensor(m)
        return
    lora.pos_mask = None if kind is None else torch.tensor(np.stack([cl.span_mask(kind, r['span'], Ltot) for r in chunk]))


# ----------------------------------------------------------------------------- capture

@torch.inference_mode()
def capture_forward(rt: qr.Runtime, chunk: list[dict], qtok: bool) -> dict:
    """Residual after every layer at the six SITES ((B, 33, 6, D) float16), per-question-token residuals at
    QTOK_LAYERS ((B, nQ, Q, D) float16, rows of equal question length only), attention mass of q0 and q1 per head
    on ATTN_GROUPS at ATTN_LAYERS ((B, nA, 2, heads, groups)) and the digit log-probabilities."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    X, P = c2.pad_batch([r['x'] for r in chunk], [r['p'] for r in chunk])
    B, Ltot = X.shape[:2]
    dev = X.device
    b = torch.arange(B, device=dev)
    site_idx = torch.tensor([[r['q0'], r['q1'], 0, r['span'][0], 0, r['qlast']] for r in chunk], device=dev)
    Wt = torch.zeros(B, Ltot, device=dev)
    Wq = torch.zeros(B, Ltot, device=dev)
    for i, r in enumerate(chunk):
        Wt[i, r['target']] = 1.0 / len(r['target'])
        Wq[i, r['question']] = 1.0 / len(r['question'])
    lm = rt.model.model.language_model
    store, qstore, attn_in, handles = {}, {}, {}, []

    def grab(l, h):
        h = h[0] if isinstance(h, tuple) else h
        v = h[b[:, None].to(h.device), site_idx.to(h.device)].float()
        v[:, 2] = torch.einsum('bl,bld->bd', Wt.to(h.device), h.float())
        v[:, 4] = torch.einsum('bl,bld->bd', Wq.to(h.device), h.float())
        store[l] = v.half().cpu()
        if qtok and l in QTOK_LAYERS:
            qi = torch.tensor([r['question'] for r in chunk], device=h.device)
            qstore[l] = h[b[:, None].to(h.device), qi].half().cpu()

    handles.append(lm.layers[0].register_forward_pre_hook(
        lambda m, a, kw: grab(0, kw.get('hidden_states', a[0] if a else None)), with_kwargs=True))
    for l in ALL_LAYERS[1:]:
        handles.append(lm.layers[l - 1].register_forward_hook(lambda m, a, o, l=l: grab(l, o)))
    for l in ATTN_LAYERS:
        def pre(module, a, kw, l=l):
            attn_in[l] = (kw['hidden_states'] if 'hidden_states' in kw else a[0], kw['position_embeddings'])
        handles.append(rt.layers[l - 1].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    normed = {}
    q = torch.tensor([[r['q0'], r['q1']] for r in chunk], device=dev)
    handles.append(lm.norm.register_forward_hook(
        lambda m, a, o: normed.__setitem__('h', o[b[:, None].to(o.device), q.to(o.device)])))
    try:
        rt.model(inputs_embeds=X, position_ids=P, logits_to_keep=1)
        dig = torch.tensor(df.digit_ids(rt), device=rt.model.lm_head.weight.device)
        logp = rt.model.lm_head(normed['h'].to(rt.model.lm_head.weight.device)).float().log_softmax(-1)[..., dig]
        # Attention rows of q0 / q1 recomputed from the hooked layer inputs (gated full attention, GQA).
        groups = [group_index(r) for r in chunk]
        att = np.zeros((B, len(ATTN_LAYERS), 2, 0, len(ATTN_GROUPS)), np.float32)
        att_list = []
        for l in ATTN_LAYERS:
            mod = rt.layers[l - 1].self_attn
            h, (cos, sin) = attn_in[l]
            hd = mod.head_dim
            k = mod.k_norm(mod.k_proj(h).view(B, Ltot, -1, hd)).transpose(1, 2)
            _, k = apply_rotary_pos_emb(k, k, cos, sin)
            k = repeat_kv(k, mod.num_key_value_groups)                               # (B, H, L, hd)
            qq = mod.q_proj(h[b[:, None], q]).view(B, 2, -1, hd * 2)
            qv, _ = torch.chunk(qq, 2, dim=-1)
            qv = mod.q_norm(qv).transpose(1, 2)                                      # (B, H, 2, hd)
            cq, sq = cos[b[:, None], q] if cos.dim() == 3 else cos, sin[b[:, None], q] if sin.dim() == 3 else sin
            qv, _ = apply_rotary_pos_emb(qv, qv, cq, sq)
            sc = (qv.float() @ k.float().transpose(2, 3)) * mod.scaling               # (B, H, 2, L)
            mask = torch.arange(Ltot, device=dev)[None, None, :] > q[:, :, None]      # (B, 2, L)
            sc = sc.masked_fill(mask[:, None], float('-inf'))
            a = sc.softmax(-1)
            per = np.zeros((B, 2, a.shape[1], len(ATTN_GROUPS)), np.float32)
            for i in range(B):
                for gi, gname in enumerate(ATTN_GROUPS):
                    idx = groups[i][gname]
                    if idx:
                        per[i, :, :, gi] = a[i, :, :, idx].sum(-1).T.cpu().numpy()
            att_list.append(per)
        att = np.stack(att_list, 1)
    finally:
        for hd_ in handles:
            hd_.remove()
    out = {'resid': torch.stack([store[l] for l in ALL_LAYERS], 1).numpy(), 'logp': logp.cpu().numpy(), 'attn': att}
    if qtok:
        out['qtok'] = torch.stack([qstore[l] for l in QTOK_LAYERS], 1).numpy()
    return out


def stage_capture(rt: qr.Runtime, out: Path, args) -> None:
    d = Path(args.features) / 'capture'
    d.mkdir(parents=True, exist_ok=True)
    ua = unit_args(args)
    rows = build(rt, tr2.disc_units(ua, 1, ('sweep_h',)) + tr2.disc_units(ua, 2, ('sweep_v', 'textconv')))
    for cond in args.conditions.split(','):
        path = d / f'{cond}.npz'
        if path.exists():
            print(f'{path} exists', flush=True)
            continue
        lora, kind = load_adapter(rt, args, cond)
        res = {'resid': [], 'logp': [], 'attn': []}
        qtok = []
        t0 = time.time()
        try:
            # Sweep rows share one prompt (equal question length); text-given rows are captured without qtok.
            for fam_rows, want_q in (([r for r in rows if r['unit'].family != 'textconv'], True),
                                     ([r for r in rows if r['unit'].family == 'textconv'], False)):
                for chunk in tr2.batches(fam_rows, args.batch):
                    set_mask(lora, kind, chunk)
                    o = capture_forward(rt, chunk, want_q and cond in args.qtok_conditions.split(','))
                    for k in res:
                        res[k].append(o[k])
                    if 'qtok' in o:
                        qtok.append(o['qtok'])
        finally:
            if lora is not None:
                lora.detach()
        order = [r for r in rows if r['unit'].family != 'textconv'] + [r for r in rows if r['unit'].family == 'textconv']
        extra = {'qtok': np.concatenate(qtok)} if qtok else {}
        np.savez(path, resid=np.concatenate(res['resid']), logp=np.concatenate(res['logp']),
                 attn=np.concatenate(res['attn']), sites=np.array(SITES), layers=np.array(ALL_LAYERS),
                 qtok_layers=np.array(QTOK_LAYERS), attn_layers=np.array(ATTN_LAYERS), attn_groups=np.array(ATTN_GROUPS),
                 truth=np.array([r['unit'].truth for r in order]), family=np.array([r['unit'].family for r in order]),
                 unit=np.array([r['unit'].id for r in order]), first_pos=np.array([r['span'][0] for r in order]),
                 question_ids=np.array(order[0]['ids'])[order[0]['question']], **extra)
        print(f'capture {cond}: {len(order)} rows ({time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- mismatch / Jacobian

def donor_index(truth: np.ndarray, shift_tens: float = 5.0) -> np.ndarray:
    """For every row the row with the same hundreds digit whose in-bin offset is closest to (o + shift) mod 10."""
    n = np.asarray(truth, float)
    k = (np.round(n) // 100).astype(int)
    o = (n - 100 * k) / 10
    want = (o + shift_tens) % 10
    out = np.zeros(len(n), int)
    for i in range(len(n)):
        cand = np.where(k == k[i])[0]
        cand = cand[cand != i]
        out[i] = cand[np.argmin(np.abs(o[cand] - want[i]))]
    return out


def jacobian_rows(rt: qr.Runtime, chunk: list[dict], layer: int) -> np.ndarray:
    """(B, 2, 10, D) float16: d logit_d / d h_q0^(layer) for the first digit (at q0) and second digit (at q1)."""
    B = len(chunk)
    D = rt.model.config.text_config.hidden_size
    delta = torch.zeros(B, D, device=rt.device, requires_grad=True)
    o = tr2.light_forward(rt, chunk, grad=True, deltas={(layer, 0): delta})
    logit = o['logit']                                                              # (B, 2, 10)
    J = np.zeros((B, 2, 10, D), np.float16)
    for role in (0, 1):
        for dgt in range(10):
            g, = torch.autograd.grad(logit[:, role, dgt].sum(), delta, retain_graph=True)
            J[:, role, dgt] = g.float().cpu().numpy()
    return J


def stage_mismatch(rt: qr.Runtime, out: Path, args) -> None:
    d = out / 'mismatch'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'mismatch.jsonl'
    done = {r['cond'] for r in read_jsonl(path)} if path.exists() else set()
    ua = unit_args(args)
    img_units = tr2.disc_units(ua, 2, ('sweep_h',))
    txt_units = {u.id[3:]: u for u in tr2.disc_units(ua, 2, ('textconv',))}
    img_units = [u for u in img_units if u.id in txt_units]
    img_rows = tr2.build_rows(rt, img_units)
    txt_rows = tr2.build_rows(rt, [txt_units[u.id] for u in img_units])
    truth = np.array([u.truth for u in img_units])
    donor = donor_index(truth)
    layers = (20, 24)
    don_t = {l: [] for l in layers}
    for ch in tr2.batches(txt_rows, args.batch):
        c = rpr.capture_positions(rt, ch, layers, (0,))
        for l in layers:
            don_t[l].append(c['resid'][l])
    don_t = {l: torch.cat(v) for l, v in don_t.items()}
    t0 = time.time()
    for l in layers:
        for name, src in ((f'same_q0@{l}', np.arange(len(truth))), (f'donor_q0@{l}', donor)):
            if name in done:
                continue
            vals = don_t[l][src]
            lps = []
            for bi, chunk in enumerate(tr2.batches(img_rows, args.batch)):
                sl = slice(bi * args.batch, bi * args.batch + len(chunk))
                o = tr2.light_forward(rt, chunk, edits=[(l, rpr.replace_edit((0,), vals[sl]))])
                lps.append(o['logp'])
            lp = np.concatenate(lps)
            append_jsonl(path, [{'cond': name, 'unit': u.id, 'truth': float(truth[i]), 'donor_truth': float(truth[src[i]]),
                                 'logp0': lp[i, 0].tolist(), 'logp1': lp[i, 1].tolist()} for i, u in enumerate(img_units)])
            print(f'mismatch {name} ({time.time() - t0:.0f}s)', flush=True)
    jpath = Path(args.features) / 'jacobian_L20.npz'
    if jpath.exists():
        print(f'{jpath} exists', flush=True)
        return
    Js = {}
    for name, rows, cond in (('text', txt_rows, 'base'), ('image', img_rows, 'base'), ('image_fixed', img_rows, 'mlp1_4_img')):
        lora, kind = load_adapter(rt, args, cond)
        parts = []
        try:
            for chunk in tr2.batches(rows, args.jac_batch):
                set_mask(lora, kind, chunk)
                parts.append(jacobian_rows(rt, chunk, 20))
        finally:
            if lora is not None:
                lora.detach()
        Js[name] = np.concatenate(parts)
        print(f'jacobian {name} ({time.time() - t0:.0f}s)', flush=True)
    np.savez(jpath, truth=truth, **Js)


# ----------------------------------------------------------------------------- adapter decomposition

LAYER_SUBSETS = {'L1': (1,), 'L2': (2,), 'L3': (3,), 'L4': (4,), 'L1_2': (1, 2), 'L1_3': (1, 2, 3), 'L2_4': (2, 3, 4),
                 'L3_4': (3, 4), 'L1_4': (1, 2, 3, 4)}
MODULE_SUBSETS = {'gate_up': ('mlp.gate_proj', 'mlp.up_proj'), 'down': ('mlp.down_proj',), 'gate': ('mlp.gate_proj',),
                  'up': ('mlp.up_proj',)}


def stage_layers(rt: qr.Runtime, out: Path, args) -> None:
    d = out / 'layers'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'layers.jsonl'
    done = {(r['adapter'], r['config']) for r in read_jsonl(path)} if path.exists() else set()
    ua = unit_args(args)
    rows = build(rt, tr2.disc_units(ua, 2, ('sweep_h', 'sweep_v')))
    t0 = time.time()
    for cond in args.conditions.split(','):
        if cond == 'base':
            continue                          # the base model has no adapter to decompose
        lora, kind = load_adapter(rt, args, cond)
        configs = []
        for ln, ls in LAYER_SUBSETS.items():
            configs.append((f'layers_{ln}', {n for n in lora.names if int(n.split('.')[0][1:]) in ls}, kind))
        for mn, ms in MODULE_SUBSETS.items():
            configs.append((f'modules_{mn}', {n for n in lora.names if any(n.endswith(m) for m in ms)}, kind))
        if kind == 'image':
            configs += [('tokens_first', None, 'first'), ('tokens_nofirst', None, 'image_nofirst')]
        if kind == 'text':
            configs = [(f'tokens_{t}', None, t) for t in TEXT_SUBSETS] + configs
        try:
            for name, active, k in configs:
                if (cond, name) in done:
                    continue
                lora.active = active
                lps = []
                for chunk in tr2.batches(rows, args.batch):
                    set_mask(lora, k, chunk)
                    lps.append(tr2.light_forward(rt, chunk)['logp'])
                lp = np.concatenate(lps)
                append_jsonl(path, [{'adapter': cond, 'config': name, 'unit': r['unit'].id, 'family': r['unit'].family,
                                     'truth': r['unit'].truth, 'logp0': lp[i, 0].tolist(), 'logp1': lp[i, 1].tolist()}
                                    for i, r in enumerate(rows)])
                fam = np.array([r['unit'].family for r in rows])
                tr_ = np.array([r['unit'].truth for r in rows])
                m = t2.digit_metrics(tr_[fam == 'sweep_h'], lp[fam == 'sweep_h', 0], lp[fam == 'sweep_h', 1])
                print(f"layers {cond} {name}: gain {m['gain']:.2f} p1 {m['p_second']:.2f} ({time.time() - t0:.0f}s)",
                      flush=True)
        finally:
            lora.active = None
            lora.detach()


# ----------------------------------------------------------------------------- steer (learned single vector)

def add_edit(site: str, v: torch.Tensor):
    """Edit for tr2.light_forward: add v (D,) to the residual at the site positions of every row."""
    def fn(h, chunk):
        m = torch.zeros(h.shape[0], h.shape[1], 1, device=h.device, dtype=h.dtype)
        for i, r in enumerate(chunk):
            s, e = r['span']
            if site == 'first':
                m[i, s] = 1
            elif site == 'image':
                m[i, s:e] = 1
            elif site == 'question':
                m[i, r['question']] = 1
            else:
                raise ValueError(site)
        return h + m * v.to(h.dtype)
    return fn


@torch.inference_mode()
def site_norms(rt: qr.Runtime, chunk: list[dict]) -> dict:
    """Median residual norm of image tokens after every layer (scale of the learned vectors)."""
    o = {}
    X, P = c2.pad_batch([r['x'] for r in chunk], [r['p'] for r in chunk])
    handles = []
    for l in range(1, 33):
        def hook(m, a, out_, l=l):
            h = out_[0] if isinstance(out_, tuple) else out_
            s, e = chunk[0]['span']
            o[l] = float(h[:, s:e].float().norm(dim=-1).median())
        handles.append(rt.layers[l - 1].register_forward_hook(hook))
    try:
        rt.model(inputs_embeds=X, position_ids=P, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return o


def stage_steer(rt: qr.Runtime, out: Path, args) -> None:
    d = out / 'steer'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'steer.jsonl'
    done = {(r['site'], r['layer']) for r in read_jsonl(path)} if path.exists() else set()
    ua = unit_args(args)
    sweep = build(rt, tr2.disc_units(ua, 1, ('sweep_h',)))
    train = sweep[0::2]
    test = sweep[1::2] + build(rt, tr2.disc_units(ua, 2, ('sweep_v',)))
    rc = build(rt, tr2.disc_units(ua, 2, ('refcoco',)))
    test += rc[0::2]
    norms = site_norms(rt, train[:8])
    digit = df.digit_ids(rt)
    rng = np.random.default_rng(SEED)
    vecs = {}
    t0 = time.time()
    for site, layers in STEER_LAYERS.items():
        if site not in args.steer_sites.split(','):
            continue
        for l in layers:
            if (site, l) in done:
                continue
            scale = norms[l]
            v = torch.zeros(rt.model.config.text_config.hidden_size, device=rt.device, requires_grad=True)
            opt = torch.optim.Adam([v], lr=args.steer_lr)
            hist = []
            for step in range(args.steer_steps):
                idx = rng.choice(len(train), args.steer_batch, replace=False)
                chunk = [train[i] for i in idx]
                o = tr2.light_forward(rt, chunk, grad=True, edits=[(l, add_edit(site, v * scale))])
                lg = o['logit']
                n = np.array([int(round(r['unit'].truth)) for r in chunk])
                t0_ = torch.tensor(n // 100, device=lg.device)
                t1_ = torch.tensor((n // 10) % 10, device=lg.device)
                loss = torch.nn.functional.cross_entropy(lg[:, 0], t0_) + torch.nn.functional.cross_entropy(lg[:, 1], t1_)
                opt.zero_grad()
                loss.backward()
                opt.step()
                hist.append(float(loss))
            vfin = (v.detach() * scale)
            vecs[f'{site}@{l}'] = vfin.float().cpu().numpy()
            recs = []
            with torch.inference_mode():
                for chunk in tr2.batches(test, args.batch):
                    lp = tr2.light_forward(rt, chunk, edits=[(l, add_edit(site, vfin))])['logp']
                    recs += [{'unit': r['unit'].id, 'family': r['unit'].family, 'truth': r['unit'].truth,
                              'logp0': lp[i, 0].tolist(), 'logp1': lp[i, 1].tolist()} for i, r in enumerate(chunk)]
            summ = {}
            for fam in ('sweep_h', 'sweep_v', 'refcoco'):
                rr = [r for r in recs if r['family'] == fam]
                summ[fam] = t2.digit_metrics(np.array([r['truth'] for r in rr]), np.array([r['logp0'] for r in rr]),
                                             np.array([r['logp1'] for r in rr]))
            append_jsonl(path, [{'site': site, 'layer': l, 'scale': scale, 'v_norm': float(vfin.norm()),
                                 'v_rel': float(vfin.norm()) / scale, 'loss': hist, 'metrics': summ, 'records': recs}])
            print(f"steer {site}@{l}: loss {hist[0]:.2f}->{np.mean(hist[-5:]):.2f} |v|/med {float(vfin.norm()) / scale:.2f} "
                  f"gain h {summ['sweep_h']['gain']:.2f} v {summ['sweep_v']['gain']:.2f} p1 h {summ['sweep_h']['p_second']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            np.savez(d / f'vec_{site}_{l}.npz', v=vecs[f'{site}@{l}'])


# ----------------------------------------------------------------------------- grad2

GRAD_GROUPS = ('system', 'first', 'image', 'question', 'template')


def stage_grad2(rt: qr.Runtime, out: Path, args) -> None:
    """Second-digit cross-entropy: Frobenius norm of the weight gradient of every linear module restricted to each
    token group (as in 3.16 stage grad), and the norm of the residual gradient dL/dh_t^(l) summed over each group.
    Groups: system (before the image), first image token, other image tokens, question (between the image and the end
    of the user turn), template (end of the user turn, assistant header and answer prefix, incl. q0 and q1)."""
    d = out / 'grad2'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'grad2.jsonl'
    done = {r['batch'] for r in read_jsonl(path)} if path.exists() else set()
    ua = unit_args(args)
    rows = build(rt, tr2.disc_units(ua, 4, ('sweep_h', 'sweep_v')) + tr2.disc_units(ua, 4, ('refcoco',)))
    rng = np.random.default_rng(SEED)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    names = {n: m for n, m in tf.target_modules(rt.layers, 'all_linear').items() if isinstance(m, torch.nn.Linear)}
    t0 = time.time()
    for bi, chunk in enumerate(tr2.batches(rows, args.grad_batch)):
        if bi >= args.grad_batches:
            break
        if bi in done:
            continue
        X, P = c2.pad_batch([r['x'] for r in chunk], [r['p'] for r in chunk])
        B, Ltot = X.shape[:2]
        grp = torch.zeros(B, Ltot, dtype=torch.long)
        for i, r in enumerate(chunk):
            s, e = r['span']
            grp[i, :s] = 1
            grp[i, s] = 2
            grp[i, s + 1:e] = 3
            grp[i, r['question']] = 4
            grp[i, r['qlast'] + 1:r['length']] = 5
        ins, outs, rg, handles = {}, {}, {}, []
        for n, m in names.items():
            def fh(mod, a, o, n=n):
                ins[n] = a[0].detach()
                o.register_hook(lambda g, n=n: outs.__setitem__(n, g.detach()))
            handles.append(m.register_forward_hook(fh))
        for l in range(1, 33):
            def lh(mod, a, o, l=l):
                h = o[0] if isinstance(o, tuple) else o
                h.register_hook(lambda g, l=l: rg.__setitem__(l, g.detach()))
            handles.append(rt.layers[l - 1].register_forward_hook(lh))
        normed = {}
        handles.append(rt.model.model.language_model.norm.register_forward_hook(lambda m, a, o: normed.__setitem__('h', o)))
        try:
            Xg = X.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                rt.model(inputs_embeds=Xg, position_ids=P, logits_to_keep=1)
                h = normed['h']
                bb = torch.arange(B, device=h.device)
                qi = torch.tensor([r['q1'] for r in chunk], device=h.device)
                logit = rt.model.lm_head(h[bb, qi]).float()
                tgt = torch.tensor([df.digit_ids(rt)[(int(round(r['unit'].truth)) // 10) % 10] for r in chunk], device=h.device)
                loss = torch.nn.functional.cross_entropy(logit, tgt, reduction='sum')
                loss.backward()
        finally:
            for hd_ in handles:
                hd_.remove()
        fro, res = {}, {}
        for gi, gname in enumerate(GRAD_GROUPS, start=1):
            msk = grp == gi
            for n in ins:
                if n in outs and msk.any():
                    x_, g_ = ins[n], outs[n]
                    mm = msk.to(x_.device)
                    fro[f'{n}|{gname}'] = float((g_[mm].float().T @ x_[mm].float()).norm())
            for l, g in rg.items():
                mm = msk.to(g.device)
                res[f'L{l}|{gname}'] = float(g[mm].float().norm())
                res[f'L{l}|{gname}|n'] = int(mm.sum())
        append_jsonl(path, [{'batch': bi, 'value': float(loss), 'n': B, 'fro': fro, 'resid': res}])
        print(f'grad2 batch {bi} ({time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- Qwen3-VL capture

@torch.inference_mode()
def stage_vlcapture(rt: qr.Runtime, out: Path, args) -> None:
    """Qwen3-VL-8B: residual at q0 (token before the first digit) and the target-token mean after every decoder layer,
    and the digit distributions at q0 / q1 (first digit forced), built from input_ids (keeps the DeepStack path)."""
    d = Path(args.features)
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'qwen3vl_capture.npz'
    if path.exists():
        print(f'{path} exists', flush=True)
        return
    ua = unit_args(args)
    units = tr2.disc_units(ua, 1, ('sweep_h',)) + tr2.disc_units(ua, 2, ('sweep_v', 'textconv'))
    digit = cv.digit_token_ids(rt)
    nL = len(rt.layers)
    resid, logp, keep = [], [], []
    t0 = time.time()
    for u in units:
        img, truth = cr.disc_image(tuple(u.source['center']), cv.DIAMETER, cv.CANVAS)
        text = qr.prompt_text(rt, 1, u.prompt) + u.prefix + str(u.ks[0])
        batch = rt.processor(text=[text], images=[img], return_tensors='pt').to(rt.device)
        ids = batch['input_ids'][0]
        if int(ids[-1]) != digit[u.ks[0]] or any(int(t) in digit for t in ids[-2:-1]):
            raise ValueError(f'Unexpected answer tokenisation for {u.id}: {rt.tokenizer.decode(ids[-4:])!r}')
        span = qr.image_spans(rt, batch)[0]
        target = [span['start'] + i for i in cr.tokens_overlapping(truth['box'], span['grid'])]
        o = rt.model(**batch, output_hidden_states=True)
        hs = o.hidden_states                                                          # nL + 1 tensors (1, L, D)
        L = ids.shape[0]
        q0, q1 = L - 2, L - 1
        resid.append(np.stack([np.stack([h[0, q0].float().cpu().numpy(), h[0, target].float().mean(0).cpu().numpy()])
                               for h in hs]).astype(np.float16))
        lg = o.logits[0, [q0, q1]].float().log_softmax(-1)[:, digit]
        logp.append(lg.cpu().numpy())
        keep.append(u)
        if len(keep) % 50 == 0:
            print(f'vlcapture {len(keep)}/{len(units)} ({time.time() - t0:.0f}s)', flush=True)
    np.savez(path, resid=np.stack(resid), logp=np.stack(logp), truth=np.array([u.truth for u in keep]),
             family=np.array([u.family for u in keep]), unit=np.array([u.id for u in keep]),
             sites=np.array(['q0', 'target']), n_layers=nL)
    print(f'vlcapture done {len(keep)} ({time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- main

def limit_units(n: int) -> None:
    """Smoke-test mode: truncate tr2.disc_units to the first n units of every family."""
    orig = tr2.disc_units

    def wrapped(args, every, families):
        units = orig(args, every, families)
        out = []
        for f in families:
            out += [u for u in units if u.family == f][:n]
        return out
    tr2.disc_units = wrapped

STAGES = {'capture': stage_capture, 'mismatch': stage_mismatch, 'layers': stage_layers, 'steer': stage_steer,
          'grad2': stage_grad2, 'vlcapture': stage_vlcapture}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--finetune-run', required=True)
    p.add_argument('--adapter-run', required=True, help='7.2 run directory with adapters/<condition>.pt')
    p.add_argument('--features', required=True)
    p.add_argument('--conditions', default='base')
    p.add_argument('--qtok-conditions', default='base,mlp1_4_text')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--jac-batch', type=int, default=8)
    p.add_argument('--grad-batch', type=int, default=6)
    p.add_argument('--grad-batches', type=int, default=12)
    p.add_argument('--steer-sites', default='first,image,question')
    p.add_argument('--steer-steps', type=int, default=60)
    p.add_argument('--steer-batch', type=int, default=8)
    p.add_argument('--steer-lr', type=float, default=0.001)
    p.add_argument('--limit', type=int, default=0, help='smoke test: keep the first N units of every family')
    args = p.parse_args()
    if args.limit:
        limit_units(args.limit)
    # 7.2 conditions plus the round-2 ones (lazy import: located_r2_run imports this module).
    from analysis_tools.application import located_r2_run as lr
    lr.register_r2()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in args.stages.split(',') if s]
    for s in stages:
        if s not in STAGES:
            raise ValueError(f'Unknown stage {s}')
    rt = qr.load(args.model)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info.setdefault('models', {})[args.model] = qr.describe(rt)
    info['runs'].append({'stages': stages, 'model': args.model, 'conditions': args.conditions,
                         'started': datetime.now(timezone.utc).isoformat(), 'gpu': torch.cuda.get_device_name(0),
                         'batch': args.batch, 'steer': [args.steer_steps, args.steer_batch, args.steer_lr]})
    meta.write_text(json.dumps(info, indent=1))
    for s in stages:
        t0 = time.time()
        STAGES[s](rt, out, args)
        append_jsonl(out / 'stage_seconds.jsonl', [{'stage': s, 'model': args.model, 'seconds': time.time() - t0,
                                                    'gpu': torch.cuda.get_device_name(0),
                                                    'finished': datetime.now(timezone.utc).isoformat()}])


if __name__ == '__main__':
    main()
