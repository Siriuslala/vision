"""Experiment 3.16 stage drivers (base model only; results: docs/analysis/position/results_2026-11-05_readout_prior_localization.md).

Stages (comma list via --stages):
  features   internal features of every layer 1-28 at q0 (token before the first digit) and at the mean of the target
             image tokens: MLP hidden activations (input of down_proj, 12288) and token-mixer pre-output features
             (input of the Gated DeltaNet out_proj / full-attention o_proj, 4096), on the dense disc sweeps (x, y),
             the text-given coordinates (3.12 'textconv') and RefCOCO test points (same units and order as the
             3.13 round-2 base capture, whose residual streams are reused);
  transplant readout adequacy: replace the q0 (or answer-prefix) residual of an image input after layer l with the
             residual of the text-given input of the same coordinate (whole vector, or only its periodic / period-100 /
             smooth subspace, or a random subspace), and the reverse; teacher-forced digit readout;
  textsrc    where the text-given code comes from: attention rows and per-head outputs at q0 of the full-attention
             layers 12-24 on text-given and image inputs, and the residual stream at every post-image text position;
  firsttok   the first image token: residual norm ratio, attention mass it receives from image queries, DeltaNet write
             strength beta, on disc sweeps and RefCOCO;
  grad       parameter-gradient leverage map: Frobenius norm of the gradient of the second-digit (and first-digit)
             cross-entropy with respect to every linear weight, split by the token group of the module input (first
             image token, other image tokens, text tokens).
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

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import coordinate_value as cv
from analysis_tools.position import coordinate_value_r2 as c2
from analysis_tools.position import digit_failure as df
from analysis_tools.position import readout_prior as rp
from analysis_tools.position import targeted_finetune_r2 as t2
from analysis_tools.position import targeted_finetune_r2_run as tr2
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

SEED = 20261105
FEATURE_LAYERS = tuple(range(1, 29))
TRANSPLANT_LAYERS = (12, 16, 19, 20, 22, 24, 26, 28)
SUBSPACE_LAYERS = (20, 24)
TEXTSRC_ATTN_LAYERS = (12, 16, 20, 24)
TEXTSRC_RESID_LAYERS = (4, 8, 12, 14, 16, 18, 19, 20, 24)


def unit_args(args) -> Namespace:
    """Arguments of the 3.13 round-2 unit builders (disc sweeps / text-given / RefCOCO test points)."""
    return Namespace(round1=args.round1, output=args.finetune_run)


def build_rows_ids(rt: qr.Runtime, units: list[df.Unit]) -> list[dict]:
    """tr2.build_rows plus the prompt token ids ('ids', list) of every row."""
    rows = tr2.build_rows(rt, units)
    for r in rows:
        u = r['unit']
        img = (Image.open(u.source['image']).convert('RGB') if 'image' in u.source
               else cr.disc_image(tuple(u.source['center']), cv.DIAMETER, cv.CANVAS)[0])
        batch = qr.build(rt, [img], u.prompt)
        r['ids'] = batch['input_ids'][0].tolist()
        if len(r['ids']) != r['prompt_len']:
            raise ValueError(f'Prompt length mismatch for {u.id}')
    return rows


def mixer_out(layer: torch.nn.Module) -> torch.nn.Module:
    """Output projection of a decoder layer's token mixer (DeltaNet out_proj or full-attention o_proj)."""
    return layer.self_attn.o_proj if hasattr(layer, 'self_attn') else layer.linear_attn.out_proj


def site_weights(chunk: list[dict], Ltot: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """q0 indices (B,) and target-token averaging weights (B, Ltot)."""
    q0 = torch.tensor([r['q0'] for r in chunk], device=device)
    Wt = torch.zeros(len(chunk), Ltot, device=device)
    for i, r in enumerate(chunk):
        Wt[i, r['target']] = 1.0 / len(r['target'])
    return q0, Wt


# ----------------------------------------------------------------------------- features

@torch.inference_mode()
def feature_forward(rt: qr.Runtime, chunk: list[dict], layers: tuple[int, ...]) -> dict:
    """MLP hidden activations and mixer pre-output features at q0 and the target-token mean for every layer.
    Returns {'mlp': (B, 2, nL, 12288) float16, 'mix': (B, 2, nL, 4096) float16, 'logp': (B, 2, 10)}."""
    X, P = c2.pad_batch([r['x'] for r in chunk], [r['p'] for r in chunk])
    B, Ltot = X.shape[:2]
    q0, Wt = site_weights(chunk, Ltot, X.device)
    b = torch.arange(B, device=X.device)
    store = {'mlp': {}, 'mix': {}}
    handles = []

    def grab(kind, l):
        def pre(module, args):
            h = args[0]
            s = h[b.to(h.device), q0.to(h.device)].float()
            t = torch.einsum('bl,bld->bd', Wt.to(h.device), h.float())
            store[kind][l] = torch.stack([s, t], 1).half().cpu()
        return pre
    for l in layers:
        layer = rt.layers[l - 1]
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(grab('mlp', l)))
        handles.append(mixer_out(layer).register_forward_pre_hook(grab('mix', l)))
    try:
        o = tr2.light_forward(rt, chunk)
    finally:
        for h in handles:
            h.remove()
    return {'mlp': torch.stack([store['mlp'][l] for l in layers], 2).numpy(),
            'mix': torch.stack([store['mix'][l] for l in layers], 2).numpy(), 'logp': o['logp']}


def stage_features(rt: qr.Runtime, out: Path, args) -> None:
    d = Path(args.features)
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'features_base.npz'
    if path.exists():
        print(f'{path} exists', flush=True)
        return
    ua = unit_args(args)
    rows = tr2.build_rows(rt, tr2.disc_units(ua, 1, ('sweep_h', 'sweep_v')) + tr2.disc_units(ua, 2, ('textconv', 'refcoco')))
    mlp, mix, logp = [], [], []
    t0 = time.time()
    for chunk in tr2.batches(rows, args.batch):
        o = feature_forward(rt, chunk, FEATURE_LAYERS)
        mlp.append(o['mlp'])
        mix.append(o['mix'])
        logp.append(o['logp'])
        print(f'features {len(mlp) * args.batch}/{len(rows)} ({time.time() - t0:.0f}s)', flush=True)
    np.savez(path, mlp=np.concatenate(mlp), mix=np.concatenate(mix), logp=np.concatenate(logp),
             truth=np.array([r['unit'].truth for r in rows]), family=np.array([r['unit'].family for r in rows]),
             unit=np.array([r['unit'].id for r in rows]), layers=np.array(FEATURE_LAYERS))
    print(f'features: {len(rows)} rows ({time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- transplant

@torch.inference_mode()
def capture_positions(rt: qr.Runtime, chunk: list[dict], layers: tuple[int, ...], offsets: tuple[int, ...]) -> dict:
    """Residual after each layer at q0 + offset for every offset: {layer: (B, len(offsets), D) float32 cpu}, plus the
    digit log-probabilities."""
    X, P = c2.pad_batch([r['x'] for r in chunk], [r['p'] for r in chunk])
    B = X.shape[0]
    idx = torch.tensor([[r['q0'] + o for o in offsets] for r in chunk], device=X.device)
    b = torch.arange(B, device=X.device)
    store, handles = {}, []
    for l in layers:
        def hook(m, a, o, l=l):
            h = o[0] if isinstance(o, tuple) else o
            store[l] = h[b[:, None].to(h.device), idx.to(h.device)].float().cpu()
        handles.append(rt.layers[l - 1].register_forward_hook(hook))
    try:
        o = tr2.light_forward(rt, chunk)
    finally:
        for h in handles:
            h.remove()
    return {'resid': store, 'logp': o['logp']}


def replace_edit(offsets: tuple[int, ...], values: torch.Tensor):
    """Edit for tr2.light_forward: set the residual at q0 + offset of every row to values (B, len(offsets), D)."""
    def fn(h, chunk):
        h = h.clone()
        for i, r in enumerate(chunk):
            for j, o in enumerate(offsets):
                h[i, r['q0'] + o] = values[i, j].to(h.device, h.dtype)
        return h
    return fn


def stage_transplant(rt: qr.Runtime, out: Path, args) -> None:
    """Readout adequacy on the base model (x axis; text-given inputs exist for the horizontal sweep only).

    For every horizontal-sweep image (every 2nd, truths 100-999) the text-given twin states the same disc centre in
    pixels. Donor residuals are captured from one input type and written into the other after layer l:
      full_q0@l     whole residual vector at q0;
      full_pre3@l   whole residual at q0 and the two answer-prefix tokens before it (the first digit is read there and
                    relayed by the DeltaNet short convolution, 3.12 round 2);
      <sub>_q0@l    only the projection onto a subspace fitted on the text-given q0 residuals at layer l
                    (rp.code_subspaces: 'periodic' = all cos/sin pairs of PERIODS, 'p100', 'smooth'), or a random
                    subspace with the dimension of 'periodic' ('rand');
      rev_q0@l      reverse direction: the image residual written into the text-given input.
    """
    d = out / 'transplant'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'transplant.jsonl'
    done = {r['cond'] for r in read_jsonl(path)} if path.exists() else set()
    ua = unit_args(args)
    img_units = tr2.disc_units(ua, 2, ('sweep_h',))
    txt_units = {u.id[3:]: u for u in tr2.disc_units(ua, 2, ('textconv',))}
    img_units = [u for u in img_units if u.id in txt_units]
    img_rows = tr2.build_rows(rt, img_units)
    txt_rows = tr2.build_rows(rt, [txt_units[u.id] for u in img_units])
    offs = (-2, -1, 0)
    layers = tuple(sorted(set(TRANSPLANT_LAYERS)))
    don_t = {l: [] for l in layers}
    don_i = {l: [] for l in layers}
    base_lp = {'image': [], 'text': []}
    t0 = time.time()
    for ch_i, ch_t in zip(tr2.batches(img_rows, args.batch), tr2.batches(txt_rows, args.batch)):
        ci = capture_positions(rt, ch_i, layers, offs)
        ct = capture_positions(rt, ch_t, layers, offs)
        for l in layers:
            don_i[l].append(ci['resid'][l])
            don_t[l].append(ct['resid'][l])
        base_lp['image'].append(ci['logp'])
        base_lp['text'].append(ct['logp'])
    don_i = {l: torch.cat(v) for l, v in don_i.items()}
    don_t = {l: torch.cat(v) for l, v in don_t.items()}
    truth = np.array([u.truth for u in img_units])
    print(f'transplant donors ready ({time.time() - t0:.0f}s)', flush=True)

    def record(cond, lp, extra=None):
        recs = [{'cond': cond, 'unit': u.id, 'truth': u.truth, 'k': u.ks[0], 'logp0': lp[i, 0].tolist(),
                 'logp1': lp[i, 1].tolist(), **(extra or {})} for i, u in enumerate(img_units)]
        append_jsonl(path, recs)
        m = t2.digit_metrics(truth, lp[:, 0], lp[:, 1])
        print(f"{cond}: gain {m['gain']:.2f} p1 {m['p_second']:.2f} p0 {m['p_first']:.2f} err {m['value_err']:.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    for name in ('base_image', 'base_text'):
        if name not in done:
            record(name, np.concatenate(base_lp[name.split('_')[1]]))

    def run(cond, rows, edits_fn):
        if cond in done:
            return
        lps = []
        for bi, chunk in enumerate(tr2.batches(rows, args.batch)):
            sl = slice(bi * args.batch, bi * args.batch + len(chunk))
            o = tr2.light_forward(rt, chunk, edits=edits_fn(sl))
            lps.append(o['logp'])
        record(cond, np.concatenate(lps))

    for l in layers:
        run(f'full_q0@{l}', img_rows, lambda sl, l=l: [(l, replace_edit((0,), don_t[l][sl, 2:3]))])
        run(f'full_pre3@{l}', img_rows, lambda sl, l=l: [(l, replace_edit(offs, don_t[l][sl]))])
    rng = np.random.default_rng(SEED)
    subinfo = {}
    for l in SUBSPACE_LAYERS:
        Ht = don_t[l][:, 2].numpy().astype(np.float64)
        subs = rp.code_subspaces(Ht, truth)
        subs['rand'] = rp.random_orthonormal(subs['periodic'].shape[0], Ht.shape[1], rng)
        # Categorical digit codes: least-squares coefficient vectors of one-hot indicators of the tens digit s (and of
        # the first digit k); and the principal subspace of the text - image difference itself (dimension 16 / 64).
        Hi_all = don_i[l][:, 2].numpy().astype(np.float64)
        n = np.round(truth).astype(int)
        for name, cols in (('tens_onehot', [(n // 10) % 10]), ('digits_onehot', [(n // 10) % 10, n // 100])):
            X1 = np.concatenate([np.eye(10)[c][:, 1:] for c in cols] + [np.ones((len(n), 1))], 1)
            coef, *_ = np.linalg.lstsq(X1, Ht, rcond=None)
            subs[name] = rp.orthonormal_rows(coef[:-1])
        diff = Ht - Hi_all
        _, _, Vt = np.linalg.svd(diff - 0 * diff.mean(0), full_matrices=False)
        subs['diff16'], subs['diff64'] = Vt[:16], Vt[:64]
        subinfo[l] = {k: int(v.shape[0]) for k, v in subs.items()}
        for name, Pm in subs.items():
            Hi = don_i[l][:, 2].numpy().astype(np.float64)
            new = torch.tensor(rp.project_replace(Hi, Ht, Pm), dtype=torch.float32)[:, None]
            run(f'{name}_q0@{l}', img_rows, lambda sl, l=l, new=new: [(l, replace_edit((0,), new[sl]))])
        run(f'rev_q0@{l}', txt_rows, lambda sl, l=l: [(l, replace_edit((0,), don_i[l][sl, 2:3]))])
    (d / 'subspaces.json').write_text(json.dumps({str(k): v for k, v in subinfo.items()}, indent=1))
    print(f'transplant done ({time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- textsrc

@torch.inference_mode()
def attention_at(rt: qr.Runtime, row: dict, query: int, layers: tuple[int, ...]) -> dict:
    """Softmax attention rows (heads, L) and per-head output contributions (heads, D) (gated head output through its
    slice of o_proj) of the full-attention layers at one query position of one row."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    store, handles = {}, []
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn

        def pre(module, a, kw, layer=layer):
            store[layer] = {'h': kw['hidden_states'] if 'hidden_states' in kw else a[0], 'pe': kw['position_embeddings']}
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        rt.model(inputs_embeds=row['x'], position_ids=row['p'], use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    out = {}
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn
        h, (cos, sin) = store[layer]['h'], store[layer]['pe']
        L, hd = h.shape[1], mod.head_dim
        qi = torch.tensor([query], device=h.device)
        qg = mod.q_proj(h[:, qi]).view(1, 1, -1, hd * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(1, L, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(1, L, -1, hd).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, qi], sin[:, qi])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        k, v = repeat_kv(k, mod.num_key_value_groups), repeat_kv(v, mod.num_key_value_groups)
        sc = (q.float() @ k.float().transpose(2, 3)) * mod.scaling
        sc = sc.masked_fill(torch.arange(L, device=h.device)[None, :] > qi[:, None], float('-inf'))
        a = sc.softmax(-1)[0, :, 0]                                              # (heads, L)
        o = (a[:, None, :] @ v[0].float()).squeeze(1) * torch.sigmoid(gate.float())[0, 0]   # (heads, hd)
        W = mod.o_proj.weight.float().view(-1, a.shape[0], hd)                    # (D, heads, hd)
        contrib = torch.einsum('dkh,kh->kd', W, o)                               # (heads, D)
        out[layer] = {'attn': a.cpu().numpy(), 'contrib': contrib.cpu().numpy()}
    return out


def number_positions(rt: qr.Runtime, ids: list[int], start: int) -> list[list[int]]:
    """Positions (>= start) of the maximal runs of digit tokens in a prompt (e.g. the px and py numbers)."""
    dig = set(df.digit_ids(rt))
    runs, cur = [], []
    for i in range(start, len(ids)):
        if ids[i] in dig:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def stage_textsrc(rt: qr.Runtime, out: Path, args) -> None:
    d = Path(args.features)
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'textsrc.npz'
    if path.exists():
        print(f'{path} exists', flush=True)
        return
    ua = unit_args(args)
    rows = build_rows_ids(rt, tr2.disc_units(ua, 2, ('textconv', 'sweep_h')))
    t0 = time.time()
    A = {l: [] for l in TEXTSRC_ATTN_LAYERS}
    C = {l: [] for l in TEXTSRC_ATTN_LAYERS}
    groups = []              # per row: dict of position groups (relative to the full row)
    post_resid, post_len, post_ids = [], [], []
    for i, r in enumerate(rows):
        res = attention_at(rt, r, r['q0'], TEXTSRC_ATTN_LAYERS)
        s, e = r['span']
        runs = number_positions(rt, r['ids'], e)
        g = {'image': list(range(s, e)), 'target': r['target'], 'px': runs[0] if len(runs) > 0 else [],
             'py': runs[1] if len(runs) > 1 else [], 'q0': [r['q0']]}
        groups.append(g)
        for l in TEXTSRC_ATTN_LAYERS:
            a = res[l]['attn']
            A[l].append(np.stack([a[:, g[k]].sum(1) if g[k] else np.zeros(a.shape[0]) for k in
                                  ('image', 'target', 'px', 'py')] + [a[:, e:r['q0'] + 1].sum(1)], 1))  # (heads, 5)
            C[l].append(res[l]['contrib'].astype(np.float16))
        if r['unit'].family == 'textconv':
            # residual at every post-image position (prompt text + answer prefix up to q0)
            X, P = r['x'], r['p']
            store, handles = {}, []
            for l in TEXTSRC_RESID_LAYERS:
                def hook(m, a_, o, l=l):
                    h = o[0] if isinstance(o, tuple) else o
                    store[l] = h[0, e:r['q0'] + 1].float().cpu().numpy().astype(np.float16)
                handles.append(rt.layers[l - 1].register_forward_hook(hook))
            try:
                with torch.inference_mode():
                    rt.model(inputs_embeds=X, position_ids=P, use_cache=False, logits_to_keep=1)
            finally:
                for h in handles:
                    h.remove()
            post_resid.append(np.stack([store[l] for l in TEXTSRC_RESID_LAYERS], 1))   # (Lpost, nL, D)
            post_len.append(post_resid[-1].shape[0])
            post_ids.append(r['ids'][e:] + [-1] * (r['q0'] + 1 - len(r['ids'])))
        if i % 20 == 0:
            print(f'textsrc {i}/{len(rows)} ({time.time() - t0:.0f}s)', flush=True)
    np.savez(path, attn_groups={l: np.stack(v) for l, v in A.items()}, layers=np.array(TEXTSRC_ATTN_LAYERS),
             contrib=np.stack([np.stack(C[l]) for l in TEXTSRC_ATTN_LAYERS], 1),        # (N, nL, heads, D)
             truth=np.array([r['unit'].truth for r in rows]), family=np.array([r['unit'].family for r in rows]),
             unit=np.array([r['unit'].id for r in rows]), groups=np.array([json.dumps(g) for g in groups]),
             post_resid=np.concatenate(post_resid), post_len=np.array(post_len),
             post_ids=np.array([json.dumps(p) for p in post_ids]), resid_layers=np.array(TEXTSRC_RESID_LAYERS))
    print(f'textsrc: {len(rows)} rows ({time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- first image token

def stage_firsttok(rt: qr.Runtime, out: Path, args) -> None:
    """Base-model profile of the first image token: residual-norm ratio to the median image token per layer, attention
    mass it receives from every image-token query at the full-attention layers (head mean), and the Gated DeltaNet
    write strength beta = sigmoid(b) (head mean) at the first versus the other image tokens."""
    d = out / 'firsttok'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'firsttok.npz'
    if path.exists():
        print(f'{path} exists', flush=True)
        return
    ua = unit_args(args)
    units = tr2.disc_units(ua, 8, ('sweep_h',)) + tr2.disc_units(ua, 8, ('refcoco',))
    rows = tr2.build_rows(rt, units)
    full = t2.FULL_LAYERS
    t0 = time.time()
    norm_ratio, attn_first, attn_first_text, beta_first, beta_other = [], [], [], [], []
    for r in rows:
        s, e = r['span']
        store, handles = {}, []
        for l in range(1, 33):
            def hook(m, a, o, l=l):
                h = o[0] if isinstance(o, tuple) else o
                store[l] = h[0, s:e].float().norm(dim=-1).cpu().numpy()
            handles.append(rt.layers[l - 1].register_forward_hook(hook))
            if hasattr(rt.layers[l - 1], 'linear_attn'):
                def bh(m, a, o, l=l):
                    store[('beta', l)] = torch.sigmoid(o[0, s:e].float()).mean(-1).cpu().numpy()
                handles.append(rt.layers[l - 1].linear_attn.in_proj_b.register_forward_hook(bh))
        att = {}
        for l in full:
            mod = rt.layers[l - 1].self_attn

            def pre(module, a, kw, l=l):
                att[l] = {'h': kw['hidden_states'] if 'hidden_states' in kw else a[0], 'pe': kw['position_embeddings']}
            handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        try:
            with torch.inference_mode():
                rt.model(inputs_embeds=r['x'], position_ids=r['p'], use_cache=False, logits_to_keep=1)
                from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
                af, aft = [], []
                for l in full:
                    mod = rt.layers[l - 1].self_attn
                    h, (cos, sin) = att[l]['h'], att[l]['pe']
                    L, hd = h.shape[1], mod.head_dim
                    qg = mod.q_proj(h).view(1, L, -1, hd * 2)
                    q, _ = torch.chunk(qg, 2, dim=-1)
                    q = mod.q_norm(q).transpose(1, 2)
                    k = mod.k_norm(mod.k_proj(h).view(1, L, -1, hd)).transpose(1, 2)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    k = repeat_kv(k, mod.num_key_value_groups)
                    sc = (q.float() @ k.float().transpose(2, 3)) * mod.scaling
                    sc = sc.masked_fill(torch.ones(L, L, device=h.device, dtype=torch.bool).triu(1), float('-inf'))
                    a = sc.softmax(-1)[0].mean(0)                                 # (L, L) head mean
                    af.append(float(a[s + 1:e, s].mean()))                         # image queries -> first image token
                    aft.append(float(a[e:, s].mean()))                             # text queries -> first image token
                attn_first.append(af)
                attn_first_text.append(aft)
        finally:
            for h in handles:
                h.remove()
        norm_ratio.append([store[l][0] / np.median(store[l][1:]) for l in range(1, 33)])
        bl = [l for l in range(1, 33) if ('beta', l) in store]
        beta_first.append([store[('beta', l)][0] for l in bl])
        beta_other.append([float(np.median(store[('beta', l)][1:])) for l in bl])
    np.savez(path, norm_ratio=np.array(norm_ratio), attn_first=np.array(attn_first), attn_first_text=np.array(attn_first_text),
             beta_first=np.array(beta_first), beta_other=np.array(beta_other), beta_layers=np.array(bl),
             full_layers=np.array(full), family=np.array([r['unit'].family for r in rows]))
    print(f'firsttok: {len(rows)} rows ({time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- gradient leverage

def stage_grad(rt: qr.Runtime, out: Path, args) -> None:
    """Parameter-gradient leverage of every linear weight for the digit readout (base model).

    Loss: cross-entropy of the true second digit at q1 (first digit forced; 'tens') and of the true first digit at q0
    ('hund'), summed over a batch of disc-sweep (x, y) and RefCOCO rows. For a linear module with input x_t and output
    gradient delta_t, the weight gradient restricted to a token group G is G_W = sum_{t in G} delta_t x_t^T; its
    Frobenius norm is the steepest first-order loss decrease per unit Frobenius norm of a weight change applied at the
    tokens of G only, and the norm of its best rank-8 approximation is the same for a rank-8 (LoRA-like) change.
    Groups: 'first' (first image token), 'image' (other image tokens), 'text' (all text tokens incl. the answer prefix).
    """
    from analysis_tools.position import targeted_finetune as tf
    d = out / 'grad'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'grad.jsonl'
    done = {r['batch'] for r in read_jsonl(path)} if path.exists() else set()
    ua = unit_args(args)
    rows = tr2.build_rows(rt, tr2.disc_units(ua, 4, ('sweep_h', 'sweep_v', 'refcoco')))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    names = tf.target_modules(rt.layers, 'all_linear')
    for p in rt.model.parameters():
        p.requires_grad_(False)
    dig = torch.tensor(df.digit_ids(rt), device=rt.device)
    t0 = time.time()
    for bi, chunk in enumerate(tr2.batches(rows, args.grad_batch)):
        if bi in done:
            continue
        X, P = c2.pad_batch([r['x'] for r in chunk], [r['p'] for r in chunk])
        B, Ltot = X.shape[:2]
        grp = torch.zeros(B, Ltot, dtype=torch.long)                 # 0 pad, 1 first image, 2 other image, 3 text
        for i, r in enumerate(chunk):
            s, e = r['span']
            grp[i, :r['length']] = 3
            grp[i, s] = 1
            grp[i, s + 1:e] = 2
        ins, outs, handles = {}, {}, []
        for n, m in names.items():
            if not isinstance(m, torch.nn.Linear):
                continue

            def fh(mod, a, o, n=n):
                ins[n] = a[0].detach()
                o.register_hook(lambda g, n=n: outs.__setitem__(n, g.detach()))
            handles.append(m.register_forward_hook(fh))
        try:
            Xg = X.detach().clone().requires_grad_(True)
            for loss_name in ('tens', 'hund'):
                ins.clear()
                outs.clear()
                normed = {}
                hh = rt.model.model.language_model.norm.register_forward_hook(lambda m, a, o: normed.__setitem__('h', o))
                try:
                    with torch.enable_grad():
                        rt.model(inputs_embeds=Xg, position_ids=P, logits_to_keep=1)
                        h = normed['h']
                        bb = torch.arange(B, device=h.device)
                        role = 1 if loss_name == 'tens' else 0
                        qi = torch.tensor([r['q1'] if role else r['q0'] for r in chunk], device=h.device)
                        logit = rt.model.lm_head(h[bb, qi]).float()
                        tgt = torch.tensor([df.digit_ids(rt)[(int(round(r['unit'].truth)) // 10) % 10] if role else
                                            df.digit_ids(rt)[int(round(r['unit'].truth)) // 100] for r in chunk],
                                           device=h.device)
                        loss = torch.nn.functional.cross_entropy(logit, tgt, reduction='sum')
                        loss.backward()
                finally:
                    hh.remove()
                fro, r8 = {}, {}
                for n in ins:
                    if n not in outs:
                        continue
                    x_, g_ = ins[n], outs[n]
                    for gi, gname in ((1, 'first'), (2, 'image'), (3, 'text')):
                        msk = (grp == gi).to(x_.device)
                        if not msk.any():
                            continue
                        G = g_[msk].float().T @ x_[msk].float()                  # (out, in)
                        fro[f'{n}|{gname}'] = float(G.norm())
                        if gname != 'first':
                            S = torch.linalg.svdvals(G) if min(G.shape) <= 64 else torch.svd_lowrank(G, q=12, niter=2)[1]
                            r8[f'{n}|{gname}'] = float(S[:8].norm())
                append_jsonl(path, [{'batch': bi, 'loss': loss_name, 'value': float(loss), 'n': B, 'fro': fro, 'r8': r8}])
                Xg.grad = None
        finally:
            for h in handles:
                h.remove()
        print(f'grad batch {bi} ({time.time() - t0:.0f}s)', flush=True)
        if bi + 1 >= args.grad_batches:
            break


# ----------------------------------------------------------------------------- inject (idealised image-token code)

INJECT_LAYERS = (4, 8, 12, 16, 19)


def stage_inject(rt: qr.Runtime, out: Path, args) -> None:
    """Idealised upstream at the image tokens: add, to every target image token of the horizontal-sweep images after
    layer l, the periodic part of the text-given code at one text source site (args.inject_site, e.g. the token after
    the py number), fitted at the same layer on the textsrc capture: c(x) = sum_T [a_T cos(2 pi x / T) + b_T sin(...)]
    with (a_T, b_T) the least-squares coefficient vectors of rp.code_subspaces' regression (periods rp.PERIODS).
    Scales 1 and 2; control: the same code evaluated at a random permutation of x (same norm, wrong phase).
    Teacher-forced digit readout as in 'transplant'."""
    tsrc = np.load(Path(args.features) / 'textsrc.npz', allow_pickle=True)
    fam, tr = tsrc['family'], tsrc['truth']
    groups = [json.loads(g) for g in tsrc['groups']]
    st = np.flatnonzero(fam == 'textconv')
    PL = tsrc['post_len']
    offs = np.concatenate([[0], np.cumsum(PL)])
    rl = list(tsrc['resid_layers'])
    site = args.inject_site
    post = tsrc['post_resid']                               # read once (every npz key access re-reads the file)
    V = []
    for j, i in enumerate(st):
        g = groups[i]
        e = g['image'][-1] + 1
        if site.startswith('q0-'):
            pos = g['q0'][0] - int(site[3:])
        elif site == 'after_py':
            pos = g['py'][-1] + 1
        else:
            pos = g[site.split('_')[0]][-1]
        V.append(post[offs[j] + pos - e])
    V = np.stack(V).astype(np.float64)                      # (n, nL, D)
    xt = tr[st]
    d = out / 'transplant'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'inject.jsonl'
    done = {r['cond'] for r in read_jsonl(path)} if path.exists() else set()
    ua = unit_args(args)
    units = tr2.disc_units(ua, 2, ('sweep_h',))
    tsrc_ids = [str(u)[3:] for u in tsrc['unit'][st]]
    units = [u for u in units if u.id in set(tsrc_ids)]
    tsrc_units = [type('U', (), {'id': i})() for i in tsrc_ids]
    rows = tr2.build_rows(rt, units)
    truth = np.array([u.truth for u in units])
    perm = np.random.default_rng(SEED).permutation(len(truth))
    t0 = time.time()
    for l in INJECT_LAYERS:
        if l not in rl:
            continue
        H = V[:, rl.index(l)]
        cols = [rp.legendre(xt, 5)] + [rp.periodic_targets(xt, T) for T in rp.PERIODS] + [np.ones((len(xt), 1))]
        coef, *_ = np.linalg.lstsq(np.concatenate(cols, 1), H, rcond=None)
        per = coef[5:5 + 2 * len(rp.PERIODS)]                # (2 nT, D)

        def code(x):
            return np.concatenate([rp.periodic_targets(x, T) for T in rp.PERIODS], 1) @ per
        # Full replacement: every target image token takes the whole residual of the text source token of the
        # text-given twin (same disc centre), after layer l; control: the source token of a random other twin.
        twin = {u.id: j for j, u in enumerate(tsrc_units)}
        for ctrl in (False, True):
            cond = f"replace_{site}@{l}{'_perm' if ctrl else ''}"
            if cond in done:
                continue
            src = torch.tensor(np.stack([H[twin[u.id]] for u in units]), dtype=torch.float32)
            if ctrl:
                src = src[torch.tensor(perm)]
            lps = []
            for bi, chunk in enumerate(tr2.batches(rows, args.batch)):
                sl = slice(bi * args.batch, bi * args.batch + len(chunk))
                a = src[sl]

                def fn(h, chunk, a=a):
                    h = h.clone()
                    for i, r in enumerate(chunk):
                        h[i, r['target']] = a[i].to(h.device, h.dtype)
                    return h
                lps.append(tr2.light_forward(rt, chunk, edits=[(l, fn)])['logp'])
            lp = np.concatenate(lps)
            append_jsonl(path, [{'cond': cond, 'unit': u.id, 'truth': u.truth, 'k': u.ks[0], 'logp0': lp[i, 0].tolist(),
                                 'logp1': lp[i, 1].tolist()} for i, u in enumerate(units)])
            m = t2.digit_metrics(truth, lp[:, 0], lp[:, 1])
            print(f"{cond}: gain {m['gain']:.2f} p1 {m['p_second']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        for scale in (1.0, 2.0):
            for ctrl in (False, True):
                cond = f"inject_{site}@{l}x{scale:g}{'_perm' if ctrl else ''}"
                if cond in done:
                    continue
                add = torch.tensor(scale * code(truth[perm] if ctrl else truth), dtype=torch.float32)
                lps = []
                for bi, chunk in enumerate(tr2.batches(rows, args.batch)):
                    sl = slice(bi * args.batch, bi * args.batch + len(chunk))
                    a = add[sl]

                    def fn(h, chunk, a=a):
                        h = h.clone()
                        for i, r in enumerate(chunk):
                            h[i, r['target']] = h[i, r['target']] + a[i].to(h.device, h.dtype)
                        return h
                    lps.append(tr2.light_forward(rt, chunk, edits=[(l, fn)])['logp'])
                lp = np.concatenate(lps)
                append_jsonl(path, [{'cond': cond, 'unit': u.id, 'truth': u.truth, 'k': u.ks[0], 'logp0': lp[i, 0].tolist(),
                                     'logp1': lp[i, 1].tolist(), 'code_norm': float(np.linalg.norm(add[i]))}
                                    for i, u in enumerate(units)])
                m = t2.digit_metrics(truth, lp[:, 0], lp[:, 1])
                print(f"{cond}: gain {m['gain']:.2f} p1 {m['p_second']:.2f} ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- main

STAGES = {'features': stage_features, 'transplant': stage_transplant, 'textsrc': stage_textsrc,
          'firsttok': stage_firsttok, 'grad': stage_grad, 'inject': stage_inject}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--round1', required=True, help='3.6 run directory (disc sweep records and cases)')
    p.add_argument('--finetune-run', required=True, help='3.13 round-2 run directory (test.jsonl)')
    p.add_argument('--features', required=True, help='scratch directory for large feature captures')
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--grad-batch', type=int, default=6)
    p.add_argument('--grad-batches', type=int, default=12)
    p.add_argument('--inject-site', default='after_py')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in args.stages.split(',') if s]
    for s in stages:
        if s not in STAGES:
            raise ValueError(f'Unknown stage {s}')
    rt = qr.load(args.model)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['model'] = qr.describe(rt)
    info['runs'].append({'stages': stages, 'started': datetime.now(timezone.utc).isoformat(), 'batch': args.batch,
                         'grad_batch': args.grad_batch, 'grad_batches': args.grad_batches})
    meta.write_text(json.dumps(info, indent=1))
    for s in stages:
        STAGES[s](rt, out, args)


if __name__ == '__main__':
    main()
