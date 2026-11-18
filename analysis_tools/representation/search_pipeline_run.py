"""Experiment 2.7 round 4, Qwen3.5-9B GPU stages (outputs under --output; round-3 stimuli are read from --round3).

  prep    (CPU) same-type images (search_pipeline.render_same_type) for every odd item of the conditions used by the
          round-3 circuit selection (80 comparison-class + 40 detector-class correctly answered pairs, target cell
          >= 7); items.jsonl.
  main    pass 0  ViT MLP activations (GELU outputs, input of linear_fc2) on corrupted images (all entities of the
                  majority type) and same-type images (all entities of the target type): per condition and block the
                  mean / std of the per-entity patch maximum, the exclusive-detector score zeta and the exclusive set
                  E (zeta > 5) of every block.  zeta.npz
          pass 1  per selected pair:
                  captures  odd, corrupted, same-type, odd with L8 H6 restricted to its own cell (h6_blk), odd with all
                            mixer outputs of layers 1-15 at the target tokens taken from the same-type image (ref_all);
                            records the decoder-input change per token, L8 H6 output and its exact split into the
                            parts read from the own cell / earlier cells / text, L8 attention per cell, the L8 H6
                            key-side logit decomposition, the gate pre-activation decomposition of the round-3 top-200
                            MLP 8-15 neurons, reader-head outputs and reader value vectors per cell.   records.npz
                  interventions (reader attention on the target, top-1 cell, answer score and digit argmax):
                            ViT (exclusive neurons / random neurons / attention output / MLP output at the target
                            patches), decoder input (subsets of target tokens, neighbours, insertion into the corrupted
                            image), reference swaps (mixer outputs at the target from the same-type image), L8 H6
                            output (same-type value, zero, position mean, corrupted value, own-cell only) and reader
                            redirection (reader value vectors of the target replaced by those of another cell).
                            rows.jsonl
  probe   reader outputs at the row / column query and reader value vectors on all atlas odd images (linear probes
          for row and column).  probe.npz
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.behavior.decomposition import read_jsonl, write_jsonl
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.representation import search_asymmetry_run as sr
from analysis_tools.representation import search_circuit_analysis as A
from analysis_tools.representation import search_circuit_heads as H
from analysis_tools.representation import search_circuit_run as R
from analysis_tools.representation import search_pipeline as sp

UPTO = 15                                     # decoder layers whose components are recorded at the target
FA_MID = (4, 8, 12)
DN_LOW = tuple(l for l in range(1, 8) if l % 4)
DN_MID = tuple(l for l in range(9, 16) if l % 4)
H6 = 6
TOP3_L8 = (6, 9, 1)                            # round-3 selection (largest blocking effect at L8)
N_VIT = 27
ZETA_MIN = 5.0


# ----------------------------------------------------------------------------- items

def selection(round3: Path) -> list[str]:
    return json.loads((round3 / 'circuit' / 'selection.json').read_text())['items']


def cond_key(it: dict) -> str:
    return f"{it['set']}|{it['pair_name']}|{it['level']}|{it['param']}|{it['odd_variant']}"


def stage_prep(args) -> None:
    items = R.Items(args.round3)
    sel = selection(args.round3)
    keys = {cond_key(items.items[i]) for i in sel}
    pool = sorted(i for i, it in items.items.items() if it['role'] == 'odd' and it['set'] in ('atlas', 'mag')
                  and cond_key(it) in keys)
    rows = sp.build_same_type(args.round3, args.output, pool)
    sel_set = set(sel)
    out = []
    for r in rows:
        it = items.items[r['id']]
        out.append({'id': r['id'], 'partner': it['partner'], 'same_type': r['path'], 'same_sha256': r['sha256'],
                    'cond': cond_key(it), 'pair': it['pair_name'], 'variant': it['odd_variant'],
                    'cls': A.feature_class(it['pair_name'], it['odd_variant']), 'tcell': A.tcell(it),
                    'target': it['target'], 'selected': r['id'] in sel_set})
    write_jsonl(args.output / 'items.jsonl', out)
    print(f'[prep] {len(out)} same-type images, {sum(o["selected"] for o in out)} selected pairs', flush=True)


class Pool:
    def __init__(self, round3: Path, output: Path):
        self.r3 = R.Items(round3)
        self.output = output
        self.rows = {r['id']: r for r in read_jsonl(output / 'items.jsonl')}

    def image(self, role: str, i: str) -> Image.Image:
        if role == 'odd':
            return self.r3.image(i)
        if role == 'corr':
            return self.r3.image(self.rows[i]['partner'])
        return Image.open(self.output / self.rows[i]['same_type']).convert('RGB')


def batch_img(rt, image: Image.Image, rc):
    a, i_row, i_col = sr.answer_ids(rt, tuple(rc))
    return sr.build_rows(rt, [image], [a]), [a], i_row, i_col


# ----------------------------------------------------------------------------- ViT hooks

def visual(rt):
    return rt.model.model.visual


@contextmanager
def vit_hooks(rt, specs: list[dict], npatch: int, record: dict | None = None, rec_patches=None, entity_patches=None):
    """ViT edits per batch row (offset b * npatch) and optional recording for batch row 0.

    spec keys: 'vit_act' [(block, patch idx, neuron idx or None, values (P, k))] replaces GELU outputs (input of
    linear_fc2); 'vit_attn' / 'vit_mlp' [(block, patch idx, values (P, 1152))] replace the attention / MLP output.
    record: dict filled with 'act' / 'attn' / 'mlp' {block: (P, .)} at rec_patches and 'emax' {block: (36, 4304)}
    (per-entity patch maximum of the GELU outputs; entity_patches: list of 36 patch-index lists)."""
    vis = visual(rt)
    hs = []
    by = defaultdict(list)
    for b, s in enumerate(specs):
        for kind in ('vit_act', 'vit_attn', 'vit_mlp'):
            for e in s.get(kind, []):
                by[(kind, e[0])].append((b, *e[1:]))
    blocks = set(k for _, k in by) | (set(range(1, N_VIT + 1)) if record is not None else set())
    for k in sorted(blocks):
        blk = vis.blocks[k - 1]

        def act_pre(m, a, k=k):
            x = a[0]
            if record is not None:
                if rec_patches is not None:
                    record.setdefault('act', {})[k] = x[torch.as_tensor(rec_patches, device=x.device)].detach().clone()
                if entity_patches is not None:
                    idx = torch.as_tensor(entity_patches, device=x.device)          # (36, 16)
                    record.setdefault('emax', {})[k] = x[idx].amax(1).float()
            lst = by.get(('vit_act', k))
            if not lst:
                return None
            x = x.clone()
            for b, pidx, nidx, v in lst:
                p = torch.as_tensor(pidx, device=x.device) + b * npatch
                if nidx is None:
                    x[p] = v.to(x.device, x.dtype)
                else:
                    sub = x[p]
                    sub[:, nidx.to(x.device)] = v.to(x.device, x.dtype)
                    x[p] = sub
            return (x,)
        hs.append(blk.mlp.linear_fc2.register_forward_pre_hook(act_pre))

        def out_hook(kind):
            def f(m, a, o, k=k):
                h = o[0] if isinstance(o, tuple) else o
                if record is not None and rec_patches is not None:
                    record.setdefault(kind[4:], {})[k] = h[torch.as_tensor(rec_patches, device=h.device)].detach().clone()
                lst = by.get((kind, k))
                if not lst:
                    return o
                h = h.clone()
                for b, pidx, v in lst:
                    p = torch.as_tensor(pidx, device=h.device) + b * npatch
                    h[p] = v.to(h.device, h.dtype)
                return (h, *o[1:]) if isinstance(o, tuple) else h
            return f
        hs.append(blk.attn.register_forward_hook(out_hook('vit_attn')))
        hs.append(blk.mlp.register_forward_hook(out_hook('vit_mlp')))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


# ----------------------------------------------------------------------------- decoder capture

@contextmanager
def capture_decoder(rt, tpos: list[int], P: dict):
    """Row-0 capture: components at the target tokens (index 0 decoder input, 2l-1 mixer l, 2l MLP l, l <= UPTO),
    full inputs and position embeddings of layers 8 / 16 / 20, o_proj inputs of layers 8 / 16 / 20 (all positions)."""
    L = P['L']
    st = {'comp': {}, 'inp': {}, 'pe': {}, 'o': {}}
    pos = torch.as_tensor(tpos)
    hs = []

    def pre0(m, a, kw):
        if a[0].shape[1] == L:
            st['comp'][0] = a[0][0, pos.to(a[0].device)].detach().clone()
            st['x0_img'] = a[0][0, P['s']:P['e']].detach().clone()
    hs.append(rt.layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
    for l in range(1, UPTO + 1):
        blk = rt.layers[l - 1]
        mix = blk.linear_attn if blk.block_type == 'linear_attention' else blk.self_attn

        def hm(m, a, o, l=l):
            h = sr._out(o)
            if h.shape[1] == L:
                st['comp'][2 * l - 1] = h[0, pos.to(h.device)].detach().clone()

        def hp(m, a, o, l=l):
            if o.shape[1] == L:
                st['comp'][2 * l] = o[0, pos.to(o.device)].detach().clone()
        hs.append(mix.register_forward_hook(hm))
        hs.append(blk.mlp.register_forward_hook(hp))
        if l >= 8:
            def ha(m, a, l=l):
                if a[0].shape[1] == L:
                    st.setdefault('act', {})[l] = a[0][0, pos.to(a[0].device)].detach().clone()
            hs.append(blk.mlp.down_proj.register_forward_pre_hook(ha))
    for l in (8, 16, 20):
        def pl(m, a, kw, l=l):
            if a[0].shape[1] == L:
                st['inp'][l] = a[0][0].detach()
                st['pe'][l] = tuple(x[0] for x in kw['position_embeddings'])
        hs.append(rt.layers[l - 1].register_forward_pre_hook(pl, with_kwargs=True))

        def po(m, a, l=l):
            if a[0].shape[1] == L:
                st['o'][l] = a[0][0].detach().clone()
        hs.append(rt.layers[l - 1].self_attn.o_proj.register_forward_pre_hook(po))
    try:
        yield st
    finally:
        for h in hs:
            h.remove()


@contextmanager
def redirect_hooks(rt, specs: list[dict], L: int):
    """Reader redirection: spec key 'redir' = [(layer, head, qpos, src tokens, dst tokens)].  At query qpos the head's
    output (o_proj input slice, after the sigmoid output gate) gains sigma(g) * sum_i a(q, src_i) (v(dst_i) - v(src_i)):
    exactly the output the head would produce if the value vectors of the source tokens were those of the
    destination tokens while its attention pattern stays unchanged."""
    by = defaultdict(list)
    for b, s in enumerate(specs):
        for e in s.get('redir', []):
            by[e[0]].append((b, *e[1:]))
    hs = []
    for l, lst in by.items():
        att = rt.layers[l - 1].self_attn
        stash = {}

        def pl(m, a, kw, stash=stash):
            if a[0].shape[1] == L:
                stash['x'] = a[0]
                stash['pe'] = kw['position_embeddings']
        hs.append(rt.layers[l - 1].register_forward_pre_hook(pl, with_kwargs=True))

        def po(m, a, l=l, lst=lst, stash=stash, att=att):
            x = a[0]
            if x.shape[1] != L:
                return None
            x = x.clone()
            hd = att.head_dim
            blk = rt.layers[l - 1]
            rows = defaultdict(list)
            for b, h, qpos, src, dst in lst:
                rows[b].append((h, qpos, tuple(src), tuple(dst)))
            for b, ent in rows.items():
                xb = stash['x'][b]
                cos, sin = (t[b] for t in stash['pe'])
                qs = sorted({q for _, q, _, _ in ent})
                qi = {q: n for n, q in enumerate(qs)}
                pr = R.attention_probs(rt, l, xb, cos, sin, qs)                      # (H, nq, L)
                xn = blk.input_layernorm(xb)
                v = att.v_proj(xn).view(L, -1, hd).float()                          # (L, KV, hd)
                g = torch.sigmoid(att.q_proj(xn[qs]).view(len(qs), -1, hd * 2)[..., hd:].float())   # (nq, H, hd)
                for h, qpos, src, dst in ent:
                    kv = h // att.num_key_value_groups
                    st_, dt_ = torch.as_tensor(src, device=x.device), torch.as_tensor(dst, device=x.device)
                    w = pr[h, qi[qpos]][st_]
                    corr = (w[:, None] * (v[dt_, kv] - v[st_, kv])).sum(0) * g[qi[qpos], h]
                    x[b, qpos, h * hd:(h + 1) * hd] += corr.to(x.dtype)
            return (x,)
        hs.append(att.o_proj.register_forward_pre_hook(po))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@contextmanager
def edits(rt, specs: list[dict], groups, P, npatch: int):
    """All intervention kinds for one batch (spec per row; 'tpos' is required by search_circuit_heads.apply_edits)."""
    L = P['L']
    comp_rows = [(b, s['comp_pos'], s['comp']) for b, s in enumerate(specs) if s.get('comp')]
    with ExitStack() as es:
        es.enter_context(vit_hooks(rt, specs, npatch))
        if comp_rows:
            es.enter_context(R.replace_components(rt, comp_rows, L))
        es.enter_context(H.apply_edits(rt, specs, groups, L))
        es.enter_context(redirect_hooks(rt, specs, L))
        yield


def run_specs(rt, base, answers, i_row, i_col, P, groups, tcell, ecell, specs, npatch, batch):
    """Teacher-forced forwards with edits; readout per row."""
    out = []
    for a in range(0, len(specs), batch):
        chunk = specs[a:a + batch]
        b = R.repeat_batch(base, len(chunk))
        with edits(rt, chunk, groups, P, npatch), R.capture_layers(rt, (16, 20), 0, None) as st2:
            logits = rt.model(**b, use_cache=False, logits_to_keep=len(answers[0]) + 1).logits
        sc_ = sr.score_rows(logits, answers * len(chunk), i_row, i_col, len(answers[0]), sr.digit_ids(rt, 6))
        for r in range(len(chunk)):
            rd = R.reader_readout(rt, st2, r, groups, P['q_row'])           # (6, 36)
            rdc = R.reader_readout(rt, st2, r, groups, P['q_col'])
            out.append({'reader': float(rd[:, tcell].mean()), 'reader_col': float(rdc[:, tcell].mean()),
                        'reader_e': float(rd[:, ecell].mean()), 'top1': bool(np.argmax(rd.sum(0)) == tcell),
                        's': sc_[r]['s'], 'arg_row': sc_[r]['arg_row'], 'arg_col': sc_[r]['arg_col']})
    return out


# ----------------------------------------------------------------------------- analysis helpers on captures

def _probs(rt, layer, x, cos, sin, tpos, groups, tcell, blocked: bool):
    """Attention probabilities (H, T, L) of the target queries; with blocked=True every image token of another cell
    is removed and the rest renormalised (the cross-cell block of search_circuit_heads.apply_edits)."""
    pr = R.attention_probs(rt, layer, x, cos, sin, tpos)
    if blocked:
        others = torch.as_tensor([t for i, g in enumerate(groups) if i != tcell for t in g], device=pr.device)
        pr = pr.clone()
        pr[:, :, others] = 0
        pr = pr / pr.sum(-1, keepdim=True)
    return pr


def head_parts(rt, st: dict, layer: int, head: int, tpos, groups, tcell, s_img: int, blocked: bool = False):
    """Exact split of one head's output (after W_O) at the target tokens into the parts read from the own cell,
    earlier cells and all other positions (text prefix); also the earlier-cell uniform mean and attention per cell.
    Returns dict of float32 tensors (mean over target tokens)."""
    blk = rt.layers[layer - 1]
    att = blk.self_attn
    hd = att.head_dim
    x = st['inp'][layer]
    cos, sin = st['pe'][layer]
    L = x.shape[0]
    pr = _probs(rt, layer, x, cos, sin, tpos, groups, tcell, blocked)       # (H, T, L)
    xn = blk.input_layernorm(x)
    kv = head // att.num_key_value_groups
    v = att.v_proj(xn).view(L, -1, hd)[:, kv].float()                       # (L, hd)
    g = torch.sigmoid(att.q_proj(xn[tpos]).view(len(tpos), -1, hd * 2)[:, head, hd:].float())   # (T, hd)
    Wo = att.o_proj.weight.float()[:, head * hd:(head + 1) * hd]            # (d, hd)
    own = groups[tcell]
    earlier = [t for gg in groups[:tcell] for t in gg]
    rest = sorted(set(range(L)) - set(own) - set(earlier))
    p = pr[head]                                                            # (T, L)

    def part(idx):
        idx_t = torch.as_tensor(idx, device=x.device)
        return ((p[:, idx_t] @ v[idx_t]) * g) @ Wo.T                        # (T, d)
    out = {'own': part(own), 'earlier': part(earlier), 'rest': part(rest)}
    out['total'] = out['own'] + out['earlier'] + out['rest']
    e_t = torch.as_tensor(earlier, device=x.device)
    out['earlier_uniform'] = (v[e_t].mean(0)[None] * g) @ Wo.T
    out['own_uniform'] = (v[torch.as_tensor(own, device=x.device)].mean(0)[None] * g) @ Wo.T
    cells = torch.stack([pr[:, :, torch.as_tensor(gg, device=x.device)].sum(-1) for gg in groups], -1)  # (H, T, 36)
    text = pr[:, :, :s_img].sum(-1, keepdim=True)
    out = {k: v_.mean(0) for k, v_ in out.items()}
    out['mass'] = torch.cat([cells, text], -1).mean(1)                       # (H, 37)
    return out


def l8_key_decomposition(rt, st_comp_img: torch.Tensor, x8: torch.Tensor, pe8, tpos, groups, s_img, e_img):
    """Additive decomposition of L8 H6 logits from the target queries over the components (decoder input, layers
    1-7) of every image token; returns (n_comp, 36) cell means and total logits with / without RoPE (36,)."""
    cos, sin = pe8
    img = slice(s_img, e_img)
    contrib, actual = R.logit_attribution(rt, 8, x8, cos, sin, tpos, st_comp_img, img, [H6])
    c = contrib[0].mean(0)                                                  # (n_comp, n_img)
    rel = [[t - s_img for t in gg] for gg in groups]
    cell = torch.stack([c[:, g].mean(-1) for g in rel], -1)
    act = actual[0].mean(0)
    cell_act = torch.stack([act[g].mean() for g in rel])
    # no-RoPE logits: queries and keys without rotation (content only)
    blk = rt.layers[7]
    att = blk.self_attn
    hd = att.head_dim
    xn = blk.input_layernorm(x8)
    q = att.q_norm(att.q_proj(xn[tpos]).view(len(tpos), -1, hd * 2)[..., :hd])[:, H6].float()
    k = att.k_norm(att.k_proj(xn[img]).view(e_img - s_img, -1, hd))[:, H6 // att.num_key_value_groups].float()
    nr = (q @ k.T).mean(0) * att.scaling
    cell_nr = torch.stack([nr[g].mean() for g in rel])
    return cell.float(), cell_act.float(), cell_nr.float(), q.mean(0).float()


def neuron_decomposition(rt, comps: dict, h6parts: torch.Tensor, a8_h6_total: torch.Tensor, nrn: dict[int, torch.Tensor]):
    """Gate pre-activation of selected MLP neurons at the target tokens, decomposed additively over residual
    components with the post-attention RMSNorm scale frozen at its actual value.

    comps: {index: (T, d)} (0 decoder input, 2l-1 mixer l, 2l MLP l); h6parts: (3, T, d) L8 H6 own / earlier / rest
    parts; nrn: {layer: neuron indices}.  Component order of the output: [input, mixer 1..15 (with L8 replaced by
    'L8 other heads'), MLP 1..14, H6 own, H6 earlier, H6 rest] = 1 + 15 + 14 + 3 = 33.
    Returns {layer: dict(contrib (33, T, k), gate (T, k), up (T, k), act (T, k))}."""
    out = {}
    for l, idx in nrn.items():
        blk = rt.layers[l - 1]
        names = [0] + [2 * m - 1 for m in range(1, l + 1)] + [2 * m for m in range(1, l)]
        res = sum(comps[c].float() for c in names)                           # residual before MLP l (T, d)
        r = torch.rsqrt(res.pow(2).mean(-1, keepdim=True) + blk.post_attention_layernorm.eps)
        w = 1.0 + blk.post_attention_layernorm.weight.float()
        Wg = blk.mlp.gate_proj.weight.float()[idx.to(res.device)]          # (k, d)
        Wu = blk.mlp.up_proj.weight.float()[idx.to(res.device)]
        stack = [comps[0].float()]
        for m in range(1, UPTO + 1):
            if m > l:
                stack.append(torch.zeros_like(comps[0], dtype=torch.float32))
            elif m == 8:
                stack.append(comps[15].float() - a8_h6_total)
            else:
                stack.append(comps[2 * m - 1].float())
        for m in range(1, UPTO):
            stack.append(comps[2 * m].float() if m < l else torch.zeros_like(comps[0], dtype=torch.float32))
        for j in range(3):
            stack.append(h6parts[j] if l >= 8 else torch.zeros_like(h6parts[j]))
        S = torch.stack(stack) * r[None] * w                                 # (33, T, d)
        contrib = S @ Wg.T                                                   # (33, T, k)
        xn = res * r * w
        gate, up = xn @ Wg.T, xn @ Wu.T
        out[l] = {'contrib': contrib, 'gate': gate, 'up': up, 'act': torch.nn.functional.silu(gate) * up}
    return out


# ----------------------------------------------------------------------------- stage main

def vit_stats(rt, pool: Pool, args) -> dict:
    """Pass 0: per condition, mean / std of the per-entity patch maximum of every ViT GELU output on
    majority-type (corrupted) and target-type (same-type) homogeneous images."""
    acc = {}
    ids = sorted(pool.rows)
    meta0 = pool.r3.items[ids[0]]['meta']
    toks = [g for g in sp.sc.entity_groups(meta0)]
    ent_p = [sp.vit_patches_of_tokens(g) for g in toks]
    for role in ('corr', 'same'):
        for a in range(0, len(ids), args.batch):
            chunk = ids[a:a + args.batch]
            imgs = [pool.image(role, i) for i in chunk]
            batch = rt.processor(text=[qr.prompt_text(rt, 1, sr.sa.DIRECT_PROMPT)] * len(imgs), images=imgs, return_tensors='pt').to(rt.device)
            npatch = int(batch['image_grid_thw'][0].prod())
            rec = {}
            ent_all = [[p + b * npatch for p in e] for b in range(len(chunk)) for e in ent_p]
            with vit_hooks(rt, [{}], npatch, record=rec, entity_patches=ent_all):
                visual(rt)(batch['pixel_values'].to(visual(rt).dtype), grid_thw=batch['image_grid_thw'])
            for k in range(1, N_VIT + 1):
                em = rec['emax'][k].view(len(chunk), 36, -1)
                for b, i in enumerate(chunk):
                    key = (pool.rows[i]['cond'], role)
                    d = acc.setdefault(key, {'n': 0, 's': torch.zeros(N_VIT, em.shape[-1], device=em.device),
                                             'ss': torch.zeros(N_VIT, em.shape[-1], device=em.device)})
                    d['s'][k - 1] += em[b].sum(0)
                    d['ss'][k - 1] += em[b].pow(2).sum(0)
                    if k == 1:
                        d['n'] += 36
        print(f'[main] vit stats {role} done', flush=True)
    out = {}
    for cond in sorted({c for c, _ in acc}):
        mt = acc[(cond, 'same')]
        mm = acc[(cond, 'corr')]
        mu_t = (mt['s'] / mt['n']).cpu().numpy()
        mu_m = (mm['s'] / mm['n']).cpu().numpy()
        sd_m = np.sqrt(np.maximum((mm['ss'] / mm['n']).cpu().numpy() - mu_m ** 2, 0))
        sd_t = np.sqrt(np.maximum((mt['ss'] / mt['n']).cpu().numpy() - mu_t ** 2, 0))
        out[cond] = {'mu_t': mu_t, 'mu_m': mu_m, 'sd_m': sd_m, 'sd_t': sd_t, 'zeta': sp.zeta(mu_t, mu_m, sd_m),
                     'n_t': mt['n'], 'n_m': mm['n']}
    return out


@torch.inference_mode()
def stage_main(rt, args) -> None:
    pool = Pool(args.round3, args.output)
    out_dir = args.output
    zpath = out_dir / 'zeta.npz'
    if zpath.exists():
        z = np.load(zpath, allow_pickle=True)
        stats = z['stats'].item()
    else:
        stats = vit_stats(rt, pool, args)
        np.savez_compressed(zpath, stats=np.asarray(stats, dtype=object))
    excl = {c: {k: np.where(s['zeta'][k - 1] > ZETA_MIN)[0] for k in range(1, N_VIT + 1)} for c, s in stats.items()}
    sel = [i for i in sorted(pool.rows) if pool.rows[i]['selected']]
    if args.limit:
        sel = sel[:args.limit]
    selection = json.loads((args.round3 / 'circuit' / 'selection.json').read_text())
    top200 = selection['neurons_top']['200']
    nrn = defaultdict(list)
    for l, i in top200:
        nrn[int(l)].append(int(i))
    nrn = {l: torch.as_tensor(sorted(v)) for l, v in nrn.items()}
    rows_path = out_dir / 'rows.jsonl'
    done = {r['id'] for r in read_jsonl(rows_path)} if rows_path.exists() else set()
    rec_path = out_dir / 'records'
    rec_path.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for n, i in enumerate(sel):
        if i in done and (rec_path / f'{n:03d}.npz').exists():
            continue
        row = pool.rows[i]
        it = pool.r3.items[i]
        tcell = row['tcell']
        ecell = sp.redirect_cell(tcell)
        rc = tuple(row['target'])
        imgs = {r: pool.image(r, i) for r in ('odd', 'corr', 'same')}
        base, answers, i_row, i_col = batch_img(rt, imgs['odd'], rc)
        P = R.positions(rt, base, len(answers[0]), i_row, i_col)
        groups = R.cell_tokens(it['meta'], P['s'])
        tpos = groups[tcell]
        epos = groups[ecell]
        npatch = int(base['image_grid_thw'][0].prod())
        tpat = sp.vit_patches_of_tokens([t - P['s'] for t in tpos])
        ent_p = [sp.vit_patches_of_tokens([t - P['s'] for t in g]) for g in groups]
        bases = {'odd': base}
        for r in ('corr', 'same'):
            bases[r] = batch_img(rt, imgs[r], rc)[0]
        cap, vit = {}, {}

        def forward(role, spec=None):
            rec = {}
            spec = spec or {}
            with ExitStack() as es:
                es.enter_context(vit_hooks(rt, [spec], npatch, record=rec, rec_patches=tpat, entity_patches=ent_p))
                if spec.get('comp'):
                    es.enter_context(R.replace_components(rt, [(0, spec['comp_pos'], spec['comp'])], P['L']))
                es.enter_context(H.apply_edits(rt, [{'tpos': tpos, **spec}], groups, P['L']))
                st = es.enter_context(capture_decoder(rt, tpos, P))
                rt.model(**bases[role], use_cache=False, logits_to_keep=1)
            return st, rec
        for role in ('odd', 'corr', 'same'):
            cap[role], vit[role] = forward(role)
        cap['blk'], _ = forward('odd', {'blk': [(8, H6)]})
        ref_vals = {c: cap['same']['comp'][c] for c in range(1, 2 * UPTO, 2)}
        cap['ref'], _ = forward('odd', {'comp': ref_vals, 'comp_pos': tpos})
        # ---------------------------------------------------------------- records
        recd = {}
        dx0 = (cap['odd']['x0_img'].float() - cap['corr']['x0_img'].float())
        recd['x0_dnorm'] = dx0.norm(dim=-1).cpu().numpy()
        recd['x0_norm'] = cap['corr']['x0_img'].float().norm(dim=-1).cpu().numpy()
        tok_rel = [t - P['s'] for t in tpos]
        order = np.argsort(-recd['x0_dnorm'][tok_rel])
        top1 = [tpos[order[0]]]
        rest3 = [tpos[j] for j in order[1:]]
        recd['vit_tmax'] = np.stack([torch.stack([vit[r]['emax'][k][tcell] for k in range(1, N_VIT + 1)]).cpu().numpy()
                                     for r in ('odd', 'corr', 'same')]).astype(np.float16)
        parts = {}
        for role in ('odd', 'corr', 'same', 'blk'):
            parts[role] = head_parts(rt, cap[role], 8, H6, tpos, groups, tcell, P['s'], blocked=(role == 'blk'))
        parts['ref'] = parts['same']
        recd['h6'] = np.stack([parts[r]['total'].cpu().numpy() for r in ('odd', 'corr', 'same', 'blk', 'ref')]).astype(np.float32)
        recd['h6_parts'] = np.stack([np.stack([parts[r][k].cpu().numpy() for k in ('own', 'earlier', 'rest', 'earlier_uniform', 'own_uniform')])
                                     for r in ('odd', 'corr', 'same')]).astype(np.float32)
        recd['l8_mass'] = np.stack([parts[r]['mass'].cpu().numpy() for r in ('odd', 'corr', 'same')]).astype(np.float32)
        # L8 head outputs (all heads, W_O applied, mean over target tokens) for odd / corr / same
        Wo8 = rt.layers[7].self_attn.o_proj.weight.float()
        recd['l8_heads'] = np.stack([torch.stack([cap[r]['o'][8][tpos][:, h * 256:(h + 1) * 256].float().mean(0) @ Wo8[:, h * 256:(h + 1) * 256].T
                                                  for h in range(16)]).cpu().numpy() for r in ('odd', 'corr', 'same')]).astype(np.float16)
        # key-side decomposition of L8 H6 (needs components at all image tokens for layers 1-7)
        kd = []
        for role in ('odd', 'corr', 'same'):
            with R.capture_layers(rt, (8,), 7, slice(P['s'], P['e'])) as stc:
                rt.model(**bases[role], use_cache=False, logits_to_keep=1)
            comps = torch.stack([stc['comp'][c][0] for c in range(15)])
            cell, cact, cnr, q = l8_key_decomposition(rt, comps, stc['inp'][8][0], tuple(x[0] for x in stc['pe'][8]),
                                                      tpos, groups, P['s'], P['e'])
            kd.append((cell.cpu().numpy(), cact.cpu().numpy(), cnr.cpu().numpy(), q.cpu().numpy()))
        recd['l8k_comp'] = np.stack([k[0] for k in kd]).astype(np.float32)
        recd['l8k_logit'] = np.stack([k[1] for k in kd]).astype(np.float32)
        recd['l8k_norope'] = np.stack([k[2] for k in kd]).astype(np.float32)
        recd['l8_q'] = np.stack([k[3] for k in kd]).astype(np.float32)
        # neuron gate decomposition (odd, corr, same, blk, ref)
        nd = {}
        for role in ('odd', 'corr', 'same', 'blk', 'ref'):
            src = 'same' if role == 'ref' else role      # in ref_all the L8 output at the target is the same-type value
            hp = head_parts_tokens(rt, cap[src], 8, H6, tpos, groups, tcell, blocked=(role == 'blk'))
            nd[role] = neuron_decomposition(rt, cap[role]['comp'], hp[:3], hp[3], nrn)
        recd['nrn_check'] = np.asarray([max(float((nd[r][l]['act'] - cap[r]['act'][l][:, nrn[l].to(cap[r]['act'][l].device)].float()).abs().max())
                                            for l in nrn) for r in ('odd', 'corr', 'same', 'blk')])
        for key in ('contrib', 'gate', 'up', 'act'):
            recd['nrn_' + key] = np.stack([np.concatenate([nd[r][l][key].cpu().numpy() for l in sorted(nrn)], -1)
                                           for r in ('odd', 'corr', 'same', 'blk', 'ref')]).astype(np.float16)
        # readers: outputs at the row / column query (o_proj input slices) and value vectors per cell
        ro, rv = [], []
        for role in ('odd', 'corr'):
            o = []
            for l, h in R.READERS:
                o.append(torch.stack([cap[role]['o'][l][q, h * 256:(h + 1) * 256] for q in (P['q_row'], P['q_col'])]))
            ro.append(torch.stack(o).float().cpu().numpy())
        for l, h in R.READERS:
            blk = rt.layers[l - 1]
            att = blk.self_attn
            xn = blk.input_layernorm(cap['odd']['inp'][l])
            v = att.v_proj(xn).view(xn.shape[0], -1, 256)[:, h // att.num_key_value_groups].float()
            rv.append(torch.stack([v[g].mean(0) for g in groups]).cpu().numpy())
        recd['reader_out'] = np.stack(ro).astype(np.float16)
        recd['reader_v'] = np.stack(rv).astype(np.float16)
        recd['tok_top1'] = np.asarray(top1)
        np.savez_compressed(rec_path / f'{n:03d}.npz', id=i, **recd)
        # ---------------------------------------------------------------- interventions
        specs, names, roles = [], [], []

        def add(name, spec, role='odd'):
            specs.append({'tpos': tpos, **spec}); names.append(name); roles.append(role)
        add('none', {})
        E = excl[row['cond']]
        act_corr = vit['corr']['act']
        ve, vr = [], []
        for k in range(1, N_VIT + 1):
            idx = E[k]
            if len(idx):
                it_ = torch.as_tensor(idx)
                ve.append((k, tpat, it_, act_corr[k][:, it_.to(act_corr[k].device)]))
                ri = torch.as_tensor(np.sort(rng.choice(np.setdiff1d(np.arange(4304), idx), size=len(idx), replace=False)))
                vr.append((k, tpat, ri, act_corr[k][:, ri.to(act_corr[k].device)]))
        add('vit_excl', {'vit_act': ve})
        add('vit_rand', {'vit_act': vr})
        add('vit_attn_same', {'vit_attn': [(k, tpat, vit['same']['attn'][k]) for k in range(1, N_VIT + 1)]})
        add('vit_attn_corr', {'vit_attn': [(k, tpat, vit['corr']['attn'][k]) for k in range(1, N_VIT + 1)]})
        add('vit_mlp_corr', {'vit_mlp': [(k, tpat, vit['corr']['mlp'][k]) for k in range(1, N_VIT + 1)]})
        bands = {'1_9': range(1, 10), '10_18': range(10, 19), '19_27': range(19, 28)}
        for bn, ks in bands.items():
            add(f'vit_mlp_corr_{bn}', {'vit_mlp': [(k, tpat, vit['corr']['mlp'][k]) for k in ks]})
            add(f'vit_attn_same_{bn}', {'vit_attn': [(k, tpat, vit['same']['attn'][k]) for k in ks]})
        # neurons ranked by the difference of homogeneous means (target type minus majority type), all blocks
        st_c = stats[row['cond']]
        for tag, sign, kk in (('up500', 1, 500), ('down500', -1, 500), ('up_all', 1, None), ('down_all', -1, None)):
            lst = []
            for k in range(1, N_VIT + 1):
                dmu = sign * (st_c['mu_t'][k - 1] - st_c['mu_m'][k - 1])
                idx = np.argsort(-dmu)[:kk] if kk else np.where(dmu > 0)[0]
                idx = idx[dmu[idx] > 0]
                if len(idx):
                    it_ = torch.as_tensor(np.sort(idx))
                    lst.append((k, tpat, it_, act_corr[k][:, it_.to(act_corr[k].device)]))
            add(f'vit_{tag}', {'vit_act': lst})
        x0c = cap['corr']['x0_img']
        x0o = cap['odd']['x0_img']
        rel = lambda ps: [p - P['s'] for p in ps]
        add('x0_all', {'comp': {0: x0c[rel(tpos)]}, 'comp_pos': tpos})
        add('x0_top1', {'comp': {0: x0c[rel(top1)]}, 'comp_pos': top1})
        add('x0_rest3', {'comp': {0: x0c[rel(rest3)]}, 'comp_pos': rest3})
        neigh = [p for g in groups for p in g if p not in tpos]
        add('x0_neigh', {'comp': {0: x0c[rel(neigh)]}, 'comp_pos': neigh})
        add('corr_none', {}, 'corr')
        add('ins_x0_all', {'comp': {0: x0o[rel(tpos)]}, 'comp_pos': tpos}, 'corr')
        add('ins_x0_top1', {'comp': {0: x0o[rel(top1)]}, 'comp_pos': top1}, 'corr')
        same_c = cap['same']['comp']
        mix = lambda ls: {2 * l - 1: same_c[2 * l - 1] for l in ls}
        add('ref_all', {'comp': mix(range(1, UPTO + 1)), 'comp_pos': tpos})
        add('ref_fa', {'comp': mix(FA_MID), 'comp_pos': tpos})
        add('ref_dn', {'comp': mix(DN_LOW + DN_MID), 'comp_pos': tpos})
        add('ref_dn1_7', {'comp': mix(DN_LOW), 'comp_pos': tpos})
        add('ref_dn9_15', {'comp': mix(DN_MID), 'comp_pos': tpos})
        o8 = {r: cap[r]['o'][8] for r in ('corr', 'same')}
        hs = slice(H6 * 256, (H6 + 1) * 256)
        add('h6_same', {'o': [(8, H6, o8['same'][tpos][:, hs])]})
        add('h6_corr', {'o': [(8, H6, o8['corr'][tpos][:, hs])]})
        add('h6_zero', {'o': [(8, H6, torch.zeros_like(o8['corr'][tpos][:, hs]))]})
        add('h6_mean', {'o': [(8, H6, o8['corr'][P['s']:P['e'], hs].mean(0, keepdim=True).expand(len(tpos), -1))]})
        add('h6_blk', {'blk': [(8, H6)]})
        add('fa8top3_same', {'o': [(8, h, o8['same'][tpos][:, h * 256:(h + 1) * 256]) for h in TOP3_L8]})
        add('fa8top3_blk', {'blk': [(8, h) for h in TOP3_L8]})
        # no reference at all: target queries of FA 4/8/12 see only their own cell and the text; DeltaNet 1-15 memory
        # reset (alpha = 0) at the target tokens for every head
        noref = {'blk': [(l, h) for l in FA_MID for h in range(16)], 'rst': [(l, h) for l in DN_LOW + DN_MID for h in range(32)]}
        add('noref', noref)
        add('noref_fa', {'blk': noref['blk']})
        add('noref_dn', {'rst': noref['rst']})
        # reader redirection: value vectors of the 4 target tokens -> those of the control cell (matched in order)
        def redir(qs, heads):
            return {'redir': [(l, h, q, tpos, epos) for (l, h) in heads for q in qs]}
        ctrl_heads = [(16, 0), (16, 9), (16, 13), (20, 0), (20, 5), (20, 14)]
        add('redir_row', redir([P['q_row']], R.READERS))
        add('redir_col', redir([P['q_col']], R.READERS))
        add('redir_both', redir([P['q_row'], P['q_col']], R.READERS))
        add('redir_ctrl', redir([P['q_row'], P['q_col']], ctrl_heads))
        add('redir_all16', redir([P['q_row'], P['q_col']], [(l, h) for l in (16, 20) for h in range(16)]))
        res_rows = []
        for role in ('odd', 'corr'):
            idx = [j for j, r in enumerate(roles) if r == role]
            b0, ans, ir, ic = batch_img(rt, imgs[role], rc)
            res = run_specs(rt, b0, ans, ir, ic, P, groups, tcell, ecell, [specs[j] for j in idx], npatch, args.batch)
            for j, r in zip(idx, res):
                res_rows.append({'id': i, 'n': n, 'cls': row['cls'], 'pair': row['pair'], 'variant': row['variant'],
                                 'tcell': tcell, 'ecell': ecell, 'image': role, 'cond': names[j],
                                 'n_vit_excl': int(sum(len(E[k]) for k in E)), **r})
        R._append(rows_path, res_rows)
        print(f'[main] {n + 1}/{len(sel)} {time.time() - t0:.0f}s', flush=True)
    (out_dir / 'run_main.json').write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()}, **qr.describe(rt),
                                                       'seconds': time.time() - t0, 'batch_size': args.batch}, indent=1))


def head_parts_tokens(rt, st: dict, layer: int, head: int, tpos, groups, tcell, blocked: bool = False):
    """Per-token version of head_parts: (4, T, d) = own / earlier / rest parts and the total of one head after W_O."""
    blk = rt.layers[layer - 1]
    att = blk.self_attn
    hd = att.head_dim
    x = st['inp'][layer]
    cos, sin = st['pe'][layer]
    L = x.shape[0]
    pr = _probs(rt, layer, x, cos, sin, tpos, groups, tcell, blocked)[head]
    xn = blk.input_layernorm(x)
    v = att.v_proj(xn).view(L, -1, hd)[:, head // att.num_key_value_groups].float()
    g = torch.sigmoid(att.q_proj(xn[tpos]).view(len(tpos), -1, hd * 2)[:, head, hd:].float())
    Wo = att.o_proj.weight.float()[:, head * hd:(head + 1) * hd]
    own = groups[tcell]
    earlier = [t for gg in groups[:tcell] for t in gg]
    rest = sorted(set(range(L)) - set(own) - set(earlier))
    out = []
    for idx in (own, earlier, rest):
        it = torch.as_tensor(idx, device=x.device)
        out.append(((pr[:, it] @ v[it]) * g) @ Wo.T)
    out.append(out[0] + out[1] + out[2])
    return torch.stack(out)


# ----------------------------------------------------------------------------- stage redirect

@torch.inference_mode()
def stage_redirect(rt, args) -> None:
    """Where the answer reads the target's position: value vectors of the target tokens replaced by those of the
    redirect cell (attention patterns fixed) for chosen full-attention heads at chosen answer-side query positions."""
    pool = Pool(args.round3, args.output)
    sel = [i for i in sorted(pool.rows) if pool.rows[i]['selected']]
    if args.limit:
        sel = sel[:args.limit]
    path = args.output / 'redirect.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    t0 = time.time()
    for n, i in enumerate(sel):
        if i in done:
            continue
        row = pool.rows[i]
        it = pool.r3.items[i]
        tcell, rc = row['tcell'], tuple(row['target'])
        ecell = sp.redirect_cell(tcell)
        base, answers, i_row, i_col = batch_img(rt, pool.image('odd', i), rc)
        P = R.positions(rt, base, len(answers[0]), i_row, i_col)
        groups = R.cell_tokens(it['meta'], P['s'])
        tpos, epos = groups[tcell], groups[ecell]
        post = list(range(P['e'], P['q_col'] + 1))
        fa_hi = [(l, h) for l in (16, 20, 24, 28, 32) for h in range(16)]
        fa_all = [(l, h) for l in (4, 8, 12, 16, 20, 24, 28, 32) for h in range(16)]
        rd = lambda heads, qs: {'tpos': tpos, 'redir': [(l, h, q, tpos, epos) for l, h in heads for q in qs]}
        specs = [{'tpos': tpos}, rd(R.READERS, [P['q_row'], P['q_col']]), rd(R.READERS, post), rd(fa_hi, post),
                 rd(fa_all, post), rd(fa_all, [P['q_row'], P['q_col']])]
        names = ['none', 'r6_digits', 'r6_post', 'fa16_32_post', 'fa_all_post', 'fa_all_digits']
        res = run_specs(rt, base, answers, i_row, i_col, P, groups, tcell, ecell, specs, 1, args.batch)
        R._append(path, [{'id': i, 'cls': row['cls'], 'tcell': tcell, 'ecell': ecell, 'cond': c, 'n_post': len(post), **r}
                         for c, r in zip(names, res)])
        print(f'[redirect] {n + 1}/{len(sel)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- stage probe

@torch.inference_mode()
def stage_probe(rt, args) -> None:
    """Reader outputs (W_O applied, summed over the six readers per layer) at the row / column query and reader value
    vectors per cell on atlas odd images."""
    items = R.Items(args.round3)
    odd = sorted([it for it in items.odd(['atlas'])], key=lambda x: x['id'])
    rec = defaultdict(list)
    t0 = time.time()
    for a in range(0, len(odd), args.batch):
        chunk = odd[a:a + args.batch]
        base, answers, i_row, i_col = R.batch_for(rt, items, [it['id'] for it in chunk], [tuple(it['target']) for it in chunk])
        P = R.positions(rt, base, len(answers[0]), i_row, i_col)
        st = {'o': {}}
        hs = []
        for l in (16, 20):
            def po(m, a_, l=l):
                if a_[0].shape[1] == P['L']:
                    st['o'][l] = a_[0][:, [P['q_row'], P['q_col']]].detach().clone()
            hs.append(rt.layers[l - 1].self_attn.o_proj.register_forward_pre_hook(po))
        try:
            with R.capture_layers(rt, (16, 20), 0, None) as st2:
                rt.model(**base, use_cache=False, logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        for b, it in enumerate(chunk):
            groups = R.cell_tokens(it['meta'], P['s'])
            outs, vals, att = [], [], []
            for l, h in R.READERS:
                Wo = rt.layers[l - 1].self_attn.o_proj.weight.float()[:, h * 256:(h + 1) * 256]
                outs.append((st['o'][l][b, :, h * 256:(h + 1) * 256].float() @ Wo.T))          # (2, d)
                blk = rt.layers[l - 1]
                xn = blk.input_layernorm(st2['inp'][l][b])
                v = blk.self_attn.v_proj(xn).view(xn.shape[0], -1, 256)[:, h // blk.self_attn.num_key_value_groups].float()
                vals.append(torch.stack([v[g].mean(0) for g in groups]))
            rd = R.reader_readout(rt, st2, b, groups, P['q_row'])
            rec['id'].append(it['id'])
            rec['target'].append(it['target'])
            rec['cls'].append(A.feature_class(it['pair_name'], it['odd_variant']))
            rec['out'].append(torch.stack(outs).half().cpu().numpy())                         # (6, 2, d)
            rec['val'].append(torch.stack(vals).half().cpu().numpy())                         # (6, 36, 256)
            rec['reader'].append(rd)
        print(f'[probe] {a + len(chunk)}/{len(odd)} {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(args.output / 'probe.npz', **{k: np.asarray(v) for k, v in rec.items()})


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['prep', 'main', 'probe', 'redirect'])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--round3', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    if args.stage == 'prep':
        stage_prep(args)
        return
    rt = qr.load(args.model)
    {'main': stage_main, 'probe': stage_probe, 'redirect': stage_redirect}[args.stage](rt, args)


if __name__ == '__main__':
    main()
