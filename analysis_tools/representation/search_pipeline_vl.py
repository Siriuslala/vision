"""Experiment 2.7 round 4, Qwen3-VL-8B GPU stages (mechanism comparison only; no accuracy comparison is reported).

Qwen3-VL-8B has 36 decoder layers that all use causal full attention (32 query heads, 8 key/value heads, head
dimension 128, no output gate) and DeepStack: the outputs of ViT blocks 8 / 16 / 24 pass through their own mergers
and are added to the image-token residual after decoder layers 1 / 2 / 3.

  behave  greedy direct answers on the round-3 atlas odd images (used only to select correctly answered pairs)
  main    phase 1  reader selection: attention of every head from the row-digit query on the target cell
                   (correct odd images); the six heads with the largest mean mass are the readers.
          phase 2  per selected pair (comparison / detector class, target cell >= 7):
                   * exact additive decomposition of the reader logit advantage of the target (odd - corrupted)
                     over the decoder input, the three DeepStack injections and every attention / MLP output;
                   * interventions at the target tokens: decoder input / DeepStack / attention output of layer l /
                     MLP output of layer l replaced by the corrupted value; target queries of layer l blocked from
                     other cells; all attention outputs replaced by same-type values (reference swap);
                     reader redirection.
          phase 3  per-head blocking at the two layers with the largest blocking effect (comparison items), then
                   output replacement (corrupted / same-type / zero) for the top-3 heads.
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

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_run as sr
from analysis_tools.representation import search_circuit_analysis as A
from analysis_tools.representation import search_circuit_heads as H
from analysis_tools.representation import search_circuit_run as R
from analysis_tools.representation import search_pipeline as sp
from analysis_tools.representation.search_pipeline_run import Pool, batch_img, cond_key

N_DS = 3


# ----------------------------------------------------------------------------- attention algebra (Qwen3-VL)

def _rot(x, cos, sin):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half
    return x * cos + rotate_half(x) * sin


def vl_q(rt, layer, x, cos, sin, qpos):
    att = rt.layers[layer - 1].self_attn
    xn = rt.layers[layer - 1].input_layernorm(x[qpos])
    q = att.q_norm(att.q_proj(xn).view(len(qpos), -1, att.head_dim)).float()
    return _rot(q, cos[qpos][:, None].float(), sin[qpos][:, None].float())          # (nq, H, hd)


def vl_k(rt, layer, x, cos, sin):
    att = rt.layers[layer - 1].self_attn
    xn = rt.layers[layer - 1].input_layernorm(x)
    k = att.k_norm(att.k_proj(xn).view(x.shape[0], -1, att.head_dim)).float()
    return _rot(k, cos[:, None].float(), sin[:, None].float())                       # (L, KV, hd)


def vl_probs(rt, layer, x, cos, sin, qpos):
    att = rt.layers[layer - 1].self_attn
    q = vl_q(rt, layer, x, cos, sin, qpos)
    k = vl_k(rt, layer, x, cos, sin).repeat_interleave(att.num_key_value_groups, dim=1)
    s = torch.einsum('qhd,khd->hqk', q, k) * att.scaling
    mask = torch.arange(x.shape[0], device=x.device)[None, :] > torch.as_tensor(qpos, device=x.device)[:, None]
    return s.masked_fill(mask[None], float('-inf')).softmax(-1)                     # (H, nq, L)


def vl_logit_attribution(rt, layer, x, cos, sin, qpos, comps, img, heads):
    """Same as search_circuit_run.logit_attribution for Qwen3-VL (RMSNorm weight w instead of 1 + w)."""
    blk = rt.layers[layer - 1]
    att = blk.self_attn
    hd = att.head_dim
    xi = x[img].float()
    r_in = torch.rsqrt(xi.pow(2).mean(-1, keepdim=True) + blk.input_layernorm.variance_epsilon)
    w_in = blk.input_layernorm.weight.float()
    kc = (comps.float() * r_in[None] * w_in) @ att.k_proj.weight.float().T
    kc = kc.view(kc.shape[0], kc.shape[1], -1, hd)
    r_k = torch.rsqrt(kc.sum(0).pow(2).mean(-1, keepdim=True) + att.k_norm.variance_epsilon)
    kc = kc * r_k[None] * att.k_norm.weight.float()
    kc = _rot(kc, cos[img][None, :, None].float(), sin[img][None, :, None].float())
    q = vl_q(rt, layer, x, cos, sin, qpos)
    kv_of = [h // att.num_key_value_groups for h in heads]
    contrib = torch.stack([torch.einsum('qd,cnd->qcn', q[:, h], kc[:, :, kv]) for h, kv in zip(heads, kv_of)]) * att.scaling
    kact = vl_k(rt, layer, x, cos, sin)[img]
    actual = torch.stack([torch.einsum('qd,nd->qn', q[:, h], kact[:, kv]) for h, kv in zip(heads, kv_of)]) * att.scaling
    return contrib, actual


# ----------------------------------------------------------------------------- hooks

@contextmanager
def capture_vl(rt, P, layers_inp, upto: int):
    """Row-0 capture: inputs + position embeddings of `layers_inp`; components at image tokens (index 0 decoder
    input, 1..3 DeepStack injections, then attention / MLP output of layers 1..upto); o_proj inputs (all positions)
    of every layer up to `upto`."""
    L, s, e = P['L'], P['s'], P['e']
    st = {'inp': {}, 'pe': {}, 'att': {}, 'mlp': {}, 'o': {}, 'x0': None, 'ds': None}
    hs = []

    def lm_pre(m, a, kw):
        ds = kw.get('deepstack_visual_embeds')
        if ds is not None:
            st['ds'] = [d.detach().clone() for d in ds]
    hs.append(rt.model.model.language_model.register_forward_pre_hook(lm_pre, with_kwargs=True))

    def pre0(m, a, kw):
        if a[0].shape[1] == L:
            st['x0'] = a[0][0, s:e].detach().clone()
    hs.append(rt.layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
    for l in range(1, upto + 1):
        blk = rt.layers[l - 1]

        def ha(m, a, o, l=l):
            h = sr._out(o)
            if h.shape[1] == L:
                st['att'][l] = h[0, s:e].detach().clone()

        def hm(m, a, o, l=l):
            if o.shape[1] == L:
                st['mlp'][l] = o[0, s:e].detach().clone()

        def ho(m, a, l=l):
            if a[0].shape[1] == L:
                st['o'][l] = a[0][0].detach().clone()
        hs += [blk.self_attn.register_forward_hook(ha), blk.mlp.register_forward_hook(hm),
               blk.self_attn.o_proj.register_forward_pre_hook(ho)]
    for l in layers_inp:
        def pl(m, a, kw, l=l):
            if a[0].shape[1] == L:
                st['inp'][l] = a[0][0].detach()
                st['pe'][l] = tuple(x[0] for x in kw['position_embeddings'])
        hs.append(rt.layers[l - 1].register_forward_pre_hook(pl, with_kwargs=True))
    try:
        yield st
    finally:
        for h in hs:
            h.remove()


@contextmanager
def vl_edits(rt, specs, groups, P):
    """spec keys: 'x0' (positions, values); 'ds' (positions relative to the image, values (3, n, d));
    'att' / 'mlp' {layer: (positions, values)}; 'o' / 'blk' as in search_circuit_heads.apply_edits;
    'redir' [(layer, head, qpos, src, dst)]."""
    L, s = P['L'], P['s']
    hs = []
    x0_rows = [(b, sp_['x0']) for b, sp_ in enumerate(specs) if 'x0' in sp_]
    if x0_rows:
        def pre0(m, a, kw):
            x = a[0]
            if x.shape[1] != L:
                return None
            x = x.clone()
            for b, (pos, v) in x0_rows:
                x[b, torch.as_tensor(pos, device=x.device)] = v.to(x.device, x.dtype)
            return (x, *a[1:]), kw
        hs.append(rt.layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
    ds_rows = [(b, sp_['ds']) for b, sp_ in enumerate(specs) if 'ds' in sp_]
    if ds_rows:
        def lm_pre(m, a, kw):
            ds = kw.get('deepstack_visual_embeds')
            if ds is None:
                return None
            n_img = ds[0].shape[0] // len(specs)
            new = [d.clone() for d in ds]
            for b, (rel, v) in ds_rows:
                idx = torch.as_tensor(rel, device=new[0].device) + b * n_img
                for j in range(N_DS):
                    new[j][idx] = v[j].to(new[j].device, new[j].dtype)
            kw['deepstack_visual_embeds'] = new
            return a, kw
        hs.append(rt.model.model.language_model.register_forward_pre_hook(lm_pre, with_kwargs=True))
    for kind in ('att', 'mlp'):
        by = defaultdict(list)
        for b, sp_ in enumerate(specs):
            for l, (pos, v) in sp_.get(kind, {}).items():
                by[l].append((b, pos, v))
        for l, lst in by.items():
            mod = rt.layers[l - 1].self_attn if kind == 'att' else rt.layers[l - 1].mlp

            def hook(m, a, o, lst=lst):
                h = sr._out(o)
                if h.shape[1] != L:
                    return o
                for b, pos, v in lst:
                    h[b, torch.as_tensor(pos, device=h.device)] = v.to(h.device, h.dtype)
                return o
            hs.append(mod.register_forward_hook(hook))
    by = defaultdict(list)
    for b, sp_ in enumerate(specs):
        for e in sp_.get('redir', []):
            by[e[0]].append((b, *e[1:]))
    for l, lst in by.items():
        att = rt.layers[l - 1].self_attn
        stash = {}

        def pl(m, a, kw, stash=stash):
            if a[0].shape[1] == L:
                stash['x'], stash['pe'] = a[0], kw['position_embeddings']
        hs.append(rt.layers[l - 1].register_forward_pre_hook(pl, with_kwargs=True))

        def po(m, a, l=l, lst=lst, stash=stash, att=att):
            x = a[0]
            if x.shape[1] != L:
                return None
            x = x.clone()
            hd = att.head_dim
            rows = defaultdict(list)
            for b, h, qpos, src, dst in lst:
                rows[b].append((h, qpos, tuple(src), tuple(dst)))
            for b, ent in rows.items():
                xb = stash['x'][b]
                cos, sin = (t[b] for t in stash['pe'])
                qs = sorted({q for _, q, _, _ in ent})
                qi = {q: n for n, q in enumerate(qs)}
                pr = vl_probs(rt, l, xb, cos, sin, qs)
                xn = rt.layers[l - 1].input_layernorm(xb)
                v = att.v_proj(xn).view(L, -1, hd).float()
                for h, qpos, src, dst in ent:
                    kv = h // att.num_key_value_groups
                    st_, dt_ = torch.as_tensor(src, device=x.device), torch.as_tensor(dst, device=x.device)
                    w = pr[h, qi[qpos]][st_]
                    x[b, qpos, h * hd:(h + 1) * hd] += (w[:, None] * (v[dt_, kv] - v[st_, kv])).sum(0).to(x.dtype)
            return (x,)
        hs.append(att.o_proj.register_forward_pre_hook(po))
    with ExitStack() as es:
        es.enter_context(H.apply_edits(rt, [{'tpos': sp_['tpos'], **{k: sp_[k] for k in ('o', 'blk') if k in sp_}}
                                            for sp_ in specs], groups, L))
        try:
            yield
        finally:
            for h in hs:
                h.remove()


def readout_vl(rt, readers, P, groups, tcell, ecell):
    """Capture-and-readout context: returns a function computing per-row readout after the forward."""
    layers = sorted({l for l, _ in readers})
    return layers


def run_vl(rt, base, answers, i_row, i_col, P, groups, tcell, ecell, specs, readers, batch):
    layers = sorted({l for l, _ in readers})
    out = []
    for a in range(0, len(specs), batch):
        chunk = specs[a:a + batch]
        b = R.repeat_batch(base, len(chunk))
        st = {'inp': {}, 'pe': {}}
        hs = []
        for l in layers:
            def pl(m, a_, kw, l=l):
                if a_[0].shape[1] == P['L']:
                    st['inp'][l] = a_[0].detach()
                    st['pe'][l] = kw['position_embeddings']
            hs.append(rt.layers[l - 1].register_forward_pre_hook(pl, with_kwargs=True))
        try:
            with vl_edits(rt, chunk, groups, P):
                logits = rt.model(**b, use_cache=False, logits_to_keep=len(answers[0]) + 1).logits
        finally:
            for h in hs:
                h.remove()
        sc_ = sr.score_rows(logits, answers * len(chunk), i_row, i_col, len(answers[0]), sr.digit_ids(rt, 6))
        for r in range(len(chunk)):
            m_t, m_e, m_c, tops = [], [], [], np.zeros(len(groups))
            for l, h in readers:
                cos, sin = (t[r] for t in st['pe'][l])
                pr = vl_probs(rt, l, st['inp'][l][r], cos, sin, [P['q_row'], P['q_col']])[h]   # (2, L)
                cells = R.cell_sum(pr, groups).float().cpu().numpy()                             # (2, 36)
                m_t.append(cells[0, tcell]); m_c.append(cells[1, tcell]); m_e.append(cells[0, ecell]); tops += cells[0]
            out.append({'reader': float(np.mean(m_t)), 'reader_col': float(np.mean(m_c)), 'reader_e': float(np.mean(m_e)),
                        'top1': bool(np.argmax(tops) == tcell), 's': sc_[r]['s'], 'arg_row': sc_[r]['arg_row'],
                        'arg_col': sc_[r]['arg_col']})
    return out


# ----------------------------------------------------------------------------- stages

@torch.inference_mode()
def stage_behave(rt, args) -> None:
    items = R.Items(args.round3)
    path = args.output / 'vl_behave.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    todo = sorted([it for it in items.odd(['atlas']) if it['id'] not in done], key=lambda x: x['id'])
    for a in range(0, len(todo), 32):
        chunk = todo[a:a + 32]
        outs = qr.generate_batch(rt, [([items.image(it['id'])], sa.DIRECT_PROMPT) for it in chunk], 24)
        R._append(path, [{'id': it['id'], 'text': o['text'], 'correct': A.lenient(it, o)[0], 'batch_size': len(chunk)}
                         for it, o in zip(chunk, outs)])
        print(f'[vl behave] {a + len(chunk)}/{len(todo)}', flush=True)


def select_vl(args, items) -> list[str]:
    beh = {r['id']: r for r in read_jsonl(args.output / 'vl_behave.jsonl')}
    pool = defaultdict(list)
    for i in sorted(beh):
        it = items.items[i]
        if beh[i]['correct'] and A.tcell(it) >= 6:
            pool[A.feature_class(it['pair_name'], it['odd_variant'])].append(i)
    rng = np.random.default_rng(args.seed)
    out = []
    for cls, n in (('comparison', args.n_cmp), ('detector', args.n_det)):
        p = pool[cls]
        out += [p[j] for j in sorted(rng.choice(len(p), size=min(n, len(p)), replace=False))]
    return out


@torch.inference_mode()
def stage_main(rt, args) -> None:
    items = R.Items(args.round3)
    out_dir = args.output / 'vl'
    out_dir.mkdir(exist_ok=True)
    nL = len(rt.layers)
    nH = rt.layers[0].self_attn.config.num_attention_heads
    sel = select_vl(args, items)
    if args.limit:
        sel = sel[:args.limit]
    # same-type images for the selected Qwen3-VL pairs
    import json as _json
    have = {r['id'] for r in read_jsonl(args.output / 'items.jsonl')}
    extra = [i for i in sel if i not in have]
    if extra:
        rows = sp.build_same_type(args.round3, args.output, extra)
        R._append(args.output / 'items.jsonl', [{
            'id': r['id'], 'partner': items.items[r['id']]['partner'], 'same_type': r['path'], 'same_sha256': r['sha256'],
            'cond': cond_key(items.items[r['id']]), 'pair': items.items[r['id']]['pair_name'],
            'variant': items.items[r['id']]['odd_variant'],
            'cls': A.feature_class(items.items[r['id']]['pair_name'], items.items[r['id']]['odd_variant']),
            'tcell': A.tcell(items.items[r['id']]), 'target': items.items[r['id']]['target'], 'selected': False}
            for r in rows])
    pool = Pool(args.round3, args.output)
    t0 = time.time()
    # ------------------------------------------------------------ phase 1: readers
    rpath = out_dir / 'readers.json'
    if rpath.exists():
        readers = [tuple(x) for x in _json.loads(rpath.read_text())['readers']]
    else:
        mass = np.zeros((nL, nH))
        for i in sel:
            it = items.items[i]
            base, answers, i_row, i_col = batch_img(rt, pool.image('odd', i), it['target'])
            P = R.positions(rt, base, len(answers[0]), i_row, i_col)
            groups = R.cell_tokens(it['meta'], P['s'])
            with capture_vl(rt, P, tuple(range(1, nL + 1)), 0) as st:
                rt.model(**base, use_cache=False, logits_to_keep=1)
            for l in range(1, nL + 1):
                pr = vl_probs(rt, l, st['inp'][l], *st['pe'][l], [P['q_row']])[:, 0]
                mass[l - 1] += pr[:, groups[A.tcell(it)]].sum(-1).float().cpu().numpy()
        mass /= len(sel)
        order = np.argsort(-mass.ravel())[:6]
        readers = [(int(o // nH) + 1, int(o % nH)) for o in order]
        rpath.write_text(_json.dumps({'readers': readers, 'mass': mass.tolist(), 'n': len(sel)}, indent=1))
        print(f'[vl] readers {readers} {time.time() - t0:.0f}s', flush=True)
    r_layers = sorted({l for l, _ in readers})
    upto = max(r_layers) - 1
    # ------------------------------------------------------------ phase 2
    rows_path = out_dir / 'rows.jsonl'
    done = {(r['id'], r['phase']) for r in read_jsonl(rows_path)} if rows_path.exists() else set()
    recs = defaultdict(list)
    rec_file = out_dir / 'records.npz'
    caches = {}
    for n, i in enumerate(sel):
        it = items.items[i]
        cls = A.feature_class(it['pair_name'], it['odd_variant'])
        tcell = A.tcell(it)
        ecell = sp.redirect_cell(tcell)
        rc = it['target']
        base, answers, i_row, i_col = batch_img(rt, pool.image('odd', i), rc)
        P = R.positions(rt, base, len(answers[0]), i_row, i_col)
        groups = R.cell_tokens(it['meta'], P['s'])
        tpos, epos = groups[tcell], groups[ecell]
        rel = [t - P['s'] for t in tpos]
        cap = {}
        for role in ('odd', 'corr', 'same'):
            b_ = batch_img(rt, pool.image(role, i), rc)[0]
            with capture_vl(rt, P, tuple(r_layers), upto) as st:
                rt.model(**b_, use_cache=False, logits_to_keep=1)
            cap[role] = st
        # mark decomposition at every reader head
        D = {}
        for role in ('odd', 'corr'):
            st = cap[role]
            comps = [st['x0']] + [st['ds'][j] for j in range(N_DS)]
            names = ['input', 'ds1', 'ds2', 'ds3']
            for l in range(1, upto + 1):
                comps += [st['att'][l], st['mlp'][l]]
                names += [f'att{l}', f'mlp{l}']
            per = []
            for l, h in readers:
                # components that exist at the input of reader layer l
                keep = [j for j, nm in enumerate(names) if nm in ('input',) or (nm.startswith('ds') and int(nm[2:]) < l)
                        or (nm[:3] in ('att', 'mlp') and int(nm[3:]) < l)]
                C = torch.stack([comps[j] for j in keep])
                contrib, actual = vl_logit_attribution(rt, l, st['inp'][l], *st['pe'][l], [P['q_row']], C,
                                                       slice(P['s'], P['e']), [h])
                c = contrib[0, 0]                                                    # (n_keep, n_img)
                relg = [[t - P['s'] for t in g] for g in groups]
                cell = torch.stack([c[:, g].mean(-1) for g in relg], -1)             # (n_keep, 36)
                adv = cell[:, tcell] - torch.cat([cell[:, :tcell], cell[:, tcell + 1:]], -1).mean(-1)
                full = torch.zeros(len(names), device=adv.device)
                full[keep] = adv
                act = actual[0, 0]
                cell_act = torch.stack([act[g].mean() for g in relg])
                err = float((c.sum(0) - act).abs().max())
                per.append((full.cpu().numpy(), float(cell_act[tcell] - torch.cat([cell_act[:tcell], cell_act[tcell + 1:]]).mean()), err))
            D[role] = per
        recs['id'].append(i); recs['cls'].append(cls); recs['tcell'].append(tcell)
        recs['adv'].append(np.stack([[p[0] for p in D[r]] for r in ('odd', 'corr')]))      # (2, 6, n_comp)
        recs['adv_actual'].append(np.asarray([[p[1] for p in D[r]] for r in ('odd', 'corr')]))
        recs['adv_err'].append(max(p[2] for r in D for p in D[r]))
        recs['x0_dnorm'].append((cap['odd']['x0'].float() - cap['corr']['x0'].float()).norm(dim=-1).cpu().numpy())
        recs['ds_dnorm'].append(np.stack([(cap['odd']['ds'][j].float() - cap['corr']['ds'][j].float()).norm(dim=-1).cpu().numpy()
                                          for j in range(N_DS)]))
        recs['names'] = names
        if (i, 'p2') in done:
            caches[i] = None
            continue
        specs, cn, roles = [], [], []

        def add(name, spec, role='odd'):
            specs.append({'tpos': tpos, **spec}); cn.append(name); roles.append(role)
        add('none', {})
        add('corr_none', {}, 'corr')
        add('x0_all', {'x0': (tpos, cap['corr']['x0'][rel])})
        add('ds_all', {'ds': (rel, torch.stack([cap['corr']['ds'][j][rel] for j in range(N_DS)]))})
        add('x0ds_all', {'x0': (tpos, cap['corr']['x0'][rel]),
                         'ds': (rel, torch.stack([cap['corr']['ds'][j][rel] for j in range(N_DS)]))})
        add('ins_x0ds_all', {'x0': (tpos, cap['odd']['x0'][rel]),
                             'ds': (rel, torch.stack([cap['odd']['ds'][j][rel] for j in range(N_DS)]))}, 'corr')
        for l in range(1, upto + 1):
            add(f'pa{l}', {'att': {l: (tpos, cap['corr']['att'][l][rel])}})
            add(f'pm{l}', {'mlp': {l: (tpos, cap['corr']['mlp'][l][rel])}})
            add(f'blk{l}', {'blk': [(l, h) for h in range(nH)]})
        add('att_all', {'att': {l: (tpos, cap['corr']['att'][l][rel]) for l in range(1, upto + 1)}})
        add('mlp_all', {'mlp': {l: (tpos, cap['corr']['mlp'][l][rel]) for l in range(1, upto + 1)}})
        add('ref_all', {'att': {l: (tpos, cap['same']['att'][l][rel]) for l in range(1, upto + 1)}})
        add('blk_all', {'blk': [(l, h) for l in range(1, upto + 1) for h in range(nH)]})
        rdr = lambda qs, hs_: {'redir': [(l, h, q, tpos, epos) for l, h in hs_ for q in qs]}
        add('redir_row', rdr([P['q_row']], readers))
        add('redir_col', rdr([P['q_col']], readers))
        add('redir_both', rdr([P['q_row'], P['q_col']], readers))
        post = list(range(P['e'], P['q_col'] + 1))
        add('redir6_post', rdr(post, readers))
        add('redir_all_post', rdr(post, [(l, h) for l in range(1, nL + 1) for h in range(nH)]))
        out_rows = []
        for role in ('odd', 'corr'):
            idx = [j for j, r in enumerate(roles) if r == role]
            b0, ans, ir, ic = batch_img(rt, pool.image(role, i), rc)
            res = run_vl(rt, b0, ans, ir, ic, P, groups, tcell, ecell, [specs[j] for j in idx], readers, args.batch)
            out_rows += [{'id': i, 'cls': cls, 'pair': it['pair_name'], 'variant': it['odd_variant'], 'tcell': tcell,
                          'image': role, 'phase': 'p2', 'cond': cn[j], **r} for j, r in zip(idx, res)]
        R._append(rows_path, out_rows)
        print(f'[vl] phase 2 {n + 1}/{len(sel)} {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(rec_file, **{k: np.asarray(v, dtype=object if k == 'names' else None) for k, v in recs.items()})
    # ------------------------------------------------------------ phase 3: heads at the top blocking layers
    rows = [r for r in read_jsonl(rows_path) if r['phase'] == 'p2' and r['image'] == 'odd']
    base_r = {r['id']: r['reader'] for r in rows if r['cond'] == 'none'}
    drop = defaultdict(list)
    for r in rows:
        if r['cls'] == 'comparison' and r['cond'].startswith('blk') and r['cond'] != 'blk_all':
            drop[int(r['cond'][3:])].append(base_r[r['id']] - r['reader'])
    top_layers = sorted(drop, key=lambda l: -np.mean(drop[l]))[:2]
    sel_path = out_dir / 'selection.json'
    for phase in ('p3', 'p4'):
        heads = None
        if phase == 'p4':
            r3 = [r for r in read_jsonl(rows_path) if r['phase'] == 'p3']
            b3 = {r['id']: r['reader'] for r in r3 if r['cond'] == 'none'}
            d3 = defaultdict(list)
            for r in r3:
                if r['cls'] == 'comparison' and r['cond'] != 'none':
                    d3[r['cond']].append(b3[r['id']] - r['reader'])
            top_heads = sorted(d3, key=lambda c: -np.mean(d3[c]))[:3]
            heads = [tuple(int(x) for x in c[1:].split('h')) for c in top_heads]
            sel_path.write_text(json.dumps({'top_layers': top_layers, 'top_heads': heads, 'readers': readers,
                                            'items': sel}, indent=1))
        done = {(r['id'], r['phase']) for r in read_jsonl(rows_path)}
        for n, i in enumerate(sel):
            if (i, phase) in done:
                continue
            it = items.items[i]
            cls = A.feature_class(it['pair_name'], it['odd_variant'])
            tcell = A.tcell(it)
            ecell = sp.redirect_cell(tcell)
            rc = it['target']
            base, answers, i_row, i_col = batch_img(rt, pool.image('odd', i), rc)
            P = R.positions(rt, base, len(answers[0]), i_row, i_col)
            groups = R.cell_tokens(it['meta'], P['s'])
            tpos = groups[tcell]
            specs, cn = [{'tpos': tpos}], ['none']
            if phase == 'p3':
                for l in top_layers:
                    for h in range(nH):
                        specs.append({'tpos': tpos, 'blk': [(l, h)]}); cn.append(f'l{l}h{h}')
            else:
                o = {}
                for role in ('corr', 'same'):
                    b_ = batch_img(rt, pool.image(role, i), rc)[0]
                    with capture_vl(rt, P, (), max(l for l, _ in heads)) as st:
                        rt.model(**b_, use_cache=False, logits_to_keep=1)
                    o[role] = st['o']
                hd = rt.layers[0].self_attn.head_dim
                for l, h in heads:
                    sl = slice(h * hd, (h + 1) * hd)
                    specs.append({'tpos': tpos, 'o': [(l, h, o['corr'][l][tpos][:, sl])]}); cn.append(f'l{l}h{h}_corr')
                    specs.append({'tpos': tpos, 'o': [(l, h, o['same'][l][tpos][:, sl])]}); cn.append(f'l{l}h{h}_same')
                    specs.append({'tpos': tpos, 'o': [(l, h, torch.zeros_like(o['corr'][l][tpos][:, sl]))]}); cn.append(f'l{l}h{h}_zero')
                    specs.append({'tpos': tpos, 'blk': [(l, h)]}); cn.append(f'l{l}h{h}_blk')
                specs.append({'tpos': tpos, 'blk': list(heads)}); cn.append('top3_blk')
                specs.append({'tpos': tpos, 'o': [(l, h, o['same'][l][tpos][:, slice(h * hd, (h + 1) * hd)]) for l, h in heads]})
                cn.append('top3_same')
            res = run_vl(rt, base, answers, i_row, i_col, P, groups, tcell, ecell, specs, readers, args.batch)
            R._append(rows_path, [{'id': i, 'cls': cls, 'pair': it['pair_name'], 'variant': it['odd_variant'],
                                   'tcell': tcell, 'image': 'odd', 'phase': phase, 'cond': c, **r} for c, r in zip(cn, res)])
            print(f'[vl] {phase} {n + 1}/{len(sel)} {time.time() - t0:.0f}s', flush=True)
    (out_dir / 'run.json').write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()}, **qr.describe(rt),
                                                  'seconds': time.time() - t0, 'batch_size': args.batch,
                                                  'readers': readers}, indent=1))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['behave', 'main'])
    p.add_argument('--model', default='Qwen3-VL-8B-Instruct')
    p.add_argument('--round3', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--n-cmp', type=int, default=60)
    p.add_argument('--n-det', type=int, default=40)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    rt = qr.load(args.model)
    {'behave': stage_behave, 'main': stage_main}[args.stage](rt, args)


if __name__ == '__main__':
    main()
