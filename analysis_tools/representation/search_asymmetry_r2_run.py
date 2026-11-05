"""Experiment 2.7 round 2 GPU stages (resumable; outputs under --output).

  behave   greedy direct answers for every round-2 odd image (generate_batch, left-padded); results_behave.jsonl
  spotcheck 20 cases re-decoded at batch size 1; spotcheck.json
  capture  one teacher-forced prefill per quadruple image (feature-present, feature-absent, homA, homB) of
           (a) the new pairs at the subtle level, (b) the salient-level Q/O and open/closed quadruples and
           (c) the round-1 subtle Q/O and open/closed quadruples.  Per image it stores
             * per entity, ViT MLP activations after GELU (max over the entity's 16 patches) at --neuron-blocks;
             * per patch, the ViT MLP activations at --patch-block for the first --patch-items quadruples;
             * per entity, residual-stream means at --state-sites;
             * ViT attention probabilities aggregated to entity x entity matrices (per head) at --vit-attn-blocks;
             * decoder full-attention probabilities aggregated to entity x entity matrices and, from the query
               position that predicts the row digit, to entities (per head) at every full-attention layer.
           capture/<source>_<set>_<pair>_<level>.npz
  steer    causal test of the oddness direction (directions.npz from the analysis stage 'direction'):
             * add alpha * m * u to the tokens of one entity of a homogeneous image (no odd entity) at a residual
               site, and read whether the teacher-forced answer points to that entity; random orthogonal direction
               as control; the answer query's attention to the edited entity is recomputed at full-attention layers;
             * remove the component along u from the target tokens of correctly answered images (R as in round 1).
           steer.jsonl

Teacher forcing uses the fixed answer text '{"row": r, "column": c}'.  S = log p(row digit) + log p(column digit).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.behavior.decomposition import read_jsonl, unit_seed
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_run as sr

FULL_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


class Sources:
    """Round-2 items (source 'r2') and round-1 items (source 'r1', ids prefixed 'r1|')."""

    def __init__(self, output: Path, round1: Path | None):
        self.roots = {'r2': output}
        self.items = {}
        if (output / 'items.jsonl').exists():
            self.items = {it['id']: {**it, 'source': 'r2'} for it in read_jsonl(output / 'items.jsonl')}
        if round1 is not None and (round1 / 'items.jsonl').exists():
            self.roots['r1'] = round1
            for it in read_jsonl(round1 / 'items.jsonl'):
                self.items['r1|' + it['id']] = {**it, 'id': 'r1|' + it['id'], 'source': 'r1', 'set': 'r1',
                                                'rows': it['grid'], 'cols': it['grid']}

    def image(self, item_id: str) -> Image.Image:
        it = self.items[item_id]
        return Image.open(self.roots[it['source']] / it['path']).convert('RGB')


def quad_key(it: dict) -> tuple:
    return (it['source'], it.get('set', 'r1'), it['pair_name'], it['level'], it['grid'], it['pair'])


# ----------------------------------------------------------------------------- behaviour

def stage_behave(rt, args, src: Sources, output: Path) -> None:
    path = output / 'results_behave.jsonl'
    done = sr._done(path)
    todo = [it for it in src.items.values() if it['source'] == 'r2' and it['variant'] in sa.ODD_VARIANTS
            and it['id'] not in done]
    groups = defaultdict(list)
    for it in todo:
        groups[tuple(it['meta']['canvas'])].append(it)
    t0 = time.time()
    for canvas, its in sorted(groups.items()):
        for i in range(0, len(its), args.batch_direct):
            chunk = its[i:i + args.batch_direct]
            out = qr.generate_batch(rt, [([src.image(it['id'])], sa.DIRECT_PROMPT) for it in chunk], 32)
            rows = []
            for it, o in zip(chunk, out):
                rec = {'id': it['id'], 'item': it['id'], 'mode': 'direct', 'batch_size': len(chunk), **o}
                rec.update(sa.score(it, rec))
                rows.append(rec)
            sr._append(path, rows)
        print(f'[behave] {canvas} {len(its)} cases, {time.time() - t0:.0f}s', flush=True)


def stage_spotcheck(rt, args, src: Sources, output: Path) -> None:
    recs = read_jsonl(output / 'results_behave.jsonl')
    random.Random(args.seed).shuffle(recs)
    rows = []
    for r in recs[:args.spotcheck]:
        g = qr.generate(rt, qr.build(rt, [src.image(r['item'])], sa.DIRECT_PROMPT), 32)
        rows.append({'id': r['id'], 'batch_text': r['text'], 'single_text': g['text'],
                     'same_text': g['text'] == r['text'],
                     'same_answer': oe.parse_answer(g['text']) == oe.parse_answer(r['text'])})
    (output / 'spotcheck.json').write_text(json.dumps(
        {'n': len(rows), 'same_answer': sum(x['same_answer'] for x in rows),
         'same_text': sum(x['same_text'] for x in rows), 'rows': rows}, indent=1))
    print(f'[spotcheck] answers {sum(x["same_answer"] for x in rows)}/{len(rows)}', flush=True)


# ----------------------------------------------------------------------------- attention helpers

def vit_attention_probs(attn_mod, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Softmax probabilities of one image's ViT attention.  hidden: (n_patch, 1152) block input after norm1;
    cos/sin: (n_patch, head_dim).  Returns (heads, n_patch, n_patch) float32."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision
    n = hidden.shape[0]
    q, k, _ = attn_mod.qkv(hidden).reshape(n, 3, attn_mod.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    sc = torch.einsum('qhd,khd->hqk', q.float(), k.float()) * attn_mod.scaling
    return sc.softmax(-1)


def full_attention_probs(rt, layer: int, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         qpos: torch.Tensor) -> torch.Tensor:
    """Causal softmax probabilities of a decoder full-attention layer for selected queries of one row.
    h: (L, d) layer input (after input norm); cos/sin: (L, rot); qpos: (nq,).  Returns (heads, nq, L)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    mod = rt.layers[layer - 1].self_attn
    L, hd = h.shape[0], mod.head_dim
    k = mod.k_norm(mod.k_proj(h[None]).view(1, L, -1, hd)).transpose(1, 2)
    _, k = apply_rotary_pos_emb(k, k, cos[None], sin[None])
    k = repeat_kv(k, mod.num_key_value_groups)
    q, _ = torch.chunk(mod.q_proj(h[None, qpos]).view(1, len(qpos), -1, hd * 2), 2, dim=-1)
    q = mod.q_norm(q).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, q, cos[None, qpos], sin[None, qpos])
    sc = (q.float() @ k.float().transpose(2, 3)) * mod.scaling
    sc = sc.masked_fill(torch.arange(L, device=h.device)[None, :] > qpos[:, None], float('-inf'))
    return sc.softmax(-1)[0]


def assignment(groups: list[list[int]] | list[torch.Tensor], n: int, device) -> torch.Tensor:
    """(n, n_groups) one-hot matrix: entry (p, g) = 1 if position p belongs to group g."""
    a = torch.zeros(n, len(groups), device=device)
    for g, idx in enumerate(groups):
        a[torch.as_tensor(idx, device=device), g] = 1.0
    return a


def entity_matrix(probs: torch.Tensor, key_assign: torch.Tensor, query_assign: torch.Tensor) -> torch.Tensor:
    """Aggregate (heads, nq, nk) probabilities to (heads, n_query_groups, n_key_groups): for every pair of groups,
    the mean over the query positions of one group of the summed probability on the key positions of the other.
    key_assign: (nk, n_key_groups), query_assign: (nq, n_query_groups), both one-hot."""
    mass = probs @ key_assign                                   # (heads, nq, n_key_groups)
    qn = query_assign / query_assign.sum(0, keepdim=True)
    return torch.einsum('qi,hqj->hij', qn, mass)


# ----------------------------------------------------------------------------- capture

def capture_quads(src: Sources, args) -> dict[tuple, dict[str, str]]:
    quads = defaultdict(dict)
    for it in src.items.values():
        if it.get('group') != 'pair' or it['grid'] != 6:
            continue
        if it['source'] == 'r2' and it['set'] in ('pairs', 'salient_hom'):
            quads[quad_key(it)][it['variant']] = it['id']
        if it['source'] == 'r1' and it['pair_name'] in args.r1_pairs and it['level'] == 'subtle' \
                and it['pair'] < args.hom_items:
            quads[quad_key(it)][it['variant']] = it['id']
    return {k: v for k, v in quads.items() if len(v) == 4}


@torch.inference_mode()
def stage_capture(rt, args, src: Sources, output: Path) -> None:
    cap = output / 'capture'
    cap.mkdir(exist_ok=True)
    by_pair = defaultdict(list)
    for k, v in sorted(capture_quads(src, args).items()):
        by_pair[k[:4]].append((k, v))
    visual = rt.model.model.visual
    digits = sr.digit_ids(rt, 9)
    for (source, set_name, pair, level), qs in sorted(by_pair.items()):
        path = cap / f'{source}_{set_name}_{pair}_{level}.npz'
        if path.exists():
            continue
        ids = [v[var] for _, v in qs for var in sa.VARIANTS]
        targets = [tuple(src.items[v['std']]['target']) for _, v in qs for _ in sa.VARIANTS]
        meta = src.items[ids[0]]['meta']
        tok = oe.entity_tokens(meta)
        n_ent, n_img = len(tok), len(ids)
        n_patch_img = min(len(ids), 4 * args.patch_items)
        A = {k: np.zeros((n_img, n_ent, 4304), np.float16) for k in args.neuron_blocks}
        P = np.zeros((n_patch_img, n_ent, 16, 4304), np.float16)
        S = {s: np.zeros((n_img, n_ent, 1152 if s.startswith('vit') else 4096), np.float16) for s in args.state_sites}
        VA = np.zeros((n_img, len(args.vit_attn_blocks), 16, n_ent, n_ent), np.float16)
        DA = np.zeros((n_img, len(FULL_LAYERS), 16, n_ent, n_ent), np.float16)
        QA = np.zeros((n_img, len(FULL_LAYERS), 16, n_ent), np.float16)
        SC = np.zeros((n_img,), np.float32)
        t0 = time.time()
        for i0 in range(0, n_img, args.batch_capture):
            chunk = list(range(i0, min(n_img, i0 + args.batch_capture)))
            answers = [sr.answer_ids(rt, targets[j]) for j in chunk]
            ans, i_row, i_col = answers[0][0], answers[0][1], answers[0][2]
            batch = sr.build_rows(rt, [src.image(ids[j]) for j in chunk], [a[0] for a in answers])
            s, e = sr.image_span(rt, batch)
            n_vis, n_patch = e - s, 4 * (e - s)
            L = batch['input_ids'].shape[1]
            tok_t = torch.tensor(tok, device=rt.device)
            patch_t = (tok_t[..., None] * 4 + torch.arange(4, device=rt.device)).flatten(1)     # (n_ent, 16)
            store, handles = {}, []
            for k in sorted(set(args.neuron_blocks) | {args.patch_block}):
                def pre(m, inp, k=k):
                    store[('act', k)] = inp[0].detach()
                handles.append(visual.blocks[k - 1].mlp.linear_fc2.register_forward_pre_hook(pre))
            for k in args.vit_attn_blocks:
                def vpre(m, a, kw, k=k):
                    store[('vattn', k)] = ((a[0] if a else kw['hidden_states']).detach(), kw['position_embeddings'])
                handles.append(visual.blocks[k - 1].attn.register_forward_pre_hook(vpre, with_kwargs=True))
            for layer in FULL_LAYERS:
                def fpre(m, a, kw, layer=layer):
                    store[('fattn', layer)] = (kw['hidden_states'].detach(), kw['position_embeddings'])
                handles.append(rt.layers[layer - 1].self_attn.register_forward_pre_hook(fpre, with_kwargs=True))
            for site in args.state_sites:
                mod = (visual.blocks[int(site[3:]) - 1] if site.startswith('vit') else
                       visual.merger if site == 'dec0' else rt.layers[int(site[3:]) - 1])
                def hook(m, i, o, site=site):
                    store[site] = sr._out(o).detach()
                handles.append(mod.register_forward_hook(hook))
            try:
                logits = rt.model(**batch, use_cache=False, logits_to_keep=len(ans) + 1).logits
            finally:
                for h in handles:
                    h.remove()
            sc = sr.score_rows(logits, [a[0] for a in answers], i_row, i_col, len(ans), digits)
            groups_dec = [torch.tensor([s + q for q in t], device=rt.device) for t in tok]
            for b, j in enumerate(chunk):
                SC[j] = sc[b]['s']
                for k in args.neuron_blocks:
                    x = store[('act', k)][b * n_patch:(b + 1) * n_patch]
                    A[k][j] = x[patch_t].float().amax(1).cpu().numpy().astype(np.float16)
                if j < n_patch_img:
                    x = store[('act', args.patch_block)][b * n_patch:(b + 1) * n_patch]
                    P[j] = x[patch_t].float().cpu().numpy().astype(np.float16)
                for site in args.state_sites:
                    x = store[site]
                    if site.startswith('vit'):
                        f = x[b * n_patch:(b + 1) * n_patch][patch_t].float().mean(1)
                    elif site == 'dec0':
                        f = x[b * n_vis:(b + 1) * n_vis][tok_t].float().mean(1)
                    else:
                        f = x[b, s:e][tok_t].float().mean(1)
                    S[site][j] = f.cpu().numpy().astype(np.float16)
                pa = assignment([patch_t[i] for i in range(n_ent)], n_patch, rt.device)
                for bi, k in enumerate(args.vit_attn_blocks):
                    hid, (cos, sin) = store[('vattn', k)]
                    sl = slice(b * n_patch, (b + 1) * n_patch)
                    pr = vit_attention_probs(visual.blocks[k - 1].attn, hid[sl], cos[sl], sin[sl])
                    VA[j, bi] = entity_matrix(pr, pa, pa).cpu().numpy().astype(np.float16)
                L0 = L - len(ans) - 1
                for li, layer in enumerate(FULL_LAYERS):
                    hid, (cos, sin) = store[('fattn', layer)]
                    qpos = torch.cat([torch.arange(s, e, device=rt.device), torch.tensor([L0 + i_row], device=rt.device)])
                    pr = full_attention_probs(rt, layer, hid[b], cos[b], sin[b], qpos)       # (heads, n_vis + 1, L)
                    ka = assignment(groups_dec, L, rt.device)
                    qa = assignment([g - s for g in groups_dec], n_vis, rt.device)
                    DA[j, li] = entity_matrix(pr[:, :-1], ka, qa).cpu().numpy().astype(np.float16)
                    QA[j, li] = (pr[:, -1] @ ka).cpu().numpy().astype(np.float16)
            del store
        np.savez(path, ids=np.array(ids), targets=np.array(targets), score=SC, patch_block=args.patch_block,
                 vit_attn_blocks=np.array(args.vit_attn_blocks), full_layers=np.array(FULL_LAYERS),
                 patch_acts=P, vit_attn=VA, dec_attn=DA, ans_attn=QA,
                 **{f'act_vit{k}': v for k, v in A.items()}, **{f'state_{s}': v for s, v in S.items()})
        print(f'[capture] {path.name}: {n_img} images, {time.time() - t0:.0f}s', flush=True)


@torch.inference_mode()
def stage_states(rt, args, src: Sources, output: Path) -> None:
    """Entity residual means at --state-sites for the target-index and layout sets (feature-absent and
    feature-present targets), one prefill per image without teacher-forced answer; states/<set>_<pair>_<layout>.npz."""
    out_dir = output / 'states'
    out_dir.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for it in src.items.values():
        if it['source'] == 'r2' and it.get('set') in ('index', 'layout'):
            groups[(it['set'], it['pair_name'], f'{it["rows"]}x{it["cols"]}')].append(it)
    visual = rt.model.model.visual
    sites = [x for x in args.state_sites if not x.startswith('vit')]
    for (set_name, pair, lay), its in sorted(groups.items()):
        path = out_dir / f'{set_name}_{pair}_{lay}.npz'
        if path.exists():
            continue
        its = sorted(its, key=lambda x: x['id'])
        tok = oe.entity_tokens(its[0]['meta'])
        S = {x: np.zeros((len(its), len(tok), 4096), np.float16) for x in sites}
        t0 = time.time()
        for i0 in range(0, len(its), args.batch_capture):
            chunk = its[i0:i0 + args.batch_capture]
            batch = sr.build_rows(rt, [src.image(it['id']) for it in chunk], [[0]] * len(chunk))
            s, e = sr.image_span(rt, batch)
            n_vis = e - s
            tok_t = torch.tensor(tok, device=rt.device)
            store, handles = {}, []
            for site in sites:
                mod = visual.merger if site == 'dec0' else rt.layers[int(site[3:]) - 1]
                def hook(m, i, o, site=site):
                    store[site] = sr._out(o).detach()
                handles.append(mod.register_forward_hook(hook))
            try:
                rt.model(**batch, use_cache=False, logits_to_keep=1)
            finally:
                for h in handles:
                    h.remove()
            for b in range(len(chunk)):
                for site in sites:
                    x = store[site]
                    f = x[b * n_vis:(b + 1) * n_vis][tok_t].float().mean(1) if site == 'dec0' else x[b, s:e][tok_t].float().mean(1)
                    S[site][i0 + b] = f.cpu().numpy().astype(np.float16)
        cols = its[0]['cols']
        np.savez(path, ids=np.array([it['id'] for it in its]), variants=np.array([it['variant'] for it in its]),
                 targets=np.array([it['target'][0] * cols + it['target'][1] for it in its]),
                 **{f'state_{x}': v for x, v in S.items()})
        print(f'[states] {path.name}: {len(its)} images, {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- steering

def residual_module(rt, site: str):
    return rt.model.model.visual.merger if site == 'dec0' else rt.layers[int(site[3:]) - 1]


def add_hooks(rt, edits: dict[str, list[tuple[int, torch.Tensor, torch.Tensor, str]]], s: int, n_vis: int, L: int):
    """edits[site] = list of (row, merged-token indices within the image, vector (d,) or (k, d), mode).
    mode 'add' adds the vector; 'set' assigns it.  Returns the hook handles."""
    handles = []
    for site, lst in edits.items():
        def hook(m, i, o, lst=lst, site=site):
            h = sr._out(o)
            for row, tk, vec, mode in lst:
                if site == 'dec0':
                    idx = row * n_vis + tk.to(h.device)
                    if mode == 'add':
                        h[idx] += vec.to(h.device, h.dtype)
                    else:
                        h[idx] = vec.to(h.device, h.dtype)
                else:
                    if h.shape[1] != L:
                        continue
                    idx = s + tk.to(h.device)
                    if mode == 'add':
                        h[row, idx] += vec.to(h.device, h.dtype)
                    else:
                        h[row, idx] = vec.to(h.device, h.dtype)
            return o
        handles.append(residual_module(rt, site).register_forward_hook(hook))
    return handles


def steer_items(src: Sources, output: Path, args) -> tuple[list[dict], list[dict]]:
    """(homogeneous images to steer, correctly answered odd images for the removal test)."""
    beh = {}
    for root, pre in ((output, ''), (src.roots.get('r1'), 'r1|')):
        if root is not None and (root / 'results_behave.jsonl').exists():
            for r in read_jsonl(root / 'results_behave.jsonl'):
                beh[pre + r['item']] = r
    quads = capture_quads(src, args)
    hom, odd = [], []
    for key, v in sorted(quads.items()):
        pair, level = key[2], key[3]
        if pair not in args.steer_pairs or level != 'subtle' or key[5] >= args.steer_items:
            continue
        rng = np.random.default_rng(unit_seed(args.seed, 'steer', key[0], pair, key[5]))
        for var in ('homA', 'homB'):
            hom.append({'item': v[var], 'pair': pair, 'entity': int(rng.integers(0, 36)), 'quad': key[5]})
        for var in ('std', 'swap'):
            if beh.get(v[var], {}).get('correct'):
                same, clean = ('homA', 'homB') if var == 'std' else ('homB', 'homA')
                odd.append({'item': v[var], 'pair': pair, 'variant': var, 'same': v[same], 'clean': v[clean],
                            'quad': key[5]})
    return hom, odd


@torch.inference_mode()
def stage_steer(rt, args, src: Sources, output: Path) -> None:
    dirs = np.load(output / 'directions.npz')
    path = output / 'steer.jsonl'
    done = sr._done(path, 'item')
    hom, odd = steer_items(src, output, args)
    digits = sr.digit_ids(rt, 9)
    t0 = time.time()
    gen = torch.Generator().manual_seed(args.seed)

    def unit(site, pair):
        u = torch.tensor(dirs[f'u|{site}|{pair}'], dtype=torch.float32)
        r = torch.randn(u.shape, generator=gen)
        r = r - (r @ u) * u
        return u, r / r.norm(), float(dirs[f'm|{site}|{pair}'])

    todo = [h for h in hom if h['item'] not in done]
    print(f'[steer] {len(todo)} homogeneous images, {len(odd)} odd images', flush=True)
    for n_done, h in enumerate(todo):
        it = src.items[h['item']]
        tok = oe.entity_tokens(it['meta'])
        e = h['entity']
        rc = (e // 6, e % 6)
        ans, i_row, i_col = sr.answer_ids(rt, rc)
        tk = torch.tensor(tok[e])
        conds = [('none', None, 0.0, 'u')]
        for site in args.steer_sites:
            for a in args.alphas:
                conds.append((f'{site}|u|{a:g}', site, a, 'u'))
            conds.append((f'{site}|rand|{max(args.alphas):g}', site, max(args.alphas), 'rand'))
        rows = []
        for i0 in range(0, len(conds), args.batch_mech):
            chunk = conds[i0:i0 + args.batch_mech]
            batch = sr.build_rows(rt, [src.image(h['item'])] * len(chunk), [ans] * len(chunk))
            s, end = sr.image_span(rt, batch)
            L = batch['input_ids'].shape[1]
            edits = defaultdict(list)
            for r, (name, site, a, kind) in enumerate(chunk):
                if site is None:
                    continue
                u, rnd, m = unit(site, h['pair'])
                vec = (u if kind == 'u' else rnd) * (a * m)
                edits[site].append((r, tk, vec, 'add'))
            store = {}
            handles = add_hooks(rt, edits, s, end - s, L)
            for layer in args.read_layers:
                def fpre(m_, a_, kw, layer=layer):
                    store[layer] = (kw['hidden_states'].detach(), kw['position_embeddings'])
                handles.append(rt.layers[layer - 1].self_attn.register_forward_pre_hook(fpre, with_kwargs=True))
            try:
                logits = rt.model(**batch, use_cache=False, logits_to_keep=len(ans) + 1).logits
            finally:
                for hd in handles:
                    hd.remove()
            res = sr.score_rows(logits, [ans] * len(chunk), i_row, i_col, len(ans), digits)
            L0 = L - len(ans) - 1
            ent_pos = torch.tensor([s + q for q in tok[e]], device=rt.device)
            for r, (c, rr) in enumerate(zip(chunk, res)):
                att = {}
                for layer in args.read_layers:
                    hid, (cos, sin) = store[layer]
                    pr = full_attention_probs(rt, layer, hid[r], cos[r], sin[r],
                                              torch.tensor([L0 + i_row], device=rt.device))
                    mass = pr[:, 0, ent_pos].sum(-1)
                    att[f'att_L{layer}_mean'] = float(mass.mean())
                    att[f'att_L{layer}_max'] = float(mass.max())
                    att[f'att_L{layer}_heads'] = [round(float(x), 5) for x in mass]
                rows.append({'id': f'{h["item"]}|{c[0]}', 'item': h['item'], 'test': 'add', 'pair': h['pair'],
                             'entity': e, 'cond': c[0], 'site': c[1], 'alpha': c[2], 'kind': c[3],
                             'hit': rr['arg_row'] == rc[0] + 1 and rr['arg_col'] == rc[1] + 1, **rr, **att})
            del store
        sr._append(path, rows)
        if n_done % 10 == 0:
            print(f'[steer] add {n_done + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)

    # removal test on correctly answered odd images
    todo = [o for o in odd if o['item'] not in done]
    for n_done, o in enumerate(todo):
        it = src.items[o['item']]
        tok = oe.entity_tokens(it['meta'])
        target = tuple(it['target'])
        t = target[0] * 6 + target[1]
        ans, i_row, i_col = sr.answer_ids(rt, target)
        imgs = [src.image(o['item']), src.image(o['same']), src.image(o['clean'])]
        base = sr.build_rows(rt, imgs, [ans] * 3)
        s, end = sr.image_span(rt, base)
        n_vis, L = end - s, base['input_ids'].shape[1]
        store, handles = {}, []
        for site in args.steer_sites:
            def hook(m_, i_, out, site=site):
                store[site] = sr._out(out).detach().float().clone()
            handles.append(residual_module(rt, site).register_forward_hook(hook))
        try:
            ref = sr.score_rows(rt.model(**base, use_cache=False, logits_to_keep=len(ans) + 1).logits,
                                [ans] * 3, i_row, i_col, len(ans), digits)
        finally:
            for hd in handles:
                hd.remove()
        tk = torch.tensor(tok[t])
        others = torch.tensor([q for i, tt in enumerate(tok) if i != t for q in tt])
        conds = []
        for site in args.steer_sites:
            u, rnd, m = unit(site, o['pair'])
            x = store[site]
            xs = x[:n_vis] if site == 'dec0' else x[0, s:end]          # row 0 = the odd image
            mu = xs[others.to(xs.device)].mean(0)
            xt = xs[tk.to(xs.device)]                                   # (4, d)
            proj = ((xt - mu) @ u.to(xt.device))[:, None] * u.to(xt.device)[None]
            conds.append((f'{site}|remove_u', site, xt - proj))
            conds.append((f'{site}|remove_rand', site, xt - proj.norm(dim=1, keepdim=True) * rnd.to(xt.device)[None]))
        batch = sr.build_rows(rt, [imgs[0]] * len(conds), [ans] * len(conds))
        edits = defaultdict(list)
        for r, (name, site, val) in enumerate(conds):
            edits[site].append((r, tk, val, 'set'))
        handles = add_hooks(rt, edits, s, n_vis, L)
        try:
            res = sr.score_rows(rt.model(**batch, use_cache=False, logits_to_keep=len(ans) + 1).logits,
                                [ans] * len(conds), i_row, i_col, len(ans), digits)
        finally:
            for hd in handles:
                hd.remove()
        rows = [{'id': f'{o["item"]}|{c[0]}', 'item': o['item'], 'test': 'remove', 'pair': o['pair'],
                 'variant': o['variant'], 'cond': c[0], 'site': c[1], 's_clean': ref[0]['s'], 's_same': ref[1]['s'],
                 's_base': ref[2]['s'], **r} for c, r in zip(conds, res)]
        sr._append(path, rows)
        if n_done % 10 == 0:
            print(f'[steer] remove {n_done + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)


def head_ablation_hooks(rt, heads: dict[int, list[int]], start: int):
    """Zero the outputs of the given full-attention heads (layer -> head indices) at sequence positions >= start
    (the teacher-forced answer span), by editing the o_proj input."""
    handles = []
    for layer, hs in heads.items():
        mod = rt.layers[layer - 1].self_attn
        hd = mod.head_dim
        def pre(m, inp, hs=hs, hd=hd):
            x = inp[0]
            for h in hs:
                x[:, start:, h * hd:(h + 1) * hd] = 0
            return (x,)
        handles.append(mod.o_proj.register_forward_pre_hook(pre))
    return handles


def projection_hooks(rt, sites: list[str], dirs: dict, rows: list[tuple[int, torch.Tensor, torch.Tensor, str]],
                     s: int, n_vis: int, L: int):
    """At every site, remove from the target tokens their component along a unit direction relative to the
    distractor mean, computed on the fly: h_t <- h_t - ((h_t - mean_d h_d) . w) w.
    rows: (row, target token indices, distractor token indices, key) with dirs[(site, key)] the unit vector."""
    handles = []
    for site in sites:
        def hook(m, i, o, site=site):
            h = sr._out(o)
            if site != 'dec0' and h.shape[1] != L:
                return o
            for row, tk, dk, key in rows:
                w = dirs[(site, key)].to(h.device, torch.float32)
                if site == 'dec0':
                    ti, di = row * n_vis + tk.to(h.device), row * n_vis + dk.to(h.device)
                    xt, mu = h[ti].float(), h[di].float().mean(0)
                    h[ti] = (xt - ((xt - mu) @ w)[:, None] * w[None]).to(h.dtype)
                else:
                    ti, di = s + tk.to(h.device), s + dk.to(h.device)
                    xt, mu = h[row, ti].float(), h[row, di].float().mean(0)
                    h[row, ti] = (xt - ((xt - mu) @ w)[:, None] * w[None]).to(h.dtype)
            return o
        handles.append(residual_module(rt, site).register_forward_hook(hook))
    return handles


@torch.inference_mode()
def stage_steer2(rt, args, src: Sources, output: Path) -> None:
    """Controls for the steering stage.
      (a) random orthogonal directions at every alpha (the steer stage only used the largest alpha);
      (b) reader-head ablation: the answer-span outputs of --reader-heads are zeroed, on homogeneous images
          edited with alpha = 1 at dec0, and on correctly answered odd images; control = the same number of other
          heads in the same layers;
      (c) projection ablation of the oddness direction at every site in --project-sites simultaneously on
          correctly answered odd images, against a random orthogonal direction per site."""
    dirs_np = np.load(output / 'directions.npz')
    path = output / 'steer2.jsonl'
    done = sr._done(path, 'item')
    hom, odd = steer_items(src, output, args)
    digits = sr.digit_ids(rt, 9)
    gen = torch.Generator().manual_seed(args.seed + 1)
    reader = defaultdict(list)
    for x in args.reader_heads:
        l, h = x.split(':')
        reader[int(l)].append(int(h))
    rng = np.random.default_rng(args.seed)
    ctrl = {l: sorted(rng.choice([h for h in range(16) if h not in hs], len(hs), replace=False).tolist())
            for l, hs in reader.items()}
    t0 = time.time()
    for n_done, hm in enumerate([h for h in hom if h['item'] not in done]):
        it = src.items[hm['item']]
        tok = oe.entity_tokens(it['meta'])
        e = hm['entity']
        rc = (e // 6, e % 6)
        ans, i_row, i_col = sr.answer_ids(rt, rc)
        tk = torch.tensor(tok[e])
        conds = []
        for site in args.steer_sites:
            u = torch.tensor(dirs_np[f'u|{site}|{hm["pair"]}'], dtype=torch.float32)
            m = float(dirs_np[f'm|{site}|{hm["pair"]}'])
            for a in args.alphas:
                r = torch.randn(u.shape, generator=gen)
                r = r - (r @ u) * u
                conds.append((f'{site}|rand|{a:g}', site, r / r.norm() * a * m, None))
        u0 = torch.tensor(dirs_np[f'u|dec0|{hm["pair"]}'], dtype=torch.float32) * float(dirs_np[f'm|dec0|{hm["pair"]}'])
        conds.append(('dec0|u|1+ablate_reader', 'dec0', u0, reader))
        conds.append(('dec0|u|1+ablate_control', 'dec0', u0, ctrl))
        rows = []
        for c in conds:
            batch = sr.build_rows(rt, [src.image(hm['item'])], [ans])
            s, end = sr.image_span(rt, batch)
            L = batch['input_ids'].shape[1]
            handles = add_hooks(rt, {c[1]: [(0, tk, c[2], 'add')]}, s, end - s, L)
            if c[3] is not None:
                handles += head_ablation_hooks(rt, c[3], L - len(ans) - 1)
            try:
                res = sr.score_rows(rt.model(**batch, use_cache=False, logits_to_keep=len(ans) + 1).logits,
                                    [ans], i_row, i_col, len(ans), digits)[0]
            finally:
                for hd in handles:
                    hd.remove()
            rows.append({'id': f'{hm["item"]}|{c[0]}', 'item': hm['item'], 'test': 'add', 'pair': hm['pair'],
                         'entity': e, 'cond': c[0], 'hit': res['arg_row'] == rc[0] + 1 and res['arg_col'] == rc[1] + 1,
                         **res})
        sr._append(path, rows)
        if n_done % 20 == 0:
            print(f'[steer2] add {n_done + 1}/{len(hom)} {time.time() - t0:.0f}s', flush=True)
    for n_done, o in enumerate([o for o in odd if o['item'] not in done]):
        it = src.items[o['item']]
        tok = oe.entity_tokens(it['meta'])
        target = tuple(it['target'])
        t = target[0] * 6 + target[1]
        ans, i_row, i_col = sr.answer_ids(rt, target)
        imgs = [src.image(o['item']), src.image(o['same']), src.image(o['clean'])]
        base = sr.build_rows(rt, imgs, [ans] * 3)
        ref = sr.score_rows(rt.model(**base, use_cache=False, logits_to_keep=len(ans) + 1).logits,
                            [ans] * 3, i_row, i_col, len(ans), digits)
        tk = torch.tensor(tok[t])
        dk = torch.tensor([q for i, tt in enumerate(tok) if i != t for q in tt])
        dirs = {}
        for site in args.project_sites:
            u = torch.tensor(dirs_np[f'u|{site}|{o["pair"]}'], dtype=torch.float32)
            r = torch.randn(u.shape, generator=gen)
            r = r - (r @ u) * u
            dirs[(site, 'u')], dirs[(site, 'rand')] = u, r / r.norm()
        conds = [('project_u_all', 'u', None), ('project_rand_all', 'rand', None),
                 ('ablate_reader', None, reader), ('ablate_control', None, ctrl)]
        rows = []
        for name, key, heads in conds:
            batch = sr.build_rows(rt, [imgs[0]], [ans])
            s, end = sr.image_span(rt, batch)
            L = batch['input_ids'].shape[1]
            handles = (projection_hooks(rt, args.project_sites, dirs, [(0, tk, dk, key)], s, end - s, L) if key else
                       head_ablation_hooks(rt, heads, L - len(ans) - 1))
            try:
                res = sr.score_rows(rt.model(**batch, use_cache=False, logits_to_keep=len(ans) + 1).logits,
                                    [ans], i_row, i_col, len(ans), digits)[0]
            finally:
                for hd in handles:
                    hd.remove()
            rows.append({'id': f'{o["item"]}|{name}', 'item': o['item'], 'test': 'remove', 'pair': o['pair'],
                         'variant': o['variant'], 'cond': name, 's_clean': ref[0]['s'], 's_same': ref[1]['s'],
                         's_base': ref[2]['s'], **res})
        sr._append(path, rows)
        if n_done % 20 == 0:
            print(f'[steer2] remove {n_done + 1}/{len(odd)} {time.time() - t0:.0f}s', flush=True)
    (output / 'steer2_heads.json').write_text(json.dumps({'reader': reader, 'control': ctrl}))


# ----------------------------------------------------------------------------- entry

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['build', 'behave', 'spotcheck', 'capture', 'states', 'steer', 'steer2'])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--round1', type=Path, default=None, help='round-1 output directory')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--new-pairs', nargs='+', default=['term', 'plus', 'dot', 'circsq', 'plusx'])
    p.add_argument('--main-subtle', type=int, default=72)
    p.add_argument('--main-salient', type=int, default=36)
    p.add_argument('--hom-items', type=int, default=24)
    p.add_argument('--salient-hom-pairs', nargs='+', default=['qo', 'closure'])
    p.add_argument('--index-pairs', nargs='+', default=['qo', 'closure', 'tilt'])
    p.add_argument('--index-items', type=int, default=216)
    p.add_argument('--layout-pairs', nargs='+', default=['qo', 'closure', 'tilt'])
    p.add_argument('--layouts', nargs='+', default=['3x12', '12x3'])
    p.add_argument('--layout-items', type=int, default=108)
    p.add_argument('--layout-present-items', type=int, default=36)
    p.add_argument('--batch-direct', type=int, default=32)
    p.add_argument('--batch-capture', type=int, default=8)
    p.add_argument('--batch-mech', type=int, default=8)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--r1-pairs', nargs='+', default=['qo', 'closure', 'curve', 'tilt'])
    p.add_argument('--neuron-blocks', type=int, nargs='+', default=[6, 9, 12, 18])
    p.add_argument('--patch-block', type=int, default=9)
    p.add_argument('--patch-items', type=int, default=12)
    p.add_argument('--state-sites', nargs='+', default=['vit12', 'vit24', 'dec0', 'dec4', 'dec8', 'dec12', 'dec16'])
    p.add_argument('--vit-attn-blocks', type=int, nargs='+', default=[3, 6, 9, 12, 18, 24])
    p.add_argument('--steer-pairs', nargs='+', default=['qo', 'closure', 'curve', 'tilt', 'dot', 'term'])
    p.add_argument('--steer-items', type=int, default=12)
    p.add_argument('--steer-sites', nargs='+', default=['dec0', 'dec8', 'dec16'])
    p.add_argument('--alphas', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    p.add_argument('--read-layers', type=int, nargs='+', default=[12, 16, 20, 24])
    p.add_argument('--reader-heads', nargs='+', default=['16:6', '16:11', '16:2', '20:2', '20:11', '20:10'],
                   help='layer:head of the full-attention heads whose answer-query attention rises most under steering')
    p.add_argument('--project-sites', nargs='+', default=['dec0', 'dec4', 'dec8', 'dec12', 'dec16'])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'build':
        from analysis_tools.representation import search_asymmetry_r2 as r2
        cfg = {'seed': args.seed, 'new_pairs': args.new_pairs,
               'main_items': {'subtle': args.main_subtle, 'salient': args.main_salient}, 'hom_items': args.hom_items,
               'salient_hom_pairs': args.salient_hom_pairs, 'index_pairs': args.index_pairs,
               'index_items': args.index_items, 'layout_pairs': args.layout_pairs, 'layouts': args.layouts,
               'layout_items': args.layout_items, 'layout_present_items': args.layout_present_items}
        r2.build(args.output, cfg)
        print(f'[build] {len(read_jsonl(args.output / "items.jsonl"))} items', flush=True)
        return
    src = Sources(args.output, args.round1)
    rt = qr.load(args.model)
    meta = {'stage': args.stage, 'args': {k: str(v) for k, v in vars(args).items()}, **qr.describe(rt),
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    {'behave': stage_behave, 'spotcheck': stage_spotcheck, 'capture': stage_capture, 'states': stage_states,
     'steer': stage_steer, 'steer2': stage_steer2}[args.stage](rt, args, src, args.output)
    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (args.output / f'run_{args.stage}.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
