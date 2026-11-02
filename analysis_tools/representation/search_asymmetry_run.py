"""Experiment 2.7 GPU stages (all resumable; outputs under --output).

  behave     greedy direct answers for every odd image (generate_batch, left-padded); results_behave.jsonl
  spotcheck  20 direct cases re-decoded at batch size 1 with qwen_runtime.generate; spotcheck.json
  capture    one prefill per quadruple image (std, swap, homA, homB) of the main-grid subtle items with donors,
             plus the experiment-2.6 gap2 / red quadruples: per entity, the ViT MLP activations (post-GELU, max
             over the entity's 16 patches) at --neuron-blocks and the residual means at --state-sites;
             capture/<source>_<pair>.npz
  edit       symmetric ViT-neuron edits in homogeneous images (needs neurons.json from the analysis): at the
             target cell, the presence-selective (A) or absence-selective (B) neurons are set to the values the
             other homogeneous image has at the same patches; teacher-forced answer scores; edit.jsonl
  mech       on correctly answered std / swap images: token-mixer output patching at the target tokens
             (ViT attention, Gated DeltaNet, full attention), distractor-token patching, Gated DeltaNet gate
             edits (alpha = 0 at target tokens, beta = 0 at raster-earlier / later entities), and full-attention
             probabilities from target and answer queries; mech.jsonl, attn.jsonl

Teacher forcing always uses the fixed answer text '{"row": r, "column": c}' so that every condition is scored on
the same tokens.  S = log p(row digit) + log p(column digit); the retained fraction R uses the no-odd donor as
the baseline, as in experiment 2.6.
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

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa

FULL_LAYERS = (4, 8, 12, 16, 20)
DEC_MIX_LAYERS = tuple(range(1, 17))
VIT_BLOCKS = 27


def _append(path: Path, rows: list[dict]) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')


def _done(path: Path, key: str = 'id') -> set[str]:
    return {r[key] for r in read_jsonl(path)} if path.exists() else set()


def _out(o):
    return o[0] if isinstance(o, tuple) else o


class Sources:
    """Items of this experiment and of experiment 2.6 (prefixed 'oe|'), with image loading by root."""

    def __init__(self, output: Path, odd_entity: Path | None):
        self.roots = {'sa': output}
        self.items = {it['id']: {**it, 'source': 'sa'} for it in sa.load_items(output).values()}
        if odd_entity is not None and (odd_entity / 'items.jsonl').exists():
            self.roots['oe'] = odd_entity
            for it in oe.load_items(odd_entity).values():
                self.items['oe|' + it['id']] = {**it, 'id': 'oe|' + it['id'], 'source': 'oe'}

    def image(self, item_id: str) -> Image.Image:
        it = self.items[item_id]
        return Image.open(self.roots[it['source']] / it['path']).convert('RGB')


def quad_key(it: dict) -> tuple:
    """(source, pair name, grid, level, pair index) identifying a quadruple."""
    if it['source'] == 'oe':
        return ('oe', it['kind'], it['grid'], it['group'], it['pair'])
    return ('sa', it['pair_name'], it['grid'], it['level'], it['pair'])


def donor_variants(it: dict) -> tuple[str, str]:
    """(same, clean): the homogeneous image whose entities equal the target, and the one equal to the distractors."""
    if it['source'] == 'oe':
        return ('homX', 'homT') if it['variant'] == 'std' else ('homT', 'homX')
    return ('homA', 'homB') if it['variant'] == 'std' else ('homB', 'homA')


# ----------------------------------------------------------------------------- behaviour

def stage_behave(rt, args, src: Sources, output: Path) -> None:
    items = [it for it in src.items.values() if it['source'] == 'sa' and it['variant'] in sa.ODD_VARIANTS]
    path = output / 'results_behave.jsonl'
    done = _done(path)
    todo = [it for it in items if it['id'] not in done]
    groups = defaultdict(list)
    for it in todo:
        groups[tuple(it['meta']['canvas'])].append(it)
    t0 = time.time()
    for canvas, its in sorted(groups.items()):
        bs = args.batch_direct if canvas[0] * canvas[1] <= 448 * 448 else max(4, args.batch_direct // 4)
        for i in range(0, len(its), bs):
            chunk = its[i:i + bs]
            out = qr.generate_batch(rt, [([src.image(it['id'])], sa.DIRECT_PROMPT) for it in chunk], 32)
            rows = []
            for it, o in zip(chunk, out):
                rec = {'id': it['id'], 'item': it['id'], 'mode': 'direct', 'batch_size': len(chunk), **o}
                rec.update(sa.score(it, rec))
                rows.append(rec)
            _append(path, rows)
        print(f'[behave] {canvas} {len(its)} cases, {time.time() - t0:.0f}s', flush=True)


def stage_spotcheck(rt, args, src: Sources, output: Path) -> None:
    recs = read_jsonl(output / 'results_behave.jsonl')
    random.Random(args.seed).shuffle(recs)
    rows = []
    for r in recs[:args.spotcheck]:
        batch = qr.build(rt, [src.image(r['item'])], sa.DIRECT_PROMPT)
        g = qr.generate(rt, batch, 32)
        rows.append({'id': r['id'], 'batch_text': r['text'], 'single_text': g['text'],
                     'same_text': g['text'] == r['text'],
                     'same_answer': oe.parse_answer(g['text']) == oe.parse_answer(r['text'])})
    (output / 'spotcheck.json').write_text(json.dumps(
        {'n': len(rows), 'same_answer': sum(x['same_answer'] for x in rows),
         'same_text': sum(x['same_text'] for x in rows), 'rows': rows}, indent=1))
    print(f'[spotcheck] answers {sum(x["same_answer"] for x in rows)}/{len(rows)}', flush=True)


# ----------------------------------------------------------------------------- batched prefill

def answer_ids(rt, rc: tuple[int, int]) -> tuple[list[int], int, int]:
    """Token ids of '{"row": r, "column": c}' (1-based r, c) and the indices of the two digit tokens."""
    text = f'{{"row": {rc[0] + 1}, "column": {rc[1] + 1}}}'
    spans = oe.digit_spans(text)
    enc = rt.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    idx = []
    for a, b in spans:
        hit = [i for i, (x, y) in enumerate(enc['offset_mapping']) if x == a and y == b]
        if len(hit) != 1:
            raise ValueError(f'answer digit in {text!r} is not a single token')
        idx.append(hit[0])
    return enc['input_ids'], idx[0], idx[1]


def build_rows(rt, images: list[Image.Image], answers: list[list[int]]) -> dict:
    """Same-size images, identical prompt, per-row teacher-forced answer ids of equal length (no padding)."""
    if len({len(a) for a in answers}) != 1:
        raise ValueError('answers in one batch must have equal token length')
    texts = [qr.prompt_text(rt, 1, sa.DIRECT_PROMPT)] * len(images)
    batch = rt.processor(text=texts, images=images, return_tensors='pt')
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist()):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError('Processor resized an image')
    n, k = batch['input_ids'].shape[0], len(answers[0])
    extra = torch.tensor(answers)
    for key, value in list(batch.items()):
        if torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == batch['input_ids'].shape[1] \
                and key != 'input_ids':
            fill = torch.ones(n, k, dtype=value.dtype) if key == 'attention_mask' else torch.zeros(n, k, dtype=value.dtype)
            batch[key] = torch.cat([value, fill], 1)
    batch['input_ids'] = torch.cat([batch['input_ids'], extra], 1)
    return batch.to(rt.device)


def image_span(rt, batch) -> tuple[int, int]:
    ids = batch['input_ids'][0]
    pos = (ids == rt.model.config.image_token_id).nonzero().flatten()
    return int(pos[0]), int(pos[-1]) + 1


def digit_ids(rt, n: int) -> list[int]:
    return [qr.token_id(rt, str(i)) for i in range(1, n + 1)]


def score_rows(logits: torch.Tensor, answers: list[list[int]], i_row: int, i_col: int, n_ans: int,
               digits: list[int]) -> list[dict]:
    """logits: (B, n_ans + 1, V) for the last n_ans + 1 positions; position j predicts answer token j."""
    lp = torch.log_softmax(logits.float(), -1)
    out = []
    dig = torch.tensor(digits, device=lp.device)
    for b, ans in enumerate(answers):
        r, c = lp[b, i_row, ans[i_row]].item(), lp[b, i_col, ans[i_col]].item()
        out.append({'s': r + c, 's_row': r, 's_col': c,
                    'arg_row': int(lp[b, i_row, dig].argmax()) + 1, 'arg_col': int(lp[b, i_col, dig].argmax()) + 1,
                    'p_row': float(lp[b, i_row, ans[i_row]].exp()), 'p_col': float(lp[b, i_col, ans[i_col]].exp())})
    return out


def vit_patch_index(tokens: list[int]) -> list[int]:
    """Patch indices (within one image) of merged tokens: each merged token holds 4 consecutive patches."""
    return [t * 4 + j for t in tokens for j in range(4)]


# ----------------------------------------------------------------------------- capture

def capture_quads(src: Sources, args) -> dict[tuple, dict[str, str]]:
    """Quadruples with all four variants: this experiment's main-grid subtle items and 2.6 gap2 / red core items."""
    quads = defaultdict(dict)
    for it in src.items.values():
        if it['source'] == 'sa' and it.get('group') == 'pair' and it['level'] == 'subtle' and it['grid'] == args.main_grid:
            quads[quad_key(it)][it['variant']] = it['id']
        if it['source'] == 'oe' and it['group'] == 'core' and it['kind'] in args.oe_kinds and it['pair'] < args.hom_items:
            quads[quad_key(it)][it['variant']] = it['id']
    return {k: v for k, v in quads.items() if len(v) == 4}


@torch.inference_mode()
def stage_capture(rt, args, src: Sources, output: Path) -> None:
    cap = output / 'capture'
    cap.mkdir(exist_ok=True)
    quads = capture_quads(src, args)
    by_pair = defaultdict(list)
    for k, v in sorted(quads.items()):
        by_pair[(k[0], k[1])].append((k, v))
    for (source, pair), qs in sorted(by_pair.items()):
        path = cap / f'{source}_{pair}.npz'
        if path.exists():
            continue
        ids = [v[var] for _, v in qs for var in ('std', 'swap', 'homA' if source == 'sa' else 'homX',
                                                   'homB' if source == 'sa' else 'homT')]
        meta = src.items[ids[0]]['meta']
        tok = oe.entity_tokens(meta)
        n_ent = len(tok)
        A = {k: np.zeros((len(ids), n_ent, 4304), np.float16) for k in args.neuron_blocks}
        S = {s: np.zeros((len(ids), n_ent, 1152 if s.startswith('vit') else 4096), np.float16) for s in args.state_sites}
        t0 = time.time()
        for i0 in range(0, len(ids), args.batch_capture):
            chunk = ids[i0:i0 + args.batch_capture]
            batch = build_rows(rt, [src.image(i) for i in chunk], [[0]] * len(chunk))
            s, e = image_span(rt, batch)
            n_vis, n_patch = e - s, 4 * (e - s)
            tok_t = torch.tensor(tok, device=rt.device)
            patch_t = (tok_t[..., None] * 4 + torch.arange(4, device=rt.device)).flatten(1)     # (n_ent, 16)
            store, handles = {}, []
            visual = rt.model.model.visual
            for k in args.neuron_blocks:
                def pre(m, inp, k=k):
                    store[('act', k)] = inp[0].detach()
                handles.append(visual.blocks[k - 1].mlp.linear_fc2.register_forward_pre_hook(pre))
            for site in args.state_sites:
                mod = (visual.blocks[int(site[3:]) - 1] if site.startswith('vit') else
                       visual.merger if site == 'dec0' else rt.layers[int(site[3:]) - 1])
                def hook(m, i, o, site=site):
                    store[site] = _out(o).detach()
                handles.append(mod.register_forward_hook(hook))
            try:
                rt.model(**batch, use_cache=False, logits_to_keep=1)
            finally:
                for h in handles:
                    h.remove()
            for b in range(len(chunk)):
                j = i0 + b
                for k in args.neuron_blocks:
                    x = store[('act', k)][b * n_patch:(b + 1) * n_patch]
                    A[k][j] = x[patch_t.to(x.device)].float().amax(1).cpu().numpy().astype(np.float16)
                for site in args.state_sites:
                    x = store[site]
                    if site.startswith('vit'):
                        f = x[b * n_patch:(b + 1) * n_patch][patch_t.to(x.device)].float().mean(1)
                    elif site == 'dec0':
                        f = x[b * n_vis:(b + 1) * n_vis][tok_t.to(x.device)].float().mean(1)
                    else:
                        f = x[b, s:e][tok_t.to(x.device)].float().mean(1)
                    S[site][j] = f.cpu().numpy().astype(np.float16)
        np.savez(path, ids=np.array(ids), targets=np.array([src.items[i]['target'] for i in ids]),
                 **{f'act_vit{k}': v for k, v in A.items()}, **{f'state_{s}': v for s, v in S.items()})
        print(f'[capture] {path.name}: {len(ids)} images, {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- generic patched forward

class Plan:
    """Per-row interventions for one batched teacher-forced forward.

    vit_act[block]:  list of (row, patch_idx LongTensor, neuron_idx LongTensor or None, values (p, n))
    vit_attn[block], mixer[layer]: list of (row, idx LongTensor, values (k, d)); idx are patch indices for the
                     ViT (within the image) and absolute sequence positions for the decoder
    gates:           per row, a dict accepted by rowcol_mechanism_run.override_gates_heads
    """

    def __init__(self, n_rows: int):
        self.vit_act, self.vit_attn, self.mixer = defaultdict(list), defaultdict(list), defaultdict(list)
        self.gates = [dict() for _ in range(n_rows)]


def run_plan(rt, batch, plan: Plan, n_ans: int):
    from analysis_tools.position.rowcol_mechanism_run import override_gates_heads
    visual = rt.model.model.visual
    s, e = image_span(rt, batch)
    n_patch = 4 * (e - s)
    L = batch['input_ids'].shape[1]
    handles = []
    for k, lst in plan.vit_act.items():
        def pre(m, inp, lst=lst):
            x = inp[0]
            for row, pidx, nidx, val in lst:
                p = row * n_patch + pidx.to(x.device)
                if nidx is None:
                    x[p] = val.to(x.device, x.dtype)
                else:
                    sub = x[p]
                    sub[:, nidx.to(x.device)] = val.to(x.device, x.dtype)
                    x[p] = sub
            return (x,)
        handles.append(visual.blocks[k - 1].mlp.linear_fc2.register_forward_pre_hook(pre))
    for k, lst in plan.vit_attn.items():
        def hook(m, i, o, lst=lst):
            for row, pidx, val in lst:
                o[row * n_patch + pidx.to(o.device)] = val.to(o.device, o.dtype)
            return o
        handles.append(visual.blocks[k - 1].attn.register_forward_hook(hook))
    for layer, lst in plan.mixer.items():
        block = rt.layers[layer - 1]
        mod = block.linear_attn if block.block_type == 'linear_attention' else block.self_attn
        def hook(m, i, o, lst=lst):
            h = _out(o)
            if h.shape[1] != L:
                return o
            for row, pos, val in lst:
                h[row, pos.to(h.device)] = val.to(h.device, h.dtype)
            return o
        handles.append(mod.register_forward_hook(hook))
    try:
        with override_gates_heads(rt, plan.gates, L):
            logits = rt.model(**batch, use_cache=False, logits_to_keep=n_ans + 1).logits
    finally:
        for h in handles:
            h.remove()
    return logits


# ----------------------------------------------------------------------------- neuron edits

@torch.inference_mode()
def stage_edit(rt, args, src: Sources, output: Path) -> None:
    sel = json.loads((output / f'neurons{args.edit_tag}.json').read_text())
    path = output / f'edit{args.edit_tag}.jsonl'
    done = _done(path, 'item')
    quads = capture_quads(src, args)
    visual = rt.model.model.visual
    digits = digit_ids(rt, 9)
    blocks = [int(b) for b in args.edit_blocks]
    t0 = time.time()
    todo = [(k, v) for k, v in sorted(quads.items()) if f'{k[0]}|{k[1]}|{k[4]}' not in done
            and f'{k[0]}|{k[1]}' in sel and k[4] < args.edit_items]
    print(f'[edit] {len(todo)} quadruples', flush=True)
    for n_done, (key, v) in enumerate(todo):
        source, pair = key[0], key[1]
        nA, nB = ('homA', 'homB') if source == 'sa' else ('homX', 'homT')
        it = src.items[v['std']]
        target = tuple(it['target'])
        g = it['grid']
        tok = oe.entity_tokens(it['meta'])
        t_idx = target[0] * g + target[1]
        pidx = torch.tensor(vit_patch_index(tok[t_idx]), device=rt.device)
        ans, i_row, i_col = answer_ids(rt, target)
        # donor activations of every edit block at the target patches, from the two homogeneous images
        imgs = {var: src.image(v[var]) for var in ('std', 'swap', nA, nB)}
        base = build_rows(rt, [imgs['std'], imgs['swap'], imgs[nA], imgs[nB]], [ans] * 4)
        s, e = image_span(rt, base)
        n_patch = 4 * (e - s)
        store, handles = {}, []
        for k in blocks:
            def pre(m, inp, k=k):
                store[k] = inp[0].detach().clone()
            handles.append(visual.blocks[k - 1].mlp.linear_fc2.register_forward_pre_hook(pre))
        try:
            logits = rt.model(**base, use_cache=False, logits_to_keep=len(ans) + 1).logits
        finally:
            for h in handles:
                h.remove()
        ref = score_rows(logits, [ans] * 4, i_row, i_col, len(ans), digits)
        act = {var: {k: store[k][b * n_patch + pidx.to(store[k].device)] for k in blocks} for b, var in enumerate(('std', 'swap', nA, nB))}
        sets = sel[f'{source}|{pair}']
        rng = np.random.default_rng(sa.unit_seed(args.seed, 'edit', source, pair, key[4]))
        conds = []
        for which in ('A', 'B'):
            neur = {k: torch.tensor(sets[str(k)][which], device=rt.device) for k in blocks}
            conds.append((f'inject_{which}', nB if which == 'A' else nA, nA if which == 'A' else nB, neur))
            conds.append((f'delete_{which}', nA if which == 'A' else nB, nB if which == 'A' else nA, neur))
        rnd = {k: torch.tensor(rng.choice(4304, len(sets[str(k)]['A']), replace=False), device=rt.device) for k in blocks}
        conds.append(('random_AtoB', nB, nA, rnd))
        conds.append(('random_BtoA', nA, nB, rnd))
        conds.append(('full_AtoB', nB, nA, {k: None for k in blocks}))
        conds.append(('full_BtoA', nA, nB, {k: None for k in blocks}))
        plan = Plan(len(conds))
        for row, (name, base_var, donor_var, neur) in enumerate(conds):
            for k in blocks:
                val = act[donor_var][k] if neur[k] is None else act[donor_var][k][:, neur[k].to(act[donor_var][k].device)]
                plan.vit_act[k].append((row, pidx, neur[k], val))
        batch = build_rows(rt, [imgs[c[1]] for c in conds], [ans] * len(conds))
        res = score_rows(run_plan(rt, batch, plan, len(ans)), [ans] * len(conds), i_row, i_col, len(ans), digits)
        rows = [{'id': f'{source}|{pair}|{key[4]}|ref|{var}', 'item': f'{source}|{pair}|{key[4]}', 'source': source,
                 'pair': pair, 'index': key[4], 'cond': f'ref_{var}', 'target': list(target), **r}
                for var, r in zip(('std', 'swap', nA, nB), ref)]
        rows += [{'id': f'{source}|{pair}|{key[4]}|{c[0]}', 'item': f'{source}|{pair}|{key[4]}', 'source': source,
                  'pair': pair, 'index': key[4], 'cond': c[0], 'base_image': c[1], 'target': list(target),
                  'n_neurons': None if c[3][blocks[0]] is None else int(len(c[3][blocks[0]])), 'blocks': blocks, **r}
                 for c, r in zip(conds, res)]
        _append(path, rows)
        if n_done % 20 == 0:
            print(f'[edit] {n_done + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- mechanism

def mech_items(src: Sources, output: Path, args) -> list[dict]:
    """Correctly answered std / swap images with both homogeneous donors, up to --mech-items per kind and variant."""
    beh = {r['item']: r for r in read_jsonl(output / 'results_behave.jsonl')}
    if 'oe' in src.roots:
        for r in read_jsonl(src.roots['oe'] / 'results_behave.jsonl'):
            if r['mode'] == 'direct':
                beh['oe|' + r['item']] = r
    by_quad = defaultdict(dict)
    for it in src.items.values():
        by_quad[quad_key(it)][it['variant']] = it['id']
    chosen = []
    for kind in args.mech_kinds:
        source, name = kind.split(':')
        for var in sa.ODD_VARIANTS:
            cands = []
            for key, v in sorted(by_quad.items()):
                if key[0] != source or key[1] != name or var not in v:
                    continue
                if source == 'sa' and (key[2] != args.main_grid or key[3] != 'subtle'):
                    continue
                if source == 'oe' and key[3] != 'core':
                    continue
                it = src.items[v[var]]
                same, clean = donor_variants(it)
                if same in v and clean in v and beh.get(it['id'], {}).get('correct'):
                    cands.append({'item': it['id'], 'same': v[same], 'clean': v[clean], 'kind': kind, 'variant': var})
            chosen.extend(cands[:args.mech_items])
    return chosen


def mech_conditions(g: int, target: tuple[int, int], tok: list[list[int]], s: int) -> list[dict]:
    """Interventions of the mech stage.  Each condition: name, list of (site, entity indices, donor), gates."""
    t = target[0] * g + target[1]
    others = [i for i in range(g * g) if i != t]
    before, after = [i for i in others if i < t], [i for i in others if i > t]
    delta = [l for l in DEC_MIX_LAYERS if l % 4]
    full = [l for l in DEC_MIX_LAYERS if l % 4 == 0]
    conds = []
    for l in DEC_MIX_LAYERS:
        conds.append({'name': f'mix_tgt_L{l}', 'mix': [(f'dec{l}', [t], 'same')]})
    groups = {'dn1_8': [l for l in delta if l <= 8], 'fa4_8': [4, 8], 'dn9_16': [l for l in delta if l > 8],
              'fa12_16': [12, 16], 'dn1_16': delta, 'fa1_16': full, 'all1_16': list(DEC_MIX_LAYERS)}
    for name, layers in groups.items():
        conds.append({'name': f'mix_tgt_{name}', 'mix': [(f'dec{l}', [t], 'same') for l in layers]})
    for name, blocks in {'v1_6': range(1, 7), 'v7_12': range(7, 13), 'v13_18': range(13, 19),
                         'v19_27': range(19, 28), 'v1_27': range(1, 28)}.items():
        conds.append({'name': f'vitattn_tgt_{name}', 'mix': [(f'vit{b}', [t], 'same') for b in blocks]})
    for name in ('dn1_16', 'fa1_16'):
        conds.append({'name': f'mix_ctx_{name}', 'mix': [(f'dec{l}', others, 'clean') for l in groups[name]]})
    # Gated DeltaNet gate edits (all 32 heads), absolute positions
    tpos = np.array([s + m for m in tok[t]])
    def pos_of(ents):
        return np.array([s + m for i in ents for m in tok[i]], dtype=int)
    ctrl = after[1] if len(after) > 1 else before[-2]            # a distractor at a similar raster position
    for name, layers in {'dn1_8': groups['dn1_8'], 'dn9_16': groups['dn9_16'], 'dn1_16': delta}.items():
        conds.append({'name': f'reset_tgt_{name}', 'gates': {'reset': {l: (tpos, None) for l in layers}}})
        conds.append({'name': f'reset_ctrl_{name}', 'gates': {'reset': {l: (pos_of([ctrl]), None) for l in layers}}})
    for name, layers in {'dn1_8': groups['dn1_8'], 'dn1_16': delta}.items():
        if before:
            conds.append({'name': f'nowrite_before_{name}', 'gates': {'nowrite': {l: (pos_of(before), None) for l in layers}}})
        if after:
            conds.append({'name': f'nowrite_after_{name}', 'gates': {'nowrite': {l: (pos_of(after), None) for l in layers}}})
    # combined: DeltaNet reads at the target cut and full-attention outputs at the target replaced
    conds.append({'name': 'reset_tgt_dn1_16+mix_tgt_fa1_16', 'gates': {'reset': {l: (tpos, None) for l in delta}},
                  'mix': [(f'dec{l}', [t], 'same') for l in full]})
    return conds


def attention_summary(rt, store: dict, queries: dict[str, list[int]], groups: dict[str, np.ndarray]) -> dict:
    """Head-averaged and head-max attention mass from query groups to key groups at the full-attention layers."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    out = {}
    for layer, rec in store.items():
        mod = rt.layers[layer - 1].self_attn
        h, (cos, sin) = rec['h'][:1], rec['pe']
        cos, sin = cos[:1], sin[:1]
        L, hd = h.shape[1], mod.head_dim
        k = mod.k_norm(mod.k_proj(h).view(1, L, -1, hd)).transpose(1, 2)
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        k = repeat_kv(k, mod.num_key_value_groups)
        for qname, qpos in queries.items():
            qi = torch.tensor(qpos, device=h.device)
            q, _ = torch.chunk(mod.q_proj(h[:, qi]).view(1, len(qpos), -1, hd * 2), 2, dim=-1)
            q = mod.q_norm(q).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, q, cos[:, qi], sin[:, qi])
            sc = (q.float() @ k.float().transpose(2, 3)) * mod.scaling
            sc = sc.masked_fill(torch.arange(L, device=h.device)[None, :] > qi[:, None], float('-inf'))
            a = sc.softmax(-1)[0]                                   # (heads, nq, L)
            for gname, gpos in groups.items():
                mass = a[:, :, torch.tensor(gpos, device=h.device, dtype=torch.long)].sum(-1).mean(1)   # (heads,)
                out[f'L{layer}|{qname}|{gname}|mean'] = float(mass.mean())
                out[f'L{layer}|{qname}|{gname}|max'] = float(mass.max())
    return out


@torch.inference_mode()
def stage_mech(rt, args, src: Sources, output: Path) -> None:
    path, apath = output / 'mech.jsonl', output / 'attn.jsonl'
    done = _done(path, 'item')
    todo = [m for m in mech_items(src, output, args) if m['item'] not in done]
    print(f'[mech] {len(todo)} images', flush=True)
    visual = rt.model.model.visual
    digits = digit_ids(rt, 9)
    t0 = time.time()
    for n_done, m in enumerate(todo):
        it = src.items[m['item']]
        g, target = it['grid'], tuple(it['target'])
        tok = oe.entity_tokens(it['meta'])
        t_idx = target[0] * g + target[1]
        ans, i_row, i_col = answer_ids(rt, target)
        n_ans = len(ans)
        imgs = [src.image(m['item']), src.image(m['same']), src.image(m['clean'])]
        base = build_rows(rt, imgs, [ans] * 3)
        s, e = image_span(rt, base)
        n_patch = 4 * (e - s)
        L = base['input_ids'].shape[1]
        conds = mech_conditions(g, target, tok, s)
        need = defaultdict(set)                     # site -> donor rows needed
        for c in conds:
            for site, ents, donor in c.get('mix', []):
                need[site].add(1 if donor == 'same' else 2)
        store, attn_store, handles = {}, {}, []
        for site in need:
            if site.startswith('vit'):
                mod = visual.blocks[int(site[3:]) - 1].attn
            else:
                block = rt.layers[int(site[3:]) - 1]
                mod = block.linear_attn if block.block_type == 'linear_attention' else block.self_attn
            def hook(m_, i, o, site=site):
                if site not in store:
                    store[site] = _out(o).detach().clone()
            handles.append(mod.register_forward_hook(hook))
        for layer in FULL_LAYERS:
            mod = rt.layers[layer - 1].self_attn
            def pre(m_, a, kw, layer=layer):
                attn_store[layer] = {'h': kw['hidden_states'].detach(), 'pe': kw['position_embeddings']}
            handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        try:
            logits = rt.model(**base, use_cache=False, logits_to_keep=n_ans + 1).logits
        finally:
            for h in handles:
                h.remove()
        ref = score_rows(logits, [ans] * 3, i_row, i_col, n_ans, digits)
        # attention edges of the clean image (row 0)
        others = [i for i in range(g * g) if i != t_idx]
        ent_pos = lambda ents: np.array([s + q for i in ents for q in tok[i]], dtype=int)
        all_ent = set(int(x) for x in ent_pos(range(g * g)))
        groups = {'target': ent_pos([t_idx]), 'before': ent_pos([i for i in others if i < t_idx]),
                  'after': ent_pos([i for i in others if i > t_idx]),
                  'background': np.array([p for p in range(s, e) if p not in all_ent], dtype=int),
                  'text_pre': np.arange(0, s), 'text_post': np.arange(e, L)}
        groups = {k: v for k, v in groups.items() if len(v)}
        L0 = L - n_ans - 1
        queries = {'target': [int(x) for x in groups['target']], 'ans_row': [L0 + i_row], 'ans_col': [L0 + i_col]}
        att = attention_summary(rt, attn_store, queries, groups)
        _append(apath, [{'item': m['item'], 'kind': m['kind'], 'variant': m['variant'], 'target': list(target),
                         'n_before': len([i for i in others if i < t_idx]), **att}])
        del attn_store

        def values(site, ents, donor):
            row = 1 if donor == 'same' else 2
            merged = [q for i in ents for q in tok[i]]
            x = store[site]
            if site.startswith('vit'):
                pidx = torch.tensor(vit_patch_index(merged), device=x.device)
                return pidx, x[row * n_patch + pidx]
            pos = torch.tensor([s + q for q in merged], device=x.device)
            return pos, x[row, pos]

        rows = []
        B = args.batch_mech
        for i0 in range(0, len(conds), B):
            chunk = conds[i0:i0 + B]
            plan = Plan(len(chunk))
            for r, c in enumerate(chunk):
                for site, ents, donor in c.get('mix', []):
                    idx, val = values(site, ents, donor)
                    (plan.vit_attn[int(site[3:])] if site.startswith('vit') else plan.mixer[int(site[3:])]).append((r, idx, val))
                if 'gates' in c:
                    plan.gates[r] = c['gates']
            batch = build_rows(rt, [imgs[0]] * len(chunk), [ans] * len(chunk))
            res = score_rows(run_plan(rt, batch, plan, n_ans), [ans] * len(chunk), i_row, i_col, n_ans, digits)
            for c, r in zip(chunk, res):
                rows.append({'id': f'{m["item"]}|{c["name"]}', 'item': m['item'], 'kind': m['kind'],
                             'variant': m['variant'], 'target': list(target), 'cond': c['name'],
                             's_clean': ref[0]['s'], 's_same': ref[1]['s'], 's_base': ref[2]['s'], 'batch_size': len(chunk), **r})
        if n_done < 2:          # identity check: an empty plan must reproduce the clean score
            batch = build_rows(rt, [imgs[0]] * 2, [ans] * 2)
            r = score_rows(run_plan(rt, batch, Plan(2), n_ans), [ans] * 2, i_row, i_col, n_ans, digits)
            rows.append({'id': f'{m["item"]}|identity', 'item': m['item'], 'cond': 'identity', 's': r[0]['s'],
                         's_clean': ref[0]['s']})
        _append(path, rows)
        del store
        if n_done % 5 == 0:
            print(f'[mech] {n_done + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- entry

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['build', 'behave', 'spotcheck', 'capture', 'edit', 'mech'])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--odd-entity', type=Path, default=None, help='experiment-2.6 output directory')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-memory', default=None, help="split the model over visible GPUs, e.g. '0:12GiB,1:12GiB'")
    p.add_argument('--pairs', nargs='+', default=list(sa.PAIRS))
    p.add_argument('--main-grid', type=int, default=6)
    p.add_argument('--main-subtle', type=int, default=72)
    p.add_argument('--main-salient', type=int, default=36)
    p.add_argument('--setsize-grids', type=int, nargs='+', default=[3, 10])
    p.add_argument('--setsize-items', type=int, default=36)
    p.add_argument('--hom-items', type=int, default=24)
    p.add_argument('--conj-grids', type=int, nargs='+', default=[3, 6, 10])
    p.add_argument('--conj-items', type=int, nargs='+', default=[36, 72, 36])
    p.add_argument('--batch-direct', type=int, default=32)
    p.add_argument('--batch-capture', type=int, default=12)
    p.add_argument('--batch-mech', type=int, default=8)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--oe-kinds', nargs='+', default=['gap2', 'red'])
    p.add_argument('--neuron-blocks', type=int, nargs='+', default=[3, 6, 9, 12, 15, 18])
    p.add_argument('--state-sites', nargs='+', default=['vit6', 'vit12', 'vit18', 'vit24', 'dec0', 'dec4', 'dec8',
                                                        'dec12', 'dec16'])
    p.add_argument('--edit-blocks', type=int, nargs='+', default=[3, 6, 9, 12])
    p.add_argument('--edit-items', type=int, default=24)
    p.add_argument('--edit-tag', default='', help="suffix of neurons<tag>.json / edit<tag>.jsonl (second edit round: '_k256')")
    p.add_argument('--mech-kinds', nargs='+', default=['oe:gap2', 'oe:red', 'oe:gap16', 'oe:color20', 'sa:qo', 'sa:closure'])
    p.add_argument('--mech-items', type=int, default=16)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'build':
        cfg = {'seed': args.seed, 'pairs': args.pairs, 'main_grid': args.main_grid,
               'main_items': {'subtle': args.main_subtle, 'salient': args.main_salient},
               'setsize_grids': args.setsize_grids, 'setsize_items': args.setsize_items, 'hom_items': args.hom_items,
               'conj_grids': args.conj_grids, 'conj_items': {str(g): n for g, n in zip(args.conj_grids, args.conj_items)}}
        sa.build(args.output, cfg)
        print(f'[build] {len(read_jsonl(args.output / "items.jsonl"))} items', flush=True)
        return
    src = Sources(args.output, args.odd_entity)
    from analysis_tools.position.coordinate_flow import load_runtime
    rt = load_runtime(args.model, args.max_memory)
    meta = {'stage': args.stage, 'args': {k: str(v) for k, v in vars(args).items()}, **qr.describe(rt),
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    {'behave': stage_behave, 'spotcheck': stage_spotcheck, 'capture': stage_capture, 'edit': stage_edit,
     'mech': stage_mech}[args.stage](rt, args, src, args.output)
    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (args.output / f'run_{args.stage}.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
