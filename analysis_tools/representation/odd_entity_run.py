"""Experiment 2.6 GPU stages: behaviour (generation), entity-state capture, context-replacement patching.

Stages (all resumable; outputs under --output):

  behave     -- greedy answers for every odd image (direct prompt) and a think-then-answer subset
                (generate_batch, left-padded); results_behave.jsonl
  spotcheck  -- 20 direct cases re-decoded with qwen_runtime.generate at batch size 1
  capture    -- one prefill per image with the direct prompt; per-entity mean states at ViT block outputs,
                the merger output (decoder input, site dec0) and decoder layer outputs, plus three text
                positions; capture/<group>_g<grid>_<kind>.npz
  patch      -- teacher-forced answer scores under context replacement (see design doc section 3.4);
                patch.jsonl

Image-token states in the decoder precede the question, so they are identical for the direct and the
think prompt; only the text positions depend on the prompt.
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

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.representation import odd_entity as oe
from analysis_tools.behavior.decomposition import read_jsonl


def _append(path: Path, rows: list[dict]) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')


def _done(path: Path) -> set[str]:
    return {r['id'] for r in read_jsonl(path)} if path.exists() else set()


def _image(output: Path, item: dict) -> Image.Image:
    return Image.open(output / item['path']).convert('RGB')


# ----------------------------------------------------------------------------- behaviour

def stage_behave(rt, args, output: Path) -> None:
    items = oe.load_items(output)
    cases = oe.behaviour_cases(list(items.values()), args.think_pairs)
    path = output / 'results_behave.jsonl'
    done = _done(path)
    todo = [c for c in cases if c['id'] not in done]
    groups = defaultdict(list)
    for c in todo:
        it = items[c['item']]
        groups[(c['mode'], tuple(it['meta']['canvas']))].append(c)
    t0 = time.time()
    for (mode, canvas), cs in sorted(groups.items()):
        bs = args.batch_direct if mode == 'direct' else args.batch_think
        if canvas[0] >= 640:
            bs = max(1, bs // 2)
        for i in range(0, len(cs), bs):
            chunk = cs[i:i + bs]
            out = qr.generate_batch(rt, [([_image(output, items[c['item']])], c['prompt']) for c in chunk],
                                    max(c['max_new_tokens'] for c in chunk))
            rows = []
            for c, o in zip(chunk, out):
                it = items[c['item']]
                rec = {'id': c['id'], 'item': c['item'], 'mode': mode, 'batch_size': len(chunk), **o}
                rec.update(oe.score(it, rec))
                rows.append(rec)
            _append(path, rows)
        print(f'[behave] {mode} {canvas} {len(cs)} cases, {time.time() - t0:.0f}s', flush=True)


def stage_spotcheck(rt, args, output: Path) -> None:
    items = oe.load_items(output)
    recs = [r for r in read_jsonl(output / 'results_behave.jsonl') if r['mode'] == 'direct']
    random.Random(args.seed).shuffle(recs)
    rows = []
    for r in recs[:args.spotcheck]:
        it = items[r['item']]
        batch = qr.build(rt, [_image(output, it)], oe.prompt_for(it, 'direct'))
        g = qr.generate(rt, batch, 32)
        rows.append({'id': r['id'], 'batch_text': r['text'], 'single_text': g['text'],
                     'same_text': g['text'] == r['text'],
                     'same_answer': oe.parse_answer(g['text']) == oe.parse_answer(r['text'])})
    (output / 'spotcheck.json').write_text(json.dumps(
        {'n': len(rows), 'same_answer': sum(x['same_answer'] for x in rows),
         'same_text': sum(x['same_text'] for x in rows), 'rows': rows}, indent=1))
    print(f'[spotcheck] answers {sum(x["same_answer"] for x in rows)}/{len(rows)}', flush=True)


# ----------------------------------------------------------------------------- batched prefill helpers

def build_batch(rt, images: list[Image.Image], text: str, answer_ids: list[int] | None = None) -> dict:
    """Identical prompts with same-size images (no padding); optional teacher-forced answer ids appended."""
    texts = [qr.prompt_text(rt, 1, text)] * len(images)
    batch = rt.processor(text=texts, images=images, return_tensors='pt')
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist()):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError('Processor resized an image')
    if answer_ids:
        n, k = batch['input_ids'].shape[0], len(answer_ids)
        extra = torch.tensor(answer_ids).repeat(n, 1)
        for key, value in list(batch.items()):
            if torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == batch['input_ids'].shape[1] \
                    and key != 'input_ids':
                fill = torch.ones(n, k, dtype=value.dtype) if key == 'attention_mask' else torch.zeros(n, k, dtype=value.dtype)
                batch[key] = torch.cat([value, fill], 1)
        batch['input_ids'] = torch.cat([batch['input_ids'], extra], 1)
    return batch.to(rt.device)


def image_start(rt, batch) -> int:
    ids = batch['input_ids'][0]
    pos = (ids == rt.model.config.image_token_id).nonzero().flatten()
    for row in batch['input_ids']:
        if not torch.equal((row == rt.model.config.image_token_id).nonzero().flatten(), pos):
            raise ValueError('Rows of a batch must share the image-token positions')
    return int(pos[0]), int(pos[-1]) + 1


def text_positions(rt, batch, end: int) -> dict[str, int]:
    """img_last: last visual token; q_last: last token of the user question; last: last prompt token."""
    ids = batch['input_ids'][0].tolist()
    im_end = rt.tokenizer.convert_tokens_to_ids('<|im_end|>')
    q_end = next(i for i in range(end, len(ids)) if ids[i] == im_end)
    return {'img_last': end - 1, 'q_last': q_end - 1, 'last': len(ids) - 1}


def _module(rt, site: str):
    visual = rt.model.model.visual
    if site.startswith('vit'):
        return visual.blocks[int(site[3:]) - 1]
    if site == 'dec0':
        return visual.merger
    if site.startswith('dec'):
        return rt.layers[int(site[3:]) - 1]
    raise ValueError(site)


def _out(o):
    return o[0] if isinstance(o, tuple) else o


# ----------------------------------------------------------------------------- capture

@torch.inference_mode()
def stage_capture(rt, args, output: Path) -> None:
    items = oe.load_items(output)
    cap = output / 'capture'
    cap.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for it in items.values():
        groups[(it['group'], it['grid'], it['kind'] if it['group'] != 'natural' else 'natural')].append(it)
    for (group, grid, kind), its in sorted(groups.items()):
        path = cap / f'{group}_g{grid:02d}_{kind}.npz'
        if path.exists():
            continue
        full = group in ('core', 'natural', 'boundary')
        vit_sites = args.vit_sites if full else args.vit_sites_aux
        dec_sites = args.dec_sites if full else args.dec_sites_aux
        sites = [f'vit{k}' for k in vit_sites] + [f'dec{k}' for k in dec_sites]
        text_sites = [f'dec{k}' for k in args.text_sites]
        its = sorted(its, key=lambda x: x['id'])
        ent_items = [it for it in its if it['variant'] not in ('homT', 'homX')]
        n_ent = its[0]['n_entities']
        E = {s: np.zeros((len(ent_items), n_ent, 1152 if s.startswith('vit') else 4096), np.float16) for s in sites}
        T = {s: np.zeros((len(its), 3, 4096), np.float16) for s in text_sites}
        ent_index = {it['id']: i for i, it in enumerate(ent_items)}
        bs = args.batch_capture if grid <= 6 else max(1, args.batch_capture // 3)
        t0 = time.time()
        for i0 in range(0, len(its), bs):
            chunk = its[i0:i0 + bs]
            batch = build_batch(rt, [_image(output, it) for it in chunk], oe.prompt_for(chunk[0], 'direct'))
            start, end = image_start(rt, batch)
            tpos = text_positions(rt, batch, end)
            tok = oe.entity_tokens(chunk[0]['meta'])
            n_vis = end - start
            n_patch = 4 * n_vis
            tok_t = torch.tensor(tok, device=rt.device)                          # (n_ent, k)
            patch_t = (tok_t[..., None] * 4 + torch.arange(4, device=rt.device)).flatten(1)  # (n_ent, 4k)
            store, handles = {}, []
            for s in set(sites) | set(text_sites):
                def hook(m, i, o, s=s):
                    if s not in store:
                        store[s] = _out(o).detach()
                handles.append(_module(rt, s).register_forward_hook(hook))
            try:
                rt.model(**batch, use_cache=False, logits_to_keep=1)
            finally:
                for h in handles:
                    h.remove()
            for b, it in enumerate(chunk):
                if it['id'] in ent_index:
                    j = ent_index[it['id']]
                    for s in sites:
                        x = store[s]
                        if s.startswith('vit'):
                            feats = x[b * n_patch:(b + 1) * n_patch][patch_t].float().mean(1)
                        elif s == 'dec0':
                            feats = x[b * n_vis:(b + 1) * n_vis][tok_t].float().mean(1)
                        else:
                            feats = x[b, start:end][tok_t].float().mean(1)
                        E[s][j] = feats.cpu().numpy().astype(np.float16)
                jt = its.index(it)
                for s in text_sites:
                    x = store[s][b]
                    T[s][jt] = torch.stack([x[tpos[k]] for k in ('img_last', 'q_last', 'last')]).float().cpu().numpy()
        np.savez(path, ent_ids=np.array([it['id'] for it in ent_items]), all_ids=np.array([it['id'] for it in its]),
                 **{f'E_{s}': v for s, v in E.items()}, **{f'T_{s}': v for s, v in T.items()})
        print(f'[capture] {path.name}: {len(its)} images, {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- patching

def answer_positions(rt, text: str) -> tuple[list[int], int, int] | None:
    """Answer token ids and the indices of the row and column digit tokens (single-digit numbers only)."""
    spans = oe.digit_spans(text)
    if spans is None:
        return None
    enc = rt.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    idx = []
    for a, b in spans:
        hit = [i for i, (x, y) in enumerate(offs) if x == a and y == b]
        if len(hit) != 1:
            return None
        idx.append(hit[0])
    return ids, idx[0], idx[1]


@torch.inference_mode()
def stage_patch(rt, args, output: Path) -> None:
    items = oe.load_items(output)
    beh = {r['item']: r for r in read_jsonl(output / 'results_behave.jsonl') if r['mode'] == 'direct'}
    path = output / 'patch.jsonl'
    done = {r['item'] for r in read_jsonl(path)} if path.exists() else set()
    sites = [f'vit{k}' for k in args.patch_vit_sites] + [f'dec{k}' for k in args.patch_dec_sites]
    by_key = {(it['group'], it['kind'], it['pair'], it['variant']): it for it in items.values()}
    todo = []
    for kind in args.patch_kinds:
        for v in oe.ODD_VARIANTS:
            # the first `patch_pairs` correctly answered images of this kind and variant, in pair order
            chosen = [it for p in range(args.core_pairs) if (it := by_key.get(('core', kind, p, v)))
                      and beh.get(it['id'], {}).get('correct')][:args.patch_pairs]
            todo.extend(it for it in chosen if it['id'] not in done)
    print(f'[patch] {len(todo)} images to patch, {len(sites)} sites', flush=True)
    t0 = time.time()
    for n_done, it in enumerate(todo):
        ans = answer_positions(rt, beh[it['id']]['text'])
        if ans is None:
            _append(path, [{'id': f'{it["id"]}|skip', 'item': it['id'], 'skip': 'answer digits not single tokens'}])
            continue
        ans_ids, i_row, i_col = ans
        same_v, clean_v = ('homX', 'homT') if it['variant'] == 'std' else ('homT', 'homX')
        donors = {'same': by_key[('core', it['kind'], it['pair'], same_v)],
                  'clean': by_key[('core', it['kind'], it['pair'], clean_v)]}
        imgs = [_image(output, it), _image(output, donors['same']), _image(output, donors['clean'])]
        base = build_batch(rt, imgs, oe.prompt_for(it, 'direct'), ans_ids)
        start, end = image_start(rt, base)
        n_vis, n_ans = end - start, len(ans_ids)
        n_patch = 4 * n_vis
        L = base['input_ids'].shape[1]

        def scores(logits):        # logits: (B, n_ans + 1, V) for positions L-n_ans-1 .. L-1
            lp = torch.log_softmax(logits.float(), -1)
            r = lp[:, i_row, ans_ids[i_row]]
            c = lp[:, i_col, ans_ids[i_col]]
            return (r + c).tolist(), r.tolist(), c.tolist()

        store, handles = {}, []
        for s in sites:
            def hook(m, i, o, s=s):
                if s not in store:
                    store[s] = _out(o).detach().clone()
            handles.append(_module(rt, s).register_forward_hook(hook))
        try:
            logits = rt.model(**base, use_cache=False, logits_to_keep=n_ans + 1).logits
        finally:
            for h in handles:
                h.remove()
        s_all, s_row, s_col = scores(logits)
        baseline = {'clean': s_all[0], 'same': s_all[1], 'clean_ctx': s_all[2]}

        tok = oe.entity_tokens(it['meta'])
        g = it['grid']
        target = it['target'][0] * g + it['target'][1]
        groups = oe.context_groups(g, g, tuple(it['target']))
        specs = [('ctx_same', grp, 'same') for grp in ('all', 'nbr', 'far', 'before', 'after')] + \
                [('ctx_clean', grp, 'clean') for grp in ('all', 'after')] + [('tgt', 'target', 'clean')]
        rows_spec = []
        for site in sites:
            for ptype, grp, donor in specs:
                ents = [target] if grp == 'target' else groups[grp]
                if not ents:
                    continue
                rows_spec.append((site, ptype, grp, donor, ents))

        def values(site, donor, ents):
            src = 1 if donor == 'same' else 2
            merged = torch.tensor([m for e in ents for m in tok[e]], device=rt.device)
            x = store[site]
            if site.startswith('vit'):
                pos = (merged[:, None] * 4 + torch.arange(4, device=rt.device)).flatten()
                return pos, x[src * n_patch + pos]
            if site == 'dec0':
                return merged, x[src * n_vis + merged]
            return start + merged, x[src, start + merged]

        out_rows = []
        B = args.batch_patch
        rep = build_batch(rt, [imgs[0]] * B, oe.prompt_for(it, 'direct'), ans_ids)
        for i0 in range(0, len(rows_spec), B):
            chunk = rows_spec[i0:i0 + B]
            edits = defaultdict(list)
            for row, (site, ptype, grp, donor, ents) in enumerate(chunk):
                pos, val = values(site, donor, ents)
                edits[site].append((row, pos, val))
            handles = []
            for site, lst in edits.items():
                def hook(m, i, o, lst=lst, site=site):
                    h = _out(o)
                    for row, pos, val in lst:
                        if site.startswith('vit'):
                            h[row * n_patch + pos] = val
                        elif site == 'dec0':
                            h[row * n_vis + pos] = val
                        else:
                            h[row, pos] = val
                    return o
                handles.append(_module(rt, site).register_forward_hook(hook))
            try:
                logits = rt.model(**rep, use_cache=False, logits_to_keep=n_ans + 1).logits
            finally:
                for h in handles:
                    h.remove()
            s_all, s_row, s_col = scores(logits)
            for row, (site, ptype, grp, donor, ents) in enumerate(chunk):
                out_rows.append({'id': f'{it["id"]}|{site}|{ptype}|{grp}', 'item': it['id'], 'kind': it['kind'],
                                 'variant': it['variant'], 'pair': it['pair'], 'target': it['target'], 'site': site,
                                 'ptype': ptype, 'group': grp, 'n_ent': len(ents), 's': s_all[row],
                                 's_row': s_row[row], 's_col': s_col[row], **{f'base_{k}': v for k, v in baseline.items()},
                                 'batch_size': B})
        # identity check: an unpatched forward of the replicated batch must reproduce the clean baseline
        if n_done < 3:
            logits = rt.model(**rep, use_cache=False, logits_to_keep=n_ans + 1).logits
            out_rows.append({'id': f'{it["id"]}|identity', 'item': it['id'], 'identity_s': scores(logits)[0][0],
                             'base_clean': baseline['clean']})
        _append(path, out_rows)
        if n_done % 10 == 0:
            print(f'[patch] {n_done + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- entry

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['build', 'behave', 'spotcheck', 'capture', 'patch'])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--core-kinds', nargs='+', default=list(oe.KINDS))
    p.add_argument('--aux-kinds', nargs='+', default=['red', 'color20', 'gap16', 'gap2'])
    p.add_argument('--core-pairs', type=int, default=72)
    p.add_argument('--aux-pairs', type=int, default=36)
    p.add_argument('--grid-sizes', type=int, nargs='+', default=[3, 10])
    p.add_argument('--boundary-per-orient', type=int, default=12)
    p.add_argument('--natural-pairs', type=int, default=48)
    p.add_argument('--refcoco-dir', default='')
    p.add_argument('--think-pairs', type=int, default=16)
    p.add_argument('--batch-direct', type=int, default=32)
    p.add_argument('--batch-think', type=int, default=16)
    p.add_argument('--batch-capture', type=int, default=24)
    p.add_argument('--batch-patch', type=int, default=32)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--vit-sites', type=int, nargs='+', default=[1, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    p.add_argument('--dec-sites', type=int, nargs='+', default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28])
    p.add_argument('--vit-sites-aux', type=int, nargs='+', default=[3, 9, 15, 21, 27])
    p.add_argument('--dec-sites-aux', type=int, nargs='+', default=[0, 4, 8, 12, 16, 24])
    p.add_argument('--text-sites', type=int, nargs='+', default=list(range(2, 33, 2)))
    p.add_argument('--patch-kinds', nargs='+', default=['red', 'color20', 'gap16', 'gap2'])
    p.add_argument('--patch-pairs', type=int, default=24)
    p.add_argument('--patch-vit-sites', type=int, nargs='+', default=[1, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    p.add_argument('--patch-dec-sites', type=int, nargs='+', default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'build':
        cfg = {k: getattr(args, k) for k in ('seed', 'core_kinds', 'aux_kinds', 'core_pairs', 'aux_pairs', 'grid_sizes',
                                              'boundary_per_orient', 'natural_pairs', 'refcoco_dir')}
        oe.build(args.output, cfg)
        print(f'[build] {len(read_jsonl(args.output / "items.jsonl"))} items', flush=True)
        return
    rt = qr.load(args.model)
    meta_path = args.output / f'run_{args.stage}.json'
    meta = {'stage': args.stage, 'args': {k: str(v) for k, v in vars(args).items()}, **qr.describe(rt),
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    {'behave': stage_behave, 'spotcheck': stage_spotcheck, 'capture': stage_capture, 'patch': stage_patch}[args.stage](
        rt, args, args.output)
    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    meta_path.write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
