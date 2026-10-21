"""Stage drivers of experiment 3.7 (ViT position and layout representations).

Stages (selected with --stages, each resumable or idempotent):
  table    PE table of the checkpoint: raw table, similarity structure (no forward pass).
  norms    PE / content norms and the causal share of PE and RoPE in every ViT block, per canvas size.
  coord    red-disc point task under ViT position edits (zero / mean / shuffle / native / scaled PE,
           zero RoPE), canvases 256-1536 pixels, batched generate.
  spot     answer-level agreement of batched generate with single generate under the edits.
  t2       recognition control: two-figure same/different discrimination (prefill Yes-No logit).
  swap     PE-only / RoPE-only / both swaps between the target region and a displaced region with the
           3.5 engine, per canvas size, also with RoPE removed and with the PE amplified (Qwen3.5 only).
  probe    linear position probes on every ViT block and the merger, per ViT edit (natural images).
  layout   object position and relative layout decodable from object, background and pooled features.
  leace    linear erasure of position at the merger; coordinate output, T2 and probes after erasure.
  refcoco  RefCOCO val bbox grounding (experiment 4.2 cases) with the PE scaled.
  reread   re-encoding a crop with RoPE offsets / original-image PE window / canvas context.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import vit_layout as vl
from analysis_tools.position.coordinate_readout_run import append_jsonl, done_ids, read_jsonl
from analysis_tools.position.position_probe_refinement import LAMBDAS, Moments, ridge_path

POINT_PROMPT = cr.DISC_QUERY + cr.SUFFIX['point']
PE_RATIO = 2.756          # mean per-entry norm ratio of the Qwen3-VL-8B and Qwen3.5-9B PE tables
CANVASES = (256, 384, 512, 768, 1024, 1536)


def code_digest() -> dict:
    here = Path(__file__).parent
    names = ('vit_layout.py', 'vit_layout_run.py', 'coordinate_readout.py')
    return {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in names}


def write_meta(out: Path, rt, args, stage: str, extra: dict | None = None) -> None:
    meta = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **qr.describe(rt), 'code': code_digest(),
            'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, **(extra or {})}
    (out / f'{stage}_meta.json').write_text(json.dumps(meta, indent=1) + '\n')


# ----------------------------------------------------------------------------- conditions

def coord_conditions(family: str, canvas: int) -> dict[str, vl.ViTEdit]:
    E = vl.ViTEdit
    conds = {'base': E(), 'pe_zero': E(pe='zero'), 'pe_mean': E(pe='mean'), 'pe_shuffle': E(pe='shuffle'),
             'pe_x2': E(pe_scale=2.0), 'pe_x4': E(pe_scale=4.0), 'rope_zero': E(rope='zero'),
             'rope_zero_pe_x4': E(rope='zero', pe_scale=4.0), 'rope_zero_pe_zero': E(pe='zero', rope='zero'),
             'rope_zero_pe_mean': E(pe='mean', rope='zero')}
    if canvas // vl.PATCH <= vl.TABLE_SIDE:
        conds['pe_native'] = E(pe='native')
    # Qwen3.5's PE table is the Qwen3-VL table scaled by about 1/2.76 (see the table stage); these scales map
    # one model's PE norm onto the other's.
    conds['pe_x2.76'] = E(pe_scale=PE_RATIO)
    if family == 'qwen3_vl':                      # cheaper comparison set
        keep = ('base', 'pe_zero', 'pe_mean', 'pe_shuffle', 'rope_zero', 'rope_zero_pe_zero', 'pe_native', 'pe_x2')
        conds = {k: v for k, v in conds.items() if k in keep}
        conds['pe_x0.36'] = E(pe_scale=1 / PE_RATIO)
    return conds


NOISE_CONDITIONS = ('base', 'pe_zero', 'pe_mean', 'pe_shuffle', 'pe_x4', 'rope_zero', 'rope_zero_pe_zero')


def score_point(case: dict, text: str) -> dict:
    _, truth = cr.disc_image(tuple(case['center']), case['diameter'], case['canvas'])
    return cr.score_disc(truth, 'point', text)


# ----------------------------------------------------------------------------- table / norms

def stage_table(rt, out: Path, args) -> None:
    root = out / 'table'
    root.mkdir(parents=True, exist_ok=True)
    w = rt.model.model.visual.pos_embed.weight.detach().float().cpu()
    torch.save(w, root / f'{args.tag}_pe_table.pt')
    write_meta(root, rt, args, f'table_{args.tag}', {'shape': list(w.shape)})
    print('saved PE table', tuple(w.shape), flush=True)


def natural_images(args, split: str, n: int) -> list[tuple[int, Image.Image]]:
    """Square 512 crops of the experiment-3.4 RefCOCO images of one split (center crop, Lanczos)."""
    src = args.natural
    cases = [c for c in json.loads((src / 'cases.json').read_text())['natural'] if c['split'] == split][:n]
    out = []
    for c in cases:
        im = Image.open(src / 'images/natural512' / f"{c['id']:04d}.png").convert('RGB')
        s = min(im.size)
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width - s) // 2 + s, (im.height - s) // 2 + s))
        out.append((c['id'], im.resize((512, 512), Image.Resampling.LANCZOS)))
    if len(out) != n:
        raise ValueError(f'Only {len(out)} {split} images available, need {n}.')
    return out


def stage_norms(rt, out: Path, args) -> None:
    """Per ViT block: token norm, and relative change caused by removing / replacing the PE or RoPE."""
    root = out / 'norms'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{args.tag}_records.jsonl'
    finished = done_ids(path)
    E = vl.ViTEdit
    edits = {'pe_zero': E(pe='zero'), 'pe_mean': E(pe='mean'), 'pe_shuffle': E(pe='shuffle'),
             'rope_zero': E(rope='zero'), 'both_zero': E(pe='zero', rope='zero')}
    items = []
    for canvas in CANVASES:
        for case in vl.disc_grid_cases(canvas, args.seed)[:4]:
            items.append((f"disc_{case['id']}", cr.disc_image(tuple(case['center']), case['diameter'], canvas)[0]))
    for i, im in natural_images(args, 'test', 8):
        items.append((f'natural_{i:04d}', im))
    depth = len(rt.model.model.visual.blocks)
    for key, image in items:
        if key in finished:
            continue
        ref = vl.vit_capture(rt, [image])[0]
        stages = ['input'] + [f'block_{k}' for k in range(1, depth + 1)] + ['merger']
        rec = {'id': key, 'grid': list(ref['grid']),
               'content_norm': float(ref['content'].norm(dim=1).mean()), 'pe_norm': float(ref['pe'].norm(dim=1).mean()),
               'pe_content_ratio': float((ref['pe'].norm(dim=1) / ref['content'].norm(dim=1)).mean()),
               'pe_content_cos': float(torch.nn.functional.cosine_similarity(ref['pe'], ref['content'], dim=1).mean()),
               'norm': {s: float(ref[s].norm(dim=1).mean()) for s in stages}, 'change': {}}
        # Share of the block output that is the (linearly carried) initial PE: projection of h on its PE.
        for name, e in edits.items():
            alt = vl.vit_capture(rt, [image], edit=e)[0]
            rec['change'][name] = {s: float(((ref[s] - alt[s]).norm(dim=1) / ref[s].norm(dim=1)).mean()) for s in stages}
        append_jsonl(path, [rec])
        print(key, rec['grid'], f"pe/content {rec['pe_content_ratio']:.3f}",
              f"merger change pe0 {rec['change']['pe_zero']['merger']:.3f} rope0 {rec['change']['rope_zero']['merger']:.3f}",
              flush=True)
    write_meta(root, rt, args, f'norms_{args.tag}')


# ----------------------------------------------------------------------------- coordinate output

def stage_coord(rt, out: Path, args) -> None:
    root = out / 'coord'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{args.tag}_records.jsonl'
    finished = done_ids(path)
    plan = [(int(c), 'white') for c in args.canvases.split(',')]
    plan += [(int(c), 'noise') for c in args.noise_canvases.split(',') if c]
    for canvas, background in plan:
        cases = [dict(c, background=background, noise_seed=args.seed + i,
                      id=c['id'] if background == 'white' else c['id'] + '_noise')
                 for i, c in enumerate(vl.disc_grid_cases(canvas, args.seed))]
        conds = coord_conditions(rt.family, canvas)
        if background == 'noise':
            conds = {k: v for k, v in conds.items() if k in NOISE_CONDITIONS}
        for cname, edit in conds.items():
            todo = [c for c in cases if f"{c['id']}|{cname}" not in finished]
            for i in range(0, len(todo), args.batch_size):
                chunk = todo[i:i + args.batch_size]
                t0 = time.time()
                items = [([vl.disc_render(c)], POINT_PROMPT) for c in chunk]
                with vl.vit_edit(rt, edit):
                    res = qr.generate_batch(rt, items, args.max_new_tokens)
                rows = [{'id': f"{c['id']}|{cname}", 'case': c, 'condition': cname, 'edit': edit.as_dict(),
                         'text': r['text'], 'finish_reason': r['finish_reason'], 'score': score_point(c, r['text'])}
                        for c, r in zip(chunk, res)]
                append_jsonl(path, rows)
                print(canvas, background, cname, f'{len(chunk)} in {time.time() - t0:.1f}s',
                      sum(r['score']['hit'] for r in rows), 'hits', flush=True)
    write_meta(root, rt, args, f'coord_{args.tag}', {'batch_size': args.batch_size})


def stage_spot(rt, out: Path, args) -> None:
    """Batched generate vs single generate on random (case, condition) pairs of the coord stage."""
    root = out / 'coord'
    recs = read_jsonl(root / f'{args.tag}_records.jsonl')
    rng = np.random.default_rng(args.seed)
    pick = [recs[i] for i in rng.choice(len(recs), min(args.n_spot, len(recs)), replace=False)]
    rows = []
    for r in pick:
        c = r['case']
        image = vl.disc_render(c)
        with vl.vit_edit(rt, vl.ViTEdit(**r['edit'])):
            g = qr.generate(rt, qr.build(rt, [image], POINT_PROMPT), args.max_new_tokens)
        a, b = cr.parse_coords(r['text'], 'point'), cr.parse_coords(g['text'], 'point')
        rows.append({'id': r['id'], 'batched': r['text'], 'single': g['text'], 'same_text': r['text'] == g['text'],
                     'same_answer': a == b, 'max_abs_diff': None if (a is None or b is None) else
                     float(np.abs(np.asarray(a) - np.asarray(b)).max())})
    (root / f'{args.tag}_spotcheck.json').write_text(json.dumps(rows, indent=1) + '\n')
    print('spot-check same answer', sum(r['same_answer'] for r in rows), '/', len(rows), flush=True)


# ----------------------------------------------------------------------------- recognition control

def stage_t2(rt, out: Path, args) -> None:
    root = out / 't2'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{args.tag}_records.jsonl'
    finished = done_ids(path)
    for canvas in [int(c) for c in args.t2_canvases.split(',')]:
        cases = vl.t2_cases(canvas, args.seed)
        conds = coord_conditions(rt.family, canvas)
        for c in cases:
            image = vl.t2_render(c)
            batch = qr.build(rt, [image], vl.T2_PROMPT)
            rows = []
            for cname, edit in conds.items():
                if f"{c['id']}|{cname}" in finished:
                    continue
                with vl.vit_edit(rt, edit):
                    s = qr.yes_no(rt, batch)
                rows.append({'id': f"{c['id']}|{cname}", 'case': c, 'condition': cname, **s})
            append_jsonl(path, rows)
        print('t2', canvas, 'done', flush=True)
    write_meta(root, rt, args, f't2_{args.tag}')


# ----------------------------------------------------------------------------- swaps (3.5 engine)

def swap_target(case: dict, grid) -> tuple[list[int], list[dict]]:
    _, truth = cr.disc_image(tuple(case['center']), case['diameter'], case['canvas'])
    H, W = grid
    target = cr.tokens_overlapping(truth['box'], grid)
    rows, cols = sorted({t // W for t in target}), sorted({t % W for t in target})
    k = max(2, H // 4)
    regions = []
    for axis in ('h', 'v'):
        c = truth['center'][0] if axis == 'h' else truth['center'][1]
        toward = 1 if c < case['canvas'] / 2 else -1
        dr, dc = (0, toward * k) if axis == 'h' else (toward * k, 0)
        moved = [(t // W + dr) * W + (t % W + dc) for t in target]
        ok = all(0 <= r + dr < H for r in rows) and all(0 <= q + dc < W for q in cols) and not set(moved) & set(target)
        regions.append({'axis': axis, 'k': k, 'valid': ok, 'region': moved if ok else None,
                        'd': [1000 * dc / W, 1000 * dr / H]})
    return target, regions


def stage_swap(rt, out: Path, args) -> None:
    if rt.family != 'qwen3_5':
        raise ValueError('The swap stage uses the 3.5 engine, which supports Qwen3.5 only.')
    root = out / 'swap'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    for canvas in [int(c) for c in args.swap_canvases.split(',')]:
        for case in vl.disc_grid_cases(canvas, args.seed):
            if case['id'] in finished:
                continue
            t0 = time.time()
            image = cr.disc_image(tuple(case['center']), case['diameter'], canvas)[0]
            prep = cr.prepare(rt, image, POINT_PROMPT)
            target, regions = swap_target(case, prep.grid)
            zero = torch.zeros_like(prep.vit_pos)
            variants = {'std': (prep.vit_pos, 1.0), 'rz': (zero, 1.0), 'x4': (prep.vit_pos, 4.0), 'rzx4': (zero, 4.0)}
            conds = []
            for vname, (vp0, scale) in variants.items():
                ip0 = (prep.interp[0], prep.interp[1] * scale)
                conds.append((f'{vname}|base', vp0, ip0, None))
                for reg in regions:
                    if not reg['valid']:
                        continue
                    kinds = ('pe', 'rope', 'both') if vname == 'std' else ('pe',)
                    for kind in kinds:
                        vp, ip = cr.swap_vit(prep, target, reg['region'], kind in ('pe', 'both'), kind in ('rope', 'both'))
                        if vname in ('rz', 'rzx4'):
                            vp = zero
                        conds.append((f"{vname}|{kind}_{reg['axis']}", vp, (ip[0], ip[1] * scale), reg['d']))
            xs = [cr.embeds(rt, prep, cr.vision_embeds(rt, prep, vit_pos=vp, interp=ip)) for _, vp, ip, _ in conds]
            res = cr.run_conditions(rt, xs, [prep.pos] * len(xs), args.max_new_tokens, batch_size=args.batch_size)
            rows = []
            for (name, _, _, d), r in zip(conds, res):
                coords = cr.parse_coords(r['text'], 'point')
                rows.append({'condition': name, 'd': d, 'text': r['text'], 'finish_reason': r['finish_reason'],
                             'center': None if coords is None else cr.answer_center(coords).tolist(),
                             'score': score_point(case, r['text'])})
            append_jsonl(path, [{'id': case['id'], 'case': case, 'grid': list(prep.grid), 'target': target,
                                 'regions': regions, 'conditions': rows}])
            print(case['id'], f'{len(conds)} conds in {time.time() - t0:.1f}s', flush=True)
    write_meta(root, rt, args, 'swap', {'batch_size': args.batch_size})


# ----------------------------------------------------------------------------- probes

def probe_conditions(family: str) -> dict[str, vl.ViTEdit]:
    E = vl.ViTEdit
    conds = {'base': E(), 'pe_zero': E(pe='zero'), 'pe_mean': E(pe='mean'), 'rope_zero': E(rope='zero'),
             'both_zero': E(pe='zero', rope='zero')}
    if family == 'qwen3_5':
        conds['pe_shuffle'] = E(pe='shuffle')
    return conds


def stage_keys(depth: int) -> list[str]:
    return ['content', 'input'] + [f'block_{k}' for k in range(1, depth + 1)] + ['merger']


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def labels_for(key: str, grid) -> torch.Tensor:
    hp, wp = grid
    xy = vl.token_xy(hp // 2, wp // 2) if key == 'merger' else vl.patch_xy(hp, wp)
    return torch.from_numpy(xy)


def stage_probe(rt, out: Path, args) -> None:
    """Ridge position probes (x, y in 0-1000) per stage and ViT edit; position-variance share."""
    root = out / 'probe'
    root.mkdir(parents=True, exist_ok=True)
    dev = rt.device
    depth = len(rt.model.model.visual.blocks)
    keys = stage_keys(depth)
    data = {s: natural_images(args, s, n) for s, n in
            (('train', args.probe_train), ('validation', args.probe_val), ('test', args.probe_test))}
    for cname, edit in probe_conditions(rt.family).items():
        target = root / f'{args.tag}_{cname}.pt'
        if target.exists():
            continue
        t0 = time.time()
        mom = {s: {k: None for k in keys} for s in ('train', 'validation')}
        possum = {k: None for k in keys}
        for split in ('train', 'validation'):
            for chunk in batches(data[split], args.vit_batch):
                caps = vl.vit_capture(rt, [im for _, im in chunk], edit=edit)
                for cap in caps:
                    for k in keys:
                        x = cap[k].double()
                        y = labels_for(k, cap['grid']).to(dev).double()
                        if mom[split][k] is None:
                            mom[split][k] = Moments(x.shape[1], 2, dev)
                        mom[split][k].add(x, y)
                        if split == 'train':
                            possum[k] = x.clone() if possum[k] is None else possum[k] + x
        ntrain = len(data['train'])
        result = {'condition': cname, 'edit': edit.as_dict(), 'stages': {}}
        fits = {}
        for k in keys:
            tr, va = mom['train'][k].state(), mom['validation'][k].state()
            tr = {a: b.to(dev) for a, b in tr.items()}
            va = {a: b.to(dev) for a, b in va.items()}
            fit = ridge_path(tr, va, [0, 1])
            best = int(fit['val_mse'].mean(1).argmin())
            fits[k] = (fit['w'][best].float(), fit['b'][best].float())
            # Position-variance share: between-slot variance of the slot means / total variance (train),
            # with the finite-sample bias of the slot means removed.
            n = float(tr['n'])
            mu = tr['s'] / n
            total = float(torch.diagonal(tr['c']).sum() / n - mu @ mu)
            means = possum[k] / ntrain
            between = float(((means - mu) ** 2).sum(1).mean())
            between = (between - total / ntrain) / (1 - 1 / ntrain)
            centered = (means - means.mean(0)).float()
            ev = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0).flip(0)   # spectrum of the slot means
            result['stages'][k] = {'lambda': float(LAMBDAS[best]), 'val_mse': fit['val_mse'][best].tolist(),
                                   'total_var': total, 'between_var': between, 'pos_share': between / total,
                                   'slot_mean_pr': float(ev.sum() ** 2 / (ev ** 2).sum()),
                                   'slot_mean_top': (ev[:20] / ev.sum()).tolist()}
        # Test pass: predictions per stage.
        preds = {k: [] for k in keys}
        for chunk in batches(data['test'], args.vit_batch):
            for cap in vl.vit_capture(rt, [im for _, im in chunk], edit=edit):
                for k in keys:
                    w, b = fits[k]
                    preds[k].append((cap[k] @ w + b).cpu().half())
        for k in keys:
            p = torch.stack(preds[k]).float()                              # (n_test, N, 2); inputs are 512x512
            y = labels_for(k, (32, 32)).float()
            err = (p - y).abs()
            units = (16 if k == 'merger' else 32) / 1000
            result['stages'][k].update(mae=float(err.mean()), rmse=float(((p - y) ** 2).mean().sqrt()),
                                       mae_token=float(err.mean() * units),
                                       mae_per_image=err.mean((1, 2)).tolist(),
                                       r2=float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum() / len(p)))
            preds[k] = p.half()
        torch.save({'result': result, 'preds': preds, 'test_ids': [i for i, _ in data['test']],
                    'possum_merger': (possum['merger'] / ntrain).float().cpu()}, target)
        (root / f'{args.tag}_{cname}.json').write_text(json.dumps(result, indent=1) + '\n')
        print(cname, f'{time.time() - t0:.0f}s', {k: round(result['stages'][k]['mae'], 1) for k in
                                                  ('content', 'input', 'block_1', 'block_8', 'block_16', 'block_27', 'merger')},
              flush=True)
    write_meta(root, rt, args, f'probe_{args.tag}')


# ----------------------------------------------------------------------------- layout

LAYOUT_BLOCKS = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27)
READOUTS = ('red', 'blue', 'background', 'bg_single', 'pooled')


def layout_features(cap: dict, case: dict, key: str, rng: np.random.Generator, masks: dict) -> np.ndarray:
    """masks: block-order patch masks of the case (vl.object_patch_masks), computed once per image."""
    x = cap[key]
    if key == 'merger':                          # merged-token masks: any patch of the 2x2 block
        background = masks['background'].reshape(-1, 4).all(1)
        masks = {m: v.reshape(-1, 4).any(1) for m, v in masks.items()}
        masks['background'] = background
        # merger rows are in raster order of merged tokens, identical to block order of 2x2 blocks
    feats = []
    for name in ('red', 'blue', 'background'):
        m = torch.from_numpy(masks[name]).to(x.device)
        if not bool(m.any()):
            raise ValueError(f"{case['id']}: empty {name} mask at {key}")
        feats.append(x[m].mean(0))
    bg = np.flatnonzero(masks['background'])
    feats.append(x[int(bg[rng.integers(len(bg))])])
    feats.append(x.mean(0))
    return torch.stack(feats).cpu().half().numpy()


def stage_layout(rt, out: Path, args) -> None:
    root = out / 'layout'
    root.mkdir(parents=True, exist_ok=True)
    keys = ['content', 'input'] + [f'block_{k}' for k in LAYOUT_BLOCKS] + ['merger']
    cases = [vl.layout_case(i, args.seed) for i in range(args.layout_n)]
    (root / 'cases.json').write_text(json.dumps(cases) + '\n')
    conds = {'base': vl.ViTEdit(), 'pe_zero': vl.ViTEdit(pe='zero'), 'rope_zero': vl.ViTEdit(rope='zero')}
    if rt.family != 'qwen3_5':
        conds = {'base': conds['base']}
    for cname, edit in conds.items():
        target = root / f'{args.tag}_{cname}_features.npz'
        if target.exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng(args.seed)
        feats = {k: [] for k in keys}
        for chunk in batches(cases, args.vit_batch):
            caps = vl.vit_capture(rt, [vl.layout_render(c) for c in chunk], edit=edit)
            for cap, c in zip(caps, chunk):
                masks = vl.object_patch_masks(c, *cap['grid'])
                for k in keys:
                    feats[k].append(layout_features(cap, c, k, rng, masks))
        np.savez(target, **{k: np.stack(v) for k, v in feats.items()})
        print('layout', cname, f'{time.time() - t0:.0f}s', flush=True)
    write_meta(root, rt, args, f'layout_{args.tag}')


# ----------------------------------------------------------------------------- erasure

def merger_tokens(rt, images: list[Image.Image], batch: int) -> torch.Tensor:
    outs = []
    for chunk in batches(images, batch):
        outs += [c['merger'] for c in vl.vit_capture(rt, chunk, blocks=[])]
    return torch.stack(outs)                                              # (n, Nv, D) float32 GPU


def stage_leace(rt, out: Path, args) -> None:
    """Fit position erasers at the merger (512x512 inputs, 16x16 tokens) and test their effect."""
    root = out / 'leace'
    root.mkdir(parents=True, exist_ok=True)
    dev = rt.device
    fit_path = root / f'{args.tag}_erasers.pt'
    if not fit_path.exists():
        nat = [im for _, im in natural_images(args, 'train', args.leace_natural)]
        syn = [vl.layout_render(vl.layout_case(i, args.seed + 1)) for i in range(args.leace_synthetic)]
        n, D, K = 0, None, 256
        sx = sxx = sxz = sxy = None
        for chunk in batches(nat + syn, args.vit_batch):
            for m in merger_tokens(rt, chunk, args.vit_batch):
                x = m.double()
                if sx is None:
                    D = x.shape[1]
                    sx, sxx = torch.zeros(D, device=dev, dtype=torch.float64), torch.zeros(D, D, device=dev, dtype=torch.float64)
                    sxz, sxy = torch.zeros(D, K, device=dev, dtype=torch.float64), torch.zeros(D, 2, device=dev, dtype=torch.float64)
                n += len(x)
                sx += x.sum(0)
                sxx.addmm_(x.T, x)
                sxz += x.T                                                # one-hot slot concept: x^T I
                sxy.addmm_(x.T, torch.from_numpy(vl.token_xy(16, 16)).to(dev))
        n_img = n // K
        sz = torch.full((K,), float(n_img), device=dev, dtype=torch.float64)
        szy = torch.from_numpy(vl.token_xy(16, 16)).to(dev).sum(0) * n_img
        er = {'onehot': vl.LeaceEraser.fit(n, sx, sxx, sz, sxz),
              'xy': vl.LeaceEraser.fit(n, sx, sxx, szy, sxy)}
        er['random255'] = vl.LeaceEraser.fit(n, sx, sxx, sz, sxz, random_rank=er['onehot'].rank, seed=args.seed)
        er['random2'] = vl.LeaceEraser.fit(n, sx, sxx, szy, sxy, random_rank=2, seed=args.seed)
        # Slot-mean ablation: replace each token's slot mean by the global mean (non-whitened, rank <= 255).
        torch.save({k: {'mean': e.mean.cpu(), 'proj': e.proj.cpu(), 'rank': e.rank, 'singular': e.singular.cpu()}
                    for k, e in er.items()} | {'slot_means': (sxz / n_img).T.cpu(), 'mean': (sx / n).cpu(),
                                               'n_images': n_img}, fit_path)
        print('fitted erasers', {k: e.rank for k, e in er.items()}, flush=True)
    saved = torch.load(fit_path)
    erasers = {k: vl.LeaceEraser(v['mean'].to(dev), v['proj'].to(dev), v['rank'], v['singular'])
               for k, v in saved.items() if isinstance(v, dict) and 'proj' in v}
    slot_shift = (saved['slot_means'] - saved['mean']).to(dev)            # (256, D)

    def make_fn(name):
        def fn(m, grid):
            if any((h, w) != (32, 32) for _, h, w in grid.tolist()):
                raise ValueError('Erasers are fitted for 512x512 inputs (16x16 merged tokens).')
            if name == 'slotmean':
                return (m.double() - slot_shift.repeat(len(grid), 1)).to(m.dtype)
            return erasers[name](m)
        return fn

    conds = ['none', 'onehot', 'xy', 'random255', 'random2', 'slotmean']
    # (a) coordinate output on 512 discs of three diameters
    path = root / f'{args.tag}_coord.jsonl'
    finished = done_ids(path)
    cases = []
    for d in [int(v) for v in args.leace_diameters.split(',')]:
        for c in vl.disc_grid_cases(512, args.seed + d):
            c = dict(c, diameter=d, id=f"{c['id']}_d{d}")
            cases.append(c)
    for cname in conds:
        todo = [c for c in cases if f"{c['id']}|{cname}" not in finished]
        for chunk in batches(todo, args.batch_size):
            items = [([cr.disc_image(tuple(c['center']), c['diameter'], 512)[0]], POINT_PROMPT) for c in chunk]
            if cname == 'none':
                res = qr.generate_batch(rt, items, args.max_new_tokens)
            else:
                with vl.merger_edit(rt, make_fn(cname)):
                    res = qr.generate_batch(rt, items, args.max_new_tokens)
            append_jsonl(path, [{'id': f"{c['id']}|{cname}", 'case': c, 'condition': cname, 'text': r['text'],
                                 'finish_reason': r['finish_reason'], 'score': score_point(c, r['text'])}
                                for c, r in zip(chunk, res)])
        print('leace coord', cname, flush=True)
    # (b) T2 recognition at 512
    path = root / f'{args.tag}_t2.jsonl'
    finished = done_ids(path)
    for c in vl.t2_cases(512, args.seed):
        batch = qr.build(rt, [vl.t2_render(c)], vl.T2_PROMPT)
        rows = []
        for cname in conds:
            if f"{c['id']}|{cname}" in finished:
                continue
            if cname == 'none':
                s = qr.yes_no(rt, batch)
            else:
                with vl.merger_edit(rt, make_fn(cname)):
                    s = qr.yes_no(rt, batch)
            rows.append({'id': f"{c['id']}|{cname}", 'case': c, 'condition': cname, **s})
        append_jsonl(path, rows)
    print('leace t2 done', flush=True)
    # (c) linear position probes refit on erased merger tokens (natural images: train / test split)
    probe_path = root / f'{args.tag}_probe.json'
    if not probe_path.exists():
        tr = merger_tokens(rt, [im for _, im in natural_images(args, 'train', args.probe_train)], args.vit_batch)
        te = merger_tokens(rt, [im for _, im in natural_images(args, 'test', args.probe_test)], args.vit_batch)
        y = torch.from_numpy(vl.token_xy(16, 16)).float().to(dev)
        result = {}
        for cname in conds:
            f = (lambda m: m) if cname == 'none' else (lambda m, fn=make_fn(cname): fn(m, torch.tensor([[1, 32, 32]])))
            xtr = torch.cat([f(m) for m in tr]).float()
            xte = torch.stack([f(m) for m in te]).float()
            mu, ym = xtr.mean(0), y.mean(0)
            xc = xtr - mu
            yy = (y.repeat(len(tr), 1) - ym)
            best = None
            nval = len(xtr) // 5                                          # first fifth (64 images) for lambda
            gram, cross = xc[nval:].T @ xc[nval:], xc[nval:].T @ yy[nval:]
            for lam in (1e1, 1e2, 1e3, 1e4, 1e5):
                w = torch.linalg.solve(gram + lam * torch.eye(xc.shape[1], device=dev), cross)
                mse = float(((xc[:nval] @ w - yy[:nval]) ** 2).mean())
                if best is None or mse < best[0]:
                    best = (mse, lam)
            lam = best[1]
            w = torch.linalg.solve(xc.T @ xc + lam * torch.eye(xc.shape[1], device=dev), xc.T @ yy)
            p = (xte - mu) @ w + ym
            err = (p - y).abs()
            result[cname] = {'lambda': lam, 'mae': float(err.mean()), 'mae_token': float(err.mean() * 16 / 1000),
                             'pred_mean_map': p.mean(0).cpu().tolist()}
            print('leace probe', cname, round(result[cname]['mae'], 1), flush=True)
        probe_path.write_text(json.dumps(result) + '\n')
    write_meta(root, rt, args, f'leace_{args.tag}')


# ----------------------------------------------------------------------------- re-read pilot

def reread_cases(seed: int, n: int) -> list[dict]:
    """A 256x256 crop (32-aligned origin) of a 512 canvas; one disc of diameter 48 inside the crop."""
    rng = np.random.default_rng(seed + 99)
    out = []
    for i in range(n):
        ox, oy = (int(v) * 32 for v in rng.integers(0, 9, 2))
        cx, cy = (float(v) for v in rng.uniform(40, 216, 2))
        out.append({'id': f'reread_{i:02d}', 'origin': [ox, oy], 'local_center': [cx, cy],
                    'center': [ox + cx, oy + cy], 'diameter': 48, 'canvas': 512, 'crop': 256})
    return out


def stage_reread(rt, out: Path, args) -> None:
    if rt.family != 'qwen3_5':
        raise ValueError('The reread pilot uses the 3.5 engine (Qwen3.5 only).')
    root = out / 'reread'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    for case in reread_cases(args.seed, args.reread_n):
        if case['id'] in finished:
            continue
        full, _ = cr.disc_image(tuple(case['center']), case['diameter'], 512)
        ox, oy = case['origin']
        crop = full.crop((ox, oy, ox + 256, oy + 256))
        prep_full = cr.prepare(rt, full, POINT_PROMPT)
        prep = cr.prepare(rt, crop, POINT_PROMPT)
        r0, c0 = oy // 16, ox // 16                                       # crop origin in patches
        off = prep.vit_pos + torch.tensor([r0, c0], device=prep.vit_pos.device)
        win = vl.window_taps(16, 16, 32, 32, r0, c0, prep.interp[0].device)
        win = (win[0], win[1].to(prep.interp[1].dtype))
        sel = torch.tensor([(oy // 32 + r) * 16 + ox // 32 + c for r in range(8) for c in range(8)])
        vis = {
            'crop_local': prep.vis,
            'rope_offset': cr.vision_embeds(rt, prep, vit_pos=off),
            'pe_window': cr.vision_embeds(rt, prep, interp=win),
            'rope_offset_pe_window': cr.vision_embeds(rt, prep, vit_pos=off, interp=win),
            'pe_window_x4': cr.vision_embeds(rt, prep, interp=(win[0], win[1] * 4)),
            'rope_offset_pe_window_x4': cr.vision_embeds(rt, prep, vit_pos=off, interp=(win[0], win[1] * 4)),
            'canvas_context': prep_full.vis[sel.to(prep_full.vis.device)],
        }
        xs = [cr.embeds(rt, prep, v) for v in vis.values()] + [cr.embeds(rt, prep_full)]
        names = list(vis) + ['full_image']
        res = cr.run_conditions(rt, xs[:-1], [prep.pos] * (len(xs) - 1), args.max_new_tokens, batch_size=args.batch_size)
        res += cr.run_conditions(rt, xs[-1:], [prep_full.pos], args.max_new_tokens, batch_size=1)
        rows = []
        for name, r in zip(names, res):
            coords = cr.parse_coords(r['text'], 'point')
            rows.append({'condition': name, 'text': r['text'], 'finish_reason': r['finish_reason'],
                         'center': None if coords is None else cr.answer_center(coords).tolist()})
        append_jsonl(path, [{'id': case['id'], 'case': case,
                             'truth_local': [1000 * v / 256 for v in case['local_center']],
                             'truth_full': [1000 * v / 512 for v in case['center']], 'conditions': rows}])
        print(case['id'], [(r['condition'], r['center']) for r in rows][:3], flush=True)
    write_meta(root, rt, args, 'reread')


def stage_refcoco(rt, out: Path, args) -> None:
    """RefCOCO val bbox grounding (cases and images of experiment 4.2) under PE scaling."""
    from analysis_tools.deepstack.deepstack_comparison import _answer_natural, score_refcoco_case
    root = out / 'refcoco'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{args.tag}_records.jsonl'
    finished = done_ids(path)
    src = args.refcoco_cases
    cases = read_jsonl(src / 'refcoco_cases.jsonl')
    scales = {'base': 1.0, 'pe_x2': 2.0, 'pe_x2.76': PE_RATIO, 'pe_x4': 4.0} if rt.family == 'qwen3_5' else \
             {'base': 1.0, 'pe_x0.36': 1 / PE_RATIO, 'pe_x2': 2.0}
    cases.sort(key=lambda c: (c['height'], c['width']))
    for cname, scale in scales.items():
        todo = [c for c in cases if f"{c['id']}|{cname}" not in finished]
        for chunk in batches(todo, args.batch_size):
            for c in chunk:
                if hashlib.sha256((src / 'refcoco_images' / c['image']).read_bytes()).hexdigest() != c['image_sha256']:
                    raise ValueError(f"Image hash mismatch for {c['image']}")
            images = [Image.open(src / 'refcoco_images' / c['image']).convert('RGB') for c in chunk]
            with vl.vit_edit(rt, vl.ViTEdit(pe_scale=scale)) if scale != 1.0 else _null():
                recs = _answer_natural(rt, images, [c['prompt'] for c in chunk], args.max_new_tokens)
            append_jsonl(path, [{'id': f"{c['id']}|{cname}", 'case_id': c['id'], 'condition': cname, 'scale': scale,
                                 'stratum': c['stratum'], **r, 'score': score_refcoco_case(c, r)}
                                for c, r in zip(chunk, recs)])
        print('refcoco', cname, flush=True)
    write_meta(root, rt, args, f'refcoco_{args.tag}', {'batch_size': args.batch_size, 'source': str(src)})


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


STAGES = {'table': stage_table, 'norms': stage_norms, 'coord': stage_coord, 'spot': stage_spot, 't2': stage_t2,
          'swap': stage_swap, 'probe': stage_probe, 'layout': stage_layout, 'leace': stage_leace,
          'reread': stage_reread, 'refcoco': stage_refcoco}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--tag', required=True, help='short model tag used in artifact names')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--natural', type=Path, required=True, help='run directory of experiment 3.4 (RefCOCO crops)')
    p.add_argument('--stages', required=True)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--canvases', default='256,384,512,768,1024,1536')
    p.add_argument('--noise-canvases', default='256,512,1024')
    p.add_argument('--t2-canvases', default='256,512,768,1024,1536')
    p.add_argument('--swap-canvases', default='256,512,768,1024')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--vit-batch', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--n-spot', type=int, default=20)
    p.add_argument('--probe-train', type=int, default=320)
    p.add_argument('--probe-val', type=int, default=80)
    p.add_argument('--probe-test', type=int, default=100)
    p.add_argument('--layout-n', type=int, default=1600)
    p.add_argument('--leace-natural', type=int, default=320)
    p.add_argument('--leace-synthetic', type=int, default=320)
    p.add_argument('--leace-diameters', default='32,64,128')
    p.add_argument('--refcoco-cases', type=Path, help='experiment 4.2 RefCOCO directory (cases and images)')
    p.add_argument('--reread-n', type=int, default=48)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    for stage in args.stages.split(','):
        print(f'=== stage {stage}', flush=True)
        STAGES[stage](rt, args.output, args)


if __name__ == '__main__':
    main()
