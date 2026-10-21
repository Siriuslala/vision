"""Stage drivers of experiment 3.7 round 2 (synthetic vs natural images, PE vs RoPE reliance).

Stages (selected with --stages; generation stages are resumable per case):
  ladder    stimulus ladder x region interventions (RoPE / PE / both swaps, swap restricted to target or
            context tokens, content swap, real move, homogeneity-selected displaced regions), point task.
  edits     whole-image ViT position edits (PE zero / mean / shuffle / scale, RoPE zero / shuffle) on RefCOCO
            and on the referent pasted on white, bbox task.
  spot      batched-vs-single agreement and hook-vs-3.5-engine equivalence checks.
  capture   merger outputs of square (squashed) natural / flipped / shuffled / synthetic images, for the
            position-subspace analysis (base; position-free ViT for the natural sets).
  erase     LEACE position erasers fitted on natural / synthetic / mixed merger tokens; RefCOCO and disc
            point grounding after erasure (Qwen3.5).
  l20       decoder layer-20 attention (q, k, v) at the first x digit for RefCOCO and discs, with and
            without merger erasure (Qwen3.5).
  smalldisc PE edits for discs of 16-64 pixels (one instance per token) on white and noise (Qwen3.5).
  layout    round-1 layout features of Qwen3-VL under pe_zero and rope_zero.
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
from analysis_tools.position import vit_layout_r2 as v2
from analysis_tools.position import vit_layout_run as vr
from analysis_tools.position.coordinate_readout_run import append_jsonl, done_ids, read_jsonl

PE_RATIO = vr.PE_RATIO
L20 = 20


def code_digest() -> dict:
    here = Path(__file__).parent
    names = ('vit_layout.py', 'vit_layout_run.py', 'vit_layout_r2.py', 'vit_layout_r2_run.py', 'coordinate_readout.py')
    return {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in names}


def write_meta(out: Path, rt, args, stage: str, extra: dict | None = None) -> None:
    meta = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **qr.describe(rt), 'code': code_digest(),
            'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, **(extra or {})}
    (out / f'{stage}_meta.json').write_text(json.dumps(meta, indent=1) + '\n')


def cases_of(args) -> list[dict]:
    path = args.output / 'cases.json'
    if not path.exists():
        path.write_text(json.dumps(v2.load_cases(args.coord_cases, args.natural / 'cases.json', args.refcoco)) + '\n')
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------- ladder

def ladder_plan(case: dict, seed: int) -> dict | None:
    """Regions of every stimulus of one case; None when some stimulus has no valid displaced region."""
    W, H = case['canvas']
    grid = (H // v2.TOKEN, W // v2.TOKEN)
    plan = {}
    for stim in v2.STIMULI:
        target = cr.tokens_overlapping(v2.target_box(case, stim), grid)
        reg = v2.primary_region(target, grid)
        if reg is None:
            return None
        plan[stim] = {'target': target, 'primary': reg}
    if v2.VERTICAL_WORDS.search(case['expression']):
        plan['natural_vflip'] = None                       # flipped expression would change its referent
    nat = plan['natural']
    image = v2.load_image(case)
    nat['variants'] = v2.homogeneity_regions(nat['target'], grid, v2.token_texture(image), nat['primary']['axis'])
    return {'grid': list(grid), 'stimuli': plan}


def stimulus_rows(case: dict, stim: str, plan: dict, seed: int) -> list[v2.RowSpec]:
    """Rows of one stimulus; row 0 is the base, gathers reference rows of this group only."""
    image, truth, prompt = v2.render(case, stim, seed=seed)
    H, W = plan['grid']
    nv, npch = H * W, H * W * 4
    T = plan['stimuli'][stim]['target']
    rows = [v2.RowSpec(image, prompt, meta={'condition': 'base', 'truth': truth})]

    def add(cond: str, region: dict, tag: str = '') -> None:
        Tp = region['region']
        perm = v2.region_patch_perm(npch, T, Tp)
        meta = {'d': region['d'], 'axis': region['axis'], 'truth': truth, 'variant': tag or 'primary'}
        name = cond + (f'_{tag}' if tag else '')
        if cond == 'rope_swap':
            rows.append(v2.RowSpec(image, prompt, rope_perm=perm, meta={'condition': name, **meta}))
        elif cond == 'pe_swap':
            rows.append(v2.RowSpec(image, prompt, pe_perm=perm, meta={'condition': name, **meta}))
        elif cond == 'vit_swap':
            rows.append(v2.RowSpec(image, prompt, pe_perm=perm, rope_perm=perm, meta={'condition': name, **meta}))
        elif cond in ('vit_swap_target', 'vit_swap_context'):
            src = next(i for i, r in enumerate(rows) if r.meta['condition'] == 'vit_swap' + (f'_{tag}' if tag else ''))
            here = len(rows)
            group = set(T) | set(Tp)
            if cond == 'vit_swap_target':      # own forward = base; T and T' tokens from the swapped forward
                g = np.array([[src - here, j] if j in group else [0, j] for j in range(nv)])
                rows.append(v2.RowSpec(image, prompt, gather=g, meta={'condition': name, **meta}))
            else:                              # own forward = swapped; T and T' tokens from the base forward
                g = np.array([[0 - here, j] if j in group else [0, j] for j in range(nv)])
                rows.append(v2.RowSpec(image, prompt, pe_perm=perm, rope_perm=perm, gather=g,
                                       meta={'condition': name, **meta}))
        elif cond == 'content_swap':
            g = v2.own_gather(nv, v2.token_swap_order(nv, T, Tp))
            rows.append(v2.RowSpec(image, prompt, gather=g, meta={'condition': name, **meta}))
        elif cond == 'move':
            dx, dy = region['dc'] * v2.TOKEN, region['dr'] * v2.TOKEN
            img2, truth2, _ = v2.render(case, stim, shift=(dx, dy), seed=seed)
            rows.append(v2.RowSpec(img2, prompt, meta={'condition': name, **meta, 'truth': truth2}))
        else:
            raise ValueError(cond)

    prim = plan['stimuli'][stim]['primary']
    for cond in ('rope_swap', 'pe_swap', 'vit_swap', 'vit_swap_target', 'vit_swap_context', 'content_swap'):
        add(cond, prim)
    if stim in v2.MOVABLE:
        add('move', prim)
    if stim == 'natural':
        for tag, reg in plan['stimuli'][stim].get('variants', {}).items():
            for cond in ('rope_swap', 'vit_swap', 'content_swap'):
                add(cond, reg, tag)
    return rows


def readouts(merger: torch.Tensor, T: list[int], Tp: list[int] | None) -> np.ndarray:
    """(3, D) float16: mean merger token over T, over T' (zeros if None) and over the remaining tokens."""
    m = merger.float()
    n = len(m)
    group = set(T) | set(Tp or [])
    ctx = [j for j in range(n) if j not in group]
    out = [m[T].mean(0), m[Tp].mean(0) if Tp else torch.zeros(m.shape[1]), m[ctx].mean(0)]
    return torch.stack(out).half().numpy()


def stage_ladder(rt, out: Path, args) -> None:
    root = out / 'ladder'
    (root / 'feats').mkdir(parents=True, exist_ok=True)
    cases = cases_of(args)
    plan_path = root / 'plans.json'
    if not plan_path.exists():
        plans = {}
        for c in cases:
            p = ladder_plan(c, args.seed)
            plans[c['id']] = p
        plan_path.write_text(json.dumps(plans) + '\n')
    plans = json.loads(plan_path.read_text())
    valid = [c for c in cases if plans[c['id']] is not None][:args.ladder_n]
    print(f'ladder: {sum(p is not None for p in plans.values())}/{len(cases)} cases valid, using {len(valid)}', flush=True)
    path = root / f'{args.tag}_rows.jsonl'
    finished = done_ids(path)
    for c in valid:
        if c['id'] in finished:
            continue
        t0 = time.time()
        rng = np.random.default_rng([args.seed, c['source_case']])       # probe-token subsample of this case
        plan = plans[c['id']]
        groups = [(s, stimulus_rows(c, s, plan, args.seed)) for s in v2.STIMULI if plan['stimuli'][s] is not None]
        # pack whole stimulus groups into batches of at most args.batch_size rows
        batches, cur = [], []
        for s, rows in groups:
            if cur and sum(len(r) for _, r in cur) + len(rows) > args.batch_size:
                batches.append(cur)
                cur = []
            cur.append((s, rows))
        batches.append(cur)
        records, feats, base_tokens = [], {}, {}
        for b in batches:
            specs, owners = [], []
            for s, rows in b:
                start = len(specs)
                specs += rows
                owners += [(s, start)] * len(rows)
            res = v2.run_rows(rt, specs, args.max_new_tokens, keep_merger=True)
            for spec, (s, start), r in zip(specs, owners, res):
                meta = spec.meta
                score = v2.score_point(meta['truth'], r['text'])
                cond = meta['condition']
                T = plan['stimuli'][s]['target']
                Tp = None
                if 'variant' in meta:
                    Tp = (plan['stimuli'][s]['primary'] if meta['variant'] == 'primary'
                          else plan['stimuli'][s]['variants'][meta['variant']])['region']
                uid = f'{s}|{cond}'
                feats[uid] = readouts(r['merger'], T, Tp if Tp else plan['stimuli'][s]['primary']['region'])
                if cond == 'base':
                    m = r['merger']
                    pick = np.sort(rng.choice(len(m), size=min(args.probe_tokens, len(m)), replace=False))
                    base_tokens[s] = {'tokens': m[pick].clone(), 'index': torch.from_numpy(pick)}
                records.append({'stimulus': s, 'condition': cond, 'variant': meta.get('variant'), 'd': meta.get('d'),
                                'axis': meta.get('axis'), 'text': r['text'], 'finish_reason': r['finish_reason'],
                                'score': score, 'truth': meta['truth']})
        torch.save({'feats': feats, 'base_tokens': base_tokens}, root / 'feats' / f"{args.tag}_{c['id']}.pt")
        append_jsonl(path, [{'id': c['id'], 'grid': plan['grid'], 'rows': records,
                             'batch_size': args.batch_size, 'seconds': round(time.time() - t0, 1)}])
        print(f"ladder {c['id']} {len(records)} rows {time.time() - t0:.1f}s", flush=True)
    write_meta(root, rt, args, f'ladder_{args.tag}')


# ----------------------------------------------------------------------------- whole-image edits

def edit_conditions(family: str) -> dict[str, dict]:
    scale = PE_RATIO if family == 'qwen3_5' else 1 / PE_RATIO
    return {'base': {}, 'pe_zero': {'pe': 'zero'}, 'pe_mean': {'pe': 'mean'}, 'pe_shuffle': {'pe': 'shuffle'},
            'pe_scale': {'pe_scale': scale}, 'rope_zero': {'rope': 'zero'}, 'rope_shuffle': {'rope': 'shuffle'},
            'rope_zero_pe_zero': {'pe': 'zero', 'rope': 'zero'}}


def score_bbox(truth: dict, text: str) -> dict:
    coords = cr.parse_coords(text, 'bbox')
    if coords is None:
        return {'valid': False, 'correct': False, 'hit': False}
    W, H = truth['canvas']
    s = np.array([W, H, W, H]) / 1000
    box = (np.asarray(coords) * s).tolist()
    iou = cr.box_iou(box, truth['box'])
    c = cr.answer_center(coords)
    t = 1000 * np.asarray(truth['center']) / [W, H]
    px = c * [W / 1000, H / 1000]
    x0, y0, x1, y1 = truth['box']
    return {'valid': True, 'coords': coords, 'iou': iou, 'correct': iou >= 0.5, 'center': c.tolist(),
            'error': float(np.linalg.norm(c - t)), 'hit': bool(x0 <= px[0] <= x1 and y0 <= px[1] <= y1)}


def stage_edits(rt, out: Path, args) -> None:
    root = out / 'edits'
    root.mkdir(parents=True, exist_ok=True)
    cases = cases_of(args)[:args.edits_n]
    conds = edit_conditions(rt.family)
    path = root / f'{args.tag}_rows.jsonl'
    finished = done_ids(path)
    for c in cases:
        if c['id'] in finished:
            continue
        t0 = time.time()
        specs = []
        for stim in ('natural', 'obj_white'):
            image, truth, _ = v2.render(c, stim, seed=args.seed)
            prompt = v2.REFCOCO_QUERY.format(expr=c['expression']) + cr.SUFFIX['bbox']
            for name, kw in conds.items():
                specs.append(v2.RowSpec(image, prompt, meta={'stimulus': stim, 'condition': name, 'truth': truth,
                                                             'seed': args.seed}, **kw))
        res = []
        for i in range(0, len(specs), args.batch_size):
            res += v2.run_rows(rt, specs[i:i + args.batch_size], args.max_new_tokens)
        rows = [{'stimulus': s.meta['stimulus'], 'condition': s.meta['condition'], 'text': r['text'],
                 'finish_reason': r['finish_reason'], 'score': score_bbox(s.meta['truth'], r['text'])}
                for s, r in zip(specs, res)]
        append_jsonl(path, [{'id': c['id'], 'truth_box': c['box'], 'rows': rows, 'batch_size': args.batch_size}])
        print(f"edits {c['id']} {time.time() - t0:.1f}s", flush=True)
    write_meta(root, rt, args, f'edits_{args.tag}')


# ----------------------------------------------------------------------------- spot checks

def stage_spot(rt, out: Path, args) -> None:
    """(1) 20 random base ladder rows: batched answer vs reference generate; (2) Qwen3.5: merger output of
    the hook-based RoPE / both swaps vs the 3.5 engine (cr.swap_vit) on 5 cases."""
    root = out / 'ladder'
    path = root / f'{args.tag}_spot.jsonl'
    if path.exists():
        return
    cases = {c['id']: c for c in cases_of(args)}
    recs = read_jsonl(root / f'{args.tag}_rows.jsonl')
    rng = np.random.default_rng(args.seed + 1)
    rows = []
    for i in rng.choice(len(recs), size=min(args.n_spot, len(recs)), replace=False):
        rec = recs[int(i)]
        base = [r for r in rec['rows'] if r['condition'] == 'base']
        r = base[int(rng.integers(len(base)))]
        image, truth, prompt = v2.render(cases[rec['id']], r['stimulus'], seed=args.seed)
        ref = qr.generate(rt, qr.build(rt, [image], prompt), args.max_new_tokens)
        sa, sb = v2.score_point(truth, ref['text']), r['score']
        rows.append({'id': rec['id'], 'stimulus': r['stimulus'], 'reference': ref['text'], 'batched': r['text'],
                     'text_equal': ref['text'].strip() == r['text'].strip(),
                     'center_diff': (float(np.linalg.norm(np.subtract(sa['center'], sb['center'])))
                                     if sa.get('lenient_valid') and sb.get('lenient_valid') else None)})
    equiv = []
    if rt.family == 'qwen3_5':
        plans = json.loads((root / 'plans.json').read_text())
        for rec in recs[:5]:
            c, plan = cases[rec['id']], plans[rec['id']]
            image, truth, prompt = v2.render(c, 'natural', seed=args.seed)
            prep = cr.prepare(rt, image, prompt)
            T, Tp = plan['stimuli']['natural']['target'], plan['stimuli']['natural']['primary']['region']
            perm = v2.region_patch_perm(len(prep.vit_pos), T, Tp)
            for name, pe, rope in (('rope', False, True), ('both', True, True)):
                vp, ip = cr.swap_vit(prep, T, Tp, pe, rope)
                ref = cr.vision_embeds(rt, prep, vit_pos=vp, interp=ip).float()
                keep = []
                spec = v2.RowSpec(image, prompt, rope_perm=perm, pe_perm=perm if pe else None)
                with v2.row_hooks(rt, [spec], keep):
                    pixel, grid = vl.pixel_inputs(rt, [image])
                    rt.model.model.visual(pixel, grid_thw=grid)
                mine = keep[0]['merger'].float().to(ref.device)
                equiv.append({'id': rec['id'], 'swap': name,
                              'rel_max_diff': float((mine - ref).abs().max() / ref.abs().max())})
    append_jsonl(path, rows + [{'equivalence': equiv}])
    n_eq = sum(r['text_equal'] for r in rows)
    print(f'spot: {n_eq}/{len(rows)} text equal; equivalence {equiv}', flush=True)


# ----------------------------------------------------------------------------- captures for the subspace

SQUARE = 512


def square_natural(args, split: str, n: int) -> list[tuple[int, Image.Image]]:
    """Experiment-3.4 RefCOCO images of one split resized (squashed) to 512x512, keeping all content."""
    cases = [c for c in json.loads((args.natural / 'cases.json').read_text())['natural'] if c['split'] == split][:n]
    out = []
    for c in cases:
        im = Image.open(args.natural / 'images/natural512' / f"{c['id']:04d}.png").convert('RGB')
        out.append((c['id'], im.resize((SQUARE, SQUARE), Image.Resampling.LANCZOS)))
    if len(out) != n:
        raise ValueError(f'Only {len(out)} {split} images available, need {n}.')
    return out


def disc_cases_512(seed: int) -> list[dict]:
    """Round-1 erasure test discs: 512 canvas, diameters 32/64/128, 6x6 jittered positions."""
    rng = np.random.default_rng(seed)
    out = []
    for d in (32, 64, 128):
        for i, (cx, cy) in enumerate(cr.grid_centers(6, SQUARE)):
            jx, jy = rng.integers(-16, 17, 2).tolist()
            cx = float(np.clip(cx + jx, d / 2, SQUARE - d / 2))
            cy = float(np.clip(cy + jy, d / 2, SQUARE - d / 2))
            out.append({'id': f'd{d}_p{i:02d}', 'canvas': SQUARE, 'diameter': d, 'center': [cx, cy]})
    return out


def capture_sets(args) -> dict[str, list[Image.Image]]:
    test = square_natural(args, 'test', args.capture_test)
    return {
        'nat_train': [im for _, im in square_natural(args, 'train', args.capture_train)],
        'nat_test': [im for _, im in test],
        'nat_vflip': [im.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for _, im in test],
        'nat_shuffled': [v2.block_shuffle(im, args.seed + i) for i, (_, im) in enumerate(test)],
        'syn_layout': [vl.layout_render(vl.layout_case(i, args.seed)) for i in range(args.capture_train)],
        'syn_disc': [vl.disc_render(c) for c in disc_cases_512(args.seed)],
    }


def stage_capture(rt, out: Path, args) -> None:
    root = out / 'capture'
    root.mkdir(parents=True, exist_ok=True)
    sets = capture_sets(args)
    plan = [(s, 'base') for s in sets] + [(s, 'both_zero') for s in ('nat_train', 'nat_test', 'nat_vflip')]
    for s, cond in plan:
        target = root / f'{args.tag}_{s}_{cond}.pt'
        if target.exists():
            continue
        t0 = time.time()
        edit = vl.ViTEdit() if cond == 'base' else vl.ViTEdit(pe='zero', rope='zero')
        outs = []
        for chunk in vr.batches(sets[s], args.vit_batch):
            outs += [c['merger'].half().cpu() for c in vl.vit_capture(rt, chunk, blocks=[], edit=edit)]
        torch.save(torch.stack(outs), target)
        print('capture', s, cond, f'{time.time() - t0:.0f}s', flush=True)
    write_meta(root, rt, args, f'capture_{args.tag}')


# ----------------------------------------------------------------------------- erasure

def moments(x: torch.Tensor, z: torch.Tensor) -> tuple:
    """Raw float64 moments of tokens x (n, D) and concept z (n, k)."""
    x, z = x.double(), z.double().to(x.device)
    return len(x), x.sum(0), x.T @ x, z.sum(0), x.T @ z


def fit_erasers(root: Path, tag: str, device: str, seed: int) -> dict:
    cap = root.parent / 'capture'
    data = {s: torch.load(cap / f'{tag}_{s}_base.pt') for s in ('nat_train', 'syn_layout')}
    nslot = data['nat_train'].shape[1]
    hw = int(round(nslot ** 0.5))
    acc = {}
    for s, x in data.items():
        for kind in ('onehot', 'xy'):
            z = vl.position_features(hw, hw, kind)
            tot = None
            for i in range(0, len(x), 16):
                xb = x[i:i + 16].to(device).float().reshape(-1, x.shape[-1])
                zb = z.repeat(len(x[i:i + 16]), 1)
                m = moments(xb, zb)
                tot = m if tot is None else tuple(a + b for a, b in zip(tot, m))
            acc[(s, kind)] = tot
    def add(a, b):
        return tuple(x + y for x, y in zip(a, b))
    erasers = {
        'onehot_nat': vl.LeaceEraser.fit(*acc[('nat_train', 'onehot')]),
        'onehot_syn': vl.LeaceEraser.fit(*acc[('syn_layout', 'onehot')]),
        'onehot_mix': vl.LeaceEraser.fit(*add(acc[('nat_train', 'onehot')], acc[('syn_layout', 'onehot')])),
        'xy_nat': vl.LeaceEraser.fit(*acc[('nat_train', 'xy')]),
        'random_255': vl.LeaceEraser.fit(*add(acc[('nat_train', 'onehot')], acc[('syn_layout', 'onehot')]),
                                         random_rank=nslot - 1, seed=seed),
    }
    return erasers


def erase_items(args) -> list[dict]:
    """RefCOCO squashed test images (point task) and the 108 erasure-test discs."""
    meta = {c['id']: c for c in json.loads((args.natural / 'cases.json').read_text())['natural']}
    items = []
    for c in cases_of(args)[:args.erase_n]:
        src = meta[c['source_case']]
        W, H = c['canvas']
        sx, sy = SQUARE / W, SQUARE / H
        b = c['box']
        box = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        items.append({'id': c['id'], 'kind': 'refcoco', 'image_path': c['image_path'], 'expression': c['expression'],
                      'truth': {'canvas': [SQUARE, SQUARE], 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]},
                      'prompt': v2.REFCOCO_QUERY.format(expr=c['expression']) + cr.SUFFIX['point'],
                      'source_size': src['input_size']})
    for d in disc_cases_512(args.seed):
        r = d['diameter'] / 2
        items.append({'id': d['id'], 'kind': 'disc', 'case': d, 'prompt': v2.DISC_PROMPT,
                      'truth': {'canvas': [SQUARE, SQUARE], 'center': d['center'], 'diameter': d['diameter'],
                                'box': [d['center'][0] - r, d['center'][1] - r, d['center'][0] + r, d['center'][1] + r]}})
    return items


def item_image(item: dict) -> Image.Image:
    if item['kind'] == 'disc':
        return vl.disc_render(item['case'])
    return Image.open(item['image_path']).convert('RGB').resize((SQUARE, SQUARE), Image.Resampling.LANCZOS)


def stage_erase(rt, out: Path, args) -> None:
    root = out / 'erase'
    root.mkdir(parents=True, exist_ok=True)
    fit_path = root / f'{args.tag}_erasers.pt'
    if not fit_path.exists():
        er = fit_erasers(root, args.tag, rt.device, args.seed)
        torch.save({k: {'mean': e.mean.cpu(), 'proj': e.proj.cpu(), 'rank': e.rank, 'singular': e.singular.cpu()}
                    for k, e in er.items()}, fit_path)
    saved = torch.load(fit_path)
    erasers = {k: vl.LeaceEraser(v['mean'].to(rt.device), v['proj'].to(rt.device), v['rank'], v['singular'])
               for k, v in saved.items()}
    items = erase_items(args)
    path = root / f'{args.tag}_rows.jsonl'
    finished = {(r['id'], r['condition']) for r in read_jsonl(path)} if path.exists() else set()
    for name in ['none'] + list(erasers):
        todo = [it for it in items if (it['id'], name) not in finished]
        t0 = time.time()
        for chunk in vr.batches(todo, args.batch_size):
            imgs = [item_image(it) for it in chunk]
            gen = [([im], it['prompt']) for im, it in zip(imgs, chunk)]
            if name == 'none':
                res = qr.generate_batch(rt, gen, args.max_new_tokens)
            else:
                fn = erasers[name]
                with vl.merger_edit(rt, lambda m, g, fn=fn: fn(m)):
                    res = qr.generate_batch(rt, gen, args.max_new_tokens)
            append_jsonl(path, [{'id': it['id'], 'kind': it['kind'], 'condition': name, 'text': r['text'],
                                 'ids': rt.tokenizer.encode(r['text'], add_special_tokens=False),
                                 'finish_reason': r['finish_reason'], 'score': v2.score_point(it['truth'], r['text']),
                                 'batch_size': args.batch_size} for it, r in zip(chunk, res)])
        print('erase', name, f'{len(todo)} items {time.time() - t0:.0f}s', flush=True)
    write_meta(root, rt, args, f'erase_{args.tag}')


# ----------------------------------------------------------------------------- decoder layer 20 attention

@torch.inference_mode()
def l20_capture(rt, x: torch.Tensor, pos: torch.Tensor, query: int, vis: tuple[int, int]) -> dict:
    """q (heads, hd) and gate at the query, rotated k / v of the image tokens (kv heads, Nv, hd), and the
    softmax row over the whole prefix, recomputed from the inputs of layer-20 self-attention."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    mod = rt.layers[L20 - 1].self_attn
    store = {}

    def pre(module, args_, kwargs):
        store['h'] = kwargs['hidden_states'] if 'hidden_states' in kwargs else args_[0]
        store['pe'] = kwargs['position_embeddings']
        raise StopForward

    h = mod.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        rt.model(inputs_embeds=x[:, :query + 1], position_ids=pos[:, :, :query + 1], use_cache=False, logits_to_keep=1)
    except StopForward:
        pass
    finally:
        h.remove()
    hs, (cos, sin) = store['h'], store['pe']
    L, hd = hs.shape[1], mod.head_dim
    qg = mod.q_proj(hs[:, query:query + 1]).view(1, 1, -1, hd * 2)
    q, gate = torch.chunk(qg, 2, dim=-1)
    q = mod.q_norm(q).transpose(1, 2)
    k = mod.k_norm(mod.k_proj(hs).view(1, L, -1, hd)).transpose(1, 2)
    v = mod.v_proj(hs).view(1, L, -1, hd).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, q, cos[:, query:query + 1], sin[:, query:query + 1])
    _, k = apply_rotary_pos_emb(k, k, cos, sin)
    groups = mod.num_key_value_groups
    kr = k.repeat_interleave(groups, 1)
    scores = (q.float() @ kr.float().transpose(2, 3))[0, :, 0] * mod.scaling          # (heads, L)
    a = scores.softmax(-1)
    s, e = vis
    return {'q': q[0, :, 0].half().cpu(), 'gate': torch.sigmoid(gate.float())[0, 0].half().cpu(),
            'k': k[0, :, s:e].half().cpu(), 'v': v[0, :, s:e].half().cpu(),
            'attn_img': a[:, s:e].half().cpu(), 'scores_img': scores[:, s:e].half().cpu()}


class StopForward(Exception):
    pass


def stage_l20(rt, out: Path, args) -> None:
    root = out / 'l20'
    root.mkdir(parents=True, exist_ok=True)
    saved = torch.load(out / 'erase' / f'{args.tag}_erasers.pt')
    eraser = vl.LeaceEraser(saved['onehot_mix']['mean'].to(rt.device), saved['onehot_mix']['proj'].to(rt.device),
                            saved['onehot_mix']['rank'], saved['onehot_mix']['singular'])
    base = {r['id']: r for r in read_jsonl(out / 'erase' / f'{args.tag}_rows.jsonl') if r['condition'] == 'none'}
    items = erase_items(args)
    for cond in ('none', 'onehot_mix'):
        target = root / f'{args.tag}_{cond}.pt'
        if target.exists():
            continue
        t0 = time.time()
        recs = []
        for it in items:
            rec = base[it['id']]
            ids = rec['ids']
            toks = cr.coordinate_tokens(rt, ids, 'point')
            if not toks:
                recs.append({'id': it['id'], 'status': 'unparsed'})
                continue
            prep = cr.prepare(rt, item_image(it), it['prompt'])
            vis = prep.vis if cond == 'none' else eraser(prep.vis)
            x, pos = cr.teacher_forced(rt, prep, ids, cr.embeds(rt, prep, vis))
            L = prep.ids.shape[0]
            query = L - 1 + toks[0][0]                       # position whose next token is the first x digit
            cap = l20_capture(rt, x, pos, query, (prep.span['start'], prep.span['end']))
            recs.append({'id': it['id'], 'kind': it['kind'], 'status': 'ok', 'grid': list(prep.grid),
                         'center': rec['score'].get('center'), 'truth': it['truth'], **cap})
        torch.save(recs, target)
        print('l20', cond, f'{time.time() - t0:.0f}s', flush=True)
    write_meta(root, rt, args, f'l20_{args.tag}')


# ----------------------------------------------------------------------------- small discs (Q4)

def stage_smalldisc(rt, out: Path, args) -> None:
    root = out / 'smalldisc'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{args.tag}_rows.jsonl'
    if path.exists():
        return
    rng = np.random.default_rng(args.seed + 5)
    conds = {'base': {}, 'pe_zero': {'pe': 'zero'}, 'pe_shuffle': {'pe': 'shuffle'}, 'pe_scale': {'pe_scale': PE_RATIO}}
    specs = []
    for d in (16, 24, 32, 64):
        for i, (cx, cy) in enumerate(cr.grid_centers(6, SQUARE)):
            jx, jy = rng.integers(-16, 17, 2).tolist()
            case = {'id': f'd{d}_p{i:02d}', 'canvas': SQUARE, 'diameter': d,
                    'center': [float(np.clip(cx + jx, d, SQUARE - d)), float(np.clip(cy + jy, d, SQUARE - d))]}
            for bg in ('white', 'noise'):
                c2 = {**case, 'background': bg, 'noise_seed': int(rng.integers(2 ** 31))}
                image = vl.disc_render(c2)
                r = d / 2
                truth = {'canvas': [SQUARE, SQUARE], 'center': case['center'], 'diameter': d,
                         'box': [case['center'][0] - r, case['center'][1] - r, case['center'][0] + r, case['center'][1] + r]}
                for name, kw in conds.items():
                    specs.append(v2.RowSpec(image, v2.DISC_PROMPT, meta={'case': c2, 'condition': name, 'truth': truth,
                                                                         'seed': args.seed}, **kw))
    t0 = time.time()
    for i in range(0, len(specs), args.batch_size):
        chunk = specs[i:i + args.batch_size]
        res = v2.run_rows(rt, chunk, args.max_new_tokens)
        append_jsonl(path, [{'case': s.meta['case'], 'condition': s.meta['condition'], 'text': r['text'],
                             'finish_reason': r['finish_reason'], 'score': v2.score_point(s.meta['truth'], r['text'])}
                            for s, r in zip(chunk, res)])
    print('smalldisc', len(specs), f'{time.time() - t0:.0f}s', flush=True)
    write_meta(root, rt, args, f'smalldisc_{args.tag}')


# ----------------------------------------------------------------------------- layout for Qwen3-VL

def stage_layout(rt, out: Path, args) -> None:
    root = out / 'layout'
    root.mkdir(parents=True, exist_ok=True)
    keys = ['content', 'input'] + [f'block_{k}' for k in vr.LAYOUT_BLOCKS] + ['merger']
    cases = [vl.layout_case(i, args.layout_seed) for i in range(args.layout_n)]
    (root / 'cases.json').write_text(json.dumps(cases) + '\n')
    for cname, edit in {'pe_zero': vl.ViTEdit(pe='zero'), 'rope_zero': vl.ViTEdit(rope='zero')}.items():
        target = root / f'{args.tag}_{cname}_features.npz'
        if target.exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng(args.layout_seed)
        feats = {k: [] for k in keys}
        for chunk in vr.batches(cases, args.vit_batch):
            caps = vl.vit_capture(rt, [vl.layout_render(c) for c in chunk], edit=edit)
            for cap, c in zip(caps, chunk):
                masks = vl.object_patch_masks(c, *cap['grid'])
                for k in keys:
                    feats[k].append(vr.layout_features(cap, c, k, rng, masks))
        np.savez(target, **{k: np.stack(v) for k, v in feats.items()})
        print('layout', cname, f'{time.time() - t0:.0f}s', flush=True)
    write_meta(root, rt, args, f'layout_{args.tag}')


STAGES = {'ladder': stage_ladder, 'edits': stage_edits, 'spot': stage_spot, 'capture': stage_capture,
          'erase': stage_erase, 'l20': stage_l20, 'smalldisc': stage_smalldisc, 'layout': stage_layout}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--natural', type=Path, required=True, help='experiment-3.4 run directory (RefCOCO images)')
    p.add_argument('--coord-cases', type=Path, required=True, help='experiment-3.5 refcoco/cases.json')
    p.add_argument('--refcoco', type=Path, required=True, help='RefCOCO parquet root')
    p.add_argument('--stages', required=True)
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--layout-seed', type=int, default=20261015, help='round-1 layout seed (same cases)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--vit-batch', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=40)
    p.add_argument('--ladder-n', type=int, default=120)
    p.add_argument('--probe-tokens', type=int, default=64)
    p.add_argument('--edits-n', type=int, default=300)
    p.add_argument('--capture-train', type=int, default=320)
    p.add_argument('--capture-test', type=int, default=100)
    p.add_argument('--erase-n', type=int, default=300)
    p.add_argument('--layout-n', type=int, default=1600)
    p.add_argument('--n-spot', type=int, default=20)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    for stage in args.stages.split(','):
        STAGES[stage](rt, args.output, args)


if __name__ == '__main__':
    main()
