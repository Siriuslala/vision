"""Experiment 3.9 stage drivers: coordinate readout tool for Qwen3.5-9B.

Stages (comma list via --stages):
  prepare    build every case (RefCOCO train / val / test pools, generic synthetic scenes, held-out
             synthetic sets) and write cases.jsonl plus resized / rendered images (CPU only).
  verify     batched-vs-single equivalence of the capture pass on a few mixed-size samples.
  capture    teacher-forced tool-call pass; residual states at the readout sites and capture layers.
  own        the model's own point and bbox answers on every evaluation case (matched prompt).
  spotcheck  answer-level agreement of `generate_batch` with the reference `generate`.
  loop       (after fitting) the full tool loop: row/column answers with tool or own coordinates in
             the tool response, direct and think-then-answer; free (model-generated) tool calls.
Fitting, evaluation and figures live in `readout_tool_analysis.py` / `readout_tool_figures.py`.
Design: docs/analysis/position/design_readout_tool.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np
from PIL import Image

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

SEED = 20261016
TRAIN_SCALES = (384, 512, 640, 768)
N_EXTRA_TRAIN_IMAGES = 2600
N_EXTRA_TEST_IMAGES = 700
N_SYNTH = {'train': 5000, 'val': 500, 'test': 500}
CHUNK = 512
MIN_PIXELS = 65536                  # processor shortest_edge: smaller images are upscaled
RC_LAYOUT_PER = 96                  # row/column images per supplement layout in the loop stage
GRID_SENTENCE = ' The figures form a grid of {rows} rows and {cols} columns.'
RC_QUESTION = 'One figure is drawn in red. Give its position.'
RC_DIRECT = (' Return only {"row": r, "column": c}, counting rows from top to bottom and columns from left to '
             'right, starting at 1. Output only the JSON, with no explanation.')
RC_TOOL = ' Call the locate tool to find the red figure first.'


def code_digest() -> dict:
    here = Path(__file__).parent
    names = ('readout_tool.py', 'readout_tool_run.py')
    return {n: hashlib.sha256((here / n).read_bytes()).hexdigest()[:16] for n in names}


def write_meta(out: Path, name: str, payload: dict) -> None:
    payload = dict(payload, created_utc=datetime.now(timezone.utc).isoformat(), code=code_digest())
    (out / f'{name}.json').write_text(json.dumps(payload, indent=1) + '\n')


# ----------------------------------------------------------------------------- prepare

def _render_synth(args):
    split, k, seed, root = args
    image, truth = rt_.generic_scene(seed)
    rel = f'synth/{split}/{k:05d}.png'
    image.save(Path(root) / rel)
    hit = truth.pop('hit')
    target = truth.pop('target')
    return {'id': f'syn_{split}_{k:05d}', 'set': 'synth', 'split': split, 'subset': f"n{min(truth['n_objects'], 8)}",
            'image': rel, 'image_root': 'run', 'target': target, 'truth': dict(truth, hit=hit), 'seed': seed}


def prepare_refcoco(args, root: Path) -> list[dict]:
    meta = json.loads((args.refcoco_source / 'cases.json').read_text())['natural']
    split34 = {c['source_sha256']: c['split'] for c in meta}
    image_of, refs = rt_.refcoco_index(args.refcoco)
    unused = sorted(d for d in image_of if d not in split34)
    rng = np.random.default_rng(SEED)
    order = [unused[i] for i in rng.permutation(len(unused))]
    split = {d: {'train': 'train', 'validation': 'val', 'test': 'test'}[s] for d, s in split34.items()}
    origin = {d: '3.4' for d in split34}
    for d in order[:N_EXTRA_TRAIN_IMAGES]:
        split[d], origin[d] = 'train', 'extra'
    for d in order[N_EXTRA_TRAIN_IMAGES:N_EXTRA_TRAIN_IMAGES + N_EXTRA_TEST_IMAGES]:
        split[d], origin[d] = 'test', 'extra'
    by_digest = {}
    for r in refs:
        if r['digest'] in split:
            by_digest.setdefault(r['digest'], []).append(r)
    (root / 'refcoco').mkdir(parents=True, exist_ok=True)
    cases, cache = [], {}
    digests = sorted(by_digest, key=lambda d: image_of[d])
    for n, d in enumerate(digests):
        shard, row = image_of[d]
        src = rt_.load_refcoco_image(args.refcoco, shard, row, cache)
        if max(src.size) / min(src.size) > 3:
            continue
        scale = int(rng.choice(TRAIN_SCALES)) if split[d] == 'train' else 512
        img = rt_.resize_long(src, scale)
        while img.width * img.height < MIN_PIXELS:      # the processor would upscale smaller images
            scale += 128
            img = rt_.resize_long(src, scale)
        rel = f'refcoco/{d[:16]}_{scale}.png'
        if not (root / rel).exists():
            img.save(root / rel)
        for r in by_digest[d]:
            truth = rt_.refcoco_truth(r, src.size, img.size)
            subset = {'testA-00000-of-00001.parquet': 'testA', 'testB-00000-of-00001.parquet': 'testB'}.get(r['shard'], r['shard'].split('-')[0])
            cases.append({'id': f"rc_{r['question_id']}_{d[:8]}", 'set': 'refcoco', 'split': split[d], 'subset': subset,
                          'origin': origin[d], 'image': rel, 'image_root': 'run', 'target': r['expression'],
                          'truth': truth, 'digest': d, 'source_size': list(src.size), 'scale': scale})
        if n % 500 == 0:
            print(f'refcoco images {n}/{len(digests)}', flush=True)
    return cases


def heldout_cases(args) -> list[dict]:
    """Experiment-3.6 discs, experiment-1.3 T6 grids, 1.3-supplement layouts and rings (all test)."""
    from analysis_tools.position import coordinate_value as cv
    from analysis_tools.position import coordinate_readout as cr
    out = []
    for c in cv.disc_cases():
        _, truth = cr.disc_image(tuple(c['center']), c['diameter'], c['canvas'])
        out.append({'id': f"disc_{c['id']}", 'set': 'disc', 'split': 'test', 'subset': c['set'], 'render': 'disc',
                    'center_px': c['center'], 'diameter': c['diameter'], 'canvas_px': c['canvas'],
                    'target': rt_.DISC_TARGET, 'truth': rt_.disc_truth(truth)})
    t6 = Path(args.t6_source)
    rows = [r for r in read_jsonl(t6 / 'cases.jsonl') if r['task'] == 'T6' and r['stage'] == 'main'
            and r['wording'] == 'primary']
    results = {r['id']: r for r in read_jsonl(t6 / 'results_main.jsonl')}
    images = {}
    for r in rows:
        key = r['image']
        e = images.setdefault(key, {'id': f"t6_{Path(key).stem}", 'set': 't6', 'split': 'test', 'subset': f"g{r['grid']:02d}",
                                    'image': str(Path(t6.name) / key), 'image_root': 'behavior', 'image_sha256': r['image_sha256'],
                                    'target': rt_.T6_TARGET, 'truth': rt_.grid_truth(r['truth']), 'prior': {}})
        e['prior'][r['format']] = {'prompt': r['prompt'], 'response': results[r['id']]['response']}
    out += list(images.values())
    sup = Path(args.supplement_source)
    for name, set_name in (('layouts', 'layout'), ('ring', 'ring')):
        rows = [r for r in read_jsonl(sup / f'cases_{name}.jsonl') if r['task'] == 'T6' and not r['thinking']]
        results = {r['id']: r for r in read_jsonl(sup / f'results_{name}.jsonl')}
        images = {}
        for r in rows:
            key = r['image']
            e = images.setdefault(key, {'id': f"{set_name}_{Path(key).stem}", 'set': set_name, 'split': 'test',
                                        'subset': r['layout'], 'image': str(Path(sup.name) / key), 'image_root': 'behavior',
                                        'image_sha256': r['image_sha256'], 'target': rt_.T6_TARGET,
                                        'truth': rt_.grid_truth(r['truth']), 'prior': {}})
            if r['id'] in results:
                e['prior'][r['format']] = {'prompt': r['prompt'], 'response': results[r['id']]['response']}
        out += list(images.values())
    return out


def fix_small_refcoco(args) -> None:
    """Re-resize RefCOCO cases whose image falls below MIN_PIXELS (same rule as prepare_refcoco).

    Applied once to the 2026-10-16 run, whose first prepare lacked the MIN_PIXELS rule (9 training cases).
    """
    path = args.output / 'cases.jsonl'
    cases = read_jsonl(path)
    image_of, refs = None, None
    cache, changed = {}, []
    for c in cases:
        W, H = c['truth']['canvas']
        if c['set'] != 'refcoco' or W * H >= MIN_PIXELS:
            continue
        if image_of is None:
            image_of, refs = rt_.refcoco_index(args.refcoco)
            refs = {(r['digest'], r['question_id']): r for r in refs}
        src = rt_.load_refcoco_image(args.refcoco, *image_of[c['digest']], cache)
        scale = c['scale']
        img = rt_.resize_long(src, scale)
        while img.width * img.height < MIN_PIXELS:
            scale += 128
            img = rt_.resize_long(src, scale)
        rel = f"refcoco/{c['digest'][:16]}_{scale}.png"
        img.save(args.images / rel)
        ref = refs[(c['digest'], c['id'].split('_')[1])]
        c.update(image=rel, scale=scale, truth=rt_.refcoco_truth(ref, src.size, img.size))
        changed.append(c['id'])
    path.write_text(''.join(json.dumps(c, sort_keys=True) + '\n' for c in cases))
    write_meta(args.output, 'fix_small_refcoco', {'changed': changed})
    print(f'fixed {len(changed)} small RefCOCO cases', flush=True)


def stage_prepare(args) -> None:
    out, root = args.output, args.images
    path = out / 'cases.jsonl'
    if path.exists():
        print('cases.jsonl exists; prepare is skipped (never overwrites a run).', flush=True)
        return
    for split in N_SYNTH:
        (root / 'synth' / split).mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (split, n) in enumerate(N_SYNTH.items()):
        jobs += [(split, k, SEED * 10 + i * 1_000_000 + k, str(root)) for k in range(n)]
    with Pool(args.workers) as pool:
        synth = pool.map(_render_synth, jobs, chunksize=32)
    print(f'synthetic scenes: {len(synth)}', flush=True)
    cases = prepare_refcoco(args, root) + synth + heldout_cases(args)
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids):
        dup = [k for k, v in Counter(ids).items() if v > 1][:5]
        raise ValueError(f'Duplicate case ids: {dup}')
    path.write_text(''.join(json.dumps(c, sort_keys=True) + '\n' for c in cases))
    write_meta(out, 'prepare', {'counts': Counter(f"{c['set']}/{c['split']}" for c in cases), 'seed': SEED,
                                'train_scales': TRAIN_SCALES, 'n_extra_train_images': N_EXTRA_TRAIN_IMAGES,
                                'n_extra_test_images': N_EXTRA_TEST_IMAGES, 'n_synth': N_SYNTH})
    print(json.dumps(Counter(f"{c['set']}/{c['split']}" for c in cases)), flush=True)


# ----------------------------------------------------------------------------- shared helpers

def load_image(case: dict, args) -> Image.Image:
    if case.get('render') == 'disc':
        from analysis_tools.position import coordinate_readout as cr
        return cr.disc_image(tuple(case['center_px']), case['diameter'], case['canvas_px'])[0]
    base = {'run': args.images, 'behavior': args.behavior_root}[case['image_root']]
    p = base / case['image']
    if 'image_sha256' in case and hashlib.sha256(p.read_bytes()).hexdigest() != case['image_sha256']:
        raise ValueError(f'Image hash mismatch for {p}')
    return Image.open(p).convert('RGB')


def n_visual(case: dict) -> int:
    W, H = case['truth']['canvas']
    return (W // 32) * (H // 32)


def batches(cases: list[dict], max_tokens: int, max_batch: int):
    """Cases sorted by visual token count, grouped under a padded-token budget."""
    cases = sorted(cases, key=n_visual)
    cur = []
    for c in cases:
        if cur and (len(cur) + 1) * max(n_visual(c), 1) + 200 * (len(cur) + 1) > max_tokens or len(cur) == max_batch:
            yield cur
            cur = []
        cur.append(c)
    if cur:
        yield cur


def load_cases(args) -> list[dict]:
    return read_jsonl(args.output / 'cases.jsonl')


# ----------------------------------------------------------------------------- capture

def capture_inputs(rt, case: dict, args):
    user = rt_.QUERY.format(target=case['target']) + rt_.TOOL_INSTRUCTION
    text = rt_.tool_prompt(rt, user) + rt_.call_text(case['target'])
    _, offsets = rt_.call_sites(rt.tokenizer, case['target'])
    return text, load_image(case, args), offsets


def stage_verify(rt, args) -> None:
    import torch
    cases = load_cases(args)
    rng = np.random.default_rng(SEED)
    pick = [cases[i] for i in rng.choice(len(cases), 8, replace=False)]
    for c in pick[:3]:
        rt_.verify_call_tokenization(rt, rt_.QUERY.format(target=c['target']) + rt_.TOOL_INSTRUCTION, c['target'])
    inputs = [capture_inputs(rt, c, args) for c in pick]
    together = rt_.capture_sites(rt, [i[0] for i in inputs], [i[1] for i in inputs], [i[2] for i in inputs])
    rows = []
    for k, i in enumerate(inputs):
        alone = rt_.capture_sites(rt, [i[0]], [i[1]], [i[2]])[0].astype(np.float32)
        b = together[k].astype(np.float32)
        rel = np.linalg.norm(b - alone, axis=-1) / np.linalg.norm(alone, axis=-1)
        rows.append({'id': pick[k]['id'], 'rel_error_max': float(rel.max()), 'rel_error_mean': float(rel.mean()),
                     'cos_min': float(((b * alone).sum(-1) / np.linalg.norm(b, axis=-1) / np.linalg.norm(alone, axis=-1)).min())})
    write_meta(args.output, 'verify', {'samples': rows})
    print(json.dumps(rows, indent=1), flush=True)
    del torch


MAP_SITE = 'target_end'              # site whose full-attention maps over visual tokens are stored


def stage_capture(rt, args) -> None:
    """Hidden states (sites x layers), attention summaries (sites x full layers x heads x 7) and the
    target_end attention maps over visual tokens (full layers x heads, Nv) for every case."""
    feat = args.features
    feat.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    done = set()
    for f in sorted(feat.glob('chunk_*.npz')):
        done |= set(np.load(f)['ids'].tolist())
    todo = [c for c in cases if c['id'] not in done]
    print(f'capture: {len(todo)} of {len(cases)} cases to do', flush=True)
    k0 = len(list(feat.glob('chunk_*.npz')))
    t0, n = time.time(), 0
    for chunk_start in range(0, len(todo), CHUNK):
        chunk = todo[chunk_start:chunk_start + CHUNK]
        ids, xs, fs, maps = [], [], [], []
        for b in batches(chunk, args.max_tokens, args.max_batch):
            inputs = [capture_inputs(rt, c, args) for c in b]
            x, f, m = rt_.capture_sites(rt, [i[0] for i in inputs], [i[1] for i in inputs], [i[2] for i in inputs],
                                        attn_layers=rt_.ATTN_LAYERS, map_site=MAP_SITE)
            ids += [c['id'] for c in b]
            xs.append(x)
            fs.append(f.astype(np.float32))
            maps += m
            n += len(b)
        lengths = np.asarray([m.shape[1] for m in maps])
        np.savez(feat / f'chunk_{k0:05d}.npz', ids=np.asarray(ids), x=np.concatenate(xs), attn=np.concatenate(fs),
                 maps=np.concatenate(maps, 1), map_len=lengths)
        k0 += 1
        print(f'captured {n}/{len(todo)}  {(time.time() - t0) / n:.3f} s/case', flush=True)
    write_meta(args.output, 'capture', {'layers': rt_.CAPTURE_LAYERS, 'sites': rt_.SITES, 'attn_layers': rt_.ATTN_LAYERS,
                                        'attn_features': rt_.ATTN_FEATURES, 'map_site': MAP_SITE,
                                        'max_tokens': args.max_tokens, 'max_batch': args.max_batch, 'n_cases': len(cases)})


# ----------------------------------------------------------------------------- answer-slot readout (variant B)

SLOT_PREFIX = '[{"point_2d": ['


def slot_text(rt, case: dict) -> str:
    from analysis_tools.common import qwen_runtime as qr
    return qr.prompt_text(rt, 1, own_prompt(case, 'point')) + SLOT_PREFIX


def slot_cases(cases: list[dict]) -> list[dict]:
    """Reduced case set for variant B: 2500 RefCOCO + 1500 synthetic training cases, every validation
    case, the whole RefCOCO test set and 200 cases of each synthetic test set."""
    rng = np.random.default_rng(SEED + 3)
    out = []
    for s, split, n in (('refcoco', 'train', 2500), ('synth', 'train', 1500), ('synth', 'test', 200), ('disc', 'test', 200),
                        ('t6', 'test', 200), ('layout', 'test', 200), ('ring', 'test', 200)):
        g = [c for c in cases if c['set'] == s and c['split'] == split]
        out += [g[i] for i in sorted(rng.choice(len(g), min(n, len(g)), replace=False))]
    return out + [c for c in cases if c['split'] == 'val' or (c['set'] == 'refcoco' and c['split'] == 'test')]


def stage_slot(rt, args) -> None:
    """Variant B: teacher-forced point prompt + canonical answer prefix; read the state that predicts
    the first digit of x (experiment 3.6 role x0) and its full-attention maps over visual tokens."""
    feat = args.features.parent / 'features_slot'
    feat.mkdir(parents=True, exist_ok=True)
    cases = slot_cases(load_cases(args))
    c0 = cases[0]
    whole = rt.tokenizer(slot_text(rt, c0), add_special_tokens=False)['input_ids']
    head = rt.tokenizer(slot_text(rt, c0)[:-len(SLOT_PREFIX)], add_special_tokens=False)['input_ids']
    tail = rt.tokenizer(SLOT_PREFIX, add_special_tokens=False)['input_ids']
    if whole != head + tail or rt.tokenizer.decode(tail[-1:]) != ' [':
        raise ValueError('Answer prefix tokenization differs from the experiment-3.6 x0 role.')
    done = set()
    for f in sorted(feat.glob('chunk_*.npz')):
        done |= set(np.load(f)['ids'].tolist())
    todo = [c for c in cases if c['id'] not in done]
    k0 = len(list(feat.glob('chunk_*.npz')))
    t0, n = time.time(), 0
    for chunk_start in range(0, len(todo), CHUNK):
        chunk = todo[chunk_start:chunk_start + CHUNK]
        ids, xs, fs, maps = [], [], [], []
        for b in batches(chunk, args.max_tokens, args.max_batch):
            x, f, m = rt_.capture_sites(rt, [slot_text(rt, c) for c in b], [load_image(c, args) for c in b],
                                        [{'x0': -1} for _ in b], sites=('x0',), attn_layers=rt_.ATTN_LAYERS, map_site='x0')
            ids += [c['id'] for c in b]
            xs.append(x)
            fs.append(f.astype(np.float32))
            maps += m
            n += len(b)
        np.savez(feat / f'chunk_{k0:05d}.npz', ids=np.asarray(ids), x=np.concatenate(xs), attn=np.concatenate(fs),
                 maps=np.concatenate(maps, 1), map_len=np.asarray([m.shape[1] for m in maps]))
        k0 += 1
        print(f'slot captured {n}/{len(todo)}  {(time.time() - t0) / n:.3f} s/case', flush=True)
    write_meta(args.output, 'slot', {'prefix': SLOT_PREFIX, 'n_cases': len(cases), 'layers': rt_.CAPTURE_LAYERS,
                                     'attn_layers': rt_.ATTN_LAYERS})


# ----------------------------------------------------------------------------- own answers

def own_cases(cases: list[dict]) -> list[dict]:
    return [c for c in cases if c['split'] == 'test']


def own_prompt(case: dict, fmt: str) -> str:
    return rt_.QUERY.format(target=case['target']) + rt_.OWN_SUFFIX[fmt]


def stage_own(rt, args) -> None:
    from analysis_tools.common import qwen_runtime as qr
    path = args.output / 'own.jsonl'
    done = {(r['id'], r['format']) for r in read_jsonl(path)}
    cases = own_cases(load_cases(args))
    t0, n = time.time(), 0
    for fmt in ('point', 'bbox'):
        todo = [c for c in cases if (c['id'], fmt) not in done]
        for b in batches(todo, args.max_tokens, args.max_batch):
            res = qr.generate_batch(rt, [([load_image(c, args)], own_prompt(c, fmt)) for c in b], args.max_new_tokens)
            append_jsonl(path, [{'id': c['id'], 'format': fmt, 'text': r['text'], 'n_tokens': r['n_tokens'],
                                 'finish_reason': r['finish_reason'], 'batch_size': len(b)} for c, r in zip(b, res)])
            n += len(b)
            if n % 512 < len(b):
                print(f'own {fmt}: {n} done, {(time.time() - t0) / n:.3f} s/answer', flush=True)


def stage_spotcheck(rt, args) -> None:
    from analysis_tools.common import qwen_runtime as qr
    own = {(r['id'], r['format']): r for r in read_jsonl(args.output / 'own.jsonl')}
    cases = {c['id']: c for c in load_cases(args)}
    rng = np.random.default_rng(SEED)
    keys = sorted(own)
    pick = [keys[i] for i in rng.choice(len(keys), args.n_spot, replace=False)]
    rows = []
    for cid, fmt in pick:
        c = cases[cid]
        ref = qr.generate(rt, qr.build(rt, [load_image(c, args)], own_prompt(c, fmt)), args.max_new_tokens)
        a, b = rt_.parse_answer(ref['text'], fmt), rt_.parse_answer(own[(cid, fmt)]['text'], fmt)
        rows.append({'id': cid, 'format': fmt, 'reference': ref['text'], 'batched': own[(cid, fmt)]['text'],
                     'text_equal': ref['text'] == own[(cid, fmt)]['text'], 'answer_equal': a == b,
                     'max_abs_diff': None if a is None or b is None else float(np.abs(np.subtract(a, b)).max())})
    write_meta(args.output, 'spotcheck', {'n': len(rows), 'answer_equal': sum(r['answer_equal'] for r in rows), 'rows': rows})
    print(f"spotcheck: {sum(r['answer_equal'] for r in rows)}/{len(rows)} answers equal", flush=True)


# ----------------------------------------------------------------------------- tool loop

def rc_cases(cases: list[dict]) -> list[dict]:
    """Row/column images: all experiment-1.3 T6 grids plus RC_LAYOUT_PER images per supplement layout."""
    t6 = [c for c in cases if c['set'] == 't6']
    rng = np.random.default_rng(SEED + 1)
    lay = []
    for name in sorted({c['subset'] for c in cases if c['set'] == 'layout'}):
        group = [c for c in cases if c['set'] == 'layout' and c['subset'] == name]
        lay += [group[i] for i in sorted(rng.choice(len(group), min(RC_LAYOUT_PER, len(group)), replace=False))]
    return t6 + lay


def rc_user(case: dict, grid: bool, tool: bool, mode: str) -> str:
    from analysis_tools.behavior.decomposition import EXPLICIT_LOCATE
    g = case['truth']['grid']
    text = RC_QUESTION + (GRID_SENTENCE.format(rows=g['rows'], cols=g['cols']) if grid else '')
    text += RC_TOOL if tool else ''
    return text + (RC_DIRECT if mode == 'direct' else EXPLICIT_LOCATE)


def tool_readout(rt, tool, items: list[tuple[str, Image.Image, str]]) -> list[dict]:
    """Run the tool on (user_text, image, target) triples: canonical call, read the site, regress."""
    texts, offs = [], []
    for user, _, target in items:
        texts.append(rt_.tool_prompt(rt, user) + rt_.call_text(target))
        offs.append(rt_.call_sites(rt.tokenizer, target)[1])
    raw = apply_tool(rt, tool, texts, [i[1] for i in items], offs)
    return [dict(rt_.ReadoutTool.format(r), raw=r.tolist()) for r in raw]


def apply_tool(rt, tool, texts: list[str], images: list[Image.Image], offs: list[dict]) -> np.ndarray:
    """One teacher-forced pass over prompt + call; the tool reads its site and returns (N, 6) raw values."""
    if tool.kind == 'pointer':
        _, _, maps = rt_.capture_sites(rt, texts, images, offs, layers=(16,), sites=(tool.site,),
                                       attn_layers=tool.layers, map_site=tool.site)
        return tool.predict_maps(maps, [rt_.grid_of(im.size) for im in images])
    X = rt_.capture_sites(rt, texts, images, offs, layers=tool.layers, sites=(tool.site,))
    return tool.predict_raw(X[:, 0].reshape(len(texts), -1).astype(np.float32))


def stage_loop(rt, args) -> None:
    from analysis_tools.common import qwen_runtime as qr
    tool = rt_.ReadoutTool.load(args.output / 'tool' / 'tool.pt')
    cases = load_cases(args)
    own = {(r['id'], r['format']): rt_.parse_answer(r['text'], r['format']) for r in read_jsonl(args.output / 'own.jsonl')}
    path = args.output / 'rowcol.jsonl'
    done = {(r['id'], r['condition'], r['mode']) for r in read_jsonl(path)}
    rc = rc_cases(cases)
    conditions = [('notool_nogrid', False, None), ('notool_grid', True, None), ('own_grid', True, 'own'), ('tool_grid', True, 'tool')]
    t0, n = time.time(), 0
    for mode in ('direct', 'think'):
        max_new = 24 if mode == 'direct' else args.max_think_tokens
        for name, grid, source in conditions:
            todo = [c for c in rc if (c['id'], name, mode) not in done]
            for b in batches(todo, args.max_tokens, args.max_batch if mode == 'direct' else args.max_batch_think):
                images = [load_image(c, args) for c in b]
                users = [rc_user(c, grid, source is not None, mode) for c in b]
                extra = [{} for _ in b]
                if source is None:
                    items = [([im], u) for im, u in zip(images, users)]
                    res = qr.generate_batch(rt, items, max_new)
                else:
                    if source == 'tool':
                        preds = tool_readout(rt, tool, [(u, im, rt_.T6_TARGET) for u, im in zip(users, images)])
                    else:
                        preds = []
                        for c in b:
                            p, bx = own[(c['id'], 'point')], own[(c['id'], 'bbox')]
                            preds.append({'point_2d': [int(round(v)) for v in p] if p else None,
                                          'bbox_2d': [int(round(v)) for v in bx] if bx else None})
                    texts = [rt_.tool_loop_text(rt, u, rt_.T6_TARGET, rt_.tool_response_text(p)) for u, p in zip(users, preds)]
                    res = generate_texts(rt, texts, images, max_new)
                    extra = [{'tool_response': p} for p in preds]
                append_jsonl(path, [dict({'id': c['id'], 'condition': name, 'mode': mode, 'text': r['text'],
                                          'n_tokens': r['n_tokens'], 'finish_reason': r['finish_reason'],
                                          'batch_size': len(b)}, **e) for c, r, e in zip(b, res, extra)])
                n += len(b)
            print(f'rowcol {mode}/{name} done; {n} answers, {(time.time() - t0) / max(n, 1):.2f} s/answer', flush=True)
    stage_freecall(rt, args, tool, cases)


def generate_texts(rt, texts: list[str], images: list[Image.Image], max_new_tokens: int) -> list[dict]:
    """Greedy left-padded batch decoding of fully rendered chat texts (same procedure as generate_batch)."""
    import torch
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    with torch.inference_mode():
        out = rt.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = rt.tokenizer.pad_token_id
    res = []
    for row in out[:, batch['input_ids'].shape[1]:]:
        ids = row.tolist()
        end = next((i + 1 for i, t in enumerate(ids) if t in eos), None)
        new = ids[:end] if end is not None else [t for t in ids if t != pad]
        res.append({'text': rt.tokenizer.decode(new, skip_special_tokens=True), 'n_tokens': len(new),
                    'finish_reason': 'eos' if end is not None else 'length'})
    return res


def stage_freecall(rt, args, tool, cases: list[dict]) -> None:
    """Model-generated calls: call rate, argument text, and the tool output read at the model's own call."""
    path = args.output / 'freecall.jsonl'
    done = {(r['id'], r['variant']) for r in read_jsonl(path)}
    rng = np.random.default_rng(SEED + 2)
    pool = []
    for s in ('refcoco', 'synth', 'disc', 't6', 'layout', 'ring'):
        group = [c for c in cases if c['set'] == s and c['split'] == 'test']
        pool += [group[i] for i in sorted(rng.choice(len(group), min(args.n_free.get(s, 0), len(group)), replace=False))]
    for variant, instr in (('instructed', rt_.TOOL_INSTRUCTION), ('uninstructed', '')):
        todo = [c for c in pool if (c['id'], variant) not in done]
        if variant == 'uninstructed':
            todo = todo[::3]
        for b in batches(todo, args.max_tokens, args.max_batch):
            images = [load_image(c, args) for c in b]
            prompts = [rt_.tool_prompt(rt, rt_.QUERY.format(target=c['target']) + instr) for c in b]
            gen = generate_texts(rt, prompts, images, 64)
            rows, cap = [], []
            for c, p, g, im in zip(b, prompts, gen, images):
                target = rt_.parse_call(g['text'])
                row = {'id': c['id'], 'variant': variant, 'text': g['text'], 'called': target is not None,
                       'target_generated': target, 'target_canonical': c['target'], 'batch_size': len(b)}
                if target is not None:
                    cap.append((len(rows), p + rt_.call_text(target), im, rt_.call_sites(rt.tokenizer, target)[1]))
                rows.append(row)
            if cap:
                raw = apply_tool(rt, tool, [x[1] for x in cap], [x[2] for x in cap], [x[3] for x in cap])
                for (k, *_), r in zip(cap, raw):
                    rows[k]['tool'] = dict(rt_.ReadoutTool.format(r), raw=r.tolist())
            append_jsonl(path, rows)
        print(f'freecall {variant} done', flush=True)


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--refcoco', type=Path, required=True)
    p.add_argument('--refcoco-source', type=Path, required=True, help='experiment-3.4 run directory (cases.json)')
    p.add_argument('--t6-source', type=Path, required=True)
    p.add_argument('--supplement-source', type=Path, required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=12000)
    p.add_argument('--max-batch', type=int, default=32)
    p.add_argument('--max-batch-think', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--max-think-tokens', type=int, default=768)
    p.add_argument('--n-spot', type=int, default=20)
    p.add_argument('--max-memory', default='', help='comma list of GiB per visible GPU; empty = single GPU')
    p.add_argument('--n-free', default='refcoco:200,synth:100,disc:100,t6:96,layout:96,ring:96')
    args = p.parse_args()
    args.n_free = {k: int(v) for k, v in (x.split(':') for x in args.n_free.split(','))}
    args.behavior_root = args.t6_source.parent
    if args.supplement_source.parent != args.behavior_root:
        raise ValueError('T6 and supplement sources must share one parent directory (image_root "behavior").')
    args.output.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(',')
    if 'prepare' in stages:
        stage_prepare(args)
        stages.remove('prepare')
    if 'fixsmall' in stages:
        fix_small_refcoco(args)
        stages.remove('fixsmall')
    if not stages:
        return
    from analysis_tools.common import qwen_runtime as qr
    rt = rt_.load_runtime(args.model, [float(v) for v in args.max_memory.split(',')] if args.max_memory else None)
    write_meta(args.output, 'runtime', dict(qr.describe(rt), max_memory_gib=args.max_memory,
                                            device_map=sorted({str(v) for v in getattr(rt.model, 'hf_device_map', {}).values()})))
    table = {'verify': stage_verify, 'capture': stage_capture, 'own': stage_own, 'spotcheck': stage_spotcheck,
             'loop': stage_loop, 'slot': stage_slot}
    for s in stages:
        t0 = time.time()
        table[s](rt, args)
        append_jsonl(args.output / 'stage_log.jsonl', [{'stage': s, 'seconds': time.time() - t0,
                                                        'utc': datetime.now(timezone.utc).isoformat()}])


if __name__ == '__main__':
    main()
