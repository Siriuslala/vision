"""Experiment 3.13 stage drivers. Design: docs/analysis/position/design_targeted_finetune.md.

Stages (comma list via --stages):
  prepare    (CPU) training and test samples: RefCOCO train/test (image split of experiment 3.9),
             graded synthetic scenes (flat/texture for training; clutter/chart_ui held out), training grids
             (sizes 4-14 except 6/10), held-out disc sweeps (3.6) and 6/10/20 grids (3.12);
  train      LoRA training of one condition (--condition), adapter saved under adapters/;
  coord      greedy point/bbox answers on the coordinate test set for the base model or one adapter;
  general    POPE / TextVQA / MMVP / GQA (200 each, the 5.2 samples) and a text-only sanity set;
  factorial  the 3.12 first-digit factorial (disc sweeps, blank canvas, 60 RefCOCO points) with the
             adapter attached: digit distributions, residuals and DLA at q0 / q1;
  spotcheck  20 coordinate items: batched decoding vs the reference `generate`.
--condition base evaluates the unmodified model.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import hashlib
import json
import math
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
from analysis_tools.position import digit_failure_run as dfr
from analysis_tools.position import readout_tool_r2 as r2
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl
from analysis_tools.common import stimuli as st

SEED = 20261021
N_TRAIN = {'refcoco': 1400, 'flat': 700, 'texture': 500, 'grid': 600}
N_TEST = {'refcoco': 400, 'flat': 100, 'clutter': 100, 'chart_ui': 100}
MAX_SYNTH_PIXELS = 768 * 768
TRAIN_GRID_SIZES = (4, 5, 7, 8, 9, 12, 14)          # 6, 10 and 20 are held out (3.12 test grids)
GRID_TARGET = 'the figure drawn in red'
FACTORIAL_REFCOCO = 60


# ----------------------------------------------------------------------------- prepare

def _save(img: Image.Image, root: Path, rel: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        img.save(p)
    return str(p)


def _graded(level: str, seed: int, root: Path, split: str, fmt: str) -> tf.Sample | None:
    img, truth, target, meta = r2.graded_scene(level, seed)
    if img.width * img.height > MAX_SYNTH_PIXELS:
        return None
    path = _save(img, root, f'images/{level}/{split}_{seed}.png')
    t = {k: truth[k] for k in ('canvas', 'box', 'center')}
    return tf.Sample(f'{level}_{split}_{seed}', level, split, path, fmt, t, tf.question(target, fmt), target, meta)


def _graded_star(args: tuple) -> tf.Sample | None:
    level, seed = args[0], args[1]
    if level == 'flat':
        # generic_scene draws the canvas first; reject oversized canvases before rendering.
        rng = np.random.default_rng(seed)
        while True:
            W, H = (int(rng.integers(8, 43)) * 32 for _ in range(2))
            if max(W, H) / min(W, H) <= 2.5:
                break
        if W * H > MAX_SYNTH_PIXELS:
            return None
    return _graded(*args)


def _train_grid(k: int, rng, root: Path) -> tf.Sample:
    while True:
        rows, cols = (int(rng.choice(TRAIN_GRID_SIZES)) for _ in range(2))
        if rows * 64 * cols * 64 <= MAX_SYNTH_PIXELS * 1.4:
            break
    target = (int(rng.integers(rows)), int(rng.integers(cols)))
    template = st.DEVELOPMENT_TEMPLATES[int(rng.integers(len(st.DEVELOPMENT_TEMPLATES)))]
    img, meta = st.generate(st.StimulusSpec(template=template, rows=rows, cols=cols, change=st.Change('color', 4),
                                            target=target, cell=64))
    inst = next(i for i in meta['instances'] if i['is_target'])
    fmt = 'point' if k % 2 == 0 else 'bbox'
    path = _save(img, root, f'images/grid/train_{k:05d}.png')
    truth = {'canvas': list(img.size), 'box': inst['tight_box'], 'center': inst['center']}
    return tf.Sample(f'grid_train_{k:05d}', 'grid', 'train', path, fmt, truth, tf.question(GRID_TARGET, fmt), GRID_TARGET,
                     {'rows': rows, 'cols': cols, 'target': target, 'template': template})


def stage_prepare(out: Path, args) -> None:
    rng = np.random.default_rng(SEED)
    train, test = [], []
    # RefCOCO: experiment-3.9 image-level split (train images never appear in its test split or in 3.12).
    r1 = Path(args.readout_tool_run)
    cases = read_jsonl(r1 / 'cases.jsonl')
    excl = {str(c['question_id']) for c in json.loads(Path(args.refcoco_cases).read_text())}
    rc_train = [c for c in cases if c['set'] == 'refcoco' and c['split'] == 'train' and c['id'].split('_')[1] not in excl]
    rc_test = [c for c in cases if c['set'] == 'refcoco' and c['split'] == 'test']
    n_excl = sum(c['set'] == 'refcoco' and c['split'] == 'train' and c['id'].split('_')[1] in excl for c in cases)
    for split, pool, n, dest in (('train', rc_train, N_TRAIN['refcoco'], train), ('test', rc_test, N_TEST['refcoco'], test)):
        for j, i in enumerate(rng.choice(len(pool), n, replace=False)):
            c = pool[int(i)]
            fmt = 'point' if j % 2 == 0 else 'bbox'
            t = {k: c['truth'][k] for k in ('canvas', 'box', 'center')}
            dest.append(tf.Sample(f"{c['id']}_{fmt}", 'refcoco', split, str(r1 / 'images' / c['image']), fmt, t,
                                  tf.question(c['target'], fmt), c['target'], {'digest': c['digest']}))
    # Graded synthetic scenes: disjoint seed ranges for train and test (rendered in parallel; scenes larger
    # than MAX_SYNTH_PIXELS are rejected, so a surplus of seeds is drawn and the first n accepted are kept).
    from multiprocessing import Pool
    for level, split, n, base, dest in (('flat', 'train', N_TRAIN['flat'], 100_000, train),
                                         ('texture', 'train', N_TRAIN['texture'], 200_000, train),
                                         ('flat', 'test', N_TEST['flat'], 900_000, test),
                                         ('clutter', 'test', N_TEST['clutter'], 910_000, test),
                                         ('chart_ui', 'test', N_TEST['chart_ui'], 920_000, test)):
        got = []
        with Pool(args.workers) as pool:          # ordered imap; leaving the block terminates queued renders
            for s in pool.imap(_graded_star, [(level, base + i, out, split, 'point') for i in range(4 * n)]):
                if s is not None:
                    got.append(s)
                if len(got) == n:
                    break
        if len(got) < n:
            raise RuntimeError(f'only {len(got)} {level} scenes accepted')
        for k, s in enumerate(got):
            if k % 2:
                s.fmt, s.prompt = 'bbox', tf.question(s.label, 'bbox')
        dest.extend(got)
        print(f'{level}/{split}: {n} scenes', flush=True)
    grng = np.random.default_rng(SEED + 1)
    seeds = [int(v) for v in grng.integers(0, 2**31, N_TRAIN['grid'])]
    with Pool(args.workers) as pool:
        train += pool.starmap(_train_grid, [(k, np.random.default_rng(seeds[k]), out) for k in range(N_TRAIN['grid'])])
    # Held-out discs: the 3.6 sweeps (every second step, as in 3.12).
    for c in [c for c in cv.disc_cases() if c['set'] in ('sweep_h', 'sweep_v')][::2]:
        img, truth, prompt = cv.disc_input(c)
        path = _save(img, out, f"images/disc/{c['id']}.png")
        test.append(tf.Sample(f"disc_{c['id']}", c['set'], 'test', path, 'point',
                              {'canvas': truth['canvas'], 'box': truth['box'], 'center': truth['center'],
                               'diameter': truth['diameter']}, prompt, 'red circle'))
    # Held-out grids: the 3.12 6x6 / 10x10 / 20x20 point questions.
    for g in df.grid_cases():
        img, truth = df.grid_input(g)
        path = _save(img, out, f"images/testgrid/{g['id']}.png")
        W, H = truth['canvas']
        cell_box = [truth['grid_origin'][0] + 64 * g['col'], truth['grid_origin'][1] + 64 * g['row']]
        cell_box += [cell_box[0] + 64, cell_box[1] + 64]
        test.append(tf.Sample(g['id'], f"grid{g['n']:02d}", 'test', path, 'point',
                              {'canvas': [W, H], 'box': cell_box, 'center': truth['center_px']}, df.GRID_PROMPT, GRID_TARGET,
                              {'col_edges_units': truth['col_edges_units'], 'n': g['n'], 'col': g['col'], 'row': g['row']}))
    tf.dump_samples(out / 'train.jsonl', train)
    tf.dump_samples(out / 'test.jsonl', test)
    from collections import Counter
    info = {'train': Counter(f'{s.family}/{s.fmt}' for s in train), 'test': Counter(f'{s.family}/{s.fmt}' for s in test),
            'refcoco_train_excluded_overlap_3_12': n_excl, 'seed': SEED}
    (out / 'prepare.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


# ----------------------------------------------------------------------------- model helpers

def load_split_sequential(name: str, max_memory: str) -> qr.Runtime:
    """Weights split over the visible GPUs in layer order (device_map='sequential'); fails instead of offloading."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    checkpoint = resolve_model(name, load_paths()['HF_HUB_CACHE'])
    mem = {int(k): v for k, v in (item.split(':') for item in max_memory.split(','))}
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16,
                                                        device_map='sequential', max_memory=mem,
                                                        attn_implementation='sdpa').eval()
    devices = {str(p.device) for p in model.parameters()}
    if any(not d.startswith('cuda') for d in devices):
        raise RuntimeError(f'Weights were offloaded ({devices}); raise --max-memory')
    model.requires_grad_(False)
    return qr.Runtime(name, checkpoint, model, processor, 'cuda:0')


def load_model(args) -> qr.Runtime:
    return load_split_sequential(args.model, args.max_memory) if args.max_memory else qr.load(args.model)


def attach(rt: qr.Runtime, args) -> tf.LoRA | None:
    if args.condition == 'base':
        return None
    path = Path(args.output) / 'adapters' / f'{args.condition}.pt'
    return tf.load_adapter(rt.layers, path, args.condition)


def cond_dir(out: Path, condition: str) -> Path:
    d = out / 'eval' / condition
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- train

def stage_train(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    if cond == 'base':
        raise ValueError('train needs a LoRA condition')
    (out / 'adapters').mkdir(exist_ok=True)
    apath = out / 'adapters' / f'{cond}.pt'
    if apath.exists():
        print(f'{apath} exists; skipping training', flush=True)
        return
    samples = tf.read_samples(out / 'train.jsonl')
    torch.manual_seed(SEED)
    rank = tf.CONDITIONS[cond].get('rank', args.rank)
    alpha = args.alpha * rank / args.rank            # keep the update scale alpha / r fixed across ranks
    lora = tf.LoRA(tf.target_modules(rt.layers, cond), rank, alpha)
    lora.attach()
    model = rt.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.config.use_cache = False
    opt = torch.optim.AdamW(lora.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    rng = np.random.default_rng(SEED)
    order = np.concatenate([rng.permutation(len(samples)) for _ in range(args.epochs)])
    n_steps = len(order) // args.step_samples
    warm = max(1, int(0.05 * n_steps))

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_steps - warm)))

    prompt_fn = lambda text: qr.prompt_text(rt, 1, text)
    log = out / 'adapters' / f'{cond}_log.jsonl'
    if log.exists():
        log.unlink()
    t0 = time.time()
    for step in range(n_steps):
        idx = order[step * args.step_samples:(step + 1) * args.step_samples]
        items = [samples[int(i)] for i in idx]
        sizes = [Image.open(s.image).size for s in items]
        tokens = [(w // 32) * (h // 32) + 150 for w, h in sizes]
        srt = np.argsort(tokens)
        micro, cur = [], []
        for i in srt:
            if cur and (len(cur) + 1) * max(tokens[j] for j in cur + [i]) > args.micro_tokens:
                micro.append(cur)
                cur = []
            cur.append(int(i))
        micro.append(cur)
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        tot, dig, n_dig = 0.0, 0.0, 0
        for mb in micro:
            ss = [items[i] for i in mb]
            batch = tf.encode_batch(rt.processor, prompt_fn, ss, [Image.open(s.image).convert('RGB') for s in ss])
            per, dce, _ = tf.batch_loss(model, batch, rt.device)
            loss = per.sum() / len(items)
            loss.backward()
            tot += float(per.sum())
            dig += float(dce.sum())
            n_dig += int(dce.numel())
        gn = torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
        opt.step()
        row = {'step': step, 'loss': tot / len(items), 'digit_ce': dig / max(1, n_dig), 'lr': lr_at(step),
               'grad_norm': float(gn), 'micro': len(micro), 'time': time.time() - t0}
        append_jsonl(log, [row])
        if step % 10 == 0:
            print(json.dumps(row), flush=True)
    state = lora.state()
    state.update({'condition': cond, 'lr': args.lr, 'epochs': args.epochs, 'step_samples': args.step_samples,
                  'n_steps': n_steps, 'n_params': lora.n_params(), 'delta_norms': lora.delta_norms(),
                  'train_seconds': time.time() - t0, 'runtime': qr.describe(rt)})
    torch.save(state, apath)
    model.gradient_checkpointing_disable()
    model.eval()
    lora.detach()
    print(f'saved {apath} ({lora.n_params()} params, {time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- evaluation

def generate_samples(rt: qr.Runtime, samples: list[tf.Sample], budget: int, max_new: int) -> list[dict]:
    """Batched greedy decoding grouped by image size (left padding inside a group is then minimal)."""
    groups: dict = {}
    for s in samples:
        groups.setdefault(Image.open(s.image).size, []).append(s)
    res = {}
    for size, group in sorted(groups.items()):
        bs = max(1, budget // ((size[0] // 32) * (size[1] // 32) + 200))
        for i in range(0, len(group), bs):
            chunk = group[i:i + bs]
            outs = qr.generate_batch(rt, [([Image.open(s.image).convert('RGB')], s.prompt) for s in chunk], max_new)
            for s, o in zip(chunk, outs):
                res[s.id] = {**o, 'batch_size': len(chunk)}
    return [res[s.id] for s in samples]


def stage_coord(rt: qr.Runtime, out: Path, args) -> None:
    d = cond_dir(out, args.condition)
    path = d / 'coord.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    samples = [s for s in tf.read_samples(out / 'test.jsonl') if s.id not in done]
    t0 = time.time()
    for i in range(0, len(samples), 64):
        chunk = samples[i:i + 64]
        outs = generate_samples(rt, chunk, args.token_budget, args.max_new_tokens)
        append_jsonl(path, [{'id': s.id, 'family': s.family, 'fmt': s.fmt, 'text': o['text'], 'finish_reason': o['finish_reason'],
                             'batch_size': o['batch_size'], **tf.score(s, o['text'])} for s, o in zip(chunk, outs)])
        print(f'coord {args.condition} {i + len(chunk)}/{len(samples)} {time.time() - t0:.0f}s', flush=True)


TEXT_SANITY = ([(f'What is {a} + {b}? Answer with only the number.', str(a + b))
                for a, b in [(17, 25), (38, 47), (123, 456), (999, 1), (64, 36), (250, 375), (81, 19), (402, 98),
                             (73, 68), (505, 495), (12, 89), (333, 444), (47, 53), (606, 207), (18, 27)]] +
               [(f'What is {a} times {b}? Answer with only the number.', str(a * b))
                for a, b in [(7, 8), (12, 12), (9, 13), (25, 4), (15, 15), (6, 17), (11, 23), (19, 3), (14, 6), (8, 125)]] +
               [(f'What is the capital of {c}? Answer with only the city name.', a)
                for c, a in [('France', 'Paris'), ('Japan', 'Tokyo'), ('Italy', 'Rome'), ('Canada', 'Ottawa'),
                             ('Australia', 'Canberra'), ('Egypt', 'Cairo'), ('Spain', 'Madrid'), ('Kenya', 'Nairobi'),
                             ('Germany', 'Berlin'), ('Brazil', 'Brasilia'), ('Norway', 'Oslo'), ('Peru', 'Lima'),
                             ('Turkey', 'Ankara'), ('Thailand', 'Bangkok'), ('Poland', 'Warsaw')]] +
               [('Spell the word "coordinate" backwards. Answer with only the reversed word.', 'etanidrooc'),
                ('How many letters are in the word "grounding"? Answer with only the number.', '9'),
                ('Which is larger, 0.9 or 0.11? Answer with only the number.', '0.9'),
                ('What is 1000 divided by 8? Answer with only the number.', '125'),
                ('Complete the sequence 2, 4, 8, 16, ... with the next number only.', '32')])


def min_area(image: Image.Image, min_pixels: int = 65536) -> Image.Image:
    """Upscale images below the processor's minimum area to sides that are multiples of 32 (no hidden resize)."""
    w, h = image.size
    if w * h >= min_pixels and w % 32 == 0 and h % 32 == 0:
        return image
    s = max(1.0, math.sqrt(min_pixels / (w * h)))
    W, H = (int(math.ceil(v * s / 32)) * 32 for v in (w, h))
    return image.resize((W, H), Image.Resampling.BICUBIC)


def stage_general(rt: qr.Runtime, out: Path, args) -> None:
    from analysis_tools.reread import cached_reread as crr
    from analysis_tools.reread import cached_reread_run as crun
    d = cond_dir(out, args.condition)
    path = d / 'general.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    ns = Namespace(seed=args.general_seed, n=args.general_n, data_dir=Path(args.data_dir), vcot_gqa=Path(args.vcot_gqa),
                   mmvp_csv=args.mmvp_csv, gqa_max_area=0.02, max_pixels=512 * 512, bv_max_pixels=None)
    t0 = time.time()
    for bench in ('pope', 'textvqa', 'mmvp', 'gqa'):
        cases = [c for c in crun.load_cases(bench, ns) if c.id not in done]
        for i in range(0, len(cases), args.general_batch):
            chunk = cases[i:i + args.general_batch]
            outs = qr.generate_batch(rt, [([min_area(crun.load_image(c, ns))], c.question) for c in chunk], args.general_max_new)
            append_jsonl(path, [{'id': c.id, 'bench': bench, 'text': o['text'], 'finish_reason': o['finish_reason'],
                                 **crun.score(c, o['text'], o['finish_reason'], None)} for c, o in zip(chunk, outs)])
        print(f'general {args.condition} {bench} {time.time() - t0:.0f}s', flush=True)
    tpath = d / 'text.jsonl'
    if not tpath.exists():
        outs = qr.generate_batch(rt, [([], q) for q, _ in TEXT_SANITY], 32)
        append_jsonl(tpath, [{'q': q, 'answer': a, 'text': o['text'], 'finish_reason': o['finish_reason'],
                              'correct': crr.vqa_normalize(o['text'].strip().split('\n')[-1]) == crr.vqa_normalize(a)}
                             for (q, a), o in zip(TEXT_SANITY, outs)])


def factorial_units(args) -> list[df.Unit]:
    """Disc sweeps (every second 3.6 step), blank canvas and 60 RefCOCO point units: the 3.12 definitions."""
    r1 = Path(args.round1)
    recs = read_jsonl(r1 / 'gen' / 'records.jsonl')
    cases = {c['id']: c for c in json.loads((r1 / 'cases.json').read_text())}
    P = cv.POINT_PROMPT
    units = []
    for r in dfr.sweep(recs, 'sweep_h', 2):
        units.append(df.Unit(r['id'], 'sweep_h', 0, r['truth_units'][0], P, df.point_prefix(0, None),
                             source={'case': r['id'], 'center': cases[r['id']]['center']}))
    for r in dfr.sweep(recs, 'sweep_v', 2):
        units.append(df.Unit(r['id'], 'sweep_v', 1, r['truth_units'][1], P, df.point_prefix(1, 500), ctx=500,
                             source={'case': r['id'], 'center': cases[r['id']]['center']}))
    units.append(df.Unit('blank_x', 'blank', 0, float('nan'), P, df.point_prefix(0, None), source={'blank': True}))
    units.append(df.Unit('blank_y', 'blank', 1, float('nan'), P, df.point_prefix(1, 500), ctx=500, source={'blank': True}))
    rc = [c for c in json.loads(Path(args.refcoco_cases).read_text()) if c['format'] == 'point'][:FACTORIAL_REFCOCO]
    for c in rc:
        W, H = c['truth']['canvas']
        units.append(df.Unit(f"rc_{c['id']}", 'refcoco', 0, 1000 * c['truth']['center'][0] / W, c['prompt'],
                             df.point_prefix(0, None), source={'image_path': c['image_path'], 'image_sha256': c['image_sha256']}))
    return units


def stage_factorial(rt: qr.Runtime, out: Path, args) -> None:
    """Same measurement as digit_failure_run.stage_factorial, restricted to factorial_units()."""
    run = cond_dir(out, args.condition) / 'fact'
    root = run / 'factorial'
    root.mkdir(parents=True, exist_ok=True)
    feat = Path(args.features) / args.condition / 'factorial'
    feat.mkdir(parents=True, exist_ok=True)
    link = run / 'features'
    if not link.exists():
        link.symlink_to(feat.parent)
    units = factorial_units(args)
    (root / 'units.json').write_text(df.dumps([u.__dict__ for u in units]))
    rpath = root / 'rows.jsonl'
    done = {r['unit'] for r in read_jsonl(rpath)} if rpath.exists() else set()
    digit_ids = df.digit_ids(rt)
    shard = len(list(feat.glob('shard_*.npz')))
    pending: list = []
    t0 = time.time()

    def flush():
        nonlocal shard, pending
        if not pending:
            return
        L = max(r[0].shape[1] for _, r in pending)
        cap = max(1, args.token_budget // L)
        parts = []
        for s in range(0, len(pending), cap):
            chunk = [r for _, r in pending[s:s + cap]]
            X, P = c2.pad_batch([r[0] for r in chunk], [r[1] for r in chunk])
            parts.append(df.readout_forward(rt, X, P, [r[2] for r in chunk], [r[3] for r in chunk]))
        res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        rows = [{'unit': u.id, 'family': u.family, 'axis': u.axis, 'truth': u.truth, 'ctx': u.ctx, 'k': r[4], 'q0': r[2],
                 'q1': r[3], 'shard': shard, 'index': i, 'logp': res['logp'][i].tolist(), 'lp_top': res['lp_top'][i].tolist(),
                 'top': [rt.tokenizer.decode([int(t)]) for t in res['top_id'][i]]} for i, (u, r) in enumerate(pending)]
        np.savez(feat / f'shard_{shard:05d}.npz', resid=res['resid'], dla_mixer=res['dla_mixer'].astype(np.float32),
                 dla_mlp=res['dla_mlp'].astype(np.float32), dla_head=res['dla_head'].astype(np.float32),
                 dla_embed=res['dla_embed'], logit=res['logit'])
        append_jsonl(rpath, rows)
        shard += 1
        pending = []

    with torch.inference_mode():
        for j, u in enumerate([u for u in units if u.id not in done]):
            rows = dfr.unit_rows(rt, u, digit_ids)
            L = rows[0][0].shape[1]
            cap = max(1, args.token_budget // L)
            if pending and (len(pending) + len(rows) > cap or abs(pending[0][1][0].shape[1] - L) > 64):
                flush()
            pending.extend((u, r) for r in rows)
            if len(pending) >= cap:
                flush()
        flush()
    print(f'factorial {args.condition} done {time.time() - t0:.0f}s', flush=True)


def stage_spotcheck(rt: qr.Runtime, out: Path, args) -> None:
    d = cond_dir(out, args.condition)
    rows = {r['id']: r for r in read_jsonl(d / 'coord.jsonl')}
    samples = [s for s in tf.read_samples(out / 'test.jsonl') if s.id in rows]
    pick = [samples[i] for i in np.random.default_rng(0).choice(len(samples), 20, replace=False)]
    res = []
    for s in pick:
        ref = qr.generate(rt, qr.build(rt, [Image.open(s.image).convert('RGB')], s.prompt), args.max_new_tokens)
        a, b = tf.parse_answer(ref['text'], s.fmt), tf.parse_answer(rows[s.id]['text'], s.fmt)
        res.append({'id': s.id, 'batched': rows[s.id]['text'], 'reference': ref['text'], 'same_coords': a == b,
                    'max_abs_diff': None if a is None or b is None else float(np.max(np.abs(np.array(a) - np.array(b))))})
    (d / 'spotcheck.json').write_text(json.dumps(res, indent=1))
    print(f"spotcheck {args.condition}: same coords {sum(r['same_coords'] for r in res)}/20", flush=True)


STAGES = {'train': stage_train, 'coord': stage_coord, 'general': stage_general, 'factorial': stage_factorial,
          'spotcheck': stage_spotcheck}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--condition', default='base', choices=['base', *tf.CONDITIONS])
    p.add_argument('--readout-tool-run', default='')
    p.add_argument('--refcoco-cases', default='')
    p.add_argument('--round1', default='')
    p.add_argument('--features', default='')
    p.add_argument('--data-dir', default='')
    p.add_argument('--vcot-gqa', default='')
    p.add_argument('--mmvp-csv', default='')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=32.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--step-samples', type=int, default=16)
    p.add_argument('--micro-tokens', type=int, default=4096)
    p.add_argument('--token-budget', type=int, default=16000)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--general-n', type=int, default=200)
    p.add_argument('--general-seed', type=int, default=20261020)
    p.add_argument('--general-batch', type=int, default=16)
    p.add_argument('--general-max-new', type=int, default=96)
    p.add_argument('--max-memory', default='')
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(',')
    if stages == ['prepare']:
        stage_prepare(out, args)
        return
    rt = load_model(args)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages, 'condition': args.condition,
                         'runtime': qr.describe(rt), 'args': {k: v for k, v in vars(args).items()},
                         'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]})
    meta.write_text(json.dumps(info, indent=1))
    lora = None
    for s in stages:
        t0 = time.time()
        if s != 'train' and lora is None:
            lora = attach(rt, args)
        STAGES[s](rt, out, args)
        print(f'stage {s} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
