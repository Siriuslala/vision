"""Experiment 7.3 stage drivers: locate (base model) -> prepare -> train -> eval / detect.

Results doc: docs/analysis/application/results_2026-11-05_located_vit_finetune.md.

Stages (comma list via --stages):
  locate     (GPU, base model, before any training) for every pair of ``LOCATE_PAIRS`` and every ViT block:
             exclusive-detector statistics on homogeneous images, the linear achievability of a new detector on
             the MLP input, deletion patching of the MLP output at the target of feature-present images, and
             transplant of a foreign detector's MLP output into the target of feature-absent images;
  prepare    (CPU) nested training sets (odd-entity search on TRAIN_PAIRS + TextVQA short answers) and the probe;
  train      LoRA on one scope (--scope, see located_finetune.parse_scope) with --n-odd samples, fixed steps;
  eval       probe, grid counting, 3.13 coordinate subset (incl. RefCOCO), general benchmarks, BabyVision
             fine-grained subset for one adapter (--name; 'base' = no adapter);
  detect     the locate detector statistics recomputed with an adapter (did the detector form where predicted?).
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.application import interp_opt as io_
from analysis_tools.application import interp_opt_run as ior
from analysis_tools.application import located_finetune as lf
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_run as tfr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_run as sr

SEED = 20261105
GRID = 6
N_QUADS = 12               # locate: quads (std, swap, homA, homB sharing target / orientation / jitter) per pair
N_FIT = 8                  # locate: homogeneous images used to fit the achievability direction (rest = test)
TRAIN_GRIDS = (4, 5, 6, 7, 8)
SWAP_FRACTION = 0.75       # share of training odd-entity items whose target lacks the part
N_ODD_MAX = 1200
PROBE_SWAP_PER_CELL = 2
PROBE_STD_PER_CELL = 1
BV_SUBTYPES = ('Find the different', 'Find the same')
TRANSPLANT_DONOR = {'gap': 'qo', 'stroke': 'closure', 'interior': 'closure', 'rotation': 'closure', 'control': 'closure'}


# ----------------------------------------------------------------------------- shared helpers

def quad_images(pair: str, k: int, level: str = 'subtle', rows: int = GRID, cols: int = GRID) -> dict:
    """The 4 variants of quad k of a pair (deterministic in pair and k)."""
    rng = np.random.default_rng([SEED, k, sum(map(ord, pair))])
    target = (int(rng.integers(rows)), int(rng.integers(cols)))
    orient = int(rng.integers(0, 4))
    jit = sa.jitter_map(rows, cols, rng)
    param = lf.LEVELS[pair][level]
    out = {v: lf.grid_image(pair, param, rows, cols, target, orient, jit, v) for v in ('std', 'swap', 'homA', 'homB')}
    return {'target': target, 'orient': orient, 'images': out}


class ViTTap:
    """Hooks on every ViT block: MLP input (post-norm2), GELU activation (fc2 input) and MLP output.

    keep: which quantities to store, each stored as the full (n_patches_total, d) tensor of the last call.
    edits: {block: fn(out) -> out} applied to the MLP output (patching / transplant).
    """

    def __init__(self, visual, keep=('inp', 'act', 'out')):
        self.visual, self.keep, self.store, self.edits, self.handles = visual, set(keep), {}, {}, []
        for b, blk in enumerate(visual.blocks, start=1):
            if 'inp' in self.keep:
                self.handles.append(blk.mlp.register_forward_pre_hook(
                    lambda m, a, b=b: self.store.__setitem__(('inp', b), a[0].detach())))
            if 'act' in self.keep:
                self.handles.append(blk.mlp.linear_fc2.register_forward_pre_hook(
                    lambda m, a, b=b: self.store.__setitem__(('act', b), a[0].detach())))
            self.handles.append(blk.mlp.register_forward_hook(self._out_hook(b)))

    def _out_hook(self, b):
        def f(m, a, o):
            if b in self.edits:
                o = self.edits[b](o)
            if 'out' in self.keep:
                self.store[('out', b)] = o.detach()
            return o
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


def vit_forward(rt, images: list[Image.Image]) -> None:
    batch = rt.processor.image_processor(images=images, return_tensors='pt')
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist()):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError('Processor resized an image')
    rt.model.model.get_image_features(batch['pixel_values'].to(rt.device), batch['image_grid_thw'].to(rt.device))


def torch_lda(pos: torch.Tensor, neg: torch.Tensor, shrink: float = 0.1) -> torch.Tensor:
    """GPU version of located_finetune.lda_direction (float64)."""
    pos, neg = pos.double(), neg.double()
    mp, mn = pos.mean(0), neg.mean(0)
    xc = torch.cat([pos - mp, neg - mn])
    cov = xc.T @ xc / max(1, len(xc) - 2)
    d = cov.shape[0]
    cov += shrink * torch.trace(cov) / d * torch.eye(d, device=cov.device, dtype=cov.dtype)
    w = torch.linalg.solve(cov, mp - mn)
    return w / (w.norm() + 1e-12)


def achievable(tgt: torch.Tensor, dis: torch.Tensor, n_fit_t: int, n_fit_d: int) -> float:
    """tgt, dis: (n_images, n_ent, 16, d); first n_fit images fit the direction, the rest test it."""
    d = tgt.shape[-1]
    w = torch_lda(tgt[:n_fit_t].reshape(-1, d), dis[:n_fit_d].reshape(-1, d))
    st = (tgt[n_fit_t:].double() @ w).amax(-1).flatten()
    sd = (dis[n_fit_d:].double() @ w).amax(-1).flatten()
    return float((st.mean() - sd.mean()) / (sd.std() + 1e-6))


def detector_stats(rt, pairs, blocks: int, with_inputs: bool, vectors: tuple = (), vec_out: dict | None = None) -> dict:
    """Per pair and ViT block: zeta summaries in both directions (+ achievability if with_inputs)."""
    visual = rt.model.model.visual
    res = {}
    tap = ViTTap(visual, keep=('inp', 'act') if with_inputs else ('act',))
    try:
        for pair in pairs:
            t0 = time.time()
            quads = [quad_images(pair, k) for k in range(N_QUADS)]
            meta = quads[0]['images']['homA'][1]
            pidx = torch.tensor(lf.patch_index(oe.entity_tokens(meta)), device=rt.device)       # (n_ent, 16)
            n_patch = 4 * meta['merged_grid'][0] * meta['merged_grid'][1]
            acts = {v: {b: [] for b in range(1, blocks + 1)} for v in ('homA', 'homB')}
            inps = {v: {b: [] for b in range(1, blocks + 1)} for v in ('homA', 'homB')}
            with torch.inference_mode():
                for v in ('homA', 'homB'):
                    for i0 in range(0, N_QUADS, 6):
                        ims = [q['images'][v][0] for q in quads[i0:i0 + 6]]
                        vit_forward(rt, ims)
                        for b in range(1, blocks + 1):
                            a = tap.store[('act', b)].reshape(len(ims), n_patch, -1)[:, pidx]   # (n, ent, 16, 4304)
                            acts[v][b].append(a.float().amax(2))
                            if with_inputs:
                                inps[v][b].append(tap.store[('inp', b)].reshape(len(ims), n_patch, -1)[:, pidx])
                        tap.store.clear()
            out = {}
            for b in range(1, blocks + 1):
                A = torch.cat(acts['homA'][b]).flatten(0, 1)          # (n_img * n_ent, 4304)
                B = torch.cat(acts['homB'][b]).flatten(0, 1)
                muA, sdA, muB, sdB = A.mean(0), A.std(0), B.mean(0), B.std(0)
                zAB = lf.zeta(muA.cpu().numpy(), muB.cpu().numpy(), sdB.cpu().numpy())
                zBA = lf.zeta(muB.cpu().numpy(), muA.cpu().numpy(), sdA.cpu().numpy())
                row = {'A|B': lf.detector_summary(zAB), 'B|A': lf.detector_summary(zBA)}
                if b in vectors and vec_out is not None:
                    vec_out[f'{pair}|{b}|A|B'] = zAB.astype(np.float32)
                    vec_out[f'{pair}|{b}|B|A'] = zBA.astype(np.float32)
                if with_inputs:
                    iA, iB = torch.cat(inps['homA'][b]), torch.cat(inps['homB'][b])
                    row['A|B']['achievable'] = achievable(iA, iB, N_FIT, N_FIT)
                    row['B|A']['achievable'] = achievable(iB, iA, N_FIT, N_FIT)
                out[b] = row
            res[pair] = out
            del acts, inps
            torch.cuda.empty_cache()
            print(f"[detect] {pair}: {time.time() - t0:.0f}s; B|A detectors by block "
                  f"{[out[b]['B|A']['n_detectors'] for b in range(1, blocks + 1)]}", flush=True)
    finally:
        tap.remove()
    return res


# ----------------------------------------------------------------------------- locate

@torch.no_grad()
def stage_locate(rt, out: Path, args) -> None:
    d = out / 'locate'
    d.mkdir(parents=True, exist_ok=True)
    nb = len(rt.model.model.visual.blocks)
    path = d / 'detectors.json'
    if not path.exists():
        path.write_text(json.dumps(detector_stats(rt, lf.LOCATE_PAIRS, nb, True), indent=1))
    # Causal role of MLP output written at each block (teacher-forced answer score of the direct-answer prompt).
    visual = rt.model.model.visual
    digits = sr.digit_ids(rt, GRID)
    tap = ViTTap(visual, keep=('out',))
    delta: dict[str, torch.Tensor] = {}                 # pair -> (blocks, 1152) mean target MLP-output difference
    rows = []
    try:
        for pair in lf.LOCATE_PAIRS:
            quads = [quad_images(pair, k) for k in range(N_QUADS)]
            meta = quads[0]['images']['std'][1]
            ent_patch = lf.patch_index(oe.entity_tokens(meta))
            n_patch = 4 * meta['merged_grid'][0] * meta['merged_grid'][1]
            acc = torch.zeros(nb, 1152, device=rt.device, dtype=torch.float64)
            for q in quads:
                tp = torch.tensor(ent_patch[q['target'][0] * GRID + q['target'][1]], device=rt.device)
                with torch.no_grad():
                    vit_forward(rt, [q['images']['std'][0], q['images']['homB'][0]])
                donor = {b: tap.store[('out', b)][n_patch + tp].clone() for b in range(1, nb + 1)}
                for b in range(1, nb + 1):
                    acc[b - 1] += (tap.store[('out', b)][tp] - donor[b]).double().mean(0)
                tap.store.clear()
                q['donor'], q['tp'] = donor, tp
            delta[pair] = (acc / len(quads)).float()
            # deletion: std image, MLP output at the target's patches of block b (or window b..b+2) <- homB value
            for q in quads:
                ans, i_row, i_col = sr.answer_ids(rt, q['target'])
                specs = [('none', ())] + [('single', (b,)) for b in range(1, nb + 1)] + \
                        [('window', tuple(range(b, b + 3))) for b in range(1, nb - 1, 3)]
                res = _patched_scores(rt, tap, q['images']['std'][0], ans, i_row, i_col, digits, n_patch, specs,
                                      lambda b, o, q=q: _replace(o, q['tp'], q['donor'][b]))
                for (kind, bl), r in zip(specs, res):
                    rows.append({'exp': 'delete', 'pair': pair, 'kind': kind, 'blocks': list(bl),
                                 'target': list(q['target']), **r})
            print(f'[locate] delete {pair} done', flush=True)
        # transplant: swap image (target lacks the part); add a foreign pair's present-detector output at block b
        for pair in lf.LOCATE_PAIRS:
            donor_pair = TRANSPLANT_DONOR[lf.family_of(pair)]
            quads = [quad_images(pair, k) for k in range(N_QUADS)]
            meta = quads[0]['images']['swap'][1]
            ent_patch = lf.patch_index(oe.entity_tokens(meta))
            n_patch = 4 * meta['merged_grid'][0] * meta['merged_grid'][1]
            for q in quads:
                tp = torch.tensor(ent_patch[q['target'][0] * GRID + q['target'][1]], device=rt.device)
                ans, i_row, i_col = sr.answer_ids(rt, q['target'])
                specs = [('none', ())] + [('single', (b,)) for b in range(1, nb + 1)]
                res = _patched_scores(rt, tap, q['images']['swap'][0], ans, i_row, i_col, digits, n_patch, specs,
                                      lambda b, o, tp=tp: _add(o, tp, delta[donor_pair][b - 1]))
                for (kind, bl), r in zip(specs, res):
                    rows.append({'exp': 'transplant', 'pair': pair, 'donor': donor_pair, 'kind': kind,
                                 'blocks': list(bl), 'target': list(q['target']), **r})
            print(f'[locate] transplant {pair} done', flush=True)
    finally:
        tap.remove()
    (d / 'causal.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    np.savez(d / 'delta.npz', **{p: v.cpu().numpy() for p, v in delta.items()})


def _replace(o: torch.Tensor, tp: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    o = o.clone()
    o[tp] = value.to(o.dtype)
    return o


def _add(o: torch.Tensor, tp: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    o = o.clone()
    o[tp] = o[tp] + vec.to(o.dtype)
    return o


def _patched_scores(rt, tap: ViTTap, image, ans, i_row, i_col, digits, n_patch, specs, edit_fn,
                    chunk: int = 10) -> list[dict]:
    """Teacher-forced answer scores of one image under several MLP-output edits (one edit spec per batch row)."""
    out = []
    for c0 in range(0, len(specs), chunk):
        part = specs[c0:c0 + chunk]
        rows_of = {}
        for r, (_, bl) in enumerate(part):
            for b in bl:
                rows_of.setdefault(b, []).append(r)
        tap.edits = {}
        for b, rs in rows_of.items():
            def f(o, b=b, rs=rs):
                o = o.clone()
                for r in rs:
                    sl = slice(r * n_patch, (r + 1) * n_patch)
                    o[sl] = edit_fn(b, o[sl])
                return o
            tap.edits[b] = f
        batch = sr.build_rows(rt, [image] * len(part), [ans] * len(part))
        with torch.no_grad():
            logits = rt.model(**{k: v.to(rt.device) if torch.is_tensor(v) else v for k, v in batch.items()},
                              use_cache=False, logits_to_keep=len(ans) + 1).logits
        tap.edits = {}
        tap.store.clear()
        out += sr.score_rows(logits, [ans] * len(part), i_row, i_col, len(ans), digits)
    return out


# ----------------------------------------------------------------------------- prepare

def _save(img: Image.Image, root: Path, rel: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return str(p)


def odd_item(root: Path, split: str, k: int, pair: str, g: int, target, variant: str, param: float, rng) -> io_.Item:
    orient = int(rng.integers(0, 4))
    jit = sa.jitter_map(g, g, rng)
    image, meta = lf.grid_image(pair, param, g, g, target, orient, jit, variant)
    path = _save(image, root, f'images/odd_{split}/{pair}_g{g}_{k:05d}_{variant}.png')
    ans = io_.ROWCOL_TEMPLATE.format(r=target[0] + 1, c=target[1] + 1) if split == 'train' else ''
    return io_.Item(f'odd_{split}_{pair}_g{g}_{k:05d}_{variant}', 'odd', split, path, sa.DIRECT_PROMPT, ans,
                    {'target': list(target), 'grid': g},
                    {'pair': pair, 'variant': variant, 'param': param, 'orient': orient, 'meta': meta,
                     'family': lf.family_of(pair), 'split': lf.split_of(pair)})


def stage_prepare(out: Path, args) -> None:
    rng = np.random.default_rng(SEED)
    odd = []
    for k in range(N_ODD_MAX):
        pair = lf.TRAIN_PAIRS[k % len(lf.TRAIN_PAIRS)]
        g = int(rng.choice(TRAIN_GRIDS))
        lo, hi = lf.LEVELS[pair]['subtle'], lf.LEVELS[pair]['salient']
        param = float(rng.uniform(lo, hi))
        variant = 'swap' if rng.random() < SWAP_FRACTION else 'std'
        target = (int(rng.integers(g)), int(rng.integers(g)))
        odd.append(odd_item(out, 'train', k, pair, g, target, variant, param, rng))
    # TextVQA short-answer items of experiment 7.1 (same images, same supervision), in a fixed random order
    vqa = [it for it in io_.read_items(Path(args.interp_run) / 'train.jsonl') if it.task == 'vqa']
    vqa = [vqa[int(i)] for i in rng.permutation(len(vqa))][:N_ODD_MAX]
    if len(vqa) < N_ODD_MAX:
        raise RuntimeError(f'only {len(vqa)} TextVQA items in the 7.1 training set')
    io_.dump_items(out / 'train_odd.jsonl', odd)
    io_.dump_items(out / 'train_vqa.jsonl', vqa)
    probe, k = [], 0
    for pair in lf.PROBE_PAIRS:
        for variant, per in (('swap', PROBE_SWAP_PER_CELL), ('std', PROBE_STD_PER_CELL)):
            for rep in range(per):
                for r in range(GRID):
                    for c in range(GRID):
                        probe.append(odd_item(out, 'probe', k, pair, GRID, (r, c), variant,
                                              float(lf.LEVELS[pair]['subtle']), rng))
                        k += 1
    io_.dump_items(out / 'probe.jsonl', probe)
    info = {'n_odd': len(odd), 'n_vqa': len(vqa), 'probe': len(probe), 'swap_fraction': SWAP_FRACTION,
            'train_pairs': lf.TRAIN_PAIRS, 'probe_pairs': lf.PROBE_PAIRS, 'seed': SEED}
    (out / 'prepare.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


# ----------------------------------------------------------------------------- adapters

def build_lora(rt, scope: str, rank: int, alpha: float) -> tf.LoRA:
    mods = lf.vit_modules(rt.model.model.visual, scope)
    if lf.parse_scope(scope)['kind'] == 'model_all':
        mods.update(tf.target_modules(rt.layers, 'all_linear'))
    return tf.LoRA(mods, rank, alpha)


def load_adapter(rt, out: Path, name: str) -> tf.LoRA | None:
    if name == 'base':
        return None
    state = torch.load(out / 'adapters' / f'{name}.pt', map_location='cpu', weights_only=False)
    lora = build_lora(rt, state['scope'], state['rank'], state['alpha'])
    lora.load(state)
    lora.attach()
    return lora


def stage_train(rt, out: Path, args) -> None:
    (out / 'adapters').mkdir(exist_ok=True)
    apath = out / 'adapters' / f'{args.name}.pt'
    if apath.exists():
        print(f'{apath} exists; skipping training', flush=True)
        return
    odd = io_.read_items(out / 'train_odd.jsonl')[:args.n_odd]
    vqa = io_.read_items(out / 'train_vqa.jsonl')[:args.n_odd]
    pool = odd + vqa
    n_visits = args.steps * args.step_samples
    rng = np.random.default_rng(SEED + args.seed)
    order = np.concatenate([rng.permutation(len(pool)) for _ in range(math.ceil(n_visits / len(pool)))])[:n_visits]
    items = [pool[int(i)] for i in order]
    torch.manual_seed(SEED + args.seed)
    lora = build_lora(rt, args.scope, args.rank, args.alpha)
    lora.attach()
    params = list(lora.parameters())
    model = rt.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.config.use_cache = False
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = max(1, int(0.05 * args.steps))
    lr_at = lambda s: args.lr * (s + 1) / warm if s < warm else \
        args.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm)))
    prompt_fn = lambda text: qr.prompt_text(rt, 1, text)
    log = out / 'adapters' / f'{args.name}_log.jsonl'
    if log.exists():
        log.unlink()
    t0 = time.time()
    try:
        for step in range(args.steps):
            chunk = items[step * args.step_samples:(step + 1) * args.step_samples]
            tokens = [(w // 32) * (h // 32) + 120 for w, h in (Image.open(it.image).size for it in chunk)]
            micro, cur = [], []
            for i in np.argsort(tokens):
                if cur and (len(cur) + 1) * max(tokens[j] for j in cur + [int(i)]) > args.micro_tokens:
                    micro.append(cur)
                    cur = []
                cur.append(int(i))
            micro.append(cur)
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            tot = {'odd': [0.0, 0], 'vqa': [0.0, 0]}
            for mb in micro:
                ss = [chunk[i] for i in mb]
                batch = io_.encode_batch(rt.processor, prompt_fn, ss, [Image.open(s.image).convert('RGB') for s in ss])
                per, _, _ = tf.batch_loss(model, batch, rt.device)
                (per.sum() / len(chunk)).backward()
                for s, v in zip(ss, per.tolist()):
                    tot[s.task][0] += v
                    tot[s.task][1] += 1
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            row = {'step': step, 'loss_odd': tot['odd'][0] / max(1, tot['odd'][1]),
                   'loss_vqa': tot['vqa'][0] / max(1, tot['vqa'][1]), 'lr': lr_at(step), 'grad_norm': float(gn),
                   'time': time.time() - t0}
            append_jsonl(log, [row])
            if step % 10 == 0:
                print(json.dumps(row), flush=True)
    finally:
        model.gradient_checkpointing_disable()
        model.eval()
        lora.detach()
    state = lora.state()
    state.update({'scope': args.scope, 'name': args.name, 'n_odd': args.n_odd, 'seed': args.seed, 'lr': args.lr,
                  'steps': args.steps, 'step_samples': args.step_samples, 'n_params': lora.n_params(),
                  'delta_norms': lora.delta_norms(), 'train_seconds': time.time() - t0,
                  'gpu': torch.cuda.get_device_name(), 'runtime': qr.describe(rt)})
    torch.save(state, apath)
    print(f'saved {apath} ({time.time() - t0:.0f}s, {lora.n_params()} params)', flush=True)


# ----------------------------------------------------------------------------- eval

def babyvision_cases(data_dir: Path) -> list[dict]:
    from eva.benchmark import load_cases, prompt_for
    rows, _ = load_cases(data_dir, 'babyvision', 'train')
    return [{'id': f"bv_{r['taskId']}", 'row': r, 'prompt': prompt_for(r, 'babyvision'), 'image': r['image_path'],
             'subtype': r['subtype']} for r in rows if r['subtype'] in BV_SUBTYPES]


def stage_eval(rt, out: Path, args) -> None:
    from eva.benchmark import score as bv_score
    name = args.name
    d = out / 'eval' / name
    d.mkdir(parents=True, exist_ok=True)
    sets = args.sets.split(',')
    gen = ior._orig_generate_batch
    lora = load_adapter(rt, out, name)
    t0 = time.time()
    try:
        if 'probe' in sets and not (d / 'probe.jsonl').exists():
            items = io_.read_items(out / 'probe.jsonl')
            res = ior.run_items(rt, gen, items, args.token_budget, 32)
            append_jsonl(d / 'probe.jsonl', [{'id': it.id, 'pair': it.meta['pair'], 'variant': it.meta['variant'],
                                              'split': it.meta['split'], 'family': it.meta['family'],
                                              'target': it.truth['target'], 'text': res[it.id]['text'],
                                              'finish_reason': res[it.id]['finish_reason'],
                                              'batch_size': res[it.id]['batch_size'],
                                              **oe.score({'target': it.truth['target']}, res[it.id])} for it in items])
            print(f'probe {name} {time.time() - t0:.0f}s', flush=True)
        if 'count' in sets and not (d / 'count.jsonl').exists():
            items = io_.read_items(Path(args.interp_run) / 'count.jsonl')
            res = ior.run_items(rt, gen, items, args.token_budget, 32)
            append_jsonl(d / 'count.jsonl', [{'id': it.id, 'truth': it.truth, 'text': res[it.id]['text'],
                                              **io_.score_count(res[it.id]['text'], it.truth)} for it in items])
            print(f'count {name} {time.time() - t0:.0f}s', flush=True)
        if 'coord' in sets and not (d / 'coord.jsonl').exists():
            ids = set(json.loads((Path(args.interp_run) / 'coord_ids.json').read_text()))
            samples = [s for s in tf.read_samples(Path(args.finetune_run) / 'test.jsonl') if s.id in ids]
            items = [io_.Item(s.id, 'coord', 'test', s.image, s.prompt) for s in samples]
            res = ior.run_items(rt, gen, items, args.token_budget, 64)
            append_jsonl(d / 'coord.jsonl', [{'id': s.id, 'family': s.family, 'fmt': s.fmt, 'text': res[s.id]['text'],
                                              'finish_reason': res[s.id]['finish_reason'], **tf.score(s, res[s.id]['text'])}
                                             for s in samples])
            print(f'coord {name} {time.time() - t0:.0f}s', flush=True)
        if 'general' in sets:
            tfr.stage_general(rt, out, Namespace(**{**vars(args), 'condition': name}))
            print(f'general {name} {time.time() - t0:.0f}s', flush=True)
        if 'babyvision' in sets and not (d / 'babyvision.jsonl').exists():
            cases = babyvision_cases(Path(args.data_dir))
            recs = []
            for i in range(0, len(cases), args.bv_batch):
                chunk = cases[i:i + args.bv_batch]
                outs = gen(rt, [([io_.fit_area(Image.open(c['image']), 1024 * 1024)], c['prompt']) for c in chunk],
                           args.bv_max_new)
                recs += [{'id': c['id'], 'subtype': c['subtype'], 'text': o['text'], 'finish_reason': o['finish_reason'],
                          'score': bv_score(c['row'], o['text'], 'babyvision')} for c, o in zip(chunk, outs)]
            append_jsonl(d / 'babyvision.jsonl', recs)
            print(f'babyvision {name} {time.time() - t0:.0f}s', flush=True)
    finally:
        if lora is not None:
            lora.detach()


def stage_detect(rt, out: Path, args) -> None:
    d = out / 'eval' / args.name
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'detectors.json'
    if path.exists():
        return
    lora = load_adapter(rt, out, args.name)
    try:
        res = detector_stats(rt, lf.PROBE_PAIRS, len(rt.model.model.visual.blocks), False)
    finally:
        if lora is not None:
            lora.detach()
    path.write_text(json.dumps(res, indent=1))


PREVIEW_CONDITIONS = {
    'single': None,
    'preview_first': 'The first image is a low-resolution preview of the second image. Answer about the second image. ',
    'blank_first': 'The first image is a low-resolution preview of the second image. Answer about the second image. ',
    'preview_after': 'The second image is a low-resolution preview of the first image. Answer about the first image. ',
}


def thumbnail(image: Image.Image, blank: bool = False) -> Image.Image:
    # half resolution, but at least the processor's minimum area (65536 pixels = 256 x 256)
    f = max(0.5, math.sqrt(65536 / (image.size[0] * image.size[1])))
    w, h = (int(math.ceil(v * f / 32)) * 32 for v in image.size)
    return io_.blank_like(image).resize((w, h)) if blank else image.resize((w, h), Image.Resampling.BICUBIC)


def stage_preview(rt, out: Path, args) -> None:
    """Part B pilot (training-free): a low-resolution preview placed before the image gives the raster-scanning
    decoder a reference for the typical entity before the first entities arrive (2.7 r2 start effect).  Controls: a
    blank first image with the same prompt, and the preview placed after the image.  Feature-absent probe items."""
    d = out / 'eval' / f'{args.name}_preview'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'preview.jsonl'
    done = {(r['id'], r['condition']) for r in read_jsonl(path)} if path.exists() else set()
    items = [it for it in io_.read_items(out / 'probe.jsonl') if it.meta['variant'] == 'swap']
    lora = load_adapter(rt, out, args.name)
    try:
        for cond, prefix in PREVIEW_CONDITIONS.items():
            todo = [it for it in items if (it.id, cond) not in done]
            for i in range(0, len(todo), 24):
                chunk = todo[i:i + 24]
                batch = []
                for it in chunk:
                    img = Image.open(it.image).convert('RGB')
                    if cond == 'single':
                        batch.append(([img], it.prompt))
                    elif cond == 'preview_after':
                        batch.append(([img, thumbnail(img)], prefix + it.prompt))
                    else:
                        batch.append(([thumbnail(img, cond == 'blank_first'), img], prefix + it.prompt))
                outs = ior._orig_generate_batch(rt, batch, 32)
                append_jsonl(path, [{'id': it.id, 'condition': cond, 'pair': it.meta['pair'], 'target': it.truth['target'],
                                     'text': o['text'], 'finish_reason': o['finish_reason'],
                                     **oe.score({'target': it.truth['target']}, o)} for it, o in zip(chunk, outs)])
            print(f'preview {cond} done', flush=True)
    finally:
        if lora is not None:
            lora.detach()


VERIFY_PAIRS = ('sqgap', 'term', 'closure', 'qo', 'dot', 'plusx')


def torch_salience(h: torch.Tensor, target: int, ridge: float = 1e-1, rank: int = 8) -> float:
    """GPU version of search_asymmetry.mahalanobis_salience (h: (n_entities, d))."""
    h = h.double()
    others = torch.cat([h[:target], h[target + 1:]])
    mu = others.mean(0)
    x = h[target] - mu
    _, sv, vt = torch.linalg.svd(others - mu, full_matrices=False)
    lam = sv ** 2 / max(1, len(others) - 1)
    floor = ridge * float(lam.mean()) + 1e-12
    proj = vt[:rank] @ x
    resid = x @ x - proj @ proj
    return float(torch.sqrt(torch.clamp((proj ** 2 / (lam[:rank] + floor)).sum() + resid / floor, min=0)))


def stage_verify(rt, out: Path, args) -> None:
    """Where did the representation change?  (1) Mahalanobis salience of the feature-absent target at every ViT
    block output (entity mean of its 16 patches); (2) for an adapter: probe accuracy with only its fc1 or only its
    fc2 LoRA active; (3) base-model per-neuron zeta vectors at the adapted blocks (for the weight analysis)."""
    d = out / 'eval' / args.name
    d.mkdir(parents=True, exist_ok=True)
    visual = rt.model.model.visual
    nb = len(visual.blocks)
    items = [it for it in io_.read_items(out / 'probe.jsonl') if it.meta['variant'] == 'swap' and it.meta['pair'] in VERIFY_PAIRS]
    lora = load_adapter(rt, out, args.name)
    res: dict = {'salience': []}
    store = {}
    hs = [blk.register_forward_hook(lambda m, a, o, b=b: store.__setitem__(b, (o[0] if isinstance(o, tuple) else o).detach()))
          for b, blk in enumerate(visual.blocks, start=1)]
    try:
        with torch.inference_mode():
            for i0 in range(0, len(items), 12):
                chunk = items[i0:i0 + 12]
                vit_forward(rt, [Image.open(it.image).convert('RGB') for it in chunk])
                meta = chunk[0].meta['meta']
                pidx = torch.tensor(lf.patch_index(oe.entity_tokens(meta)), device=rt.device)
                n_patch = 4 * meta['merged_grid'][0] * meta['merged_grid'][1]
                for j, it in enumerate(chunk):
                    t = it.truth['target'][0] * GRID + it.truth['target'][1]
                    row = {'id': it.id, 'pair': it.meta['pair'], 'target': it.truth['target'], 'sal': []}
                    for b in range(1, nb + 1):
                        h = store[b][j * n_patch:(j + 1) * n_patch][pidx].float().mean(1)
                        row['sal'].append(torch_salience(h, t))
                    res['salience'].append(row)
                store.clear()
    finally:
        for h in hs:
            h.remove()
    if lora is not None:
        probe = [it for it in io_.read_items(out / 'probe.jsonl') if it.meta['variant'] == 'swap']
        res['paths'] = {}
        for mode in ('fc1', 'fc2'):
            lora.active = {n for n in lora.names if n.endswith(f'linear_{mode}')}
            r = ior.run_items(rt, ior._orig_generate_batch, probe, args.token_budget, 32)
            res['paths'][mode] = [{'id': it.id, 'pair': it.meta['pair'], 'split': it.meta['split'],
                                   'correct': oe.score({'target': it.truth['target']}, r[it.id])['correct']} for it in probe]
            print(f'verify {args.name} path {mode} done', flush=True)
        lora.active = set()                  # base model for the neuron vectors
        blocks = sorted({int(n.split('.')[0][1:]) for n in lora.names if n.startswith('V')})
        if len(blocks) <= 6:
            vec: dict = {}
            detector_stats(rt, lf.PROBE_PAIRS, nb, False, tuple(blocks), vec)
            np.savez(d / 'neuron_zeta_base.npz', **vec)
        lora.detach()
    (d / 'verify.json').write_text(json.dumps(res))


def stage_write(rt, out: Path, args) -> None:
    """Where does the adapter write a target-specific signal?  For feature-absent probe images, the change of
    every ViT block output caused by the adapter, D = h_tuned - h_base (entity mean over its 16 patches), is
    projected on v_k = mean(D_target - mean D_distractors) fitted on even images; the deviation of the target's
    projection from the distractors' (in distractor standard deviations) is reported on odd images (zeta_write).
    For adapted MLPs the fc2-path contribution of each base neuron j is c_j = |s B2 A2[:, j]| * mean over images of
    (mean distractor activation - target activation) (entity maximum over patches), i.e. how much of the adapter's
    output at the target comes from neurons that are active on the distractors and silent on the target."""
    d = out / 'eval' / args.name
    visual = rt.model.model.visual
    nb = len(visual.blocks)
    lora = load_adapter(rt, out, args.name)
    if lora is None:
        raise ValueError('stage write needs an adapter')
    items = [it for it in io_.read_items(out / 'probe.jsonl') if it.meta['variant'] == 'swap' and it.meta['pair'] in VERIFY_PAIRS]
    store: dict = {}
    hs = [blk.register_forward_hook(lambda m, a, o, b=b: store.__setitem__(('h', b), (o[0] if isinstance(o, tuple) else o).detach()))
          for b, blk in enumerate(visual.blocks, start=1)]
    hs.append(visual.merger.register_forward_hook(lambda m, a, o: store.__setitem__(('h', 'merger'), o.detach())))
    mlp_blocks = sorted({int(n.split('.')[0][1:]) for n in lora.names if n.startswith('V') and 'fc2' in n})
    for b in mlp_blocks:
        hs.append(visual.blocks[b - 1].mlp.linear_fc2.register_forward_pre_hook(
            lambda m, a, b=b: store.__setitem__(('a', b), a[0].detach())))
    keys = list(range(1, nb + 1)) + ['merger']
    D = {k: [] for k in keys}               # per image (n_ent, d) change of the block output
    T = []
    dact = {b: [] for b in mlp_blocks}      # per image (4304,) mean distractor minus target activation (base)
    try:
        with torch.inference_mode():
            for i0 in range(0, len(items), 12):
                chunk = items[i0:i0 + 12]
                imgs = [Image.open(it.image).convert('RGB') for it in chunk]
                meta = chunk[0].meta['meta']
                pidx = torch.tensor(lf.patch_index(oe.entity_tokens(meta)), device=rt.device)
                n_patch = 4 * meta['merged_grid'][0] * meta['merged_grid'][1]
                res = {}
                for mode in ('base', 'tuned'):
                    lora.active = set() if mode == 'base' else None
                    vit_forward(rt, imgs)
                    res[mode] = dict(store)
                    store.clear()
                for j, it in enumerate(chunk):
                    t = it.truth['target'][0] * GRID + it.truth['target'][1]
                    T.append(t)
                    for k in keys:
                        if k == 'merger':
                            tok = torch.tensor(oe.entity_tokens(meta), device=rt.device)
                            n_tok = n_patch // 4
                            x = (res['tuned'][('h', k)] - res['base'][('h', k)])[j * n_tok:(j + 1) * n_tok][tok].float().mean(1)
                        else:
                            x = (res['tuned'][('h', k)] - res['base'][('h', k)])[j * n_patch:(j + 1) * n_patch][pidx].float().mean(1)
                        D[k].append(x.cpu())
                    for b in mlp_blocks:
                        a = res['base'][('a', b)][j * n_patch:(j + 1) * n_patch][pidx].float().amax(1)    # (n_ent, 4304)
                        others = torch.cat([a[:t], a[t + 1:]])
                        dact[b].append((others.mean(0) - a[t]).cpu())
    finally:
        for h in hs:
            h.remove()
        lora.active = None
    out_res: dict = {'zeta_write': {}, 'norm_ratio': {}}
    n = len(T)
    fit, test = list(range(0, n, 2)), list(range(1, n, 2))
    for k in keys:
        X = D[k]
        diff = torch.stack([X[i][T[i]] - torch.cat([X[i][:T[i]], X[i][T[i] + 1:]]).mean(0) for i in fit]).mean(0)
        if diff.norm() < 1e-8:
            out_res['zeta_write'][str(k)] = 0.0
            out_res['norm_ratio'][str(k)] = 0.0
            continue
        v = diff / diff.norm()
        zs = []
        for i in test:
            s_ = X[i] @ v
            o = torch.cat([s_[:T[i]], s_[T[i] + 1:]])
            zs.append(float((s_[T[i]] - o.mean()) / (o.std() + 1e-6)))
        out_res['zeta_write'][str(k)] = float(np.median(zs))
        out_res['norm_ratio'][str(k)] = float(np.mean([X[i][T[i]].norm() / torch.cat([X[i][:T[i]], X[i][T[i] + 1:]]).norm(dim=1).mean()
                                                       for i in test]))
    contrib = {}
    sc = lora.scale
    for b in mlp_blocks:
        key = lora._key(f'V{b}.mlp.linear_fc2')
        g = (sc * lora.B[key].detach().float() @ lora.A[key].detach().float()).norm(dim=0).cpu()     # (4304,)
        da = torch.stack(dact[b]).mean(0)
        contrib[str(b)] = {'c': (g * da).tolist(), 'g': g.tolist(), 'dact': da.tolist()}
    torch.save(contrib, d / 'write_contrib.pt')
    (d / 'write.json').write_text(json.dumps(out_res, indent=1))
    lora.detach()
    print(f"write {args.name}: {out_res['zeta_write']}", flush=True)


STAGES = {'write': stage_write, 'verify': stage_verify, 'preview': stage_preview, 'locate': stage_locate, 'train': stage_train, 'eval': stage_eval, 'detect': stage_detect}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--name', default='base', help="adapter name ('base' = no adapter)")
    p.add_argument('--scope', default='', help='LoRA scope (located_finetune.parse_scope)')
    p.add_argument('--n-odd', type=int, default=1200)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--sets', default='probe,count,coord,general,babyvision')
    p.add_argument('--interp-run', required=True, help='7.1 run directory (TextVQA items, count set, coord ids)')
    p.add_argument('--finetune-run', required=True, help='3.13 run directory (coordinate test samples)')
    p.add_argument('--data-dir', default='')
    p.add_argument('--vcot-gqa', default='')
    p.add_argument('--mmvp-csv', default='')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=32.0)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--step-samples', type=int, default=16)
    p.add_argument('--micro-tokens', type=int, default=3072)
    p.add_argument('--token-budget', type=int, default=16000)
    p.add_argument('--general-n', type=int, default=200)
    p.add_argument('--general-seed', type=int, default=20261020)
    p.add_argument('--general-batch', type=int, default=16)
    p.add_argument('--general-max-new', type=int, default=96)
    p.add_argument('--bv-batch', type=int, default=11)
    p.add_argument('--bv-max-new', type=int, default=1536)
    p.add_argument('--max-memory', default='')
    args = p.parse_args()
    import torch as _torch
    _torch.set_num_threads(16)            # CPU cap (common rules)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(',')
    if stages == ['prepare']:
        stage_prepare(out, args)
        return
    if 'train' in stages and not args.scope:
        raise ValueError('--scope is required for training')
    rt = tfr.load_model(args)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages, 'name': args.name,
                         'gpu': torch.cuda.get_device_name(), 'runtime': qr.describe(rt), 'args': vars(args),
                         'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]})
    meta.write_text(json.dumps(info, indent=1))
    for s in stages:
        t0 = time.time()
        STAGES[s](rt, out, args)
        print(f'stage {s} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
