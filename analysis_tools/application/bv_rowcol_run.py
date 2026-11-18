"""Experiment 7.4 stage drivers (results: docs/analysis/application/results_2026-11-11_babyvision_rowcol.md).

Stages (comma list via --stages):
  prep        (CPU) run directory: replica grids (evaluation set, training set), BabyVision cases (links to the 7.2
              cases + recoloured-cell reasoning cases), the mixed training files, links to the 7.2 base evaluations;
  bv          BabyVision answers of one condition: official question (reasoning), odd row/column, odd point,
              recoloured-cell point / row-column / reasoning;
  rep         replica answers: odd row/column, odd point, marked row/column (direct), odd and marked reasoning;
  up2         (base only) the same BabyVision and replica direct tasks with images upscaled 2x (resolution check);
  spot        (base only) answer agreement of batched vs single-sample greedy decoding on 20 replica prompts;
  train       LoRA training of one condition on its training file (multi-part adapter: decoder part restricted to
              image tokens + ViT part);
  coord, general, ordinal, ordinal_cot   3.13 coordinate test set, POPE/TextVQA/MMVP/GQA + text-only, the 72 grid
              images of 3.13 (direct and reasoning row/column);
  capture     residual stream at q0 / target tokens (7.2 periodic-code check) on the disc sweeps and RefCOCO;
  gatestat    DeltaNet alpha / beta of the gate layers at figure-line onsets vs other figure tokens (7.3C check).
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.application import bv_rowcol as bv
from analysis_tools.application import coord_locate as cl
from analysis_tools.application import coord_locate_run as clr
from analysis_tools.application import counting_gate as cg
from analysis_tools.application import located_finetune as lf
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_r2_run as tr2
from analysis_tools.position import targeted_finetune_run as tfr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

SEED = 20261111
N_EVAL_PER_FAMILY = 40
# Training mixture (pre-registered counts; see results doc section 3).
MIX = {'coord_313': 200, 'pt_mark': 200, 'pt_odd': 100, 'extent': 100, 'rc_mark': 200, 'rc_odd': 200,
       'cot_mark': 200, 'cot_odd': 200, 'instruct': 600}
COORD_313_STRATA = {('refcoco', 'point'): 50, ('refcoco', 'bbox'): 50, ('flat', 'point'): 20, ('flat', 'bbox'): 20,
                    ('grid', 'point'): 15, ('grid', 'bbox'): 15, ('texture', 'point'): 15, ('texture', 'bbox'): 15}
REP_COT_MAX = 768
ORD_COT_PROMPT = ('One figure is drawn in red. In which row and column is it? Count rows from top to bottom and '
                  'columns from left to right, starting at 1. Think step by step, then give your final answer on its '
                  'own last line as "Final answer: (row,column)".')


# ----------------------------------------------------------------------------- prep

def _render(job):
    idx, split, family, seed, root = job
    rep, img, marked = bv.make_replica(idx, split, family, seed)
    d = Path(root) / 'replicas' / split
    rep.image = str(d / f'{rep.id}.png')
    rep.marked_image = str(d / f'{rep.id}_mark.png')
    img.save(rep.image)
    marked.save(rep.marked_image)
    return bv.replica_dict(rep)


def render_set(out: Path, split: str, families: tuple[str, ...], n_per: int, seed0: int, workers: int) -> list[dict]:
    path = out / 'replicas' / f'{split}.jsonl'
    if path.exists():
        return [json.loads(l) for l in path.read_text().splitlines()]
    (out / 'replicas' / split).mkdir(parents=True, exist_ok=True)
    jobs = [(i, split, f, seed0 + 1000 * fi + i, str(out)) for fi, f in enumerate(families) for i in range(n_per)]
    with ProcessPoolExecutor(workers) as ex:
        reps = list(ex.map(_render, jobs, chunksize=8))
    path.write_text(''.join(json.dumps(r) + '\n' for r in reps))
    return reps


def as_rep(d: dict) -> bv.Replica:
    d = dict(d)
    d['canvas'], d['cell'], d['target'] = tuple(d['canvas']), tuple(d['cell']), tuple(d['target'])
    return bv.Replica(**d)


def _sample(sid: str, family: str, image: str, prompt: str, answer: str, meta: dict | None = None) -> tf.Sample:
    return tf.Sample(sid, family, 'train', image, 'text', {}, prompt, '', {'answer_text': answer + '\n', **(meta or {})})


def stage_prep(out: Path, args) -> None:
    src72 = Path(args.run72)
    for name in ('test.jsonl',):
        dst = out / name
        if not dst.exists():
            dst.symlink_to((src72 / name).resolve())
    # Base evaluations of 7.2 that use identical prompts and decoding (coordinate, general, text-only, ordinal).
    (out / 'eval' / 'base').mkdir(parents=True, exist_ok=True)
    for f in ('coord.jsonl', 'general.jsonl', 'text.jsonl', 'ordinal.jsonl'):
        s, dst = src72 / 'eval' / 'base' / f, out / 'eval' / 'base' / f
        if s.exists() and not dst.exists():
            dst.symlink_to(s.resolve())
    # BabyVision cases: the 7.2 cases plus BabyVision-style reasoning questions on the recoloured cells.
    cases = [json.loads(l) for l in (src72 / 'babyvision' / 'cases.jsonl').read_text().splitlines()]
    ann = {it['taskId']: it for it in json.loads(Path(args.bv_annotation).read_text())['items']}
    extra = []
    for c in cases:
        if c['task'] != 'red_rowcol':
            continue
        it = ann[c['taskId']]
        color = c['prompt'].split(' is drawn in ')[1].split('.')[0]
        q = bv.MARK_QUESTIONS[0].format(n=it['rows'] * it['cols'], noun=it['item_noun'], R=it['rows'], C=it['cols'],
                                         color=color) + bv.FINAL_SUFFIX
        extra.append({**c, 'id': c['id'].replace('red_rowcol', 'red_cot'), 'task': 'red_cot', 'prompt': q})
    if not (out / 'bv_annotation.json').exists():
        (out / 'bv_annotation.json').symlink_to(Path(args.bv_annotation).resolve())
    # Diagnostic variant (added after the first results): the official question prefixed with the grid size, which
    # only 5 of the 16 official questions state.
    for c in cases:
        if c['task'] == 'official':
            extra.append({**c, 'id': c['id'].replace('official', 'official_dims'), 'task': 'official_dims',
                          'prompt': f"The grid has {c['rows']} rows and {c['cols']} columns. " + c['prompt']})
    (out / 'bv_cases.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cases + extra))
    # Replicas.
    ev = render_set(out, 'eval', bv.FAMILIES, N_EVAL_PER_FAMILY, SEED, args.workers)
    n_train_img = sum(MIX[k] for k in ('pt_mark', 'pt_odd', 'extent', 'rc_mark', 'rc_odd', 'cot_mark', 'cot_odd'))
    per = int(math.ceil(n_train_img / len(bv.TRAIN_FAMILIES)))
    trn = render_set(out, 'train', bv.TRAIN_FAMILIES, per, SEED + 500000, args.workers)
    print(f'replicas: eval {len(ev)}, train {len(trn)}', flush=True)
    # Training mixture.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(trn))
    reps = [as_rep(trn[i]) for i in order]
    samples, nocot = [], []
    cur = 0

    def take(n):
        nonlocal cur
        r = reps[cur:cur + n]
        cur += n
        if len(r) < n:
            raise ValueError('not enough training replicas')
        return r
    for r in take(MIX['pt_mark']):
        samples.append(_sample(f'ptm_{r.id}', 'rep_pt_mark', r.marked_image, bv.question(r, 'mark_pt'),
                               bv.point_answer(r, f'One {r.noun} is drawn in {r.mark_color}.').rstrip('\n')))
    for r in take(MIX['pt_odd']):
        samples.append(_sample(f'pto_{r.id}', 'rep_pt_odd', r.image, bv.question(r, 'odd_pt'),
                               bv.point_answer(r, f'One {r.noun} differs from all the others.')))
    for r in take(MIX['extent']):
        img = r.image if rng.random() < 0.5 else r.marked_image
        samples.append(_sample(f'ext_{r.id}', 'rep_extent', img, bv.EXTENT_PROMPT.format(noun=r.noun), bv.extent_answer(r)))
    for r in take(MIX['rc_mark']):
        samples.append(_sample(f'rcm_{r.id}', 'rep_rc_mark', r.marked_image, bv.question(r, 'mark_rc'), bv.rowcol_json(r)))
    for r in take(MIX['rc_odd']):
        samples.append(_sample(f'rco_{r.id}', 'rep_rc_odd', r.image, bv.question(r, 'odd_rc'), bv.rowcol_json(r)))
    nocot = list(samples)
    bad = 0
    for kind, n in (('mark', MIX['cot_mark']), ('odd', MIX['cot_odd'])):
        for j, r in enumerate(take(n)):
            text, rc = bv.mediated_answer(r, marked=kind == 'mark')
            if rc != (r.target[0] + 1, r.target[1] + 1):
                bad += 1
            img = r.marked_image if kind == 'mark' else r.image
            samples.append(_sample(f'cot{kind[0]}_{r.id}', f'rep_cot_{kind}', img, bv.question(r, f'{kind}_cot', j), text))
            # Data ablation: the same image with the direct row/column question instead of the reasoning answer.
            nocot.append(_sample(f'rc{kind[0]}2_{r.id}', f'rep_rc_{kind}', img, bv.question(r, f'{kind}_rc'), bv.rowcol_json(r)))
    if bad:
        raise ValueError(f'{bad} reasoning answers give a wrong ordinal; fix the extent definition')
    base72 = tf.read_samples(src72 / 'train.jsonl')
    coord = []
    for (fam, fmt), n in COORD_313_STRATA.items():
        pool = [s for s in base72 if s.family == fam and s.fmt == fmt]
        coord += [pool[i] for i in rng.choice(len(pool), n, replace=False)]
    instr_pool = [s for s in tf.read_samples(Path(args.finetune_run) / 'train_scaled.jsonl') if s.fmt == 'text']
    instr = [instr_pool[i] for i in rng.choice(len(instr_pool), MIX['instruct'], replace=False)]
    tf.dump_samples(out / 'train.jsonl', coord + samples + instr)
    tf.dump_samples(out / 'train_nocot.jsonl', coord + nocot + instr)
    counts = {}
    for s in coord + samples + instr:
        counts[s.family] = counts.get(s.family, 0) + 1
    (out / 'train_counts.json').write_text(json.dumps(counts, indent=1))
    print('train.jsonl', len(coord + samples + instr), counts, flush=True)


# ----------------------------------------------------------------------------- adapters

class Adapter:
    """Multi-part adapter of one 7.4 condition: a decoder LoRA (optionally restricted to image tokens) and a ViT LoRA."""

    def __init__(self, rt: qr.Runtime, cond: str, rank: int, alpha: float):
        spec = bv.CONDITIONS[cond]
        self.cond, self.spec = cond, spec
        dec = {}
        if spec.get('all_linear'):
            dec.update(tf.target_modules(rt.layers, 'all_linear'))
        for l in spec['mlp']:
            for n in tf.MLP_NAMES:
                dec[f'L{l}.{n}'] = dict(rt.layers[l - 1].named_modules())[n]
        if spec['gate']:
            dec.update(cg.dn_modules(rt.layers, 'gate:' + ','.join(map(str, spec['gate']))))
        self.dec = tf.LoRA(dec, rank, alpha) if dec else None
        self.vit = tf.LoRA(lf.vit_modules(rt.model.model.visual, spec['vit']), rank, alpha) if spec['vit'] else None
        self.rt = rt

    def parts(self):
        return [p for p in (self.dec, self.vit) if p is not None]

    def attach(self, hook: bool = True) -> None:
        for p in self.parts():
            p.attach()
        if self.dec is not None and self.spec['positions'] and hook:
            cl.attach_positions(self.rt.model, self.dec, self.rt.model.config.image_token_id, self.spec['positions'])

    def detach(self) -> None:
        for p in self.parts():
            p.detach()

    def param_groups(self) -> list[dict]:
        groups = []
        if self.dec is not None:
            for part, lr in (('all_linear', bv.LR['all_linear']), ('mlp', bv.LR['mlp']), ('gate', bv.LR['gate'])):
                keys = [k for k in self.dec.names if (part == 'all_linear' and self.spec.get('all_linear')) or
                        (part == 'mlp' and not self.spec.get('all_linear') and '.mlp.' in k) or
                        (part == 'gate' and not self.spec.get('all_linear') and 'in_proj_' in k)]
                ps = [self.dec.A[self.dec._key(k)] for k in keys] + [self.dec.B[self.dec._key(k)] for k in keys]
                if ps:
                    groups.append({'params': ps, 'lr': lr, 'base_lr': lr, 'part': part})
        if self.vit is not None:
            groups.append({'params': list(self.vit.parameters()), 'lr': bv.LR['vit'], 'base_lr': bv.LR['vit'], 'part': 'vit'})
        return groups

    def n_params(self) -> int:
        return sum(p.n_params() for p in self.parts())

    def state(self) -> dict:
        return {'condition': self.cond, 'spec': self.spec, 'dec': self.dec.state() if self.dec else None,
                'vit': self.vit.state() if self.vit else None, 'n_params': self.n_params()}

    def load(self, state: dict) -> None:
        if state['condition'] != self.cond:
            raise ValueError(f"adapter of {state['condition']} is not {self.cond}")
        if self.dec is not None:
            self.dec.load(state['dec'])
        if self.vit is not None:
            self.vit.load(state['vit'])


def load_adapter(rt: qr.Runtime, out: Path, cond: str, args, hook: bool = True) -> Adapter | None:
    if cond == 'base':
        return None
    state = torch.load(out / 'adapters' / f'{cond}.pt', map_location='cpu', weights_only=False)
    ad = Adapter(rt, cond, state['rank'], state['alpha'])
    ad.load(state)
    ad.attach(hook)
    return ad


def stage_train(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    (out / 'adapters').mkdir(exist_ok=True)
    apath = out / 'adapters' / f'{cond}.pt'
    if apath.exists():
        print(f'{apath} exists; skipping training', flush=True)
        return
    spec = bv.CONDITIONS[cond]
    samples = tf.read_samples(out / spec.get('train_file', 'train.jsonl'))
    seed = spec['seed']
    torch.manual_seed(seed)
    ad = Adapter(rt, cond, args.rank, args.alpha)
    ad.attach()
    model = rt.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.config.use_cache = False
    groups = ad.param_groups()
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.999))
    rng = np.random.default_rng(seed)
    epochs = spec.get('epochs', args.epochs)
    order = np.concatenate([rng.permutation(len(samples)) for _ in range(epochs)])
    n_steps = len(order) // args.step_samples
    warm = max(1, int(0.05 * n_steps))

    def lr_scale(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_steps - warm)))
    prompt_fn = lambda text: qr.prompt_text(rt, 1, text)
    log = out / 'adapters' / f'{cond}_log.jsonl'
    if log.exists():
        log.unlink()
    params = [p for g in groups for p in g['params']]
    t0 = time.time()
    for step in range(n_steps):
        items = [samples[int(i)] for i in order[step * args.step_samples:(step + 1) * args.step_samples]]
        sizes = [Image.open(s.image).size for s in items]
        tokens = [(w // 32) * (h // 32) + 150 + len(s.meta.get('answer_text', '')) // 3 for (w, h), s in zip(sizes, items)]
        micro, cur = [], []
        for i in np.argsort(tokens):
            if cur and (len(cur) + 1) * max(tokens[j] for j in cur + [int(i)]) > args.micro_tokens:
                micro.append(cur)
                cur = []
            cur.append(int(i))
        micro.append(cur)
        for g in opt.param_groups:
            g['lr'] = g['base_lr'] * lr_scale(step)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for mb in micro:
            ss = [items[i] for i in mb]
            batch = tf.encode_batch(rt.processor, prompt_fn, ss, [Image.open(s.image).convert('RGB') for s in ss])
            per, _, _ = tf.batch_loss(model, batch, rt.device)
            (per.sum() / len(items)).backward()
            tot += float(per.sum())
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        row = {'step': step, 'loss': tot / len(items), 'lr_scale': lr_scale(step), 'grad_norm': float(gn),
               'micro': len(micro), 'time': time.time() - t0}
        append_jsonl(log, [row])
        if step % 10 == 0:
            print(json.dumps(row), flush=True)
    state = ad.state()
    state.update({'rank': args.rank, 'alpha': args.alpha, 'lr': {g['part']: g['base_lr'] for g in groups},
                  'epochs': epochs, 'step_samples': args.step_samples, 'n_steps': n_steps,
                  'train_seconds': time.time() - t0, 'n_train': len(samples), 'train_file': spec.get('train_file', 'train.jsonl'),
                  'delta_norms': {**(ad.dec.delta_norms() if ad.dec else {}), **(ad.vit.delta_norms() if ad.vit else {})},
                  'gpu': torch.cuda.get_device_name(0), 'runtime': qr.describe(rt)})
    torch.save(state, apath)
    model.gradient_checkpointing_disable()
    model.eval()
    ad.detach()
    print(f'saved {apath} ({ad.n_params()} params, {time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- evaluation helpers

def upscale(img: Image.Image, factor: int) -> Image.Image:
    return img if factor == 1 else img.resize((img.width * factor, img.height * factor), Image.Resampling.BICUBIC)


def run_gen(rt, items: list[tuple[str, Image.Image, str]], max_new: int, bs: int) -> dict[str, dict]:
    """items (id, image, prompt), grouped by image size (fewer padded tokens); greedy, left-padded batches."""
    res = {}
    order = sorted(range(len(items)), key=lambda i: items[i][1].size)
    for i in range(0, len(order), bs):
        chunk = [items[j] for j in order[i:i + bs]]
        outs = qr.generate_batch(rt, [([im], p) for _, im, p in chunk], max_new)
        for (cid, _, _), o in zip(chunk, outs):
            res[cid] = {**o, 'batch_size': len(chunk)}
    return res


def score_rc(text: str, finish: str, truth, cot: bool) -> dict:
    rc = bv.final_rowcol(text) if cot else bv.json_rowcol(text)
    fmt = bv.format_class(text, finish, rc)
    return {'rc': rc, 'format': fmt, 'correct': rc is not None and list(rc) == list(truth),
            'row_ok': rc is not None and rc[0] == truth[0], 'col_ok': rc is not None and rc[1] == truth[1]}


def stage_bv(rt, out: Path, args, factor: int = 1) -> None:
    from eva.benchmark import load_cases, score as bv_score
    cond = args.condition
    d = tfr.cond_dir(out, cond)
    path = d / ('babyvision.jsonl' if factor == 1 else f'babyvision_up{factor}.jsonl')
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    cases = [json.loads(l) for l in (out / 'bv_cases.jsonl').read_text().splitlines()]
    tasks = ('official', 'odd_rowcol', 'odd_point', 'red_point', 'red_rowcol', 'red_cot', 'official_dims') if factor == 1 else \
        ('official', 'odd_rowcol', 'red_rowcol', 'red_cot')
    rows = {r['taskId']: r for r in load_cases(Path(args.data_dir), 'babyvision', 'train')[0]}
    t0 = time.time()
    for task in tasks:
        group = [c for c in cases if c['task'] == task and c['id'] not in done]
        if not group:
            continue
        cot = task in ('official', 'red_cot', 'official_dims')
        max_new = args.bv_max_new if cot else args.max_new_tokens
        res = run_gen(rt, [(c['id'], upscale(Image.open(c['image']).convert('RGB'), factor), c['prompt']) for c in group],
                      max_new, args.cot_batch if cot else args.batch)
        recs = []
        for c in group:
            o = res[c['id']]
            rec = {'id': c['id'], 'task': task, 'taskId': c['taskId'], 'text': o['text'], 'finish_reason': o['finish_reason'],
                   'n_tokens': o['n_tokens'], 'batch_size': o['batch_size']}
            if task.endswith('point'):
                pt = tf.parse_answer(o['text'], 'point')
                rec.update(cl.point_hit(pt, c['truth']['box'], tuple(c['canvas'])))
                rec['coords'] = pt
            else:
                rec.update(score_rc(o['text'], o['finish_reason'], c['truth']['rc'], cot))
                if task in ('official', 'official_dims'):
                    rec['strict'] = bool(bv_score(rows[c['taskId']], o['text'], 'babyvision')['correct'])
            recs.append(rec)
        append_jsonl(path, recs)
        print(f'bv{factor} {cond} {task}: {sum(r.get("correct", r.get("hit", False)) for r in recs)}/{len(recs)} '
              f'({time.time() - t0:.0f}s)', flush=True)


REP_TASKS = ('odd_rc', 'mark_rc', 'odd_pt', 'odd_cot', 'mark_cot')


def stage_rep(rt, out: Path, args, factor: int = 1) -> None:
    cond = args.condition
    d = tfr.cond_dir(out, cond)
    path = d / ('replica.jsonl' if factor == 1 else f'replica_up{factor}.jsonl')
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    reps = [as_rep(json.loads(l)) for l in (out / 'replicas' / 'eval.jsonl').read_text().splitlines()]
    tasks = REP_TASKS if factor == 1 else ('odd_rc', 'mark_rc', 'odd_cot', 'mark_cot')
    t0 = time.time()
    for task in tasks:
        items = []
        for k, r in enumerate(reps):
            cid = f'{r.id}|{task}'
            if cid in done:
                continue
            img = Image.open(r.marked_image if task.startswith('mark') else r.image).convert('RGB')
            items.append((cid, upscale(img, factor), bv.question(r, task, k)))
        if not items:
            continue
        cot = task.endswith('cot')
        res = run_gen(rt, items, REP_COT_MAX if cot else args.max_new_tokens, args.cot_batch if cot else args.batch)
        byid = {r.id: r for r in reps}
        recs = []
        for cid, _, _ in items:
            r = byid[cid.split('|')[0]]
            o = res[cid]
            truth = [r.target[0] + 1, r.target[1] + 1]
            rec = {'id': cid, 'rep': r.id, 'task': task, 'family': r.family, 'density': r.density(), 'rows': r.rows,
                   'cols': r.cols, 'text': o['text'], 'finish_reason': o['finish_reason'], 'n_tokens': o['n_tokens'],
                   'batch_size': o['batch_size'], 'truth': truth}
            if task == 'odd_pt':
                pt = tf.parse_answer(o['text'], 'point')
                b = r.boxes[r.target[0] * r.cols + r.target[1]]
                rec.update(cl.point_hit(pt, b, r.canvas))
                cell = bv.cell_of_point(pt, r.rows, r.cols, r.boxes, r.canvas)
                rec.update({'coords': pt, 'cell': None if cell is None else [cell[0] + 1, cell[1] + 1],
                            'cell_hit': cell is not None and [cell[0] + 1, cell[1] + 1] == truth})
            else:
                rec.update(score_rc(o['text'], o['finish_reason'], truth, cot))
            recs.append(rec)
        append_jsonl(path, recs)
        key = 'cell_hit' if task == 'odd_pt' else 'correct'
        print(f'rep{factor} {cond} {task}: {sum(r[key] for r in recs)}/{len(recs)} ({time.time() - t0:.0f}s)', flush=True)


def stage_spot(rt, out: Path, args) -> None:
    """Batched vs batch-size-1 greedy answers on 20 replica prompts (10 direct, 10 reasoning)."""
    path = tfr.cond_dir(out, 'base') / 'spotcheck.json'
    if path.exists():
        return
    reps = [as_rep(json.loads(l)) for l in (out / 'replicas' / 'eval.jsonl').read_text().splitlines()][::10][:20]
    items = [(f'{r.id}|{t}', Image.open(r.image).convert('RGB'), bv.question(r, t)) for r, t in
             zip(reps, ['odd_rc'] * 10 + ['odd_cot'] * 10)]
    batched = run_gen(rt, items, REP_COT_MAX, args.cot_batch)
    agree = []
    for cid, im, p in items:
        single = qr.generate(rt, qr.build(rt, [im], p), REP_COT_MAX)
        cot = cid.endswith('cot')
        a = bv.final_rowcol(batched[cid]['text']) if cot else bv.json_rowcol(batched[cid]['text'])
        b = bv.final_rowcol(single['text']) if cot else bv.json_rowcol(single['text'])
        agree.append({'id': cid, 'batched': a, 'single': b, 'same_answer': a == b,
                      'same_text': batched[cid]['text'] == single['text']})
    path.write_text(json.dumps({'n': len(agree), 'answer_agreement': sum(x['same_answer'] for x in agree),
                                'text_identical': sum(x['same_text'] for x in agree), 'batch': args.cot_batch,
                                'items': agree}, indent=1))
    print(f"spot: answers agree {sum(x['same_answer'] for x in agree)}/{len(agree)}", flush=True)


def _ordinal_cot(rt, out: Path, args) -> None:
    d = tfr.cond_dir(out, args.condition)
    path = d / 'ordinal_cot.jsonl'
    if path.exists():
        return
    samples = [s for s in tf.read_samples(out / 'test.jsonl') if s.family in ('grid06', 'grid10', 'grid20')]
    res = run_gen(rt, [(s.id, Image.open(s.image).convert('RGB'), ORD_COT_PROMPT) for s in samples], 512, args.cot_batch)
    recs = []
    for s in samples:
        o = res[s.id]
        truth = [s.meta['row'] + 1, s.meta['col'] + 1]
        recs.append({'id': s.id, 'family': s.family, 'text': o['text'], 'finish_reason': o['finish_reason'],
                     'truth': truth, **score_rc(o['text'], o['finish_reason'], truth, True)})
    append_jsonl(path, recs)
    print(f"ordinal_cot {args.condition}: {sum(r['correct'] for r in recs)}/{len(recs)}", flush=True)


def stage_capture(rt, out: Path, args) -> None:
    """7.2 periodic-code check: residuals at q0 and the target-token mean on the disc sweeps (x, y) and RefCOCO."""
    cond = args.condition
    dpath = Path(args.features) / 'capture'
    dpath.mkdir(parents=True, exist_ok=True)
    path = dpath / f'{cond}.npz'
    if path.exists():
        return
    ad = load_adapter(rt, out, cond, args, hook=False)
    ua = Namespace(round1=args.round1, output=args.finetune_run)
    try:
        rows = tr2.build_rows(rt, tr2.disc_units(ua, 1, ('sweep_h', 'sweep_v')) + tr2.disc_units(ua, 2, ('textconv', 'refcoco')))
        resid, logp = [], []
        for chunk in tr2.batches(rows, args.batch):
            if ad is not None and ad.dec is not None:
                Ltot = max(r['length'] for r in chunk)
                kind = ad.spec['positions']
                ad.dec.pos_mask = None if kind is None else torch.tensor(np.stack([cl.span_mask(kind, r['span'], Ltot) for r in chunk]))
            o = tr2.light_forward(rt, chunk, clr.CAPTURE_LAYERS)
            resid.append(np.stack([o['resid'][l].numpy()[:, [0, 2]] for l in clr.CAPTURE_LAYERS], 2).astype(np.float16))
            logp.append(o['logp'])
    finally:
        if ad is not None:
            ad.detach()
    np.savez(path, resid=np.concatenate(resid), logp=np.concatenate(logp), sites=np.array(['q0', 'target']),
             truth=np.array([r['unit'].truth for r in rows]), family=np.array([r['unit'].family for r in rows]),
             unit=np.array([r['unit'].id for r in rows]), layers=np.array(clr.CAPTURE_LAYERS))
    print(f'capture {cond}: {len(rows)} rows', flush=True)


def stage_gatestat(rt, out: Path, args) -> None:
    """Mean DeltaNet alpha / beta per head of the gate layers (and all DeltaNet layers) at image tokens of the replica
    evaluation grids, split into the first token row / column of each entity row / column ('onset') and other
    entity tokens. A per-condition change of the onset-minus-other contrast at layers 6-10 is the 7.3C signature
    of a changed counting gate."""
    cond = args.condition
    d = tfr.cond_dir(out, cond)
    path = d / 'gatestat.json'
    if path.exists():
        return
    ad = load_adapter(rt, out, cond, args)
    reps = [as_rep(json.loads(l)) for l in (out / 'replicas' / 'eval.jsonl').read_text().splitlines()][::2]
    store, hs = {}, []
    for l in cg.DN_LAYERS:
        mod = rt.layers[l - 1].linear_attn
        hs.append(mod.in_proj_a.register_forward_hook(lambda m, a, o, l=l: store.__setitem__(('a', l), o.detach())))
        hs.append(mod.in_proj_b.register_forward_hook(lambda m, a, o, l=l: store.__setitem__(('b', l), o.detach())))
    acc = {}
    try:
        with torch.inference_mode():
            for r in reps:
                img = Image.open(r.image).convert('RGB')
                batch = qr.build(rt, [img], bv.question(r, 'odd_rc'))
                span = qr.image_spans(rt, batch)[0]
                H, W = span['grid']
                cls = onset_classes(r, H, W)
                rt.model(**batch, use_cache=False, logits_to_keep=1)
                for l in cg.DN_LAYERS:
                    mod = rt.layers[l - 1].linear_attn
                    a = store[('a', l)][0, span['start']:span['end']].float()
                    b = store[('b', l)][0, span['start']:span['end']].float()
                    alpha = torch.exp(-mod.A_log.float().exp() * torch.nn.functional.softplus(a + mod.dt_bias.float()))
                    beta = b.sigmoid()
                    for k in (1, 2, 3):
                        m = torch.tensor(cls == k, device=alpha.device)
                        if m.any():
                            s = acc.setdefault((l, k), [torch.zeros(cg.N_HEADS, device=alpha.device),
                                                        torch.zeros(cg.N_HEADS, device=alpha.device), 0])
                            s[0] += alpha[m].sum(0)
                            s[1] += beta[m].sum(0)
                            s[2] += int(m.sum())
                store.clear()
    finally:
        for h in hs:
            h.remove()
        if ad is not None:
            ad.detach()
    res = [{'layer': k[0], 'cls': k[1], 'alpha': (v[0] / v[2]).tolist(), 'beta': (v[1] / v[2]).tolist(), 'n': v[2]}
           for k, v in acc.items()]
    path.write_text(json.dumps(res))
    print(f'gatestat {cond}: {len(reps)} images', flush=True)


def onset_classes(r: bv.Replica, H: int, W: int) -> np.ndarray:
    """Per merged token (raster order): 1 = token whose 32-pixel row contains the top edge of an entity row,
    2 = token whose 32-pixel column contains the left edge of an entity column (and not class 1), 3 = other tokens
    inside the grid extent, 0 = margin."""
    cls = np.zeros((H, W), np.int64)
    x0, y0, x1, y1 = r.extent
    ox, oy = 12, 12
    row_tops = {int((oy + i * r.cell[1]) // 32) for i in range(r.rows)}
    col_lefts = {int((ox + j * r.cell[0]) // 32) for j in range(r.cols)}
    for ty in range(H):
        for tx in range(W):
            if not (x0 <= tx * 32 + 16 <= x1 + 16 and y0 <= ty * 32 + 16 <= y1 + 16):
                continue
            cls[ty, tx] = 1 if ty in row_tops else (2 if tx in col_lefts else 3)
    return cls.reshape(-1)


def with_adapter(fn):
    def run(rt, out, args):
        ad = load_adapter(rt, out, args.condition, args)
        try:
            fn(rt, out, args)
        finally:
            if ad is not None:
                ad.detach()
    return run


STAGES = {
    'train': stage_train,
    'bv': with_adapter(stage_bv),
    'rep': with_adapter(stage_rep),
    'up2': with_adapter(lambda rt, out, args: (stage_bv(rt, out, args, 2), stage_rep(rt, out, args, 2))),
    'spot': stage_spot,
    'coord': with_adapter(lambda rt, out, args: tfr.stage_coord(rt, out, args)),
    'general': with_adapter(lambda rt, out, args: tfr.stage_general(rt, out, args)),
    'ordinal': with_adapter(lambda rt, out, args: clr._ordinal(rt, out, args)),
    'ordinal_cot': with_adapter(_ordinal_cot),
    'capture': stage_capture,
    'gatestat': stage_gatestat,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--condition', default='base')
    p.add_argument('--run72', required=True)
    p.add_argument('--finetune-run', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--data-dir', default='')
    p.add_argument('--bv-annotation', default='')
    p.add_argument('--vcot-gqa', default='')
    p.add_argument('--mmvp-csv', default='')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=32.0)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--step-samples', type=int, default=16)
    p.add_argument('--micro-tokens', type=int, default=4096)
    p.add_argument('--token-budget', type=int, default=16000)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--general-n', type=int, default=200)
    p.add_argument('--general-seed', type=int, default=20261020)
    p.add_argument('--general-batch', type=int, default=16)
    p.add_argument('--general-max-new', type=int, default=96)
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--cot-batch', type=int, default=16)
    p.add_argument('--bv-max-new', type=int, default=1024)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in args.stages.split(',') if s]
    if stages == ['prep']:
        stage_prep(out, args)
        return
    conds = args.condition.split(',')
    for c in conds:
        if c != 'base' and c not in bv.CONDITIONS:
            raise ValueError(f'Unknown condition {c}')
    if len(conds) > 1 and 'train' in stages:
        raise ValueError('train takes one condition')
    rt = qr.load(args.model)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['model'] = qr.describe(rt)
    info['runs'].append({'stages': stages, 'condition': args.condition, 'started': datetime.now(timezone.utc).isoformat(),
                         'gpu': torch.cuda.get_device_name(0), 'batch': args.batch, 'cot_batch': args.cot_batch})
    meta.write_text(json.dumps(info, indent=1))
    for cond in conds:
        args.condition = cond
        for s in stages:
            t0 = time.time()
            STAGES[s](rt, out, args)
            append_jsonl(out / 'stage_seconds.jsonl', [{'condition': cond, 'stage': s, 'seconds': time.time() - t0,
                                                        'gpu': torch.cuda.get_device_name(0),
                                                        'finished': datetime.now(timezone.utc).isoformat()}])


if __name__ == '__main__':
    main()
