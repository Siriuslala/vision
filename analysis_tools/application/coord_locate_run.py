"""Experiment 7.2 stage drivers (results: docs/analysis/application/results_2026-11-05_located_finetune.md).

Stages (comma list via --stages):
  link        (CPU) run directory: links to the 3.13 round-2 training / test samples and to the cited 3.13 adapters and
              evaluations; nested training subsets train_n<N>.jsonl (data efficiency);
  bvprep      (CPU) BabyVision "Find the different" cases on the 16 annotated grids;
  train       LoRA training of one condition (3.13 trainer settings: rank 8, alpha 32, lr 1e-4, 16 samples per step,
              cosine schedule) with the 7.2 position masks; --epochs sets the passes over the condition's training file;
  coord       greedy coordinate answers on the 3.13 test split (997 items);
  general     POPE / TextVQA / MMVP / GQA (200 each) and the 45 text-only questions (3.13 protocol);
  digits      teacher-forced digit readout (first digit forced) on the disc sweeps (x, y; every 2nd step) and RefCOCO
              test points: P(true first digit), P(true second digit), second-digit gain, the first-digit margin m;
  capture     residual stream at q0 and the target-token mean for selected layers (geometry check of the prediction);
  babyvision  answers to the BabyVision cases.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.application import coord_locate as cl
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_r2 as t2
from analysis_tools.position import targeted_finetune_r2_run as tr2
from analysis_tools.position import targeted_finetune_run as tfr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

SEED = 20261105
SUBSET_SIZES = (200, 800)
LINKED_CONDITIONS = ('all_linear', 'early1_4_img', 'mlp_all')
CAPTURE_LAYERS = (0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 28, 32)
BV_POINTS_PER_IMAGE = 6


# ----------------------------------------------------------------------------- link / data

def stage_link(out: Path, args) -> None:
    src = Path(args.finetune_run)
    for name in ('train.jsonl', 'test.jsonl'):
        dst = out / name
        if not dst.exists():
            dst.symlink_to(src / name)
    for sub in ('adapters', 'eval'):
        (out / sub).mkdir(exist_ok=True)
    for cond in LINKED_CONDITIONS:
        for suffix in ('.pt', '_log.jsonl'):
            dst = out / 'adapters' / f'{cond}{suffix}'
            if not dst.exists() and (src / 'adapters' / f'{cond}{suffix}').exists():
                dst.symlink_to((src / 'adapters' / f'{cond}{suffix}').resolve())
        (out / 'eval' / cond).mkdir(exist_ok=True)
        for f in ('coord.jsonl', 'general.jsonl', 'text.jsonl', 'spotcheck.json'):
            s, dst = src / 'eval' / cond / f, out / 'eval' / cond / f
            if s.exists() and not dst.exists():
                dst.symlink_to(s.resolve())
    base = out / 'eval' / 'base'
    base.mkdir(exist_ok=True)
    for f in ('coord.jsonl', 'general.jsonl', 'text.jsonl'):
        s, dst = src / 'eval' / 'base' / f, base / f
        if s.exists() and not dst.exists():
            dst.symlink_to(s.resolve())
    samples = tf.read_samples(out / 'train.jsonl')
    subs = cl.nested_subsets([s.family + '|' + s.fmt for s in samples], SUBSET_SIZES, SEED)
    for n, idx in subs.items():
        path = out / f'train_n{n}.jsonl'
        if not path.exists():
            tf.dump_samples(path, [samples[i] for i in idx])
    print('linked; subsets', {n: len(v) for n, v in subs.items()}, flush=True)


def stage_bvprep(out: Path, args) -> None:
    """BabyVision cases on the 16 'Find the different' grids (annotation of experiment 1.3)."""
    from analysis_tools.behavior.decomposition import JSON_ONLY, LOCATE_SUFFIX
    from analysis_tools.behavior.decomposition_babyvision import FOREGROUND_THRESHOLD, pad32
    from eva.benchmark import load_cases, prompt_for
    rows = {r['taskId']: r for r in load_cases(Path(args.data_dir), 'babyvision', 'train')[0]}
    ann = json.loads(Path(args.bv_annotation).read_text())
    d = out / 'babyvision'
    (d / 'images').mkdir(parents=True, exist_ok=True)
    cases = []
    for item in ann['items']:
        tid, nr, nc, noun = item['taskId'], item['rows'], item['cols'], item['item_noun']
        row = rows[tid]
        image = Image.open(row['image_path']).convert('RGB')
        bg = item['background']
        base = pad32(image, bg)
        bpath = d / 'images' / f'{tid}.png'
        base.save(bpath)
        target = tuple(item['answer_rc'])
        t_index = target[0] * nc + target[1]
        note = (' ' + item['note']) if item['note'] else ''
        canvas = list(base.size)
        common = {'taskId': tid, 'rows': nr, 'cols': nc, 'canvas': canvas}
        cases.append({'id': f'{tid}|official', 'task': 'official', 'image': str(bpath), 'prompt': prompt_for(row, 'babyvision'),
                      'truth': {'rc': [target[0] + 1, target[1] + 1]}, **common})
        q5 = f'One {noun} differs from all the others.' + note
        cases.append({'id': f'{tid}|odd_rowcol', 'task': 'odd_rowcol', 'image': str(bpath),
                      'prompt': q5 + LOCATE_SUFFIX['row_column'] + JSON_ONLY, 'truth': {'rc': [target[0] + 1, target[1] + 1]}, **common})
        cases.append({'id': f'{tid}|odd_point', 'task': 'odd_point', 'image': str(bpath),
                      'prompt': q5 + LOCATE_SUFFIX['point'] + JSON_ONLY,
                      'truth': {'box': item['tight_boxes'][t_index], 'rc': [target[0] + 1, target[1] + 1]}, **common})
        rng = np.random.default_rng(SEED + tid)
        others = [k for k in range(nr * nc) if k != t_index]
        picks = [t_index] + [int(k) for k in rng.choice(others, BV_POINTS_PER_IMAGE - 1, replace=False)]
        # Marker colour: of red / green / blue, the one farthest from the image's median foreground and background
        # colours (several grids are drawn in crimson, where a red marker would be ambiguous).
        arr = np.asarray(image).astype(int)
        fg = arr[np.abs(arr - np.array(bg)).max(-1) > FOREGROUND_THRESHOLD]
        ref = [np.median(fg, 0) if len(fg) else np.array(bg), np.array(bg)]
        palette = {'red': (255, 0, 0), 'green': (0, 170, 0), 'blue': (0, 0, 255)}
        cname = max(palette, key=lambda n: min(np.linalg.norm(np.array(palette[n]) - r) for r in ref))
        color = palette[cname]
        for j, k in enumerate(picks):
            a = np.asarray(image).astype(int).copy()
            x0, y0, x1, y1 = item['cell_boxes'][k]
            cell = a[y0:y1, x0:x1]
            cell[np.abs(cell - np.array(bg)).max(-1) > FOREGROUND_THRESHOLD] = color
            img = pad32(Image.fromarray(a.astype(np.uint8)), bg)
            p = d / 'images' / f'{tid}_{cname}_{j}.png'
            img.save(p)
            rc = [k // nc + 1, k % nc + 1]
            q6 = f'One {noun} is drawn in {cname}.'
            for task, suffix in (('red_point', LOCATE_SUFFIX['point']), ('red_rowcol', LOCATE_SUFFIX['row_column'])):
                cases.append({'id': f'{tid}|{task}|{j}', 'task': task, 'image': str(p), 'prompt': q6 + suffix + JSON_ONLY,
                              'truth': {'box': item['tight_boxes'][k], 'rc': rc}, **common})
    (d / 'cases.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cases))
    print(f'{len(cases)} BabyVision cases', flush=True)


ROWCOL_PROMPT = ('One figure is drawn in red. Give its position. Return only {"row": r, "column": c}, counting rows from '
                 'top to bottom and columns from left to right, starting at 1. Output only the JSON, with no explanation.')


def stage_fixprep(out: Path, args) -> None:
    """Format-drift repair data (7.2): the 3200 coordinate samples + one row/column question per 3.13 training grid
    image (answer in the base model's plain JSON format, the target cell from the grid metadata) + the 1671
    self-distilled instruction samples of 3.13 round 2 (answers are the base model's own greedy outputs)."""
    src = Path(args.finetune_run)
    base = tf.read_samples(out / 'train.jsonl')
    rowcol = []
    for smp in base:
        if smp.family == 'grid' and smp.fmt == 'point':
            r, c = smp.meta['target']
            rowcol.append(tf.Sample(f'rowcol_{smp.id}', 'grid_rowcol', 'train', smp.image, 'text', smp.truth, ROWCOL_PROMPT,
                                    '', {'answer_text': json.dumps({'row': int(r) + 1, 'column': int(c) + 1}) + '\n'}))
    instruct = [smp for smp in tf.read_samples(src / 'train_scaled.jsonl') if smp.fmt == 'text']
    tf.dump_samples(out / 'train_fix.jsonl', base + rowcol + instruct)
    print(f'train_fix: {len(base)} coordinate + {len(rowcol)} row/column + {len(instruct)} instruction samples', flush=True)


# ----------------------------------------------------------------------------- adapters

def make_lora(rt: qr.Runtime, cond: str, rank: int, alpha: float) -> tf.LoRA:
    spec = tf.CONDITIONS[cond]
    r = spec.get('rank', rank)
    return tf.LoRA(tf.target_modules(rt.layers, cond), r, alpha * r / rank)


def attach_trained(rt: qr.Runtime, out: Path, cond: str, with_hook: bool = True) -> tf.LoRA | None:
    """Load an adapter of out/adapters and attach it with its position restriction (input_ids hook when with_hook;
    analysis passes with inputs_embeds set lora.pos_mask per batch)."""
    if cond == 'base':
        return None
    state = torch.load(out / 'adapters' / f'{cond}.pt', map_location='cpu', weights_only=False)
    if state.get('condition') != cond:
        raise ValueError(f'adapter of {state.get("condition")} is not {cond}')
    lora = tf.LoRA(tf.target_modules(rt.layers, cond), state['rank'], state['alpha'])
    lora.load(state)
    lora.attach()
    kind = cl.condition_positions(cond)
    if kind is not None and with_hook:
        cl.attach_positions(rt.model, lora, rt.model.config.image_token_id, kind)
    return lora


def stage_train(rt: qr.Runtime, out: Path, args) -> None:
    """The 3.13 trainer (targeted_finetune_run.stage_train) with the 7.2 position masks."""
    cond = args.condition
    apath = out / 'adapters' / f'{cond}.pt'
    if apath.exists():
        print(f'{apath} exists; skipping training', flush=True)
        return
    spec = tf.CONDITIONS[cond]
    samples = tf.read_samples(out / spec.get('train_file', 'train.jsonl'))
    epochs = spec.get('epochs', args.epochs)
    base_lr = spec.get('lr', args.lr)
    torch.manual_seed(spec.get('seed', tfr.SEED))
    lora = make_lora(rt, cond, args.rank, args.alpha)
    lora.attach()
    model = rt.model
    kind = spec.get('positions')
    if kind is not None:
        cl.attach_positions(model, lora, model.config.image_token_id, kind)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.config.use_cache = False
    opt = torch.optim.AdamW(lora.parameters(), lr=base_lr, weight_decay=0.0, betas=(0.9, 0.999))
    rng = np.random.default_rng(spec.get('seed', tfr.SEED))
    order = np.concatenate([rng.permutation(len(samples)) for _ in range(epochs)])
    n_steps = len(order) // args.step_samples
    warm = max(1, int(0.05 * n_steps))

    def lr_at(s):
        if s < warm:
            return base_lr * (s + 1) / warm
        return base_lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_steps - warm)))
    prompt_fn = lambda text: qr.prompt_text(rt, 1, text)
    log = out / 'adapters' / f'{cond}_log.jsonl'
    if log.exists():
        log.unlink()
    t0 = time.time()
    for step in range(n_steps):
        items = [samples[int(i)] for i in order[step * args.step_samples:(step + 1) * args.step_samples]]
        sizes = [Image.open(s.image).size for s in items]
        tokens = [(w // 32) * (h // 32) + 150 for w, h in sizes]
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
        tot, dig, n_dig = 0.0, 0.0, 0
        for mb in micro:
            ss = [items[i] for i in mb]
            batch = tf.encode_batch(rt.processor, prompt_fn, ss, [Image.open(s.image).convert('RGB') for s in ss])
            per, dce, _ = tf.batch_loss(model, batch, rt.device)
            (per.sum() / len(items)).backward()
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
    state.update({'condition': cond, 'lr': base_lr, 'epochs': epochs, 'step_samples': args.step_samples,
                  'n_steps': n_steps, 'n_params': lora.n_params(), 'delta_norms': lora.delta_norms(),
                  'train_seconds': time.time() - t0, 'n_train': len(samples), 'positions': kind,
                  'gpu': torch.cuda.get_device_name(0), 'runtime': qr.describe(rt)})
    torch.save(state, apath)
    model.gradient_checkpointing_disable()
    model.eval()
    lora.detach()
    print(f'saved {apath} ({lora.n_params()} params, {time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- evaluation

def with_adapter(fn):
    def run(rt, out, args):
        lora = attach_trained(rt, out, args.condition)
        try:
            fn(rt, out, args)
        finally:
            if lora is not None:
                lora.detach()
    return run


def _coord(rt, out, args):
    tfr.stage_coord(rt, out, args)


def _general(rt, out, args):
    tfr.stage_general(rt, out, args)


def stage_digits(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    d = tfr.cond_dir(out, cond)
    # --eval-positions: evaluate the adapter with its update restricted to another token set (analysis splice):
    # 'question' = text tokens of the prompt (before the answer prefix), 'prefix' = answer-prefix tokens.
    path = d / ('digits.jsonl' if not args.eval_positions else f'digits_{args.eval_positions}.jsonl')
    if path.exists():
        print(f'{path} exists', flush=True)
        return
    lora = attach_trained(rt, out, cond, with_hook=False)
    kind = cl.condition_positions(cond)
    ua = Namespace(round1=args.round1, output=args.finetune_run)
    rows = tr2.build_rows(rt, tr2.disc_units(ua, 2, ('sweep_h', 'sweep_v', 'refcoco')))
    res = []
    try:
        for chunk in tr2.batches(rows, args.batch):
            if lora is not None:
                Ltot = max(r['length'] for r in chunk)
                if args.eval_positions:
                    lora.pos_mask = torch.tensor(np.stack([t2.position_mask(args.eval_positions.replace('prefix', 'answer'),
                                                                            r['length'], r['span'], r['prompt_len'], Ltot)
                                                           for r in chunk]))
                else:
                    lora.pos_mask = None if kind is None else torch.tensor(np.stack([cl.span_mask(kind, r['span'], Ltot)
                                                                                      for r in chunk]))
            o = tr2.light_forward(rt, chunk)
            for r, lp in zip(chunk, o['logp']):
                u = r['unit']
                res.append({'unit': u.id, 'family': u.family, 'truth': u.truth, 'k': u.ks[0],
                            'logp0': lp[0].tolist(), 'logp1': lp[1].tolist()})
    finally:
        if lora is not None:
            lora.detach()
    append_jsonl(path, res)
    for fam in ('sweep_h', 'sweep_v', 'refcoco'):
        rr = [r for r in res if r['family'] == fam]
        m = t2.digit_metrics(np.array([r['truth'] for r in rr]), np.array([r['logp0'] for r in rr]),
                             np.array([r['logp1'] for r in rr]))
        print(f"digits {cond} {fam}: gain {m['gain']:.2f} p1 {m['p_second']:.2f} p0 {m['p_first']:.2f}", flush=True)


def stage_capture(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    dpath = Path(args.features) / 'capture'
    dpath.mkdir(parents=True, exist_ok=True)
    path = dpath / f'{cond}.npz'
    if path.exists():
        print(f'{path} exists', flush=True)
        return
    lora = attach_trained(rt, out, cond, with_hook=False)
    kind = cl.condition_positions(cond)
    ua = Namespace(round1=args.round1, output=args.finetune_run)
    rows = tr2.build_rows(rt, tr2.disc_units(ua, 1, ('sweep_h', 'sweep_v')) + tr2.disc_units(ua, 2, ('textconv', 'refcoco')))
    resid, logp = [], []
    try:
        for chunk in tr2.batches(rows, args.batch):
            if lora is not None:
                Ltot = max(r['length'] for r in chunk)
                lora.pos_mask = None if kind is None else torch.tensor(np.stack([cl.span_mask(kind, r['span'], Ltot)
                                                                                  for r in chunk]))
            o = tr2.light_forward(rt, chunk, CAPTURE_LAYERS)
            resid.append(np.stack([o['resid'][l].numpy()[:, [0, 2]] for l in CAPTURE_LAYERS], 2).astype(np.float16))
            logp.append(o['logp'])
    finally:
        if lora is not None:
            lora.detach()
    np.savez(path, resid=np.concatenate(resid), logp=np.concatenate(logp), sites=np.array(['q0', 'target']),
             truth=np.array([r['unit'].truth for r in rows]), family=np.array([r['unit'].family for r in rows]),
             unit=np.array([r['unit'].id for r in rows]), layers=np.array(CAPTURE_LAYERS))
    print(f'capture {cond}: {len(rows)} rows', flush=True)


def _babyvision(rt, out, args):
    from eva.benchmark import load_cases, score as bv_score
    d = tfr.cond_dir(out, args.condition)
    path = d / 'babyvision.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    cases = [json.loads(l) for l in (out / 'babyvision' / 'cases.jsonl').read_text().splitlines()]
    cases = [c for c in cases if c['id'] not in done]
    rows = {r['taskId']: r for r in load_cases(Path(args.data_dir), 'babyvision', 'train')[0]}
    t0 = time.time()
    for task in ('red_point', 'red_rowcol', 'odd_point', 'odd_rowcol', 'official'):
        group = [c for c in cases if c['task'] == task]
        max_new = args.bv_max_new if task == 'official' else args.max_new_tokens
        for i in range(0, len(group), args.bv_batch):
            chunk = group[i:i + args.bv_batch]
            outs = qr.generate_batch(rt, [([Image.open(c['image']).convert('RGB')], c['prompt']) for c in chunk], max_new)
            recs = []
            for c, o in zip(chunk, outs):
                rec = {'id': c['id'], 'task': task, 'taskId': c['taskId'], 'text': o['text'],
                       'finish_reason': o['finish_reason'], 'batch_size': len(chunk)}
                if task.endswith('point'):
                    rec.update(cl.point_hit(tf.parse_answer(o['text'], 'point'), c['truth']['box'], tuple(c['canvas'])))
                elif task == 'official':
                    s = bv_score(rows[c['taskId']], o['text'], 'babyvision')
                    rc = cl.lenient_rowcol(o['text'])
                    rec.update({'strict': bool(s['correct']), 'rc': rc, 'correct': rc is not None and list(rc) == c['truth']['rc']})
                else:
                    rc = cr.parse_rowcol(o['text'])
                    rec.update({'rc': rc, 'correct': rc is not None and list(rc) == c['truth']['rc'],
                                'row_ok': rc is not None and rc[0] == c['truth']['rc'][0],
                                'col_ok': rc is not None and rc[1] == c['truth']['rc'][1]})
                recs.append(rec)
            append_jsonl(path, recs)
        print(f'babyvision {args.condition} {task} ({time.time() - t0:.0f}s)', flush=True)


def lenient_rowcol_json(text: str) -> tuple[int, int] | None:
    """Row / column from a JSON-like answer, tolerant to code fences, list wrapping and a wrong closing bracket
    (the fine-tuned answer template drift '[{"row": 5, "column": 8]')."""
    import re
    mr, mc = re.search(r'"row"\s*:\s*(\d+)', text), re.search(r'"column"\s*:\s*(\d+)', text)
    return (int(mr.group(1)), int(mc.group(1))) if mr and mc else None


def _ordinal(rt, out, args):
    """Direct row / column answers on the 72 grid test images of 3.13 (6x6, 10x10, 20x20; the red figure is the
    target): strict parse (base JSON format) and lenient parse (format drift tolerated), row / column correctness and
    the signed row / column error."""
    d = tfr.cond_dir(out, args.condition)
    path = d / 'ordinal.jsonl'
    if path.exists():
        print(f'{path} exists', flush=True)
        return
    samples = [smp for smp in tf.read_samples(out / 'test.jsonl') if smp.family in ('grid06', 'grid10', 'grid20')]
    recs = []
    for i in range(0, len(samples), 8):
        chunk = samples[i:i + 8]
        outs = qr.generate_batch(rt, [([Image.open(smp.image).convert('RGB')], ROWCOL_PROMPT) for smp in chunk],
                                 args.max_new_tokens)
        for smp, o in zip(chunk, outs):
            strict, len_ = cr.parse_rowcol(o['text']), lenient_rowcol_json(o['text'])
            tr_, tc = smp.meta['row'] + 1, smp.meta['col'] + 1
            recs.append({'id': smp.id, 'family': smp.family, 'text': o['text'], 'finish_reason': o['finish_reason'],
                         'batch_size': len(chunk), 'strict': strict, 'lenient': len_, 'truth': [tr_, tc],
                         'strict_correct': strict is not None and list(strict) == [tr_, tc],
                         'correct': len_ is not None and list(len_) == [tr_, tc],
                         'row_err': None if len_ is None else len_[0] - tr_, 'col_err': None if len_ is None else len_[1] - tc})
    append_jsonl(path, recs)
    print(f"ordinal {args.condition}: strict {sum(r['strict_correct'] for r in recs)}/{len(recs)} lenient "
          f"{sum(r['correct'] for r in recs)}/{len(recs)}", flush=True)


STAGES = {'train': stage_train, 'coord': with_adapter(_coord), 'general': with_adapter(_general), 'digits': stage_digits,
          'capture': stage_capture, 'babyvision': with_adapter(_babyvision),
          'ordinal': with_adapter(_ordinal)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--condition', default='base')
    p.add_argument('--finetune-run', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--data-dir', default='')
    p.add_argument('--bv-annotation', default='')
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
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--bv-batch', type=int, default=8)
    p.add_argument('--bv-max-new', type=int, default=1024)
    p.add_argument('--eval-positions', default='', choices=['', 'question', 'prefix'])
    args = p.parse_args()
    cl.register()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in args.stages.split(',') if s]
    if stages == ['link']:
        stage_link(out, args)
        return
    if stages == ['bvprep']:
        stage_bvprep(out, args)
        return
    if stages == ['fixprep']:
        stage_fixprep(out, args)
        return
    conds = args.condition.split(',')
    for c in conds:
        if c != 'base' and c not in tf.CONDITIONS:
            raise ValueError(f'Unknown condition {c}')
    if len(conds) > 1 and 'train' in stages:
        raise ValueError('train takes one condition')
    rt = qr.load(args.model)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['model'] = qr.describe(rt)
    info['runs'].append({'stages': stages, 'condition': args.condition, 'started': datetime.now(timezone.utc).isoformat(),
                         'gpu': torch.cuda.get_device_name(0), 'batch': args.batch, 'bv_batch': args.bv_batch})
    meta.write_text(json.dumps(info, indent=1))
    for cond in conds:
        args.condition = cond
        for s in stages:
            t0 = time.time()
            STAGES[s](rt, out, args)
            # Several queues share the run directory: per-process timing log (experiment.json is only appended at start).
            append_jsonl(out / 'stage_seconds.jsonl', [{'condition': cond, 'stage': s, 'seconds': time.time() - t0,
                                                        'gpu': torch.cuda.get_device_name(0),
                                                        'finished': datetime.now(timezone.utc).isoformat()}])


if __name__ == '__main__':
    main()
