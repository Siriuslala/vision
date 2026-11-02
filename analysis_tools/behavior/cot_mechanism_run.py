"""Experiment 1.7 GPU stages (resumable; every stage appends JSONL records and skips finished ids).

Stages
------
gen      fresh generations on the four task sets in direct / think / answer_first modes (thinking off),
         plus a small thinking-on subset; spot-check of batched decoding against qwen_runtime.generate.
faith    faithfulness tests on the think generations: early answering after truncating the CoT at
         token fractions, and editing a stated row / count (late claim and first claim) then continuing.
mech     matched pairs (A, B): decode A under decode-time interventions (image mask, image-KV swap,
         text-KV swap, DeltaNet-state swap, their combinations, and the complement B-prefill with A's
         image KV), with attention probes on clean runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.behavior import cot_mechanism as cm
from analysis_tools.behavior import cot_mechanism_engine as ce
from analysis_tools.common import qwen_runtime as qr


def append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def code_digest() -> dict:
    here = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob('cot_mechanism*.py'))}


MAX_NEW = {'direct': 64, 'think': 1024, 'answer_first': 768, 'thinking_on': 4096}


# ----------------------------------------------------------------------------- gen

def stage_gen(rt, out: Path, n: int, batch_size: int, tasks: list[str], n_thinking: int) -> None:
    path = out / 'gen/records.jsonl'
    done = {r['id'] for r in read(path)}
    for task in tasks:
        items = cm.build_items(task, n)
        rendered = {it.id: it.render() for it in items}
        modes = ['direct', 'think', 'answer_first']
        jobs = [(it, m) for m in modes for it in items]
        if task in ('rowcol', 'point') and n_thinking:
            jobs += [(it, 'thinking_on') for it in items[:n_thinking]]
        for mode in modes + ['thinking_on']:
            sel = [(it, m) for it, m in jobs if m == mode and f'{it.id}|{m}' not in done]
            bs = batch_size if mode != 'thinking_on' else max(1, min(8, batch_size))
            for i in range(0, len(sel), bs):
                chunk = sel[i:i + bs]
                t0 = time.time()
                p = cm.prompt(task, 'think' if mode == 'thinking_on' else mode)
                res = ce.continue_batch(rt, [(rendered[it.id][0], p, '') for it, _ in chunk], MAX_NEW[mode],
                                        thinking=mode == 'thinking_on')
                dt = time.time() - t0
                append(path, [{'id': f'{it.id}|{m}', 'item': it.id, 'task': task, 'mode': m, 'template': it.template,
                               'truth': cm.truth_of(it, rendered[it.id][1]), 'prompt': p, 'batch_size': len(chunk),
                               'seconds': dt / len(chunk), **r} for (it, m), r in zip(chunk, res)])
                print(f'gen {task} {mode} {i + len(chunk)}/{len(sel)} {dt:.1f}s', flush=True)


def stage_spotcheck(rt, out: Path, n: int = 20) -> None:
    """Answer-level agreement of batched decoding with qwen_runtime.generate at batch size 1."""
    path = out / 'gen/spotcheck.json'
    if path.exists():
        return
    recs = [r for r in read(out / 'gen/records.jsonl') if r['mode'] in ('direct', 'think')]
    rng = np.random.default_rng(0)
    pick = [recs[i] for i in rng.choice(len(recs), size=min(n, len(recs)), replace=False)]
    rows = []
    for r in pick:
        it = next(x for x in cm.build_items(r['task'], 512) if x.id == r['item'])
        img = it.render()[0]
        ref = qr.generate(rt, qr.build(rt, [img], r['prompt']), MAX_NEW[r['mode']])
        a, _ = cm.final_answer(r['task'], r['text'])
        b, _ = cm.final_answer(r['task'], ref['text'])
        rows.append({'id': r['id'], 'answer_batched': a, 'answer_bs1': b, 'answer_equal': a == b,
                     'text_equal': ref['text'] == r['text']})
    path.write_text(json.dumps({'n': len(rows), 'answer_agree': sum(x['answer_equal'] for x in rows),
                                'text_agree': sum(x['text_equal'] for x in rows), 'rows': rows}, indent=1))


# ----------------------------------------------------------------------------- faith

FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
N_EDIT_ITEMS = 32      # think responses per task that receive the edit tests (GPU budget)
EDIT_MAX_NEW = 512     # continuation budget after an edit; think responses end within 373 tokens


def _body_tokens(rt, task: str, text: str) -> list[int]:
    _, pos = cm.final_answer(task, text)
    body = text[:pos].rstrip()
    # drop a trailing "Final answer:" style lead-in so the forced cue is not duplicated
    for lead in ('Final Answer:', 'Final answer:', '**Final Answer:**', '**Final answer:**', 'Answer:'):
        if body.endswith(lead):
            body = body[:-len(lead)].rstrip()
    return rt.tokenizer.encode(body, add_special_tokens=False)


def stage_faith(rt, out: Path, batch_size: int, tasks: list[str]) -> None:
    gen = [r for r in read(out / 'gen/records.jsonl') if r['mode'] == 'think' and r['task'] in tasks
           and r['finish_reason'] == 'eos']
    items = {}
    for task in tasks:
        for it in cm.build_items(task, 512):
            items[it.id] = it
    path = out / f"faith/records_{'_'.join(tasks)}.jsonl"     # one file per process/task set
    done = {r['id'] for f in (out / 'faith').glob('records*.jsonl') for r in read(f)}
    jobs = []
    n_edit = Counter()
    for r in gen:
        task = r['task']
        toks = _body_tokens(rt, task, r['text'])
        for f in FRACTIONS:
            k = int(round(f * len(toks)))
            prefix = rt.tokenizer.decode(toks[:k]) + cm.FORCE_CUE[task]
            jobs.append({'id': f"{r['id']}|trunc{f:.1f}", 'src': r['id'], 'test': 'truncate', 'fraction': f,
                         'task': task, 'item': r['item'], 'prefix': prefix, 'max_new': 16})
        # edits of a stated row (or declared row total for count): last claim and first claim
        axis = 'row'
        span = cm.locate_claim(task, r['text'], axis)
        first = cm.locate_first_claim(task, r['text'], axis)
        rows_n = r['truth']['rows']
        if n_edit[task] >= N_EDIT_ITEMS:
            continue
        n_edit[task] += 1
        for which, sp in (('last', span), ('first', first)):
            if sp is None:
                continue
            delta = 1 if sp[2] + 1 <= (rows_n + 2 if task == 'count' else rows_n) else -1
            if sp[2] + delta < 1:
                continue
            prefix, new = cm.edit_claim(r['text'], sp, delta)
            jobs.append({'id': f"{r['id']}|edit_{which}", 'src': r['id'], 'test': f'edit_{which}', 'task': task,
                         'item': r['item'], 'prefix': prefix, 'orig_value': sp[2], 'new_value': new, 'max_new': EDIT_MAX_NEW})
            # control: the same truncation point without an edit
            jobs.append({'id': f"{r['id']}|keep_{which}", 'src': r['id'], 'test': f'keep_{which}', 'task': task,
                         'item': r['item'], 'prefix': r['text'][:sp[1]], 'orig_value': sp[2], 'new_value': sp[2],
                         'max_new': EDIT_MAX_NEW})
    jobs = [j for j in jobs if j['id'] not in done]
    rendered = {}
    for group in ('truncate', 'edit'):
        sel = [j for j in jobs if (j['test'] == 'truncate') == (group == 'truncate')]
        sel.sort(key=lambda j: (j['task'], len(j['prefix'])))
        bs = batch_size * (2 if group == 'truncate' else 1)
        for i in range(0, len(sel), bs):
            chunk = sel[i:i + bs]
            for j in chunk:
                if j['item'] not in rendered:
                    rendered[j['item']] = items[j['item']].render()[0]
            t0 = time.time()
            res = ce.continue_batch(rt, [(rendered[j['item']], cm.prompt(j['task'], 'think'), j['prefix']) for j in chunk],
                                    max(j['max_new'] for j in chunk))
            append(path, [{**{k: v for k, v in j.items() if k != 'prefix'}, 'prefix_chars': len(j['prefix']),
                           'prefix_tail': j['prefix'][-120:], 'batch_size': len(chunk), **x} for j, x in zip(chunk, res)])
            print(f'faith {group} {i + len(chunk)}/{len(sel)} {time.time() - t0:.1f}s', flush=True)


# ----------------------------------------------------------------------------- mech

CONDITIONS = (
    # name, base ('A' or 'B'), donor, channels, mask_image, record attention
    ('clean_A', 'A', None, (), False, True),
    ('clean_B', 'B', None, (), False, False),
    ('mask_img', 'A', None, (), True, False),
    ('img_from_B', 'A', 'B', ('img',), False, True),
    ('txt_from_B', 'A', 'B', ('txt',), False, False),
    ('delta_from_B', 'A', 'B', ('delta',), False, False),
    ('txt_delta_from_B', 'A', 'B', ('txt', 'delta'), False, False),
    ('B_prefill_img_from_A', 'B', 'A', ('img',), False, False),
    ('all_from_B', 'A', 'B', ('img', 'txt', 'delta'), False, False),
)


def _save_probe(path: Path, probe: ce.Probe, res: list[dict]) -> None:
    """Stack per-step probe tensors into arrays (steps, B, layers, heads) and maps (steps, B, layers, n_img)."""
    steps = probe.steps
    L = list(ce.FULL_LAYERS)
    arr = {k: np.stack([np.stack([s[l][k].numpy() for l in L], 1) for s in steps]).astype(np.float16)
           for k in ('img', 'pre', 'txt', 'gen')}
    arr['map'] = np.stack([np.stack([s[l]['map'].numpy() for l in L], 1) for s in steps])
    arr['map_h11'] = np.stack([s[20]['map_h11'].numpy() for s in steps])
    arr['n_tokens'] = np.array([r['n_tokens'] for r in res])
    np.savez_compressed(path, **arr)


FLOOR_ACC = 0.25   # a task whose clean_A batch-0 accuracy is below this cannot show which image the answer follows


def _floor(task: str, recs: list[dict]) -> bool:
    if task == 'point':      # point answers are scored by the nearer target, so no floor applies
        return False
    rows = [r for r in recs if r['task'] == task and r['condition'] == 'clean_A' and r['batch_id'].endswith('|b0|clean_A')]
    if not rows:
        return False
    ok = 0
    for r in rows:
        a = cm.final_answer(task, r['text'])[0]
        t = r['truth_a']
        if task == 'point':
            x0, y0, x1, y1 = t['cell_box_norm']
            ok += a is not None and x0 <= a[0] <= x1 and y0 <= a[1] <= y1
        else:
            ok += a is not None and a[0] == t['row'] and a[1] == t['col']
    return ok / len(rows) < FLOOR_ACC


def stage_mech(rt, out: Path, n_pairs: int, batch_size: int, tasks: list[str], max_new: int) -> None:
    path = out / 'mech/records.jsonl'
    prior = read(path)
    done = {r['batch_id'] for r in prior}
    (out / 'mech/probes').mkdir(parents=True, exist_ok=True)
    for task in tasks:
        a_items = cm.build_items(task, n_pairs, seed=29)
        pairs = [(a, cm.pair_item(a)) for a in a_items]
        for i in range(0, len(pairs), batch_size):
            if i > 0 and _floor(task, read(path)):
                print(f'mech {task}: clean_A accuracy of batch 0 below {FLOOR_ACC}; remaining batches skipped', flush=True)
                break
            chunk = pairs[i:i + batch_size]
            todo = [c for c in CONDITIONS if f'{task}|b{i}|{c[0]}' not in done]
            if not todo:
                continue
            ra = [a.render() for a, _ in chunk]
            rb = [b.render() for _, b in chunk]
            p = cm.prompt(task, 'think')
            batch_a = ce.build_batch(rt, [x[0] for x in ra], [p] * len(chunk))
            batch_b = ce.build_batch(rt, [x[0] for x in rb], [p] * len(chunk))
            sp = ce.spans(rt, batch_a)
            if ce.spans(rt, batch_b) != sp:
                raise ValueError('pair members have different token layouts')
            cache_a, delta_a = ce.prefill(rt, batch_a)
            cache_b, delta_b = ce.prefill(rt, batch_b)
            if not torch.equal(delta_a, delta_b):
                raise ValueError('rope deltas differ between pair members')
            caches = {'A': (batch_a, cache_a), 'B': (batch_b, cache_b)}
            for name, base, donor, channels, mask, rec in todo:
                t0 = time.time()
                bt, cache = caches[base]
                res, probe = ce.run_condition(rt, bt, cache, delta_a, sp, max_new,
                                              donor=caches[donor][1] if donor else None, channels=set(channels),
                                              mask_image=mask, record=rec)
                if rec:
                    _save_probe(out / f'mech/probes/{task}_b{i}_{name}.npz', probe, res)
                rows = []
                for k, ((a, b), r) in enumerate(zip(chunk, res)):
                    rows.append({'id': f'{task}|b{i}|{name}|{a.id}', 'batch_id': f'{task}|b{i}|{name}', 'task': task,
                                 'condition': name, 'item_a': a.id, 'item_b': b.id, 'row_in_batch': k,
                                 'truth_a': cm.truth_of(a, ra[k][1]), 'truth_b': cm.truth_of(b, rb[k][1]),
                                 'text': r['text'], 'ids': r['ids'], 'finish_reason': r['finish_reason'],
                                 'n_tokens': r['n_tokens'], 'batch_size': len(chunk), 'sp': sp})
                append(path, rows)
                append(path.parent / 'done.jsonl', [{'id': f'{task}|b{i}|{name}'}])
                print(f'mech {task} b{i} {name} {time.time() - t0:.1f}s', flush=True)
            del cache_a, cache_b, caches
            torch.cuda.empty_cache()


EDIT_CONDITIONS = (
    # name, prefill used as base, donor of the image KV
    ('edited_all', 'edited', None),
    ('clean_prefill_img_from_edited', 'clean', 'edited'),
    ('edited_prefill_img_from_clean', 'edited', 'clean'),
)


def stage_mech_edit(rt, out: Path, n_pairs: int, batch_size: int, max_new: int) -> None:
    """Link to experiment 3.10: prefill with DeltaNet alpha=1 at the image tokens of layers 9-16 (the
    row-count recurrence), then decode with image KV taken from the edited or the clean prefill."""
    from analysis_tools.position import rowcol_mechanism as rm
    from analysis_tools.position import rowcol_mechanism_run as rmr
    path = out / 'mech_edit/records.jsonl'
    done = {r['batch_id'] for r in read(path)}
    task = 'rowcol'
    items = cm.build_items(task, n_pairs, seed=29)
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        todo = [c for c in EDIT_CONDITIONS if f'{task}|b{i}|{c[0]}' not in done]
        if not todo:
            continue
        rendered = [a.render() for a in chunk]
        batch = ce.build_batch(rt, [x[0] for x in rendered], [cm.prompt(task, 'think')] * len(chunk))
        sp = ce.spans(rt, batch)
        clean, delta = ce.prefill(rt, batch)
        pos = np.arange(sp['img_start'], sp['img_end'])
        ov = {'nodecay': {l: (pos, None) for l in rm.MID_LINEAR}}
        with rmr.override_gates_heads(rt, [ov] * len(chunk), sp['P'] - 1):
            edited, _ = ce.prefill(rt, batch)
        caches = {'clean': clean, 'edited': edited}
        for name, base, donor in todo:
            t0 = time.time()
            res, _ = ce.run_condition(rt, batch, caches[base], delta, sp, max_new,
                                      donor=caches[donor] if donor else None, channels={'img'} if donor else set())
            append(path, [{'id': f'{task}|b{i}|{name}|{a.id}', 'batch_id': f'{task}|b{i}|{name}', 'condition': name,
                           'item_a': a.id, 'truth_a': cm.truth_of(a, rendered[k][1]), 'text': r['text'],
                           'finish_reason': r['finish_reason'], 'n_tokens': r['n_tokens'], 'batch_size': len(chunk),
                           'edit_layers': list(rm.MID_LINEAR)}
                          for k, (a, r) in enumerate(zip(chunk, res))])
            print(f'mech_edit b{i} {name} {time.time() - t0:.1f}s', flush=True)
        del clean, edited, caches
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stage', required=True, choices=('gen', 'spotcheck', 'faith', 'mech', 'mech_edit'))
    ap.add_argument('--model', default='Qwen3.5-9B')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-memory', default='', help='split load, e.g. "0:12GiB,1:12GiB" (device_map=auto)')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--n-pairs', type=int, default=32)
    ap.add_argument('--n-thinking', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--tasks', default='rowcol,odd,point,count')
    ap.add_argument('--mech-max-new', type=int, default=768)
    args = ap.parse_args()
    if args.max_memory:
        mm = {int(k): v for k, v in (x.split(':') for x in args.max_memory.split(','))}
        rt = ce.load_split(args.model, mm)
    else:
        rt = qr.load(args.model, device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = args.out / f'run_{args.stage}.json'
    meta.write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()}, 'runtime': qr.describe(rt),
                                'code': code_digest(), 'started': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))
    tasks = args.tasks.split(',')
    if args.stage == 'gen':
        stage_gen(rt, args.out, args.n, args.batch_size, tasks, args.n_thinking)
        stage_spotcheck(rt, args.out)
    elif args.stage == 'spotcheck':
        stage_spotcheck(rt, args.out)
    elif args.stage == 'faith':
        stage_faith(rt, args.out, args.batch_size, tasks)
    elif args.stage == 'mech':
        stage_mech(rt, args.out, args.n_pairs, args.batch_size, tasks, args.mech_max_new)
    elif args.stage == 'mech_edit':
        stage_mech_edit(rt, args.out, args.n_pairs, args.batch_size, args.mech_max_new)


if __name__ == '__main__':
    main()
