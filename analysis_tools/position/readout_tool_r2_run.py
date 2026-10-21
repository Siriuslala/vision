"""Experiment 3.9 round 2 stage drivers.

Stages (comma list via --stages):
  prepare   round-2 case list: round-1 cases with question phrasings (training: random among 6;
            evaluation: canonical) and alternative RefCOCO expressions for training; the graded
            synthetic family (train / val / test); a held-out-phrasing test subset (CPU only).
  pca       PCA bases (PCA_DIM) of visual-token states at CONTENT_LAYERS from 256 training images.
  capture   prompt + call prefix (with the approx_point argument) inside greedy generation: slot
            hidden states, attention maps at target_end and slot, PCA content features, first-digit
            log-probabilities and the model's rough point.  Supports --shard k/n for several GPUs.
  own       the model's own point and bbox answers on the graded test sets (round-1 prompt).
  spotcheck answer-level agreement of batched with single-sample generation (20 own answers).
  freecall  model-generated calls on a test subset: call rate, argument agreement with the canonical
            call (identical arguments give the identical slot state).
  dissoc    coordinate / ordinal dissociation: pointer-head attention steering and counting-head gate
            ablation, each measured on point answers and on row/column answers (see design doc).
Fitting, evaluation and figures: readout_tool_r2_analysis.py / readout_tool_r2_figures.py.
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
from analysis_tools.position import readout_tool_r2 as r2
from analysis_tools.position import readout_tool_run as r1run
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

SEED = 20261020
N_GRADED = {'train': 600, 'val': 100, 'test': 240}
N_HELDOUT_PHRASING = {'refcoco': 300, 'synth': 100, 'disc': 100, 't6': 96, 'layout': 100, 'ring': 100, 'graded': 50}
CHUNK = 384


def code_digest() -> dict:
    here = Path(__file__).parent
    return {n: hashlib.sha256((here / n).read_bytes()).hexdigest()[:16]
            for n in ('readout_tool.py', 'readout_tool_r2.py', 'readout_tool_r2_run.py')}


def write_meta(out: Path, name: str, payload: dict) -> None:
    payload = dict(payload, created_utc=datetime.now(timezone.utc).isoformat(), code=code_digest())
    (out / f'{name}.json').write_text(json.dumps(payload, indent=1, default=str) + '\n')


# ----------------------------------------------------------------------------- prepare

def _render_graded(job):
    level, split, k, seed, root, backgrounds = job
    image, truth, target, meta = r2.graded_scene(level, seed, [Path(b) for b in backgrounds])
    rel = f'graded/{level}/{split}/{k:05d}.png'
    image.save(Path(root) / rel)
    return {'id': f'g_{level}_{split}_{k:05d}', 'set': f'g_{level}', 'split': split, 'subset': meta.get('kind', level),
            'image': rel, 'image_root': 'r2', 'target': target, 'truth': truth, 'seed': int(seed), 'meta': meta}


def refcoco_expressions(refcoco: Path) -> dict[str, list[str]]:
    """question_id -> all referring expressions of the record."""
    import pyarrow.parquet as pq
    out = {}
    for shard in sorted((refcoco / 'data').glob('*.parquet')):
        t = pq.read_table(shard, columns=['question_id', 'answer'])
        for q, a in zip(t.column('question_id').to_pylist(), t.column('answer').to_pylist()):
            if a:
                out.setdefault(str(q), list(a))
    return out


def stage_prepare(args) -> None:
    path = args.output / 'cases.jsonl'
    if path.exists():
        print('cases.jsonl exists; prepare is skipped (never overwrites a run).', flush=True)
        return
    rng = np.random.default_rng(SEED)
    r1 = read_jsonl(args.r1 / 'cases.jsonl')
    exprs = refcoco_expressions(args.refcoco)
    cases = []
    for c in r1:
        c = dict(c, image_root='r1' if c.get('image_root') == 'run' else c.get('image_root'))
        if c['split'] == 'train':
            c['phrasing'] = int(rng.integers(len(r2.PHRASINGS)))
            if c['set'] == 'refcoco':
                alts = exprs.get(c['id'].split('_')[1], [c['target']])
                c['target_r1'] = c['target']
                c['target'] = alts[int(rng.integers(len(alts)))]
        else:
            c['phrasing'] = 0
        cases.append(c)
    bg = {s: sorted({c['image'] for c in r1 if c['set'] == 'refcoco' and c['split'] == s}) for s in ('train', 'test')}
    jobs = []
    for li, level in enumerate(r2.LEVELS):
        for si, (split, n) in enumerate(N_GRADED.items()):
            if level == 'flat' and split != 'test':
                continue                     # flat training scenes are the round-1 synth set
            pool = [str(args.r1_images / b) for b in bg['test' if split == 'test' else 'train']]
            for k in range(n):
                jobs.append((level, split, k, SEED * 100 + li * 10_000_000 + si * 1_000_000 + k, str(args.images), pool))
    for level in r2.LEVELS:
        for split in N_GRADED:
            (args.images / 'graded' / level / split).mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        graded = pool.map(_render_graded, jobs, chunksize=8)
    for g in graded:
        g['phrasing'] = int(rng.integers(len(r2.PHRASINGS))) if g['split'] == 'train' else 0
    cases += graded
    held = []
    for s, n in N_HELDOUT_PHRASING.items():
        sets = [f'g_{l}' for l in r2.LEVELS] if s == 'graded' else [s]
        for st in sets:
            pool_ = [c for c in cases if c['set'] == st and c['split'] == 'test']
            for i in sorted(rng.choice(len(pool_), min(n, len(pool_)), replace=False)):
                held.append(dict(pool_[i], id=pool_[i]['id'] + '_hp', phrasing='heldout', base_id=pool_[i]['id']))
    cases += held
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate case ids')
    path.write_text(''.join(json.dumps(c, sort_keys=True) + '\n' for c in cases))
    counts = Counter(f"{c['set']}/{c['split']}" + ('/hp' if c['phrasing'] == 'heldout' else '') for c in cases)
    write_meta(args.output, 'prepare', {'counts': counts, 'seed': SEED, 'n_graded': N_GRADED, 'heldout': N_HELDOUT_PHRASING})
    print(json.dumps(counts, indent=0), flush=True)


# ----------------------------------------------------------------------------- shared helpers

def load_image(case: dict, args) -> Image.Image:
    if case.get('render') == 'disc':
        from analysis_tools.position import coordinate_readout as cr
        return cr.disc_image(tuple(case['center_px']), case['diameter'], case['canvas_px'])[0]
    base = {'r1': args.r1_images, 'r2': args.images, 'behavior': args.behavior_root,
            'ordinal': args.behavior_root.parent / ORDINAL_RUN}[case['image_root']]
    p = base / case['image']
    if 'image_sha256' in case and hashlib.sha256(p.read_bytes()).hexdigest() != case['image_sha256']:
        raise ValueError(f'Image hash mismatch for {p}')
    return Image.open(p).convert('RGB')


def user_text(case: dict) -> str:
    tpl = r2.HELDOUT_PHRASING if case['phrasing'] == 'heldout' else r2.PHRASINGS[case['phrasing']]
    return tpl.format(target=case['target'])


def load_cases(args) -> list[dict]:
    return read_jsonl(args.output / 'cases.jsonl')


# ----------------------------------------------------------------------------- pca

def stage_pca(rt, args) -> None:
    import torch
    out = args.output / 'pca.pt'
    if out.exists():
        print('pca.pt exists', flush=True)
        return
    cases = [c for c in load_cases(args) if c['split'] == 'train']
    rng = np.random.default_rng(SEED + 5)
    pick = [cases[i] for i in rng.choice(len(cases), 256, replace=False)]
    states = {l: [] for l in r2.CONTENT_LAYERS}
    for b in r1run.batches(pick, args.max_tokens, args.max_batch):
        texts = [r2.tool_prompt_r2(rt, user_text(c)) + r2.call_prefix(c['target']) for c in b]
        got = r2.visual_states(rt, texts, [load_image(c, args) for c in b])
        for l in r2.CONTENT_LAYERS:
            states[l] += [s[::3] for s in got[l]]
    pca = {l: r2.fit_pca(states[l]) for l in r2.CONTENT_LAYERS}
    torch.save(pca, out)
    write_meta(args.output, 'pca', {'n_images': len(pick), 'explained': {l: pca[l]['explained'] for l in pca},
                                    'n_tokens': {l: int(sum(len(s) for s in states[l])) for l in states}})
    print({l: pca[l]['explained'] for l in pca}, flush=True)


# ----------------------------------------------------------------------------- capture

def stage_capture(rt, args) -> None:
    import torch
    feat = args.features
    feat.mkdir(parents=True, exist_ok=True)
    pca = torch.load(args.output / 'pca.pt')
    cases = load_cases(args)
    k_shard, n_shard = args.shard
    cases = [c for i, c in enumerate(cases) if i % n_shard == k_shard]
    c0 = cases[0]
    r2.verify_prefix_tokenization(rt, user_text(c0), c0['target'])
    done = set()
    for f in sorted(feat.glob(f'chunk_s{k_shard}_*.npz')):
        done |= set(np.load(f)['ids'].tolist())
    todo = [c for c in cases if c['id'] not in done]
    print(f'capture shard {k_shard}/{n_shard}: {len(todo)} of {len(cases)} to do', flush=True)
    k0 = len(list(feat.glob(f'chunk_s{k_shard}_*.npz')))
    t0, n = time.time(), 0
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        rows, ids = [], []
        for b in r1run.batches(chunk, args.max_tokens, args.max_batch):
            texts = [r2.tool_prompt_r2(rt, user_text(c)) + r2.call_prefix(c['target']) for c in b]
            offs = [r2.prefix_sites(rt.tokenizer, c['target']) for c in b]
            rows += r2.capture_r2(rt, texts, [load_image(c, args) for c in b], offs, pca)
            ids += [c['id'] for c in b]
            n += len(b)
        save_chunk(feat / f'chunk_s{k_shard}_{k0:05d}.npz', ids, rows)
        k0 += 1
        print(f'captured {n}/{len(todo)}  {(time.time() - t0) / n:.3f} s/case', flush=True)
    write_meta(args.output, f'capture_s{k_shard}', {'n_cases': len(cases), 'max_tokens': args.max_tokens, 'max_batch': args.max_batch,
                                                    'hidden_layers': r2.HIDDEN_LAYERS, 'content_layers': r2.CONTENT_LAYERS,
                                                    'attn_layers': rt_.ATTN_LAYERS, 'sites': r2.SITES_R2})


def save_chunk(path: Path, ids: list[str], rows: list[dict]) -> None:
    lens = np.asarray([r['maps']['slot'].shape[1] for r in rows])
    arrays = {'ids': np.asarray(ids), 'hidden': np.stack([r['hidden'] for r in rows]),
              'digit_logp': np.stack([r['digit_logp'] for r in rows]), 'cont': np.asarray([r['cont'] for r in rows]),
              'map_len': lens, 'grid': np.asarray([r['grid'] for r in rows])}
    for s in r2.SITES_R2:
        arrays[f'maps_{s}'] = np.concatenate([r['maps'][s] for r in rows], 1)
    for l in r2.CONTENT_LAYERS:
        arrays[f'content_{l}'] = np.concatenate([r['content'][l] for r in rows], 0)
    np.savez(path, **arrays)


# ----------------------------------------------------------------------------- own answers

def own_cases(cases: list[dict]) -> list[dict]:
    return [c for c in cases if c['set'].startswith('g_') and c['split'] == 'test' and c['phrasing'] == 0]


def stage_own(rt, args) -> None:
    from analysis_tools.common import qwen_runtime as qr
    path = args.output / 'own.jsonl'
    done = {(r['id'], r['format']) for r in read_jsonl(path)}
    cases = own_cases(load_cases(args))
    for fmt in ('point', 'bbox'):
        todo = [c for c in cases if (c['id'], fmt) not in done]
        for b in r1run.batches(todo, args.max_tokens, args.max_batch):
            res = qr.generate_batch(rt, [([load_image(c, args)], r1run.own_prompt(c, fmt)) for c in b], 48)
            append_jsonl(path, [{'id': c['id'], 'format': fmt, 'text': r['text'], 'n_tokens': r['n_tokens'],
                                 'finish_reason': r['finish_reason'], 'batch_size': len(b)} for c, r in zip(b, res)])
        print(f'own {fmt} done', flush=True)


def stage_spotcheck(rt, args) -> None:
    from analysis_tools.common import qwen_runtime as qr
    own = {(r['id'], r['format']): r for r in read_jsonl(args.output / 'own.jsonl')}
    cases = {c['id']: c for c in load_cases(args)}
    rng = np.random.default_rng(SEED)
    keys = sorted(own)
    rows = []
    for i in rng.choice(len(keys), 20, replace=False):
        cid, fmt = keys[i]
        c = cases[cid]
        ref = qr.generate(rt, qr.build(rt, [load_image(c, args)], r1run.own_prompt(c, fmt)), 48)
        a, b = rt_.parse_answer(ref['text'], fmt), rt_.parse_answer(own[(cid, fmt)]['text'], fmt)
        rows.append({'id': cid, 'format': fmt, 'answer_equal': a == b,
                     'max_abs_diff': None if a is None or b is None else float(np.abs(np.subtract(a, b)).max())})
    write_meta(args.output, 'spotcheck', {'n': len(rows), 'answer_equal': sum(r['answer_equal'] for r in rows), 'rows': rows})
    print(f"spotcheck {sum(r['answer_equal'] for r in rows)}/20", flush=True)


# ----------------------------------------------------------------------------- free calls

def stage_freecall(rt, args) -> None:
    path = args.output / 'freecall.jsonl'
    done = {r['id'] for r in read_jsonl(path)}
    cases = [c for c in load_cases(args) if c['split'] == 'test' and c['phrasing'] == 0]
    rng = np.random.default_rng(SEED + 7)
    pool = []
    for s in sorted({c['set'] for c in cases}):
        g = [c for c in cases if c['set'] == s]
        pool += [g[i] for i in sorted(rng.choice(len(g), min(40, len(g)), replace=False))]
    todo = [c for c in pool if c['id'] not in done]
    for b in r1run.batches(todo, args.max_tokens, args.max_batch):
        texts = [r2.tool_prompt_r2(rt, user_text(c)) for c in b]
        gen = r1run.generate_texts(rt, texts, [load_image(c, args) for c in b], 96)
        rows = []
        for c, g in zip(b, gen):
            t = g['text']
            k = t.find('<tool_call>')
            target = approx = None
            if k >= 0:
                import re
                m = re.search(r'<parameter=target>\n(.*?)\n</parameter>\s*<parameter=approx_point>\n\[(.*)', t[k:], re.S)
                if m:
                    target, approx = m.group(1), r2.parse_approx(m.group(2))
            rows.append({'id': c['id'], 'set': c['set'], 'text': t, 'called': k >= 0, 'target_generated': target,
                         'target_canonical': c['target'], 'same_prefix': target == c['target'], 'approx': approx,
                         'reasoning_before_call': k > 0, 'batch_size': len(b)})
        append_jsonl(path, rows)
    print('freecall done', flush=True)


# ----------------------------------------------------------------------------- tool on the experiment-1.5 stimuli

ORDINAL_RUN = 'behavior/2026-10-16_ordinal_mechanism'


def ordinal_cases(args) -> list[dict]:
    """Experiment-1.5 grid variants as round-2 cases (truth box = target cell box)."""
    root = args.behavior_root.parent / ORDINAL_RUN
    meta = {r['image'].split('/')[-1]: r for r in read_jsonl(root / 'cases.jsonl') if r['mode'] == 'direct'}
    out = []
    for r in read_jsonl(root / 'images.jsonl'):
        t = meta[r['path'].split('/')[-1]]['truth']
        box = [float(v) for v in t['target_cell_box']]
        out.append({'id': r['image_id'], 'set': 'ordinal15', 'split': 'test', 'subset': r['image_id'].split(':')[1],
                    'image': r['path'], 'image_root': 'ordinal', 'image_sha256': r['sha256'], 'target': rt_.T6_TARGET,
                    'phrasing': 0, 'truth': {'canvas': list(t['canvas']), 'box': box,
                                             'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                                             'hit': {'kind': 'box', 'box': box}}})
    return out


def stage_ordtool(rt, args) -> None:
    import torch
    feat = args.features.parent / 'features_ord'
    feat.mkdir(parents=True, exist_ok=True)
    pca = torch.load(args.output / 'pca.pt')
    cases = ordinal_cases(args)
    (args.output / 'ordinal_cases.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cases))
    done = set()
    for f in sorted(feat.glob('chunk_*.npz')):
        done |= set(np.load(f)['ids'].tolist())
    todo = [c for c in cases if c['id'] not in done]
    k0 = len(list(feat.glob('chunk_*.npz')))
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        rows, ids = [], []
        for b in r1run.batches(chunk, args.max_tokens, args.max_batch):
            texts = [r2.tool_prompt_r2(rt, user_text(c)) + r2.call_prefix(c['target']) for c in b]
            offs = [r2.prefix_sites(rt.tokenizer, c['target']) for c in b]
            rows += r2.capture_r2(rt, texts, [load_image(c, args) for c in b], offs, pca)
            ids += [c['id'] for c in b]
        save_chunk(feat / f'chunk_{k0:05d}.npz', ids, rows)
        k0 += 1
        print(f'ordtool {start + len(chunk)}/{len(todo)}', flush=True)


# ----------------------------------------------------------------------------- digit-slot residual capture (3.12 follow-up)

DIGIT_LAYERS = (20, 24, 26, 27, 28)
N_DIGIT = {'train': {'refcoco': 2000, 'synth': 1500, 'graded': 200}, 'test_per_set': 200, 'test_refcoco': 600}


def digit_offsets(tokenizer, target: str, cont: str):
    """Teacher-forced suffix (call prefix + the model's own rough point up to ']') and the negative offsets
    of q1x / q1y, the first digit tokens of x and y (each predicts the next digit; experiment-3.12 role q1)."""
    m = r2.APPROX_RE.match(cont)
    if m is None:
        return None
    own = cont[:m.end()]
    text = r2.call_prefix(target) + own
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    base = len(r2.call_prefix(target))
    cx = base + m.start(1) + (1 if own[m.start(1)] == '-' else 0)
    cy = base + m.start(2) + (1 if own[m.start(2)] == '-' else 0)
    n = len(enc['input_ids'])
    kx = next(i for i, (a, b) in enumerate(enc['offset_mapping']) if a <= cx < b)
    ky = next(i for i, (a, b) in enumerate(enc['offset_mapping']) if a <= cy < b)
    if tokenizer.decode(enc['input_ids'][kx:kx + 1]).strip() != text[cx] or tokenizer.decode(enc['input_ids'][ky:ky + 1]).strip() != text[cy]:
        raise ValueError(f'digit tokenization differs for {own!r}')
    return own, {'q1x': kx - n, 'q1y': ky - n}


def digit_cases(args) -> list[dict]:
    cases = load_cases(args)
    rng = np.random.default_rng(SEED + 9)
    pick = []
    for s in sorted({c['set'] for c in cases}):
        g = [c for c in cases if c['set'] == s and c['phrasing'] != 'heldout']
        tr = [c for c in g if c['split'] == 'train']
        key = 'graded' if s.startswith('g_') else s
        n = N_DIGIT['train'].get(key, 0)
        pick += [tr[i] for i in sorted(rng.choice(len(tr), min(n, len(tr)), replace=False))] if tr and n else []
        pick += [c for c in g if c['split'] == 'val']
        te = [c for c in g if c['split'] == 'test']
        k = min(N_DIGIT['test_refcoco'] if s == 'refcoco' else N_DIGIT['test_per_set'], len(te))
        pick += [te[i] for i in sorted(rng.choice(len(te), k, replace=False))] if te else []
    return pick


def stage_digitcap(rt, args) -> None:
    """Residual states at q1x / q1y (layers DIGIT_LAYERS) after teacher-forcing the model's own rough point."""
    import torch
    feat = args.features.parent / 'features_digit'
    feat.mkdir(parents=True, exist_ok=True)
    cont = {}
    for f in sorted(args.features.glob('chunk_s*_*.npz')):
        d = np.load(f)
        cont.update(zip(d['ids'].tolist(), d['cont'].tolist()))
    k_shard, n_shard = args.shard
    cases = [c for i, c in enumerate(digit_cases(args)) if c['id'] in cont and i % n_shard == k_shard]
    done = set()
    for f in sorted(feat.glob(f'chunk_s{k_shard}_*.npz')):
        done |= set(np.load(f)['ids'].tolist())
    todo = [c for c in cases if c['id'] not in done]
    print(f'digitcap: {len(todo)} of {len(cases)} to do', flush=True)
    k0 = len(list(feat.glob(f'chunk_s{k_shard}_*.npz')))
    skipped = []
    for start in range(0, len(todo), CHUNK):
        ids, X, owns = [], [], []
        for b in r1run.batches(todo[start:start + CHUNK], args.max_tokens, args.max_batch):
            texts, offs, imgs, keep = [], [], [], []
            for c in b:
                got = digit_offsets(rt.tokenizer, c['target'], cont[c['id']])
                if got is None:
                    skipped.append(c['id'])
                    continue
                texts.append(r2.tool_prompt_r2(rt, user_text(c)) + r2.call_prefix(c['target']) + got[0])
                offs.append(got[1])
                imgs.append(load_image(c, args))
                keep.append(c)
                owns.append(got[0])
            if not keep:
                continue
            x = rt_.capture_sites(rt, texts, imgs, offs, layers=DIGIT_LAYERS, sites=('q1x', 'q1y'))
            ids += [c['id'] for c in keep]
            X.append(x)
        np.savez(feat / f'chunk_s{k_shard}_{k0:05d}.npz', ids=np.asarray(ids), x=np.concatenate(X), own=np.asarray(owns))
        k0 += 1
        print(f'digitcap {start + CHUNK}/{len(todo)}', flush=True)
    write_meta(args.output, f'digitcap_s{k_shard}', {'layers': DIGIT_LAYERS, 'n': len(cases), 'skipped_unparsed': skipped})
    del torch


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--r1', type=Path, required=True, help='round-1 run directory (cases.jsonl, images)')
    p.add_argument('--refcoco', type=Path, required=True)
    p.add_argument('--behavior-root', type=Path, required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=8000)
    p.add_argument('--max-batch', type=int, default=32)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--max-memory', default='', help='comma list of GiB per visible GPU; empty = one GPU')
    args = p.parse_args()
    args.shard = tuple(int(v) for v in args.shard.split('/'))
    args.r1_images = args.r1 / 'images'
    args.output.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(',')
    if 'prepare' in stages:
        stage_prepare(args)
        stages.remove('prepare')
    if not stages:
        return
    from analysis_tools.common import qwen_runtime as qr
    rt = rt_.load_runtime(args.model, [float(v) for v in args.max_memory.split(',')] if args.max_memory else None)
    write_meta(args.output, f'runtime_s{args.shard[0]}', qr.describe(rt))
    table = {'pca': stage_pca, 'capture': stage_capture, 'own': stage_own, 'spotcheck': stage_spotcheck,
             'freecall': stage_freecall, 'ordtool': stage_ordtool, 'digitcap': stage_digitcap}
    if 'dissoc' in stages:
        from analysis_tools.position import readout_tool_r2_dissoc as dz
        table['dissoc'] = dz.stage_dissoc
    if 'ordcoord' in stages:
        from analysis_tools.position import readout_tool_r2_dissoc as dz
        table['ordcoord'] = dz.stage_ordcoord
    for s in stages:
        t0 = time.time()
        table[s](rt, args)
        append_jsonl(args.output / 'stage_log.jsonl', [{'stage': s, 'shard': list(args.shard), 'seconds': time.time() - t0,
                                                        'utc': datetime.now(timezone.utc).isoformat()}])


if __name__ == '__main__':
    main()
