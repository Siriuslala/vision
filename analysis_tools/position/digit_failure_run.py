"""Experiment 3.12 stage drivers (GPU). Design: docs/analysis/position/design_digit_failure.md.

Stages (comma list via --stages):
  grid_gen   greedy answers of the T6-like grid point questions (equal-length batches, no padding) and a
             20-case spot check of the batched decoder against the reference `generate`.
  factorial  first-digit factorial on every unit family (see `digit_failure.py`): forced first digit k,
             digit distributions, residuals and direct logit attribution at q0 / q1.
  attn       recomputed softmax attention of the full-attention layers at q0 and q1 for a subset of units
             (attention mass on the image span, on the forced first digit and on the rest of the text).
Round-1 records of experiment 3.6 provide the disc sweeps (--round1).
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
from analysis_tools.position import coordinate_value as cv
from analysis_tools.position import coordinate_value_r2 as c2
from analysis_tools.position import digit_failure as df
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

COUNT_RANGE = range(10, 51)
COUNT_SEEDS = 3
REFCOCO_N = 120
TEXTPOS_RANGE = range(10, 100)
CTX_VALUES = (200, 800)


def code_digest() -> dict:
    root = Path(__file__).parent
    names = ('digit_failure.py', 'digit_failure_run.py', 'coordinate_value.py', 'coordinate_readout.py')
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()[:16] for n in names}


def sweep(recs: list[dict], name: str, every: int) -> list[dict]:
    rs = sorted([r for r in recs if r['set'] == name], key=lambda r: r['truth_units'][name == 'sweep_v'])
    return rs[::every]


def build_units(args) -> list[df.Unit]:
    r1 = Path(args.round1)
    recs = read_jsonl(r1 / 'gen' / 'records.jsonl')
    cases = {c['id']: c for c in json.loads((r1 / 'cases.json').read_text())}
    P = cv.POINT_PROMPT
    units: list[df.Unit] = []
    for r in sweep(recs, 'sweep_h', 2):
        units.append(df.Unit(r['id'], 'sweep_h', 0, r['truth_units'][0], P, df.point_prefix(0, None),
                             source={'case': r['id'], 'center': cases[r['id']]['center']}))
    for r in sweep(recs, 'sweep_v', 2):
        units.append(df.Unit(r['id'], 'sweep_v', 1, r['truth_units'][1], P, df.point_prefix(1, 500), ctx=500,
                             source={'case': r['id'], 'center': cases[r['id']]['center']}))
    for r in sweep(recs, 'sweep_v', 4):
        for v in CTX_VALUES:
            units.append(df.Unit(f"ctx{v}_{r['id']}", 'ctx', 1, r['truth_units'][1], P, df.point_prefix(1, v), ctx=v,
                                 source={'case': r['id'], 'center': cases[r['id']]['center']}))
    units.append(df.Unit('blank_x', 'blank', 0, float('nan'), P, df.point_prefix(0, None), source={'blank': True}))
    units.append(df.Unit('blank_y', 'blank', 1, float('nan'), P, df.point_prefix(1, 500), ctx=500, source={'blank': True}))
    rc = [c for c in json.loads(Path(args.refcoco_cases).read_text()) if c['format'] == 'point'][:REFCOCO_N]
    for c in rc:
        W, H = c['truth']['canvas']
        units.append(df.Unit(f"rc_{c['id']}", 'refcoco', 0, 1000 * c['truth']['center'][0] / W, c['prompt'],
                             df.point_prefix(0, None),
                             source={'image_path': c['image_path'], 'image_sha256': c['image_sha256']}))
    for g in df.grid_cases():
        _, truth = df.grid_input(g)
        units.append(df.Unit(g['id'], 'grid', 0, truth['center_units'][0], df.GRID_PROMPT, df.point_prefix(0, None),
                             source={'grid': g, 'truth': truth}))
    for n in COUNT_RANGE:
        for s in range(COUNT_SEEDS):
            units.append(df.Unit(f'count_{n:02d}_{s}', 'count', 0, float(n), c2.COUNT_QUERY, '', ks=(1, 2, 3, 4, 5),
                                 source={'n': n, 'seed': 1000 * n + s}))
    for r in sweep(recs, 'sweep_h', 4):
        px, py = (int(round(v)) for v in cases[r['id']]['center'])
        units.append(df.Unit(f"tc_{r['id']}", 'textconv', 0, 1000 * px / c2.CANVAS, c2.TEXT_QUERY.format(px=px, py=py),
                             df.point_prefix(0, None), source={'case': r['id'], 'center': cases[r['id']]['center']}))
    for p in TEXTPOS_RANGE:
        units.append(df.Unit(f'textpos_{p:02d}', 'textpos', 0, float(p), df.text_position_prompt(p), '',
                             source={'position': p}))
    return units


def unit_image(u: df.Unit) -> Image.Image | None:
    if u.family in ('sweep_h', 'sweep_v', 'ctx', 'textconv'):
        img, _ = cr.disc_image(tuple(u.source['center']), cv.DIAMETER, cv.CANVAS)
        return img
    if u.family == 'blank':
        return Image.new('RGB', (cv.CANVAS, cv.CANVAS), 'white')
    if u.family == 'refcoco':
        p = Path(u.source['image_path'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != u.source['image_sha256']:
            raise ValueError(f'Image hash mismatch for {p}')
        return Image.open(p).convert('RGB')
    if u.family == 'grid':
        img, _ = df.grid_input(u.source['grid'])
        return img
    if u.family == 'count':
        img, _ = c2.count_image(u.source['n'], u.source['seed'])
        return img
    return None


def unit_rows(rt, u: df.Unit, digit_ids: list[int]) -> list[tuple]:
    """(x, pos, q0, q1, k) for every forced first digit of a unit."""
    pre = df.prefix_ids(rt, u.prefix)
    img = unit_image(u)
    if img is None:
        x0, p0, L = df.text_prepare(rt, u.prompt)
    else:
        prep = cr.prepare(rt, img, u.prompt)
        x0, p0, L = cr.embeds(rt, prep), prep.pos, len(prep.ids)
    rows = []
    for k in u.ks:
        x, p = df.append_tokens(rt, x0, p0, pre + [digit_ids[k]])
        q0 = L + len(pre) - 1
        rows.append((x, p, q0, q0 + 1, k))
    return rows


def stage_factorial(rt, out: Path, args) -> None:
    root = out / 'factorial'
    root.mkdir(parents=True, exist_ok=True)
    feat = Path(args.features) / 'factorial'
    feat.mkdir(parents=True, exist_ok=True)
    units = build_units(args)
    upath = root / 'units.json'
    if not upath.exists():
        upath.write_text(df.dumps([u.__dict__ for u in units]))
    rpath = root / 'rows.jsonl'
    done = {r['unit'] for r in read_jsonl(rpath)} if rpath.exists() else set()
    todo = [u for u in units if u.id not in done]
    digit_ids = df.digit_ids(rt)
    shard = len(list(feat.glob('shard_*.npz')))
    t0 = time.time()
    pending: list[tuple] = []          # (unit, row tuple); always whole units

    def run_rows(rows: list[tuple], cap: int) -> dict:
        parts = []
        for s in range(0, len(rows), cap):
            chunk = rows[s:s + cap]
            X, P = c2.pad_batch([r[0] for r in chunk], [r[1] for r in chunk])
            parts.append(df.readout_forward(rt, X, P, [r[2] for r in chunk], [r[3] for r in chunk]))
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    def flush():
        nonlocal shard, pending
        if not pending:
            return
        L = max(r[0].shape[1] for _, r in pending)
        res = run_rows([r for _, r in pending], max(1, args.token_budget // L))
        rows = []
        for i, (u, r) in enumerate(pending):
            rows.append({'unit': u.id, 'family': u.family, 'axis': u.axis, 'truth': u.truth, 'ctx': u.ctx, 'k': r[4],
                         'q0': r[2], 'q1': r[3], 'shard': shard, 'index': i,
                         'logp': res['logp'][i].tolist(), 'lp_top': res['lp_top'][i].tolist(),
                         'top': [rt.tokenizer.decode([int(t)]) for t in res['top_id'][i]]})
        np.savez(feat / f'shard_{shard:05d}.npz', resid=res['resid'], dla_mixer=res['dla_mixer'].astype(np.float32),
                 dla_mlp=res['dla_mlp'].astype(np.float32), dla_head=res['dla_head'].astype(np.float32),
                 dla_embed=res['dla_embed'], logit=res['logit'])
        append_jsonl(rpath, rows)          # rows are written only after their shard exists
        shard += 1
        pending = []

    for j, u in enumerate(todo):
        rows = unit_rows(rt, u, digit_ids)
        L = rows[0][0].shape[1]
        cap = max(1, args.token_budget // L)
        if pending and (len(pending) + len(rows) > cap or abs(pending[0][1][0].shape[1] - L) > 64):
            flush()
        pending.extend((u, r) for r in rows)
        if len(pending) >= cap:
            flush()
        if j % 50 == 0:
            print(f'factorial {j}/{len(todo)} units, {time.time() - t0:.0f}s', flush=True)
    flush()
    print(f'factorial done {time.time() - t0:.0f}s', flush=True)


def stage_grid_gen(rt, out: Path, args) -> None:
    root = out / 'grid'
    root.mkdir(parents=True, exist_ok=True)
    gpath = root / 'gen.jsonl'
    done = {r['id'] for r in read_jsonl(gpath)} if gpath.exists() else set()
    cases = [c for c in df.grid_cases() if c['id'] not in done]
    t0 = time.time()
    for n in df.GRID_SIZES:
        group = [c for c in cases if c['n'] == n]
        bs = max(1, args.token_budget // {6: 260, 10: 520, 20: 1720}[n])
        for s in range(0, len(group), bs):
            chunk = group[s:s + bs]
            xs, ps, truths = [], [], []
            for c in chunk:
                img, truth = df.grid_input(c)
                prep = cr.prepare(rt, img, df.GRID_PROMPT)
                xs.append(cr.embeds(rt, prep))
                ps.append(prep.pos)
                truths.append((truth, len(prep.ids)))
            if len({x.shape[1] for x in xs}) != 1:
                raise ValueError('Grid prompts of one size must have equal length')
            res = cr.greedy(rt, torch.cat(xs), torch.cat(ps, 1), args.max_new_tokens)
            rows = []
            for c, (truth, L), r in zip(chunk, truths, res):
                coords = cr.parse_coords(r['text'], 'point')
                rows.append({'id': c['id'], **c, 'truth': truth, 'prompt_len': L, 'text': r['text'], 'ids': r['ids'],
                             'finish_reason': r['finish_reason'], 'coords': coords, 'batch_size': len(chunk)})
            append_jsonl(gpath, rows)
        print(f'grid_gen {n}x{n} done {time.time() - t0:.0f}s', flush=True)
    # Spot check: batched manual greedy vs the reference generate (batch size 1).
    spath = root / 'spotcheck.jsonl'
    if not spath.exists():
        gen = {r['id']: r for r in read_jsonl(gpath)}
        rng = np.random.default_rng(0)
        ids = sorted(gen)
        pick = [ids[i] for i in rng.choice(len(ids), 20, replace=False)]
        rows = []
        for i in pick:
            c = gen[i]
            img, _ = df.grid_input(c)
            ref = qr.generate(rt, qr.build(rt, [img], df.GRID_PROMPT), args.max_new_tokens)
            rows.append({'id': i, 'batched': c['text'], 'reference': ref['text'],
                         'same_text': ref['text'] == c['text'],
                         'same_coords': cr.parse_coords(ref['text'], 'point') == c['coords']})
        append_jsonl(spath, rows)
        print(f"spotcheck same coords {sum(r['same_coords'] for r in rows)}/20", flush=True)


def stage_attn(rt, out: Path, args) -> None:
    """Attention mass at q0 / q1 on the image span and on the forced first-digit token."""
    root = out / 'attn'
    root.mkdir(parents=True, exist_ok=True)
    apath = root / 'attn.jsonl'
    done = {(r['unit'], r['k']) for r in read_jsonl(apath)} if apath.exists() else set()
    units = [u for u in build_units(args) if u.family in ('sweep_h', 'sweep_v', 'blank', 'textconv', 'count', 'refcoco')]
    digit_ids = df.digit_ids(rt)
    t0 = time.time()
    for j, u in enumerate(units):
        if u.family in ('sweep_h', 'sweep_v', 'refcoco', 'count') and j % 2:
            continue                                   # every other unit is enough for attention statistics
        ks = [df.first_and_second(u.truth)[0] if u.family != 'count' else int(u.truth) // 10] if u.family != 'blank' else [4, 5]
        pre = df.prefix_ids(rt, u.prefix)
        prep = cr.prepare(rt, unit_image(u), u.prompt)
        L = len(prep.ids)
        span = (prep.span['start'], prep.span['end'])
        for k in ks:
            if (u.id, k) in done:
                continue
            x, p = df.append_tokens(rt, cr.embeds(rt, prep), prep.pos, pre + [digit_ids[k]])
            q0 = L + len(pre) - 1
            res = df.attention_mass(rt, x, p, [q0, q0 + 1], span, q0 + 1)
            append_jsonl(apath, [{'unit': u.id, 'family': u.family, 'truth': u.truth, 'k': k, 'layers': list(df.FULL_LAYERS),
                                  'image': [m.T.tolist() for m in res['image']], 'digit': [m.T.tolist() for m in res['digit']],
                                  'check': res['check']}])
        if j % 40 == 0:
            print(f'attn {j}/{len(units)} {time.time() - t0:.0f}s', flush=True)


def load_split(name: str, max_memory: str) -> qr.Runtime:
    """Load the checkpoint with its weights split over several visible GPUs (same dtype and attention)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    checkpoint = resolve_model(name, load_paths()['HF_HUB_CACHE'])
    mem = {int(k): v for k, v in (item.split(':') for item in max_memory.split(','))}
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16,
                                                        device_map='auto', max_memory=mem,
                                                        attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    return qr.Runtime(name, checkpoint, model, processor, 'cuda:0')


STAGES = {'grid_gen': stage_grid_gen, 'factorial': stage_factorial, 'attn': stage_attn}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--refcoco-cases', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--token-budget', type=int, default=12000)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--max-memory', default='', help='optional weight split, e.g. "0:13GiB,1:13GiB" (visible indices)')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rt = load_split(args.model, args.max_memory) if args.max_memory else qr.load(args.model)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages, 'runtime': qr.describe(rt),
                         'code': code_digest(), 'token_budget': args.token_budget, 'max_new_tokens': args.max_new_tokens})
    meta.write_text(json.dumps(info, indent=1))
    for s in args.stages.split(','):
        t0 = time.time()
        STAGES[s](rt, out, args)
        print(f'stage {s} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
