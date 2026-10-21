"""Stage drivers of experiment 3.8 (positional role of sequence order).

Stages (each resumable, one JSONL record per image):
  behavior   slot-order interventions on 8x8 grids (two templates) and on 8x4 / 4x8 grids;
  restore    mixer-output restoration of block-move conditions, by layer group and by single layer;
  probe      layer-wise linear probes of slot and token coordinates on rolled blank canvases and grids,
             and of the sequence index on sequences of identical visual tokens;
  fail20     order interventions on the experiment-1.3 20x20 T6 row/column cases;
  spotcheck  manual greedy decoder against the reference generate() on unmodified prompts.

Usage: python -m analysis_tools.position.sequence_order_run --stages behavior,restore ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import sequence_order as so
from analysis_tools.position.coordinate_readout_run import t6_cases, t6_input, score_t6

TEMPLATES = ('person', 'ring')


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def done_ids(path: Path) -> set[str]:
    return {r['id'] for r in read_jsonl(path)}


def code_digest() -> dict:
    here = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(here.glob('sequence_order*.py')) + [here / 'coordinate_readout.py']}


# ----------------------------------------------------------------------------- cases

def grid_cases() -> list[dict]:
    cases = []
    for tpl in TEMPLATES:
        for r in range(8):
            for c in range(8):
                cases.append({'id': f'g8x8_{tpl}_r{r}c{c}', 'template': tpl, 'rows': 8, 'cols': 8, 'target': [r, c]})
    for rows, cols in ((8, 4), (4, 8)):
        for r in range(rows):
            for c in range(cols):
                cases.append({'id': f'g{rows}x{cols}_person_r{r}c{c}', 'template': 'person', 'rows': rows, 'cols': cols,
                              'target': [r, c]})
    return cases


def blank_vis(rt, size: tuple[int, int]) -> torch.Tensor:
    """Merged visual embeddings (Nv, D) of a white canvas (width, height)."""
    from PIL import Image
    prep = cr.prepare(rt, Image.new('RGB', size, (255, 255, 255)), so.PROMPT)
    return prep.vis


def conditions(case: dict, H: int, W: int, full: bool) -> list[dict]:
    """Slot-order conditions for one grid case. Each entry: name, family, order (or insertion spec), params."""
    r, c = case['target']
    R, C = case['rows'], case['cols']
    T = so.cell_tokens((r, c), (H, W))
    out = [{'name': 'identity', 'family': 'identity', 'order': so.order_identity(H, W)}]
    for tag, (dr, dc) in (('down', (1, 0)), ('up', (-1, 0)), ('right', (0, 1)), ('left', (0, -1))):
        if 0 <= r + dr < R and 0 <= c + dc < C:
            out.append({'name': f'block_{tag}', 'family': 'block', 'axis': 'v' if dr else 'h', 'd': dr or dc,
                        'order': so.order_block_move(H, W, T, so.cell_tokens((r + dr, c + dc), (H, W)))})
    for k in (1, -1):
        out.append({'name': f'roll_rows{k:+d}', 'family': 'roll_rows', 'axis': 'v', 'd': k,
                    'order': so.order_roll_rows(H, W, so.CELL_TOKENS * k)})
        out.append({'name': f'roll_within{k:+d}', 'family': 'roll_within', 'axis': 'h', 'd': k,
                    'order': so.order_roll_within_rows(H, W, so.CELL_TOKENS * k)})
    out.append({'name': 'shuffle_others', 'family': 'shuffle', 'order': so.order_shuffle_others(H, W, T, seed=r * 31 + c)})
    if not full:
        return out
    out.append({'name': 'transpose', 'family': 'global', 'order': so.order_transpose(H, W)})
    out.append({'name': 'reverse_rows', 'family': 'global', 'order': so.order_reverse_rows(H, W)})
    out.append({'name': 'mirror_rows', 'family': 'global', 'order': so.order_mirror_rows(H, W)})
    # Vertical context carry: the moved block takes `left` / `right` tokens of its rows along.
    carries = [(2, 2), (4, 4), (8, 8), (W, W), (2, 0), (4, 0), (8, 0), (W, 0), (0, 2), (0, 4), (0, 8), (0, W)]
    for dr in (1, -1):
        if 0 <= r + dr < R:
            for left, right in carries:
                out.append({'name': f'carry_v{dr:+d}_L{left}_R{right}', 'family': 'carry_v', 'axis': 'v', 'd': dr,
                            'left': left, 'right': right, 'order': so.order_carry(H, W, (r, c), dr, left, right)})
    # Horizontal context carry: the target moves one cell with `carry` trailing tokens of its rows.
    for dc in (1, -1):
        for carry in (2, 4):
            try:
                order = so.order_segment_shift(H, W, (r, c), dc, carry)
            except ValueError:
                continue
            out.append({'name': f'carry_h{dc:+d}_{carry}', 'family': 'carry_h', 'axis': 'h', 'd': dc, 'carry': carry,
                        'order': order})
    return out


def insertion_conditions(case: dict, H: int, W: int, blank: torch.Tensor) -> list[dict]:
    """Extra blank-canvas tokens inserted into the image span (counting tests)."""
    r, _ = case['target']
    mid = blank[(H // 2) * W + W // 2][None]
    edge = blank[(H // 2) * W][None]
    rows2 = blank[(H // 2) * W:(H // 2 + 2) * W]
    return [
        {'name': 'insert_mid_start', 'family': 'insert', 'extra': mid.expand(2 * W, -1), 'at': 0, 'index_shift': 2 * W},
        {'name': 'insert_rows_start', 'family': 'insert', 'extra': rows2, 'at': 0, 'index_shift': 2 * W},
        {'name': 'insert_rows_before', 'family': 'insert', 'extra': rows2, 'at': so.CELL_TOKENS * r * W,
         'index_shift': 2 * W},
        {'name': 'insert_edge_start', 'family': 'insert', 'extra': edge.expand(so.CELL_TOKENS, -1), 'at': 0,
         'index_shift': so.CELL_TOKENS},
    ]


def describe_condition(cond: dict, T: list[int], W: int) -> dict:
    meta = {k: v for k, v in cond.items() if k not in ('order', 'extra')}
    if 'order' in cond:
        meta['slot_cell'] = so.slot_cell(cond['order'], T, W)
    else:
        inv_slot = np.array(T) + (cond['index_shift'] if cond['at'] <= min(T) else 0)
        meta['slot_cell'] = [float(((inv_slot // W).mean() - 0.5) / so.CELL_TOKENS),
                             float(((inv_slot % W).mean() - 0.5) / so.CELL_TOKENS)]
    return meta


def run_grouped(rt, prep, conds: list[dict], args, n_rows: int, n_cols: int) -> list[dict]:
    """Build and decode conditions, batching only conditions of equal sequence length."""
    built = []
    for cond in conds:
        if 'order' in cond:
            seq = so.permuted(prep, cond['order'])
        else:
            seq = so.with_inserted(prep, so.permuted(prep, so.order_identity(*prep.grid)), cond['extra'], cond['at'])
        built.append(so.build(rt, prep, seq))
    results = [None] * len(conds)
    lengths = sorted({x.shape[1] for x, _ in built})
    for L in lengths:
        idx = [i for i, (x, _) in enumerate(built) if x.shape[1] == L]
        res = so.decode(rt, [built[i][0] for i in idx], [built[i][1] for i in idx], args.batch_size, args.max_new_tokens)
        for i, r in zip(idx, res):
            results[i] = so.read_rowcol(rt, r, n_rows, n_cols)
    return results


# ----------------------------------------------------------------------------- stages

def stage_behavior(rt, out: Path, args) -> None:
    path = out / 'behavior' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    blanks = {}
    for case in grid_cases():
        if case['id'] in finished:
            continue
        t0 = time.time()
        image, meta = so.grid_image(case['template'], case['rows'], case['cols'], tuple(case['target']))
        prep = cr.prepare(rt, image, so.PROMPT)
        H, W = prep.grid
        T = so.cell_tokens(tuple(case['target']), (H, W))
        full = case['rows'] == case['cols'] == 8
        conds = conditions(case, H, W, full)
        if full:
            size = (image.width, image.height)
            if size not in blanks:
                blanks[size] = blank_vis(rt, size)
            conds += insertion_conditions(case, H, W, blanks[size])
        answers = run_grouped(rt, prep, conds, args, case['rows'], case['cols'])
        rows = [{**describe_condition(c, T, W), **a} for c, a in zip(conds, answers)]
        append_jsonl(path, [{**case, 'token_grid': [H, W], 'image_sha256': meta['sha256'], 'conditions': rows,
                             'batch_size': args.batch_size, 'seconds': round(time.time() - t0, 1)}])
        print(f"behavior {case['id']} {len(conds)} conds {time.time() - t0:.1f}s", flush=True)


def edit_conditions(case: dict, H: int, W: int) -> list[dict]:
    """Duplicate or delete whole cell rows / cell-column segments of figures (instance-counting tests).

    Each condition is a token index list (duplicates allowed, deletions omit tokens); `count` is the
    shift of the target's ordinal (in cells) predicted by counting figure rows / columns from the image start.
    """
    r, c = case['target']
    R, C = case['rows'], case['cols']
    g = np.arange(H * W).reshape(H, W)
    rows_of = lambda k: list(range(so.CELL_TOKENS * k, so.CELL_TOKENS * (k + 1)))
    donor = r - 1 if r >= 1 else r + 1                       # a non-target cell row
    out = []
    def seq_rows(row_list):
        return np.concatenate([g[t] for t in row_list])
    token_rows = list(range(H))
    t0 = so.CELL_TOKENS * r
    out.append({'name': 'dup_row_before', 'axis': 'v', 'count': 1,
                'index': seq_rows(token_rows[:t0] + rows_of(donor) + token_rows[t0:])})
    out.append({'name': 'dup_row_start', 'axis': 'v', 'count': 1,
                'index': seq_rows(rows_of(donor) + token_rows)})
    if r >= 1:
        out.append({'name': 'delete_row0', 'axis': 'v', 'count': -1,
                    'index': seq_rows([t for t in token_rows if t not in rows_of(0)])})
    if r >= 2:
        out.append({'name': 'delete_row_before', 'axis': 'v', 'count': -1,
                    'index': seq_rows([t for t in token_rows if t not in rows_of(r - 1)])})
    # Column analogues act on the target's two token rows only.
    cdonor = c - 1 if c >= 1 else c + 1
    def edit_row(t, kind):
        row = list(g[t])
        seg = row[so.CELL_TOKENS * cdonor:so.CELL_TOKENS * (cdonor + 1)]
        if kind == 'dup':
            k = so.CELL_TOKENS * c
            return row[:k] + seg + row[k:]
        if kind == 'dup_start':
            return seg + row
        k = so.CELL_TOKENS * (c - 1)
        return row[:k] + row[k + so.CELL_TOKENS:]
    for kind, name, cnt, ok in (('dup', 'dup_col_before', 1, True), ('dup_start', 'dup_col_start', 1, True),
                                ('del', 'delete_col_before', -1, c >= 1)):
        if ok:
            idx = np.concatenate([edit_row(t, kind) if t in rows_of(r) else list(g[t]) for t in range(H)])
            out.append({'name': name, 'axis': 'h', 'count': cnt, 'index': idx})
    return out


def stage_behavior2(rt, out: Path, args) -> None:
    path = out / 'behavior' / 'records_edit.jsonl'
    finished = done_ids(path)
    for case in [c for c in grid_cases() if c['rows'] == c['cols'] == 8]:
        if case['id'] in finished:
            continue
        t0 = time.time()
        image, _ = so.grid_image(case['template'], 8, 8, tuple(case['target']))
        prep = cr.prepare(rt, image, so.PROMPT)
        H, W = prep.grid
        conds = [{'name': 'identity', 'count': 0, 'index': so.order_identity(H, W)}] + edit_conditions(case, H, W)
        built = [so.build(rt, prep, so.permuted(prep, c['index'])) for c in conds]
        answers = [None] * len(conds)
        for L in sorted({x.shape[1] for x, _ in built}):
            idx = [i for i, (x, _) in enumerate(built) if x.shape[1] == L]
            res = so.decode(rt, [built[i][0] for i in idx], [built[i][1] for i in idx], args.batch_size, args.max_new_tokens)
            for i, rr in zip(idx, res):
                answers[i] = so.read_rowcol(rt, rr, 8, 8)
        rows = [{**{k: v for k, v in c.items() if k != 'index'}, 'n_visual': len(c['index']), **a} for c, a in zip(conds, answers)]
        append_jsonl(path, [{**case, 'token_grid': [H, W], 'conditions': rows, 'batch_size': args.batch_size,
                             'seconds': round(time.time() - t0, 1)}])
        print(f"behavior2 {case['id']} {len(conds)} conds {time.time() - t0:.1f}s", flush=True)


RESTORE_GROUPS = {
    'none': set(),
    'all': set(range(1, 33)),
    'linear_all': set(so.LINEAR_LAYERS),
    'full_all': set(so.FULL_LAYERS),
    'linear_1_12': {l for l in so.LINEAR_LAYERS if l <= 12},
    'linear_13_24': {l for l in so.LINEAR_LAYERS if 13 <= l <= 24},
    'linear_25_32': {l for l in so.LINEAR_LAYERS if l >= 25},
    'full_1_12': {4, 8, 12},
    'full_13_32': {16, 20, 24, 28, 32},
    'layers_1_8': set(range(1, 9)),
    'layers_9_16': set(range(9, 17)),
    'layers_17_32': set(range(17, 33)),
}


def stage_restore(rt, out: Path, args) -> None:
    """Mixer restoration of block moves; single-layer sweeps on the person template only."""
    path = out / 'restore' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    for case in [c for c in grid_cases() if c['rows'] == c['cols'] == 8]:
        if case['id'] in finished:
            continue
        t0 = time.time()
        image, _ = so.grid_image(case['template'], 8, 8, tuple(case['target']))
        prep = cr.prepare(rt, image, so.PROMPT)
        H, W = prep.grid
        span = (prep.span['start'], prep.span['end'])
        r, c = case['target']
        T = so.cell_tokens((r, c), (H, W))
        x0, p0 = so.build(rt, prep, so.permuted(prep, so.order_identity(H, W)))
        clean = so.capture_mixers(rt, x0, p0, span)
        jobs = []                     # (name, meta, order, layers, slots)
        moves = [('down', 1, 0), ('up', -1, 0), ('right', 0, 1), ('left', 0, -1)]
        for tag, dr, dc in moves:
            if not (0 <= r + dr < 8 and 0 <= c + dc < 8):
                continue
            dst = so.cell_tokens((r + dr, c + dc), (H, W))
            order = so.order_block_move(H, W, T, dst)
            meta = {'move': tag, 'axis': 'v' if dr else 'h', 'd': dr or dc, 'slot_cell': so.slot_cell(order, T, W)}
            for g, layers in RESTORE_GROUPS.items():
                jobs.append((f'{tag}|{g}', {**meta, 'group': g}, order, layers, None))
            # Restore every layer at the moved target slots only, or at all other slots only.
            jobs.append((f'{tag}|all_target_slots', {**meta, 'group': 'all_target_slots'}, order, RESTORE_GROUPS['all'], dst))
            others = [i for i in range(H * W) if i not in set(dst)]
            jobs.append((f'{tag}|all_other_slots', {**meta, 'group': 'all_other_slots'}, order, RESTORE_GROUPS['all'], others))
            if case['template'] == 'person' and tag in ('down', 'right'):
                for layer in range(1, 33):
                    jobs.append((f'{tag}|single{layer}', {**meta, 'group': f'single{layer}', 'layer': layer}, order, {layer}, None))
        xs, ps = [], []
        for _, _, order, _, _ in jobs:
            x, p = so.build(rt, prep, so.permuted(prep, order))
            xs.append(x)
            ps.append(p)
        L = xs[0].shape[1]
        hooks = lambda a, b: so.restore_mixers(rt, clean, span, [(j[2], j[3], j[4]) for j in jobs[a:b]], L)
        res = so.decode(rt, xs, ps, args.batch_size, args.max_new_tokens, hooks=hooks)
        rows = [{'name': n, **m, **so.read_rowcol(rt, rr, 8, 8)} for (n, m, _, _, _), rr in zip(jobs, res)]
        append_jsonl(path, [{**case, 'token_grid': [H, W], 'conditions': rows, 'batch_size': args.batch_size,
                             'seconds': round(time.time() - t0, 1)}])
        print(f"restore {case['id']} {len(jobs)} jobs {time.time() - t0:.1f}s", flush=True)


def stage_restore_check(rt, out: Path, args) -> None:
    """Numerical check: restoring every layer makes visual residuals equal the clean ones (re-indexed)."""
    image, _ = so.grid_image('person', 8, 8, (3, 5))
    prep = cr.prepare(rt, image, so.PROMPT)
    H, W = prep.grid
    span = (prep.span['start'], prep.span['end'])
    T = so.cell_tokens((3, 5), (H, W))
    order = so.order_block_move(H, W, T, so.cell_tokens((4, 5), (H, W)))
    x0, p0 = so.build(rt, prep, so.permuted(prep, so.order_identity(H, W)))
    clean = so.capture_mixers(rt, x0, p0, span)
    ref = so.capture_residuals(rt, x0, p0, span, [8, 16, 32])
    x1, p1 = so.build(rt, prep, so.permuted(prep, order))
    with so.restore_mixers(rt, clean, span, [(order, RESTORE_GROUPS['all'], None)], x1.shape[1]):
        got = so.capture_residuals(rt, x1, p1, span, [8, 16, 32])
    raw = so.capture_residuals(rt, x1, p1, span, [8, 16, 32])
    o = torch.as_tensor(order)
    rep = {}
    for layer in (8, 16, 32):
        scale = ref[layer].float().norm(dim=-1).mean()
        rep[layer] = {'restored_rel_err': float((got[layer].float() - ref[layer][o].float()).norm(dim=-1).mean() / scale),
                      'unrestored_rel_err': float((raw[layer].float() - ref[layer][o].float()).norm(dim=-1).mean() / scale)}
    (out / 'restore').mkdir(parents=True, exist_ok=True)
    (out / 'restore' / 'check.json').write_text(json.dumps(rep, indent=1))
    print('restore check', rep, flush=True)


def stage_spotcheck(rt, out: Path, args) -> None:
    rng = np.random.default_rng(args.seed)
    cases = grid_cases()
    pick = [cases[i] for i in rng.choice(len(cases), args.n_spot, replace=False)]
    rows = []
    for case in pick:
        image, _ = so.grid_image(case['template'], case['rows'], case['cols'], tuple(case['target']))
        prep = cr.prepare(rt, image, so.PROMPT)
        x, p = so.build(rt, prep, so.permuted(prep, so.order_identity(*prep.grid)))
        mine = so.read_rowcol(rt, so.decode(rt, [x], [p], 1, args.max_new_tokens)[0], case['rows'], case['cols'])
        ref = qr.generate(rt, qr.build(rt, [image], so.PROMPT), args.max_new_tokens)
        rows.append({'id': case['id'], 'manual': mine['text'], 'reference': ref['text'],
                     'agree': cr.parse_rowcol(mine['text']) == cr.parse_rowcol(ref['text'])})
    (out / 'spotcheck.json').write_text(json.dumps({'n': len(rows), 'agree': sum(r['agree'] for r in rows), 'rows': rows}, indent=1))
    print('spotcheck', sum(r['agree'] for r in rows), '/', len(rows), flush=True)


# ----------------------------------------------------------------------------- probes

PROBE_LAYERS = list(range(0, 33))


def stage_probe(rt, out: Path, args) -> None:
    """Capture residuals for the probe datasets and fit layer-wise ridge probes (results in probe/results.json)."""
    from PIL import Image
    root = out / 'probe'
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1)
    results = {}
    # (a) rolled inputs: blank canvas and 8x8 grids, slot (R, C) vs token (r, c) coordinates.
    for name in ('blank', 'grid'):
        feats, labels, groups = {l: [] for l in PROBE_LAYERS}, [], []
        for i in range(args.n_probe):
            if name == 'blank':
                image = Image.new('RGB', (512, 512), (255, 255, 255))
            else:
                image, _ = so.grid_image(TEMPLATES[i % 2], 8, 8, tuple(rng.integers(0, 8, 2).tolist()))
            prep = cr.prepare(rt, image, so.PROMPT)
            H, W = prep.grid
            kr, kc = (0, 0) if i == 0 else tuple(rng.integers(0, [H, W]).tolist())
            g = np.roll(np.roll(np.arange(H * W).reshape(H, W), kr, 0), kc, 1)
            order = g.ravel()
            x, p = so.build(rt, prep, so.permuted(prep, order))
            res = so.capture_residuals(rt, x, p, (prep.span['start'], prep.span['end']), PROBE_LAYERS)
            for l in PROBE_LAYERS:
                feats[l].append(res[l])
            slots = np.arange(H * W)
            labels.append(np.stack([slots // W, slots % W, order // W, order % W], 1))
            groups.append(np.full(H * W, i))
        y = torch.as_tensor(np.concatenate(labels), dtype=torch.float64)
        grp = np.concatenate(groups)
        test_groups = np.arange(args.n_probe)[::4]
        test = torch.as_tensor(np.where(np.isin(grp, test_groups))[0])
        train = torch.as_tensor(rng.permutation(np.where(~np.isin(grp, test_groups))[0]))
        out_rows = {}
        for l in PROBE_LAYERS:
            X = torch.cat(feats[l]).to('cuda').double()
            out_rows[l] = {k: so.ridge_r2(X, y[:, j].to('cuda'), train.to('cuda'), test.to('cuda'))
                           for j, k in enumerate(('slot_row', 'slot_col', 'token_row', 'token_col'))}
            print(f'probe {name} layer {l} {out_rows[l]}', flush=True)
        results[name] = out_rows
    # (b) identical visual tokens: every slot holds the mean blank-canvas embedding.
    blank = blank_vis(rt, (512, 512))
    image = Image.new('RGB', (512, 512), (255, 255, 255))
    prep = cr.prepare(rt, image, so.PROMPT)
    H, W = prep.grid
    s, e = prep.span['start'], prep.span['end']
    for variant in ('mrope', 'flat'):
        seq = so.Sequence(blank.mean(0, keepdim=True).expand(H * W, -1).clone(), prep.pos[1:, 0, s:e].clone())
        if variant == 'flat':
            seq.mrope[:] = seq.mrope[0:1, :1]          # every h / w id equals the image's temporal id
        x, p = so.build(rt, prep, seq)
        res = so.capture_residuals(rt, x, p, (s, e), PROBE_LAYERS)
        torch.save({l: res[l] for l in (0, 4, 8, 16, 24, 32)}, root / f'identical_{variant}.pt')
        n = torch.arange(H * W, dtype=torch.float64)
        perm = torch.as_tensor(rng.permutation(H * W))
        test, train = perm[:H * W // 4], perm[H * W // 4:]
        rows = {}
        for l in PROBE_LAYERS:
            X = res[l].to('cuda').double()
            targets = {'index': n, 'log_index': torch.log1p(n), 'row': n // W, 'col': n % W}
            rows[l] = {k: so.ridge_r2(X, v.to('cuda'), train.to('cuda'), test.to('cuda')) for k, v in targets.items()}
            d = X - X.mean(0)
            rows[l]['spread'] = float(d.norm(dim=-1).mean() / X.norm(dim=-1).mean())
        results[f'identical_{variant}'] = rows
        print(f'probe identical {variant}', rows[16], flush=True)
    (root / 'results.json').write_text(json.dumps(results, indent=1))


# ----------------------------------------------------------------------------- 20x20 failure cases

def stage_fail20(rt, out: Path, args) -> None:
    path = out / 'fail20' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    cases = [c for c in t6_cases(Path(args.t6_source), grids=(20,)) if c['format'] == 'rowcol']
    for case in cases:
        if case['id'] in finished:
            continue
        t0 = time.time()
        image, truth, text = t6_input(case)
        prep = cr.prepare(rt, image, text)
        H, W = prep.grid
        r, c = truth['target_rc']
        T = so.cell_tokens((r, c), (H, W))
        conds = [{'name': 'identity', 'order': so.order_identity(H, W)}]
        for k in (1, -1):
            conds.append({'name': f'roll_rows{k:+d}', 'axis': 'v', 'd': k, 'order': so.order_roll_rows(H, W, 2 * k)})
            conds.append({'name': f'roll_within{k:+d}', 'axis': 'h', 'd': k, 'order': so.order_roll_within_rows(H, W, 2 * k)})
        for tag, dc in (('right', 1), ('left', -1)):
            if 0 <= c + dc < 20:
                conds.append({'name': f'block_{tag}', 'axis': 'h', 'd': dc,
                              'order': so.order_block_move(H, W, T, so.cell_tokens((r, c + dc), (H, W)))})
            for carry in (2, 4):
                try:
                    conds.append({'name': f'carry_h{dc:+d}_{carry}', 'axis': 'h', 'd': dc, 'carry': carry,
                                  'order': so.order_segment_shift(H, W, (r, c), dc, carry)})
                except ValueError:
                    pass
        # Scramble the target's own two token rows (target slots fixed): removes in-row sequence context.
        rows_tokens = [R * W + C for R in range(2 * r, 2 * r + 2) for C in range(W)]
        g = np.arange(H * W)
        movable = [t for t in rows_tokens if t not in set(T)]
        g[movable] = np.random.default_rng(r * 97 + c).permutation(movable)
        conds.append({'name': 'scramble_target_rows', 'order': g})
        xs, ps = [], []
        for cond in conds:
            x, p = so.build(rt, prep, so.permuted(prep, cond['order']))
            xs.append(x)
            ps.append(p)
        res = so.decode(rt, xs, ps, args.batch_size_large, args.max_new_tokens)
        rows = []
        for cond, rr in zip(conds, res):
            meta = {k: v for k, v in cond.items() if k != 'order'}
            meta['slot_cell'] = so.slot_cell(cond['order'], T, W)
            rows.append({**meta, 'text': rr['text'], 'finish_reason': rr['finish_reason'], 'score': score_t6(case, rr['text'])})
        append_jsonl(path, [{'id': case['id'], 'truth': truth, 'token_grid': [H, W], 'conditions': rows,
                             'batch_size': args.batch_size_large, 'seconds': round(time.time() - t0, 1)}])
        print(f"fail20 {case['id']} {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--t6-source', required=True)
    p.add_argument('--stages', default='spotcheck,behavior')
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--batch-size-large', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=24)
    p.add_argument('--n-spot', type=int, default=20)
    p.add_argument('--n-probe', type=int, default=40)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    meta_path = out / 'experiment.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {'runs': []}
    meta.update({'runtime': qr.describe(rt), 'prompt': so.PROMPT})
    meta['runs'].append({'stages': args.stages, 'args': vars(args), 'code_sha256': code_digest(),
                         'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    meta_path.write_text(json.dumps(meta, indent=1))
    stages = {'spotcheck': stage_spotcheck, 'behavior': stage_behavior, 'behavior2': stage_behavior2, 'restore_check': stage_restore_check,
              'restore': stage_restore, 'probe': stage_probe, 'fail20': stage_fail20}
    for name in args.stages.split(','):
        stages[name](rt, out, args)


if __name__ == '__main__':
    main()
