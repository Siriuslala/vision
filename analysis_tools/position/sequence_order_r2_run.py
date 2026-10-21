"""Stage drivers of experiment 3.8 round 2 (Gated DeltaNet mechanism of the row ordinal).

Stages (resumable; one JSONL record per image unless noted):
  gates      alpha_t / beta_t of all 24 DeltaNet layers on the visual span: 8x8 grids (identity order and
             a layout with a blank cell row inserted), white canvas, RefCOCO test images;
  probe      cell-level features under dissociating layouts; identity-trained row / column probes applied to
             every layout, and pooled probes of the figure-row ordinal vs the ViT row (residual and per head);
  intervene  gate overrides (state reset, no decay, no write) with greedy decoding of the row / column answer;
  restore2   mixer-output restoration of block moves at subsets of slots ordered by distance from the target;
  transplant the target's tokens are replaced by the red figure's tokens from an image whose target sits one
             cell away (the ViT position changes, the sequence order and every count stay);
  think      block moves under the think-then-answer prompt and the direct prompt (paired subset);
  spotcheck  batched manual decoder against the reference generate().

Usage: python -m analysis_tools.position.sequence_order_r2_run --stages gates,probe ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import sequence_order as so
from analysis_tools.position import sequence_order_r2 as s2
from analysis_tools.position.sequence_order_run import append_jsonl, done_ids, read_jsonl, blank_vis

TEMPLATES = ('person', 'ring')
RES_LAYERS = [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 28, 32]
HEAD_LAYERS = list(s2.MID_LINEAR)
SAVE_FEATURE_LAYERS = (8, 12, 16, 24)


def code_digest() -> dict:
    here = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(here.glob('sequence_order*.py')) + [here / 'coordinate_readout.py']}


def cases8() -> list[dict]:
    return [{'id': f'g8x8_{t}_r{r}c{c}', 'template': t, 'target': [r, c]}
            for t in TEMPLATES for r in range(8) for c in range(8)]


def natural_images(src: Path, split: str, n: int) -> list[tuple[int, Image.Image]]:
    """Square 512 center crops of the experiment-3.4 RefCOCO images of one split."""
    cases = [c for c in json.loads((src / 'cases.json').read_text())['natural'] if c['split'] == split][:n]
    out = []
    for c in cases:
        im = Image.open(src / 'images/natural512' / f"{c['id']:04d}.png").convert('RGB')
        s = min(im.size)
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width - s) // 2 + s, (im.height - s) // 2 + s))
        out.append((c['id'], im.resize((512, 512), Image.Resampling.LANCZOS)))
    if len(out) != n:
        raise ValueError(f'Only {len(out)} {split} images available, need {n}.')
    return out


# ----------------------------------------------------------------------------- gates

def stage_gates(rt, out: Path, args) -> None:
    root = out / 'gates'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'done.json').exists():
        return
    layers = list(so.LINEAR_LAYERS)
    blank = blank_vis(rt, (512, 512))
    items = []                                   # (kind, id, image, layout or None)
    for case in cases8():
        image, _ = so.grid_image(case['template'], 8, 8, tuple(case['target']))
        items.append(('grid', case['id'], image, None))
    for case in cases8()[::4]:                   # 32 grids with a blank cell row inserted after cell row 3
        image, _ = so.grid_image(case['template'], 8, 8, tuple(case['target']))
        items.append(('grid_blankrow', case['id'], image, 'insert_blank_mid'))
    items.append(('white', 'white', Image.new('RGB', (512, 512), (255, 255, 255)), None))
    for nid, im in natural_images(Path(args.natural), 'test', args.n_natural):
        items.append(('natural', f'refcoco_{nid}', im, None))
    alpha, beta, masks, meta = [], [], [], []
    lay = s2.layout_conditions()
    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        xs, ps, spans = [], [], []
        for kind, iid, image, lname in chunk:
            prep = cr.prepare(rt, image, so.PROMPT)
            H, W = prep.grid
            if lname is None:
                seq = so.permuted(prep, so.order_identity(H, W))
                fm = s2.figure_mask(image, (H, W))
                index = np.arange(H * W)
            else:
                L = s2.make_layout(lay[lname], W)
                seq = s2.layout_sequence(prep, L, blank[(H // 2) * W + W // 2])
                fm0 = s2.figure_mask(image, (H, W))
                fm = np.where(L.index >= 0, fm0[L.index.clip(0)], False)
                index = L.index
            x, p = so.build(rt, prep, seq)
            xs.append(x)
            ps.append(p)
            spans.append((prep.span['start'], prep.span['start'] + len(index)))
            masks.append(fm)
            meta.append({'kind': kind, 'id': iid, 'index': index.tolist(), 'grid': [H, W]})
        for L in sorted({x.shape[1] for x in xs}):
            idx = [k for k, x in enumerate(xs) if x.shape[1] == L]
            store = {}
            span = spans[idx[0]]
            with s2.record_gates(rt, layers, span, store, L), torch.inference_mode():
                rt.model(inputs_embeds=torch.cat([xs[k] for k in idx]), position_ids=torch.cat([ps[k] for k in idx], 1),
                         use_cache=False, logits_to_keep=1)
            for j, k in enumerate(idx):
                alpha.append((i + k, torch.stack([store[l][0][j] for l in layers]).half()))
                beta.append((i + k, torch.stack([store[l][1][j] for l in layers]).half()))
        print(f'gates {i + len(chunk)}/{len(items)}', flush=True)
    alpha = [a for _, a in sorted(alpha, key=lambda t: t[0])]
    beta = [b for _, b in sorted(beta, key=lambda t: t[0])]
    torch.save({'layers': layers, 'alpha': alpha, 'beta': beta, 'mask': masks, 'meta': meta}, root / 'gates.pt')
    (root / 'done.json').write_text(json.dumps({'n': len(meta), 'batch_size': args.batch_size}))


# ----------------------------------------------------------------------------- probes

def stage_probe(rt, out: Path, args) -> None:
    root = out / 'probe'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'results.json').exists():
        return
    rng = np.random.default_rng(args.seed)
    picks = [c for c in cases8()]
    picks = [picks[i] for i in sorted(rng.choice(len(picks), args.n_probe, replace=False))]
    lay = s2.layout_conditions()
    blank = blank_vis(rt, (512, 512))
    cells, feats_res, feats_head = [], {l: [] for l in RES_LAYERS}, {l: [] for l in HEAD_LAYERS}
    t0 = time.time()
    for ii, case in enumerate(picks):
        image, _ = so.grid_image(case['template'], 8, 8, tuple(case['target']))
        prep = cr.prepare(rt, image, so.PROMPT)
        H, W = prep.grid
        for lname, rows in lay.items():
            L = s2.make_layout(rows, W)
            x, p = so.build(rt, prep, s2.layout_sequence(prep, L, blank[(H // 2) * W + W // 2]))
            st = s2.capture_grouped(rt, x, p, prep.span['start'], L.cell_of_slot, RES_LAYERS, HEAD_LAYERS)
            for l in RES_LAYERS:
                feats_res[l].append(st['res'][l])
            for l in HEAD_LAYERS:
                feats_head[l].append(st['head'][l])
            for k, cell in enumerate(L.cells):
                cells.append({'image': ii, 'case': case['id'], 'template': case['template'], 'layout': lname,
                              'cell': k, **cell})
        print(f'probe capture {ii + 1}/{len(picks)} {time.time() - t0:.0f}s', flush=True)
    res = {l: torch.cat(v).float() for l, v in feats_res.items()}
    head = {l: torch.cat(v).float() for l, v in feats_head.items()}
    torch.save({'cells': cells, 'res': {l: res[l].half() for l in SAVE_FEATURE_LAYERS}}, root / 'features.pt')
    dev = 'cuda'
    lay_arr = np.array([c['layout'] for c in cells])
    img = np.array([c['image'] for c in cells])
    var = {k: torch.tensor([float(c[k]) for c in cells], dtype=torch.float64) for k in
           ('vit_row', 'vit_col', 'ord_row', 'ord_col', 'slot_row')}
    results = {'transfer': {}, 'pooled': {}, 'pooled_head': {}, 'alpha': {}}
    preds = {}
    ident = torch.as_tensor(lay_arr == 'identity')
    # (a) identity-trained probes applied to every layout.
    for l in RES_LAYERS:
        X = res[l].to(dev)
        for axis, key in (('row', 'vit_row'), ('col', 'vit_col')):
            m, a = s2.ridge_select(X[ident.to(dev)], var[key][ident].to(dev), img[ident.numpy()])
            preds[f'{axis}_L{l}'] = ridge_predict_cpu(m, X)
            results['alpha'][f'{axis}_L{l}'] = a
    # (b) pooled probes on the row layouts (ordinal and ViT row dissociated), held-out images.
    row_lay = ~np.isin(lay_arr, ['dup_col_mid', 'delete_col_mid', 'roll_within+1'])
    col_lay = np.isin(lay_arr, ['identity', 'dup_col_mid', 'delete_col_mid', 'roll_within+1'])
    test_img = np.unique(img)[::4]
    def pooled(X, sel, keys):
        tr = sel & ~np.isin(img, test_img)
        te = sel & np.isin(img, test_img)
        out = {}
        for k in keys:
            trt, tet = torch.as_tensor(tr), torch.as_tensor(te)
            m, _ = s2.ridge_select(X[trt.to(X.device)], var[k][trt].to(X.device), img[tr])
            out[k] = s2.r2_score(s2.ridge_predict(m, X[tet.to(X.device)]), var[k][tet].to(X.device))
        return out
    for l in RES_LAYERS:
        X = res[l].to(dev)
        results['pooled'][l] = {'row': pooled(X, row_lay, ('ord_row', 'vit_row', 'slot_row')),
                                'col': pooled(X, col_lay, ('ord_col', 'vit_col'))}
        print('pooled', l, results['pooled'][l], flush=True)
    for l in HEAD_LAYERS:
        results['pooled_head'][l] = []
        for h in range(head[l].shape[1]):
            X = head[l][:, h].to(dev)
            results['pooled_head'][l].append({'row': pooled(X, row_lay, ('ord_row', 'vit_row')),
                                              'col': pooled(X, col_lay, ('ord_col', 'vit_col'))})
        print('pooled heads', l, flush=True)
    torch.save({'cells': cells, 'preds': preds}, root / 'preds.pt')
    (root / 'results.json').write_text(json.dumps(results, indent=1))


def ridge_predict_cpu(model, X) -> torch.Tensor:
    return s2.ridge_predict(model, X).float().cpu()


# ----------------------------------------------------------------------------- gate interventions

def intervention_jobs(case: dict, H: int, W: int, start: int) -> list[dict]:
    """Gate-override jobs for one 8x8 case. Each job: name, base order, override dict, metadata."""
    r, c = case['target']
    T = so.cell_tokens((r, c), (H, W))
    groups = {'mid': s2.MID_LINEAR, 'early': s2.EARLY_LINEAR, 'late': s2.LATE_LINEAR, 'all': so.LINEAR_LAYERS}
    ident = so.order_identity(H, W)
    def rowstarts(n):
        return start + np.arange(0, n, W)
    def all_pos(order):
        return start + np.arange(len(order))
    def half_pos(order, half):
        # Top (half 0) or bottom (half 1) token row of every cell row; every token row holds W tokens here.
        slots = np.arange(len(order))
        return start + slots[(slots // W) % so.CELL_TOKENS == half]
    def target_pos(order):
        inv = np.argsort(order) if len(order) == H * W else None
        return start + inv[T]
    jobs = [{'name': 'none', 'base': 'identity', 'order': ident, 'ov': {}}]
    for g in ('mid', 'early', 'late'):
        jobs.append({'name': f'reset_rowstart|{g}', 'base': 'identity', 'order': ident,
                     'ov': {'layers': groups[g], 'reset': rowstarts(H * W)}})
        jobs.append({'name': f'nowrite_all|{g}', 'base': 'identity', 'order': ident,
                     'ov': {'layers': groups[g], 'nowrite': all_pos(ident)}})
    for half, tag in ((0, 'top'), (1, 'bottom')):
        jobs.append({'name': f'nowrite_{tag}|mid', 'base': 'identity', 'order': ident,
                     'ov': {'layers': groups['mid'], 'nowrite': half_pos(ident, half)}})
    jobs.append({'name': 'nodecay|mid', 'base': 'identity', 'order': ident,
                 'ov': {'layers': groups['mid'], 'nodecay': start + np.arange(H * W)}})
    for g in ('mid', 'all'):
        jobs.append({'name': f'nowrite_target|{g}', 'base': 'identity', 'order': ident,
                     'ov': {'layers': groups[g], 'nowrite': target_pos(ident)}})
        for k in sorted({r, r - 1}):
            if k >= 1:
                jobs.append({'name': f'reset_at_k|{g}', 'k': k, 'base': 'identity', 'order': ident,
                             'ov': {'layers': groups[g], 'reset': np.array([start + CTW(k, W)])}})
    if r >= 2:
        delete = np.concatenate([np.arange(H * W).reshape(H, W)[t] for t in range(H) if t // so.CELL_TOKENS != r - 1])
        for name, ov in (('none', {}), ('reset_rowstart|mid', {'layers': groups['mid'], 'reset': rowstarts(len(delete))}),
                         ('nowrite_all|mid', {'layers': groups['mid'], 'nowrite': all_pos(delete)})):
            jobs.append({'name': name, 'base': 'delete_row_before', 'order': delete, 'ov': ov})
    if r <= 6:
        down = so.order_block_move(H, W, T, so.cell_tokens((r + 1, c), (H, W)))
        for name, ov in (('none', {}), ('reset_rowstart|mid', {'layers': groups['mid'], 'reset': rowstarts(H * W)}),
                         ('nowrite_all|mid', {'layers': groups['mid'], 'nowrite': all_pos(down)}),
                         ('nowrite_target|mid', {'layers': groups['mid'], 'nowrite': target_pos(down)})):
            jobs.append({'name': name, 'base': 'block_down', 'order': down, 'ov': ov})
    if c <= 6:
        right = so.order_block_move(H, W, T, so.cell_tokens((r, c + 1), (H, W)))
        for name, ov in (('none', {}), ('reset_rowstart|mid', {'layers': groups['mid'], 'reset': rowstarts(H * W)}),
                         ('nowrite_target|mid', {'layers': groups['mid'], 'nowrite': target_pos(right)})):
            jobs.append({'name': name, 'base': 'block_right', 'order': right, 'ov': ov})
    return jobs


def intervention_jobs2(case: dict, H: int, W: int, start: int) -> list[dict]:
    """Supplementary gate overrides on content-preserving and content-changing row edits (all DeltaNet groups)."""
    r, c = case['target']
    T = so.cell_tokens((r, c), (H, W))
    groups = {'early': s2.EARLY_LINEAR, 'mid': s2.MID_LINEAR, 'late': s2.LATE_LINEAR, 'all': so.LINEAR_LAYERS}
    g = np.arange(H * W).reshape(H, W)
    bases = {'identity': so.order_identity(H, W)}
    if r >= 2:
        bases['delete_row_before'] = np.concatenate([g[t] for t in range(H) if t // so.CELL_TOKENS != r - 1])
    if r >= 1:
        bases['roll_rows-1'] = so.order_roll_rows(H, W, -so.CELL_TOKENS)
    if r <= 6:
        bases['block_down'] = so.order_block_move(H, W, T, so.cell_tokens((r + 1, c), (H, W)))
    jobs = []
    for base, order in bases.items():
        n = len(order)
        jobs.append({'name': 'none', 'base': base, 'order': order, 'ov': {}})
        for gname in ('early', 'mid', 'all'):
            jobs.append({'name': f'nowrite_all|{gname}', 'base': base, 'order': order,
                         'ov': {'layers': groups[gname], 'nowrite': start + np.arange(n)}})
            jobs.append({'name': f'reset_rowstart|{gname}', 'base': base, 'order': order,
                         'ov': {'layers': groups[gname], 'reset': start + np.arange(0, n, W)}})
    return jobs


def stage_intervene2(rt, out: Path, args) -> None:
    path = out / 'intervene2' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    for case in cases8():
        if case['id'] in finished:
            continue
        t0 = time.time()
        image, _ = so.grid_image(case['template'], 8, 8, tuple(case['target']))
        prep = cr.prepare(rt, image, so.PROMPT)
        H, W = prep.grid
        jobs = intervention_jobs2(case, H, W, prep.span['start'])
        built = [so.build(rt, prep, so.permuted(prep, j['order'])) for j in jobs]
        results = [None] * len(jobs)
        for L in sorted({x.shape[1] for x, _ in built}):
            idx = [i for i, (x, _) in enumerate(built) if x.shape[1] == L]
            hooks = lambda a, b, idx=idx, L=L: s2.override_gates(rt, [jobs[idx[i]]['ov'] for i in range(a, b)], L)
            res = so.decode(rt, [built[i][0] for i in idx], [built[i][1] for i in idx], args.batch_size,
                            args.max_new_tokens, hooks=hooks)
            for i, rr in zip(idx, res):
                results[i] = so.read_rowcol(rt, rr, 8, 8)
        rows = [{**{k: v for k, v in j.items() if k not in ('order', 'ov')}, **a} for j, a in zip(jobs, results)]
        append_jsonl(path, [{**case, 'conditions': rows, 'batch_size': args.batch_size, 'seconds': round(time.time() - t0, 1)}])
        print(f"intervene2 {case['id']} {len(jobs)} jobs {time.time() - t0:.1f}s", flush=True)


def CTW(k: int, W: int) -> int:
    """Slot of the first token of cell row k."""
    return so.CELL_TOKENS * k * W


def stage_intervene(rt, out: Path, args) -> None:
    path = out / 'intervene' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    for case in cases8():
        if case['id'] in finished:
            continue
        t0 = time.time()
        image, _ = so.grid_image(case['template'], 8, 8, tuple(case['target']))
        prep = cr.prepare(rt, image, so.PROMPT)
        H, W = prep.grid
        jobs = intervention_jobs(case, H, W, prep.span['start'])
        built = [so.build(rt, prep, so.permuted(prep, j['order'])) for j in jobs]
        results = [None] * len(jobs)
        for L in sorted({x.shape[1] for x, _ in built}):
            idx = [i for i, (x, _) in enumerate(built) if x.shape[1] == L]
            hooks = lambda a, b, idx=idx, L=L: s2.override_gates(rt, [jobs[idx[i]]['ov'] for i in range(a, b)], L)
            res = so.decode(rt, [built[i][0] for i in idx], [built[i][1] for i in idx], args.batch_size,
                            args.max_new_tokens, hooks=hooks)
            for i, rr in zip(idx, res):
                results[i] = so.read_rowcol(rt, rr, 8, 8)
        rows = [{**{k: v for k, v in j.items() if k not in ('order', 'ov')}, **a} for j, a in zip(jobs, results)]
        append_jsonl(path, [{**case, 'conditions': rows, 'batch_size': args.batch_size,
                             'seconds': round(time.time() - t0, 1)}])
        print(f"intervene {case['id']} {len(jobs)} jobs {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- restoration by distance

def stage_restore2(rt, out: Path, args) -> None:
    path = out / 'restore2' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    all_layers = set(range(1, 33))
    mid = set(range(9, 17))
    for case in cases8():
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
        jobs = []
        for tag, dr, dc in (('down', 1, 0), ('up', -1, 0), ('right', 0, 1), ('left', 0, -1)):
            if not (0 <= r + dr < 8 and 0 <= c + dc < 8):
                continue
            dst = so.cell_tokens((r + dr, c + dc), (H, W))
            order = so.order_block_move(H, W, T, dst)
            first = min(dst)
            row0 = (first // W) // so.CELL_TOKENS               # slot cell row of the moved target
            tset = set(dst)
            others = [i for i in range(H * W) if i not in tset]
            subsets = {
                'none': [],
                'target': list(dst),
                'others': others,
                'before_target': [i for i in others if i < first],
                'after_samerow': [i for i in others if i > first and (i // W) // so.CELL_TOKENS == row0],
                'later_rows': [i for i in others if (i // W) // so.CELL_TOKENS > row0],
            }
            after = [i for i in others if i > first]
            for k in (2, 4, 8, 16, 32):
                subsets[f'next{k}'] = after[:k]
            meta = {'move': tag, 'axis': 'v' if dr else 'h', 'd': dr or dc, 'slot_cell': so.slot_cell(order, T, W)}
            for name, sl in subsets.items():
                jobs.append((f'{tag}|{name}', {**meta, 'subset': name, 'layers': 'all', 'n_slots': len(sl)},
                             order, all_layers if sl else set(), sl if sl else None))
            jobs.append((f'{tag}|after_samerow_mid', {**meta, 'subset': 'after_samerow', 'layers': '9-16',
                                                      'n_slots': len(subsets['after_samerow'])},
                         order, mid, subsets['after_samerow']))
        xs, ps = [], []
        for _, _, order, _, _ in jobs:
            x, p = so.build(rt, prep, so.permuted(prep, order))
            xs.append(x)
            ps.append(p)
        L = xs[0].shape[1]
        hooks = lambda a, b: so.restore_mixers(rt, clean, span, [(j[2], j[3], j[4]) for j in jobs[a:b]], L)
        res = so.decode(rt, xs, ps, args.batch_size, args.max_new_tokens, hooks=hooks)
        rows = [{'name': n, **m, **so.read_rowcol(rt, rr, 8, 8)} for (n, m, _, _, _), rr in zip(jobs, res)]
        append_jsonl(path, [{**case, 'conditions': rows, 'batch_size': args.batch_size,
                             'seconds': round(time.time() - t0, 1)}])
        print(f"restore2 {case['id']} {len(jobs)} jobs {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- ViT-position transplant

def stage_transplant(rt, out: Path, args) -> None:
    path = out / 'transplant' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    for case in cases8():
        if case['id'] in finished:
            continue
        t0 = time.time()
        r, c = case['target']
        image, _ = so.grid_image(case['template'], 8, 8, (r, c))
        prep = cr.prepare(rt, image, so.PROMPT)
        H, W = prep.grid
        T = so.cell_tokens((r, c), (H, W))
        conds = [{'name': 'identity', 'vis': prep.vis}]
        for tag, dr, dc in (('down', 1, 0), ('up', -1, 0), ('right', 0, 1), ('left', 0, -1)):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < 8 and 0 <= cc < 8):
                continue
            donor_img, _ = so.grid_image(case['template'], 8, 8, (rr, cc))
            donor = cr.prepare(rt, donor_img, so.PROMPT)
            vis = prep.vis.clone()
            vis[T] = donor.vis[so.cell_tokens((rr, cc), (H, W))]
            conds.append({'name': f'transplant_{tag}', 'axis': 'v' if dr else 'h', 'd': dr or dc, 'vis': vis})
        s, e = prep.span['start'], prep.span['end']
        built = [so.build(rt, prep, so.Sequence(cd['vis'], prep.pos[1:, 0, s:e])) for cd in conds]
        res = so.decode(rt, [b[0] for b in built], [b[1] for b in built], args.batch_size, args.max_new_tokens)
        rows = [{**{k: v for k, v in cd.items() if k != 'vis'}, **so.read_rowcol(rt, rr_, 8, 8)} for cd, rr_ in zip(conds, res)]
        append_jsonl(path, [{**case, 'conditions': rows, 'batch_size': args.batch_size,
                             'seconds': round(time.time() - t0, 1)}])
        print(f"transplant {case['id']} {time.time() - t0:.1f}s", flush=True)


def stage_transplant_ctx(rt, out: Path, args) -> None:
    """Context transplant: every token outside the target cell comes from an image whose red figure sits one cell
    away (its red cell replaced by the same cell of an image without a red figure); the target cell keeps the
    tokens of the original image. The red figure is seen only at the original slot, while the ViT context of all
    other tokens was computed with the red figure at the neighbouring cell."""
    from analysis_tools.common import stimuli as st
    path = out / 'transplant_ctx' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    normals = {}
    for case in cases8():
        if case['id'] in finished:
            continue
        t0 = time.time()
        r, c = case['target']
        tpl = case['template']
        image, _ = so.grid_image(tpl, 8, 8, (r, c))
        prep = cr.prepare(rt, image, so.PROMPT)
        H, W = prep.grid
        T = so.cell_tokens((r, c), (H, W))
        if tpl not in normals:
            img_n, meta_n = st.generate(st.StimulusSpec(tpl, 8, 8, target=None, cell=so.CELL, seed=0))
            if meta_n['grid_origin'] != [0, 0]:
                raise ValueError('Normal grid must fill the canvas.')
            normals[tpl] = cr.prepare(rt, img_n, so.PROMPT).vis
        conds = [{'name': 'identity', 'vis': prep.vis}]
        for tag, dr, dc in (('down', 1, 0), ('up', -1, 0), ('right', 0, 1), ('left', 0, -1)):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < 8 and 0 <= cc < 8):
                continue
            donor_img, _ = so.grid_image(tpl, 8, 8, (rr, cc))
            donor = cr.prepare(rt, donor_img, so.PROMPT)
            vis = donor.vis.clone()
            D = so.cell_tokens((rr, cc), (H, W))
            vis[D] = normals[tpl][D]
            vis[T] = prep.vis[T]
            conds.append({'name': f'context_{tag}', 'axis': 'v' if dr else 'h', 'd': dr or dc, 'vis': vis})
        s_, e_ = prep.span['start'], prep.span['end']
        built = [so.build(rt, prep, so.Sequence(cd['vis'], prep.pos[1:, 0, s_:e_])) for cd in conds]
        res = so.decode(rt, [b[0] for b in built], [b[1] for b in built], args.batch_size, args.max_new_tokens)
        rows = [{**{k: v for k, v in cd.items() if k != 'vis'}, **so.read_rowcol(rt, x, 8, 8)} for cd, x in zip(conds, res)]
        append_jsonl(path, [{**case, 'conditions': rows, 'batch_size': args.batch_size, 'seconds': round(time.time() - t0, 1)}])
        print(f"transplant_ctx {case['id']} {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- think-then-answer

def stage_think(rt, out: Path, args) -> None:
    path = out / 'think' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    rng = np.random.default_rng(args.seed + 7)
    pool = [c for c in cases8() if 1 <= c['target'][0] <= 6 and 1 <= c['target'][1] <= 6]
    picks = [pool[i] for i in sorted(rng.choice(len(pool), args.n_think, replace=False))]
    for case in picks:
        if case['id'] in finished:
            continue
        t0 = time.time()
        r, c = case['target']
        image, _ = so.grid_image(case['template'], 8, 8, (r, c))
        rec = {**case, 'conditions': []}
        for mode, prompt, max_new in (('direct', so.PROMPT, args.max_new_tokens), ('think', s2.THINK_PROMPT, args.max_new_think)):
            prep = cr.prepare(rt, image, prompt)
            H, W = prep.grid
            T = so.cell_tokens((r, c), (H, W))
            conds = [('identity', None, 0, so.order_identity(H, W))]
            for tag, dr, dc in (('down', 1, 0), ('up', -1, 0), ('right', 0, 1), ('left', 0, -1)):
                conds.append((f'block_{tag}', 'v' if dr else 'h', dr or dc,
                              so.order_block_move(H, W, T, so.cell_tokens((r + dr, c + dc), (H, W)))))
            built = [so.build(rt, prep, so.permuted(prep, o)) for *_, o in conds]
            res = so.decode(rt, [b[0] for b in built], [b[1] for b in built], args.batch_size, max_new)
            for (name, axis, d, order), rr in zip(conds, res):
                rc = s2.parse_last_rowcol(rr['text'])
                rec['conditions'].append({'mode': mode, 'name': name, 'axis': axis, 'd': d,
                                          'slot_cell': so.slot_cell(order, T, W), 'text': rr['text'],
                                          'finish_reason': rr['finish_reason'],
                                          'row': None if rc is None else rc[0] - 1, 'col': None if rc is None else rc[1] - 1})
        rec['batch_size'] = args.batch_size
        rec['seconds'] = round(time.time() - t0, 1)
        append_jsonl(path, [rec])
        print(f"think {case['id']} {time.time() - t0:.1f}s", flush=True)


def stage_spotcheck(rt, out: Path, args) -> None:
    """Batched manual decoder (batch size args.batch_size) vs reference generate() on unmodified prompts."""
    rng = np.random.default_rng(args.seed + 3)
    cases = cases8()
    pick = [cases[i] for i in rng.choice(len(cases), args.n_spot, replace=False)]
    xs, ps, refs = [], [], []
    for case in pick:
        image, _ = so.grid_image(case['template'], 8, 8, tuple(case['target']))
        prep = cr.prepare(rt, image, so.PROMPT)
        x, p = so.build(rt, prep, so.permuted(prep, so.order_identity(*prep.grid)))
        xs.append(x)
        ps.append(p)
        refs.append(qr.generate(rt, qr.build(rt, [image], so.PROMPT), args.max_new_tokens)['text'])
    mine = so.decode(rt, xs, ps, args.batch_size, args.max_new_tokens)
    rows = [{'id': c['id'], 'manual': m['text'], 'reference': ref,
             'agree': cr.parse_rowcol(m['text']) == cr.parse_rowcol(ref)} for c, m, ref in zip(pick, mine, refs)]
    (out / 'spotcheck.json').write_text(json.dumps({'n': len(rows), 'agree': sum(r['agree'] for r in rows),
                                                    'batch_size': args.batch_size, 'rows': rows}, indent=1))
    print('spotcheck', sum(r['agree'] for r in rows), '/', len(rows), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--natural', required=True)
    p.add_argument('--stages', default='spotcheck')
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=24)
    p.add_argument('--max-new-think', type=int, default=512)
    p.add_argument('--n-spot', type=int, default=20)
    p.add_argument('--n-probe', type=int, default=32)
    p.add_argument('--n-natural', type=int, default=32)
    p.add_argument('--n-think', type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(8)        # CPU tensor ops oversubscribe the shared machine with the default thread count
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    meta_path = out / 'experiment.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {'runs': []}
    meta.update({'runtime': qr.describe(rt), 'prompt': so.PROMPT, 'think_prompt': s2.THINK_PROMPT})
    meta['runs'].append({'stages': args.stages, 'args': vars(args), 'code_sha256': code_digest(),
                         'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    meta_path.write_text(json.dumps(meta, indent=1))
    stages = {'transplant_ctx': stage_transplant_ctx, 'intervene2': stage_intervene2, 'gates': stage_gates, 'probe': stage_probe, 'intervene': stage_intervene, 'restore2': stage_restore2,
              'transplant': stage_transplant, 'think': stage_think, 'spotcheck': stage_spotcheck}
    for name in args.stages.split(','):
        stages[name](rt, out, args)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
