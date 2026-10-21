"""Stage drivers of experiment 3.10 (one mechanism for row and column ordinals). Resumable JSONL records.

Stages:
  spotcheck  unpadded batched greedy decoder (the one used by every stage) against the reference generate();
  gates      prefill of the experiment-1.5 slot grids (H32, H64, V32, V64; 320 images each): alpha_t and
             beta_t of all 24 Gated DeltaNet layers at every visual token, and residual-stream means of the
             cells on the target's short-axis line and on a second line (running-count probe features);
  gapgate    gate overrides at "boundary tokens" (the first figure tokens after an interior blank line) for
             the heads selected from the gate stage (`boundary_heads.json`, written by the analysis), with
             matched-position and random-head controls, plus an artificial boundary inserted into base images;
  headclass  head-class-specific gate overrides on the experiment-3.8 8x8 grids (no decay / no write in
             layers 9-16 for integrator, row-reset, cell-row-onset heads separately);
  implicit   direct-answer vs think-then-answer (thinking off) under the same prefill gate overrides, on the
             8x8 grids and on the 1.5 V64 base images; a small thinking-on comparison;
  order      default ({"row", "column"}) vs swapped ({"column", "row"}) output order: answers and the layer-16 /
             layer-20 attention of the query that predicts the first digit of each value.

Usage: python -m analysis_tools.position.rowcol_mechanism_run --stages gates,...
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import rowcol_mechanism as rm
from analysis_tools.position import sequence_order as so
from analysis_tools.position import sequence_order_r2 as s2
from analysis_tools.position.sequence_order_run import append_jsonl, done_ids, read_jsonl

RES_LAYERS = (4, 8, 12, 16, 20, 24, 28)
ATTN_LAYERS = (16, 20)
GRID_TEMPLATES = ('person', 'ring')
BIG = s2.BIG


def code_digest() -> dict:
    here = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob('rowcol_mechanism*.py'))}


def cases8() -> list[dict]:
    return [{'id': f'g8x8_{t}_r{r}c{c}', 'template': t, 'target': [r, c]}
            for t in GRID_TEMPLATES for r in range(8) for c in range(8)]


def ordinal_image(run: Path, case: dict) -> Image.Image:
    return Image.open(run / case['image']).convert('RGB')


# ----------------------------------------------------------------------------- gate overrides with per-layer heads

@contextmanager
def override_gates_heads(rt: qr.Runtime, rows: list[dict], prompt_len: int):
    """Prefill-only gate overrides with a head set per layer.

    rows: one dict per batch row with optional keys 'reset', 'nodecay', 'nowrite', each a dict
    {1-based DeltaNet layer: (absolute positions array, heads list or None for all 32 heads)}.
    reset sets alpha_t = 0, nodecay sets alpha_t = 1, nowrite sets beta_t = 0 (same encoding as s2.override_gates).
    """
    layers = sorted({l for row in rows for key in ('reset', 'nodecay', 'nowrite') for l in row.get(key, {})})
    handles = []

    def items(layer, key):
        out = []
        for b, row in enumerate(rows):
            spec = row.get(key, {}).get(layer)
            if spec is None or len(spec[0]) == 0 or (spec[1] is not None and len(spec[1]) == 0):
                continue
            pos = torch.as_tensor(np.asarray(spec[0]), dtype=torch.long)
            heads = None if spec[1] is None else torch.as_tensor(spec[1], dtype=torch.long)
            out.append((b, pos, heads))
        return out

    def edit(out, plan):
        if out.shape[1] != prompt_len:
            return out
        out = out.clone()
        for (b, pos, heads), value in plan:
            pos = pos.to(out.device)
            if heads is None:
                out[b, pos] = value
            else:
                sub = out[b, pos]
                sub[:, heads.to(out.device)] = value
                out[b, pos] = sub
        return out
    for layer in layers:
        mod = s2.linear_block(rt, layer)
        a_plan = [(it, BIG) for it in items(layer, 'reset')] + [(it, -BIG) for it in items(layer, 'nodecay')]
        b_plan = [(it, -BIG) for it in items(layer, 'nowrite')]
        if a_plan:
            handles.append(mod.in_proj_a.register_forward_hook(lambda m, a, o, p=a_plan: edit(o, p)))
        if b_plan:
            handles.append(mod.in_proj_b.register_forward_hook(lambda m, a, o, p=b_plan: edit(o, p)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def decode_jobs(rt, preps: list, jobs: list[dict], batch_size: int, max_new: int) -> list[dict]:
    """Greedy decoding of jobs {'prep': index into preps, 'ov': override row}; equal prompt lengths per batch."""
    out = [None] * len(jobs)
    by_len = {}
    for i, j in enumerate(jobs):
        by_len.setdefault(len(preps[j['prep']].ids), []).append(i)
    for L, idx in by_len.items():
        for a in range(0, len(idx), batch_size):
            chunk = idx[a:a + batch_size]
            x = torch.cat([cr.embeds(rt, preps[jobs[i]['prep']]) for i in chunk], 0)
            p = torch.cat([preps[jobs[i]['prep']].pos for i in chunk], 1)
            with override_gates_heads(rt, [jobs[i].get('ov', {}) for i in chunk], L):
                res = cr.greedy(rt, x, p, max_new)
            for i, r in zip(chunk, res):
                out[i] = r
    return out


def answer_fields(text: str, finish: str) -> dict:
    rc = rm.parse_rowcol_any(text)
    return {'text': text, 'finish_reason': finish, 'row': None if rc is None else rc[0],
            'col': None if rc is None else rc[1]}


# ----------------------------------------------------------------------------- spot check

def stage_spotcheck(rt, out: Path, args) -> None:
    rng = np.random.default_rng(args.seed + 1)
    o_cases = rm.load_ordinal_cases(Path(args.ordinal_run), ('H64', 'V64'))
    picks = [o_cases[i] for i in rng.choice(len(o_cases), args.n_spot // 2, replace=False)]
    items = [(ordinal_image(Path(args.ordinal_run), c), rm.PROMPTS['direct' if k % 2 else 'swapped'], c['id'])
             for k, c in enumerate(picks)]
    for case in [cases8()[i] for i in rng.choice(128, args.n_spot - len(picks), replace=False)]:
        items.append((so.grid_image(case['template'], 8, 8, tuple(case['target']))[0], rm.PROMPTS['direct'], case['id']))
    rows = []
    for im, prompt, cid in items:
        prep = cr.prepare(rt, im, prompt)
        mine = decode_jobs(rt, [prep], [{'prep': 0}], 1, args.max_new_tokens)[0]['text']
        ref = qr.generate(rt, qr.build(rt, [im], prompt), args.max_new_tokens)['text']
        rows.append({'id': cid, 'prompt': prompt[:60], 'manual': mine, 'reference': ref,
                     'agree': rm.parse_rowcol_any(mine) == rm.parse_rowcol_any(ref), 'same_text': mine == ref})
    # Batched decoding (the batch size used by the stages) against batch size 1 on the 8x8 grids.
    preps = [cr.prepare(rt, so.grid_image(c['template'], 8, 8, tuple(c['target']))[0], rm.PROMPTS['direct'])
             for c in cases8()[:args.batch_size]]
    b1 = [decode_jobs(rt, preps, [{'prep': i}], 1, args.max_new_tokens)[0]['text'] for i in range(len(preps))]
    bn = [r['text'] for r in decode_jobs(rt, preps, [{'prep': i} for i in range(len(preps))], args.batch_size,
                                         args.max_new_tokens)]
    summary = {'n': len(rows), 'agree': sum(r['agree'] for r in rows), 'same_text': sum(r['same_text'] for r in rows),
               'batch_vs_single_agree': sum(rm.parse_rowcol_any(a) == rm.parse_rowcol_any(b) for a, b in zip(b1, bn)),
               'batch_n': len(b1), 'batch_size': args.batch_size, 'rows': rows}
    (out / 'spotcheck.json').write_text(json.dumps(summary, indent=1))
    print('spotcheck', summary['agree'], '/', summary['n'], 'batch', summary['batch_vs_single_agree'], flush=True)


# ----------------------------------------------------------------------------- gates and running-count features

def stage_gates(rt, out: Path, args) -> None:
    root = out / 'gates'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    run = Path(args.ordinal_run)
    cases = rm.load_ordinal_cases(run, tuple(args.conditions.split(',')))
    for cond in args.conditions.split(','):
        todo = [c for c in cases if c['condition'] == cond and c['id'] not in finished]
        for a in range(0, len(todo), args.prefill_batch):
            t0 = time.time()
            chunk = todo[a:a + args.prefill_batch]
            preps = [cr.prepare(rt, ordinal_image(run, c), rm.DIRECT_PROMPT) for c in chunk]
            geos = [rm.geometry(c['truth'], c['axis']) for c in chunk]
            s, e = preps[0].span['start'], preps[0].span['end']
            L = len(preps[0].ids)
            if any(len(p.ids) != L or p.span['start'] != s for p in preps):
                raise ValueError('prompts of one condition must share their length and image span')
            lines = []
            for g in geos:
                o2 = (g.target[1] + 3) % 6
                lines.append([g.cell_tokens(slot, o) for o in (g.target[1], o2) for slot in range(21)])
            feats = {l: [] for l in RES_LAYERS}
            handles = []
            for l in RES_LAYERS:
                def hook(m, i, o, l=l):
                    h = o[0] if isinstance(o, tuple) else o
                    if h.shape[1] != L:
                        return
                    vis = h[:, s:e].float()
                    for b, cells in enumerate(lines):
                        idx = torch.tensor(cells, device=vis.device)            # (42, k*k)
                        feats[l].append(vis[b][idx].mean(1).half().cpu())       # (42, D)
                handles.append(rt.layers[l - 1].register_forward_hook(hook))
            gates = {}
            x = torch.cat([cr.embeds(rt, p) for p in preps], 0)
            pos = torch.cat([p.pos for p in preps], 1)
            try:
                with torch.inference_mode(), s2.record_gates(rt, rm.LINEAR_LAYERS, (s, e), gates, L):
                    rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
            finally:
                for h in handles:
                    h.remove()
            for b, c in enumerate(chunk):
                alpha = torch.stack([gates[l][0][b] for l in rm.LINEAR_LAYERS]).half()   # (24, Nv, 32)
                beta = torch.stack([gates[l][1][b] for l in rm.LINEAR_LAYERS]).half()
                res = torch.stack([feats[l][b] for l in RES_LAYERS])                     # (7, 42, D)
                name = c['id'].split('|')[0]
                torch.save({'alpha': alpha, 'beta': beta, 'res': res}, root / f'{name}.pt')
            append_jsonl(path, [{'id': c['id'], 'file': c['id'].split('|')[0] + '.pt', 'grid': list(g.grid),
                                 'span': [s, e], 'res_layers': list(RES_LAYERS), 'batch_size': len(chunk)}
                                for c, g in zip(chunk, geos)])
            print(f'gates {cond} {a + len(chunk)}/{len(todo)} {time.time() - t0:.1f}s', flush=True)


# ----------------------------------------------------------------------------- boundary-gate interventions

def random_heads(sets: dict[int, list[int]], rng: np.random.Generator, exclude: dict[int, list[int]]) -> dict:
    out = {}
    for l, hs in sets.items():
        pool = [h for h in range(32) if h not in exclude.get(l, [])]
        out[l] = sorted(rng.choice(pool, len(hs), replace=False).tolist())
    return out


def stage_gapgate(rt, out: Path, args) -> None:
    root = out / 'gapgate'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    sel = json.loads((out / 'boundary_heads.json').read_text())
    run = Path(args.ordinal_run)
    integrators = rm.head_sets(Path(args.gate_stats))['integrator']
    rng = np.random.default_rng(args.seed + 5)
    cases = [c for c in rm.load_ordinal_cases(run, tuple(args.conditions.split(',')))
             if c['variant'] in ('insL', 'delL', 'base') and c['id'] not in finished]
    for cond in args.conditions.split(','):
        axis = cond[0]
        heads = {int(l): h for l, h in sel[axis]['heads'].items() if h}
        rand = random_heads(heads, rng, heads)
        todo = [c for c in cases if c['condition'] == cond]
        for a in range(0, len(todo), args.gap_chunk):
            t0 = time.time()
            chunk = todo[a:a + args.gap_chunk]
            preps, jobs, meta = [], [], []
            for i, c in enumerate(chunk):
                prep = cr.prepare(rt, ordinal_image(run, c), rm.DIRECT_PROMPT)
                preps.append(prep)
                g = rm.geometry(c['truth'], c['axis'])
                s = prep.span['start']
                if c['variant'] == 'base':
                    # Artificial boundary: the gates of the selected heads at line `target - 3` are set as if a
                    # blank line preceded it (state erased); only targets at slot >= 5 (c >= 6) are used.
                    if g.target[0] >= 5:
                        line = g.target[0] - 3
                        ga = rm.SlotGeometry(g.axis, g.cell, tuple(p for p in g.present if p != line - 1),
                                             g.target, g.grid, g.origin)
                        btok = s + ga.boundary_tokens()
                        conds = [('none', {}),
                                 ('reset_boundary', {'reset': {l: (btok, h) for l, h in heads.items()}}),
                                 ('reset_random', {'reset': {l: (btok, h) for l, h in rand.items()}})]
                        extra = {'artificial_line': int(line), 'pred_restart': 4}
                    else:
                        conds, extra = [('none', {})], {}
                else:
                    btok = s + g.boundary_tokens()
                    ctok = s + g.matched_control_tokens()
                    conds = [('none', {}),
                             ('nodecay_boundary', {'nodecay': {l: (btok, h) for l, h in heads.items()}}),
                             ('nodecay_control', {'nodecay': {l: (ctok, h) for l, h in heads.items()}}),
                             ('nodecay_random', {'nodecay': {l: (btok, h) for l, h in rand.items()}}),
                             ('reset_integrators', {'reset': {l: (btok, h) for l, h in integrators.items() if h}})]
                    extra = {}
                for name, ov in conds:
                    jobs.append({'prep': i, 'ov': ov})
                    meta.append((c, name, extra))
            res = decode_jobs(rt, preps, jobs, args.batch_size, args.max_new_tokens)
            rows = {}
            for (c, name, extra), r in zip(meta, res):
                rec = rows.setdefault(c['id'], {'id': c['id'], 'condition': c['condition'], 'variant': c['variant'],
                                                'c': c['c'], 'truth': {k: c['truth'][k] for k in ('count', 'gap', 'slot')},
                                                **extra, 'runs': {}})
                f = answer_fields(r['text'], r['finish_reason'])
                f['long'], f['short'] = rm.long_short(c['axis'], None if f['row'] is None else (f['row'], f['col']))
                rec['runs'][name] = f
            append_jsonl(path, [{**v, 'batch_size': args.batch_size} for v in rows.values()])
            print(f'gapgate {cond} {a + len(chunk)}/{len(todo)} {time.time() - t0:.1f}s', flush=True)


def stage_gapgate2(rt, out: Path, args) -> None:
    """Follow-up of gapgate on H grids: no decay over the whole blank run plus the boundary token, for the boundary
    heads, for all heads in all 24 DeltaNet layers, and for all heads in layers 9-16 only."""
    root = out / 'gapgate2'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    sel = json.loads((out / 'boundary_heads.json').read_text())
    run = Path(args.ordinal_run)
    heads = {int(l): h for l, h in sel['H']['heads'].items() if h}
    conds_h = [c for c in tuple(args.conditions.split(',')) if c.startswith('H')]
    cases = [c for c in rm.load_ordinal_cases(run, tuple(conds_h))
             if c['variant'] in ('insL', 'delL') and c['id'] not in finished]
    for a in range(0, len(cases), args.gap_chunk):
        t0 = time.time()
        chunk = cases[a:a + args.gap_chunk]
        preps, jobs, meta = [], [], []
        for i, c in enumerate(chunk):
            prep = cr.prepare(rt, ordinal_image(run, c), rm.DIRECT_PROMPT)
            preps.append(prep)
            g = rm.geometry(c['truth'], c['axis'])
            s = prep.span['start']
            blank = [t for b in g.interior_blanks() for o in range(6) for t in g.cell_tokens(b, o)]
            # The blank cells cover the whole blank column only inside the grid; that is the blank run in every token row.
            span = s + np.array(sorted(set(blank) | set(g.boundary_tokens().tolist())))
            conds = [('none', {}),
                     ('nodecay_run|boundary_heads', {'nodecay': {l: (span, h) for l, h in heads.items()}}),
                     ('nodecay_run|all_heads_all_layers', {'nodecay': {l: (span, None) for l in rm.LINEAR_LAYERS}}),
                     ('nodecay_run|all_heads_mid', {'nodecay': {l: (span, None) for l in rm.MID_LINEAR}}),
                     ('nowrite_run|all_heads_all_layers', {'nowrite': {l: (span, None) for l in rm.LINEAR_LAYERS}})]
            for name, ov in conds:
                jobs.append({'prep': i, 'ov': ov})
                meta.append((c, name))
        res = decode_jobs(rt, preps, jobs, args.batch_size, args.max_new_tokens)
        rows = {}
        for (c, name), r in zip(meta, res):
            rec = rows.setdefault(c['id'], {'id': c['id'], 'condition': c['condition'], 'variant': c['variant'],
                                            'c': c['c'], 'truth': {k: c['truth'][k] for k in ('count', 'gap', 'slot')},
                                            'runs': {}})
            f = answer_fields(r['text'], r['finish_reason'])
            f['long'], f['short'] = rm.long_short(c['axis'], None if f['row'] is None else (f['row'], f['col']))
            rec['runs'][name] = f
        append_jsonl(path, [{**v, 'batch_size': args.batch_size} for v in rows.values()])
        print(f'gapgate2 {a + len(chunk)}/{len(cases)} {time.time() - t0:.1f}s', flush=True)


def stage_gapswap(rt, out: Path, args) -> None:
    """H grids: replace the visual embeddings of the cell after the blank line (the boundary cell, every short-axis
    row except the target's) with the embeddings of the same token positions in the base image of the same family,
    where that cell has a figure on its left. Control: the same swap at the matched control cell."""
    root = out / 'gapswap'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    run = Path(args.ordinal_run)
    conds_h = tuple(c for c in args.conditions.split(',') if c.startswith('H'))
    all_cases = rm.load_ordinal_cases(run, conds_h)
    bases = {(c['condition'], c['base']): c for c in all_cases if c['variant'] == 'base'}
    cases = [c for c in all_cases if c['variant'] in ('insL', 'delL') and c['id'] not in finished]
    for a in range(0, len(cases), args.gap_chunk):
        t0 = time.time()
        chunk = cases[a:a + args.gap_chunk]
        preps, jobs, meta = [], [], []
        for c in chunk:
            g = rm.geometry(c['truth'], c['axis'])
            b = bases[(c['condition'], c['base'])]
            prep = cr.prepare(rt, ordinal_image(run, c), rm.DIRECT_PROMPT)
            src = cr.prepare(rt, ordinal_image(run, b), rm.DIRECT_PROMPT)
            if src.vis.shape != prep.vis.shape:
                raise ValueError('base and variant images differ in size')
            line = g.interior_blanks()[0] + 1
            others = [o for o in range(6) if o != g.target[1]]
            def cell_set(ln):
                return sorted(t for o in others for t in g.cell_tokens(ln, o))
            out_rows = {}
            for name, ln in (('swap_boundary_cell', line), ('swap_control_cell', line + 3)):
                if ln > 19 or ln not in g.present:          # base images hold figures on lines 0-19 only
                    continue
                idx = torch.tensor(cell_set(ln), device=prep.vis.device)
                vis = prep.vis.clone()
                vis[idx] = src.vis[idx]
                p2 = cr.Prepared(prep.batch, prep.ids, prep.span, prep.vit_pos, prep.interp, prep.pos, vis)
                out_rows[name] = p2
            for name, pp in [('none', prep)] + list(out_rows.items()):
                preps.append(pp)
                jobs.append({'prep': len(preps) - 1})
                meta.append((c, name))
        res = decode_jobs(rt, preps, jobs, args.batch_size, args.max_new_tokens)
        rows = {}
        for (c, name), r in zip(meta, res):
            rec = rows.setdefault(c['id'], {'id': c['id'], 'condition': c['condition'], 'variant': c['variant'],
                                            'c': c['c'], 'truth': {k: c['truth'][k] for k in ('count', 'gap', 'slot')},
                                            'runs': {}})
            f = answer_fields(r['text'], r['finish_reason'])
            f['long'], f['short'] = rm.long_short(c['axis'], None if f['row'] is None else (f['row'], f['col']))
            rec['runs'][name] = f
        append_jsonl(path, [{**v, 'batch_size': args.batch_size} for v in rows.values()])
        print(f'gapswap {a + len(chunk)}/{len(cases)} {time.time() - t0:.1f}s', flush=True)


# ----------------------------------------------------------------------------- head-class interventions (8x8)

def stage_headclass(rt, out: Path, args) -> None:
    root = out / 'headclass'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    sets = rm.head_sets(Path(args.gate_stats))
    mid = {name: rm.restrict(v, rm.MID_LINEAR) for name, v in sets.items()}
    allh = {l: None for l in rm.MID_LINEAR}
    todo = [c for c in cases8() if c['id'] not in finished]
    for a in range(0, len(todo), args.gap_chunk):
        t0 = time.time()
        chunk = todo[a:a + args.gap_chunk]
        preps, jobs, meta = [], [], []
        for i, c in enumerate(chunk):
            prep = cr.prepare(rt, so.grid_image(c['template'], 8, 8, tuple(c['target']))[0], so.PROMPT)
            preps.append(prep)
            s, e = prep.span['start'], prep.span['end']
            pos = np.arange(s, e)
            conds = [('none', {})]
            for cls in ('integrator', 'row_reset', 'onset', 'other'):
                conds.append((f'nodecay|{cls}', {'nodecay': {l: (pos, h) for l, h in mid[cls].items()}}))
                conds.append((f'nowrite|{cls}', {'nowrite': {l: (pos, h) for l, h in mid[cls].items()}}))
            conds.append(('nodecay|all', {'nodecay': {l: (pos, h) for l, h in allh.items()}}))
            conds.append(('nowrite|all', {'nowrite': {l: (pos, h) for l, h in allh.items()}}))
            for name, ov in conds:
                jobs.append({'prep': i, 'ov': ov})
                meta.append((c, name))
        res = decode_jobs(rt, preps, jobs, args.batch_size, args.max_new_tokens)
        rows = {}
        for (c, name), r in zip(meta, res):
            rows.setdefault(c['id'], {**c, 'runs': {}})['runs'][name] = answer_fields(r['text'], r['finish_reason'])
        append_jsonl(path, [{**v, 'batch_size': args.batch_size,
                             'n_heads': {k: sum(len(h) for h in mid[k].values()) for k in mid}} for v in rows.values()])
        print(f'headclass {a + len(chunk)}/{len(todo)} {time.time() - t0:.1f}s', flush=True)


# ----------------------------------------------------------------------------- implicit counting

def implicit_conditions(s: int, e: int) -> list[tuple[str, dict]]:
    pos = np.arange(s, e)
    allh = {l: (pos, None) for l in rm.MID_LINEAR}
    return [('none', {}), ('nodecay|mid', {'nodecay': allh}), ('nowrite|mid', {'nowrite': allh})]


def stage_implicit(rt, out: Path, args) -> None:
    root = out / 'implicit'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    run = Path(args.ordinal_run)
    items = [{'id': c['id'], 'set': 'g8x8', 'image': ('grid', c['template'], tuple(c['target'])),
              'truth_row': c['target'][0] + 1, 'truth_col': c['target'][1] + 1} for c in cases8()]
    for c in rm.load_ordinal_cases(run, ('V64',)):
        if c['variant'] == 'base':
            g = rm.geometry(c['truth'], 'V')
            items.append({'id': c['id'].split('|')[0], 'set': 'V64', 'image': ('file', c['image']),
                          'truth_row': c['truth']['count'], 'truth_col': g.target[1] + 1})
    modes = [('direct', rm.DIRECT_PROMPT, False, args.max_new_tokens), ('think', rm.THINK_PROMPT, False, args.max_new_think)]
    for mode, prompt, thinking, max_new in modes:
        todo = [it for it in items if f"{it['id']}|{mode}" not in finished]
        for a in range(0, len(todo), args.batch_size):
            t0 = time.time()
            chunk = todo[a:a + args.batch_size]
            preps = []
            for it in chunk:
                im = (so.grid_image(it['image'][1], 8, 8, it['image'][2])[0] if it['image'][0] == 'grid'
                      else Image.open(run / it['image'][1]).convert('RGB'))
                preps.append(cr.prepare(rt, im, prompt, thinking))
            jobs, meta = [], []
            for i, (it, p) in enumerate(zip(chunk, preps)):
                for name, ov in implicit_conditions(p.span['start'], p.span['end']):
                    jobs.append({'prep': i, 'ov': ov})
                    meta.append((it, name))
            res = decode_jobs(rt, preps, jobs, args.batch_size, max_new)
            rows = {}
            for (it, name), r in zip(meta, res):
                rec = rows.setdefault(it['id'], {'id': f"{it['id']}|{mode}", 'item': it['id'], 'set': it['set'],
                                                 'mode': mode, 'truth_row': it['truth_row'],
                                                 'truth_col': it['truth_col'], 'runs': {}})
                rec['runs'][name] = answer_fields(r['text'], r['finish_reason'])
            append_jsonl(path, [{**v, 'batch_size': args.batch_size} for v in rows.values()])
            print(f'implicit {mode} {a + len(chunk)}/{len(todo)} {time.time() - t0:.1f}s', flush=True)
    # Thinking-on comparison: a small subset of 8x8 grids with targets in rows 4-8, none vs no decay.
    rng = np.random.default_rng(args.seed + 9)
    pool = [it for it in items if it['set'] == 'g8x8' and it['truth_row'] >= 4]
    picks = [pool[i] for i in sorted(rng.choice(len(pool), args.n_thinking_on, replace=False))]
    todo = [it for it in picks if f"{it['id']}|thinking_on" not in finished]
    for a in range(0, len(todo), args.batch_size // 2):
        t0 = time.time()
        chunk = todo[a:a + args.batch_size // 2]
        preps = [cr.prepare(rt, so.grid_image(it['image'][1], 8, 8, it['image'][2])[0], rm.DIRECT_PROMPT, True)
                 for it in chunk]
        jobs, meta = [], []
        for i, (it, p) in enumerate(zip(chunk, preps)):
            for name, ov in implicit_conditions(p.span['start'], p.span['end'])[:2]:
                jobs.append({'prep': i, 'ov': ov})
                meta.append((it, name))
        res = decode_jobs(rt, preps, jobs, args.batch_size, args.max_new_thinking_on)
        rows = {}
        for (it, name), r in zip(meta, res):
            rec = rows.setdefault(it['id'], {'id': f"{it['id']}|thinking_on", 'item': it['id'], 'set': it['set'],
                                             'mode': 'thinking_on', 'truth_row': it['truth_row'],
                                             'truth_col': it['truth_col'], 'runs': {}})
            rec['runs'][name] = answer_fields(r['text'], r['finish_reason'])
        append_jsonl(path, [{**v, 'batch_size': args.batch_size} for v in rows.values()])
        print(f'implicit thinking_on {a + len(chunk)}/{len(todo)} {time.time() - t0:.1f}s', flush=True)


# ----------------------------------------------------------------------------- output order

def stage_order(rt, out: Path, args) -> None:
    root = out / 'order'
    (root / 'maps').mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    run = Path(args.ordinal_run)
    items = []
    for c in cases8():
        r, col = c['target']
        items.append({'id': c['id'], 'set': 'g8x8', 'image': ('grid', c['template'], (r, col)),
                      'truth_row': r + 1, 'truth_col': col + 1, 'target_tokens': None})
    for c in rm.load_ordinal_cases(run, ('H64', 'V64', 'H32', 'V32')):
        if c['variant'] != 'base':
            continue
        g = rm.geometry(c['truth'], c['axis'])
        tr, tc = (g.target[1] + 1, c['truth']['count']) if c['axis'] == 'H' else (c['truth']['count'], g.target[1] + 1)
        items.append({'id': c['id'].split('|')[0], 'set': c['condition'], 'image': ('file', c['image']),
                      'truth_row': tr, 'truth_col': tc, 'target_tokens': g.cell_tokens(*g.target)})
    for order in ('direct', 'swapped'):
        prompt = rm.PROMPTS[order]
        todo = [it for it in items if f"{it['id']}|{order}" not in finished]
        for a in range(0, len(todo), args.batch_size):
            t0 = time.time()
            chunk = todo[a:a + args.batch_size]
            preps = []
            for it in chunk:
                im = (so.grid_image(it['image'][1], 8, 8, it['image'][2])[0] if it['image'][0] == 'grid'
                      else Image.open(run / it['image'][1]).convert('RGB'))
                preps.append(cr.prepare(rt, im, prompt))
            res = decode_jobs(rt, preps, [{'prep': i} for i in range(len(chunk))], args.batch_size, args.max_new_tokens)
            recs = []
            for it, prep, r in zip(chunk, preps, res):
                rec = {'id': f"{it['id']}|{order}", 'item': it['id'], 'set': it['set'], 'order': order,
                       'truth_row': it['truth_row'], 'truth_col': it['truth_col'],
                       **answer_fields(r['text'], r['finish_reason']), 'grid': list(prep.grid)}
                T = it['target_tokens'] or so.cell_tokens((it['truth_row'] - 1, it['truth_col'] - 1), prep.grid)
                rec['target_tokens'] = [int(t) for t in T]
                coord = cr.coordinate_tokens(rt, r['ids'], 'rowcol')
                if rec['row'] is None or len(coord) != 2:
                    rec['status'] = 'no_coordinate_tokens'
                    recs.append(rec)
                    continue
                x, pos = cr.teacher_forced(rt, prep, r['ids'][:-1])
                L = len(prep.ids)
                queries = [L + coord[0][0] - 1, L + coord[1][0] - 1]           # predict first digit of row, of column
                s, e = prep.span['start'], prep.span['end']
                ro = cr.attention_readout(rt, x, pos, queries, (s, e), layers=ATTN_LAYERS)
                maps = np.stack([ro.attn[l][:, :, s:e].numpy() for l in ATTN_LAYERS])   # (2, 16, 2, Nv)
                np.save(root / 'maps' / f"{it['id']}_{order}.npy", maps.astype(np.float16))
                rec['status'] = 'ok'
                rec['row_first'] = coord[0][0] < coord[1][0]
                recs.append(rec)
            append_jsonl(path, [{**v, 'batch_size': args.batch_size} for v in recs])
            print(f'order {order} {a + len(chunk)}/{len(todo)} {time.time() - t0:.1f}s', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--ordinal-run', required=True, help='experiment-1.5 run directory (images, cases, answers)')
    p.add_argument('--gate-stats', required=True, help='experiment-3.8 r2 gates/stats.json (head classes)')
    p.add_argument('--stages', default='spotcheck')
    p.add_argument('--conditions', default='H32,H64,V32,V64')
    p.add_argument('--seed', type=int, default=20261020)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--prefill-batch', type=int, default=8)
    p.add_argument('--gap-chunk', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=24)
    p.add_argument('--max-new-think', type=int, default=768)
    p.add_argument('--max-new-thinking-on', type=int, default=2048)
    p.add_argument('--n-thinking-on', type=int, default=8)
    p.add_argument('--n-spot', type=int, default=20)
    args = p.parse_args()
    torch.set_num_threads(8)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    meta_path = out / 'experiment.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {'runs': []}
    meta.update({'runtime': qr.describe(rt), 'prompts': rm.PROMPTS, 'grid_prompt': so.PROMPT})
    meta['runs'].append({'stages': args.stages, 'args': vars(args), 'code_sha256': code_digest(),
                         'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    meta_path.write_text(json.dumps(meta, indent=1))
    stages = {'spotcheck': stage_spotcheck, 'gates': stage_gates, 'gapgate': stage_gapgate, 'gapgate2': stage_gapgate2, 'gapswap': stage_gapswap,
              'headclass': stage_headclass, 'implicit': stage_implicit, 'order': stage_order}
    for name in args.stages.split(','):
        with torch.inference_mode():
            stages[name](rt, out, args)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
