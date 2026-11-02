"""GPU stages of experiment 3.14 (causal mask vs linear attention). Resumable JSONL records.

Every stage works on one model and one "mode" tag:
  Qwen3-VL-8B:  ds_on (native DeepStack) or ds_off (no DeepStack injection);
  Qwen3.5-9B:   none, freeze, nodecay, nowrite (prefill gate edits of all 24 DeltaNet layers on the image span).

Stages:
  gen        greedy answers on the 3.11 cases (10x10 grid row/col and column/row, 10x10 point, disc point,
             RefCOCO point) under the mode; unpadded equal-length batches.
  spotcheck  Qwen3-VL ds_on only: the inputs_embeds decoder with injected DeepStack against the native
             qwen_runtime.generate (pixel_values path).
  kv         paired image-KV subset patching (true move and decoder-only splice) as in 3.11, over every
             full-attention layer of the model, and the attention mass of every head on token groups at the
             query predicting each value's first digit.
  ordinal    the 1.5 slot grids: greedy answers plus residual means of the 21 cells on the target's
             long-axis line at chosen layers (running-count probe), under the mode or under a full-attention
             knockout (Qwen3.5 only: --knockouts).

Usage: python -m analysis_tools.position.causal_vs_linear_run --model ... --mode ... --stages gen kv ...
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import causal_vs_linear as cv
from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import rowcol_mechanism as rm
from analysis_tools.position.coordinate_readout_supp import refcoco_swaps

sr = cf.sr
read_jsonl, append_jsonl, done_ids = sr.read_jsonl, sr.append_jsonl, sr.done_ids
KV_FULL = ('t6g10_rowcol', 't6g10_point', 'disc_point', 'rc_point')
ORD_LAYERS = {'qwen3_5': (8, 12, 16, 20), 'qwen3_vl': (9, 18, 27)}


def code_digest() -> dict:
    here = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob('causal_vs_linear*.py'))}


def tag_of(args) -> str:
    return f"{'q35' if 'Qwen3.5' in args.model else 'vl'}_{args.mode}"


def ds_on(rt, args) -> bool:
    return rt.family == 'qwen3_vl' and args.mode == 'ds_on'


def mode_context(rt, args, span, stack: ExitStack, positions: list[list[int]] | None = None):
    """Enter the gate edit of the mode (Qwen3.5) on the image span (or on per-row target positions)."""
    if rt.family == 'qwen3_5':
        if not cv.parse_mode(args.mode)[2]:
            positions = None
        stack.enter_context(cv.gate_override(rt, args.mode, span, positions=positions))


# ----------------------------------------------------------------------------- decoding

def decode(rt, preps: list, args, knock: tuple = (), kind: str = 'img', hooks=None,
           targets: list[list[int]] | None = None) -> list[dict]:
    """Greedy answers of equal-length prompts in batches, under the mode (and an optional knockout).

    targets: per prompt, absolute positions of the target tokens (used by target-only gate modes).
    """
    out = []
    for a in range(0, len(preps), args.batch_size):
        chunk = preps[a:a + args.batch_size]
        tchunk = None if targets is None else targets[a:a + args.batch_size]
        span = (chunk[0].span['start'], chunk[0].span['end'])
        x = torch.cat([cr.embeds(rt, p) for p in chunk], 0)
        pos = torch.cat([p.pos for p in chunk], 1)
        with ExitStack() as st:
            mode_context(rt, args, span, st, tchunk)
            st.enter_context(cv.deepstack_rows(rt, [p.meta['ds'] for p in chunk] if ds_on(rt, args) else None, span))
            st.enter_context(cv.attention_knockout(rt, knock, span, chunk[0].grid, kind))
            if hooks is not None:
                st.enter_context(hooks(len(chunk[0].ids), span))
            res = cr.greedy(rt, x, pos, args.max_new_tokens)
        for r in res:
            r['batch_size'] = len(chunk)
        out += res
    return out


def group_by_length(items: list, preps: list) -> list[list[int]]:
    by = {}
    for i, p in enumerate(preps):
        by.setdefault((len(p.ids), p.span['start'], p.span['end']), []).append(i)
    return list(by.values())


# ----------------------------------------------------------------------------- gen / spotcheck

def stage_gen(rt, out: Path, args) -> None:
    path = out / f'gen_{tag_of(args)}.jsonl'
    done = done_ids(path)
    cases = cf.load_cases(Path(args.flow_run))
    todo = [c for c in cases.values() if c['cond'] in args.conditions and c['id'] not in done]
    for a in range(0, len(todo), 64):
        chunk = todo[a:a + 64]
        preps = [cv.prepare(rt, sr.load_image(c), c['prompt']) for c in chunk]
        rows = []
        for idx in group_by_length(chunk, preps):
            tg = [[preps[i].span['start'] + t for t in cr.tokens_overlapping(chunk[i]['truth']['box'], preps[i].grid)]
                  for i in idx]
            res = decode(rt, [preps[i] for i in idx], args, targets=tg)
            for i, r in zip(idx, res):
                c = chunk[i]
                vals = cf.answer_values(rt, r['ids'], c['format'])
                rows.append({'id': c['id'], 'cond': c['cond'], 'mode': args.mode, 'text': r['text'], 'ids': r['ids'],
                             'finish_reason': r['finish_reason'], 'batch_size': r['batch_size'],
                             'values': vals, 'score': cf.score_values(c, vals)})
        append_jsonl(path, rows)
        print(f'gen {tag_of(args)} {a + len(chunk)}/{len(todo)}', flush=True)
        del preps
        torch.cuda.empty_cache()


def stage_spotcheck(rt, out: Path, args) -> None:
    """Answer agreement of the inputs_embeds decoder (with injected DeepStack) and the native generate."""
    path = out / f'spotcheck_{tag_of(args)}.jsonl'
    done = done_ids(path)
    cases = cf.load_cases(Path(args.flow_run))
    gen = read_jsonl(out / f'gen_{tag_of(args)}.jsonl')
    rng = np.random.default_rng(args.seed)
    for k in sorted(rng.choice(len(gen), min(args.spotcheck, len(gen)), replace=False)):
        g = gen[k]
        if g['id'] in done:
            continue
        c = cases[g['id']]
        ref = qr.generate(rt, qr.build(rt, [sr.load_image(c)], c['prompt']), args.max_new_tokens)
        ids = rt.tokenizer.encode(ref['text'], add_special_tokens=False)
        vr = cf.answer_values(rt, ids, c['format'])
        val = lambda v: None if v is None else [(d['key'], d['value']) for d in v]
        append_jsonl(path, [{'id': g['id'], 'reference': ref['text'], 'manual': g['text'],
                             'text_equal': ref['text'].strip() == g['text'].strip(),
                             'answer_equal': val(vr) == val(g['values'])}])


# ----------------------------------------------------------------------------- kv

def pair_inputs(rt, case: dict, prep, splice: bool, use_ds: bool):
    """Run-B inputs for each feasible displacement (true move, optional splice) or RefCOCO content swap."""
    xa = cr.embeds(rt, prep)
    dsa = prep.meta.get('ds') if use_ds else None
    target = cr.tokens_overlapping(case['truth']['box'], prep.grid)
    out = []
    if case['set'] in ('disc', 't6'):
        for mv in sr.moves(case, prep.grid, target, 2, 4):
            pb = cv.prepare(rt, sr.run_b_image(case, mv), case['prompt'])
            if len(pb.ids) != len(prep.ids):
                raise ValueError(f"Run B length differs for {case['id']}")
            xb = cr.embeds(rt, pb)
            dsb = pb.meta.get('ds') if use_ds else None
            tag = f"{mv['axis']}{mv['sign']:+d}"
            out.append({'tag': tag, 'kind': 'move', 'axis': mv['axis'], 'xb': xb, 'dsb': dsb, 'target': target,
                        'moved': mv['shifted']})
            if splice:
                D = sorted(set(target) | set(mv['shifted']))
                xs = xa.clone()
                ab = torch.tensor(D) + prep.span['start']
                xs[0, ab] = xb[0, ab]
                dss = None if dsa is None else [cv.splice_rows(a, b, D) for a, b in zip(dsa, dsb)]
                out.append({'tag': tag + '_splice', 'kind': 'splice', 'axis': mv['axis'], 'xb': xs, 'dsb': dss,
                            'target': target, 'moved': mv['shifted']})
    else:
        for reg in refcoco_swaps(target, prep.grid):
            if not reg['valid']:
                continue
            order = cr.swap_index(prep.grid[0] * prep.grid[1], target, reg['region'])
            dsb = None if dsa is None else [d[order.to(d.device)] for d in dsa]
            out.append({'tag': f"{reg['axis']}swap", 'kind': 'swap', 'axis': reg['axis'],
                        'xb': cr.embeds(rt, prep, order=order), 'dsb': dsb, 'target': target, 'moved': reg['region']})
    return xa, dsa, out


def attention_summary(rt, xa_tf, pos, queries: list[int], grid, target, span, ds) -> tuple[np.ndarray, np.ndarray]:
    """Per layer, head and query: mass on the profile groups (last column = text); head-mean image maps."""
    s, e = span
    rows = cv.attention_rows(rt, xa_tf, pos, queries, ds=ds, span=span)
    masks = cv.profile_masks(grid, target)
    names = list(masks)
    mass, maps = [], []
    for l in cv.full_layers(rt):
        a = rows[l]                                                  # (heads, Q, L)
        img = a[:, :, s:e].numpy()
        m = [img[:, :, masks[n]].sum(-1) for n in names] + [a[:, :, :s].sum(-1).numpy() + a[:, :, e:].sum(-1).numpy()]
        mass.append(np.stack(m, -1))
        maps.append(img.mean(0))
    return np.stack(mass).astype(np.float32), np.stack(maps).astype(np.float16)


@torch.inference_mode()
def stage_kv(rt, out: Path, args) -> None:
    tag = tag_of(args)
    path = out / f'kv_{tag}.jsonl'
    adir = out / f'kv_attn_{tag}'
    adir.mkdir(exist_ok=True)
    cases = cf.load_cases(Path(args.flow_run))
    gen = read_jsonl(out / f'gen_{tag}.jsonl')
    full = [c for c in args.conditions if c in KV_FULL]
    todo = cf.limit_per_condition(gen, full, args.n_kv) + \
        cf.limit_per_condition(gen, [c for c in args.conditions if c not in KV_FULL], args.n_kv_order)
    digits = cf.digit_ids(rt)
    use_ds = ds_on(rt, args)
    for g in todo:
        if g['id'] in done_ids(path) or not g['score']['valid']:
            continue
        c = cases[g['id']]
        t0 = time.time()
        prep = cv.prepare(rt, sr.load_image(c), c['prompt'])
        L = len(prep.ids)
        s, e = prep.span['start'], prep.span['end']
        ans = g['ids']
        splice = c['cond'] in KV_FULL
        is_full = splice and not args.reduced
        target = cr.tokens_overlapping(c['truth']['box'], prep.grid)
        xa, dsa, pairs = pair_inputs(rt, c, prep, splice=splice, use_ds=use_ds)
        xa_tf, pos = cr.teacher_forced(rt, prep, ans[:-1], x=xa)
        queries = cf.first_digit_queries(g['values'], L)
        with ExitStack() as st:
            mode_context(rt, args, (s, e), st, [[s + t for t in target]])
            mass, maps = attention_summary(rt, xa_tf, pos, queries, prep.grid, target, (s, e), dsa)
            np.savez_compressed(adir / f"{g['id']}.npz", mass=mass, maps=maps,
                                groups=np.array(list(cv.profile_masks(prep.grid, target)) + ['text']))
        recs = []
        for pr in pairs:
            # target-only modes edit the changed tokens D (the target in run A and in run B) in every row
            D = [s + t for t in sorted(set(pr['target']) | set(pr['moved']))]
            with ExitStack() as st:
                mode_context(rt, args, (s, e), st, [D])
                xb_tf, _ = cr.teacher_forced(rt, prep, ans[:-1], x=pr['xb'])
                conds = cv.kv_conditions(rt, prep.grid, pr['target'], pr['moved'], s, e, L, full=is_full)
                chunks = []
                for i in range(0, len(conds), args.kv_rows):
                    part = conds[i:i + args.kv_rows]
                    lg = cv.tf_kv_logits(rt, xb_tf, xa_tf, pos, L, [(pp, ll) for _, pp, ll in part], (s, e),
                                         ds_b=pr['dsb'], ds_a=dsa)
                    chunks.append(cf.ld_table(lg, ans, g['values'], digits, b_row=0))
                recs.append({'tag': pr['tag'], 'kind': pr['kind'], 'axis': pr['axis'], 'target': pr['target'],
                             'moved': pr['moved'], 'values': cf._merge(chunks, [n for n, _, _ in conds], args.kv_rows)})
        append_jsonl(path, [{'id': g['id'], 'cond': c['cond'], 'mode': args.mode, 'correct': g['score']['correct'],
                             'grid': list(prep.grid), 'span': [s, e], 'prompt_len': L, 'values_meta': g['values'],
                             'queries': queries, 'pairs': recs, 'seconds': round(time.time() - t0, 1)}])
        print(f"kv {tag} {g['id']} {len(pairs)} pairs {time.time() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- ordinal slot grids

def line_cells(geo: rm.SlotGeometry) -> list[list[int]]:
    """Token lists of the 21 cells on the target's long-axis line."""
    return [geo.cell_tokens(slot, geo.target[1]) for slot in range(21)]


def stage_ordinal(rt, out: Path, args) -> None:
    run = Path(args.ordinal_run)
    knocks = args.knockouts or ['none']
    layers = ORD_LAYERS[rt.family]
    cases = rm.load_ordinal_cases(run, tuple(args.ordinal_conditions))
    cases = [c for c in cases if c['variant'] in args.ordinal_variants]
    for kn in knocks:
        name = f'{tag_of(args)}_{kn}'
        path = out / 'ordinal' / f'{name}.jsonl'
        fdir = out / 'ordinal' / name
        fdir.mkdir(parents=True, exist_ok=True)
        done = done_ids(path)
        if kn == 'none':
            knock, kind = (), 'img'
        else:
            kind, spec = kn.split(':')                 # e.g. img:4 / img:4,8,12,16 / local:4,8,12,16
            knock = tuple(int(v) for v in spec.split(','))
        for cond in args.ordinal_conditions:
            todo = [c for c in cases if c['condition'] == cond and c['id'] not in done]
            for a in range(0, len(todo), args.batch_size):
                t0 = time.time()
                chunk = todo[a:a + args.batch_size]
                preps = [cv.prepare(rt, Image.open(run / c['image']).convert('RGB'), rm.DIRECT_PROMPT) for c in chunk]
                geos = [rm.geometry(c['truth'], c['axis']) for c in chunk]
                L = len(preps[0].ids)
                s, e = preps[0].span['start'], preps[0].span['end']
                feats = {l: [] for l in layers}

                def hooks(L_, span, feats=feats, geos=geos):
                    from contextlib import contextmanager

                    @contextmanager
                    def cm():
                        hs = []
                        for l in layers:
                            def hook(m, i, o, l=l):
                                h = o[0] if isinstance(o, tuple) else o
                                if h.shape[1] != L_ or len(feats[l]) >= len(geos):
                                    return
                                vis = h[:, s:e].float()
                                for b, g in enumerate(geos):
                                    idx = torch.tensor(line_cells(g), device=vis.device)
                                    feats[l].append(vis[b][idx].mean(1).half().cpu())
                            hs.append(rt.layers[l - 1].register_forward_hook(hook))
                        try:
                            yield
                        finally:
                            for h in hs:
                                h.remove()
                    return cm()
                res = decode(rt, preps, args, knock=knock, kind=kind, hooks=hooks)
                rows = []
                for b, (c, r) in enumerate(zip(chunk, res)):
                    rc = rm.parse_rowcol_any(r['text']) if r['finish_reason'] == 'eos' else None
                    lo, sh = rm.long_short(c['axis'], rc)
                    fname = c['id'].split('|')[0] + '.pt'
                    torch.save(torch.stack([feats[l][b] for l in layers]), fdir / fname)
                    rows.append({'id': c['id'], 'condition': cond, 'variant': c['variant'], 'base': c['base'],
                                 'knockout': kn, 'mode': args.mode, 'text': r['text'], 'finish_reason': r['finish_reason'],
                                 'answer_long': lo, 'answer_short': sh, 'file': fname, 'layers': list(layers),
                                 'batch_size': r['batch_size']})
                append_jsonl(path, rows)
                print(f'ordinal {name} {cond} {a + len(chunk)}/{len(todo)} {time.time() - t0:.1f}s', flush=True)


# ----------------------------------------------------------------------------- main

def use_torch_deltanet() -> None:
    """Route the Qwen3.5 chunked Gated DeltaNet prefill to the transformers pure-torch implementation.

    The hub/FLA kernel needs more shared memory than A40 GPUs provide; the torch path computes the same
    recurrence (in float32). Runs that use it record the flag in run_log.jsonl.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    fn = m.torch_chunk_gated_delta_rule
    m.torch_chunk_gated_delta_rule = getattr(fn, '__wrapped__', fn)

STAGES = {'gen': stage_gen, 'spotcheck': stage_spotcheck, 'kv': stage_kv, 'ordinal': stage_ordinal}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stages', nargs='+', required=True, choices=list(STAGES))
    p.add_argument('--model', required=True)
    p.add_argument('--mode', required=True, help='ds_on / ds_off (Qwen3-VL) or none / freeze / nodecay / nowrite (Qwen3.5)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--flow-run', type=Path, required=True, help='3.11 run directory (cases.json)')
    p.add_argument('--ordinal-run', type=Path, required=True, help='1.5 run directory (slot grids)')
    p.add_argument('--conditions', nargs='+', default=list(KV_FULL) + ['t6g10_colrow'])
    p.add_argument('--n-kv', type=int, default=24)
    p.add_argument('--n-kv-order', type=int, default=20)
    p.add_argument('--reduced', action='store_true', help='main KV groups only (splice pairs are kept)')
    p.add_argument('--ordinal-conditions', nargs='+', default=['H32', 'H64', 'V32', 'V64'])
    p.add_argument('--ordinal-variants', nargs='+', default=['base', 'shift', 'insL', 'delL', 'delR'])
    p.add_argument('--knockouts', nargs='*', default=[], help="none, img:4, img:4,8,12,16, local:4,8,12,16 ...")
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--kv-rows', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--max-memory', default='', help="split the model over visible GPUs, e.g. '0:10GiB,1:14GiB'")
    p.add_argument('--torch-deltanet', action='store_true',
                   help='use the pure-torch Gated DeltaNet prefill (GPUs whose shared memory is too small for the FLA kernel)')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.torch_deltanet:
        use_torch_deltanet()
    rt = cf.load_runtime(args.model, args.max_memory)
    if rt.family == 'qwen3_vl' and args.mode not in ('ds_on', 'ds_off'):
        raise ValueError('Qwen3-VL modes: ds_on, ds_off')
    if rt.family == 'qwen3_5' and args.mode not in cv.GATE_MODES:
        raise ValueError(f'Qwen3.5 modes: {cv.GATE_MODES}')
    for st in args.stages:
        append_jsonl(args.output / 'run_log.jsonl', [{
            'stage': st, 'tag': tag_of(args), 'args': {k: str(v) for k, v in vars(args).items()},
            'runtime': qr.describe(rt), 'code_sha256': code_digest(), 'device': torch.cuda.get_device_name(0),
            'started': datetime.now(timezone.utc).isoformat()}])
        STAGES[st](rt, args.output, args)


if __name__ == '__main__':
    main()
