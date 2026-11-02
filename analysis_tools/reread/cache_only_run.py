"""Experiment 5.3 runner: cache-only re-read (no new ViT forward) vs the native crop-and-re-encode zoom tool.

Stages (comma list):
  verify          merger re-implementation vs the model's visual tokens, and padded batch vs batch size 1;
  lean_<bench>    accuracy of every condition under the lean tool protocol (single forced call, prefilled
                  answer marker) on one benchmark: textvqa, gqa, mmvp, pope, syn_color, syn_gap, syn_loc;
  native_<bench>  the new cache-only conditions under experiment 5.2's native multi-round protocol
                  (executed by cached_reread_run.run_bench), for comparison with 5.2's crop results;
  timing          batch-size-1 lean pipeline with KV-cache continuation, timed per stage, with peak memory.

Benchmarks, case sampling and scoring are those of experiment 5.2 (same seed, hence the same questions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.reread import cache_only as co
from analysis_tools.reread import cached_reread as cr
from analysis_tools.reread import cached_reread_run as crr
from analysis_tools.reread import reread_pilot as rp

LEAN_CONDS = ['base', 'full2x', 'crop1', 'crop2', 'c27', 'c26', 'c25', 's27', 's26', 's25', 'u27', 'b27',
              'c27n', 's27n', 'h27']
LEAN_ORACLE = ['o_crop2', 'o_c27', 'o_s27']
NATIVE_CONDS = ['s27', 'u27', 'c26', 'c25']
TIMING_CONDS = ['base', 'crop1', 'crop2', 'c27', 's27', 'u27', 'c25', 's25', 'full2x', 'h27']


# ----------------------------------------------------------------------------- features and tool output

def features(rt, image: Image.Image, conds: list[co.Condition]) -> dict:
    """The single ViT pass (plus the 2x pass only when the reference condition h27 is requested)."""
    out = {'f1': co.encode(rt, image, co.cache_blocks(conds))}
    if any(c.kind == 'cache' and c.scale == 2 for c in conds):
        out['f2'] = rp.encode(rt, [cr.upscale(image, 2)])[0]
    return out


def second_segment(rt, cond: co.Condition, image: Image.Image, feats: dict, win: rp.Window) -> tuple[cr.VSeg, dict]:
    """Tool output for one window and its cost: new ViT patches, ViT FLOPs, merger FLOPs."""
    vcfg = rt.model.config.vision_config
    if cond.kind == 'crop':
        k = cr.crop_scale(win, cond.scale)
        e = rp.encode(rt, [cr.upscale(rp.crop(image, win), k)])[0]
        n = 4 * e['grid'][0] * e['grid'][1]
        return cr.VSeg(e['vis'], e['grid'], 'new'), {'crop_scale': k, 'vit_patches': n,
                                                      'vit_flops': co.vit_flops(n, vcfg), 'merger_flops': 0.0}
    if cond.scale == 2:                                   # reference: slice of the 2x-image cache (5.2's h27)
        f = feats['f2']
        wh = rp.Window(win.r0 * 2, win.c0 * 2, win.h * 2, win.w * 2)
        idx = torch.tensor(wh.indices(f['grid'][1]), device=f['vis'].device)
        seg = cr.VSeg(f['vis'][idx], (wh.h, wh.w), cond.ids, (float(win.r0), float(win.c0)), 0.5)
        return seg, {'crop_scale': None, 'vit_patches': 0, 'vit_flops': 0.0, 'merger_flops': 0.0}
    capped = cond.merge != 'grid' and cr.crop_scale(win, 2) < 2
    if capped:
        # The native tool shrinks its upscaling factor when the 2x crop exceeds CROP_MAX_PIXELS; the 4x-token
        # readers obey the same cap and return the standard merge (K tokens) for such windows.
        cond = co.Condition(cond.name, 'cache', 1, cond.site, cond.ids, cond.oracle, block=cond.block, merge='grid')
    seg = co.cache_segment(rt, cond, feats['f1'], win)
    reuse = cond.block == co.LAST_BLOCK and cond.merge == 'grid'
    mf = 0.0 if reuse else co.merger_flops(seg.grid[0] * seg.grid[1], vcfg)
    return seg, {'crop_scale': None, 'vit_patches': 0, 'vit_flops': 0.0, 'merger_flops': mf, 'capped': capped}


def base_vit(rt, grid: tuple[int, int], cond: co.Condition) -> tuple[int, float]:
    """Patches and FLOPs of the window-independent ViT passes of a condition (full image, plus 2x if any)."""
    vcfg = rt.model.config.vision_config
    n1 = 4 * grid[0] * grid[1]
    if cond.kind == 'full2x':
        return 4 * n1, co.vit_flops(4 * n1, vcfg)
    if cond.kind == 'cache' and cond.scale == 2:
        return n1 + 4 * n1, co.vit_flops(n1, vcfg) + co.vit_flops(4 * n1, vcfg)
    return n1, co.vit_flops(n1, vcfg)


def oracle_call_ids(rt, box: list[float]) -> list[int]:
    text = co.CALL_PREFIX + ', '.join(str(round(v)) for v in box) + ']\n</parameter>\n</function>\n</tool_call>'
    return rt.tokenizer(text, add_special_tokens=False)['input_ids']


# ----------------------------------------------------------------------------- lean accuracy stage

def run_lean(rt, bench: str, out: Path, args, closed: bool = False) -> None:
    """Lean protocol: the call turn is prefilled up to the bbox value (no reasoning before the call), the tool
    is executed once, and the answer turn is prefilled with "Final answer:". base and full2x get the same
    answer prefill. The call is generated once per case and shared by all non-oracle tool conditions.

    closed=True (stage lean2_<bench>): the call is decoded only up to the closing ']' of the bbox and closed with
    the fixed tail of the call format, so no label text is decoded and every call is well formed."""
    tag = 'lean2' if closed else 'lean'
    root = out / f'{tag}_{bench}'
    root.mkdir(parents=True, exist_ok=True)
    cases = crr.load_cases(bench, args)
    names = LEAN_CONDS + (LEAN_ORACLE if bench in crr.ORACLE_BENCHES else [])
    if args.conds:
        names = [c for c in args.conds if c in co.CONDITIONS and (bench in crr.ORACLE_BENCHES or not co.CONDITIONS[c].oracle)]
    conds = [co.CONDITIONS[k] for k in names]
    path = root / 'results.jsonl'
    done = {(r['case_id'], r['cond']) for r in crr.read_jsonl(path)}
    yn = (qr.token_id(rt, ' Yes'), qr.token_id(rt, ' No')) if bench in ('syn_color', 'syn_gap') else None
    stop_tool, stop_ans = cr.stop_ids(rt, True), cr.stop_ids(rt, False)
    pre_call, pre_ans = co.lean_call_prefix_ids(rt), co.answer_prefix_ids(rt)
    t_start = time.time()
    for ci in range(0, len(cases), args.chunk):
        chunk = [c for c in cases[ci:ci + args.chunk] if any((c.id, k) not in done for k in names)]
        if not chunk:
            continue
        images = [crr.load_image(c, args) for c in chunk]
        feats = [features(rt, im, conds) for im in images]
        s1 = [cr.stage1_ids(rt, c.question) + pre_call for c in chunk]
        calls = [None] * len(chunk)
        if any(c.kind in ('crop', 'cache') and not c.oracle for c in conds):
            items1 = [cr.assemble(rt, ids, [cr.VSeg(f['f1']['vis'], f['f1']['grid'])]) for ids, f in zip(s1, feats)]
            stops1 = (stop_tool | co.bracket_stop_ids(rt)) if closed else stop_tool
            calls = cr.run_batched(rt, items1, args.max_call, stops1, args.token_budget, args.max_batch)
            if closed:
                for cl in calls:
                    cl['ids'], cl['text'] = co.close_call(rt, cl['ids'])
            del items1
        jobs, rows = [], []
        for j, (c, im, f) in enumerate(zip(chunk, images, feats)):
            g = f['f1']['grid']
            for cond in conds:
                if (c.id, cond.name) in done:
                    continue
                n_p, vf = base_vit(rt, g, cond)
                row = {'case_id': c.id, 'bench': bench, 'grid': list(g), 'cond': cond.name, 'protocol': tag,
                       'vit_patches': n_p, 'vit_flops': vf, 'merger_flops': 0.0, 'n_calls': 0}
                if cond.kind in ('base', 'full2x'):
                    seg = cr.VSeg(f['f1']['vis'], g)
                    if cond.kind == 'full2x':
                        e = rp.encode(rt, [cr.upscale(im, 2)])[0]
                        seg = cr.VSeg(e['vis'], e['grid'])
                    ids, segs = cr.plain_ids(rt, c.question) + pre_ans, [seg]
                else:
                    if cond.oracle:
                        box = crr.oracle_window_box(c)
                        call_ids, call_text = oracle_call_ids(rt, box), None
                    else:
                        call_ids, call_text = pre_call + calls[j]['ids'], co.CALL_PREFIX + calls[j]['text']
                        box = cr.parse_call(call_text)
                    row['call'] = call_text
                    if box is None:
                        # The forced prefix makes this rare; a malformed call is scored as a wrong answer.
                        row.update(text='', finish_reason='bad_call', no_call=True, L=0, n_extra_visual=0,
                                   **crr.score(c, '', 'bad_call', None))
                        rows.append(row)
                        continue
                    win = cr.box_to_window(box, g)
                    seg, cost = second_segment(rt, cond, im, f, win)
                    ids = s1[j][:-len(pre_call)] + call_ids + cr.suffix_ids(rt, win.norm_box(g), c.fmt) + pre_ans
                    segs = [cr.VSeg(f['f1']['vis'], g), seg]
                    row.update(n_calls=1, bbox=box, window=win.as_list(), window_box=win.norm_box(g),
                               coverage=crr.coverage(win, g, crr.evidence_box(c)), crop_scale=cost['crop_scale'],
                               seg_grid=list(seg.grid), capped=cost.get('capped', False))
                    row['vit_patches'] += cost['vit_patches']
                    row['vit_flops'] += cost['vit_flops']
                    row['merger_flops'] = cost['merger_flops']
                it = cr.assemble(rt, ids, segs)
                it['x'] = it['x'].cpu()                # assembled prompts wait on the CPU until their batch
                row.update(L=it['L'], n_extra_visual=it['n_extra_visual'])
                jobs.append((row, it, c))
        if jobs:
            res = cr.run_batched(rt, [it for _, it, _ in jobs], args.max_new, stop_ans, args.token_budget,
                                 args.max_batch, yn)
            for (row, it, c), r in zip(jobs, res):
                text = co.ANSWER_PREFIX + ' ' + r['text']
                row.update(text=r['text'], finish_reason=r['finish_reason'], n_answer_tokens=r['n_tokens'],
                           batch_size=r['batch_size'], **crr.score(c, text, r['finish_reason'], r.get('s')))
                if c.bench == 'syn_loc' and row.get('window'):
                    row['frame_error'] = crr.frame_error(c, row.get('pred'), rp.Window(*row['window']), tuple(row['grid']))
                if r.get('s') is not None:
                    row['s'] = r['s']
                rows.append(row)
        crr.append_jsonl(path, rows)
        del feats, jobs
        torch.cuda.empty_cache()
        print(f'[{tag} {bench}] {min(ci + args.chunk, len(cases))}/{len(cases)} cases, {time.time() - t_start:.0f}s',
              flush=True)
    crr.write_meta(root, rt, args, f'{tag}_{bench}', {'conditions': names, 'n_cases': len(cases),
                                                     'token_budget': args.token_budget, 'max_batch': args.max_batch,
                                                     'code_5_3': code_digest()})


def code_digest() -> dict:
    import hashlib
    files = ['analysis_tools/reread/cache_only.py', 'analysis_tools/reread/cache_only_run.py']
    return {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16] for f in files}


# ----------------------------------------------------------------------------- native protocol (5.2 engine)

def run_native(rt, bench: str, out: Path, args) -> None:
    """New cache-only conditions under 5.2's native multi-round protocol.

    cached_reread_run.run_bench looks up conditions in cached_reread.CONDITIONS and calls its module-level
    needs / features / second_segment. The 5.3 conditions are duck-type compatible with 5.2's Condition; they
    are registered under their own names (which do not collide with 5.2's) and the three hooks are pointed at
    the cache-only readers for the duration of this call, then restored.
    """
    names = args.conds or NATIVE_CONDS
    clash = [k for k in names if k in cr.CONDITIONS and cr.CONDITIONS[k] != co.CONDITIONS.get(k)]
    saved = (crr.needs, crr.features, crr.second_segment, dict(cr.CONDITIONS))
    try:
        for k in names:
            if k not in clash:
                cr.CONDITIONS[k] = co.CONDITIONS[k]
        crr.needs = lambda ks: {'conds': [cr.CONDITIONS[k] for k in ks]}
        crr.features = lambda rt_, im, need: features(rt_, im, [_as53(c) for c in need['conds']])
        crr.second_segment = lambda rt_, cond, im, f, win: second_segment(rt_, _as53(cond), im, f, win)
        sub = argparse.Namespace(**{**vars(args), 'conds': names})
        crr.run_bench(rt, bench, out / 'native', sub)
    finally:
        crr.needs, crr.features, crr.second_segment = saved[:3]
        cr.CONDITIONS.clear()
        cr.CONDITIONS.update(saved[3])


def _as53(cond) -> co.Condition:
    if isinstance(cond, co.Condition):
        return cond
    if cond.kind == 'crop':
        return co.Condition(cond.name, 'crop', cond.scale)
    raise ValueError(f'Condition {cond.name!r} is not a 5.3 condition.')


# ----------------------------------------------------------------------------- timing (batch size 1)

def _sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def timed_lean(rt, case: cr.Case, image: Image.Image, cond: co.Condition, args, closed: bool = False) -> dict:
    """End-to-end lean pipeline at batch size 1 with KV-cache continuation, timed per stage.

    Stages: vit (window-independent ViT passes, with the cache hooks), call (first prefill and the forced
    call decoding), tool (crop re-encoding or cache read with merger), answer (continuation prefill and
    answer decoding). base and full2x have only vit and answer.
    """
    stop_tool, stop_ans = cr.stop_ids(rt, True), cr.stop_ids(rt, False)
    pre_call, pre_ans = co.lean_call_prefix_ids(rt), co.answer_prefix_ids(rt)
    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t = {'t0': _sync_time()}
    rec = {'case_id': case.id, 'bench': case.bench, 'cond': cond.name}
    if cond.kind in ('base', 'full2x'):
        e = rp.encode(rt, [cr.upscale(image, 2 if cond.kind == 'full2x' else 1)])[0]
        t['vit'] = _sync_time()
        it = cr.assemble(rt, cr.plain_ids(rt, case.question) + pre_ans, [cr.VSeg(e['vis'], e['grid'])])
        gen, cache = crr.greedy_cached(rt, it['x'], it['pos'], None, args.max_new, stop_ans)
        t['call'] = t['tool'] = t['vit']
        t['answer'] = _sync_time()
        L, extra = it['L'] + len(gen), 0
    else:
        f = features(rt, image, [cond])
        g = f['f1']['grid']
        t['vit'] = _sync_time()
        ids = cr.stage1_ids(rt, case.question) + pre_call
        it = cr.assemble(rt, ids, [cr.VSeg(f['f1']['vis'], g)])
        gen, cache = crr.greedy_cached(rt, it['x'], it['pos'], None, args.max_call,
                                       (stop_tool | co.bracket_stop_ids(rt)) if closed else stop_tool)
        n_dec = len(gen)                         # decoded tokens; all but the last are in the KV cache
        if closed:
            gen, _ = co.close_call(rt, gen)        # the tail tokens are fed with the tool response below
        t['call'] = _sync_time()
        box = cr.parse_call(co.CALL_PREFIX + rt.tokenizer.decode(gen))
        if box is None:
            raise RuntimeError(f'Unparsable forced call for {case.id}.')
        fed = it['L'] + n_dec - 1
        win = cr.box_to_window(box, g)
        seg, _ = second_segment(rt, cond, image, f, win)
        t['tool'] = _sync_time()
        ids = ids + gen + cr.suffix_ids(rt, win.norm_box(g), case.fmt) + pre_ans
        it = cr.assemble(rt, ids, [cr.VSeg(f['f1']['vis'], g), seg])
        extra = it['n_extra_visual']
        ans, cache = crr.greedy_cached(rt, it['x'][fed:], it['pos'][:, fed:], cache, args.max_new, stop_ans)
        t['answer'] = _sync_time()
        L = it['L'] + len(ans)
    rec.update(seconds=t['answer'] - t['t0'], t_vit=t['vit'] - t['t0'], t_call=t['call'] - t['vit'],
               t_tool=t['tool'] - t['call'], t_answer=t['answer'] - t['tool'],
               peak_extra_bytes=torch.cuda.max_memory_allocated() - base_mem, L_final=L, n_extra_visual=extra,
               **cr.kv_bytes(rt.model.config, L))
    del cache
    return rec


def run_timing(rt, out: Path, args, closed: bool = False) -> None:
    root = out / ('timing2' if closed else 'timing')
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'timing.jsonl'
    done = {(r['case_id'], r['cond']) for r in crr.read_jsonl(path)}
    conds = [co.CONDITIONS[k] for k in (args.conds or TIMING_CONDS)]
    warm = True
    for bench in args.timing_benches:
        for c in crr.load_cases(bench, args)[:args.timing_n]:
            im = crr.load_image(c, args)
            if warm:                                       # untimed warm-up of every code path
                for cond in conds:
                    timed_lean(rt, c, im, cond, args, closed)
                warm = False
            rows = []
            # rotate the condition order per case so that load drift on the shared GPU does not favour one
            k0 = len(done) % len(conds)
            for cond in conds[k0:] + conds[:k0]:
                if (c.id, cond.name) not in done:
                    rows.append(timed_lean(rt, c, im, cond, args, closed))
                    done.add((c.id, cond.name))
            crr.append_jsonl(path, rows)
            torch.cuda.empty_cache()
        print(f'[timing] {bench} done', flush=True)
    crr.write_meta(root, rt, args, 'timing', {'conditions': [c.name for c in conds], 'code_5_3': code_digest()})


# ----------------------------------------------------------------------------- verification

@torch.inference_mode()
def run_verify(rt, out: Path, args) -> None:
    """(1) merger re-implementation on captured block-27 states vs the model's visual tokens; (2) 'shift' tokens
    at even offsets equal the standard tokens; (3) padded batch vs batch size 1 on lean answer items."""
    root = out / 'verify'
    root.mkdir(parents=True, exist_ok=True)
    rep = {'grid_vs_vis_maxabs': [], 'shift_even_vs_grid_maxabs': [], 'vis_absmax': []}
    items = []
    pre_call, pre_ans = co.lean_call_prefix_ids(rt), co.answer_prefix_ids(rt)
    cases = crr.load_cases('textvqa', args)[:args.verify_n] + crr.load_cases('syn_loc', args)[:4]
    merger = rt.model.model.visual.merger
    for c in cases:
        im = crr.load_image(c, args)
        f = co.encode(rt, im, (25, 27))
        H, W = f['grid']
        full = rp.Window(0, 0, H, W)
        tok, gr, _ = co.read_window(merger, f['patch_27'], full, 'grid')
        rep['grid_vs_vis_maxabs'].append(float((tok - f['vis']).abs().max()))
        rep['vis_absmax'].append(float(f['vis'].abs().max()))
        win = cr.box_to_window([300, 300, 700, 700], f['grid'])
        sh, sg, _ = co.read_window(merger, f['patch_27'], win, 'shift')
        sh = sh.view(sg[0], sg[1], -1)[::2, ::2].reshape(-1, sh.shape[-1])
        ref = f['vis'][torch.tensor(win.indices(W), device=f['vis'].device)]
        rep['shift_even_vs_grid_maxabs'].append(float((sh - ref).abs().max()))
        for k in ('crop2', 'c27', 's27', 's25'):
            cond = co.CONDITIONS[k]
            feats = features(rt, im, [cond])
            seg, _ = second_segment(rt, cond, im, feats, win)
            ids = (cr.stage1_ids(rt, c.question) + oracle_call_ids(rt, win.norm_box(f['grid']))
                   + cr.suffix_ids(rt, win.norm_box(f['grid']), c.fmt) + pre_ans)
            items.append(cr.assemble(rt, ids, [cr.VSeg(feats['f1']['vis'], f['grid']), seg]))
    stops = cr.stop_ids(rt, False)
    single = [cr.greedy_padded(rt, [it], args.max_new, stops)[0]['text'] for it in items]
    batched = [r['text'] for r in cr.run_batched(rt, items, args.max_new, stops, args.token_budget, args.max_batch)]
    rep['batch_agree'] = sum(a == b for a, b in zip(single, batched)) / len(items)
    rep['batch_answer_agree'] = sum(cr.vqa_normalize(a) == cr.vqa_normalize(b) for a, b in zip(single, batched)) / len(items)
    rep['batch_pairs'] = list(zip(single, batched))
    rep['n_items'] = len(items)
    (root / 'verify.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ('grid_vs_vis_maxabs', 'shift_even_vs_grid_maxabs', 'batch_agree',
                                          'batch_answer_agree', 'n_items')}), flush=True)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', default='Qwen3.5-9B')
    ap.add_argument('--stages', required=True)
    ap.add_argument('--seed', type=int, default=20261020)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--vcot-gqa', type=Path, required=True)
    ap.add_argument('--mmvp-csv', type=Path, required=True)
    ap.add_argument('--n-natural', type=int, default=200)
    ap.add_argument('--n-synth', type=int, default=48)
    ap.add_argument('--grid-n', type=int, default=20)
    ap.add_argument('--loc-k', type=int, default=5)
    ap.add_argument('--gqa-max-area', type=float, default=0.02)
    ap.add_argument('--max-pixels', type=int, default=512 * 512)
    ap.add_argument('--max-call', type=int, default=40)
    ap.add_argument('--max-new', type=int, default=32)
    ap.add_argument('--native-max-call', type=int, default=160)
    ap.add_argument('--native-max-calls', type=int, default=3)
    ap.add_argument('--native-max-new', type=int, default=256)
    ap.add_argument('--chunk', type=int, default=8)
    ap.add_argument('--token-budget', type=int, default=24000)
    ap.add_argument('--max-batch', type=int, default=16)
    ap.add_argument('--conds', nargs='*', default=None)
    ap.add_argument('--timing-n', type=int, default=20)
    ap.add_argument('--timing-benches', nargs='*', default=['textvqa', 'gqa', 'mmvp', 'pope', 'syn_loc'])
    ap.add_argument('--verify-n', type=int, default=8)
    ap.add_argument('--max-memory', type=float, nargs='*', default=None)
    args = ap.parse_args()
    # 5.2's loaders and scorers read these fields
    args.n = {b: args.n_natural for b in crr.NATURAL} | {b: args.n_synth for b in crr.SYNTH}
    args.bv_max_pixels, args.max_new_bv = 1024 * 1024, args.native_max_new
    args.output.mkdir(parents=True, exist_ok=True)
    rt = crr.load_runtime(args)                    # one GPU, or a layer split when --max-memory is given
    for stage in args.stages.split(','):
        t = time.time()
        if stage == 'verify':
            run_verify(rt, args.output, args)
        elif stage in ('timing', 'timing2'):
            run_timing(rt, args.output, args, stage == 'timing2')
        elif stage.startswith('lean_'):
            run_lean(rt, stage[5:], args.output, args)
        elif stage.startswith('lean2_'):
            run_lean(rt, stage[6:], args.output, args, closed=True)
        elif stage.startswith('native_'):
            nat = argparse.Namespace(**{**vars(args), 'max_call': args.native_max_call,
                                        'max_calls': args.native_max_calls, 'max_new': args.native_max_new,
                                        'chunk': 16})
            run_native(rt, stage[7:], args.output, nat)
        else:
            raise ValueError(f'Unknown stage {stage!r}.')
        print(f'stage {stage} finished in {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
