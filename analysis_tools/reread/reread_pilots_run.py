"""Experiment 5.4 runner: learned 2x cache (option 1) and high-resolution documents (option 2).

Stages (comma list):
  train          train the learned 2x modules of all variants jointly on the same batches (ViT only, no decoder);
  fidelity       held-out feature fidelity of the learned modules vs u27 / b27 / h27, against two targets: the
                 2x-cache tokens (training target) and the 2x crop re-encode of random windows (ViT only);
  opt1_textvqa   TextVQA accuracy of crop1 / crop2 / cache readers / learned readers (closed-call lean protocol);
  opt2_doc       DocVQA accuracy of the option-2 conditions (closed-call lean protocol);
  opt2_timing    batch-size-1 end-to-end timing of the option-2 conditions with KV-cache continuation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.reread import cache_only as co
from analysis_tools.reread import cached_reread as cr
from analysis_tools.reread import cached_reread_run as crr
from analysis_tools.reread import reread_pilot as rp
from analysis_tools.reread import reread_pilots as P


def code_digest() -> dict:
    import hashlib
    files = ['analysis_tools/reread/reread_pilots.py', 'analysis_tools/reread/reread_pilots_run.py']
    return {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16] for f in files}


def write_meta(root: Path, args, stage: str, extra: dict) -> None:
    meta = {'stage': stage, 'args': {k: str(v) for k, v in vars(args).items()}, 'code': code_digest(),
            'torch': torch.__version__, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **extra}
    (root / f'meta_{stage}.json').write_text(json.dumps(meta, indent=1, default=str))


def checkpoint_path(args) -> Path:
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    return resolve_model(args.model, load_paths()['HF_HUB_CACHE'])


# ----------------------------------------------------------------------------- option 1: training

def run_train(args) -> None:
    root = args.output / 'train'
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    visual, ip = P.load_vision(checkpoint_path(args), 'cuda:0')
    merger = visual.merger
    files = P.training_files(args.data_dir, args.n_train_textvqa, args.n_train_coco, args.seed)
    dl = torch.utils.data.DataLoader(P.PairDataset(files, ip, args.max_pixels), batch_size=args.batch,
                                     shuffle=False, num_workers=args.workers, collate_fn=P.collate_pairs,
                                     persistent_workers=False, prefetch_factor=4)
    names = args.variants
    hp = {'c': args.width, 'n_lo': args.n_lo, 'n_hi': args.n_hi}
    mods = {n: P.Upsampler2x(len(P.VARIANTS[n]), **hp).cuda() for n in names}
    opts = {n: torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01) for n, m in mods.items()}
    steps = args.epochs * len(dl)
    scheds = {n: torch.optim.lr_scheduler.OneCycleLR(o, args.lr, total_steps=steps, pct_start=0.05)
              for n, o in opts.items()}
    blocks = tuple(sorted({b for n in names for b in P.VARIANTS[n]}))
    log = (root / 'loss.jsonl').open('a')
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        for p1, g1, p2, g2 in dl:
            f1 = P.vit_states(visual, p1, g1, blocks)
            f2 = P.vit_states(visual, p2, g2)
            # relative error: summed squared error over all tokens / summed squared deviation of the target
            # tokens from their mean, so 1.0 = predicting the batch mean and 0.0 = exact
            tgt = torch.cat([f['vis'] for f in f2]).float()
            denom = (tgt - tgt.mean(0)).pow(2).sum()
            u_err = sum(P.token_mse(P.merge_grid(merger, f['patch_27'].repeat_interleave(2, 0).repeat_interleave(2, 1)),
                                    t['vis']).sum() for f, t in zip(f1, f2))
            rec = {'step': step, 'epoch': ep, 'rel_u27': float(u_err / denom)}
            for n, m in mods.items():
                opts[n].zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    err = sum(P.token_mse(P.merge_grid(merger, m([f[f'patch_{k}'] for k in P.VARIANTS[n]])),
                                          t['vis']).sum() for f, t in zip(f1, f2))
                loss = err / denom
                P.check_finite(loss, f'{n} loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opts[n].step()
                scheds[n].step()
                rec[f'rel_{n}'] = float(loss)
            log.write(json.dumps(rec) + '\n')
            if step % 25 == 0:
                log.flush()
                print(f'[train] step {step}/{steps} {time.time() - t0:.0f}s ' +
                      ' '.join(f'{k}={v:.4f}' for k, v in rec.items() if k.startswith('rel_')), flush=True)
            step += 1
    log.close()
    for n, m in mods.items():
        torch.save({'state': m.state_dict(), 'hparams': hp, 'blocks': P.VARIANTS[n], 'n_params': P.n_params(m)},
                   root / f'{n}.pt')
    write_meta(root, args, 'train', {'n_files': len(files), 'steps': steps, 'seconds': time.time() - t0,
                                      'n_params': {n: P.n_params(m) for n, m in mods.items()}, 'hparams': hp})


# ----------------------------------------------------------------------------- option 1: fidelity

@torch.no_grad()
def run_fidelity(args) -> None:
    """Held-out TextVQA val images (not the 200 evaluation questions' sampling, but the same pool) with one random
    window each (4-12 merged tokens per side). Per token squared error of each reader against (a) the 2x-cache
    tokens of the window and (b) the 2x crop re-encode tokens of the window."""
    root = args.output / 'fidelity'
    root.mkdir(parents=True, exist_ok=True)
    visual, ip = P.load_vision(checkpoint_path(args), 'cuda:0')
    merger = visual.merger
    mods = P.load_modules(args.output / 'train', args.variants, 'cuda:0')
    rows = crr.read_jsonl(args.data_dir / 'TextVQA/data/val_data.jsonl')
    imgs = sorted({r['image'] for r in rows})
    rng = np.random.default_rng(args.seed + 7)
    pick = [imgs[i] for i in rng.choice(len(imgs), size=args.n_fidelity, replace=False)]
    blocks = tuple(sorted({b for n in args.variants for b in P.VARIANTS[n]} | {27}))
    out = []
    for name in pick:
        im = cr.resize_budget(Image.open(args.data_dir / 'TextVQA/data/val_images' / name).convert('RGB'),
                              args.max_pixels)
        f = P.vit_states(visual, *P.pixel_inputs(ip, [im]), blocks)[0]
        f2 = P.vit_states(visual, *P.pixel_inputs(ip, [cr.upscale(im, 2)]))[0]
        H, W = f['grid']
        h, w = int(rng.integers(4, min(12, H) + 1)), int(rng.integers(4, min(12, W) + 1))
        win = rp.Window(int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1)), h, w)
        crop = P.vit_states(visual, *P.pixel_inputs(ip, [cr.upscale(rp.crop(im, win), 2)]))[0]['vis']
        sw = rp.Window(2 * win.r0, 2 * win.c0, 2 * h, 2 * w)
        idx = torch.tensor(sw.indices(2 * W), device=crop.device)
        readers = {'h27': f2['vis'][idx],
                   'u27': co.read_window(merger, f['patch_27'], win, 'nearest')[0],
                   'b27': co.read_window(merger, f['patch_27'], win, 'bilinear')[0],
                   # each 1x visual token copied to its 2x2 sub-positions (token-space nearest upsampling)
                   'rep27': f['vis'].view(H, W, -1)[win.r0:win.r0 + h, win.c0:win.c0 + w]
                   .repeat_interleave(2, 0).repeat_interleave(2, 1).reshape(4 * h * w, -1)}
        for n, m in mods.items():
            readers[n] = P.learned_tokens(m, merger, f, P.VARIANTS[n])[idx]
        rec = {'image': name, 'grid': [H, W], 'window': win.as_list()}
        for tname, tgt in (('cache2x', f2['vis'][idx]), ('crop2', crop)):
            t = tgt.float()
            rec[f'{tname}_ss'] = float((t - t.mean(0)).pow(2).sum())
            rec[f'{tname}_norm2'] = float(t.pow(2).sum())
            for rn, v in readers.items():
                if rn == 'h27' and tname == 'cache2x':
                    continue
                rec[f'{tname}_{rn}_se'] = float(P.token_mse(v, t).sum())
                rec[f'{tname}_{rn}_cos'] = float(torch.nn.functional.cosine_similarity(v.float(), t, -1).mean())
        out.append(rec)
    (root / 'fidelity.jsonl').write_text('\n'.join(json.dumps(r) for r in out) + '\n')
    write_meta(root, args, 'fidelity', {'n': len(out)})


# ----------------------------------------------------------------------------- LLM stages: shared pieces

def case_images(case: cr.Case, args, conds: list[P.PCond]) -> dict:
    """The images a case needs: lo always; hi for documents; up (bicubic 2x of lo) when a condition reads it."""
    srcs = {c.view for c in conds} | {c.src for c in conds if c.tool in ('crop', 'cache')}
    if case.bench == 'docvqa':
        lo, hi = P.doc_images(case.image_path, args.doc_max_pixels)
        ims = {'lo': lo, 'hi': hi}
    else:
        if 'hi' in srcs:
            raise ValueError('The hi image is defined for documents only.')
        ims = {'lo': crr.load_image(case, args)}
    if 'up' in srcs:
        ims['up'] = cr.upscale(ims['lo'], 2)
    return ims


def case_features(rt, ims: dict, conds: list[P.PCond], modules: dict, ip) -> tuple[dict, dict, dict]:
    """Window-independent ViT passes (one per needed image) and learned 2x tokens. Returns feats, learned, cost."""
    visual = rt.model.model.visual
    need = set().union(*(c.passes() for c in conds))
    var = sorted({c.variant for c in conds if c.tool == 'learned'})
    lo_blocks = tuple(sorted({b for v in var for b in P.VARIANTS[v]} |
                             ({27} if any(c.merge != 'grid' and c.src == 'lo' for c in conds) else set())))
    feats, cost = {}, {}
    for p in sorted(need):
        feats[p] = P.vit_states(visual, *P.pixel_inputs(ip, [ims[p]]), lo_blocks if p == 'lo' else ())[0]
        cost[p] = 4 * feats[p]['grid'][0] * feats[p]['grid'][1]
    learned = {v: P.learned_tokens(modules[v], visual.merger, feats['lo'], P.VARIANTS[v]) for v in var}
    return feats, learned, cost


def run_lean2(rt, bench: str, cases: list[cr.Case], conds: list[P.PCond], modules: dict, root: Path, args) -> None:
    """Closed-call lean protocol of experiment 5.3 (lean2): the call turn is prefilled up to the bbox value and
    decoded only to the closing bracket, the call is closed with the fixed tail, the tool runs once, and the
    answer turn is prefilled with "Final answer:". One call per (case, view), shared by all tool conditions with
    that view."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'results.jsonl'
    done = {(r['case_id'], r['cond']) for r in crr.read_jsonl(path)}
    ip = rt.processor.image_processor
    stop_tool, stop_ans = cr.stop_ids(rt, True), cr.stop_ids(rt, False)
    stops1 = stop_tool | co.bracket_stop_ids(rt)
    pre_call, pre_ans = co.lean_call_prefix_ids(rt), co.answer_prefix_ids(rt)
    vcfg = rt.model.config.vision_config
    t0 = time.time()
    for ci in range(0, len(cases), args.chunk):
        chunk = [c for c in cases[ci:ci + args.chunk] if any((c.id, k.name) not in done for k in conds)]
        if not chunk:
            continue
        data = []
        for c in chunk:
            ims = case_images(c, args, conds)
            feats, learned, cost = case_features(rt, ims, conds, modules, ip)
            data.append((c, ims, feats, learned, cost))
        calls = {}
        for view in sorted({k.view for k in conds if k.tool}):
            items = [cr.assemble(rt, cr.stage1_ids(rt, c.question) + pre_call,
                                 [cr.VSeg(f[view]['vis'], f[view]['grid'])]) for c, _, f, _, _ in data]
            res = cr.run_batched(rt, items, args.max_call, stops1, args.token_budget, args.max_batch)
            for (c, *_), r in zip(data, res):
                ids, text = co.close_call(rt, r['ids'])
                calls[(c.id, view)] = (pre_call + ids, co.CALL_PREFIX + text)
            del items
        jobs, rows = [], []
        for c, ims, feats, learned, cost in data:
            for k in conds:
                if (c.id, k.name) in done:
                    continue
                fv = feats[k.view]
                g = fv['grid']
                n_vit = sum(cost[p] for p in k.passes())
                row = {'case_id': c.id, 'bench': bench, 'cond': k.name, 'grid': list(g), 'vit_patches': n_vit,
                       'vit_flops': sum(co.vit_flops(cost[p], vcfg) for p in k.passes())}
                if k.tool is None:
                    ids, segs = cr.plain_ids(rt, c.question) + pre_ans, [cr.VSeg(fv['vis'], g)]
                else:
                    call_ids, call_text = calls[(c.id, k.view)]
                    box = cr.parse_call(call_text)
                    row['call'] = call_text
                    if box is None:
                        row.update(text='', finish_reason='bad_call', correct=0.0, L=0, n_extra_visual=0)
                        rows.append(row)
                        continue
                    win = cr.box_to_window(box, g)
                    seg, tc = P.window_tokens(k, win, ims, feats, learned, rt.model.model.visual,
                                              rt.model.model.visual.merger, ip)
                    row['vit_patches'] += tc['vit_patches']
                    row['vit_flops'] += co.vit_flops(tc['vit_patches'], vcfg) if tc['vit_patches'] else 0.0
                    ids = (cr.stage1_ids(rt, c.question) + call_ids + cr.suffix_ids(rt, win.norm_box(g), c.fmt)
                           + pre_ans)
                    segs = [cr.VSeg(fv['vis'], g), seg]
                    row.update(bbox=box, window=win.as_list(), window_area=P.window_area(win, g),
                               coverage=crr.coverage(win, g, c.meta.get('evidence_box') or c.oracle_box),
                               seg_grid=list(seg.grid), capped=tc['capped'])
                it = cr.assemble(rt, ids, segs)
                it['x'] = it['x'].cpu()
                row.update(L=it['L'], n_extra_visual=it['n_extra_visual'])
                jobs.append((row, it, c))
        if jobs:
            res = cr.run_batched(rt, [it for _, it, _ in jobs], args.max_new, stop_ans, args.token_budget,
                                 args.max_batch)
            for (row, it, c), r in zip(jobs, res):
                text = co.ANSWER_PREFIX + ' ' + r['text']
                sc = P.score_doc(c, text) if bench == 'docvqa' else crr.score(c, text, r['finish_reason'], None)
                row.update(text=r['text'], finish_reason=r['finish_reason'], batch_size=r['batch_size'], **sc)
                rows.append(row)
        crr.append_jsonl(path, rows)
        del data, jobs
        torch.cuda.empty_cache()
        print(f'[{bench}] {min(ci + args.chunk, len(cases))}/{len(cases)} cases {time.time() - t0:.0f}s', flush=True)
    write_meta(root, args, bench, {'conditions': [k.name for k in conds], 'n_cases': len(cases),
                                   'token_budget': args.token_budget, 'max_batch': args.max_batch,
                                   'model': qr.describe(rt)})


# ----------------------------------------------------------------------------- timing (batch size 1)

def _t() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def timed(rt, case: cr.Case, ims: dict, k: P.PCond, modules: dict, args) -> dict:
    """Stages: vit (window-independent passes and the learned module), call (first prefill + closed call
    decoding), tool (crop re-encode or cache slice), answer (continuation prefill on the KV cache + answer)."""
    ip = rt.processor.image_processor
    stop_tool, stop_ans = cr.stop_ids(rt, True), cr.stop_ids(rt, False)
    pre_call, pre_ans = co.lean_call_prefix_ids(rt), co.answer_prefix_ids(rt)
    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t = {'t0': _t()}
    feats, learned, _ = case_features(rt, ims, [k], modules, ip)
    fv = feats[k.view]
    g = fv['grid']
    t['vit'] = _t()
    rec = {'case_id': case.id, 'cond': k.name}
    if k.tool is None:
        it = cr.assemble(rt, cr.plain_ids(rt, case.question) + pre_ans, [cr.VSeg(fv['vis'], g)])
        ans, cache = crr.greedy_cached(rt, it['x'], it['pos'], None, args.max_new, stop_ans)
        t['call'] = t['tool'] = t['vit']
        L, extra = it['L'] + len(ans), 0
    else:
        ids = cr.stage1_ids(rt, case.question) + pre_call
        it = cr.assemble(rt, ids, [cr.VSeg(fv['vis'], g)])
        gen, cache = crr.greedy_cached(rt, it['x'], it['pos'], None, args.max_call, stop_tool | co.bracket_stop_ids(rt))
        n_dec = len(gen)
        gen, _ = co.close_call(rt, gen)
        t['call'] = _t()
        box = cr.parse_call(co.CALL_PREFIX + rt.tokenizer.decode(gen))
        if box is None:
            raise RuntimeError(f'Unparsable forced call for {case.id}.')
        fed = it['L'] + n_dec - 1
        win = cr.box_to_window(box, g)
        seg, _ = P.window_tokens(k, win, ims, feats, learned, rt.model.model.visual, rt.model.model.visual.merger, ip)
        t['tool'] = _t()
        ids = ids + gen + cr.suffix_ids(rt, win.norm_box(g), case.fmt) + pre_ans
        it = cr.assemble(rt, ids, [cr.VSeg(fv['vis'], g), seg])
        extra = it['n_extra_visual']
        ans, cache = crr.greedy_cached(rt, it['x'][fed:], it['pos'][:, fed:], cache, args.max_new, stop_ans)
        t['answer'] = _t()
        L = it['L'] + len(ans)
    t.setdefault('answer', _t())
    rec.update(seconds=t['answer'] - t['t0'], t_vit=t['vit'] - t['t0'], t_call=t['call'] - t['vit'],
               t_tool=t['tool'] - t['call'], t_answer=t['answer'] - t['tool'], L_final=L, n_extra_visual=extra,
               peak_extra_bytes=torch.cuda.max_memory_allocated() - base_mem, **cr.kv_bytes(rt.model.config, L))
    del cache
    return rec


def run_timing(rt, cases: list[cr.Case], conds: list[P.PCond], modules: dict, root: Path, args) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'timing.jsonl'
    done = {(r['case_id'], r['cond']) for r in crr.read_jsonl(path)}
    warm = True
    for i, c in enumerate(cases):
        ims = case_images(c, args, conds)
        if warm:
            for k in conds:
                timed(rt, c, ims, k, modules, args)
            warm = False
        k0 = i % len(conds)             # rotate the order so load drift on a shared machine favours no condition
        rows = [timed(rt, c, ims, k, modules, args) for k in conds[k0:] + conds[:k0] if (c.id, k.name) not in done]
        crr.append_jsonl(path, rows)
        torch.cuda.empty_cache()
    write_meta(root, args, 'timing', {'conditions': [k.name for k in conds], 'gpu': torch.cuda.get_device_name(0)})


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', default='Qwen3.5-9B')
    ap.add_argument('--stages', required=True)
    ap.add_argument('--seed', type=int, default=20261020)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--max-pixels', type=int, default=512 * 512)
    # option 1
    ap.add_argument('--variants', nargs='*', default=list(P.VARIANTS))
    ap.add_argument('--n-train-textvqa', type=int, default=12000)
    ap.add_argument('--n-train-coco', type=int, default=4000)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--width', type=int, default=384)
    ap.add_argument('--n-lo', type=int, default=4)
    ap.add_argument('--n-hi', type=int, default=2)
    ap.add_argument('--n-fidelity', type=int, default=300)
    ap.add_argument('--n-textvqa', type=int, default=200)
    ap.add_argument('--opt1-conds', nargs='*', default=list(P.OPT1))
    # option 2
    ap.add_argument('--doc-meta', type=Path, default=None)
    ap.add_argument('--doc-images', type=Path, default=None)
    ap.add_argument('--n-doc', type=int, default=150)
    ap.add_argument('--doc-max-pixels', type=int, default=1024 * 1024)
    ap.add_argument('--doc-tag', default='', help='suffix of the option-2 output directories (e.g. a second budget)')
    ap.add_argument('--opt2-conds', nargs='*', default=[k for k in P.OPT2 if not k.startswith('lo_l27')])
    ap.add_argument('--timing-n', type=int, default=20)
    # decoding
    ap.add_argument('--max-call', type=int, default=40)
    ap.add_argument('--max-new', type=int, default=32)
    ap.add_argument('--chunk', type=int, default=8)
    ap.add_argument('--token-budget', type=int, default=24000)
    ap.add_argument('--max-batch', type=int, default=16)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(',')
    rt = None
    for stage in stages:
        t = time.time()
        if stage == 'train':
            run_train(args)
        elif stage == 'fidelity':
            run_fidelity(args)
        elif stage in ('opt1_textvqa', 'opt2_doc', 'opt2_timing'):
            if rt is None:
                rt = qr.load(args.model)
            if stage == 'opt1_textvqa':
                conds = [P.OPT1[k] for k in args.opt1_conds]
                mods = P.load_modules(args.output / 'train', sorted({k.variant for k in conds if k.tool == 'learned'}),
                                      rt.device)
                args.n = {'textvqa': args.n_textvqa}
                cases = crr.load_cases('textvqa', args)
                run_lean2(rt, 'textvqa', cases, conds, mods, args.output / 'opt1_textvqa', args)
            else:
                conds = [P.OPT2[k] for k in args.opt2_conds]
                mods = P.load_modules(args.output / 'train', sorted({k.variant for k in conds if k.tool == 'learned'}),
                                      rt.device)
                cases = P.doc_cases(args.doc_meta, args.doc_images, args.n_doc, args.seed)
                if stage == 'opt2_doc':
                    run_lean2(rt, 'docvqa', cases, conds, mods, args.output / f'opt2_doc{args.doc_tag}', args)
                else:
                    run_timing(rt, cases[:args.timing_n], conds, mods, args.output / f'opt2_timing{args.doc_tag}', args)
        else:
            raise ValueError(f'Unknown stage {stage!r}.')
        print(f'stage {stage} finished in {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
