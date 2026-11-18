"""Experiment 7.3 round 2 stage drivers (ViT detector formation for feature-absent targets).

Results doc: docs/analysis/application/results_2026-11-11_vit_detector_formation.md.
Stimuli, training data and probe are those of experiment 7.3 (--src-run = its run directory); all outputs of this
round go to --output.

Stages (comma list via --stages):
  relocate  (GPU, base model, before any training of this round) per ViT block k:
            (1) readout gradient: d s / d o_k at the target's 16 patches of feature-absent (swap) images, where s is the
                teacher-forced log-probability of the correct row and column digits and o_k the block-k MLP output;
            (2) shared absence direction: shrinkage LDA on the block-k MLP input separating B-entity patches (part
                absent) from A-entity patches (part present), pooled over the training pairs; leave-one-pair-out
                deviation of the left-out pair's absent target among its distractors in swap images;
            (3) write-then-read simulation: one new rectified unit at block k reading the left-out-pair-free absence
                direction and writing along the left-out-pair-free mean readout gradient; change of s and of the hit
                rate on the left-out pair's swap images.  The pre-registered window ranking is written to
                relocate/prereg.json before any training of this round.
  train     one adapter: --method lora (rank, alpha) | full (DenseDelta on fc1, fc2) | width (WidthExtension, --width)
            on --scope (located_finetune.parse_scope grammar, MLP scopes), same data / steps as 7.3;
  probe     the 7.3 odd-entity probe (1188 items) for one adapter;
  detect    per adapter: exclusive-detector counts of existing neurons at the tuned blocks, per-new-neuron zeta
            (width), and the per-neuron split of the adapter's target-specific write (distributed vs concentrated);
  badcase   dose probe (part size x raster position) on held-out features for one adapter;
  vl        Qwen3-VL-8B ViT detector statistics and DeepStack-tap separability (mechanism comparison only);
  general   coordinate, counting and general benchmarks for one adapter (7.3 stage_eval with this round's loader).
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.application import detector_formation as dfm
from analysis_tools.application import interp_opt as io_
from analysis_tools.application import interp_opt_run as ior
from analysis_tools.application import located_finetune as lf
from analysis_tools.application import located_finetune_run as lfr
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_run as tfr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_run as sr

GRID = lfr.GRID
N_QUADS = lfr.N_QUADS
UNIT_THRESHOLD = 3.0        # write-then-read unit fires above 3 sd of the distractor-class patch responses
UNIT_SATURATION = 3.0       # ... and saturates 3 sd above its threshold
SCALES = (1.0, 3.0)         # write norms in units of the median presence-detector write (7.3 locate delta)
ANALYSIS_PAIRS = ('sqgap', 'term', 'closure', 'qo', 'dot', 'plusx')
DOSE_PAIRS = {'closure': (2, 4, 8, 12), 'qo': (3, 6, 10, 14), 'dot': (2, 4, 6, 8), 'plusx': (8, 15, 30, 45)}


# ----------------------------------------------------------------------------- helpers

def _quads(pair: str) -> list[dict]:
    return [lfr.quad_images(pair, k) for k in range(N_QUADS)]


def _meta(quads):
    meta = quads[0]['images']['homA'][1]
    return meta, lf.patch_index(oe.entity_tokens(meta)), 4 * meta['merged_grid'][0] * meta['merged_grid'][1]


class InputTap:
    """Stores the MLP input (post norm2) of every ViT block of the last call."""

    def __init__(self, visual):
        self.store, self.handles = {}, []
        for b, blk in enumerate(visual.blocks, start=1):
            self.handles.append(blk.mlp.register_forward_pre_hook(
                lambda m, a, b=b: self.store.__setitem__(b, a[0].detach())))

    def remove(self):
        for h in self.handles:
            h.remove()


def _class_stats(x: torch.Tensor) -> tuple:
    """Sum, scatter (around the mean) and count of patch features x (n, d), float64."""
    x = x.double()
    m = x.mean(0)
    xc = x - m
    return m, xc.T @ xc, len(x)


def _pooled_direction(stats: list[tuple], shrink: float = 0.1) -> torch.Tensor:
    """Shared absence direction from per-pair stats [(mB, SB, nB, mA, SA, nA)]: shrinkage LDA with the mean over
    pairs of (mB - mA) and the within-(pair, class) pooled covariance; unit norm."""
    diff = torch.stack([s[0] - s[3] for s in stats]).mean(0)
    cov = sum(s[1] + s[4] for s in stats)
    n = sum(s[2] + s[5] for s in stats) - 2 * len(stats)
    cov = cov / max(1, n)
    d = cov.shape[0]
    cov = cov + shrink * torch.trace(cov) / d * torch.eye(d, device=cov.device, dtype=cov.dtype)
    w = torch.linalg.solve(cov, diff)
    return (w / w.norm()).float()


# ----------------------------------------------------------------------------- relocate (pre-registration)

@torch.no_grad()
def _absence_stats(rt, pairs) -> dict:
    """Per pair and block: (mB, SB, nB, mA, SA, nA) of MLP-input patches of homB (B entities) and homA images."""
    visual = rt.model.model.visual
    nb = len(visual.blocks)
    tap = InputTap(visual)
    res = {}
    try:
        for pair in pairs:
            quads = _quads(pair)
            _, ent_patch, n_patch = _meta(quads)
            pidx = torch.tensor(ent_patch, device=rt.device).flatten()
            feats = {}
            for v in ('homB', 'homA'):
                xs = {b: [] for b in range(1, nb + 1)}
                for i0 in range(0, N_QUADS, 6):
                    ims = [q['images'][v][0] for q in quads[i0:i0 + 6]]
                    lfr.vit_forward(rt, ims)
                    for b in range(1, nb + 1):
                        xs[b].append(tap.store[b].reshape(len(ims), n_patch, -1)[:, pidx].reshape(-1, tap.store[b].shape[-1]))
                    tap.store.clear()
                feats[v] = {b: _class_stats(torch.cat(xs[b])) for b in xs}
            res[pair] = {b: (*feats['homB'][b], *feats['homA'][b]) for b in range(1, nb + 1)}
            print(f'[relocate] stats {pair}', flush=True)
    finally:
        tap.remove()
    return res


@torch.no_grad()
def _swap_inputs(rt, pair: str) -> tuple[dict, list[int], np.ndarray]:
    """MLP inputs of every block on the 12 swap images of a pair: {block: (12, n_ent, 16, d)}, target indices."""
    visual = rt.model.model.visual
    quads = _quads(pair)
    _, ent_patch, n_patch = _meta(quads)
    pidx = torch.tensor(ent_patch, device=rt.device)
    tap = InputTap(visual)
    out = {b: [] for b in range(1, len(visual.blocks) + 1)}
    try:
        for i0 in range(0, N_QUADS, 6):
            ims = [q['images']['swap'][0] for q in quads[i0:i0 + 6]]
            lfr.vit_forward(rt, ims)
            for b in out:
                out[b].append(tap.store[b].reshape(len(ims), n_patch, -1)[:, pidx])
            tap.store.clear()
    finally:
        tap.remove()
    return {b: torch.cat(v) for b, v in out.items()}, [q['target'][0] * GRID + q['target'][1] for q in quads], ent_patch


def _readout_gradients(rt, pair: str) -> torch.Tensor:
    """(12, n_blocks, 2, d): d s / d o_k summed over the target's patches and averaged over distractor entities."""
    visual = rt.model.model.visual
    nb = len(visual.blocks)
    quads = _quads(pair)
    _, ent_patch, n_patch = _meta(quads)
    digits = sr.digit_ids(rt, GRID)
    zs: dict = {}

    def mk(b):
        def f(m, a, o):
            z = torch.zeros_like(o, requires_grad=True)
            zs[b] = z
            return o + z
        return f
    hs = [blk.mlp.register_forward_hook(mk(b)) for b, blk in enumerate(visual.blocks, start=1)]
    out = []
    try:
        for q in quads:
            ans, i_row, i_col = sr.answer_ids(rt, q['target'])
            batch = sr.build_rows(rt, [q['images']['swap'][0]], [ans])
            with torch.enable_grad():
                logits = rt.model(**batch, use_cache=False, logits_to_keep=len(ans) + 1).logits
                lp = torch.log_softmax(logits.float(), -1)
                s = lp[0, i_row, ans[i_row]] + lp[0, i_col, ans[i_col]]
                grads = torch.autograd.grad(s, [zs[b] for b in range(1, nb + 1)])
            t = q['target'][0] * GRID + q['target'][1]
            tp = torch.tensor(ent_patch[t], device=rt.device)
            others = torch.tensor(np.delete(ent_patch, t, axis=0), device=rt.device)
            g = torch.stack([torch.stack([gr[tp].float().sum(0), gr[others].float().sum(1).mean(0)]) for gr in grads])
            out.append(g.cpu())
            zs.clear()
    finally:
        for h in hs:
            h.remove()
    return torch.stack(out)


def _unit_scores(rt, pair: str, specs: list, chunk: int = 10) -> list[dict]:
    """Teacher-forced scores on the 12 swap images of `pair` under unit insertions.

    specs: list of None (no edit) or dicts {block, w, mu, sd, g, c}: at every patch of the image the block-k MLP
    output receives c * clamp((w^T x - mu) / sd - UNIT_THRESHOLD, 0, UNIT_SATURATION) / UNIT_SATURATION * g,
    x = MLP input: a saturating rectified unit that is silent below 3 sd and writes at most c per patch.
    Returns per spec the list of 12 per-image score dicts.
    """
    visual = rt.model.model.visual
    quads = _quads(pair)
    _, _, n_patch = _meta(quads)
    digits = sr.digit_ids(rt, GRID)
    res = [[] for _ in specs]
    cur: dict = {}

    def mk(b):
        def f(m, a, o):
            if b not in cur:
                return o
            o = o.clone()
            x = a[0]
            for r, sp in cur[b]:
                sl = slice(r * n_patch, (r + 1) * n_patch)
                z = (x[sl].float() @ sp['w'] - sp['mu']) / sp['sd']
                resp = torch.clamp(z - UNIT_THRESHOLD, 0.0, UNIT_SATURATION) / UNIT_SATURATION
                o[sl] = o[sl] + (sp['c'] * resp[:, None] * sp['g'][None]).to(o.dtype)
            return o
        return f
    hs = [blk.mlp.register_forward_hook(mk(b)) for b, blk in enumerate(visual.blocks, start=1)]
    try:
        for q in quads:
            ans, i_row, i_col = sr.answer_ids(rt, q['target'])
            for c0 in range(0, len(specs), chunk):
                part = specs[c0:c0 + chunk]
                cur.clear()
                for r, sp in enumerate(part):
                    if sp is not None:
                        cur.setdefault(sp['block'], []).append((r, sp))
                batch = sr.build_rows(rt, [q['images']['swap'][0]] * len(part), [ans] * len(part))
                with torch.no_grad():
                    logits = rt.model(**batch, use_cache=False, logits_to_keep=len(ans) + 1).logits
                for j, sc in enumerate(sr.score_rows(logits, [ans] * len(part), i_row, i_col, len(ans), digits)):
                    res[c0 + j].append({'s': sc['s'], 'hit': int(sc['arg_row'] == q['target'][0] + 1
                                                                 and sc['arg_col'] == q['target'][1] + 1)})
    finally:
        cur.clear()
        for h in hs:
            h.remove()
    return res


def stage_relocate(rt, out: Path, args) -> None:
    d = out / 'relocate'
    d.mkdir(parents=True, exist_ok=True)
    rt.model.requires_grad_(False)
    nb = len(rt.model.model.visual.blocks)
    blocks = list(range(1, nb + 1))
    train, held = list(lf.TRAIN_PAIRS), list(lf.HELDOUT_FEATURES + lf.HELDOUT_FAMILY)
    t0 = time.time()
    # (1) readout gradients
    gpath = d / 'grads.pt'
    if gpath.exists():
        grads = torch.load(gpath)
    else:
        grads = {p: _readout_gradients(rt, p) for p in train + held}
        torch.save(grads, gpath)
    print(f'[relocate] gradients {time.time() - t0:.0f}s', flush=True)
    # (2) absence statistics and directions
    stats = _absence_stats(rt, train)
    delta = np.load(Path(args.src_run) / 'locate' / 'delta.npz')
    c_unit = float(np.median([np.linalg.norm(delta[p][b - 1]) for p in delta.files for b in range(7, nb + 1)]))
    rows, sep = [], []
    folds = [(p, [q for q in train if q != p]) for p in train] + [(p, train) for p in held]
    for test_pair, fit_pairs in folds:
        X, T, _ = _swap_inputs(rt, test_pair)
        specs = [None]
        for b in blocks:
            w = _pooled_direction([stats[q][b] for q in fit_pairs]).to(rt.device)
            # standardisation and normaliser from the fit pairs' A patches / B entities (patch stats via the means)
            mu = float(np.mean([float(stats[q][b][3].float().to(rt.device) @ w) for q in fit_pairs]))
            var = np.mean([float(w @ (stats[q][b][4].float().to(rt.device) / stats[q][b][5]) @ w) for q in fit_pairs])
            sd = float(math.sqrt(var))
            gfit = torch.stack([grads[q][:, b - 1, 0] for q in fit_pairs]).flatten(0, 1).mean(0)
            g = (gfit / gfit.norm()).to(rt.device)
            # leave-one-pair-out separability on swap images of the test pair (entity max over patches)
            s = (X[b].float() @ w).amax(-1)                     # (12, n_ent)
            zs = []
            for i, t in enumerate(T):
                o = torch.cat([s[i, :t], s[i, t + 1:]])
                zs.append(float((s[i, t] - o.mean()) / (o.std() + 1e-6)))
            gt = grads[test_pair][:, b - 1]
            sep.append({'pair': test_pair, 'fold': 'lopo' if test_pair in train else 'heldout', 'block': b,
                        'zeta_swap': float(np.median(zs)), 'grad_norm': float(gt[:, 0].mean(0).norm()),
                        'grad_norm_specific': float((gt[:, 0] - gt[:, 1]).mean(0).norm()),
                        'grad_cos_fit': float(torch.nn.functional.cosine_similarity(gt[:, 0].mean(0), gfit.cpu(), 0)),
                        'mu': mu, 'sd': sd})
            for c in SCALES:
                specs.append({'block': b, 'w': w, 'mu': mu, 'sd': sd, 'g': g, 'c': c * c_unit, 'scale': c})
        res = _unit_scores(rt, test_pair, specs)
        base = res[0]
        for sp, r in zip(specs, res):
            rows.append({'pair': test_pair, 'fold': 'lopo' if test_pair in train else 'heldout',
                         'block': None if sp is None else sp['block'], 'scale': 0.0 if sp is None else sp['scale'],
                         'ds': float(np.mean([x['s'] - y['s'] for x, y in zip(r, base)])),
                         'dhit': float(np.mean([x['hit'] - y['hit'] for x, y in zip(r, base)])),
                         'hit': float(np.mean([x['hit'] for x in r]))})
        print(f'[relocate] write-then-read {test_pair} {time.time() - t0:.0f}s', flush=True)
    (d / 'separability.json').write_text(json.dumps(sep, indent=1))
    (d / 'write_read.json').write_text(json.dumps({'c_unit': c_unit, 'rows': rows}, indent=1))
    # pre-registered criterion and window ranking (training pairs, leave-one-pair-out, scale 1)
    crit = {b: float(np.mean([r['ds'] for r in rows if r['fold'] == 'lopo' and r['block'] == b and r['scale'] == 1.0]))
            for b in blocks}
    ws = dfm.window_scores(crit, dfm.WINDOWS)
    pre = {'time': datetime.now(timezone.utc).isoformat(), 'criterion': 'mean over 5 leave-one-pair-out folds of the '
           'teacher-forced log-prob gain of the write-then-read unit at scale 1', 'block_scores': crit,
           'window_scores': {f'{w[0]}-{w[-1]}': v for w, v in ws.items()},
           'window_ranking': [f'{w[0]}-{w[-1]}' for w, _ in sorted(ws.items(), key=lambda kv: -kv[1])],
           'best_block': max(crit, key=crit.get)}
    ppath = d / 'prereg.json'
    if ppath.exists():
        raise RuntimeError(f'{ppath} exists; the pre-registration is never overwritten')
    ppath.write_text(json.dumps(pre, indent=1))
    print(json.dumps(pre), flush=True)


@torch.no_grad()
def stage_oddness(rt, out: Path, args) -> None:
    """Shared contextual oddness at the MLP input of every ViT block (base model or --name adapter).

    For swap (part-absent target) and std (part-present target) images of every probe pair, entity vectors are the
    16-patch means of the block-k MLP input, centred per image (x_e - mean over the image's entities).  A shrinkage
    LDA direction separating the target from the distractors is fitted on the training pairs except one and the
    left-out pair's median per-image deviation (s_t - mean s_d) / sd_d is reported ('lopo'); held-out pairs use the
    direction of all five training pairs ('heldout'); 'within' fits and tests on alternate images of the same pair.
    The same is computed at the merger output (token means of the target's 4 tokens)."""
    d = out / 'eval' / args.name
    d.mkdir(parents=True, exist_ok=True)
    visual = rt.model.model.visual
    nb = len(visual.blocks)
    ad = load_adapter(rt, out, args.name)
    tap = InputTap(visual)
    mstore: dict = {}
    hm = visual.merger.register_forward_hook(lambda m, a, o: mstore.__setitem__('m', o.detach()))
    keys = list(range(1, nb + 1)) + ['merger']
    feats: dict = {}
    try:
        for pair in lf.PROBE_PAIRS:
            quads = _quads(pair)
            meta = quads[0]['images']['swap'][1]
            ent_patch = lf.patch_index(oe.entity_tokens(meta))
            pidx = torch.tensor(ent_patch, device=rt.device)
            tok = torch.tensor(oe.entity_tokens(meta), device=rt.device)
            n_patch = 4 * meta['merged_grid'][0] * meta['merged_grid'][1]
            for v in ('swap', 'std'):
                acc = {k: [] for k in keys}
                for i0 in range(0, N_QUADS, 6):
                    ims = [q['images'][v][0] for q in quads[i0:i0 + 6]]
                    lfr.vit_forward(rt, ims)
                    for b in range(1, nb + 1):
                        acc[b].append(tap.store[b].reshape(len(ims), n_patch, -1)[:, pidx].float().mean(2))
                    acc['merger'].append(mstore['m'].reshape(len(ims), n_patch // 4, -1)[:, tok].float().mean(2))
                    tap.store.clear()
                for k in keys:
                    x = torch.cat(acc[k])
                    feats[(pair, v, k)] = (x - x.mean(1, keepdim=True)).cpu()
            print(f'[oddness] {pair}', flush=True)
    finally:
        tap.remove()
        hm.remove()
        if ad is not None:
            ad.detach()
    targets = {p: [q['target'][0] * GRID + q['target'][1] for q in _quads(p)] for p in lf.PROBE_PAIRS}

    def fit(pairs, v, k, idx=None):
        tg, ds = [], []
        for p in pairs:
            X = feats[(p, v, k)]
            for i, t in enumerate(targets[p]):
                if idx is not None and i not in idx:
                    continue
                tg.append(X[i, t])
                ds.append(torch.cat([X[i, :t], X[i, t + 1:]]))
        return lfr.torch_lda(torch.stack(tg).to(rt.device), torch.cat(ds).to(rt.device)).float().cpu()

    def test(p, v, k, w, idx=None):
        X = feats[(p, v, k)] @ w
        zs = []
        for i, t in enumerate(targets[p]):
            if idx is not None and i not in idx:
                continue
            o = torch.cat([X[i, :t], X[i, t + 1:]])
            zs.append(float((X[i, t] - o.mean()) / (o.std() + 1e-6)))
        return float(np.median(zs))
    def meandiff(pairs, v, k):
        # unwhitened mean offset of the target from its image's entity mean (features are already centred per image)
        u = torch.stack([feats[(p, v, k)][i, t] for p in pairs for i, t in enumerate(targets[p])]).mean(0)
        return u / u.norm()

    rows = []
    train = list(lf.TRAIN_PAIRS)
    dir_path = out / 'eval' / 'base' / 'oddness_dirs.pt'
    base_dirs = torch.load(dir_path) if args.name != 'base' else {}
    dirs = {}
    for v in ('std', 'swap'):             # std first: swap rows also use the present-target directions
        for k in keys:
            w_all = fit(train, v, k)
            dirs[(v, k, 'all')] = w_all
            for p in lf.PROBE_PAIRS:
                w = fit([q for q in train if q != p], v, k) if p in train else w_all
                dirs[(v, k, p)] = w
                ev, od = set(range(0, N_QUADS, 2)), set(range(1, N_QUADS, 2))
                row = {'pair': p, 'variant': v, 'block': k, 'fold': 'lopo' if p in train else 'heldout',
                       'zeta': test(p, v, k, w), 'within': test(p, v, k, fit([p], v, k, ev), od)}
                fit_pairs = [q for q in train if q != p] if p in train else train
                dirs[('meandiff', v, k, p)] = meandiff(fit_pairs, v, k)
                ref = base_dirs if base_dirs else dirs
                if base_dirs:      # adapter features read along the direction fitted on the base model
                    row['zeta_basedir'] = test(p, v, k, base_dirs[(v, k, p)])
                if v == 'swap' and ('meandiff', 'std', k, p) in ref:
                    # absent targets read along the base model's PRESENT-target directions (LDA and mean offset)
                    row['zeta_presentdir'] = test(p, v, k, ref[('std', k, p)])
                    row['zeta_present_meandiff'] = test(p, v, k, ref[('meandiff', 'std', k, p)])
                rows.append(row)
    if args.name == 'base':
        torch.save(dirs, dir_path)
    (d / 'oddness.json').write_text(json.dumps(rows, indent=1))
    print(f'oddness {args.name} done', flush=True)


def stage_readout(rt, out: Path, args) -> None:
    """First-order account of how an adapter's change of the decoder input moves the answer.

    For the 12 swap images of every probe pair: g_i = d s / d m_i is the gradient of the base model's teacher-forced
    log-probability s of the correct row and column digits with respect to the merger output m_i of image token i;
    D_i = m_i(tuned) - m_i(base).  The first-order change of s is sum_i g_i^T D_i, split into the target's 4 tokens
    and all other image tokens; it is compared with the actual change of s (tuned minus base).  Also reported:
    cos(D_target - mean D_other, g_target)."""
    d = out / 'eval' / args.name
    d.mkdir(parents=True, exist_ok=True)
    rt.model.requires_grad_(False)
    visual = rt.model.model.visual
    ad = load_adapter(rt, out, args.name)
    if ad is None:
        raise ValueError('readout needs an adapter')
    store: dict = {}

    def hook(m, a, o):
        if store.get('z') is not None:
            return o + store['z']
        store['m'] = o.detach()
        return o
    h = visual.merger.register_forward_hook(hook)
    digits = sr.digit_ids(rt, GRID)
    rows = []
    try:
        for pair in lf.PROBE_PAIRS:
            for q in _quads(pair):
                img = q['images']['swap'][0]
                meta = q['images']['swap'][1]
                tok = oe.entity_tokens(meta)
                t = q['target'][0] * GRID + q['target'][1]
                ans, i_row, i_col = sr.answer_ids(rt, q['target'])
                batch = sr.build_rows(rt, [img], [ans])
                res = {}
                for mode in ('tuned', 'base'):
                    ad.active = set() if mode == 'base' else None
                    store.clear()
                    with torch.no_grad():
                        lfr.vit_forward(rt, [img])
                    m = store['m'].clone()
                    store['z'] = torch.zeros_like(m, requires_grad=True) if mode == 'base' else None
                    with torch.enable_grad() if mode == 'base' else torch.no_grad():
                        logits = rt.model(**batch, use_cache=False, logits_to_keep=len(ans) + 1).logits
                        lp = torch.log_softmax(logits.float(), -1)
                        sc = lp[0, i_row, ans[i_row]] + lp[0, i_col, ans[i_col]]
                        g = torch.autograd.grad(sc, store['z'])[0].float() if mode == 'base' else None
                    res[mode] = (m.float(), float(sc), g)
                    store['z'] = None
                D = res['tuned'][0] - res['base'][0]
                g = res['base'][2]
                tt = torch.tensor(tok[t], device=D.device)
                mask = torch.ones(len(D), dtype=torch.bool, device=D.device)
                mask[tt] = False
                oth = torch.tensor([x for i, e in enumerate(tok) if i != t for x in e], device=D.device)
                contrast = D[tt].mean(0) - D[oth].mean(0)
                rows.append({'pair': pair, 'target': list(q['target']), 'ds_actual': res['tuned'][1] - res['base'][1],
                             'fo_target': float((g[tt] * D[tt]).sum()), 'fo_other': float((g[mask] * D[mask]).sum()),
                             'cos_contrast_g': float(torch.nn.functional.cosine_similarity(contrast, g[tt].sum(0), 0)),
                             'norm_D_target': float(D[tt].norm(dim=1).mean()), 'norm_D_other': float(D[oth].norm(dim=1).mean()),
                             'norm_m': float(res['base'][0][tt].norm(dim=1).mean())})
            print(f'[readout] {pair}', flush=True)
    finally:
        h.remove()
        ad.active = None
        ad.detach()
    (d / 'readout.json').write_text(json.dumps(rows, indent=1))


# ----------------------------------------------------------------------------- adapters

def build_adapter(rt, method: str, scope: str, args):
    visual = rt.model.model.visual
    if scope == 'vit_merger':
        if method != 'lora':
            raise ValueError('the merger is trained with LoRA only')
        lora = tf.LoRA({'merger.linear_fc1': visual.merger.linear_fc1, 'merger.linear_fc2': visual.merger.linear_fc2},
                       args.rank, args.alpha)
        lora.kind = 'lora'
        return lora
    s = lf.parse_scope(scope)
    if s['kind'] != 'vit_mlp':
        raise ValueError('this round trains MLP scopes (or vit_merger) only')
    if method == 'lora':
        lora = lfr.build_lora(rt, scope, args.rank, args.alpha)
        lora.kind = 'lora'
        return lora
    if method == 'full':
        return dfm.DenseDelta(lf.vit_modules(visual, scope))
    if method == 'width':
        return dfm.WidthExtension({f'V{b}.mlp': visual.blocks[b - 1].mlp for b in s['blocks']}, args.width, args.seed)
    raise ValueError(method)


SRC_RUN: Path | None = None      # set by main(); adapters named 'r1_<name>' are 7.3 round-1 LoRA adapters


def load_adapter(rt, out: Path, name: str):
    if name == 'base':
        return None
    if name.startswith('r1_'):
        ad = lfr.load_adapter(rt, SRC_RUN, name[3:])
        ad.kind = 'lora'
        return ad
    state = torch.load(out / 'adapters' / f'{name}.pt', map_location='cpu', weights_only=False)
    method = state['method']
    ns = Namespace(rank=state.get('rank', 8), alpha=state.get('alpha', 32.0), width=state.get('m', 0),
                   seed=state.get('seed', 1))
    ad = build_adapter(rt, method, state['scope'], ns)
    ad.load(state)
    ad.attach()
    return ad


def stage_train(rt, out: Path, args) -> None:
    (out / 'adapters').mkdir(exist_ok=True)
    apath = out / 'adapters' / f'{args.name}.pt'
    if apath.exists():
        print(f'{apath} exists; skipping training', flush=True)
        return
    src = Path(args.src_run)
    odd = io_.read_items(src / 'train_odd.jsonl')[:args.n_odd]
    vqa = io_.read_items(src / 'train_vqa.jsonl')[:args.n_odd]
    pool = odd + vqa
    n_visits = args.steps * args.step_samples
    rng = np.random.default_rng(lfr.SEED + args.seed)            # identical data order to 7.3 for the same seed
    order = np.concatenate([rng.permutation(len(pool)) for _ in range(math.ceil(n_visits / len(pool)))])[:n_visits]
    items = [pool[int(i)] for i in order]
    torch.manual_seed(lfr.SEED + args.seed)
    rt.model.requires_grad_(False)
    ad = build_adapter(rt, args.method, args.scope, args)
    ad.attach()
    params = list(ad.parameters())
    model = rt.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.config.use_cache = False
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = max(1, int(0.05 * args.steps))
    lr_at = lambda s: args.lr * (s + 1) / warm if s < warm else \
        args.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm)))
    prompt_fn = lambda text: qr.prompt_text(rt, 1, text)
    log = out / 'adapters' / f'{args.name}_log.jsonl'
    if log.exists():
        log.unlink()
    t0 = time.time()
    try:
        for step in range(args.steps):
            chunk = items[step * args.step_samples:(step + 1) * args.step_samples]
            tokens = [(w // 32) * (h // 32) + 120 for w, h in (Image.open(it.image).size for it in chunk)]
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
            tot = {'odd': [0.0, 0], 'vqa': [0.0, 0]}
            for mb in micro:
                ss = [chunk[i] for i in mb]
                batch = io_.encode_batch(rt.processor, prompt_fn, ss, [Image.open(s.image).convert('RGB') for s in ss])
                per, _, _ = tf.batch_loss(model, batch, rt.device)
                (per.sum() / len(chunk)).backward()
                for s, v in zip(ss, per.tolist()):
                    tot[s.task][0] += v
                    tot[s.task][1] += 1
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            row = {'step': step, 'loss_odd': tot['odd'][0] / max(1, tot['odd'][1]),
                   'loss_vqa': tot['vqa'][0] / max(1, tot['vqa'][1]), 'lr': lr_at(step), 'grad_norm': float(gn),
                   'time': time.time() - t0}
            append_jsonl(log, [row])
            if step % 10 == 0:
                print(json.dumps(row), flush=True)
    finally:
        model.gradient_checkpointing_disable()
        model.eval()
        ad.detach()
    state = ad.state()
    if args.method == 'lora':
        state['method'] = 'lora'
    state.update({'method': args.method, 'scope': args.scope, 'name': args.name, 'n_odd': args.n_odd,
                  'seed': args.seed, 'lr': args.lr, 'steps': args.steps, 'n_params': ad.n_params(),
                  'delta_norms': ad.delta_norms(), 'train_seconds': time.time() - t0,
                  'gpu': torch.cuda.get_device_name(), 'runtime': qr.describe(rt)})
    if args.method == 'width':
        state['m'] = args.width
    torch.save(state, apath)
    print(f'saved {apath} ({time.time() - t0:.0f}s, {ad.n_params()} params)', flush=True)


# ----------------------------------------------------------------------------- probe / general

def stage_probe(rt, out: Path, args) -> None:
    d = out / 'eval' / args.name
    d.mkdir(parents=True, exist_ok=True)
    if (d / 'probe.jsonl').exists():
        return
    ad = load_adapter(rt, out, args.name)
    t0 = time.time()
    try:
        items = io_.read_items(Path(args.src_run) / 'probe.jsonl')
        res = ior.run_items(rt, ior._orig_generate_batch, items, args.token_budget, 32)
        append_jsonl(d / 'probe.jsonl', [{'id': it.id, 'pair': it.meta['pair'], 'variant': it.meta['variant'],
                                          'split': it.meta['split'], 'family': it.meta['family'],
                                          'target': it.truth['target'], 'text': res[it.id]['text'],
                                          'batch_size': res[it.id]['batch_size'],
                                          **oe.score({'target': it.truth['target']}, res[it.id])} for it in items])
    finally:
        if ad is not None:
            ad.detach()
    print(f'probe {args.name} {time.time() - t0:.0f}s', flush=True)


def stage_general(rt, out: Path, args) -> None:
    """7.3 stage_eval (coord, count, general + text) with this round's adapter loader and output directory."""
    orig = lfr.load_adapter
    lfr.load_adapter = lambda rt_, out_, name: load_adapter(rt_, out, name)
    try:
        lfr.stage_eval(rt, out, Namespace(**{**vars(args), 'sets': 'count,coord,general',
                                             'interp_run': args.interp_run, 'finetune_run': args.finetune_run}))
    finally:
        lfr.load_adapter = orig


# ----------------------------------------------------------------------------- detect (representation change)

@torch.no_grad()
def stage_detect(rt, out: Path, args) -> None:
    """For one adapter (or base): (a) exclusive-detector counts of existing neurons at all blocks (homA / homB);
    (b) per-new-neuron zeta (width); (c) on feature-absent probe images of ANALYSIS_PAIRS, the per-neuron split of
    the target-specific write v^T(dO_t - mean dO_d) at every tuned block, and the response distribution of the
    written direction; for 'base' the same split for the presence signal (std images) at blocks 7-27."""
    d = out / 'eval' / args.name
    d.mkdir(parents=True, exist_ok=True)
    visual = rt.model.model.visual
    nb = len(visual.blocks)
    ad = load_adapter(rt, out, args.name)
    res: dict = {}
    try:
        if not (d / 'detectors.json').exists():
            (d / 'detectors.json').write_text(json.dumps(lfr.detector_stats(rt, lf.PROBE_PAIRS, nb, False), indent=1))
        apath = (SRC_RUN / 'adapters' / f'{args.name[3:]}.pt') if args.name.startswith('r1_') else \
            (out / 'adapters' / f'{args.name}.pt')
        scope = torch.load(apath, map_location='cpu', weights_only=False)['scope'] if ad is not None else ''
        tuned = sorted(lf.parse_scope(scope)['blocks']) if ad is not None and scope != 'vit_merger' else []
        if scope in ('vit_all', 'model_all'):
            tuned = [7, 13, 14, 15, 19, 25]
        res['context_zeta'] = _context_zeta(rt, ad, tuned if ad is not None else BASE_BLOCKS)
        # (b) new neurons: zeta on homogeneous images (entity max over patches), both directions
        if ad is not None and ad.kind == 'width':
            ad.record = True
            res['new_neuron_zeta'] = {}
            for pair in lf.PROBE_PAIRS:
                quads = _quads(pair)
                _, ent_patch, n_patch = _meta(quads)
                pidx = torch.tensor(ent_patch, device=rt.device)
                acts = {}
                for v in ('homA', 'homB'):
                    acc = {n: [] for n in ad.names}
                    for i0 in range(0, N_QUADS, 6):
                        ims = [q['images'][v][0] for q in quads[i0:i0 + 6]]
                        lfr.vit_forward(rt, ims)
                        for n in ad.names:
                            acc[n].append(ad.acts[n].reshape(len(ims), n_patch, -1)[:, pidx].amax(2).flatten(0, 1))
                    acts[v] = {n: torch.cat(acc[n]) for n in ad.names}
                row = {}
                for n in ad.names:
                    A, B = acts['homA'][n], acts['homB'][n]
                    row[n] = {'B|A': lf.zeta(B.mean(0).cpu().numpy(), A.mean(0).cpu().numpy(), A.std(0).cpu().numpy()).tolist(),
                              'A|B': lf.zeta(A.mean(0).cpu().numpy(), B.mean(0).cpu().numpy(), B.std(0).cpu().numpy()).tolist(),
                              'muA': A.mean(0).tolist(), 'muB': B.mean(0).tolist()}
                res['new_neuron_zeta'][pair] = row
            ad.record = False
        # (c) per-neuron split of the target-specific write (MLP-scoped adapters and base)
        if ad is None or (tuned and getattr(ad, 'kind', '') in ('lora', 'full', 'width') and scope.startswith('vit_mlp')):
            res['split'] = _write_split(rt, out, args, ad, tuned)
    finally:
        if ad is not None:
            ad.detach()
    (d / 'detect.json').write_text(json.dumps(res))
    print(f'detect {args.name} done', flush=True)


BASE_BLOCKS = [7, 10, 13, 14, 15, 19, 25]


@torch.no_grad()
def _context_zeta(rt, ad, blocks: list[int]) -> dict:
    """Per-neuron selectivity for the odd target inside an image: on the 12 swap (absent-target) and 12 std
    (present-target) images of every probe pair, a_e = max over the entity's 16 patches of a neuron's activation
    (existing neurons: input of linear_fc2, including the adapter's change; new neurons: WidthExtension output) and
    zeta_swap = median over images of (a_t - mean_d a_d) / (sd_d a_d + 0.01).  A neuron with zeta_swap > 5 is a
    contextual exclusive detector: it fires on the odd entity and not on its neighbours in the same image."""
    visual = rt.model.model.visual
    store: dict = {}
    hs = [visual.blocks[b - 1].mlp.linear_fc2.register_forward_pre_hook(
        lambda m, a, b=b: store.__setitem__(b, a[0].detach())) for b in blocks]
    if ad is not None and getattr(ad, 'kind', '') == 'width':
        ad.record = True
    res: dict = {}
    try:
        for pair in lf.PROBE_PAIRS:
            quads = _quads(pair)
            meta = quads[0]['images']['swap'][1]
            ent_patch = lf.patch_index(oe.entity_tokens(meta))
            pidx = torch.tensor(ent_patch, device=rt.device)
            n_patch = 4 * meta['merged_grid'][0] * meta['merged_grid'][1]
            T = [q['target'][0] * GRID + q['target'][1] for q in quads]
            for v in ('swap', 'std'):
                acc: dict = {}
                for i0 in range(0, N_QUADS, 6):
                    ims = [q['images'][v][0] for q in quads[i0:i0 + 6]]
                    lfr.vit_forward(rt, ims)
                    for b in blocks:
                        acc.setdefault(('old', b), []).append(store[b].reshape(len(ims), n_patch, -1)[:, pidx].float().amax(2))
                        if ad is not None and getattr(ad, 'kind', '') == 'width' and f'V{b}.mlp' in ad.acts:
                            acc.setdefault(('new', b), []).append(
                                ad.acts[f'V{b}.mlp'].reshape(len(ims), n_patch, -1)[:, pidx].float().amax(2))
                    store.clear()
                for (kind, b), xs in acc.items():
                    X = torch.cat(xs)                                         # (12, n_ent, n_units)
                    zs = []
                    for i, t in enumerate(T):
                        o = torch.cat([X[i, :t], X[i, t + 1:]])
                        zs.append((X[i, t] - o.mean(0)) / (o.std(0) + 0.01))
                    z = torch.stack(zs).median(0).values.cpu().numpy()
                    top = np.argsort(z)[::-1][:5]
                    res[f'{pair}|{v}|{kind}|{b}'] = {'n_gt5': int((z > 5).sum()), 'max': float(z.max()),
                                                     'top5': z[top].tolist(), 'top5_idx': top.tolist(),
                                                     'z': z.astype(np.float16).tolist() if kind == 'new' else None}
            print(f'[context] {pair}', flush=True)
    finally:
        for h in hs:
            h.remove()
        if ad is not None and getattr(ad, 'kind', '') == 'width':
            ad.record = False
    return res


def _write_split(rt, out: Path, args, ad, tuned: list[int]) -> dict:
    """Per tuned block (base: blocks 7, 13, 19, 25 for the presence signal), on probe images of ANALYSIS_PAIRS.

    The adapter's change of the block-k MLP output at entity e (16-patch mean) is dO_e = O_tuned - O_base with the
    same block input (the input itself comes from the tuned model, so dO_e isolates the block's own write).  The
    target-specific direction v is mean_i(dO_t - mean_{e != t} dO_e) on even images; on odd images the projection
    difference v^T(dO_t - mean dO_d) is split exactly into
      existing neuron j (fc1 path):  (v^T W2[:, j]) (da_j,t - mean da_j,d),  da = GELU change of neuron j,
      existing neuron j (fc2 LoRA/full path): (v^T dW2[:, j]) (a_j,t - mean a_j,d),  a = tuned activation,
      new neuron i (width):           (v^T W_out[:, i]) (n_i,t - mean n_i,d).
    For base (no adapter) the presence signal of std images is split with dO = O_base (v from std images):
      (v^T W2[:, j]) (a_j,t - mean a_j,d)."""
    visual = rt.model.model.visual
    probe = [it for it in io_.read_items(Path(args.src_run) / 'probe.jsonl') if it.meta['pair'] in ANALYSIS_PAIRS]
    variant = 'swap' if ad is not None else 'std'
    items = [it for it in probe if it.meta['variant'] == variant]
    blocks = tuned if ad is not None else [7, 13, 19, 25]
    store: dict = {}
    hs = []
    for b in blocks:
        mlp = visual.blocks[b - 1].mlp
        hs.append(mlp.register_forward_pre_hook(lambda m, a, b=b: store.__setitem__(('x', b), a[0].detach())))
    rows = {b: {'t': [], 'd': [], 'entity': []} for b in blocks}
    try:
        for i0 in range(0, len(items), 12):
            chunk = items[i0:i0 + 12]
            imgs = [Image.open(it.image).convert('RGB') for it in chunk]
            meta = chunk[0].meta['meta']
            pidx = torch.tensor(lf.patch_index(oe.entity_tokens(meta)), device=rt.device)
            n_patch = 4 * meta['merged_grid'][0] * meta['merged_grid'][1]
            lfr.vit_forward(rt, imgs)               # tuned model (or base): block inputs
            for b in blocks:
                mlp = visual.blocks[b - 1].mlp
                x = store[('x', b)]
                pre = mlp.linear_fc1(x).float()
                a_base = mlp.act_fn(pre)
                parts = {}
                if ad is None:
                    parts['base'] = a_base
                elif ad.kind == 'lora':
                    k1 = ad._key(f'V{b}.mlp.linear_fc1')
                    pre_t = pre + ad.scale * (x.float() @ ad.A[k1].t()) @ ad.B[k1].t()
                    a_t = mlp.act_fn(pre_t)
                    parts['fc1'] = a_t - a_base
                    parts['fc2'] = a_t
                elif ad.kind == 'full':
                    k1 = ad._key(f'V{b}.mlp.linear_fc1')
                    pre_t = pre + x.float() @ ad.W[k1].t() + ad.b[k1]
                    a_t = mlp.act_fn(pre_t)
                    parts['fc1'] = a_t - a_base
                    parts['fc2'] = a_t
                else:
                    k = ad._key(f'V{b}.mlp')
                    parts['new'] = mlp.act_fn(x.float() @ ad.W_in[k].t() + ad.b_in[k])
                for j, it in enumerate(chunk):
                    t = it.truth['target'][0] * GRID + it.truth['target'][1]
                    sl = slice(j * n_patch, (j + 1) * n_patch)
                    ent = {name: v[sl][pidx].mean(1) for name, v in parts.items()}       # (n_ent, n_units)
                    rows[b]['entity'].append({name: v.cpu() for name, v in ent.items()})
                    rows[b]['t'].append(t)
            store.clear()
    finally:
        for h in hs:
            h.remove()
    res = {}
    for b in blocks:
        visualb = visual.blocks[b - 1].mlp
        W2 = visualb.linear_fc2.weight.float().cpu()
        cols = {'base': W2, 'fc1': W2}
        if ad is not None and ad.kind == 'lora':
            k2 = ad._key(f'V{b}.mlp.linear_fc2')
            cols['fc2'] = (ad.scale * ad.B[k2].float() @ ad.A[k2].float()).detach().cpu()
        if ad is not None and ad.kind == 'full':
            cols['fc2'] = ad.W[ad._key(f'V{b}.mlp.linear_fc2')].detach().float().cpu()
        if ad is not None and ad.kind == 'width':
            cols['new'] = ad.W_out[ad._key(f'V{b}.mlp')].detach().float().cpu()
        E, T = rows[b]['entity'], rows[b]['t']
        names = list(E[0])
        # write per entity: dO_e = sum over parts of cols[part] @ ent[part]
        dO = [sum(cols[nm] @ e[nm].T for nm in names).T for e in E]               # (n_ent, d) per image
        fit, test = range(0, len(E), 2), range(1, len(E), 2)
        diff = torch.stack([dO[i][T[i]] - torch.cat([dO[i][:T[i]], dO[i][T[i] + 1:]]).mean(0) for i in fit]).mean(0)
        v = diff / diff.norm()
        proj = {nm: (v @ cols[nm]).numpy() for nm in names}                         # (n_units,)
        contrib = {nm: [] for nm in names}
        zs, resp_t, resp_d = [], [], []
        for i in test:
            s_ = dO[i] @ v
            o = torch.cat([s_[:T[i]], s_[T[i] + 1:]])
            zs.append(float((s_[T[i]] - o.mean()) / (o.std() + 1e-6)))
            resp_t.append(float(s_[T[i]]))
            resp_d += o.tolist()
            for nm in names:
                e = E[i][nm].numpy()
                contrib[nm].append(dfm.neuron_contributions(proj[nm], e[T[i]], np.delete(e, T[i], 0).mean(0)))
        c_all = np.concatenate([np.mean(contrib[nm], 0) for nm in names])
        res[str(b)] = {'zeta': float(np.median(zs)), 'n_eff': dfm.effective_count(c_all),
                       'top10_share': dfm.top_share(c_all, 10), 'top100_share': dfm.top_share(c_all, 100),
                       'part_totals': {nm: float(np.mean(contrib[nm], 0).sum()) for nm in names},
                       'contrib': {nm: np.mean(contrib[nm], 0).astype(np.float32).tolist() for nm in names},
                       'resp_t': resp_t, 'resp_d': resp_d[:4000]}
    return res


# ----------------------------------------------------------------------------- badcase (dose probe)

def stage_badcase(rt, out: Path, args) -> None:
    """Feature-absent targets at four part sizes per held-out pair, every cell of a 6 x 6 grid once per size."""
    d = out / 'eval' / args.name
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'badcase.jsonl'
    if path.exists():
        return
    items_path = out / 'badcase_items.jsonl'
    if not items_path.exists():
        rng = np.random.default_rng(lfr.SEED + 7)
        items, k = [], 0
        for pair, levels in DOSE_PAIRS.items():
            for lv in levels:
                for r in range(GRID):
                    for c in range(GRID):
                        it = lfr.odd_item(out, 'dose', k, pair, GRID, (r, c), 'swap', float(lv), rng)
                        it.meta['level'] = lv
                        items.append(it)
                        k += 1
        io_.dump_items(items_path, items)
    items = io_.read_items(items_path)
    ad = load_adapter(rt, out, args.name)
    try:
        res = ior.run_items(rt, ior._orig_generate_batch, items, args.token_budget, 32)
    finally:
        if ad is not None:
            ad.detach()
    append_jsonl(path, [{'id': it.id, 'pair': it.meta['pair'], 'level': it.meta['level'], 'target': it.truth['target'],
                         'text': res[it.id]['text'], **oe.score({'target': it.truth['target']}, res[it.id])} for it in items])
    print(f'badcase {args.name} done', flush=True)


# ----------------------------------------------------------------------------- Qwen3-VL (mechanism comparison)

@torch.no_grad()
def stage_vl(rt, out: Path, args) -> None:
    """Detector counts and absence separability in the ViT of the loaded model (run with --model Qwen3-VL-8B),
    plus separability of the absent target among distractors at the final merger and at each DeepStack merger."""
    d = out / 'vl' / args.model
    d.mkdir(parents=True, exist_ok=True)
    visual = rt.model.model.visual
    nb = len(visual.blocks)
    if not (d / 'detectors.json').exists():
        (d / 'detectors.json').write_text(json.dumps(lfr.detector_stats(rt, lf.PROBE_PAIRS, nb, True), indent=1))
    ds_idx = list(getattr(visual, 'deepstack_visual_indexes', []) or [])
    store: dict = {}
    hs = [visual.merger.register_forward_hook(lambda m, a, o: store.__setitem__('merger', o.detach()))]
    for j, mod in enumerate(getattr(visual, 'deepstack_merger_list', []) or []):
        hs.append(mod.register_forward_hook(lambda m, a, o, j=j: store.__setitem__(f'deepstack{j}', o.detach())))
    res = {'deepstack_blocks_1based': [i + 1 for i in ds_idx], 'pairs': {}}
    try:
        for pair in lf.PROBE_PAIRS:
            quads = _quads(pair)
            meta = quads[0]['images']['swap'][1]
            tok = torch.tensor(oe.entity_tokens(meta), device=rt.device)            # (n_ent, 4)
            n_tok = meta['merged_grid'][0] * meta['merged_grid'][1]
            sal: dict = {}
            for i0 in range(0, N_QUADS, 6):
                part = quads[i0:i0 + 6]
                lfr.vit_forward(rt, [q['images']['swap'][0] for q in part])
                for key, h in store.items():
                    for j, q in enumerate(part):
                        e = h[j * n_tok:(j + 1) * n_tok][tok].float().mean(1)
                        t = q['target'][0] * GRID + q['target'][1]
                        sal.setdefault(key, []).append(lfr.torch_salience(e, t))
                store.clear()
            res['pairs'][pair] = {k: float(np.median(v)) for k, v in sal.items()}
            print(f'[vl] {pair} {res["pairs"][pair]}', flush=True)
    finally:
        for h in hs:
            h.remove()
    (d / 'taps.json').write_text(json.dumps(res, indent=1))


STAGES = {'readout': stage_readout, 'oddness': stage_oddness, 'relocate': stage_relocate, 'train': stage_train, 'probe': stage_probe, 'general': stage_general,
          'detect': stage_detect, 'badcase': stage_badcase, 'vl': stage_vl}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--src-run', required=True, help='7.3 run directory (data, probe, locate delta, base results)')
    p.add_argument('--stages', required=True)
    p.add_argument('--name', default='base')
    p.add_argument('--method', default='lora', choices=('lora', 'full', 'width'))
    p.add_argument('--scope', default='')
    p.add_argument('--width', type=int, default=0)
    p.add_argument('--n-odd', type=int, default=1200)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=32.0)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--step-samples', type=int, default=16)
    p.add_argument('--micro-tokens', type=int, default=3072)
    p.add_argument('--token-budget', type=int, default=16000)
    p.add_argument('--interp-run', default='')
    p.add_argument('--finetune-run', default='')
    p.add_argument('--data-dir', default='')
    p.add_argument('--vcot-gqa', default='')
    p.add_argument('--mmvp-csv', default='')
    p.add_argument('--general-n', type=int, default=200)
    p.add_argument('--general-seed', type=int, default=20261020)
    p.add_argument('--general-batch', type=int, default=16)
    p.add_argument('--general-max-new', type=int, default=96)
    p.add_argument('--bv-batch', type=int, default=11)
    p.add_argument('--bv-max-new', type=int, default=1536)
    p.add_argument('--max-memory', default='')
    args = p.parse_args()
    torch.set_num_threads(16)
    global SRC_RUN
    SRC_RUN = Path(args.src_run)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if 'train' in args.stages and not args.scope:
        raise ValueError('--scope is required for training')
    if args.method == 'width' and 'train' in args.stages and args.width <= 0:
        raise ValueError('--width must be positive for method width')
    rt = tfr.load_model(args)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages, 'name': args.name,
                         'gpu': torch.cuda.get_device_name(), 'runtime': qr.describe(rt), 'args': vars(args),
                         'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]})
    meta.write_text(json.dumps(info, indent=1))
    names = args.name.split(',')          # several adapters may share one model load (analysis stages)
    if len(names) > 1 and 'train' in args.stages:
        raise ValueError('train takes a single --name')
    for name in names:
        args.name = name
        for s in args.stages.split(','):
            t0 = time.time()
            STAGES[s](rt, out, args)
            print(f'stage {s} ({name}) done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
