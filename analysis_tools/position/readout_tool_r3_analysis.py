"""Experiment 3.9 round 3 analysis on the round-2 capture (no new model runs).

  annotate  RefCOCO segmentation masks and other annotated objects of every RefCOCO test case
            (for salient-part, clutter, occlusion and selection-error analyses).
  zeroshot  zero-training attention decoders on every head: the experiment-3.11 peak decoder
            (3x3 neighbourhood centroid of the argmax token), the full-map centroid, and box decoders
            from the attention extent or from content similarity grown from the attention peak.
            Thresholds are chosen on the validation split.  Also the position of the attention peak
            inside the RefCOCO target (salient-part bias).
  fit       convolutional pointer readers with reduced inputs (selected heads, content layers,
            extent channel, rough point) trained exactly like the round-2 tool.
  evaluate  paired reader-vs-model metrics per test set, accuracy against input cost, and RefCOCO
            failure modes (object size, clutter, occlusion, selection errors, box extent).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position import readout_tool_r2 as r2
from analysis_tools.position import readout_tool_r2_analysis as a2
from analysis_tools.position import readout_tool_r3 as r3
from analysis_tools.position.coordinate_readout_run import read_jsonl
from analysis_tools.position.readout_tool_analysis import mcnemar_exact, wilson

H11 = r3.head_index(20, 11)
FINAL_R2 = 'cnn_slot_content_approx'


def load_all(run: Path) -> tuple[dict, dict, dict, np.ndarray]:
    cases = {c['id']: c for c in read_jsonl(run / 'cases.jsonl')}
    t0 = time.time()
    cap = a2.load_capture(run, cases)
    print(f'loaded {len(cap["ids"])} cases in {time.time() - t0:.0f} s', flush=True)
    S = a2.splits(cap, cases)
    Y = np.stack([rt_.targets_units(cases[i]['truth']) for i in cap['ids']])
    return cases, cap, S, Y


def hit_of(truth: dict, p: np.ndarray) -> bool:
    return bool(rt_.inside(rt_.to_pixels(p, truth['canvas']), truth['hit']))


def stage_edges(run: Path, out: Path) -> None:
    """Zero-training edge-group decoder (3.11 round 2) at the tool slot: centre, box IoU and hit per test set."""
    cases, cap, S, Y = load_all(run)
    ids = cap['ids']
    tl = r3.parse_heads(','.join(r3.EDGE_TL))
    br = r3.parse_heads(','.join(r3.EDGE_BR))
    pairs = {'groups': (tl, br), 'L24H7+L24H5': (r3.parse_heads('L24H7'), r3.parse_heads('L24H5')),
             'L24H7,L20H0+L24H5,L20H10': (r3.parse_heads('L24H7,L20H0'), r3.parse_heads('L24H5,L20H10'))}
    use = np.nonzero(S['val'] | S['test'])[0]
    boxes = {k: np.full((len(ids), 4), np.nan) for k in pairs}
    for i in use:
        for k, (a, b) in pairs.items():
            boxes[k][i] = r3.edge_group_box(cap['maps_slot'][i], cap['grid'][i], a, b)
    res = {'groups': {'top_left': list(r3.EDGE_TL), 'bottom_right': list(r3.EDGE_BR)}, 'by_set': {}}
    for s in a2.EVAL_SETS:
        m = S['test'] & (S['sets'] == s)
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        row = {'n': int(m.sum())}
        for k, B in boxes.items():
            C = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], 1)
            hits = [hit_of(cases[ids[i]]['truth'], C[i]) for i in idx]
            e = np.linalg.norm(C[idx] - Y[idx, :2], axis=1)
            row[k] = {'hit': float(np.mean(hits)), 'hit_ci': wilson(int(np.sum(hits)), len(hits)), 'err_median': float(np.median(e)),
                      'err_mean': float(e.mean()), 'iou': float(np.mean([r3.box_iou_units(B[i], Y[i, 2:]) for i in idx]))}
        res['by_set'][s] = row
    np.savez(out / 'edges_points.npz', ids=np.asarray(ids), **{f'box_{k}': v for k, v in boxes.items()})
    (out / 'edges.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({s: {k: v for k, v in r.items()} for s, r in res['by_set'].items()}, indent=0, default=float)[:5000], flush=True)


# ----------------------------------------------------------------------------- annotate

def stage_annotate(run: Path, out: Path, refcoco: Path) -> None:
    cases = [c for c in read_jsonl(run / 'cases.jsonl') if c['set'] == 'refcoco' and c['split'] in ('test', 'val')]
    qids = {c['id'].split('_')[1] for c in cases}
    ann = r3.refcoco_annotations(refcoco, qids)
    rows = {}
    for c in cases:
        a = ann[c['id'].split('_')[1]]
        src = tuple(c['source_size'])
        W, H = c['truth']['canvas']
        mask = r3.polygon_mask(a['segmentation'], src, (W, H))
        if mask.sum() == 0:
            raise ValueError(f'empty segmentation for {c["id"]}')
        yy, xx = np.nonzero(mask)
        target = rt_.to_units(c['truth']['box'], (W, H))
        others = [r3.xywh_to_units(b, src) for b in a['others']]
        gh, gw = rt_.grid_of((W, H))
        # mask coverage of each merged token cell (32 x 32 pixels) for pointer-on-object analyses
        cov = mask[:gh * 32, :gw * 32].reshape(gh, 32, gw, 32).mean((1, 3)).reshape(-1)
        rows[c['id']] = {'mask_centroid': rt_.to_units([xx.mean(), yy.mean()], (W, H)).tolist(),
                         'mask_frac_of_box': float(mask.sum() / max((c['truth']['box'][2] - c['truth']['box'][0]) *
                                                                    (c['truth']['box'][3] - c['truth']['box'][1]), 1)),
                         'others': [o.tolist() for o in others], 'n_others': len(others),
                         'occlusion': r3.occlusion_fraction(target, others), 'token_mask': cov.round(3).tolist(),
                         'subset': c['subset']}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'refcoco_annotations.json').write_text(json.dumps(rows))
    print(f'annotated {len(rows)} RefCOCO cases', flush=True)


# ----------------------------------------------------------------------------- zeroshot

def peak_points_all_heads(maps: np.ndarray, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """maps (H, Nv) -> peak-3x3 centroid (H, 2) and full-map centroid (H, 2), vectorized over heads."""
    gh, gw = grid
    H = maps.shape[0]
    a = maps.astype(np.float64).reshape(H, gh, gw)
    pad = np.pad(a, ((0, 0), (1, 1), (1, 1)))
    k = a.reshape(H, -1).argmax(1)
    r, c = k // gw, k % gw
    dr, dc = np.mgrid[0:3, 0:3]
    win = pad[np.arange(H)[:, None, None], r[:, None, None] + dr, c[:, None, None] + dc]       # (H, 3, 3)
    cw, ch = r3.cell_size(grid)
    xs = (c[:, None, None] + dc - 1 + 0.5) * cw
    ys = (r[:, None, None] + dr - 1 + 0.5) * ch
    s = win.sum((1, 2))
    peak = np.stack([(win * xs).sum((1, 2)) / s, (win * ys).sum((1, 2)) / s], 1)
    cen = r2.token_centers(grid)
    w = a.reshape(H, -1)
    mean = (w @ cen) / w.sum(1, keepdims=True)
    return peak, mean


def stage_zeroshot(run: Path, out: Path) -> None:
    cases, cap, S, Y = load_all(run)
    ids = cap['ids']
    use = np.nonzero(S['val'] | S['test'])[0]
    nh = 128
    peak = np.full((len(ids), nh, 2), np.nan)
    mean = np.full((len(ids), nh, 2), np.nan)
    peak_te = np.full((len(ids), 2), np.nan)
    for i in use:
        peak[i], mean[i] = peak_points_all_heads(cap['maps_slot'][i], cap['grid'][i])
        peak_te[i] = r3.peak_centroid(cap['maps_target_end'][i][H11], cap['grid'][i])
    sets = S['sets']
    groups = np.asarray([a2.val_group(s) for s in sets])
    err = lambda P, m: np.linalg.norm(P[m] - Y[m, :2], axis=-1)
    # per-head validation error (mean over the three validation groups, as for reader selection)
    val_err = {}
    for name, P in (('peak', peak), ('mean', mean)):
        v = np.stack([np.nanmean(np.linalg.norm(P[S['val'] & (groups == g)] - Y[S['val'] & (groups == g), None, :2], axis=-1), 0)
                      for g in a2.VAL_GROUPS])                                                          # (3, 128)
        val_err[name] = {'by_group': v.tolist(), 'mean': v.mean(0).tolist()}
    rank = np.argsort(val_err['peak']['mean'])
    best = int(rank[0])
    print('best heads (peak, val):', [(r3.head_name(int(h)), round(val_err['peak']['mean'][h], 1)) for h in rank[:12]], flush=True)
    # box decoders for H11 and the best head: thresholds chosen on validation IoU (mean over groups)
    c16 = cap['content_16']
    sample = np.concatenate([c16[i][::5] for i in np.nonzero(S['train'])[0][::7]]).astype(np.float32)
    cmu, csd = sample.mean(0), sample.std(0) + 1e-3

    def boxes(i, h, kind, tau):
        m = cap['maps_slot'][i][h].astype(np.float32)
        g = cap['grid'][i]
        if kind == 'attn':
            reg = r3.attention_extent(m, g, tau)
        else:
            reg = r3.content_extent(m, (c16[i].astype(np.float32) - cmu) / csd, g, tau)
        return r3.region_box(reg, g)

    taus = {'attn': [0.05, 0.1, 0.2, 0.3, 0.5], 'content': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
    val_idx = np.nonzero(S['val'])[0]
    chosen, box_pred = {}, {}
    for h in sorted({H11, best}):
        for kind, grid_t in taus.items():
            scores = []
            for tau in grid_t:
                ious = np.asarray([r3.box_iou_units(boxes(i, h, kind, tau), Y[i, 2:]) for i in val_idx])
                scores.append(np.mean([ious[groups[val_idx] == g].mean() for g in a2.VAL_GROUPS]))
            tau = grid_t[int(np.argmax(scores))]
            chosen[f'{r3.head_name(h)}_{kind}'] = {'tau': tau, 'val_iou_by_tau': dict(zip(map(str, grid_t), scores))}
            B = np.full((len(ids), 4), np.nan)
            for i in use:
                B[i] = boxes(i, h, kind, tau)
            box_pred[f'{r3.head_name(h)}_{kind}'] = B
    # test summary per set
    test = S['test']
    res = {'best_head_peak': r3.head_name(best), 'rank_peak': [r3.head_name(int(h)) for h in rank[:20]],
           'val_err': val_err, 'box_thresholds': chosen, 'by_set': {}}
    decoders = {'H11 peak (slot)': peak[:, H11], 'H11 peak (target_end)': peak_te, 'H11 map mean (slot)': mean[:, H11],
                f'{r3.head_name(best)} peak (slot)': peak[:, best]}
    for k, B in box_pred.items():
        decoders[f'{k} box centre'] = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], 1)
    for s in a2.EVAL_SETS:
        m = test & (sets == s)
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        row = {'n': int(m.sum())}
        for name, P in decoders.items():
            hits = [hit_of(cases[ids[i]]['truth'], P[i]) for i in idx]
            e = err(P, m)
            row[name] = {'hit': float(np.mean(hits)), 'hit_ci': wilson(int(np.sum(hits)), len(hits)),
                         'err_mean': float(e.mean()), 'err_median': float(np.median(e))}
        for k, B in box_pred.items():
            row[f'{k} box IoU'] = float(np.mean([r3.box_iou_units(B[i], Y[i, 2:]) for i in idx]))
        res['by_set'][s] = row
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'zeroshot_points.npz', ids=np.asarray(ids), peak=peak.astype(np.float32), mean=mean.astype(np.float32),
             peak_te=peak_te, **{f'box_{k}': v for k, v in box_pred.items()})
    (out / 'zeroshot.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({s: {k: (v['err_median'] if isinstance(v, dict) else v) for k, v in r.items()} for s, r in res['by_set'].items()},
                     indent=0, default=float)[:4000], flush=True)


# ----------------------------------------------------------------------------- reduced-input readers

# name -> (head spec, content layers, extent channel, rough point)
CONFIGS = {
    'k0_content_approx': ('', (0, 16), False, True),                   # = round-2 attention-free reader
    'k1_H11': ('L20H11', (), False, False),
    'k1_H11_content': ('L20H11', (0, 16), False, False),
    'k1_H11_content_approx': ('L20H11', (0, 16), False, True),
    'k1_H11_content_extent_approx': ('L20H11', (0, 16), True, True),
    'k2_content_approx': ('L20H11,L16H0', (0, 16), False, True),
    'k4_content_approx': ('L20H11,L20H10,L24H5,L32H8', (0, 16), False, True),
    'k4_content_extent_approx': ('L20H11,L20H10,L24H5,L32H8', (0, 16), True, True),
    'k4_c16_extent_approx': ('L20H11,L20H10,L24H5,L32H8', (16,), True, True),
    'k8_content_approx': ('L20H11,L20H10,L24H5,L32H8,L16H0,L24H12,L24H4,L24H0', (0, 16), False, True),
    'L20_content_approx': ('L20', (0, 16), False, True),
    'L16L20_content_approx': ('L16,L20', (0, 16), False, True),
    'all_content_approx': ('all', (0, 16), False, True),                # = round-2 final tool (refit, same seed)
    'all_content_extent_approx': ('all', (0, 16), True, True),
    # experiment 3.11 round 2: on RefCOCO, L24 H7 reads the box's min edge and L24 H5 the max edge
    'edge3_content_approx': ('L20H11,L24H7,L24H5', (0, 16), False, True),
    'edge3_content_extent_approx': ('L20H11,L24H7,L24H5', (0, 16), True, True),
    # experiment 3.11 round 2 (final): edge-reader groups on RefCOCO.  Top-left: L20 H0, L24 H1/H2/H7/H12;
    # bottom-right: L20 H10, L24 H5/H6/H8/H9, L28 H0.  edge5 = H11 + the two strongest heads of each group.
    'edge5': ('L20H11,L24H7,L20H0,L24H5,L20H10', (), False, False),
    'edge5_content_approx': ('L20H11,L24H7,L20H0,L24H5,L20H10', (0, 16), False, True),
    'edge12_content_approx': ('L20H11,L20H0,L24H1,L24H2,L24H7,L24H12,L20H10,L24H5,L24H6,L24H8,L24H9,L28H0', (0, 16), False, True),
    # data-driven: the four heads with the lowest zero-training peak-decoder validation error
    'rank4_content_approx': ('L24H15,L20H14,L20H2,L24H0', (0, 16), False, True),
}


def config_channels(name: str) -> dict[str, int]:
    spec, layers, extent, approx = CONFIGS[name]
    d = {'heads': len(r3.parse_heads(spec))}
    for l in layers:
        d[f'content_{l}'] = r2.PCA_DIM
    d['extent'] = int(extent)
    d['approx'] = 3 * int(approx)
    return d


def make_items_r3(cap: dict, cases: dict, name: str, cstats: dict) -> list[dict]:
    spec, layers, extent, approx = CONFIGS[name]
    rows = r3.parse_heads(spec)
    ext_rows = rows or r3.parse_heads('L20H11')            # the extent channel needs a seed map
    items = []
    for i, cid in enumerate(cap['ids']):
        g = cap['grid'][i]
        y = rt_.targets_units(cases[cid]['truth'])
        parts = []
        if rows:
            parts.append(r3.head_features(cap['maps_slot'][i], rows, g))
        for l in layers:
            parts.append(cap[f'content_{l}'][i].astype(np.float32))
        if extent:
            parts.append(r3.extent_features(cap['maps_slot'][i], ext_rows, cap['content_16'][i], *cstats['content_16']))
        if approx:
            parts.append(r2.approx_features(r2.parse_approx(cap['cont'][i]), g))
        items.append({'F': np.concatenate(parts, 1).astype(np.float16), 'grid': g, 'q': rt_.token_overlap(y[2:], g),
                      'y': y, 'id': cid})
    return items


def content_stats(cap: dict, S: dict) -> dict:
    out = {}
    for l in r2.CONTENT_LAYERS:
        x = np.concatenate([cap[f'content_{l}'][i][::5] for i in np.nonzero(S['train'])[0][::7]]).astype(np.float32)
        out[f'content_{l}'] = (x.mean(0), x.std(0) + 1e-3)
    return out


def flip_negate(name: str) -> tuple[int, ...]:
    d = config_channels(name)
    if not d['approx']:
        return ()
    return (sum(d.values()) - 3,)                           # the dx channel of the rough-point group


def stage_fit(run: Path, out: Path, device: str, only: list[str], tag: str) -> None:
    cases, cap, S, Y = load_all(run)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ids.json').write_text(json.dumps(cap['ids']))
    cstats = content_stats(cap, S)
    log_path, preds_path = out / f'readers_{tag}.json', out / f'predictions_{tag}.npz'
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    preds = dict(np.load(preds_path)) if preds_path.exists() else {}
    sets = S['sets']
    for full in only:
        # 'config@s1' = same config, training seed 1; 'config@nocc' = without the clutter and chart_ui
        # training scenes (the families experiment 3.13 holds out)
        name, _, variant = full.partition('@')
        seed = int(variant[1:]) if variant.startswith('s') else 0
        if variant not in ('', 'nocc') and not variant.startswith('s'):
            raise ValueError(f'unknown variant {full}')
        if full in log:
            print(f'{name}: done', flush=True)
            continue
        items = make_items_r3(cap, cases, name, cstats)
        keep = S['train'] & ~np.isin(sets, ('g_clutter', 'g_chart_ui')) if variant == 'nocc' else S['train']
        train = [it for it, t in zip(items, keep) if t]
        vkeep = S['val'] & ~np.isin(sets, ('g_clutter', 'g_chart_ui')) if variant == 'nocc' else S['val']
        val = [it for it, v in zip(items, vkeep) if v]
        t1 = time.time()
        print(f'fit {full}: {len(train)} train, {len(val)} val, {items[0]["F"].shape[1]} channels', flush=True)
        params, hist = r2.fit_grid_pointer(train, val, seed=seed, device=device, flip_negate=flip_negate(name),
                                           log=lambda s: print(s, flush=True))
        params['kind'] = 'cnn'
        params['config'] = CONFIGS[name]
        m = r2.load_grid_pointer(params).to(device)
        P = r2.predict_grid_pointer(m, params['mean'], params['std'], items, device)
        preds[full] = P
        log[full] = {'config': CONFIGS[name], 'seed': seed, 'channels': config_channels(name), 'n_train': len(train),
                     'fit_seconds': time.time() - t1, 'n_params': int(sum(p.numel() for p in m.parameters())),
                     'val': a2.val_metric(P[S['val']], Y[S['val']], sets[S['val']]),
                     'test': a2.errors_by_set(P[S['test']], Y[S['test']], sets[S['test']]),
                     'best_epoch': hist.get('best_epoch'), 'val_curve': hist.get('val_error_by_epoch')}
        torch.save(params, out / f'{full}.pt')
        np.savez(preds_path, **preds)
        log_path.write_text(json.dumps(log, indent=1))
        print(full, json.dumps({'val': log[full]['val']}), flush=True)
        del items, train, val


# ----------------------------------------------------------------------------- evaluate

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run', type=Path, required=True, help='round-2 run directory (capture, cases, own answers)')
    p.add_argument('--r1', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='round-3 output directory')
    p.add_argument('--stage', required=True)
    p.add_argument('--refcoco', type=Path, default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--only', default='')
    p.add_argument('--tag', default='main')
    a = p.parse_args()
    if a.stage == 'annotate':
        stage_annotate(a.run, a.out, a.refcoco)
    elif a.stage == 'zeroshot':
        stage_zeroshot(a.run, a.out)
    elif a.stage == 'edges':
        stage_edges(a.run, a.out)
    elif a.stage == 'fit':
        stage_fit(a.run, a.out, a.device, [s for s in a.only.split(',') if s] or list(CONFIGS), a.tag)
    elif a.stage == 'evaluate':
        from analysis_tools.position.readout_tool_r3_eval import stage_evaluate
        stage_evaluate(a.run, a.r1, a.out)
    else:
        raise ValueError(a.stage)


if __name__ == '__main__':
    main()
