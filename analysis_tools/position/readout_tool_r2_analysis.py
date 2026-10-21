"""Experiment 3.9 round 2 analysis: fit the readers on the round-2 capture and evaluate them.

  fit       readers on the same training set (RefCOCO train + flat synthetic train + graded train):
            attention readers (round-1 per-token pointer, convolutional pointer at target_end or slot,
            with merger / decoder content features, with the model's rough point), attention-free
            readers (slot hidden-state ridge, learned query over content features, residual correction
            of the rough point), and a data ablation without the graded training scenes.  Selection
            uses the validation sets only.  Writes tool_r2/readers.json and per-reader test predictions.
  evaluate  paired reader-vs-model metrics per evaluation set (point hit, centre error, bbox IoU),
            held-out phrasing, digit statistics, free-call agreement, the dissociation experiment and
            a cost model.  Writes results_r2.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position import readout_tool_r2 as r2
from analysis_tools.position.coordinate_readout_run import read_jsonl
from analysis_tools.position.readout_tool_analysis import (ALPHAS, binom_two_sided, boot_ci, digit_stats, mcnemar_exact,
                                                           wilson)

R1_SETS = ('refcoco', 'synth', 'disc', 't6', 'layout', 'ring')
GRADED_SETS = tuple(f'g_{l}' for l in r2.LEVELS)
EVAL_SETS = R1_SETS + GRADED_SETS
VAL_GROUPS = ('refcoco', 'synth', 'graded')


def val_group(s: str) -> str:
    return 'graded' if s.startswith('g_') else s


# ----------------------------------------------------------------------------- loading

def load_capture(run: Path, cases: dict) -> dict:
    """All capture chunks -> dict of per-case lists / arrays in one order."""
    out = defaultdict(list)
    for f in sorted((run / 'features').glob('chunk_s*_*.npz')):
        d = np.load(f)
        n = len(d['ids'])
        cut = np.cumsum(np.concatenate([[0], d['map_len']]))
        out['ids'] += d['ids'].tolist()
        out['hidden'].append(d['hidden'])
        out['digit_logp'].append(d['digit_logp'])
        out['cont'] += d['cont'].tolist()
        out['grid'] += [tuple(int(v) for v in g) for g in d['grid']]
        for s in r2.SITES_R2:
            m = d[f'maps_{s}']
            out[f'maps_{s}'] += [m[:, a:b] for a, b in zip(cut[:-1], cut[1:])]
        for l in r2.CONTENT_LAYERS:
            m = d[f'content_{l}']
            out[f'content_{l}'] += [m[a:b] for a, b in zip(cut[:-1], cut[1:])]
        del n
    out['hidden'] = np.concatenate(out['hidden'])
    out['digit_logp'] = np.concatenate(out['digit_logp'])
    missing = set(cases) - set(out['ids'])
    if missing:
        raise ValueError(f'{len(missing)} cases lack captured features, e.g. {sorted(missing)[:3]}')
    for i, cid in enumerate(out['ids']):
        if tuple(rt_.grid_of(cases[cid]['truth']['canvas'])) != out['grid'][i]:
            raise ValueError(f'grid mismatch for {cid}')
    return dict(out)


def make_items(cap: dict, cases: dict, feats: tuple[str, ...]) -> list[dict]:
    """Per-case reader items with per-token features F (Nv, C) from the named feature groups:
    'attn_target_end', 'attn_slot' (log-ratio attention), 'content_0', 'content_16' (PCA), 'approx'."""
    items = []
    for i, cid in enumerate(cap['ids']):
        c = cases[cid]
        g = cap['grid'][i]
        y = rt_.targets_units(c['truth'])
        parts = []
        for f in feats:
            if f.startswith('attn_'):
                parts.append(r2.attention_features(cap[f'maps_{f[5:]}'][i], g))
            elif f.startswith('content_'):
                parts.append(cap[f][i].astype(np.float32))
            elif f == 'approx':
                parts.append(r2.approx_features(r2.parse_approx(cap['cont'][i]), g))
            else:
                raise ValueError(f)
        items.append({'F': np.concatenate(parts, 1).astype(np.float16), 'grid': g, 'q': rt_.token_overlap(y[2:], g),
                      'y': y, 'id': cid})
    return items


def splits(cap: dict, cases: dict) -> dict[str, np.ndarray]:
    sp = np.asarray([cases[i]['split'] for i in cap['ids']])
    hp = np.asarray([cases[i]['phrasing'] == 'heldout' for i in cap['ids']])
    sets = np.asarray([cases[i]['set'] for i in cap['ids']])
    return {'train': sp == 'train', 'val': sp == 'val', 'test': (sp == 'test') & ~hp, 'heldout': hp, 'sets': sets}


def errors_by_set(P: np.ndarray, Y: np.ndarray, sets: np.ndarray) -> dict:
    out = {}
    for s in np.unique(sets):
        m = sets == s
        out[str(s)] = float(np.linalg.norm(P[m, :2] - Y[m, :2], axis=1).mean())
    return out


def val_metric(P, Y, sets) -> dict:
    g = np.asarray([val_group(s) for s in sets])
    out = {k: float(np.linalg.norm(P[g == k, :2] - Y[g == k, :2], axis=1).mean()) for k in VAL_GROUPS if (g == k).any()}
    out['mean'] = float(np.mean([out[k] for k in VAL_GROUPS if k in out]))
    return out


# ----------------------------------------------------------------------------- fit

READERS = {
    # name: (kind, feature groups, flip-negated channel names)
    'ptr_r1_target_end': ('r1pointer', ('attn_target_end',)),
    'cnn_target_end': ('cnn', ('attn_target_end',)),
    'cnn_slot': ('cnn', ('attn_slot',)),
    'cnn_slot_content': ('cnn', ('attn_slot', 'content_0', 'content_16')),
    'cnn_slot_content_approx': ('cnn', ('attn_slot', 'content_0', 'content_16', 'approx')),
    'cnn_content_approx': ('cnn', ('content_0', 'content_16', 'approx')),          # no attention
}


def feature_dims(feats) -> dict:
    d = {'attn_target_end': 128, 'attn_slot': 128, 'content_0': r2.PCA_DIM, 'content_16': r2.PCA_DIM, 'approx': 3}
    return {f: d[f] for f in feats}


def flip_negate_channels(feats) -> tuple[int, ...]:
    """Signed-x channels: the approx dx offset; PCA content has no defined mirror and is left as is."""
    k, out = 0, []
    for f, n in feature_dims(feats).items():
        if f == 'approx':
            out.append(k)
        k += n
    return tuple(out)


def predict_items(name: str, params: dict, items: list[dict], device: str) -> np.ndarray:
    kind = READERS[name.split('@')[0]][0] if name.split('@')[0] in READERS else params['kind']
    if kind == 'r1pointer':
        m = rt_.PointerScorer(params['dim'], params['hidden'])
        m.load_state_dict(params['state'])
        m.eval()
        r1items = [{'F': it['F_r1'], 'centers': r2.token_centers(it['grid']), 'grid': it['grid']} for it in items]
        return rt_.pointer_predict_items(m, params['mean'], params['std'], r1items, params['tau'])
    m = r2.load_grid_pointer(params).to(device)
    return r2.predict_grid_pointer(m, params['mean'], params['std'], items, device)


def stage_fit(run: Path, device: str, only: list[str] | None = None, tag: str = 'main') -> None:
    out = run / 'tool_r2'
    out.mkdir(exist_ok=True)
    cases = {c['id']: c for c in read_jsonl(run / 'cases.jsonl')}
    t0 = time.time()
    cap = load_capture(run, cases)
    print(f'loaded {len(cap["ids"])} cases in {time.time() - t0:.0f} s', flush=True)
    (out / 'ids.json').write_text(json.dumps(cap['ids']))
    S = splits(cap, cases)
    Y = np.stack([rt_.targets_units(cases[i]['truth']) for i in cap['ids']])
    sets = S['sets']
    log_path = out / f'readers_{tag}.json'          # one file pair per fitting process (parallel fits)
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    preds_path = out / f'predictions_{tag}.npz'
    preds = dict(np.load(preds_path)) if preds_path.exists() else {}
    no_graded = np.asarray([not s.startswith('g_') for s in sets])
    plan = list(READERS) + ['cnn_slot_content_approx@no_graded', 'cnn_slot@no_graded']
    for name in plan:
        if only and name not in only:
            continue
        if name in log:
            print(f'{name}: done', flush=True)
            continue
        base, variant = (name.split('@') + [''])[:2]
        kind, feats = READERS[base]
        items = make_items(cap, cases, feats)
        tr = S['train'] & (no_graded if variant == 'no_graded' else True)
        train = [it for it, t in zip(items, tr) if t]
        val = [it for it, v in zip(items, S['val']) if v]
        t1 = time.time()
        print(f'fit {name}: {len(train)} train, {len(val)} val', flush=True)
        if kind == 'r1pointer':
            for it in items:
                nv = it['F'].shape[0]
                raw = (np.exp(it['F'].astype(np.float32).T) - 1e-3) / nv       # undo the log-ratio transform
                it['F_r1'] = rt_.token_features(np.clip(raw, 0, None), it['grid']).astype(np.float16)
                it['centers'] = r2.token_centers(it['grid'])
            r1tr = [dict(F=it['F_r1'], centers=it['centers'], q=it['q'], y=it['y'], grid=it['grid']) for it in train]
            r1va = [dict(F=it['F_r1'], centers=it['centers'], q=it['q'], y=it['y'], grid=it['grid']) for it in val]
            params, hist = rt_.fit_pointer(r1tr, r1va, seed=0)
            params['kind'] = 'r1pointer'
        else:
            params, hist = r2.fit_grid_pointer(train, val, seed=0, device=device, flip_negate=flip_negate_channels(feats),
                                               log=lambda s: print(s, flush=True))
            params['kind'] = 'cnn'
        P = predict_items(base, params, items, device)
        preds[name] = P
        log[name] = {'feats': list(feats), 'variant': variant, 'n_train': len(train), 'fit_seconds': time.time() - t1,
                     'val': val_metric(P[S['val']], Y[S['val']], sets[S['val']]),
                     'test': errors_by_set(P[S['test']], Y[S['test']], sets[S['test']]),
                     'heldout_phrasing': errors_by_set(P[S['heldout']], Y[S['heldout']], sets[S['heldout']]),
                     'best_epoch': hist.get('best_epoch'), 'val_curve': hist.get('val_error_by_epoch')}
        torch.save(params, out / f'{name}.pt')
        np.savez(preds_path, **preds)
        log_path.write_text(json.dumps(log, indent=1))
        print(name, json.dumps({'val': log[name]['val'], 'test': log[name]['test']}), flush=True)
        del items, train, val
    if not only or 'attnfree' in only:
        fit_attention_free(cap, cases, S, Y, out, preds, log, device)
        np.savez(preds_path, **preds)
        log_path.write_text(json.dumps(log, indent=1))


def fit_attention_free(cap, cases, S, Y, out, preds, log, device) -> None:
    """Hidden-state ridge at the slot, residual ridge on the rough point, the rough point itself,
    and the learned query over decoder content features."""
    sets = S['sets']
    approx = np.asarray([r2.parse_approx(c) or [np.nan, np.nan] for c in cap['cont']], float)
    valid = ~np.isnan(approx).any(1)
    A = np.where(valid[:, None], np.clip(approx, 0, 1000), 500.0)
    P = np.concatenate([A, A, A], 1)
    preds['approx_point'] = P
    log['approx_point'] = {'val': val_metric(P[S['val']], Y[S['val']], sets[S['val']]),
                           'test': errors_by_set(P[S['test']], Y[S['test']], sets[S['test']]),
                           'heldout_phrasing': errors_by_set(P[S['heldout']], Y[S['heldout']], sets[S['heldout']]),
                           'parse_rate': float(valid.mean())}
    best = None
    for k, layer in enumerate(r2.HIDDEN_LAYERS):
        X = cap['hidden'][:, k].astype(np.float32)
        m, _ = rt_.fit_ridge(X[S['train']], Y[S['train']], ALPHAS, X[S['val']], Y[S['val']])
        P = m.predict(X)
        v = val_metric(P[S['val']], Y[S['val']], sets[S['val']])
        log[f'ridge_slot_L{layer}'] = {'val': v, 'test': errors_by_set(P[S['test']], Y[S['test']], sets[S['test']]),
                                       'heldout_phrasing': errors_by_set(P[S['heldout']], Y[S['heldout']], sets[S['heldout']])}
        preds[f'ridge_slot_L{layer}'] = P
        if best is None or v['mean'] < best[0]:
            best = (v['mean'], layer, X)
        # residual correction of the rough point from the same state
        R = Y - np.concatenate([A, A, A], 1)
        mr, _ = rt_.fit_ridge(np.concatenate([X, A], 1)[S['train']], R[S['train']], ALPHAS,
                              np.concatenate([X, A], 1)[S['val']], R[S['val']])
        P = np.concatenate([A, A, A], 1) + mr.predict(np.concatenate([X, A], 1))
        log[f'resid_slot_L{layer}'] = {'val': val_metric(P[S['val']], Y[S['val']], sets[S['val']]),
                                       'test': errors_by_set(P[S['test']], Y[S['test']], sets[S['test']]),
                                       'heldout_phrasing': errors_by_set(P[S['heldout']], Y[S['heldout']], sets[S['heldout']])}
        preds[f'resid_slot_L{layer}'] = P
        print(f'ridge/resid L{layer}', log[f'ridge_slot_L{layer}']['val'], log[f'resid_slot_L{layer}']['val'], flush=True)
    log['ridge_slot_best_layer'] = best[1]
    # learned query over decoder content (layer 16 PCA) from the slot hidden state of the best layer
    name = 'query_content16'
    if name not in log:
        hk = r2.HIDDEN_LAYERS.index(best[1])
        items = make_items(cap, cases, ('content_16',))
        for i, it in enumerate(items):
            it['h'] = cap['hidden'][i, hk]
        params, hist = fit_query_pointer(items, S, device)
        P = predict_query_pointer(params, items, device)
        preds[name] = P
        log[name] = {'hidden_layer': best[1], 'val': val_metric(P[S['val']], Y[S['val']], sets[S['val']]),
                     'test': errors_by_set(P[S['test']], Y[S['test']], sets[S['test']]),
                     'heldout_phrasing': errors_by_set(P[S['heldout']], Y[S['heldout']], sets[S['heldout']]),
                     'best_epoch': hist['best_epoch']}
        torch.save(params, out / f'{name}.pt')
        print(name, log[name]['val'], flush=True)


def stage_fit_digit(run: Path) -> None:
    """Residual correction of the model's rough point from the decoder state at q1 (experiment-3.12 role):
    for x, ridge on h(q1x) predicts truth_x - approx_x; for y, h(q1y) predicts truth_y - approx_y.
    Point only (the rough point carries no box).  Predictions are aligned with ids.json (NaN elsewhere)."""
    out = run / 'tool_r2'
    cases = {c['id']: c for c in read_jsonl(run / 'cases.jsonl')}
    ids_all = json.loads((out / 'ids.json').read_text())
    pos = {k: i for i, k in enumerate(ids_all)}
    ids, X, own = [], [], []
    for f in sorted((run / 'features_digit').glob('chunk_*.npz')):
        d = np.load(f)
        ids += d['ids'].tolist()
        X.append(d['x'])
        own += d['own'].tolist()
    X = np.concatenate(X)                                        # (N, 2 sites, layers, D)
    A = np.asarray([r2.parse_approx(o) for o in own], float).clip(0, 1000)
    Y = np.stack([rt_.targets_units(cases[i]['truth']) for i in ids])
    sp = np.asarray([cases[i]['split'] for i in ids])
    sets = np.asarray([cases[i]['set'] for i in ids])
    tr, va, te = sp == 'train', sp == 'val', sp == 'test'
    from analysis_tools.position.readout_tool_r2_run import DIGIT_LAYERS
    log, preds = {}, {}
    for k, layer in enumerate(DIGIT_LAYERS):
        P = np.full((len(ids), 6), np.nan)
        for axis, site in ((0, 0), (1, 1)):
            Z = np.concatenate([X[:, site, k].astype(np.float32), A], 1)
            R = (Y[:, axis] - A[:, axis])[:, None].repeat(6, 1)
            m, _ = rt_.fit_ridge(Z[tr], R[tr], ALPHAS, Z[va], R[va])
            P[:, axis] = A[:, axis] + m.predict(Z)[:, 0]
        name = f'resid_q1_L{layer}'
        full = np.full((len(ids_all), 6), np.nan)
        full[[pos[i] for i in ids]] = P
        preds[name] = full
        log[name] = {'val': val_metric(P[va], Y[va], sets[va]), 'test': errors_by_set(P[te], Y[te], sets[te]),
                     'n_train': int(tr.sum()), 'point_only': True}
        print(name, json.dumps(log[name]), flush=True)
    full = np.full((len(ids_all), 6), np.nan)
    full[[pos[i] for i in ids], :2] = A
    preds['approx_point_digit_subset'] = full
    Pa = np.concatenate([A, np.full((len(ids), 4), np.nan)], 1)
    log['approx_point_digit_subset'] = {'val': val_metric(Pa[va], Y[va], sets[va]), 'test': errors_by_set(Pa[te], Y[te], sets[te]),
                                        'point_only': True}
    (out / 'readers_digit.json').write_text(json.dumps(log, indent=1))
    np.savez(out / 'predictions_digit.npz', **preds)


def fit_query_pointer(items, S, device, epochs: int = 20, lr: float = 1e-3, batch: int = 32, seed: int = 0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = [it for it, t in zip(items, S['train']) if t]
    val = [it for it, v in zip(items, S['val']) if v]
    sample = np.concatenate([it['F'][::7] for it in train[::3]]).astype(np.float32)
    mean, std = sample.mean(0), sample.std(0) + 1e-3
    model = r2.QueryPointer(train[0]['h'].shape[0], train[0]['F'].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    order = sorted(range(len(train)), key=lambda i: train[i]['grid'][0] * train[i]['grid'][1])
    groups = [order[k:k + batch] for k in range(0, len(order), batch)]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(groups), pct_start=0.1)
    best = (float('inf'), None, -1)
    hist = []
    for ep in range(epochs):
        model.train()
        for gi in rng.permutation(len(groups)):
            b = [train[i] for i in groups[gi]]
            X, V, Q, Y, grids = r2.make_grid_batch(b, mean, std, device)
            H = torch.tensor(np.stack([it['h'] for it in b]), dtype=torch.float32, device=device)
            alpha, point, box = r2.grid_decode(model(H, X), V, grids)
            qn = Q / Q.sum(1, keepdim=True).clamp_min(1e-6)
            loss = -(qn * torch.log(alpha.clamp_min(1e-9))).sum(1).mean() + (point - Y[:, :2]).abs().sum(1).mean() / 100 \
                + (box - Y[:, 2:]).abs().sum(1).mean() / 100
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        params = {'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'mean': mean, 'std': std,
                  'dh': train[0]['h'].shape[0], 'dv': train[0]['F'].shape[1]}
        P = predict_query_pointer(params, val, device, model)
        err = float(np.mean(np.linalg.norm(P[:, :2] - np.stack([it['y'][:2] for it in val]), axis=1)))
        hist.append(err)
        print(f'  query epoch {ep}: {err:.1f}', flush=True)
        if err < best[0]:
            best = (err, params, ep)
    return best[1], {'best_epoch': best[2], 'val_error_by_epoch': hist}


@torch.no_grad()
def predict_query_pointer(params, items, device, model=None, batch: int = 32) -> np.ndarray:
    if model is None:
        model = r2.QueryPointer(params['dh'], params['dv']).to(device)
        model.load_state_dict(params['state'])
    model.eval()
    out = np.zeros((len(items), 6))
    order = sorted(range(len(items)), key=lambda i: items[i]['grid'][0] * items[i]['grid'][1])
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        b = [items[i] for i in idx]
        X, V, _, _, grids = r2.make_grid_batch(b, params['mean'], params['std'], device)
        H = torch.tensor(np.stack([it['h'] for it in b]), dtype=torch.float32, device=device)
        _, point, box = r2.grid_decode(model(H, X), V, grids)
        out[idx, :2] = point.cpu().numpy()
        out[idx, 2:] = box.cpu().numpy()
    return out


# ----------------------------------------------------------------------------- evaluate

def own_answers(run: Path, r1: Path) -> dict:
    own = defaultdict(dict)
    for path in (r1 / 'own.jsonl', run / 'own.jsonl'):
        for r in read_jsonl(path):
            own[r['id']][r['format']] = rt_.parse_answer(r['text'], r['format'])
    return own


def score_rows(cases: dict, ids: list[str], P: np.ndarray, own: dict) -> list[dict]:
    rows = []
    for cid, p in zip(ids, P):
        c = cases[cid]
        t = c['truth']
        base = c.get('base_id', cid)
        q = rt_.ReadoutTool.format(np.nan_to_num(p)) if not np.isnan(p[2:]).any() else \
            {'point_2d': [int(v) for v in np.clip(np.rint(p[:2]), 0, 1000)], 'bbox_2d': None}
        tp = rt_.score_point(t, q['point_2d'])
        tb = rt_.score_box(t, q['bbox_2d']) if q['bbox_2d'] is not None else {'iou': float('nan')}
        o = own.get(base, {})
        op = rt_.score_point(t, o.get('point'))
        ob = rt_.score_box(t, o.get('bbox'))
        obc = rt_.score_point(t, rt_.answer_center(o['bbox'])) if o.get('bbox') else {'valid': False, 'hit': False}
        rows.append({'id': cid, 'set': c['set'], 'subset': c['subset'], 'tool_hit': tp['hit'], 'tool_error': tp['error'],
                     'tool_iou': tb['iou'], 'tool_point': q['point_2d'], 'tool_box': q['bbox_2d'], 'tool_raw': p.tolist(),
                     'own_valid': op['valid'], 'own_hit': op['hit'], 'own_hit_any': op['hit_any'], 'own_error': op.get('error'),
                     'own_point': op.get('point'), 'own_box': ob.get('box'), 'own_iou': ob['iou'],
                     'own_box_center_error': obc.get('error'), 'own_box_center_hit': obc['hit'],
                     'area_frac': float((t['box'][2] - t['box'][0]) * (t['box'][3] - t['box'][1]) / (t['canvas'][0] * t['canvas'][1]))})
    return rows


def set_summary(rows: list[dict]) -> dict:
    n = len(rows)
    kt, ko = sum(r['tool_hit'] for r in rows), sum(r['own_hit'] for r in rows)
    b = sum(r['tool_hit'] and not r['own_hit'] for r in rows)
    c = sum(r['own_hit'] and not r['tool_hit'] for r in rows)
    te = np.asarray([r['tool_error'] for r in rows])
    v = [r for r in rows if r['own_valid']]
    d = np.asarray([r['own_error'] - r['tool_error'] for r in v])
    ti, oi = np.asarray([r['tool_iou'] for r in rows], float), np.asarray([r['own_iou'] for r in rows], float)
    if np.isnan(ti).all():                     # point-only reader
        ti = oi * np.nan
    bc = np.asarray([r['own_box_center_error'] for r in rows if r['own_box_center_error'] is not None])
    return {'n': n, 'tool_hit': kt / n, 'tool_hit_ci': wilson(kt, n), 'own_hit': ko / n, 'own_hit_ci': wilson(ko, n),
            'own_hit_any': sum(r['own_hit_any'] for r in rows) / n, 'only_tool': b, 'only_own': c, 'p_mcnemar_hit': mcnemar_exact(b, c),
            'tool_err_mean': float(te.mean()), 'tool_err_median': float(np.median(te)),
            'own_err_mean': float(np.mean([r['own_error'] for r in v])) if v else float('nan'),
            'own_err_median': float(np.median([r['own_error'] for r in v])) if v else float('nan'),
            'own_box_center_err_median': float(np.median(bc)) if len(bc) else float('nan'),
            'diff_err_mean': float(d.mean()) if len(d) else float('nan'), 'diff_err_ci': boot_ci(d) if len(d) else [float('nan')] * 2,
            'p_sign_err': binom_two_sided(int((d > 0).sum()), int((d != 0).sum())) if len(d) else float('nan'),
            'tool_iou': float(ti.mean()), 'own_iou': float(oi.mean()), 'tool_iou50': float((ti >= 0.5).mean()),
            'own_iou50': float((oi >= 0.5).mean()),
            'diff_iou_ci': boot_ci(ti - oi) if not np.isnan(ti).any() else [float('nan')] * 2,
            'p_sign_iou': binom_two_sided(int((ti > oi).sum()), int((ti != oi).sum())) if not np.isnan(ti).any() else float('nan')}


POINT_PREFIX = re.compile(r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)')


def parse_point_prefix(text: str) -> list[float] | None:
    """First point_2d pair in a possibly truncated answer (the dissociation run stops after 24 tokens)."""
    m = POINT_PREFIX.search(text)
    return [float(m.group(1)), float(m.group(2))] if m else None


def parse_rc(text: str):
    return rt_.parse_rowcol_free(text)


def dissoc_eval(run: Path, cases: dict) -> dict:
    rows = read_jsonl(run / 'dissoc.jsonl')
    if not rows:
        return {}
    by = defaultdict(dict)
    for r in rows:
        by[(r['id'], r['task'])][r['condition']] = r
    out = defaultdict(lambda: defaultdict(list))
    pair = defaultdict(dict)
    for (cid, task), conds in by.items():
        t = cases[cid]['truth']
        g = t['grid']
        if 'base' not in conds:
            continue

        def cell(r):
            if task == 'point':
                p = parse_point_prefix(r['text'])
                if p is None:
                    return None
                s = rt_.score_point(t, p)
                return (s['pred_rc'][0] + 1, s['pred_rc'][1] + 1)
            return parse_rc(r['text'])
        b = cell(conds['base'])
        truth_rc = (g['target_rc'][0] + 1, g['target_rc'][1] + 1)          # target_rc is 0-based
        out[task]['base_correct'].append(b == truth_rc if b else False)
        for cond in ('steer_x', 'steer_y', 'gate'):
            if cond not in conds:
                continue
            x = cell(conds[cond])
            if b is None or x is None:
                out[task][cond + '_invalid'].append(1)
                continue
            dr, dc = x[0] - b[0], x[1] - b[1]
            pair[(cid, cond)][task] = (dr, dc)
            out[task][cond + '_drow'].append(dr)
            out[task][cond + '_dcol'].append(dc)
            out[task][cond + '_correct'].append(x == truth_rc)
            if cond.startswith('steer'):
                sd = conds[cond]['steer_dir']
                out[task][cond + '_follow'].append((dr, dc) == tuple(sd))
                out[task][cond + '_moved'].append((dr, dc) != (0, 0))
    res = {'paired': {}}
    for cond in ('steer_x', 'steer_y', 'gate'):
        both = [v for (cid, c), v in pair.items() if c == cond and len(v) == 2]
        if len(both) < 3:
            continue
        pr = np.asarray([v['point'][0] for v in both], float)
        rr = np.asarray([v['rowcol'][0] for v in both], float)
        pc = np.asarray([v['point'][1] for v in both], float)
        rc = np.asarray([v['rowcol'][1] for v in both], float)
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
        res['paired'][cond] = {'n': len(both), 'r_drow': corr(pr, rr), 'r_dcol': corr(pc, rc),
                               'same_drow': float((pr == rr).mean()), 'same_dcol': float((pc == rc).mean()),
                               'both_row_moved': float(((pr != 0) & (rr != 0)).mean()),
                               'only_point_row_moved': float(((pr != 0) & (rr == 0)).mean()),
                               'only_ordinal_row_moved': float(((pr == 0) & (rr != 0)).mean())}
    for task, d in out.items():
        res[task] = {}
        for k, v in d.items():
            v = np.asarray(v, float)
            res[task][k] = {'n': int(len(v)), 'mean': float(v.mean()) if len(v) else float('nan')}
            if k.endswith(('_follow', '_moved', '_correct')) or k == 'base_correct':
                res[task][k]['ci'] = wilson(int(v.sum()), len(v))
    return res


def alpha_examples(run: Path, out: Path, cases: dict, name: str, rows: list[dict], device: str = 'cpu') -> None:
    """Pointer distributions of the final reader on one random test image per showcase set."""
    rng = np.random.default_rng(11)
    want = []
    for s in ('refcoco', 'g_photo_bg', 'g_clutter', 'layout', 'disc'):
        g = [r['id'] for r in rows if r['set'] == s and not r['id'].endswith('_hp')]
        want.append(g[int(rng.integers(len(g)))])
    sub = {k: cases[k] for k in want}
    cap = defaultdict(list)
    for f in sorted((run / 'features').glob('chunk_s*_*.npz')):
        d = np.load(f)
        ids = d['ids'].tolist()
        hit = [i for i, k in enumerate(ids) if k in sub]
        if not hit:
            continue
        cut = np.cumsum(np.concatenate([[0], d['map_len']]))
        arrays = {k: d[k] for k in d.files}
        for i in hit:
            cap['ids'].append(ids[i])
            cap['cont'].append(str(arrays['cont'][i]))
            cap['grid'].append(tuple(int(v) for v in arrays['grid'][i]))
            for st in r2.SITES_R2:
                cap[f'maps_{st}'].append(arrays[f'maps_{st}'][:, cut[i]:cut[i + 1]])
            for l in r2.CONTENT_LAYERS:
                cap[f'content_{l}'].append(arrays[f'content_{l}'][cut[i]:cut[i + 1]])
    items = make_items(dict(cap), sub, READERS[name.split('@')[0]][1])
    params = torch.load(out / f'{name}.pt', weights_only=False)
    m = r2.load_grid_pointer(params)
    _, alphas = r2.predict_grid_pointer(m, params['mean'], params['std'], items, device, return_alpha=True)
    ex = [{'id': it['id'], 'alpha': a.tolist()} for it, a in zip(items, alphas)]
    (out / 'alpha_examples.json').write_text(json.dumps(ex))


def cost_model(n_visual: int = 1024, seq_len: int = 1300) -> dict:
    """Analytic FLOPs and measured CPU latency of the tool for one call (one image of n_visual tokens).

    Attention row: for each of the 8 full-attention layers, 16 query heads x head_dim 256 dot products
    against seq_len cached keys plus the softmax (the query itself is computed by the forward pass).
    Reader: the convolutional pointer on a square token grid.  Model reference: 2 x 9e9 FLOPs per token.
    """
    heads, hd, layers = 16, 256, len(rt_.ATTN_LAYERS)
    attn = layers * heads * hd * 2 * seq_len
    g = int(round(math.sqrt(n_visual)))
    res = {}
    for name, cin in (('attention only', 128), ('attention + content + rough point', 128 + 2 * r2.PCA_DIM + 3)):
        m = r2.GridPointer(cin).eval()
        conv = sum(p.numel() for n, p in m.named_parameters() if n.endswith('weight')) * 2 * g * g
        x = torch.randn(1, cin, g, g)
        with torch.no_grad():
            for _ in range(3):
                m(x)
            t0 = time.perf_counter()
            for _ in range(20):
                m(x)
        res[name] = {'reader_flops': float(conv), 'reader_params': int(sum(p.numel() for p in m.parameters())),
                     'reader_cpu_ms': (time.perf_counter() - t0) / 20 * 1000}
    pca = 2 * 4096 * r2.PCA_DIM * 2 * n_visual
    return {'n_visual': n_visual, 'seq_len': seq_len, 'attention_row_flops': float(attn), 'pca_projection_flops': float(pca),
            'model_prefill_flops': float(2 * 9e9 * seq_len), 'model_flops_per_generated_token': 2 * 9e9,
            'map_bytes_fp16': 2 * layers * heads * n_visual, 'content_bytes_fp16': 2 * 2 * r2.PCA_DIM * n_visual,
            'readers': res}


def ordinal_tool_points(run: Path, name: str, device: str = 'cpu') -> dict:
    """Reader `name` applied to the capture on the experiment-1.5 grids: image_id -> point (0-1000)."""
    feat = run / 'features_ord'
    if not feat.exists() or not (run / 'tool_r2' / f'{name}.pt').exists():
        return {}
    cases = {c['id']: c for c in read_jsonl(run / 'ordinal_cases.jsonl')}
    cap = defaultdict(list)
    for f in sorted(feat.glob('chunk_*.npz')):
        d = np.load(f)
        cut = np.cumsum(np.concatenate([[0], d['map_len']]))
        arrays = {k: d[k] for k in d.files}
        for i, cid in enumerate(arrays['ids'].tolist()):
            cap['ids'].append(cid)
            cap['cont'].append(str(arrays['cont'][i]))
            cap['grid'].append(tuple(int(v) for v in arrays['grid'][i]))
            for st in r2.SITES_R2:
                cap[f'maps_{st}'].append(arrays[f'maps_{st}'][:, cut[i]:cut[i + 1]])
            for l in r2.CONTENT_LAYERS:
                cap[f'content_{l}'].append(arrays[f'content_{l}'][cut[i]:cut[i + 1]])
    items = make_items(dict(cap), cases, READERS[name.split('@')[0]][1])
    params = torch.load(run / 'tool_r2' / f'{name}.pt', weights_only=False)
    P = predict_items(name, params, items, device)
    approx = {cid: r2.parse_approx(c) for cid, c in zip(cap['ids'], cap['cont'])}
    return {'tool': {cid: p[:2].tolist() for cid, p in zip(cap['ids'], P)}, 'approx': approx}


def ordcoord_eval(run: Path, tool_points: dict | None = None) -> dict:
    """Experiment-1.5 stimulus variants: change of the point answer (in cells along the varied axis)
    and of the direct ordinal answer, relative to the base image of the same base index."""
    rows = read_jsonl(run / 'ordcoord.jsonl')
    if not rows:
        return {}
    root = run.parent.parent / 'behavior' / '2026-10-16_ordinal_mechanism'
    imgs = {r['image_id']: r for r in read_jsonl(root / 'images.jsonl')}
    cases = {r['image'].split('/')[-1]: r for r in read_jsonl(root / 'cases.jsonl') if r['mode'] == 'direct'}
    answers = {r['id']: r['response'] for r in read_jsonl(root / 'gen' / 'results_direct.jsonl')}
    rec = {}
    for r in rows:
        im = imgs[r['image_id']]
        name = im['path'].split('/')[-1]
        c = cases[name]
        t = c['truth']
        axis = t['axis']
        p = parse_point_prefix(r['text'])
        W, H = t['canvas']
        coord = None if p is None else (p[0] * W / 1000 if axis == 'H' else p[1] * H / 1000) / t['cell']   # in cells
        rc = parse_rc(answers.get(c['id'], ''))
        ordv = None if rc is None else (rc[1] if axis == 'H' else rc[0])
        cond, base, var = c['condition'], c['base'], name.rsplit('_', 1)[1].split('.')[0]
        box = t['target_cell_box']
        truth_cell = ((box[0] + box[2]) / 2 if axis == 'H' else (box[1] + box[3]) / 2) / t['cell']
        extra = {}
        for key in ('tool', 'approx'):
            q = (tool_points or {}).get(key, {}).get(r['image_id'])
            extra[key] = None if q is None else (q[0] * W / 1000 if axis == 'H' else q[1] * H / 1000) / t['cell']
        rec[(cond, base, var)] = dict({'coord': coord, 'ord': ordv, 'truth_cell': truth_cell}, **extra)
    out = defaultdict(lambda: defaultdict(list))
    for (cond, base, var), v in rec.items():
        b = rec.get((cond, base, 'base'))
        if var == 'base' or b is None:
            continue
        if None not in (v['coord'], b['coord']):
            out[(cond, var)]['dcoord'].append(v['coord'] - b['coord'])
            out[(cond, var)]['dtruth'].append(v['truth_cell'] - b['truth_cell'])
        if None not in (v['ord'], b['ord']):
            out[(cond, var)]['dord'].append(v['ord'] - b['ord'])
        for key in ('tool', 'approx'):
            if None not in (v.get(key), b.get(key)):
                out[(cond, var)][f'd{key}'].append(v[key] - b[key])
    res = {}
    for (cond, var), d in sorted(out.items()):
        dc, do = np.asarray(d['dcoord']), np.asarray(d['dord'])
        res[f'{cond}/{var}'] = {'n_coord': int(len(dc)), 'dcoord_mean': float(dc.mean()) if len(dc) else float('nan'),
                                'dcoord_median': float(np.median(dc)) if len(dc) else float('nan'),
                                'dcoord_within_half_of_truth': float((np.abs(dc - np.asarray(d['dtruth'])) < 0.5).mean()) if len(dc) else float('nan'),
                                'dtruth_mean': float(np.mean(d['dtruth'])) if len(dc) else float('nan'),
                                'n_ord': int(len(do)), 'dord_mean': float(do.mean()) if len(do) else float('nan'),
                                'dord_eq0': float((do == 0).mean()) if len(do) else float('nan'),
                                'dord_eqm1': float((do == -1).mean()) if len(do) else float('nan'),
                                'dord_eqp1': float((do == 1).mean()) if len(do) else float('nan'),
                                **{f'd{k}_mean': float(np.mean(d[f'd{k}'])) if d.get(f'd{k}') else float('nan') for k in ('tool', 'approx')},
                                **{f'd{k}_median': float(np.median(d[f'd{k}'])) if d.get(f'd{k}') else float('nan') for k in ('tool', 'approx')}}
    return res


def stage_evaluate(run: Path, r1: Path) -> None:
    out = run / 'tool_r2'
    cases = {c['id']: c for c in read_jsonl(run / 'cases.jsonl')}
    log, preds = {}, {}
    for f in sorted(out.glob('readers_*.json')):
        log.update(json.loads(f.read_text()))
    for f in sorted(out.glob('predictions_*.npz')):
        preds.update(dict(np.load(f)))
    ids = json.loads((out / 'ids.json').read_text())
    own = own_answers(run, r1)
    test = [k for k, i in enumerate(ids) if cases[i]['split'] == 'test']
    readers = [k for k in log if isinstance(log[k], dict) and 'val' in log[k]]
    res = {'selection': {k: log[k]['val'] for k in readers}, 'by_reader': {}}
    for name in readers:
        tk = [k for k in test if not np.isnan(preds[name][k, 0])]
        rows = score_rows(cases, [ids[k] for k in tk], preds[name][tk], own)
        main = [r for r in rows if not r['id'].endswith('_hp')]
        res['by_reader'][name] = {s: set_summary([r for r in main if r['set'] == s]) for s in EVAL_SETS
                                  if any(r['set'] == s for r in main)}
        hp = [r for r in rows if r['id'].endswith('_hp')]
        if hp:
            base_err = {r['id']: r['tool_error'] for r in main}
            res['by_reader'][name]['_heldout_phrasing'] = {
                s: {'n': len(g), 'err_heldout': float(np.mean([r['tool_error'] for r in g])),
                    'err_canonical_same_cases': float(np.mean([base_err[r['id'][:-3]] for r in g]))}
                for s in EVAL_SETS for g in [[r for r in hp if r['set'] == s]] if g}
        if name == res.get('final'):
            pass
    tools = [k for k in readers if not log[k].get('point_only') and not k.startswith('approx_point')]
    res['final'] = min(tools, key=lambda k: log[k]['val']['mean'])
    final_rows = score_rows(cases, [ids[k] for k in test], preds[res['final']][test], own)
    (out / 'final_rows.json').write_text(json.dumps(final_rows))
    vals = lambda key: np.concatenate([np.asarray(r[key]) for r in final_rows if r[key] is not None and not r['id'].endswith('_hp')])
    res['digits'] = {'tool': digit_stats(vals('tool_point')),
                     'own': digit_stats(np.concatenate([np.asarray(r['own_point']) for r in final_rows if r['own_point']]))}
    res['dissoc'] = dissoc_eval(run, cases)
    res['ordcoord'] = ordcoord_eval(run, ordinal_tool_points(run, res['final']))
    res['ridge_best'] = log.get('ridge_slot_best_layer')
    res['cost'] = cost_model()
    if READERS.get(res['final'].split('@')[0], ('',))[0] == 'cnn':
        alpha_examples(run, out, cases, res['final'], final_rows)
    fc = read_jsonl(run / 'freecall.jsonl')
    if fc:
        res['freecall'] = {'n': len(fc), 'called': sum(r['called'] for r in fc), 'same_prefix': sum(r['same_prefix'] for r in fc),
                           'approx_parsed': sum(r['approx'] is not None for r in fc),
                           'reasoning_before_call': sum(r['reasoning_before_call'] for r in fc)}
    (out / 'results_r2.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({'final': res['final']}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--r1', type=Path, required=True)
    p.add_argument('--stage', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--only', default='')
    p.add_argument('--tag', default='main')
    a = p.parse_args()
    if a.stage == 'fit':
        stage_fit(a.run, a.device, [s for s in a.only.split(',') if s] or None, a.tag)
    elif a.stage == 'fit_digit':
        stage_fit_digit(a.run)
    elif a.stage == 'evaluate':
        stage_evaluate(a.run, a.r1)
    else:
        raise ValueError(a.stage)


if __name__ == '__main__':
    main()
