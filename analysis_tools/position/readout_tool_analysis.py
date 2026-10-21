"""Experiment 3.9 analysis: fit the readout regressors and evaluate the tool against the model.

  fit       ridge for every (site, layer) on the mixed training set (RefCOCO train + generic synthetic
            train), selection on the mixed validation set only; multi-layer ridge and MLP candidates at
            the selected site; data ablations (RefCOCO only, synthetic only); an in-domain disc ceiling.
            Writes tool/tool.pt (the selected regressor) and predictions for every test case.
  evaluate  paired tool-vs-model metrics per set and condition, defect diagnostics (digit snapping,
            column offsets, coordinate conventions), row/column tool-loop results and free calls.
Held-out evaluation sets (discs, T6 grids, layouts, rings, RefCOCO test) are never used for fitting
or selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position.coordinate_readout_run import read_jsonl

ALPHAS = (1e1, 1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6)
EVAL_SETS = ('refcoco', 'synth', 'disc', 't6', 'layout', 'ring')


# ----------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float('nan')] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for discordant counts b (only A correct) and c (only B correct)."""
    return binom_two_sided(b, b + c)


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test with p = 0.5 (doubling the smaller tail)."""
    if n == 0:
        return 1.0
    m = min(k, n - k)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(m + 1)]
    top = max(logs)
    tail = math.exp(top) * sum(math.exp(v - top) for v in logs)
    return float(min(1.0, 2 * tail))


def boot_ci(x: np.ndarray, stat=np.mean, n: int = 2000, seed: int = 0) -> list[float]:
    if len(x) == 0:
        return [float('nan')] * 2
    rng = np.random.default_rng(seed)
    v = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ----------------------------------------------------------------------------- features

def load_features(run: Path, cases: dict):
    """Hidden states (N, sites, layers, D) float16, attention summaries (N, sites, La, H, 7) and the
    target_end attention maps (list of (La*H, Nv) float16), in the order of the returned ids."""
    ids, xs, fs, maps = [], [], [], []
    for f in sorted((run / 'features').glob('chunk_*.npz')):
        d = np.load(f)
        ids += d['ids'].tolist()
        xs.append(d['x'])
        fs.append(d['attn'])
        cut = np.cumsum(np.concatenate([[0], d['map_len']]))
        m = d['maps']                     # read once; NpzFile re-reads the member on every access
        maps += [m[:, a:b] for a, b in zip(cut[:-1], cut[1:])]
    missing = set(cases) - set(ids)
    if missing:
        raise ValueError(f'{len(missing)} cases have no captured features, e.g. {sorted(missing)[:3]}')
    return ids, np.concatenate(xs), np.concatenate(fs), maps


def pointer_items(C: list[dict], maps: list[np.ndarray]) -> list[dict]:
    items = []
    for c, m in zip(C, maps):
        g = rt_.grid_of(c['truth']['canvas'])
        y = rt_.targets_units(c['truth'])
        items.append({'F': rt_.token_features(m, g).astype(np.float16), 'centers': rt_.token_centers((1, 2 * g[0], 2 * g[1])).numpy(),
                      'q': rt_.token_overlap(y[2:], g), 'y': y, 'grid': g})
    return items


def view(X: np.ndarray, site: str, layers) -> np.ndarray:
    s = rt_.SITES.index(site)
    li = [rt_.CAPTURE_LAYERS.index(l) for l in layers]
    return X[:, s, li].reshape(len(X), -1).astype(np.float32)


def val_score(P: np.ndarray, Y: np.ndarray, groups: np.ndarray) -> dict:
    out = {}
    for g in np.unique(groups):
        m = groups == g
        out[str(g)] = float(np.linalg.norm(P[m, :2] - Y[m, :2], axis=1).mean())
    out['mean'] = float(np.mean([out[str(g)] for g in np.unique(groups)]))
    return out


def set_errors(P: np.ndarray, Y: np.ndarray, sets: np.ndarray) -> dict:
    return {s: float(np.linalg.norm(P[sets == s, :2] - Y[sets == s, :2], axis=1).mean()) for s in EVAL_SETS if (sets == s).any()}


# ----------------------------------------------------------------------------- fit

def stage_fit(run: Path) -> None:
    out = run / 'tool'
    out.mkdir(exist_ok=True)
    cases = {c['id']: c for c in read_jsonl(run / 'cases.jsonl')}
    ids, X, A, maps = load_features(run, cases)
    C = [cases[i] for i in ids]
    Y = np.stack([rt_.targets_units(c['truth']) for c in C])
    split = np.asarray([c['split'] for c in C])
    sets = np.asarray([c['set'] for c in C])
    tr, va, te = split == 'train', split == 'val', split == 'test'
    log = {'n_train': int(tr.sum()), 'n_val': int(va.sum()), 'n_test': int(te.sum()),
           'train_by_set': {s: int((tr & (sets == s)).sum()) for s in ('refcoco', 'synth')},
           'val_by_set': {s: int((va & (sets == s)).sum()) for s in ('refcoco', 'synth')}}
    print(json.dumps(log), flush=True)
    sweep = []
    for site in rt_.SITES:
        for layer in rt_.CAPTURE_LAYERS:
            t0 = time.time()
            Z = view(X, site, [layer])
            model, curve = rt_.fit_ridge(Z[tr], Y[tr], ALPHAS, Z[va], Y[va])
            P = model.predict(Z)
            row = {'site': site, 'layer': layer, 'alpha': model.alpha, 'val': val_score(P[va], Y[va], sets[va]),
                   'test': set_errors(P[te], Y[te], sets[te]), 'alpha_curve': curve,
                   'val_iou': float(np.mean([rt_.box_iou(p[2:], y[2:]) for p, y in zip(P[va], Y[va])]))}
            sweep.append(row)
            print(f"{site:11s} L{layer:02d} alpha={model.alpha:g} val={row['val']['mean']:.1f} "
                  f"test={ {k: round(v, 1) for k, v in row['test'].items()} } ({time.time() - t0:.0f}s)", flush=True)
    (out / 'sweep.json').write_text(json.dumps(sweep, indent=1))
    best = min(sweep, key=lambda r: r['val']['mean'])
    site = best['site']
    ranked = sorted([r for r in sweep if r['site'] == site], key=lambda r: r['val']['mean'])
    top3 = tuple(sorted(r['layer'] for r in ranked[:3]))
    candidates = {}

    def record(name, kind, layers, P, extra):
        candidates[name] = {'kind': kind, 'site': site, 'layers': list(layers), 'val': val_score(P[va], Y[va], sets[va]),
                            'test': set_errors(P[te], Y[te], sets[te]), **extra}
        print(name, json.dumps(candidates[name]['val']), flush=True)

    models = {}
    for name, layers in (('ridge_single', (best['layer'],)), ('ridge_top3', top3)):
        Z = view(X, site, layers)
        m, curve = rt_.fit_ridge(Z[tr], Y[tr], ALPHAS, Z[va], Y[va])
        P = m.predict(Z)
        record(name, 'ridge', layers, P, {'alpha': m.alpha})
        models[name] = (rt_.ReadoutTool('ridge', site, layers, {k: getattr(m, k) for k in ('mean', 'scale', 'weight', 'bias', 'alpha')}), P)
    import torch
    torch.set_num_threads(16)
    for name, layers in (('mlp_single', (best['layer'],)), ('mlp_top3', top3)):
        Z = view(X, site, layers)
        m, hist = rt_.fit_mlp(Z[tr], Y[tr], Z[va], Y[va], seed=0)
        with torch.no_grad():
            P = m(torch.tensor(Z)).numpy()
        record(name, 'mlp', layers, P, hist)
        tool = rt_.ReadoutTool.__new__(rt_.ReadoutTool)
        tool.kind, tool.site, tool.layers, tool.mlp = 'mlp', site, tuple(layers), m
        models[name] = (tool, P)
    # Attention-summary ridge: per-head visual mass, centroid, spread and peak at all sites.
    Z = A.reshape(len(A), -1).astype(np.float32)
    m, _ = rt_.fit_ridge(Z[tr], Y[tr], ALPHAS, Z[va], Y[va])
    record('attn_summary_ridge', 'ridge', (), m.predict(Z), {'alpha': m.alpha})
    # Training-free single head: attention centroid of the best head (chosen on validation) at target_end.
    s_t = rt_.SITES.index('target_end')
    cen = A[:, s_t, :, :, 1:3]                                                      # (N, La, H, 2)
    err = np.linalg.norm(cen[va] - Y[va, None, None, :2], axis=-1).mean(0)
    li, hi = np.unravel_index(err.argmin(), err.shape)
    P1 = np.concatenate([cen[:, li, hi], np.repeat(cen[:, li, hi], 2, 1)], 1)
    record('single_head_centroid', 'head', (rt_.ATTN_LAYERS[li],), P1, {'head': int(hi)})
    # Pointer readout on the target_end attention maps.
    items = pointer_items(C, maps)
    itr = [it for it, t in zip(items, tr) if t]
    iva = [it for it, v in zip(items, va) if v]
    params, hist = rt_.fit_pointer(itr, iva, seed=0)
    ptool = rt_.ReadoutTool('pointer', 'target_end', rt_.ATTN_LAYERS, params)
    PP = rt_.pointer_predict_items(ptool.pointer, ptool.pmean, ptool.pstd, items, ptool.tau)
    record('pointer', 'pointer', rt_.ATTN_LAYERS, PP, hist)
    models['pointer'] = (ptool, PP)
    for name, mask in (('pointer_refcoco_only', tr & (sets == 'refcoco')), ('pointer_synth_only', tr & (sets == 'synth'))):
        prm, h2 = rt_.fit_pointer([it for it, t in zip(items, mask) if t], iva, seed=0)
        t2 = rt_.ReadoutTool('pointer', 'target_end', rt_.ATTN_LAYERS, prm)
        record(name, 'pointer', rt_.ATTN_LAYERS, rt_.pointer_predict_items(t2.pointer, t2.pmean, t2.pstd, items, t2.tau), h2)
    final = min(models, key=lambda k: candidates[k]['val']['mean'])
    tool, P = models[final]
    allpred = {name: {i: rt_.ReadoutTool.format(p) for i, p, t in zip(ids, Pm, te) if t} for name, (_, Pm) in models.items()}
    (out / 'predictions_candidates.json').write_text(json.dumps(allpred))
    tool.save(out / 'tool.pt', {'candidate': final, 'val': candidates[final]['val']})
    # Data ablations with the selected readout input (ridge, so that only the data changes).
    ablation = {}
    best_hidden = min(('ridge_single', 'ridge_top3'), key=lambda k: candidates[k]['val']['mean'])
    Z = view(X, site, tuple(candidates[best_hidden]['layers']))
    for name, mask in (('refcoco_only', tr & (sets == 'refcoco')), ('synth_only', tr & (sets == 'synth')), ('mixed', tr)):
        m, _ = rt_.fit_ridge(Z[mask], Y[mask], ALPHAS, Z[va], Y[va])
        Q = m.predict(Z)
        ablation[name] = {'val': val_score(Q[va], Y[va], sets[va]), 'test': set_errors(Q[te], Y[te], sets[te]), 'alpha': m.alpha}
        print('ablation', name, json.dumps(ablation[name]), flush=True)
    # In-domain ceiling: fit on disc grid2d, test on the disc sweeps (the experiment-3.6 protocol).
    subset = np.asarray([c.get('subset', '') for c in C])
    g2, sw = (sets == 'disc') & (subset == 'grid2d'), (sets == 'disc') & (subset != 'grid2d')
    ceiling = []
    for layer in rt_.CAPTURE_LAYERS:
        Zl = view(X, site, [layer])
        m, _ = rt_.fit_ridge(Zl[g2], Y[g2], ALPHAS, Zl[sw], Y[sw])   # alpha picked on the sweeps: optimistic ceiling
        Q = m.predict(Zl[sw])
        ceiling.append({'layer': layer, 'sweep_error': float(np.linalg.norm(Q[:, :2] - Y[sw, :2], axis=1).mean()), 'alpha': m.alpha})
    preds = {i: {'raw': p.tolist(), **rt_.ReadoutTool.format(p)} for i, p, t in zip(ids, P, te) if t}
    (out / 'predictions.json').write_text(json.dumps(preds))
    (out / 'fit.json').write_text(json.dumps({'log': log, 'best_single': {k: best[k] for k in ('site', 'layer', 'alpha', 'val')},
                                              'top3': list(top3), 'candidates': candidates, 'final': final,
                                              'ablation': ablation, 'disc_ceiling': ceiling}, indent=1))
    print('final', final, json.dumps(candidates[final]['test']), flush=True)


# ----------------------------------------------------------------------------- variant B (answer slot)

def stage_fit_slot(run: Path) -> None:
    """Variant B readout at the x0 answer slot: per-layer ridge on hidden states and the pointer on the
    x0 attention maps, trained on the reduced slot training set; compared on the same test cases with
    the selected tool (target_end pointer)."""
    out = run / 'tool'
    cases = {c['id']: c for c in read_jsonl(run / 'cases.jsonl')}
    ids, X, A, maps = [], [], [], []
    for f in sorted((run / 'features_slot').glob('chunk_*.npz')):
        d = np.load(f)
        ids += d['ids'].tolist()
        X.append(d['x'])
        cut = np.cumsum(np.concatenate([[0], d['map_len']]))
        m = d['maps']
        maps += [m[:, a:b] for a, b in zip(cut[:-1], cut[1:])]
    X = np.concatenate(X)
    C = [cases[i] for i in ids]
    Y = np.stack([rt_.targets_units(c['truth']) for c in C])
    split = np.asarray([c['split'] for c in C])
    sets = np.asarray([c['set'] for c in C])
    tr, va, te = split == 'train', split == 'val', split == 'test'
    res = {'n': {'train': int(tr.sum()), 'val': int(va.sum()), 'test': int(te.sum())}, 'ridge': []}
    for k, layer in enumerate(rt_.CAPTURE_LAYERS):
        Z = X[:, 0, k].astype(np.float32)
        m, _ = rt_.fit_ridge(Z[tr], Y[tr], ALPHAS, Z[va], Y[va])
        P = m.predict(Z)
        res['ridge'].append({'layer': layer, 'alpha': m.alpha, 'val': val_score(P[va], Y[va], sets[va]),
                             'test': set_errors(P[te], Y[te], sets[te])})
        print('slot ridge', layer, json.dumps(res['ridge'][-1]['test']), flush=True)
    items = pointer_items(C, maps)
    params, hist = rt_.fit_pointer([it for it, t in zip(items, tr) if t], [it for it, v in zip(items, va) if v], seed=0)
    tool = rt_.ReadoutTool('pointer', 'x0', rt_.ATTN_LAYERS, params)
    P = rt_.pointer_predict_items(tool.pointer, tool.pmean, tool.pstd, items, tool.tau)
    res['pointer'] = {'val': val_score(P[va], Y[va], sets[va]), 'test': set_errors(P[te], Y[te], sets[te]), **hist}
    print('slot pointer', json.dumps(res['pointer']['test']), flush=True)
    # same test cases under the selected tool (target_end pointer, trained on the full training set)
    final = json.loads((out / 'predictions.json').read_text())
    tid = [i for i, t in zip(ids, te) if t]
    Pf = np.asarray([final[i]['raw'] for i in tid])
    res['final_on_same_cases'] = set_errors(Pf, Y[te], sets[te])
    res['predictions'] = {i: rt_.ReadoutTool.format(p) for i, p, t in zip(ids, P, te) if t}
    (out / 'slot.json').write_text(json.dumps(res, indent=1))
    print('final tool on same cases', json.dumps(res['final_on_same_cases']), flush=True)


# ----------------------------------------------------------------------------- evaluate

def paired(rows: list[dict], key_t: str, key_o: str) -> dict:
    n = len(rows)
    kt = sum(r[key_t] for r in rows)
    ko = sum(r[key_o] for r in rows)
    b = sum(r[key_t] and not r[key_o] for r in rows)
    c = sum(r[key_o] and not r[key_t] for r in rows)
    return {'n': n, 'tool': kt / n if n else float('nan'), 'tool_ci': wilson(kt, n), 'own': ko / n if n else float('nan'),
            'own_ci': wilson(ko, n), 'only_tool': b, 'only_own': c, 'p_mcnemar': mcnemar_exact(b, c)}


def error_stats(rows: list[dict]) -> dict:
    t = np.asarray([r['tool_error'] for r in rows])
    v = [r for r in rows if r['own_valid']]
    tv = np.asarray([r['tool_error'] for r in v])
    ov = np.asarray([r['own_error'] for r in v])
    out = {'n': len(rows), 'tool_mean': float(t.mean()) if len(t) else float('nan'), 'tool_mean_ci': boot_ci(t),
           'tool_median': float(np.median(t)) if len(t) else float('nan'), 'n_own_valid': len(v),
           'own_mean': float(ov.mean()) if len(ov) else float('nan'), 'own_mean_ci': boot_ci(ov),
           'own_median': float(np.median(ov)) if len(ov) else float('nan'),
           'tool_mean_on_own_valid': float(tv.mean()) if len(tv) else float('nan'),
           'tool_better_frac': float((tv < ov).mean()) if len(tv) else float('nan')}
    if len(tv):
        d = ov - tv
        out['diff_mean_own_minus_tool'] = float(d.mean())
        out['diff_mean_ci'] = boot_ci(d)
        out['p_sign'] = binom_two_sided(int((d > 0).sum()), int((d != 0).sum()))
    return out


def case_rows(run: Path) -> list[dict]:
    cases = {c['id']: c for c in read_jsonl(run / 'cases.jsonl')}
    preds = json.loads((run / 'tool' / 'predictions.json').read_text())
    own = defaultdict(dict)
    for r in read_jsonl(run / 'own.jsonl'):
        own[r['id']][r['format']] = r
    rows = []
    for cid, p in preds.items():
        c = cases[cid]
        t = c['truth']
        tp, tb = rt_.score_point(t, p['point_2d']), rt_.score_box(t, p['bbox_2d'])
        op = rt_.score_point(t, rt_.parse_answer(own[cid]['point']['text'], 'point')) if 'point' in own[cid] else {'valid': False, 'hit': False, 'hit_any': False}
        ob_coords = rt_.parse_answer(own[cid]['bbox']['text'], 'bbox') if 'bbox' in own[cid] else None
        ob = rt_.score_box(t, ob_coords)
        obc = rt_.score_point(t, rt_.answer_center(ob_coords)) if ob_coords else {'valid': False, 'hit': False}
        row = {'id': cid, 'set': c['set'], 'subset': c['subset'], 'canvas': t['canvas'],
               'truth_center': rt_.to_units(t['center'], t['canvas']).tolist(),
               'truth_box': rt_.to_units(t['box'], t['canvas']).tolist(),
               'tool_point': p['point_2d'], 'tool_box': p['bbox_2d'], 'tool_raw': p['raw'],
               'tool_hit': tp['hit'], 'tool_error': tp['error'], 'tool_iou': tb['iou'], 'tool_dx': tp['dx'], 'tool_dy': tp['dy'],
               'own_valid': op['valid'], 'own_point': op.get('point'), 'own_hit': op['hit'], 'own_hit_any': op['hit_any'],
               'own_hit_pix_only': op.get('hit_pix_only', False), 'own_error': op.get('error'),
               'own_dx': op.get('dx'), 'own_dy': op.get('dy'),
               'own_box_valid': ob['valid'], 'own_box': ob.get('box'), 'own_iou': ob['iou'],
               'own_box_center_hit': obc['hit'], 'own_box_center_error': obc.get('error'),
               'area_frac': float((t['box'][2] - t['box'][0]) * (t['box'][3] - t['box'][1]) / (t['canvas'][0] * t['canvas'][1]))}
        if 'nearest_correct' in tp:
            row['tool_nearest'] = tp['nearest_correct']
            row['own_nearest'] = op.get('nearest_correct', False)
        if 'grid' in t:
            g = t['grid']
            row['grid'] = g
            row['tool_rc'] = tp['pred_rc']
            row['own_rc'] = op.get('pred_rc')
        for fmt, pr in c.get('prior', {}).items():
            if fmt == 'point':
                sp = rt_.score_point(t, rt_.parse_answer(pr['response'], 'point'))
                row['prior_point_valid'], row['prior_point_hit'], row['prior_point_error'] = sp['valid'], sp['hit'], sp.get('error')
                row['prior_point_hit_any'] = sp['hit_any']
                row['prior_point'] = sp.get('point')
        rows.append(row)
    return rows


def digit_stats(values: np.ndarray) -> dict:
    v = np.rint(values).astype(int)
    tens = (v // 10) % 10
    return {'n': int(len(v)), 'mult10': float((v % 10 == 0).mean()), 'mult50': float((v % 50 == 0).mean()),
            'top_values': [[int(a), int(b)] for b, a in sorted(((int(b), int(a)) for a, b in zip(*np.unique(v, return_counts=True))), reverse=True)[:8]],
            'tens_entropy_bits': float(-sum(p * math.log2(p) for p in np.bincount(tens, minlength=10) / len(v) if p > 0))}


def rowcol_eval(run: Path, cases: dict, tool_rows: dict) -> dict:
    rows = read_jsonl(run / 'rowcol.jsonl')
    if not rows:
        return {}
    out = defaultdict(dict)
    per = defaultdict(list)
    for r in rows:
        c = cases[r['id']]
        g = c['truth']['grid']
        tr, tc = g['target_rc']
        rc = rt_.parse_rowcol_free(r['text'])
        rec = {'id': r['id'], 'group': c['set'] if c['set'] == 't6' and False else c['subset'], 'valid': rc is not None,
               'correct': rc is not None and (rc[0] - 1, rc[1] - 1) == (tr, tc),
               'd_row': None if rc is None else rc[0] - 1 - tr, 'd_col': None if rc is None else rc[1] - 1 - tc,
               'n_tokens': r['n_tokens'], 'truncated': r['finish_reason'] != 'eos'}
        if 'tool_response' in r and r['tool_response'].get('point_2d'):
            p = rt_.score_point(c['truth'], r['tool_response']['point_2d'])
            rec['response_cell_correct'] = p['pred_rc'] == [tr, tc]
            # Naive arithmetic that assumes the grid fills the canvas: index = floor(u * n / 1000).
            u = np.asarray(r['tool_response']['point_2d'], float)
            rec['naive_correct'] = [int(u[1] * g['rows'] // 1000), int(u[0] * g['cols'] // 1000)] == [tr, tc]
            rec['followed_response_cell'] = rc is not None and [rc[0] - 1, rc[1] - 1] == p['pred_rc']
        per[(r['condition'], r['mode'])].append(rec)
    summary = {}
    for (cond, mode), recs in sorted(per.items()):
        n = len(recs)
        k = sum(x['correct'] for x in recs)
        s = {'n': n, 'acc': k / n, 'acc_ci': wilson(k, n), 'valid': sum(x['valid'] for x in recs) / n,
             'truncated': sum(x['truncated'] for x in recs), 'mean_tokens': float(np.mean([x['n_tokens'] for x in recs])),
             'col_plus1': sum(x['d_col'] == 1 for x in recs) / n, 'row_plus1': sum(x['d_row'] == 1 for x in recs) / n}
        for key in ('response_cell_correct', 'naive_correct', 'followed_response_cell'):
            vals = [x[key] for x in recs if key in x]
            if vals:
                s[key] = sum(vals) / len(vals)
        groups = defaultdict(list)
        for x in recs:
            groups[x['group']].append(x['correct'])
        s['by_group'] = {g: {'n': len(v), 'acc': sum(v) / len(v)} for g, v in sorted(groups.items())}
        summary[f'{cond}|{mode}'] = s
    # paired tests against the no-tool grid-size condition of the same mode
    for mode in ('direct', 'think'):
        base = {x['id']: x['correct'] for x in per.get(('notool_grid', mode), [])}
        for cond in ('own_grid', 'tool_grid', 'notool_nogrid'):
            cur = {x['id']: x['correct'] for x in per.get((cond, mode), [])}
            common = sorted(set(base) & set(cur))
            b = sum(cur[i] and not base[i] for i in common)
            c = sum(base[i] and not cur[i] for i in common)
            if common:
                summary[f'{cond}|{mode}']['vs_notool_grid'] = {'n': len(common), 'only_cond': b, 'only_base': c,
                                                               'p_mcnemar': mcnemar_exact(b, c)}
    return summary


def freecall_eval(run: Path, cases: dict, tool_rows: dict) -> dict:
    rows = read_jsonl(run / 'freecall.jsonl')
    out = {}
    for variant in ('instructed', 'uninstructed'):
        rs = [r for r in rows if r['variant'] == variant]
        if not rs:
            continue
        called = [r for r in rs if r['called']]
        d = {'n': len(rs), 'called': len(called), 'call_rate_ci': wilson(len(called), len(rs)),
             'target_equal_canonical': sum(r['target_generated'] == r['target_canonical'] for r in called)}
        by_set = defaultdict(list)
        for r in called:
            if 'tool' not in r:
                continue
            c = cases[r['id']]
            sp = rt_.score_point(c['truth'], r['tool']['point_2d'])
            base = tool_rows.get(r['id'])
            by_set[c['set']].append((sp['error'], sp['hit'], base['tool_error'] if base else float('nan'),
                                     base['tool_hit'] if base else False))
        d['by_set'] = {s: {'n': len(v), 'free_error': float(np.mean([x[0] for x in v])), 'free_hit': float(np.mean([x[1] for x in v])),
                           'canonical_error': float(np.mean([x[2] for x in v])), 'canonical_hit': float(np.mean([x[3] for x in v]))}
                       for s, v in sorted(by_set.items())}
        out[variant] = d
    return out


def stage_evaluate(run: Path) -> None:
    cases = {c['id']: c for c in read_jsonl(run / 'cases.jsonl')}
    rows = case_rows(run)
    (run / 'eval_cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    res = {'by_set': {}, 'by_subset': {}}
    for s in EVAL_SETS:
        rs = [r for r in rows if r['set'] == s]
        if not rs:
            continue
        d = {'hit': paired(rs, 'tool_hit', 'own_hit'), 'hit_any_own': sum(r['own_hit_any'] for r in rs) / len(rs),
             'own_pix_only': sum(r['own_hit_pix_only'] for r in rs), 'own_invalid': sum(not r['own_valid'] for r in rs),
             'error': error_stats(rs),
             'iou': {'tool_mean': float(np.mean([r['tool_iou'] for r in rs])), 'own_mean': float(np.mean([r['own_iou'] for r in rs])),
                     'tool_iou50': float(np.mean([r['tool_iou'] >= 0.5 for r in rs])),
                     'own_iou50': float(np.mean([r['own_iou'] >= 0.5 for r in rs])),
                     'own_box_invalid': sum(not r['own_box_valid'] for r in rs),
                     'own_box_center_hit': float(np.mean([r['own_box_center_hit'] for r in rs]))}}
        if 'tool_nearest' in rs[0]:
            d['nearest'] = paired(rs, 'tool_nearest', 'own_nearest')
        pri = [r for r in rs if 'prior_point_hit' in r]
        if pri:
            d['prior_point'] = {'n': len(pri), 'hit': float(np.mean([r['prior_point_hit'] for r in pri])),
                                'hit_any': float(np.mean([r['prior_point_hit_any'] for r in pri])),
                                'own_matched_hit': float(np.mean([r['own_hit'] for r in pri]))}
        res['by_set'][s] = d
        for sub in sorted({r['subset'] for r in rs}):
            ss = [r for r in rs if r['subset'] == sub]
            res['by_subset'][f'{s}/{sub}'] = {'hit': paired(ss, 'tool_hit', 'own_hit'), 'error': error_stats(ss),
                                              'own_pix_only': sum(r['own_hit_pix_only'] for r in ss),
                                              'own_hit_any': sum(r['own_hit_any'] for r in ss) / len(ss),
                                              'tool_iou': float(np.mean([r['tool_iou'] for r in ss])),
                                              'own_iou': float(np.mean([r['own_iou'] for r in ss]))}
            if 'tool_nearest' in ss[0]:
                res['by_subset'][f'{s}/{sub}']['nearest'] = paired(ss, 'tool_nearest', 'own_nearest')
    # digit-level defects on all held-out synthetic sets
    syn = [r for r in rows if r['set'] in ('disc', 't6', 'layout', 'ring')]
    own_vals = np.asarray([v for r in syn if r['own_valid'] for v in r['own_point']])
    tool_vals = np.asarray([v for r in syn for v in r['tool_point']])
    truth_vals = np.asarray([v for r in syn for v in r['truth_center']])
    res['digits'] = {'own': digit_stats(own_vals), 'tool': digit_stats(tool_vals), 'truth': digit_stats(truth_vals)}
    rc_ = [r for r in rows if r['set'] == 'refcoco']
    res['digits_refcoco'] = {'own': digit_stats(np.asarray([v for r in rc_ if r['own_valid'] for v in r['own_point']])),
                             'tool': digit_stats(np.asarray([v for r in rc_ for v in r['tool_point']]))}
    # column / row offsets of point answers on grids (cell units)
    offs = {}
    for s in ('t6', 'layout'):
        for who in ('own', 'tool'):
            rs = [r for r in rows if r['set'] == s and r.get(f'{who}_rc') is not None]
            dc = [r[f'{who}_rc'][1] - r['grid']['target_rc'][1] for r in rs]
            dr = [r[f'{who}_rc'][0] - r['grid']['target_rc'][0] for r in rs]
            offs[f'{s}/{who}'] = {'n': len(rs), 'col_plus1': float(np.mean(np.asarray(dc) == 1)), 'col_minus1': float(np.mean(np.asarray(dc) == -1)),
                                  'row_plus1': float(np.mean(np.asarray(dr) == 1)), 'row_minus1': float(np.mean(np.asarray(dr) == -1)),
                                  'mean_dx_units': float(np.mean([r[f'{who}_dx'] for r in rs])),
                                  'mean_dy_units': float(np.mean([r[f'{who}_dy'] for r in rs]))}
    res['grid_offsets'] = offs
    # error vs target size on RefCOCO
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01]
    res['refcoco_by_area'] = []
    for a, b in zip(bins[:-1], bins[1:]):
        ss = [r for r in rc_ if a <= r['area_frac'] < b]
        if ss:
            res['refcoco_by_area'].append({'area': [a, b], 'n': len(ss), 'hit': paired(ss, 'tool_hit', 'own_hit'),
                                           'error': error_stats(ss)})
    tool_rows = {r['id']: r for r in rows}
    res['rowcol'] = rowcol_eval(run, cases, tool_rows)
    res['freecall'] = freecall_eval(run, cases, tool_rows)
    res['fit'] = json.loads((run / 'tool' / 'fit.json').read_text())
    worst = defaultdict(list)
    for s in EVAL_SETS:
        rs = sorted([r for r in rows if r['set'] == s], key=lambda r: -r['tool_error'])
        worst[s] = [r['id'] for r in rs[:6]]
    res['worst'] = worst
    (run / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    for s, d in res['by_set'].items():
        e = d['error']
        print(f"{s:8s} n={d['hit']['n']:5d} hit tool {d['hit']['tool']:.3f} own {d['hit']['own']:.3f} p={d['hit']['p_mcnemar']:.2g} | "
              f"err tool {e['tool_mean']:.1f} own {e['own_mean']:.1f} | IoU tool {d['iou']['tool_mean']:.2f} own {d['iou']['own_mean']:.2f}")
    print(json.dumps(res['digits'], indent=0)[:800])
    print(json.dumps(offs, indent=0)[:1500])
    for k, v in res['rowcol'].items():
        print(k, round(v['acc'], 3), v.get('vs_notool_grid', ''), {kk: round(vv, 3) for kk, vv in v.items() if kk in ('response_cell_correct', 'naive_correct', 'followed_response_cell', 'col_plus1')})
    print(json.dumps(res['freecall'], indent=0)[:1500])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stage', choices=('fit', 'fit_slot', 'evaluate'), required=True)
    a = p.parse_args()
    {'fit': stage_fit, 'fit_slot': stage_fit_slot, 'evaluate': stage_evaluate}[a.stage](a.run)


if __name__ == '__main__':
    main()
