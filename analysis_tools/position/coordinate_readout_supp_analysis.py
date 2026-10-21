"""Metrics and figures of the supplementary stages of experiment 3.5.

Reads patching/, ablation/records_redundancy.jsonl, refcoco/ and insertion/ from the run directory
written by `coordinate_readout_supp.py`, writes `results_supp.json`, and draws the figures used in
the results document (`c4_*`, `rc_*`, `c5_*`).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.position.coordinate_readout_analysis import (FULL_LAYERS, bootstrap_mean, displacement_weight, prop,
                                                                read_jsonl, select_heads)
from analysis_tools.position.coordinate_readout_figures import (AQUA, BLUE, GREEN, INK, MAGENTA, MUTED, ORANGE, RED,
                                                               VIOLET, YELLOW, save)

PI5 = [(20, 10), (16, 11), (20, 14), (16, 15), (16, 6)]


def per_image(rows: list[tuple[str, float]]) -> list[float]:
    by = defaultdict(list)
    for i, v in rows:
        by[i].append(v)
    return [float(np.mean(v)) for v in by.values()]


# ----------------------------------------------------------------------------- C4 patching

def patch_metrics(recs: list[dict]) -> dict:
    """w of the patched output relative to the in-batch unpatched run A, toward T' (d) and toward B's output."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in recs:
        sub = 'correct' if r['baseline_correct'] else 'incorrect'
        by_pair = defaultdict(dict)
        for c in r['conditions']:
            by_pair[c['pair']][c['set']] = c
        for pair, cs in by_pair.items():
            none, bfree = cs.get('none'), cs.get('B_free')
            if none is None or none['center'] is None:
                continue
            dB = None if bfree is None or bfree['center'] is None else np.asarray(bfree['center']) - np.asarray(none['center'])
            k = int(pair[1:])
            for name, c in cs.items():
                if name in ('none', 'B_free'):
                    continue
                key = (name, k)
                if c['center'] is None:
                    acc[key]['invalid'].append(r['id'])
                    continue
                dy = np.asarray(c['center']) - np.asarray(none['center'])
                w, e = displacement_weight(dy, c['d'])
                for s in (sub, 'all', 'axis_' + pair[0]):
                    acc[key][s].append((r['id'], w))
                acc[key]['e'].append((r['id'], e))
                if dB is not None and np.linalg.norm(dB) > 20:
                    acc[key]['wB'].append((r['id'], float(dy @ dB / (dB @ dB))))
                acc[key]['equalB'].append(bfree is not None and c['center'] == bfree['center'])
            if bfree is not None and bfree['center'] is not None:
                w, _ = displacement_weight(dB, cs['none']['d'])
                acc[('B_free', k)]['all'].append((r['id'], w))
                acc[('B_free', k)]['axis_' + pair[0]].append((r['id'], w))
    out = {}
    for (name, k), v in acc.items():
        out[f'{name}|k{k}'] = {s: bootstrap_mean(per_image(v[s])) for s in ('all', 'correct', 'incorrect', 'wB', 'e', 'axis_h', 'axis_v')
                               if v.get(s)}
        if v.get('equalB'):
            out[f'{name}|k{k}']['equal_B'] = prop(int(sum(v['equalB'])), len(v['equalB']))
        out[f'{name}|k{k}']['invalid'] = len(v.get('invalid', []))
    return out


# ----------------------------------------------------------------------------- C4 redundancy

def redund_metrics(recs: list[dict], cases: dict) -> dict:
    out = {}
    names = [c['set'] for c in recs[0]['conditions']]
    for name in names:
        derr, drec, moved, hit, n_valid, pred, true = [], [], [], [], 0, [], []
        for r in recs:
            cs = {c['set']: c for c in r['conditions']}
            b, c = cs['none'], cs[name]
            hit.append(bool(c['score'].get('hit')))
            if not (b['score']['valid'] and c['score']['valid']):
                continue
            n_valid += 1
            pred.append(c['score']['center'])
            case = cases[r['id']]
            image_truth = (1000 * np.asarray(case['center']) / case['canvas']).tolist()
            true.append(image_truth)
            derr.append(c['score']['error'] - b['score']['error'])
            drec.append(c['recognition'] - b['recognition'])
            moved.append(float(np.linalg.norm(np.asarray(c['score']['center']) - np.asarray(b['score']['center']))))
        out[name] = {'n_heads': recs[0]['conditions'][names.index(name)]['n_heads'], 'd_error': bootstrap_mean(derr),
                     'd_recognition': bootstrap_mean(drec), 'displacement': bootstrap_mean(moved),
                     'hit': prop(int(sum(hit)), len(hit)), 'valid': prop(n_valid, len(recs))}
        if len(pred) > 2:
            P, T = np.asarray(pred), np.asarray(true)
            # Center coordinates use the drawn center; partially clipped discs differ by at most a few units.
            out[name]['corr_truth'] = [float(np.corrcoef(P[:, a], T[:, a])[0, 1]) for a in (0, 1)]
            out[name]['pred_sd'] = P.std(0).tolist()
    return out


def post_metrics(recs: list[dict]) -> dict:
    """Mean ablation from the answer positions versus from the first post-image token."""
    out = {}
    keys = sorted({(c['set'], c['from']) for c in recs[0]['conditions']})
    for name, where in keys:
        pred, true, hit, moved = [], [], [], []
        for r in recs:
            cs = {(c['set'], c['from']): c for c in r['conditions']}
            c, b = cs[(name, where)], cs[('none', where)]
            hit.append(bool(c['score'].get('hit')))
            if c['score']['valid']:
                pred.append(c['score']['center'])
                true.append(r['truth_center'])
                if b['score']['valid']:
                    moved.append(float(np.linalg.norm(np.asarray(c['score']['center']) - np.asarray(b['score']['center']))))
        P, T = np.asarray(pred), np.asarray(true)
        out[f'{name}|{where}'] = {'hit': prop(int(sum(hit)), len(hit)), 'valid': prop(len(pred), len(recs)),
                                  'displacement': bootstrap_mean(moved),
                                  'corr_truth': [float(np.corrcoef(P[:, a], T[:, a])[0, 1]) for a in (0, 1)] if len(P) > 2 else None}
    return out


# ----------------------------------------------------------------------------- RefCOCO

def convention_table(cases: dict, base: list[dict]) -> dict:
    """Strict and either-convention point hits of the C0 point answers (plan.md section 6 item 9).

    Target region: tight box of the disc, or the target cell for T6 grids. Uses the shared helper.
    """
    from analysis_tools.behavior.coordinate_convention import chance_rates, convention_hits
    acc = defaultdict(lambda: defaultdict(list))
    for r in base:
        c = cases[r['id']]
        if r['format'] != 'point':
            continue
        if c['set'] == 't6':
            box, canvas, key = c['truth']['target_cell_box'], c['truth']['canvas'], f"t6_{c['grid']}x{c['grid']}"
        else:
            box, canvas = r['truth']['box'], r['truth']['canvas']
            key = c['set'] if c['set'] != 'size' else f"size_{c['canvas']}"
        pt = tuple(r['score']['coords']) if r['score'].get('valid') else None
        h = convention_hits(pt, canvas, box)
        for k in ('h_norm', 'h_pix', 'h_any', 'pixel_only'):
            acc[key][k].append(h[k])
        for k, v in chance_rates(canvas, box).items():
            acc[key][k].append(v)
    return {key: {**{k: prop(int(sum(v[k])), len(v[k])) for k in ('h_norm', 'h_pix', 'h_any', 'pixel_only')},
                  **{k: float(np.mean(v[k])) for k in ('p_norm', 'p_pix', 'p_any')}} for key, v in acc.items()}


def rc0_metrics(base: list[dict], spot: list[dict]) -> dict:
    from analysis_tools.position.coordinate_readout_supp import point_conventions
    out = {}
    for fmt in ('point', 'bbox'):
        rs = [r for r in base if r['format'] == fmt]
        # Convention diagnostics are recomputed with the shared helper (plan.md section 6 item 9).
        for r in rs:
            if r['score']['valid']:
                r['score'].update(point_conventions(r['score']['center'], r['truth']['box'], r['truth']['canvas']))
        s = [r['score'] for r in rs]
        v = [x for x in s if x['valid']]
        e = {'n': len(rs), 'valid': prop(len(v), len(rs)), 'hit_norm': prop(sum(x['h_norm'] for x in v), len(rs)),
             'hit_pix': prop(sum(x['h_pix'] for x in v), len(rs)), 'hit_any': prop(sum(x['h_any'] for x in v), len(rs)),
             'pix_only': prop(sum(x['h_pix'] and not x['h_norm'] for x in v), len(rs)),
             'chance_norm': float(np.mean([x['chance_norm'] for x in v])), 'chance_pix': float(np.mean([x['chance_pix'] for x in v])),
             'chance_any': float(np.mean([x['chance_any'] for x in v])), 'error': bootstrap_mean([x['error'] for x in v]),
             'max_coord_over_1000': int(sum(max(x['coords']) > 1000 for x in v))}
        if fmt == 'bbox':
            e['iou50'] = prop(sum(x['iou'] >= 0.5 for x in v), len(rs))
            e['iou50_pix'] = prop(sum(x['iou_pix'] >= 0.5 for x in v), len(rs))
            e['iou_mean'] = bootstrap_mean([x['iou'] for x in v])
        # Regression slope of the predicted center on the true center (0-1000).
        if v:
            tc = np.array([1000 * np.asarray(r['truth']['center']) / np.asarray(r['truth']['canvas']) for r in rs if r['score']['valid']])
            pc = np.array([x['center'] for x in v])
            e['slope_xy'] = [float(np.polyfit(tc[:, a], pc[:, a], 1)[0]) for a in (0, 1)]
        out[fmt] = e
    out['spotcheck'] = {'n': len(spot), 'answer_equal': int(sum(r['answer_equal'] for r in spot)),
                        'text_equal': int(sum(r['text_equal'] for r in spot))}
    return out


def rc1_metrics(att: list[dict]) -> dict:
    out = {}
    for fmt in ('point', 'bbox'):
        rs = [r for r in att if r['format'] == fmt and r.get('status') == 'ok']
        if not rs:
            continue
        e = {'n': len(rs), 'n_correct': sum(r['correct'] for r in rs)}
        prof = {}
        for q in ('q_end', 'think_end', 'last', 'first'):
            vm, mx = [], []
            for r in rs:
                names = r['query_names']
                idx = [i for i, n in enumerate(names) if n.endswith('d0') and n.startswith('v')] if q == 'first' else [names.index(q)]
                vm.append(np.asarray(r['vis_mass'])[:, :, idx].mean((1, 2)))
                mx.append(np.log(np.maximum(np.asarray(r['E'])[:, :, idx], 1e-6)).mean(-1).max(1))
            prof[q] = {'vis_mass': np.mean(vm, 0).tolist(), 'max_logE': np.mean(mx, 0).tolist()}
        e['layer_profile'] = prof
        # Candidate heads from the disc task, scored on RefCOCO (no reselection), and heads selected on RefCOCO itself.
        logE, hit = [], []
        for r in rs:
            idx = [i for i, n in enumerate(r['query_names']) if n.endswith('d0') and n.startswith('v')]
            logE.append(np.log(np.maximum(np.asarray(r['E'])[:, :, idx], 1e-6)).mean(-1))
            hit.append(np.asarray(r['hit'])[:, :, idx].mean(-1))
        logE, hit = np.mean(logE, 0), np.mean(hit, 0)
        e['pi5'] = [{'layer': l, 'head': h, 'logE': float(logE[FULL_LAYERS.index(l), h]), 'hit': float(hit[FULL_LAYERS.index(l), h]),
                     'rank': int((logE > logE[FULL_LAYERS.index(l), h]).sum()) + 1} for l, h in PI5]
        e['logE_map'] = logE.tolist()
        e['selected_on_refcoco'] = {k: v for k, v in select_heads(rs).items() if k in ('heads', 'validation', 'n_selection')}
        # Attention centroid of the five heads vs truth center and output (point answers only).
        if fmt == 'point':
            rows = []
            for r in rs:
                if 'center' not in r['score']:
                    continue                  # unparsable answer: no output coordinate to compare
                c = np.asarray(r['centroid'])
                q = [i for i, n in enumerate(r['query_names']) if n.endswith('d0') and n.startswith('v')]
                cen = np.mean([c[FULL_LAYERS.index(l), h, q[0]] for l, h in PI5], 0)
                rows.append({'correct': r['correct'], 'centroid': cen.tolist(), 'truth': r['truth_center'],
                             'output': r['score']['center']})
            for sub in ('correct', 'incorrect'):
                g = [x for x in rows if x['correct'] == (sub == 'correct')]
                if len(g) < 3:
                    continue
                C, T, O = (np.array([x[k] for x in g]) for k in ('centroid', 'truth', 'output'))
                e[f'centroid_{sub}'] = {'n': len(g), 'corr_truth': [float(np.corrcoef(C[:, a], T[:, a])[0, 1]) for a in (0, 1)],
                                        'corr_output': [float(np.corrcoef(C[:, a], O[:, a])[0, 1]) for a in (0, 1)],
                                        'dist_truth': float(np.linalg.norm(C - T, axis=1).mean()),
                                        'dist_output': float(np.linalg.norm(C - O, axis=1).mean())}
        rho_c = [np.asarray(r['rho']).mean(0) for r in rs if r['correct'] and np.isfinite(r['rho']).all()]
        share = [np.asarray(r['grad_share_image']).mean(0) for r in rs if np.isfinite(r['grad_share_image']).all()]
        e['rho_correct'] = np.mean(rho_c, 0).tolist() if rho_c else None
        e['image_share'] = np.mean(share, 0).tolist() if share else None
        e['target_fraction'] = float(np.mean([len(r['target_tokens']) / (r['grid'][0] * r['grid'][1]) for r in rs]))
        out[fmt] = e
    return out


RC_FAMILIES = ('s2shift', 's1shift', 's2swap', 's1swap_rope', 's1swap_both', 's4perm', 'c3content')


def rc2_metrics(recs: list[dict]) -> dict:
    fam_rows = defaultdict(lambda: defaultdict(list))
    axis_rows = defaultdict(list)
    gap_diff, rope_shift, identical = [], [], 0
    n_swap = {'h': 0, 'v': 0}
    for r in recs:
        cs = {c['condition']: c for c in r['conditions']}
        std = cs['standard']
        if std['center'] is None:
            continue
        for g in r['regions']:
            n_swap[g['axis']] += g['valid']
        if cs['gap']['center'] is not None:
            gap_diff.append(float(np.linalg.norm(np.asarray(cs['gap']['center']) - np.asarray(std['center']))))
            identical += cs['gap']['center'] == std['center']
        if cs['s1rope_shift']['center'] is not None:
            rope_shift.append(float(np.linalg.norm(np.asarray(cs['s1rope_shift']['center']) - np.asarray(std['center']))))
        sub = 'correct' if r['baseline_correct'] else 'incorrect'
        for name, c in cs.items():
            fam = next((f for f in RC_FAMILIES if name.startswith(f + '_')), None)
            if fam is None or c['d'] is None or c['center'] is None:
                continue
            ref = cs[c.get('ref', 'standard')]
            if ref['center'] is None:
                continue
            dy = np.asarray(c['center']) - np.asarray(ref['center'])
            w, e = displacement_weight(dy, c['d'])
            for s in (sub, 'all'):
                fam_rows[fam][s].append((r['id'], w))
            fam_rows[fam]['e'].append((r['id'], e))
            ax = 0 if abs(c['d'][0]) > 0 else 1
            axis_rows[(fam, 'xy'[ax])].append((r['id'], dy[ax] / c['d'][ax]))
    out = {'n_images': len(recs), 'n_swap_valid': n_swap, 'gap_center_diff': bootstrap_mean(gap_diff),
           'gap_identical': prop(identical, len(gap_diff)), 'rope_shift_center_diff': bootstrap_mean(rope_shift), 'families': {}}
    for fam, subs in fam_rows.items():
        out['families'][fam] = {s: bootstrap_mean(per_image(v)) for s, v in subs.items()}
    out['by_axis'] = {f'{f}|{a}': bootstrap_mean(per_image(v)) for (f, a), v in sorted(axis_rows.items())}
    return out


# ----------------------------------------------------------------------------- C5

def c5_metrics(recs: list[dict]) -> dict:
    out = {}
    for grid in sorted({r['grid'] for r in recs}):
        rs = [r for r in recs if r['grid'] == grid]
        names = [c['condition'] for c in rs[0]['conditions']]
        e = {'n': len(rs)}
        for sub in ('all', 'correct'):
            g = [r for r in rs if sub == 'all' or {c['condition']: c for c in r['conditions']}['standard']['score'].get('correct')]
            ent = {'n': len(g)}
            for name in names:
                dr, dc, ch, inv, corr = [], [], 0, 0, 0
                for r in g:
                    cs = {c['condition']: c for c in r['conditions']}
                    s0, s = cs['standard']['score'], cs[name]['score']
                    if not (s0.get('valid') and s.get('valid')):
                        inv += 1
                        continue
                    dr.append(s['pred_rc'][0] - s0['pred_rc'][0])
                    dc.append(s['pred_rc'][1] - s0['pred_rc'][1])
                    ch += (dr[-1], dc[-1]) != (0, 0)
                    corr += s['correct']
                ent[name] = {'d_row': bootstrap_mean(dr), 'd_col': bootstrap_mean(dc), 'changed': prop(ch, len(dr)),
                             'correct': prop(corr, len(g)), 'invalid': inv,
                             'row_plus1': prop(sum(x == 1 for x in dr), len(dr)), 'col_plus1': prop(sum(x == 1 for x in dc), len(dc))}
            e[sub] = ent
        out[str(grid)] = e
    return out


def row_attention_profile(run: Path, att: list[dict], cases: dict, heads: list) -> dict:
    """Attention mass of the 6x6 row-column pointing heads per grid row (column) relative to the target,
    at the row (column) digit query, and the share on rows above / below the target row."""
    out = {}
    for r in att:
        if r['set'] != 't6' or r['format'] != 'rowcol':
            continue
        maps = np.load(run / 'attention' / 'maps' / f"{r['id']}.npz")['attn'].astype(float)
        H, W = r['grid']
        tr, tc = cases[r['id']]['truth']['target_rc']
        n = H // 2
        for qi, axis, t in ((3, 'row', tr), (4, 'column', tc)):
            prof = np.full(2 * n - 1, np.nan)
            prof[n - 1 - t:n - 1 - t + n] = 0
            for l, h in heads:
                a = maps[FULL_LAYERS.index(l), h, qi].reshape(H, W)
                v = (a.sum(1) if axis == 'row' else a.sum(0)).reshape(-1, 2).sum(1)
                v = v / max(v.sum(), 1e-12)
                prof[n - 1 - t:n - 1 - t + n] += v / len(heads)
            key = f"g{n}|{axis}|{'correct' if r['correct'] else 'incorrect'}"
            out.setdefault(key, []).append(prof)
    res = {}
    for k, v in out.items():
        n = (len(v[0]) + 1) // 2
        P = np.array(v)
        # Offsets outside the grid of a sample are NaN and excluded from the mean at that offset.
        res[k] = {'n': len(v), 'profile': np.nanmean(P, 0).tolist(), 'offsets': list(range(-(n - 1), n)),
                  'before': float(np.nansum(P[:, :n - 1], 1).mean()), 'target': float(P[:, n - 1].mean()),
                  'after': float(np.nansum(P[:, n:], 1).mean())}
    return res


# ----------------------------------------------------------------------------- figures

def fig_patch(res: dict, out: Path) -> None:
    p = res['patch']
    groups = [('single heads', [f'h{l}_{h}' for l, h in PI5], BLUE),
              ('head sets', ['pi5', 'cand7', 'top10', 'top20', 'top40', 'full_all'], VIOLET),
              ('random 5', ['rand5L_0', 'rand5L_1', 'rand5L_2'], MUTED),
              ('one full layer', [f'full_L{l}' for l in FULL_LAYERS], AQUA),
              ('Gated DeltaNet', ['lin_early', 'lin_mid', 'lin_late', 'lin_all'], ORANGE),
              ('all mixers', ['mixers_all'], INK)]
    labels = {'pi5': '5 candidates', 'cand7': '7 candidates', 'top10': 'top 10', 'top20': 'top 20', 'top40': 'top 40',
              'full_all': 'all 128 full', 'lin_early': 'DeltaNet 1-11', 'lin_mid': 'DeltaNet 13-23', 'lin_late': 'DeltaNet 25-31',
              'lin_all': 'all 24 DeltaNet', 'mixers_all': 'all mixers'}
    names, cols = [], []
    for _, ns, c in groups:
        names += [n for n in ns if f'{n}|k4' in p]
        cols += [c] * len([n for n in ns if f'{n}|k4' in p])
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(names))
    for i, (n, c) in enumerate(zip(names, cols)):
        e = p[f'{n}|k4']['all']
        ax.bar(i, e['mean'], color=c, width=0.7)
        ax.errorbar(i, e['mean'], yerr=[[e['mean'] - e['ci'][0]], [e['ci'][1] - e['mean']]], color=INK, lw=1, capsize=2)
    ax.axhline(1, color=MUTED, ls='--', lw=1)
    if 'B_free|k4' in p:
        ax.axhline(p['B_free|k4']['all']['mean'], color=RED, ls=':', lw=1.2, label='run B itself')
        ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(x)
    def label(n: str) -> str:
        if n in labels:
            return labels[n]
        if n.startswith('full_L'):
            return f'all heads of L{n[6:]}'
        if n.startswith('rand5L_'):
            return f'random 5 (draw {int(n[-1]) + 1})'
        l, h = n[1:].split('_')
        return f'L{l} H{h}'
    ax.set_xticklabels([label(n) for n in names],
                       rotation=50, ha='right', fontsize=8)
    ax.set_ylabel('patching weight w (toward T\')')
    ax.set_title('Output patching from run B (disc at T\', 4 tokens away) into run A at the answer positions', fontsize=9)
    save(fig, out / 'c4_patching.png')


def fig_l20(res: dict, out: Path) -> None:
    """Layer-20 sweep: patching one head only, and patching all heads but one."""
    p = res['patch_L20']
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    hs = np.arange(16)
    for ax, kind, title in ((axes[0], 'only', 'patch one head of layer 20'), (axes[1], 'drop', 'patch all heads of layer 20 except one')):
        for off, axis, col, lab in ((-0.2, 'axis_h', BLUE, 'horizontal pairs (x)'), (0.2, 'axis_v', RED, 'vertical pairs (y)')):
            vals = [p[f'{kind}_{h}|k4'][axis] for h in hs]
            ax.bar(hs + off, [v['mean'] for v in vals], width=0.4, color=col, label=lab)
            ax.errorbar(hs + off, [v['mean'] for v in vals], yerr=[[v['mean'] - v['ci'][0] for v in vals], [v['ci'][1] - v['mean'] for v in vals]],
                        fmt='none', color=INK, lw=0.8, capsize=1.5)
        for axis, col in (('axis_h', BLUE), ('axis_v', RED)):
            ax.axhline(p['full_L20|k4'][axis]['mean'], color=col, ls=':', lw=1.2)
        ax.set_xticks(hs)
        ax.set_xlabel('head of layer 20')
        ax.set_title(title + ' (dotted: all 16 heads)', fontsize=9)
    axes[0].set_ylabel('patching weight w')
    axes[0].legend(fontsize=8)
    save(fig, out / 'c4_layer20.png')


def fig_redund(res: dict, out: Path) -> None:
    """Mean ablation at the answer positions: top-k heads by pointing score versus random heads."""
    r = res['redundancy']
    ks = [5, 10, 20, 40, 64, 128]
    top = ['top5', 'top10', 'top20', 'top40', 'top64', 'full_all']
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    panels = (('displacement', 'output displacement (0-1000)', lambda e: e['mean']),
              ('hit', 'hit rate', lambda e: e['p']))
    for ax, (metric, label, get) in zip(axes[:2], panels):
        ax.plot(ks, [get(r[n][metric]) for n in top], '-o', color=BLUE, ms=5, label='top-k by pointing score')
        for i in range(3):
            pts = [(k, get(r[f'rand{k}_{i}'][metric])) for k in ks if f'rand{k}_{i}' in r]
            ax.plot(*zip(*pts), 'o', color=MUTED, ms=4, label='random k outside the top 40' if i == 0 else None)
        ax.plot(16, get(r['full_L20'][metric]), 's', color=ORANGE, ms=7, label='all 16 heads of layer 20')
        ax.plot(5, get(r['pi5'][metric]), 'D', color=VIOLET, ms=6, label='5 candidate heads')
        if metric == 'hit':
            ax.axhline(r['none']['hit']['p'], color=GREEN, ls='--', lw=1, label='no ablation')
        ax.set_ylabel(label)
    for axis, col, mk in ((0, BLUE, '-o'), (1, RED, '-s')):
        axes[2].plot(ks, [r[n]['corr_truth'][axis] for n in top], mk, color=col, ms=5,
                     label=f"{'x' if axis == 0 else 'y'}: top-k by pointing score")
        axes[2].plot(16, r['full_L20']['corr_truth'][axis], 'D', color=col, ms=6, mfc='white',
                     label=f"{'x' if axis == 0 else 'y'}: all heads of layer 20")
    axes[2].set_ylabel('correlation of output with true center')
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.set_xlabel('number of mean-ablated full-attention heads')
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    save(fig, out / 'c4_redundancy.png')


def fig_rc_examples(run: Path, rcases: dict, base: list[dict], out: Path) -> None:
    pts = [r for r in base if r['format'] == 'point'][:300]
    rng = np.random.default_rng(3)
    pick = [pts[i] for i in rng.choice(len(pts), 4, replace=False)]
    fig, grid = plt.subplots(2, 4, figsize=(15, 7))
    for ax2, r in zip(grid[1], pick):
        # Row 2: mean attention of the five disc candidate heads at the first digit of x, over visual tokens.
        c = rcases[r['id']]
        im = Image.open(c['image_path']).convert('RGB')
        maps = np.load(run / 'refcoco' / 'attention' / 'maps' / f"{r['id']}.npz")['attn'].astype(float)
        H, W = r['grid']
        a = np.mean([maps[FULL_LAYERS.index(l), h, 3] for l, h in PI5], 0).reshape(H, W)
        ax2.imshow(im)
        ax2.imshow(a / a.max(), extent=(0, im.width, im.height, 0), cmap='magma', alpha=0.6)
        b = r['truth']['box']
        ax2.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=AQUA, lw=1.5))
        ax2.set_title('5 candidate heads, query: first digit of x', fontsize=8)
        ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)
    axes = grid[0]
    for ax, r in zip(axes, pick):
        c = rcases[r['id']]
        im = Image.open(c['image_path']).convert('RGB')
        ax.imshow(im)
        b = r['truth']['box']
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=AQUA, lw=2))
        if r['score']['valid']:
            W, H = r['truth']['canvas']
            x, y = r['score']['center']
            ax.plot(x * W / 1000, y * H / 1000, 'x', color=RED, ms=12, mew=3)
        ax.set_title(f"\"{c['expression']}\"  {'hit' if r['score'].get('hit') else 'miss'}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    save(fig, out / 'rc_inputs.png')


def fig_rc_profile(res: dict, disc: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    p = res['rc1']['point']['layer_profile']
    for q, c, lab in (('first', BLUE, 'coordinate first digit'), ('q_end', MUTED, 'question end'), ('last', ORANGE, 'last prompt token')):
        axes[0].plot(FULL_LAYERS, p[q]['vis_mass'], '-o', color=c, ms=4, label=f'RefCOCO, {lab}')
    dp = disc['layer_profile']
    axes[0].plot(FULL_LAYERS, dp['first']['vis_mass'] if 'first' in dp else dp['v0d0']['vis_mass'], '--', color=BLUE, lw=1.2,
                 label='single disc, coordinate first digit')
    axes[0].set_ylabel('image attention mass (mean over heads)')
    axes[0].set_xlabel('full-attention layer')
    axes[0].set_xticks(FULL_LAYERS)
    axes[0].legend(fontsize=7)
    rho = res['rc1']['point']['rho_correct']
    share = res['rc1']['point']['image_share']
    L = np.arange(1, len(rho) + 1)
    axes[1].plot(L, rho, color=BLUE, label='RefCOCO, target share of image attribution (correct)')
    axes[1].plot(L, share, color=ORANGE, label='RefCOCO, image share of all attribution')
    axes[1].plot(L, disc['rho_correct']['mean'], '--', color=BLUE, lw=1.2, label='single disc, target share')
    axes[1].axhline(res['rc1']['point']['target_fraction'], color=MUTED, ls=':', lw=1, label='RefCOCO target area fraction')
    for l in FULL_LAYERS:
        axes[1].axvline(l, color='#ececea', lw=0.8, zorder=0)
    axes[1].set_xlabel('decoder layer (input)')
    axes[1].legend(fontsize=7)
    save(fig, out / 'rc_attention.png')


def fig_rc_weights(res: dict, disc_c2: dict, out: Path) -> None:
    fams = [('s2shift', 'MRoPE global shift'), ('s1shift', 'learned PE shift'), ('s2swap', 'MRoPE swap'),
            ('s1swap_rope', 'ViT RoPE swap'), ('s1swap_both', 'ViT PE+RoPE swap'), ('s4perm', 'sequence permutation'),
            ('c3content', 'content swap')]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (f, lab) in enumerate(fams):
        for off, src, col, name in ((-0.18, res['rc2']['families'].get(f, {}).get('all'), ORANGE, 'RefCOCO'),
                                    (0.18, disc_c2['families'].get(f, {}).get('all', {}).get('w'), BLUE, 'single disc')):
            if not src:
                continue
            ax.bar(i + off, src['mean'], width=0.34, color=col, label=name if i == 0 else None)
            ax.errorbar(i + off, src['mean'], yerr=[[src['mean'] - src['ci'][0]], [src['ci'][1] - src['mean']]], color=INK, lw=1, capsize=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([l for _, l in fams], fontsize=8, rotation=25, ha='right')
    ax.set_ylabel('displacement weight w')
    ax.legend(fontsize=8)
    save(fig, out / 'rc_weights.png')


def fig_c5(res: dict, out: Path) -> None:
    conds = [('pix_v_target', 'band above target row'), ('pix_v_top', 'band at top edge'), ('pix_v_bottom', 'band at bottom edge'),
             ('id_v_target', 'MRoPE rows +2 below insertion'), ('pix_h_target', 'band left of target column'),
             ('pix_h_left', 'band at left edge'), ('pix_h_right', 'band at right edge'), ('id_h_target', 'MRoPE cols +2 right of insertion')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, key, lab in ((axes[0], 'd_row', 'mean change of row answer'), (axes[1], 'd_col', 'mean change of column answer')):
        for j, (g, col) in enumerate((('6', BLUE), ('10', ORANGE), ('20', VIOLET))):
            if g not in res['c5']:
                continue
            e = res['c5'][g]['correct']
            vals = [e[c][key] for c, _ in conds]
            xs = np.arange(len(conds)) + (j - 1) * 0.26
            ax.bar(xs, [v['mean'] for v in vals], width=0.26, color=col, label=f'{g}x{g} (n={e["n"]})')
            ax.errorbar(xs, [v['mean'] for v in vals], yerr=[[v['mean'] - v['ci'][0] for v in vals], [v['ci'][1] - v['mean'] for v in vals]],
                        fmt='none', color=INK, lw=0.8, capsize=1.5)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([l for _, l in conds], rotation=40, ha='right', fontsize=8)
        ax.set_title(lab, fontsize=9)
    axes[0].set_ylabel('answer change (cells), baseline-correct grids')
    axes[0].legend(fontsize=8)
    save(fig, out / 'c5_insertion.png')


def fig_c5_example(run: Path, cases: dict, recs: list[dict], out: Path) -> None:
    from analysis_tools.position import coordinate_readout as cr
    from analysis_tools.position.coordinate_readout_run import t6_input
    r = next(x for x in recs if x['grid'] == 6 and x['truth_rc'][0] >= 2)
    image, truth, _ = t6_input(cases[r['id']])
    cs = {c['condition']: c for c in r['conditions']}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    imgs = [('standard', image), ('pix_v_target', cr.insert_band(image, 'v', truth['target_rc'][0] * truth['cell'], truth['cell'])),
            ('pix_h_target', cr.insert_band(image, 'h', truth['target_rc'][1] * truth['cell'], truth['cell']))]
    for ax, (name, im) in zip(axes, imgs):
        ax.imshow(im)
        ans = cs[name]['text'].strip().replace('\n', ' ')
        title = {'standard': 'original grid', 'pix_v_target': 'blank band above the target row',
                 'pix_h_target': 'blank band left of the target column'}[name]
        ax.set_title(f'{title}\nanswer: {ans.replace("```json", "").replace("```", "").strip()}', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for side in ('top', 'right', 'bottom', 'left'):
            ax.spines[side].set_visible(True)
    save(fig, out / 'c5_example.png')


def fig_row_profile(res: dict, out: Path) -> None:
    prof = res['row_profile']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, axis in zip(axes, ('row', 'column')):
        for g, col in ((6, BLUE), (10, ORANGE), (20, VIOLET)):
            e = prof.get(f'g{g}|{axis}|correct')
            if not e:
                continue
            off = np.array(e['offsets'])
            m = np.abs(off) <= 5
            ax.plot(off[m], np.array(e['profile'])[m], '-o', color=col, ms=4, label=f"{g}x{g} correct (n={e['n']})")
        ax.set_xlabel(f'grid {axis} relative to the target {axis}')
        ax.set_title(f'query: {axis} digit', fontsize=9)
    axes[0].set_ylabel('share of image attention (pointing heads)')
    axes[0].legend(fontsize=8)
    save(fig, out / 'c5_row_attention.png')


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args()
    run, figs = args.run, args.figures
    figs.mkdir(parents=True, exist_ok=True)
    main_res = json.loads((run / 'results.json').read_text())
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    res = {}
    patch = read_jsonl(run / 'patching' / 'records.jsonl')
    if patch:
        res['patch'] = patch_metrics(patch)
        l20 = read_jsonl(run / 'patching' / 'records_L20.jsonl')
        if l20:
            res['patch_L20'] = patch_metrics(l20)
            res['patch_L20_n_images'] = len(l20)
        res['patch_n_images'] = len(patch)
    red = read_jsonl(run / 'ablation' / 'records_redundancy.jsonl')
    if red:
        res['redundancy'] = redund_metrics(red, cases)
        res['redundancy_n_images'] = len(red)
    post = read_jsonl(run / 'ablation' / 'records_post_image.jsonl')
    if post:
        res['post_image'] = post_metrics(post)
        res['post_image_n_images'] = len(post)
    res['c0_conventions'] = convention_table(cases, read_jsonl(run / 'baseline' / 'records.jsonl'))
    rbase = read_jsonl(run / 'refcoco' / 'baseline.jsonl')
    if rbase:
        res['rc0'] = rc0_metrics(rbase, read_jsonl(run / 'refcoco' / 'spotcheck.jsonl'))
    ratt = read_jsonl(run / 'refcoco' / 'attention' / 'records.jsonl')
    if ratt:
        tc = {r['id']: (1000 * np.asarray(r['truth']['center']) / np.asarray(r['truth']['canvas'])).tolist() for r in rbase}
        for r in ratt:
            if r.get('status') == 'ok':
                r['truth_center'] = tc[r['id']]
        res['rc1'] = rc1_metrics(ratt)
        res['rc1_failed'] = [r['id'] for r in ratt if r.get('status') != 'ok']
    rint = read_jsonl(run / 'refcoco' / 'interventions.jsonl')
    if rint:
        res['rc2'] = rc2_metrics(rint)
    ins = read_jsonl(run / 'insertion' / 'records.jsonl')
    if ins:
        res['c5'] = c5_metrics(ins)
    att = [r for r in read_jsonl(run / 'attention' / 'records.jsonl') if r.get('status') == 'ok' and r['set'] == 't6']
    heads = main_res['c1']['t6g06_rowcol']['candidates']['heads']
    res['row_profile'] = row_attention_profile(run, att, cases, heads)
    (run / 'results_supp.json').write_text(json.dumps(res, indent=1, default=float))
    print('wrote', run / 'results_supp.json')
    if 'patch' in res:
        fig_patch(res, figs)
    if 'redundancy' in res:
        fig_redund(res, figs)
    if 'patch_L20' in res:
        fig_l20(res, figs)
    if rbase:
        fig_rc_examples(run, {c['id']: c for c in json.loads((run / 'refcoco' / 'cases.json').read_text())}, rbase, figs)
    if 'rc1' in res:
        fig_rc_profile(res, main_res['c1']['disc_point'], figs)
    if 'rc2' in res:
        fig_rc_weights(res, main_res['c2'], figs)
    if 'c5' in res:
        fig_c5(res, figs)
        fig_c5_example(run, cases, ins, figs)
    fig_row_profile(res, figs)


if __name__ == '__main__':
    main()
