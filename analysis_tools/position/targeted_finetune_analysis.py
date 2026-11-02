"""Experiment 3.13 CPU analysis (no model weights are loaded). Design: docs/analysis/position/design_targeted_finetune.md.

Sections of results.json:
  training    loss curves and adapter sizes / update norms per condition;
  coord       coordinate metrics per test group and condition, paired comparisons with the base model;
  general     POPE / TextVQA / MMVP / GQA accuracy and the text-only sanity set, paired with the base model;
  mechanism   the 3.12 measurements on every condition: second-digit slope, image x first-digit interaction share,
              blank-canvas prior P(9|4) / P(0|5), digit probabilities, the q1 ridge probe and DLA;
  change      what moved relative to the base model: per-component DLA differences and residual-stream changes
              at q0 / q1 per layer (relative norm and base-probe transfer).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis_tools.position import digit_failure as df
from analysis_tools.position import digit_failure_analysis as dfa
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position.coordinate_readout_run import read_jsonl
from analysis_tools.position.coordinate_value_analysis import ridge_fit, ridge_predict
from analysis_tools.position.coordinate_value_r2_analysis import mcnemar_p, mean_ci, prop, sign_flip_p

GROUPS = {'disc_x': ('sweep_h',), 'disc_y': ('sweep_v',), 'grid06': ('grid06',), 'grid10': ('grid10',),
          'grid20': ('grid20',), 'clutter': ('clutter',), 'chart_ui': ('chart_ui',), 'flat': ('flat',),
          'refcoco': ('refcoco',)}
HELD_OUT = ('disc_x', 'disc_y', 'grid06', 'grid10', 'grid20', 'clutter', 'chart_ui', 'refcoco')


# ----------------------------------------------------------------------------- coordinate metrics

def coord_rows(run: Path, cond: str) -> dict:
    return {r['id']: r for r in read_jsonl(run / 'eval' / cond / 'coord.jsonl')}


def group_metrics(samples: list[tf.Sample], rows: dict, group: str) -> dict:
    fams = GROUPS[group]
    ss = [s for s in samples if s.family in fams]
    if not ss:
        return {}
    n = len(ss)
    valid = [rows[s.id]['valid'] for s in ss]
    hit = [rows[s.id]['hit'] for s in ss]
    err = np.array([rows[s.id]['error'] if rows[s.id]['valid'] else np.nan for s in ss])
    vals = np.concatenate([rows[s.id]['coords'] for s in ss if rows[s.id]['valid']]) if any(valid) else np.array([])
    out = {'n': n, 'valid': prop(sum(valid), n), 'hit': prop(sum(hit), n), 'error': mean_ci(err),
           'median_error': float(np.nanmedian(err)), 'round10': tf.round_share(vals), 'round50': tf.round_share(vals, 50)}
    ok = [s for s in ss if rows[s.id]['valid']]
    pred = np.array([rows[s.id]['center'] for s in ok])
    truth = np.array([rows[s.id]['truth_units'] for s in ok])
    if group.startswith('disc'):
        a = 0 if group == 'disc_x' else 1
        out['plateau'], out['plateau_n'] = tf.plateau_share(pred[:, a], truth[:, a])
        out['slope'] = tf.slope(pred[:, a], truth[:, a])
        out['slope_low'] = tf.slope(pred[:, a], truth[:, a], 0, 380)          # 3.12 compression range (x < 380)
        out['mean_signed_low'] = float(np.mean((pred[:, a] - truth[:, a])[truth[:, a] < 380]))
        out['curve'] = {'truth': truth[:, a].tolist(), 'pred': pred[:, a].tolist()}
    else:
        p2, t2 = pred.ravel(), truth.ravel()
        out['plateau'], out['plateau_n'] = tf.plateau_share(p2, t2)
        out['slope'] = tf.slope(p2, t2)
        out['slope_x'], out['slope_y'] = tf.slope(pred[:, 0], truth[:, 0]), tf.slope(pred[:, 1], truth[:, 1])
    if group.startswith('grid'):
        edges = [np.array(s.meta['col_edges_units']) for s in ok]
        out['edge_closer'], out['edge_median_cells'] = tf.edge_share(pred[:, 0], edges)
        cols = np.array([np.searchsorted(e, p, side='right') - 1 for e, p in zip(edges, pred[:, 0])])
        true_c = np.array([s.meta['col'] for s in ok])
        out['col_correct'] = prop(int((cols == true_c).sum()), len(ok))
        out['col_plus1'] = prop(int((cols == true_c + 1).sum()), len(ok))
    ious = [rows[s.id].get('iou', np.nan) for s in ss if s.fmt == 'bbox']
    if ious:
        ious = np.array([v if v is not None else 0.0 for v in ious], float)
        ious[np.isnan(ious)] = 0.0                     # invalid bbox answers count as IoU 0
        out['iou'] = mean_ci(ious)
        out['iou50'] = prop(int((ious >= 0.5).sum()), len(ious))
    return out


def paired(samples, base: dict, rows: dict, group: str) -> dict:
    ss = [s for s in samples if s.family in GROUPS[group]]
    hb = np.array([base[s.id]['hit'] for s in ss])
    hc = np.array([rows[s.id]['hit'] for s in ss])
    b, c = int((hb & ~hc).sum()), int((~hb & hc).sum())
    eb = np.array([base[s.id].get('error', np.nan) for s in ss], float)
    ec = np.array([rows[s.id].get('error', np.nan) for s in ss], float)
    d = ec - eb
    return {'hit_diff': float(hc.mean() - hb.mean()), 'mcnemar_p': mcnemar_p(b, c), 'base_only': b, 'cond_only': c,
            'error_diff': mean_ci(d), 'error_p': sign_flip_p(d)}


def coord(run: Path, conds: list[str]) -> dict:
    samples = tf.read_samples(run / 'test.jsonl')
    R = {}
    base = coord_rows(run, 'base')
    for cond in conds:
        rows = coord_rows(run, cond)
        missing = [s.id for s in samples if s.id not in rows]
        if missing:
            raise ValueError(f'{cond}: {len(missing)} test items have no coordinate answer')
        R[cond] = {g: group_metrics(samples, rows, g) for g in GROUPS}
        if cond != 'base':
            R[cond]['paired'] = {g: paired(samples, base, rows, g) for g in GROUPS}
        heldout = [s for s in samples if any(s.family in GROUPS[g] for g in HELD_OUT)]
        R[cond]['all_heldout_hit'] = prop(sum(rows[s.id]['hit'] for s in heldout), len(heldout))
        R[cond]['truncated'] = int(sum(r['finish_reason'] == 'length' for r in rows.values()))
    return R


# ----------------------------------------------------------------------------- general ability

def general(run: Path, conds: list[str]) -> dict:
    R = {}
    base = {r['id']: r for r in read_jsonl(run / 'eval' / 'base' / 'general.jsonl')}
    btext = read_jsonl(run / 'eval' / 'base' / 'text.jsonl')
    for cond in conds:
        rows = {r['id']: r for r in read_jsonl(run / 'eval' / cond / 'general.jsonl')}
        out = {}
        for bench in ('pope', 'textvqa', 'mmvp', 'gqa'):
            ids = sorted(i for i, r in rows.items() if r['bench'] == bench)
            acc = np.array([rows[i]['correct'] for i in ids], float)
            o = {'acc': mean_ci(acc), 'n': len(ids), 'truncated': int(sum(rows[i]['finish_reason'] == 'length' for i in ids)),
                 'unparsed': int(sum(not rows[i].get('parsed', True) for i in ids))}
            if cond != 'base':
                d = acc - np.array([base[i]['correct'] for i in ids], float)
                o['diff'] = mean_ci(d)
                o['p'] = sign_flip_p(d)
                o['same_text'] = float(np.mean([rows[i]['text'] == base[i]['text'] for i in ids]))
            out[bench] = o
        text = read_jsonl(run / 'eval' / cond / 'text.jsonl')
        out['text'] = {'acc': prop(sum(r['correct'] for r in text), len(text)),
                       'same_as_base': float(np.mean([a['text'] == b['text'] for a, b in zip(text, btext)]))}
        R[cond] = out
    return R


# ----------------------------------------------------------------------------- mechanism (3.12 measurements)

def blank_prior(U: dict) -> dict:
    out = {}
    for axis, name in ((0, 'blank_x'), (1, 'blank_y')):
        u = U[name]
        P1 = np.exp(u['logp1'])
        ks = u['ks']
        out[name] = {'P9_after4': float(P1[ks.index(4), 9]), 'P0_after5': float(P1[ks.index(5), 0]),
                     'first_digit': dfa.softmax_digits(u['logp0']).tolist(),
                     'max_other': float(max(P1[j].max() for j, k in enumerate(ks) if k not in (4, 5)))}
    return out


def digit_probs(U: dict) -> dict:
    """q0 first-digit and q1 second-digit (at the true first digit) probabilities of the true digit."""
    out = {}
    for fam in ('sweep_h', 'sweep_v', 'refcoco'):
        ids = [i for i in dfa.family(U, fam) if 100 <= U[i]['truth'] < 1000]
        p0, p1, a1 = [], [], []
        for i in ids:
            u = U[i]
            f, s = dfa.digits_of(u['truth'], 100)
            p0.append(dfa.softmax_digits(u['logp0'])[f])
            if f in u['ks']:
                P = np.exp(u['logp1'][u['ks'].index(f)])
                p1.append(P[s])
                a1.append(abs(int(P.argmax()) - s) <= 1)
        out[fam] = {'p_first_true': float(np.mean(p0)), 'p_second_true': float(np.mean(p1)),
                    'second_within1': float(np.mean(a1)), 'n': len(ids)}
    return out


def margin_vs_offset(U: dict) -> dict:
    """3.12 round-2 test: is E[second digit | true first digit] explained by the q0 neighbour margins
    (log p0[k+1] - log p0[k], log p0[k-1] - log p0[k]; truths 200-899 as in 3.12 round 2) or by the true offset (x - 100k) / 10?
    Returns the R^2 of the two linear regressions per family."""
    out = {}
    for fam in ('sweep_h', 'sweep_v', 'refcoco'):
        E, M, O = [], [], []
        for i in dfa.family(U, fam):
            u = U[i]
            if not 200 <= u['truth'] < 900:          # 3.12 round-2 range (both neighbours of k exist)
                continue
            f = int(u['truth'] // 100)
            P = np.exp(u['logp1'][u['ks'].index(f)])
            l0 = np.log(dfa.softmax_digits(u['logp0']))
            up, dn = l0[f + 1] - l0[f], l0[f - 1] - l0[f]
            E.append(float((P * np.arange(10)).sum()))
            M.append([max(up, -20.0), max(dn, -20.0)])
            O.append((u['truth'] - 100 * f) / 10)
        E, M, O = np.array(E), np.array(M), np.array(O)[:, None]

        def r2(X):
            X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
            coef, *_ = np.linalg.lstsq(X1, E, rcond=None)
            return float(1 - ((E - X1 @ coef) ** 2).sum() / ((E - E.mean()) ** 2).sum())
        out[fam] = {'r2_margins': r2(M), 'r2_offset': r2(O), 'n': int(len(E))}
    return out


def anova(U: dict) -> dict:
    out = {}
    for fam in ('sweep_h', 'sweep_v', 'refcoco'):
        ids = dfa.family(U, fam)
        out[fam] = dfa.anova2(np.stack([U[i]['logp1'] for i in ids]))
    return out


def mechanism(run: Path, conds: list[str]) -> tuple[dict, dict]:
    R, A = {}, {}
    for cond in conds:
        fr = run / 'eval' / cond / 'fact'
        U = dfa.load(fr)
        shards = dfa.load_shard_arrays(fr / 'features', ('resid', 'dla_mixer', 'dla_mlp', 'dla_head', 'dla_embed'))
        clamp, _ = dfa.clamp(U)
        dla, dA = dfa.dla(U, shards)
        R[cond] = {'clamp': {f: clamp[f] for f in clamp}, 'anova': anova(U), 'blank': blank_prior(U),
                   'digits': digit_probs(U), 'margin_vs_offset': margin_vs_offset(U), 'oracle': dfa.oracle(U, shards),
                   'prior_writers': {f: dla[f]['prior_writers'] for f in dla}}
        for k, v in dA.items():
            A[f'{cond}/{k}'] = v
        # Residuals at the true first digit for the change analysis.
        for fam in ('sweep_h', 'sweep_v'):
            ids = [i for i in dfa.family(U, fam) if 100 <= U[i]['truth'] < 1000 and int(U[i]['truth'] // 100) in U[i]['ks']]
            rows = [U[i]['rows'][U[i]['ks'].index(int(U[i]['truth'] // 100))] for i in ids]
            A[f'{cond}/{fam}_resid'] = np.stack([shards[s]['resid'][j] for s, j in rows]).astype(np.float32)  # (N, 2, 7, D)
            A[f'{cond}/{fam}_truth'] = np.array([U[i]['truth'] for i in ids])
            A[f'{cond}/{fam}_ids'] = np.array(ids)
        del shards
    return R, A


def change(A: dict, conds: list[str]) -> dict:
    """Per-component DLA change and residual change relative to the base model."""
    out = {}
    names = list(A['base/dla_names'])
    for cond in conds:
        if cond == 'base':
            continue
        o = {}
        for fam in ('sweep_h', 'sweep_v'):
            for role in ('q0', 'q1'):
                d = A[f'{cond}/{fam}_{role}_tuned'] - A[f'base/{fam}_{role}_tuned']
                top = np.argsort(-np.abs(d))[:8]
                o[f'{fam}_{role}_tuned_delta_top'] = {names[t]: round(float(d[t]), 3) for t in top}
                o[f'{fam}_{role}_tuned_total'] = [float(A[f'base/{fam}_{role}_tuned'].sum()), float(A[f'{cond}/{fam}_{role}_tuned'].sum())]
            for k, dg in ((4, 9), (5, 0)):
                d = A[f'{cond}/{fam}_prior_k{k}_d{dg}'] - A[f'base/{fam}_prior_k{k}_d{dg}']
                top = np.argsort(-np.abs(d))[:8]
                o[f'{fam}_prior_k{k}_d{dg}_delta_top'] = {names[t]: round(float(d[t]), 3) for t in top}
                o[f'{fam}_prior_k{k}_d{dg}_total'] = [float(A[f'base/{fam}_prior_k{k}_d{dg}'].sum()),
                                                     float(A[f'{cond}/{fam}_prior_k{k}_d{dg}'].sum())]
            # Residual change at q0 / q1 per stored layer (same images and forced digits in both models).
            if not np.array_equal(A[f'{cond}/{fam}_ids'], A[f'base/{fam}_ids']):
                raise ValueError('factorial units differ between conditions')
            Rb, Rc = A[f'base/{fam}_resid'], A[f'{cond}/{fam}_resid']
            rel = np.linalg.norm(Rc - Rb, axis=-1) / np.linalg.norm(Rb, axis=-1)          # (N, 2, 7)
            o[f'{fam}_resid_rel_change'] = {role: dict(zip([f'L{l}' for l in df.RESID_LAYERS], rel[:, r].mean(0).round(4).tolist()))
                                            for r, role in enumerate(('q0', 'q1'))}
            # Probe transfer: ridge fitted on base q1 residuals (5-fold) applied to the condition's residuals.
            x = A[f'base/{fam}_truth']
            tr = {}
            order = np.random.default_rng(0).permutation(len(x))
            for li, l in enumerate(df.RESID_LAYERS):
                pb, pc, pcc = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
                for f in range(5):
                    te = np.zeros(len(x), bool)
                    te[order[f::5]] = True
                    m = dfa.ridge_fit(Rb[~te, 1, li], x[~te, None])
                    pb[te] = dfa.ridge_predict(m, Rb[te, 1, li])[:, 0]
                    pc[te] = dfa.ridge_predict(m, Rc[te, 1, li])[:, 0]
                    m2 = dfa.ridge_fit(Rc[~te, 1, li], x[~te, None])
                    pcc[te] = dfa.ridge_predict(m2, Rc[te, 1, li])[:, 0]
                tr[f'L{l}'] = {'base_on_base': float(np.abs(pb - x).mean()), 'base_on_cond': float(np.abs(pc - x).mean()),
                               'cond_on_cond': float(np.abs(pcc - x).mean())}
            o[f'{fam}_probe_transfer_q1'] = tr
        out[cond] = o
    return out


def training(run: Path, conds: list[str]) -> dict:
    out = {}
    for cond in conds:
        if cond == 'base':
            continue
        log = read_jsonl(run / 'adapters' / f'{cond}_log.jsonl')
        st = torch.load(run / 'adapters' / f'{cond}.pt', map_location='cpu', weights_only=False)
        norms = st['delta_norms']
        per_layer = {}
        for n, v in norms.items():
            l = int(n.split('.')[0][1:])
            per_layer[l] = per_layer.get(l, 0.0) + v ** 2
        out[cond] = {'n_params': st['n_params'], 'n_steps': st['n_steps'], 'train_seconds': st['train_seconds'],
                     'loss': [r['loss'] for r in log], 'digit_ce': [r['digit_ce'] for r in log],
                     'first10_digit_ce': float(np.mean([r['digit_ce'] for r in log[:10]])),
                     'last10_digit_ce': float(np.mean([r['digit_ce'] for r in log[-10:]])),
                     'delta_norm_per_layer': {l: float(np.sqrt(v)) for l, v in sorted(per_layer.items())}}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    p.add_argument('--conditions', nargs='+', required=True)
    p.add_argument('--figures', required=True)
    p.add_argument('--skip', nargs='*', default=[], help='sections to skip (e.g. general mechanism)')
    args = p.parse_args()
    run = Path(args.run)
    conds = [c for c in args.conditions if (run / 'eval' / c / 'coord.jsonl').exists()]
    print('conditions with results:', conds, flush=True)
    res = {'conditions': conds}
    res['training'] = training(run, conds)
    res['coord'] = coord(run, conds)
    if 'general' not in args.skip:
        res['general'] = general(run, [c for c in conds if (run / 'eval' / c / 'general.jsonl').exists()])
    arrays = {}
    if 'mechanism' not in args.skip:
        mc = [c for c in conds if (run / 'eval' / c / 'fact' / 'factorial' / 'rows.jsonl').exists()]
        res['mechanism'], arrays = mechanism(run, mc)
        res['change'] = change(arrays, mc)
    (run / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    np.savez(run / 'analysis.npz', **{k: v for k, v in arrays.items() if not k.endswith('_resid')})
    print('analysis written', flush=True)
    from analysis_tools.position import targeted_finetune_figures as F
    F.main_figures(run, Path(args.figures))


if __name__ == '__main__':
    main()
