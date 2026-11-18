"""Experiment 7.3 round 2: tables and figures (CPU).

Reads this round's run directory (--run) and the 7.3 round-1 run directory (--src-run); writes summary.json to the run
directory and figures to --figures.  Results doc: docs/analysis/application/results_2026-11-11_vit_detector_formation.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.application import detector_formation as dfm
from analysis_tools.application import interp_opt as io_
from analysis_tools.application import located_finetune as lf

TRAIN = lf.TRAIN_PAIRS
HFEAT = lf.HELDOUT_FEATURES
HFAM = lf.HELDOUT_FAMILY
WIN = [f'{w[0]}-{w[-1]}' for w in dfm.WINDOWS]
# 3-block MLP LoRA windows: round-1 (seed 1, and seed 2 for 13-15 / 25-27) and round-2 adapter names
R1_WINDOW = {'1-3': ['early_n1200_s1'], '4-6': ['w04_n1200_s1'], '7-9': ['w07_n1200_s1'], '10-12': ['w10_n1200_s1'],
             '13-15': ['loc_n1200_s1', 'loc_n1200_s2'], '16-18': ['w16_n1200_s1'], '19-21': ['w19_n1200_s1'],
             '22-24': ['w22_n1200_s1'], '25-27': ['late_n1200_s1', 'late_n1200_s2']}
EARLIER_BINS = ((0, 5), (6, 11), (12, 23), (24, 35))


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def probe_rows(run: Path, src: Path, name: str) -> list[dict] | None:
    for p in (run / 'eval' / name / 'probe.jsonl', src / 'eval' / name / 'probe.jsonl'):
        if p.exists():
            return read_jsonl(p)
    return None


def counts(rows: list[dict]) -> dict:
    sel = lambda f: [r for r in rows if r['variant'] == 'swap' and f(r['pair'])]
    out = {'train': sel(lambda p: p in TRAIN), 'hfeat': sel(lambda p: p in HFEAT), 'hfam': sel(lambda p: p in HFAM),
           'present': [r for r in rows if r['variant'] == 'std']}
    out['train_hfeat'] = out['train'] + out['hfeat']
    return {k: int(sum(bool(r['correct']) for r in v)) for k, v in out.items()} | \
        {f'n_{k}': len(v) for k, v in out.items()}


def window_runs(run: Path, src: Path) -> dict:
    res = {}
    for i, w in enumerate(WIN):
        names = list(R1_WINDOW[w])
        start = dfm.WINDOWS[i][0]
        for s in (2, 3):
            n = f'lora_w{start:02d}_s{s}'
            if (run / 'eval' / n / 'probe.jsonl').exists():
                names.append(n)
        res[w] = {n: counts(probe_rows(run, src, n)) for n in names if probe_rows(run, src, n) is not None}
    return res


def perm_p_spearman(x, y, n_perm: int = 20000, seed: int = 0) -> float:
    """One-sided permutation p-value of the Spearman correlation (H1: positive)."""
    rng = np.random.default_rng(seed)
    obs = dfm.spearman(x, y)
    y = np.asarray(y)
    hits = sum(dfm.spearman(x, rng.permutation(y)) >= obs - 1e-12 for _ in range(n_perm))
    return float((hits + 1) / (n_perm + 1))


def criteria(run: Path, src: Path) -> dict:
    pre1 = json.loads((run / 'relocate' / 'prereg.json').read_text())
    pre2 = json.loads((run / 'relocate' / 'prereg_v2.json').read_text())
    old = json.loads((src / 'locate' / 'summary.json').read_text())
    tdh = old['transplant_dhit']
    lin = old['absent_achievable']
    blk = lambda v: {b: float(v[b - 1]) for b in range(1, 28)}
    return {'C1 write-then-read (prereg v1)': {w: pre1['window_scores'][w] for w in WIN},
            'C2 shared oddness (prereg v2)': {w: pre2['window_C2'][w] for w in WIN},
            'C2_HF shared oddness, held-out family': {w: pre2['window_C2_HF'][w] for w in WIN},
            'round-1 transplant gain': {f'{k[0]}-{k[-1]}': v for k, v in dfm.window_scores(blk(tdh), dfm.WINDOWS).items()},
            'round-1 within-pair linear achievability': {f'{k[0]}-{k[-1]}': v for k, v in
                                                         dfm.window_scores(blk(lin), dfm.WINDOWS).items()}}


def analyse(run: Path, src: Path) -> dict:
    res: dict = {}
    wr = window_runs(run, src)
    res['windows'] = wr
    mean = {w: {k: float(np.mean([c[k] / c[f'n_{k}'] for c in wr[w].values()])) for k in ('train_hfeat', 'hfam', 'train', 'hfeat')}
            for w in WIN}
    spread = {w: {k: float(np.ptp([c[k] / c[f'n_{k}'] for c in wr[w].values()])) for k in ('train_hfeat', 'hfam')} for w in WIN}
    res['window_mean'], res['window_spread'] = mean, spread
    res['window_n_seeds'] = {w: len(wr[w]) for w in WIN}
    crit = criteria(run, src)
    res['criteria'] = crit
    res['spearman'] = {}
    for cname, cv in crit.items():
        x = [cv[w] for w in WIN]
        for target in ('train_hfeat', 'hfam'):
            y = [mean[w][target] for w in WIN]
            y7 = [mean[w][target] for w in WIN[2:]]
            res['spearman'][f'{cname} | {target}'] = {
                'rho': dfm.spearman(x, y), 'p_perm': perm_p_spearman(x, y),
                'rho_7_27': dfm.spearman(x[2:], y7)}
    # within the 7-27 plateau: are window differences larger than seed differences?  Permutation test of the variance
    # of window means (labels of the 21 runs permuted), and the seed spread of every window.
    rng = np.random.default_rng(0)
    res['plateau_test'] = {}
    for target in ('train_hfeat', 'hfam'):
        groups = [[c[target] / c[f'n_{target}'] for c in wr[w].values()] for w in WIN[2:]]
        vals = np.concatenate(groups)
        sizes = [len(g) for g in groups]
        stat = lambda v: float(np.var([np.mean(x) for x in np.split(v, np.cumsum(sizes)[:-1])]))
        obs = stat(vals)
        hits = sum(stat(rng.permutation(vals)) >= obs for _ in range(20000))
        res['plateau_test'][target] = {'var_of_window_means': obs, 'p_perm': (hits + 1) / 20001}
    res['full_scan'] = {n: counts(probe_rows(run, src, n)) for n in FULL_SCAN if probe_rows(run, src, n) is not None}
    res['formation'] = formation_summary(run, src, FORMATION)
    res['oddness_windows'] = {}
    for w in WIN:
        for n in list(wr[w]):
            nm = n if n.startswith('lora_') else 'r1_' + n
            od = oddness_summary(run, nm)
            if od is not None:
                res['oddness_windows'][nm] = {'window': w, 'merger_swap_train': od['swap|train|merger'],
                                              'merger_swap_hfeat': od['swap|hfeat|merger'],
                                              'merger_swap_hfam': od['swap|hfam|merger'], 'acc': wr[w][n]}
    res['base_oddness'] = oddness_summary(run, 'base')
    res['readout'] = {}
    for n in FORMATION + ['width_w13_m8_s1']:
        rows = _jl(run / 'eval' / n / 'readout.json')
        if rows is None:
            continue
        res['readout'][n] = {g: {k: float(np.mean([r[k] for r in rows if r['pair'] in P])) for k in
                                 ('ds_actual', 'fo_target', 'fo_other', 'cos_contrast_g', 'norm_D_target', 'norm_D_other', 'norm_m')}
                             for g, P in (('train', TRAIN), ('hfeat', HFEAT), ('hfam', HFAM))}
    res['general'] = {}
    for n in ('base', 'width_w13_m38_s1', 'full_b14_lr1e4_s1', 'merger_s1'):
        d = (src if n == 'base' else run) / 'eval' / n
        if not (d / 'general.jsonl').exists():
            continue
        g = read_jsonl(d / 'general.jsonl')
        out = {b: float(np.mean([r['correct'] for r in g if r['bench'] == b])) for b in ('pope', 'textvqa', 'mmvp', 'gqa')}
        co = read_jsonl(d / 'coord.jsonl')
        out['coord_synthetic_hit'] = int(sum(bool(r['hit']) for r in co if r['family'] != 'refcoco'))
        out['coord_synthetic_n'] = sum(r['family'] != 'refcoco' for r in co)
        out['refcoco_hit'] = int(sum(bool(r['hit']) for r in co if r['family'] == 'refcoco'))
        out['coord_invalid'] = int(sum(not r.get('valid', True) for r in co))
        out['text_correct'] = int(sum(bool(r['correct']) for r in read_jsonl(d / 'text.jsonl')))
        out['count_correct'] = int(sum(bool(r['correct']) for r in read_jsonl(d / 'count.jsonl')))
        if n != 'base':
            gb = {r['id']: r for r in read_jsonl(src / 'eval' / 'base' / 'general.jsonl')}
            for b in ('pope', 'textvqa', 'mmvp', 'gqa'):
                diffs = [r['correct'] - gb[r['id']]['correct'] for r in g if r['bench'] == b and r['id'] in gb]
                bs = [np.mean(rng.choice(diffs, len(diffs))) for _ in range(2000)]
                out[f'{b}_delta'] = [float(np.mean(diffs)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            cb = {r['id']: bool(r['hit']) for r in read_jsonl(src / 'eval' / 'base' / 'coord.jsonl')}
            ca = {r['id']: bool(r['hit']) for r in co if r['family'] != 'refcoco'}
            b01 = sum((not cb[i]) and ca[i] for i in ca if i in cb)
            b10 = sum(cb[i] and not ca[i] for i in ca if i in cb)
            out['coord_mcnemar'] = [b01, b10, io_.mcnemar_exact(b01, b10)]
        res['general'][n] = out
    res['badcase'] = badcase_summary(run, src, ['base', 'r1_loc_n1200_s1', 'r1_late_n1200_s1', 'r1_vitall_n1200_s1',
                                                'width_w13_m38_s1', 'full_b14_lr1e4_s1', 'lora_r64_w13_s1', 'width_w13_m8_s1', 'merger_s1'])
    return res


FORMATION = ['r1_loc_n1200_s1', 'lora_r64_w13_s1', 'width_w13_m8_s1', 'width_w13_m38_s1', 'width_w13_m38_s2',
             'full_b14_s1', 'full_b14_lr1e4_s1', 'merger_s1', 'r1_late_n1200_s1', 'r1_vitall_n1200_s1']
FULL_SCAN = [f'full_b{b:02d}_lr1e4_s1' for b in (5, 14, 26)]


def _jl(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def oddness_summary(run: Path, name: str) -> dict | None:
    rows = _jl(run / 'eval' / name / 'oddness.json')
    if rows is None:
        return None
    key = 'zeta' if name == 'base' else 'zeta_basedir'
    out = {}
    for k in ['merger'] + list(range(1, 28)):
        for grp, pairs in (('train', TRAIN), ('hfeat', HFEAT), ('hfam', HFAM)):
            for v in ('swap', 'std'):
                z = [r[key] for r in rows if r['block'] == k and r['variant'] == v and r['pair'] in pairs]
                out[f'{v}|{grp}|{k}'] = float(np.median(z))
    return out


def formation_summary(run: Path, src: Path, names: list[str]) -> dict:
    base_det = _jl(run / 'eval' / 'base' / 'detect.json')
    res = {}
    for n in names:
        pr = probe_rows(run, src, n[3:] if n.startswith('r1_') else n)
        if pr is None:
            continue
        r: dict = {'probe': counts(pr)}
        meta = None
        for ap in (run / 'adapters' / f'{n}.pt',):
            if ap.exists():
                import torch
                st = torch.load(ap, map_location='cpu', weights_only=False)
                meta = {k: st.get(k) for k in ('method', 'scope', 'n_params', 'lr', 'seed', 'm', 'rank')}
        r['meta'] = meta
        od = oddness_summary(run, n)
        if od is not None:
            r['oddness'] = {k: od[k] for k in od if k.split('|')[2] in ('merger', '27', '15', '13')}
        det = _jl(run / 'eval' / n / 'detect.json')
        if det is not None:
            cz = det['context_zeta']
            blocks = sorted({int(k.split('|')[3]) for k in cz})
            for kind in ('old', 'new'):
                for grp, pairs in (('train', TRAIN), ('hfeat', HFEAT), ('hfam', HFAM)):
                    vals = [cz[f'{p}|swap|{kind}|{b}'] for p in pairs for b in blocks if f'{p}|swap|{kind}|{b}' in cz]
                    if vals:
                        r[f'ctx_{kind}_{grp}'] = {'n_gt5_median_per_block': float(np.median([v['n_gt5'] for v in vals])),
                                                  'n_gt5_total': int(sum(v['n_gt5'] for v in vals)),
                                                  'max': float(max(v['max'] for v in vals))}
                        if kind == 'old' and base_det is not None:
                            bv = [base_det['context_zeta'].get(f'{p}|swap|old|{b}') for p in pairs for b in blocks]
                            bv = [x for x in bv if x is not None]
                            if bv:
                                r[f'ctx_base_{grp}'] = {'n_gt5_total': int(sum(v['n_gt5'] for v in bv)),
                                                        'max': float(max(v['max'] for v in bv))}
            if 'new_neuron_zeta' in det:
                nz = det['new_neuron_zeta']
                for grp, pairs in (('train', TRAIN), ('hfeat', HFEAT), ('hfam', HFAM)):
                    zz = np.concatenate([np.asarray(nz[p][m]['B|A']) for p in pairs for m in nz[p]])
                    r[f'new_hom_BA_{grp}'] = {'max': float(zz.max()), 'n_gt5': int((zz > 5).sum()), 'n': int(zz.size)}
            if 'split' in det:
                r['split'] = {b: {k: v[k] for k in ('zeta', 'n_eff', 'top10_share', 'top100_share', 'part_totals')}
                              for b, v in det['split'].items()}
        res[n] = r
    return res


def mcnemar_vs(run: Path, src: Path, a: str, b: str, sel) -> dict:
    A = {r['id']: bool(r['correct']) for r in probe_rows(run, src, a) if sel(r)}
    B = {r['id']: bool(r['correct']) for r in probe_rows(run, src, b) if sel(r)}
    ids = sorted(set(A) & set(B))
    b01 = sum((not A[i]) and B[i] for i in ids)
    b10 = sum(A[i] and not B[i] for i in ids)
    return {'n': len(ids), 'a_wrong_b_right': b01, 'a_right_b_wrong': b10, 'p': io_.mcnemar_exact(b01, b10)}


def badcase_summary(run: Path, src: Path, names: list[str]) -> dict:
    res = {}
    for n in names:
        pr = probe_rows(run, src, n[3:] if n.startswith('r1_') else n)
        r: dict = {}
        if pr is not None:
            for lo, hi in EARLIER_BINS:
                rows = [x for x in pr if x['variant'] == 'swap' and x['pair'] in TRAIN + HFEAT
                        and lo <= x['target'][0] * 6 + x['target'][1] <= hi]
                k = sum(bool(x['correct']) for x in rows)
                r[f'earlier_{lo}_{hi}'] = {'k': k, 'n': len(rows), 'ci': io_.wilson(k, len(rows))[1:]}
        bc = run / 'eval' / n / 'badcase.jsonl'
        if bc.exists():
            rows = read_jsonl(bc)
            for pair in sorted({x['pair'] for x in rows}):
                for lv in sorted({x['level'] for x in rows if x['pair'] == pair}):
                    sel = [x for x in rows if x['pair'] == pair and x['level'] == lv]
                    k = sum(bool(x['correct']) for x in sel)
                    r[f'dose|{pair}|{lv}'] = {'k': k, 'n': len(sel)}
                    early = [x for x in sel if x['target'][0] * 6 + x['target'][1] <= 5]
                    r[f'dose_early|{pair}|{lv}'] = {'k': sum(bool(x['correct']) for x in early), 'n': len(early)}
        res[n] = r
    return res


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def fig_relocate(run: Path, src: Path, res: dict, path: Path) -> None:
    """Base-model block profiles: separability (static absence vs contextual oddness), neurons, readout, simulation."""
    plt = _plt()
    sep = json.loads((run / 'relocate' / 'separability.json').read_text())
    wr = json.loads((run / 'relocate' / 'write_read.json').read_text())['rows']
    od = res['base_oddness']
    old = json.loads((src / 'locate' / 'summary.json').read_text())
    B = np.arange(1, 28)
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    a = ax[0]
    stat = [np.median([r['zeta_swap'] for r in sep if r['block'] == b and r['fold'] == 'lopo']) for b in B]
    a.plot(B, [od[f'std|train|{b}'] for b in B], color='#1f77b4', marker='o', ms=3, label='part present, shared oddness (LOPO)')
    a.plot(B, [od[f'swap|train|{b}'] for b in B], color='#d62728', marker='o', ms=3, label='part absent, shared oddness (LOPO)')
    a.plot(B, [od[f'swap|hfam|{b}'] for b in B], color='#9467bd', marker='o', ms=3, label='part absent, held-out family')
    a.plot(B, stat, color='#7f7f7f', marker='s', ms=3, label='part absent, shared static B-vs-A direction (LOPO)')
    a.axhline(5, color='k', ls=':', lw=0.8)
    a.set_yscale('symlog', linthresh=1)
    a.set_xlabel('ViT block (MLP input)')
    a.set_ylabel('target deviation from distractors (sd)')
    a.set_title('(a) linearly readable target signal shared across features')
    a.legend(fontsize=7.5, loc='lower right')
    a = ax[1]
    a.plot(B, old['present_n_detectors'], color='#1f77b4', marker='o', ms=3, label='part present')
    a.plot(B, old['absent_n_detectors'], color='#d62728', marker='o', ms=3, label='part absent')
    a.set_yscale('symlog', linthresh=1)
    a.set_xlabel('ViT block')
    a.set_ylabel('exclusive detector neurons (median over pairs)')
    a.set_title('(b) single neurons with zeta > 5 (round 1)')
    a.legend(fontsize=8)
    a = ax[2]
    a.plot(B, [np.mean([r['grad_norm'] for r in sep if r['block'] == b and r['fold'] == 'lopo']) for b in B],
           color='#2ca02c', marker='o', ms=3)
    a.set_xlabel('ViT block (MLP output at the target)')
    a.set_ylabel('norm of mean d log p / d o_k')
    a.set_title('(c) answer sensitivity to a write at block k')
    a = ax[3]
    for sc, col in ((1.0, '#ff7f0e'), (3.0, '#8c564b')):
        a.plot(B, [np.mean([r['ds'] for r in wr if r['block'] == b and r['fold'] == 'lopo' and r['scale'] == sc]) for b in B],
               color=col, marker='o', ms=3, label=f'write-then-read unit, norm {sc:g} c')
    a.plot(B, old['transplant_dscore'], color='#7f7f7f', marker='s', ms=3, label='round-1 transplant')
    a.axhline(0, color='k', lw=0.6)
    a.set_xlabel('ViT block')
    a.set_ylabel('gain of log p(correct row, column) (nats)')
    a.set_title('(d) simulated new unit vs round-1 transplant')
    a.legend(fontsize=7.5)
    for a in ax:
        a.axvspan(0.5, 6.5, color='0.92', zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_windows(res: dict, path: Path) -> None:
    plt = _plt()
    wr, crit = res['windows'], res['criteria']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    panels = [('C2 shared oddness (prereg v2)', 'train_hfeat', 'training + held-out features (n=504)'),
              ('C2_HF shared oddness, held-out family', 'hfam', 'held-out family (n=288)'),
              ('round-1 transplant gain', 'train_hfeat', 'training + held-out features (n=504)')]
    cmap = plt.get_cmap('viridis')
    for a, (cname, target, lab) in zip(ax, panels):
        for i, w in enumerate(WIN):
            x = crit[cname][w]
            ys = [c[target] / c[f'n_{target}'] for c in wr[w].values()]
            col = cmap(i / 8)
            a.scatter([x] * len(ys), ys, color=col, s=14, alpha=0.6)
            a.errorbar(x, np.mean(ys), yerr=[[np.mean(ys) - min(ys)], [max(ys) - np.mean(ys)]], color=col, marker='o',
                       ms=7, capsize=3, label=f'blocks {w}')
        sp = res['spearman'][f'{cname} | {target}']
        a.set_title(f"{cname}\nSpearman {sp['rho']:.2f} (perm. p {sp['p_perm']:.3f})", fontsize=9.5)
        a.set_xlabel('criterion value of the window (base model, before training)')
        a.set_ylabel(f'absent-target accuracy, {lab}')
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_distributed(run: Path, res: dict, path: Path) -> None:
    """(a) schematic; (b) cumulative share of the target-specific write vs number of neurons; (c) response along v."""
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    a = ax[0]
    rng = np.random.default_rng(3)
    n = 40
    det = np.zeros(n)
    det[17] = 1.0
    dist = np.abs(rng.normal(0.16, 0.08, n))
    a.bar(np.arange(n) - 0.2, det, width=0.4, color='#1f77b4', label='detector: one unit, large selective response')
    a.bar(np.arange(n) + 0.2, dist, width=0.4, color='#d62728', label='distributed linear signal: many small changes')
    a.set_xlabel('MLP neurons (illustration)')
    a.set_ylabel('contribution to target minus distractor (a.u.)')
    a.set_title('(a) two ways to make the target stand out (schematic)')
    a.legend(fontsize=7.5)
    a = ax[1]
    curves = [('base', '13', 'base model, part present, block 13', '#1f77b4', '-'),
              ('base', '19', 'base model, part present, block 19', '#1f77b4', '--'),
              ('r1_loc_n1200_s1', '14', 'LoRA r8 blocks 13-15 (block 14)', '#d62728', '-'),
              ('lora_r64_w13_s1', '14', 'LoRA r64 blocks 13-15 (block 14)', '#ff7f0e', '-'),
              ('full_b14_lr1e4_s1', '14', 'full fine-tuning block 14', '#2ca02c', '-'),
              ('width_w13_m38_s1', '15', 'new neurons 38/block (block 15)', '#9467bd', '-')]
    for name, b, lab, col, ls in curves:
        det = _jl(run / 'eval' / name / 'detect.json')
        if det is None or 'split' not in det or b not in det['split']:
            continue
        c = np.abs(np.concatenate([np.asarray(v) for v in det['split'][b]['contrib'].values()]))
        c = np.sort(c)[::-1]
        cum = np.cumsum(c) / c.sum()
        a.plot(np.arange(1, len(c) + 1), cum, color=col, ls=ls,
               label=f"{lab}: N_eff {det['split'][b]['n_eff']:.0f}")
    a.set_xscale('log')
    a.axhline(1, color='k', lw=0.5)
    a.set_xlabel('number of neurons (sorted by contribution)')
    a.set_ylabel('cumulative share of sum |c_j|')
    a.set_title('(b) how many neurons carry the write')
    a.legend(fontsize=7)
    a = ax[2]
    for name, b, lab, col in (('base', '13', 'base, part present, block 13', '#1f77b4'),
                              ('r1_loc_n1200_s1', '14', 'LoRA r8, part absent, block 14', '#d62728'),
                              ('full_b14_lr1e4_s1', '14', 'full FT, part absent, block 14', '#2ca02c'),
                              ('width_w13_m38_s1', '15', 'new neurons, part absent, block 15', '#9467bd')):
        det = _jl(run / 'eval' / name / 'detect.json')
        if det is None or 'split' not in det or b not in det['split']:
            continue
        sp = det['split'][b]
        d = np.asarray(sp['resp_d'])
        mu, sd = d.mean(), d.std()
        a.hist((d - mu) / sd, bins=60, range=(-5, 10), density=True, histtype='step', color=col, lw=1.2,
               label=f'{lab}: distractors')
        t = (np.asarray(sp['resp_t']) - mu) / sd
        a.hist(t, bins=60, range=(-5, 10), density=True, histtype='stepfilled', alpha=0.3, color=col,
               label=f'{lab}: target')
    a.set_xlabel('projection of the MLP write on v (distractor sd units)')
    a.set_ylabel('density')
    a.set_title('(c) response of the written direction')
    a.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_newneurons(run: Path, path: Path) -> None:
    """Per new neuron: homogeneous-image zeta^{B|A} vs in-image zeta_swap (absent target), per pair."""
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, name in zip(ax, ('width_w13_m38_s1', 'width_w13_m8_s1')):
        det = _jl(run / 'eval' / name / 'detect.json')
        if det is None:
            continue
        cmap = {p: c for p, c in zip(lf.PROBE_PAIRS, plt.get_cmap('tab20').colors)}
        for p in lf.PROBE_PAIRS:
            for m in det['new_neuron_zeta'][p]:
                b = m.split('.')[0][1:]
                hz = np.asarray(det['new_neuron_zeta'][p][m]['B|A'])
                cz = det['context_zeta'].get(f'{p}|swap|new|{b}')
                if cz is None or cz['z'] is None:
                    continue
                a.scatter(hz, np.asarray(cz['z'], dtype=float), s=8, color=cmap[p], alpha=0.7,
                          label=p if m.startswith('V13') else None)
        a.axhline(5, color='k', ls=':', lw=0.8)
        a.axvline(5, color='k', ls=':', lw=0.8)
        a.set_xscale('symlog', linthresh=1)
        a.set_yscale('symlog', linthresh=1)
        a.set_xlabel('zeta on homogeneous images (one B among As)')
        a.set_ylabel('zeta inside swap images (target vs neighbours)')
        a.set_title(f'{name}: each point = one new neuron x one pair', fontsize=9.5)
    ax[0].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_badcase(res: dict, path: Path) -> None:
    plt = _plt()
    bc = res['badcase']
    names = [('base', 'base model', 'k'), ('r1_loc_n1200_s1', 'LoRA r8 blocks 13-15', '#d62728'),
             ('r1_late_n1200_s1', 'LoRA r8 blocks 25-27', '#ff7f0e'), ('width_w13_m38_s1', 'new neurons blocks 13-15', '#9467bd'),
             ('full_b14_lr1e4_s1', 'full fine-tuning block 14', '#2ca02c'), ('r1_vitall_n1200_s1', 'full-ViT LoRA', '#1f77b4')]
    pairs = sorted({k.split('|')[1] for k in bc.get('base', {}) if k.startswith('dose|')})
    fig, ax = plt.subplots(1, 1 + len(pairs), figsize=(4.2 * (1 + len(pairs)), 4.2))
    a = ax[0]
    xs = np.arange(len(EARLIER_BINS))
    for j, (n, lab, col) in enumerate(names):
        if n not in bc or 'earlier_0_5' not in bc[n]:
            continue
        ys = [bc[n][f'earlier_{lo}_{hi}']['k'] / bc[n][f'earlier_{lo}_{hi}']['n'] for lo, hi in EARLIER_BINS]
        lo_ = [y - bc[n][f'earlier_{l}_{h}']['ci'][0] for y, (l, h) in zip(ys, EARLIER_BINS)]
        hi_ = [bc[n][f'earlier_{l}_{h}']['ci'][1] - y for y, (l, h) in zip(ys, EARLIER_BINS)]
        a.errorbar(xs + 0.06 * (j - 2.5), ys, yerr=[lo_, hi_], color=col, marker='o', capsize=2, label=lab)
    a.set_xticks(xs, ['0-5', '6-11', '12-23', '24-35'])
    a.set_xlabel('entities before the target in raster order')
    a.set_ylabel('absent-target accuracy (training + held-out features)')
    a.set_title('(a) raster position')
    a.legend(fontsize=7)
    for a, pair in zip(ax[1:], pairs):
        for n, lab, col in names:
            if n not in bc:
                continue
            lv = sorted(float(k.split('|')[2]) for k in bc[n] if k.startswith(f'dose|{pair}|'))
            if not lv:
                continue
            ys = [bc[n][f'dose|{pair}|{v:g}']['k'] / bc[n][f'dose|{pair}|{v:g}']['n'] for v in lv]
            a.plot(lv, ys, color=col, marker='o', label=lab)
        a.set_xlabel(f'{pair}: part size ({"deg" if pair == "plusx" else "px"})')
        a.set_ylabel('absent-target accuracy (n=36 per point)')
        a.set_title(f'(dose) {pair}')
        a.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_vl(run: Path, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    models = [('Qwen3.5-9B', '#d62728'), ('Qwen3-VL-8B-Instruct', '#1f77b4')]
    taps_all = {}
    for m, col in models:
        det = _jl(run / 'vl' / m / 'detectors.json')
        taps = _jl(run / 'vl' / m / 'taps.json')
        if det is None:
            continue
        B = sorted(int(b) for b in det[TRAIN[0]])
        for key, a, lab in (('n_detectors', ax[0], 'exclusive detector neurons'), ('achievable', ax[1], 'linear zeta')):
            for d_, ls in (('A|B', '-'), ('B|A', '--')):
                pairs = TRAIN + HFEAT
                y = [np.median([det[p][str(b)][d_][key] for p in pairs]) for b in B]
                a.plot(B, y, color=col, ls=ls, marker='o', ms=2.5,
                       label=f"{m}, part {'present' if d_ == 'A|B' else 'absent'}")
            a.set_yscale('symlog', linthresh=1)
            a.set_xlabel('ViT block')
            a.set_ylabel(lab + ' (median)')
        if taps is not None:
            taps_all[m] = taps
    ax[0].legend(fontsize=7)
    ax[0].set_title('(a) single-neuron detectors')
    ax[1].set_title('(b) linear readability on homogeneous images')
    if 'Qwen3-VL-8B-Instruct' in taps_all:
        a = ax[2]
        t = taps_all['Qwen3-VL-8B-Instruct']
        keys = sorted({k for p in t['pairs'] for k in t['pairs'][p]})
        xs = np.arange(len(keys))
        for grp, pairs, col in (('training', TRAIN, '#2ca02c'), ('held-out feature', HFEAT, '#d62728'), ('held-out family', HFAM, '#9467bd')):
            a.plot(xs, [np.median([t['pairs'][p][k] for p in pairs]) for k in keys], marker='o', color=col, label=grp)
        if 'Qwen3.5-9B' in taps_all:
            t5 = taps_all['Qwen3.5-9B']
            a.scatter([keys.index('merger')], [np.median([t5['pairs'][p]['merger'] for p in TRAIN + HFEAT])],
                      marker='*', s=120, color='k', label='Qwen3.5-9B merger (training + held-out features)')
        lbl = {f'deepstack{j}': f"DeepStack {j}\n(block {b})" for j, b in enumerate(t['deepstack_blocks_1based'])}
        a.set_xticks(xs, [lbl.get(k, 'final merger') for k in keys])
        a.set_ylabel('Mahalanobis salience of the absent target')
        a.set_title('(c) Qwen3-VL: absent target at the tokens sent to the LLM')
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_readout(res: dict, path: Path) -> None:
    plt = _plt()
    names = [('r1_loc_n1200_s1', 'LoRA r8 13-15'), ('r1_late_n1200_s1', 'LoRA r8 25-27'), ('lora_r64_w13_s1', 'LoRA r64 13-15'),
             ('width_w13_m8_s1', 'new neurons 8/block'), ('width_w13_m38_s1', 'new neurons 38/block'),
             ('full_b14_lr1e4_s1', 'full FT block 14'), ('merger_s1', 'LoRA merger'), ('r1_vitall_n1200_s1', 'full-ViT LoRA')]
    names = [x for x in names if x[0] in res['readout']]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    for a, g, title in ((ax[0], 'train', 'training features'), (ax[1], 'hfeat', 'held-out features')):
        x = np.arange(len(names))
        r = [res['readout'][n][g] for n, _ in names]
        a.bar(x - 0.27, [v['ds_actual'] for v in r], 0.27, color='k', label='actual change of log p')
        a.bar(x, [v['fo_target'] for v in r], 0.27, color='#d62728', label="first order: change at the target's 4 tokens")
        a.bar(x + 0.27, [v['fo_other'] for v in r], 0.27, color='#7f7f7f', label='first order: change at all other image tokens')
        a.axhline(0, color='k', lw=0.6)
        a.set_xticks(x, [l for _, l in names], rotation=30, ha='right', fontsize=8)
        a.set_ylabel('nats (mean over 60 / 24 absent-target images)')
        a.set_title(f'decoder-input account of the gain, {title}')
    ax[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_pipeline(path: Path) -> None:
    """Schematic of the measured processing chain (numbers quoted from this round's base-model measurements)."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.2)
    ax.axis('off')
    boxes = [
        (0.2, 2.6, 1.8, 1.2, 'image\n(16 x 16 px patches)', '#eeeeee'),
        (2.4, 2.6, 2.2, 1.2, 'ViT blocks 1-6\npart detectors begin\nto form', '#eeeeee'),
        (5.0, 3.6, 3.2, 1.5, 'ViT blocks 7-27, PART PRESENT\ntens of neurons fire only on\nthe odd entity (zeta > 5)', '#c6dbef'),
        (5.0, 1.2, 3.2, 1.5, 'ViT blocks 7-27, PART ABSENT\nno selective neuron; target differs\nalong a shared direction (zeta 6-7)', '#fcbba1'),
        (8.7, 2.5, 2.0, 1.4, 'merger\n4 patches -> 1 token\npresent 26 / absent 4', '#eeeeee'),
        (11.1, 2.5, 2.3, 1.4, 'decoder layers 8-15\nFA L8 H6 template,\nMLP writes key mark', '#eeeeee'),
        (13.8, 2.5, 2.0, 1.4, 'reader heads\nL16 / L20\n-> row, column', '#eeeeee'),
    ]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=c, ec='k', lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=8.5)
    arr = dict(arrowstyle='->', lw=1.3)
    for (x0, y0), (x1, y1) in [((2.0, 3.2), (2.4, 3.2)), ((4.6, 3.4), (5.0, 4.3)), ((4.6, 3.0), (5.0, 2.0)),
                               ((8.2, 4.3), (8.7, 3.5)), ((8.2, 2.0), (8.7, 2.9)), ((10.7, 3.2), (11.1, 3.2)),
                               ((13.4, 3.2), (13.8, 3.2))]:
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=arr)
    ax.add_patch(FancyBboxPatch((2.6, 0.05), 9.2, 0.8, boxstyle='round,pad=0.05', fc='#fee391', ec='#b15928', lw=1.2))
    ax.text(7.2, 0.45, 'fine-tuning anywhere from block 7 to the merger: adds, at the target tokens, a small component\n'
            'along the direction the decoder reads (many neurons, small changes; no new selective unit)', ha='center',
            va='center', fontsize=8)
    ax.annotate('', xy=(9.7, 2.45), xytext=(9.0, 0.9), arrowprops=dict(arrowstyle='->', lw=1.2, color='#b15928'))
    ax.annotate('', xy=(6.6, 1.15), xytext=(6.6, 0.9), arrowprops=dict(arrowstyle='->', lw=1.2, color='#b15928'))
    ax.text(8, 5.9, 'Finding the odd entity: measured processing chain (Qwen3.5-9B, base model; adapter effect in yellow)',
            ha='center', fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--src-run', required=True)
    ap.add_argument('--figures', required=True)
    a = ap.parse_args()
    run, src, figs = Path(a.run), Path(a.src_run), Path(a.figures)
    figs.mkdir(parents=True, exist_ok=True)
    res = analyse(run, src)
    (run / 'summary.json').write_text(json.dumps(res, indent=1))
    for fn, args_ in ((fig_relocate, (run, src, res, figs / 'relocate_profiles.png')), (fig_windows, (res, figs / 'window_criteria.png')),
                      (fig_distributed, (run, res, figs / 'distributed_vs_detector.png')),
                      (fig_newneurons, (run, figs / 'new_neurons.png')), (fig_badcase, (res, figs / 'badcase.png')),
                      (fig_vl, (run, figs / 'vl_detectors.png')), (fig_readout, (res, figs / 'readout.png')), (fig_pipeline, (figs / 'pipeline.png',))):
        try:
            fn(*args_)
        except (KeyError, FileNotFoundError, ValueError, TypeError) as e:     # partial data while runs are pending
            print(f'figure {fn.__name__} skipped: {type(e).__name__} {e}')
    print(json.dumps({k: res[k] for k in ('window_mean', 'window_n_seeds', 'spearman')}, indent=1))


if __name__ == '__main__':
    main()
