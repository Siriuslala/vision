"""Experiment 3.12 round 2 figures (reads results.json / arrays.npz of digit_failure_r2_analysis).

Also regenerates the round-1 theory figure (Figure 1 of results_2026-10-20_digit_failure.md) with the red
band labels, from the round-1 analysis arrays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.position import digit_failure_figures as F

plt, C, GREY, INK, DIGIT_CMAP, save = F.plt, F.C, F.GREY, F.INK, F.DIGIT_CMAP, F.save
FAMS = (('sweep_h', 'disc x'), ('sweep_v', 'disc y'))


def fig_geometry(R: dict, A: dict, figs: Path) -> None:
    """Q1: the q0 digit-logit trajectory is a 2-D arc; digits own angular sectors."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for row, (fam, label) in enumerate(FAMS):
        g = R['geometry'][fam]
        x, z1, z2, th = A[f'{fam}_x'], A[f'{fam}_z1'], A[f'{fam}_z2'], A[f'{fam}_theta']
        vt, s, L0 = A[f'{fam}_vt'], A[f'{fam}_s'], A[f'{fam}_L0']
        ax = axes[row, 0]
        sc = ax.scatter(z1, z2, c=x, cmap='viridis', s=14, vmin=100, vmax=1000)
        rad = np.hypot(z1, z2).max() * 1.15
        for d in range(1, 10):
            ph = np.radians(g['digit_phase_deg'][d])
            ax.plot([0, rad * np.cos(ph)], [0, rad * np.sin(ph)], color=GREY, lw=0.6, ls='--')
            ax.text(1.08 * rad * np.cos(ph), 1.08 * rad * np.sin(ph), str(d), color='red', ha='center', va='center',
                    fontsize=9, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlabel('$z_1$ (nats)')
        ax.set_ylabel('$z_2$ (nats)')
        ax.set_title(f'{label}: q0 digit-logit trajectory (top-2 SVD plane)', fontsize=9)
        fig.colorbar(sc, ax=ax, fraction=0.046, label='true coordinate x (0-1000)')
        ax = axes[row, 1]
        ax.scatter(x, np.degrees(th), c=x, cmap='viridis', s=10, vmin=100, vmax=1000)
        ax.set_xlabel('true coordinate x')
        ax.set_ylabel('angle θ (degrees, from x = 100)')
        ax.set_title(f"angle vs x: Spearman {g['angle_spearman']:.3f}, {g['deg_per_100_units']:.0f}° per 100 units", fontsize=9)
        ax = axes[row, 2]
        rec = (np.column_stack([z1, z2]) @ np.diag(1 / s[:2])) @ (vt[:2] * s[:2, None]) + A[f'{fam}_mu']
        o = np.argsort(th)
        for d in range(1, 10):
            ax.plot(np.degrees(th[o]), rec[o, d], color=DIGIT_CMAP(d / 9), lw=1.2)
        env = rec.max(1)
        ax.plot(np.degrees(th[o]), env[o], color=INK, lw=0.8, ls=':', label='upper envelope')
        win = rec.argmax(1)
        span = env.max() - env.min()
        for d in np.unique(win):
            idx = o[np.isin(o, np.where(win == d)[0])]
            m = idx[len(idx) // 2]
            ax.text(np.degrees(th[m]), env[m] + 0.04 * span, str(int(d)), color='red', ha='center', fontsize=9,
                    fontweight='bold')
        ax.set_xlabel('angle θ (degrees)')
        ax.set_ylabel('rank-2 digit logit (centred, nats)')
        ax.set_title('digit logits are cosines of θ (rank-2 reconstruction)', fontsize=9)
        ax.legend(fontsize=7, loc='lower center')
        ax = axes[row, 3]
        ev0 = g['q0_ev'][:3]
        ev1 = np.mean([g['q1'][str(k)]['ev'] for k in range(1, 10)], 0)
        w = 0.35
        ax.bar(np.arange(3) - w / 2, ev0, w, color=C[0], label='q0 (first digit)')
        ax.bar(np.arange(3) + w / 2, ev1, w, color=C[1], label='q1 (second digit, mean over k)')
        ax.set_xticks(range(3), ['rank 1', 'rank 2', 'rank 3'])
        ax.set_ylabel('share of image-driven logit variance')
        ax.set_title('dimensionality of the logit trajectory', fontsize=9)
        ax.legend(fontsize=7)
    save(fig, figs / 'geometry.png')


def fig_shift(R: dict, figs: Path) -> None:
    """Two-stage test at q1: moving the image by +100 and k by +1 together vs separately."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    names = [('d_image_only', 'image +100 only'), ('d_k_only', 'k +1 only'), ('d_shift_other', 'both (k away from 4/5)'),
             ('d_shift_centre', 'both (k = 3, 4 or 5)')]
    w = 0.38
    for j, (fam, label) in enumerate(FAMS):
        st = R['q1_model'][fam]['shift_test']
        ax.bar(np.arange(4) + (j - 0.5) * w, [st[n] for n, _ in names], w, color=C[j], label=label)
    ax.set_xticks(range(4), [n for _, n in names])
    ax.set_ylabel('mean squared change of q1 digit logits (nats²)')
    ax.legend(fontsize=7)
    save(fig, figs / 'shift.png')


def _bar_conditions(ax, tab: dict, names: list[tuple[str, str]], color: str) -> None:
    vals = [tab[n]['r'] for n, _ in names]
    lo = [tab[n]['r'] - tab[n]['ci'][0] for n, _ in names]
    hi = [tab[n]['ci'][1] - tab[n]['r'] for n, _ in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=color, xerr=[lo, hi], error_kw={'lw': 0.8, 'ecolor': INK})
    ax.set_yticks(y, [lab for _, lab in names])
    ax.axvline(0, color=GREY, lw=0.6)
    ax.axvline(1, color=GREY, lw=0.6, ls=':')
    ax.set_xlim(-0.1, 1.15)
    ax.set_xlabel('recovery of the run-B digit-logit change')


def fig_sources(R: dict, A: dict, figs: Path) -> None:
    """Q3: causal sources of the first (q0) and second (q1) digit."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={'width_ratios': [1.1, 1.1, 1]})
    common = [('img', 'image KV, 8 full-attn layers'), ('img@20', 'image KV, layer 20 only'),
              ('H11<-img@20', 'L20 H11 output (path via image KV)'), ('post', 'post-image text KV'),
              ('rec', 'DeltaNet recurrent states'), ('convlast', 'DeltaNet conv slot of previous token'),
              ('convlast@25-32', '  same, layers 25-31 only'), ('convlast@1-8', '  same, layers 1-8 only'),
              ('delta', 'all DeltaNet states'), ('img+delta', 'image KV + DeltaNet states')]
    q1_names = common[:4] + [('q0', 'q0 token KV (full attn)')] + common[4:]
    for ax, role, names, color, title in ((axes[0], 'q0', common, C[0], 'first digit (step q0)'),
                                         (axes[1], 'q1', q1_names, C[1], 'second digit (step q1)')):
        p = R['patch'][role]
        _bar_conditions(ax, p['conditions'], names, color)
        ax.set_title(f"{title}: {p['n_pairs']} disc pairs", fontsize=9)
    ax = axes[2]
    for role, color, ls, label in (('q0_x', C[0], '-', 'first digit, x'), ('q0_y', C[0], '--', 'first digit, y'),
                                   ('q1_x', C[1], '-', 'second digit, x'), ('q1_y', C[1], '--', 'second digit, y')):
        t = A[f'tail_{role}']
        ax.plot(t[:, 0], t[:, 1], color=color, ls=ls, marker='o', ms=3, label=label)
        ax.fill_between(t[:, 0], t[:, 2], t[:, 3], color=color, alpha=0.2, lw=0)
    ax.axvline(0, color=GREY, lw=0.6, ls='--')
    ax.text(0.1, 0.05, 'q0', color=GREY, fontsize=8)
    ax.set_xlabel('first position seeing run-B image KV (rel. to q0)')
    ax.set_ylabel('recovery of the run-B digit-logit change')
    ax.set_title('where along the answer the image is read', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    save(fig, figs / 'sources.png')


def fig_attention(R: dict, figs: Path) -> None:
    """Attention mass at q1 on token groups (correlational counterpart of fig_sources)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    layers = [4, 8, 12, 16, 20, 24, 28, 32]
    groups = [('image', 'image tokens', C[0]), ('post', 'post-image text', C[2]), ('q0', 'q0 token', C[4]),
              ('q1', 'q1 itself (first digit)', C[1])]
    for ax, q in zip(axes, ('q0', 'q1')):
        a = R['attention']['sweep_h'][q]
        for g, lab, col in groups:
            if q == 'q0' and g == 'q1':
                continue
            ax.plot(layers, a[g], color=col, marker='o', ms=3, label=lab)
        ax.set_xlabel('full-attention layer')
        ax.set_title(f'disc x, query {q} (head mean)', fontsize=9)
    axes[0].set_ylabel('attention mass')
    axes[1].legend(fontsize=7)
    save(fig, figs / 'attention.png')


def fig_margin(R: dict, A: dict, figs: Path) -> None:
    """What q1 reads: E[second digit] against the true offset and against the q0 neighbour margin."""
    fams = [('sweep_h', 'disc x'), ('sweep_v', 'disc y'), ('refcoco', 'RefCOCO x'), ('grid', 'grid x'),
            ('textconv', 'text-given value')]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), sharey=True)
    for j, (fam, label) in enumerate(fams):
        m = R['margin'][fam]
        E, o, d = A[f'margin_{fam}_E'], A[f'margin_{fam}_o'], A[f'margin_{fam}_d']
        ax = axes[0, j]
        ax.scatter(o, E, c=o, cmap='viridis', s=10, vmin=0, vmax=10)
        ax.plot([0, 10], [0, 9.5], color=GREY, ls='--', lw=0.7)
        ax.set_title(f"{label}: R² offset {m['r2_offset']:.2f}", fontsize=9)
        ax.set_xlabel('true in-bin offset (second-digit units)')
        ax = axes[1, j]
        sc = ax.scatter(d, E, c=o, cmap='viridis', s=10, vmin=0, vmax=10)
        ax.set_title(f"R² margins {m['r2_margins']:.2f} (both {m['r2_both']:.2f})", fontsize=9)
        ax.set_xlabel('q0 margin log P(k+1) − log P(k−1) (nats)')
    axes[0, 0].set_ylabel('expected second digit at q1')
    axes[1, 0].set_ylabel('expected second digit at q1')
    fig.colorbar(sc, ax=axes[:, -1].tolist(), fraction=0.04, label='true in-bin offset')
    save(fig, figs / 'margin.png')


def fig_steer(R: dict, A: dict, figs: Path) -> None:
    if not R.get('steer'):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, l in zip(axes, (24, 28)):
        for key, col, lab in ((f'margin@{l}', C[0], 'margin direction'), (f'random@{l}', GREY, 'random direction')):
            dd, dE = A[f'steer_all_{key}_dd'], A[f'steer_all_{key}_dE']
            ax.scatter(dd, dE, s=6, color=col, alpha=0.5, label=f"{lab} (slope {R['steer']['all']['slopes'][key]['slope']:.2f})")
        ax.axhline(0, color=GREY, lw=0.5)
        ax.axvline(0, color=GREY, lw=0.5)
        ax.set_xlabel('change of q0 margin log P(k+1) − log P(k−1) (nats)')
        ax.set_title(f'steering the q0 residual at layer {l}', fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('change of expected second digit at q1')
    save(fig, figs / 'steer.png')


def fig_prior(R: dict, A: dict, figs: Path) -> None:
    """Q4: the centre prior along the sweeps, by family, and under ablation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    for j, (fam, label) in enumerate(FAMS):
        b = R['prior_behaviour'][fam]['p_prior_by_bin']
        mids = [int(k.split('-')[0]) + 50 for k in b]
        ax.plot(mids, [v['p9_k4'] for v in b.values()], color=C[j], marker='o', ms=3, label=f'{label}: P(9 | k=4)')
        ax.plot(mids, [v['p0_k5'] for v in b.values()], color=C[j], marker='s', ms=3, ls='--', label=f'{label}: P(0 | k=5)')
    ax.axvspan(400, 600, color=GREY, alpha=0.15, lw=0)
    ax.set_xlabel('true coordinate x (bin centre)')
    ax.set_ylabel('probability (forced first digit k)')
    ax.legend(fontsize=6)
    ax = axes[1]
    fams = ['blank', 'sweep_h', 'sweep_v', 'grid', 'refcoco', 'textconv', 'count']
    pd = R['prior_dla']
    vals = [pd[f]['k4_d9']['mean'] if 'k4_d9' in pd.get(f, {}) else np.nan for f in fams]
    sds = [pd[f]['k4_d9']['sd_images'] if 'k4_d9' in pd.get(f, {}) else 0 for f in fams]
    ax.bar(range(len(fams)), vals, yerr=sds, color=C[0], error_kw={'lw': 0.8})
    ax.set_xticks(range(len(fams)), fams, rotation=30)
    ax.set_ylabel('direct prior logit for 9 after 4 (nats)')
    ax.set_title('MLP 27-31 + L32 H8, mean ± SD over inputs', fontsize=9)
    ax = axes[2]
    ab = R.get('ablation', {})
    if ab:
        conds = ab['conditions']
        for j, (fam, label) in enumerate(FAMS):
            if fam not in ab:
                continue
            ax.plot(range(len(conds)), [ab[fam][c].get('plateau', np.nan) for c in conds], color=C[j], marker='o', ms=3,
                    label=f'{label}: 500-plateau share')
        if 'textconv' in ab:
            ax.plot(range(len(conds)), [ab['textconv'][c]['p_true'] for c in conds], color=C[2], marker='s', ms=3,
                    label='text-given value: P(true 2nd digit)')
        ax.set_xticks(range(len(conds)), conds, rotation=40, fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_title('mean ablation at q1', fontsize=9)
        ax.legend(fontsize=6)
    save(fig, figs / 'prior.png')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--figures', required=True)
    p.add_argument('--round1-figures', required=True)
    args = p.parse_args()
    run, figs = Path(args.run), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    R = json.loads((run / 'results.json').read_text())
    A = dict(np.load(run / 'arrays.npz', allow_pickle=True))
    fig_geometry(R, A, figs)
    fig_shift(R, figs)
    fig_sources(R, A, figs)
    fig_attention(R, figs)
    fig_margin(R, A, figs)
    fig_steer(R, A, figs)
    fig_prior(R, A, figs)
    r1 = Path(args.round1)
    F.fig_theory(dict(np.load(r1 / 'analysis.npz', allow_pickle=True)), json.loads((r1 / 'results.json').read_text()),
                 Path(args.round1_figures))


if __name__ == '__main__':
    main()
