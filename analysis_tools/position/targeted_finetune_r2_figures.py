"""Figures of experiment 3.13 round 2 (from results_r2.json and analysis_r2.npz; no model is loaded)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.position.coordinate_value_r2_figures import manifold_figure

C1, C2, C3, C4, C5, C6, C7, C8 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'
GREY = '0.55'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 160,
                     'savefig.bbox': 'tight', 'axes.grid': True, 'grid.color': '0.92', 'grid.linewidth': 0.6})

# Condition -> (label, family colour). Families: reference, readout stage only, readout stage frozen,
# one component type everywhere, one upstream site, scaled.
COND = {
    'base': ('base', GREY), 'all_linear': ('all-linear', C1),
    'mlp28_32': ('MLP 28-32', C2), 'mlp28_32_r64': ('MLP 28-32 r64', C2), 'mlp28_32_lr1e3': ('MLP 28-32 lr 1e-3', C2),
    'late25_32': ('MLP+DN 25-32', C2),
    'all_but_mlp28_32': ('all-linear, MLP 28-32 frozen', C3), 'all_but_25_32': ('all-linear, layers 25-32 frozen', C3),
    'mlp_all': ('all MLPs', C4), 'fa_attn': ('full attention q/k/v/o', C4), 'dn_proj': ('DeltaNet projections', C4),
    'fa_mlp': ('full attention + MLPs', C4),
    'mixer16_24': ('mixers 16-24', C5), 'mlp10_14': ('MLP 10-14', C5),
    'scaled_all_linear': ('scaled all-linear', C7), 'scaled_all_but_25_32': ('scaled, 25-32 frozen', C7),
    'early1_4_img': ('layers 1-4 @ image tokens', C8), 'all_linear_img': ('all layers @ image tokens', C8),
    'scaled_early1_4_img': ('scaled, layers 1-4 @ image', C7),
    'all_linear+band1_4': ('splice: tuned layers 1-4 only', C6), 'all_linear+full@image': ('splice: tuned @ image only', C6),
}
FAMILY_LEGEND = [(GREY, 'no training'), (C1, 'all linear layers'), (C2, 'readout stage only (layers 25-32)'),
                 (C3, 'all linear, readout stage frozen'), (C4, 'one component type, all layers'),
                 (C5, 'one upstream site (round 1)'), (C8, 'located: image-token positions only'), (C7, 'scaled data')]
ORDER = ['base', 'all_linear', 'all_but_mlp28_32', 'all_but_25_32', 'mlp28_32', 'mlp28_32_r64', 'mlp28_32_lr1e3',
         'late25_32', 'mlp_all', 'fa_mlp', 'fa_attn', 'dn_proj', 'mixer16_24', 'mlp10_14', 'all_linear_img', 'early1_4_img',
         'scaled_all_linear', 'scaled_early1_4_img']


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


def _family_legend(ax, loc='lower right', families=None):
    from matplotlib.patches import Patch
    fams = FAMILY_LEGEND if families is None else [f for f in FAMILY_LEGEND if f[0] in families]
    ax.legend(handles=[Patch(color=c, label=l) for c, l in fams], loc=loc, fontsize=7, frameon=False)


# ----------------------------------------------------------------------------- controls

def fig_controls(R: dict, figs: Path) -> None:
    conds = [c for c in ORDER if c in R['coord'] and not c.startswith('scaled')]
    mech = R.get('mechanism', {})
    panels = [('disc x mean centre error (0-1000 units)', lambda c: R['coord'][c]['disc_x']['error']['mean']),
              ('20x20 grid hit rate', lambda c: R['coord'][c]['grid20']['hit']['p']),
              ('second-digit gain (ideal 1)', lambda c: mech[c]['clamp']['sweep_h']['slope_E_offset_in_range']
               if c in mech else np.nan)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.32 * len(conds) + 1.2), sharey=True)
    y = np.arange(len(conds))[::-1]
    for ax, (title, f) in zip(axes, panels):
        vals = [f(c) for c in conds]
        ax.barh(y, vals, color=[COND[c][1] for c in conds], height=0.7)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v, yi, f' {v:.2f}' if v < 5 else f' {v:.1f}', va='center', fontsize=7, color='0.2')
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(y, [COND[c][0] for c in conds])
    from matplotlib.patches import Patch
    fams = [f for f in FAMILY_LEGEND if f[0] in {COND[c][1] for c in conds}]
    fig.legend(handles=[Patch(color=c, label=l) for c, l in fams], loc='lower center', ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    _save(fig, figs / 'controls.png')


# ----------------------------------------------------------------------------- splices

def fig_splice_depth(R: dict, figs: Path, cond: str = 'all_linear') -> None:
    S = R['splice'].get(cond)
    if not S:
        return
    ks = list(range(4, 32, 4))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for ax, key, title in zip(axes, ('gain', 'p_second', 'value_err'),
                              ('second-digit gain', 'P(true second digit)', 'teacher-forced value error (units)')):
        base, full = S['base']['pooled'][key], S['full']['pooled'][key]
        kk = [0] + ks + [32]
        fwd = [base] + [S[f'fwd{k}']['pooled'][key] for k in ks] + [full]
        rev = [full] + [S[f'rev{k}']['pooled'][key] for k in ks] + [base]
        ax.plot(kk, fwd, '-o', color=C1, ms=4, lw=2, label='tuned layers 1..k, base layers k+1..32')
        ax.plot(kk, rev, '-s', color=C2, ms=4, lw=2, label='base layers 1..k, tuned layers k+1..32')
        ax.axhline(base, color=GREY, ls=':', lw=1)
        ax.axhline(full, color=C1, ls=':', lw=1)
        ax.set_xlabel('splice layer k')
        ax.set_title(title, fontsize=9)
        ax.set_xticks(kk)
    axes[0].legend(fontsize=7, frameon=False, loc='center right')
    _save(fig, figs / f'splice_depth_{cond}.png')


def fig_splice_bars(R: dict, figs: Path, cond: str = 'all_linear') -> None:
    S = R['splice'].get(cond)
    if not S:
        return
    groups = [('band', [f'band{a}_{a + 3}' for a in range(1, 33, 4)], C5),
              ('type', ['mlp', 'fa', 'dn', 'mixer', 'mlp_le24', 'mlp_ge25', 'mixer_le24', 'mixer_ge25'], C4),
              ('position', ['full@image', 'full@question', 'full@answer', 'full@text', 'fwd24@image', 'fwd24@question',
                            'fwd24@answer', 'fwd24@text'], C3)]
    labels = {'mlp': 'MLP (all layers)', 'fa': 'full attention', 'dn': 'DeltaNet', 'mixer': 'both mixers',
              'mlp_le24': 'MLP 1-24', 'mlp_ge25': 'MLP 25-32', 'mixer_le24': 'mixers 1-24', 'mixer_ge25': 'mixers 25-32',
              'full@image': 'all layers @ image tokens', 'full@question': 'all layers @ question tokens',
              'full@answer': 'all layers @ answer prefix', 'full@text': 'all layers @ all text',
              'fwd24@image': 'layers 1-24 @ image tokens', 'fwd24@question': 'layers 1-24 @ question tokens',
              'fwd24@answer': 'layers 1-24 @ answer prefix', 'fwd24@text': 'layers 1-24 @ all text'}
    names = [(n, c) for _, ns, c in groups for n in ns if n in S]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.3 * len(names) + 1.2), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, key, title in zip(axes, ('gain', 'p_second'), ('share of the second-digit gain change reproduced',
                                                            'share of the P(true second digit) change reproduced')):
        vals = [S[n]['recovery'][key] for n, _ in names]
        ax.barh(y, vals, color=[c for _, c in names], height=0.7)
        if key == 'gain':
            lo = [S[n]['pooled']['gain_ci'][0] for n, _ in names]
            hi = [S[n]['pooled']['gain_ci'][1] for n, _ in names]
            b, f = S['base']['pooled']['gain'], S['full']['pooled']['gain']
            ax.errorbar(vals, y, xerr=[np.array(vals) - (np.array(lo) - b) / (f - b), (np.array(hi) - b) / (f - b) - np.array(vals)],
                        fmt='none', ecolor='0.3', lw=0.8, capsize=2)
        ax.axvline(0, color='0.3', lw=0.8)
        ax.axvline(1, color=C1, ls=':', lw=1)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(y, [labels.get(n, n.replace('band', 'layers ').replace('_', '-')) for n, _ in names])
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=C5, label='4-layer band'), Patch(color=C4, label='component type'),
                            Patch(color=C3, label='token positions')], fontsize=7, frameon=False, loc='lower right')
    _save(fig, figs / f'splice_bars_{cond}.png')


# ----------------------------------------------------------------------------- geometry

def fig_spectrum(R: dict, figs: Path) -> None:
    G = R['geometry']
    conds = [c for c in ('base', 'all_linear', 'all_but_25_32', 'mlp10_14', 'mlp28_32_lr1e3') if c in G]
    if not conds:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, site in zip(axes, ('q0', 'q1', 'target')):
        for c in conds:
            e = G[c]['sweep_h'][site]['extra']['100']
            ax.plot(range(len(e)), e, '-', lw=2, color=COND[c][1] if c != 'mlp10_14' else C5, label=COND[c][0])
        if 'textconv' in G.get('base', {}):
            e = G['base']['textconv'][site]['extra']['100']
            ax.plot(range(len(e)), e, '--', lw=1.5, color='0.2', label='base, text-given value')
        ax.axhline(G[conds[0]]['sweep_h'][site]['chance'], color=GREY, ls=':', lw=1)
        ax.set_title({'q0': 'q0 (token before the first digit)', 'q1': 'q1 (first-digit token)',
                      'target': 'target image tokens (mean)'}[site], fontsize=9)
        ax.set_xlabel('layer')
    axes[0].set_ylabel('extra R$^2$ of a period-100 component')
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, figs / 'spectrum_period100.png')
    # Full spectrum at q0 for base vs all-linear (heatmaps).
    have = [c for c in ('base', 'all_linear', 'all_but_25_32') if c in G]
    fig, axes = plt.subplots(1, len(have), figsize=(4.2 * len(have) + 0.8, 3.2))
    axes = np.atleast_1d(axes)
    periods = list(G[have[0]]['sweep_h']['q0']['extra'])
    vmax = max(max(max(G[c]['sweep_h']['q0']['extra'][T]) for T in periods) for c in have)
    for ax, c in zip(axes, have):
        M = np.array([G[c]['sweep_h']['q0']['extra'][T] for T in periods])
        im = ax.imshow(M, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=vmax)
        ax.set_yticks(range(len(periods)), periods)
        ax.set_xlabel('layer')
        ax.set_title(f'{COND[c][0]}: q0, disc x sweep', fontsize=9)
        ax.grid(False)
    axes[0].set_ylabel('period T (0-1000 units)')
    cb = fig.colorbar(im, ax=list(axes), fraction=0.03)
    cb.set_label('extra R$^2$ of cos/sin(2$\\pi$x/T)')
    _save(fig, figs / 'spectrum_q0.png')


def fig_manifolds(A: dict, figs: Path) -> None:
    for site in ('q0', 'target'):
        for conds in (('base', 'all_linear'), ('base', 'all_linear+band1_4')) if site == 'target' else (('base', 'all_linear'),):
            panels = []
            for c in conds:
                for l in (16, 20, 24, 28):
                    k = f'{c}/sweep_h/{site}/L{l}/pca3'
                    if k not in A:
                        continue
                    var = A[f'{c}/sweep_h/{site}/L{l}/var3']
                    panels.append({'P': A[k], 'c': A[f'{c}/sweep_h/x'], 'title': f'{COND[c][0]}, layer {l}',
                                   'subtitle': f'{site}; PC1-3 {100 * var.sum():.0f}% var.'})
            if panels:
                manifold_figure(panels, figs / f"manifold_{site}{'_hybrid' if conds[1] != 'all_linear' else ''}.png",
                                'true x (0-1000 units)',
                                suptitle=f'Qwen3.5-9B residual stream at {site}, disc x sweep (PCA 3-D)')


def fig_clock(A: dict, R: dict, figs: Path) -> None:
    conds = [c for c in ('base', 'all_linear', 'all_but_25_32', 'mlp10_14') if f'{c}/sweep_h/q0/L24/clock' in A]
    if not conds:
        return
    fig, axes = plt.subplots(2, len(conds), figsize=(3.3 * len(conds), 6.4))
    axes = np.array(axes).reshape(2, -1)
    norm = mcolors.Normalize(0, 1000)
    for j, c in enumerate(conds):
        x = A[f'{c}/sweep_h/x']
        P = A[f'{c}/sweep_h/q0/L24/clock']
        ax = axes[0, j]
        sc = ax.scatter(P[:, 0], P[:, 1], c=x, cmap='viridis', norm=norm, s=14, edgecolors='0.3', linewidths=0.2)
        ax.set_aspect('equal')
        ax.set_title(f'{COND[c][0]}: q0, layer 24', fontsize=9)
        ax.set_xlabel('clock plane axis 1')
        ax.set_ylabel('clock plane axis 2')
        g = R['geometry'][c]['sweep_h']['q0']
        ax.text(0.02, 0.98, f"period-100 extra R$^2$ {g['extra']['100'][24]:.2f}", transform=ax.transAxes, va='top', fontsize=7)
        ax = axes[1, j]
        ang = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
        ax.scatter(x % 100, ang, c=x, cmap='viridis', norm=norm, s=10)
        ax.set_xlabel('x mod 100 (units)')
        ax.set_ylabel('clock angle (degrees)')
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02).set_label('true x (0-1000 units)')
    _save(fig, figs / 'clock_q0_L24.png')


def fig_alignment(R: dict, figs: Path) -> None:
    Al = R.get('alignment', {})
    if 'base' not in Al or 'all_linear' not in Al:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    tags = [t for t in Al['base'] if t.endswith('q1<-q0')]
    layers = [int(t.split('/')[0][1:]) for t in tags]
    for ax, (tagsuf, title) in zip(axes, (('q0<-q0', 'first-digit readout (q0 logits, q0 residual)'),
                                          ('q1<-q0', 'second-digit readout (q1 logits, q0 residual)'),
                                          ('q1<-q1', 'second-digit readout (q1 logits, q1 residual)'))):
        for rd, cd, col, ls, lab in (('base', 'base', GREY, '-', 'base code, base readout'),
                                     ('all_linear', 'all_linear', C1, '-', 'tuned code, tuned readout'),
                                     ('base', 'all_linear', C1, '--', 'tuned code, base readout'),
                                     ('all_linear', 'base', GREY, '--', 'base code, tuned readout')):
            v = [Al[rd][f'L{l}/{tagsuf}'][f'code_{cd}']['gain_spread'] for l in layers
                 if f'code_{cd}' in Al[rd][f'L{l}/{tagsuf}']]
            if len(v) == len(layers):
                ax.plot(layers, v, ls, color=col, marker='o', ms=4, lw=2, label=lab)
        ax.set_xticks(layers)
        ax.set_xlabel('layer of the residual')
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel('readout gain: s.d. over digits of\nlogit change per 100 units (nats)')
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, figs / 'alignment_gain.png')


def fig_emergence(R: dict, figs: Path) -> None:
    """Cross-validated decodability of the period-100 clock (and of x) per layer at the target tokens and at q0."""
    G = R['geometry']
    conds = [c for c in ('base', 'all_linear', 'all_linear+band1_4', 'early1_4_img', 'mlp10_14', 'all_but_25_32',
                         'mlp28_32_lr1e3') if c in G]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, (fam, site, title) in zip(axes, (('sweep_h', 'target', 'disc sweep: target image tokens'),
                                             ('sweep_h', 'q0', 'disc sweep: q0'), ('refcoco', 'q0', 'RefCOCO: q0'))):
        for c in conds:
            if fam not in G[c]:
                continue
            v = G[c][fam][site]['decode100']['periodic']
            ls = '--' if c == 'all_linear+band1_4' else '-'
            ax.plot(range(len(v)), v, ls, lw=2, color=COND[c][1] if c != 'mlp10_14' else C5, label=COND[c][0])
        ax.axhline(0, color='0.3', lw=0.8)
        ax.set_ylim(-0.3, 1.02)
        ax.set_xlabel('layer')
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel('held-out R$^2$ of cos/sin(2$\\pi$x/100)')
    axes[0].legend(fontsize=7, frameon=False, loc='upper left')
    _save(fig, figs / 'clock_emergence.png')


# Scaled-data figure: colour = trainable location (shared with the round-1-data condition), hatching = scaled data,
# so that the three scaled conditions stay distinguishable.
SCALED_STYLE = {
    'base': (GREY, ''), 'all_linear': (C1, ''), 'early1_4_img': (C8, ''),
    'scaled_all_linear': (C1, '///'), 'scaled_early1_4_img': (C8, '///'), 'scaled_all_but_25_32': (C3, '///'),
}


def fig_scaled(R: dict, figs: Path) -> None:
    conds = [c for c in ('base', 'all_linear', 'early1_4_img', 'scaled_all_linear', 'scaled_early1_4_img', 'scaled_all_but_25_32')
             if c in R['coord']]
    groups = [('disc_x', 'error', 'disc x error'), ('grid20', 'error', '20x20 grid error'), ('chart_ui', 'error', 'chart/UI error'),
              ('refcoco', 'error', 'RefCOCO error'), ('clutter', 'hit', 'clutter hit rate'), ('refcoco', 'iou', 'RefCOCO IoU')]
    fig, axes = plt.subplots(1, len(groups), figsize=(15, 3.2))
    for ax, (g, k, title) in zip(axes, groups):
        vals = []
        for c in conds:
            v = R['coord'][c][g][k]
            vals.append(v['mean'] if 'mean' in v else v['p'])
        for i, c in enumerate(conds):
            ax.bar(i, vals[i], color=SCALED_STYLE[c][0], hatch=SCALED_STYLE[c][1], edgecolor='white', linewidth=0, width=0.7)
        if k == 'hit':
            lo = [R['coord'][c][g][k]['ci'][0] for c in conds]
            hi = [R['coord'][c][g][k]['ci'][1] for c in conds]
            ax.errorbar(range(len(conds)), vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt='none', ecolor='0.3', lw=0.8)
            ax.set_ylim(0.6, 1.0)
        elif 'ci' in R['coord'][conds[0]][g][k]:
            lo = [R['coord'][c][g][k]['ci'][0] for c in conds]
            hi = [R['coord'][c][g][k]['ci'][1] for c in conds]
            ax.errorbar(range(len(conds)), vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt='none', ecolor='0.3', lw=0.8)
        ax.set_xticks(range(len(conds)), [COND[c][0] for c in conds], rotation=60, ha='right', fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='x', visible=False)
    _save(fig, figs / 'scaled_coord.png')
    gen = R.get('general', {})
    gc = [c for c in conds if c in gen]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    benches = ('pope', 'textvqa', 'mmvp', 'gqa')
    w = 0.8 / len(gc)
    for j, c in enumerate(gc):
        vals = [gen[c][b]['acc']['mean'] for b in benches]
        lo = [gen[c][b]['acc']['ci'][0] for b in benches]
        hi = [gen[c][b]['acc']['ci'][1] for b in benches]
        xs = np.arange(len(benches)) + (j - (len(gc) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.9, color=SCALED_STYLE[c][0], hatch=SCALED_STYLE[c][1], edgecolor='white', linewidth=0,
               label=COND[c][0])
        ax.errorbar(xs, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt='none', ecolor='0.3', lw=0.8)
    ax.set_xticks(range(len(benches)), ['POPE', 'TextVQA', 'MMVP', 'GQA'])
    ax.set_ylabel('accuracy (200 questions each)')
    ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax.set_ylim(0, 1.05)
    _save(fig, figs / 'scaled_general.png')


def fig_causal(R: dict, figs: Path) -> None:
    """Clock ablation / injection and the constant-shift test (teacher-forced P(true second digit) and gain)."""
    C, S = R.get('clock', {}), R.get('steer', {})
    if not C or not S:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    names = list(C)
    y = np.arange(len(names))[::-1]
    for key, off, col, lab in (('p_second', 0.2, C1, 'P(true second digit)'), ('gain', -0.2, C2, 'second-digit gain')):
        axes[0].barh(y + off, [C[n][key] for n in names], height=0.38, color=col, label=lab)
    axes[0].set_yticks(y, [n.replace('all_linear/', 'tuned: ').replace('base/', 'base: ') for n in names], fontsize=7)
    axes[0].set_title('clock plane ablation (tuned) and injection (base)', fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, loc='lower right')
    names = list(S)
    y = np.arange(len(names))[::-1]
    for key, off, col, lab in (('p_second', 0.2, C1, 'P(true second digit)'), ('gain', -0.2, C2, 'second-digit gain')):
        axes[1].barh(y + off, [S[n][key] for n in names], height=0.38, color=col, label=lab)
    axes[1].set_yticks(y, names, fontsize=7)
    axes[1].set_title('base + constant layer-4 shift at image tokens (disc sweeps)', fontsize=9)
    for ax in axes:
        ax.grid(axis='y', visible=False)
        ax.set_xlim(0, 1.05)
    _save(fig, figs / 'clock_causal.png')


def fig_margin(R: dict, figs: Path) -> None:
    """3.12 round-3 quantity: resolution of the first-digit logit difference m per condition (from the captures)."""
    G = R['geometry']
    conds = [c for c in ORDER + ['all_linear+band1_4'] if c in G]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(conds) + 1))
    y = np.arange(len(conds))[::-1]
    for fam, off, col in (('sweep_h', 0.2, C1), ('sweep_v', -0.2, C2)):
        ax.barh(y + off, [G[c][fam]['margin']['resolution'] for c in conds], height=0.38, color=col,
                label={'sweep_h': 'disc x sweep', 'sweep_v': 'disc y sweep'}[fam])
    ax.set_yticks(y, [COND[c][0] for c in conds], fontsize=7)
    ax.set_xlabel('resolution of m = l0(k+1) - l0(k-1): 10 x residual s.d. / slope (units)')
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis='y', visible=False)
    _save(fig, figs / 'margin_resolution.png')


def main_figures(run: Path, figs: Path) -> None:
    figs.mkdir(parents=True, exist_ok=True)
    R = json.loads((run / 'results_r2.json').read_text())
    A = dict(np.load(run / 'analysis_r2.npz'))
    for f in (fig_controls, fig_splice_depth, fig_splice_bars, fig_spectrum, fig_alignment, fig_emergence, fig_scaled,
              fig_causal, fig_margin):
        try:
            f(R, figs)
        except KeyError as e:          # a section whose runs are not finished yet
            print(f'{f.__name__}: missing {e}', flush=True)
    fig_manifolds(A, figs)
    fig_clock(A, R, figs)
    print('figures written', flush=True)
