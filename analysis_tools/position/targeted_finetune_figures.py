"""Experiment 3.13 figures (reads results.json / analysis.npz written by targeted_finetune_analysis)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.position import digit_failure as df
from analysis_tools.position import targeted_finetune as tf

COLORS = {'base': '#8a8984', 'all_linear': '#2a78d6', 'mlp28_32': '#eb6834', 'mixer16_24': '#1baf7a', 'mlp28_32_r64': '#6250d6', 'mlp10_14': '#eda100', "late25_32": "#e87ba4"}
LABELS = {"late25_32": "(b') MLP+DeltaNet-in 25-32", 'base': 'base', 'all_linear': '(a) all linear', 'mlp28_32': '(b) MLP 28-32', 'mlp28_32_r64': '(b-r64) MLP 28-32, rank 64', 'mixer16_24': '(c) mixer 16-24',
          'mlp10_14': '(d) MLP 10-14'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'axes.axisbelow': True})


def _bars(ax, groups, conds, value, err=None, title='', ylabel='', ref=None):
    w = 0.8 / len(conds)
    x = np.arange(len(groups))
    for j, c in enumerate(conds):
        v = np.array([value(c, g) for g in groups], float)
        e = None
        if err is not None:
            lo_hi = np.array([err(c, g) for g in groups], float)
            e = np.abs(lo_hi.T - v)
        ax.bar(x + (j - (len(conds) - 1) / 2) * w, v, w * 0.92, color=COLORS[c], label=LABELS[c], yerr=e,
               error_kw={'elinewidth': 0.7, 'capsize': 1.5})
    if ref is not None:
        ax.axhline(ref, color='#52514e', lw=0.8, ls=':')
    ax.set_xticks(x, groups, rotation=30, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def fig_inputs(run: Path, out: Path) -> None:
    tr = tf.read_samples(run / 'train.jsonl')
    te = tf.read_samples(run / 'test.jsonl')
    pick = [next(s for s in tr if s.family == f) for f in ('refcoco', 'flat', 'texture', 'grid')]
    pick += [next(s for s in te if s.family == f) for f in ('sweep_h', 'grid10', 'clutter', 'chart_ui')]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for ax, s in zip(axes.ravel(), pick):
        ax.imshow(Image.open(s.image))
        W, H = s.truth['canvas']
        b = s.truth['box']
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='#e34948', lw=1.2))
        ax.plot(*s.truth['center'], marker='+', color='#e34948', ms=10, mew=1.5)
        role = 'train' if s.split == 'train' else 'held-out test'
        ax.set_title(f'{role}: {s.family} ({s.fmt})\n"{s.label[:38]}"', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    ans = tf.answer_text(pick[0].fmt, pick[0].truth, pick[0].label)
    fig.text(0.01, 0.005, 'Training answer (first panel): ' + ans.replace('\n', ' ').replace('\t', '\\t '), fontsize=7, family='monospace')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / 'inputs.png', dpi=130)
    plt.close(fig)


def fig_disc(R: dict, conds: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, g, name in zip(axes, ('disc_x', 'disc_y'), ('x (horizontal sweep)', 'y (vertical sweep)')):
        for c in conds:
            cur = R['coord'][c][g]['curve']
            o = np.argsort(cur['truth'])
            ax.plot(np.array(cur['truth'])[o], np.array(cur['pred'])[o], color=COLORS[c], lw=1.4 if c != 'base' else 2.2,
                    label=LABELS[c], alpha=0.95)
        ax.plot([0, 1000], [0, 1000], color='#52514e', ls='--', lw=0.8, label='identity')
        ax.axhspan(480, 520, color='#d9d8d4', alpha=0.5, lw=0, label='480-520 plateau band')
        ax.set_xlabel('true coordinate (0-1000)')
        ax.set_ylabel('model output (0-1000)')
        ax.set_title(f'single red disc, {name}')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'disc_curves.png', dpi=130)
    plt.close(fig)


def fig_coord(R: dict, conds: list[str], out: Path) -> None:
    C = R['coord']
    groups = ['disc_x', 'disc_y', 'grid06', 'grid10', 'grid20', 'clutter', 'chart_ui', 'flat', 'refcoco']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    _bars(axes[0, 0], groups, conds, lambda c, g: C[c][g]['hit']['p'], lambda c, g: C[c][g]['hit']['ci'],
          'hit rate (centre inside target)', 'share')
    _bars(axes[0, 1], groups, conds, lambda c, g: C[c][g]['error']['mean'], lambda c, g: C[c][g]['error']['ci'],
          'mean centre error', '0-1000 units')
    _bars(axes[1, 0], groups, conds, lambda c, g: C[c][g]['round10'], None, 'share of coordinate values that are multiples of 10',
          'share', ref=0.1)
    _bars(axes[1, 1], groups, conds, lambda c, g: C[c][g]['plateau'], None,
          'share of outputs in [480, 520] when truth in [400, 610)', 'share', ref=0.19)
    axes[0, 0].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out / 'coord_metrics.png', dpi=130)
    plt.close(fig)


def fig_mechanism(R: dict, conds: list[str], out: Path) -> None:
    M = R['mechanism']
    conds = [c for c in conds if c in M]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    fams = ['sweep_h', 'sweep_v', 'refcoco']
    _bars(axes[0], fams, conds, lambda c, f: M[c]['clamp'][f]['slope_E_offset_in_range'], None,
          'second-digit gain\n(slope of E[d2] on offset)', 'slope', ref=1.0)
    _bars(axes[1], fams, conds, lambda c, f: M[c]['anova'][f]['interaction'], None,
          'image x first-digit share\nof second-digit logit variance', 'share')
    bl = ['P(9|4) x', 'P(0|5) x', 'P(9|4) y', 'P(0|5) y']
    get = {'P(9|4) x': ('blank_x', 'P9_after4'), 'P(0|5) x': ('blank_x', 'P0_after5'),
           'P(9|4) y': ('blank_y', 'P9_after4'), 'P(0|5) y': ('blank_y', 'P0_after5')}
    _bars(axes[2], bl, conds, lambda c, b: M[c]['blank'][get[b][0]][get[b][1]], None,
          'blank-canvas centre prior\n(second digit after forced first digit)', 'probability', ref=0.1)
    _bars(axes[3], ['sweep_h', 'sweep_v'], conds, lambda c, f: M[c]['oracle'][f]['model']['within1'], None,
          'second digit within +-1 (model)\ndotted: base ridge probe, q1, L32', 'share')
    for f, xx in (('sweep_h', 0), ('sweep_v', 1)):
        v = M['base']['oracle'][f]['L32']['ridge_coordinate']['within1']
        axes[3].plot([xx - 0.4, xx + 0.4], [v, v], color='#52514e', ls=':', lw=1.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=len(conds), loc='lower center', frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / 'mechanism.png', dpi=130)
    plt.close(fig)


def fig_dla(run: Path, R: dict, conds: list[str], out: Path) -> None:
    A = np.load(run / 'analysis.npz')
    conds = [c for c in conds if c != 'base' and f'{c}/sweep_h_q1_tuned' in A]
    rows = [('sweep_h_q1_tuned', 'image-driven logit toward true 2nd digit (q1)'),
            ('sweep_h_q0_tuned', 'image-driven logit toward true 1st digit (q0)'),
            ('sweep_h_prior_k4_d9', 'prior logit "9 after 4" (q1)')]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 9), sharex=True)
    layers = np.arange(1, 33)
    w = 0.8 / (2 * len(conds))
    for ax, (key, title) in zip(axes, rows):
        base = A[f'base/{key}']
        for j, c in enumerate(conds):
            d = A[f'{c}/{key}'] - base
            off = (j - (len(conds) - 1) / 2) * 2 * w
            ax.bar(layers + off - w / 2, d[:32], w, color=COLORS[c], alpha=0.45, hatch='//', ec=COLORS[c], lw=0)
            ax.bar(layers + off + w / 2, d[32:64], w, color=COLORS[c], label=LABELS[c])
        ax.axhline(0, color='#52514e', lw=0.6)
        ax.set_title(f'change vs base: {title}; solid = MLP, hatched = token mixer (x sweep, nats)', fontsize=9)
        ax.set_ylabel('delta logit (nats)')
    axes[-1].set_xlabel('layer (1-based)')
    axes[-1].set_xticks(layers)
    axes[0].legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out / 'dla_change.png', dpi=130)
    plt.close(fig)


def fig_change(R: dict, conds: list[str], out: Path) -> None:
    Ch, T = R.get('change', {}), R['training']
    conds = [c for c in conds if c in Ch]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    L = [f'L{l}' for l in df.RESID_LAYERS]
    for c in conds:
        for role, ls in (('q0', '--'), ('q1', '-')):
            v = [Ch[c]['sweep_h_resid_rel_change'][role][k] for k in L]
            axes[0].plot(df.RESID_LAYERS, v, ls, color=COLORS[c], marker='o', ms=3, label=f'{LABELS[c]} {role}')
        tr = Ch[c]['sweep_h_probe_transfer_q1']
        axes[1].plot(df.RESID_LAYERS, [tr[k]['base_on_cond'] for k in L], color=COLORS[c], marker='o', ms=3, label=LABELS[c])
        n = T[c]['delta_norm_per_layer']
        axes[2].bar([int(k) for k in n], list(n.values()), color=COLORS[c], alpha=0.75, label=LABELS[c])
    tr0 = Ch[conds[0]]['sweep_h_probe_transfer_q1']
    axes[1].plot(df.RESID_LAYERS, [tr0[k]['base_on_base'] for k in L], color=COLORS['base'], lw=2.2, label='base probe on base')
    axes[0].set(title='relative residual change at q0 (dashed) / q1 (solid)', xlabel='layer', ylabel='||h_ft - h_base|| / ||h_base||')
    axes[1].set(title='ridge probe fitted on base q1 residuals,\napplied to fine-tuned residuals', xlabel='layer', ylabel='MAE (0-1000 units)')
    axes[2].set(title='LoRA update norm per layer', xlabel='layer', ylabel='||scale * B A||_F (summed over modules)')
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'representation_change.png', dpi=130)
    plt.close(fig)


def fig_general(R: dict, conds: list[str], out: Path) -> None:
    G = R['general']
    conds = [c for c in conds if c in G]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    benches = ['pope', 'textvqa', 'mmvp', 'gqa', 'text']
    _bars(ax, benches, conds, lambda c, b: G[c][b]['acc']['p'] if b == 'text' else G[c][b]['acc']['mean'],
          lambda c, b: G[c][b]['acc']['ci'], 'general ability (200 questions per benchmark; text = 45 text-only)', 'accuracy')
    ax.legend(fontsize=7, ncol=3, loc='lower right')
    fig.tight_layout()
    fig.savefig(out / 'general.png', dpi=130)
    plt.close(fig)


def fig_training(R: dict, conds: list[str], out: Path) -> None:
    T = R['training']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    k = 10
    for c in conds:
        if c not in T:
            continue
        for ax, key in zip(axes, ('loss', 'digit_ce')):
            v = np.convolve(T[c][key], np.ones(k) / k, mode='valid')
            ax.plot(np.arange(len(v)) + k - 1, v, color=COLORS[c], label=LABELS[c])
    axes[0].set(title='training loss (all answer tokens, 10-step mean)', xlabel='step', ylabel='cross-entropy (nats)')
    axes[1].set(title='cross-entropy on digit tokens (10-step mean)', xlabel='step', ylabel='nats')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'training.png', dpi=130)
    plt.close(fig)


def fig_grid(run: Path, R: dict, conds: list[str], out: Path) -> None:
    from analysis_tools.position.coordinate_readout_run import read_jsonl
    te = {s.id: s for s in tf.read_samples(run / 'test.jsonl') if s.family == 'grid10'}
    show = [c for c in ('base', 'all_linear', 'mlp28_32') if c in conds]
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.8), squeeze=False)
    for ax, c in zip(axes[0], show):
        rows = {r['id']: r for r in read_jsonl(run / 'eval' / c / 'coord.jsonl')}
        e = np.array(next(iter(te.values())).meta['col_edges_units'])
        for v in e:
            ax.axvline(v, color='#d9d8d4', lw=0.6)
            ax.axhline(v, color='#d9d8d4', lw=0.6)
        xs = [(rows[i]['truth_units'][0], rows[i]['center'][0]) for i in te if rows[i]['valid']]
        ax.scatter(*zip(*xs), s=18, color=COLORS[c], label='point answer (x)')
        ax.plot([0, 1000], [0, 1000], '--', color='#52514e', lw=0.8, label='identity')
        ax.set(title=f'10x10 grid, {LABELS[c]}', xlabel='true x (0-1000)', ylabel='output x')
        ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'grid10.png', dpi=130)
    plt.close(fig)


def main_figures(run: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    R = json.loads((run / 'results.json').read_text())
    conds = R['conditions']
    fig_inputs(run, out)
    fig_disc(R, conds, out)
    fig_coord(R, conds, out)
    fig_grid(run, R, conds, out)
    if len(conds) > 1:
        fig_training(R, conds, out)
    if 'general' in R:
        fig_general(R, conds, out)
    if 'mechanism' in R and 'base' in R['mechanism']:
        fig_mechanism(R, conds, out)
        if R.get('change'):
            fig_dla(run, R, conds, out)
            fig_change(R, conds, out)
    print('figures written to', out, flush=True)
