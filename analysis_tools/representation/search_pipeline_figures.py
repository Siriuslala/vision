"""Experiment 2.7 round 4 figures (called by search_pipeline_analysis.main)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import read_jsonl

C = {'comparison': '#d9822b', 'detector': '#2a6fbb'}
GREY = '#888888'
LAB = {'comparison': 'comparison class', 'detector': 'detector class'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5})


def _bars(ax, table, conds, labels, key='reader', ref=None, title=''):
    x = np.arange(len(conds))
    w = 0.38
    for j, c in enumerate(('comparison', 'detector')):
        m = [table.get(k, {}).get(c, {}).get(key, np.nan) for k in conds]
        e = [table.get(k, {}).get(c, {}).get(key + '_ci', 0) for k in conds]
        ax.bar(x + (j - 0.5) * w, m, w * 0.92, yerr=e, color=C[c], label=LAB[c], capsize=2, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('reader attention on target')
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8)


def _hbars(ax, table, conds, labels, title=''):
    """Horizontal grouped bars of reader attention (mean, 95% interval) for both classes."""
    y = np.arange(len(conds))
    h = 0.38
    for j, c in enumerate(('comparison', 'detector')):
        m = [table.get(k, {}).get(c, {}).get('reader', np.nan) for k in conds]
        e = [table.get(k, {}).get(c, {}).get('reader_ci', 0) for k in conds]
        ax.barh(y + (j - 0.5) * h, m, h * 0.92, xerr=e, color=C[c], label=LAB[c], capsize=2, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('reader attention on target (mean of 6 reader heads)')
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='lower right')


def fig_examples(output: Path, round3: Path, figs: Path, items: dict):
    sel = [r for r in items.values() if r.get('selected')]
    pick = [next(r for r in sel if r['cls'] == 'comparison' and r['pair'] == 'closure'),
            next(r for r in sel if r['cls'] == 'detector' and r['pair'] == 'qo')]
    fig, axs = plt.subplots(2, 3, figsize=(8.4, 5.8))
    from analysis_tools.behavior.decomposition import read_jsonl as rj
    r3 = {it['id']: it for it in rj(round3 / 'items.jsonl')}
    for i, r in enumerate(pick):
        imgs = [Image.open(round3 / r3[r['id']]['path']), Image.open(round3 / r3[r['partner']]['path']),
                Image.open(output / r['same_type'])]
        for j, (im, t) in enumerate(zip(imgs, ('odd image', 'corrupted image', 'same-type image'))):
            ax = axs[i, j]
            ax.imshow(im)
            rr, cc = r['target']
            ax.add_patch(plt.Rectangle((cc * 64, rr * 64), 64, 64, fill=False, ec='#d9822b', ls='--', lw=1.5))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f'{LAB[r["cls"]]}: {t}', fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'examples.png', dpi=150)
    plt.close(fig)


def fig_vit(summary, recs, items, figs: Path):
    v = summary.get('vit', {})
    tab = summary.get('interventions', {})
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.9])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :])]
    ax = axs[0]
    for c in ('comparison', 'detector'):
        m = v.get('excl_by_class_mean', {}).get(c)
        if m:
            ax.plot(np.arange(1, 28), m, color=C[c], lw=2, label=LAB[c])
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel('ViT block'); ax.set_ylabel('exclusive neurons (zeta > 5)')
    ax.set_title('(a) exclusive ViT detectors of the target type', fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    blocks = [6, 9, 12, 18, 24]
    for c in ('comparison', 'detector'):
        d = v.get('excl_act', {}).get(c, {})
        ks = [k for k in blocks if str(k) in d or k in d]
        if not ks:
            continue
        val = np.array([d.get(str(k), d.get(k)) for k in ks])     # (k, 3)
        ax.plot(ks, val[:, 0], color=C[c], lw=2, marker='o', label=f'{LAB[c]}: target, odd image')
        ax.plot(ks, val[:, 1], color=C[c], lw=1.2, ls='--', marker='x', label=f'{LAB[c]}: same cell, corrupted')
    ax.set_xlabel('ViT block'); ax.set_ylabel('mean GELU output of the exclusive set')
    ax.set_title('(b) exclusive detectors fire only on the target', fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axs[2]
    ex = {}
    for r in recs:
        c = items[str(r['id'])]['cls']
        if c not in ex:
            ex[c] = r
    if 'detector' in ex:
        d = ex['detector']['x0_dnorm'].astype(float).reshape(12, 12) / np.median(ex['detector']['x0_norm'])
        im = ax.imshow(d, cmap='viridis')
        t = items[str(ex['detector']['id'])]['target']
        ax.add_patch(plt.Rectangle((t[1] * 2 - 0.5, t[0] * 2 - 0.5), 2, 2, fill=False, ec='white', ls='--'))
        plt.colorbar(im, ax=ax, fraction=0.046, label='|change of decoder input| / median |input|')
        ax.set_title('(c) decoder-input change per token\n(one detector-class pair)', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    conds = ['none', 'vit_excl', 'vit_rand', 'vit_up500', 'vit_up_all', 'vit_down500', 'vit_down_all',
             'vit_mlp_corr_1_9', 'vit_mlp_corr_10_18', 'vit_mlp_corr_19_27', 'vit_mlp_corr',
             'vit_attn_same_1_9', 'vit_attn_same_10_18', 'vit_attn_same_19_27', 'vit_attn_same', 'vit_attn_corr',
             'x0_all', 'x0_top1', 'x0_rest3', 'x0_neigh', 'corr_none', 'ins_x0_all', 'ins_x0_top1']
    labels = ['odd image, no intervention', 'ViT: exclusive neurons (zeta > 5) -> corrupted', 'ViT: random neurons (same count) -> corrupted',
              'ViT: 500 neurons/block more active on target type -> corr.', 'ViT: all neurons more active on target type -> corr.',
              'ViT: 500 neurons/block more active on majority type -> corr.', 'ViT: all neurons more active on majority type -> corr.',
              'ViT: MLP output, blocks 1-9 -> corr.', 'ViT: MLP output, blocks 10-18 -> corr.', 'ViT: MLP output, blocks 19-27 -> corr.',
              'ViT: MLP output, all blocks -> corr.', 'ViT: attention output, blocks 1-9 -> same-type', 'ViT: attention output, blocks 10-18 -> same-type',
              'ViT: attention output, blocks 19-27 -> same-type', 'ViT: attention output, all blocks -> same-type',
              'ViT: attention output, all blocks -> corr.', 'decoder input, 4 target tokens -> corr.',
              'decoder input, most-changed target token -> corr.', 'decoder input, other 3 target tokens -> corr.',
              'decoder input, all 140 other tokens -> corr.', 'corrupted image, no intervention',
              'corrupted image + odd decoder input at 4 target tokens', 'corrupted image + odd decoder input at the most-changed token']
    _hbars(axs[3], tab, conds, labels, title='(d) interventions at the target (ViT: the 16 target patches; decoder input: target tokens)')
    fig.tight_layout()
    fig.savefig(figs / 'vit.png', dpi=150)
    plt.close(fig)


def fig_reference(summary, figs: Path):
    tab = summary.get('interventions', {})
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    conds = ['none', 'ref_all', 'ref_fa', 'ref_dn', 'ref_dn1_7', 'ref_dn9_15', 'h6_corr', 'h6_same', 'h6_zero', 'h6_mean',
             'h6_blk', 'fa8top3_same', 'fa8top3_blk', 'noref_fa', 'noref_dn', 'noref', 'corr_none']
    labels = ['odd image, no intervention', 'all mixer outputs (FA 4/8/12 + DeltaNet 1-15) -> same-type',
              'FA 4/8/12 outputs -> same-type', 'DeltaNet 1-15 outputs -> same-type', 'DeltaNet 1-7 outputs -> same-type',
              'DeltaNet 9-15 outputs -> same-type', 'L8 H6 output -> corrupted', 'L8 H6 output -> same-type', 'L8 H6 output -> 0',
              'L8 H6 output -> mean over image tokens', 'L8 H6 attention restricted to own cell + text',
              'L8 H6/H9/H1 outputs -> same-type', 'L8 H6/H9/H1 attention restricted to own cell + text',
              'all FA 4/8/12 heads restricted to own cell + text', 'DeltaNet 1-15 memory reset (alpha = 0)',
              'both restrictions (no cross-cell reading at all)', 'corrupted image, no intervention']
    _hbars(axs[0], tab, conds, labels, title='(a) what the target tokens read from other cells (odd image; interventions at the 4 target tokens)')
    prof = summary.get('h6_distance', {})
    if prof:
        ks = sorted(int(k) for k in prof)[:30]
        axs[1].plot(ks, [prof[str(k)] if str(k) in prof else prof[k] for k in ks], color=C['comparison'], lw=2, marker='o', ms=3)
        axs[1].set_xlabel('raster distance from the target (cells)')
        axs[1].set_ylabel('L8 H6 attention mass per earlier cell')
        axs[1].set_title('(b) L8 H6 spreads over all earlier cells', fontsize=9)
    h = summary.get('h6', {})
    keys = [('cos_odd_corr', 'odd vs corr.'), ('cos_odd_same', 'odd vs same-type'), ('cos_odd_blk', 'odd vs own-only'),
            ('cos_earlier_uniform', 'earlier part vs\nuniform mean')]
    x = np.arange(len(keys))
    for j, c in enumerate(('comparison', 'detector')):
        m = [h.get(c, {}).get(k, (np.nan, 0))[0] for k, _ in keys]
        e = [h.get(c, {}).get(k, (np.nan, 0))[1] for k, _ in keys]
        axs[2].bar(x + (j - 0.5) * 0.38, m, 0.35, yerr=e, color=C[c], label=LAB[c], capsize=2)
    axs[2].set_xticks(x); axs[2].set_xticklabels([l for _, l in keys], fontsize=8, rotation=20, ha='right')
    axs[2].set_ylabel('cosine of L8 H6 output vectors'); axs[2].set_ylim(0, 1.05)
    axs[2].set_title('(c) L8 H6 output at the target', fontsize=9)
    axs[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'reference.png', dpi=150)
    plt.close(fig)


def fig_neurons(summary, figs: Path):
    ns = summary.get('neurons', {})
    if not ns:
        return
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for j, c in enumerate(('comparison', 'detector')):
        d = ns.get(c)
        if not d:
            continue
        ax = axs[j]
        ao, ar = np.array(d['per_neuron']['A_own']), np.array(d['per_neuron']['A_ref'])
        ax.scatter(ao, ar, s=10, color=C[c], alpha=0.7, label=f'{LAB[c]} (200 neurons)')
        lim = np.nanmax(np.abs(np.concatenate([ao, ar]))) * 1.05
        ax.plot([-lim, lim], [lim, -lim], color=GREY, ls='--', lw=1, label='A_ref = -A_own (subtraction)')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel('A_own = gate(odd) - gate(corrupted)')
        ax.set_ylabel('A_ref = gate(reference swapped) - gate(odd)')
        ax.set_title(f'({"ab"[j]}) {LAB[c]}: slope {d["slope_ref_on_own"]:.2f}', fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    ax = axs[2]
    groups = list(ns.get('comparison', ns.get('detector'))['dcon_own'].keys())
    y = np.arange(len(groups))
    for j, c in enumerate(('comparison', 'detector')):
        if c in ns:
            ax.barh(y + (j - 0.5) * 0.38, [ns[c]['dcon_own'][g] for g in groups], 0.35, color=C[c], label=LAB[c] + ': odd - corr.')
    if 'comparison' in ns:
        ax.scatter([ns['comparison']['dcon_ref'][g] for g in groups], y - 0.19, marker='x', color='k', s=18,
                   label='comparison: ref. swap - odd', zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(groups, fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('change of gate pre-activation (mean over neurons)')
    ax.set_title('(c) which residual components change the gate', fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / 'neurons.png', dpi=150)
    plt.close(fig)


def fig_readers(summary, figs: Path):
    rd = {**summary.get('redirect', {}), **summary.get('redirect_post', {})}
    vp = summary.get('value_probe', {})
    vl = summary.get('vl', {}).get('redirect', {})
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, tab, name in ((axs[0], rd, 'Qwen3.5-9B'), (axs[1], vl, 'Qwen3-VL-8B')):
        conds = [c for c in ('none', 'redir_both', 'r6_post', 'redir6_post', 'redir_all16', 'fa_all_digits', 'fa16_32_post',
                             'fa_all_post', 'redir_all_post') if c in tab]
        x = np.arange(len(conds))
        for j, (k, lab, col, hatch) in enumerate((('row_target', 'row = target row', C['detector'], ''),
                                                  ('row_redirect', 'row = redirect-cell row', C['detector'], '//'),
                                                  ('col_target', 'column = target column', C['comparison'], ''),
                                                  ('col_redirect', 'column = redirect-cell column', C['comparison'], '//'))):
            m = [np.mean([tab[c][cl][k] * tab[c][cl]['n'] for cl in tab[c]]) / np.mean([tab[c][cl]['n'] for cl in tab[c]]) for c in conds]
            ax.bar(x + (j - 1.5) * 0.2, m, 0.19, color=col, hatch=hatch, edgecolor='white', label=lab)
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('fraction of pairs (argmax digit)'); ax.set_ylim(0, 1.05)
        ax.set_title(f'{name}: reader value vectors of the target\nreplaced by those of another cell', fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    if vp:
        hs = sorted(vp, key=int)
        names = ['L16 H6', 'L16 H11', 'L16 H2', 'L20 H2', 'L20 H11', 'L20 H10']
        x = np.arange(len(hs))
        axs[2].bar(x - 0.2, [vp[h]['row'] for h in hs], 0.38, color=C['detector'], label='row of the cell')
        axs[2].bar(x + 0.2, [vp[h]['col'] for h in hs], 0.38, color=C['comparison'], label='column of the cell')
        axs[2].axhline(1 / 6, color=GREY, ls='--', lw=1, label='chance')
        axs[2].set_xticks(x); axs[2].set_xticklabels(names[:len(hs)], fontsize=8)
        axs[2].set_ylabel('held-out accuracy'); axs[2].set_ylim(0, 1.05)
        axs[2].set_title('Qwen3.5: cell position decoded from reader values', fontsize=9)
        axs[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / 'readers.png', dpi=150)
    plt.close(fig)


def fig_vl(summary, figs: Path):
    vl = summary.get('vl', {})
    p2 = vl.get('p2')
    if not p2:
        return
    layers = sorted(int(k[2:]) for k in p2 if k.startswith('pa') and k[2:].isdigit())
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={'width_ratios': [1.4, 1.4, 1]})
    for j, (pref, name) in enumerate((('pa', 'attention output -> corr.'), ('pm', 'MLP output -> corr.'),
                                      ('blk', 'target queries blocked from other cells'))):
        for c in ('comparison', 'detector'):
            d = [p2.get(f'{pref}{l}', {}).get(c, {}).get('drop', np.nan) for l in layers]
            ls = ['-', '--', ':'][j]
            axs[0 if pref != 'blk' else 1].plot(layers, d, color=C[c], ls=ls, lw=1.6, label=f'{LAB[c]}: {name}')
    for ax, t in ((axs[0], '(a) Qwen3-VL: replacing one layer output at the target'),
                  (axs[1], '(b) Qwen3-VL: blocking cross-cell attention of the target')):
        ax.set_xlabel('decoder layer'); ax.set_ylabel('drop of reader attention on target')
        ax.set_title(t, fontsize=9); ax.legend(frameon=False, fontsize=7)
    mk = vl.get('mark', {})
    if mk:
        groups = {'input': lambda n: n == 'input', 'DeepStack': lambda n: n.startswith('ds'),
                  'attention': lambda n: n.startswith('att'), 'MLP': lambda n: n.startswith('mlp')}
        x = np.arange(len(groups))
        for j, c in enumerate(('comparison', 'detector')):
            if c in mk:
                by = mk[c]['by_comp']
                axs[2].bar(x + (j - 0.5) * 0.38, [sum(v for n, v in by.items() if f(n)) for f in groups.values()], 0.35,
                           color=C[c], label=LAB[c])
        axs[2].set_xticks(x); axs[2].set_xticklabels(list(groups), fontsize=8)
        axs[2].set_ylabel('contribution to the reader logit advantage (nats)')
        axs[2].set_title('(c) Qwen3-VL: who writes the reader-key advantage', fontsize=9)
        axs[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'vl.png', dpi=150)
    plt.close(fig)


def fig_pipeline(figs: Path):
    """Schematic of the measured pipeline (no data; the doc text gives the evidence for every box)."""
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7.2); ax.axis('off')

    def box(x, y, w, h, text, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', fc=col, ec='#444444', lw=0.8, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.6)

    def arrow(x1, y1, x2, y2, col='#444444'):
        ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
    det, cmp_, sh = '#d6e4f5', '#f7e1c8', '#e8e8e8'
    ax.text(0.1, 6.95, 'detector-class target (e.g. a Q among Os)', color=C['detector'], fontsize=9, weight='bold')
    ax.text(0.1, 3.65, 'comparison-class target (e.g. an O among Qs)', color=C['comparison'], fontsize=9, weight='bold')
    box(0.1, 5.3, 2.3, 1.4, 'ViT MLP: rectified detector\nneurons fire only on the\ntail patch of the target\n(zero on majority entities)', det)
    box(2.8, 5.3, 2.3, 1.4, 'merger: the target merged\ntoken carries the detectors\'\nwritten vector; majority\ntokens do not', det)
    box(5.5, 5.3, 3.0, 1.4, 'decoder MLP 1-15 at the target\ntokens read this vector from the\ntarget\'s own residual (no other\ncell needed) and write along the\nreader key direction', det)
    box(0.1, 2.0, 2.3, 1.4, 'ViT: the target lacks the\nfeature; no neuron fires only\non it; its merged token\ndiffers by a missing vector', cmp_)
    box(2.8, 2.0, 2.6, 1.4, 'decoder L4/L8/L12 attention\n(L8 H6 largest) and DeltaNet\n9-15: the target query reads\nthe earlier cells -> reference\nvector (majority content)', cmp_)
    box(5.8, 2.0, 2.7, 1.4, 'MLP 8-15 neurons at the target:\ngate = (own content) + (reference)\nfires only if own content lacks\nwhat the reference has;\nwrites along the reader key', cmp_)
    box(9.0, 3.6, 2.6, 1.6, 'L16 H6/H11/H2, L20 H2/H11/H10:\nnearly fixed query from the\nrow / column digit position;\nlargest q.k on the target\n-> attention on the target', sh)
    box(12.0, 3.6, 2.8, 1.6, 'reader OV copies the value of\nthe attended tokens (cell\ncolumn / row code built by\nDeltaNet accumulators) into\nthe answer position\n-> row and column digits', sh)
    for (x1, y1, x2, y2) in ((2.4, 6.0, 2.8, 6.0), (5.1, 6.0, 5.5, 6.0), (2.4, 2.7, 2.8, 2.7), (5.4, 2.7, 5.8, 2.7),
                             (8.5, 6.0, 9.0, 4.6), (8.5, 2.7, 9.0, 4.2), (11.6, 4.4, 12.0, 4.4)):
        arrow(x1, y1, x2, y2)
    arrow(1.25, 5.3, 1.25, 3.4, GREY)
    ax.text(1.35, 4.3, 'same ViT, same\nmerger; only the\ndetector differs', fontsize=7, color=GREY)
    box(2.8, 0.3, 5.7, 1.1, 'every earlier entity token: its value vectors (L8) and its DeltaNet writes form the reference;\n'
        'the same computation runs at every entity token, so the reference exists whether or not the target is odd', sh)
    arrow(4.0, 1.4, 4.0, 2.0, GREY)
    fig.savefig(figs / 'pipeline.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def make_all(output: Path, round3: Path, figs: Path, summary: dict, recs, rows, items):
    fig_examples(output, round3, figs, items)
    if recs:
        fig_vit(summary, recs, items, figs)
        fig_reference(summary, figs)
        fig_neurons(summary, figs)
    fig_readers(summary, figs)
    fig_vl(summary, figs)
    fig_pipeline(figs)
