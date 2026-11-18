"""Figures of experiment 2.7 round 3.  All numbers come from summary.json (search_circuit_analysis), which is computed
from the trace, knock and patch files (model activations under the stated interventions; nothing is fitted)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.representation import search_circuit_analysis as A

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
CLS_COLOR = {'detector': A.C_DET, 'comparison': A.C_CMP}
CLS_LABEL = {'detector': 'ViT-detector targets', 'comparison': 'comparison targets'}
GROUP_LABEL = ['decoder input\n(merger)', 'DeltaNet\n1-7', 'DeltaNet\n9-15', 'full attn\n4,8,12', 'MLP\n1-7', 'MLP\n8-15']


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_examples(items, output: Path, figs: Path):
    def find(**kw):
        return next(it for it in sorted(items.values(), key=lambda x: x['id'])
                    if all(it.get(k) == v for k, v in kw.items()))
    odd = find(set='atlas', pair_name='closure', level='subtle', variant='swap', role='odd')
    picks = [(odd, 'atlas: closed circle among open (odd)'),
             (items[odd['partner']], 'atlas: corrupted counterpart'),
             (find(set='prev', pair_name='qo', level='subtle', variant='swap', role='odd', n_prev=5),
              'prev: O among Q, 5 earlier entities'),
             (find(set='shift', pair_name='qo', role='corrupted', k_blank=12), 'shift: homogeneous, 12 blank cells')]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4))
    for ax, (it, title) in zip(axes, picks):
        ax.imshow(Image.open(output / it['path']))
        if it['role'] == 'odd' or it['set'] == 'atlas':
            r, c = it['target']
            ax.add_patch(plt.Rectangle((c * 64, r * 64), 64, 64, fill=False, ec=A.C_CMP, ls='--', lw=1.5))
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    _save(fig, figs / 'examples.png')


def fig_prior(summary, figs: Path):
    rp, bm = summary['reader_prior'], summary['baseline_mark']
    fig = plt.figure(figsize=(13, 3.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.35)
    maps = [('hom_map', 'homogeneous images'), ('shift_k6', 'first 6 cells blank'), ('shift_k12', 'first 12 cells blank')]
    for g, (k, title) in enumerate(maps):
        ax = fig.add_subplot(gs[0, g])
        M = np.asarray(rp[k] if k == 'hom_map' else rp[k]['map'], float).ravel()
        if k != 'hom_map':
            M[:int(k.split('k')[1])] = np.nan
        im = ax.imshow(M.reshape(6, 6), cmap='viridis', vmin=0, vmax=3)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(6), range(1, 7)); ax.set_yticks(range(6), range(1, 7))
        ax.set_xlabel('column'); ax.set_ylabel('row')
    cax = fig.add_axes([0.075, -0.06, 0.4, 0.035])
    fig.colorbar(im, cax=cax, orientation='horizontal').set_label('reader attention / uniform share (blank cells grey)')
    ax = fig.add_subplot(gs[0, 3])
    for ci, cls in enumerate(('detector', 'comparison')):
        for mi, (m, hatch) in enumerate((('baseline', '//'), ('mark', None))):
            x = np.arange(4) + (ci * 2 + mi - 1.5) * 0.2
            y = [bm[f'{cls}|{b}'][m] for b in range(4)]
            e = [bm[f'{cls}|{b}'][m + '_se'] * 1.96 for b in range(4)]
            ax.bar(x, y, 0.19, yerr=e, capsize=2, color=CLS_COLOR[cls] if m == 'mark' else 'white',
                   edgecolor=CLS_COLOR[cls], hatch=hatch, label=f'{CLS_LABEL[cls]}: {m}')
    ax.axhline(0, color='#888888', lw=0.8)
    ax.set_xticks(range(4), A.RBIN_LABEL, fontsize=8)
    ax.set_ylabel('target logit minus other cells (nats)')
    ax.set_title('L16 readers: position baseline and target mark', fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    _save(fig, figs / 'prior.png')


def fig_attribution(summary, figs: Path):
    att, bm = summary['attribution'], summary['baseline_mark']
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={'width_ratios': [1.1, 1.6, 1.1]})
    for ci, cls in enumerate(('detector', 'comparison')):
        vals = [A.group_attr(np.asarray(v['L16']), 16) for v in att.values() if v['class'] == cls]
        m = np.array([[d.get(g, 0) for g in A.GROUPS] for d in vals])
        x = np.arange(len(A.GROUPS)) + (ci - 0.5) * 0.38
        axes[0].bar(x, m.mean(0), 0.36, color=CLS_COLOR[cls], label=f'{CLS_LABEL[cls]} ({len(vals)} conditions)')
        for j in range(len(A.GROUPS)):
            axes[0].scatter(np.full(len(vals), x[j]), m[:, j], s=7, color='k', alpha=0.45, zorder=3, lw=0)
        layer = np.array([np.asarray(v['L16'], float) for v in att.values() if v['class'] == cls]).mean(0)
        axes[1].plot(range(len(layer)), layer, color=CLS_COLOR[cls], lw=1.8, marker='o', ms=3, label=CLS_LABEL[cls])
        for gi, (g, ls) in enumerate((('dn_mid', '-'), ('mlp_mid', '--'), ('input', ':'))):
            axes[2].plot(range(4), [bm[f'{cls}|{b}']['groups'][g] for b in range(4)], color=CLS_COLOR[cls], ls=ls,
                         marker='o', ms=3, label=f'{CLS_LABEL[cls]}: {["DeltaNet 9-15", "MLP 8-15", "decoder input"][gi]}')
    axes[0].axhline(0, color='#888888', lw=0.8)
    axes[0].set_xticks(range(len(A.GROUPS)), GROUP_LABEL, fontsize=7.5)
    axes[0].set_ylabel('share of the target mark (nats)')
    axes[0].set_title('component groups (dots: feature conditions)', fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    names = A.comp_names(16)
    axes[1].axhline(0, color='#888888', lw=0.8)
    axes[1].set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    axes[1].set_ylabel('share of the target mark (nats)')
    axes[1].set_title('single components (mean over conditions)', fontsize=9)
    axes[1].legend(fontsize=7, frameon=False)
    axes[2].set_xticks(range(4), A.RBIN_LABEL, fontsize=7.5)
    axes[2].set_ylabel('share of the target mark (nats)')
    axes[2].set_title('by raster position of the target', fontsize=9)
    axes[2].legend(fontsize=6, frameon=False)
    _save(fig, figs / 'attribution.png')


GLOBAL_KNOCKS = ['none', 'dn_off', 'dn_off_1_7', 'dn_off_9_15', 'fa_off', 'fa_off_4', 'fa_off_8', 'fa_off_12_16',
                 'both_off', 'prime_same', 'prime_other', 'prime_blank']
LOCAL = ['none', 'fa8_tgt', 'fa_tgt', 'fa_dst', 'dn17_tgt', 'dn_tgt', 'dn_dst']


def fig_knock(summary, figs: Path):
    kn = summary['knock']
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), gridspec_kw={'width_ratios': [1.3, 1.3, 0.9]}, sharey=True)
    panels = [(axes[0], 'early', GLOBAL_KNOCKS, 'targets in cells 1-6'), (axes[1], 'late', GLOBAL_KNOCKS, 'targets after cell 6'),
              (axes[2], 'late', LOCAL, 'local knockouts, targets after cell 6')]
    for ax, band, conds, title in panels:
        conds = [c for c in conds if any(f'atlas|{cl}|{band}|{c}' in kn for cl in CLS_COLOR)]
        for ci, cls in enumerate(('detector', 'comparison')):
            xs, ys, es = [], [], []
            for i, c in enumerate(conds):
                v = kn.get(f'atlas|{cls}|{band}|{c}')
                if v is None:
                    continue
                xs.append(i + (ci - 0.5) * 0.38); ys.append(v['reader_target']); es.append(1.96 * v['reader_target_se'])
            ax.bar(xs, ys, 0.36, yerr=es, capsize=2, color=CLS_COLOR[cls], label=CLS_LABEL[cls])
        ax.set_xticks(range(len(conds)), conds, rotation=45, ha='right', fontsize=7)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel('reader attention on target (mean of 6 heads)')
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, figs / 'knock.png')


def fig_start(summary, figs: Path):
    ss, src = summary['start_sources'], summary['dn_sources']
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    if 'fa8_own' in ss:
        axes[0].plot(range(1, 37), ss['fa8_own'], color=A.C_DET, marker='o', ms=3, label='own entity')
        axes[0].plot(range(1, 37), ss['fa8_earlier'], color=A.C_CMP, marker='o', ms=3, label='earlier entities')
        axes[0].set_xlabel('raster index of the entity (homogeneous images)')
        axes[0].set_ylabel('attention mass (head mean)')
        axes[0].set_title(f'full attention L8 (n = {ss["fa8_n"]} images)', fontsize=9)
        axes[0].legend(fontsize=7, frameon=False)
    H = np.asarray(ss['dn_rho_layer_mean'])
    im = axes[1].imshow(H, aspect='auto', cmap='viridis', extent=[0.5, 36.5, len(A.DN_LAYERS) - 0.5, -0.5])
    axes[1].set_yticks(range(len(A.DN_LAYERS)), A.DN_LAYERS, fontsize=6)
    axes[1].set_xlabel('raster index of the entity (homogeneous images)'); axes[1].set_ylabel('DeltaNet layer')
    axes[1].set_title(f'DeltaNet relative prediction error (n = {ss["dn_rho_n"]})', fontsize=9)
    fig.colorbar(im, ax=axes[1]).set_label('|e| / |v| (head mean)')
    heads = src['odd']['heads']
    parts = ['own cell', 'same row', 'previous row', 'older rows', 'text prefix']
    cols = ['#08306b', '#2171b5', '#6baed6', '#c6dbef', '#bbbbbb']
    left = np.zeros(len(heads))
    for pi, (pn, col) in enumerate(zip(parts, cols)):
        v = np.array([heads[h][pi] for h in heads])
        axes[2].barh(range(len(heads)), v, left=left, color=col, label=pn, edgecolor='white', lw=1)
        left += v
    axes[2].set_yticks(range(len(heads)), [f'L{h.split(":")[0]} H{h.split(":")[1]}' for h in heads])
    axes[2].set_xlabel('share of the delta-rule prediction at the target')
    axes[2].set_title('where the prediction at the target comes from', fontsize=9)
    axes[2].legend(fontsize=6.5, frameon=False, loc='lower right')
    _save(fig, figs / 'start.png')


def fig_modes(summary, figs: Path):
    md = summary['mag_dose']
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for col, (pair, unit) in enumerate((('qo', 'Q tail length (px)'), ('closure', 'gap (px)'))):
        for var, color in (('std', A.C_DET), ('swap', A.C_CMP)):
            for band, ls in ((0, '--'), (1, '-')):
                pts = sorted((float(k.split('|')[1]), v) for k, v in md.items()
                             if k.split('|')[0] == pair and k.split('|')[2] == var and int(k.split('|')[3]) == band)
                if not pts:
                    continue
                x = [p[0] for p in pts]
                lab = f'{"present" if var == "std" else "absent"}, {"cells 1-6" if band == 0 else "after cell 6"}'
                axes[col].errorbar(x, [p[1]['acc'] for p in pts], yerr=[[p[1]['acc'] - p[1]['lo'] for p in pts],
                                   [p[1]['hi'] - p[1]['acc'] for p in pts]], color=color, ls=ls, marker='o', ms=3,
                                   capsize=2, label=lab)
                axes[col + 2].plot(x, [p[1]['mark'] for p in pts], color=color, ls=ls, marker='o', ms=3, label=lab)
        axes[col].set_xlabel(unit); axes[col].set_ylim(-0.03, 1.03)
        axes[col].set_ylabel('accuracy (generation)')
        axes[col].set_title(f'{A.PAIR_LABEL[pair]}: behaviour', fontsize=9)
        axes[col + 2].set_xlabel(unit); axes[col + 2].set_ylabel('target mark at L16 readers (nats)')
        axes[col + 2].set_title(f'{A.PAIR_LABEL[pair]}: reader mark', fontsize=9)
        axes[col + 2].axhline(0, color='#888888', lw=0.8)
    axes[0].legend(fontsize=6.5, frameon=False)
    _save(fig, figs / 'modes.png')


def fig_patch(summary, figs: Path):
    pa = summary['patch']
    comps = ['embed'] + [f'{t}{l}' for l in range(1, 16) for t in ('mix', 'mlp')] + ['dn_mix_1_15', 'fa_mix_4_12', 'mlp_1_15']
    lab = ['input'] + [(('fa' if l % 4 == 0 else 'dn') + str(l)) if t == 'mix' else f'mlp{l}' for l in range(1, 16) for t in ('mix', 'mlp')] \
        + ['all DeltaNet 1-15', 'all FA 4,8,12', 'all MLP 1-15']
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for ci, cls in enumerate(('detector', 'comparison')):
        ys = [pa.get(f'{cls}|{c}', {}).get('d_reader', np.nan) for c in comps]
        ax.bar(np.arange(len(comps)) + (ci - 0.5) * 0.4, ys, 0.38, color=CLS_COLOR[cls], label=CLS_LABEL[cls])
    ax.axhline(0, color='#888888', lw=0.8)
    ax.set_xticks(range(len(comps)), lab, rotation=90, fontsize=7)
    ax.set_ylabel('change of reader attention on target')
    ax.set_title('noising one component at the target tokens (value from the corrupted image)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, figs / 'patch.png')


def make_all(items, beh, trace, knock, patch, summary, figs: Path, output: Path):
    fig_examples(items, output, figs)
    fig_prior(summary, figs)
    fig_attribution(summary, figs)
    fig_start(summary, figs)
    if beh:
        fig_modes(summary, figs)
    if knock:
        fig_knock(summary, figs)
    if patch:
        fig_patch(summary, figs)
    if summary.get('circuit'):
        fig_heads(summary, figs)
        fig_neurons(summary, figs)
        fig_diagram(summary, figs)


# ----------------------------------------------------------------------------- circuit figures

def fig_heads(summary, figs: Path):
    c = summary['circuit']
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.45)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    for ax, kind, title in ((axes[0], 'blk', 'block cross-cell reads'), (axes[1], 'pat', 'output := corrupted')):
        M = np.array([[c['drop_comparison'].get(f'fa{l}h{h}_{kind}', [np.nan])[0] for h in range(16)] for l in (4, 8, 12)])
        Md = np.array([[c['drop_detector'].get(f'fa{l}h{h}_{kind}', [np.nan])[0] for h in range(16)] for l in (4, 8, 12)])
        im = ax.imshow(np.vstack([M, Md]), cmap='viridis', vmin=0, vmax=0.12, aspect='auto')
        ax.set_yticks(range(6), ['L4 cmp', 'L8 cmp', 'L12 cmp', 'L4 det', 'L8 det', 'L12 det'])
        ax.set_xticks(range(16), range(16), fontsize=7); ax.set_xlabel('head')
        ax.set_title(f'FA heads at target: {title}', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8).set_label('drop of reader attention')
    D = np.array([[c['drop_comparison'].get(f'dn{l}h{h}_rst', [np.nan])[0] for h in range(32)] for l in (9, 10, 11, 13, 14, 15)])
    im = axes[2].imshow(D, cmap='viridis', vmin=0, vmax=max(0.03, np.nanmax(D)), aspect='auto')
    axes[2].set_yticks(range(6), [f'DN{l}' for l in (9, 10, 11, 13, 14, 15)])
    axes[2].set_xticks(range(0, 32, 2), range(0, 32, 2), fontsize=7); axes[2].set_xlabel('value head')
    axes[2].set_title('DeltaNet, target only: memory reset (comparison targets)', fontsize=9)
    fig.colorbar(im, ax=axes[2], shrink=0.8).set_label('drop of reader attention')
    _save(fig, figs / 'heads.png')


def fig_neurons(summary, figs: Path):
    c = summary['circuit']
    s = c['second']
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), gridspec_kw={'width_ratios': [1.4, 1.4, 1.0]})
    for ax, phase, title in ((axes[0], 'nec', 'necessity (odd image, target := corrupted)'),
                             (axes[1], 'suf', 'sufficiency (corrupted image, target := odd)')):
        labels, conds = [], []
        if phase == 'nec':
            conds = ['odd_none'] + [f'nec_nrn_{t}{k}' for k in (10, 50, 200) for t in ('top', 'rand')] + ['nec_fa8top_blk', 'nec_dntop_rst']
        else:
            conds = ['corr_none'] + [f'suf_nrn_{t}{k}' for k in (10, 50, 200) for t in ('top', 'rand')] + \
                ['suf_fa8top', 'suf_dntop', 'suf_heads', 'suf_mlp8_15', 'suf_res15']
        for ci, cls in enumerate(('detector', 'comparison')):
            y = [s.get(f'{cls}|{k}', {}).get('reader', np.nan) for k in conds]
            e = [1.96 * s.get(f'{cls}|{k}', {}).get('reader_se', 0) for k in conds]
            ax.bar(np.arange(len(conds)) + (ci - 0.5) * 0.38, y, 0.36, yerr=e, capsize=2, color=CLS_COLOR[cls], label=CLS_LABEL[cls])
        ax.set_xticks(range(len(conds)), conds, rotation=50, ha='right', fontsize=7)
        ax.set_ylabel('reader attention on target'); ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    C = np.array(c['neuron_corr'])
    order = np.argsort([0 if k == 'detector' else 1 for k in c['neuron_condition_class']], kind='stable')
    im = axes[2].imshow(C[np.ix_(order, order)], cmap='viridis', vmin=0, vmax=1)
    labs = [c['neuron_conditions'][i].replace('|subtle', '').replace('|salient', ' sal.') for i in order]
    axes[2].set_xticks(range(len(labs)), labs, rotation=90, fontsize=5.5); axes[2].set_yticks(range(len(labs)), labs, fontsize=5.5)
    axes[2].set_title('neuron attribution: correlation between conditions', fontsize=9)
    fig.colorbar(im, ax=axes[2], shrink=0.8).set_label('Pearson r (MLP 8-15, 98k neurons)')
    _save(fig, figs / 'neurons.png')


def fig_diagram(summary, figs: Path):
    """Schematic of the measured circuit; every number printed on it is a measured effect from summary.json
    (reader attention on the target, mean of the six reader heads; comparison-class targets unless stated)."""
    c = summary['circuit']
    s, dc = c['second'], c['drop_comparison']
    b = c['base_comparison']
    rq = c['reader_qk']
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.axis('off'); ax.set_xlim(0, 14); ax.set_ylim(0, 6.4)

    def box(x, y, w, h, text, col):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=col, ec='#333333', lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.6)

    def arrow(x0, y0, x1, y1, text='', col='#333333'):
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', color=col, lw=1.4))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, fontsize=6.8, ha='center', color=col)
    g = lambda k: s.get(k, {}).get('reader', float('nan'))
    box(0.1, 4.6, 2.3, 1.2, 'target token, ViT detector fires\n(present feature)', '#dbe9f6')
    box(0.1, 2.5, 2.3, 1.2, 'target token, no detector\n(absent feature)', '#fbe3cf')
    box(0.1, 0.3, 2.3, 1.2, 'raster-earlier entities\n(majority template)', '#f2f2f2')
    box(3.2, 0.9, 2.9, 1.6, 'FA L8 H6 (+H9; L12 H4, L4 H14)\ntarget query reads earlier entities\n'
        f'block H6 at target: {b:.2f}->{b - dc["fa8h6_blk"][0]:.2f}\nblock H6,H9,H1: {b:.2f}->{g("comparison|nec_fa8top_blk"):.2f}\n'
        f'(detector: {c["base_detector"]:.2f}->{g("detector|nec_fa8top_blk"):.2f})', '#fbe3cf')
    box(3.2, 3.4, 2.9, 0.9, 'DeltaNet 9-15 heads (distributed)\nL14 H2, L13 H5, L15 H17 ...\n'
        f'top-5 reset at target: {b:.2f}->{g("comparison|nec_dntop_rst"):.2f}', '#fbe3cf')
    box(7.0, 2.3, 2.8, 2.0, 'MLP 8-15 at target token\n(top 200 of 98,304 neurons;\nlargest in MLP 15, 11, 13)\n'
        f'necessity: {b:.2f}->{g("comparison|nec_nrn_top200"):.2f}\n'
        f'sufficiency: {g("comparison|corr_none"):.2f}->{g("comparison|suf_nrn_top200"):.2f}\n'
        f'all MLP 8-15: ->{g("comparison|suf_mlp8_15"):.2f}', '#e8e8e8')
    box(10.5, 2.3, 2.2, 2.0, 'readers L16 H6/H11/H2,\nL20 H2/H11/H10\nfixed query (cos {:.2f})\n'
        'key change . query:\n{:.1f} nats / L16 head'.format(rq['L16H6']['q_cos_across_items'],
                                                             rq['L16H11']['comparison']['dlogit']), '#e8e8e8')
    box(13.1, 2.9, 0.8, 0.8, 'answer', '#ffffff')
    arrow(2.4, 5.2, 7.0, 4.3, 'local: own content only (mixers at target not needed)', A.C_DET)
    arrow(2.4, 3.05, 7.0, 3.05, 'own content', A.C_CMP)
    arrow(2.4, 0.9, 3.2, 1.5, 'keys / values', '#666666')
    arrow(2.4, 1.2, 3.2, 3.5, 'state', '#666666')
    arrow(6.1, 1.7, 7.0, 2.6, 'template', A.C_CMP)
    arrow(6.1, 3.85, 7.0, 3.7, '', A.C_CMP)
    arrow(9.8, 3.3, 10.5, 3.3, 'key mark')
    arrow(12.7, 3.3, 13.1, 3.3)
    ax.text(0.1, 6.15, 'Measured circuit (reader attention on target; comparison-class targets, n = 80, unless stated)', fontsize=9)
    _save(fig, figs / 'circuit.png')
