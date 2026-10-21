"""Figures of experiment 3.7 round 2. Reads the run directory and its results.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from analysis_tools.position import vit_layout_r2 as v2
from analysis_tools.position import vit_layout_figures as vf
from analysis_tools.position.coordinate_readout import tokens_overlapping

SERIES = vf.SERIES
MODEL = {'q35': 'Qwen3.5-9B', 'q3vl': 'Qwen3-VL-8B'}
STIM_LABEL = {'disc_white': 'disc\nwhite', 'disc_checker': 'disc\nchecker', 'disc_noise': 'disc\nnoise',
              'disc_shuffled': 'disc\nshuffled', 'disc_scene': 'disc\nscene', 'obj_white': 'object\nwhite',
              'obj_shuffled': 'object\nshuffled', 'natural': 'natural', 'natural_vflip': 'natural\nv-flip'}
save = vf.save


def bar_group(ax, labels, series: dict, colors, ylabel, width=0.13):
    xs = np.arange(len(labels))
    k = len(series)
    for i, (name, vals) in enumerate(series.items()):
        m = [v['mean'] if v else np.nan for v in vals]
        lo = [v['mean'] - v['ci'][0] if v else 0 for v in vals]
        hi = [v['ci'][1] - v['mean'] if v else 0 for v in vals]
        ax.bar(xs + (i - (k - 1) / 2) * width, m, width, yerr=[lo, hi], color=colors[i], label=name,
               error_kw={'lw': 0.7, 'capsize': 1.5})
    ax.set_xticks(xs, labels, fontsize=7)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel(ylabel)


# ----------------------------------------------------------------------------- inputs

def fig_inputs(out: Path, run: Path) -> None:
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    plans = json.loads((run / 'ladder' / 'plans.json').read_text())
    cid = next(k for k, p in plans.items() if p and p['stimuli']['natural_vflip'] and p['stimuli']['natural'].get('variants'))
    case, plan = cases[cid], plans[cid]
    fig, ax = plt.subplots(2, 5, figsize=(15, 5.6))
    ax = ax.ravel()
    H, W = plan['grid']
    for i, s in enumerate(v2.STIMULI):
        img, truth, prompt = v2.render(case, s, seed=20261016)
        ax[i].imshow(img)
        for toks, color, ls in ((plan['stimuli'][s]['target'], '#eb6834', '-'),
                                (plan['stimuli'][s]['primary']['region'], '#2a78d6', '--')):
            rs = [t // W for t in toks]
            cs = [t % W for t in toks]
            ax[i].add_patch(Rectangle((min(cs) * 32, min(rs) * 32), (max(cs) - min(cs) + 1) * 32,
                                      (max(rs) - min(rs) + 1) * 32, fill=False, ec=color, lw=1.6, ls=ls))
        ax[i].set_title(s, fontsize=9)
        ax[i].axis('off')
    ax[9].axis('off')
    ax[9].legend(handles=[Patch(fc='none', ec='#eb6834', label='target tokens T'),
                          Patch(fc='none', ec='#2a78d6', ls='--', label="displaced region T'")], loc='center', fontsize=9)
    ax[9].text(0.5, 0.15, f'expression: "{case["expression"]}"', ha='center', fontsize=8, transform=ax[9].transAxes)
    save(fig, out, 'inputs.png')


# ----------------------------------------------------------------------------- ladder

LADDER_CONDS = [('rope_swap', 'RoPE swap'), ('vit_swap', 'PE+RoPE swap (vit_swap)'),
                ('vit_swap_target', 'vit_swap, target tokens only'), ('vit_swap_context', 'vit_swap, context tokens only'),
                ('content_swap', 'content swap (slot)'), ('move', 'real move')]


def fig_ladder(out: Path, res: dict) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(14, 7.5))
    for a, tag in zip(ax, ('q35', 'q3vl')):
        m = res[tag]['ladder']['stimuli']
        stims = [s for s in v2.STIMULI if s in m]
        series = {lab: [m[s]['conditions'].get(c) for s in stims] for c, lab in LADDER_CONDS}
        bar_group(a, [STIM_LABEL[s] for s in stims], series, SERIES, 'displacement weight w')
        a.set_title(f'{MODEL[tag]}: output displacement weight per stimulus (bars: mean over cases; whiskers: bootstrap 95% CI)', fontsize=9)
        a.set_ylim(-0.15, 1.25)
    ax[0].legend(ncol=6, fontsize=7, loc='upper center')
    save(fig, out, 'ladder_w.png')


def fig_ladder_probe(out: Path, res: dict) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(14, 7))
    for row, tag in enumerate(('q35', 'q3vl')):
        lad = res[tag]['ladder']
        stims = [s for s in v2.STIMULI if s in lad['probe']]
        pr, st = lad['probe'], lad['stimuli']
        bar_group(ax[row, 0], [STIM_LABEL[s] for s in stims],
                  {'probe: target tokens encode T\' (w_probe, vit_swap)': [pr[s]['vit_swap']['w_probe_T'] for s in stims],
                   'output follows (w, vit_swap_target)': [st[s]['conditions']['vit_swap_target'] for s in stims]},
                  [SERIES[2], SERIES[0]], 'displacement weight', width=0.35)
        ax[row, 0].set_title(f'{MODEL[tag]}: target tokens', fontsize=9)
        bar_group(ax[row, 1], [STIM_LABEL[s] for s in stims],
                  {'probe: context encodes target at T\' (w_ctx, vit_swap)': [pr[s]['vit_swap']['w_ctx'] for s in stims],
                   'output follows (w, vit_swap_context)': [st[s]['conditions']['vit_swap_context'] for s in stims]},
                  [SERIES[3], SERIES[1]], 'displacement weight', width=0.35)
        r2 = [pr[s]['ctx_base_r2_x'] for s in stims]
        ax[row, 1].plot(np.arange(len(stims)), r2, 'kD', ms=4, label='context probe R² of target x (base images)')
        ax[row, 1].set_title(f'{MODEL[tag]}: context tokens', fontsize=9)
        for a in ax[row]:
            a.set_ylim(-0.3, 1.1)
            a.legend(fontsize=7, loc='upper right')
    save(fig, out, 'ladder_probe.png')


def fig_texture(out: Path, res: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    t = res['coord35_texture']
    labels = ['low', 'middle', 'high']
    bar_group(ax[0], labels, {'RoPE swap': t['rope']['tertiles'], 'PE+RoPE swap': t['both']['tertiles'],
                              'content swap': t['content']['tertiles']}, SERIES[:3], 'w', width=0.25)
    ax[0].set_xlabel("texture tertile of T' (gray-level std inside tokens)")
    ax[0].set_title('3.5 RefCOCO swaps (Qwen3.5) by texture of the displaced region', fontsize=9)
    ax[0].legend(fontsize=7)
    tags = [g for g in ('q35', 'q3vl') if res[g]['ladder']]
    groups = [f'{MODEL[g]}\n{c}' for g in tags for c in ('RoPE swap', 'vit_swap', 'content swap')]
    vals = {v: [] for v in ('primary', 'homog', 'textured')}
    for g in tags:
        cs = res[g]['ladder']['stimuli']['natural']['conditions']
        for c in ('rope_swap', 'vit_swap', 'content_swap'):
            vals['primary'].append(cs.get(c))
            vals['homog'].append(cs.get(f'{c}_homog'))
            vals['textured'].append(cs.get(f'{c}_textured'))
    bar_group(ax[1], groups, {"3.5 rule T'": vals['primary'], "most homogeneous T'": vals['homog'],
                              "most textured T'": vals['textured']}, [SERIES[7], SERIES[2], SERIES[1]], 'w', width=0.25)
    ax[1].set_title("natural images: swaps into homogeneous vs textured regions", fontsize=9)
    ax[1].legend(fontsize=7)
    save(fig, out, 'texture.png')


# ----------------------------------------------------------------------------- edits

EDIT_LABELS = ['pe_zero', 'pe_mean', 'pe_shuffle', 'pe_scale', 'rope_zero', 'rope_shuffle', 'rope_zero_pe_zero']


def fig_edits(out: Path, res: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    for a, stim in zip(ax, ('natural', 'obj_white')):
        series = {}
        for tag in ('q35', 'q3vl'):
            e = res[tag]['edits'].get(stim, {})
            series[MODEL[tag]] = [e.get(c, {}).get('delta_iou') for c in EDIT_LABELS]
        bar_group(a, [c.replace('_', '\n', 1) for c in EDIT_LABELS], series, SERIES[:2], 'mean IoU change vs base', width=0.35)
        a.set_title(f'{stim}: whole-image position edits (bbox task, 300 RefCOCO test cases)', fontsize=9)
        a.legend(fontsize=7)
    save(fig, out, 'edits.png')


# ----------------------------------------------------------------------------- subspace

def fig_subspace(out: Path, res: dict) -> None:
    s = res['q35']['subspace']
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    for name, color in (('nat_train_base', SERIES[0]), ('syn_layout_base', SERIES[1]), ('syn_disc_base', SERIES[2]),
                        ('nat_train_both_zero', SERIES[7])):
        sp = s['families'][name]['spectrum'][:15]
        ax[0].plot(np.arange(1, len(sp) + 1), sp, marker='o', ms=3, color=color, label=f"{name} (PR {s['families'][name]['pr']:.1f})")
    ax[0].plot(np.arange(1, 16), s['clean_spectrum']['nat_train'][:15], ls='--', color=SERIES[0], label='nat_train, outlier slots removed')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('principal component of the slot means')
    ax[0].set_ylabel('variance share')
    ax[0].legend(fontsize=6)
    ax[0].set_title('spectrum of the slot-mean matrix M (merger, 16x16 tokens)', fontsize=9)
    pairs = ['nat_halves', 'nat_train|nat_test', 'nat_train|nat_vflip', 'nat_train|nat_shuffled', 'nat_train|syn_layout',
             'nat_train|syn_disc']
    ks = ['k2', 'k5', 'k10', 'k20']
    xs = np.arange(len(pairs))
    for i, k in enumerate(ks):
        ax[1].bar(xs + (i - 1.5) * 0.2, [s['overlap'][k][p] for p in pairs], 0.2, color=SERIES[i], label=f'k = {k[1:]}')
    ax[1].plot(xs, [s['overlap']['k20']['random']] * len(xs), 'k_', ms=12, label='random subspace (k = 20)')
    ax[1].set_xticks(xs, [p.replace('|', '\nvs ') for p in pairs], fontsize=6)
    ax[1].set_ylabel('subspace overlap O_k')
    ax[1].legend(fontsize=6)
    ax[1].set_title('overlap of top-k slot-mean subspaces', fontsize=9)
    for name, color in (('nat_train_clean', SERIES[0]), ('syn_layout', SERIES[1])):
        sp = s['spatial'][name]
        p = np.array(sp['power']).mean(0)
        ax[2].plot(sp['freq'], p, marker='o', ms=3, color=color, label=f'{name}: mean of top-10 PCs')
    for j, f in enumerate(s['rope']['cycles_per_token']):
        if 0.05 < f <= 0.5:
            ax[2].axvline(f, color='#7a5195', lw=0.8, ls=':', label='ViT RoPE frequencies' if j == 0 else None)
    ax[2].set_xlabel('spatial frequency (cycles per merged token)')
    ax[2].set_ylabel('share of PC power')
    ax[2].legend(fontsize=6)
    ax[2].set_title('spatial spectrum of the position PCs', fontsize=9)
    save(fig, out, 'subspace.png')


def shaded_cloud(ax, P, c, cmap, elev, azim, title=None):
    lo = P.min(0)
    z0 = lo[2] - 0.25 * (P[:, 2].max() - lo[2])
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    y0, y1 = P[:, 1].min(), P[:, 1].max()
    pad = 0.1 * max(x1 - x0, y1 - y0)
    floor = [[(x0 - pad, y0 - pad, z0), (x1 + pad, y0 - pad, z0), (x1 + pad, y1 + pad, z0), (x0 - pad, y1 + pad, z0)]]
    ax.add_collection3d(Poly3DCollection(floor, facecolor=(0.97, 0.97, 0.97, 0.6), edgecolor='0.35', lw=0.6))
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), z0), s=14, c='0.6', alpha=0.25, depthshade=False, edgecolors='none')
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=22, c=c, cmap=cmap, depthshade=True, edgecolors='k', linewidths=0.15)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.8), zoom=1.25)
    if title:
        ax.set_title(title, fontsize=9)
    return sc


def fig_manifold(out: Path, run: Path) -> None:
    st = torch.load(run / 'capture' / 'q35_slot_stats.pt', weights_only=False)
    from analysis_tools.position.vit_layout_r2_analysis import centers
    xy = centers((16, 16))
    fig = plt.figure(figsize=(16, 8))
    cols = [('nat_train_base', 0, 'x'), ('nat_train_base', 1, 'y'), ('syn_layout_base', 0, 'x'), ('syn_layout_base', 1, 'y')]
    for j, (fam, axis, lab) in enumerate(cols):
        d = st[fam]
        dev = d['means_norm']
        keep = (dev <= 3 * dev.median()).numpy()
        Z = d['pcs'][:, :20].numpy()[keep]                 # top-20 PC scores of the slot means, clean slots
        Z = Z - Z.mean(0)
        u, sv, _ = np.linalg.svd(Z, full_matrices=False)
        P = u[:, :3] * sv[:3]                             # 3D PCA of the clean slot means
        c = xy[keep, axis]
        col_axes = []
        for i, (elev, azim, view) in enumerate(((22, -55, 'side view'), (88, -90, 'top view'))):
            ax = fig.add_subplot(2, 4, i * 4 + j + 1, projection='3d')
            sc = shaded_cloud(ax, P, c, 'viridis', elev, azim, f'{fam.replace("_base", "")}: colour = token {lab}\n{view}')
            col_axes.append(ax)
    fig.suptitle('Qwen3.5-9B merger output: slot means of the 16x16 token positions, 3D PCA (high-norm outlier slots removed)', fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.93, top=0.9, bottom=0.02, wspace=0.05, hspace=0.12)
    cb = fig.colorbar(sc, cax=fig.add_axes([0.945, 0.25, 0.012, 0.5]))
    cb.set_label('token x or y named in the panel title (0-1000)', fontsize=8)
    for a in fig.axes:
        if a.name == '3d':
            bb = a.get_position()
            fig.add_artist(Rectangle((bb.x0, bb.y0), bb.width, bb.height, transform=fig.transFigure, fill=False, ec='0.6', lw=0.6))
    fig.savefig(out / 'manifold.png', dpi=150)
    plt.close(fig)
    print('figure manifold.png')


def fig_content_probe(out: Path, res: dict) -> None:
    pr = res['q35']['subspace']['probes']
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    rows = (np.arange(16) + .5) / 16 * 1000
    for a, (key, title) in zip(ax, (('nat_train_base->nat_test_base', 'full ViT, upright test'),
                                    ('nat_train_both_zero->nat_test_both_zero', 'position-free ViT, upright test'),
                                    ('nat_train_both_zero->nat_vflip_both_zero', 'position-free ViT, flipped test'))):
        mm = np.array(pr[key]['mean_map']).reshape(16, 16, 2)
        a.plot(rows, mm[..., 1].mean(1), 'o-', color=SERIES[0], ms=3, label='predicted y (mean over row)')
        a.plot(rows, mm[..., 0].mean(0), 's-', color=SERIES[1], ms=3, label='predicted x (mean over column)')
        a.plot([0, 1000], [0, 1000], 'k:', lw=0.8, label='identity')
        a.set_xlabel('true token coordinate (0-1000)')
        a.set_ylabel('probe prediction')
        a.set_title(f"{title}\nMAE x {pr[key]['mae_x']:.0f}, y {pr[key]['mae_y']:.0f}; corr y {pr[key]['corr_y']:.2f}", fontsize=8)
        a.set_ylim(-50, 1050)
    ax[0].legend(fontsize=7)
    save(fig, out, 'content_probe.png')


def fig_erase(out: Path, res: dict) -> None:
    e = res['q35']['erase']
    conds = ['none', 'random_255', 'xy_nat', 'onehot_nat', 'onehot_syn', 'onehot_mix']
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for a, kind in zip(ax, ('refcoco', 'disc')):
        m = e[kind]
        p = [m[c]['hit']['p'] for c in conds]
        lo = [m[c]['hit']['p'] - m[c]['hit']['ci'][0] for c in conds]
        hi = [m[c]['hit']['ci'][1] - m[c]['hit']['p'] for c in conds]
        a.bar(np.arange(len(conds)), p, yerr=[lo, hi], color=[SERIES[7], SERIES[7], SERIES[3], SERIES[0], SERIES[1], SERIES[2]],
              error_kw={'lw': 0.7, 'capsize': 2})
        a.set_xticks(np.arange(len(conds)), conds, fontsize=7)
        a.set_ylabel('hit rate (Wilson 95% CI)')
        for i, c in enumerate(conds):
            a.text(i, 0.02, f"{m[c]['error']['mean']:.0f}", ha='center', fontsize=7, color='w')
        a.set_title(f"{kind} ({m['none']['hit']['n']} images): point hit after merger erasure (number: mean center error)", fontsize=8)
        a.set_ylim(0, 1.05)
    save(fig, out, 'erase.png')


def fig_l20(out: Path, res: dict) -> None:
    l = res['q35']['l20']
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    heads = np.arange(16)
    for kind, color in (('refcoco', SERIES[0]), ('disc', SERIES[1])):
        for cond, ls in (('none', '-'), ('onehot_mix', '--')):
            if kind in l.get(cond, {}):
                h = l[cond][kind]['heads']
                ax[0].plot(heads, [h[str(i)]['logit_var_share_pos'] for i in heads], ls=ls, marker='o', ms=3, color=color,
                           label=f'{kind}, {"no erasure" if cond == "none" else "merger erased"}')
    ax[0].axvline(11, color='0.6', lw=0.8)
    ax[0].set_xlabel('layer-20 head'); ax[0].set_ylabel('share of logit variance from position component')
    ax[0].legend(fontsize=6); ax[0].set_title('positional share of attention logits over image tokens', fontsize=9)
    labels, vals = [], {'full key': [], 'position component only': [], 'content component only': []}
    for kind in ('refcoco', 'disc'):
        h = l['none'][kind]['heads']['11']
        labels.append(kind)
        vals['full key'].append(h['enrich_full']); vals['position component only'].append(h['enrich_pos'])
        vals['content component only'].append(h['enrich_con'])
    xs = np.arange(2)
    for i, (k, v) in enumerate(vals.items()):
        ax[1].bar(xs + (i - 1) * 0.25, v, 0.25, color=SERIES[i], label=k)
    ax[1].set_xticks(xs, labels); ax[1].set_ylabel('target enrichment of softmax')
    ax[1].set_yscale('log'); ax[1].legend(fontsize=7); ax[1].set_title('L20 H11: which key component selects the target', fontsize=9)
    groups, series = [], {'o from position values': [], 'o from content values': [], 'full o': []}
    for cond in ('none', 'onehot_mix'):
        for kind in ('refcoco', 'disc'):
            h = l[cond][kind]['heads']['11']
            groups.append(f'{kind}\n{"no erasure" if cond == "none" else "erased"}')
            series['o from position values'].append(h['r2_o_pos'][0]); series['o from content values'].append(h['r2_o_con'][0])
            series['full o'].append(h['r2_o_all'][0])
    xs = np.arange(len(groups))
    for i, (k, v) in enumerate(series.items()):
        ax[2].bar(xs + (i - 1) * 0.25, v, 0.25, color=SERIES[i + 3], label=k)
    ax[2].set_xticks(xs, groups, fontsize=7); ax[2].set_ylabel('CV R² of target x')
    ax[2].set_ylim(0, 1.05); ax[2].legend(fontsize=7); ax[2].set_title('L20 H11 output: target x readable from', fontsize=9)
    save(fig, out, 'l20.png')


def fig_smalldisc(out: Path, res: dict) -> None:
    m = res['q35']['smalldisc']
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ds = [16, 24, 32, 64]
    for a, bg in zip(ax, ('white', 'noise')):
        for i, c in enumerate(('base', 'pe_zero', 'pe_shuffle', 'pe_scale')):
            e = [m[f'{d}|{bg}|{c}']['error'] for d in ds]
            a.errorbar(ds, [x['mean'] for x in e], yerr=[[x['mean'] - x['ci'][0] for x in e], [x['ci'][1] - x['mean'] for x in e]],
                       marker='o', ms=4, color=SERIES[i], label=c if c != 'pe_scale' else 'pe x2.76', capsize=2)
        a.set_xscale('log', base=2); a.set_xticks(ds, [str(d) for d in ds])
        a.set_xlabel('disc diameter (pixels; one merged token = 32)'); a.set_ylabel('center error (0-1000)')
        a.set_title(f'{bg} background, 36 positions per point', fontsize=9)
        a.legend(fontsize=7)
    save(fig, out, 'smalldisc.png')


def fig_layout_q3vl(out: Path, res: dict, round1: Path) -> None:
    r1 = json.loads((round1 / 'results.json').read_text())
    lay = {'base': r1['q3vl']['layout']['base'], **res['q3vl'].get('layout', {})}
    vf.fig_layout(out, {'q3vl': {'layout': lay}})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--round1', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    res = json.loads((args.run / 'results.json').read_text())
    fig_inputs(args.figures, args.run)
    fig_ladder(args.figures, res)
    fig_ladder_probe(args.figures, res)
    fig_texture(args.figures, res)
    fig_edits(args.figures, res)
    fig_subspace(args.figures, res)
    fig_manifold(args.figures, args.run)
    fig_content_probe(args.figures, res)
    fig_erase(args.figures, res)
    fig_l20(args.figures, res)
    fig_smalldisc(args.figures, res)
    if res['q3vl'].get('layout'):
        fig_layout_q3vl(args.figures, res, args.round1)


if __name__ == '__main__':
    main()
