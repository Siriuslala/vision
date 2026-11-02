"""Figures of experiment 3.11 round 2 (coordinate readout on natural images).

Reads summary.json / salient_rows.json (coordinate_flow_natural_analysis.py), the round-2 records and the
2.5e readout maps; writes PNG files into docs/analysis/position/figures/coordinate_flow_natural/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
from PIL import Image

from analysis_tools.position import coordinate_flow_natural_analysis as A
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.representation import sees_right as sr

FULL_LAYERS = cr.FULL_LAYERS
LI = A.LI
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#008300', '#e34948']
GREY = '#8c8b87'
DIVERGING = LinearSegmentedColormap.from_list('bgo', ['#2a78d6', '#f0efec', '#eb6834'])
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 10})
GROUP_LABEL = {'rc': 'RefCOCO', 'ms_small': 'synthetic 4 shapes, 64 px', 'ms_large': 'synthetic 3 shapes, 160 px'}


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def head_list(spec: list[str]) -> list[tuple[int, int]]:
    return [tuple(int(t) for t in s.split('.')) for s in spec]


# ----------------------------------------------------------------------------- Q2: salient part / edges

def overlay(ax, img, p, grid, box, mask, center, peak, model, title, vmax):
    H, W = grid
    ax.imshow(img)
    m = p.reshape(H, W)
    im = ax.imshow(m, cmap='viridis', alpha=0.6, extent=(0, W * 32, H * 32, 0), vmin=0, vmax=vmax, interpolation='nearest')
    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec='white', lw=1.4))
    if mask is not None:
        mk = np.array(mask + [mask[0]])
        ax.plot(mk[:, 0], mk[:, 1], color='#ff4fd8', lw=0.9)
    ax.plot(*center, marker='+', color='white', ms=12, mew=2)
    ax.plot(*peak, marker='x', color='#e34948', ms=9, mew=2.2)
    if model is not None:
        ax.plot(*model, marker='o', mfc='none', mec='#eda100', ms=9, mew=2)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def fig_gallery(run: Path, source: Path, out: Path, S: dict, rows: list[dict], name='salient_gallery.png',
                flip: bool = False) -> None:
    """RefCOCO examples: attention of H11 (y and x query) and of the edge-head groups (x query)."""
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    cls = S['edges']['classes']
    mn, mx = head_list(cls['min_edge']), head_list(cls['max_edge'])
    rng = np.random.default_rng(3)
    by = {k: [r for r in rows if r['category'] == k] for k in ('person', 'animal', 'other')}
    pick = []
    for k, n in (('person', 4), ('animal', 2), ('other', 2)):
        rs = sorted(by[k], key=lambda r: r['area_frac'])
        qs = np.linspace(0.15, 0.9, n)
        pick += [rs[int(q * (len(rs) - 1))] for q in qs]
    fig, axes = plt.subplots(len(pick), 4, figsize=(11, 2.75 * len(pick)))
    for i, r in enumerate(pick):
        key = r['id'].replace('rc_point_', 'rc')
        c = cases[key]
        img = sr.load_image(c)
        a = np.load(source / 'readout' / 'maps' / f"{r['id']}.npz")['attn'].astype(np.float32)[:, :, :2]
        p = a / a.sum(-1, keepdims=True)
        grid = tuple(r['grid'])
        center = np.array(c['truth']['center'])
        model = None if r['model_u'] is None else np.array(r['model_u']) * np.array(r['canvas']) / 1000
        maps = [(p[LI[20], 11, 1], 'L20 H11, y query'), (p[LI[20], 11, 0], 'L20 H11, x query'),
                (np.mean([p[LI[l], h, 0] for l, h in mn], 0), 'min-edge heads, x query'),
                (np.mean([p[LI[l], h, 0] for l, h in mx], 0), 'max-edge heads, x query')]
        for j, (m, t) in enumerate(maps):
            pk = A.local_peak(m, grid)[1]
            im = overlay(axes[i, j], img, m, grid, c['truth']['box'], c['mask'], center, pk, model,
                         f'{t}\n"{c["expression"][:28]}" ({r["category"]}, area {r["area_frac"]:.2f})' if j == 0 else t,
                         vmax=float(m.max()))
            if j == 3:
                cb = fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.02)
                cb.set_label('share of image attn.\n(per token)', fontsize=7)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color='white', lw=1.4, label='target box', markeredgecolor='k'),
               Line2D([], [], color='#ff4fd8', lw=0.9, label='COCO mask'),
               Line2D([], [], color='k', marker='+', ls='', ms=10, label='box centre (white)'),
               Line2D([], [], color='#e34948', marker='x', ls='', ms=9, label='attention peak (3x3 local centre)'),
               Line2D([], [], color='#eda100', marker='o', mfc='none', ls='', ms=9, label="model's answer")]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, out, name)


def fig_head_edges(S: dict, out: Path, key='rc_2p5e', name='head_edges.png', title='RefCOCO (2.5e answers)') -> None:
    """Edge regression coefficients and box-normalized attention centroid of every head, layers 16-32."""
    prof = S['edges'][key]
    b = np.array(prof['beta'])
    uv = np.array(prof['uv_median'])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for k, (ax, lab) in enumerate(zip(axes[:2], ('x query: peak x on (x_min, x_max)', 'y query: peak y on (y_min, y_max)'))):
        for li, l in enumerate(FULL_LAYERS):
            if l < 16:
                continue
            ax.scatter(b[li, :, k, 0], b[li, :, k, 1], s=22, color=C[li - 3], label=f'layer {l}', edgecolor='white', lw=0.5)
            for h in range(16):
                if abs(b[li, h, k, 0] - b[li, h, k, 1]) > 0.55:
                    ax.annotate(f'{l}.{h}', (b[li, h, k, 0], b[li, h, k, 1]), fontsize=6.5, xytext=(2, 2),
                                textcoords='offset points')
        ax.plot([0, 1], [1, 0], color=GREY, lw=0.8, ls='--')
        ax.plot([0.5], [0.5], marker='s', color='k', ms=5)
        ax.set_xlabel('coefficient on min edge ($b_{min}$)')
        ax.set_ylabel('coefficient on max edge ($b_{max}$)')
        ax.set_title(lab)
        ax.set_xlim(-0.35, 1.1)
        ax.set_ylim(-0.35, 1.1)
    axes[0].legend(fontsize=7, frameon=False, loc='upper right')
    ax = axes[2]
    for li, l in enumerate(FULL_LAYERS):
        if l < 16:
            continue
        ax.scatter(uv[li, :, 0], uv[li, :, 1], s=22, color=C[li - 3], edgecolor='white', lw=0.5)
        for h in range(16):
            if abs(uv[li, h, 0] - 0.5) > 0.15 or abs(uv[li, h, 1] - 0.5) > 0.2:
                ax.annotate(f'{l}.{h}', (uv[li, h, 0], uv[li, h, 1]), fontsize=6.5, xytext=(2, 2), textcoords='offset points')
    ax.axhline(0.5, color=GREY, lw=0.6)
    ax.axvline(0.5, color=GREY, lw=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel('u at x query (0 = left edge, 1 = right edge)')
    ax.set_ylabel('v at y query (0 = top edge, 1 = bottom edge)')
    ax.set_title('median in-box attention centroid')
    fig.suptitle(f'{title}: n = {prof["n"]} images; black square = centre reader', fontsize=10)
    fig.tight_layout()
    save(fig, out, name)


def fig_salient_uv(rows: list[dict], out: Path) -> None:
    """2D histograms of the H11 peak position in box-normalized coordinates, by category and query."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
    bins = np.linspace(-0.2, 1.2, 15)
    for i, q in enumerate(('y', 'x')):
        for j, k in enumerate(('person', 'animal', 'other')):
            uv = np.array([r['maps']['H11'][q]['peak_uv'] for r in rows if r['category'] == k])
            h, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=[bins, bins])
            ax = axes[i, j]
            im = ax.imshow(h.T / len(uv), origin='upper', extent=(bins[0], bins[-1], bins[-1], bins[0]), cmap='viridis')
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec='white', lw=1.2))
            ax.plot(0.5, 0.5, marker='+', color='white', ms=10, mew=2)
            ax.set_title(f'{k}, {q} query (n = {len(uv)})')
            ax.set_xlabel('u (box-normalized x)')
            ax.set_ylabel('v (box-normalized y)')
            cb = fig.colorbar(im, ax=ax, fraction=0.046)
            cb.set_label('fraction of images', fontsize=7)
    fig.suptitle('L20 H11 attention peak relative to the target box (white frame = box, + = box centre)', fontsize=10)
    fig.tight_layout()
    save(fig, out, 'salient_uv.png')


def fig_size(S: dict, out: Path) -> None:
    """Peak-to-centre distance against object area (H11 and the edge-group means), model error for reference."""
    sal = S['salient']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (name, col) in enumerate((('H11', C[0]), ('L20H2', C[1]), ('L20mean', C[2]))):
        for q, ls in (('y', '-'), ('x', '--')):
            ba = sal[name][q]['by_area']
            mid = [f"{b['lo']:.2f}-{min(b['hi'], 1):.2f}\n(n={b['n']})" for b in ba]
            axes[0].plot(range(len(ba)), [b['peak_err_median'] for b in ba], ls=ls, marker='o', color=col,
                         label=f'{name}, {q} query')
            axes[1].plot(range(len(ba)), [b['peak_err_rel_median'] for b in ba], ls=ls, marker='o', color=col)
    axes[0].plot(range(len(ba)), [b['model_err_median'] for b in ba], color='k', marker='s', label="model's answer")
    for ax in axes:
        ax.set_xticks(range(len(ba)))
        ax.set_xticklabels(mid, fontsize=7)
        ax.set_xlabel('box area / image area')
    axes[0].set_ylabel('median distance to box centre (0-1000 units)')
    axes[1].set_ylabel('median distance / box diagonal')
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, out, 'salient_size.png')


# ----------------------------------------------------------------------------- Q1: what / where

def fig_patch_heads(S: dict, out: Path) -> None:
    P = S.get('patch', {})
    keys = [k for k in ('rc:where', 'rc:what', 'ms_large:where', 'ms_large:what', 'ms_small:where', 'ms_small:what') if k in P]
    if not keys:
        return
    fig, axes = plt.subplots(len(keys) // 2 + len(keys) % 2, 2, figsize=(12, 2.6 * (len(keys) + 1) // 2), squeeze=False)
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.7)
    for ax, k in zip(axes.ravel(), keys):
        hm = np.array(P[k]['heads'])
        im = ax.imshow(hm, cmap=DIVERGING, norm=norm, aspect='auto')
        for li in range(8):
            for h in range(16):
                if abs(hm[li, h]) >= 0.05:
                    ax.text(h, li, f'{hm[li, h]:.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=5.5)
        ax.set_yticks(range(8))
        ax.set_yticklabels(FULL_LAYERS)
        ax.set_xticks(range(16))
        g, kind = k.split(':')
        ax.set_title(f"{GROUP_LABEL[g]}, {kind} pair (n = {P[k]['n_values']} values); all heads "
                     f"{P[k]['full_all']['mean']:.2f}, sum of singles {P[k]['sum_single']:.2f}", fontsize=8)
        ax.set_xlabel('head')
        ax.set_ylabel('layer')
        cb = fig.colorbar(im, ax=ax, fraction=0.03)
        cb.set_label('recovered fraction r', fontsize=7)
    for ax in axes.ravel()[len(keys):]:
        ax.axis('off')
    fig.tight_layout()
    save(fig, out, 'patch_heads.png')


def fig_what_where(S: dict, out: Path) -> None:
    P = S.get('patch', {})
    groups = [g for g in ('rc', 'ms_large', 'ms_small') if f'{g}:what' in P and f'{g}:where' in P]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 4.0), squeeze=False)
    for ax, g in zip(axes[0], groups):
        a, b = np.array(P[f'{g}:where']['heads']), np.array(P[f'{g}:what']['heads'])
        for li, l in enumerate(FULL_LAYERS):
            ax.scatter(a[li], b[li], s=18, color=C[li % len(C)], label=f'layer {l}', edgecolor='white', lw=0.4)
            for h in range(16):
                if max(abs(a[li, h]), abs(b[li, h])) > 0.06:
                    ax.annotate(f'{l}.{h}', (a[li, h], b[li, h]), fontsize=6.5, xytext=(2, 2), textcoords='offset points')
        lim = [min(a.min(), b.min(), 0) - 0.03, max(a.max(), b.max()) + 0.05]
        ax.plot(lim, lim, color=GREY, lw=0.7, ls='--')
        ax.axhline(0, color=GREY, lw=0.5)
        ax.axvline(0, color=GREY, lw=0.5)
        ax.set_xlabel('r, where pair (target content moved)')
        ax.set_ylabel('r, what pair (expression swapped)')
        ax.set_title(GROUP_LABEL[g])
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc='upper left')
    fig.tight_layout()
    save(fig, out, 'what_where.png')


def fig_ablate(S: dict, out: Path) -> None:
    Ab = S.get('ablate', {})
    if not Ab:
        return
    gks = [k for k in ('rc@answer', 'ms_large@answer', 'ms_small@answer', 'disc@answer', 't6g10@answer') if k in Ab]
    sets = [s for s in Ab[gks[0]] if s != 'none']
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    w = 0.8 / len(gks)
    for gi, gk in enumerate(gks):
        g = gk.split('@')[0]
        lab = GROUP_LABEL.get(g, {'disc': 'single disc', 't6g10': 'T6 10x10 grid'}.get(g, g))
        x = np.arange(len(sets) + 1) + gi * w
        names = ['none'] + sets
        acc = [Ab[gk][s]['correct'] for s in names]
        axes[0].bar(x, [a['p'] for a in acc], width=w * 0.92, color=C[gi], label=lab,
                    yerr=np.array([[a['p'] - a['ci'][0], a['ci'][1] - a['p']] for a in acc]).T, capsize=2,
                    error_kw={'lw': 0.7})
        for k, (ax, key) in enumerate(zip(axes[1:], ('shift_x', 'shift_y'))):
            v = [Ab[gk][s][key] for s in sets]
            ax.bar(np.arange(len(sets)) + gi * w, [d['mean'] for d in v], width=w * 0.92, color=C[gi],
                   yerr=np.array([[d['mean'] - d['ci'][0], d['ci'][1] - d['mean']] for d in v]).T, capsize=2,
                   error_kw={'lw': 0.7})
    axes[0].set_xticks(np.arange(len(sets) + 1) + 0.4 - w / 2)
    axes[0].set_xticklabels(['none'] + sets, rotation=30, fontsize=7)
    axes[0].set_ylabel('hit rate (Wilson 95% CI)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(gks), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    for ax, t in zip(axes[1:], ('x', 'y')):
        ax.set_xticks(np.arange(len(sets)) + 0.4 - w / 2)
        ax.set_xticklabels(sets, rotation=30, fontsize=7)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_ylabel(f'answer shift in {t} / box {"width" if t == "x" else "height"}\n(ablated minus intact, mean, 95% CI)')
    fig.tight_layout()
    save(fig, out, 'ablate.png')


def fig_flip(S: dict, out: Path) -> None:
    F = S.get('flip')
    if not F:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={'width_ratios': [1.1, 1, 1]})
    ax = axes[0]
    bins = F['bins']
    lab = ['<0', '0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1', '>1']
    x = np.arange(len(lab))
    for j, (cat, ls) in enumerate((('person', '-'), ('other', ':'))):
        for cond, col in (('upright', C[0]), ('flipped', C[1])):
            e = F['h11_peak_v'][cat][cond]
            ax.plot(x, np.array(e['hist']) / e['n'], ls=ls, marker='o', color=col, label=f'{cat}, {cond} (n = {e["n"]})')
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=7)
    ax.set_xlabel('v of the L20 H11 peak at the y query (0 = top edge of box, 1 = bottom edge)')
    ax.set_ylabel('fraction of images')
    ax.legend(fontsize=7, frameon=False)
    for ax, (q, idx) in zip(axes[1:], (('y', 1), ('x', 0))):
        heads = F['heads']
        x = np.arange(len(heads))
        for j, (cond, col) in enumerate((('upright', C[0]), ('flipped', C[1]))):
            v = [F[cond][hd][q] for hd in heads]
            ax.bar(x + (j - 0.5) * 0.38, [d['mean'] for d in v], width=0.36, color=col, label=cond,
                   yerr=np.array([[d['mean'] - d['ci'][0], d['ci'][1] - d['mean']] for d in v]).T, capsize=2,
                   error_kw={'lw': 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels(heads, rotation=30, fontsize=7)
        ax.axhline(0.5, color=GREY, lw=0.6, ls='--')
        ax.set_ylim(0, 1)
        ax.set_ylabel(('v (0 = top edge of box, 1 = bottom edge)' if q == 'y' else 'u (0 = left edge, 1 = right edge)'))
        ax.set_title(f'{q} query: in-box attention centroid, persons (n = {F["n_person"]})')
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, out, 'flip.png')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    S = json.loads((args.run / 'summary.json').read_text())
    rows = json.loads((args.run / 'salient_rows.json').read_text())
    fig_gallery(args.run, args.source, args.out, S, rows)
    fig_head_edges(S, args.out)
    if 'ms_large' in S.get('edges_ms', {}):
        S['edges']['ms_large'] = S['edges_ms']['ms_large']
        fig_head_edges(S, args.out, key='ms_large', name='head_edges_ms_large.png', title='synthetic 160 px shapes')
    fig_salient_uv(rows, args.out)
    fig_size(S, args.out)
    fig_patch_heads(S, args.out)
    fig_what_where(S, args.out)
    fig_ablate(S, args.out)
    fig_flip(S, args.out)


if __name__ == '__main__':
    main()
