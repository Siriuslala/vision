"""Figures of experiment 3.11 round 3 (edge-reading heads); reads summary.json / arrays.npz of the analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from analysis_tools.position import edge_heads_analysis as ea

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # validated categorical slots
DIVERGING = LinearSegmentedColormap.from_list('bo', ['#1c5cab', '#86b6ef', '#e6e6e3', '#f2a07c', '#b8461b'])
STEP_LABEL = {'pt_x': 'point x', 'pt_y': 'point y', 'bb_x1': 'box x_min', 'bb_y1': 'box y_min',
              'bb_x2': 'box x_max', 'bb_y2': 'box y_max'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def head_label(h: str) -> str:
    l, k = h.split('.')
    return f'L{l} H{k}'


def set_label(name: str) -> str:
    names = {'edge_min': 'left/top group (5)', 'edge_max': 'right/bottom group (6)', 'H11': 'L20 H11'}
    return names[name] if name in names else head_label(name)


# ----------------------------------------------------------------------------- density per step

def fig_density(S, arr, out: Path, rows: list[str], name: str, key: str = 'density') -> None:
    dens = arr[key]
    sets = S['density_sets'] if key == 'density' else ['edge_min', 'edge_max', 'H11']
    lo, hi = ea.DENSITY_RANGE
    fig, axes = plt.subplots(len(rows), 6, figsize=(12, 1.95 * len(rows) + 0.6), squeeze=False)
    for r, nm in enumerate(rows):
        k = sets.index(nm)
        vmax = np.nanpercentile(dens[k], 99)
        for s in range(6):
            ax = axes[r, s]
            im = ax.imshow(dens[k, s], extent=[lo, hi, hi, lo], cmap='viridis', vmin=0, vmax=vmax, interpolation='nearest')
            ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec='white', lw=1.2))
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.tick_params(labelsize=7, length=2)
            if r == 0:
                ax.set_title(STEP_LABEL[S['steps'][s]])
            if s == 0:
                ax.set_ylabel(f'{set_label(nm)}\nv', fontsize=8)
            if r == len(rows) - 1:
                ax.set_xlabel('u', fontsize=8)
        cb = fig.colorbar(im, ax=axes[r].tolist(), fraction=0.015, pad=0.01)
        cb.set_label('attention / uniform', fontsize=7)
        cb.ax.tick_params(labelsize=7)
    fig.savefig(out / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- regression heatmaps

def fig_regression(S, out: Path) -> None:
    heads = S['key_heads']
    beta = np.array([S['beta_key'][h] for h in heads])          # (H, 6, 2, 2)
    r2 = np.array([S['r2_key'][h] for h in heads])               # (H, 6, 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(heads) + 1.6))
    for k, ax in enumerate(axes):
        e = beta[:, :, k, 1] - beta[:, :, k, 0]
        e = np.where(r2[:, :, k] >= 0.3, e, np.nan)                 # R^2 < 0.3: the edges do not explain the peak
        cmap = DIVERGING.copy()
        cmap.set_bad('#f4f4f2')
        im = ax.imshow(e, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        for i in range(len(heads)):
            for s in range(6):
                ax.text(s, i, f'{beta[i, s, k, 0]:.2f}/{beta[i, s, k, 1]:.2f}\nR²={r2[i, s, k]:.2f}', ha='center',
                        va='center', fontsize=6.2, color='black')
        ax.set_xticks(range(6), [STEP_LABEL[s] for s in S['steps']], rotation=30)
        ax.set_yticks(range(len(heads)), [head_label(h) for h in heads])
        ax.set_title(['peak x regressed on (x_min, x_max)', 'peak y regressed on (y_min, y_max)'][k])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cb.set_label('b_max − b_min  (−1 = min edge, +1 = max edge)')
    fig.savefig(out / 'regression_steps.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_regression_all(S, arr, out: Path) -> None:
    """All heads of layers 16-32: (b_min, b_max) of the on-axis peak coordinate at every step."""
    beta = arr['beta']                                             # (6, 2, 8, 16, 2)
    layers = [l for l in ea.FULL_LAYERS if l >= 16]
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.9), sharex=True, sharey=True)
    for s, ax in enumerate(axes):
        k = ea.STEPS[s][3]
        for c, l in enumerate(layers):
            b = beta[s, k, ea.LI[l]]
            ax.scatter(b[:, 0], b[:, 1], s=12, color=SERIES[c], label=f'L{l}', lw=0)
        b = beta[s, k, ea.LI[20], 11]
        ax.scatter([b[0]], [b[1]], s=60, facecolor='none', edgecolor='black', lw=1.2, label='L20 H11')
        ax.plot([0, 1], [1, 0], color='#bdbdb8', ls='--', lw=0.8)
        ax.scatter([0.5], [0.5], marker='s', color='black', s=14, label='centre reader')
        ax.set_title(f"{STEP_LABEL[S['steps'][s]]} (peak {'xy'[k]})", fontsize=8)
        ax.set_xlabel('b_min')
        ax.set_xlim(-0.4, 1.3)
        ax.set_ylim(-0.4, 1.3)
    axes[0].set_ylabel('b_max')
    axes[-1].legend(frameon=False, fontsize=6.5, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out / 'regression_all.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- band fractions

def fig_bands(S, out: Path) -> None:
    plan = [(h, st) for h in ('24.5', '20.10', '24.9', '28.0') for st in ('pt_x', 'bb_x2', 'pt_y', 'bb_y2')]
    plan += [(h, st) for h in ('24.7', '20.0') for st in ('bb_x1', 'pt_x', 'bb_y1', 'pt_y')]
    cats = [('corner', SERIES[1]), ('edge only', SERIES[0]), ('other edge only', SERIES[2]), ('neither', '#bdbdb8')]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for i, (h, st) in enumerate(plan):
        d = S['bands'][h][st]
        mx = h in ea.EDGE_MAX
        axis_x = st.endswith('x') or st.endswith('x1') or st.endswith('x2')
        corner = d['corner_br'] if mx else d['corner_tl']
        if axis_x:
            own = d['right_only'] if mx else d['left_only']
            other = d['bottom_only'] if mx else d['top_only']
        else:
            own = d['bottom_only'] if mx else d['top_only']
            other = d['right_only'] if mx else d['left_only']
        vals = [corner, own, other, 1 - corner - own - other]
        bottom = 0
        for (lab, col), v in zip(cats, vals):
            ax.bar(i, v, bottom=bottom, color=col, width=0.8, edgecolor='white', lw=1, label=lab if i == 0 else None)
            bottom += v
    ax.set_xticks(range(len(plan)), [f'{head_label(h)}\n{STEP_LABEL[st]}' for h, st in plan], fontsize=6.5)
    ax.set_ylabel('fraction of images')
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    fig.savefig(out / 'bands.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- extreme points

def fig_extreme(S, out: Path) -> None:
    keys = [k for k in ('max_pt_x', 'max_bb_x2', 'min_bb_x1', 'max_pt_y', 'max_bb_y2', 'min_bb_y1') if k in S['extreme']]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.7 * len(keys), 2.9), squeeze=False)
    for ax, k in zip(axes[0], keys):
        d = S['extreme'][k]
        p = np.array(d['points'])
        ax.scatter(p[:, 1], p[:, 0], s=8, color=SERIES[0], alpha=0.5, lw=0, label='image')
        ax.plot([0, 1], [0, 1], color='#7a7a75', lw=1, ls='--', label='y = x')
        grp, step = k.split('_', 1)
        off = d['off_axis']
        ax.set_xlabel(f"{off} of the {d['edge']}most mask pixels")
        ax.set_ylabel(f'{off} of group peak')
        ax.set_title(f"{'right/bottom' if grp == 'max' else 'left/top'} group, {STEP_LABEL[step]}\n"
                     f"r={d['r_extreme']:.2f} (centroid {d['r_centroid']:.2f}), n={d['n']}", fontsize=8)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.3, 1.3)
    axes[0, 0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'extreme_points.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- stability

def fig_stability(S, out: Path) -> None:
    plan = [('24.5', 'pt_x', 0), ('20.10', 'pt_x', 0), ('24.5', 'pt_y', 1), ('20.10', 'pt_y', 1),
            ('24.7', 'pt_x', 0), ('24.7', 'pt_y', 1), ('24.7', 'bb_x2', 0), ('24.5', 'bb_x1', 0)]
    st = S['stability']
    groups = [('family', f) for f in st['family']] + [('size', s) for s in st['size']]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharey=True)
    for ax, (h, step, k) in zip(axes.ravel(), plan):
        s = S['steps'].index(step)
        for gi, (kind, g) in enumerate(groups):
            b = np.array(st[kind][g]['beta'][h])[s, k]
            ax.scatter([b[0]], [gi], color=SERIES[0], s=22, label='b_min' if gi == 0 else None, zorder=3)
            ax.scatter([b[1]], [gi], color=SERIES[1], s=22, marker='s', label='b_max' if gi == 0 else None, zorder=3)
        ax.axvline(0.5, color='#bdbdb8', lw=0.8)
        ax.axvline(0, color='#bdbdb8', lw=0.8, ls=':')
        ax.axvline(1, color='#bdbdb8', lw=0.8, ls=':')
        ax.set_title(f"{head_label(h)}, {STEP_LABEL[step]}, peak {'xy'[k]}", fontsize=8)
        ax.set_xlim(-0.4, 1.3)
    labels = [f"{g} (n={st[kind][g]['n']})" if kind == 'family' else
              f"area {st[kind][g]['area_range'][0]:.3f}–{st[kind][g]['area_range'][1]:.3f}" for kind, g in groups]
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(groups)), labels, fontsize=7)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'stability.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- galleries

def overlay(ax, img, att, grid, box, mask, title, peak=None):
    ax.imshow(np.asarray(img) // 2 + 64)                            # dimmed photo so the attention stands out
    H, W = grid
    a = att.reshape(H, W) / att.max()
    rgba = plt.get_cmap('viridis')(a)
    rgba[..., 3] = np.clip(a, 0, 1) ** 0.6 * 0.9                    # transparent where attention is low
    ax.imshow(rgba, extent=[0, W * ea.TP, H * ea.TP, 0], interpolation='nearest')
    im = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec='white', lw=1.2))
    if mask is not None:
        ax.contour(mask, levels=[0.5], colors=['#e87ba4'], linewidths=0.8)
    if peak is not None:
        ax.plot(peak[0], peak[1], 'x', color='red', ms=6, mew=1.6)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def fig_gallery(run: Path, items: list[dict], fam: str, out: Path, n: int = 6) -> None:
    sel = [it for it in items if it['family'] == fam and not np.isnan(it['att'][0, 0, 0, 0])
           and not np.isnan(it['att'][2, 0, 0, 0])]
    if len(sel) < 2:
        return
    sel.sort(key=lambda it: it['area_frac'])
    sel = [sel[int(i)] for i in np.linspace(0, len(sel) - 1, min(n, len(sel)))]
    cols = [('H11', ['20.11'], 0), ('H11', ['20.11'], 1), ('right/bottom group', ea.EDGE_MAX, 0),
            ('right/bottom group', ea.EDGE_MAX, 1), ('left/top group', ea.EDGE_MIN, 0), ('left/top group', ea.EDGE_MIN, 1)]
    fig, axes = plt.subplots(len(sel), 7, figsize=(15, 2.2 * len(sel)))
    for r, it in enumerate(sel):
        c = it['case']
        img = Image.open(c['image_path']).convert('RGB')
        mask = ea.mask_of(c)
        ax = axes[r, 0]
        ax.imshow(img)
        b = it['box']
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='yellow', lw=1.2))
        ax.set_title(f"{c['expression'][:28]} ({it['area_frac']:.3f})", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for k, (nm, heads, s) in enumerate(cols):
            a = np.mean([it['att'][s, ea.hid(h)[0], ea.hid(h)[1]] for h in heads], 0)
            pk = ea.peaks(a[None], it['grid'])[0]
            im = overlay(axes[r, k + 1], img, a, it['grid'], b, mask, f'{nm}, {STEP_LABEL[ea.STEP_NAMES[s]]}', pk)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.01, pad=0.005)
    cb.set_label('attention / max over image tokens')
    fig.savefig(out / f'gallery_{fam}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- salient part

def fig_parts(S, out: Path, head: str = '20.11', min_n: int = 5) -> None:
    pf = S['parts_family'][head]
    fams = sorted(pf['pt_y'])
    if not fams:
        return
    fig, axes = plt.subplots(len(fams), 2, figsize=(11, 1.9 * len(fams)), squeeze=False)
    for i, fam in enumerate(fams):
        for j, st in enumerate(('pt_x', 'pt_y')):
            d = pf[st].get(fam, {})
            parts = sorted([(k, v) for k, v in d.items() if v['n'] >= min_n and v['area'] >= 0.01],
                           key=lambda kv: -kv[1]['area'])[:10]
            ax = axes[i, j]
            x = np.arange(len(parts))
            ax.bar(x - 0.2, [v['area'] for _, v in parts], 0.4, color='#bdbdb8', label='area share')
            ax.bar(x + 0.2, [v['share'] for _, v in parts], 0.4, color=SERIES[0], label='attention share')
            for xi, (_, v) in zip(x, parts):
                ax.text(xi, max(v['area'], v['share']) + 0.01, f"×{v['enrichment']:.1f}", ha='center', fontsize=6.5)
            ax.set_xticks(x, [k for k, _ in parts], rotation=30, fontsize=7)
            ax.set_title(f'{fam}, {head_label(head)}, {STEP_LABEL[st]}', fontsize=8)
            ax.set_ylabel('share of object')
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f'parts_{head.replace(".", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_salient_uv(items, out: Path) -> None:
    fams = sorted({it['family'] for it in items})
    fig, axes = plt.subplots(2, len(fams), figsize=(1.9 * len(fams) + 1, 4.2), squeeze=False)
    li, hh = ea.hid('20.11')
    for j, fam in enumerate(fams):
        for k in range(2):
            vals = []
            for it in items:
                if it['family'] != fam or np.isnan(it['att'][k, 0, 0, 0]):
                    continue
                pk = ea.peaks(it['att'][k, li, hh][None], it['grid'])[0]
                b = it['box']
                vals.append([(pk[0] - b[0]) / (b[2] - b[0]), (pk[1] - b[1]) / (b[3] - b[1])])
            v = np.array(vals)
            hst, _, _ = np.histogram2d(v[:, 1], v[:, 0], bins=10, range=[[-0.25, 1.25], [-0.25, 1.25]])
            im = axes[k, j].imshow(hst / len(v), extent=[-0.25, 1.25, 1.25, -0.25], cmap='viridis', vmin=0, vmax=0.12)
            axes[k, j].add_patch(Rectangle((0, 0), 1, 1, fill=False, ec='white', lw=1))
            axes[k, j].set_title(f'{fam} (n={len(v)})\nH11, {STEP_LABEL[ea.STEP_NAMES[k]]}', fontsize=7)
            axes[k, j].tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02)
    cb.set_label('fraction of images per bin')
    fig.savefig(out / 'salient_uv.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- path patching and ablation

COND_LABEL = {'H11_B': 'H11 from B', 'H11_mean': 'H11 mean-ablated', 'L16H1H2_B': 'L16 H1+H2 from B',
              'L20H3_B': 'L20 H3 from B (control)', 'L20noH11_B': 'L20 other 15 heads from B', 'L20_B': 'L20 all heads from B',
              'dn_pre20_B': 'DeltaNet L1-19 from B', 'dn_pre20_noH11_B': 'DeltaNet L1-19 from B, H11 kept A'}


def fig_path(S, out: Path) -> None:
    P = S['path']
    splits = [s for s in ('rc', 'mc') if s in P]
    if not splits:
        return
    heads = [(24, h) for h in (1, 2, 7, 12, 5, 6, 8, 9, 11)] + [(28, 0), (20, 11), (20, 10)]
    fig = plt.figure(figsize=(18, 4.6))
    gs = fig.add_gridspec(1, len(splits) + 3, width_ratios=[1] * len(splits) + [0.04, 0.75, 0.9], wspace=0.3)
    axes = [fig.add_subplot(gs[0, k]) for k in range(len(splits))]
    for ax, sp in zip(axes, splits):
        d = P[sp]
        conds = d['conds'][1:]
        s = np.array(d['s'])[1:]                                   # (R-1, 4, 16)
        M = np.array([[s[c, d['layers'].index(l), h] for (l, h) in heads] for c in range(len(conds))])
        im = ax.imshow(M.T, cmap=DIVERGING, vmin=-1, vmax=1, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(i, j, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6)
        ax.set_xticks(range(len(conds)), [COND_LABEL[c] for c in conds], rotation=40, ha='right', fontsize=7)
        ax.set_yticks(range(len(heads)), [f'L{l} H{h}' for l, h in heads], fontsize=7)
        ax.set_title(f"{'RefCOCO' if sp == 'rc' else 'PACO'}: attention recovery s (n={d['n_pairs']} pairs)", fontsize=8)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, len(splits)]))
    cb.set_label('attention recovery s')
    ax = fig.add_subplot(gs[0, -1])                                # gs[0, -2] is left empty for the tick labels
    for k, sp in enumerate(splits):
        d = P[sp]
        if d['r'] is None:
            continue
        r = np.array(d['r'])[1:]
        lo, hi = np.array(d['r_ci'][0])[1:], np.array(d['r_ci'][1])[1:]
        y = np.arange(len(r)) + (k - 0.5) * 0.35
        ax.barh(y, r, 0.33, color=SERIES[k], label=f"{'RefCOCO' if sp == 'rc' else 'PACO'} (n={d['n_values']} values)")
        ax.errorbar(r, y, xerr=[r - lo, hi - r], fmt='none', ecolor='#555', lw=0.8)
    ax.set_yticks(range(len(P[splits[0]]['conds']) - 1), [COND_LABEL[c] for c in P[splits[0]]['conds'][1:]], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color='#888', lw=0.8)
    ax.set_xlabel('first-digit logit recovery r (95% bootstrap CI)')
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    fig.savefig(out / 'path.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_ablate(S, out: Path) -> None:
    Z = S['ablate'].get('size')
    if not Z:
        return
    A = {f"area {d['area_range'][0]:.3f}–{d['area_range'][1]:.3f}": {k: v['hit'] for k, v in d.items() if k != 'area_range'}
         for d in Z.values()}
    fams = list(A)
    sets = list(next(iter(A.values())))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    w = 0.8 / len(sets)
    for k, s in enumerate(sets):
        v = np.array([A[f][s] for f in fams])
        x = np.arange(len(fams)) + (k - (len(sets) - 1) / 2) * w
        ax.bar(x, v[:, 0], w, color=SERIES[k % len(SERIES)] if s != 'none' else '#7a7a75', label=s)
        ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt='none', ecolor='#444', lw=0.7)
    ax.set_xticks(range(len(fams)), fams, fontsize=8)
    ax.set_xlabel('object area / image area (quartiles of the 872 objects, 218 each)')
    ax.set_ylabel('point hit rate (Wilson 95% CI)')
    ax.set_ylim(0.75, 1.02)
    ax.legend(ncol=len(sets), frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    fig.savefig(out / 'ablate.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    S = json.loads((args.run / 'summary.json').read_text())
    arr = np.load(args.run / 'arrays.npz')
    fig_density(S, arr, args.out, ['24.7', '20.0', '24.5', '20.10', '24.9', '28.0'], 'density_edge_heads.png')
    fig_density(S, arr, args.out, ['20.11', '24.11', '20.2', 'edge_min', 'edge_max'], 'density_h11_groups.png')
    fig_density(S, arr, args.out, ['edge_min', 'edge_max', 'H11'], 'density_rc.png', key='density_rc')
    fig_density(S, arr, args.out, ['edge_min', 'edge_max', 'H11'], 'density_mc.png', key='density_mc')
    fig_regression(S, args.out)
    fig_regression_all(S, arr, args.out)
    fig_bands(S, args.out)
    fig_extreme(S, args.out)
    fig_stability(S, args.out)
    fig_parts(S, args.out)
    fig_parts(S, args.out, head='20.10')
    fig_path(S, args.out)
    fig_ablate(S, args.out)
    items, _ = ea.load_items(args.run)
    fig_salient_uv(items, args.out)
    for fam in sorted({it['family'] for it in items}):
        fig_gallery(args.run, items, fam, args.out)
    print('figures written', flush=True)


if __name__ == '__main__':
    main()
