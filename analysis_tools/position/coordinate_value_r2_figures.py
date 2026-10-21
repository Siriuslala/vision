"""Experiment 3.6 round 2 figures (reads results.json / analysis.npz of the round-2 run).

Manifold figures follow the PI reference style: a 3D PCA point cloud drawn as shaded spheres coloured by
the continuous variable (viridis), a grey shadow projected onto a framed floor parallelogram, an oblique
side view (top row) and a top-down view (bottom row), each inside a thin frame, no ticks, and a colorbar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
CMAP = 'viridis'


# ----------------------------------------------------------------------------- manifold style

def _shade(rgba: np.ndarray, f: float) -> np.ndarray:
    out = rgba.copy()
    out[:, :3] = np.clip(rgba[:, :3] * f if f < 1 else rgba[:, :3] + (1 - rgba[:, :3]) * (f - 1), 0, 1)
    return out


def manifold_view(ax, P: np.ndarray, c: np.ndarray, norm, elev: float, azim: float, size: float = 22,
                  cmap: str = CMAP) -> None:
    """One view of a 3D point cloud with floor shadow and framed floor parallelogram (isotropic axes)."""
    X = P - P.mean(0)
    X = X / np.abs(X).max()
    lo, hi = X.min(0), X.max(0)
    pad = 0.12
    z0 = lo[2] - 0.35
    x0, x1, y0, y1 = lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad
    floor = [[(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]]
    ax.computed_zorder = False                       # draw order: floor, shadow, spheres, highlights
    poly = Poly3DCollection(floor, facecolor=(0.97, 0.97, 0.97, 0.6), edgecolor='0.35', linewidth=0.6, zorder=0)
    ax.add_collection3d(poly)
    ax.scatter(X[:, 0], X[:, 1], np.full(len(X), z0 + 1e-3), s=size * 1.3, c='0.55', alpha=0.10, edgecolors='none',
               depthshade=False, zorder=1)
    rgba = plt.get_cmap(cmap)(norm(c))
    order = np.argsort(c)
    ax.scatter(X[order, 0], X[order, 1], X[order, 2], s=size, c=rgba[order], edgecolors=_shade(rgba[order], 0.55),
               linewidths=0.35, depthshade=True, zorder=2)
    ax.scatter(X[order, 0], X[order, 1], X[order, 2], s=size * 0.18, c=_shade(rgba[order], 1.55), edgecolors='none',
               alpha=0.8, depthshade=True, zorder=3)
    ax.view_init(elev=elev, azim=azim)
    r = 1.0 + pad
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(z0, r)
    ax.set_box_aspect((1, 1, (r - z0) / (2 * r)))
    ax.set_axis_off()


def frame(fig, ax, inset: float = 0.004) -> None:
    b = ax.get_position()
    fig.add_artist(Rectangle((b.x0 + inset, b.y0 + inset), b.width - 2 * inset, b.height - 2 * inset,
                             transform=fig.transFigure, fill=False, edgecolor='0.3', linewidth=0.6))


def manifold_figure(panels: list[dict], path: Path, cbar_label: str, suptitle: str | None = None,
                    vmin: float = 0, vmax: float = 1000) -> None:
    """panels: dicts with 'P' (n, 3) PCA coordinates, 'c' (n,) colour values, 'title', 'subtitle'."""
    n = len(panels)
    fig = plt.figure(figsize=(3.0 * n + 0.8, 6.4))
    norm = mcolors.Normalize(vmin, vmax)
    axes = []
    for j, p in enumerate(panels):
        a1 = fig.add_axes([0.02 + j * (0.9 / n), 0.50, 0.9 / n - 0.01, 0.40], projection='3d')
        a2 = fig.add_axes([0.02 + j * (0.9 / n), 0.06, 0.9 / n - 0.01, 0.40], projection='3d')
        manifold_view(a1, p['P'], p['c'], norm, elev=22, azim=-58)
        manifold_view(a2, p['P'], p['c'], norm, elev=89.9, azim=-90)
        fig.text(0.02 + j * (0.9 / n) + (0.9 / n - 0.01) / 2, 0.99, p['title'], ha='center', va='bottom', fontsize=10)
        fig.text(0.02 + j * (0.9 / n) + (0.9 / n - 0.01) / 2, 0.915, p['subtitle'], ha='center', va='bottom', fontsize=8,
                 color='0.35')
        axes += [a1, a2]
    for a in axes:
        frame(fig, a)
    fig.text(0.005, 0.70, 'side view', rotation=90, va='center', fontsize=8, color='0.35')
    fig.text(0.005, 0.26, 'top view', rotation=90, va='center', fontsize=8, color='0.35')
    cax = fig.add_axes([0.935, 0.20, 0.012, 0.6])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label(cbar_label)
    if suptitle:
        fig.suptitle(suptitle, y=1.07, fontsize=11)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- categorical palette (fixed order)

C1, C2, C3, C4, C5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
GREY = '0.45'


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- Q1 geometry

def fig_manifolds(A: dict, figs: Path) -> None:
    names = {'resid20': ('residual stream', 'L20 output'), 'attn20': ('attention block', 'L20 output'),
             'mlp28': ('MLP block', 'L28 output'), 'head20h11': ('head L20 H11', '256-d, before o_proj'),
             'resid12': ('residual stream', 'L12 output'), 'resid28': ('residual stream', 'L28 output'),
             'resid32': ('residual stream', 'L32 output'), 'mlp24': ('MLP block', 'L24 output')}
    for axis, role, sweep in (('x', 'x0', 'sweep_h'), ('y', 'y0', 'sweep_v')):
        t = A[f'geo_{axis}_t']
        for tag, keys in (('sites', ('resid20', 'attn20', 'mlp28', 'head20h11')),
                          ('layers', ('resid12', 'resid20', 'resid28', 'resid32'))):
            panels = []
            for k in keys:
                v = A[f'geo_{axis}_{k}_var3']
                panels.append({'P': A[f'geo_{axis}_{k}_pca3'], 'c': t, 'title': names[k][0],
                               'subtitle': f'Qwen3.5-9B, {names[k][1]}, role {role}\nPC1-3 {100 * v.sum():.0f}% var.'})
            manifold_figure(panels, figs / f'manifold_{tag}_{axis}.png', f'true {axis} (0-1000)',
                            f'{sweep}: state predicting the first digit of {axis}, coloured by true {axis}')
    tg = A['geo_grid_t']
    panels = []
    for role in ('x0', 'y0'):
        for a, nm in ((0, 'x'), (1, 'y')):
            v = A[f'geo_grid_{role}_resid20_var3']
            panels.append({'P': A[f'geo_grid_{role}_resid20_pca3'], 'c': tg[:, a], 'title': f'role {role}, colour = true {nm}',
                           'subtitle': f'Qwen3.5-9B, residual L20, grid2d\nPC1-3 {100 * v.sum():.0f}% var.'})
    manifold_figure(panels, figs / 'manifold_grid.png', 'true coordinate (0-1000)',
                    '29x29 grid of disc positions: the same states coloured by true x and by true y')


def fig_geometry_metrics(R: dict, figs: Path) -> None:
    g = R['geometry']
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
    metrics = (('signal_fraction', 'share of variance explained\nby the curve of the coordinate', False),
               ('pr', 'participation ratio of the curve', False), ('tortuosity', 'arc length / chord', False),
               ('probe_mae', 'probe MAE on sweeps (units)', True), ('delta_median', 'local threshold delta (units)', True))
    for ax, (m, lab, logy) in zip(axs.ravel(), metrics):
        for site, col in (('resid', C1), ('attn', C2), ('mlp', C3)):
            for axis, ls in (('x', '-'), ('y', '--')):
                L = [l for l in range(1, 33) if f'{site}{l}' in g[axis]]
                ax.plot(L, [g[axis][f'{site}{l}'][m] for l in L], ls, color=col, lw=1.4, label=f'{site}, {axis}')
        ax.set_xlabel('layer')
        ax.set_ylabel(lab)
        if logy:
            ax.set_yscale('log')
        ax.grid(alpha=0.25)
    axs[0, 0].legend(fontsize=7, ncol=2)
    ax = axs[1, 2]
    for axis, ls in (('x', '-'), ('y', '--')):
        for key, col in (('resid20', C1), ('resid28', C4)):
            m = g[axis][key]
            s = np.array(m['speed'])
            ax.plot(m['speed_t'], s / s.mean(), ls, color=col, lw=1.2, label=f'{key}, {axis}')
    ax.axvspan(400, 610, color='0.85', zorder=0, label='x plateau 498/500')
    ax.set_xlabel('true coordinate (0-1000)')
    ax.set_ylabel('local speed |d mu/dt| / mean')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.suptitle('Geometry of the coordinate curve at the first-digit position (dominant answer prefix)', y=1.0)
    fig.tight_layout()
    _save(fig, figs / 'geometry_metrics.png')


def fig_compression(A: dict, R: dict, figs: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, axis in zip(axs, 'xy'):
        t, e = A[f'geo_{axis}_t'], A[f'geo_{axis}_emitted']
        ax.plot([0, 1000], [0, 1000], color=GREY, lw=0.8, ls=':', label='identity')
        ax.plot(t, e, '.', color=C2, ms=4, label='model output')
        for key, col in (('resid20', C1), ('resid28', C3)):
            ax.plot(A[f'geo_{axis}_{key}_probe_t'], A[f'geo_{axis}_{key}_probe'], '.', color=col, ms=2.5,
                    label=f'linear probe, {key}')
        s = R['geometry'][f'model_slope_{axis}']
        ax.set_title(f'{axis}: model slope {s["lo"]:.2f} (t<380) / {s["hi"]:.2f}; '
                     f'probe L20 {R["geometry"][axis]["resid20"]["probe_slope_lo"]:.2f} / '
                     f'{R["geometry"][axis]["resid20"]["probe_slope_hi"]:.2f}', fontsize=8.5)
        ax.set_xlabel(f'true {axis} (0-1000)')
        ax.set_ylabel(f'{axis} (0-1000)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, figs / 'compression.png')


# ----------------------------------------------------------------------------- Q3 probe versus lens

def fig_reconcile(A: dict, R: dict, figs: Path) -> None:
    rc = R['reconcile']
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), gridspec_kw={'height_ratios': [2, 1]})
    titles = {'x0': 'x, first digit (role x0)', 'y0': 'y, first digit (role y0)',
              'x1': 'x, second digit (role x1)', 'y1': 'y, second digit (role y1)'}
    L = np.arange(33)
    for j, role in enumerate(('x0', 'y0', 'x1', 'y1')):
        ax = axs[0, j]
        r = rc[role]
        for key, col, lab in (('full_probe', C1, 'ridge on full residual (4096-d)'), ('lens_probe', C2, 'classifier on 10 lens logits'),
                              ('random10', C3, 'classifier on 10 random directions'), ('lens_argmax', C5, 'logit lens argmax')):
            ax.plot(L, A[f'rec_{role}_{key}'], color=col, lw=1.5, label=lab)
        ax.axhline(r['model']['p'], color='k', lw=1, ls='--', label='model output')
        ax.axhline(r['majority_baseline'], color=GREY, lw=1, ls=':', label='majority class')
        ax.set_ylim(0, 1)
        ax.set_title(f"{titles[role]}, n = {r['n_test']}")
        ax.set_ylabel('exact digit accuracy')
        ax.grid(alpha=0.25)
        ax2 = axs[1, j]
        ax2.semilogy(L, np.maximum(A[f'rec_{role}_beta_in_U'], 1e-5), color=C4, lw=1.5, label='share of coordinate direction in digit span')
        ax2.axhline(r['beta_in_U_random_expectation'], color=GREY, ls=':', lw=1, label='random direction (9/4096)')
        ax2.set_xlabel('layer')
        ax2.set_ylabel('squared-norm share')
        ax2.grid(alpha=0.25)
    axs[0, 0].legend(fontsize=7, loc='upper left')
    axs[1, 0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    _save(fig, figs / 'reconcile.png')


def fig_unembed(A: dict, R: dict, figs: Path) -> None:
    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_axes([0.05, 0.12, 0.36, 0.78])
    cos = A['ug_unembed_cos']
    im = ax.imshow(cos, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('digit')
    ax.set_ylabel('digit')
    ax.set_title(f"cosine of centred digit unembeddings\nr(cos, |i-j|) = {R['unembed']['unembed']['corr_cos_vs_numeric_distance']:.2f}",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label='cosine')
    ax3 = fig.add_axes([0.50, 0.05, 0.40, 0.88], projection='3d')
    P = A['ug_unembed_pca3']
    norm = mcolors.Normalize(0, 9)
    manifold_view(ax3, P, np.arange(10), norm, elev=22, azim=-58, size=90)
    X = (P - P.mean(0)) / np.abs(P - P.mean(0)).max()
    for d in range(10):
        ax3.text(X[d, 0], X[d, 1], X[d, 2] + 0.12, str(d), fontsize=9, ha='center')
    v = R['unembed']['unembed']['pc_var']
    ax3.set_title(f'digit unembeddings, PCA (PC1-3 {100 * sum(v[:3]):.0f}% var.)', fontsize=9)
    cax = fig.add_axes([0.92, 0.2, 0.012, 0.6])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, label='digit value')
    _save(fig, figs / 'unembed.png')


# ----------------------------------------------------------------------------- Q2 formats

FMT_LABEL = {'s1000': '0-1000', 'text': 'text to 0-1000', 's100': '0-100', 's10': '0-10', 'pix': 'pixels', 'frac': 'fraction 0.xx',
             'count': 'counting'}


def fig_formats(R: dict, figs: Path) -> None:
    """Digit sharpness grouped by digit bin width (visual coordinates) and for the text-conversion control."""
    tab = [r for r in R['formats']['digit_table'] if r['n'] >= 20 and r['kind'] != 'count' and r['fmt'] != 'pix']
    digit_name = {('s1000', 'third_last'): 'hundreds', ('s1000', 'second_last'): 'tens', ('s1000', 'last'): 'units',
                  ('text', 'third_last'): 'hundreds', ('text', 'second_last'): 'tens', ('text', 'last'): 'units',
                  ('frac', 'second_last'): 'tenths', ('frac', 'last'): 'hundredths', ('s100', 'second_last'): 'tens',
                  ('s100', 'last'): 'units', ('s10', 'last'): 'single digit'}
    groups = [('bin 100 units', C1, lambda r: r['fmt'] != 'text' and abs(r['bin_units'] - 100) < 1),
              ('bin 10 units', C2, lambda r: r['fmt'] != 'text' and abs(r['bin_units'] - 10) < 1),
              ('bin 1 unit', C3, lambda r: r['fmt'] != 'text' and abs(r['bin_units'] - 1) < 0.5),
              ('text-to-0-1000 control', C4, lambda r: r['fmt'] == 'text')]
    rows = []
    for gname, col, f in groups:
        for r in sorted([r for r in tab if f(r)], key=lambda r: (r['kind'], r['fmt'])):
            src = 'RefCOCO' if r['kind'] == 'refcoco' else 'disc'
            rows.append((f"{src}, {FMT_LABEL[r['fmt']]}: {digit_name.get((r['fmt'], r['slot']), r['slot'])} (n={r['n']})", r, col, gname))
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.32 * len(rows) + 1.5), sharey=True)
    y = np.arange(len(rows))
    for ax, metric, lab in ((axs[0], 'p_true', 'P(true digit | true preceding digits)'),
                            (axs[1], 'entropy_bits', 'entropy of the digit distribution (bits)')):
        for i, (name, r, col, g) in enumerate(rows):
            m, ci = r[metric]['mean'], r[metric]['ci']
            ax.barh(i, m, color=col, height=0.7, label=g if (ax is axs[0] and (i == 0 or rows[i - 1][3] != g)) else None)
            ax.errorbar(m, i, xerr=[[m - ci[0]], [ci[1] - m]], fmt='none', color='k', elinewidth=0.7)
        ax.set_xlabel(lab)
        ax.grid(alpha=0.25, axis='x')
    axs[0].axvline(0.1, color=GREY, ls=':', lw=1)
    axs[1].axvline(np.log2(10), color=GREY, ls=':', lw=1)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([r[0] for r in rows], fontsize=7)
    axs[0].invert_yaxis()
    axs[0].legend(fontsize=7, loc='center right')
    axs[0].set_title('dotted: uniform over 10 digits', fontsize=8)
    fig.tight_layout()
    _save(fig, figs / 'formats.png')


def fig_counting(R: dict, rows: list[dict], figs: Path) -> None:
    cnt = [r for r in rows if r['kind'] == 'count']
    if not cnt:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axs[0]
    for lab, col, sel in (('single-digit counts', C3, lambda r: r['n_digits'] == 1),
                          ('tens digit of 10-50', C1, lambda r: r['n_digits'] == 2 and r['label'] == 'second_last'),
                          ('units digit of 10-50', C2, lambda r: r['n_digits'] == 2 and r['label'] == 'last')):
        xs = [r for r in cnt if sel(r)]
        ax.bar(lab, np.mean([r['p_true'] for r in xs]), color=col, width=0.6)
        ax.text(lab, np.mean([r['p_true'] for r in xs]) + 0.02, f"n={len(xs)}", ha='center', fontsize=7)
    ax.set_ylabel('mean P(true digit | true preceding digits)')
    ax.set_ylim(0, 1)
    ax.set_title('counting: digit sharpness', fontsize=9)
    ax.tick_params(axis='x', labelsize=7)
    ax = axs[1]
    pr = R['formats']['probes']
    for key, col, lab in (('count_count_n0_last', C2, 'units slot'), ('count_count_n0_second_last', C1, 'tens slot')):
        if key in pr:
            L = [int(k) for k in pr[key] if k != 'n']
            ax.plot(L, [pr[key][str(l)]['mae_units'] for l in L], 'o-', color=col, label=f'{lab} (n={pr[key]["n"]})')
    ax.set_xlabel('layer')
    ax.set_ylabel('probe MAE of the true count (5-fold CV)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, figs / 'counting.png')


def fig_prior(A: dict, figs: Path) -> None:
    if 'prior_s1000' not in A:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios': [2, 1]})
    im = axs[0].imshow(A['prior_s1000'], cmap='viridis', vmin=0, aspect='auto')
    axs[0].set_yticks(range(9))
    axs[0].set_yticklabels([str(d) for d in range(1, 10)])
    axs[0].set_xticks(range(10))
    axs[0].set_xlabel('second digit')
    axs[0].set_ylabel('forced first digit')
    axs[0].set_title('blank canvas: P(second digit | first digit), 0-1000 prompt', fontsize=9)
    fig.colorbar(im, ax=axs[0], label='probability')
    axs[1].bar(range(10), A['sweep_second_digit_mean'], color=C1, width=0.7)
    axs[1].axhline(0.1, color=GREY, ls=':', lw=1, label='uniform')
    axs[1].set_xticks(range(10))
    axs[1].set_xlabel('second digit')
    axs[1].set_ylabel('mean probability')
    axs[1].set_title('disc sweeps: mean second-digit distribution', fontsize=9)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, figs / 'prior.png')


# ----------------------------------------------------------------------------- causal tests

def fig_ablation(R: dict, figs: Path, key: str = 'ablate', fname: str = 'ablation.png') -> None:
    a = R[key]['conditions']
    names = [n for n in a if n != 'clean']
    fig, axs = plt.subplots(1, 3, figsize=(15, 0.28 * len(names) + 1.6), sharey=True)
    y = np.arange(len(names))
    base = a['clean']['first_acc']['p']
    for ax, metric, lab in ((axs[0], 'first_acc', 'first-digit accuracy'),
                            (axs[1], 'delta_p_true', 'change of P(true first digit)'),
                            (axs[2], 'delta_mae', 'change of |x - truth| (units)')):
        for i, n in enumerate(names):
            d = a[n][metric]
            if metric == 'first_acc':
                v, lo, hi = d['p'], d['ci'][0], d['ci'][1]
            else:
                v, lo, hi = d['mean'], d['ci'][0], d['ci'][1]
            p = a[n]['mcnemar_p'] if metric == 'first_acc' else a[n][f'{metric}_p']
            col = C1 if p < 0.01 else '0.65'
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt='o', color=col, ms=4, elinewidth=0.8)
        ax.axvline(base if metric == 'first_acc' else 0, color='k', lw=0.8, ls='--')
        ax.set_xlabel(lab)
        ax.grid(alpha=0.25, axis='x')
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(names, fontsize=7)
    axs[0].invert_yaxis()
    axs[0].set_title(f"dashed: clean ({base:.2f}); blue: p < 0.01", fontsize=8)
    fig.tight_layout()
    _save(fig, figs / fname)


def fig_pathpatch(A: dict, R: dict, figs: Path) -> None:
    pp = R['pathpatch']
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.6), gridspec_kw={'width_ratios': [1.1, 1.1, 1.4]})
    for row, axis in enumerate('xy'):
        if axis not in pp:
            continue
        r = pp[axis]
        for ax, kind, L in ((axs[row, 0], 'mlp', range(20, 33)), (axs[row, 1], 'mixer', range(21, 33))):
            m, ci = A[f'pp_{axis}_{kind}'], A[f'pp_{axis}_{kind}_ci']
            ax.bar(list(L), m, color=C1 if kind == 'mlp' else C2, width=0.7)
            ax.errorbar(list(L), m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt='none', color='k', elinewidth=0.7)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xlabel('layer')
            ax.set_ylabel('share of the L20 H11 effect')
            ax.set_title(f"{axis}: {kind} output at the first-digit position\nset to its sender-patched value "
                         f"(pairs with effect > 1 nat: {r['n_sender_effect_gt1']})", fontsize=8.5)
            ax.grid(alpha=0.25, axis='y')
        ax = axs[row, 2]
        H = A[f'pp_{axis}_heads']
        lim = max(0.05, np.abs(H).max())
        im = ax.imshow(H, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
        ax.set_yticks(range(3))
        ax.set_yticklabels(['L24', 'L28', 'L32'])
        ax.set_xticks(range(16))
        ax.set_xlabel('head')
        ax.set_title(f"{axis}: single heads (share of the L20 H11 effect)\n"
                     f"sender total effect {r['sender_total_effect_normalized']['mean']:.2f} of B-A", fontsize=8.5)
        fig.colorbar(im, ax=ax, label='share of sender effect')
    fig.tight_layout()
    _save(fig, figs / 'pathpatch.png')


# ----------------------------------------------------------------------------- Q5 neurons

def fig_neurons(A: dict, R: dict, figs: Path) -> None:
    nr = R['neurons']
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    for row, role in enumerate(('x0', 'y0')):
        t = A[f'nr_{role}_t']
        C = A[f'nr_{role}_curves'][:12]
        W = A[f'nr_{role}_writes'][:12]
        top = nr[f'{role}_top_neurons'][:12]
        ax = axs[row, 0]
        Call = A[f'nr_{role}_curves'][:30]
        Z = (Call - Call.mean(1, keepdims=True)) / (Call.std(1, keepdims=True) + 1e-9)
        info30 = nr[f'{role}_top_neurons'][:30]
        order = np.argsort([x['peak_t'] for x in info30])
        im = ax.imshow(Z[order], aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5,
                       extent=[t[0], t[-1], len(order) - 0.5, -0.5])
        e = A[f'nr_{role}_emitted']
        ch = np.where(np.diff([int(str(int(v))[0]) for v in e]) != 0)[0]
        for k in ch:
            ax.axvline((t[k] + t[k + 1]) / 2, color='k', lw=0.6, ls=':')
        for b in range(100, 1000, 100):
            ax.plot([b, b], [-0.5, -1.3], color=C2, lw=1.5, clip_on=False)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"L{info30[k]['layer']} N{info30[k]['neuron']} w{info30[k]['write_digit_pos']}" for k in order], fontsize=5)
        ax.set_xlabel(f'true {role[0]} (0-1000); dotted: emitted first digit changes; orange ticks: true hundreds')
        ax.set_title(f'{role}: activation of the 30 most important neurons (L24-32), sorted by peak', fontsize=8.5)
        fig.colorbar(im, ax=ax, label='activation z-score along the sweep')
        ax = axs[row, 1]
        im = ax.imshow(W / np.abs(W).max(1, keepdims=True), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([f"L{x['layer']} N{x['neuron']}" for x in top], fontsize=6.5)
        ax.set_xticks(range(10))
        ax.set_xlabel('digit')
        ax.set_title(f'{role}: digit write vectors (row-normalized)', fontsize=8.5)
        fig.colorbar(im, ax=ax, label='relative logit write')
        ax = axs[row, 2]
        rec = nr[f'{role}_topk_variance_explained']
        ks = [10, 30, 100, 300, 1000]
        for l, col in ((24, C1), (28, C2), (29, C3), (30, C4), (32, C5)):
            ax.plot(ks, [rec[str(l)][str(k)] for k in ks], 'o-', color=col, label=f'MLP {l}')
        ax.set_xscale('log')
        ax.set_xlabel('top-k neurons kept (others at mean)')
        ax.set_ylabel('digit-logit variance explained')
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        ax.set_title(f'{role}: sparsity of the direct digit write (of 12288 neurons)', fontsize=8.5)
    fig.tight_layout()
    _save(fig, figs / 'neurons.png')


def fig_inputs(run: Path, figs: Path) -> None:
    """Example inputs of the format experiment with the model's greedy answer and the true answer."""
    from analysis_tools.position import coordinate_value_r2_run as rr
    from analysis_tools.position.coordinate_readout_run import read_jsonl
    cases = {c['id']: c for c in json.loads((run / 'formats' / 'cases.json').read_text())}
    gen = {r['id']: r for r in read_jsonl(run / 'formats' / 'gen.jsonl')}
    pick = ['s1000_sweep_h_152', 'frac_sweep_h_152', 's100_sweep_h_152', 'text_sweep_h_152', 'count_23_0']
    rc = sorted(i for i in cases if i.startswith('rc_s1000_'))[:1]
    pick = [i for i in pick if i in cases] + rc
    fig, axs = plt.subplots(1, len(pick), figsize=(2.9 * len(pick), 3.6))
    for ax, i in zip(axs, pick):
        c = cases[i]
        img, _ = rr.format_input(c)
        ax.imshow(img)
        ax.set_axis_off()
        ax.add_patch(Rectangle((0, 0), img.size[0] - 1, img.size[1] - 1, fill=False, edgecolor='0.4', lw=0.8))
        ans = gen[i]['text'].strip().replace('```json', '').replace('```', '').replace('\n', ' ').replace('\t', '')
        ans = ' '.join((ans.split(', "label"')[0] + (']}]' if '"label"' in ans else '')).split())
        fmt = 'count' if c['kind'] == 'count' else c['fmt']
        ax.set_title(f"{c['kind']}, {FMT_LABEL[fmt]}", fontsize=8.5)
        ax.text(0.5, -0.06, f"model: {ans[:34]}\ntrue: {', '.join(c['true_strings'])}", transform=ax.transAxes,
                ha='center', va='top', fontsize=7.5, family='monospace')
    fig.tight_layout()
    _save(fig, figs / 'inputs.png')


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--round1', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    R = json.loads((args.run / 'results.json').read_text())
    A = dict(np.load(args.run / 'analysis.npz'))
    if 'geometry' in R:
        fig_manifolds(A, args.figures)
        fig_geometry_metrics(R, args.figures)
        fig_compression(A, R, args.figures)
    if 'reconcile' in R:
        fig_reconcile(A, R, args.figures)
    if 'unembed' in R:
        fig_unembed(A, R, args.figures)
    if 'formats' in R:
        fig_inputs(args.run, args.figures)
        rows = json.loads((args.run / 'format_rows.json').read_text())
        fig_formats(R, args.figures)
        fig_counting(R, rows, args.figures)
        fig_prior(A, args.figures)
    if 'ablate' in R:
        fig_ablation(R, args.figures)
    if 'neuron_ablate' in R:
        fig_ablation(R, args.figures, 'neuron_ablate', 'neuron_ablation.png')
    if 'pathpatch' in R:
        fig_pathpatch(A, R, args.figures)
    if 'neurons' in R:
        fig_neurons(A, R, args.figures)


if __name__ == '__main__':
    main()
