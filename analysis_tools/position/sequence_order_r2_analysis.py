"""Statistics and figures of experiment 3.8 round 2 (Gated DeltaNet mechanism of the row ordinal).

Usage: python -m analysis_tools.position.sequence_order_r2_analysis --run RUN_DIR --round1 R1_DIR --figures FIG_DIR
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from analysis_tools.position import sequence_order_r2 as s2
from analysis_tools.position.sequence_order_analysis import (BLUE, ORANGE, AQUA, YELLOW, VIOLET, INK, MUTED, GRID,
                                                             boot, read_jsonl, value, wilson)

CMAP = 'viridis'
MID = list(s2.MID_LINEAR)


# ----------------------------------------------------------------------------- gates

def load_gates(run: Path) -> dict:
    d = torch.load(run / 'gates' / 'gates.pt', weights_only=False)
    kinds = np.array([m['kind'] for m in d['meta']])
    out = {'layers': d['layers'], 'meta': d['meta'], 'mask': d['mask'], 'kinds': kinds}
    for k in ('grid', 'white', 'natural'):
        idx = np.where(kinds == k)[0]
        out[k] = (torch.stack([d['alpha'][i] for i in idx]).float().numpy(),
                  torch.stack([d['beta'][i] for i in idx]).float().numpy())      # (n, 24, 256, 32)
    idx = np.where(kinds == 'grid_blankrow')[0]
    out['blankrow'] = ([d['alpha'][i].float().numpy() for i in idx], [d['beta'][i].float().numpy() for i in idx],
                       [np.array(d['meta'][i]['index']) for i in idx])
    return out


def head_classes(a_grid: np.ndarray) -> np.ndarray:
    """(24, 32) labels from the mean grid alpha (24, 256, 32):
    3 = cell-row-onset head, 2 = row-reset head, 1 = integrator, 0 = other.

    Cell-row-onset head: at token column 0, alpha differs by more than 0.2 between the top and the bottom
    token row of a cell row (the head erases its state once per figure row, not once per token row).
    Row-reset head: alpha at token column 0 below 0.6 while alpha at columns >= 2 is above 0.85.
    Integrator: alpha above 0.95 at column 0 and above 0.97 on average (memory longer than ~30 tokens
    at every position, including row starts).
    """
    C = np.arange(256) % 16
    a0 = a_grid[:, C == 0].mean(1)
    a2 = a_grid[:, C >= 2].mean(1)
    lab = np.zeros(a0.shape, int)
    lab[(a0 < 0.6) & (a2 > 0.85)] = 2
    lab[(a0 > 0.95) & (a_grid.mean(1) > 0.97)] = 1
    R = np.arange(256) // 16
    even0 = a_grid[:, (C == 0) & (R % 2 == 0)].mean(1)
    odd0 = a_grid[:, (C == 0) & (R % 2 == 1)].mean(1)
    lab[np.abs(even0 - odd0) > 0.2] = 3
    return lab


def gate_stats(g: dict) -> dict:
    C = np.arange(256) % 16
    R = np.arange(256) // 16
    layers = g['layers']
    res = {'layers': layers}
    for k in ('grid', 'white', 'natural'):
        a, b = g[k][0].mean(0), g[k][1].mean(0)               # (24, 256, 32)
        res[k] = {'d_alpha_rowstart': (a[:, C == 0].mean(1) - a[:, C > 0].mean(1)).tolist(),
                  'd_beta_rowstart': (b[:, C == 0].mean(1) - b[:, C > 0].mean(1)).tolist(),
                  'alpha_mean': a.mean(1).tolist(),
                  'd_beta_top_bottom': (b[:, R % 2 == 0].mean(1) - b[:, R % 2 == 1].mean(1)).tolist(),
                  'alpha_by_col': a.reshape(len(layers), 16, 16, 32).mean(1).tolist(),
                  'beta_by_col': b.reshape(len(layers), 16, 16, 32).mean(1).tolist()}
    lab = head_classes(g['grid'][0].mean(0))
    res['classes'] = lab.tolist()
    li = [layers.index(l) for l in MID]
    res['class_counts_mid'] = {'onset': int((lab[li] == 3).sum()), 'reset': int((lab[li] == 2).sum()),
                               'integrator': int((lab[li] == 1).sum()), 'n_heads': int(lab[li].size)}
    res['class_counts_all'] = {'onset': int((lab == 3).sum()), 'reset': int((lab == 2).sum()),
                               'integrator': int((lab == 1).sum()), 'n_heads': int(lab.size)}
    for k in ('grid', 'white', 'natural'):
        a_ = g[k][0].mean(0)
        e0 = a_[:, (C == 0) & (R % 2 == 0)].mean(1)
        o0 = a_[:, (C == 0) & (R % 2 == 1)].mean(1)
        res[k]['parity_d_alpha_rowstart'] = (e0 - o0).tolist()
        res[k]['n_parity_heads'] = int((np.abs(e0 - o0) > 0.2).sum())
        res[k]['n_parity_heads_mid'] = int((np.abs(e0 - o0)[li] > 0.2).sum())
        b_ = g[k][1].mean(0)
        top = b_[:, R == 0].mean(1) - b_[:, R > 0].mean(1)
        res[k]['n_top_writer_heads'] = int((top > 0.2).sum())
    da = {k: np.array(res[k]['d_alpha_rowstart']).ravel() for k in ('grid', 'white', 'natural')}
    res['corr_d_alpha'] = {'grid_white': float(np.corrcoef(da['grid'], da['white'])[0, 1]),
                           'grid_natural': float(np.corrcoef(da['grid'], da['natural'])[0, 1])}
    # Rows starts in natural images: share of heads whose alpha drops by more than 0.2 at token column 0.
    res['n_drop_heads'] = {k: int((da[k] < -0.2).sum()) for k in da}
    # Inserted blank cell row: beta / alpha on blank tokens vs on figure tokens (mean over images).
    al, be, idx = g['blankrow']
    db, dal = [], []
    for a_, b_, ix in zip(al, be, idx):
        blank = ix < 0
        db.append(b_[:, blank].mean(1) - b_[:, ~blank].mean(1))
        dal.append(a_[:, blank].mean(1) - a_[:, ~blank].mean(1))
    res['blankrow_d_beta'] = np.mean(db, 0).tolist()
    res['blankrow_d_alpha'] = np.mean(dal, 0).tolist()
    # Effective memory length 1 / (1 - alpha), median over the visual span, per head.
    a = g['grid'][0].mean(0)
    res['memory_median'] = np.median(1 / (1 - np.clip(a, 0, 0.99999)), axis=1).tolist()
    return res


def fig_gate_heatmaps(gs: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    norm = mcolors.Normalize(-0.7, 0.1)
    for ax, k, title in zip(axes, ('grid', 'white', 'natural'), ('8x8 grids (128)', 'white canvas (1)', 'RefCOCO test (32)')):
        m = np.array(gs[k]['d_alpha_rowstart'])
        im = ax.imshow(m, aspect='auto', cmap=CMAP, norm=norm, interpolation='nearest')
        ax.set_yticks(range(len(gs['layers'])), gs['layers'], fontsize=6)
        ax.set_xlabel('head')
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        for l in MID:
            i = gs['layers'].index(l)
            ax.add_patch(Rectangle((-0.5, i - 0.5), 32, 1, fill=False, edgecolor='white', linewidth=0.5))
    axes[0].set_ylabel('DeltaNet layer')
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes, shrink=0.9)
    cb.set_label('Δα at row start (α at token column 0 minus α at columns 1–15)')
    fig.savefig(path)
    plt.close(fig)


def fig_gate_profiles(gs: dict, path: Path) -> None:
    layers = gs['layers']
    lab = np.array(gs['classes'])
    li = [layers.index(l) for l in MID]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    styles = {'grid': ('-', '8x8 grids'), 'white': (':', 'white canvas'), 'natural': ('--', 'RefCOCO')}
    for cls, color, name in ((3, AQUA, 'cell-row-onset heads'), (2, ORANGE, 'row-reset heads'), (1, BLUE, 'integrator heads'),
                             (0, MUTED, 'other heads')):
        sel = [(i, h) for i in li for h in range(32) if lab[i, h] == cls]
        if not sel:
            continue
        for k, (ls, kname) in styles.items():
            A = np.array(gs[k]['alpha_by_col'])
            Bt = np.array(gs[k]['beta_by_col'])
            axes[0].plot(range(16), np.mean([A[i, :, h] for i, h in sel], 0), ls=ls, color=color)
            axes[1].plot(range(16), np.mean([Bt[i, :, h] for i, h in sel], 0), ls=ls, color=color)
        axes[0].plot([], [], color=color, label=f'{name} ({len(sel)})')
    for k, (ls, kname) in styles.items():
        axes[0].plot([], [], ls=ls, color=INK, label=kname)
    axes[0].set_ylabel('decay α (mean over heads, rows, images)')
    axes[1].set_ylabel('write strength β')
    for ax in axes:
        ax.set_xlabel('token column C within the token row')
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title('DeltaNet layers 9–16: α along a token row', fontsize=10)
    axes[1].set_title('β along a token row', fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def fig_gate_maps(g: dict, gs: dict, heads: list[tuple[int, int]], path: Path) -> None:
    layers = gs['layers']
    fig, axes = plt.subplots(2, len(heads), figsize=(3.1 * len(heads), 5.9), constrained_layout=True)
    na, nb = mcolors.Normalize(0, 1), mcolors.Normalize(0, 1)
    for j, (l, h) in enumerate(heads):
        i = layers.index(l)
        a = g['grid'][0][:, i, :, h].mean(0).reshape(16, 16)
        b = g['grid'][1][:, i, :, h].mean(0).reshape(16, 16)
        axes[0, j].imshow(a, cmap=CMAP, norm=na)
        axes[1, j].imshow(b, cmap=CMAP, norm=nb)
        kind = {3: 'cell-row onset', 2: 'row-reset', 1: 'integrator', 0: 'other'}[gs['classes'][i][h]]
        axes[0, j].set_title(f'layer {l} head {h} ({kind})', fontsize=9)
        for ax in axes[:, j]:
            ax.set_xticks([0, 15])
            ax.set_yticks([0, 15])
            ax.grid(False)
    axes[0, 0].set_ylabel('α: token row')
    axes[1, 0].set_ylabel('β: token row')
    for ax in axes[1]:
        ax.set_xlabel('token column')
    fig.colorbar(plt.cm.ScalarMappable(norm=na, cmap=CMAP), ax=axes[0], shrink=0.8).set_label('decay α')
    fig.colorbar(plt.cm.ScalarMappable(norm=nb, cmap=CMAP), ax=axes[1], shrink=0.8).set_label('write strength β')
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- probes

ROW_LAYOUTS = ['delete_row_mid', 'delete_row0', 'dup_row_mid', 'swap_rows_34', 'roll_rows+1', 'roll_rows-1']
COL_LAYOUTS = ['dup_col_mid', 'delete_col_mid', 'roll_within+1']
BLANK_LAYOUTS = ['insert_blank_start', 'insert_blank_mid']


def probe_stats(run: Path) -> dict:
    d = torch.load(run / 'probe' / 'preds.pt', weights_only=False)
    cells, preds = d['cells'], d['preds']
    lay = np.array([c['layout'] for c in cells])
    img = np.array([c['image'] for c in cells])
    key = {(c['image'], c['vit_row'], c['vit_col']): i for i, c in enumerate(cells) if c['layout'] == 'identity'}
    base_idx = np.array([key[(c['image'], c['vit_row'], c['vit_col'])] for c in cells])
    layers = sorted({int(k.split('_L')[1]) for k in preds})
    out = {'layers': layers, 'row': {}, 'col': {}}
    for axis, layouts, vit, ordk in (('row', ROW_LAYOUTS + BLANK_LAYOUTS, 'vit_row', 'ord_row'),
                                     ('col', COL_LAYOUTS, 'vit_col', 'ord_col')):
        v = np.array([c[vit] for c in cells], float)
        o = np.array([c[ordk] for c in cells], float)
        s = np.array([c['slot_row'] for c in cells], float)
        for name in layouts:
            sel = lay == name
            ref = s if name in BLANK_LAYOUTS else o
            # Cells displaced by exactly one cell (wrapped rows of the roll layouts are excluded).
            aff = sel & (np.abs(ref - v) > 0.5) & (np.abs(ref - v) < 1.5)
            rows = []
            for l in layers:
                p = preds[f'{axis}_L{l}'].numpy().astype(float)
                shift = (p - p[base_idx]) / np.where(aff, ref - v, np.nan)
                per_img = [np.nanmean(shift[aff & (img == i)]) for i in np.unique(img[aff])]
                rows.append(boot(per_img))
            out[axis][name] = {'n_cells': int(aff.sum()), 'by_layer': rows,
                               'reference': 'slot_row' if name in BLANK_LAYOUTS else ordk}
    res = json.loads((run / 'probe' / 'results.json').read_text())
    out['pooled'] = res['pooled']
    out['pooled_head'] = res['pooled_head']
    return out


def fig_probe_transfer(ps: dict, path: Path) -> None:
    layers = ps['layers']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    colors = {'delete_row_mid': BLUE, 'delete_row0': AQUA, 'dup_row_mid': VIOLET, 'swap_rows_34': ORANGE,
              'roll_rows+1': YELLOW, 'roll_rows-1': '#8a8a8a', 'insert_blank_start': INK, 'insert_blank_mid': '#b3b3b3',
              'dup_col_mid': VIOLET, 'delete_col_mid': BLUE, 'roll_within+1': YELLOW}
    labels = {'delete_row_mid': 'delete cell row 3', 'delete_row0': 'delete top row', 'dup_row_mid': 'duplicate row 3',
              'swap_rows_34': 'swap rows 3/4', 'roll_rows+1': 'roll rows +1 (bottom row first)',
              'roll_rows-1': 'roll rows −1 (top row last)', 'insert_blank_start': 'blank row at start (vs slot row)',
              'insert_blank_mid': 'blank row after row 3 (vs slot row)', 'dup_col_mid': 'duplicate column 3',
              'delete_col_mid': 'delete column 3', 'roll_within+1': 'roll within rows +1'}
    for ax, axis in zip(axes, ('row', 'col')):
        for name, st in ps[axis].items():
            m = [r['mean'] for r in st['by_layer']]
            lo = [r['ci'][0] for r in st['by_layer']]
            hi = [r['ci'][1] for r in st['by_layer']]
            ls = '--' if name in BLANK_LAYOUTS else '-'
            ax.plot(layers, m, ls=ls, color=colors[name], marker='o', ms=3, label=labels[name])
            ax.fill_between(layers, lo, hi, color=colors[name], alpha=0.12, linewidth=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axhline(1, color=MUTED, lw=0.8, ls=':')
        ax.axvspan(9, 16, color=GRID, alpha=0.6, zorder=0)
        ax.set_xlabel('layer (0 = decoder input)')
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel('probe follow ratio (shift / predicted shift)')
    axes[0].set_title('row probe trained on raster order, tested on edited layouts', fontsize=10)
    axes[1].set_title('column probe trained on raster order', fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def fig_probe_heads(ps: dict, gs: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    layers = sorted(int(l) for l in ps['pooled'])
    for key, color, name in (('ord_row', BLUE, 'figure-row ordinal'), ('vit_row', ORANGE, 'ViT row'),
                             ('slot_row', MUTED, 'slot row')):
        axes[0].plot(layers, [ps['pooled'][str(l)]['row'][key] for l in layers], color=color, marker='o', ms=3, label=name)
    for key, color, name in (('ord_col', BLUE, 'figure-column ordinal'), ('vit_col', ORANGE, 'ViT column')):
        axes[0].plot(layers, [ps['pooled'][str(l)]['col'][key] for l in layers], color=color, ls='--', marker='s', ms=3,
                     label=name)
    axes[0].set_xlabel('layer')
    axes[0].set_ylabel('held-out R²')
    axes[0].set_title('pooled probes on dissociating layouts (residual)', fontsize=10)
    axes[0].legend(fontsize=7, frameon=False)
    lab = np.array(gs['classes'])
    marks = {3: ('s', AQUA, 'cell-row-onset head'), 2: ('^', ORANGE, 'row-reset head'), 1: ('o', BLUE, 'integrator head'),
             0: ('.', MUTED, 'other head')}
    for l in [str(x) for x in MID]:
        i = gs['layers'].index(int(l))
        for h, st in enumerate(ps['pooled_head'][l]):
            m, c, _ = marks[lab[i, h]]
            axes[1].scatter(st['col']['ord_col'] - st['col']['vit_col'], st['row']['ord_row'] - st['row']['vit_row'],
                            marker=m, color=c, s=18, alpha=0.8)
    axes[1].legend(handles=[Line2D([], [], marker=m, color=c, ls='', label=n) for m, c, n in marks.values()],
                   fontsize=7, frameon=False)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].axvline(0, color=MUTED, lw=0.8)
    axes[1].set_xlabel('R²(column ordinal) − R²(ViT column)')
    axes[1].set_ylabel('R²(row ordinal) − R²(ViT row)')
    axes[1].set_title('single heads, DeltaNet layers 9–16 (read-out o_t)', fontsize=10)
    fig.savefig(path)
    plt.close(fig)


# Manifold views follow the style of the PI reference figure (shaded spheres, floor shadow, two views).
def _shade(rgba: np.ndarray, f: float) -> np.ndarray:
    out = rgba.copy()
    out[:, :3] = np.clip(out[:, :3] * f, 0, 1)
    return out


def manifold_view(ax, X: np.ndarray, c: np.ndarray, markers: np.ndarray, norm, elev: float, azim: float) -> None:
    lo, hi = X.min(0), X.max(0)
    pad = 0.12
    z0 = lo[2] - 0.35
    x0, x1, y0, y1 = lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad
    ax.computed_zorder = False
    ax.add_collection3d(Poly3DCollection([[(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]],
                                         facecolor=(0.97, 0.97, 0.97, 0.6), edgecolor='0.35', linewidth=0.6, zorder=0))
    ax.scatter(X[:, 0], X[:, 1], np.full(len(X), z0 + 1e-3), s=26, c='0.55', alpha=0.10, edgecolors='none',
               depthshade=False, zorder=1)
    rgba = plt.get_cmap(CMAP)(norm(c))
    for mk in np.unique(markers):
        s = markers == mk
        ax.scatter(X[s, 0], X[s, 1], X[s, 2], s=24, marker=mk, c=rgba[s], edgecolors=_shade(rgba[s], 0.55),
                   linewidths=0.35, depthshade=True, zorder=2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(z0, hi[2] + pad)
    ax.set_axis_off()


def frame(fig, ax, inset: float = 0.004) -> None:
    b = ax.get_position()
    fig.add_artist(Rectangle((b.x0 + inset, b.y0 + inset), b.width - 2 * inset, b.height - 2 * inset,
                             transform=fig.transFigure, fill=False, edgecolor='0.3', linewidth=0.6))


def fig_manifold(run: Path, path: Path) -> None:
    """Cell features (mean over probe images) of raster order (circles) and 'delete cell row 3' (triangles)."""
    d = torch.load(run / 'probe' / 'features.pt', weights_only=False)
    cells = d['cells']
    lay = np.array([c['layout'] for c in cells])
    panels = []
    for l in (8, 16, 24):
        F_ = d['res'][l].float().numpy()
        pts, col, mk = [], [], []
        for name, marker in (('identity', 'o'), ('delete_row_mid', '^')):
            groups = {}
            for i in np.where(lay == name)[0]:
                groups.setdefault((cells[i]['vit_row'], cells[i]['vit_col']), []).append(i)
            for (vr, vc), ids in sorted(groups.items()):
                pts.append(F_[ids].mean(0))
                col.append(cells[ids[0]]['ord_row'])
                mk.append(marker)
        P = np.array(pts)
        P = P - P.mean(0)
        U, S, Vt = np.linalg.svd(P, full_matrices=False)
        X = P @ Vt[:3].T
        X = X / np.abs(X).max()
        panels.append((l, X, np.array(col, float), np.array(mk)))
    n = len(panels)
    fig = plt.figure(figsize=(3.2 * n + 1.0, 6.6))
    norm = mcolors.Normalize(0, 7)
    axes = []
    for j, (l, X, c, mk) in enumerate(panels):
        a1 = fig.add_axes([0.03 + j * (0.88 / n), 0.49, 0.88 / n - 0.01, 0.40], projection='3d')
        a2 = fig.add_axes([0.03 + j * (0.88 / n), 0.06, 0.88 / n - 0.01, 0.40], projection='3d')
        manifold_view(a1, X, c, mk, norm, 22, -58)
        manifold_view(a2, X, c, mk, norm, 89.9, -90)
        xc = 0.03 + j * (0.88 / n) + (0.88 / n - 0.01) / 2
        fig.text(xc, 0.945, f'layer {l}', ha='center', fontsize=10)
        fig.text(xc, 0.915, 'Qwen3.5-9B residual, cell means (PCA 3D)', ha='center', fontsize=7, color='0.35')
        axes += [a1, a2]
    for a in axes:
        frame(fig, a)
    fig.text(0.008, 0.69, 'side view', rotation=90, va='center', fontsize=8, color='0.35')
    fig.text(0.008, 0.26, 'top view', rotation=90, va='center', fontsize=8, color='0.35')
    cax = fig.add_axes([0.935, 0.25, 0.012, 0.5])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax).set_label('figure-row ordinal (counted, 0-based)')
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color='0.4', label='raster order'),
                        Line2D([], [], marker='^', ls='', color='0.4', label='cell row 3 deleted')],
               loc='lower right', fontsize=7, frameon=False)
    fig.savefig(path)
    plt.close(fig)


def propagation_stats(run: Path, round1: Path) -> dict:
    """How far a row edit propagates: probe shift of later cell rows (this run) and round-1 answer shifts by target row."""
    d = torch.load(run / 'probe' / 'preds.pt', weights_only=False)
    cells, preds = d['cells'], d['preds']
    lay = np.array([c['layout'] for c in cells])
    tpl = np.array([c['template'] for c in cells])
    img = np.array([c['image'] for c in cells])
    key = {(c['image'], c['vit_row'], c['vit_col']): i for i, c in enumerate(cells) if c['layout'] == 'identity'}
    base = np.array([key[(c['image'], c['vit_row'], c['vit_col'])] for c in cells])
    vr = np.array([c['vit_row'] for c in cells])
    orr = np.array([c['ord_row'] for c in cells])
    out = {'probe': {}, 'behavior': {}}
    for l in (8, 16, 32):
        p = preds[f'row_L{l}'].numpy().astype(float)
        for t in ('person', 'ring', 'all'):
            tsel = np.ones(len(cells), bool) if t == 'all' else tpl == t
            for name, rows, sign in (('delete_row_mid', range(4, 8), -1), ('dup_row_mid', range(4, 8), 1),
                                     ('roll_rows+1', range(0, 7), 1)):
                vals = []
                for r in rows:
                    s_ = tsel & (lay == name) & (vr == r)
                    if name == 'dup_row_mid':
                        s_ &= orr == r + 1
                    per = [float(((p[s_ & (img == i)] - p[base[s_ & (img == i)]]) * sign).mean()) for i in np.unique(img[s_])]
                    vals.append(boot(per))
                out['probe'][f'{name}|{t}|L{l}'] = vals
    recs = read_jsonl(round1 / 'behavior' / 'records_edit.jsonl')
    for name in ('delete_row0', 'delete_row_before', 'dup_row_start', 'dup_row_before'):
        for t in ('person', 'ring'):
            acc = {}
            for r in recs:
                if r['template'] != t:
                    continue
                c = {x['name']: x for x in r['conditions']}
                if name not in c or value(c[name], 'row') is None or value(c['identity'], 'row') is None:
                    continue
                acc.setdefault(r['target'][0], []).append((value(c[name], 'row') - value(c['identity'], 'row')) / c[name]['count'])
            out['behavior'][f'{name}|{t}'] = {int(k): boot(v) for k, v in sorted(acc.items())}
    return out


def fig_propagation(pr: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    ax = axes[0]
    for t, color in (('person', BLUE), ('ring', ORANGE)):
        for name, ls, lab in (('delete_row0', '-', 'top row deleted'), ('dup_row_start', ':', 'row duplicated at start')):
            st = pr['behavior'][f'{name}|{t}']
            ks = sorted(st)
            ax.plot(ks, [st[k]['mean'] for k in ks], ls=ls, marker='o', ms=3, color=color, label=f'{t}: {lab}')
    ax.set_xlabel('target row r (0-based)')
    ax.set_ylabel('answer follow ratio (round 1, 8 images per point)')
    ax.set_title('answer: does a row edit at the image start reach row r?', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for ax, name, title in ((axes[1], 'delete_row_mid', 'probe: cell row 3 deleted'), (axes[2], 'dup_row_mid', 'probe: cell row 3 duplicated')):
        for l, ls in ((8, ':'), (16, '-'), (32, '--')):
            for t, color in (('person', BLUE), ('ring', ORANGE)):
                st = pr['probe'][f'{name}|{t}|L{l}']
                ax.plot(range(1, 5), [v['mean'] for v in st], ls=ls, marker='o', ms=3, color=color)
        ax.set_xlabel('cell rows after the edit (1 = first row after it)')
        ax.set_ylabel('probe shift toward the count (cells; count predicts 1)')
        ax.set_title(title, fontsize=9)
        ax.axhline(1, color=MUTED, lw=0.8, ls=':')
        ax.axhline(0, color=MUTED, lw=0.8)
    axes[1].legend(handles=[Line2D([], [], color=BLUE, label='person'), Line2D([], [], color=ORANGE, label='ring'),
                            Line2D([], [], color=INK, ls=':', label='layer 8'), Line2D([], [], color=INK, ls='-', label='layer 16'),
                            Line2D([], [], color=INK, ls='--', label='layer 32')], fontsize=7, frameon=False)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- interventions

def ok(a: dict, key: str, truth: int) -> bool:
    return a.get(key) == truth


def intervene_stats(run: Path) -> dict:
    recs = read_jsonl(run / 'intervene' / 'records.jsonl')
    out = {'n_images': len(recs)}
    by = {}
    for rec in recs:
        r, c = rec['target']
        base = {cd['base']: cd for cd in rec['conditions'] if cd['name'] == 'none'}
        for cd in rec['conditions']:
            k = (cd['base'], cd['name'] + (f"|k" if 'k' in cd else ''))
            e = by.setdefault(k, {'row_ok': [], 'col_ok': [], 'valid': [], 'f_row': [], 'f_col': [], 'exact_pred': [],
                                  'row_shift': [], 'col_shift': []})
            e['valid'].append(bool(cd.get('valid')))
            e['row_ok'].append(ok(cd, 'row', r))
            e['col_ok'].append(ok(cd, 'col', c))
            # Order effects are measured against the raster-order input under the same intervention, so that
            # a uniform deficit caused by the intervention itself is not counted as following the slot.
            same = next((x for x in rec['conditions'] if x['base'] == 'identity' and x['name'] == cd['name']
                         and x.get('k') == cd.get('k')), base['identity'])
            er, eb = value(cd, 'row'), value(same, 'row')
            ec, ebc = value(cd, 'col'), value(same, 'col')
            e0r, e0c = value(base['identity'], 'row'), value(base['identity'], 'col')
            if cd['base'] == 'identity' and er is not None and e0r is not None:
                e['row_shift'].append(er - e0r)
            if cd['base'] == 'identity' and ec is not None and e0c is not None:
                e['col_shift'].append(ec - e0c)
            if cd['base'] == 'delete_row_before' and er is not None and eb is not None:
                e['f_row'].append((er - eb) / -1.0)
                e['exact_pred'].append(cd.get('row') == r - 1)
            if cd['base'] == 'block_down' and er is not None and eb is not None:
                e['f_row'].append(er - eb)
            if cd['base'] == 'block_right' and ec is not None and ebc is not None:
                e['f_col'].append(ec - ebc)
            if 'k' in cd and er is not None and e0r is not None:
                e['f_row'].append((er - e0r) / -cd['k'])
                e['exact_pred'].append(cd.get('row') == r - cd['k'])
    for (base, name), e in by.items():
        n = len(e['row_ok'])
        out[f'{base}/{name}'] = {
            'n': n, 'valid': sum(e['valid']),
            'row_acc': sum(e['row_ok']) / n, 'row_acc_ci': wilson(sum(e['row_ok']), n),
            'col_acc': sum(e['col_ok']) / n, 'col_acc_ci': wilson(sum(e['col_ok']), n),
            'f_row': boot(e['f_row']) if e['f_row'] else None, 'f_col': boot(e['f_col']) if e['f_col'] else None,
            'exact_pred': (sum(e['exact_pred']), len(e['exact_pred'])) if e['exact_pred'] else None,
            'mean_row_shift': float(np.mean(e['row_shift'])) if e['row_shift'] else None,
            'mean_abs_row_shift': float(np.mean(np.abs(e['row_shift']))) if e['row_shift'] else None,
            'mean_abs_col_shift': float(np.mean(np.abs(e['col_shift']))) if e['col_shift'] else None}
    return out


def deficit_by_row(run: Path, stage: str = 'intervene') -> dict:
    """Mean shift of the expected row (intervention minus no intervention, raster order) by target row."""
    recs = read_jsonl(run / stage / 'records.jsonl')
    acc = {}
    for rec in recs:
        c = {x['name']: x for x in rec['conditions'] if x['base'] == 'identity'}
        b = value(c['none'], 'row')
        for name, cd in c.items():
            a = value(cd, 'row')
            if 'k' in cd or a is None or b is None:
                continue
            acc.setdefault(name, {}).setdefault(rec['target'][0], []).append(a - b)
    return {n: {int(r): boot(v) for r, v in sorted(d.items())} for n, d in acc.items()}


def fig_deficit(dr: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True, sharey=True)
    sets = [[('reset_rowstart|early', AQUA, 'reset at token-row starts, L1–8'),
             ('reset_rowstart|mid', BLUE, 'reset at token-row starts, L9–16'),
             ('reset_rowstart|late', MUTED, 'reset at token-row starts, L17–24'),
             ('nodecay|mid', VIOLET, 'α=1 on image tokens, L9–16')],
            [('nowrite_all|early', AQUA, 'β=0 on image tokens, L1–8'), ('nowrite_all|mid', BLUE, 'β=0 on image tokens, L9–16'),
             ('nowrite_all|late', MUTED, 'β=0 on image tokens, L17–24'), ('nowrite_top|mid', ORANGE, 'β=0 on top token rows, L9–16'),
             ('nowrite_bottom|mid', YELLOW, 'β=0 on bottom token rows, L9–16')]]
    for ax, items in zip(axes, sets):
        for name, color, label in items:
            if name not in dr:
                continue
            st = dr[name]
            ks = sorted(st)
            m = np.array([st[k]['mean'] for k in ks])
            ax.plot(ks, m, marker='o', ms=3, color=color, label=label)
            ax.fill_between(ks, [st[k]['ci'][0] for k in ks], [st[k]['ci'][1] for k in ks], color=color, alpha=0.12, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel('target row r (0-based)')
        ax.legend(fontsize=7, frameon=False, loc='lower left')
    axes[0].set_ylabel('shift of expected row (rows)')
    axes[0].set_title('erasing or freezing the state: deficit grows with r', fontsize=10)
    axes[1].set_title('removing writes: deficit grows with r', fontsize=10)
    fig.savefig(path)
    plt.close(fig)


IDENT_JOBS = [('none', 'none'), ('reset_rowstart|early', 'reset at row starts, L1–8'),
              ('reset_rowstart|mid', 'reset at row starts, L9–16'), ('reset_rowstart|late', 'reset at row starts, L17–24'),
              ('nowrite_all|early', 'β=0 all image tokens, L1–8'), ('nowrite_all|mid', 'β=0 all image tokens, L9–16'),
              ('nowrite_all|late', 'β=0 all image tokens, L17–24'), ('nowrite_top|mid', 'β=0 top token rows, L9–16'),
              ('nowrite_bottom|mid', 'β=0 bottom token rows, L9–16'), ('nodecay|mid', 'α=1 all image tokens, L9–16'),
              ('nowrite_target|mid', 'β=0 target tokens, L9–16'), ('nowrite_target|all', 'β=0 target tokens, all DeltaNet')]


def fig_intervene(st: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True, gridspec_kw={'width_ratios': [1.6, 1]})
    ax = axes[0]
    y = np.arange(len(IDENT_JOBS))
    for off, key, color, name in ((-0.18, 'row', BLUE, 'row correct'), (0.18, 'col', ORANGE, 'column correct')):
        vals = [st[f'identity/{j}'][f'{key}_acc'] for j, _ in IDENT_JOBS]
        ci = np.array([st[f'identity/{j}'][f'{key}_acc_ci'] for j, _ in IDENT_JOBS])
        ax.barh(y + off, vals, height=0.34, color=color, label=name)
        ax.errorbar(vals, y + off, xerr=np.clip([np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)], 0, None), fmt='none', ecolor=INK, lw=0.8)
    ax.set_yticks(y, [n for _, n in IDENT_JOBS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('fraction correct (raster order, 128 images, Wilson 95% CI)')
    ax.legend(fontsize=8, frameon=True, loc='upper left', bbox_to_anchor=(0.0, -0.12), ncol=2)
    ax.set_title('gate interventions on the raster-order input', fontsize=10)
    ax = axes[1]
    rows = [('delete_row_before', 'none', 'delete row r−1'), ('delete_row_before', 'reset_rowstart|mid', '+ reset row starts L9–16'),
            ('delete_row_before', 'nowrite_all|mid', '+ β=0 L9–16'), ('block_down', 'none', 'block move down'),
            ('block_down', 'reset_rowstart|mid', '+ reset row starts L9–16'), ('block_down', 'nowrite_all|mid', '+ β=0 L9–16'),
            ('block_down', 'nowrite_target|mid', '+ β=0 target L9–16'), ('block_right', 'none', 'block move right (column)'),
            ('block_right', 'reset_rowstart|mid', '+ reset row starts L9–16'), ('block_right', 'nowrite_target|mid', '+ β=0 target L9–16')]
    y = np.arange(len(rows))
    for i, (b, j, n) in enumerate(rows):
        e = st.get(f'{b}/{j}')
        if e is None:
            continue
        f = e['f_col'] if b == 'block_right' else e['f_row']
        color = ORANGE if b == 'block_right' else BLUE
        ax.barh(i, f['mean'], color=color, height=0.6)
        ax.errorbar(f['mean'], i, xerr=np.clip([[f['mean'] - f['ci'][0]], [f['ci'][1] - f['mean']]], 0, None), fmt='none', ecolor=INK, lw=0.8)
    ax.set_yticks(y, [n for *_, n in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(1, color=MUTED, lw=0.8, ls=':')
    ax.set_xlabel('follow ratio (reference: raster order\nunder the same intervention; 95% CI)')
    ax.set_title('order effects under gate interventions', fontsize=10)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- restoration, transplant, think

def restore2_stats(run: Path) -> dict:
    recs = read_jsonl(run / 'restore2' / 'records.jsonl')
    out = {'n_images': len(recs)}
    acc = {}
    for rec in recs:
        conds = rec['conditions']
        r, c = rec['target']
        for cd in conds:
            ax = 0 if cd['axis'] == 'v' else 1
            key = ('row', 'col')[ax]
            d = cd['slot_cell'][ax] - (r, c)[ax]
            a = value(cd, key)
            truth = (r, c)[ax]
            name = cd['name'].split('|')[1]
            acc.setdefault((cd['move'], name), []).append(None if a is None else (a - truth) / d)
    for (move, name), v in acc.items():
        out[f'{move}|{name}'] = boot(v)
    return out


SUBSETS = [('none', 'no restoration'), ('target', 'target slots'), ('others', 'all other slots'),
           ('before_target', 'slots before target'), ('after_samerow', 'after target, same cell row'),
           ('after_samerow_mid', 'same, layers 9–16 only'), ('later_rows', 'later cell rows'),
           ('next2', 'next 2 slots'), ('next4', 'next 4'), ('next8', 'next 8'), ('next16', 'next 16'), ('next32', 'next 32')]


def fig_restore2(st: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True, constrained_layout=True)
    for ax, move in zip(axes, ('down', 'up', 'right', 'left')):
        color = BLUE if move in ('down', 'up') else ORANGE
        for i, (k, n) in enumerate(SUBSETS):
            e = st.get(f'{move}|{k}')
            if e is None or not np.isfinite(e['mean']):
                continue
            ax.barh(i, e['mean'], color=color, height=0.6)
            ax.errorbar(e['mean'], i, xerr=np.clip([[e['mean'] - e['ci'][0]], [e['ci'][1] - e['mean']]], 0, None), fmt='none', ecolor=INK, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f'block move {move} ({"row" if color == BLUE else "column"} follow)', fontsize=10)
        ax.set_xlabel('follow ratio (bootstrap 95% CI)')
    axes[0].set_yticks(range(len(SUBSETS)), [n for _, n in SUBSETS], fontsize=8)
    axes[0].invert_yaxis()
    fig.savefig(path)
    plt.close(fig)


def transplant_stats(run: Path, stage: str = 'transplant') -> dict:
    recs = read_jsonl(run / stage / 'records.jsonl')
    acc = {}
    for rec in recs:
        base = rec['conditions'][0]
        for cd in rec['conditions'][1:]:
            for key in ('row', 'col'):
                a, b = value(cd, key), value(base, key)
                on_axis = (key == 'row') == (cd['axis'] == 'v')
                acc.setdefault((cd['name'], key), []).append(None if a is None or b is None else
                                                             ((a - b) / cd['d'] if on_axis else a - b))
    return {'n_images': len(recs), **{f'{n}|{k}': boot(v) for (n, k), v in acc.items()}}


def intervene2_stats(run: Path) -> dict:
    """Row answers under supplementary gate overrides: follow ratios against the raster-order input under the same
    override (expected rows), and the raw shift of the raster-order input itself."""
    recs = read_jsonl(run / 'intervene2' / 'records.jsonl')
    acc = {}
    pred = {'delete_row_before': -1, 'roll_rows-1': -1, 'block_down': 1}
    for rec in recs:
        r = rec['target'][0]
        conds = {(x['base'], x['name']): x for x in rec['conditions']}
        for (base, name), cd in conds.items():
            a = value(cd, 'row')
            ref = value(conds[('identity', name)], 'row')
            ref0 = value(conds[('identity', 'none')], 'row')
            e = acc.setdefault(f'{base}/{name}', {'f': [], 'f_raw': [], 'ok': [], 'shift': []})
            e['ok'].append(cd.get('row') == r)
            if base == 'identity':
                if a is not None and ref0 is not None:
                    e['shift'].append(a - ref0)
                continue
            if a is not None and ref is not None:
                e['f'].append((a - ref) / pred[base])
            if a is not None and ref0 is not None:
                e['f_raw'].append((a - ref0) / pred[base])
    out = {'n_images': len(recs)}
    for k, e in acc.items():
        n = len(e['ok'])
        out[k] = {'n': n, 'row_acc': sum(e['ok']) / n, 'row_acc_ci': wilson(sum(e['ok']), n),
                  'f': boot(e['f']) if e['f'] else None, 'f_raw': boot(e['f_raw']) if e['f_raw'] else None,
                  'shift': boot(e['shift']) if e['shift'] else None}
    return out


def think_stats(run: Path) -> dict:
    recs = read_jsonl(run / 'think' / 'records.jsonl')
    out = {'n_images': len(recs)}
    table = {}
    for rec in recs:
        r, c = rec['target']
        for cd in rec['conditions']:
            k = (cd['mode'], cd['name'])
            e = table.setdefault(k, {'valid': 0, 'correct': 0, 'follow': [], 'n': 0, 'len': []})
            e['n'] += 1
            e['len'].append(len(cd['text']))
            if cd['row'] is None:
                e['follow'].append(None)
                continue
            e['valid'] += 1
            e['correct'] += int(cd['row'] == r and cd['col'] == c)
            if cd['axis'] is not None:
                ax = 0 if cd['axis'] == 'v' else 1
                ans = (cd['row'], cd['col'])[ax]
                e['follow'].append(int(round(ans - (r, c)[ax]) == cd['d']))
    for (mode, name), e in table.items():
        fl = [x for x in e['follow'] if x is not None]
        out[f'{mode}|{name}'] = {'n': e['n'], 'valid': e['valid'], 'correct': e['correct'],
                                 'follow': sum(fl), 'n_follow': len(fl), 'follow_ci': wilson(sum(fl), len(fl)),
                                 'mean_chars': float(np.mean(e['len']))}
    # Paired McNemar on "answer follows the slot" between direct and think, per move.
    by_img = {}
    for rec in recs:
        r, c = rec['target']
        for cd in rec['conditions']:
            if cd['axis'] is None:
                continue
            ax = 0 if cd['axis'] == 'v' else 1
            f = None if cd['row'] is None else int(round((cd['row'], cd['col'])[ax] - (r, c)[ax]) == cd['d'])
            by_img.setdefault((rec['id'], cd['name']), {})[cd['mode']] = f
    for name in ('block_down', 'block_up', 'block_right', 'block_left'):
        pairs = [v for (i, n), v in by_img.items() if n == name and v.get('direct') is not None and v.get('think') is not None]
        b = sum(1 for v in pairs if v['direct'] == 1 and v['think'] == 0)
        c_ = sum(1 for v in pairs if v['direct'] == 0 and v['think'] == 1)
        out[f'mcnemar|{name}'] = {'direct_only': b, 'think_only': c_, 'p': mcnemar(b, c_), 'n_pairs': len(pairs)}
    return out


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ----------------------------------------------------------------------------- schematic

def fig_schematic(path: Path) -> None:
    """Plain-language schematic of the DeltaNet recurrence on a raster-ordered grid and of block moves."""
    fig = plt.figure(figsize=(14, 7.4))
    RED, CELL_A, CELL_B = '#e8826b', '#dfe9f7', '#e3f2e9'
    # (a) the image as cells of 2x2 tokens, read in raster order.
    ax = fig.add_axes([0.01, 0.40, 0.22, 0.55])
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 8.9)
    ax.axis('off')
    for R in range(8):
        for C in range(8):
            cell = (R // 2, C // 2)
            fc = RED if cell == (1, 1) else (CELL_A if (R // 2) % 2 == 0 else CELL_B)
            ax.add_patch(Rectangle((0.3 + C, 7.9 - R), 0.92, 0.92, facecolor=fc, edgecolor='white', lw=0.6))
        ax.annotate('', xy=(8.45, 8.35 - R), xytext=(0.15, 8.35 - R),
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.5, alpha=0.6))
    for k in range(4):
        ax.text(0.05, 7.85 - 2 * k, f'cell row {k}', rotation=90, fontsize=7, color=MUTED, va='center', ha='right')
    ax.set_title('(a) image = 4x4 cells, each 2x2 tokens;\ntokens enter the sequence row by row', fontsize=9)
    # (b) gate behaviour along the sequence for cell rows 0 and 1 (four token rows of 8 tokens).
    ax = fig.add_axes([0.27, 0.40, 0.72, 0.55])
    n = 32
    ax.set_xlim(-7.5, n + 0.5)
    ax.set_ylim(-0.3, 10.6)
    ax.axis('off')
    for i in range(n):
        R, C = i // 8, i % 8
        fc = RED if (R // 2, C // 2) == (1, 1) else (CELL_A if R < 2 else CELL_B)  # same target cell (1, 1) as panel (a)
        ax.add_patch(Rectangle((i + 0.05, 9.4), 0.9, 0.8, facecolor=fc, edgecolor='white'))
    for R in range(4):
        ax.plot([R * 8, R * 8], [0.0, 10.3], color=MUTED, lw=0.9 if R % 2 == 0 else 0.5, ls='-' if R % 2 == 0 else ':')
        ax.text(R * 8 + 4, 10.35, f'token row {R}', ha='center', fontsize=7, color=MUTED)
    t = np.arange(n) + 0.5
    reset, onset, est = [], [], []
    m1 = m2 = 0.0
    for i in range(n):
        if i % 8 == 0:
            m1 *= 0.1
        if i % 16 == 0:
            m2 *= 0.1
        m1 = 0.8 * m1 + 0.25
        m2 = 0.9 * m2 + 0.12
        reset.append(m1)
        onset.append(m2)
        est.append(float(i // 16))
    ax.plot(t, 6.6 + 1.6 * np.array(reset), color=ORANGE, lw=1.8)
    ax.plot(t, 3.6 + 1.6 * np.array(onset), color=AQUA, lw=1.8)
    ax.step(t, 0.6 + 1.6 * np.array(est), where='mid', color=BLUE, lw=1.8)
    ax.text(-7.3, 7.6, 'row-reset heads\nα ≈ 0.3–0.5 at every\ntoken-row start', fontsize=8, color=ORANGE, va='center')
    ax.text(-7.3, 4.6, 'cell-row-onset heads\nα ≈ 0.3 only at the first\ntoken row of a cell row', fontsize=8, color=AQUA, va='center')
    ax.text(-7.3, 1.5, 'long-memory heads\n(α ≈ 1): row estimate\ncarried across rows', fontsize=8, color=BLUE, va='center')
    ax.text(n / 2, 9.0, 'memory of each head type along the sequence (schematic; heights in arbitrary units)',
            ha='center', fontsize=8, color=INK)
    ax.set_title('(b) what the gated, decaying state can keep: within-row history vs. a row estimate passed across rows', fontsize=9)
    # (c) block moves.
    for j, (title, body) in enumerate((
        ('block move down (row follows, 0.90)',
         'The target tokens are read inside the next cell row. The long-memory state there already\n'
         'holds that row\'s estimate (one more row); the tokens after the target read the row out.\n'
         'Cutting the state at token-row starts in layers 9–16 halves this effect (0.90 -> 0.53).'),
        ('block move right (column follows partly, 0.55 / 0.29)',
         'The row estimate is unchanged. The within-row history before the target changes by one\n'
         'cell, while the ViT context of the other tokens still places the red figure at its old\n'
         'column. Row-start resets in layers 9–16 leave this effect intact (0.55 -> 0.49).'))):
        ax = fig.add_axes([0.03 + 0.49 * j, 0.14, 0.46, 0.20])
        ax.axis('off')
        ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.95, boxstyle='round,pad=0.01', facecolor='#fafafa',
                                    edgecolor=GRID, transform=ax.transAxes))
        ax.text(0.04, 0.85, f'(c{j + 1}) {title}', fontsize=9, transform=ax.transAxes, va='top', weight='bold')
        ax.text(0.04, 0.55, body, fontsize=8.2, transform=ax.transAxes, va='top')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(8)
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--figures', required=True)
    args = p.parse_args()
    run, figs = Path(args.run), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    results = {}
    if (run / 'gates' / 'gates.pt').exists():
        g = load_gates(run)
        gs = gate_stats(g)
        results['gates'] = {k: v for k, v in gs.items() if k not in ('grid', 'white', 'natural')}
        for k in ('grid', 'white', 'natural'):
            results['gates'][f'{k}_counts'] = {q: gs[k][q] for q in ('n_parity_heads', 'n_parity_heads_mid', 'n_top_writer_heads')}
        results['gates']['summary'] = {k: {'max_drop': float(np.min(gs[k]['d_alpha_rowstart']))} for k in ('grid', 'white', 'natural')}
        (run / 'gates' / 'stats.json').write_text(json.dumps(gs))
        fig_gate_heatmaps(gs, figs / 'gates_rowstart.png')
        fig_gate_profiles(gs, figs / 'gates_profile.png')
        lab = np.array(gs['classes'])
        li = [gs['layers'].index(l) for l in MID]
        da = np.array(gs['grid']['d_alpha_rowstart'])
        reset = min(((l, h) for l in MID for h in range(32)), key=lambda t: da[gs['layers'].index(t[0]), t[1]])
        mem = np.array(gs['memory_median'])
        integ = max(((l, h) for l in MID for h in range(32) if lab[gs['layers'].index(l), h] == 1),
                    key=lambda t: mem[gs['layers'].index(t[0]), t[1]], default=(15, 0))
        par = np.abs(np.array(gs['grid']['parity_d_alpha_rowstart']))
        topb = max(((l, h) for l in MID for h in range(32)), key=lambda t: par[gs['layers'].index(t[0]), t[1]])
        results['gates']['example_heads'] = {'reset': reset, 'integrator': integ, 'onset': topb}
        fig_gate_maps(g, gs, [reset, integ, topb], figs / 'gates_maps.png')
    if (run / 'probe' / 'preds.pt').exists():
        ps = probe_stats(run)
        results['probe'] = ps
        fig_probe_transfer(ps, figs / 'probe_transfer.png')
        if 'gates' in results:
            fig_probe_heads(ps, gs, figs / 'probe_heads.png')
        fig_manifold(run, figs / 'manifold_rows.png')
        results['propagation'] = propagation_stats(run, Path(args.round1))
        fig_propagation(results['propagation'], figs / 'propagation.png')
    if (run / 'intervene' / 'records.jsonl').exists():
        results['intervene'] = intervene_stats(run)
        fig_intervene(results['intervene'], figs / 'intervene.png')
        results['deficit'] = deficit_by_row(run)
        fig_deficit(results['deficit'], figs / 'deficit.png')
    if (run / 'restore2' / 'records.jsonl').exists():
        results['restore2'] = restore2_stats(run)
        fig_restore2(results['restore2'], figs / 'restore2.png')
    if (run / 'transplant' / 'records.jsonl').exists():
        results['transplant'] = transplant_stats(run)
    if (run / 'transplant_ctx' / 'records.jsonl').exists():
        results['transplant_ctx'] = transplant_stats(run, 'transplant_ctx')
    if (run / 'intervene2' / 'records.jsonl').exists():
        results['intervene2'] = intervene2_stats(run)
    if (run / 'think' / 'records.jsonl').exists():
        results['think'] = think_stats(run)
    fig_schematic(figs / 'schematic.png')
    (run / 'results.json').write_text(json.dumps(results, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    print('wrote', run / 'results.json')


if __name__ == '__main__':
    main()
