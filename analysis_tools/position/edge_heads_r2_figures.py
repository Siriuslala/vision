"""Figures of experiment 3.11 round 4 (edge_heads_r2_analysis.py outputs); PNGs for the results doc.

Every attention overlay shows the image-normalized attention of one head at one query (from model activations),
the target box (white) and the inner edge bands of the region protocol (dashed), with a colorbar in units of
'attention per token / uniform attention per token' (1 = uniform)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np

from analysis_tools.position import edge_heads as eh
from analysis_tools.position import edge_heads_r2 as e2
from analysis_tools.position import edge_heads_r2_analysis as an
from analysis_tools.representation import sees_right as sr

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GRAY = '#9a9994'
STEP_TITLE = {'pt_x': 'point x', 'pt_y': 'point y', 'bb_x1': 'box x_min', 'bb_y1': 'box y_min',
              'bb_x2': 'box x_max', 'bb_y2': 'box y_max'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def draw_box(ax, box, grid, color='white'):
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec=color, lw=1.4))
    c0, c1, r0, r1 = e2.box_cells(box, grid)
    tp, b = e2.TP, e2.BAND
    if c1 - c0 + 1 >= 2 * b + 1 and r1 - r0 + 1 >= 2 * b + 1:
        for x0, y0, w, h in (((c0) * tp, r0 * tp, b * tp, (r1 - r0 + 1) * tp), ((c1 - b + 1) * tp, r0 * tp, b * tp, (r1 - r0 + 1) * tp),
                             (c0 * tp, r0 * tp, (c1 - c0 + 1) * tp, b * tp), (c0 * tp, (r1 - b + 1) * tp, (c1 - c0 + 1) * tp, b * tp)):
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec='#f0efec', lw=0.6, ls='--'))


def overlay(ax, img, p, grid, box, vmax, title=None):
    """p: (Nv,) image attention summing to 1; shown as p * Nv (1 = uniform)."""
    H, W = grid
    ax.imshow(np.asarray(img) * 0.45 / 255 + 0.0)
    m = (p * len(p)).reshape(H, W)
    im = ax.imshow(m, cmap='viridis', alpha=np.clip(m / vmax, 0.15, 0.85), vmin=0, vmax=vmax,
                   extent=(0, W * e2.TP, H * e2.TP, 0), interpolation='nearest')
    draw_box(ax, box, grid)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)
    return im


# ----------------------------------------------------------------------------- protocol figure

def fig_protocol(r3_cases: dict, case_id: str, out: Path) -> None:
    c = r3_cases[case_id]
    img = sr.load_image(c)
    W, H = c['canvas']
    grid = (H // e2.TP, W // e2.TP)
    lab = e2.region_labels(c['truth']['box'], grid).reshape(grid)
    colors = {'interior': '#c3c2b7', 'left': CAT[0], 'right': CAT[1], 'top': CAT[2], 'bottom': CAT[3],
              'top_left': CAT[6], 'top_right': CAT[4], 'bottom_left': CAT[5], 'bottom_right': CAT[7],
              'out_ring': '#52514e', 'out_col0': '#0b0b0b', 'out_far': '#fcfcfb'}
    cmap = ListedColormap([colors[r] for r in e2.REGIONS])
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    axs[0].imshow(img)
    draw_box(axs[0], c['truth']['box'], grid, 'yellow')
    axs[0].set_title(f"input ({c['expression']}); yellow = target box")
    axs[1].imshow(lab, cmap=cmap, vmin=-0.5, vmax=len(e2.REGIONS) - 0.5, extent=(0, W, H, 0))
    axs[1].set_title('region of every merged token (32 x 32 px)')
    for a in axs:
        a.set_xticks([]), a.set_yticks([])
    axs[1].legend(handles=[Patch(fc=colors[r], ec='#52514e', label=r) for r in e2.REGIONS], loc='center left',
                  bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'protocol.png', dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- head x value table

def fig_head_table(summary: dict, heads: list[str], out: Path, name: str = 'head_value_table.png',
                   causal: dict | None = None) -> None:
    """Rows heads, columns value steps; stacked bars of the across-edge shares (min band / middle / max band)."""
    steps = list(an.STEP_NAMES)
    reg = summary['r3_regions']
    fig, axs = plt.subplots(1, len(steps), figsize=(2.0 * len(steps) + 1.5, 0.32 * len(heads) + 1.4), sharey=True)
    for j, st in enumerate(steps):
        ax = axs[j]
        for i, h in enumerate(heads):
            d = reg[st]['heads'][h]
            a = d['across']
            left = 0
            for k, col in enumerate((CAT[0], '#c3c2b7', CAT[1])):
                ax.barh(i, a[k], left=left, color=col, height=0.72, edgecolor='white', linewidth=1)
                left += a[k]
            ax.text(1.02, i, f"{d['inbox']:.2f}", va='center', fontsize=7, color='#52514e')
            if causal is not None and h in causal.get(st, {}):
                ax.text(-0.04, i, f"{causal[st][h]:.2f}", va='center', ha='right', fontsize=7, color='#0b0b0b')
        e = reg[st]['heads'][heads[0]]['across_e']
        ax.axvline(e[0], color='#0b0b0b', lw=0.6, ls=':')
        ax.axvline(e[0] + e[1], color='#0b0b0b', lw=0.6, ls=':')
        ax.set_xlim(0, 1.18)
        ax.set_title(STEP_TITLE[st])
        ax.set_xticks([0, 0.5, 1])
    axs[0].set_yticks(range(len(heads)))
    axs[0].set_yticklabels(heads)
    axs[0].invert_yaxis()
    fig.legend(handles=[Patch(fc=CAT[0], label='min band (left at x / top at y)'), Patch(fc='#c3c2b7', label='middle'),
                        Patch(fc=CAT[1], label='max band (right at x / bottom at y)')],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.06, 'share of in-box attention (dotted: token-count shares)', ha='center')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / name, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- r3 overlays

def fig_r3_gallery(round3: Path, items: list[dict], heads: list[str], out: Path, name: str) -> None:
    """Rows: heads; columns: (image, value step) pairs for one or two images."""
    cases = eh.load_cases(round3)
    cols = []
    for it in items:
        for fmt, vi, st, _ in an.R3_STEPS:
            if fmt == it['format']:
                cols.append((it, vi, st))
    fig, axs = plt.subplots(len(heads), len(cols), figsize=(1.75 * len(cols) + 0.8, 1.75 * len(heads) + 0.6))
    vmax = 12
    im = None
    for j, (it, vi, st) in enumerate(cols):
        img = sr.load_image(cases[it['case']])
        for i, h in enumerate(heads):
            l, hh = h[1:].split('H')
            p = it['att'][vi, an.LI[int(l)], int(hh)]
            im = overlay(axs[i, j], img, p, it['grid'], it['box'], vmax, STEP_TITLE[st] if i == 0 else None)
            if j == 0:
                axs[i, j].set_ylabel(h, fontsize=8)
    for i, h in enumerate(heads):
        axs[i, 0].set_yticks([])
        axs[i, 0].set_ylabel(h)
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01)
    cb.set_label('attention / uniform attention per token')
    fig.savefig(out / name, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- probe figures

def fig_scan(summary: dict, tags: list[str], titles: list[str], out: Path, name: str) -> None:
    """Single-head patching r (layers x heads) for each tag; diverging colour, 0 = gray."""
    fig, axs = plt.subplots(1, len(tags), figsize=(3.3 * len(tags) + 0.8, 3.0))
    axs = np.atleast_1d(axs)
    vmax = 0.3
    im = None
    for ax, tag, tt in zip(axs, tags, titles):
        d = summary['scan'].get(tag)
        if d is None:
            ax.set_visible(False)
            continue
        m = np.array([[d['mean'].get(f'{l}.{h}', np.nan) for h in range(16)] for l in e2.FULL_LAYERS])
        im = ax.imshow(m, cmap='PuOr_r', norm=TwoSlopeNorm(0, -vmax / 3, vmax), aspect='auto')
        for li in range(8):
            for h in range(16):
                if m[li, h] >= 0.05:
                    ax.text(h, li, f'{m[li, h]:.2f}'[1:], ha='center', va='center', fontsize=5.5,
                            color='white' if m[li, h] > 0.15 else '#0b0b0b')
        ax.set_yticks(range(8))
        ax.set_yticklabels([f'L{l}' for l in e2.FULL_LAYERS])
        ax.set_xticks(range(0, 16, 3))
        ax.set_xlabel('head')
        ax.set_title(f"{tt} (n={d['n']}; all heads {d['full_B'][0]:.2f})", fontsize=8)
    cb = fig.colorbar(im, ax=list(axs), fraction=0.02, pad=0.01)
    cb.set_label('first-digit logit recovery r')
    fig.savefig(out / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_digit_slots(summary: dict, heads: list[str], out: Path, st_set: str = 'nat') -> None:
    """Across-edge shares and in-box attention of key heads at the four query slots of each value step."""
    steps = list(an.STEP_NAMES)
    slots = [-1, 0, 1, 2]
    fig, axs = plt.subplots(2, len(steps), figsize=(2.3 * len(steps) + 1, 5.2), sharey='row')
    for j, st in enumerate(steps):
        for i, h in enumerate(heads):
            sk, inb, xs = [], [], []
            for s_ in slots:
                d = summary['profiles'].get(f'{st_set}:{st}:{s_}')
                if d is None or h not in d['heads']:
                    continue
                a = d['heads'][h]['across']
                sk.append(a[2] - a[0])
                inb.append(d['heads'][h]['inbox'])
                xs.append(s_)
            axs[0, j].plot(xs, sk, marker='o', ms=4, lw=1.5, color=CAT[i % 8], label=h)
            axs[1, j].plot(xs, inb, marker='o', ms=4, lw=1.5, color=CAT[i % 8], label=h)
        axs[0, j].axhline(0, color=GRAY, lw=0.6)
        axs[0, j].set_title(STEP_TITLE[st])
        for r_ in range(2):
            axs[r_, j].set_xticks(slots)
            axs[r_, j].set_xticklabels(['pre', 'd1', 'd2', 'd3'])
    axs[0, 0].set_ylabel('max-band share - min-band share')
    axs[1, 0].set_ylabel('attention inside box')
    axs[0, -1].legend(fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(out / 'digit_slots.png', dpi=150)
    plt.close(fig)


def fig_behaviour(run: Path, summary: dict, out: Path) -> None:
    """Answer shift (pixels) against the moved-edge displacement for one-edge moves."""
    beh = summary['behaviour']
    panels = [('point', 'x', ('left', 'right')), ('point', 'y', ('top', 'bottom')),
              ('bbox', 'x_min', ('left', 'right')), ('bbox', 'x_max', ('left', 'right'))]
    fig, axs = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    for r_, st in enumerate(('str', 'rect')):
        for j, (fmt, key, sides) in enumerate(panels):
            ax = axs[r_, j]
            for k, side in enumerate(sides):
                d = beh.get(f'{st}:{fmt}:{side}:{key}')
                if d is None:
                    continue
                xs = sorted(int(x) for x in d['median_shift_by_delta'])
                ys = [d['median_shift_by_delta'][str(x)][0] for x in xs]
                ax.plot(xs, ys, marker='o', ms=5, lw=2, color=CAT[k], label=f"{side} edge moves: slope {d['slope']:.2f}")
            ax.plot([-100, 100], [-50, 50], color=GRAY, lw=0.8, ls='--', label='slope 0.5')
            ax.plot([-100, 100], [-100, 100], color=GRAY, lw=0.8, ls=':', label='slope 1')
            ax.axhline(0, color=GRAY, lw=0.5)
            ax.set_title(f"{'natural' if st == 'str' else 'rectangle'}: {fmt} {key}", fontsize=9)
            ax.legend(fontsize=6.5, frameon=False, loc='upper left')
            if r_ == 1:
                ax.set_xlabel('signed displacement of the moved edge (px)')
        axs[r_, 0].set_ylabel('median answer shift (px)')
    fig.tight_layout()
    fig.savefig(out / 'behaviour.png', dpi=150)
    plt.close(fig)


GROUP_ORDER = ['H11_B', 'min{a}_B', 'max{a}_B', 'ctr{a}_B', 'edges{a}_B', 'L20_B', 'full_B', 'dn_B',
               'max{a}_mix0.5', 'max{a}_mix2', 'edges{a}_B_freeze_late_heads', 'edges{a}_B_freeze_dn21',
               'edges{a}_B_freeze_mlp21_27', 'edges{a}_B_freeze_mlp28_32', 'q_edges{a}_B']
GROUP_LABEL = {'H11_B': 'L20H11', 'min{a}_B': 'min-edge group', 'max{a}_B': 'max-edge group', 'ctr{a}_B': 'centre group',
               'edges{a}_B': 'min+max groups', 'L20_B': 'all L20 heads', 'full_B': 'all 128 heads',
               'dn_B': 'all DeltaNet', 'max{a}_mix0.5': 'max group x0.5', 'max{a}_mix2': 'max group x2',
               'edges{a}_B_freeze_late_heads': 'edges, freeze L24-32 heads', 'edges{a}_B_freeze_dn21': 'edges, freeze DeltaNet 21-32',
               'edges{a}_B_freeze_mlp21_27': 'edges, freeze MLP 21-27', 'edges{a}_B_freeze_mlp28_32': 'edges, freeze MLP 28-32',
               'q_edges{a}_B': 'edge heads query only'}


def fig_groups(summary: dict, out: Path) -> None:
    conds = summary['conditions']
    rows = [('str:right:x', 'x', 'natural, right edge +96'), ('str:left:x', 'x', 'natural, left edge -96'),
            ('str:bottom:y', 'y', 'natural, bottom edge +96'), ('str:top:y', 'y', 'natural, top edge -96'),
            ('rect:right:x', 'x', 'rectangle, right edge +96'), ('nat:point:x', 'x', 'natural, translation (x)'),
            ('nat:point:y', 'y', 'natural, translation (y)')]
    rows = [r for r in rows if r[0] in conds and conds[r[0]].get('n', 0) > 0]
    fig, ax = plt.subplots(figsize=(11, 0.33 * len(GROUP_ORDER) * 1.0 + 2.5))
    width = 0.8 / max(len(rows), 1)
    for k, (tag, a, label) in enumerate(rows):
        d = conds[tag]
        ys, lo, hi = [], [], []
        for g in GROUP_ORDER:
            v = d.get(g.format(a=a), [np.nan] * 3)
            ys.append(v[0]), lo.append(v[0] - v[1]), hi.append(v[2] - v[0])
        pos = np.arange(len(GROUP_ORDER)) + (k - len(rows) / 2 + 0.5) * width
        ax.barh(pos, ys, height=width * 0.9, xerr=[lo, hi], color=CAT[k % 8], label=f"{label} (n={d['n']})",
                error_kw={'lw': 0.6})
    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_yticklabels([GROUP_LABEL[g] for g in GROUP_ORDER])
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.axvline(1, color=GRAY, lw=0.6, ls=':')
    ax.set_xlabel('first-digit logit recovery r (patched from run B; mean, 95% bootstrap CI)')
    ax.legend(fontsize=7.5, frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(out / 'groups.png', dpi=150)
    plt.close(fig)


def fig_comp(summary: dict, out: Path) -> None:
    """L20 H11 across the composite design: necessity, sufficiency, precision, relay, by size."""
    cp, ab = summary['comp_probe'], summary['comp_ablation']
    combos = [('shape', 'plain'), ('shape', 'photo'), ('cutout', 'plain'), ('cutout', 'photo')]
    sizes = list(e2.COMP_SIZES)
    metrics = [('necessity: hit drop after mean-ablating H11', None),
               ('sufficiency: r of H11 alone, y value (where pair)', 'r_y:H11_B'),
               ('sufficiency: r of H11 alone, x value (where pair)', 'r_x:H11_B'),
               ('H11 attention inside target box', 'h11_mass_target'),
               ('readers lose target mass when H11 is ablated', 'relay')]
    fig, axs = plt.subplots(2, 5, figsize=(18, 6.4), sharex=True)
    for r_, nd in enumerate((0, 2)):
        for j, (title, key) in enumerate(metrics):
            ax = axs[r_, j]
            for k, (ot, bg) in enumerate(combos):
                ys = []
                for sz in sizes:
                    cell = f'{ot}|{bg}|{sz}|{nd}'
                    if key is None:
                        d = ab.get(cell, {})
                        ys.append(d['none']['hit'][0] - d['H11']['hit'][0] if 'H11' in d else np.nan)
                    elif key == 'relay':
                        d = cp.get(f'where:{cell}', {})
                        vals = [d[f'relay_drop:{h}'][0] for h in ('L24H7', 'L24H5', 'L24H9', 'L24H2', 'L28H0') if f'relay_drop:{h}' in d]
                        ys.append(np.mean(vals) if vals else np.nan)
                    else:
                        ys.append(cp.get(f'where:{cell}', {}).get(key, [np.nan])[0])
                ax.plot(sizes, ys, marker='o', ms=5, lw=2, color=CAT[k], label=f'{ot} on {bg}')
            ax.set_xscale('log', base=2)
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(s_) for s_ in sizes])
            ax.axhline(0, color=GRAY, lw=0.5)
            ax.set_title(f'{title}\n({nd} distractors)', fontsize=8)
            if r_ == 1:
                ax.set_xlabel('target size (px, longest side)')
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'comp_h11.png', dpi=150)
    plt.close(fig)


def fig_qswap(summary: dict, out: Path) -> None:
    cp = summary['comp_probe']
    combos = [('shape', 'plain'), ('shape', 'photo'), ('cutout', 'plain'), ('cutout', 'photo')]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, (ot, bg) in enumerate(combos):
        a_, b_, n_ = [], [], []
        for sz in e2.COMP_SIZES:
            d = cp.get(f'where:{ot}|{bg}|{sz}|2', {})
            a_.append(d.get('q11_mass_A', [np.nan])[0]), b_.append(d.get('q11_mass_B', [np.nan])[0])
        axs[0].plot(e2.COMP_SIZES, a_, marker='o', color=CAT[k], lw=2, label=f'{ot} on {bg}: on run-A target')
        axs[0].plot(e2.COMP_SIZES, b_, marker='s', color=CAT[k], lw=1, ls='--', label=f'{ot} on {bg}: on run-B location')
        w_ = []
        for sz in e2.COMP_SIZES:
            d = cp.get(f'what:{ot}|{bg}|{sz}|2', {})
            w_.append(d.get('q11_mass_B', [np.nan])[0])
        axs[1].plot(e2.COMP_SIZES, w_, marker='o', color=CAT[k], lw=2, label=f'{ot} on {bg}')
    axs[0].set_title('where pair: H11 query taken from run B (target moved)', fontsize=9)
    axs[1].set_title('what pair: H11 query taken from run B (other referent)\nattention on the other referent', fontsize=9)
    for ax in axs:
        ax.set_xscale('log', base=2)
        ax.set_xticks(e2.COMP_SIZES)
        ax.set_xticklabels([str(s_) for s_ in e2.COMP_SIZES])
        ax.set_xlabel('target size (px)')
        ax.set_ylabel('H11 attention mass in box')
    axs[0].legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(out / 'qswap.png', dpi=150)
    plt.close(fig)


def fig_probe_overlays(run: Path, round3: Path, cases: dict, recs: list[dict], out: Path, which: list[str], name: str,
                       heads: list[str]) -> None:
    """Rows: probe jobs; columns: run-A attention of the listed heads at the x and y first-digit queries."""
    r3 = eh.load_cases(round3)
    sel = [r for r in recs if r['id'] in which]
    fig, axs = plt.subplots(len(sel), 2 * len(heads) + 1, figsize=(1.6 * (2 * len(heads) + 1), 1.75 * len(sel) + 0.5))
    axs = np.atleast_2d(axs)
    im = None
    for i, rec in enumerate(sel):
        c = cases[rec['case']]
        img, box = e2.render(c, r3)
        arr = an.probe_arrays(run, rec)
        att = arr['att_a'].astype(np.float32)
        q0 = [k for k, (key, s_) in enumerate(rec['digit_queries']) if s_ == 0]
        axs[i, 0].imshow(img)
        draw_box(axs[i, 0], box, tuple(rec['grid']), 'yellow')
        axs[i, 0].set_xticks([]), axs[i, 0].set_yticks([])
        axs[i, 0].set_ylabel(c.get('expr', '')[:22], fontsize=7)
        for j, h in enumerate(heads):
            l, hh = map(int, h[1:].split('H'))
            for v in range(2):
                ax = axs[i, 1 + 2 * j + v]
                p = att[q0[v], e2.ATT_LAYERS.index(l), hh]
                im = overlay(ax, img, p, tuple(rec['grid']), box, 12, f"{h} {'x' if v == 0 else 'y'}" if i == 0 else None)
    cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01)
    cb.set_label('attention / uniform attention per token')
    fig.savefig(out / name, dpi=120, bbox_inches='tight')
    plt.close(fig)


def fig_digit_overlay(run: Path, round3: Path, cases: dict, rec: dict, heads: list[str], out: Path, name: str) -> None:
    """Rows: heads; columns: every query slot (token before the value, then the query of each digit) of every value."""
    r3 = eh.load_cases(round3)
    c = cases[rec['case']]
    img, box = e2.render(c, r3)
    att = an.probe_arrays(run, rec)['att_a'].astype(np.float32)
    cols = list(enumerate(rec['digit_queries']))
    vals = {v['key']: str(v['value']) for v in rec['values']}
    fig, axs = plt.subplots(len(heads), len(cols), figsize=(1.45 * len(cols) + 0.6, 1.5 * len(heads) + 0.6))
    im = None
    for j, (qi, (key, slot)) in enumerate(cols):
        lab = 'before' if slot < 0 else f"digit {slot + 1} ('{vals[key][slot] if slot < len(vals[key]) else ''}')"
        for i, h in enumerate(heads):
            l, hh = map(int, h[1:].split('H'))
            im = overlay(axs[i, j], img, att[qi, e2.ATT_LAYERS.index(l), hh], tuple(rec['grid']), box, 12,
                         f'{key}: {lab}' if i == 0 else None)
            if j == 0:
                axs[i, j].set_ylabel(h)
    cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01)
    cb.set_label('attention / uniform attention per token')
    fig.savefig(out / name, dpi=110, bbox_inches='tight')
    plt.close(fig)


def fig_mediation(summary: dict, out: Path) -> None:
    """r of the edge groups with one MLP layer frozen (left) or all MLPs of layers 20-32 frozen except one (right)."""
    med = summary['mediation']
    tags = [t for t in ('str:right:x', 'str:left:x', 'str:bottom:y', 'str:top:y', 'rect:right:x', 'rect:bottom:y', 'nat:x', 'nat:y')
            if t in med and med[t].get('n', 0) > 0]
    layers = list(e2.MED_LAYERS)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for k, t in enumerate(tags):
        a = t.split(':')[-1]
        d = med[t]
        base = d[f'edges{a}_B'][0]
        fr = [d[f'edges{a}_B_freeze_mlp{l}'][0] for l in layers]
        on = [d[f'edges{a}_B_only_mlp{l}'][0] for l in layers]
        axs[0].plot(layers, fr, marker='o', ms=4, lw=1.6, color=CAT[k % 8], label=f"{t} (n={d['n']}, no freeze {base:.2f})")
        axs[1].plot(layers, on, marker='o', ms=4, lw=1.6, color=CAT[k % 8])
        axs[1].axhline(d[f'edges{a}_B_freeze_mlp20_32'][0], color=CAT[k % 8], lw=0.8, ls=':')
    axs[0].set_title('edge groups from B, one MLP layer frozen to run A', fontsize=9)
    axs[1].set_title('edge groups from B, all MLPs 20-32 frozen except one\n(dotted: all frozen)', fontsize=9)
    for ax in axs:
        ax.set_xlabel('MLP layer')
        ax.axhline(0, color=GRAY, lw=0.5)
    axs[0].set_ylabel('first-digit logit recovery r')
    axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'mediation.png', dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--round3', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--parts', nargs='+', default=['r3', 'probe'])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run / 'summary.json').read_text())
    if 'r3' in args.parts:
        r3c = eh.load_cases(args.round3)
        items = an.load_r3(args.round3)
        by = {(it['case'], it['format']): it for it in items}
        both = sorted({cid for cid, f in by if (cid, 'point') in by and (cid, 'bbox') in by})
        picks = []
        for g in ('vocpart', 'rc', 'stuff', 'paco'):
            cand = [cid for cid in both if r3c[cid]['group'] == g and 0.08 < r3c[cid]['area_frac'] < 0.3]
            picks += cand[:1]
        fig_protocol(r3c, picks[0], args.out)
        heads = ['L20H0', 'L24H2', 'L24H7', 'L24H12', 'L24H1', 'L20H10', 'L24H5', 'L24H6', 'L24H8', 'L24H9', 'L28H0',
                 'L20H11', 'L16H1', 'L16H2', 'L20H8', 'L20H3']
        fig_head_table(summary, heads, args.out)
        gal_heads = ['L20H0', 'L24H7', 'L20H10', 'L24H5', 'L28H0', 'L20H11', 'L16H1']
        for k, cid in enumerate(picks):
            sel = [by[(cid, f)] for f in ('point', 'bbox') if (cid, f) in by]
            if sel:
                fig_r3_gallery(args.round3, sel, gal_heads, args.out, f'gallery_{k}.png')
    if 'probe' in args.parts and 'scan' in summary:
        fig_scan(summary, [f'nat:{s_}' for s_ in an.STEP_NAMES], [STEP_TITLE[s_] for s_ in an.STEP_NAMES], args.out,
                 'scan_nat.png')
        fig_scan(summary, ['str:left', 'str:right', 'str:top', 'str:bottom', 'rect:right', 'rect:bottom'],
                 ['left edge moves', 'right edge moves', 'top edge moves', 'bottom edge moves', 'rect right', 'rect bottom'],
                 args.out, 'scan_edges.png')
        fig_scan(summary, ['comp:pt_x', 'comp:pt_y'], ['composite 160 px, point x', 'composite 160 px, point y'],
                 args.out, 'scan_comp.png')
        fig_digit_slots(summary, ['L20H0', 'L24H7', 'L20H10', 'L24H5', 'L28H0', 'L20H11', 'L16H1'], args.out)
        fig_groups(summary, args.out)
        cases = e2.load_cases(args.run)
        recs = an.load_probe(args.run)
        for fmt in ('point', 'bbox'):
            cand = [r for r in recs if r['set'] == 'nat' and r['format'] == fmt and e2.box_span_ok(r['box_a'], tuple(r['grid']))
                    and all(len(v['toks']) == 3 for v in r['values'])]
            if cand:
                cand.sort(key=lambda r: (r['box_a'][2] - r['box_a'][0]) * (r['box_a'][3] - r['box_a'][1]))
                fig_digit_overlay(args.run, args.round3, cases, cand[int(len(cand) * 0.7)], ['L20H0', 'L24H7', 'L20H10', 'L24H5', 'L20H11'],
                                  args.out, f'digits_{fmt}.png')
    if 'probe' in args.parts and 'mediation' in summary:
        fig_mediation(summary, args.out)
    if 'probe' in args.parts and 'behaviour' in summary:
        fig_behaviour(args.run, summary, args.out)
    if 'probe' in args.parts and 'comp_probe' in summary:
        fig_comp(summary, args.out)
        fig_qswap(summary, args.out)
        cases = e2.load_cases(args.run)
        recs = an.load_probe(args.run)
        for nm, filt in (('overlay_comp.png', lambda r: r['set'] == 'comp' and r['kind'] == 'where' and r['format'] == 'point'),
                         ('overlay_nat.png', lambda r: r['set'] == 'nat' and r['format'] == 'point')):
            cand = [r for r in recs if filt(r)]
            if nm == 'overlay_comp.png':
                pick = []
                for ot in ('shape', 'cutout'):
                    for bg in ('plain', 'photo'):
                        for sz in (32, 160):
                            pick += [r['id'] for r in cand if cases[r['case']]['objtype'] == ot and cases[r['case']]['bg'] == bg
                                     and cases[r['case']]['size'] == sz and cases[r['case']]['ndistr'] == 2][:1]
            else:
                pick = [r['id'] for r in cand[::max(1, len(cand) // 6)]][:6]
            fig_probe_overlays(args.run, args.round3, cases, recs, args.out, pick, nm,
                               ['L20H11', 'L20H0', 'L24H7', 'L20H10', 'L24H5'])


if __name__ == '__main__':
    main()
