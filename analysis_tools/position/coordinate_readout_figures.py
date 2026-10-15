"""Figures of experiment 3.5 (coordinate readout path).

Reads the run directory (stage artifacts and results.json produced by
`coordinate_readout_analysis.py`) and writes PNG figures for the results document.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position.coordinate_readout_analysis import FULL_LAYERS, load, positions
from analysis_tools.position.coordinate_readout_run import case_input, disc_input, t6_input

# Categorical slots in fixed order (reference palette, light mode).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 150,
                     'savefig.bbox': 'tight', 'legend.frameon': False, 'lines.linewidth': 2})


def save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print('figure', path.name)


# ----------------------------------------------------------------------------- inputs

def fig_inputs(d: dict, out: Path) -> None:
    picks = [('grid_d64_p14_point', 'disc 64 px, 6x6 position grid'), ('trans_d32_h+3_point', 'translation series, 32 px'),
             ('size1280_q5_point', 'canvas-size series, 1280 px')]
    t6 = next(c for c in d['cases'].values() if c['set'] == 't6' and c['grid'] == 20 and c['format'] == 'rowcol')
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (cid, title) in zip(axes, picks):
        image, truth, _ = disc_input(d['cases'][cid])
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
    image, truth, _ = t6_input(t6)
    axes[3].imshow(image)
    axes[3].set_title(f"T6 20x20 grid, target row {truth['target_rc'][0] + 1} col {truth['target_rc'][1] + 1}", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for side in ('top', 'right', 'bottom', 'left'):
            ax.spines[side].set_visible(True)
    save(fig, out / 'inputs.png')


# ----------------------------------------------------------------------------- C0

def fig_c0(d: dict, res: dict, out: Path) -> None:
    pos = positions(d)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {32: BLUE, 64: ORANGE, 128: AQUA}
    for ax, axis, label in ((axes[0], 'h', 'x (horizontal series)'), (axes[1], 'v', 'y (vertical series)')):
        i = 0 if axis == 'h' else 1
        for dia in (32, 64, 128):
            pts = sorted((pos[k][1][i], pos[k][0][i]) for k, c in d['cases'].items()
                         if c['set'] == 'trans' and c.get('axis') == axis and c['diameter'] == dia and k in pos)
            if pts:
                t, o = np.array(pts).T
                ax.plot(t, o, '-o', color=colors[dia], ms=4, label=f'diameter {dia} px')
        ax.plot([0, 1000], [0, 1000], '--', color=MUTED, lw=1, label='output = truth')
        ax.set_xlabel(f'true {label}, 0-1000'); ax.set_ylabel('model output, 0-1000')
        ax.legend(fontsize=8)
    ax = axes[2]
    sizes = sorted({c['canvas'] for c in d['cases'].values() if c['set'] == 'size'})
    norm, pix = [], []
    for s in sizes:
        e = res['c0'][f'size|{s}|point']
        norm.append(e['error']['mean']); pix.append(e['error_pixels']['mean'])
    ax.plot(sizes, norm, '-o', color=BLUE, label='center error, 0-1000 units')
    ax.plot(sizes, pix, '-s', color=ORANGE, label='center error, pixels')
    ax.set_xlabel('canvas side (pixels)'); ax.set_ylabel('mean center error')
    ax.set_title('canvas-size series (7 positions each)', fontsize=9)
    ax.legend(fontsize=8)
    axes[0].set_title('translation series, horizontal', fontsize=9)
    axes[1].set_title('translation series, vertical', fontsize=9)
    save(fig, out / 'c0_translation_size.png')


def fig_c0_values(d: dict, out: Path) -> None:
    """Frequency of individual output coordinate values: disc point answers and T6 point answers."""
    from collections import Counter
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
    panels = ((axes[0], lambda c: c['set'] in ('grid', 'trans', 'size'), 'disc point answers (x and y pooled)'),
              (axes[1], lambda c: c['set'] == 't6', 'T6 point answers, 6x6 / 10x10 / 20x20 (x and y pooled)'))
    for ax, sel, title in panels:
        cnt = Counter()
        for r in d['baseline']:
            c = d['cases'][r['id']]
            if sel(c) and r['format'] == 'point' and r['score'].get('valid'):
                cnt.update(int(v) for v in r['score']['coords'])
        xs = np.array(sorted(cnt)); ys = np.array([cnt[x] for x in xs])
        ax.vlines(xs, 0, ys, color=BLUE, lw=1.5)
        for x, y in zip(xs, ys):
            if y >= max(5, ys.max() * 0.15):
                ax.annotate(str(x), (x, y), textcoords='offset points', xytext=(0, 2), ha='center', fontsize=7, color=INK)
        ax.set_ylabel('count'); ax.set_title(f'{title}, {ys.sum()} values', fontsize=9)
    axes[1].set_xlabel('output coordinate value (0-1000)')
    save(fig, out / 'c0_output_values.png')


# ----------------------------------------------------------------------------- C1

C1_TASKS = [('disc_point', 'disc, point'), ('disc_bbox', 'disc, bbox'), ('t6g06_rowcol', 'T6 6x6, row-column'),
            ('t6g20_rowcol', 'T6 20x20, row-column'), ('t6g10_point', 'T6 10x10, point')]
HEATMAP_TASKS = C1_TASKS[:4]


def fig_c1_heads(d: dict, res: dict, out: Path) -> None:
    tasks = [t for t in HEATMAP_TASKS if t[0] in res['c1'] and res['c1'][t[0]]['candidates']]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.6), squeeze=False)
    fig.subplots_adjust(wspace=0.45)
    for ax, (task, title) in zip(axes[0], tasks):
        m = np.asarray(res['c1'][task]['candidates']['logE_map_selection'])
        im = ax.imshow(m, cmap='Blues', vmin=0, vmax=max(3, m.max()), aspect='auto')
        for l, h in res['c1'][task]['candidates']['heads']:
            ax.add_patch(plt.Rectangle((h - 0.5, FULL_LAYERS.index(l) - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.5))
        ax.set_yticks(range(8)); ax.set_yticklabels(FULL_LAYERS); ax.set_xticks(range(0, 16, 3))
        ax.set_xlabel('head'); ax.set_ylabel('full-attention layer'); ax.grid(False)
        ax.set_title(f"{title}: first digits (n={res['c1'][task]['candidates']['n_selection']})", fontsize=9)
        fig.colorbar(im, ax=ax, label='mean log E')
    save(fig, out / 'c1_head_enrichment.png')


def fig_c1_profile(d: dict, res: dict, out: Path) -> None:
    tasks = [t for t in C1_TASKS if t[0] in res['c1']]
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.4 * len(tasks), 6.4), squeeze=False, sharex=True)
    names = {'q_end': ('question end', MUTED), 'think_end': ('</think>', YELLOW), 'last': ('last prompt token', VIOLET),
             'first': ('coordinate first digits', BLUE)}
    for j, (task, title) in enumerate(tasks):
        prof = res['c1'][task]['layer_profile']
        for k, (lab, col) in names.items():
            if k in prof:
                axes[0, j].plot(FULL_LAYERS, prof[k]['vis_mass'], '-o', color=col, ms=4, label=lab)
                axes[1, j].plot(FULL_LAYERS, prof[k]['max_logE'], '-o', color=col, ms=4, label=lab)
        axes[0, j].set_title(title, fontsize=9)
        axes[0, j].set_ylabel('attention mass on image (mean over heads)')
        axes[1, j].set_ylabel('max over heads of mean log E')
        axes[1, j].set_xlabel('layer'); axes[1, j].set_xticks(FULL_LAYERS)
    axes[0, 0].legend(fontsize=8)
    save(fig, out / 'c1_layer_profile.png')


def fig_c1_rho(d: dict, res: dict, out: Path) -> None:
    tasks = [t for t in C1_TASKS if t[0] in res['c1']]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 3.6), squeeze=False, sharey=True)
    for ax, (task, title) in zip(axes[0], tasks):
        e = res['c1'][task]
        for tag, col in (('correct', BLUE), ('incorrect', ORANGE)):
            if f'rho_{tag}' in e:
                ax.plot(range(1, 33), e[f'rho_{tag}']['mean'], color=col, label=f"{tag} (n={e[f'rho_{tag}']['n']})")
        if 'image_share_correct' in e:
            ax.plot(range(1, 33), e['image_share_correct'], ':', color=MUTED, label='image share of |A|, correct')
        for l in FULL_LAYERS:
            ax.axvline(l, color=GRID, lw=1, zorder=0)
        ax.set_title(title, fontsize=9); ax.set_xlabel('layer (input residual)')
    axes[0, 0].set_ylabel(r'target share $\rho^\ell$')
    axes[0, 0].legend(fontsize=8)
    save(fig, out / 'c1_attribution_rho.png')


def fig_c1_centroid(d: dict, res: dict, out: Path) -> None:
    tasks = [t for t in C1_TASKS if t[0] in res['c1'] and 'centroid' in res['c1'][t[0]]]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4), squeeze=False)
    for ax, (task, title) in zip(axes[0], tasks):
        rows = [r for r in res['c1'][task]['centroid']['rows'] if not r['correct']]
        if not rows:
            continue
        axis = 0
        t = np.array([r['truth'][axis] for r in rows]); o = np.array([r['output'][axis] for r in rows])
        c = np.array([r['centroid'][axis] for r in rows])
        ax.scatter(t, c, s=14, color=BLUE, label='candidate-head centroid')
        ax.scatter(t, o, s=14, color=ORANGE, marker='x', label='model output')
        ax.plot([0, 1000], [0, 1000], '--', color=MUTED, lw=1)
        ax.set_xlabel('true x (0-1000)'); ax.set_ylabel('x (0-1000)')
        ax.set_title(f'{title}, incorrect samples (n={len(rows)})', fontsize=9)
    axes[0, 0].legend(fontsize=8)
    save(fig, out / 'c1_centroid.png')


def fig_c1_overlay(d: dict, res: dict, out: Path, run: Path) -> None:
    """Attention of the top candidate head at the first digit of each coordinate, over the input image."""
    pos = positions(d)
    picks = []
    att = {r['id']: r for r in d['attention']}
    for task, want_correct in (('disc_point', True), ('t6g20_rowcol', False)):
        if task not in res['c1'] or not res['c1'][task]['candidates']:
            continue
        recs = [r for r in d['attention'] if (('disc_' if r['set'] in ('grid', 'trans') else f"t6g{r['grid'][0] // 2:02d}_")
                                                + r['format']) == task and r['correct'] == want_correct]
        if task == 't6g20_rowcol':
            recs = [r for r in recs if r['score'].get('d_col') == 1 and r['score'].get('d_row') == 0] or recs
        # Grid tasks use heads selected on the 6x6 row-column task, where nearly all answers are correct.
        source = 't6g06_rowcol' if task.startswith('t6') and 't6g06_rowcol' in res['c1'] else task
        if recs:
            picks.append((task, recs[0], res['c1'][source]['candidates']['heads'][0]))
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 3, figsize=(12, 4.2 * len(picks)), squeeze=False)
    for row, (task, rec, (layer, head)) in enumerate(picks):
        case = d['cases'][rec['id']]
        image, truth, _ = case_input(case)
        maps = np.load(run / 'attention' / 'maps' / f"{rec['id']}.npz")
        H, W = rec['grid']
        li = FULL_LAYERS.index(layer)
        labels = ['last prompt token'] + (['row value', 'column value'] if case['format'] == 'rowcol' else ['x value', 'y value'])
        for j, k in enumerate([2, 3, 4]):
            a = maps['attn'][li, head, k].astype(float).reshape(H, W)
            ax = axes[row, j]
            ax.imshow(np.asarray(image.convert('L')), cmap='gray', vmin=0, vmax=255 * 1.6)
            heat = np.kron(a / a.max(), np.ones((32, 32)))
            rgba = plt.get_cmap('inferno_r')(heat)
            rgba[..., 3] = np.clip(heat * 1.5, 0, 0.85)
            ax.imshow(rgba)
            o, t = pos[rec['id']]
            size = np.asarray(image.size)
            ax.plot(*(t * size / 1000), 'o', mfc='none', mec=AQUA, ms=12, mew=2, label='truth')
            ax.plot(*(o * size / 1000), 'x', color=YELLOW, ms=12, mew=2, label='output')
            ax.set_title(f'L{layer} H{head}, query: {labels[j]}', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        axes[row, 0].set_ylabel(rec['id'], fontsize=8)
    axes[0, 0].legend(fontsize=8, loc='lower left')
    save(fig, out / 'c1_attention_overlay.png')


# ----------------------------------------------------------------------------- C2

C2_FAMILIES = [('s2shift', 'S2 global shift'), ('s2swap', 'S2 region swap'), ('s1shift', 'S1 learned-PE shift'),
               ('s1swap_pe', 'S1 swap: PE'), ('s1swap_rope', 'S1 swap: ViT RoPE'), ('s1swap_both', 'S1 swap: PE+RoPE'),
               ('conflict_both', 'S1+S2 swap'), ('s4perm', 'S4 sequence swap'), ('c3content', 'C3 content swap'),
               ('real', 'disc actually moved')]


def fig_c2_bars(res: dict, out: Path) -> None:
    fams = [(k, lab) for k, lab in C2_FAMILIES if k in res['c2']['families']]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (k, lab) in enumerate(fams):
        for j, (sub, col) in enumerate((('correct', BLUE), ('incorrect', ORANGE))):
            e = res['c2']['families'][k].get(sub)
            if not e:
                continue
            m, (lo, hi) = e['w']['mean'], e['w']['ci']
            ax.bar(i + (j - 0.5) * 0.36, m, 0.34, color=col, label=f"baseline {sub}" if i == 0 else None)
            ax.errorbar(i + (j - 0.5) * 0.36, m, [[m - lo], [hi - m]], color=INK, lw=1, capsize=2)
    ax.axhline(1, ls='--', color=MUTED, lw=1); ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels([lab for _, lab in fams], rotation=30, ha='right')
    ax.set_ylabel('displacement weight w (95% CI over images)')
    ax.legend(fontsize=8)
    save(fig, out / 'c2_weights.png')


def fig_c2_shift(d: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, fam, title in ((axes[0], 's2shift', 'S2: MRoPE ids of image tokens shifted'),
                           (axes[1], 's1shift', 'S1: learned PE window shifted')):
        for axis, col in (('h', BLUE), ('v', ORANGE)):
            pts = []
            for rec in d['interventions']:
                conds = {c['condition']: c for c in rec['conditions']}
                for name, c in conds.items():
                    if not name.startswith(fam + '_') or c.get('axis') != axis or c['center'] is None:
                        continue
                    ref = conds[c['ref']]
                    if ref['center'] is None:
                        continue
                    i = 0 if axis == 'h' else 1
                    pts.append((c['d'][i], c['center'][i] - ref['center'][i]))
            if pts:
                p = np.array(pts)
                xs = np.unique(p[:, 0])
                ax.scatter(p[:, 0] + (5 if axis == 'v' else -5), p[:, 1], s=6, alpha=0.25, color=col)
                ax.plot(xs, [p[p[:, 0] == x, 1].mean() for x in xs], '-o', color=col, ms=4,
                        label='x displacement' if axis == 'h' else 'y displacement')
        lim = [-520, 520]
        ax.plot(lim, lim, '--', color=MUTED, lw=1, label='w = 1')
        ax.plot(lim, [0, 0], ':', color=MUTED, lw=1, label='w = 0')
        ax.set_xlabel('predicted displacement d (0-1000)'); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel('observed output displacement (0-1000)')
    axes[0].legend(fontsize=8)
    save(fig, out / 'c2_global_shift.png')


def fig_c2_arrows(d: dict, out: Path) -> None:
    recs = [r for r in d['interventions'] if r['baseline_correct'] and r['diameter'] == 64]
    if not recs:
        return
    def typical(r):
        # Representative image: ViT-RoPE swap weights closest to the population mean (about 0.7) on both axes.
        cs = {c['condition']: c for c in r['conditions']}
        ws = []
        for tag in ('h6', 'v6'):
            c, b = cs.get(f's1swap_rope_{tag}'), cs['standard']
            if not c or c['center'] is None or b['center'] is None:
                return 9.0
            dd = np.asarray(c['d']); ws.append((np.asarray(c['center']) - np.asarray(b['center'])) @ dd / (dd @ dd))
        return float(np.mean(np.abs(np.asarray(ws) - 0.7)))
    rec = min(recs, key=typical)
    conds = {c['condition']: c for c in rec['conditions']}
    case = d['cases'][rec['id']]
    image, _, _ = disc_input(case)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, tag in zip(axes, ('h6', 'v6') if 's2swap_h6' in conds else ('h4', 'v4')):
        ax.imshow(image); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        base = np.asarray(conds['standard']['center']) * 512 / 1000
        ax.plot(*base, 'o', color=INK, ms=7, label='baseline output')
        for name, col in ((f's2swap_{tag}', BLUE), (f's1swap_rope_{tag}', ORANGE), (f's1swap_pe_{tag}', YELLOW),
                          (f's4perm_{tag}', VIOLET), (f'c3content_{tag}', MAGENTA), (f'real_{tag}', AQUA)):
            c = conds.get(name)
            if not c or c['center'] is None:
                continue
            p = np.asarray(c['center']) * 512 / 1000
            ax.annotate('', p, base, arrowprops=dict(arrowstyle='->', color=col, lw=2))
            ax.plot(*p, 'o', color=col, ms=6, label=name.rsplit('_', 1)[0])
        reg = next(g for g in rec['regions'] if f"{g['axis']}{g['k']}" == tag)
        H, W = rec['grid']
        for t in reg['region']:
            ax.add_patch(plt.Rectangle(((t % W) * 32, (t // W) * 32), 32, 32, fill=False, ec=MUTED, lw=0.8, ls='--'))
        ax.set_title(f"T' = target region moved {reg['k']} tokens ({'right/left' if reg['axis'] == 'h' else 'down/up'})", fontsize=9)
    axes[0].legend(fontsize=7, loc='lower left')
    save(fig, out / 'c2_arrows.png')


# ----------------------------------------------------------------------------- failure analysis

FAIL_FAMILIES = [('s2swap', 'MRoPE swap'), ('s1rope', 'ViT RoPE swap'), ('s1both', 'ViT PE+RoPE swap'),
                 ('s4perm', 'sequence swap'), ('c3content', 'content swap')]
DISC_FAMILY = {'s2swap': 's2swap', 's1rope': 's1swap_rope', 's1both': 's1swap_both', 's4perm': 's4perm',
               'c3content': 'c3content'}


def grid_condition_names(fam: str, direction: str) -> list[str]:
    """Condition names of one-cell swaps in the fail / fail2 stages."""
    return {'s2swap': [f's2swap_{direction}'], 's1rope': [f's1rope_swap_{direction}'],
            's1both': [f's1swap_both_{direction}1'], 's4perm': [f's4perm_{direction}1'],
            'c3content': [f'c3content_{direction}1']}[fam]


def fig_fail_follow(res: dict, out: Path) -> None:
    """Fraction of a region swap followed by the answer, split by swap axis, for discs and T6 grids."""
    blocks = [('disc', 'disc point (C2, 2-6 tokens)')] + [
        (k, k.replace('g', 'T6 ', 1).replace('|', ' ')) for k in ('g10|point', 'g20|point', 'g6|rowcol', 'g10|rowcol', 'g20|rowcol')
        if k in res['failure']]
    colors = [MUTED, ORANGE, MAGENTA, YELLOW, AQUA, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    width = 0.8 / len(blocks)
    labelled = set()
    for ax, (axis, dirs, title) in zip(axes, (('x', ('right', 'left'), 'horizontal swaps (answer x / column)'),
                                             ('y', ('down', 'up'), 'vertical swaps (answer y / row)'))):
        for j, ((block, label), col) in enumerate(zip(blocks, colors)):
            for i, (fam, _) in enumerate(FAIL_FAMILIES):
                if block == 'disc':
                    e = res['c2'].get('weights_by_axis', {}).get(f'{DISC_FAMILY[fam]}|{axis}')
                    vals = [e['mean']] if e else []
                else:
                    conds = res['failure'][block]['conditions']
                    vals = [conds[n]['follow_ratio']['mean'] for dname in dirs for n in grid_condition_names(fam, dname)
                            if n in conds and 'follow_ratio' in conds[n] and conds[n]['follow_ratio']['n'] >= 10]
                if vals:
                    ax.bar(i + (j - (len(blocks) - 1) / 2) * width, np.mean(vals), width * 0.92, color=col,
                           label=label if label not in labelled else None)
                    labelled.add(label)
        ax.axhline(1, ls='--', color=MUTED, lw=1); ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(range(len(FAIL_FAMILIES))); ax.set_xticklabels([lab for _, lab in FAIL_FAMILIES], rotation=15)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel('answer displacement / swap displacement')
    axes[0].legend(fontsize=8)
    save(fig, out / 'fail_swap_follow.png')


def fig_fail_candidates(res: dict, out: Path) -> None:
    block = res['failure'].get('g20|rowcol', {}).get('candidates')
    if not block:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, axis in zip(axes, ('col', 'row')):
        rows = block[axis]['rows']
        for g, col, lab in (('correct', BLUE, 'answer correct'), ('plus1', ORANGE, 'answer = truth + 1'),
                            ('other', MUTED, 'other error')):
            sub = [r for r in rows if (r['d'] == 0 if g == 'correct' else r['d'] == 1 if g == 'plus1' else r['d'] not in (0, 1))]
            if sub:
                ax.scatter([r['truth'] for r in sub], [r['margin_plus1'] for r in sub], s=18, color=col, label=f'{lab} (n={len(sub)})')
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel(f'true {"column" if axis == "col" else "row"} (1-based)')
        ax.set_title(f'20x20 grid: log p(truth+1) - log p(truth), {"column" if axis == "col" else "row"}', fontsize=9)
    axes[0].set_ylabel('log-probability margin')
    axes[0].legend(fontsize=8)
    save(fig, out / 'fail_candidate_margin.png')


def fig_fail_column_profile(d: dict, res: dict, out: Path, run: Path) -> None:
    """Attention mass of pointing heads per grid column (row) relative to the target, at the column (row) digit."""
    if 't6g06_rowcol' not in res['c1'] or not res['c1']['t6g06_rowcol']['candidates']:
        return
    heads = res['c1']['t6g06_rowcol']['candidates']['heads']           # selected on 6x6, applied to 20x20
    groups = {}
    for r in d['attention']:
        if r['set'] != 't6' or r['format'] != 'rowcol' or r['grid'][0] != 40:
            continue
        s = r['score']
        g = 'correct' if s['correct'] else 'column +1' if (s.get('d_col'), s.get('d_row')) == (1, 0) else 'other errors'
        maps = np.load(run / 'attention' / 'maps' / f"{r['id']}.npz")['attn'].astype(float)
        tr, tc = d['cases'][r['id']]['truth']['target_rc']
        H, W = r['grid']
        for qi, axis, t in ((4, 'column', tc), (3, 'row', tr)):
            prof = np.full(9, np.nan)
            acc = np.zeros(9)
            for l, h in heads:
                a = maps[FULL_LAYERS.index(l), h, qi].reshape(H, W)
                v = (a.sum(0) if axis == 'column' else a.sum(1)).reshape(-1, 2).sum(1)   # mass per grid column/row
                for k in range(-4, 5):
                    if 0 <= t + k < len(v):
                        acc[k + 4] += v[t + k] / len(heads)
                    else:
                        prof[k + 4] = np.nan
            prof = np.where(np.isnan(prof) & (acc == 0), np.nan, acc)
            groups.setdefault((g, axis), []).append(prof)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, axis in zip(axes, ('column', 'row')):
        for g, col in (('correct', BLUE), ('column +1', ORANGE), ('other errors', MUTED)):
            rows = groups.get((g, axis))
            if rows:
                ax.plot(range(-4, 5), np.nanmean(rows, 0), '-o', color=col, ms=4, label=f'{g} (n={len(rows)})')
        ax.set_xlabel(f'grid {axis} relative to the target {axis}')
        ax.set_title(f'20x20 row-column, query: {axis} digit', fontsize=9)
    axes[0].set_ylabel('attention mass (mean over pointing heads)')
    axes[0].legend(fontsize=8)
    save(fig, out / 'fail_column_profile.png')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    d = load(args.run)
    res = json.loads((args.run / 'results.json').read_text())
    fig_inputs(d, args.figures)
    fig_c0(d, res, args.figures)
    fig_c0_values(d, args.figures)
    fig_c1_heads(d, res, args.figures)
    fig_c1_profile(d, res, args.figures)
    fig_c1_rho(d, res, args.figures)
    fig_c1_centroid(d, res, args.figures)
    fig_c1_overlay(d, res, args.figures, args.run)
    if d['interventions']:
        fig_c2_bars(res, args.figures)
        fig_c2_shift(d, args.figures)
        fig_c2_arrows(d, args.figures)
    if d['failure']:
        fig_fail_follow(res, args.figures)
        fig_fail_candidates(res, args.figures)
    fig_fail_column_profile(d, res, args.figures, args.run)


if __name__ == '__main__':
    main()
