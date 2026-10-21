"""Figures of experiment 3.11 (information flow of coordinate readout).

Reads summary.json (coordinate_flow_analysis.py), the 3.11 run records and the 2.5e readout maps; writes
PNG files into docs/analysis/position/figures/coordinate_flow/. Continuous quantities use viridis (or
a diverging blue-grey-orange map for signed shares); categorical series use a fixed hue order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
import numpy as np

from analysis_tools.position import coordinate_readout as cr
from analysis_tools.representation import sees_right as sr

FULL_LAYERS = cr.FULL_LAYERS
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#008300', '#e34948']
GREY = '#8c8b87'
DIVERGING = LinearSegmentedColormap.from_list('bgo', ['#2a78d6', '#f0efec', '#eb6834'])
COND_LABEL = {'disc_point': 'disc point', 't6g10_rowcol': 'T6 10x10 row/col', 't6g10_point': 'T6 10x10 point',
              'rc_point': 'RefCOCO point', 't6g20_rowcol': 'T6 20x20 row/col'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 10})


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


# ----------------------------------------------------------------------------- Q1

def fig_heads(S: dict, out: Path) -> None:
    """Per-head recovered fraction (8 layers x 16 heads) for x and y, per condition."""
    H = S.get('q1_heads', {})
    conds = [c for c in ('disc_point', 't6g10_rowcol', 't6g10_point', 'rc_point') if c in H]
    if not conds:
        return
    fig, axes = plt.subplots(len(conds), 2, figsize=(12.5, 3.0 * len(conds) + 0.6), squeeze=False)
    fig.subplots_adjust(hspace=0.55)
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.8)
    for i, c in enumerate(conds):
        for j, (axis, lab) in enumerate((('h', 'x / column'), ('v', 'y / row'))):
            ax = axes[i, j]
            if axis not in H[c]:
                ax.axis('off')
                continue
            m = np.array(H[c][axis]['single'], float)
            im = ax.imshow(m, cmap=DIVERGING, norm=norm, aspect='auto')
            ax.set_yticks(range(8), [f'L{l}' for l in FULL_LAYERS])
            ax.set_xticks(range(16))
            t = H[c][axis]['table']
            ax.set_title(f"{COND_LABEL[c]}, {lab} (n={t['full_all']['n']})\n"
                         f"all heads {t['full_all']['mean']:.2f}, "
                         f"L20 {t['L20']['mean']:.2f}, sum of singles {H[c][axis]['sum_single']:.2f}", fontsize=8)
            for li in range(8):
                for h in range(16):
                    if abs(m[li, h]) >= 0.05:
                        ax.text(h, li, f'{m[li, h]:.2f}', ha='center', va='center', fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel('head')
    cb = fig.colorbar(im, ax=axes, shrink=0.6)
    cb.set_label('recovered fraction r (single-head output patching)')
    save(fig, out, 'heads_single.png')


def fig_loo(S: dict, out: Path) -> None:
    """Drop of the all-head recovered fraction when one head of layers 16/20/24 is left out."""
    H = S.get('q1_heads', {})
    conds = [c for c in ('disc_point', 't6g10_rowcol', 't6g10_point', 'rc_point') if c in H]
    if not conds:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3.0), squeeze=False, sharey=True)
    for i, c in enumerate(conds):
        ax = axes[0, i]
        for j, axis in enumerate(('h', 'v')):
            if axis not in H[c]:
                continue
            drop = H[c][axis]['loo_drop']
            names = [f'{l}_{h}' for l in (16, 20, 24) for h in range(16)]
            v = [drop.get(n, np.nan) for n in names]
            ax.bar(np.arange(48) + (j - 0.5) * 0.4, v, width=0.4, color=C[j], label='x / column' if axis == 'h' else 'y / row')
        ax.set_xticks([7.5, 23.5, 39.5], ['L16', 'L20', 'L24'])
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_title(COND_LABEL[c])
        if i == 0:
            ax.set_ylabel('drop of r when one head is left out\n(all 128 heads patched minus the head)')
    axes[0, 0].legend(frameon=False)
    save(fig, out, 'heads_loo.png')


def fig_rowcol_attention(S: dict, out: Path) -> None:
    """Attention along the target row at the horizontal and vertical queries, and the row-start share."""
    A = S.get('q1_rowcol_attention', {})
    conds = [c for c in ('disc_point', 't6g10_rowcol', 't6g20_rowcol', 'rc_point') if c in A]
    fig, axes = plt.subplots(2, len(conds), figsize=(4.2 * len(conds), 7.2), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.3)
    for i, c in enumerate(conds):
        for j, q in enumerate(('h', 'v')):
            ax = axes[j, i]
            rel = np.array(A[c][q]['rel_profile'], float)                 # (8, 16, 81), offset -40 .. 40
            off = np.arange(-40, 41)
            keep = (off >= -8) & (off <= 8)
            ax.plot(off[keep], rel[4, 11][keep], color=C[0], lw=2, marker='o', ms=3, label='L20 H11')
            ax.plot(off[keep], np.nanmean(rel[4], 0)[keep], color=C[1], lw=2, marker='o', ms=3, label='layer-20 head mean')
            ax.axvline(0, color=GREY, lw=0.8, ls='--')
            col0 = np.array(A[c][q]['col0_mass'], float)
            ax.set_title(f"{COND_LABEL[c]}, {'x/column' if q == 'h' else 'y/row'} query\n"
                         f"col-0 share: L20 {col0[4].mean():.2f}, H11 {col0[4, 11]:.2f}, uniform {A[c][q]['u_col0']:.2f}",
                         fontsize=8)
            ax.set_xlabel('column offset from target (same row)', fontsize=8)
            if i == 0:
                ax.set_ylabel('share of image attention per token')
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, out, 'rowcol_profile.png')


def fig_rowstart_maps(source: Path, out: Path) -> None:
    """Average layer-20 attention map, aligned on the target, for both queries (synthetic and natural)."""
    cases = {c['id']: c for c in json.loads((source / 'cases.json').read_text())}
    recs = [r for r in sr.read_jsonl(source / 'readout' / 'records.jsonl') if r.get('status') == 'ok']
    conds = ('disc_point', 't6g10_rowcol', 'rc_point')
    fig, axes = plt.subplots(2, len(conds) * 2, figsize=(15, 5.4))
    from analysis_tools.representation.sees_right_analysis import axis_of
    im = None
    for i, cond in enumerate(conds):
        rs = [r for r in recs if cases[r['id']]['cond'] == cond]
        for w, reader in enumerate(('H11', 'mean')):
            acc = {'h': np.zeros((33, 33)), 'v': np.zeros((33, 33))}
            cnt = {'h': np.zeros((33, 33)), 'v': np.zeros((33, 33))}
            for r in rs:
                c = cases[r['id']]
                a = np.load(source / 'readout' / 'maps' / f"{r['id']}.npz")['attn'].astype(np.float32)
                H, W = r['grid']
                T = np.array(r['target_tokens'])
                tr, tc = int(np.median(T // W)), int(np.median(T % W))
                p = a / a.sum(-1, keepdims=True)
                for k in range(2):
                    q = 'h' if axis_of(c['format'], k) == 0 else 'v'
                    m = (p[4, 11, k] if reader == 'H11' else p[4, :, k].mean(0)).reshape(H, W)
                    rr, cc = np.meshgrid(np.arange(H) - tr + 16, np.arange(W) - tc + 16, indexing='ij')
                    ok = (rr >= 0) & (rr < 33) & (cc >= 0) & (cc < 33)
                    np.add.at(acc[q], (rr[ok], cc[ok]), m[ok])
                    np.add.at(cnt[q], (rr[ok], cc[ok]), 1)
            for j, q in enumerate(('h', 'v')):
                ax = axes[j, 2 * i + w]
                mm = acc[q] / np.maximum(cnt[q], 1)
                mm[cnt[q] < 5] = np.nan
                im = ax.imshow(mm[8:25, 8:25], cmap='viridis', norm=LogNorm(1e-4, 0.3), extent=(-8.5, 8.5, 8.5, -8.5))
                ax.plot(0, 0, marker='s', mfc='none', mec='#e34948', ms=8, mew=1.5)
                ax.set_title(f"{COND_LABEL[cond]}\n{'L20 H11' if reader == 'H11' else 'L20 mean'}, "
                             f"{'x/col' if q == 'h' else 'y/row'} query", fontsize=8)
                ax.set_xticks([-8, 0, 8])
                ax.set_yticks([-8, 0, 8])
    fig.supxlabel('token column offset from target')
    fig.supylabel('token row offset from target')
    cb = fig.colorbar(im, ax=axes, shrink=0.7)
    cb.set_label('mean share of image attention per token (log scale)')
    save(fig, out, 'aligned_attention.png')


# ----------------------------------------------------------------------------- Q2

def fig_q2(S: dict, out: Path) -> None:
    Q = S.get('q2', {})
    conds = list(Q)
    names = [('model', 'model answer', GREY), ('H11_local', 'L20 H11 peak', C[0]), ('L20mean_local', 'L20 mean peak', C[1]),
             ('best', 'best head peak (cross-fitted)', C[2])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for j, (split, title) in enumerate((('all', 'all answers'), ('model_correct', 'model correct'), ('model_wrong', 'model wrong'))):
        ax = axes[j]
        for i, (key, lab, col) in enumerate(names):
            ys, lo, hi = [], [], []
            for c in conds:
                if key == 'model':
                    p = Q[c]['model'] if split == 'all' else {'p': 1.0 if split == 'model_correct' else 0.0, 'ci': [np.nan, np.nan]}
                else:
                    p = Q[c][key][split]
                ys.append(p['p'])
                lo.append(p['p'] - p['ci'][0] if np.isfinite(p['ci'][0]) else 0)
                hi.append(p['ci'][1] - p['p'] if np.isfinite(p['ci'][1]) else 0)
            if key == 'model' and split != 'all':
                continue
            ax.errorbar(np.array(ys), np.arange(len(conds)) + (i - 1.5) * 0.18, xerr=[lo, hi], fmt='o', ms=4,
                        color=col, label=lab, elinewidth=1)
        ax.set_yticks(range(len(conds)), [f"{c} (n={Q[c]['n']}, model wrong {Q[c]['H11']['model_wrong']['n']})" for c in conds], fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel('hit rate (Wilson 95% CI)')
        ax.set_title(title)
        ax.invert_yaxis()
    axes[0].legend(frameon=False, fontsize=7, loc='lower left')
    save(fig, out, 'attention_answer.png')


# ----------------------------------------------------------------------------- Q3 / Q4

def _share_bar(ax, tab: dict, names: list[str], color: str, offset: float, label: str) -> None:
    m = [tab[n]['mean'] if n in tab else np.nan for n in names]
    lo = [tab[n]['mean'] - tab[n]['ci'][0] if n in tab else 0 for n in names]
    hi = [tab[n]['ci'][1] - tab[n]['mean'] if n in tab else 0 for n in names]
    ax.bar(np.arange(len(names)) + offset, m, width=0.38, color=color, yerr=[lo, hi], error_kw={'lw': 0.8}, label=label)


GROUP_ORDER = ['all_img', 'obj', 'all_but_obj', 'above', 'row_left', 'inbox', 'row_right', 'below_left', 'below_cols',
               'below_right', 'ring1', 'ring2', 'ring3_4', 'ring5p', 'text']


def fig_kv_groups(S: dict, out: Path) -> None:
    Q = S.get('q3', {})
    conds = [c for c in ('disc_point', 't6g10_rowcol', 't6g10_point', 'rc_point') if c in Q]
    if not conds:
        return
    fig, axes = plt.subplots(len(conds), 2, figsize=(13, 3.2 * len(conds) + 0.5), squeeze=False, sharey=True)
    for i, c in enumerate(conds):
        for j, axis in enumerate(('h', 'v')):
            ax = axes[i, j]
            kinds = [('move', 'true move (ViT + decoder)', C[0]), ('splice', 'splice (decoder only)', C[1])] if c != 'rc_point' \
                else [('swap', 'content swap (decoder only)', C[1])]
            for k, (kind, lab, col) in enumerate(kinds):
                key = f'{kind}_{axis}'
                if key in Q[c]:
                    n = Q[c][key]['all_img']['n']
                    _share_bar(ax, Q[c][key], GROUP_ORDER, col, (k - 0.5) * 0.4 if len(kinds) == 2 else 0, f'{lab} (n={n})')
            ax.axhline(0, color=GREY, lw=0.8)
            ax.axhline(1, color=GREY, lw=0.6, ls=':')
            ax.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER, rotation=60, fontsize=7)
            ax.set_title(f"{COND_LABEL[c]}, {'x / column (horizontal move)' if axis == 'h' else 'y / row (vertical move)'}", fontsize=9)
            ax.legend(frameon=False, fontsize=7)
            if j == 0:
                ax.set_ylabel('recovered fraction r\n(mean, bootstrap 95% CI)')
    fig.subplots_adjust(hspace=0.75)
    save(fig, out, 'kv_groups.png')


def fig_kv_rows(run: Path, out: Path) -> None:
    """Recovered fraction of single token rows / columns, aligned on the changed tokens D (splice and move)."""
    recs = sr.read_jsonl(run / 'kv.jsonl')
    conds = [c for c in ('disc_point', 't6g10_rowcol', 't6g10_point', 'rc_point') if any(r['cond'] == c for r in recs)]
    if not conds:
        return
    fig, axes = plt.subplots(2, len(conds), figsize=(3.6 * len(conds), 5.6), squeeze=False, sharey='row')
    for i, c in enumerate(conds):
        for j, (kind_axis, line_kind) in enumerate((('rows', 'row'), ('cols', 'col'))):
            ax = axes[j, i]
            for k, (kind, axis, col, lab) in enumerate([('move', 'v', C[0], 'move, y'), ('move', 'h', C[2], 'move, x'),
                                                        ('splice', 'v', C[1], 'splice, y'), ('splice', 'h', C[3], 'splice, x'),
                                                        ('swap', 'v', C[1], 'swap, y'), ('swap', 'h', C[3], 'swap, x')]):
                acc = {}
                for r in recs:
                    if r['cond'] != c:
                        continue
                    W = r['grid'][1]
                    for pr in r['pairs']:
                        if pr['kind'] != kind or pr['axis'] != axis:
                            continue
                        D = sorted(set(pr['target']) | set(pr['moved']))
                        ref = min(t // W for t in D) if line_kind == 'row' else min(t % W for t in D)
                        for v in pr['values']:
                            if v['axis'] != axis or v['dA'] is None or v['dA'] == v['dB']:
                                continue
                            gap = v['LD_B'] - v['LD']['none']
                            if abs(gap) < 2:
                                continue
                            for name, ld in v['LD'].items():
                                if name.startswith(line_kind) and name[len(line_kind):].isdigit():
                                    off = int(name[len(line_kind):]) - ref
                                    acc.setdefault(off, []).append((ld - v['LD']['none']) / gap)
                if not acc:
                    continue
                offs = sorted(o for o in acc if -6 <= o <= 10)
                ax.plot(offs, [np.mean(np.clip(acc[o], -3, 3)) for o in offs], marker='o', ms=3, color=col, lw=1.6,
                        ls='-' if kind == 'move' else '--', label=lab)
            ax.axhline(0, color=GREY, lw=0.8)
            ax.axvline(0, color=GREY, lw=0.8, ls=':')
            ax.set_xlabel(f"token {line_kind} offset from the first {line_kind} of D")
            ax.set_title(f"{COND_LABEL[c]}: single token {line_kind}s", fontsize=9)
            if i == 0:
                ax.set_ylabel('recovered fraction r (mean)')
            ax.legend(frameon=False, fontsize=6)
    fig.subplots_adjust(hspace=0.45)
    save(fig, out, 'kv_rows_cols.png')


def fig_layer_group(S: dict, out: Path) -> None:
    """Layer x token-group heatmaps per emitted value (first digit), per answer format."""
    Q = S.get('q4', {})
    keys = [k for k in Q if '|' in k and (k.endswith('|move') or k.endswith('|swap'))]
    if not keys:
        return
    order = ['disc_point', 'disc_point_xyk', 'disc_point_yxk', 'disc_bbox', 'disc_bbox_yxk', 't6g10_rowcol', 't6g10_colrow',
             't6g10_point', 't6g10_point_yxk', 'rc_point', 'rc_point_yxk']
    panels = []
    for cond in order:
        for kind in ('move', 'swap'):
            k = f'{cond}|{kind}'
            if k in Q:
                for val in sorted(Q[k], key=lambda s: int(s.split(':')[0])):
                    if Q[k][val].get('obj@20', {}).get('n', 0) >= 5:      # bbox max values rarely change first digit
                        panels.append((cond, val, Q[k][val]))
    groups = ['obj', 'all_but_obj', 'all_img']
    ncol = 6
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.0 * nrow + 0.6), squeeze=False)
    fig.subplots_adjust(hspace=0.9, wspace=0.15)
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=1.0)
    im = None
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (cond, val, row) in zip(axes.ravel(), panels):
        ax.axis('on')
        m = np.array([[row.get(f'{g}@{l}', {}).get('mean', np.nan) for l in FULL_LAYERS] for g in groups])
        im = ax.imshow(m, cmap=DIVERGING, norm=norm, aspect='auto')
        n = row.get('obj@20', {}).get('n', 0)
        idx, key = val.split(':')
        ax.set_title(f'{cond}\n#{int(idx) + 1} {key} (n={n})', fontsize=7)
        first_col = any(ax is a for a in axes[:, 0])
        ax.set_yticks(range(3), ['target', 'non-target', 'all image'] if first_col else ['', '', ''], fontsize=6)
        ax.set_xticks(range(8), [str(l) for l in FULL_LAYERS], fontsize=6)
        ax.set_xlabel('layer', fontsize=6, labelpad=1)
    cb = fig.colorbar(im, ax=axes, shrink=0.5)
    cb.set_label('recovered fraction r (image KV of one layer from run B)')
    save(fig, out, 'layer_group_per_value.png')


def fig_order(S: dict, out: Path) -> None:
    """Target-object share and non-target share per emitted value under the two emission orders."""
    Q = S.get('q4', {})
    pairs = [('disc_point_xyk', 'disc_point_yxk'), ('disc_bbox', 'disc_bbox_yxk'), ('t6g10_rowcol', 't6g10_colrow'),
             ('t6g10_point', 't6g10_point_yxk'), ('rc_point', 'rc_point_yxk')]
    pairs = [p for p in pairs if f'{p[0]}|move' in Q or f'{p[0]}|swap' in Q]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.3 * len(pairs), 3.4), squeeze=False, sharey=True)
    for i, (a, b) in enumerate(pairs):
        ax = axes[0, i]
        labels, xs = [], 0
        for j, cond in enumerate((a, b)):
            k = f'{cond}|move' if f'{cond}|move' in Q else f'{cond}|swap'
            for val in sorted(Q.get(k, {}), key=lambda s: int(s.split(':')[0])):
                row = Q[k][val]
                for g, (grp, col) in enumerate((('obj', C[0]), ('all_but_obj', C[1]))):
                    if grp in row:
                        ax.bar(xs + (g - 0.5) * 0.38, row[grp]['mean'], width=0.38, color=col,
                               label=('target tokens' if g == 0 else 'all other image tokens') if (i == 0 and xs == 0) else None)
                idx, key = val.split(':')
                labels.append((xs, f"{key}\n#{int(idx) + 1}"))
                xs += 1
            xs += 0.6
        ax.set_xticks([x for x, _ in labels], [l.replace('_', '\n') for _, l in labels], fontsize=6)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_title(f'{a}  vs  {b}', fontsize=8)
        if i == 0:
            ax.set_ylabel('recovered fraction r (mean)')
    fig.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.06))
    save(fig, out, 'order.png')


def fig_step_attention(S: dict, out: Path) -> None:
    """Attention destination (target / target rows / target columns / other image / text) per emitted value."""
    A = S.get('q4', {}).get('attention', {})
    conds = [c for c in ('disc_point_xyk', 'disc_point_yxk', 't6g10_rowcol', 't6g10_colrow', 'rc_point', 'rc_point_yxk') if c in A]
    if not conds:
        return
    gnames = ['target', 'target rows', 'target columns', 'other image', 'text']
    fig, axes = plt.subplots(1, len(conds), figsize=(2.8 * len(conds), 3.6), squeeze=False, sharey=True)
    for i, c in enumerate(conds):
        ax = axes[0, i]
        vals = sorted(A[c], key=lambda s: int(s.split(':')[0]))
        for j, val in enumerate(vals):
            m = np.array(A[c][val])                                     # (8, 16, 5)
            share = m[4].mean(0)                                         # layer-20 head mean
            bottom = 0
            for g in range(5):
                ax.bar(j, share[g], bottom=bottom, color=(C + [GREY])[g if g < 4 else 8 - 0], width=0.6,
                       label=gnames[g] if (i == 0 and j == 0) else None, edgecolor='white', linewidth=1)
                bottom += share[g]
        ax.set_xticks(range(len(vals)), [v.split(':')[1] for v in vals])
        ax.set_title(c, fontsize=8)
        if i == 0:
            ax.set_ylabel('layer-20 attention share (head mean)')
    fig.legend(frameon=False, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))
    save(fig, out, 'step_attention.png')


# ----------------------------------------------------------------------------- Q5

def fig_flow(S: dict, out: Path) -> None:
    F = S.get('q5', {})
    conds = [c for c in ('disc_point', 't6g10_point', 'rc_point') if c in F]
    if not conds:
        return
    fig, axes = plt.subplots(2, len(conds), figsize=(4.4 * len(conds), 7.2), squeeze=False)
    pos_lab = {'target_mean': 'target tokens', 'img_mean': 'all image tokens', 'img_last': 'last image token',
               'after_target': 'token after target', 'text_mean': 'post-image text', 'prompt_last': 'last prompt token',
               'q_x': 'x digit query', 'q_y': 'y digit query'}
    for i, c in enumerate(conds):
        f = F[c]
        ax = axes[0, i]
        r2 = np.array(f['r2'])
        for p, name in enumerate(f['positions']):
            ax.plot(range(r2.shape[0]), np.clip(r2[:, p], -0.2, 1), color=(C + [GREY])[p], lw=1.6, label=pos_lab[name])
        for l in FULL_LAYERS:
            ax.axvline(l, color=GREY, lw=0.4, ls=':')
        ax.set_ylim(-0.2, 1.02)
        ax.set_xlabel('site (0 = decoder input, k = output of layer k)')
        ax.set_title(f"{COND_LABEL[c]} (n={f['n']}): linear readability of (x, y)", fontsize=9)
        if i == 0:
            ax.set_ylabel('cross-validated R$^2$ (ridge)')
        ax.legend(frameon=False, fontsize=6)
        ax = axes[1, i]
        sh = f['img_share']
        for j, (k, lab) in enumerate((('text', 'post-image prompt text (mean)'), ('prompt_last', 'last prompt token'),
                                      ('answer_other', 'other answer tokens'), ('q_x', 'x digit query'), ('q_y', 'y digit query'))):
            m = np.array(sh[k]).mean(1)
            ax.plot(FULL_LAYERS, m, marker='o', ms=4, color=C[j], lw=1.8, label=lab)
        ax.set_ylim(0, 1)
        ax.set_xticks(FULL_LAYERS)
        ax.set_xlabel('full-attention layer')
        if i == 0:
            ax.set_ylabel('share of attention on image tokens\n(mean over 16 heads)')
        ax.legend(frameon=False, fontsize=6)
    save(fig, out, 'flow.png')


# ----------------------------------------------------------------------------- Q6

def fig_field(run: Path, source: Path, out: Path, S: dict, cond: str = 't6g20_rowcol', site: int = 16) -> None:
    """Decoded x and y of every visual token (held-out ridge field) for two correct and two column +1 images."""
    f = run / 'field' / f'{cond}.npz'
    q6 = S.get('q6', {}).get(cond, {}).get(str(site))
    if not f.exists() or q6 is None:
        return
    z = np.load(f)
    D_all = z[q6['lambda_key']]
    cases = {c['id']: c for c in json.loads((source / 'cases.json').read_text())}
    gen = {g['id']: g for g in sr.read_jsonl(source / 'gen.jsonl')}
    ids = [str(i) for i in z['ids']]
    ok = [i for i, cid in enumerate(ids) if gen[cid]['score']['correct']][:2]
    bad = [i for i, cid in enumerate(ids) if gen[cid]['score'].get('d_col') == 1][:2]
    sel = ok + bad
    fig, axes = plt.subplots(2, len(sel), figsize=(3.0 * len(sel) + 1.0, 6.0), squeeze=False)
    im = None
    for j, i in enumerate(sel):
        H, W = z['grids'][i]
        D = D_all[i].reshape(H, W, 2)
        cid = ids[i]
        target = cr.tokens_overlapping(cases[cid]['truth']['box'], (int(H), int(W)))
        tr, tc = np.array(target) // W, np.array(target) % W
        for k in range(2):
            ax = axes[k, j]
            im = ax.imshow(D[:, :, k], cmap='viridis', vmin=0, vmax=1000)
            ax.add_patch(plt.Rectangle((tc.min() - 0.5, tr.min() - 0.5), tc.max() - tc.min() + 1, tr.max() - tr.min() + 1,
                                       fill=False, ec='#e34948', lw=1.5))
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"decoded {'x' if k == 0 else 'y'}", fontsize=9)
        sc = gen[cid]['score']
        axes[0, j].set_title(f"{'correct' if sc['correct'] else 'column +1'}\n{cid}", fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.7)
    cb.set_label('decoded coordinate of the token (0-1000)')
    fig.suptitle(f'Held-out position field, {cond}, site {site}', fontsize=9)
    save(fig, out, f'field_{cond}.png')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    S = json.loads((args.run / 'summary.json').read_text())
    for fn in (fig_heads, fig_loo, fig_rowcol_attention, fig_q2, fig_kv_groups, fig_layer_group, fig_order,
               fig_step_attention, fig_flow):
        fn(S, args.out)
    fig_rowstart_maps(args.source, args.out)
    fig_kv_rows(args.run, args.out)
    for cond in ("t6g20_rowcol", "t6g10_point"):
        fig_field(args.run, args.source, args.out, S, cond)


if __name__ == '__main__':
    main()
