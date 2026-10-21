"""Figures of experiment 1.5 (ordinal mechanism); called from ordinal_mechanism_analysis.analyze."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.behavior import ordinal_mechanism as om

# categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GREY = '#8a8985'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'legend.frameon': False})


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_variants(output: Path, out: Path, cond: str = 'H64', base: int = 30) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(9, 9.5))
    cases = {c['id']: c for c in om.read_jsonl(output / 'cases.jsonl')}
    for ax, v in zip(axes, om.VARIANTS):
        c = cases[f'{cond}_b{base:03d}_{v}|direct']
        ax.imshow(Image.open(output / c['image']))
        t = c['truth']
        ax.set_title(f"{v}: count = {t['count']}, slot = {t['slot']}, ratio = {t['ratio']:.2f}", fontsize=10, loc='left')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    save(fig, out / 'variants.png')


DELTA_BINS = [(-99, -2, '≤ −2'), (-1, -1, '−1'), (0, 0, '0'), (1, 1, '+1'), (2, 99, '≥ +2')]
DELTA_COLORS = ['#104281', '#5598e7', '#d6d5d0', '#eb6834', '#9c3a12']


def fig_paired(summary: dict, out: Path) -> None:
    keys = [k for k in summary['routes']]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.3 * len(keys), 3.6), sharey=True)
    for ax, key in zip(np.atleast_1d(axes), keys):
        e = summary['routes'][key]
        bottom = np.zeros(4)
        for (lo, hi, lab), col in zip(DELTA_BINS, DELTA_COLORS):
            vals = []
            for v in ('shift', 'insL', 'delL', 'delR'):
                h = {int(a): b for a, b in e[v]['delta_hist'].items()}
                n = max(e[v]['n_paired'], 1)
                vals.append(sum(b for a, b in h.items() if lo <= a <= hi) / n)
            ax.bar(range(4), vals, bottom=bottom, color=col, edgecolor='white', linewidth=1, label=lab)
            bottom += vals
        ax.set_xticks(range(4), ['shift', 'insL', 'delL', 'delR'], rotation=45)
        ax.set_title(key.replace('|', '\n'), fontsize=9)
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel('fraction of paired answers')
    np.atleast_1d(axes)[-1].legend(title='answer − base answer', bbox_to_anchor=(1.02, 1), loc='upper left')
    save(fig, out / 'paired_change.png')


C_BINS = ((3, 5), (6, 8), (9, 12), (13, 18))


def fig_follow_by_c(summary: dict, out: Path, mode: str = 'direct') -> None:
    """Paired change per target-ordinal bin, pooled over cell sizes per axis (rows: H, V; columns: variants)."""
    fb = summary['follow_by_c']
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    for i, ax_name in enumerate(('H', 'V')):
        for j, v in enumerate(('shift', 'insL', 'delL', 'delR')):
            ax = axes[i, j]
            bottom = np.zeros(len(C_BINS))
            ns = []
            for lo, hi in C_BINS:
                ns.append(sum(len(l) for k, dd in fb.items() if k.startswith(ax_name) and k.endswith(f'|{mode}|{v}')
                              for c, l in dd.items() if lo <= int(c) <= hi))
            for (dlo, dhi, lab), col in zip(DELTA_BINS, DELTA_COLORS):
                vals = []
                for (lo, hi), n in zip(C_BINS, ns):
                    d = [x for k, dd in fb.items() if k.startswith(ax_name) and k.endswith(f'|{mode}|{v}')
                         for c, l in dd.items() if lo <= int(c) <= hi for x in l]
                    vals.append(sum(dlo <= x <= dhi for x in d) / max(n, 1))
                ax.bar(range(len(C_BINS)), vals, bottom=bottom, color=col, edgecolor='white', linewidth=1, label=lab)
                bottom += vals
            ax.set_xticks(range(len(C_BINS)), [f'{lo}–{hi}\n(n={n})' for (lo, hi), n in zip(C_BINS, ns)], fontsize=8)
            ax.set_title(f'{"columns (H)" if ax_name == "H" else "rows (V)"}: {v}', fontsize=10)
            ax.grid(False)
        axes[i, 0].set_ylabel('fraction of paired answers')
    for ax in axes[1]:
        ax.set_xlabel('target ordinal c (bins)')
    axes[0, -1].legend(title='answer − base answer', bbox_to_anchor=(1.02, 1), loc='upper left')
    save(fig, out / f'follow_by_c_{mode}.png')


def fig_base_offsets(summary: dict, out: Path) -> None:
    bo = summary['base_offsets']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, sgn, title in ((axes[0], 1, 'P(answer = c + 1), base images'), (axes[1], -1, 'P(answer = c − 1), base images')):
        for i, key in enumerate(sorted(bo)):
            cs = sorted(int(c) for c in bo[key])
            y = [np.mean(np.array(bo[key][c] if c in bo[key] else bo[key][str(c)]) == sgn) for c in cs]
            ax.plot(cs, y, '-o', ms=4, lw=1.5, color=CAT[i % len(CAT)], label=key.replace('|', ' '))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('target ordinal c')
    axes[0].set_ylabel('fraction (4 bases per c direct, 2 think)')
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    save(fig, out / 'base_offsets.png')


def fig_probes(summary: dict, out: Path) -> None:
    for key, res in summary['probes'].items():
        layers = res['layers']
        names = [('unit', 'continuous coordinate'), ('count', 'count ordinal'), ('slot', 'slot index'),
                 ('answer', "model's answer"), ('gap_left', 'blank line precedes target')]
        fig, axes = plt.subplots(1, len(names) + 1, figsize=(3.0 * (len(names) + 1), 3.2), sharey=True)
        for ax, (t, title) in zip(axes, names):
            for i, role in enumerate(res['roles']):
                ax.plot(layers, res['r2'][t][role], '-o', ms=3, lw=1.5, color=CAT[i], label=role)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('layer')
        ax = axes[-1]
        for i, role in enumerate(res['roles']):
            ax.plot(layers, res['partial'][role], '-o', ms=3, lw=1.5, color=CAT[i], label=role)
        ax.set_title('count | coordinate (residual)', fontsize=10)
        ax.set_xlabel('layer')
        axes[0].set_ylabel('cross-validated $R^2$ (within condition)')
        axes[0].set_ylim(-0.1, 1.02)
        axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='position')
        fig.suptitle(f"Ridge probes, {key.replace('|', ', ')}", fontsize=11)
        save(fig, out / f"probes_{key.replace('|', '_')}.png")


def fig_classifier(summary: dict, out: Path) -> None:
    for key, res in summary['probes'].items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
        for ax, kind in zip(axes, ('linear', 'mlp')):
            for i, role in enumerate(res['roles']):
                rr = res['classifier'][role][kind]
                L = [r['layer'] for r in rr]
                ax.plot(L, [r['count_acc'] for r in rr], '-o', ms=3, lw=1.5, color=CAT[i], label=f'{role}: count')
                ax.plot(L, [r['control_acc'] for r in rr], '--', lw=1.2, color=CAT[i], label=f'{role}: control')
            ax.set_title(f'{kind} classifier', fontsize=10)
            ax.set_xlabel('layer')
        axes[0].set_ylabel('cross-validated accuracy')
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        fig.suptitle(f"Count-ordinal classifiers and control task, {key.replace('|', ', ')}", fontsize=11)
        save(fig, out / f"classifier_{key.replace('|', '_')}.png")


def fig_steer(summary: dict, out: Path) -> None:
    """Per axis: mean change of the ordinal vs steered layer (pooled over conditions), and the fraction of
    changed answers per target-ordinal bin at layer 24."""
    groups = [('m', '1', 'mean difference, 1 unit'), ('m', '3', 'mean difference, 3 units'),
              ('a', '10', 'linear answer direction, 10 units'), ('x36n', '', '3.6 x direction, norm of m 3'),
              ('rand', '', 'random, norm of m 3')]
    for mode, st in summary['steer'].items():
        pooled = st.get('pooled', {}) if st else {}
        if not pooled:
            continue
        axes_ = sorted(pooled)
        fig, axes = plt.subplots(2, len(axes_), figsize=(4.2 * len(axes_), 6.6), squeeze=False,
                                 gridspec_kw={'hspace': 0.45, 'wspace': 0.32})
        for col, axis in enumerate(axes_):
            e = pooled[axis]
            ax = axes[0, col]
            for gi, (g, mag, lab) in enumerate(groups):
                for sign, ls in ((1, '-'), (-1, '--')):
                    key = g + ('+' if sign > 0 else '-') + mag
                    ys = [e['by_layer'].get(f'{l}|{key}', {}).get('mean_change', np.nan) for l in (16, 20, 24, 28)]
                    if np.all(np.isnan(ys)):
                        continue
                    ax.plot((16, 20, 24, 28), ys, ls, marker='o', ms=4, lw=1.5, color=CAT[gi],
                            label=f'{lab} ({"+" if sign > 0 else "−"})')
            ax.axhline(0, color=GREY, lw=0.8)
            ax.set_title(f"{'columns (H)' if axis == 'H' else 'rows (V)'}, n = {e['n']}", fontsize=10)
            ax.set_xticks((16, 20, 24, 28))
            ax.set_xlabel('steered layer')
            ax.set_ylabel('mean change of the ordinal')
            ax = axes[1, col]
            bins = sorted({k.split('|')[2] for k in e['by_c']}, key=lambda b: int(b.split('-')[0]))
            x = np.arange(len(bins))
            labs = [('m+3', 0), ('m-3', 0), ('a+10', 2), ('x36n+', 3), ('rand+', 4)]
            labs = [(l, c) for l, c in labs if f'24|{l}|{bins[0]}' in e['by_c']]
            w = 0.8 / len(labs)
            for i, (lab, ci) in enumerate(labs):
                ys = [e['by_c'][f'24|{lab}|{b}']['p_changed'] for b in bins]
                ax.bar(x + (i - (len(labs) - 1) / 2) * w, [np.nan if y is None else y for y in ys], w * 0.9,
                       color=CAT[ci], hatch='///' if lab == 'm-3' else None, edgecolor='white', label=lab)
            ax.set_xticks(x, [f'{b}\n(n = {e["by_c"][f"24|m+3|{b}"]["n"]})' for b in bins])
            ax.set_xlabel('target ordinal c')
            ax.set_ylabel('fraction of answers changed (layer 24)')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7, loc='upper left')
        handles = {}
        for ax in axes[0]:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(lab, h)
        axes[0, -1].legend(handles.values(), handles.keys(), bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        fig.suptitle(f'Steering the state that predicts the long-axis ordinal ({mode})', fontsize=11)
        save(fig, out / f'steer_{mode}.png')


def fig_onset(summary: dict, out: Path) -> None:
    om_ = summary['onset_models']['conditions']
    conds = [c for c in om_ if c.endswith(('n16p0', 'n20p0', 'n24p0'))]
    cols = 3
    rows = int(np.ceil(len(conds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.9 * rows), sharey=True, gridspec_kw={'hspace': 0.45})
    names = [('subitize', 'subitizing + approximation'), ('ratio', 'ratio with biased N'),
             ('anchored', 'two-anchor proportion'), ('digit_prior', 'digit prior')]
    for ax, cond in zip(axes.flat, conds):
        e = om_[cond]
        cs = sorted(int(c) for c in e['observed_plus1'])
        ax.bar(cs, [e['observed_plus1'][str(c)] if str(c) in e['observed_plus1'] else e['observed_plus1'][c] for c in cs],
               color='#d6d5d0', width=0.8, label='observed')
        for i, (m, lab) in enumerate(names):
            p = e['models'][m]['pred_plus1']
            ax.plot(cs, [p[str(c)] if str(c) in p else p[c] for c in cs], lw=1.6, color=CAT[i], label=lab)
        ax.set_title(f"{cond} (n = {e['n_samples']})", fontsize=9)
        ax.grid(False)
    for ax in axes.flat[len(conds):]:
        ax.axis('off')
    axes.flat[0].set_ylabel('P(answer = c + 1)')
    axes.flat[cols - 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('true column c')
    save(fig, out / 'onset_models.png')


def fig_row_bias(summary: dict, out: Path) -> None:
    tb = summary['row_biases']['tables']
    keys = list(tb)
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(keys) + 1))
    y = np.arange(len(keys))
    for i, (k, lab) in enumerate((('row+1', 'row +1'), ('row-1', 'row −1'), ('col+1', 'column +1'),
                                  ('col-1', 'column −1'))):
        ax.barh(y + (i - 1.5) * 0.2, [tb[key][k]['p'] for key in keys], height=0.2, color=CAT[i], label=lab)
    ax.set_yticks(y, keys, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('fraction of row/column answers')
    ax.legend(loc='lower right')
    save(fig, out / 'row_bias.png')


def fig_strategies(summary: dict, out: Path) -> None:
    st = summary['strategies']
    if not st:
        return
    keys = list(om.STRATEGY_PATTERNS) + ['full_enumeration']
    fig, ax = plt.subplots(figsize=(9, 3))
    x = np.arange(len(keys))
    for i, cond in enumerate(st):
        ax.bar(x + (i - 1) * 0.27, [st[cond][k]['p'] for k in keys], width=0.27, color=CAT[i], label=cond)
    ax.set_xticks(x, keys, rotation=20)
    ax.set_ylabel('fraction of responses')
    ax.legend()
    save(fig, out / 'strategies.png')


def make_all(output: Path, summary: dict, rows: list[dict], out: Path) -> None:
    fig_variants(output, out)
    fig_paired(summary, out)
    fig_follow_by_c(summary, out, 'direct')
    fig_follow_by_c(summary, out, 'think')
    fig_base_offsets(summary, out)
    fig_probes(summary, out)
    fig_classifier(summary, out)
    fig_steer(summary, out)
    fig_onset(summary, out)
    fig_row_bias(summary, out)
    fig_strategies(summary, out)
    cases = {c['id']: c for c in om.read_jsonl(output / 'cases.jsonl')}
    fig_manifold(output, cases, out, [('H64', 'direct', 24, 'col0'), ('H64', 'direct', 16, 'vis'),
                                      ('V64', 'direct', 24, 'row0'), ('V64', 'direct', 16, 'vis')])


# ----------------------------------------------------------------------------- manifold (PI reference style)

def _manifold_panel(fig, spec, P: np.ndarray, color: np.ndarray, view: str, title: str | None, subtitle: str | None):
    ax = fig.add_subplot(spec, projection='3d')
    lo, hi = P.min(0), P.max(0)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    floor = lo[2] - 0.35 * (hi[2] - lo[2])
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]])
    ax.plot(corners[:, 0], corners[:, 1], floor, color='#55544f', lw=0.8)
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), floor), s=16, c='#8a8985', alpha=0.28, edgecolors='none')
    order = np.argsort(P[:, 2] if view == 'top' else P[:, 1])
    sc = ax.scatter(P[order, 0], P[order, 1], P[order, 2], c=color[order], cmap='viridis', s=22, alpha=0.95,
                    edgecolors='#20201e', linewidths=0.2, depthshade=True)
    if view == 'top':
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=22, azim=-58)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_zlim(floor, hi[2])
    ax.xaxis.pane.set_visible(False); ax.yaxis.pane.set_visible(False); ax.zaxis.pane.set_visible(False)
    ax.grid(False)
    ax.set_axis_off()
    if title:
        ax.set_title(title + ('\n' + subtitle if subtitle else ''), fontsize=10)
    return ax, sc


def fig_manifold(output: Path, cases: dict, out: Path, panels: list[tuple[str, str, int, str]]) -> None:
    """3D PCA point clouds (side view top row, top view bottom row) coloured by the count ordinal k.

    panels: (condition list joined by '+', mode, layer, role).
    """
    from analysis_tools.behavior.ordinal_mechanism_analysis import load_features
    fig = plt.figure(figsize=(4.2 * len(panels), 8.2))
    gs = fig.add_gridspec(2, len(panels), hspace=0.02, wspace=0.02)
    roles = ['last', 'row0', 'col0', 'vis']
    sc = None
    for j, (conds, mode, layer, role) in enumerate(panels):
        ids, H, _ = load_features(output, conds.split('+'), mode, layers=[layer], roles=[roles.index(role)])
        X = H[:, 0, 0]
        cs = [cases[i] for i in ids]
        cond = np.array([c['condition'] for c in cs])
        for cn in np.unique(cond):
            X[cond == cn] -= X[cond == cn].mean(0)
        k = np.array([c['truth']['count'] for c in cs])
        _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        P = (X - X.mean(0)) @ Vt[:3].T
        _, sc = _manifold_panel(fig, gs[0, j], P, k, 'side', f'{conds}, {role}', f'Qwen3.5-9B, layer {layer}, {mode}')
        _manifold_panel(fig, gs[1, j], P, k, 'top', None, None)
        for row in range(2):
            box = fig.add_subplot(gs[row, j])
            box.set_xticks([]); box.set_yticks([])
            box.patch.set_alpha(0)
            for s in box.spines.values():
                s.set_visible(True); s.set_linewidth(0.6); s.set_color('#8a8985')
            box.grid(False)
    cb = fig.colorbar(sc, ax=fig.axes, shrink=0.5, pad=0.02)
    cb.set_label('count ordinal k (instance lines up to the target)')
    save(fig, out / 'manifold.png')
