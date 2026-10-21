"""Experiment 3.6 figures (static PNG for the analysis documents)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from analysis_tools.position import coordinate_value as cv
from analysis_tools.position.coordinate_readout_run import read_jsonl

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUES = LinearSegmentedColormap.from_list('blues', ['#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b'])
DIVERGING = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#eb6834'])
ROUND = (250, 333, 500, 667, 750)

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 2,
                     'legend.frameon': False, 'figure.dpi': 130, 'savefig.bbox': 'tight'})


def save(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path.name)


def fig_tokenizer(res: dict, out: Path):
    tok = res['tokenizer']
    names = list(tok)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(names) + 1.0))
    ax.axis('off')
    for i, n in enumerate(names):
        y = len(names) - i
        ax.text(0, y, n, ha='left', va='center', fontsize=9, color=INK, family='monospace')
        ax.text(0.17, y, f"{tok[n]['n_not_one_digit_per_token']}/{tok[n]['n_values']}", ha='left', va='center',
                fontsize=9, color=MUTED)
        x = 0.27
        for piece in tok[n]['tokens_357']:
            digit = piece.strip().isdigit()
            w = 0.012 * max(len(repr(piece)) - 2, 1) + 0.012
            ax.add_patch(plt.Rectangle((x, y - 0.32), w, 0.64, facecolor='#cde2fb' if digit else '#f0efec',
                                       edgecolor='#2a78d6' if digit else GRID, lw=1))
            ax.text(x + w / 2, y, repr(piece)[1:-1], ha='center', va='center', fontsize=8, family='monospace')
            x += w + 0.004
    ax.text(0, len(names) + 1, 'context', fontsize=9, weight='bold')
    ax.text(0.17, len(names) + 1, 'violations', fontsize=9, weight='bold')
    ax.text(0.27, len(names) + 1, 'tokens of the context with n = 357 (digit tokens highlighted)', fontsize=9, weight='bold')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.3, len(names) + 1.5)
    save(fig, out / 'tokenizer.png')


def fig_inputs(run: Path, out: Path):
    recs = {r['id']: r for r in read_jsonl(run / 'gen' / 'records.jsonl')}
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    pick = ['g2d_x128_y096', 'g2d_x304_y336', 'sweep_h_400']
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    for ax, rid in zip(axes, pick):
        image, truth, _ = cv.disc_input(cases[rid])
        ax.imshow(image)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color(MUTED)
        t = recs[rid]['truth_units']
        e = recs[rid]['score'].get('coords')
        ax.set_title(f'truth ({t[0]:.0f}, {t[1]:.0f})  output ({e[0]:.0f}, {e[1]:.0f})', fontsize=9)
        px = np.array(e) * 512 / 1000
        ax.plot(*px, marker='x', color=INK, ms=9, mew=2)
    fig.text(0.02, -0.04, 'Prompt: ' + cv.POINT_PROMPT, fontsize=8, wrap=True, color=MUTED)
    ex = recs[pick[1]]
    x, y = (str(int(v)) for v in ex['score']['coords'])
    roles = [f'x{k}: state at {prev!r} predicts {d!r}' for k, (prev, d) in enumerate(zip([' ['] + list(x[:-1]), x))]
    roles += [f'y{k}: state at {prev!r} predicts {d!r}' for k, (prev, d) in enumerate(zip([' '] + list(y[:-1]), y))]
    fig.text(0.02, -0.10, 'Answer: ' + ex['text'].replace('\n', ' '), fontsize=8, color=MUTED)
    fig.text(0.02, -0.16, 'Roles (middle example): ' + '; '.join(roles), fontsize=8, color=MUTED)
    save(fig, out / 'inputs.png')


def fig_behaviour(A: dict, res: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, name, lab in zip(axes, ('sweep_h', 'sweep_v'), ('x', 'y')):
        t, e = A[f'{name}_truth'], A[f'{name}_emitted']
        ax.plot([0, 1000], [0, 1000], color=MUTED, lw=1, ls='--', label='output = truth')
        ax.step(t, e, where='mid', color=SERIES[0], label='model output')
        for v in ROUND:
            ax.axhline(v, color=SERIES[1], lw=0.6, alpha=0.5)
        ax.set_xlabel(f'true {lab} (0-1000)')
        ax.set_ylabel(f'output {lab}')
        ax.set_title(f'1D sweep along {lab} (step 2 px = 3.9 units)', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
    axes[1].text(1000, 0, 'thin orange lines: 250, 333, 500, 667, 750', fontsize=7, color=MUTED, ha='right', va='bottom')
    save(fig, out / 'staircase.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, a in zip(axes, 'xy'):
        b = [d for d in res['behaviour']['bias'][a] if d['mean'] is not None]
        c = np.array([(d['bin'][0] + d['bin'][1]) / 2 for d in b])
        m = np.array([d['mean'] for d in b])
        lo = np.array([d['ci'][0] for d in b])
        hi = np.array([d['ci'][1] for d in b])
        ax.axhline(0, color=MUTED, lw=1)
        ax.fill_between(c, lo, hi, color=SERIES[0], alpha=0.2, lw=0)
        ax.plot(c, m, color=SERIES[0], marker='o', ms=4)
        ax.set_xlabel(f'true {a} (bins of 62.5 units)')
        ax.set_ylabel(f'mean signed error in {a}')
        ax.set_xlim(0, 1000)
    save(fig, out / 'bias.png')
    T, E = A['all_truth'], A['all_emitted']
    fig, ax = plt.subplots(figsize=(11, 3.4))
    bins = np.arange(0, 1001, 5)
    ax.hist(np.rint(T.ravel()), bins=bins, color=MUTED, alpha=0.5, label='truth (rounded)')
    ax.hist(E.ravel(), bins=bins, color=SERIES[0], alpha=0.85, label='model output')
    ax.set_yscale('log')
    ax.set_xlabel('coordinate value (both axes, all disc answers)')
    ax.set_ylabel('count (log)')
    ax.legend(fontsize=8)
    save(fig, out / 'values.png')


def fig_force(res: dict, out: Path):
    f = res['force']
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.36
    for j, a in enumerate('xy'):
        ms = [f[a][f'forced_{k}']['mean_abs_error'] for k in (0, 1, 2)]
        ci = np.array([f[a][f'forced_{k}']['ci'] for k in (0, 1, 2)])
        x = np.arange(3) + (j - 0.5) * w
        ax.bar(x, ms, w * 0.92, color=SERIES[j], label=a)
        ax.errorbar(x, ms, yerr=[np.array(ms) - ci[:, 0], ci[:, 1] - np.array(ms)], fmt='none', ecolor=INK, lw=1)
        for xi, m in zip(x, ms):
            ax.text(xi, m + 1, f'{m:.1f}', ha='center', fontsize=8, color=INK)
    ax.set_xticks(range(3), ['free', 'true first digit forced', 'true first two digits forced'])
    ax.set_ylabel('mean |output - truth|')
    ax.legend(fontsize=8)
    save(fig, out / 'force.png')


def fig_probes(A: dict, res: dict, out: Path):
    names = res['probes']['x0']['targets']
    # Reference: mean |output - truth| of the model on the same test images (both sweeps).
    ex = np.concatenate([np.abs(A['sweep_h_emitted'] - A['sweep_h_truth']), np.abs(A['sweep_v_other'] - 500.0)])
    ey = np.concatenate([np.abs(A['sweep_v_emitted'] - A['sweep_v_truth']), np.abs(A['sweep_h_other'] - 500.0)])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, role in zip(axes, ('last', 'x0', 'x1', 'y0')):
        tab = A[f'probe_{role}']
        for j, (n, lab) in enumerate((('true', 'probe → true coordinate'), ('emitted', 'probe → output value'))):
            ax.plot(np.arange(33), tab[:, names.index(n), 1], color=SERIES[j], label=lab)
        ref = ey.mean() if role.startswith('y') else ex.mean()
        ax.axhline(ref, color=INK, ls='--', lw=1, label='model output error')
        ax.axhline(62.5, color=MUTED, ls=':', lw=1, label='one merged token (62.5)')
        ax.set_yscale('log')
        ax.set_title(f'role {role}' + (' (target: y)' if role.startswith('y') else ' (target: x)'), fontsize=10)
        ax.set_xlabel('layer (0 = embedding)')
    axes[0].set_ylabel('held-out MAE (0-1000 units, log)')
    axes[-1].legend(fontsize=7, loc='upper right')
    save(fig, out / 'probe_layers.png')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, role in zip(axes, ('x0', 'x1', 'x2')):
        tab = A[f'probe_{role}']
        for j, (T, lab) in enumerate(((100, 'period 100 (tens digit)'), (10, 'period 10 (units digit)'))):
            r = (tab[:, names.index(f'cos{T}'), 0] + tab[:, names.index(f'sin{T}'), 0]) / 2
            ax.plot(np.arange(33), np.clip(r, -0.2, 1), color=SERIES[j], label=lab)
        ax.set_title(f'role {role}', fontsize=10)
        ax.set_xlabel('layer')
    axes[0].set_ylabel('R² of cos/sin(2π v / T), v = output')
    axes[-1].legend(fontsize=8)
    save(fig, out / 'fourier.png')


def fig_geometry(A: dict, res: dict, out: Path):
    layers = A['pca_layers']
    for key, lab in (('sweep_h_x0', 'x'), ('sweep_v_y0', 'y'), ('sweep_h_x1', 'x')):
        P, t, e = A[f'geo_{key}_pca'], A[f'geo_{key}_truth'], A[f'geo_{key}_emitted']
        var = A[f'geo_{key}_pcavar']
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.6))
        for ax, l, p, v in zip(axes.ravel(), layers, P, var):
            ax.plot(p[:, 0], p[:, 1], color=GRID, lw=0.8, zorder=1)
            sc = ax.scatter(p[:, 0], p[:, 1], c=t, cmap=BLUES, s=12, vmin=0, vmax=1000, zorder=2, lw=0)
            for val in ROUND:
                m = e == val
                if m.any():
                    ax.scatter(p[m, 0], p[m, 1], facecolor='none', edgecolor=SERIES[1], s=26, lw=0.8, zorder=3)
            ax.set_title(f'L{l}  PC1-2 {100 * v[:2].sum():.0f}% var', fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(sc, ax=axes, shrink=0.6, label=f'true {lab}')
        fig.suptitle(f'{key}: residual states along the sweep (orange rings: output ∈ {{250, 333, 500, 667, 750}})',
                     fontsize=10)
        save(fig, out / f'pca_{key}.png')
    P, T = A['geo_grid2d_x0_pca'], A['geo_grid2d_truth']
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6))
    for col, l in enumerate((8, 16, 24, 32)):
        k = list(layers).index(l)
        for row, a in enumerate('xy'):
            ax = axes[row, col]
            sc = ax.scatter(P[k][:, 0], P[k][:, 1], c=T[:, row], cmap=BLUES, s=6, vmin=0, vmax=1000, lw=0)
            ax.set_title(f'L{l}, color = true {a}', fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(sc, ax=axes, shrink=0.6, label='true coordinate')
    fig.suptitle('grid2d (29×29 disc positions): state at role x0, PC1-PC2', fontsize=10)
    save(fig, out / 'pca_grid2d_x0.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw={'width_ratios': [1.3, 1.3, 1]})
    for ax, key in zip(axes[:2], ('sweep_h_x0', 'sweep_v_y0')):
        S, t, e = A[f'geo_{key}_steps'], A[f'geo_{key}_truth'], A[f'geo_{key}_emitted']
        im = ax.imshow(np.log2(S.T), aspect='auto', origin='lower', cmap=DIVERGING, vmin=-2, vmax=2,
                       extent=[t[0], t[-1], -0.5, 32.5])
        ch = np.where(np.diff(e) != 0)[0]
        ax.scatter((t[ch] + t[ch + 1]) / 2, np.full(len(ch), 33.2), marker='v', s=10, color=INK, clip_on=False)
        ax.set_xlabel('true coordinate'), ax.set_ylabel('layer')
        ax.set_title(f'{key}: log2(step / median), ▼ output changes', fontsize=9)
        ax.grid(False)
    fig.colorbar(im, ax=axes[:2], shrink=0.8)
    ax = axes[2]
    for j, key in enumerate(('sweep_h_x0', 'sweep_v_y0')):
        g = res['geometry'][key]
        ax.plot(np.arange(33), g['auroc_step_at_value_change'], color=SERIES[j], label=f'{key}: output value change')
        ax.plot(np.arange(33), g['auroc_step_at_first_digit_change'], color=SERIES[j], ls='--',
                label=f'{key}: first digit change')
    ax.axhline(0.5, color=MUTED, lw=1)
    ax.set_xlabel('layer'), ax.set_ylabel('AUROC (step length)')
    ax.legend(fontsize=7)
    save(fig, out / 'steps.png')


def fig_tuning(A: dict, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, key in zip(axes, ('sweep_h_x0', 'sweep_h_x1', 'sweep_h_x2', 'sweep_v_y0')):
        L, t = A[f'tuning_{key}'], A[f'tuning_{key}_truth']
        p = np.exp(L)
        for d in range(10):
            if np.nanmax(p[:, d]) < 0.05:
                continue
            color = SERIES[d % 8] if d < 8 else MUTED
            ax.plot(t, p[:, d], color=color, lw=1.4)
            k = np.nanargmax(p[:, d])
            ax.text(t[k], p[k, d] + 0.02, str(d), color=INK, fontsize=8, ha='center')
        ax.set_title(f'{key}: P(next digit)', fontsize=9)
        ax.set_xlabel('true coordinate')
    axes[0].set_ylabel('probability')
    save(fig, out / 'tuning.png')


def fig_lens(A: dict, res: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    ax = axes[0]
    for j, role in enumerate(('x0', 'y0')):
        d = res['lens'][f'lens_{role}']
        ax.plot(np.arange(33), d['agree_emitted'], color=SERIES[j], label=f'{role}: = output digit')
        ax.plot(np.arange(33), d['agree_true'], color=SERIES[j], ls='--', label=f'{role}: = true digit')
    ax.set_xlabel('layer'), ax.set_ylabel('logit-lens argmax agreement')
    ax.legend(fontsize=7)
    for ax, role in zip(axes[1:], ('x0', 'y0')):
        x = np.arange(1, 33)
        ax.bar(x - 0.2, A[f'dla_{role}_attn'], 0.4, color=SERIES[0], label='attention / DeltaNet block')
        ax.bar(x + 0.2, A[f'dla_{role}_mlp'], 0.4, color=SERIES[1], label='MLP block')
        ax.set_xlabel('layer'), ax.set_ylabel('direct logit margin of output digit')
        ax.set_title(f'role {role}', fontsize=10)
        ax.legend(fontsize=7)
    save(fig, out / 'lens_dla.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, (key, title, cmap, lim) in zip(axes, (('heads_x0_r2', 'x0: head output → true x (R²)', BLUES, (0, 1)),
                                                 ('heads_y0_r2', 'y0: head output → true y (R²)', BLUES, (0, 1)),
                                                 ('heads_x0_dla', 'x0: head direct logit margin', DIVERGING, None))):
        M = A[key]
        if lim is None:
            m = np.abs(M).max()
            lim = (-m, m)
        im = ax.imshow(np.clip(M, *lim), cmap=cmap, vmin=lim[0], vmax=lim[1], aspect='auto')
        ax.set_yticks(range(len(cv.FULL_LAYERS)), [f'L{l}' for l in cv.FULL_LAYERS])
        ax.set_xticks(range(16))
        ax.set_xlabel('head')
        ax.set_title(title, fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, out / 'heads.png')


def fig_steer(A: dict, out: Path):
    if 'steer_layers' not in A:
        return
    layers, deltas = A['steer_layers'], A['steer_deltas']
    fig, axes = plt.subplots(2, len(layers), figsize=(3 * len(layers), 6.2), sharex=True, sharey=True)
    for row, method in enumerate(('linear', 'manifold')):
        tab = A[f'steer_{method}']
        for li, l in enumerate(layers):
            ax = axes[row, li]
            ax.plot([deltas[0], deltas[-1]], [deltas[0], deltas[-1]], color=MUTED, ls='--', lw=1)
            for j in range(tab.shape[1]):
                y = tab[li, j]
                ok = ~np.isnan(y)
                ax.plot(deltas[ok], y[ok], color=BLUES(0.3 + 0.7 * A['steer_truth'][j] / 1000), lw=0.9, alpha=0.8)
            ax.plot(deltas, np.nanmean(tab[li], 0), color=SERIES[1], lw=2)
            ax.set_title(f'{method}, L{l}', fontsize=9)
            if row == 1:
                ax.set_xlabel('requested shift Δ')
        axes[row, 0].set_ylabel('output x shift')
    save(fig, out / 'steer.png')


def fig_interp(A: dict, out: Path):
    if 'interp_layers' not in A:
        return
    layers, alphas = A['interp_layers'], A['interp_alphas']
    offs = sorted(int(k.split('_')[1]) for k in A if k.startswith('interp_') and k.split('_')[1].isdigit() and k.count('_') == 1)
    fig, axes = plt.subplots(len(offs), len(layers), figsize=(3 * len(layers), 3.0 * len(offs)), sharex=True, squeeze=False)
    for r, off in enumerate(offs):
        V, T = A[f'interp_{off}'], A[f'interp_{off}_truth']
        for c, l in enumerate(layers):
            ax = axes[r, c]
            for j in range(V.shape[1]):
                ax.plot(alphas, V[c, j], color=BLUES(0.3 + 0.7 * T[j, 0] / 1000), lw=1, marker='o', ms=2)
            for v in ROUND:
                ax.axhline(v, color=SERIES[1], lw=0.5, alpha=0.5)
            ax.set_title(f'L{l}, donor = base + {off}', fontsize=9)
            if r == len(offs) - 1:
                ax.set_xlabel('interpolation weight α')
        axes[r, 0].set_ylabel('output x')
    save(fig, out / 'interp.png')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    res = json.loads((args.run / 'results.json').read_text())
    A = dict(np.load(args.run / 'analysis.npz'))
    fig_tokenizer(res, args.figures)
    fig_inputs(args.run, args.figures)
    fig_behaviour(A, res, args.figures)
    if 'force' in res:
        fig_force(res, args.figures)
    if 'probes' in res:
        fig_probes(A, res, args.figures)
    if 'geometry' in res:
        fig_geometry(A, res, args.figures)
    if 'tuning_sweep_h_x0' in A:
        fig_tuning(A, args.figures)
    if 'lens' in res:
        fig_lens(A, res, args.figures)
    fig_steer(A, args.figures)
    fig_interp(A, args.figures)


if __name__ == '__main__':
    main()
