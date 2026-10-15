"""Figures for experiment 3.4 (position probe refinement); reads results.json and predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.position import position_probe_refinement as pr

# Categorical slots in fixed order (validated reference palette, light mode).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, MUTED, GRID = '#222222', '#6b6b6b', '#e4e4e4'
PROBES = [('rp256', 'RP256 ridge', 'o'), ('rp1024', 'RP1024 ridge', 's'), ('full', 'Full ridge', 'D'),
          ('mlp64', 'MLP64', '^'), ('mlp256', 'MLP256', 'v'), ('mlp1024', 'MLP1024', 'P')]
FULL_ATTENTION = (4, 8, 12, 16, 20, 24, 28, 32)
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'legend.frameon': False,
                     'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def stage_x(stage: str) -> float:
    if stage == 'patch_content':
        return -3
    if stage.startswith('vit_'):
        return int(stage.split('_')[1])
    if stage == 'merger':
        return 0
    return pr.decoder_layer(stage)


def probe_value(res: dict, stage: str, probe: str, key: str = 'mae'):
    if probe in ('rp256', 'rp1024', 'full'):
        m = res['ridge']['test512'][stage][probe]['all']
    else:
        m = res['mlp']['test512'].get(stage, {}).get(probe)
        if m is None:
            return None, None
    ci = m.get(f'{key}_ci95')
    return m[key], ci


def layer_curves(res: dict, stages: list[str], out: Path, key: str, ylabel: str, name: str) -> None:
    vit = [s for s in stages if pr.is_vision(s)]
    dec = ['merger'] + [s for s in stages if s.startswith('decoder_')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1, 1.25]), sharey=True)
    for ax, group, title in ((axes[0], vit, 'ViT (patch tokens)'), (axes[1], dec, 'Merger and decoder (merged tokens)')):
        for k, (probe, label, marker) in enumerate(PROBES):
            pts = [(stage_x(s), *probe_value(res, s, probe, key)) for s in group]
            pts = [p for p in pts if p[1] is not None]
            if not pts:
                continue
            xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            ax.plot(xs, ys, color=SERIES[k], lw=2 if probe in ('full', 'mlp256') else 1.4, marker=marker,
                    ms=4 if len(xs) > 12 else 6, label=label)
            if probe == 'full' and pts[0][2] is not None:
                lo = [p[2][0] for p in pts]
                hi = [p[2][1] for p in pts]
                ax.fill_between(xs, lo, hi, color=SERIES[k], alpha=.18, lw=0)
        base = res['ridge']['test512']['patch_content']['full']['all'][key]
        ax.axhline(base, color=MUTED, ls='--', lw=1)
        ax.text(1 if group is dec else 10, base, ' patch content (full ridge)', color=MUTED, va='bottom', fontsize=8)
        ax.set_title(title, color=INK)
        if group is dec:
            for layer in FULL_ATTENTION:
                ax.axvline(layer, color=GRID, lw=3, zorder=0)
            ax.set_xticks([0] + list(FULL_ATTENTION))
            ax.set_xticklabels(['merger'] + [f'{l}\nFA' for l in FULL_ATTENTION])
            ax.set_xlabel('decoder layer (FA = full attention)')
        else:
            ax.set_xticks([-3, 0, 9, 18, 27])
            ax.set_xticklabels(['content', '+pos', '9', '18', '27'])
            ax.set_xlabel('ViT block')
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=6, fontsize=8)
    fig.savefig(out / name)
    plt.close(fig)


def fa_breakdown(res: dict, out: Path) -> None:
    """Token-unit error around each early full-attention layer (decoder layers 3-17)."""
    layers = [3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
    names = {pr.decoder_layer(s): s for s in res['ridge']['test512'] if s.startswith('decoder_')}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, probe in ((2, 'full'), (4, 'mlp256')):
        vals = [probe_value(res, names[l], probe, 'token_mae') for l in layers]
        x = np.array([i for i, v in enumerate(vals) if v[0] is not None])
        vals = [v for v in vals if v[0] is not None]
        if not vals:
            continue
        ys = np.array([v[0] for v in vals])
        err = np.array([[v[0] - v[1][0], v[1][1] - v[0]] for v in vals]).T
        ax.errorbar(x + (k - 3) * .12, ys, yerr=err, fmt=PROBES[k][2], color=SERIES[k], ms=6, capsize=2, label=PROBES[k][1])
    for i, l in enumerate(layers):
        if l in FULL_ATTENTION:
            ax.axvspan(i - .4, i + .4, color=GRID, zorder=0)
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([f'{l}\nFA' if l in FULL_ATTENTION else str(l) for l in layers])
    ax.set_xlabel('decoder layer (shaded: full attention; others: Gated DeltaNet)')
    ax.set_ylabel('token-unit MAE (merged tokens)')
    ax.legend()
    fig.savefig(out / 'fa_vs_deltanet.png')
    plt.close(fig)


def capacity(res: dict, out: Path) -> None:
    stages = res['meta']['lc_stage_names']
    fig, axes = plt.subplots(1, len(stages), figsize=(3 * len(stages), 3.2), sharey=True)
    for ax, s in zip(axes, stages):
        dim = 4096
        params = dict(rp256=258 * 2, rp1024=1026 * 2, full=(dim + 1) * 2)
        for k, (probe, label, marker) in enumerate(PROBES):
            v, _ = probe_value(res, s, probe)
            if v is None:
                continue
            p = params.get(probe) or res['mlp']['test512'][s][probe]['params']
            ax.scatter(p, v, color=SERIES[k], marker=marker, s=40, label=label, zorder=3)
            # Same probe family on patch content, drawn at the decoder probe's parameter count for alignment.
            c, _ = probe_value(res, 'patch_content', probe)
            ax.scatter(p, c, facecolor='white', edgecolor=SERIES[k], marker=marker, s=40, zorder=3)
        ax.set_xscale('log')
        ax.set_title(s.replace('_linear_attention', ' (DeltaNet)').replace('_full_attention', ' (FA)'), fontsize=9)
        ax.set_xlabel('probe parameters')
    axes[0].set_ylabel('test MAE (0-1000); hollow = patch content')
    axes[-1].legend(fontsize=7, loc='center right')
    fig.savefig(out / 'capacity.png')
    plt.close(fig)


def learning_curves(res: dict, out: Path) -> None:
    lc = res['learning_curve']
    stages = list(lc)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, fam, title in ((axes[0], 'ridge', 'Full ridge (all tokens)'), (axes[1], 'mlp256', 'MLP256 (64 tokens / image)')):
        for k, s in enumerate(stages):
            d = lc[s][fam]
            sizes = sorted({int(n.split('_n')[1].split('_')[0]) if fam == 'ridge' else int(n[1:].split('_')[0])
                            for n in d if (fam != 'ridge' or n.startswith('lc_full'))})
            rows = []
            for n in sizes:
                vals = [v for kk, v in d.items() if (kk.startswith(f'lc_full_n{n}_') if fam == 'ridge' else kk.startswith(f'n{n}_'))]
                rows.append((n, np.mean(vals), min(vals), max(vals)))
            rows = np.array(rows)
            label = s.replace('_linear_attention', '').replace('_full_attention', ' FA')
            ax.plot(rows[:, 0], rows[:, 1], color=SERIES[k], marker='o', ms=5, lw=2, label=label)
            ax.fill_between(rows[:, 0], rows[:, 2], rows[:, 3], color=SERIES[k], alpha=.2, lw=0)
        ax.set_xscale('log')
        ax.set_xticks(pr.LC_SIZES)
        ax.set_xticklabels([str(v) for v in pr.LC_SIZES])
        ax.set_xlabel('training images')
        ax.set_title(title)
    axes[0].set_ylabel('test MAE (0-1000)')
    axes[1].legend(fontsize=8)
    fig.savefig(out / 'learning_curves.png')
    plt.close(fig)


def spatial_maps(res: dict, run: Path, out: Path) -> None:
    cases = json.loads((run / 'cases.json').read_text())['natural']
    test = [c for c in cases if c['split'] == 'test']
    cols = json.loads((run / 'predictions' / 'columns.json').read_text())
    preds = np.load(run / 'predictions' / 'test512.npz')
    stages = ['merger', [s for s in cols if pr.decoder_layer(s) == 14][0], [s for s in cols if pr.decoder_layer(s) == 32][0]]
    lab = np.concatenate([pr.token_labels(tuple(c['patch_hw']), False)[1] for c in test])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, s in zip(axes, stages):
        c = {n: i for i, n in enumerate(cols[s])}
        p = preds[s][:, [c['full_image_x'], c['full_image_y']]]
        err = np.abs(p - lab).mean(1)
        cell = np.clip((lab // 125).astype(int), 0, 7)
        grid = np.zeros((8, 8))
        np.add.at(grid, (cell[:, 1], cell[:, 0]), err)
        grid /= np.maximum(np.bincount(cell[:, 1] * 8 + cell[:, 0], minlength=64).reshape(8, 8), 1)
        im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=250)
        ax.set_title(f'{s.split("_attention")[0]}: MAE {err.mean():.1f}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.colorbar(im, ax=axes, label='mean coordinate error (0-1000)', shrink=.8)
    fig.savefig(out / 'spatial_error.png')
    plt.close(fig)


def relative_fig(res: dict, out: Path) -> None:
    rel = res['relative']
    stages = sorted(rel, key=lambda s: (not pr.is_vision(s), stage_x(s)))
    x = np.arange(len(stages))
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    series = [('abs_full_ridge', 'displacement from full-ridge absolute', 2, 'D'), ('r1', 'R1 difference ridge', 0, 'o'),
              ('r2', 'R2 pair MLP', 1, 's'), ('r3', 'R3 Fourier bilinear', 3, '^')]
    for key, label, k, m in series:
        ax.plot(x, [rel[s][key]['mae'] for s in stages], color=SERIES[k], marker=m, lw=1.6, label=label)
    ax.plot(x, [probe_value(res, s, 'full')[0] for s in stages], color=MUTED, ls=':', marker='.', label='absolute coordinate MAE (full ridge)')
    ax.axhline(np.mean([rel[s]['mean_abs_delta'] for s in stages]), color=MUTED, ls='--', lw=1)
    ax.text(0, np.mean([rel[s]['mean_abs_delta'] for s in stages]), ' predict zero displacement', color=MUTED, fontsize=8, va='bottom')
    ax.set_ylabel('displacement MAE (0-1000)')
    ax.legend(fontsize=8, ncol=2)
    g = np.array([rel[s]['gamma'] for s in stages])
    ci = np.array([rel[s]['gamma_ci95'] for s in stages])
    bx.bar(x, g, color=[SERIES[0] if v > 0 else '#b0b0b0' for v in g], width=.6)
    bx.errorbar(x, g, yerr=np.abs(ci.T - g), fmt='none', color=INK, capsize=2, lw=1)
    bx.axhline(0, color=INK, lw=.8)
    bx.set_ylabel(r'$\Gamma$ = R1 - min(R2, R3)')
    bx.set_xticks(x)
    bx.set_xticklabels([s.replace('_linear_attention', '').replace('_full_attention', ' FA').replace('decoder_', 'D')
                        for s in stages], rotation=60, fontsize=8)
    fig.savefig(out / 'relative.png')
    plt.close(fig)


def synthetic_summary(res: dict, out: Path) -> None:
    stages = [s for s in res['ridge']['synthetic']]
    stages = sorted(stages, key=lambda s: (not pr.is_vision(s), stage_x(s)))
    dec = ['merger'] + [s for s in stages if s.startswith('decoder_')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, kind_list, title in ((axes[0], [('uniform', 'uniform canvas'), ('pattern', 'repeated pattern (dev templates)'),
                                            ('heldout', 'repeated pattern (held-out templates)')], 'Transfer: natural-image full ridge'),
                                 (axes[1], None, 'Synthetic-internal probes')):
        if kind_list:
            for k, (kind, label) in enumerate(kind_list):
                ax.plot([stage_x(s) for s in dec], [res['ridge']['synthetic'][s]['full'][kind]['mae'] for s in dec],
                        color=SERIES[k], marker='o', ms=4, label=label)
            ax.plot([stage_x(s) for s in dec], [res['ridge']['test512'][s]['full']['all']['mae'] for s in dec],
                    color=MUTED, ls=':', label='natural test images')
        else:
            syn = res['synthetic']
            fs = [s for s in dec if s in syn]
            items = [('internal_ridge', 'in_distribution', 'ridge, dev templates (test images)'),
                     ('internal_ridge', 'heldout', 'ridge, held-out templates'),
                     ('internal_mlp256', 'heldout', 'MLP256, held-out templates')]
            for k, (probe, sub, label) in enumerate(items):
                ax.plot([stage_x(s) for s in fs], [syn[s][probe][sub]['mae'] for s in fs], color=SERIES[k], marker='o', ms=4, label=label)
            ys = [np.mean([v['mae'] for v in syn[s]['uniform_out_ridge'].values()]) for s in fs]
            ax.plot([stage_x(s) for s in fs], ys, color=SERIES[3], marker='s', ms=4, label='ridge, uniform canvas (leave size group out)')
            ys = [np.mean([v['mae'] for v in syn[s]['grid_out_ridge'].values()]) for s in fs]
            ax.plot([stage_x(s) for s in fs], ys, color=SERIES[4], marker='^', ms=4, label='ridge, pattern (leave grid out)')
        ax.axhline(250, color=MUTED, ls='--', lw=1)
        for layer in FULL_ATTENTION:
            ax.axvline(layer, color=GRID, lw=3, zorder=0)
        ax.set_xticks([0] + list(FULL_ATTENTION))
        ax.set_xticklabels(['merger'] + [str(l) for l in FULL_ATTENTION])
        ax.set_xlabel('decoder layer (shaded: full attention)')
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('MAE (0-1000); dashed: constant-center level')
    fig.savefig(out / 'synthetic_summary.png')
    plt.close(fig)


def synthetic_arrows(run: Path, out: Path) -> None:
    cases = json.loads((run / 'cases.json').read_text())['synthetic']
    cols = json.loads((run / 'predictions' / 'columns.json').read_text())
    preds = np.load(run / 'predictions' / 'synthetic.npz')
    pick = [next(i for i, c in enumerate(cases) if c['kind'] == 'uniform' and c['color'] == 'white' and c['grid'] == 'square_small'),
            next(i for i, c in enumerate(cases) if c['kind'] == 'pattern' and c['template'] == 'ring' and c['grid'] == '10x10'
                 and c['offset'] == 'phase0')]
    stages = ['merger', [s for s in cols if pr.decoder_layer(s) == 14][0], [s for s in cols if pr.decoder_layer(s) == 32][0]]
    offsets = preds['offsets_merged']
    # Third row: the same uniform canvas read by the synthetic-internal ridge (trained on the other uniform size groups).
    internal = {s: np.load(run / 'probes' / 'synthetic' / f'{s}.npz')['uniform_out_ridge'] for s in stages}
    rows = [(i, 'transfer') for i in pick] + [(pick[0], 'internal')]
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 11))
    for r, (i, mode) in enumerate(rows):
        c = cases[i]
        image = Image.open(run / 'images/synthetic' / f'{i:04d}.png')
        grid, lab = pr.token_labels(tuple(c['patch_hw']), False)
        W, H = image.size
        for j, s in enumerate(stages):
            ax = axes[r, j]
            cc = {n: k for k, n in enumerate(cols[s])}
            if mode == 'transfer':
                p = preds[s][offsets[i]:offsets[i + 1]][:, [cc['full_image_x'], cc['full_image_y']]]
            else:
                p = internal[s][offsets[i]:offsets[i + 1]]
            ax.imshow(image, extent=(0, 1000, 1000, 0), aspect='auto', alpha=.6)
            keep = (grid[:, 0] % 2 == 0) & (grid[:, 1] % 2 == 0)
            ax.quiver(lab[keep, 0], lab[keep, 1], p[keep, 0] - lab[keep, 0], p[keep, 1] - lab[keep, 1], angles='xy',
                      scale_units='xy', scale=1, color=SERIES[1], width=.004)
            ax.scatter(lab[keep, 0], lab[keep, 1], s=4, color=SERIES[0])
            ax.set_xlim(0, 1000)
            ax.set_ylim(1000, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f'{c["template"]} {W}x{H} | {s.split("_attention")[0]} | {mode} | MAE {np.abs(p - lab).mean():.0f}', fontsize=8)
    fig.savefig(out / 'synthetic_arrows.png')
    plt.close(fig)


def examples(run: Path, out: Path) -> None:
    cases = json.loads((run / 'cases.json').read_text())
    nat = [c for c in cases['natural'] if c['split'] == 'test'][:3]
    syn = cases['synthetic']
    picks = [next(i for i, c in enumerate(syn) if c['kind'] == 'pattern' and c['grid'] == g and c['offset'] == o)
             for g, o in (('6x6', 'phase0'), ('20x20', 'jitter4'))]
    picks.append(next(i for i, c in enumerate(syn) if c['held_out'] and c['grid'] == '10x10'))
    picks.append(next(i for i, c in enumerate(syn) if c['kind'] == 'uniform' and c['color'] == 'gray'))
    fig, axes = plt.subplots(1, 7, figsize=(17, 2.9))
    items = [(run / 'images/natural512' / f'{c["id"]:04d}.png', f'RefCOCO test {c["input_size"][0]}x{c["input_size"][1]}') for c in nat]
    items += [(run / 'images/synthetic' / f'{i:04d}.png', f'{syn[i]["template"]} {syn[i]["grid"]} {syn[i]["offset"]}') for i in picks]
    for ax, (path, title) in zip(axes, items):
        image = Image.open(path)
        ax.imshow(image)
        w, h = image.size
        for v in range(0, w, 32):
            ax.axvline(v, color='white', lw=.3, alpha=.6)
        for v in range(0, h, 32):
            ax.axhline(v, color='white', lw=.3, alpha=.6)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.savefig(out / 'examples.png')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    res = json.loads((args.output / 'results.json').read_text())
    res['meta']['lc_stage_names'] = list(res['learning_curve'])
    stages = sorted(res['ridge']['test512'], key=lambda s: (not pr.is_vision(s), stage_x(s)))
    layer_curves(res, stages, args.figures, 'mae', 'test MAE (0-1000 image coordinates)', 'layer_mae.png')
    layer_curves(res, stages, args.figures, 'token_mae', 'test MAE in token units', 'layer_token_mae.png')
    fa_breakdown(res, args.figures)
    capacity(res, args.figures)
    learning_curves(res, args.figures)
    spatial_maps(res, args.output, args.figures)
    relative_fig(res, args.figures)
    synthetic_summary(res, args.figures)
    synthetic_arrows(args.output, args.figures)
    examples(args.output, args.figures)


if __name__ == '__main__':
    main()
