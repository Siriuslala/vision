"""Figures of experiment 2.5e ("sees right but answers wrong").

Reads the run directory (cases.json, gen.jsonl, readout maps, metrics.npz, probe.npz, summary.json,
causal_rows.json) and writes PNG figures for results_2026-10-16_sees_right.md.

Colour conventions (defined again in the document captions):
  heatmaps   viridis, logarithmic, share of the query's image attention per visual token, fixed range
             [1e-4, 1] in every figure, with a colorbar;
  red        target box on the input and target tokens on heatmaps;
  blue x     model output on the input (white x on heatmaps);
  bars       blue = correct answers, orange = wrong answers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

from analysis_tools.representation import sees_right as sr
from analysis_tools.representation.sees_right_analysis import FULL_LAYERS, RESID_LAYERS, axis_of

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
NORM = LogNorm(1e-4, 1.0)
CB_LABEL = "share of the query's image attention per visual token (log)"
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 130,
                     'savefig.bbox': 'tight', 'legend.frameon': False, 'lines.linewidth': 2})
COND_LABEL = {'disc_point': 'disc point', 't6g06_rowcol': 'T6 6x6 row/col', 't6g06_point': 'T6 6x6 point',
              't6g10_rowcol': 'T6 10x10 row/col', 't6g10_point': 'T6 10x10 point', 't6g20_rowcol': 'T6 20x20 row/col',
              't6g20_point': 'T6 20x20 point', 't6g20_rowcol_think': 'T6 20x20 row/col, think',
              't5g03_rowcol': 'T5 3x3 row/col', 't5g03_point': 'T5 3x3 point', 't5g06_rowcol': 'T5 6x6 row/col',
              't5g06_point': 'T5 6x6 point', 't5g10_rowcol': 'T5 10x10 row/col', 't5g10_point': 'T5 10x10 point',
              'rc_point': 'RefCOCO point'}
ORDER = list(COND_LABEL)


def save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print('figure', path.name)


def frame(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(True)


def heat(ax, share: np.ndarray, size, target: list[int], W: int, out_px=None):
    im = ax.imshow(np.clip(share, 1e-4, 1), cmap='viridis', norm=NORM, extent=(0, size[0], size[1], 0),
                   interpolation='nearest')
    for tok in target:
        r, c = divmod(tok, W)
        ax.add_patch(Rectangle((c * 32, r * 32), 32, 32, fill=False, ec=RED, lw=1.0))
    if out_px is not None:
        ax.plot(*out_px, 'x', color='white', ms=8, mew=2)
    frame(ax)
    return im


# ----------------------------------------------------------------------------- galleries

def gallery(run: Path, out: Path, cond: str, cases: dict, gen: dict, recs: dict, rng, n_each: int = 3) -> None:
    """Random correct and wrong samples of one condition: input, L20 H11 and layer-20 head mean at both
    coordinate-value queries (positions predicting the first digit)."""
    ids = [i for i, r in recs.items() if r['status'] == 'ok' and cases[i]['cond'] == cond]
    right = [i for i in ids if gen[i]['score']['correct']]
    wrong = [i for i in ids if not gen[i]['score']['correct']]
    pick = [(i, True) for i in rng.permutation(right)[:n_each]] + [(i, False) for i in rng.permutation(wrong)[:n_each]]
    if not pick:
        return
    names = ['row', 'column'] if cases[pick[0][0]]['format'] == 'rowcol' else ['x', 'y']
    fig, axes = plt.subplots(len(pick), 5, figsize=(16, 3.2 * len(pick)), squeeze=False)
    im = None
    for row, (cid, ok) in enumerate(pick):
        c, r, s = cases[cid], recs[cid], gen[cid]['score']
        image = sr.load_image(c)
        size = np.asarray(image.size, float)
        H, W = r['grid']
        maps = np.load(run / 'readout' / 'maps' / f'{cid}.npz')['attn'].astype(np.float32)
        o = np.asarray(s['out_units']) * size / 1000
        ax = axes[row, 0]
        ax.imshow(image)
        b = c['truth']['box']
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=RED, lw=1.5))
        ax.plot(*o, 'x', color=BLUE, ms=10, mew=2.5)
        ans = (f"answer r{s['pred_rc'][0] + 1} c{s['pred_rc'][1] + 1}, truth r{c['truth']['target_rc'][0] + 1} "
               f"c{c['truth']['target_rc'][1] + 1}") if c['format'] == 'rowcol' else f"answer {[int(v) for v in s['out_units']]}"
        ax.set_title(f"{'correct' if ok else 'wrong'}: {ans}", fontsize=8)
        frame(ax)
        L20 = FULL_LAYERS.index(20)
        for j, (h, k) in enumerate([(11, 0), (11, 1), (None, 0), (None, 1)], start=1):
            a = maps[L20, :, k].mean(0) if h is None else maps[L20, h, k]
            share = (a / a.sum()).reshape(H, W)
            im = heat(axes[row, j], share, size, r['target_tokens'], W, o)
            tm = share.ravel()[r['target_tokens']].sum()
            axes[row, j].set_title(f"{'L20 H11' if h is not None else 'L20 head mean'}, {names[k]} digit\n"
                                   f"target share {tm:.2f}", fontsize=8)
        if c.get('expression'):
            axes[row, 0].set_xlabel(c['expression'][:40], fontsize=7)
    cb = fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), fraction=0.02, pad=0.01)
    cb.set_label(CB_LABEL)
    fig.suptitle(f'{COND_LABEL[cond]}: random samples; query = position predicting the first digit. '
                 'Red box: target; blue x (white on heatmaps): model output', fontsize=10, y=0.995)
    save(fig, out / f'gallery_{cond}.png')


def layer_head_grid(run: Path, out: Path, cases: dict, gen: dict, recs: dict, cid: str, k: int) -> None:
    """All 8 full layers x 16 heads (+ head mean) of one sample at one coordinate-value query."""
    c, r, s = cases[cid], recs[cid], gen[cid]['score']
    image = sr.load_image(c)
    size = np.asarray(image.size, float)
    H, W = r['grid']
    maps = np.load(run / 'readout' / 'maps' / f'{cid}.npz')['attn'].astype(np.float32)
    o = np.asarray(s['out_units']) * size / 1000
    fig, axes = plt.subplots(8, 17, figsize=(26, 13.5))
    im = None
    for li, l in enumerate(FULL_LAYERS):
        for h in range(17):
            a = maps[li, :, k].mean(0) if h == 16 else maps[li, h, k]
            im = heat(axes[li, h], (a / a.sum()).reshape(H, W), size, r['target_tokens'], W, o)
            if li == 0:
                axes[li, h].set_title('mean' if h == 16 else f'H{h}', fontsize=9)
        axes[li, 0].set_ylabel(f'L{l}', fontsize=10)
    names = ['row', 'column'] if c['format'] == 'rowcol' else ['x', 'y']
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.01, pad=0.005)
    cb.set_label(CB_LABEL)
    fig.suptitle(f"{COND_LABEL[c['cond']]}, {cid}, {'correct' if s['correct'] else 'wrong'}; query: {names[k]} digit. "
                 'Red: target tokens; white x: output', fontsize=11)
    save(fig, out / f'heads_{cid}_v{k}.png')


# ----------------------------------------------------------------------------- summaries

def fig_peak_classes(S: dict, out: Path) -> None:
    """Peak class fractions (target / adjacent / elsewhere) per condition, correct vs wrong, for the
    x / column query and the y / row query separately; diamond = peak inside the target's token
    column (x query) or token row (y query)."""
    conds = [c for c in ORDER if c in S['attention']]
    panels = [('L20H11', 'x', 'L20 H11, x / column query'), ('L20H11', 'y', 'L20 H11, y / row query'),
              ('best', 'x', 'best head (cross-fitted), x / column'), ('best', 'y', 'best head (cross-fitted), y / row')]
    fig, axes = plt.subplots(1, 4, figsize=(21, 7.5), sharey=True)
    cls_cols = [(BLUE, 'peak on target tokens'), (AQUA, 'adjacent'), (MUTED, 'elsewhere')]
    for ax, (rd, axis, title) in zip(axes, panels):
        y = 0
        ticks, labels = [], []
        for cond in conds:
            res = S['attention'][cond]['readers'][rd]
            rowcol = 'rowcol' in cond
            q = ('pred1' if rowcol else 'pred0') if axis == 'x' else ('pred0' if rowcol else 'pred1')
            for corr in ('correct', 'wrong'):
                if corr not in res:
                    continue
                r = res[corr][q]
                left = 0
                for (col, lab), key in zip(cls_cols, ('peak_target', 'peak_adjacent', 'peak_elsewhere')):
                    v = r[key]['p']
                    ax.barh(y, v, left=left, color=col, height=0.8, label=lab if y == 0 else None, edgecolor='white')
                    left += v
                ax.plot(r['peak_axis_on_target']['p'], y, 'D', color=YELLOW, mec=INK, ms=6,
                        label='peak in target token column / row' if y == 0 else None)
                ticks.append(y); labels.append(f"{COND_LABEL[cond]} - {corr} (n={r['peak_target']['n']})")
                y += 1
            y += 0.6
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(0, 1); ax.set_xlabel('fraction of answers')
        ax.set_title(title, fontsize=10)
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower center', bbox_to_anchor=(2.2, -0.14), ncol=4, fontsize=9)
    save(fig, out / 'peak_classes.png')


def _xy_query(cond: str, axis: str) -> int:
    rowcol = 'rowcol' in cond
    return (1 if rowcol else 0) if axis == 'x' else (0 if rowcol else 1)


def fig_target_mass(S: dict, out: Path) -> None:
    conds = [c for c in ORDER if c in S['attention']]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
    for ax, axis, title in ((axes[0], 'x', 'L20 H11, x / column query'), (axes[1], 'y', 'L20 H11, y / row query')):
        for j, (corr, col) in enumerate((('correct', BLUE), ('wrong', ORANGE))):
            xs, ms, lo, hi = [], [], [], []
            for i, cond in enumerate(conds):
                res = S['attention'][cond]['readers']['L20H11'].get(corr)
                if not res:
                    continue
                v = res[f'pred{_xy_query(cond, axis)}']['tmass']
                xs.append(i + (j - 0.5) * 0.38); ms.append(v['mean']); lo.append(v['ci'][0]); hi.append(v['ci'][1])
            ax.bar(xs, ms, 0.36, color=col, label=f'{corr} answers')
            ax.errorbar(xs, ms, [np.subtract(ms, lo), np.subtract(hi, ms)], fmt='none', ecolor=INK, lw=1, capsize=2)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=40, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel('target share of image attention (95% CI)')
    axes[0].legend(fontsize=9)
    save(fig, out / 'target_mass.png')


def fig_layer_profile(S: dict, out: Path) -> None:
    """Best head per layer: solid = x / column query, peak on target tokens; dashed = y / row query, peak
    in the target's token row. Blue correct, orange wrong."""
    conds = [c for c in ORDER if c in S['attention']]
    n = len(conds)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, cond in zip(axes.ravel(), conds):
        prof = S['attention'][cond]['layer_profile']
        qx, qy = _xy_query(cond, 'x'), _xy_query(cond, 'y')
        for corr, col in (('correct', BLUE), ('wrong', ORANGE)):
            if corr not in prof:
                continue
            pt = np.asarray(prof[corr]['peak_target'])[:, :16, qx].max(1)
            pa = np.asarray(prof[corr]['peak_axis'])[:, :16, qy].max(1)
            ax.plot(FULL_LAYERS, pt, '-o', color=col, ms=4, label=f'{corr}: x/col query, peak on target')
            ax.plot(FULL_LAYERS, pa, '--s', color=col, ms=3, lw=1.4, label=f'{corr}: y/row query, peak in target row')
        ax.set_title(COND_LABEL[cond], fontsize=9)
        ax.set_ylim(0, 1.02)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('full-attention layer')
    for ax in axes[:, 0]:
        ax.set_ylabel('fraction (best head)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    save(fig, out / 'layer_profile.png')


def fig_probe(S: dict, F: dict, P: dict, out: Path) -> None:
    """Left: held-out probe error vs layer (wrong answers whose axis value differs): distance of the decoding
    to the truth and to the output. Right: decoded vs truth / output at layer 20 for 20x20 row/col columns."""
    conds = [c for c in ORDER if c in S['probe']]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    frac = []
    for cond in conds:
        vals = []
        for li, l in enumerate(RESID_LAYERS):
            ks = [S['probe'][cond][f'L{l}'][f'v{k}'].get('wrong', {}).get('closer_truth') for k in range(2)]
            ks = [x for x in ks if x]
            vals.append(np.mean([x['p'] for x in ks]) if ks else np.nan)
        frac.append(vals)
    frac = np.asarray(frac)
    im = ax.imshow(frac, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([COND_LABEL[c] for c in conds], fontsize=8)
    ax.set_xticks(range(len(RESID_LAYERS))); ax.set_xticklabels([f'L{l}' for l in RESID_LAYERS])
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            if np.isfinite(frac[i, j]):
                ax.text(j, i, f'{frac[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if frac[i, j] < 0.6 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('fraction of wrong answers whose decoding\nis closer to the truth than to the output')
    ax.set_title('probe at the digit state (wrong answers, wrong axis)', fontsize=10)
    ax = axes[1]
    sel = np.where((F['cond'] == 't6g20_rowcol') & ~F['correct'])[0]
    li = RESID_LAYERS.index(20)
    if len(sel):
        dec = P['pos'][sel, li, 1]
        t = F['true_rc'][sel, 1] + 1
        o = F['pred_rc'][sel, 1] + 1
        ax.scatter(t, dec, s=18, color=BLUE, label='decoded vs true column')
        ax.scatter(o, dec, s=18, color=ORANGE, marker='x', label='decoded vs output column')
        ax.plot([1, 21], [1, 21], '--', color=MUTED, lw=1)
        ax.set_xlabel('column (1-based)'); ax.set_ylabel('probe-decoded column, layer 20')
        ax.set_title(f'T6 20x20 row/col, wrong answers (n={len(sel)})', fontsize=10)
        ax.legend(fontsize=8)
    save(fig, out / 'probe.png')


def fig_bias(S: dict, out: Path) -> None:
    conds = [c for c in ('t6g20_rowcol', 't6g10_point', 't6g20_point', 't5g06_rowcol', 't5g10_rowcol') if c in S['bias']]
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 3.6), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        for k in (0, 1):
            fmt = 'rowcol' if 'rowcol' in cond else 'point'
            is_x = axis_of(fmt, k) == 0
            col, name = (GREEN, 'column / x') if is_x else (VIOLET, 'row / y')
            curve = S['bias'][cond][f'v{k}']['bias_curve_truth']
            xs = sorted(int(x) for x in curve)
            ax.plot([x + 1 for x in xs], [curve[str(x)] if str(x) in curve else curve[x] for x in xs], '-o', ms=3,
                    color=col, label=name)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(COND_LABEL[cond], fontsize=9)
        ax.set_xlabel('true index (1-based)')
    axes[0, 0].set_ylabel('mean output - truth (cells)')
    for ax in axes[0]:
        ax.legend(fontsize=8)
    save(fig, out / 'bias_curves.png')


def fig_causal(S: dict, out: Path) -> None:
    if 'causal' not in S:
        return
    conds = [c for c in ORDER if c in S['causal']]
    names = [('patch_L20H11', 'patch L20 H11'), ('patch_L20', 'patch layer 20'), ('patch_full_all', 'patch 128 full heads'),
             ('patch_mixers_all', 'patch all mixers'), ('v_L20', 'V copy L20'), ('v_L16_20_24', 'V copy L16/20/24'),
             ('v_full', 'V copy 8 layers'), ('k_swap_full', 'K swap 8 layers'), ('kv_swap_full', 'K+V swap (no-op)'),
             ('B_img_v', 'B image V'), ('B_img_kv', 'B image K+V'), ('B_text_kv', 'B text K+V'), ('B_lin', 'B DeltaNet state'),
             ('B_text_kv_lin', 'B text K+V + state'), ('B_all', 'B whole cache')]
    M = np.full((len(conds), len(names)), np.nan)
    for i, cond in enumerate(conds):
        for j, (n, _) in enumerate(names):
            e = S['causal'][cond].get(n, {}).get('all')
            if e:
                M[i, j] = e['w']['mean']
    fig, ax = plt.subplots(figsize=(15, 0.6 * len(conds) + 2.5))
    im = ax.imshow(M, cmap='viridis', vmin=-0.2, vmax=1.2, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if M[i, j] < 0.7 else INK)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([b for _, b in names], rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([COND_LABEL[c] for c in conds], fontsize=9)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('follow ratio w (output shift / target shift)')
    save(fig, out / 'causal.png')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = sr.load_cases(args.run)
    gen = {g['id']: g for g in sr.read_jsonl(args.run / 'gen.jsonl')}
    recs = {r['id']: r for r in sr.read_jsonl(args.run / 'readout' / 'records.jsonl')}
    S = json.loads((args.run / 'summary.json').read_text())
    F = dict(np.load(args.run / 'metrics.npz', allow_pickle=False))
    P = dict(np.load(args.run / 'probe.npz', allow_pickle=False))
    rng = np.random.default_rng(args.seed)
    for cond in ORDER:
        if any(cases[i]['cond'] == cond for i in recs):
            gallery(args.run, args.out, cond, cases, gen, recs, rng)
    # Layer x head grids: one wrong sample each from disc, 20x20 row/col (column +1) and RefCOCO.
    for cond, k in (('disc_point', 1), ('t6g20_rowcol', 1), ('t6g20_rowcol', 0), ('rc_point', 1)):
        ids = [i for i, r in recs.items() if r['status'] == 'ok' and cases[i]['cond'] == cond and not gen[i]['score']['correct']]
        if cond == 't6g20_rowcol':
            ids = [i for i in ids if gen[i]['score'].get('d_col') == 1 and gen[i]['score'].get('d_row') == 0] or ids
        if ids:
            layer_head_grid(args.run, args.out, cases, gen, recs, sorted(ids)[0], k)
    fig_peak_classes(S, args.out)
    fig_target_mass(S, args.out)
    fig_layer_profile(S, args.out)
    fig_probe(S, F, P, args.out)
    fig_bias(S, args.out)
    fig_causal(S, args.out)


if __name__ == '__main__':
    main()
