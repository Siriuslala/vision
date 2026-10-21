"""Figures of experiment 3.9 round 2 (docs/analysis/position/figures/readout_tool_r2/).

Colours: tool = blue (#2b6cb0), model = orange (#dd6b20), ground truth = green (#2f855a); continuous
quantities use viridis with a labelled colorbar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position import readout_tool_r2 as r2
from analysis_tools.position.coordinate_readout_run import read_jsonl

TOOL, MODEL, TRUTH = '#2b6cb0', '#dd6b20', '#2f855a'
ORDER = ('disc', 't6', 'layout', 'ring', 'g_flat', 'synth', 'g_texture', 'g_clutter', 'g_chart_ui', 'g_photo_bg', 'refcoco')
LABEL = {'disc': 'single disc', 't6': 'T6 grid', 'layout': 'row/col layout', 'ring': 'ring', 'g_flat': 'flat',
         'synth': 'flat (r1 test)', 'g_texture': 'texture', 'g_clutter': 'clutter', 'g_chart_ui': 'chart / UI',
         'g_photo_bg': 'photo background', 'refcoco': 'RefCOCO'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def image_of(case: dict, run: Path, r1: Path) -> Image.Image:
    if case.get('render') == 'disc':
        from analysis_tools.position import coordinate_readout as cr
        return cr.disc_image(tuple(case['center_px']), case['diameter'], case['canvas_px'])[0]
    base = {'r1': r1 / 'images', 'r2': run / 'images', 'behavior': run.parent.parent / 'behavior'}[case['image_root']]
    return Image.open(base / case['image']).convert('RGB')


def draw_box(ax, box, color, ls, label=None):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=2, ls=ls, label=label))


def overlay(ax, row, case, run, r1, title):
    im = image_of(case, run, r1)
    ax.imshow(im)
    W, H = im.size
    t = case['truth']
    draw_box(ax, t['box'], TRUTH, '-')
    ax.plot(*t['center'], 'o', color=TRUTH, ms=6)
    draw_box(ax, rt_.to_pixels(row['tool_box'], (W, H)), TOOL, '--')
    ax.plot(*rt_.to_pixels(row['tool_point'], (W, H)), 'x', color=TOOL, ms=9, mew=2.5)
    if row.get('own_box'):
        draw_box(ax, rt_.to_pixels(row['own_box'], (W, H)), MODEL, ':')
    if row.get('own_point'):
        ax.plot(*rt_.to_pixels(row['own_point'], (W, H)), '+', color=MODEL, ms=10, mew=2.5)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def legend_handles():
    return [Line2D([], [], color=TRUTH, lw=2, marker='o', label='ground truth box / centre'),
            Line2D([], [], color=TOOL, lw=2, ls='--', marker='x', label='tool box / point'),
            Line2D([], [], color=MODEL, lw=2, ls=':', marker='+', label='model box / point')]


def fig_examples(rows, cases, run, r1, out):
    sets = ('g_flat', 'g_texture', 'g_clutter', 'g_chart_ui', 'g_photo_bg', 'refcoco', 'layout', 'disc')
    fig, axes = plt.subplots(2, len(sets), figsize=(2.6 * len(sets), 5.6))
    rng = np.random.default_rng(3)
    for j, s in enumerate(sets):
        g = [r for r in rows if r['set'] == s and not r['id'].endswith('_hp')]
        for i, r in enumerate([g[k] for k in rng.choice(len(g), 2, replace=False)]):
            c = cases[r['id']]
            overlay(axes[i, j], r, c, run, r1, f"{LABEL[s]}\n{c['target'][:28]}")
    fig.legend(handles=legend_handles(), loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / 'examples.png', dpi=130)
    plt.close(fig)


READER_ROWS = (
    ('cnn_target_end', 'attention @ target_end (round-1 site)'),
    ('cnn_slot', 'attention @ slot'),
    ('cnn_slot_content', 'attention @ slot + content'),
    ('cnn_slot_content_approx', 'attention @ slot + content + rough point (final)'),
    ('cnn_slot@no_graded', 'attention @ slot, no graded training'),
    ('cnn_slot_content_approx@no_graded', 'final inputs, no graded training'),
    ('query_content16', 'no attention: learned query x content'),
    ('cnn_content_approx', 'no attention: content + rough point'),
    ('resid_q1_L27', 'no attention: q1 hidden-state residual (L27)'),
    ('resid_slot_L20', 'no attention: slot hidden-state residual (L20)'),
    ('ridge_slot_L20', 'no attention: slot hidden-state ridge (L20)'),
    ('approx_point', "model's rough point in the call"),
)


def fig_readers(res, out):
    readers = [k for k, _ in READER_ROWS if k in res['by_reader']]
    labels = dict(READER_ROWS)
    sets = [s for s in ORDER if s in res['by_reader'][readers[0]]]
    M = np.array([[res['by_reader'][r].get(s, {}).get('tool_err_median', np.nan) for s in sets] for r in readers])
    ref = res['by_reader'][res['final']]
    own = np.array([ref[s]['own_err_median'] for s in sets])
    obc = np.array([ref[s]['own_box_center_err_median'] for s in sets])
    M = np.vstack([M, own[None], obc[None]])
    names = [labels[r] for r in readers] + ['model answer: point', 'model answer: bbox centre']
    fig, ax = plt.subplots(figsize=(1.0 * len(sets) + 3, 0.42 * len(names) + 1.6))
    im = ax.imshow(np.log10(M), cmap='viridis_r', aspect='auto', vmin=np.log10(3), vmax=np.log10(300))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=8,
                    color='white' if np.log10(M[i, j]) > 1.6 else 'black')
    ax.set_xticks(range(len(sets)), [LABEL[s] for s in sets], rotation=35, ha='right')
    ax.set_yticks(range(len(names)), names)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    ticks = [3, 10, 30, 100, 300]
    cb.set_ticks(np.log10(ticks), labels=[str(t) for t in ticks])
    cb.set_label('median centre error (0-1000 units, log scale)')
    ax.axhline(len(names) - 2.5, color='white', lw=3)
    ax.axhline(5.5, color='white', lw=3)
    fig.tight_layout()
    fig.savefig(out / 'readers.png', dpi=140)
    plt.close(fig)


def fig_applicability(res, out):
    final = res['final']
    d = res['by_reader'][final]
    sets = [s for s in ORDER if s in d]
    x = np.arange(len(sets))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for k, (who, col, off) in enumerate((('tool', TOOL, -0.15), ('own', MODEL, 0.15))):
        v = np.array([d[s][f'{who}_hit'] for s in sets])
        ci = np.array([d[s][f'{who}_hit_ci'] for s in sets])
        axes[0].errorbar(x + off, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt='o', color=col, capsize=3,
                         label='tool' if who == 'tool' else 'model')
        axes[1].plot(x + off, [d[s][f'{who}_err_median'] for s in sets], 'o', color=col, ms=7)
        axes[2].plot(x + off, [d[s][f'{who}_iou'] for s in sets], 'o', color=col, ms=7)
    axes[0].set_ylabel('point hit rate (Wilson 95% CI)')
    axes[1].set_ylabel('median centre error (0-1000 units)')
    axes[1].set_yscale('log')
    axes[2].set_ylabel('mean bbox IoU')
    for ax in axes:
        ax.set_xticks(x, [LABEL[s] for s in sets], rotation=40, ha='right')
        ax.axvspan(-0.5, 3.5, color='0.92', zorder=0)
    axes[0].legend(frameon=False, loc='lower left')
    axes[0].text(1.5, 1.02, 'sparse / repeated', ha='center', fontsize=8, transform=axes[0].get_xaxis_transform())
    axes[0].text(7.5, 1.02, 'realism axis', ha='center', fontsize=8, transform=axes[0].get_xaxis_transform())
    fig.tight_layout()
    fig.savefig(out / 'applicability.png', dpi=140)
    plt.close(fig)


def fig_dissoc(res, out):
    d = res.get('dissoc')
    if not d:
        return
    conds = ('steer_x', 'steer_y', 'gate')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (axis, key) in zip(axes, (('column change', '_dcol'), ('row change', '_drow'))):
        for k, (task, col) in enumerate((('point', TOOL), ('rowcol', MODEL))):
            v = [d[task].get(c + key, {}).get('mean', np.nan) for c in conds]
            ax.bar(np.arange(3) + (k - 0.5) * 0.36, v, 0.34, color=col,
                   label='point answer (cell of the point)' if task == 'point' else 'row/column answer')
        ax.axhline(0, color='0.3', lw=1)
        ax.set_xticks(range(3), ['steer to right/left\nneighbour', 'steer to lower/upper\nneighbour', 'DeltaNet alpha=1\n(layers 9-16)'])
        ax.set_title(f'mean {axis} vs unperturbed answer (cells)')
    axes[0].set_ylabel('mean signed change (cells)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'dissoc.png', dpi=140)
    plt.close(fig)


def fig_pointer(run, r1, out, cases, rows):
    """Pointer distributions alpha of the final reader on a few test images (saved by the fit stage)."""
    p = run / 'tool_r2' / 'alpha_examples.json'
    if not p.exists():
        return
    ex = json.loads(p.read_text())
    fig, axes = plt.subplots(1, len(ex), figsize=(3.2 * len(ex), 3.4))
    rowmap = {r['id']: r for r in rows}
    for ax, e in zip(np.atleast_1d(axes), ex):
        c = cases[e['id']]
        im = image_of(c, run, r1)
        ax.imshow(im)
        a = np.asarray(e['alpha'])
        h = ax.imshow(np.ma.masked_less(a, 0.02 * a.max()), cmap='viridis', alpha=0.75,
                      extent=(0, im.width, im.height, 0), vmin=0, vmax=a.max())
        draw_box(ax, c['truth']['box'], TRUTH, '-')
        if e['id'] in rowmap:
            ax.plot(*rt_.to_pixels(rowmap[e['id']]['tool_point'], im.size), 'x', color='white', ms=9, mew=2.5)
        ax.set_title(f"{LABEL.get(c['set'], c['set'])}: {c['target'][:24]}", fontsize=8)
        ax.axis('off')
        cb = fig.colorbar(h, ax=ax, fraction=0.04)
        cb.set_label('pointer weight alpha (per token)', fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'pointer.png', dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--r1', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    a = p.parse_args()
    a.figures.mkdir(parents=True, exist_ok=True)
    res = json.loads((a.run / 'tool_r2' / 'results_r2.json').read_text())
    rows = json.loads((a.run / 'tool_r2' / 'final_rows.json').read_text())
    cases = {c['id']: c for c in read_jsonl(a.run / 'cases.jsonl')}
    fig_examples(rows, cases, a.run, a.r1, a.figures)
    fig_readers(res, a.figures)
    fig_applicability(res, a.figures)
    fig_dissoc(res, a.figures)
    fig_pointer(a.run, a.r1, a.figures, cases, rows)


if __name__ == '__main__':
    main()
