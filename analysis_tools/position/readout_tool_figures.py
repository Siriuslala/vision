"""Experiment 3.9 figures (reads results.json, eval_cases.jsonl, tool/*.json, rowcol.jsonl).

Colour convention used in every figure (also defined in the results document):
tool = blue (#2a78d6), model's own answer = orange (#eb6834), ground truth = green (#1baf7a).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from analysis_tools.position.coordinate_readout_run import read_jsonl

TOOL, OWN, TRUTH, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
SITE_COLORS = {'prompt_end': '#6250d6', 'target_end': '#eda100', 'call_end': '#2a78d6'}
SET_LABEL = {'refcoco': 'RefCOCO test', 'synth': 'generic synthetic test', 'disc': 'single disc (3.6)',
             't6': 'T6 grids (1.3)', 'layout': 'layouts (1.3 supp.)', 'ring': 'rings (1.3 supp.)'}
SETS = list(SET_LABEL)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.axisbelow': True})


def ci_err(v, ci):
    v = np.asarray(v, float)
    return [np.clip(v - ci[:, 0], 0, None), np.clip(ci[:, 1] - v, 0, None)]


def load(run: Path):
    res = json.loads((run / 'results.json').read_text())
    rows = read_jsonl(run / 'eval_cases.jsonl')
    cases = {c['id']: c for c in read_jsonl(run / 'cases.jsonl')}
    return res, rows, cases


def image_of(case: dict, run: Path) -> Image.Image:
    if case.get('render') == 'disc':
        from analysis_tools.position import coordinate_readout as cr
        return cr.disc_image(tuple(case['center_px']), case['diameter'], case['canvas_px'])[0]
    base = {'run': run / 'images', 'behavior': run.parent.parent / 'behavior'}[case['image_root']]
    return Image.open(base / case['image']).convert('RGB')


def overlay(ax, row: dict, case: dict, run: Path, title: str) -> None:
    im = image_of(case, run)
    W, H = im.size
    ax.imshow(im)
    sx, sy = W / 1000, H / 1000

    def box(b, color, ls):
        ax.add_patch(Rectangle((b[0] * sx, b[1] * sy), (b[2] - b[0]) * sx, (b[3] - b[1]) * sy, fill=False,
                               edgecolor=color, lw=1.6, ls=ls))
    box(row['truth_box'], TRUTH, '-')
    box(row['tool_box'], TOOL, '--')
    if row.get('own_box'):
        box(row['own_box'], OWN, ':')
    ax.plot(row['truth_center'][0] * sx, row['truth_center'][1] * sy, 'o', color=TRUTH, ms=7, mec='white', mew=1.2)
    ax.plot(row['tool_point'][0] * sx, row['tool_point'][1] * sy, 'X', color=TOOL, ms=9, mec='white', mew=1.2)
    if row.get('own_point'):
        ax.plot(row['own_point'][0] * sx, row['own_point'][1] * sy, 'P', color=OWN, ms=9, mec='white', mew=1.2)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GRAY)


def legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=TRUTH, marker='o', ls='-', label='ground truth (centre, tight box)'),
            Line2D([], [], color=TOOL, marker='X', ls='--', label='tool (point_2d, bbox_2d)'),
            Line2D([], [], color=OWN, marker='P', ls=':', label="model's own answer (point, bbox)")]


def fig_inputs(res, rows, cases, run, out):
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for ax, s in zip(axes.flat, SETS):
        rs = [r for r in rows if r['set'] == s]
        r = rs[int(rng.integers(len(rs)))]
        c = cases[r['id']]
        overlay(ax, r, c, run, f"{SET_LABEL[s]} | {r['subset']}\ntarget: \"{c['target'][:40]}\"\n"
                               f"tool err {r['tool_error']:.0f}, own err {r['own_error'] if r['own_error'] is None else round(r['own_error'])}")
    fig.legend(handles=legend_handles(), loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / 'inputs.png', dpi=130)
    plt.close(fig)


def fig_sites(run, out):
    sweep = json.loads((run / 'tool' / 'sweep.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for site, col in SITE_COLORS.items():
        r = [x for x in sweep if x['site'] == site]
        L = [x['layer'] for x in r]
        axes[0].plot(L, [x['val']['refcoco'] for x in r], '-o', color=col, ms=4, label=site)
        axes[1].plot(L, [x['val']['synth'] for x in r], '-o', color=col, ms=4, label=site)
    axes[0].set_title('validation: RefCOCO')
    axes[1].set_title('validation: generic synthetic')
    fit = json.loads((run / 'tool' / 'fit.json').read_text())
    site = fit['best_single']['site']
    r = [x for x in sweep if x['site'] == site]
    cols = {'refcoco': '#2a78d6', 'synth': '#1baf7a', 'disc': '#eb6834', 't6': '#eda100', 'layout': '#e87ba4', 'ring': '#6250d6'}
    for s, col in cols.items():
        axes[2].plot([x['layer'] for x in r], [x['test'].get(s, np.nan) for x in r], '-o', ms=4, color=col, label=SET_LABEL[s])
    axes[2].plot([x['layer'] for x in fit['disc_ceiling']], [x['sweep_error'] for x in fit['disc_ceiling']], '--', color='#eb6834',
                 label='single disc, in-domain ridge (ceiling)')
    for s_, col in cols.items():
        v = fit['candidates']['pointer']['test'].get(s_)
        if v is not None:
            axes[2].axhline(v, color=col, ls=':', lw=1.4)
    axes[2].plot([], [], ':', color=GRAY, label='pointer tool (dotted, same colour per set)')
    axes[2].set_title(f'test sets: hidden-state ridge at {site} (solid) vs pointer (dotted)')
    for ax in axes:
        ax.set_xlabel('decoder layer')
        ax.set_ylabel('mean centre error (0-1000 units)')
        ax.set_yscale('log')
    axes[0].legend(frameon=False, title='readout site')
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'sites_layers.png', dpi=130)
    plt.close(fig)


def fig_main(res, out):
    by = res['by_set']
    sets = [s for s in SETS if s in by]
    y = np.arange(len(sets))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for k, (who, col, dy) in enumerate((('tool', TOOL, -0.12), ('own', OWN, 0.12))):
        h = [by[s]['hit'][who] for s in sets]
        ci = np.asarray([by[s]['hit'][f'{who}_ci'] for s in sets])
        axes[0].errorbar(h, y + dy, xerr=ci_err(h, ci), fmt='o', color=col, ms=7, capsize=3,
                         label='tool' if who == 'tool' else "model's own")
        e = [by[s]['error'][f'{who}_mean'] for s in sets]
        eci = np.asarray([by[s]['error'][f'{who}_mean_ci'] for s in sets])
        axes[1].errorbar(e, y + dy, xerr=ci_err(e, eci), fmt='o', color=col, ms=7, capsize=3)
        axes[1].plot([by[s]['error'][f'{who}_median'] for s in sets], y + dy, '|', color=col, ms=12, mew=2)
        iou = [by[s]['iou'][f'{who}_mean'] for s in sets]
        axes[2].plot(iou, y + dy, 'o', color=col, ms=7)
    axes[0].set_xlabel('point hit rate (Wilson 95% CI)')
    axes[1].set_xlabel('centre error, 0-1000 units (dot: mean with bootstrap CI; bar: median)')
    axes[1].set_xscale('log')
    axes[2].set_xlabel('mean bbox IoU')
    axes[0].set_yticks(y, [SET_LABEL[s] for s in sets])
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(out / 'main.png', dpi=130)
    plt.close(fig)


def fig_sweep(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, sub, k, lab in ((axes[0], 'sweep_h', 0, 'x'), (axes[1], 'sweep_v', 1, 'y')):
        rs = sorted([r for r in rows if r['set'] == 'disc' and r['subset'] == sub], key=lambda r: r['truth_center'][k])
        t = [r['truth_center'][k] for r in rs]
        ax.plot(t, t, '-', color=TRUTH, lw=1.2, label='ground truth (y = x)')
        ax.plot(t, [r['own_point'][k] if r['own_point'] else np.nan for r in rs], '-', color=OWN, lw=1.6, label="model's own")
        ax.plot(t, [r['tool_raw'][k] for r in rs], '-', color=TOOL, lw=1.6, label='tool (continuous output)')
        ax.set_xlabel(f'true {lab} (0-1000)')
        ax.set_ylabel(f'answered {lab} (0-1000)')
        ax.set_title(f'single-disc {sub.replace("_", " ")} (225 steps of 3.9 units)')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'sweep.png', dpi=130)
    plt.close(fig)


def fig_digits(rows, out):
    syn = [r for r in rows if r['set'] in ('disc', 't6', 'layout', 'ring')]
    own = np.rint([v for r in syn if r['own_valid'] for v in r['own_point']]).astype(int)
    tool = np.asarray([v for r in syn for v in r['tool_point']]).astype(int)
    truth = np.rint([v for r in syn for v in r['truth_center']]).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.arange(0, 101, 2)
    for v, col, lab in ((own, OWN, "model's own"), (tool, TOOL, 'tool'), (truth, TRUTH, 'ground truth')):
        axes[0].hist(v % 100, bins=bins, histtype='step', lw=1.8, color=col, density=True, label=lab)
        cnt = np.bincount((v // 10) % 10, minlength=10) / len(v)
        axes[1].plot(range(10), cnt, '-o', color=col, ms=5, label=lab)
    axes[0].set_xlabel('value mod 100 (last two digits), 2-unit bins')
    axes[0].set_ylabel('density')
    axes[0].set_title('held-out synthetic sets: snapping to multiples of 10 / 50')
    axes[1].set_xlabel('tens digit')
    axes[1].set_ylabel('fraction of coordinates')
    axes[1].set_title('tens-digit distribution')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'digits.png', dpi=130)
    plt.close(fig)


def fig_grid_offsets(rows, out):
    groups = [('t6', 'g20'), ('layout', 'g10x20'), ('layout', 'g20x10'), ('layout', 'f6c128')]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (s, sub) in zip(axes, groups):
        rs = [r for r in rows if r['set'] == s and r['subset'] == sub]
        for who, col, dx in (('own', OWN, -0.15), ('tool', TOOL, 0.15)):
            cols = defaultdict(list)
            for r in rs:
                if r.get(f'{who}_dx') is None:
                    continue
                cols[r['grid']['target_rc'][1] + 1].append(r[f'{who}_dx'] / 1000 * r['canvas'][0] / r['grid']['cell'])
            xs = sorted(cols)
            m = [np.mean(cols[x]) for x in xs]
            ax.plot(np.asarray(xs) + dx, m, 'o-', color=col, ms=4, label="model's own" if who == 'own' else 'tool')
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.axhline(1, color=GRAY, lw=0.8, ls=':')
        ax.set_title(f'{SET_LABEL[s]}: {sub}')
        ax.set_xlabel('true column (1-based)')
        ax.set_ylabel('mean x offset of point (cells)')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'grid_offsets.png', dpi=130)
    plt.close(fig)


def fig_rowcol(res, out):
    rc = res.get('rowcol', {})
    if not rc:
        return
    conds = ['notool_nogrid', 'notool_grid', 'own_grid', 'tool_grid']
    labels = ['no tool', 'no tool\n+ grid size', 'own coords\n+ grid size', 'tool coords\n+ grid size']
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    x = np.arange(len(conds))
    for mode, col, dx in (('direct', '#2a78d6', -0.1), ('think', '#e87ba4', 0.1)):
        a = [rc.get(f'{c}|{mode}', {}).get('acc', np.nan) for c in conds]
        ci = np.asarray([rc.get(f'{c}|{mode}', {}).get('acc_ci', [np.nan, np.nan]) for c in conds])
        axes[0].errorbar(x + dx, a, yerr=ci_err(a, ci), fmt='o', ms=7, capsize=3, color=col,
                         label='direct answer' if mode == 'direct' else 'think then answer')
    for c, mk in (('own_grid', 's'), ('tool_grid', 'D')):
        v = rc.get(f'{c}|direct', {}).get('response_cell_correct')
        if v is not None:
            axes[0].plot(conds.index(c), v, mk, mfc='none', mec='#0b0b0b', ms=10,
                         label=f"{'own' if c == 'own_grid' else 'tool'} coordinates converted with true grid geometry")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel('row/column accuracy (Wilson 95% CI)')
    axes[0].set_title('all row/column images (T6 grids + layouts)')
    axes[0].legend(frameon=False, fontsize=8)
    groups = sorted(rc.get('tool_grid|direct', {}).get('by_group', {}))
    gx = np.arange(len(groups))
    for c, col in zip(conds, ('#8a8984', '#6250d6', OWN, TOOL)):
        bg = rc.get(f'{c}|direct', {}).get('by_group', {})
        axes[1].plot(gx, [bg.get(g, {}).get('acc', np.nan) for g in groups], 'o-', color=col, ms=5, label=c)
    axes[1].set_xticks(gx, groups, rotation=45)
    axes[1].set_ylabel('accuracy (direct answer)')
    axes[1].set_title('by layout, direct answer')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'rowcol.png', dpi=130)
    plt.close(fig)


def fig_area(res, rows, out):
    rs = [r for r in rows if r['set'] == 'refcoco']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    b = res['refcoco_by_area']
    xs = np.arange(len(b))
    for who, col, dx in (('tool', TOOL, -0.1), ('own', OWN, 0.1)):
        axes[0].plot(xs + dx, [x['error'][f'{who}_median'] for x in b], 'o-', color=col, label="model's own" if who == 'own' else 'tool')
        axes[1].plot(xs + dx, [x['hit'][who] for x in b], 'o-', color=col)
    for ax in axes:
        ax.set_xticks(xs, [f"{x['area'][0]:.2f}-{min(x['area'][1], 1):.2f}\nn={x['n']}" for x in b])
        ax.set_xlabel('target box area / image area')
    axes[0].set_ylabel('median centre error (0-1000)')
    axes[1].set_ylabel('point hit rate')
    axes[0].legend(frameon=False)
    fig.suptitle('RefCOCO test: dependence on target size')
    fig.tight_layout()
    fig.savefig(out / 'refcoco_area.png', dpi=130)
    plt.close(fig)


def fig_failures(res, rows, cases, run, out):
    byid = {r['id']: r for r in rows}
    fig, axes = plt.subplots(2, 6, figsize=(21, 9.5))
    for j, s in enumerate(SETS):
        for i, cid in enumerate(res['worst'][s][:2]):
            r = byid[cid]
            ax = axes[i, j]
            own = 'invalid' if r['own_error'] is None else f"{r['own_error']:.0f}"
            overlay(ax, r, cases[cid], run, f"{SET_LABEL[s]} | {r['subset']}\n\"{cases[cid]['target'][:34]}\"\n"
                                            f"tool err {r['tool_error']:.0f}, own err {own}")
    fig.legend(handles=legend_handles(), loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / 'failures.png', dpi=110)
    plt.close(fig)


def fig_subsets(res, out):
    sub = res['by_subset']
    keys = [k for k in sub if k.split('/')[0] in ('disc', 't6', 'layout', 'ring')]
    y = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * len(keys) + 1.5), sharey=True)
    for who, col, dy in (('tool', TOOL, -0.15), ('own', OWN, 0.15)):
        h = [sub[k]['hit'][who] for k in keys]
        ci = np.asarray([sub[k]['hit'][f'{who}_ci'] for k in keys])
        axes[0].errorbar(h, y + dy, xerr=ci_err(h, ci), fmt='o', ms=5, capsize=2, color=col,
                         label="model's own (strict)" if who == 'own' else 'tool')
        axes[1].plot([sub[k]['error'][f'{who}_median'] for k in keys], y + dy, 'o', ms=5, color=col)
    axes[0].plot([sub[k]['own_hit_any'] for k in keys], y + 0.15, '|', color=OWN, ms=10, mew=2, label="model's own, either convention")
    axes[0].set_yticks(y, keys)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('point hit rate (target cell / disc)')
    axes[1].set_xlabel('median centre error (0-1000)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'subsets.png', dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    a = p.parse_args()
    a.figures.mkdir(parents=True, exist_ok=True)
    res, rows, cases = load(a.run)
    fig_inputs(res, rows, cases, a.run, a.figures)
    fig_sites(a.run, a.figures)
    fig_main(res, a.figures)
    fig_sweep(rows, a.figures)
    fig_digits(rows, a.figures)
    fig_grid_offsets(rows, a.figures)
    fig_rowcol(res, a.figures)
    fig_area(res, rows, a.figures)
    fig_failures(res, rows, cases, a.run, a.figures)
    fig_subsets(res, a.figures)


if __name__ == '__main__':
    main()
