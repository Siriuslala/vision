"""Figures of experiment 3.9 round 3 (docs/analysis/position/figures/readout_tool_r3/).

Colours (fixed per entity in every figure): model = orange (#eb6834), full 128-head tool = blue
(#2a78d6), one-head tool (L20 H11 + content + rough point) = aqua (#1baf7a), L20 H11 attention peak
(experiment-3.11 decoder) = yellow (#eda100), ground truth = dark grey (#333333).  Continuous
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

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position import readout_tool_r3 as r3
from analysis_tools.position.coordinate_readout_run import read_jsonl
from analysis_tools.position.readout_tool_r2_figures import LABEL, image_of

MODEL, FULL, ONE, PEAK, TRUTH = '#eb6834', '#2a78d6', '#1baf7a', '#eda100', '#333333'
SETS = ('refcoco', 'g_photo_bg', 'g_clutter', 'g_texture', 'g_flat', 'g_chart_ui', 'disc', 'ring', 't6', 'layout')
HELDOUT = ('disc', 'ring', 't6', 'layout')          # image families absent from the tool's training data
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

LADDER = [
    ('zs_H11_peak_target_end', 'H11 peak @ target_end (zero-training)'),
    ('zs_H11_peak', 'H11 peak @ slot (3.11 decoder)'),
    ('zs_L20H11_attn', 'H11 attention-extent box centre'),
    ('zs_L20H11_content', 'H11 peak + content-grown box centre'),
    ('k1_H11', 'learned: H11 map only'),
    ('k1_H11_content', 'learned: H11 + content'),
    ('k1_H11_content_approx', 'learned: H11 + content + rough point'),
    ('k4_content_approx', 'learned: 4 heads + content + rough point'),
    ('L20_content_approx', 'learned: layer 20 (16 heads) + content + rough'),
    ('all_content_approx', 'learned: 128 heads + content + rough (round 2)'),
    ('k0_content_approx', 'learned: content + rough point, no attention'),
    ('approx_point', "model's rough point in the call"),
]


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path, flush=True)


def fig_ladder(res: dict, figs: Path) -> None:
    rows = [(k, l) for k, l in LADDER if k in res['readers']] + [('MODEL_POINT', 'model direct answer: point'),
                                                                 ('MODEL_BOXC', 'model direct answer: box centre')]
    M = np.full((len(rows), len(SETS)), np.nan)
    H = np.full_like(M, np.nan)
    ref = res['readers']['all_content_approx']['test']
    for i, (k, _) in enumerate(rows):
        for j, s in enumerate(SETS):
            if k == 'MODEL_POINT':
                M[i, j], H[i, j] = ref[s]['own_err_median'], ref[s]['own_hit']
            elif k == 'MODEL_BOXC':
                M[i, j] = ref[s]['own_box_center_err_median']
            else:
                t = res['readers'][k]['test'][s]
                M[i, j], H[i, j] = t['tool_err_median'], t['tool_hit']
    fig, ax = plt.subplots(figsize=(12.5, 0.5 * len(rows) + 1.6))
    im = ax.imshow(np.log10(M), cmap='viridis_r', vmin=0.3, vmax=2.5, aspect='auto')
    for i in range(len(rows)):
        for j in range(len(SETS)):
            txt = f'{M[i, j]:.0f}' + ('' if np.isnan(H[i, j]) else f'\n{H[i, j]:.2f}')
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.5, color='white' if np.log10(M[i, j]) > 1.5 else 'black')
    ax.set_xticks(range(len(SETS)), [LABEL[s] + (' *' if s in HELDOUT else '') for s in SETS], rotation=30, ha='right')
    ax.set_yticks(range(len(rows)), [l for _, l in rows])
    for y in (3.5, len(rows) - 3.5):
        ax.axhline(y, color='white', lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    ticks = [2, 5, 10, 20, 50, 100, 200]
    cb.set_ticks(np.log10(ticks), labels=[str(t) for t in ticks])
    cb.set_label('median centre error (0-1000 units)')
    ax.set_title('From the 3.11 attention-peak decoder to the learned tool (cell: median error / point hit rate)')
    save(fig, figs / 'ladder.png')


def fig_heads(res: dict, figs: Path) -> None:
    """Accuracy against the number of attention heads in the reader input (all with content + rough point)."""
    pts = [(0, 'k0_content_approx'), (1, 'k1_H11_content_approx'), (2, 'k2_content_approx'), (4, 'k4_content_approx'),
           (8, 'k8_content_approx'), (16, 'L20_content_approx'), (32, 'L16L20_content_approx'), (128, 'all_content_approx')]
    pts = [(k, n) for k, n in pts if n in res['readers']]
    xs = np.asarray([max(k, 0.5) for k, _ in pts])
    seeds = res.get('seeds', {})
    panels = [('validation mean centre error', lambda n: res['readers'][n]['val']['mean'] if res['readers'][n].get('val') else np.nan,
               'val_mean'),
              ('RefCOCO test: median centre error', lambda n: res['readers'][n]['test']['refcoco']['tool_err_median'], 'rc_median'),
              ('RefCOCO test: box IoU', lambda n: res['readers'][n]['test']['refcoco']['tool_iou'], 'rc_iou'),
              ('held-out families (disc, ring, T6, layout): hit rate',
               lambda n: np.mean([res['readers'][n]['test'][s]['tool_hit'] for s in HELDOUT]), 'heldout_hit')]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (title, f, key) in zip(axes, panels):
        y = np.asarray([f(n) for _, n in pts], float)
        ax.plot(xs, y, '-o', color=FULL, lw=2, ms=6, label='seed 0')
        for k, n in pts:
            if n in seeds and key in seeds[n]:
                v = np.asarray(seeds[n][key])
                ax.errorbar(max(k, 0.5), v.mean(), yerr=[[v.mean() - v.min()], [v.max() - v.mean()]], fmt='s', color=ONE,
                            ms=6, capsize=3, label='3 seeds: mean, range')
        if key != 'val_mean' and key != 'heldout_hit':
            m = res['readers']['all_content_approx']['test']['refcoco']
            ax.axhline(m['own_err_median'] if key == 'rc_median' else m['own_iou'], color=MODEL, ls='--', lw=1.5, label='model')
        if key == 'heldout_hit':
            m = res['readers']['all_content_approx']['test']
            ax.axhline(np.mean([m[s]['own_hit'] for s in HELDOUT]), color=MODEL, ls='--', lw=1.5, label='model')
        ax.set_xscale('log', base=2)
        ax.set_xticks(xs, ['0'] + [str(k) for k, _ in pts[1:]])
        ax.set_xlabel('number of attention heads in the input')
        ax.set_title(title, fontsize=10)
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, frameon=False)
    fig.suptitle('Reader accuracy against input size (every reader also gets layer-0/16 content features and the rough point)')
    save(fig, figs / 'heads_curve.png')


def fig_salient(res: dict, figs: Path) -> None:
    sal = res['failure']['salient']
    show = [('H11 peak', 'L20 H11 attention peak at the slot', PEAK), ('tool:k1_H11_content_approx', 'one-head tool point', ONE),
            ('model point', 'model point answer', MODEL)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={'wspace': 0.35})
    bins = np.linspace(-0.25, 1.25, 31)
    for ax, (k, title, _) in zip(axes, show):
        uv = np.asarray(sal[k]['uv'])
        Hh, _, _ = np.histogram2d(np.clip(uv[:, 0], -0.25, 1.25), np.clip(uv[:, 1], -0.25, 1.25), bins=[bins, bins])
        im = ax.imshow(Hh.T / len(uv), origin='upper', extent=[bins[0], bins[-1], bins[-1], bins[0]], cmap='viridis')
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec='white', lw=1.5))
        ax.plot(0.5, 0.5, '+', color='white', ms=12, mew=2)
        ax.set_title(f'{title}\nmedian (u, v) = ({sal[k]["u_median"]:.2f}, {sal[k]["v_median"]:.2f}); on mask {sal[k]["on_mask"]:.2f}', fontsize=9)
        ax.set_xlabel('u (0 = left edge, 1 = right edge)')
        if k == show[0][0]:
            ax.set_ylabel('v (0 = top edge, 1 = bottom edge)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label('fraction of cases per bin')
    fig.suptitle('Where the readouts land inside the RefCOCO target box (white frame = box, white cross = box centre; n = 1562)')
    save(fig, figs / 'salient.png')


def fig_failure(res: dict, figs: Path) -> None:
    f = res['failure']
    who = [('model', 'model point answer', MODEL), ('all_content_approx', '128-head tool', FULL),
           ('k1_H11_content_approx', 'one-head tool', ONE), ('zs_H11_peak', 'H11 peak (3.11 decoder)', PEAK)]
    panels = [('area', 'target box area / image area'), ('others', 'other annotated objects in the image'),
              ('occlusion', 'max fraction of target box covered by another box')]
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8), gridspec_kw={'width_ratios': [1, 1, 1, 1.0], 'wspace': 0.3})
    for ax, (key, xl) in zip(axes, panels):
        bins = [b for b in f['model'][key] if b['n'] > 0]
        x = np.arange(len(bins))
        for i, (k, lab, c) in enumerate(who):
            bb = [b for b in f[k][key] if b['n'] > 0]
            y = np.asarray([b['rate'] for b in bb])
            ci = np.asarray([b['ci'] for b in bb])
            ax.errorbar(x + (i - 1.5) * 0.12, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt='o-', color=c, ms=5, lw=1.5, capsize=2, label=lab)
        lab = {'area': lambda b: f'{b["lo"]:.2f}-{min(b["hi"], 1):.2f}', 'others': lambda b: f'{b["lo"]}' if b['hi'] - b['lo'] == 1 else f'{b["lo"]}-{b["hi"] - 1}' if b['hi'] < 100 else f'>={b["lo"]}',
               'occlusion': lambda b: '0' if b['hi'] < 1e-3 else ('>0-0.25' if b['lo'] < 1e-3 else f'{b["lo"]:.2f}-{min(b["hi"], 1):.2f}')}[key]
        ax.set_xticks(x, [lab(b) + f'\nn={b["n"]}' for b in bins], fontsize=8)
        ax.set_xlabel(xl)
        ax.set_ylabel('point hit rate (Wilson 95% CI)')
        ax.set_ylim(0.6, 1.01)
    axes[0].legend(fontsize=8, frameon=False, loc='lower right')
    ax = axes[3]
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.08, pos.y0, pos.width - 0.08, pos.height])
    cats = ['target|target', 'target|background', 'target|distractor', 'background|target', 'distractor|target',
            'background|background', 'distractor|distractor']
    names = {'target|target': 'both hit', 'target|background': 'tool hit, model off-object', 'target|distractor': 'tool hit, model other object',
             'background|target': 'tool off-object, model hit', 'distractor|target': 'tool other object, model hit',
             'background|background': 'both off-object', 'distractor|distractor': 'both other object'}
    for i, (k, lab, c) in enumerate(who[1:3]):
        sel = f[k]['selection']
        v = [sel.get(cc, 0) for cc in cats[1:]]
        ax.barh(np.arange(len(v)) + (i - 0.5) * 0.38, v, height=0.36, color=c, label=lab)
    ax.set_yticks(range(len(cats) - 1), [names[c] for c in cats[1:]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('number of RefCOCO test cases (both hit: %d / %d)' % (f[who[1][0]]['selection'].get('target|target', 0), f['n']))
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    fig.suptitle('RefCOCO test failure modes (n = 1562): hit rate by object size, number of annotated objects, overlap; joint tool / model outcomes')
    save(fig, figs / 'failure.png')


def fig_examples(r2run: Path, r1: Path, out: Path, figs: Path, res: dict) -> None:
    """RefCOCO examples with the L20 H11 slot attention map, chosen by failure type."""
    rows = json.loads((out / 'rows_focus.json').read_text())
    cases = {c['id']: c for c in read_jsonl(r2run / 'cases.jsonl')}
    one = {r['id']: r for r in rows['k1_H11_content_approx'] if r['set'] == 'refcoco'}
    full = {r['id']: r for r in rows['all_content_approx'] if r['set'] == 'refcoco'}
    sal = res['failure']['salient']['H11 peak']
    peak = dict(zip(sal['ids'], np.asarray(sal['uv'])))
    rng = np.random.default_rng(3)
    groups = {
        'salient part (H11 peak far from centre, tool hits)': [i for i, r in one.items() if r['tool_hit'] and r['tool_error'] < 30 and peak[i][1] < 0.2],
        'box extent error (point hits, IoU < 0.5)': [i for i, r in one.items() if r['tool_hit'] and r['tool_iou'] < 0.5],
        'tool picks another object, model hits': [i for i, r in one.items() if not r['tool_hit'] and r['own_hit']],
        'both miss': [i for i, r in one.items() if not r['tool_hit'] and not r['own_hit']],
    }
    pick = {g: [ids[k] for k in rng.choice(len(ids), min(3, len(ids)), replace=False)] for g, ids in groups.items() if ids}
    want = {i for v in pick.values() for i in v}
    maps = load_maps(r2run, want)
    fig, axes = plt.subplots(len(pick), 3, figsize=(13, 3.6 * len(pick)))
    axes = np.atleast_2d(axes)
    for gi, (g, ids) in enumerate(pick.items()):
        for j, cid in enumerate(ids):
            ax = axes[gi, j]
            c = cases[cid]
            img = image_of(c, r2run, r1)
            W, Hh = img.size
            gh, gw = rt_.grid_of((W, Hh))
            ax.imshow(img)
            m = maps[cid].reshape(gh, gw)
            share = m / m.sum()
            im = ax.imshow(np.ma.masked_less(share, 0.1 * share.max()), extent=[0, gw * 32, gh * 32, 0], cmap='viridis',
                           alpha=0.6, vmin=0, vmax=share.max())
            t = c['truth']['box']
            ax.add_patch(plt.Rectangle((t[0], t[1]), t[2] - t[0], t[3] - t[1], fill=False, ec='white', lw=2.5))
            for row, col, ls, mk in ((one[cid], ONE, '--', 'x'),):
                if row['tool_box']:
                    b = rt_.to_pixels(row['tool_box'], (W, Hh))
                    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=2, ls=ls))
                ax.plot(*rt_.to_pixels(row['tool_point'], (W, Hh)), mk, color=col, ms=10, mew=2.5)
            if one[cid]['own_box']:
                b = rt_.to_pixels(one[cid]['own_box'], (W, Hh))
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=MODEL, lw=2, ls=':'))
            pk = rt_.to_pixels(r3.peak_centroid(maps[cid], (gh, gw)), (W, Hh))
            ax.plot(*pk, '*', color=PEAK, ms=13, mec='black')
            ax.set_title(f'"{c["target"][:38]}"\ntool IoU {one[cid]["tool_iou"]:.2f}, model IoU {one[cid]["own_iou"]:.2f}', fontsize=8)
            ax.axis('off')
            if j == 0:
                ax.text(-0.04, 0.5, g, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01).set_label('H11 attention share', fontsize=7)
    handles = [Line2D([], [], color='0.6', lw=2.5, label='ground-truth box (white)'),
               Line2D([], [], color=ONE, lw=2, ls='--', marker='x', label='one-head tool box / point'),
               Line2D([], [], color=MODEL, lw=2, ls=':', label='model box answer'),
               Line2D([], [], color=PEAK, marker='*', ls='', ms=12, mec='black', label='H11 peak (3.11 decoder)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    save(fig, figs / 'examples.png')


def load_maps(r2run: Path, want: set[str]) -> dict[str, np.ndarray]:
    """L20 H11 slot attention of selected cases from the round-2 capture chunks."""
    out = {}
    h = r3.head_index(20, 11)
    for f in sorted((r2run / 'features').glob('chunk_s*_*.npz')):
        d = np.load(f)
        ids = d['ids'].tolist()
        hit = [i for i, c in enumerate(ids) if c in want]
        if not hit:
            continue
        cut = np.cumsum(np.concatenate([[0], d['map_len']]))
        m = d['maps_slot'][h]
        for i in hit:
            out[ids[i]] = m[cut[i]:cut[i + 1]].astype(np.float32)
        if len(out) == len(want):
            break
    return out


def fig_vec(out: Path, figs: Path) -> None:
    p = out / 'vec_summary.json'
    if not p.exists():
        return
    d = json.loads(p.read_text())
    names = list(d['readers'])
    sets = [s for s in SETS if s in d['readers'][names[0]]]
    M = np.asarray([[d['readers'][n][s]['median'] for s in sets] for n in names])
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(names) + 1.5))
    im = ax.imshow(np.log10(M), cmap='viridis_r', vmin=0.3, vmax=2.5, aspect='auto')
    for i in range(len(names)):
        for j in range(len(sets)):
            ax.text(j, i, f'{M[i, j]:.0f}\n{d["readers"][names[i]][sets[j]]["hit"]:.2f}', ha='center', va='center', fontsize=7.5,
                    color='white' if np.log10(M[i, j]) > 1.5 else 'black')
    ax.set_xticks(range(len(sets)), [LABEL[s] + (' *' if s in HELDOUT else '') for s in sets], rotation=30, ha='right')
    ax.set_yticks(range(len(names)), [d['labels'].get(n, n) for n in names])
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    ticks = [2, 5, 10, 20, 50, 100, 200]
    cb.set_ticks(np.log10(ticks), labels=[str(t) for t in ticks])
    cb.set_label('median centre error (0-1000 units)')
    ax.set_title('Flash-attention-compatible inputs on the capture subset (cell: median error / point hit rate)')
    save(fig, figs / 'vec_readers.png')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--r1', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--only', default='')
    a = p.parse_args()
    a.figures.mkdir(parents=True, exist_ok=True)
    res = json.loads((a.out / 'results_r3.json').read_text())
    todo = [s for s in a.only.split(',') if s] or ['ladder', 'heads', 'salient', 'failure', 'examples', 'vec', 'training']
    if 'ladder' in todo:
        fig_ladder(res, a.figures)
    if 'heads' in todo:
        fig_heads(res, a.figures)
    if 'salient' in todo:
        fig_salient(res, a.figures)
    if 'failure' in todo:
        fig_failure(res, a.figures)
    if 'examples' in todo:
        fig_examples(a.run, a.r1, a.out, a.figures, res)
    if 'vec' in todo:
        fig_vec(a.out, a.figures)
    if 'training' in todo:
        from analysis_tools.position.readout_tool_r3_compare import fig_training
        fig_training(a.out, a.figures)


if __name__ == '__main__':
    main()
