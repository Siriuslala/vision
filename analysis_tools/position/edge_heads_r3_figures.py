"""Figures of experiment 3.11 round 5 (edge_heads_r3.py, edge_heads_r3_analysis.py)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from analysis_tools.position import edge_heads as eh
from analysis_tools.position import edge_heads_r3 as e3

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GRAY = '#9a9994'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
KIND_LABEL = {'shape_white': 'shape / white', 'shape_grey': 'shape / grey', 'shape_noise': 'shape / noise',
              'shape_photo': 'shape / photo', 'texshape_white': 'textured shape / white', 'cutout_white': 'cut-out / white',
              'cutout_grey': 'cut-out / grey', 'cutout_photo': 'cut-out / photo', 'grid10': '10x10 grid'}
KINDS = list(KIND_LABEL)


def bar_ci(ax, x, v, color, label=None, width=0.38):
    m = np.array([t[0] for t in v])
    lo = np.array([t[1] for t in v])
    hi = np.array([t[2] for t in v])
    ax.bar(x, m, width, color=color, label=label, edgecolor='white', linewidth=1)
    ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt='none', ecolor='#333333', elinewidth=0.8, capsize=2)


def fig_avg(S: dict, tag: str, out: Path) -> None:
    sl = S[tag]['slopes']
    conds = ['B', 'grp_moved', 'grp_other', 'both', 'all_but_edges', 'all']
    labels = ['whole\nimage B', 'readers of\nthe moved\nedge*', 'readers of\nthe fixed\nedge*',
              'both\nreader groups', 'all other\nheads', 'all\nheads']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, setname, title in zip(axes, ('str', 'rect'), ('natural objects (one edge moved)', 'solid rectangle on white')):
        x = np.arange(len(conds))
        for j, (f, col) in enumerate((('point', CAT[0]), ('bbox', CAT[1]))):
            d = sl.get(f'{setname}:{f}:any', {})
            v = [d.get(c, [np.nan] * 4) for c in conds]
            bar_ci(ax, x + (j - 0.5) * 0.4, v, col, {'point': 'centre (point answer)', 'bbox': 'moved box edge'}[f])
        ax.axhline(0.5, color=GRAY, lw=0.8, ls='--')
        ax.axhline(1.0, color=GRAY, lw=0.8, ls=':')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(title)
    axes[0].set_ylabel('median slope of the written value\n(pixels per pixel of edge shift)')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(out / f'avg_slopes_{tag}.png', dpi=170)
    plt.close(fig)


def fig_neurons(S: dict, out: Path) -> None:
    rec = S['q35']['recovery']
    ks = [8, 32, 128, 512, 2048, 8192]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    ax = axes[0]
    for f, col in (('point', CAT[0]), ('bbox', CAT[1])):
        d = rec[f'all:{f}']
        m = [d[f'top{k}'][0] for k in ks]
        lo = [d[f'top{k}'][1] for k in ks]
        hi = [d[f'top{k}'][2] for k in ks]
        ax.plot(ks, m, '-o', color=col, ms=4, lw=2, label={'point': 'centre (point)', 'bbox': 'moved box edge'}[f])
        ax.fill_between(ks, lo, hi, color=col, alpha=0.15, lw=0)
        ax.axhline(d['mlp24_30'][0], color=col, ls='--', lw=1)
        ax.plot([2048], [d['rand2048'][0]], 'x', color=col, ms=7)
    ax.set_xscale('log')
    ax.set_xlabel('number of MLP 24-30 neurons patched from run B (top-k by |attribution|)')
    ax.set_ylabel('first-digit logit recovery r')
    ax.set_title('per-pair top-k neurons (dashed: all neurons of MLP 24-30;\ncross: 2048 random neurons)', fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    n = S['q35']['neurons']
    for f, col in (('point', CAT[0]), ('bbox', CAT[1])):
        ax.plot(n['k'], n['concentration'][f], '-o', color=col, ms=4, lw=2, label=f)
    ax.set_xscale('log')
    ax.set_xlabel('top-k neurons (by |attribution|)')
    ax.set_ylabel('share of the summed attribution')
    ax.set_title(f"attribution concentration; cosine of attribution vectors\nacross pairs {n['cos_across_pairs']['point']:.2f} (point), "
                 f"{n['cos_across_pairs']['bbox']:.2f} (box)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / 'avg_neurons_q35.png', dpi=170)
    plt.close(fig)


def fig_mix(S: dict, tag: str, out: Path) -> None:
    mx = S[tag]['mix']
    kinds = [k for k in KINDS if k in mx]
    rows = [('H11', 'L20 H11 alone'), ('top1', 'largest single head'), ('n5', 'heads with r >= 0.05 (count / 10)'),
            ('all_but_H11', 'all heads except H11'), ('kv_target', 'image K/V: target tokens'), ('kv_ring', 'image K/V: ring'),
            ('kv_row_left', 'image K/V: same rows, left'), ('kv_row_right', 'image K/V: same rows, right'),
            ('kv_rest', 'image K/V: rest')]
    if tag != 'q35':
        rows = [r for r in rows if r[0] not in ('H11', 'all_but_H11')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(rows) + 1.6))
    for ax, axis in zip(axes, ('y', 'x')):
        M = np.full((len(rows), len(kinds)), np.nan)
        for j, k in enumerate(kinds):
            d = mx[k].get(axis, {})
            for i, (c, _) in enumerate(rows):
                if c == 'top1':
                    M[i, j] = d.get('top_singles', [[None, np.nan]])[0][1]
                elif c == 'n5':
                    M[i, j] = d.get('n_heads_r_ge_0.05', np.nan) / 10
                elif c in d:
                    M[i, j] = d[c][0]
        im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if M[i, j] < 0.55 else 'black')
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([KIND_LABEL[k] for k in kinds], rotation=35, ha='right', fontsize=7.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[1] for r in rows] if axis == 'y' else [], fontsize=8)
        ax.set_title(f'{axis} value (first digit)')
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label('first-digit logit recovery r (count row: heads / 10)')
    fig.savefig(out / f'mix_recovery_{tag}.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def strict_hit(r: dict) -> bool:
    """Point answer inside the target box under the 0-1000 convention or the pixel convention."""
    v = {d['key']: d['value'] for d in r['values']}
    W, H = r['canvas']
    b = r['box_a']
    for x, y in ((v['x'] * W / 1000, v['y'] * H / 1000), (v['x'], v['y'])):
        if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
            return True
    return False


def fig_mix_examples(run: Path, round3: Path, out: Path) -> None:
    """Run-A attention of L20 H11 and two edge heads over the image tokens at the y first-digit query, for one
    item of four kinds (attention probabilities from the model, renormalised over the image tokens)."""
    r3 = eh.load_cases(round3)
    cases = {c['id']: c for c in json.loads((run / 'mix_cases.json').read_text())}
    recs = [json.loads(l) for l in open(run / 'mix_q35.jsonl')]
    recs = [r for r in recs if 'lds' in r]
    kinds = ['grid10', 'shape_white', 'shape_photo', 'cutout_white']
    heads = [(20, 11), (20, 10), (24, 5)]
    layers = [12, 16, 20, 24, 28, 32]
    fig, axes = plt.subplots(len(kinds), 1 + len(heads), figsize=(2.3 * (1 + len(heads)) + 0.6, 2.3 * len(kinds)))
    for i, k in enumerate(kinds):
        pool = [x for x in recs if x['kind'] == k and cases[x['id']].get('ndistr', 0) > 0]
        r = next((x for x in pool if strict_hit(x)), pool[0])          # grids: no correct answer exists
        c = cases[r['id']]
        img, box = e3.render_mix(c, r3, False)
        z = np.load(run / 'mix_q35' / f"{r['id']}.npz")
        att = z['att_a'].astype(np.float32)
        Hn, Wn = r['grid']
        W = c['canvas'][0]
        axes[i, 0].imshow(img)
        axes[i, 0].add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec='#1baf7a', lw=1.5))
        axes[i, 0].set_title(f"{KIND_LABEL[k]}\n'{c['expr']}'", fontsize=7.5)
        for j, (l, h) in enumerate(heads):
            p = att[layers.index(l), h, 1]
            p = p / p.sum()
            ax = axes[i, j + 1]
            ax.imshow(img, alpha=0.35)
            im = ax.imshow(p.reshape(Hn, Wn), cmap='viridis', alpha=0.85, extent=(0, W, W, 0), vmin=0, vmax=max(0.05, p.max()))
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec='white', lw=1))
            if i == 0:
                ax.set_title(f'L{l} H{h}', fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.text(0.5, 0.005, 'colour: attention probability on each image token at the query of the y value (first digit); '
             'green / white box: target', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / 'mix_examples_q35.png', dpi=150)
    plt.close(fig)


def fig_transplant(S: dict, out: Path) -> None:
    """Left: attention-pattern transplants between L20 H10 and L20 H11 (which share their value vectors, key/value
    head 2 of layer 20); right: recovery of all heads of layer 20 vs layer 24 and of H11 alone (y value)."""
    m2, mx = S['q35']['mix2'], S['q35']['mix']
    kinds = [k for k in KINDS if k in m2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    x = np.arange(len(kinds))
    series = [('L20 H10 (own attention)', lambda k: m2[k]['y']['own:L20H10'], CAT[1]),
              ('L20 H11 with L20 H10 attention', lambda k: m2[k]['y']['pat_of:L20H10'], CAT[3]),
              ('L20 H10 with L20 H11 attention', lambda k: m2[k]['y']['h11pat:L20H10'], CAT[4]),
              ('L20 H11 (own attention)', lambda k: mx[k]['y']['H11'], CAT[0])]
    w = 0.2
    for i, (lab, fn, col) in enumerate(series):
        bar_ci(ax, x + (i - 1.5) * w, [fn(k) for k in kinds], col, lab, width=w)
    ax.set_xticks(x)
    ax.set_xticklabels([KIND_LABEL[k] for k in kinds], rotation=35, ha='right', fontsize=7.5)
    ax.set_ylabel('first-digit logit recovery r (y value)')
    ax.set_title('same values (key/value head 2 of layer 20), swapped attention', fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax.set_ylim(0, 1.15)
    ax = axes[1]
    series = [('all 16 heads of layer 20', lambda k: mx[k]['y']['L20'], CAT[0]),
              ('all 16 heads of layer 24', lambda k: mx[k]['y']['L24'], CAT[2]),
              ('L20 H11 alone', lambda k: mx[k]['y']['H11'], CAT[3])]
    w = 0.26
    for i, (lab, fn, col) in enumerate(series):
        bar_ci(ax, x + (i - 1) * w, [fn(k) for k in kinds], col, lab, width=w)
    ax.set_xticks(x)
    ax.set_xticklabels([KIND_LABEL[k] for k in kinds], rotation=35, ha='right', fontsize=7.5)
    ax.set_title('which layer delivers the y read (head-output patching from run B)', fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(out / 'mix_transplant_q35.png', dpi=170)
    plt.close(fig)


def fig_flow(out: Path) -> None:
    """Schematic of the process narrative (not data): from image tokens to the first digit of a centre value."""
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=color, ec='#444444', lw=0.8, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.6, wrap=True)

    def arrow(x0, y0, x1, y1, text=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10, lw=1, color='#444444'))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha='center', fontsize=6.8, color='#333333')
    box(0.1, 3.6, 2.2, 1.3, 'image tokens (K/V fixed)\nleft-edge band, right-edge band,\ntop / bottom bands, background', '#dfe9f7')
    box(0.1, 1.4, 2.2, 1.3, 'text tokens: expression\n("the dog" / "the red square")\nGated DeltaNet carries the\nreferent identity forward', '#f6e3d8')
    box(2.9, 3.85, 2.6, 1.05, 'left / top reader group\n(L20 H0, L24 H2, L24 H7, ...)\nreads the left edge at x,\nthe top edge at y', '#d6efe4')
    box(2.9, 2.55, 2.6, 1.05, 'right / bottom reader group\n(L20 H10, L24 H5, L24 H9, L28 H0)\nreads the right edge at x,\nthe bottom edge at y', '#d6efe4')
    box(2.9, 1.0, 2.6, 1.15, 'L20 H11: query selects the referent\nby colour / shape; on uniform\nbackgrounds it is the dominant\nreader of y; on grids the only one', '#fbe8b5')
    box(6.1, 2.2, 2.3, 1.9, 'residual stream at the\nanswer position:\nh = h0 + u_left + u_right\n+ u_H11 + ...\n(each head adds its own\nvector: the summation)', '#ececec')
    box(9.0, 2.2, 2.8, 1.9, 'MLP 24-30 (+ L32 relay heads,\nunembedding): decode one\nposition from the reports in h.\nleft + right reports -> midpoint;\nreports of one edge -> that edge;\nconflicting reports -> majority', '#e6def7')
    box(9.5, 0.4, 1.8, 0.9, 'first digit of the value\n(tens / units relayed by\nDeltaNet short convolution)', '#ffffff')
    arrow(2.3, 4.25, 2.9, 4.35, 'K/V')
    arrow(2.3, 4.0, 2.9, 3.1)
    arrow(2.3, 3.8, 2.9, 1.6)
    arrow(2.3, 2.05, 2.9, 1.5, 'query')
    arrow(2.3, 2.4, 2.9, 2.8, 'query')
    arrow(5.5, 4.35, 6.1, 3.7, 'W_O o')
    arrow(5.5, 3.05, 6.1, 3.15, 'W_O o')
    arrow(5.5, 1.6, 6.1, 2.5, 'W_O o')
    arrow(8.4, 3.15, 9.0, 3.15)
    arrow(10.4, 2.2, 10.4, 1.3)
    ax.text(6, 5.05, 'Schematic (not data): how Qwen3.5-9B writes the first digit of a centre coordinate', ha='center', fontsize=9)
    fig.savefig(out / 'flow.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--round3', type=Path, required=True)
    p.add_argument('--round4', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    S = json.loads((args.run / 'summary.json').read_text())
    for tag in ('q35', 'vl'):
        if S.get(tag, {}).get('slopes'):
            fig_avg(S, tag, args.out)
        if S.get(tag, {}).get('mix'):
            fig_mix(S, tag, args.out)
    fig_flow(args.out)
    if S['q35'].get('mix2'):
        fig_transplant(S, args.out)
    if (args.run / 'mix_q35.jsonl').exists():
        fig_mix_examples(args.run, args.round3, args.out)
    if S['q35'].get('neurons'):
        fig_neurons(S, args.out)


if __name__ == '__main__':
    main()
