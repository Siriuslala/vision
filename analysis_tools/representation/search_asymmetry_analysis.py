"""Experiment 2.7 CPU analysis: behaviour atlas, ViT presence neurons, symmetric neuron edits, target salience,
mixer / gate interventions, attention edges, and figures.

Stages:
  select   neuron sets for the edit stage (neurons.json) from capture/*.npz
  analyze  every table (summary.json) and figure

Colour conventions (repeated in the document captions): blue = std (the odd entity has feature A, "presence"),
orange = swap (the odd entity lacks it, "absence"); heatmaps use viridis with a labelled colorbar.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, MUTED, GRIDC = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRIDC, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 130,
                     'savefig.bbox': 'tight', 'legend.frameon': False, 'lines.linewidth': 2})
PAIR_LABEL = {'qo': 'Q vs O', 'tilt': 'tilted vs vertical', 'length': 'long vs short', 'curve': 'curved vs straight',
              'closure': 'open vs closed', 'sat': 'saturated vs desat.', 'size': 'large vs small',
              'gap2': '2.6 gap 2 px', 'red': '2.6 red', 'gap16': '2.6 gap 16 px', 'color20': '2.6 colour dE20'}


def wrow(k: int, n: int) -> dict:
    p, lo, hi = oe.wilson(k, n)
    return {'k': int(k), 'n': int(n), 'p': p, 'lo': min(lo, p), 'hi': max(hi, p)}


def load_capture(output: Path, name: str):
    z = np.load(output / 'capture' / f'{name}.npz')
    ids = [str(x) for x in z['ids']]
    n = len(ids) // 4
    return z, ids, n


# ----------------------------------------------------------------------------- neuron selection

def neuron_stats(z, n: int, block: int, g: int = 6) -> dict:
    """Selectivity of every ViT MLP neuron from the homogeneous images, and target deviations in std / swap.

    Layout of z: images ordered as (std, swap, homA, homB) per quadruple.  act: (4n, n_ent, 4304) max over the
    16 patches of each entity.  Returns d' between A and B entities, and per-image target deviations.
    """
    act = z[f'act_vit{block}'].astype(np.float32)
    targets = z['targets']
    std, swap, homa, homb = act[0::4], act[1::4], act[2::4], act[3::4]
    dsel = sa.selectivity(homa.reshape(-1, act.shape[-1]), homb.reshape(-1, act.shape[-1]))
    t_idx = [int(t[0]) * g + int(t[1]) for t in targets[0::4]]
    dev_std = np.stack([sa.target_dprime(std[i], t) for i, t in enumerate(t_idx)])     # (n, 4304)
    dev_swap = np.stack([sa.target_dprime(swap[i], t) for i, t in enumerate(t_idx)])
    return {'dsel': dsel, 'dev_std': dev_std, 'dev_swap': dev_swap, 'mean_a': homa.mean((0, 1)),
            'mean_b': homb.mean((0, 1)), 'sd_a': homa.std((0, 1)), 'sd_b': homb.std((0, 1)), 't_idx': t_idx}


def stage_select(output: Path, top_k: int, blocks: list[int], tag: str = '') -> dict:
    sel = {}
    for path in sorted((output / 'capture').glob('*.npz')):
        name = path.stem
        z, ids, n = load_capture(output, name)
        source, pair = name.split('_', 1)
        sel[f'{source}|{pair}'] = {}
        for b in blocks:
            st = neuron_stats(z, n, b)
            order = np.argsort(st['dsel'])
            sel[f'{source}|{pair}'][str(b)] = {'A': [int(i) for i in order[::-1][:top_k]],
                                               'B': [int(i) for i in order[:top_k]],
                                               'dA': float(st['dsel'][order[::-1][:top_k]].mean()),
                                               'dB': float(-st['dsel'][order[:top_k]].mean())}
    (output / f'neurons{tag}.json').write_text(json.dumps(sel))
    return sel


# ----------------------------------------------------------------------------- behaviour

def behaviour(output: Path, items: dict, odd_entity: Path | None) -> dict:
    """Accuracy per (pair, level, grid, variant) with Wilson intervals, paired McNemar tests, raster-row profile."""
    recs = {r['item']: r for r in read_jsonl(output / 'results_behave.jsonl')}
    cells = defaultdict(list)
    for iid, r in recs.items():
        it = items[iid]
        cells[(it['pair_name'], it['level'], it['grid'], it['variant'])].append((it['pair'], r, it))
    table = {}
    for (pair, level, g, v), lst in sorted(cells.items()):
        k = sum(r['correct'] for _, r, _ in lst)
        kn = sum(r['near'] for _, r, _ in lst)
        n = len(lst)
        rows = defaultdict(lambda: [0, 0])
        for _, r, it in lst:
            rows[it['target'][0]][0] += r['correct']
            rows[it['target'][0]][1] += 1
        table[f'{pair}|{level}|{g}|{v}'] = {**wrow(k, n), 'near': kn / n, 'invalid': sum(not r['valid'] for _, r, _ in lst),
                                            'default11': sum(r['valid'] and r['pred'] == [0, 0] and it['target'] != [0, 0]
                                                             for _, r, it in lst),
                                            'by_row': {int(a): b for a, b in sorted(rows.items())}}
    tests = {}
    for key in table:
        pair, level, g, v = key.split('|')
        if v != 'std' or f'{pair}|{level}|{g}|swap' not in table:
            continue
        a = {p: r['correct'] for p, r, _ in cells[(pair, level, int(g), 'std')]}
        b = {p: r['correct'] for p, r, _ in cells[(pair, level, int(g), 'swap')]}
        common = sorted(set(a) & set(b))
        b_only = sum(a[p] and not b[p] for p in common)
        c_only = sum(b[p] and not a[p] for p in common)
        tests[f'{pair}|{level}|{g}'] = {'std_only': b_only, 'swap_only': c_only, 'p': sa.mcnemar_exact(b_only, c_only),
                                        'diff': table[key]['p'] - table[f'{pair}|{level}|{g}|swap']['p']}
    out = {'table': table, 'tests': tests}
    if odd_entity is not None:
        oe_recs = [r for r in read_jsonl(odd_entity / 'results_behave.jsonl') if r['mode'] == 'direct']
        oe_items = oe.load_items(odd_entity)
        ref = defaultdict(lambda: [0, 0])
        for r in oe_recs:
            it = oe_items[r['item']]
            if it['group'] == 'core':
                ref[(it['kind'], it['variant'])][0] += r['correct']
                ref[(it['kind'], it['variant'])][1] += 1
        out['odd_entity'] = {f'{k}|{v}': wrow(*ref[(k, v)]) for k, v in ref}
    return out


def fig_examples(output: Path, items: dict, figs: Path, pairs: list[str]) -> None:
    by = {(it['pair_name'], it['level'], it['grid'], it['pair'], it['variant']): it for it in items.values()}
    cols = [('std', 'subtle'), ('swap', 'subtle'), ('std', 'salient'), ('swap', 'salient')]
    fig, axes = plt.subplots(len(pairs) + 1, 4, figsize=(9.6, 2.5 * (len(pairs) + 1)))
    for i, pair in enumerate(pairs):
        for j, (v, level) in enumerate(cols):
            it = by[(pair, level, 6, 0, v)]
            ax = axes[i, j]
            ax.imshow(Image.open(output / it['path']))
            x0, y0, x1, y1 = next(ins['cell_box'] for ins in it['meta']['instances']
                                  if [ins['row'], ins['col']] == it['target'])
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ORANGE, lw=1.6, ls='--'))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            unit = sa.PAIRS[pair]['unit']
            ax.set_title(f'{PAIR_LABEL[pair]}: {v}, {level} ({it["param"]} {unit})', fontsize=8)
    for j, cond in enumerate(sa.CONJ_CONDITIONS + ('',)):
        ax = axes[-1, j]
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if not cond:
            ax.axis('off')
            continue
        it = by[(cond, '-', 6, 0, 'std')]
        ax.imshow(Image.open(output / it['path']))
        x0, y0, x1, y1 = next(ins['cell_box'] for ins in it['meta']['instances'] if [ins['row'], ins['col']] == it['target'])
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ORANGE, lw=1.6, ls='--'))
        ax.set_title(f'{cond} (red circle target)', fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'examples.png')
    plt.close(fig)


def fig_atlas(beh: dict, figs: Path, pairs: list[str]) -> None:
    t = beh['table']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, level in zip(axes, ('subtle', 'salient')):
        names = list(pairs) + ([k for k in ('gap2', 'gap16', 'color20', 'red')] if level == 'subtle' and 'odd_entity' in beh else [])
        for i, name in enumerate(names):
            for dx, v, col in ((-0.17, 'std', BLUE), (0.17, 'swap', ORANGE)):
                if name in sa.PAIRS:
                    r = t.get(f'{name}|{level}|6|{v}')
                else:
                    r = beh['odd_entity'].get(f'{name}|{v}')
                if r is None:
                    continue
                ax.bar(i + dx, r['p'], 0.32, color=col, label=v if i == 0 else None)
                ax.errorbar(i + dx, r['p'], yerr=[[r['p'] - r['lo']], [r['hi'] - r['p']]], color=INK, lw=1, capsize=2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([PAIR_LABEL[n] for n in names], rotation=35, ha='right', fontsize=8)
        ax.set_title(f'6x6 grid, {level} level' + (' (+ 2.6 pairs)' if level == 'subtle' else ''))
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('accuracy (exact row and column)')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2)
    fig.tight_layout()
    fig.savefig(figs / 'atlas.png')
    plt.close(fig)


def fig_setsize(beh: dict, figs: Path, pairs: list[str]) -> None:
    t = beh['table']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cmap = plt.get_cmap('tab10')
    for i, pair in enumerate(pairs):
        for v, ls in (('std', '-'), ('swap', '--')):
            xs, ys = [], []
            for g in (3, 6, 10):
                r = t.get(f'{pair}|subtle|{g}|{v}')
                if r:
                    xs.append(g * g); ys.append(r['p'])
            axes[0].plot(xs, ys, ls, marker='o', color=cmap(i), label=PAIR_LABEL[pair] if v == 'std' else None)
    axes[0].set_xscale('log'); axes[0].set_xticks([9, 36, 100]); axes[0].set_xticklabels(['9', '36', '100'])
    axes[0].set_xlabel('number of entities'); axes[0].set_ylabel('accuracy')
    axes[0].set_title('feature pairs, subtle level (solid: std, dashed: swap)')
    axes[0].legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.14))
    for cond, col in zip(sa.CONJ_CONDITIONS, (VIOLET, RED, GREEN)):
        xs, ys, lo, hi = [], [], [], []
        for g in (3, 6, 10):
            r = t.get(f'{cond}|-|{g}|std')
            if r:
                xs.append(g * g); ys.append(r['p']); lo.append(r['p'] - r['lo']); hi.append(r['hi'] - r['p'])
        axes[1].errorbar(xs, ys, yerr=[lo, hi], marker='o', color=col, capsize=3, label=cond)
    axes[1].set_xscale('log'); axes[1].set_xticks([9, 36, 100]); axes[1].set_xticklabels(['9', '36', '100'])
    axes[1].set_xlabel('number of entities'); axes[1].set_ylim(0, 1.05)
    axes[1].set_title('red circle among green circles / red squares')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figs / 'setsize.png')
    plt.close(fig)


# ----------------------------------------------------------------------------- neurons and salience

def neurons(output: Path, blocks: list[int], thresholds=(1.0, 2.0)) -> dict:
    """Counts and strengths of A- and B-selective ViT neurons, and the target's local deviation in std / swap."""
    out = {}
    for path in sorted((output / 'capture').glob('*.npz')):
        z, ids, n = load_capture(output, path.stem)
        res = {}
        for b in blocks:
            st = neuron_stats(z, n, b)
            d = st['dsel']
            order = np.argsort(d)
            top_a, top_b = order[::-1][:32], order[:32]
            r = {f'nA>{t}': int((d > t).sum()) for t in thresholds}
            r.update({f'nB>{t}': int((d < -t).sum()) for t in thresholds})
            r['dA_top32'], r['dB_top32'] = float(d[top_a].mean()), float(-d[top_b].mean())
            # target deviation (in units of the distractor spread) on the top neurons of each direction
            r['dev_std_A'], r['dev_swap_A'] = float(st['dev_std'][:, top_a].mean()), float(st['dev_swap'][:, top_a].mean())
            r['dev_std_B'], r['dev_swap_B'] = float(st['dev_std'][:, top_b].mean()), float(st['dev_swap'][:, top_b].mean())
            # activation levels: A-neurons on A and on B entities (presence = on vs off), and their spread
            r['muA_on_A'], r['muA_on_B'] = float(st['mean_a'][top_a].mean()), float(st['mean_b'][top_a].mean())
            r['sdA_on_A'], r['sdA_on_B'] = float(st['sd_a'][top_a].mean()), float(st['sd_b'][top_a].mean())
            r['muB_on_B'], r['muB_on_A'] = float(st['mean_b'][top_b].mean()), float(st['mean_a'][top_b].mean())
            r['sdB_on_B'], r['sdB_on_A'] = float(st['sd_b'][top_b].mean()), float(st['sd_a'][top_b].mean())
            # number of image-level target "pop-outs": top-32 A neurons where the target is the argmax (std)
            # or the argmin (swap) entity, averaged over images
            res[str(b)] = r
        out[path.stem] = res
    return out


def salience(output: Path, sites: list[str]) -> dict:
    """Mahalanobis salience of the target against the distractors, std vs swap, per state site."""
    out = {}
    for path in sorted((output / 'capture').glob('*.npz')):
        z, ids, n = load_capture(output, path.stem)
        targets = z['targets']
        res = {}
        for site in sites:
            key = f'state_{site}'
            if key not in z:
                continue
            x = z[key].astype(np.float32)
            x = (x - x.reshape(-1, x.shape[-1]).mean(0)) / (x.reshape(-1, x.shape[-1]).std(0) + 1e-6)
            m_std, m_swap = [], []
            for q in range(n):
                t = int(targets[4 * q][0]) * 6 + int(targets[4 * q][1])
                m_std.append(sa.mahalanobis_salience(x[4 * q], t))
                m_swap.append(sa.mahalanobis_salience(x[4 * q + 1], t))
            m_std, m_swap = np.array(m_std), np.array(m_swap)
            res[site] = {'std': float(m_std.mean()), 'swap': float(m_swap.mean()),
                         'log_ratio': float(np.mean(np.log(m_std / m_swap))),
                         'frac_std_gt_swap': float((m_std > m_swap).mean())}
        out[path.stem] = res
    return out


def popout_stat(act: np.ndarray, entity: int, k: int = 10) -> tuple[float, float]:
    """Mean of the k largest positive and k largest negative per-neuron deviations of one entity from the others.

    act: (n_ent, n_neurons).  The deviation is (z_e - mean_others) / (std_others + 0.01); the floor 0.01 keeps
    neurons that are silent on every other entity from producing infinite scores.
    """
    others = np.delete(act, entity, axis=0)
    d = (act[entity] - others.mean(0)) / (others.std(0) + 1e-2)
    d.sort()
    return float(d[-k:].mean()), float(-d[:k].mean())


def popout(output: Path, blocks: list[int], k: int = 10, g: int = 6) -> dict:
    """Best local pop-out of the target among all ViT MLP neurons, std vs swap, with a distractor null.

    The null applies the same statistic to the entity 7 cells after the target (raster order, wrapping) in the
    same std and swap images, i.e. an ordinary distractor.
    """
    out = {}
    for path in sorted((output / 'capture').glob('*.npz')):
        z, ids, n = load_capture(output, path.stem)
        tg = z['targets']
        res = {}
        for b in blocks:
            a = z[f'act_vit{b}'].astype(np.float32)
            pos = {'std': [], 'swap': [], 'null': []}
            neg = {'std': [], 'swap': []}
            for q in range(n):
                t = int(tg[4 * q][0]) * g + int(tg[4 * q][1])
                c = (t + 7) % (g * g)
                for v, i in (('std', 0), ('swap', 1)):
                    p_, n_ = popout_stat(a[4 * q + i], t, k)
                    pos[v].append(p_); neg[v].append(n_)
                pos['null'].append(float(np.mean([popout_stat(a[4 * q + i], c, k)[0] for i in (0, 1)])))
            res[str(b)] = {f'pos_{v}': float(np.mean(x)) for v, x in pos.items()}
            res[str(b)].update({f'neg_{v}': float(np.mean(x)) for v, x in neg.items()})
            res[str(b)]['pos_swap_sem'] = float(np.std(pos['swap']) / math.sqrt(n))
            res[str(b)]['pos_std_sem'] = float(np.std(pos['std']) / math.sqrt(n))
        out[path.stem] = res
    return out


def fig_popout(pop: dict, beh: dict, figs: Path, block: int = 9) -> None:
    """Left: target pop-out (std vs swap) per pair with the distractor null.  Right: pop-out vs accuracy."""
    stems = sorted(pop)
    b = str(block)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    x = np.arange(len(stems))
    ax.bar(x - 0.2, [pop[s_][b]['pos_std'] for s_ in stems], 0.4, color=BLUE, label='std target (has A)')
    ax.bar(x + 0.2, [pop[s_][b]['pos_swap'] for s_ in stems], 0.4, color=ORANGE, label='swap target (lacks A)')
    ax.scatter(x, [pop[s_][b]['pos_null'] for s_ in stems], marker='_', s=300, color=INK, zorder=5,
               label='ordinary distractor (null)')
    ax.set_xticks(x); ax.set_xticklabels([PAIR_LABEL[_name(s_)] for s_ in stems], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('top-10 positive neuron deviation (z units)')
    ax.set_title(f'local pop-out of the target in ViT block {b} MLP neurons')
    ax.legend(fontsize=8)
    ax = axes[1]
    t = beh['table']
    for s_ in stems:
        name = _name(s_)
        for v, col in (('std', BLUE), ('swap', ORANGE)):
            if s_.startswith('oe_'):
                r = beh.get('odd_entity', {}).get(f'{name}|{v}')
            else:
                r = t.get(f'{name}|subtle|6|{v}')
            if r is None:
                continue
            ax.errorbar(pop[s_][b][f'pos_{v}'], r['p'], yerr=[[r['p'] - r['lo']], [r['hi'] - r['p']]], fmt='o',
                        color=col, capsize=2)
            ax.annotate(PAIR_LABEL[name], (pop[s_][b][f'pos_{v}'], r['p']), fontsize=6, xytext=(3, 3),
                        textcoords='offset points')
    ax.set_xscale('log'); ax.set_xlabel('target pop-out (z units, log scale)'); ax.set_ylabel('accuracy (6x6)')
    ax.set_title('pop-out predicts search accuracy (blue std, orange swap)')
    fig.tight_layout()
    fig.savefig(figs / 'popout.png')
    plt.close(fig)


def fig_onoff(output: Path, sel: dict, figs: Path, stems=('sa_qo', 'sa_curve'), block: int = 12) -> None:
    """Activation distributions of the single most A-selective and most B-selective neuron over homA / homB entities."""
    fig, axes = plt.subplots(1, 2 * len(stems), figsize=(4 * 2 * len(stems), 3.4))
    for i, stem in enumerate(stems):
        z, ids, n = load_capture(output, stem)
        a = z[f'act_vit{block}'].astype(np.float32)
        homa, homb = a[2::4].reshape(-1, a.shape[-1]), a[3::4].reshape(-1, a.shape[-1])
        for j, which in enumerate(('A', 'B')):
            nid = sel[stem.replace('_', '|', 1)][str(block)][which][0]
            ax = axes[2 * i + j]
            lo, hi = min(homa[:, nid].min(), homb[:, nid].min()), max(homa[:, nid].max(), homb[:, nid].max())
            bins = np.linspace(lo, hi, 40)
            ax.hist(homb[:, nid], bins, color=ORANGE, alpha=0.7, label='B entities (homB)')
            ax.hist(homa[:, nid], bins, color=BLUE, alpha=0.7, label='A entities (homA)')
            ax.set_title(f'{PAIR_LABEL[_name(stem)]}: top {which}-selective neuron #{nid}', fontsize=9)
            ax.set_xlabel(f'activation (max over 16 patches), ViT block {block}')
            if i == 0 and j == 0:
                ax.set_ylabel('entities'); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / 'onoff.png')
    plt.close(fig)


def fig_maps(output: Path, sel: dict, figs: Path, stem: str = 'sa_qo', q: int = 0, block: int = 12) -> None:
    """Per-entity activation of the top A-selective neuron and per-entity pop-out, in one std and one swap image."""
    z, ids, n = load_capture(output, stem)
    a = z[f'act_vit{block}'].astype(np.float32)
    nid = sel[stem.replace('_', '|', 1)][str(block)]['A'][0]
    tg = z['targets'][4 * q]
    items = sa.load_items(output)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    for r, (v, i) in enumerate((('std', 0), ('swap', 1))):
        ax = axes[r, 0]
        ax.imshow(Image.open(output / items[ids[4 * q + i]]['path'])); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f'{v} input (target row {tg[0] + 1}, col {tg[1] + 1})', fontsize=9)
        act = a[4 * q + i]
        im = axes[r, 1].imshow(act[:, nid].reshape(6, 6), cmap='viridis')
        axes[r, 1].set_title(f'neuron #{nid} activation per entity', fontsize=9)
        fig.colorbar(im, ax=axes[r, 1], label='activation (max over patches)')
        pop = np.array([popout_stat(act, e)[0] for e in range(36)]).reshape(6, 6)
        im = axes[r, 2].imshow(pop, cmap='viridis')
        axes[r, 2].set_title('top-10 positive deviation per entity', fontsize=9)
        fig.colorbar(im, ax=axes[r, 2], label='z units')
        for ax in axes[r, 1:]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.add_patch(plt.Rectangle((tg[1] - 0.5, tg[0] - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2, ls='--'))
    fig.suptitle(f'{PAIR_LABEL[_name(stem)]}, ViT block {block}; orange dashed box = odd entity', fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / 'maps.png')
    plt.close(fig)


# ----------------------------------------------------------------------------- neuron edits

def edits(output: Path, tag: str = '') -> dict:
    rows = read_jsonl(output / f'edit{tag}.jsonl')
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        hit = r['arg_row'] == r['target'][0] + 1 and r['arg_col'] == r['target'][1] + 1
        cond = r['cond'] if not r['cond'].startswith('ref_') else {'ref_homX': 'ref_homA', 'ref_homT': 'ref_homB'}.get(r['cond'], r['cond'])
        by[f"{r['source']}_{r['pair']}"][cond].append((r['index'], hit, r['s'], r['p_row'] * r['p_col']))
    out = {}
    for pair, conds in by.items():
        res = {}
        for c, lst in conds.items():
            k = sum(h for _, h, _, _ in lst)
            res[c] = {**wrow(k, len(lst)), 'S': float(np.mean([x for _, _, x, _ in lst])),
                      'P': float(np.mean([x for _, _, _, x in lst]))}
        # paired comparison inject_A vs delete_A (same items)
        for a, b in (('inject_A', 'delete_A'), ('inject_B', 'delete_B'), ('inject_A', 'inject_B'), ('full_AtoB', 'full_BtoA')):
            if a in conds and b in conds:
                ha = {i: h for i, h, _, _ in conds[a]}
                hb = {i: h for i, h, _, _ in conds[b]}
                common = sorted(set(ha) & set(hb))
                x = sum(ha[i] and not hb[i] for i in common)
                y = sum(hb[i] and not ha[i] for i in common)
                res[f'{a}_vs_{b}'] = {'a_only': x, 'b_only': y, 'p': sa.mcnemar_exact(x, y)}
        out[pair] = res
    return out


# ----------------------------------------------------------------------------- mechanism

def bootstrap_ratio(num: np.ndarray, den: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    r = float(num.sum() / den.sum())
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    bs = num[idx].sum(1) / den[idx].sum(1)
    return r, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mech(output: Path, min_gap: float = 1.0) -> dict:
    rows = [r for r in read_jsonl(output / 'mech.jsonl') if r['cond'] != 'identity']
    ident = [r for r in read_jsonl(output / 'mech.jsonl') if r['cond'] == 'identity']
    groups = defaultdict(list)
    for r in rows:
        if r['s_clean'] - r['s_base'] < min_gap:
            continue
        groups[(r['kind'], r['variant'], r['cond'])].append(r)
    out = {'identity_max_abs': max((abs(r['s'] - r['s_clean']) for r in ident), default=None), 'R': {}, 'n': {}}
    for (kind, var, cond), lst in sorted(groups.items()):
        num = np.array([r['s'] - r['s_base'] for r in lst])
        den = np.array([r['s_clean'] - r['s_base'] for r in lst])
        out['R'][f'{kind}|{var}|{cond}'] = bootstrap_ratio(num, den)
        out['n'][f'{kind}|{var}'] = len(lst)
    out['excluded_small_gap'] = len({r['item'] for r in rows if r['s_clean'] - r['s_base'] < min_gap})
    return out


def attention(output: Path) -> dict:
    rows = read_jsonl(output / 'attn.jsonl')
    acc = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if k.count('|') == 3:
                acc[(r['kind'], r['variant'], k)].append(v)
        # per-token normalisation of the target -> before mass
        for layer in (4, 8, 12, 16, 20):
            if r['n_before']:
                acc[(r['kind'], r['variant'], f'L{layer}|target|before|per_entity')].append(
                    r[f'L{layer}|target|before|mean'] / r['n_before'])
    return {f'{k}|{v}|{key}': float(np.mean(x)) for (k, v, key), x in acc.items()}


# ----------------------------------------------------------------------------- figures (mechanism)

def _name(stem: str) -> str:
    return stem.split('_', 1)[1]


def fig_neurons(neu: dict, sal: dict, beh: dict, figs: Path, blocks: list[int]) -> None:
    stems = sorted(neu)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(stems))
    b = str(blocks[len(blocks) // 2])
    ax = axes[0]
    ax.bar(x - 0.2, [neu[s][b]['nA>2.0'] for s in stems], 0.4, color=BLUE, label="A-selective (d' > 2)")
    ax.bar(x + 0.2, [neu[s][b]['nB>2.0'] for s in stems], 0.4, color=ORANGE, label="B-selective (d' < -2)")
    ax.set_xticks(x); ax.set_xticklabels([PAIR_LABEL[_name(s)] for s in stems], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('number of ViT MLP neurons (of 4304)'); ax.set_title(f'selective neurons, ViT block {b}')
    ax.set_yscale('symlog', linthresh=10); ax.legend(fontsize=8)
    ax = axes[1]
    early = [k for k in blocks if k <= 12]
    for i, s_ in enumerate(stems):
        col = plt.get_cmap('tab10')(i)
        ax.plot(early, [neu[s_][str(k)]['dev_std_A'] for k in early], '-', marker='o', color=col,
                label=PAIR_LABEL[_name(s_)])
        ax.plot(early, [-neu[s_][str(k)]['dev_swap_A'] for k in early], '--', marker='s', color=col)
    ax.set_yscale('log'); ax.set_xlabel('ViT block'); ax.set_ylabel('|target deviation| on top-32 A neurons (z)')
    ax.set_title('local signal: solid = std (target on), dashed = swap (target off)', fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax = axes[2]
    sites = list(next(iter(sal.values())).keys())
    for i, s_ in enumerate(stems):
        ax.plot(range(len(sites)), [sal[s_][t]['log_ratio'] for t in sites], marker='o',
                color=plt.get_cmap('tab10')(i), label=PAIR_LABEL[_name(s_)])
    ax.axhline(0, color=MUTED, ls=':')
    ax.set_xticks(range(len(sites))); ax.set_xticklabels([t.replace('vit', 'V').replace('dec', 'D') for t in sites])
    ax.set_ylabel('mean log(salience std / salience swap)'); ax.set_title('target Mahalanobis salience')
    fig.tight_layout()
    fig.savefig(figs / 'neurons.png')
    plt.close(fig)


def fig_edits(ed: dict, figs: Path, tag: str = '', what: str = 'blocks 3, 6, 9, 12, 32 neurons per block') -> None:
    pairs = sorted(ed)
    conds = [('inject_A', BLUE, 'inject A (presence)'), ('delete_A', ORANGE, 'delete A (absence)'),
             ('inject_B', AQUA, 'inject B'), ('delete_B', YELLOW, 'delete B'), ('random_AtoB', MUTED, 'random set A->B'),
             ('full_AtoB', VIOLET, 'all neurons A into B'), ('full_BtoA', MAGENTA, 'all neurons B into A')]
    fig, ax = plt.subplots(figsize=(14, 4.2))
    w = 0.8 / len(conds)
    for j, (c, col, lab) in enumerate(conds):
        for i, p in enumerate(pairs):
            r = ed[p].get(c)
            if r is None:
                continue
            ax.bar(i + (j - len(conds) / 2 + 0.5) * w, r['p'], w, color=col, label=lab if i == 0 else None)
            ax.errorbar(i + (j - len(conds) / 2 + 0.5) * w, r['p'], yerr=[[r['p'] - r['lo']], [r['hi'] - r['p']]],
                        color=INK, lw=0.8, capsize=1.5)
    for i, p in enumerate(pairs):
        for c, mk in (('ref_std', '_'), ('ref_swap', 'x')):
            if c in ed[p]:
                ax.scatter(i, ed[p][c]['p'], marker=mk, color=INK, s=80, zorder=5,
                           label={'ref_std': 'pixel std (teacher-forced)', 'ref_swap': 'pixel swap'}[c] if i == 0 else None)
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels([PAIR_LABEL[_name(p)] for p in pairs], rotation=20, ha='right')
    ax.set_ylabel('hit rate: row and column argmax = edited entity'); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.set_title(f'symmetric ViT-neuron edits at one entity of a homogeneous 6x6 image ({what})')
    fig.tight_layout()
    fig.savefig(figs / f'edits{tag}.png')
    plt.close(fig)


def fig_mech_layers(me: dict, figs: Path, kinds: list[str]) -> None:
    R = me['R']
    fig, axes = plt.subplots(2, len(kinds), figsize=(3.2 * len(kinds), 6.6), sharey='row')
    for j, kind in enumerate(kinds):
        ax = axes[0, j]
        for var, col in (('std', BLUE), ('swap', ORANGE)):
            ys = [R.get(f'{kind}|{var}|mix_tgt_L{l}', (np.nan,) * 3) for l in range(1, 17)]
            ax.plot(range(1, 17), [y[0] for y in ys], marker='o', color=col, label=var)
            ax.fill_between(range(1, 17), [y[1] for y in ys], [y[2] for y in ys], color=col, alpha=0.15)
        for l in (4, 8, 12, 16):
            ax.axvline(l, color=GRIDC, lw=3, zorder=0)
        ax.set_title(PAIR_LABEL[kind.split(':')[1]], fontsize=9)
        ax.set_xlabel('decoder layer (grey: full attention)')
        ax = axes[1, j]
        names = ['vitattn_tgt_v1_6', 'vitattn_tgt_v7_12', 'vitattn_tgt_v13_18', 'vitattn_tgt_v19_27', 'mix_tgt_dn1_8',
                 'mix_tgt_fa4_8', 'mix_tgt_dn9_16', 'mix_tgt_fa12_16', 'mix_tgt_all1_16', 'mix_ctx_dn1_16', 'mix_ctx_fa1_16']
        labels = ['ViT 1-6', 'ViT 7-12', 'ViT 13-18', 'ViT 19-27', 'DN 1-8', 'FA 4,8', 'DN 9-16', 'FA 12,16',
                  'all 1-16', 'ctx DN', 'ctx FA']
        for dx, var, col in ((-0.2, 'std', BLUE), (0.2, 'swap', ORANGE)):
            for i, n_ in enumerate(names):
                r = R.get(f'{kind}|{var}|{n_}')
                if r is None:
                    continue
                ax.bar(i + dx, r[0], 0.4, color=col)
                ax.errorbar(i + dx, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], color=INK, lw=0.8, capsize=1.5)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
        ax.axhline(1, color=MUTED, ls=':', lw=1); ax.axhline(0, color=MUTED, lw=0.8)
    axes[0, 0].set_ylabel('retained R (one layer patched)')
    axes[1, 0].set_ylabel('retained R (layer group)')
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(figs / 'mixer_patching.png')
    plt.close(fig)


def fig_gates(me: dict, figs: Path, kinds: list[str]) -> None:
    R = me['R']
    names = ['reset_tgt_dn1_8', 'reset_ctrl_dn1_8', 'reset_tgt_dn9_16', 'reset_tgt_dn1_16', 'reset_ctrl_dn1_16',
             'nowrite_before_dn1_8', 'nowrite_after_dn1_8', 'nowrite_before_dn1_16', 'nowrite_after_dn1_16',
             'reset_tgt_dn1_16+mix_tgt_fa1_16']
    labels = ['a=0 target DN1-8', 'a=0 control DN1-8', 'a=0 target DN9-16', 'a=0 target DN1-16', 'a=0 control DN1-16',
              'b=0 before DN1-8', 'b=0 after DN1-8', 'b=0 before DN1-16', 'b=0 after DN1-16', 'a=0 target + FA patch']
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.2 * len(kinds), 4.4), sharey=True)
    for j, kind in enumerate(kinds):
        ax = axes[j]
        for dx, var, col in ((-0.2, 'std', BLUE), (0.2, 'swap', ORANGE)):
            for i, n_ in enumerate(names):
                r = R.get(f'{kind}|{var}|{n_}')
                if r is None:
                    continue
                ax.bar(i + dx, r[0], 0.4, color=col, label=var if i == 0 else None)
                ax.errorbar(i + dx, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], color=INK, lw=0.8, capsize=1.5)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(labels, rotation=70, ha='right', fontsize=7)
        ax.axhline(1, color=MUTED, ls=':', lw=1); ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(PAIR_LABEL[kind.split(':')[1]], fontsize=9)
    axes[0].set_ylabel('retained R'); axes[0].legend()
    fig.tight_layout()
    fig.savefig(figs / 'gates.png')
    plt.close(fig)


def fig_attention(att: dict, figs: Path, kinds: list[str]) -> None:
    layers = (4, 8, 12, 16, 20)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [('target', 'before', 'mean', 'target query -> all raster-earlier entity tokens (head mean)'),
              ('target', 'target', 'mean', 'target query -> its own tokens (head mean)'),
              ('ans_row', 'target', 'max', 'row-digit query -> target tokens (max over heads)')]
    cmap = plt.get_cmap('tab10')
    for ax, (q, g, stat, title) in zip(axes, panels):
        for i, kind in enumerate(kinds):
            for var, ls in (('std', '-'), ('swap', '--')):
                ys = [att.get(f'{kind}|{var}|L{l}|{q}|{g}|{stat}', np.nan) for l in layers]
                ax.plot(layers, ys, ls, marker='o', color=cmap(i), label=PAIR_LABEL[kind.split(':')[1]] if var == 'std' else None)
        ax.set_xticks(layers); ax.set_xlabel('full-attention layer'); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel('attention probability mass')
    axes[0].legend(fontsize=7, title='solid std, dashed swap', title_fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / 'attention.png')
    plt.close(fig)


def analyze(args) -> None:
    out, figs = args.output, args.figures
    figs.mkdir(parents=True, exist_ok=True)
    items = sa.load_items(out)
    pairs = [p for p in sa.PAIRS if any(it.get('pair_name') == p for it in items.values())]
    summary = {}
    beh = behaviour(out, items, args.odd_entity)
    summary['behaviour'] = beh
    fig_examples(out, items, figs, pairs)
    fig_atlas(beh, figs, pairs)
    fig_setsize(beh, figs, pairs)
    if (out / 'capture').exists():
        summary['neurons'] = neurons(out, args.blocks)
        summary['salience'] = salience(out, ['vit6', 'vit12', 'vit18', 'vit24', 'dec0', 'dec4', 'dec8', 'dec12', 'dec16'])
        fig_neurons(summary['neurons'], summary['salience'], beh, figs, args.blocks)
        summary['popout'] = popout(out, args.blocks)
        fig_popout(summary['popout'], beh, figs)
        sel = json.loads((out / 'neurons.json').read_text())
        fig_onoff(out, sel, figs)
        fig_maps(out, sel, figs)
    if (out / 'edit.jsonl').exists():
        summary['edits'] = edits(out)
        if (out / 'edit_k256.jsonl').exists():
            summary['edits_k256'] = edits(out, '_k256')
            fig_edits(summary['edits_k256'], figs, '_k256', 'blocks 3-18, 256 neurons per block')
        fig_edits(summary['edits'], figs)
    if (out / 'mech.jsonl').exists():
        summary['mech'] = mech(out)
        kinds = sorted({k.split('|')[0] for k in summary['mech']['R']})
        fig_mech_layers(summary['mech'], figs, kinds)
        fig_gates(summary['mech'], figs, kinds)
        summary['attention'] = attention(out)
        fig_attention(summary['attention'], figs, kinds)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('[analyze] done', flush=True)


# ----------------------------------------------------------------------------- entry

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['select', 'analyze'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--odd-entity', type=Path, default=None)
    p.add_argument('--top-k', type=int, default=32)
    p.add_argument('--tag', default='')
    p.add_argument('--blocks', type=int, nargs='+', default=[3, 6, 9, 12, 15, 18])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    if args.stage == 'select':
        sel = stage_select(args.output, args.top_k, args.blocks, args.tag)
        print(f'[select] {len(sel)} pairs', flush=True)
        return
    analyze(args)


if __name__ == '__main__':
    main()
