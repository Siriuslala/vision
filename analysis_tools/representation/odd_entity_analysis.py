"""Experiment 2.6 CPU analysis: behaviour tables, oddness-vs-feature split, raster dependence,
text-side majority readout, boundary maps, patching curves, and figures.

Colour conventions (repeated in the document captions):
  blue   = std images (odd entity carries the change X), orange = swap images (odd entity is the template T);
  kinds: red = red, yellow = dE20, pink = dE10, blue = gap 16, aqua = gap 2, violet = natural crops;
  heatmaps use viridis (continuous scores) with a labelled colorbar.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.representation import odd_entity as oe

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 130,
                     'savefig.bbox': 'tight', 'legend.frameon': False, 'lines.linewidth': 2})
KIND_COLOR = {'red': RED, 'color20': YELLOW, 'color10': MAGENTA, 'gap16': BLUE, 'gap2': AQUA, 'natural': VIOLET}
KIND_LABEL = {'red': 'red', 'color20': 'colour dE20', 'color10': 'colour dE10', 'gap16': 'gap 16 px',
              'gap2': 'gap 2 px', 'natural': 'natural crops'}


def site_order(s: str) -> tuple:
    return (0 if s.startswith('vit') else 1, int(s[3:]))


def site_label(s: str) -> str:
    return f'V{s[3:]}' if s.startswith('vit') else (f'D{s[3:]}' if s != 'dec0' else 'D0')


def wilson_row(k: int, n: int) -> dict:
    p, lo, hi = oe.wilson(k, n)
    return {'k': k, 'n': n, 'p': p, 'lo': lo, 'hi': hi}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ----------------------------------------------------------------------------- behaviour

NONE_RE = re.compile(r'no (?:visible )?differences?|(?:all|are) (?:\w+ )*identical|none of the', re.I)
MAJORITY_RE = re.compile(r'\b(all (?:the )?other|all (?:the )?remaining|the (?:other|remaining) \w+ (?:are|have|look)|'
                         r'most (?:of the )?\w*|the rest|majority|every other|others (?:are|have|look))', re.I)
FEATURE_RE = re.compile(r'\b(gap|break|broken|missing|incomplete|open|red|colou?r|brown|dark|lighter|complete|'
                        r'intact|closed|black)\b', re.I)


def code_reasoning(text: str) -> dict:
    """Rule-based coding of think-then-answer text: does it state a majority template, name a feature?"""
    return {'majority': bool(MAJORITY_RE.search(text)), 'feature': bool(FEATURE_RE.search(text)),
            'n_chars': len(text)}


def behaviour(output: Path, items: dict) -> dict:
    recs = read_jsonl(output / 'results_behave.jsonl')
    cells = defaultdict(list)
    paired = defaultdict(dict)
    rows = defaultdict(lambda: defaultdict(list))
    for r in recs:
        it = items[r['item']]
        kind = 'natural' if it['group'] == 'natural' else it['kind']
        key = (it['group'], it['grid'], kind, it['variant'], r['mode'])
        cells[key].append(r)
        paired[(it['group'], it['grid'], kind, r['mode'])].setdefault(it['pair'], {})[it['variant']] = r['correct']
        if it['group'] == 'core' and r['mode'] == 'direct':
            rows[(kind, it['variant'])][it['target'][0]].append(r['correct'])
    table = []
    for key, rs in sorted(cells.items()):
        k = sum(r['correct'] for r in rs)
        wrong = [r for r in rs if not r['correct']]
        table.append({'group': key[0], 'grid': key[1], 'kind': key[2], 'variant': key[3], 'mode': key[4],
                      **wilson_row(k, len(rs)), 'invalid': sum(not r['valid'] for r in rs),
                      'truncated': sum(r['finish_reason'] != 'eos' for r in rs),
                      'claims_none': sum(r['pred'] is None and bool(NONE_RE.search(r['text'])) for r in rs),
                      'adjacent_of_wrong': sum(bool(r.get('adjacent')) for r in wrong), 'n_wrong': len(wrong)})
    tests = []
    for key, d in sorted(paired.items()):
        both = [v for v in d.values() if 'std' in v and 'swap' in v]
        b = sum(v['std'] and not v['swap'] for v in both)
        c = sum(v['swap'] and not v['std'] for v in both)
        tests.append({'group': key[0], 'grid': key[1], 'kind': key[2], 'mode': key[3], 'n_pairs': len(both),
                      'std_only': b, 'swap_only': c, 'p_mcnemar': mcnemar_exact(b, c)})
    coding = defaultdict(list)
    for r in recs:
        if r['mode'] == 'think':
            it = items[r['item']]
            kind = 'natural' if it['group'] == 'natural' else it['kind']
            coding[(kind, it['variant'])].append({**code_reasoning(r['text']), 'correct': r['correct']})
    think_coding = []
    for (kind, v), cs in sorted(coding.items()):
        maj = [c for c in cs if c['majority']]
        think_coding.append({'kind': kind, 'variant': v, 'n': len(cs), 'majority': len(maj),
                             'feature': sum(c['feature'] for c in cs),
                             'acc_with_majority': wilson_row(sum(c['correct'] for c in maj), len(maj)),
                             'acc_without': wilson_row(sum(c['correct'] for c in cs if not c['majority']),
                                                       len(cs) - len(maj))})
    by_row = {f'{k}|{v}': {str(r): wilson_row(sum(x), len(x)) for r, x in sorted(d.items())}
              for (k, v), d in rows.items()}
    return {'table': table, 'std_vs_swap': tests, 'by_target_row': by_row, 'think_coding': think_coding}


# ----------------------------------------------------------------------------- representation

def load_capture(output: Path, name: str):
    path = output / 'capture' / f'{name}.npz'
    return np.load(path, allow_pickle=False) if path.exists() else None


def paired_entities(z, items: dict, site: str):
    """Standardised entity states of matched std/swap images: (H_std, H_swap, targets, pairs)."""
    ids = [str(x) for x in z['ent_ids']]
    E = z[f'E_{site}'].astype(np.float32)
    by = defaultdict(dict)
    for i, iid in enumerate(ids):
        it = items[iid]
        if it['variant'] in oe.ODD_VARIANTS:
            by[it['pair']][it['variant']] = i
    pairs = sorted(p for p, d in by.items() if len(d) == 2)
    g = items[ids[0]]['grid']
    idx_std = [by[p]['std'] for p in pairs]
    idx_swap = [by[p]['swap'] for p in pairs]
    targets = np.array([items[ids[by[p]['std']]]['target'][0] * g + items[ids[by[p]['std']]]['target'][1]
                        for p in pairs])
    allv = E[idx_std + idx_swap].reshape(-1, E.shape[-1])
    mu, sd = allv.mean(0), allv.std(0) + 1e-6
    H = (E - mu) / sd
    return H[idx_std], H[idx_swap], targets, pairs, (mu, sd)


def split_metrics(Hs, Hw, targets, folds: int = 4, n_boot: int = 300, seed: int = 0) -> dict:
    n = len(targets)
    Ds = np.stack([oe.contrast(Hs[i], targets[i]) for i in range(n)])
    Dw = np.stack([oe.contrast(Hw[i], targets[i]) for i in range(n)])
    full = oe.oddness_split(Ds, Dw)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        sp = oe.oddness_split(Ds[b], Dw[b])
        boots.append((sp['cos'], sp['rho']))
    boots = np.array(boots)
    # cross-validated top-1 with the oddness and the feature directions
    order = rng.permutation(n)
    hit = {k: np.zeros(n, bool) for k in ('odd_std', 'odd_swap', 'feat_std', 'feat_swap')}
    marg = {k: np.zeros(n) for k in ('odd_std', 'odd_swap')}
    for f in range(folds):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        sp = oe.oddness_split(Ds[train], Dw[train])
        for i in test:
            hit['odd_std'][i] = oe.top1(Hs[i], sp['u_odd'], targets[i])
            hit['odd_swap'][i] = oe.top1(Hw[i], sp['u_odd'], targets[i])
            hit['feat_std'][i] = oe.top1(Hs[i], sp['u_feat'], targets[i])
            hit['feat_swap'][i] = oe.top1(Hw[i], -sp['u_feat'], targets[i])
            marg['odd_std'][i] = oe.margin(Hs[i], sp['u_odd'], targets[i])
            marg['odd_swap'][i] = oe.margin(Hw[i], sp['u_odd'], targets[i])
    out = {'n_pairs': n, 'cos': full['cos'], 'cos_lo': float(np.percentile(boots[:, 0], 2.5)),
           'cos_hi': float(np.percentile(boots[:, 0], 97.5)), 'rho': full['rho'],
           'rho_lo': float(np.percentile(boots[:, 1], 2.5)), 'rho_hi': float(np.percentile(boots[:, 1], 97.5)),
           'norm_odd': full['norm_odd'], 'norm_feat': full['norm_feat'],
           'outlier_std': float(np.mean([oe.outlier_top1(Hs[i], targets[i]) for i in range(n)])),
           'outlier_swap': float(np.mean([oe.outlier_top1(Hw[i], targets[i]) for i in range(n)]))}
    for k, v in hit.items():
        out[f'top1_{k}'] = float(v.mean())
    out['_hit_std'] = hit['odd_std'].tolist()
    out['_hit_swap'] = hit['odd_swap'].tolist()
    out['_margin_std'] = marg['odd_std'].tolist()
    out['_margin_swap'] = marg['odd_swap'].tolist()
    return out, full


def representation(output: Path, items: dict, seed: int) -> dict:
    res = {}
    dirs = {}
    for f in sorted((output / 'capture').glob('*.npz')):
        name = f.stem
        if name.startswith('boundary'):
            continue
        z = np.load(f)
        sites = sorted([k[2:] for k in z.files if k.startswith('E_')], key=site_order)
        res[name] = {}
        for s in sites:
            Hs, Hw, targets, pairs, stats = paired_entities(z, items, s)
            m, full = split_metrics(Hs, Hw, targets, seed=seed)
            m['target_rows'] = (targets // items[str(z['ent_ids'][0])]['grid']).tolist()
            res[name][s] = m
            if name.startswith('core'):
                dirs[(name, s)] = (full['u_odd'], full['u_feat'], stats)
        print(f'[analyze] representation {name}', flush=True)
    return res, dirs


def raster_summary(rep: dict, sites=('vit18', 'dec0', 'dec8', 'dec16')) -> dict:
    """Oddness margin and CV top-1 of the target by target row (6x6 core), with a row-1 vs rows 2-6 contrast."""
    out = {}
    for name, r in rep.items():
        if not name.startswith('core'):
            continue
        out[name] = {}
        for s in sites:
            rows = np.array(r[s]['target_rows'])
            d = {}
            for v in ('std', 'swap'):
                m, h = np.array(r[s][f'_margin_{v}']), np.array(r[s][f'_hit_{v}'])
                d[v] = {'margin_by_row': [float(m[rows == q].mean()) for q in range(6)],
                        'top1_by_row': [wilson_row(int(h[rows == q].sum()), int((rows == q).sum())) for q in range(6)],
                        'spearman_row_margin': float(np.corrcoef(np.argsort(np.argsort(rows)),
                                                                 np.argsort(np.argsort(m)))[0, 1])}
            out[name][s] = d
    return out


def text_readout(output: Path, items: dict) -> dict:
    """Majority readout at text positions: project std/swap states on the homX - homT direction.

    t = +1 means "like the all-X image", -1 "like the all-T image".  majority_index = (t_swap - t_std) / 2 is +1
    if the position encodes the majority feature and -1 if it encodes the odd entity's feature.
    """
    out = {}
    for f in sorted((output / 'capture').glob('core_*.npz')):
        z = np.load(f)
        ids = [str(x) for x in z['all_ids']]
        var = np.array([items[i]['variant'] for i in ids])
        out[f.stem] = {}
        for key in sorted([k for k in z.files if k.startswith('T_')], key=lambda k: int(k[5:])):
            T = z[key].astype(np.float32)
            row = {}
            for j, pos in enumerate(('img_last', 'q_last', 'last')):
                X = T[:, j]
                mu, sd = X.mean(0), X.std(0) + 1e-6
                X = (X - mu) / sd
                a, b = X[var == 'homX'].mean(0), X[var == 'homT'].mean(0)
                w = a - b
                mid, half = (a + b) @ w / 2, (a - b) @ w / 2
                t = {v: float(((X[var == v] @ w) - mid).mean() / half) for v in ('std', 'swap')}
                row[pos] = {**t, 'majority_index': (t['swap'] - t['std']) / 2, 'presence': (t['swap'] + t['std']) / 2}
            out[f.stem][key[2:]] = row
    return out


def boundary(output: Path, items: dict, dirs: dict) -> dict:
    out = {}
    for f in sorted((output / 'capture').glob('boundary_*.npz')):
        z = np.load(f)
        kind = f.stem.split('_', 2)[2]
        ids = [str(x) for x in z['ent_ids']]
        out[kind] = {}
        for key in sorted([k for k in z.files if k.startswith('E_')], key=lambda k: site_order(k[2:])):
            s = key[2:]
            if (f'core_g06_{kind}', s) not in dirs:
                continue
            u_odd, u_feat, (mu, sd) = dirs[(f'core_g06_{kind}', s)]
            H = (z[key].astype(np.float32) - mu) / sd
            maps = {'LR': [], 'TB': []}
            for i, iid in enumerate(ids):
                it = items[iid]
                sc = H[i] @ u_odd
                sc = (sc - sc.mean()) / (sc.std() + 1e-9)
                m = sc.reshape(6, 6)
                if not it['x_first']:            # put the X half first (left / top) for averaging
                    m = m[:, ::-1] if it['variant'] == 'bLR' else m[::-1, :]
                maps[it['variant'][1:]].append(m)
            res = {}
            for o, ms in maps.items():
                M = np.mean(ms, 0)
                prof = M.mean(0) if o == 'LR' else M.mean(1)      # along the axis crossing the boundary
                res[o] = {'map': M.tolist(), 'profile': prof.tolist(),
                          'boundary_minus_interior': float((prof[2] + prof[3]) / 2 - (prof[0] + prof[5]) / 2),
                          # same-side comparison removes feature leakage and image-edge effects:
                          # boundary line (2 X side, 3 T side) minus the adjacent non-edge line (1, 4)
                          'local_index': float(((prof[2] - prof[1]) + (prof[3] - prof[4])) / 2)}
            out[kind][s] = res
    return out


def patching(output: Path, n_boot: int, seed: int) -> dict:
    path = output / 'patch.jsonl'
    if not path.exists():
        return {}
    rows = [r for r in read_jsonl(path) if 'ptype' in r]
    ident = [r for r in read_jsonl(path) if 'identity_s' in r]
    groups = defaultdict(list)
    for r in rows:
        base = r['base_same'] if r['ptype'] == 'ctx_same' else r['base_clean_ctx']
        groups[(r['kind'], r['variant'], r['ptype'], r['group'], r['site'])].append(
            (r['s'] - base, r['base_clean'] - base))
    rng = np.random.default_rng(seed)
    out = []
    for key, v in sorted(groups.items()):
        a = np.array(v)
        num, den = a[:, 0], a[:, 1]
        keep = den > 1.0                          # pairs whose clean answer beats the donor baseline by > 1 nat
        num, den = num[keep], den[keep]
        if len(num) == 0:
            continue
        ratio = num.sum() / den.sum()
        bs = []
        for _ in range(n_boot):
            b = rng.integers(0, len(num), len(num))
            bs.append(num[b].sum() / den[b].sum())
        out.append({'kind': key[0], 'variant': key[1], 'ptype': key[2], 'group': key[3], 'site': key[4],
                    'n': int(len(num)), 'n_dropped': int((~keep).sum()), 'retained': float(ratio),
                    'lo': float(np.percentile(bs, 2.5)), 'hi': float(np.percentile(bs, 97.5)),
                    'median': float(np.median(num / den)), 'mean_gap': float(den.mean())})
    max_dev = max([abs(r['identity_s'] - r['base_clean']) for r in ident], default=None)
    return {'rows': out, 'identity_max_dev': max_dev}


# ----------------------------------------------------------------------------- figures

def fig_examples(output: Path, items: dict, figs: Path) -> None:
    pick = [('core', 'red', 0, 'std', 'std (red)'), ('core', 'red', 0, 'swap', 'swap (red)'),
            ('core', 'gap2', 1, 'std', 'std (gap 2 px)'), ('core', 'gap2', 1, 'swap', 'swap (gap 2 px)'),
            ('core', 'color20', 2, 'homX', 'homX donor (dE20)'), ('hetero', 'gap16', 0, 'std', 'hetero (gap 16)'),
            ('boundary', 'red', 0, 'bLR', 'boundary LR (red)'), ('natural', None, 0, 'std', 'natural std')]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    for ax, (g, k, p, v, title) in zip(axes.flat, pick):
        it = next(i for i in items.values() if i['group'] == g and (k is None or i['kind'] == k) and i['pair'] == p
                  and i['variant'] == v)
        ax.imshow(Image.open(output / it['path']))
        if it.get('target') is not None:
            x0, y0, x1, y1 = it['meta']['instances'][it['target'][0] * it['grid'] + it['target'][1]]['cell_box']
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ORANGE, lw=2, ls='--'))
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.suptitle('Input examples (orange dashed box = odd entity, display only)', fontsize=11)
    fig.savefig(figs / 'examples.png')
    plt.close(fig)


def fig_behaviour(beh: dict, figs: Path) -> None:
    T = {(r['group'], r['grid'], r['kind'], r['variant'], r['mode']): r for r in beh['table']}
    panels = [('6x6, direct', [('core', 6, k, 'direct') for k in oe.KINDS] + [('natural', 3, 'natural', 'direct')]),
              ('6x6, think-then-answer', [('core', 6, k, 'think') for k in oe.KINDS] + [('natural', 3, 'natural', 'think')]),
              ('grid size and heterogeneity, direct',
               [(g, s, k, 'direct') for k in ('red', 'color20', 'gap16', 'gap2')
                for g, s in (('grid', 3), ('core', 6), ('grid', 10), ('hetero', 6))])]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10.5))
    for ax, (title, keys) in zip(axes, panels):
        xs, labels = [], []
        for i, (g, s, k, m) in enumerate(keys):
            for j, (v, col) in enumerate((('std', BLUE), ('swap', ORANGE))):
                r = T.get((g, s, k, v, m))
                if r is None:
                    continue
                x = i + (j - 0.5) * 0.36
                ax.bar(x, r['p'], 0.34, color=col, label={'std': 'std: odd entity carries X',
                                                         'swap': 'swap: odd entity is the template'}[v] if i == 0 else None)
                ax.errorbar(x, r['p'], [[max(0, r['p'] - r['lo'])], [max(0, r['hi'] - r['p'])]], color=INK, lw=1, capsize=2)
            labels.append(f'{KIND_LABEL[k]}\n{g} {s}x{s}' if 'grid' in title else KIND_LABEL[k])
            xs.append(i)
        ax.set_xticks(xs, labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('row/column accuracy')
        ax.set_title(title, fontsize=10, loc='left')
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('Locating the odd entity: std vs swap (Wilson 95% CI)', fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / 'behaviour.png')
    plt.close(fig)


def fig_split(rep: dict, figs: Path) -> None:
    names = [f'core_g06_{k}' for k in oe.KINDS if f'core_g06_{k}' in rep] + \
            (['natural_g03_natural'] if 'natural_g03_natural' in rep else [])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for name in names:
        kind = name.split('_')[-1]
        sites = sorted(rep[name], key=site_order)
        x = np.arange(len(sites))
        r = rep[name]
        col = KIND_COLOR[kind]
        axes[0].plot(x, [r[s]['cos'] for s in sites], color=col, label=KIND_LABEL[kind], marker='o', ms=4)
        axes[0].fill_between(x, [r[s]['cos_lo'] for s in sites], [r[s]['cos_hi'] for s in sites], color=col, alpha=.15, lw=0)
        axes[1].plot(x, [r[s]['top1_odd_std'] for s in sites], color=col, marker='o', ms=4)
        axes[1].plot(x, [r[s]['top1_odd_swap'] for s in sites], color=col, marker='s', ms=4, ls='--')
        axes[2].plot(x, [r[s]['outlier_std'] for s in sites], color=col, marker='o', ms=4)
        axes[2].plot(x, [r[s]['outlier_swap'] for s in sites], color=col, marker='s', ms=4, ls='--')
    for ax in axes:
        ax.set_xticks(x, [site_label(s) for s in sites], rotation=60, fontsize=8)
        ax.axvline(9.5, color=MUTED, lw=1, ls=':')
        # V27 output: 97% of the variance sits in 5 massive-activation dimensions (removed by the merger norm)
        ax.axvspan(8.5, 9.5, color=GRID, alpha=.6, lw=0)
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel('cos(mean D_std, mean D_swap)')
    axes[0].set_title('+1 = oddness code, -1 = feature code', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('CV top-1 of the oddness direction')
    axes[1].set_title('circle/solid = std, square/dashed = swap', fontsize=10)
    axes[2].set_ylabel('top-1 of the farthest-from-mean entity')
    axes[2].set_title('template-free outlier detector', fontsize=10)
    for ax in axes[1:]:
        ax.set_ylim(0, 1.05)
        ax.axhline(1 / 36, color=MUTED, lw=1, ls=':')
    fig.suptitle('Odd entity states along ViT (V) and decoder (D) sites, 6x6 grids (natural crops: 3x3)', fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / 'oddness_vs_feature.png')
    plt.close(fig)


def fig_raster(rep: dict, figs: Path) -> None:
    kinds = [k for k in ('red', 'color20', 'gap16', 'gap2') if f'core_g06_{k}' in rep]
    fig, axes = plt.subplots(2, len(kinds), figsize=(4.2 * len(kinds), 6.4), squeeze=False)
    for j, k in enumerate(kinds):
        r = rep[f'core_g06_{k}']
        sites = sorted(r, key=site_order)
        for i, v in enumerate(('std', 'swap')):
            M = np.zeros((6, len(sites)))
            for c, s in enumerate(sites):
                rows = np.array(r[s]['target_rows'])
                m = np.array(r[s][f'_margin_{v}'])
                M[:, c] = [m[rows == q].mean() for q in range(6)]
            ax = axes[i, j]
            im = ax.imshow(M, aspect='auto', cmap='viridis')
            ax.set_yticks(range(6), [str(q + 1) for q in range(6)])
            ax.set_xticks(range(len(sites)), [site_label(s) for s in sites], rotation=70, fontsize=7)
            ax.set_title(f'{KIND_LABEL[k]}, {v}', fontsize=10)
            ax.grid(False)
            if j == 0:
                ax.set_ylabel('target row (raster order)')
            fig.colorbar(im, ax=ax, label='oddness margin (SD of others)', shrink=.85)
    fig.suptitle('Oddness margin of the target by target row (CV oddness direction)', fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / 'raster_margin.png')
    plt.close(fig)


def fig_patch(pat: dict, figs: Path) -> None:
    rows = pat.get('rows', [])
    if not rows:
        return
    kinds = [k for k in ('red', 'color20', 'gap16', 'gap2') if any(r['kind'] == k for r in rows)]
    curves = [('ctx_same', 'all', INK, 'context -> target feature (all 35)'),
              ('ctx_same', 'nbr', BLUE, 'context -> target feature (8 neighbours)'),
              ('ctx_same', 'far', AQUA, 'context -> target feature (far)'),
              ('ctx_same', 'before', YELLOW, 'context -> target feature (raster before)'),
              ('ctx_same', 'after', MAGENTA, 'context -> target feature (raster after)'),
              ('ctx_clean', 'all', VIOLET, 'context -> no-odd image (all 35)'),
              ('tgt', 'target', RED, 'target -> majority template')]
    fig, axes = plt.subplots(2, len(kinds), figsize=(4.4 * len(kinds), 7.4), squeeze=False, sharey=True)
    for j, k in enumerate(kinds):
        for i, v in enumerate(('std', 'swap')):
            ax = axes[i, j]
            sub = [r for r in rows if r['kind'] == k and r['variant'] == v]
            sites = sorted({r['site'] for r in sub}, key=site_order)
            x = np.arange(len(sites))
            n = 0
            for pt, g, col, lab in curves:
                d = {r['site']: r for r in sub if r['ptype'] == pt and r['group'] == g}
                if not d:
                    continue
                n = max(n, max(r['n'] for r in d.values()))
                ax.plot(x, [d[s]['retained'] if s in d else np.nan for s in sites], color=col, marker='o', ms=3,
                        lw=1.6, label=lab if (i == 0 and j == 0) else None)
            ax.axhline(0, color=MUTED, lw=1)
            ax.axhline(1, color=MUTED, lw=1, ls=':')
            ax.axvline(9.5, color=MUTED, lw=1, ls=':')
            ax.set_xticks(x, [site_label(s) for s in sites], rotation=70, fontsize=7)
            ax.set_title(f'{KIND_LABEL[k]}, {v} (n={n})', fontsize=10)
            if j == 0:
                ax.set_ylabel('retained answer score')
    fig.legend(loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Context replacement: share of log p(correct row, column) kept, by patch site', fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / 'patch_curves.png')
    plt.close(fig)


def fig_raster_patch(pat: dict, figs: Path, sites=('vit1', 'dec0', 'dec4')) -> None:
    """Retained answer score when the entities before vs after the target (raster order) take the target's feature."""
    rows = pat.get('rows', [])
    if not rows:
        return
    conds = [(k, v) for k in ('red', 'color20', 'gap16', 'gap2') for v in ('std', 'swap')]
    fig, axes = plt.subplots(1, len(sites), figsize=(5.2 * len(sites), 4.0), sharey=True)
    for ax, site in zip(axes, sites):
        for j, (grp, col) in enumerate((('before', YELLOW), ('after', MAGENTA))):
            for i, (k, v) in enumerate(conds):
                r = next((r for r in rows if r['kind'] == k and r['variant'] == v and r['ptype'] == 'ctx_same'
                          and r['group'] == grp and r['site'] == site), None)
                if r is None:
                    continue
                x = i + (j - .5) * .38
                ax.bar(x, r['retained'], .36, color=col, label=f'raster {grp} replaced' if i == 0 else None)
                ax.errorbar(x, r['retained'], [[max(0, r['retained'] - r['lo'])], [max(0, r['hi'] - r['retained'])]],
                            color=INK, lw=1, capsize=2)
        ax.axhline(0, color=MUTED, lw=1)
        ax.axhline(1, color=MUTED, lw=1, ls=':')
        short = {'red': 'red', 'color20': 'dE20', 'gap16': 'gap16', 'gap2': 'gap2'}
        ax.set_xticks(range(len(conds)), [f'{short[k]}\n{v}' for k, v in conds], fontsize=8)
        ax.set_title(f'patch site {site_label(site)}', fontsize=10)
    axes[0].set_ylabel('retained answer score')
    axes[1].legend(fontsize=9, loc='lower center')
    fig.suptitle('Entities before vs after the odd entity in raster order, set to its feature (bootstrap 95% CI)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / 'raster_patch.png')
    plt.close(fig)


def fig_boundary(bd: dict, figs: Path, sites=('vit9', 'vit27', 'dec8', 'dec16')) -> None:
    kinds = [k for k in ('red', 'color20', 'gap16', 'gap2') if k in bd]
    if not kinds:
        return
    fig, axes = plt.subplots(len(kinds), len(sites), figsize=(3.2 * len(sites), 3.0 * len(kinds)), squeeze=False)
    for i, k in enumerate(kinds):
        for j, s in enumerate(sites):
            ax = axes[i, j]
            if s not in bd[k]:
                ax.axis('off')
                continue
            M = np.array(bd[k][s]['LR']['map'])
            im = ax.imshow(M, cmap='viridis', vmin=-1.5, vmax=1.5)
            ax.axvline(2.5, color='white', lw=1.5, ls='--')
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f'{KIND_LABEL[k]}, {site_label(s)}', fontsize=9)
            fig.colorbar(im, ax=ax, shrink=.8, label='oddness score (z within image)')
    fig.suptitle('Half/half images (X on the left of the white dashed line): oddness score per entity', fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / 'boundary_maps.png')
    plt.close(fig)


def fig_text(txt: dict, figs: Path) -> None:
    if not txt:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, pos in zip(axes, ('img_last', 'q_last', 'last')):
        for name, d in txt.items():
            kind = name.split('_')[-1]
            sites = sorted(d, key=lambda s: int(s[3:]))
            ax.plot([int(s[3:]) for s in sites], [d[s][pos]['majority_index'] for s in sites], color=KIND_COLOR[kind],
                    marker='o', ms=3, label=KIND_LABEL[kind])
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel('decoder layer')
        ax.set_title({'img_last': 'last visual token', 'q_last': 'last question token',
                      'last': 'last prompt token'}[pos], fontsize=10)
    axes[0].set_ylabel('majority index\n(+1 majority feature, -1 odd feature)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Which feature do text-side states encode? Projection on the all-X minus all-T direction', fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / 'text_majority.png')
    plt.close(fig)


def fig_maps(output: Path, items: dict, dirs: dict, figs: Path, kind='gap2', pair=0,
             sites=('vit6', 'vit18', 'vit27', 'dec8', 'dec16')) -> None:
    z = load_capture(output, f'core_g06_{kind}')
    if z is None:
        return
    ids = [str(x) for x in z['ent_ids']]
    fig, axes = plt.subplots(2, len(sites) + 1, figsize=(2.9 * (len(sites) + 1), 5.8))
    for i, v in enumerate(('std', 'swap')):
        j0 = next(j for j, iid in enumerate(ids) if items[iid]['pair'] == pair and items[iid]['variant'] == v)
        it = items[ids[j0]]
        axes[i, 0].imshow(Image.open(output / it['path']))
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'{v} input', fontsize=9)
        for j, s in enumerate(sites):
            u_odd, _, (mu, sd) = dirs[(f'core_g06_{kind}', s)]
            sc = ((z[f'E_{s}'][j0].astype(np.float32) - mu) / sd) @ u_odd
            sc = (sc - sc.mean()) / sc.std()
            ax = axes[i, j + 1]
            im = ax.imshow(sc.reshape(6, 6), cmap='viridis', vmin=-2, vmax=5)
            r, c = it['target']
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec=ORANGE, lw=2))
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            ax.set_title(site_label(s), fontsize=9)
    fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), shrink=.8, label='oddness score (z within image)')
    fig.suptitle(f'Per-entity oddness score, {KIND_LABEL[kind]} (orange box = odd entity; '
                 'oddness direction fitted on all pairs, illustration only)', fontsize=10)
    fig.savefig(figs / f'oddness_maps_{kind}.png')
    plt.close(fig)


# ----------------------------------------------------------------------------- entry

def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if not str(k).startswith('_')}
    if isinstance(d, list):
        return [_strip(v) for v in d]
    return d


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-boot', type=int, default=1000)
    args = p.parse_args(argv)
    args.figures.mkdir(parents=True, exist_ok=True)
    items = oe.load_items(args.output)
    summary = {}
    beh = behaviour(args.output, items)
    summary['behaviour'] = beh
    fig_examples(args.output, items, args.figures)
    fig_behaviour(beh, args.figures)
    if (args.output / 'capture').exists():
        rep, dirs = representation(args.output, items, args.seed)
        summary['representation'] = rep
        summary['raster'] = raster_summary(rep)
        summary['text'] = text_readout(args.output, items)
        summary['boundary'] = boundary(args.output, items, dirs)
        fig_split(rep, args.figures)
        fig_raster(rep, args.figures)
        fig_text(summary['text'], args.figures)
        fig_boundary(summary['boundary'], args.figures)
        for k in ('gap2', 'red', 'color20'):
            fig_maps(args.output, items, dirs, args.figures, kind=k)
    summary['patch'] = patching(args.output, args.n_boot, args.seed)
    fig_patch(summary['patch'], args.figures)
    fig_raster_patch(summary['patch'], args.figures)
    spot = args.output / 'spotcheck.json'
    if spot.exists():
        s = json.loads(spot.read_text())
        summary['spotcheck'] = {k: s[k] for k in ('n', 'same_answer', 'same_text')}
    (args.output / 'summary.json').write_text(json.dumps(_strip(summary), indent=1))
    print('[analyze] done', flush=True)


if __name__ == '__main__':
    main()
