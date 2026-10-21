"""Figures of experiment 3.7 (ViT position and layout). Reads the run directory and results.json."""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import vit_layout as vl
from analysis_tools.position.vit_layout_run import reread_cases

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5195', '#5c5c5c']
MODELS = {'q35': ('Qwen3.5-9B', 'model.safetensors-00004-of-00004.safetensors'),
          'q3vl': ('Qwen3-VL-8B-Instruct', 'model-00004-of-00004.safetensors')}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'legend.frameon': False})


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print('figure', name)


def load_table(tag: str) -> torch.Tensor:
    name, shard = MODELS[tag]
    snap = glob.glob(os.path.join(os.environ['HF_HUB_CACHE'], f'models--Qwen--{name}', 'snapshots', '*'))[0]
    with safe_open(os.path.join(snap, shard), 'pt') as f:
        return f.get_tensor('model.visual.pos_embed.weight').float()


def pca_rgb(x: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    xc = x - x.mean(0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    p = u[:, :k] * s[:k]
    p = (p - p.min(0)) / (p.max(0) - p.min(0))
    return p, s ** 2 / (s ** 2).sum()


def table_stats(t: torch.Tensor) -> dict:
    x = t.numpy()
    side = 48
    norms = np.linalg.norm(x, axis=1)
    xn = x / norms[:, None]
    cos = xn @ xn.T
    r, c = np.divmod(np.arange(side * side), side)
    dist = np.sqrt((r[:, None] - r[None]) ** 2 + (c[:, None] - c[None]) ** 2)
    bins = np.arange(0, 68, 2)
    prof = [float(cos[(dist >= a) & (dist < a + 2)].mean()) for a in bins]
    _, var = pca_rgb(x)
    # 2D DCT energy of the table along the grid axes: share in the lowest 8x8 frequencies.
    g = x.reshape(side, side, -1)
    k = np.arange(side)
    dct = np.cos(np.pi * (k[:, None] + .5) * k[None] / side)
    coef = np.einsum('ia,jb,ijd->abd', dct, dct, g - g.mean((0, 1)))
    energy = (coef ** 2).sum(-1)
    return {'norm_mean': float(norms.mean()), 'norm_min': float(norms.min()), 'norm_max': float(norms.max()),
            'pca_var_top10': var[:10].tolist(), 'dct_low8_share': float(energy[:8, :8].sum() / energy.sum()),
            'cos_vs_dist': {'bins': bins.tolist(), 'mean_cos': prof},
            'mean_vector_norm': float(np.linalg.norm(x.mean(0)))}


def fig_table(out: Path, run: Path) -> dict:
    tables = {tag: load_table(tag) for tag in MODELS}
    stats = {tag: table_stats(t) for tag, t in tables.items()}
    a, b = tables['q35'], tables['q3vl']
    stats['cross'] = {'entry_cos_mean': float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean()),
                      'rel_diff': float((a - b).norm() / a.norm())}
    fig, ax = plt.subplots(2, 5, figsize=(14, 5.8))
    refs = [(0, 0), (24, 24), (12, 36), (47, 10)]
    for row, tag in enumerate(MODELS):
        x = tables[tag].numpy()
        rgb, var = pca_rgb(x)
        ax[row, 0].imshow(rgb.reshape(48, 48, 3))
        ax[row, 0].set_title(f'{MODELS[tag][0]}\nPCA 1-3 as RGB ({100 * var[:3].sum():.0f}% var)')
        im = ax[row, 1].imshow(np.linalg.norm(x, axis=1).reshape(48, 48), cmap='Blues')
        ax[row, 1].set_title('entry norm')
        fig.colorbar(im, ax=ax[row, 1], fraction=.046)
        xn = x / np.linalg.norm(x, axis=1, keepdims=True)
        for j, (rr, cc) in enumerate(refs[:2]):
            im = ax[row, 2 + j].imshow((xn @ xn[rr * 48 + cc]).reshape(48, 48), cmap='RdBu_r', vmin=-1, vmax=1)
            ax[row, 2 + j].plot(cc, rr, 'k+', ms=10)
            ax[row, 2 + j].set_title(f'cosine to entry ({rr},{cc})')
        fig.colorbar(im, ax=ax[row, 3], fraction=.046)
        for a_ in ax[row, :4]:
            a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
        s = stats[tag]['cos_vs_dist']
        ax[row, 4].plot(np.array(s['bins']) + 1, s['mean_cos'], color=SERIES[0], lw=2)
        ax[row, 4].set_xlabel('grid distance (table cells)')
        ax[row, 4].set_ylabel('mean cosine')
        ax[row, 4].set_title('similarity vs distance')
    save(fig, out, 'pe_table.png')
    return stats


def visual_tensors(tag: str) -> dict:
    name, _ = MODELS[tag]
    snap = glob.glob(os.path.join(os.environ['HF_HUB_CACHE'], f'models--Qwen--{name}', 'snapshots', '*'))[0]
    index = json.loads(Path(snap, 'model.safetensors.index.json').read_text())['weight_map']
    out = {}
    for shard in sorted({v for k, v in index.items() if k.startswith('model.visual.')}):
        with safe_open(os.path.join(snap, shard), 'pt') as f:
            for k in f.keys():
                if k.startswith('model.visual.') and 'deepstack' not in k:
                    out[k[len('model.visual.'):]] = f.get_tensor(k).float()
    return out


def fig_weights(out: Path) -> dict:
    """Qwen3.5-9B vs Qwen3-VL-8B ViT weights: cosine and norm ratio per tensor."""
    a, b = visual_tensors('q35'), visual_tensors('q3vl')
    rows = []
    for k in a:
        if k in b and a[k].shape == b[k].shape:
            x, y = a[k].flatten(), b[k].flatten()
            rows.append({'name': k, 'cos': float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
                         'norm_ratio': float(x.norm() / y.norm())})
    groups = {'pos_embed': [r for r in rows if r['name'] == 'pos_embed.weight'],
              'patch_embed': [r for r in rows if r['name'].startswith('patch_embed')],
              'blocks (weight matrices)': [r for r in rows if r['name'].startswith('blocks') and r['name'].endswith('weight')
                                           and 'norm' not in r['name']],
              'blocks (norms, biases)': [r for r in rows if r['name'].startswith('blocks') and
                                         ('norm' in r['name'] or r['name'].endswith('bias'))],
              'merger': [r for r in rows if r['name'].startswith('merger')]}
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for j, (g, rs) in enumerate(groups.items()):
        ax[0].scatter([r['cos'] for r in rs], [r['norm_ratio'] for r in rs], s=14 if g != 'pos_embed' else 60,
                      color=SERIES[j], label=g, zorder=3 if g == 'pos_embed' else 2)
    ax[0].set_xlabel('cosine(Qwen3.5, Qwen3-VL)'); ax[0].set_ylabel('norm ratio Qwen3.5 / Qwen3-VL')
    ax[0].axhline(1, color='#999999', lw=.8); ax[0].legend(fontsize=7); ax[0].set_title('ViT tensors of the two checkpoints')
    blk = [r for r in groups['blocks (weight matrices)']]
    depth = [int(r['name'].split('.')[1]) + 1 for r in blk]
    ax[1].scatter(depth, [r['cos'] for r in blk], s=10, color=SERIES[2])
    ax[1].set_xlabel('ViT block'); ax[1].set_ylabel('cosine'); ax[1].set_title('weight-matrix cosine by depth')
    save(fig, out, 'weights_q35_vs_q3vl.png')
    return {g: {'n': len(rs), 'cos_median': float(np.median([r['cos'] for r in rs])) if rs else None,
                'norm_ratio_median': float(np.median([r['norm_ratio'] for r in rs])) if rs else None,
                'norm_ratio_min': float(min(r['norm_ratio'] for r in rs)) if rs else None} for g, rs in groups.items()}


def fig_norms(out: Path, run: Path) -> dict:
    res = {}
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for tag, ls in (('q35', '-'), ('q3vl', '--')):
        p = run / 'norms' / f'{tag}_records.jsonl'
        if not p.exists():
            continue
        recs = [json.loads(l) for l in p.read_text().splitlines()]
        disc = [r for r in recs if r['id'].startswith('disc')]
        canv = sorted({r['grid'][0] * 16 for r in disc})
        ratio = [np.mean([r['pe_content_ratio'] for r in disc if r['grid'][0] * 16 == c]) for c in canv]
        nat = [r for r in recs if r['id'].startswith('natural')]
        res[tag] = {'disc_ratio_by_canvas': dict(zip(map(str, canv), map(float, ratio))),
                    'natural_ratio': float(np.mean([r['pe_content_ratio'] for r in nat])),
                    'natural_cos': float(np.mean([r['pe_content_cos'] for r in nat])),
                    'natural_pe_norm': float(np.mean([r['pe_norm'] for r in nat])),
                    'natural_content_norm': float(np.mean([r['content_norm'] for r in nat]))}
        ax[0].plot(canv, ratio, ls, marker='o', color=SERIES[0] if tag == 'q35' else SERIES[2], label=f'{tag} disc images')
        ax[0].axhline(res[tag]['natural_ratio'], ls=':', color=SERIES[0] if tag == 'q35' else SERIES[2],
                      label=f'{tag} natural 512')
        order = lambda k: -1 if k == 'input' else (99 if k == 'merger' else int(k.split('_')[1]))
        stages = sorted(nat[0]['norm'], key=order)
        xs = np.arange(len(stages))
        ax[1].plot(xs, [np.mean([r['norm'][s] for r in nat]) for s in stages], ls, color=SERIES[0], label=tag)
        res[tag]['change_natural'] = {}
        for j, name in enumerate(('pe_zero', 'pe_mean', 'pe_shuffle', 'rope_zero', 'both_zero')):
            v = [np.mean([r['change'][name][s] for r in nat]) for s in stages]
            res[tag]['change_natural'][name] = dict(zip(stages, map(float, v)))
            ax[2].plot(xs, v, ls, color=SERIES[j], label=f'{tag} {name}' if tag == 'q35' or j == 0 else None)
        ticks = [0, 1, 5, 10, 15, 20, 25, 27, 28]
        for a_ in ax[1:]:
            a_.set_xticks(ticks, [stage_label(stages[i]) for i in ticks], rotation=0)
            a_.set_xlabel('stage: In = ViT input, 1-27 = blocks, M = merger')
    ax[0].set_xlabel('canvas (pixels)'); ax[0].set_ylabel('||PE|| / ||patch content||'); ax[0].set_title('PE size at the ViT input')
    ax[0].axvline(768, color='#999999', lw=.8); ax[0].legend(fontsize=7)
    ax[1].set_yscale('log'); ax[1].set_title('mean token norm by ViT stage'); ax[1].legend(fontsize=7)
    ax[2].set_title('relative change ||h - h_edit|| / ||h|| (natural)'); ax[2].legend(fontsize=7, ncol=2)
    save(fig, out, 'norms.png')
    return res


COND_ORDER = ['base', 'pe_native', 'pe_mean', 'pe_shuffle', 'pe_zero', 'pe_x0.36', 'pe_x2', 'pe_x2.76', 'pe_x4',
              'rope_zero', 'rope_zero_pe_x4', 'rope_zero_pe_mean', 'rope_zero_pe_zero']
COLOR = dict(zip(COND_ORDER, ['#222222', '#8c564b', '#1baf7a', '#eda100', '#e87ba4', '#9ecae1', '#6baed6',
                               '#2171b5', '#08306b', '#eb6834', '#d62728', '#9467bd', '#7f7f7f']))
STYLE = {c: '--' if c.startswith('rope_zero') else '-' for c in COND_ORDER}


def fig_coord(out: Path, res: dict) -> None:
    for tag in ('q35', 'q3vl'):
        m = res.get(tag, {}).get('coord')
        if not m:
            continue
        white = {k: v for k, v in m.items() if 'noise' not in k.split('|')[0]}
        noise = {k.replace('noise', ''): v for k, v in m.items() if 'noise' in k.split('|')[0]}
        canv = sorted({int(k.split('|')[0]) for k in white})
        conds = [c for c in COND_ORDER if any(k.endswith('|' + c) for k in white)]
        fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
        for row, data in enumerate((white, noise)):
            for c in conds:
                xs = [s for s in canv if f'{s}|{c}' in data]
                if not xs:
                    continue
                kw = dict(ls=STYLE[c], marker='o', ms=4, color=COLOR[c], label=c)
                # lenient diagnostic parse (identical repeated points accepted); equals strict for Qwen3.5
                ax[row, 0].plot(xs, [data[f'{s}|{c}']['lenient']['hit']['p'] for s in xs], **kw)
                ax[row, 1].plot(xs, [data[f'{s}|{c}']['lenient']['error']['mean'] for s in xs], **kw)
                ax[row, 2].plot(xs, [data[f'{s}|{c}']['lenient'].get('slope_x', np.nan) for s in xs], **kw)
            bg = 'white canvas' if row == 0 else 'gray-noise background'
            for a_, t in zip(ax[row], ('hit rate (point inside disc)', 'center error (0-1000)', 'regression slope, x')):
                a_.set_xscale('log', base=2); a_.set_xticks(canv, [str(c) for c in canv]); a_.set_title(f'{t}; {bg}', fontsize=9)
                a_.axvline(768, color='#999999', lw=.8); a_.set_xlabel('canvas (pixels)')
            ax[row, 2].axhline(1, color='#999999', lw=.8)
        ax[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f'{tag}: red-disc point output under ViT position edits (36 images per point; grey line = 768, native PE size)')
        save(fig, out, f'coord_{tag}.png')


def fig_coord_scatter(out: Path, run: Path) -> None:
    p = run / 'coord' / 'q35_records.jsonl'
    if not p.exists():
        return
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    conds = ['base', 'pe_zero', 'pe_mean', 'pe_shuffle', 'rope_zero', 'rope_zero_pe_zero']
    fig, ax = plt.subplots(1, len(conds), figsize=(15, 2.9))
    for a_, c in zip(ax, conds):
        rows = [r for r in recs if r['condition'] == c and r['case']['canvas'] == 512]
        for r in rows:
            t = [1000 * v / 512 for v in r['case']['center']]
            if r['score']['valid']:
                a_.annotate('', xy=r['score']['center'], xytext=t, arrowprops=dict(arrowstyle='->', color=SERIES[0], lw=.7))
            a_.plot(*t, 'o', color=SERIES[1], ms=3)
        a_.set_xlim(0, 1000); a_.set_ylim(1000, 0); a_.set_aspect('equal'); a_.set_title(c, fontsize=8)
    fig.suptitle('512 canvas: truth (orange) -> output (arrow head)')
    save(fig, out, 'coord_scatter_q35.png')


def fig_t2(out: Path, res: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, tag in enumerate(('q35', 'q3vl')):
        m = res.get(tag, {}).get('t2')
        if not m:
            continue
        canv = sorted({int(k.split('|')[0]) for k in m})
        conds = [c for c in COND_ORDER if any(k.endswith('|' + c) for k in m)]
        for j, c in enumerate(conds):
            xs = [s for s in canv if f'{s}|{c}' in m]
            ax[i].plot(xs, [m[f'{s}|{c}']['auroc'] for s in xs], STYLE[c], marker='o', ms=4, color=COLOR[c], label=c)
        ax[i].set_xscale('log', base=2); ax[i].set_xticks(canv, [str(c) for c in canv]); ax[i].axhline(.5, color='#999999', lw=.8)
        ax[i].set_title(f'{tag}: T2 same/different AUROC (48 pairs)'); ax[i].set_xlabel('canvas (pixels)')
        ax[i].legend(fontsize=7, ncol=2)
    save(fig, out, 't2.png')


def fig_swap(out: Path, res: dict) -> None:
    m = res.get('q35', {}).get('swap')
    if not m:
        return
    canv = sorted({int(k.split('|')[0]) for k in m if k != 'base_hits'})
    lines = [('std|pe', 'PE swap'), ('std|rope', 'RoPE swap'), ('std|both', 'PE + RoPE swap'),
             ('x4|pe', 'PE swap, PE x4'), ('rz|pe', 'PE swap, RoPE removed'), ('rzx4|pe', 'PE swap, RoPE removed, PE x4')]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for j, (k, lab) in enumerate(lines):
        xs = [c for c in canv if f'{c}|{k}' in m]
        y = [m[f'{c}|{k}']['w']['mean'] for c in xs]
        lo = [m[f'{c}|{k}']['w']['ci'][0] for c in xs]
        hi = [m[f'{c}|{k}']['w']['ci'][1] for c in xs]
        ax.plot(xs, y, marker='o', color=SERIES[j], label=lab)
        ax.fill_between(xs, lo, hi, color=SERIES[j], alpha=.15, lw=0)
    ax.set_xscale('log', base=2); ax.set_xticks(canv, [str(c) for c in canv]); ax.axvline(768, color='#999999', lw=.8)
    ax.axhline(0, color='#999999', lw=.8); ax.set_xlabel('canvas (pixels)'); ax.set_ylabel('displacement weight w')
    ax.set_title('Qwen3.5: output follows the swapped position source'); ax.legend(fontsize=7)
    save(fig, out, 'swap_weights.png')


def stage_label(s: str) -> str:
    return {'content': 'C', 'input': 'In', 'merger': 'M'}.get(s, s.replace('block_', ''))


def fig_probe(out: Path, res: dict) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    for i, tag in enumerate(('q35', 'q3vl')):
        m = res.get(tag, {}).get('probe')
        if not m:
            continue
        for j, (c, v) in enumerate(m.items()):
            stages = list(v)
            xs = np.arange(len(stages))
            ax[0, i].plot(xs, [v[s]['mae'] for s in stages], marker='o', ms=3, color=SERIES[j], label=c)
            ax[1, i].plot(xs, [100 * v[s]['pos_share'] for s in stages], marker='o', ms=3, color=SERIES[j], label=c)
        for a_ in ax[:, i]:
            a_.set_xticks(xs, [stage_label(s) for s in stages], fontsize=6)
        ax[0, i].set_title(f'{tag}: linear position probe, test MAE (0-1000)'); ax[0, i].legend(fontsize=7)
        ax[0, i].axhline(250, color='#999999', lw=.8)
        ax[1, i].set_title(f'{tag}: position-variance share (%) of tokens'); ax[1, i].set_yscale('log')
        ax[1, i].set_xlabel('stage: C = patch content, In = content + PE, 1-27 = ViT blocks, M = merger')
    save(fig, out, 'probe_layers.png')


def fig_probe_maps(out: Path, run: Path) -> None:
    stages = ['input', 'block_1', 'block_4', 'block_13', 'block_27', 'merger']
    conds = [c for c in ('base', 'pe_zero', 'rope_zero', 'both_zero') if (run / 'probe' / f'q35_{c}.pt').exists()]
    if not conds:
        return
    fig, ax = plt.subplots(len(conds), len(stages), figsize=(2.3 * len(stages), 2.3 * len(conds)))
    ax = np.atleast_2d(ax)
    for i, c in enumerate(conds):
        d = torch.load(run / 'probe' / f'q35_{c}.pt')
        for j, s in enumerate(stages):
            p = d['preds'][s].float().mean(0).numpy()                      # (N, 2) mean prediction per slot
            if s == 'merger':
                g = p.reshape(16, 16, 2)
            else:
                g = vl.to_raster(torch.from_numpy(p), 32, 32).numpy().reshape(32, 32, 2)
            a_ = ax[i, j]
            for k in range(g.shape[0]):
                a_.plot(g[k, :, 0], g[k, :, 1], color=SERIES[0], lw=.5)
                a_.plot(g[:, k, 0], g[:, k, 1], color=SERIES[0], lw=.5)
            mae = d['result']['stages'][s]['mae']
            a_.set_xlim(0, 1000); a_.set_ylim(1000, 0); a_.set_aspect('equal'); a_.set_xticks([]); a_.set_yticks([])
            a_.set_title(f'{c} | {stage_label(s)} | MAE {mae:.0f}', fontsize=7)
    fig.suptitle('Mean probe prediction per slot (grid lines connect neighbouring slots), Qwen3.5, 100 test images')
    save(fig, out, 'probe_maps.png')


def fig_layout(out: Path, res: dict) -> None:
    for tag in ('q35', 'q3vl'):
        m = res.get(tag, {}).get('layout')
        if not m:
            continue
        conds = list(m)
        targets = [('red_x', 'r2', 'red x (R2)'), ('blue_x', 'r2', 'other object x (R2)'),
                   ('red_left', 'p', 'red left of blue (acc.)'), ('red_bigger', 'p', 'red bigger (acc.)')]
        fig, ax = plt.subplots(len(conds), 4, figsize=(15, 3.2 * len(conds)), squeeze=False)
        for i, c in enumerate(conds):
            keys = sorted({k.split('|')[0] for k in m[c]}, key=lambda s: (s != 'content', s != 'input', s == 'merger',
                                                                           int(s.split('_')[1]) if s.startswith('block') else 0))
            xs = np.arange(len(keys))
            for j, (lab, field, title) in enumerate(targets):
                for r, readout in enumerate(('red', 'blue', 'background', 'bg_single', 'pooled')):
                    y = [m[c][f'{k}|{readout}'][lab][field] for k in keys]
                    ax[i, j].plot(xs, y, marker='o', ms=3, color=SERIES[r], label=readout)
                ax[i, j].set_xticks(xs, [stage_label(k) for k in keys], fontsize=6)
                ax[i, j].set_title(f'{tag} {c}: {title}', fontsize=8)
                ax[i, j].set_ylim(-.1 if field == 'r2' else .4, 1.02)
            ax[i, 0].legend(fontsize=7)
        save(fig, out, f'layout_{tag}.png')


def fig_leace(out: Path, res: dict, run: Path) -> None:
    m = res.get('q35', {}).get('leace')
    p = run / 'leace' / 'q35_erasers.pt'
    if not m or not p.exists():
        return
    d = torch.load(p)
    sm = (d['slot_means'] - d['slot_means'].mean(0)).double().numpy()
    u, s, vt = np.linalg.svd(sm, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(2, 6)
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(np.arange(1, 41), np.cumsum(var[:40]), marker='o', ms=3, color=SERIES[0], label='merger slot means (PCA)')
    sing = d['onehot']['singular'].double().numpy()
    ax0.plot(np.arange(1, 41), np.cumsum(sing[:40] ** 2) / (sing ** 2).sum(), marker='s', ms=3, color=SERIES[1],
             label='whitened cross-covariance (LEACE)')
    ax0.set_xlabel('number of components'); ax0.set_ylabel('cumulative share'); ax0.legend(fontsize=7)
    ax0.set_title('Dimensionality of the position signal at the merger')
    for j in range(4):
        a_ = fig.add_subplot(gs[0, 2 + j])
        a_.imshow((u[:, j] * s[j]).reshape(16, 16), cmap='RdBu_r')
        a_.set_title(f'PC{j + 1} of slot means ({100 * var[j]:.0f}%)', fontsize=8); a_.set_xticks([]); a_.set_yticks([])
        a_.grid(False)
    conds = ['none', 'random2', 'xy', 'random255', 'onehot', 'slotmean']
    ax1 = fig.add_subplot(gs[1, :2])
    ax2 = fig.add_subplot(gs[1, 2:4])
    ax3 = fig.add_subplot(gs[1, 4:])
    xs = np.arange(len(conds))
    hits = [m['coord'][f'512|{c}']['hit'] for c in conds]
    ax1.bar(xs, [h['p'] for h in hits], color=SERIES[0], width=.6)
    ax1.errorbar(xs, [h['p'] for h in hits], yerr=np.abs(np.array([h['ci'] for h in hits]).T - [h['p'] for h in hits]),
                 fmt='none', color='#333333', lw=.8)
    ax1.set_xticks(xs, conds, fontsize=7); ax1.set_title('disc point hit rate (108 images)')
    ax2.bar(xs, [m['coord'][f'512|{c}'].get('corr_x', 0) for c in conds], width=.35, color=SERIES[0], label='x')
    ax2.bar(xs + .35, [m['coord'][f'512|{c}'].get('corr_y', 0) for c in conds], width=.35, color=SERIES[1], label='y')
    ax2.set_xticks(xs + .17, conds, fontsize=7); ax2.set_title('corr(output, truth)'); ax2.legend(fontsize=7)
    ax3.bar(xs, [m['t2'][f'512|{c}']['auroc'] for c in conds], color=SERIES[2], width=.6)
    if 'probe' in m:
        ax3b = ax3.twinx()
        ax3b.plot(xs, [m['probe'][c]['mae'] for c in conds], 'o', color=SERIES[3])
        ax3b.set_ylabel('refit linear probe MAE', color=SERIES[3])
    ax3.set_xticks(xs, conds, fontsize=7); ax3.set_ylim(.4, 1.02); ax3.set_title('T2 AUROC (bars) and probe MAE (dots)')
    save(fig, out, 'leace.png')


def fig_reread(out: Path, res: dict) -> None:
    m = res.get('q35', {}).get('reread')
    if not m:
        return
    order = ['crop_local', 'rope_offset', 'pe_window', 'rope_offset_pe_window', 'pe_window_x4',
             'rope_offset_pe_window_x4', 'canvas_context', 'full_image']
    order = [o for o in order if o in m]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = np.arange(len(order))
    w = [m[o]['w']['mean'] for o in order]
    ci = np.array([m[o]['w']['ci'] for o in order]).T
    ax.bar(xs, w, color=SERIES[0], width=.6)
    ax.errorbar(xs, w, yerr=np.abs(ci - w), fmt='none', color='#333333', lw=.8)
    ax.set_xticks(xs, order, rotation=25, ha='right', fontsize=7); ax.axhline(1, color='#999999', lw=.8)
    ax.set_ylabel('w: 0 = crop coordinates, 1 = original-image coordinates')
    ax.set_title('Re-encoding a 256 crop of a 512 canvas (48 cases)')
    save(fig, out, 'reread.png')


def fig_refcoco(out: Path, res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x0 = 0
    ticks, labels = [], []
    for tag in ('q35', 'q3vl'):
        m = res.get(tag, {}).get('refcoco')
        if not m:
            continue
        for c in [c for c in COND_ORDER if c in m]:
            a = m[c]['acc']
            ax.bar(x0, a['p'], color=COLOR[c], width=.7)
            ax.errorbar(x0, a['p'], yerr=[[a['p'] - a['ci'][0]], [a['ci'][1] - a['p']]], color='#333333', lw=.8)
            ticks.append(x0); labels.append(f'{tag}\n{c}'); x0 += 1
        x0 += .8
    ax.set_xticks(ticks, labels, fontsize=7); ax.set_ylim(.6, 1); ax.set_ylabel('IoU >= 0.5 (751 expressions)')
    ax.set_title('RefCOCO val grounding with the ViT PE rescaled')
    save(fig, out, 'refcoco.png')


def fig_inputs(out: Path) -> None:
    fig, ax = plt.subplots(1, 6, figsize=(15, 2.8))
    for a_, canvas in zip(ax[:2], (256, 1536)):
        c = vl.disc_grid_cases(canvas, 20261015)[8]
        a_.imshow(cr.disc_image(tuple(c['center']), c['diameter'], canvas)[0]); a_.set_title(f'disc, canvas {canvas}', fontsize=8)
    c = vl.t2_cases(512, 20261015)[0]
    ax[2].imshow(vl.t2_render(c)); ax[2].set_title('T2 pair (different), 512', fontsize=8)
    ax[3].imshow(vl.layout_render(vl.layout_case(0, 20261015))); ax[3].set_title('layout image, 512', fontsize=8)
    rc = reread_cases(20261015, 1)[0]
    full, _ = cr.disc_image(tuple(rc['center']), rc['diameter'], 512)
    ax[4].imshow(full); ox, oy = rc['origin']
    ax[4].add_patch(plt.Rectangle((ox, oy), 256, 256, fill=False, ec=SERIES[0], lw=1.5)); ax[4].set_title('re-read: full 512', fontsize=8)
    ax[5].imshow(full.crop((ox, oy, ox + 256, oy + 256))); ax[5].set_title('re-read: 256 crop', fontsize=8)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
    save(fig, out, 'inputs.png')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    res = json.loads((args.run / 'results.json').read_text())
    extra = {'pe_table': fig_table(args.figures, args.run), 'weights': fig_weights(args.figures),
             'norms': fig_norms(args.figures, args.run)}
    (args.run / 'results_figures.json').write_text(json.dumps(extra, indent=1) + '\n')
    fig_inputs(args.figures)
    fig_coord(args.figures, res)
    fig_coord_scatter(args.figures, args.run)
    fig_t2(args.figures, res)
    fig_swap(args.figures, res)
    fig_probe(args.figures, res)
    fig_probe_maps(args.figures, args.run)
    fig_layout(args.figures, res)
    fig_leace(args.figures, res, args.run)
    fig_reread(args.figures, res)
    fig_refcoco(args.figures, res)


if __name__ == '__main__':
    main()
