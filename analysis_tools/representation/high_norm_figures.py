"""Figures of experiment 2.2 (high-norm tokens). Reads <run>/<tag>/ stage outputs and analysis/results.json."""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
from safetensors import safe_open

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5195', '#5c5c5c']
MODELS = {'q35': 'Qwen3.5-9B', 'q3vl': 'Qwen3-VL-8B-Instruct'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False})
TAU = 3.0


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('figure', name)


def load_table(tag: str) -> np.ndarray:
    snap = glob.glob(os.path.join(os.environ['HF_HUB_CACHE'], f'models--Qwen--{MODELS[tag]}', 'snapshots', '*'))[0]
    index = json.loads(Path(snap, 'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(os.path.join(snap, index['model.visual.pos_embed.weight']), 'pt') as f:
        return f.get_tensor('model.visual.pos_embed.weight').float().numpy()


def noaxis(ax):
    ax.set_xticks([]); ax.set_yticks([])


def mark_hn(ax, ratio: np.ndarray, label: bool = False) -> None:
    """Hollow red circles on tokens whose norm exceeds TAU times the image median."""
    r, c = np.nonzero(np.asarray(ratio, float) > TAU)
    ax.scatter(c, r, s=22, facecolor='none', edgecolor='#ff2d2d', lw=.8, label='norm > 3 x median' if label else None)


# ----------------------------------------------------------------------------- norm maps per layer

def fig_norm_maps(out: Path, root: Path, tag: str) -> None:
    sv = {n: torch.load(root / 'survey' / f'{n}.pt', weights_only=False) for n in ('blank', 'probe', 'natural')}
    picks = [('white 512', next(r for r in sv['blank'] if r['id'] == 'white512')),
             ('synthetic 512', next((r for r in sv['probe'] if r['labels']['canvas'] == 512 and 'decoder' in r),
                                    next(r for r in sv['probe'] if 'decoder' in r))),
             ('natural', next(r for r in sv['natural'] if 'decoder' in r))]
    vit_blocks = [3, 6, 9, 10, 12, 18, 23, 26, 27]
    dec_layers = [1, 8, 16, 24, 31, 32]
    ncol = len(vit_blocks) + 1 + len(dec_layers)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(1.35 * ncol, 1.5 * len(picks) + .6))
    norm = LogNorm(vmin=.3, vmax=40)
    for i, (name, r) in enumerate(picks):
        for j, b in enumerate(vit_blocks):
            im = axes[i, j].imshow(np.clip(r['ratio'][b - 1].astype(float), .3, 40), cmap='viridis', norm=norm)
            mark_hn(axes[i, j], r['ratio'][b - 1], label=(i == 0 and j == 0))
            axes[i, j].set_title(f'ViT {b}' if i == 0 else '', fontsize=8)
        axes[i, len(vit_blocks)].imshow(np.clip(r['merger_ratio'].astype(float), .3, 40), cmap='viridis', norm=norm)
        axes[i, len(vit_blocks)].set_title('merger' if i == 0 else '', fontsize=8)
        d = r['decoder']['visual']
        for j, l in enumerate(dec_layers):
            m = d[l - 1] / np.median(d[l - 1])
            axes[i, len(vit_blocks) + 1 + j].imshow(np.clip(m, .3, 40), cmap='viridis', norm=norm)
            mark_hn(axes[i, len(vit_blocks) + 1 + j], m)
            axes[i, len(vit_blocks) + 1 + j].set_title(f'dec {l}' if i == 0 else '', fontsize=8)
        axes[i, 0].set_ylabel(name, fontsize=8)
        for a in axes[i]:
            noaxis(a)
    mark_hn_merger = [mark_hn(axes[i, len(vit_blocks)], picks[i][1]['merger_ratio']) for i in range(len(picks))]
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper left', fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=.015, pad=.01)
    cb.set_label('token norm / image median norm (log scale)')
    fig.suptitle(f'{MODELS[tag]}: token-norm maps by layer (ViT patches, merged tokens, decoder visual tokens)', fontsize=10)
    save(fig, out, f'norm_maps_{tag}.png')


def fig_emergence(out: Path, res: dict, tag: str) -> None:
    sv = res['survey']
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    names = [n for n in ('blank', 'noise', 'probe', 'grid', 'natural') if n in sv]
    labels = {'blank': 'blank canvases', 'noise': 'pixel noise', 'probe': 'random-placement synthetic', 'grid': '1.3 grids',
              'natural': 'RefCOCO natural'}
    for k, n in enumerate(names):
        y = sv[n]['max_ratio_median']
        x = np.arange(1, len(y) + 1)
        ax[0].plot(x, y, '-o', ms=3, color=SERIES[k], label=labels[n])
        ax[0].fill_between(x, sv[n]['max_ratio_q10'], sv[n]['max_ratio_q90'], color=SERIES[k], alpha=.12)
        ax[1].plot(x, sv[n]['n_hn3_mean'], '-o', ms=3, color=SERIES[k], label=labels[n])
        if 'decoder' in sv[n]:
            d = sv[n]['decoder']['visual_max_over_median_median']
            ax[2].plot(np.arange(1, len(d) + 1), d, '-o', ms=3, color=SERIES[k], label=labels[n])
    ax[0].set_yscale('log'); ax[0].axhline(TAU, color='#999', ls='--', lw=.8)
    ax[0].set_xlabel('ViT block'); ax[0].set_ylabel('max norm / median (median over images)')
    ax[0].set_title('largest token norm per block'); ax[0].legend(fontsize=7)
    ax[1].set_yscale('symlog'); ax[1].set_xlabel('ViT block'); ax[1].set_ylabel('tokens with norm > 3 x median')
    ax[1].set_title('high-norm token count per image')
    ax[2].axhline(TAU, color='#999', ls='--', lw=.8)
    ax[2].set_xlabel('decoder layer'); ax[2].set_ylabel('max / median norm of visual tokens'); ax[2].set_title('decoder visual tokens')
    fig.suptitle(f'{MODELS[tag]}: emergence of high-norm tokens (dashed: threshold 3; band: 10-90% of images)', fontsize=10)
    fig.tight_layout()
    save(fig, out, f'emergence_{tag}.png')


def fig_decoder_prefix(out: Path, res: dict, tag: str) -> None:
    sv = res['survey']
    n = 'natural' if 'natural' in sv else 'blank'
    d = sv[n]['decoder']
    pref = np.array(d['prefix_norm_median'])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for j, t in enumerate(d['prefix_tokens']):
        ax.plot(np.arange(1, len(pref) + 1), pref[:, j], '-o', ms=3, color=SERIES[j], label=repr(t))
    ax.plot(np.arange(1, len(pref) + 1), d['visual_median_norm'], '-', color='k', label='visual tokens (median)')
    ax.set_yscale('log'); ax.set_xlabel('decoder layer'); ax.set_ylabel('hidden-state norm')
    ax.legend(fontsize=7); ax.set_title(f'{MODELS[tag]}: prefix-token norms vs visual tokens ({n} images)')
    fig.tight_layout()
    save(fig, out, f'decoder_prefix_{tag}.png')


def fig_freq_maps(out: Path, res: dict, tag: str) -> None:
    fm = res['freq_maps']
    keys = sorted(k for k in fm if k.startswith('probe|'))
    keys += sorted((k for k in fm if k.startswith('grid|')), key=lambda k: int(k.split('|')[1].split('x')[0]))
    keys += sorted((k for k in fm if k.startswith('natural|')), key=lambda k: -fm[k]['n'])[:2]
    keys += [k for k in fm if k.startswith('noise|') and fm[k]['n'] >= 8][:1]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(2.1 * len(keys), 2.6))
    axes = np.atleast_1d(axes)
    for a, k in zip(axes, keys):
        f = np.array(fm[k]['f'])
        im = a.imshow(f, cmap='viridis', vmin=0, vmax=1)
        a.set_title(f"{k.replace('|', ' ').replace('probe', 'random-placement')}\nN={fm[k]['n']}", fontsize=7)
        noaxis(a)
    cb = fig.colorbar(im, ax=axes, fraction=.02, pad=.01)
    cb.set_label('fraction of images with a high-norm token (ViT block 12)')
    save(fig, out, f'freq_maps_{tag}.png')


def fig_table_positions(out: Path, res: dict, tag: str) -> None:
    t = load_table(tag)
    n = np.linalg.norm(t, axis=1).reshape(48, 48)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(n / np.median(n), cmap='viridis')
    cb = fig.colorbar(im, ax=ax, fraction=.046)
    cb.set_label('PE table entry norm / median')
    sizes = sorted({tuple(r['grid']) for r in res['blank_table_positions'] if r['image'].startswith('white')})
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h']
    for k, g in enumerate(sizes):
        pts = [r for r in res['blank_table_positions'] if tuple(r['grid']) == g and r['image'].startswith('white')]
        ax.scatter([p['table_col'] for p in pts], [p['table_row'] for p in pts], marker=markers[k % len(markers)],
                   facecolor='none', edgecolor=SERIES[k % len(SERIES)], s=40, lw=1.2, label=f'{g[0]}x{g[1]} patches')
    ax.set_xlim(-1, 48); ax.set_ylim(48, -1)
    ax.set_xlabel('PE table column'); ax.set_ylabel('PE table row')
    ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.25, .5), title='white canvas', title_fontsize=7)
    ax.set_title(f'{MODELS[tag]}: high-norm positions (block 12)\nin PE-table coordinates', fontsize=9)
    save(fig, out, f'table_positions_{tag}.png')


def fig_mechanism(out: Path, res: dict, root: Path, tag: str) -> None:
    n = res['neurons']
    dec = n['decomposition']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    parts = ['in', 'attn', 'mlp', 'out']
    lab = {'in': 'block input', 'attn': 'attention update', 'mlp': 'MLP update', 'out': 'block output'}
    x = np.arange(len(parts))
    ax[0].bar(x - .2, [dec[p]['hn'] for p in parts], .4, color=SERIES[1], label='high-norm tokens')
    ax[0].bar(x + .2, [dec[p]['normal'] for p in parts], .4, color=SERIES[0], label='other tokens')
    ax[0].set_xticks(x, [lab[p] for p in parts]); ax[0].set_yscale('log'); ax[0].set_ylabel('mean L2 norm')
    ax[0].set_title(f'ViT block {n["block"]}: which sublayer writes the large norm'); ax[0].legend(fontsize=7)
    A = np.load(root / 'neurons' / 'act_hn_top64.npy').astype(float)
    B = np.load(root / 'neurons' / 'act_normal_top64.npy').astype(float)
    order = np.load(root / 'neurons' / 'neuron_order.npy')
    k = 14
    pos = np.arange(k)
    ax[1].boxplot([A[:, i] for i in range(k)], positions=pos - .2, widths=.35, showfliers=False,
                  patch_artist=True, boxprops=dict(facecolor=SERIES[1], alpha=.6))
    ax[1].boxplot([B[:, i] for i in range(k)], positions=pos + .2, widths=.35, showfliers=False,
                  patch_artist=True, boxprops=dict(facecolor=SERIES[0], alpha=.6))
    ax[1].set_xticks(pos, [str(int(o)) for o in order[:k]], rotation=60, fontsize=7)
    ax[1].set_xlabel('MLP neuron index (ranked by activation difference)'); ax[1].set_ylabel('activation (fc2 input)')
    reg = set(n['register_neurons'])
    for i in range(k):
        if int(order[i]) in reg:
            ax[1].get_xticklabels()[i].set_color(SERIES[1])
    ax[1].set_title('register neurons (orange labels): high-norm (orange) vs other (blue) tokens', fontsize=8)
    fig.tight_layout()
    save(fig, out, f'mechanism_{tag}.png')


def fig_source(out: Path, root: Path, res: dict, tag: str) -> None:
    recs = torch.load(root / 'source' / 'records.pt', weights_only=False)
    conds = [c for c in ('base', 'pe_zero', 'pe_mean', 'pe_shuffle0', 'pe_hflip', 'pe_vflip', 'pe_outlier_shrink', 'pe_x0.5',
                         'pe_x2', 'rope_zero', 'rope_shuffle', 'ab_hn_mean', 'ab_neuron_zero', 'ab_rand_mean') if c in recs]
    ids = ['white512', 'noise512_0', next(r['id'] for r in recs['base'] if r['id'].startswith('p')),
           next(r['id'] for r in recs['base'] if r['id'].startswith('rc'))]
    fig, axes = plt.subplots(len(ids), len(conds), figsize=(1.25 * len(conds), 1.35 * len(ids) + .5))
    norm = LogNorm(vmin=.3, vmax=40)
    for j, c in enumerate(conds):
        by = {r['id']: r for r in recs[c]}
        for i, iid in enumerate(ids):
            r = by[iid]
            k = r['blocks'].index(12)
            im = axes[i, j].imshow(np.clip(r['ratio'][k].astype(float), .3, 40), cmap='viridis', norm=norm)
            mark_hn(axes[i, j], r['ratio'][k], label=(i == 0 and j == 0))
            noaxis(axes[i, j])
            if i == 0:
                axes[i, j].set_title(c.replace('_', ' '), fontsize=7)
            if j == 0:
                axes[i, j].set_ylabel(iid, fontsize=7)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper left', fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=.015, pad=.01)
    cb.set_label('norm / image median at ViT block 12 (log)')
    fig.suptitle(f'{MODELS[tag]}: high-norm tokens under position-source interventions and ablations', fontsize=10)
    save(fig, out, f'source_{tag}.png')
    pc = root / 'source' / 'phase_content.pt'
    if not pc.exists():
        return
    pc = torch.load(pc, weights_only=False)
    keys = list(pc['window'])
    fig, axes = plt.subplots(2, max(len(keys) + 1, 6), figsize=(14, 5))
    norm = LogNorm(vmin=.3, vmax=40)
    full = pc['full640'].astype(float)
    axes[0, 0].imshow(np.clip(full, .3, 40), cmap='viridis', norm=norm); axes[0, 0].set_title('white 640 (40x40)', fontsize=7)
    mark_hn(axes[0, 0], full, label=True)
    for j, k in enumerate(keys):
        r0, c0 = map(int, k.split('_'))
        axes[0, 0].add_patch(plt.Rectangle((c0 - .5, r0 - .5), 32, 32, fill=False, ec=SERIES[j % 8], lw=.8))
        axes[0, j + 1].imshow(np.clip(pc['window'][k].astype(float), .3, 40), cmap='viridis', norm=norm)
        mark_hn(axes[0, j + 1], pc['window'][k])
        axes[0, j + 1].set_title(f'512 with PE window\noffset ({r0},{c0})', fontsize=7, color=SERIES[j % 8])
    for j, c in enumerate(pc['content'][:axes.shape[1]]):
        im = axes[1, j].imshow(np.clip(c['ratio'].astype(float), .3, 40), cmap='viridis', norm=norm)
        mark_hn(axes[1, j], c['ratio'])
        x0, y0, x1, y1 = c['box']
        axes[1, j].add_patch(plt.Rectangle((x0 / 16 - .5, y0 / 16 - .5), (x1 - x0) / 16, (y1 - y0) / 16, fill=False, ec='w', lw=1))
        r, cc = divmod(c['target_raster'], 32)
        axes[1, j].plot(cc, r, '+', color='#ffd400', ms=9, label='target patch' if j == 0 else None)
        axes[1, j].set_title(('canonical' if c['is_canonical'] else 'control') + f' ({r},{cc})', fontsize=7)
    for a in axes.flat:
        noaxis(a)
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc='upper left', fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=.015, pad=.01)
    cb.set_label('norm / image median at ViT block 12 (log)')
    fig.suptitle('top: PE phase windows (coloured frames mark the window in the 640 canvas); bottom: noise tile (white frame) '
                 'placed over a target patch (yellow cross)', fontsize=9)
    save(fig, out, f'phase_content_{tag}.png')


def fig_attention(out: Path, root: Path, res: dict, tag: str) -> None:
    at = res['attention']['vit']
    blocks = sorted(int(k) for k in at)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(blocks, [at[str(b)]['mass_hn_mean'] if str(b) in at else at[b]['mass_hn_mean'] for b in blocks], '-o', ms=3,
               color=SERIES[1], label='high-norm tokens (block-12 set)')
    ax[0].plot(blocks, [at[str(b)]['mass_rand_mean'] for b in blocks], '-o', ms=3, color=SERIES[0], label='same number of random tokens')
    ax[0].set_xlabel('ViT block'); ax[0].set_ylabel('attention mass on the set (mean over queries, heads, images)')
    ax[0].set_title('ViT attention received'); ax[0].legend(fontsize=7)
    vit = torch.load(root / 'attention' / 'vit.pt', weights_only=False)
    r = vit[0]
    from analysis_tools.position import vit_layout as vl
    hp, wp = r['grid']
    for j, b in enumerate((12, 20)):
        if b not in r['res']:
            continue
        rec = r['res'][b]['received'].mean(0)
        m = vl.to_raster(torch.tensor(rec)[:, None], hp, wp)[:, 0].view(hp, wp).numpy()
        im = ax[1 + j].imshow(m * hp * wp, cmap='viridis', norm=LogNorm(vmin=.1, vmax=max(m.max() * hp * wp, 1)))
        ax[1 + j].set_title(f"{r['id']}: attention received at ViT block {b}", fontsize=8)
        noaxis(ax[1 + j])
        cb = fig.colorbar(im, ax=ax[1 + j], fraction=.046)
        cb.set_label('received attention x number of tokens (1 = uniform)')
    fig.tight_layout()
    save(fig, out, f'attention_vit_{tag}.png')
    dec = res['attention'].get('decoder')
    if dec:
        layers = sorted(int(k) for k in dec)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
        for k, (key, lab) in enumerate((('first', 'first token <|im_start|>'), ('prefix', 'all 4 prefix tokens'),
                                        ('visual_all', 'all visual tokens'), ('visual_hn', 'visual tokens with ViT high-norm patches'))):
            ax[0].plot(layers, [dec[str(l)][f'last_{key}'] for l in layers], '-o', ms=3, color=SERIES[k], label=lab)
            ax[1].plot(layers, [dec[str(l)][f'visual_{key}'] for l in layers], '-o', ms=3, color=SERIES[k], label=lab)
        ax[0].set_title('query: last prompt token'); ax[1].set_title('queries: 16 random visual tokens')
        for a in ax:
            a.set_xlabel('decoder full-attention layer'); a.set_ylabel('attention mass'); a.set_ylim(0, 1)
        ax[1].legend(fontsize=7)
        fig.suptitle(f'{MODELS[tag]}: decoder attention mass (mean over heads and images)', fontsize=10)
        fig.tight_layout()
        save(fig, out, f'attention_decoder_{tag}.png')


def fig_probes(out: Path, res: dict, tag: str) -> None:
    pr = res.get('probes')
    if not pr:
        return
    sites = list(pr)
    metrics = [('content_r2', 'r2', 'own-patch content R^2'), ('position_r2', 'r2', 'position R^2'),
               ('template', 'balanced_acc', 'template bal. acc.'), ('count', 'balanced_acc', 'instance count (3 bins) bal. acc.'),
               ('canvas', 'balanced_acc', 'canvas size bal. acc.'), ('anomaly', 'balanced_acc', 'anomaly presence bal. acc.')]
    groups = [('hn', 'high-norm tokens', SERIES[1]), ('canon', 'same positions, not high-norm', SERIES[2]),
              ('random', 'random other tokens', SERIES[0]), ('pooled', 'image mean of other tokens', SERIES[7])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    nat = res.get('probes_natural', {})
    if nat:
        a = axes.flat[0]
        x = np.arange(len(sites))
        for k, (g, lab, col) in enumerate((('hn', 'high-norm tokens', SERIES[1]),
                                            ('matched', 'texture-matched other tokens', SERIES[3]),
                                            ('random', 'random other tokens', SERIES[0]))):
            a.bar(x + (k - 1) * .27, [nat.get(s, {}).get(g, {}).get('content_r2', {}).get('r2', np.nan) for s in sites], .27,
                  color=col, label=lab)
        a.set_xticks(x, [f'ViT {s}' if s != 'merger' else 'merger' for s in sites])
        a.set_title('own-patch content R^2 (RefCOCO natural images)', fontsize=9)
        a.legend(fontsize=6, loc='lower left')
        metrics = metrics[1:]
        axes_iter = list(axes.flat)[1:]
    else:
        axes_iter = list(axes.flat)
    for a, (m, key, title) in zip(axes_iter, metrics):
        x = np.arange(len(sites))
        for k, (g, lab, col) in enumerate(groups):
            vals = []
            for s in sites:
                d = pr[s].get(g, {}).get('raw', {}).get(m)
                vals.append(d[key] if d else np.nan)
            a.bar(x + (k - 1.5) * .2, vals, .2, color=col, label=lab)
        a.set_xticks(x, [f'ViT {s}' if s != 'merger' else 'merger' for s in sites])
        a.set_title(title + ' (random-placement synthetic)', fontsize=9)
        if key == 'balanced_acc':
            ch = {'template': .25, 'count': 1 / 3, 'canvas': 1 / 3, 'anomaly': .5}[m]
            a.axhline(ch, color='#999', ls='--', lw=.8)
    axes_iter[0].legend(fontsize=7)
    fig.suptitle(f'{MODELS[tag]}: linear probes on raw features (dashed: chance; test images held out)', fontsize=10)
    fig.tight_layout()
    save(fig, out, f'probes_{tag}.png')


def fig_posvar(out: Path, res: dict, tag: str) -> None:
    pv = res.get('posvar')
    if not pv:
        return
    groups = sorted({k.split('|')[0] for k in pv})
    conds = ['base', 'ab_hn_mean', 'ab_rand_mean', 'ab_neuron_zero', 'pe_zero']
    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 3.6))
    axes = np.atleast_1d(axes)
    for a, g in zip(axes, groups):
        sites = list(pv[f'{g}|base'])
        x = np.arange(len(sites))
        for k, c in enumerate(conds):
            key = f'{g}|{c}'
            if key not in pv:
                continue
            a.plot(x, [pv[key][s]['all']['r2'] for s in sites], '-o', ms=3, color=SERIES[k], label=c)
            if c == 'base':
                a.plot(x, [pv[key][s]['excl_canonical']['r2'] for s in sites], '--', color=SERIES[k], lw=1,
                       label='base, canonical positions excluded')
        a.set_xticks(x, [s.replace('vit_', 'ViT ').replace('dec_', 'dec ') for s in sites], rotation=45, fontsize=7)
        a.axhline(0, color='#999', lw=.8); a.set_ylabel('split-half $R^2_{pos}$'); a.set_title(g)
    axes[0].legend(fontsize=7)
    fig.suptitle(f'{MODELS[tag]}: position-explained variance by site and condition', fontsize=10)
    fig.tight_layout()
    save(fig, out, f'posvar_{tag}.png')


def fig_ablation(out: Path, res: dict, tag: str) -> None:
    ab = res.get('ablation')
    if not ab or 'base' not in ab:
        return
    tasks = list(ab['base'])
    conds = [c for c in ('hn_mean', 'hn_zero', 'neuron_zero', 'rand_mean0', 'rand_mean1') if c in ab]
    fig, ax = plt.subplots(figsize=(8, .45 * len(tasks) + 1.5))
    y = np.arange(len(tasks))
    for k, c in enumerate(conds):
        d = [ab[c].get(t, {}).get('delta', np.nan) for t in tasks]
        lo, hi = [], []
        for t in tasks:
            e = ab[c].get(t)
            if e is None:
                lo.append(0); hi.append(0); continue
            b0 = ab['base'][t]['acc']
            lo.append(e['acc'] - e['ci'][0]); hi.append(e['ci'][1] - e['acc'])
        ax.errorbar(np.array(d) * 100, y + (k - len(conds) / 2) * .13, xerr=np.array([lo, hi]) * 100, fmt='o', ms=4,
                    color=SERIES[[1, 3, 2, 0, 7][k]], label=c, capsize=2, lw=.8)
    ax.axvline(0, color='#999', lw=.8)
    ax.set_yticks(y, [f"{t} (base {ab['base'][t]['acc'] * 100:.0f}%, n={ab['base'][t]['n']})" for t in tasks], fontsize=7)
    ax.set_xlabel('accuracy change vs base (percentage points; bars: Wilson 95% half-widths of the condition)')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title(f'{MODELS[tag]}: task effect of ViT high-norm ablations', fontsize=10)
    save(fig, out, f'ablation_{tag}.png')


def fig_blue_box(out: Path, res: dict, tag: str) -> None:
    bb = res.get('blue_box')
    if not bb:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    v = bb['vit_band_ratio_mean']
    ax[0].plot(np.arange(1, len(v) + 1), v, '-o', ms=3, color=SERIES[0], label='ViT patches on the box band')
    ax[0].axhline(bb['merger_band_ratio_mean'], color=SERIES[1], ls='--', label='merger tokens on the box band')
    ax[0].axhline(1, color='#999', lw=.8)
    ax[0].set_xlabel('ViT block'); ax[0].set_ylabel('norm ratio marked / unmarked'); ax[0].legend(fontsize=7)
    ax[0].set_title(f"blue box effect on token norm (N={bb['n_pairs']} pairs)")
    if 'decoder_band_ratio_mean' in bb:
        d = bb['decoder_band_ratio_mean']
        ax[1].plot(np.arange(1, len(d) + 1), d, '-o', ms=3, color=SERIES[0], label='marked / unmarked, band tokens')
        ax[1].plot(np.arange(1, len(d) + 1), bb['decoder_band_over_median_marked_mean'], '-o', ms=3, color=SERIES[2],
                   label='band tokens / image median (marked)')
        ax[1].axhline(1, color='#999', lw=.8); ax[1].axhline(TAU, color='#999', ls='--', lw=.8)
        ax[1].set_xlabel('decoder layer'); ax[1].set_ylabel('norm ratio'); ax[1].legend(fontsize=7)
        ax[1].set_title('decoder visual tokens on the box band')
    fig.tight_layout()
    save(fig, out, f'blue_box_{tag}.png')


def fig_inputs(out: Path, root: Path) -> None:
    """Input examples: random-placement probe image, natural image, blue-box T3 grid, 3.5 disc and the 2.1 grid."""
    from PIL import Image
    from analysis_tools.position import coordinate_readout as cr
    from analysis_tools.representation.high_norm_run import probe_case, probe_render
    from analysis_tools.representation.visual_detail import figure_image
    cases = [json.loads(l) for l in (root / 'ablation' / 'cases.jsonl').read_text().splitlines()]
    nat = next(c for c in cases if c['task'] == 'refcoco')
    t3 = next(c for c in cases if c['task_tag'] == 'T3_box' and int(c['grid']) == 6)
    dec_root = Path(os.environ['WORK_DIR']) / 'analysis' / 'behavior' / '2026-10-13_decomposition'
    ims = [('random-placement synthetic (512)', probe_render(probe_case(3, 20261016, 512))),
           ('RefCOCO natural: ' + nat['expression'], Image.open(nat['image_path']).convert('RGB')),
           ('T3 blue box (6x6)', Image.open(dec_root / t3['image']).convert('RGB')),
           ('3.5 disc (512, d=32)', cr.disc_image((300, 180), 32)[0]),
           ('2.1 grid (384, gap 8)', figure_image(14, 8))]
    fig, ax = plt.subplots(1, len(ims), figsize=(3 * len(ims), 3.3))
    for a, (t, im) in zip(ax, ims):
        a.imshow(im); a.set_title(t, fontsize=8); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color('0.6')
    save(fig, out, 'inputs.png')


def fig_compare(out: Path, results: dict) -> None:
    if not all(t in results for t in ('q35', 'q3vl')):
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for k, t in enumerate(('q35', 'q3vl')):
        sv = results[t]['survey']
        for j, n in enumerate(('blank', 'natural')):
            if n not in sv:
                continue
            y = sv[n]['max_ratio_median']
            ax[0].plot(np.arange(1, len(y) + 1), y, ['-', '--'][j], color=SERIES[k], label=f'{MODELS[t]}, {n}')
            ax[1].plot(np.arange(1, len(y) + 1), sv[n]['n_hn3_mean'], ['-', '--'][j], color=SERIES[k], label=f'{MODELS[t]}, {n}')
        at = results[t].get('attention', {}).get('vit', {})
        if at:
            bl = sorted(int(b) for b in at)
            ax[2].plot(bl, [at[str(b)]['mass_hn_mean'] for b in bl], '-o', ms=3, color=SERIES[k], label=MODELS[t])
    ax[0].set_yscale('log'); ax[0].set_xlabel('ViT block'); ax[0].set_ylabel('max norm / median'); ax[0].legend(fontsize=7)
    ax[1].set_yscale('symlog'); ax[1].set_xlabel('ViT block'); ax[1].set_ylabel('tokens > 3 x median')
    ax[2].set_xlabel('ViT block'); ax[2].set_ylabel('attention mass on high-norm tokens'); ax[2].legend(fontsize=7)
    fig.suptitle('Qwen3.5-9B vs Qwen3-VL-8B: high-norm tokens in the shared-ancestry ViT', fontsize=10)
    fig.tight_layout()
    save(fig, out, 'compare_models.png')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--tags', default='q35,q3vl')
    args = p.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    results = {}
    for tag in args.tags.split(','):
        root = args.run / tag
        rp = root / 'analysis' / 'results.json'
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        results[tag] = res
        steps = [('survey', lambda: (fig_norm_maps(args.figures, root, tag), fig_emergence(args.figures, res, tag),
                                     fig_decoder_prefix(args.figures, res, tag), fig_freq_maps(args.figures, res, tag),
                                     fig_table_positions(args.figures, res, tag), fig_probes(args.figures, res, tag),
                                     fig_blue_box(args.figures, res, tag))),
                 ('neurons', lambda: fig_mechanism(args.figures, res, root, tag)),
                 ('source', lambda: fig_source(args.figures, root, res, tag)),
                 ('attention', lambda: fig_attention(args.figures, root, res, tag)),
                 ('posvar', lambda: fig_posvar(args.figures, res, tag)),
                 ('ablation', lambda: fig_ablation(args.figures, res, tag))]
        for key, fn in steps:
            if key in res:
                fn()
    fig_compare(args.figures, results)
    if (args.run / 'q35' / 'ablation' / 'cases.jsonl').exists():
        fig_inputs(args.figures, args.run / 'q35')


if __name__ == '__main__':
    main()
