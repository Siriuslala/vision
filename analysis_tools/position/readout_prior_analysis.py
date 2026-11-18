"""Experiment 3.16 analysis (CPU): tables, summary.json and figures of the prior localization.

Inputs: the 3.13 round-2 base capture (residual streams at q0 / q1 / target tokens, every layer), and the 3.16 stages
(features, transplant, textsrc, firsttok, grad). Missing stage outputs are reported, never replaced by defaults.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.position import readout_prior as rp
from analysis_tools.position import targeted_finetune_r2 as t2
from analysis_tools.position.coordinate_readout_run import read_jsonl

C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
     'green': '#008300', 'gray': '#8a8a8a', 'black': '#222222'}
CLOCK_LAYERS = (0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 30, 32)
FAMS = ('sweep_h', 'sweep_v', 'refcoco', 'textconv')


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def margin_section(cap, figs: Path, summ: dict) -> None:
    fam, tr, lp = cap['family'], cap['truth'], cap['logp']
    summ['margin'] = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, f in zip(axes, ('sweep_h', 'refcoco', 'textconv')):
        s = fam == f
        d = rp.margin_decomposition(tr[s], lp[s, 0])
        summ['margin'][f] = d
        t = tr[s]
        sel = (t >= 200) & (t < 900)
        t, l0 = t[sel], lp[s, 0][sel]
        k = np.round(t).astype(int) // 100
        m = l0[np.arange(len(t)), k + 1] - l0[np.arange(len(t)), k - 1]
        o = (t - 100 * k) / 10
        sc = ax.scatter(o, m, c=k, cmap='viridis', s=12, vmin=2, vmax=8)
        ax.set_title(f"{ {'sweep_h': 'disc x', 'refcoco': 'RefCOCO x', 'textconv': 'text-given x'}[f]}: "
                     f"bias share {d['bias_share']:.2f}, res {d['resolution']:.0f} -> {d['resolution_unbiased']:.0f}",
                     fontsize=9)
        ax.set_xlabel('in-bin offset o = (x - 100k) / 10')
        style(ax)
    axes[0].set_ylabel('m = l0(k+1) - l0(k-1)  (nats)')
    cb = fig.colorbar(sc, ax=axes, shrink=0.85)
    cb.set_label('true first digit k')
    fig.savefig(figs / 'margin_bias.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    summ['margin']['sweep_v'] = rp.margin_decomposition(tr[fam == 'sweep_v'], lp[fam == 'sweep_v', 0])


def clock_section(cap, figs: Path, summ: dict) -> None:
    R, fam, tr = cap['resid'], cap['family'], cap['truth']
    layers = list(cap['layers'])
    res = {}
    curves = [('sweep_h', 2, 'disc x: target tokens'), ('sweep_h', 0, 'disc x: q0'), ('sweep_h', 1, 'disc x: q1'),
              ('sweep_v', 2, 'disc y: target tokens'), ('sweep_v', 0, 'disc y: q0'),
              ('textconv', 0, 'text-given x: q0'), ('textconv', 1, 'text-given x: q1'),
              ('refcoco', 2, 'RefCOCO x: target tokens'), ('refcoco', 0, 'RefCOCO x: q0')]
    for f, site, name in curves:
        s = fam == f
        res[name] = {l: rp.clock_decodability(R[s, site, layers.index(l)].astype(np.float32), tr[s]) for l in CLOCK_LAYERS}
    summ['clock_residual'] = res
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    groups = [['disc x: target tokens', 'disc x: q0', 'disc x: q1'], ['disc y: target tokens', 'disc y: q0'],
              ['text-given x: q0', 'text-given x: q1', 'RefCOCO x: target tokens', 'RefCOCO x: q0']]
    cols = [C['blue'], C['orange'], C['aqua'], C['yellow']]
    for ax, g, title in zip(axes, groups, ('image, horizontal disc sweep', 'image, vertical disc sweep',
                                           'text-given value and RefCOCO')):
        for name, c in zip(g, cols):
            ax.plot(CLOCK_LAYERS, [max(-1, res[name][l]) for l in CLOCK_LAYERS], '-o', ms=3, lw=2, color=c, label=name)
        ax.axhline(0, color=C['gray'], lw=1)
        ax.axvline(20, color=C['gray'], lw=1, ls=':')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('layer (residual after the layer)')
        ax.legend(fontsize=7, frameon=False)
        style(ax)
    axes[0].set_ylabel('block-CV R^2 of cos/sin(2 pi x / 100)\n(clipped at -1)')
    fig.savefig(figs / 'clock_residual.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def transplant_section(run: Path, figs: Path, summ: dict) -> None:
    path = run / 'transplant' / 'transplant.jsonl'
    if not path.exists():
        print('transplant: missing', flush=True)
        return
    recs = read_jsonl(path)
    by = {}
    for r in recs:
        by.setdefault(r['cond'], []).append(r)
    out = {}
    for cond, rr in by.items():
        m = t2.digit_metrics(np.array([r['truth'] for r in rr]), np.array([r['logp0'] for r in rr]),
                             np.array([r['logp1'] for r in rr]))
        p1 = []
        for r in rr:
            lp1 = np.array(r['logp1'])
            p = np.exp(lp1 - lp1.max())
            p /= p.sum()
            p1.append(p[(int(round(r['truth'])) // 10) % 10])
        lo, hi = np.percentile([np.mean(np.random.default_rng(i).choice(p1, len(p1))) for i in range(1000)], [2.5, 97.5])
        out[cond] = {**m, 'p_second_ci': [float(lo), float(hi)]}
    summ['transplant'] = out
    layers = sorted({int(c.split('@')[1]) for c in out if c.startswith('full_q0@')})
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, key, lab in zip(axes, ('p_second', 'gain'), ('P(true second digit)', 'second-digit gain')):
        ax.plot(layers, [out[f'full_q0@{l}'][key] for l in layers], '-o', color=C['blue'], lw=2, label='text -> image, q0')
        ax.plot(layers, [out[f'full_pre3@{l}'][key] for l in layers], '-o', color=C['aqua'], lw=2,
                label='text -> image, q0 and 2 prefix tokens')
        ax.axhline(out['base_image'][key], color=C['gray'], ls='--', lw=1.2, label='image input, no transplant')
        ax.axhline(out['base_text'][key], color=C['black'], ls=':', lw=1.2, label='text-given input, no transplant')
        for name, mk, c in (('periodic', 's', C['orange']), ('p100', '^', C['yellow']), ('smooth', 'v', C['magenta']),
                            ('digits_onehot', 'D', C['gray']), ('diff16', 'P', C['black']), ('diff64', '*', C['blue']),
                            ('rand', 'x', C['green'])):
            pts = [(l, out[f'{name}_q0@{l}'][key]) for l in (20, 24) if f'{name}_q0@{l}' in out]
            if pts:
                ax.plot(*zip(*pts), mk, ms=8, color=c, ls='none', label=f'subspace only: {name}')
        ax.set_xlabel('layer after which the residual is replaced')
        ax.set_ylabel(lab)
        style(ax)
    axes[1].legend(fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.savefig(figs / 'transplant.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def textsrc_section(feat: Path, figs: Path, summ: dict) -> None:
    path = feat / 'textsrc.npz'
    if not path.exists():
        print('textsrc: missing', flush=True)
        return
    z = np.load(path, allow_pickle=True)
    fam, tr = z['family'], z['truth']
    layers = list(z['layers'])
    A = z['attn_groups'].item()
    out = {'attention': {}, 'head_clock': {}, 'post_clock': {}}
    names = ('image', 'target', 'px', 'py', 'text')
    for f in ('textconv', 'sweep_h'):
        s = fam == f
        out['attention'][f] = {int(l): {n: A[l][s][:, :, i].mean(0).tolist() for i, n in enumerate(names)} for l in layers}
    # per-head contribution at q0: block-CV clock decodability (text-given and image)
    Cn = z['contrib']                                        # (N, nL, heads, D)
    for f in ('textconv', 'sweep_h'):
        s = fam == f
        out['head_clock'][f] = {int(l): [rp.clock_decodability(Cn[s, li, h].astype(np.float32), tr[s])
                                         for h in range(Cn.shape[2])] for li, l in enumerate(layers)}
        out['head_ruler'] = out.get('head_ruler', {})
        out['head_ruler'][f] = {int(l): [rp.ruler_decodability(Cn[s, li, h].astype(np.float32), tr[s])
                                         for h in range(Cn.shape[2])] for li, l in enumerate(layers)}
    # post-image positions of text-given rows: align the px / py digit runs and the tail (identical tokens)
    groups = [json.loads(g) for g in z['groups']]
    st = np.flatnonzero(fam == 'textconv')
    PR, PL = z['post_resid'], z['post_len']
    rl = list(z['resid_layers'])
    offs = np.concatenate([[0], np.cumsum(PL)])
    sites = {}
    for j, i in enumerate(st):
        g = groups[i]
        e = g['image'][-1] + 1
        block = PR[offs[j]:offs[j + 1]]
        q0 = g['q0'][0]
        rec = {'px_last': block[g['px'][-1] - e], 'py_last': block[g['py'][-1] - e], 'after_py': block[g['py'][-1] + 1 - e]}
        for back in range(0, 40, 1):
            rec[f'q0-{back}'] = block[q0 - back - e]
        for k, v in rec.items():
            sites.setdefault(k, []).append(v)
    x = tr[st]
    px = np.round(x * 512 / 1000)
    for k, v in sites.items():
        V = np.stack(v)                                       # (n, nL, D)
        out['post_clock'][k] = {int(l): rp.clock_decodability(V[:, li].astype(np.float32), x) for li, l in enumerate(rl)}
        out['post_clock_px'] = out.get('post_clock_px', {})
        out['post_clock_px'][k] = {int(l): rp.block_cv_r2(V[:, li].astype(np.float32), rp.periodic_targets(px, 100),
                                                          np.floor(px / 100).astype(int)) for li, l in enumerate(rl)}
    summ['textsrc'] = out
    # figure: attention groups at q0 (head mean) and per-head clock decodability at layer 20
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    ax = axes[0]
    w = 0.35
    for i, (f, c) in enumerate((('textconv', C['blue']), ('sweep_h', C['orange']))):
        vals = [np.mean(out['attention'][f][l]['px']) + np.mean(out['attention'][f][l]['py']) for l in layers]
        img = [np.mean(out['attention'][f][l]['target']) for l in layers]
        ax.bar(np.arange(len(layers)) + (i - 0.5) * w, vals, w * 0.9, color=c, label=f'{f}: number tokens in text')
        ax.bar(np.arange(len(layers)) + (i - 0.5) * w, img, w * 0.9, bottom=vals, color=c, alpha=0.45,
               label=f'{f}: target image tokens')
    ax.set_xticks(range(len(layers)), [f'L{l}' for l in layers])
    ax.set_ylabel('attention mass at q0 (head mean)')
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = axes[1]
    for f, c in (('textconv', C['blue']), ('sweep_h', C['orange'])):
        ax.plot(range(16), np.maximum(-1, out['head_clock'][f][20]), 'o-', color=c, lw=1.5, label=f'{f}: clock')
    ax.axhline(0, color=C['gray'], lw=1)
    ax.set_xlabel('head of layer 20')
    ax.set_ylabel('block-CV R^2 of the head output at q0')
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = axes[2]
    for k, c in (('px_last', C['blue']), ('py_last', C['magenta']), ('after_py', C['aqua']), ('q0-0', C['orange'])):
        ax.plot(rl, [max(-1, out['post_clock'][k][l]) for l in rl], 'o-', color=c, lw=1.5, label=f'{k}: x (0-1000)')
    ax.plot(rl, [max(-1, out['post_clock_px']['px_last'][l]) for l in rl], 's--', color=C['blue'], lw=1,
            label='px_last: pixel value px')
    ax.axhline(0, color=C['gray'], lw=1)
    ax.set_xlabel('layer')
    ax.set_ylabel('block-CV R^2 of the period-100 code')
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.savefig(figs / 'textsrc.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def features_section(feat: Path, figs: Path, summ: dict) -> None:
    path = feat / 'features_base.npz'
    if not path.exists():
        print('features: missing', flush=True)
        return
    z = np.load(path, mmap_mode='r', allow_pickle=True)
    fam, tr, layers = z['family'], z['truth'], list(z['layers'])
    out = {}
    for kind in ('mlp', 'mix'):
        F = z[kind]
        for f in ('sweep_h', 'sweep_v', 'refcoco'):
            s = np.flatnonzero(fam == f)
            for site, sname in ((1, 'target'), (0, 'q0')):
                key = f'{kind}|{f}|{sname}'
                out[key] = {int(l): rp.clock_decodability(np.asarray(F[s, site, li], np.float32), tr[s])
                            for li, l in enumerate(layers)}
                out[key + '|ruler'] = {int(l): rp.ruler_decodability(np.asarray(F[s, site, li], np.float32), tr[s])
                                       for li, l in enumerate(layers)}
                print(key, ' '.join(f'{l}:{v:.2f}' for l, v in out[key].items()), flush=True)
    summ['reachability'] = out
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, f in zip(axes, ('sweep_h', 'sweep_v')):
        for (kind, site), c, ls in ((('mlp', 'target'), C['blue'], '-'), (('mix', 'target'), C['aqua'], '-'),
                                    (('mlp', 'q0'), C['orange'], '--'), (('mix', 'q0'), C['yellow'], '--')):
            v = out[f'{kind}|{f}|{site}']
            ax.plot(layers, [max(-1, v[l]) for l in layers], ls, marker='o', ms=3, lw=1.8, color=c,
                    label=f"{'MLP hidden' if kind == 'mlp' else 'mixer pre-output'} @ {site}")
        ax.axhline(0, color=C['gray'], lw=1)
        ax.set_title({'sweep_h': 'disc x', 'sweep_v': 'disc y'}[f], fontsize=10)
        ax.set_xlabel('layer')
        style(ax)
    axes[0].set_ylabel('block-CV R^2 of cos/sin(2 pi x / 100)\nfrom the component features (clipped at -1)')
    axes[1].legend(fontsize=7, frameon=False)
    fig.savefig(figs / 'reachability.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def firsttok_section(run: Path, figs: Path, summ: dict) -> None:
    path = run / 'firsttok' / 'firsttok.npz'
    if not path.exists():
        print('firsttok: missing', flush=True)
        return
    z = np.load(path)
    fam = z['family']
    out = {}
    for f in ('sweep_h', 'refcoco'):
        s = fam == f
        out[f] = {'norm_ratio': np.median(z['norm_ratio'][s], 0).tolist(), 'attn_first': z['attn_first'][s].mean(0).tolist(),
                  'attn_first_text': z['attn_first_text'][s].mean(0).tolist(),
                  'beta_first': z['beta_first'][s].mean(0).tolist(), 'beta_other': z['beta_other'][s].mean(0).tolist()}
    out['full_layers'] = z['full_layers'].tolist()
    out['beta_layers'] = z['beta_layers'].tolist()
    summ['firsttok'] = out
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for f, c in (('sweep_h', C['blue']), ('refcoco', C['orange'])):
        axes[0].plot(range(1, 33), out[f]['norm_ratio'], 'o-', ms=3, color=c, label=f)
        axes[1].plot(out['full_layers'], out[f]['attn_first'], 'o-', color=c, label=f'{f}: image queries')
        axes[1].plot(out['full_layers'], out[f]['attn_first_text'], 's--', color=c, label=f'{f}: text queries')
        axes[2].plot(out['beta_layers'], out[f]['beta_first'], 'o-', ms=3, color=c, label=f'{f}: first image token')
        axes[2].plot(out['beta_layers'], out[f]['beta_other'], ':', color=c, label=f'{f}: median other image token')
    axes[0].set_ylabel('norm / median image-token norm')
    axes[1].set_ylabel('attention to the first image token')
    axes[2].set_ylabel('DeltaNet write strength beta')
    for ax in axes:
        ax.set_xlabel('layer')
        ax.legend(fontsize=7, frameon=False)
        style(ax)
    axes[0].set_yscale('log')
    fig.savefig(figs / 'firsttok.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def grad_section(run: Path, figs: Path, summ: dict) -> None:
    path = run / 'grad' / 'grad.jsonl'
    if not path.exists():
        print('grad: missing', flush=True)
        return
    recs = read_jsonl(path)
    out = {}
    for loss in ('tens', 'hund'):
        rr = [r for r in recs if r['loss'] == loss]
        for key in ('fro', 'r8'):
            tab = rp.leverage_table(rr, key)
            agg = {}
            for k, v in tab.items():
                name, grp = k.split('|')
                layer = int(name.split('.')[0][1:])
                kind = 'mlp' if '.mlp.' in name else ('fa' if '.self_attn.' in name else 'dn')
                agg.setdefault((grp, kind), {}).setdefault(layer, 0.0)
                agg[(grp, kind)][layer] += v ** 2
            out[f'{loss}|{key}'] = {f'{g}|{k}': {l: float(np.sqrt(v)) for l, v in sorted(d.items())} for (g, k), d in agg.items()}
            out[f'{loss}|{key}|module'] = tab
    summ['grad'] = out
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax, grp in zip(axes, ('image', 'first', 'text')):
        for kind, c in (('mlp', C['blue']), ('dn', C['aqua']), ('fa', C['orange'])):
            d = out['tens|fro'].get(f'{grp}|{kind}', {})
            if d:
                ax.plot(list(d), list(d.values()), 'o-', ms=3, color=c, label=kind)
        ax.set_title(f'tokens: {grp}', fontsize=10)
        ax.set_xlabel('layer')
        ax.set_yscale('log')
        ax.legend(fontsize=7, frameon=False)
        style(ax)
    axes[0].set_ylabel('||grad of second-digit CE wrt W||_F\n(per component type, per layer)')
    fig.savefig(figs / 'grad_leverage.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--capture', required=True)
    p.add_argument('--figures', required=True)
    p.add_argument('--sections', default='margin,clock,transplant,textsrc,features,firsttok,grad')
    args = p.parse_args()
    run, feat, figs = Path(args.run), Path(args.features), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    spath = run / 'summary.json'
    summ = json.loads(spath.read_text()) if spath.exists() else {}
    secs = args.sections.split(',')
    if 'margin' in secs or 'clock' in secs:
        cap = np.load(Path(args.capture) / 'base.npz', allow_pickle=True)
        if 'margin' in secs:
            margin_section(cap, figs, summ)
        if 'clock' in secs:
            clock_section(cap, figs, summ)
    if 'transplant' in secs:
        transplant_section(run, figs, summ)
    if 'textsrc' in secs:
        textsrc_section(feat, figs, summ)
    if 'features' in secs:
        features_section(feat, figs, summ)
    if 'firsttok' in secs:
        firsttok_section(run, figs, summ)
    if 'grad' in secs:
        grad_section(run, figs, summ)
    spath.write_text(json.dumps(summ, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    print('wrote', spath, flush=True)


if __name__ == '__main__':
    main()
