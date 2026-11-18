"""Experiment 7.2 round 2 analysis (CPU): summary.json and figures.

Inputs: the round-2 stages of this run (restore, qswap, prompts, confirmatory training evaluations), the 3.16 round-2
captures (residual streams of the base model and of the 7.2 adapters at q0, q1, target tokens, the first image token
and question tokens, attention masses) and the 7.2 round-1 evaluations. Missing inputs of a requested part raise an
error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

from analysis_tools.position import readout_prior as rp
from analysis_tools.position import readout_prior_r2 as r2
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_r2 as t2
from analysis_tools.position.coordinate_readout_run import read_jsonl
from analysis_tools.position.coordinate_value_r2_figures import manifold_figure

C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
     'green': '#008300', 'gray': '#8a8a8a', 'black': '#222222'}
SITE = {'q0': 0, 'q1': 1, 'target': 2, 'first': 3, 'qmean': 4, 'qlast': 5}
CONDS = ('base', 'mlp1_4_img', 'mlp1_4_first', 'mlp1_4_text')
CCOL = {'base': C['gray'], 'mlp1_4_img': C['orange'], 'mlp1_4_first': C['blue'], 'mlp1_4_text': C['aqua']}


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def cap(feat: Path, cond: str):
    p = feat / 'capture' / f'{cond}.npz'
    if not p.exists():
        raise FileNotFoundError(f'{p} (3.16 round-2 capture stage)')
    return np.load(p)


def metrics(rr: list[dict]) -> dict:
    return t2.digit_metrics(np.array([r['truth'] for r in rr]), np.array([r['logp0'] for r in rr]),
                            np.array([r['logp1'] for r in rr]))


# ----------------------------------------------------------------------------- representation change by layer

def change_section(feat: Path, figs: Path, summ: dict) -> None:
    zb = cap(feat, 'base')
    layers = [int(v) for v in zb['layers']]
    m = np.where(zb['family'] == 'sweep_h')[0]
    x = zb['truth'][m]
    out = {'layers': layers}
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.6), sharex=True)
    for j, site in enumerate(('first', 'target', 'qmean', 'q0')):
        for cond in CONDS[1:]:
            z = cap(feat, cond)
            sp = [r2.change_split(zb['resid'][m, l, SITE[site]], z['resid'][m, l, SITE[site]]) for l in range(len(layers))]
            cl_ = [rp.clock_decodability(z['resid'][m, l, SITE[site]].astype(np.float64), x) for l in range(len(layers))]
            out[f'{cond}|{site}'] = {'const': [d['const'] for d in sp], 'vary': [d['vary'] for d in sp], 'clock': cl_}
            axes[0, j].plot(layers, [d['const'] for d in sp], color=CCOL[cond], ls='-', label=f'{cond}: constant offset')
            axes[0, j].plot(layers, [d['vary'] for d in sp], color=CCOL[cond], ls='--', label=f'{cond}: x-dependent part')
            axes[1, j].plot(layers, np.maximum(cl_, -1), color=CCOL[cond], marker='o', ms=2.5, label=cond)
        clb = [rp.clock_decodability(zb['resid'][m, l, SITE[site]].astype(np.float64), x) for l in range(len(layers))]
        out[f'base|{site}'] = {'clock': clb}
        axes[1, j].plot(layers, np.maximum(clb, -1), color=CCOL['base'], marker='o', ms=2.5, label='base')
        axes[0, j].set_title({'first': 'first image token', 'target': 'target image tokens (mean)',
                              'qmean': 'question tokens (mean)', 'q0': 'q0'}[site], fontsize=9)
        axes[1, j].axhline(0, color=C['gray'], lw=0.8)
        axes[1, j].set_xlabel('layer (output of)')
        for a in axes[:, j]:
            a.axvline(4, color=C['gray'], ls=':', lw=0.8)
            a.axvline(20, color=C['gray'], ls=':', lw=0.8)
            style(a)
    axes[0, 0].set_ylabel('|change| / |base residual|')
    axes[1, 0].set_ylabel('R$^2$ of cos/sin(2$\\pi$x/100)\n(leave-one-hundred-out, clipped)')
    axes[0, 1].legend(fontsize=6.5, frameon=False)
    axes[1, 1].legend(fontsize=6.5, frameon=False)
    fig.savefig(figs / 'change_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    summ['change'] = out


# ----------------------------------------------------------------------------- clock plane and manifolds

def clock_section(feat: Path, figs: Path, summ: dict) -> None:
    out = {}
    panels = [('target', 20), ('target', 24), ('q0', 24)]
    fig, axes = plt.subplots(len(panels), len(CONDS), figsize=(3.3 * len(CONDS) + 0.8, 3.2 * len(panels)))
    norm = mcolors.Normalize(0, 100)
    for i, (site, l) in enumerate(panels):
        for j, cond in enumerate(CONDS):
            z = cap(feat, cond)
            layers = [int(v) for v in z['layers']]
            m = np.where(z['family'] == 'sweep_h')[0]
            x = z['truth'][m]
            P, ang = r2.clock_plane(z['resid'][m, layers.index(l), SITE[site]].astype(np.float64), x)
            slope, rr = r2.circular_slope(ang, x)
            rad = np.linalg.norm(P, axis=1)
            out[f'{cond}|{site}|{l}'] = {'slope_deg_per_unit': slope, 'abs_r': rr, 'radius_mean': float(rad.mean()),
                                         'radius_cv': float(rad.std() / rad.mean())}
            ax = axes[i, j]
            ax.scatter(P[:, 0], P[:, 1], c=x % 100, cmap='viridis', norm=norm, s=10)
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=C['gray'], ls=':', lw=0.8))
            ax.set_aspect('equal')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_title(f'{cond}, {site} L{l}\nangle slope {slope:.2f} deg/unit, |r| {rr:.2f}', fontsize=8)
            if i == len(panels) - 1:
                ax.set_xlabel('held-out prediction of cos(2$\\pi$x/100)')
            if j == 0:
                ax.set_ylabel('held-out prediction of sin(2$\\pi$x/100)')
            style(ax)
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax)
    cb.set_label('x mod 100 (coordinate units)')
    fig.savefig(figs / 'clock_plane.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    summ['clock_plane'] = out


def manifold_section(feat: Path, figs: Path, summ: dict) -> None:
    info = {}
    for site, l in (('target', 20), ('target', 24), ('q0', 24)):
        for colour in ('mod100', 'x'):
            panels = []
            for cond in ('base', 'mlp1_4_img', 'mlp1_4_first', 'mlp1_4_text'):
                z = cap(feat, cond)
                layers = [int(v) for v in z['layers']]
                m = np.where(z['family'] == 'sweep_h')[0]
                x = z['truth'][m]
                H = z['resid'][m, layers.index(l), SITE[site]].astype(np.float64)
                H = H - H.mean(0)
                U, S, Vt = np.linalg.svd(H, full_matrices=False)
                ev = (S ** 2) / (S ** 2).sum()
                info[f'{cond}|{site}|{l}'] = ev[:6].tolist()
                panels.append({'P': U[:, :3] * S[:3], 'c': x % 100 if colour == 'mod100' else x, 'title': cond,
                               'subtitle': f'{site}, layer {l}; PC1-3 explain {ev[:3].sum():.2f}'})
            manifold_figure(panels, figs / f'manifold_{site}_L{l}_{colour}.png',
                            'x mod 100' if colour == 'mod100' else 'coordinate x', vmin=0,
                            vmax=100 if colour == 'mod100' else 1000,
                            suptitle=f'Qwen3.5-9B, horizontal disc sweep, residual at {site} after layer {l} (3D PCA)')
    summ['manifold_pc_variance'] = info


# ----------------------------------------------------------------------------- attention and question tokens

def attention_section(feat: Path, figs: Path, summ: dict) -> None:
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax, (fam, title) in zip(axes, (('sweep_h', 'image input (disc x)'), ('textconv', 'text-given input'),
                                        ('sweep_v', 'image input (disc y)'))):
        width = 0.2
        groups = None
        for j, cond in enumerate(CONDS):
            z = cap(feat, cond)
            groups = list(z['attn_groups'])
            al = [int(v) for v in z['attn_layers']]
            m = np.where(z['family'] == fam)[0]
            A = z['attn'][m]                                                    # (N, nA, 2, H, G)
            mass = A[:, al.index(20 if fam != 'sweep_v' else 24), 0].mean((0, 1))  # q0, head-mean
            out[f'{cond}|{fam}'] = {'layer': 20 if fam != 'sweep_v' else 24, 'q0_mass': dict(zip(groups, mass.tolist()))}
            for li, l in enumerate(al):
                out[f'{cond}|{fam}|L{l}'] = {'q0': dict(zip(groups, A[:, li, 0].mean((0, 1)).tolist())),
                                             'q1': dict(zip(groups, A[:, li, 1].mean((0, 1)).tolist()))}
            ax.bar(np.arange(len(groups)) + (j - 1.5) * width, mass, width, color=CCOL[cond], label=cond)
        ax.set_xticks(range(len(groups)), groups, rotation=30, fontsize=8)
        ax.set_title(f'{title}: q0 attention, layer {20 if fam != "sweep_v" else 24} (head mean)', fontsize=9)
        style(ax)
    axes[0].set_ylabel('attention mass')
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(figs / 'attention_groups.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    summ['attention'] = out


def qtok_section(feat: Path, figs: Path, summ: dict, tokenizer_words: list[str] | None = None) -> None:
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), sharey=True)
    for ax, cond in zip(axes, ('base', 'mlp1_4_text')):
        z = cap(feat, cond)
        if 'qtok' not in z.files:
            raise ValueError(f'capture of {cond} has no per-question-token residuals')
        m = np.where(z['family'] == 'sweep_h')[0]
        x = z['truth'][m]
        Q = z['qtok'][:len(np.where(z['family'] != 'textconv')[0])]           # rows in sweep order
        fam_sweep = z['family'][z['family'] != 'textconv']
        Q = Q[fam_sweep == 'sweep_h']
        ql = [int(v) for v in z['qtok_layers']]
        R = np.array([[rp.clock_decodability(Q[:, li, t].astype(np.float64), x) for t in range(Q.shape[2])]
                      for li in range(len(ql))])
        out[cond] = {'layers': ql, 'r2': R.tolist()}
        im = ax.imshow(np.clip(R, -1, 1), aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        ax.set_yticks(range(len(ql)), ql)
        ax.set_xlabel('question token index (0 = first token after the image)')
        ax.set_title(f'{cond}: period-100 code at each question token', fontsize=9)
    axes[0].set_ylabel('layer (output of)')
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label('leave-one-hundred-out R$^2$ (clipped)')
    fig.savefig(figs / 'question_tokens.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    summ['qtok'] = out


# ----------------------------------------------------------------------------- restore / swap / prompts

def restore_section(run: Path, figs: Path, summ: dict) -> None:
    recs = read_jsonl(run / 'restore' / 'restore.jsonl')
    by: dict = {}
    for r in recs:
        by.setdefault(r['adapter'], {}).setdefault(r['config'], []).append(r)
    out = {a: {c: {f: metrics([r for r in rr if r['family'] == f]) for f in ('sweep_h', 'sweep_v')} for c, rr in cs.items()}
           for a, cs in by.items()}
    summ['restore'] = out
    adapters = list(out)
    fig, axes = plt.subplots(1, len(adapters), figsize=(5.2 * len(adapters), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    cols = {'system': C['green'], 'question': C['aqua'], 'first': C['blue'], 'image_nofirst': C['magenta'], 'image': C['orange'], 'target': C['yellow']}
    for ax, a in zip(axes, adapters):
        sites = sorted({c.split('@')[0] for c in out[a] if '@' in c})
        for site in sites:
            ls_ = sorted(int(c.split('@')[1]) for c in out[a] if c.startswith(site + '@'))
            ax.plot(ls_, [out[a][f'{site}@{l}']['sweep_h']['p_second'] for l in ls_], color=cols[site], marker='o', ms=3,
                    label=f'restore {site}')
        ax.axhline(out[a]['none']['sweep_h']['p_second'], color=C['black'], ls='--', lw=1, label='adapter, no restoration')
        ax.axhline(0.15, color=C['gray'], ls=':', lw=1, label='base model')
        ax.set_title(a, fontsize=9)
        ax.set_xlabel('layer after which the group is restored to its base value')
        style(ax)
    axes[0].set_ylabel('P(true second digit), disc x')
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(figs / 'restore_layers.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def qswap_section(run: Path, figs: Path, summ: dict) -> None:
    recs = read_jsonl(run / 'qswap' / 'qswap.jsonl')
    by: dict = {}
    for r in recs:
        by.setdefault(r['adapter'], {}).setdefault(r['config'], []).append(r)
    out = {a: {c: r2.donor_shift(np.array([r['truth'] for r in rr]), np.array([r['donor_truth'] for r in rr]),
                                 np.array([r['logp1'] for r in rr])) for c, rr in cs.items()} for a, cs in by.items()}
    summ['qswap'] = out
    fig, axes = plt.subplots(1, len(out), figsize=(5.2 * len(out), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, out):
        for site, col in (('question', C['aqua']), ('first', C['blue']), ('image_nofirst', C['magenta'])):
            ls_ = sorted(int(c.split('@')[1]) for c in out[a] if c.startswith(site + '@'))
            if not ls_:
                continue
            ax.plot(ls_, [out[a][f'{site}@{l}']['b_donor'] for l in ls_], color=col, marker='o', ms=3, label=f'{site}: donor slope')
            ax.plot(ls_, [out[a][f'{site}@{l}']['b_own'] for l in ls_], color=col, ls='--', marker='s', ms=3, label=f'{site}: own slope')
        ax.set_title(a, fontsize=9)
        ax.set_xlabel('layer after which the group is taken from the donor input')
        style(ax)
    axes[0].set_ylabel('slope of E[s] on the in-bin offset')
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(figs / 'donor_swap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def prompts_section(run: Path, round1: Path, summ: dict) -> None:
    out = {}
    for d in sorted((run / 'eval').glob('*/prompts.jsonl')):
        cond = d.parent.name
        recs = read_jsonl(d)
        orig = {r['id']: r for r in read_jsonl(round1 / 'eval' / cond / 'coord.jsonl')} if cond != 'base' or \
            (round1 / 'eval' / 'base' / 'coord.jsonl').exists() else {}
        ids = sorted({r['id'] for r in recs})
        res = {}
        for variant in ('orig', 'para', 'zh', 'qfirst'):
            rr = [orig[i] for i in ids if i in orig] if variant == 'orig' else [r for r in recs if r['variant'] == variant]
            for fam in ('sweep_h', 'grid20', 'chart_ui'):
                ff = [r for r in rr if r['family'] == fam]
                if not ff:
                    continue
                err = [r['error'] for r in ff if r.get('valid') and r.get('error') is not None]
                res[f'{variant}|{fam}'] = {'n': len(ff), 'valid': len(err), 'mean_err': float(np.mean(err)) if err else None,
                                           'hit': float(np.mean([bool(r.get('hit')) for r in ff]))}
        out[cond] = res
    summ['prompts'] = out


def confirm_section(run: Path, summ: dict) -> None:
    out = {}
    for d in sorted((run / 'eval').glob('*')):
        cond = d.name
        res = {}
        if (d / 'digits.jsonl').exists():
            rr = read_jsonl(d / 'digits.jsonl')
            res['digits'] = {f: metrics([r for r in rr if r['family'] == f]) for f in ('sweep_h', 'sweep_v', 'refcoco')}
        if (d / 'coord.jsonl').exists():
            rr = read_jsonl(d / 'coord.jsonl')
            fams = sorted({r['family'] for r in rr})
            res['coord'] = {}
            for fam in fams:
                ff = [r for r in rr if r['family'] == fam and r['fmt'] == 'point']
                err = [r['error'] for r in ff if r.get('valid') and r.get('error') is not None]
                if ff:
                    res['coord'][fam] = {'n': len(ff), 'valid': len(err), 'mean_err': float(np.mean(err)) if err else None,
                                         'hit': float(np.mean([bool(r.get('hit')) for r in ff]))}
        if (d / 'ordinal.jsonl').exists():
            rr = read_jsonl(d / 'ordinal.jsonl')
            res['ordinal'] = {'correct': int(sum(r['correct'] for r in rr)), 'n': len(rr)}
        if res:
            out[cond] = res
    summ['confirm'] = out


PARTS = ('change', 'clock', 'manifold', 'attention', 'qtok', 'restore', 'qswap', 'prompts', 'confirm')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--round1-run', required=True)
    p.add_argument('--rp2-run', required=True)
    p.add_argument('--rp2-features', required=True)
    p.add_argument('--capture72', required=True)
    p.add_argument('--figures', required=True)
    p.add_argument('--parts', default=','.join(PARTS))
    args = p.parse_args()
    run, feat, figs = Path(args.run), Path(args.rp2_features), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    sp = run / 'summary.json'
    summ = json.loads(sp.read_text()) if sp.exists() else {}
    for part in args.parts.split(','):
        print(f'analysis part {part}', flush=True)
        {'change': lambda: change_section(feat, figs, summ), 'clock': lambda: clock_section(feat, figs, summ),
         'manifold': lambda: manifold_section(feat, figs, summ), 'attention': lambda: attention_section(feat, figs, summ),
         'qtok': lambda: qtok_section(feat, figs, summ), 'restore': lambda: restore_section(run, figs, summ),
         'qswap': lambda: qswap_section(run, figs, summ), 'prompts': lambda: prompts_section(run, Path(args.round1_run), summ),
         'confirm': lambda: confirm_section(run, summ)}[part]()
        sp.write_text(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
