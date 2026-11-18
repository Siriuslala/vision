"""Experiment 3.16 round 2 analysis (CPU): summary.json and figures.

Inputs: the round-2 GPU stages (capture, mismatch, layers, steer, grad2, vlcapture), the 3.16 round-1 run (gradient
records, margin decomposition) and, as a fallback for the spectrum, the 7.2 captures. Requested parts whose inputs are
missing raise an error; nothing is replaced by defaults.
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
from analysis_tools.position import targeted_finetune_r2 as t2
from analysis_tools.position.coordinate_readout_run import read_jsonl

C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
     'green': '#008300', 'gray': '#8a8a8a', 'black': '#222222'}
PERIOD_LABELS = ['1000', '500', '200', '100', '50', '33.3', '25', '20', '10']
ROUND1 = 'analysis/position/2026-11-05_readout_prior'
SITE = {'q0': 0, 'q1': 1, 'target': 2, 'first': 3, 'qmean': 4, 'qlast': 5}


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def load_capture(feat: Path, cond: str):
    p = feat / 'capture' / f'{cond}.npz'
    if not p.exists():
        raise FileNotFoundError(f'{p} (run the capture stage)')
    return np.load(p)


def rows(z, fam: str) -> np.ndarray:
    return np.where(z['family'] == fam)[0]


# ----------------------------------------------------------------------------- spectrum

def spectrum_section(feat: Path, figs: Path, summ: dict) -> None:
    zb, zi = load_capture(feat, 'base'), load_capture(feat, 'mlp1_4_img')
    layers = [int(v) for v in zb['layers']]
    panels = [('image input, target tokens (base)', zb, 'sweep_h', 'target'),
              ('image input, q0 (base)', zb, 'sweep_h', 'q0'),
              ('text-given input, q0 (base)', zb, 'textconv', 'q0'),
              ('text-given input, q1 (base)', zb, 'textconv', 'q1'),
              ('image input, target tokens (mlp1_4_img)', zi, 'sweep_h', 'target'),
              ('image input, q0 (mlp1_4_img)', zi, 'sweep_h', 'q0')]
    summ['spectrum'] = {'periods': list(r2.SPECTRUM_PERIODS), 'layers': layers}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True, sharey=True)
    for ax, (title, z, fam, site) in zip(axes.ravel(), panels):
        idx = rows(z, fam)
        x = z['truth'][idx]
        S = np.array([r2.spectrum(z['resid'][idx, l, SITE[site]].astype(np.float64), x) for l in range(len(layers))]).T
        summ['spectrum'][title] = S.tolist()
        im = ax.imshow(np.clip(S, -1, 1), aspect='auto', cmap='viridis', vmin=-1, vmax=1, origin='upper')
        ax.set_title(title, fontsize=9)
        ax.set_yticks(range(len(PERIOD_LABELS)), PERIOD_LABELS)
        ax.set_xticks(range(0, len(layers), 4), [layers[i] for i in range(0, len(layers), 4)])
        ax.axvline(layers.index(20), color='w', ls=':', lw=0.8)
    for ax in axes[:, 0]:
        ax.set_ylabel('period T (coordinate units)')
    for ax in axes[-1]:
        ax.set_xlabel('layer (output of)')
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label('leave-one-hundred-out R$^2$ of cos/sin(2$\\pi$x/T) (clipped to [-1, 1])')
    fig.savefig(figs / 'spectrum.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- digit decoding

def decoding_section(feat: Path, figs: Path, summ: dict) -> None:
    zb, zi = load_capture(feat, 'base'), load_capture(feat, 'mlp1_4_img')
    layers = [int(v) for v in zb['layers']]
    sets = [('text-given q0 (base)', zb, 'textconv', 'q0', C['blue'], '-'),
            ('text-given q1 (base)', zb, 'textconv', 'q1', C['blue'], '--'),
            ('image q0 (base)', zb, 'sweep_h', 'q0', C['orange'], '-'),
            ('image q1 (base)', zb, 'sweep_h', 'q1', C['orange'], '--'),
            ('image q0 (mlp1_4_img)', zi, 'sweep_h', 'q0', C['aqua'], '-'),
            ('image q1 (mlp1_4_img)', zi, 'sweep_h', 'q1', C['aqua'], '--')]
    sel = [l for l in layers if l % 2 == 0 or l in (19, 21)]
    summ['decoding'] = {'layers': sel}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for name, z, fam, site, col, ls in sets:
        idx = rows(z, fam)
        x = z['truth'][idx]
        blk = [r2.digit_decoding(z['resid'][idx, l, SITE[site]].astype(np.float64), x, 1, True) for l in sel]
        inter = [r2.digit_decoding(z['resid'][idx, l, SITE[site]].astype(np.float64), x, 1, False) for l in sel]
        summ['decoding'][name] = {'block': blk, 'interleaved': inter}
        axes[0].plot(sel, blk, color=col, ls=ls, marker='o', ms=3, label=name)
        axes[1].plot(sel, inter, color=col, ls=ls, marker='o', ms=3, label=name)
    for ax, t in zip(axes, ('whole hundreds held out (one readout for all hundreds)', 'interleaved folds')):
        ax.axhline(0.1, color=C['gray'], ls=':', lw=1)
        ax.axvline(20, color=C['gray'], ls=':', lw=0.8)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel('layer (output of)')
        style(ax)
    axes[0].set_ylabel('tens-digit accuracy of linear readout + arg max')
    axes[1].legend(fontsize=7, frameon=False, loc='upper left')
    fig.savefig(figs / 'digit_decoding.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- simulated readouts (derivation)

def simulation_section(work: Path, figs: Path, summ: dict) -> None:
    m = json.loads((work / ROUND1 / 'summary.json').read_text())['margin']['sweep_h']
    slope = m['slope_within']
    bias_t = np.array([m['bias'][str(k)] for k in range(2, 9)]) / slope            # measured per-hundred bias (tens)
    noise_t = m['sd_within'] / slope
    x = np.arange(100, 1000, 0.5) + 0.25          # avoid sampling exactly on digit boundaries
    s = (np.floor(x).astype(int) // 10) % 10
    sims = {'monotone': r2.simulate_codes(x)['monotone'],
            'conditional (measured bias)': None, 'circle': r2.simulate_codes(x, noise=0.0)['circle']}
    # conditional code with the measured per-hundred bias of the base model (hundreds 2-8) and its within-bin noise
    k = np.floor(x).astype(int) // 100
    rng = np.random.default_rng(0)
    c = np.zeros(10)
    c[2:9] = bias_t
    o_hat = (x - 100 * k) / 10 + c[k] + rng.normal(0, noise_t, len(x))
    sims['conditional (measured bias)'] = np.clip(np.floor(o_hat), 0, 9).astype(int)
    noisy_circle = r2.simulate_codes(x, noise=0.3)['circle']
    summ['simulation'] = {name: {'accuracy': float((p == s).mean()), 'changes': r2.argmax_changes(p)}
                          for name, p in {**sims, 'circle (noise 0.3 tens)': noisy_circle}.items()}
    summ['simulation']['true_changes'] = r2.argmax_changes(s)
    summ['simulation']['measured_bias_tens'] = bias_t.tolist()
    summ['simulation']['measured_noise_tens'] = float(noise_t)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4), sharey=True)
    sel = (x >= 200) & (x < 500)
    for ax, (name, p) in zip(axes, sims.items()):
        ax.plot(x[sel], s[sel], color=C['gray'], lw=2.5, alpha=0.5, label='true tens digit s')
        ax.plot(x[sel], p[sel], color=C['blue'], lw=1, label='arg max of the linear readout')
        acc = summ['simulation'][name]['accuracy']
        ax.set_title(f'{name}: accuracy {acc:.2f}', fontsize=9)
        ax.set_xlabel('coordinate x')
        style(ax)
    axes[0].set_ylabel('second digit')
    axes[0].legend(fontsize=7, frameon=False, loc='upper right')
    fig.savefig(figs / 'derivation_simulation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- text-path validity

def validity_section(run: Path, feat: Path, figs: Path, summ: dict) -> None:
    recs = read_jsonl(run / 'mismatch' / 'mismatch.jsonl')
    out = {}
    for cond in sorted({r['cond'] for r in recs}):
        rr = [r for r in recs if r['cond'] == cond]
        out[cond] = r2.donor_shift(np.array([r['truth'] for r in rr]), np.array([r['donor_truth'] for r in rr]),
                                   np.array([r['logp1'] for r in rr]))
        out[cond].update(t2.digit_metrics(np.array([r['truth'] for r in rr]), np.array([r['logp0'] for r in rr]),
                                          np.array([r['logp1'] for r in rr])))
    summ['mismatch'] = out
    J = np.load(feat / 'jacobian_L20.npz')
    summ['jacobian'] = {}
    for a, b in (('text', 'image'), ('image', 'image_fixed'), ('text', 'image_fixed')):
        for role, rn in ((1, 'second'), (0, 'first')):
            summ['jacobian'][f'{a}|{b}|{rn}'] = r2.jacobian_alignment(J[a][:, role], J[b][:, role])
    zb, zi = load_capture(feat, 'base'), load_capture(feat, 'mlp1_4_img')
    layers = [int(v) for v in zb['layers']]
    it, ih = rows(zb, 'textconv'), rows(zb, 'sweep_h')
    xt, xh = zb['truth'][it], zb['truth'][ih]
    cross = {'layers': layers}
    for site in ('q0', 'q1'):
        cross[site] = [r2.cross_decoding(zb['resid'][it, l, SITE[site]].astype(np.float64), xt,
                                         zi['resid'][ih, l, SITE[site]].astype(np.float64), xh) for l in range(len(layers))]
        cross[site + '_self_text'] = [rp.clock_decodability(zb['resid'][it, l, SITE[site]].astype(np.float64), xt)
                                      for l in range(len(layers))]
        cross[site + '_self_fixed'] = [rp.clock_decodability(zi['resid'][ih, l, SITE[site]].astype(np.float64), xh)
                                       for l in range(len(layers))]
    summ['cross_decoding'] = cross
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    ax = axes[0]
    names = [c for c in ('same_q0@20', 'donor_q0@20', 'same_q0@24', 'donor_q0@24') if c in out]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [out[c]['b_own'] for c in names], 0.4, color=C['orange'], label='slope on own offset')
    ax.bar(xs + 0.2, [out[c]['b_donor'] for c in names], 0.4, color=C['blue'], label='slope on donor offset')
    ax.set_xticks(xs, names, fontsize=8)
    ax.set_ylabel('least-squares slope of E[s]')
    ax.set_title('transplanted text-given q0: own vs donor coordinate', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = axes[1]
    keys = [k for k in summ['jacobian'] if k.endswith('second')]
    ax.bar(np.arange(len(keys)) - 0.2, [summ['jacobian'][k]['matched_cos'] for k in keys], 0.4, color=C['blue'],
           label='matched inputs')
    ax.bar(np.arange(len(keys)) + 0.2, [summ['jacobian'][k]['overlap'] for k in keys], 0.4, color=C['aqua'],
           label='top-9 subspace overlap')
    ax.axhline(summ['jacobian'][keys[0]]['random_overlap'], color=C['gray'], ls=':', lw=1)
    ax.set_xticks(range(len(keys)), [k.replace('|second', '').replace('|', ' vs ') for k in keys], fontsize=8)
    ax.set_ylabel('cosine / mean squared cosine')
    ax.set_title('readout Jacobian d(second-digit logits)/d h$_{q0}^{(20)}$', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = axes[2]
    for site, col in (('q0', C['blue']), ('q1', C['orange'])):
        ax.plot(layers, np.maximum(cross[site], -1), color=col, marker='o', ms=3, label=f'{site}: text-fitted readout on fixed image')
        ax.plot(layers, np.maximum(cross[site + '_self_fixed'], -1), color=col, ls='--', lw=1, label=f'{site}: fixed image, own readout')
    ax.axhline(0, color=C['gray'], lw=0.8)
    ax.set_xlabel('layer (output of)')
    ax.set_ylabel('R$^2$ of cos/sin(2$\\pi$x/100) (clipped at -1)')
    ax.set_title('cross-decoding: base text-given -> mlp1_4_img image', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.savefig(figs / 'text_validity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- gradients by layer

def _comb(acc: dict, layer: int, mods: list[str], grp: str) -> float:
    return float(np.sqrt(sum(np.mean(acc[f'L{layer}.{m}|{grp}']) ** 2 for m in mods if f'L{layer}.{m}|{grp}' in acc)))


def grad_section(work: Path, run: Path, figs: Path, summ: dict) -> None:
    recs = [r for r in read_jsonl(work / ROUND1 / 'grad' / 'grad.jsonl') if r['loss'] == 'tens']
    acc: dict = {}
    for r in recs:
        for k, v in r['fro'].items():
            acc.setdefault(k, []).append(v)
    M = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
    first = [_comb(acc, l, M, 'first') for l in range(1, 33)]
    other = [_comb(acc, l, M, 'image') for l in range(1, 33)]
    summ['grad_layers'] = {'first_mlp': first, 'other_image_mlp': other,
                           'ratio': [f / o if o > 0 else None for f, o in zip(first, other)]}
    g2 = read_jsonl(run / 'grad2' / 'grad2.jsonl')
    res: dict = {}
    for r in g2:
        for k, v in r['resid'].items():
            res.setdefault(k, []).append(v)
    groups = ('system', 'first', 'image', 'question', 'template')
    rg = {g: [float(np.sqrt(np.mean(np.square(res[f'L{l}|{g}'])))) for l in range(1, 33)] for g in groups}
    rgn = {g: [float(np.mean(res[f'L{l}|{g}|n'])) for l in range(1, 33)] for g in groups}
    acc2: dict = {}
    for r in g2:
        for k, v in r['fro'].items():
            acc2.setdefault(k, []).append(v)
    mlp2 = {g: [_comb(acc2, l, M, g) for l in range(1, 33)] for g in groups}
    summ['grad2'] = {'resid': rg, 'resid_ntokens': rgn, 'mlp': mlp2, 'batches': len(g2)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    L = np.arange(1, 33)
    ax = axes[0]
    ax.semilogy(L, first, color=C['orange'], marker='o', ms=3, label='first image token (1 token)')
    ax.semilogy(L, other, color=C['blue'], marker='o', ms=3, label='other image tokens (255 tokens, combined)')
    ax.set_xlabel('layer')
    ax.set_ylabel('|gradient of MLP weights| (Frobenius)')
    ax.set_title('second-digit CE: MLP weight gradient by token group', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = axes[1]
    ax.plot(L, summ['grad_layers']['ratio'][:31] + [np.nan], color=C['black'], marker='o', ms=3)
    ax.axhline(1, color=C['gray'], ls=':', lw=1)
    ax.set_xlabel('layer')
    ax.set_ylabel('first / other-image ratio')
    ax.set_yscale('log')
    ax.set_title('ratio of the two curves', fontsize=9)
    style(ax)
    ax = axes[2]
    cols = {'system': C['gray'], 'first': C['orange'], 'image': C['blue'], 'question': C['aqua'], 'template': C['magenta']}
    for g in groups:
        ax.semilogy(L, rg[g], color=cols[g], marker='o', ms=3, label=g)
    ax.set_xlabel('layer (output of)')
    ax.set_ylabel('|dL/dh| summed over the group (L2)')
    ax.set_title('residual-stream gradient by token group', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.savefig(figs / 'grad_layers.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- adapter decomposition / steer

def _metrics(rr: list[dict]) -> dict:
    return t2.digit_metrics(np.array([r['truth'] for r in rr]), np.array([r['logp0'] for r in rr]),
                            np.array([r['logp1'] for r in rr]))


def layers_section(run: Path, figs: Path, summ: dict) -> None:
    recs = read_jsonl(run / 'layers' / 'layers.jsonl')
    out: dict = {}
    for r in recs:
        out.setdefault(r['adapter'], {}).setdefault(r['config'], []).append(r)
    summ['adapter_layers'] = {a: {c: {f: _metrics([r for r in rr if r['family'] == f]) for f in ('sweep_h', 'sweep_v')}
                                  for c, rr in cs.items()} for a, cs in out.items()}
    adapters = list(summ['adapter_layers'])
    fig, axes = plt.subplots(1, len(adapters), figsize=(7 * len(adapters), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, adapters):
        cs = list(summ['adapter_layers'][a])
        xs = np.arange(len(cs))
        ax.bar(xs - 0.2, [summ['adapter_layers'][a][c]['sweep_h']['gain'] for c in cs], 0.4, color=C['blue'], label='gain (disc x)')
        ax.bar(xs + 0.2, [summ['adapter_layers'][a][c]['sweep_h']['p_second'] for c in cs], 0.4, color=C['orange'],
               label='P(true second digit) (disc x)')
        ax.set_xticks(xs, [c.replace('layers_', '').replace('modules_', 'mod ').replace('tokens_', 'tok ') for c in cs],
                      rotation=45, fontsize=7)
        ax.set_title(f'{a}: update kept only in the listed subset', fontsize=9)
        ax.axhline(0.34, color=C['gray'], ls=':', lw=1)
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(figs / 'adapter_layers.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def steer_section(run: Path, figs: Path, summ: dict) -> None:
    recs = read_jsonl(run / 'steer' / 'steer.jsonl')
    out: dict = {}
    for r in recs:
        out.setdefault(r['site'], {})[r['layer']] = {'metrics': r['metrics'], 'v_rel': r['v_rel'],
                                                      'loss_first': r['loss'][0], 'loss_last': float(np.mean(r['loss'][-5:]))}
    summ['steer'] = out
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    cols = {'first': C['orange'], 'image': C['blue'], 'question': C['aqua']}
    for ax, fam in zip(axes, ('sweep_h', 'sweep_v', 'refcoco')):
        for site, d in out.items():
            ls_ = sorted(d)
            ax.plot(ls_, [d[l]['metrics'][fam]['gain'] for l in ls_], color=cols[site], marker='o', ms=3, label=f'{site}: gain')
            ax.plot(ls_, [d[l]['metrics'][fam]['p_second'] for l in ls_], color=cols[site], ls='--', marker='s', ms=3,
                    label=f'{site}: P(true 2nd)')
        ax.set_title({'sweep_h': 'disc x (held-out half)', 'sweep_v': 'disc y (not fitted)', 'refcoco': 'RefCOCO x (not fitted)'}[fam],
                     fontsize=9)
        ax.set_xlabel('layer after which the vector is added')
        style(ax)
    axes[0].set_ylabel('second-digit gain / probability')
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(figs / 'steer_layers.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- first token / Qwen3-VL

def firsttok_section(feat: Path, summ: dict) -> None:
    z = load_capture(feat, 'base')
    idx = np.where(z['family'] != 'textconv')[0]
    H = z['resid'][idx, :, SITE['first']].astype(np.float64)                # (N, 33, D)
    dev = np.linalg.norm(H - H.mean(0), axis=2).max(0) / np.linalg.norm(H.mean(0), axis=1)
    summ['first_token_x_independence'] = {'max_rel_deviation_by_layer': dev.tolist()}


def qwen3vl_section(feat: Path, figs: Path, summ: dict) -> None:
    p = feat / 'qwen3vl_capture.npz'
    if not p.exists():
        raise FileNotFoundError(f'{p} (run vlcapture)')
    z = np.load(p)
    nL = int(z['n_layers'])
    out = {'layers': list(range(nL + 1))}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax, (fam, site, title) in zip(axes, (('sweep_h', 1, 'image, target tokens'), ('sweep_h', 0, 'image, q0'),
                                             ('textconv', 0, 'text-given, q0'))):
        idx = np.where(z['family'] == fam)[0]
        x = z['truth'][idx]
        S = np.array([r2.spectrum(z['resid'][idx, l, site].astype(np.float64), x) for l in range(nL + 1)]).T
        out[title] = S.tolist()
        im = ax.imshow(np.clip(S, -1, 1), aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        ax.set_title(f'Qwen3-VL-8B: {title}', fontsize=9)
        ax.set_yticks(range(len(PERIOD_LABELS)), PERIOD_LABELS)
        ax.set_xlabel('layer (output of)')
    axes[0].set_ylabel('period T')
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label('leave-one-hundred-out R$^2$ (clipped)')
    fig.savefig(figs / 'qwen3vl_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    for fam in ('sweep_h', 'sweep_v', 'textconv'):
        idx = np.where(z['family'] == fam)[0]
        out[f'digits_{fam}'] = t2.digit_metrics(z['truth'][idx], z['logp'][idx, 0], z['logp'][idx, 1])
    summ['qwen3vl'] = out


PARTS = ('spectrum', 'decoding', 'simulation', 'validity', 'grad', 'layers', 'steer', 'firsttok', 'qwen3vl')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--capture72', required=True)
    p.add_argument('--figures', required=True)
    p.add_argument('--model-name', default='Qwen3.5-9B')
    p.add_argument('--parts', default=','.join(PARTS))
    args = p.parse_args()
    run, feat, figs = Path(args.run), Path(args.features), Path(args.figures)
    work = run.parents[2]
    figs.mkdir(parents=True, exist_ok=True)
    sp = run / 'summary.json'
    summ = json.loads(sp.read_text()) if sp.exists() else {}
    for part in args.parts.split(','):
        print(f'analysis part {part}', flush=True)
        if part == 'spectrum':
            spectrum_section(feat, figs, summ)
        elif part == 'decoding':
            decoding_section(feat, figs, summ)
        elif part == 'simulation':
            simulation_section(work, figs, summ)
        elif part == 'validity':
            validity_section(run, feat, figs, summ)
        elif part == 'grad':
            grad_section(work, run, figs, summ)
        elif part == 'layers':
            layers_section(run, figs, summ)
        elif part == 'steer':
            steer_section(run, figs, summ)
        elif part == 'firsttok':
            firsttok_section(feat, summ)
        elif part == 'qwen3vl':
            qwen3vl_section(feat, figs, summ)
        else:
            raise ValueError(part)
        sp.write_text(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
