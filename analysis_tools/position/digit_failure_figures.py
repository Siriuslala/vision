"""Experiment 3.12 figures (reads results.json and analysis.npz written by digit_failure_analysis)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# Categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red).
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GREY = '#8a8985'
INK = '#0b0b0b'
DIGIT_CMAP = plt.get_cmap('viridis')
plt.rcParams.update({'figure.constrained_layout.use': True, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': INK, 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'legend.frameon': False})


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path.name, flush=True)


def digit_colorbar(fig, ax, label='digit'):
    sm = plt.cm.ScalarMappable(norm=Normalize(-0.5, 9.5), cmap=DIGIT_CMAP)
    cb = fig.colorbar(sm, ax=ax, ticks=range(10), fraction=0.04, pad=0.02)
    cb.set_label(label)
    return cb


# ----------------------------------------------------------------------------- 0. inputs

RUN: Path | None = None


def fig_inputs(A: dict, R: dict, figs: Path) -> None:
    from analysis_tools.position import digit_failure as df
    from analysis_tools.position import digit_failure_run as dr
    units = {u['id']: u for u in json.loads((RUN / 'factorial' / 'units.json').read_text())}
    picks = [('sweep_h_160', 'disc (sweep_h), x = 312.5'), ('blank_x', 'blank canvas (content-free prior)'),
             (next(i for i in units if i.startswith('rc_')), 'RefCOCO'), ('grid10_r02_c04', 'grid 10x10, target column 5'),
             ('count_23_0', 'count, n = 23')]
    fig, axes = plt.subplots(1, len(picks) + 1, figsize=(19, 3.6), gridspec_kw={'width_ratios': [1] * len(picks) + [1.4]})
    for ax, (uid, title) in zip(axes, picks):
        u = dict(units[uid])
        unit = df.Unit(**{k: (tuple(v) if k == 'ks' else v) for k, v in u.items()})
        ax.imshow(dr.unit_image(unit))
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(GREY)
        if uid.startswith('rc_'):
            ax.set_title('RefCOCO: ' + unit.prompt.split(':')[1].split('.')[0].strip(), fontsize=8)
    ax = axes[-1]
    ax.axis('off')
    text = ('Answer prefix (all point questions):\n```json\\n[\\n    {"point_2d": [  + forced k\n\n'
            'Pure-text position task (row shortened):\n' + df.text_position_prompt(37, 100)[:60] + '...\n'
            + 'o o o ... x (37th) ... o\n... what is the position of the x?\n\n'
            'Text-given pixel: "Its center is at pixel (160, 256)\nof this 512x512 image. Convert to 0-1000."')
    ax.text(0, 0.95, text, va='top', fontsize=7.5, family='monospace', wrap=True)
    save(fig, figs / 'inputs.png')


# ----------------------------------------------------------------------------- 1. theory

def fig_theory(A: dict, R: dict, figs: Path) -> None:
    x = A['toy_x']
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.6), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    panels = [('toy_hundreds_z_logits', 'toy_hundreds', 'hundreds digit, readout of z only'),
              ('toy_tens_z_logits', 'toy_tens', 'tens digit, readout of z only'),
              ('toy_tens_zk_logits', 'toy_tens', 'tens digit, readout of z and the emitted first digit k')]
    for j, (lk, tk, title) in enumerate(panels):
        lg = A[lk]
        ax = axes[0, j]
        for d in range(10):
            ax.plot(x, lg[:, d], color=DIGIT_CMAP(d / 9), lw=1.2)
        ax.plot(x, lg.max(1), color=INK, lw=0.8, ls=':', label='upper envelope (arg-max)')
        ax.set_title(title, fontsize=9)
        ax.set_ylabel('logit' if j == 0 else '')
        if j == 0:
            ax.legend(loc='upper left', fontsize=7)
        ax = axes[1, j]
        ax.scatter(x, A[tk], s=2, color=GREY, label='required digit')
        ax.scatter(x, lg.argmax(1), s=2, c=lg.argmax(1), cmap=DIGIT_CMAP, vmin=0, vmax=9, label='readout arg-max')
        acc = R['theory'][f"demo_{lk[4:-7]}_acc"]
        ax.set_title(f'accuracy {acc:.2f}', fontsize=8)
        ax.set_xlabel('true coordinate x (0-1000)')
        ax.set_ylabel('digit' if j == 0 else '')
        ax.set_yticks([0, 5, 9])
        if j == 0:
            ax.legend(loc='upper left', fontsize=7, markerscale=4)
    digit_colorbar(fig, axes[:, 2].tolist())
    save(fig, figs / 'theory.png')


# ----------------------------------------------------------------------------- 2. landscape

def _heat(ax, fig, t, P, label, ylabel='digit'):
    o = np.argsort(t)
    im = ax.imshow(P[o].T, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1,
                   extent=[t[o][0], t[o][-1], -0.5, 9.5], interpolation='nearest')
    ax.set_ylabel(ylabel)
    ax.set_yticks(range(10))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(label)
    return o


def fig_landscape(A: dict, R: dict, figs: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.2), gridspec_kw={'width_ratios': [1, 1, 0.55]}, layout='constrained')
    for r, (fam, name) in enumerate((('sweep_h', 'x'), ('sweep_v', 'y'))):
        t, P0, P1 = A[f'{fam}_truth'], A[f'{fam}_P0'], A[f'{fam}_P1true']
        ax = axes[r, 0]
        o = _heat(ax, fig, t, P0, 'probability')
        ax.plot(t[o], np.clip(np.round(t[o]) // 100, 0, 9), color='white', lw=1, ls='--', label='true first digit')
        ax.set_title(f'{name}: first digit P(d | image)', fontsize=9)
        ax.legend(loc='upper left', fontsize=7, labelcolor='white')
        ax = axes[r, 1]
        ok = np.isfinite(P1[:, 0])
        o = _heat(ax, fig, t[ok], P1[ok], 'probability')
        tt = t[ok][o]
        ax.scatter(tt, (np.round(tt) // 10) % 10, s=4, color='white', label='true second digit')
        ax.set_title(f'{name}: second digit P(d | image, true first digit)', fontsize=9)
        ax.legend(loc='upper left', fontsize=7, labelcolor='white', markerscale=2)
        ax = axes[r, 2]
        if f'{fam}_prior1' in A:
            pr = np.exp(A[f'{fam}_prior1'])
            im = ax.imshow(pr, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1,
                           extent=[-0.5, 9.5, 0.5, 9.5])
            ax.set_xlabel('second digit')
            ax.set_ylabel('forced first digit k')
            ax.set_title('blank canvas: P(second | k)', fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02).set_label('probability')
    for ax in axes[:, :2].ravel():
        ax.set_xlabel('true coordinate (0-1000)')
    save(fig, figs / 'landscape.png')


# ----------------------------------------------------------------------------- 3. clamp

CLAMP_FAMS = [('sweep_h', 'disc x (image)'), ('refcoco', 'RefCOCO x (image)'), ('count', 'count (image)'),
              ('textconv', 'pixel given in text'), ('textpos', 'text position (pure text)')]


def fig_clamp(A: dict, R: dict, figs: Path) -> None:
    fig, axes = plt.subplots(1, len(CLAMP_FAMS), figsize=(16, 3.4), sharey=True)
    for ax, (fam, title) in zip(axes, CLAMP_FAMS):
        o, E = A[f'{fam}_offset'], A[f'{fam}_E']
        m = (o > -30) & (o < 40)
        ax.scatter(o[m], E[m], s=3, color=C[0], alpha=0.35, label='model E[second digit]')
        g = np.linspace(-30, 40, 400)
        ax.plot(g, np.clip(np.floor(g), 0, 9), color=GREY, lw=1.5, label='magnitude-conditional ideal')
        ax.axvspan(0, 10, color='#f0efec', zorder=0, label='first digit correct')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('offset o = (truth - k w) / (w / 10)')
    axes[0].set_ylabel('expected second digit')
    axes[0].legend(loc='upper left', fontsize=7, markerscale=3)
    save(fig, figs / 'clamp.png')


# ----------------------------------------------------------------------------- 4. variance

VAR_FAMS = [('sweep_h', 'disc x'), ('sweep_v', 'disc y'), ('refcoco', 'RefCOCO x'), ('grid', 'grid x'),
            ('count', 'count'), ('textconv', 'text pixel'), ('textpos', 'text position')]


def fig_variance(A: dict, R: dict, figs: Path) -> None:
    V = R['variance']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.6))
    names = [n for f, n in VAR_FAMS]
    bottom = np.zeros(len(VAR_FAMS))
    for j, (key, lab) in enumerate((('image', 'image'), ('k', 'forced first digit k'), ('interaction', 'image x k'))):
        vals = np.array([V[f][key] for f, _ in VAR_FAMS])
        a1.bar(names, vals, bottom=bottom, color=C[j], label=lab, edgecolor='white', linewidth=1)
        bottom += vals
    a1.set_ylabel('share of second-digit logit variance')
    a1.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    a1.tick_params(axis='x', rotation=30)
    x = np.arange(len(VAR_FAMS))
    for j, (key, lab) in enumerate((('js_image_swap', 'swap image, keep k'), ('js_k_swap', 'swap k, keep image'))):
        m = np.array([V[f][key]['mean'] for f, _ in VAR_FAMS])
        lo = np.array([V[f][key]['ci'][0] for f, _ in VAR_FAMS])
        hi = np.array([V[f][key]['ci'][1] for f, _ in VAR_FAMS])
        a2.bar(x + (j - 0.5) * 0.38, m, 0.36, yerr=[m - lo, hi - m], color=C[j], label=lab, capsize=2)
    a2.axhline(V['ctx']['js_ctx_swap']['mean'], color=C[2], ls='--', lw=1.2, label='disc y: swap x in prefix (200/500/800)')
    a2.set_xticks(x, names, rotation=30)
    a2.set_ylabel('Jensen-Shannon divergence (bits)')
    a2.legend(fontsize=7)
    save(fig, figs / 'variance.png')


# ----------------------------------------------------------------------------- 5. evidence and sharpness law

LAW_FAMS = [('sweep_h', 'disc x', 'o'), ('sweep_v', 'disc y', 'o'), ('refcoco', 'RefCOCO x', 's'), ('grid', 'grid x', 'D'),
            ('count', 'count', '^'), ('textconv', 'text pixel', 'P'), ('textpos', 'text position', 'X')]


def fig_evidence(A: dict, R: dict, figs: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={'width_ratios': [0.6, 1.2, 1]})
    ax = axes[0]
    for j, fam in enumerate(('sweep_h', 'sweep_v')):
        ax.bar(np.arange(10) + (j - 0.5) * 0.38, np.exp(A[f'{fam}_prior0']), 0.36, color=C[j],
               label=f'blank canvas, {"x" if j == 0 else "y"}')
    ax.set_xticks(range(10))
    ax.set_xlabel('first digit')
    ax.set_ylabel('probability')
    ax.set_title('content-free prior', fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    t, E = A['sweep_h_E0_truth'], A['sweep_h_E0']
    o = np.argsort(t)
    Ec = E - E.mean(1, keepdims=True)
    im = ax.imshow(Ec[o].T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-8, vmax=8,
                   extent=[t[o][0], t[o][-1], -0.5, 9.5], interpolation='nearest')
    ax.plot(t[o], np.clip(np.round(t[o]) // 100, 0, 9), color=INK, lw=1, ls='--', label='true first digit')
    ax.scatter(t[o], np.argmax(E[o][:, 1:], 1) + 1, s=4, color=C[3], label='arg-max of evidence')
    ax.set_yticks(range(10))
    ax.set_xlabel('true x (0-1000)')
    ax.set_ylabel('first digit')
    ax.set_title('disc x: evidence log P(d | image) - log P(d | blank)', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label('evidence, centred (nats)')
    ax = axes[2]
    ax.plot(A['law_ratio'], A['law_p'], color=GREY, lw=1.5, label='Gaussian evidence, no prior')
    ax.axhline(0.1, color=GREY, ls=':', lw=1)
    Ev = R['evidence']
    for j, (fam, name, mk) in enumerate(LAW_FAMS):
        if fam not in Ev:
            continue
        for digit, fill in (('first', C[j % len(C)]), ('second', 'none')):
            e = Ev[fam][digit]
            if e.get('p_true') is None:
                continue
            ax.scatter(e['width'] / e['sigma'], e['p_true'], marker=mk, s=40, facecolors=fill,
                       edgecolors=C[j % len(C)], label=f'{name}' if digit == 'first' else None)
    ax.set_xscale('log')
    ax.set_xlabel('digit width / fitted readout resolution sigma')
    ax.set_ylabel('P(true digit)')
    ax.set_title('digit sharpness law (filled: first digit, open: second)', fontsize=9)
    ax.legend(fontsize=6.5, loc='upper left')
    save(fig, figs / 'evidence.png')


# ----------------------------------------------------------------------------- 5b. first-digit boundaries

def fig_boundaries(A: dict, R: dict, figs: Path) -> None:
    b = np.arange(200, 1000, 100)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for j, (fam, name) in enumerate((('sweep_h', 'x'), ('sweep_v', 'y'))):
        if f'{fam}_bound_raw_shift' not in A:
            continue
        axes[0].plot(b, A[f'{fam}_bound_raw_shift'], color=C[j], marker='o', ms=4, lw=1.5, label=f'{name}, raw')
        axes[0].plot(b, A[f'{fam}_bound_calibrated_shift'], color=C[j], marker='s', ms=4, lw=1.2, ls='--',
                     label=f'{name}, blank-calibrated')
        axes[1].plot(b, A[f'{fam}_bound_raw_width'], color=C[j], marker='o', ms=4, lw=1.5, label=name)
    axes[0].axhline(0, color=GREY, lw=0.8)
    axes[0].set_ylabel('boundary location - true boundary (units)')
    axes[0].set_title('where the first digit switches (k-1 -> k)', fontsize=9)
    axes[1].axhline(10, color=GREY, ls=':', lw=1, label='width of one second-digit bin')
    axes[1].set_ylabel('logistic transition width s (units)')
    axes[1].set_title('how sharply it switches', fontsize=9)
    for ax in axes:
        ax.set_xlabel('true hundreds boundary (0-1000)')
        ax.legend(fontsize=7)
    save(fig, figs / 'boundaries.png')


# ----------------------------------------------------------------------------- 6. decoding

def fig_decoding(A: dict, R: dict, figs: Path) -> None:
    variants = [('greedy', 'greedy', C[1]), ('blank_calibrated', 'blank-calibrated (alpha = 1)', C[0]),
                ('expected', 'expected value', C[2])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (fam, name) in zip(axes, (('sweep_h', 'x'), ('sweep_v', 'y'))):
        t = A[f'{fam}_dec_truth']
        o = np.argsort(t)
        ax.plot([100, 1000], [100, 1000], color=GREY, ls='--', lw=1, label='identity')
        for key, lab, col in variants:
            mae = R['decoding'][fam][key]['mae']['mean']
            ax.plot(t[o], A[f'{fam}_{key}'][o], color=col, lw=1.5, label=f'{lab} (MAE {mae:.1f})')
        ax.axvspan(400, 610, color='#f0efec', zorder=0, label='500 plateau region')
        ax.set_xlabel(f'true {name} (0-1000)')
        ax.set_title(f'disc {name}: emulated decoding at 10-unit resolution', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
    axes[0].set_ylabel('decoded value')
    save(fig, figs / 'decoding.png')


# ----------------------------------------------------------------------------- 7. DLA

def fig_dla(A: dict, R: dict, figs: Path) -> None:
    names = list(A['dla_names'])
    layers = np.arange(1, 33)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    ax = axes[0]
    for j, (key, lab) in enumerate((('image_var', 'image-driven variance'), ('k_var', 'first-digit-driven variance'))):
        v = A[f'sweep_h_q1_{key}']
        ax.plot(layers, v[:32], color=C[j], lw=1.5, ls='--', label=f'{lab}, token mixer')
        ax.plot(layers, v[32:64], color=C[j], lw=1.5, label=f'{lab}, MLP')
    ax.set_xlabel('layer')
    ax.set_ylabel('variance of direct contribution (nats$^2$)')
    ax.set_title('disc x, second-digit position q1', fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    for j, (role, lab) in enumerate((('q0', 'first digit at q0'), ('q1', 'second digit at q1'))):
        tuned = A[f'sweep_h_{role}_tuned']
        ax.plot(layers, tuned[:32] + tuned[32:64], color=C[j], lw=1.5, label=lab)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel('layer')
    ax.set_ylabel('logit toward the true digit (nats)')
    ax.set_title('image-driven contribution toward the truth (mixer + MLP)', fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[2]
    for j, key in enumerate(('k4_d9', 'k5_d0')):
        c = A[f'sweep_h_prior_{key}']
        ax.plot(layers, c[:32], color=C[j], ls='--', lw=1.5, label=f'{key}: token mixer')
        ax.plot(layers, c[32:64], color=C[j], lw=1.5, label=f'{key}: MLP')
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel('layer')
    ax.set_ylabel('extra logit of the prior digit (nats)')
    ax.set_title('who writes "4 -> 9" and "5 -> 0" at q1', fontsize=9)
    ax.legend(fontsize=7)
    save(fig, figs / 'dla.png')


# ----------------------------------------------------------------------------- 8. oracle

def fig_oracle(A: dict, R: dict, figs: Path) -> None:
    O = R['oracle']
    fams = [('sweep_h', 'disc x'), ('sweep_v', 'disc y'), ('count', 'count'), ('textconv', 'text pixel')]
    fig, axes = plt.subplots(1, len(fams), figsize=(15, 3.4), sharey=True)
    layers = [int(k[1:]) for k in O['sweep_h'] if k.startswith('L')]
    for ax, (fam, name) in zip(axes, fams):
        if fam not in O:
            continue
        for j, (key, lab) in enumerate((('ridge_coordinate', 'full-dim ridge on the value'), ('pc32', 'softmax on 32 PCs'),
                                        ('pc8', 'softmax on 8 PCs'), ('pc2', 'softmax on 2 PCs'))):
            ax.plot(layers, [O[fam][f'L{l}'][key]['acc'] for l in layers], color=C[j], marker='o', ms=3, lw=1.3, label=lab)
        ax.axhline(O[fam]['model']['acc'], color=INK, ls='--', lw=1, label='model output')
        ax.axhline(0.1, color=GREY, ls=':', lw=1, label='chance')
        ax.set_title(f'{name} (n = {O[fam]["n"]})', fontsize=9)
        ax.set_xlabel('layer')
    axes[0].set_ylabel('second-digit accuracy at q1')
    axes[0].legend(fontsize=6.5)
    save(fig, figs / 'oracle.png')


# ----------------------------------------------------------------------------- 9. grids

def fig_grid(A: dict, R: dict, figs: Path) -> None:
    sizes = [n for n in (6, 10, 20) if f'grid{n}_tx' in A]
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.2))
    for ax, n in zip(np.atleast_1d(axes), sizes):
        tx, ox, edges = A[f'grid{n}_tx'], A[f'grid{n}_ox'], A[f'grid{n}_edges']
        for e in edges:
            ax.axhline(e, color='#e1e0dc', lw=0.7, zorder=0)
            ax.axvline(e, color='#e1e0dc', lw=0.7, zorder=0)
        ax.plot([0, 1000], [0, 1000], color=GREY, ls='--', lw=1, label='identity')
        ax.scatter(tx, ox, s=14, color=C[0], label='model x output')
        ax.set_xlabel('true target x (0-1000)')
        ax.set_ylabel('output x')
        r = R['grid'][str(n)]
        ax.set_title(f'{n}x{n}: column +1 {r["col_plus1"]["p"]:.2f}, closer to a grid line {r["frac_closer_to_edge"]:.2f}',
                     fontsize=9)
        ax.legend(fontsize=7, loc='upper left', title='light lines: cell edges', title_fontsize=7)
    save(fig, figs / 'grid.png')


# ----------------------------------------------------------------------------- 10. attention

def fig_attention(A: dict, R: dict, figs: Path) -> None:
    if 'sweep_h_img_q0' not in A:
        return
    layers = [4, 8, 12, 16, 20, 24, 28, 32]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for j, fam in enumerate(('sweep_h', 'count', 'textconv')):
        if f'{fam}_img_q0' not in A:
            continue
        axes[0].plot(layers, A[f'{fam}_img_q0'].mean(1), color=C[j], ls='--', marker='o', ms=3, label=f'{fam}, q0')
        axes[0].plot(layers, A[f'{fam}_img_q1'].mean(1), color=C[j], marker='o', ms=3, label=f'{fam}, q1')
        axes[1].plot(layers, A[f'{fam}_dig_q1'].mean(1), color=C[j], marker='o', ms=3, label=fam)
    axes[0].set_ylabel('attention mass on image tokens (head mean)')
    axes[1].set_ylabel('attention mass of q1 on the first-digit token')
    for ax in axes:
        ax.set_xlabel('full-attention layer')
        ax.legend(fontsize=7)
    save(fig, figs / 'attention.png')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--figures', required=True)
    args = p.parse_args()
    global RUN
    run, figs = Path(args.run), Path(args.figures)
    RUN = run
    figs.mkdir(parents=True, exist_ok=True)
    R = json.loads((run / 'results.json').read_text())
    A = dict(np.load(run / 'analysis.npz', allow_pickle=True))
    for f in FIGURES:
        f(A, R, figs)


FIGURES = [fig_inputs, fig_theory, fig_landscape, fig_clamp, fig_variance, fig_evidence, fig_boundaries, fig_decoding, fig_dla, fig_oracle,
           fig_grid, fig_attention]

if __name__ == '__main__':
    main()
