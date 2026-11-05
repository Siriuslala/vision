"""Experiment 3.12 round 3: CPU analysis and figures.

Inputs: the round-3 GPU run (`digit_failure_r3.py`: units.jsonl, relay.jsonl, patch3.jsonl), the round-2 run
(arrays.npz, results.json, weights.npz) and nothing else. Outputs: results_r3.json in the round-3 run and the
figures cones.png, svd_angle.png, relay.png, units.png, units_sources.png, chain.png in the figure directory.

Conceptual figures (cones.png, chain.png) are simulations or diagrams and are labelled as such in the doc.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.position import digit_failure_figures as F
from analysis_tools.position.coordinate_readout_run import read_jsonl
from analysis_tools.position.digit_failure_r2_analysis import ratio_ci, recovery, to_jsonable

plt, C, GREY, INK, DIGIT_CMAP, save = F.plt, F.C, F.GREY, F.INK, F.DIGIT_CMAP, F.save
RNG = np.random.default_rng(0)
FAMS = (('sweep_h', 'disc x'), ('sweep_v', 'disc y'))


def softmax(lp: np.ndarray) -> np.ndarray:
    p = np.exp(lp - lp.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def expect(lp: np.ndarray) -> np.ndarray:
    return (softmax(lp) * np.arange(10)).sum(-1)


def lin_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and R^2 of a least-squares line."""
    b = np.polyfit(x, y, 1)
    r2 = 1 - ((y - np.polyval(b, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(b[0]), float(r2)


# ----------------------------------------------------------------------------- conceptual figures

def fig_cones(figs: Path) -> None:
    """Simulation: argmax regions of linear logits in a 2-D hidden space are cones; a curve visits them."""
    K = 6
    phases = np.radians(np.arange(K) * 360 / K + 15)
    W = np.column_stack([np.cos(phases), np.sin(phases)])                  # (K, 2) readout rows
    g = np.linspace(-1.2, 1.2, 401)
    X, Y = np.meshgrid(g, g)
    lab = (np.stack([X, Y], -1) @ W.T).argmax(-1)
    cols = [DIGIT_CMAP(c / (K - 1)) for c in range(K)]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    xs = np.linspace(0, 1, 600)
    curves = (('one turn (θ covers 340°)', 0.3 + 0.7 * xs, np.radians(-10 + 340 * xs)),
              ('three turns (θ covers 1080°)', 0.35 + 0.65 * xs, np.radians(-10 + 1080 * xs)))
    for row, (name, rho, th) in enumerate(curves):
        ax = axes[row, 0]
        ax.contourf(X, Y, lab, levels=np.arange(-0.5, K), colors=cols, alpha=0.28)
        for c in range(K):
            ax.annotate('', xy=0.55 * W[c], xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=cols[c], lw=1.6))
            ax.text(1.05 * np.cos(phases[c]), 1.05 * np.sin(phases[c]), f'digit {c}', ha='center', va='center',
                    fontsize=8, color=INK)
        if row == 0:                                                          # half-space boundaries of digit 0
            for d in (1, K - 1):
                nrm = W[0] - W[d]
                tng = np.array([-nrm[1], nrm[0]]) / np.linalg.norm(nrm)
                ax.plot([-1.2 * tng[0], 1.2 * tng[0]], [-1.2 * tng[1], 1.2 * tng[1]], color=INK, ls='--', lw=0.8)
            ax.text(-1.15, 1.17, 'dashed: boundaries of two half-spaces;\ntheir intersection = cone of digit 0',
                    fontsize=7, ha='left', va='top')
        n = np.column_stack([rho * np.cos(th), rho * np.sin(th)])
        sc = ax.scatter(n[:, 0], n[:, 1], c=xs, cmap='viridis', s=3)
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('hidden-state coordinate $n_1$')
        ax.set_ylabel('hidden-state coordinate $n_2$')
        ax.set_title(f'regions where each digit wins; curve: {name}', fontsize=9)
        fig.colorbar(sc, ax=ax, fraction=0.046, label='input coordinate x (a.u., 0-1)')
        L = n @ W.T
        ax = axes[row, 1]
        for c in range(K):
            ax.plot(xs, L[:, c], color=cols[c], lw=1.4, label=f'digit {c}')
        ax.plot(xs, L.max(1), color=INK, ls=':', lw=1.0, label='upper envelope')
        ax.set_xlabel('input coordinate x (a.u.)')
        ax.set_ylabel('logit $\\ell_c = w_c^\\top n$')
        ax.set_title('logits along the curve', fontsize=9)
        if row == 0:
            ax.legend(fontsize=7, ncol=2, frameon=False, loc='lower left')
        ax = axes[row, 2]
        ax.scatter(xs, L.argmax(1), c=[cols[c] for c in L.argmax(1)], s=4)
        ax.set_yticks(range(K))
        ax.set_xlabel('input coordinate x (a.u.)')
        ax.set_ylabel('emitted digit (argmax)')
        ax.set_title('one band per digit' if row == 0 else 'each digit recurs once per turn (sawtooth)', fontsize=9)
    save(fig, figs / 'cones.png')


def fig_svd_angle(A: dict, R: dict, figs: Path) -> None:
    """Model activations: SVD spectrum of the q0 logit trajectory and the angle construction."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    ax = axes[0]
    w = 0.38
    for j, (fam, label) in enumerate(FAMS):
        s = np.linalg.svd(centre2(A[f'{fam}_L0']), compute_uv=False)
        share = s ** 2 / (s ** 2).sum()
        ax.bar(np.arange(1, 7) + (j - 0.5) * w, share[:6], width=w, color=C[j], label=f'{label}, q0')
    q1 = np.mean([R['geometry'][f]['q1'][str(k)]['ev'][0] for f, _ in FAMS for k in range(1, 10)])
    ax.axhline(q1, color=C[2], ls='--', lw=1, label=f'q1 (other matrix, per forced k): first share, mean {q1:.2f}')
    ax.set_xlabel('singular direction j')
    ax.set_ylabel('variance share $\\sigma_j^2 / \\sum_i \\sigma_i^2$')
    ax.set_title('spectrum of the centred logit matrix $\\tilde L$', fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    fam = 'sweep_h'
    x, z1, z2 = A[f'{fam}_x'], A[f'{fam}_z1'], A[f'{fam}_z2']
    ax = axes[1]
    sc = ax.scatter(z1, z2, c=x, cmap='viridis', s=14, vmin=100, vmax=1000)
    i = int(np.argmin(np.abs(x - 430)))
    ax.plot([0, z1[i]], [0, z2[i]], color=INK, lw=1)
    ax.plot([0, 12], [0, 0], color=GREY, lw=0.8)
    a = np.arctan2(z2[i], z1[i])
    tt = np.linspace(0, a, 50) if a > 0 else np.linspace(0, a, 50)
    ax.plot(3 * np.cos(tt), 3 * np.sin(tt), color='red', lw=1.2)
    ax.text(3.6 * np.cos(a / 2), 3.6 * np.sin(a / 2), 'θ', color='red', fontsize=11)
    ax.text(z1[i] / 2 + 0.4, z2[i] / 2, 'ρ', fontsize=11)
    ax.scatter([z1[i]], [z2[i]], s=60, facecolor='none', edgecolor='red', lw=1.5, label=f'image with x = {x[i]:.0f}')
    ax.set_aspect('equal')
    ax.axhline(0, color=GREY, lw=0.4)
    ax.axvline(0, color=GREY, lw=0.4)
    ax.set_xlabel('$z_1$ (nats)')
    ax.set_ylabel('$z_2$ (nats)')
    ax.set_title('disc x: each image is a point $(z_1, z_2)$; θ = atan2($z_2$, $z_1$)', fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc='lower left')
    fig.colorbar(sc, ax=ax, fraction=0.046, label='true coordinate x (0-1000)')
    ax = axes[2]
    for j, (fam, label) in enumerate(FAMS):
        x, th = A[f'{fam}_x'], np.degrees(A[f'{fam}_theta'])
        r = np.hypot(A[f'{fam}_z1'], A[f'{fam}_z2'])
        keep = x >= 150
        ax.scatter(x[keep], th[keep] - th[keep].min(), s=9, color=C[j], label=f'{label} (x ≥ 150)')
        ax.scatter(x[~keep], th[~keep] - th[keep].min(), s=12, marker='x', color=GREY,
                   label='x < 150 (answers are often two-digit numbers)' if j == 0 else None)
    ax.set_xlabel('true coordinate x')
    ax.set_ylabel('unwrapped angle θ (degrees, from x = 150)')
    ax.set_title('the angle grows monotonically with x', fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    save(fig, figs / 'svd_angle.png')


def centre2(L: np.ndarray) -> np.ndarray:
    """Centre a logit matrix (N, 10) over digits (row means) and over images (column means)."""
    L = L - L.mean(-1, keepdims=True)
    return L - L.mean(0)


def angle_stats(A: dict) -> dict:
    """Angle statistics restricted to x >= 150, where the trajectory radius is large enough."""
    out = {}
    for fam, _ in FAMS:
        x, th = A[f'{fam}_x'], A[f'{fam}_theta']
        r = np.hypot(A[f'{fam}_z1'], A[f'{fam}_z2'])
        L = A[f'{fam}_L0']
        s = np.linalg.svd(centre2(L), compute_uv=False)
        keep = x >= 150
        slope = np.polyfit(x[keep], th[keep], 1)[0]
        o = np.argsort(x[keep])
        am = L[keep].argmax(1)[o]
        seq = [int(am[0])]
        for a in am[1:]:
            if a != seq[-1]:
                seq.append(int(a))
        out[fam] = {'singular_values': s[:5].round(2).tolist(), 'share': (s ** 2 / (s ** 2).sum())[:5].round(3).tolist(),
                    'n': int(len(x)), 'n_keep': int(keep.sum()),
                    'radius_small_lt150': float(r[~keep].max()), 'radius_min_keep': float(r[keep].min()),
                    'deg_per_100_keep': float(np.degrees(slope) * 100),
                    'angle_range_keep_deg': float(np.degrees(th[keep].max() - th[keep].min())),
                    'argmax_sequence_keep': seq}
    return out


# ----------------------------------------------------------------------------- adjacent difference vs offset

ADJ_FAMS = (('sweep_h', 'disc x'), ('sweep_v', 'disc y'), ('refcoco', 'RefCOCO x'), ('grid', '10x10 grid x'), ('textconv', 'text-given value'))


def adjdiff_decomposition(A: dict) -> dict:
    """Round-2 natural rows (true k forced, truths 200-899): how the tens-digit gain splits into two factors.

    m = l0(k+1) - l0(k-1) is the first-digit adjacent difference at q0, o the true offset in tens-digit units,
    E the expected tens digit at q1. If E depends on o only through m, the measured gain dE/do equals
    (dE/dm) * (dm/do); the scatter of m around its line in o, divided by dm/do, is the offset resolution
    that a reader of m can reach (in coordinate units, x10).
    """
    out = {}
    for fam, _ in ADJ_FAMS:
        m, o, E = A[f'margin_{fam}_d'], A[f'margin_{fam}_o'], A[f'margin_{fam}_E']
        b_mo = np.polyfit(o, m, 1)
        sd_m = float((m - np.polyval(b_mo, o)).std())
        dE_dm = float(np.polyfit(m, E, 1)[0])
        dE_do = float(np.polyfit(o, E, 1)[0])
        out[fam] = {'n': int(len(m)), 'dm_do': float(b_mo[0]), 'sd_m_given_o': sd_m, 'dE_dm': dE_dm, 'dE_do': dE_do,
                    'product': float(dE_dm * b_mo[0]), 'resolution_units': float(10 * sd_m / abs(b_mo[0])),
                    'r2_m_on_o': lin_fit(o, m)[1], 'r2_E_on_m': lin_fit(m, E)[1]}
    # The same slope / scatter for m computed from the rank-2 (arc-plane) reconstruction of the q0 logits.
    for fam, _ in FAMS:
        x, L, vt, s, mu = A[f'{fam}_x'], A[f'{fam}_L0'], A[f'{fam}_vt'], A[f'{fam}_s'], A[f'{fam}_mu']
        z = np.column_stack([A[f'{fam}_z1'], A[f'{fam}_z2']])
        rec = (z / s[:2]) @ (vt[:2] * s[:2, None]) + mu
        sel = (x >= 200) & (x < 900)
        k = (x[sel] // 100).astype(int)
        o = (x[sel] - 100 * k) / 10
        for name, M in (('full', L[sel]), ('rank2', rec[sel])):
            m = M[np.arange(len(k)), k + 1] - M[np.arange(len(k)), k - 1]
            b = np.polyfit(o, m, 1)
            sd = float((m - np.polyval(b, o)).std())
            out[fam][f'recomputed_{name}'] = {'n': int(len(k)), 'dm_do': float(b[0]), 'sd': sd,
                                              'resolution_units': float(10 * sd / b[0])}
    return out


def fig_adjdiff(A: dict, D: dict, figs: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for j, (fam, label) in enumerate(ADJ_FAMS[:3]):
        m, o, E = A[f'margin_{fam}_d'], A[f'margin_{fam}_o'], A[f'margin_{fam}_E']
        d = D[fam]
        axes[0].scatter(o, m, s=10, color=C[j], alpha=0.55,
                        label=f"{label}: slope {d['dm_do']:.2f} nats/unit, scatter SD {d['sd_m_given_o']:.1f} nats")
        xs = np.linspace(0, 10, 50)
        axes[0].plot(xs, np.polyval(np.polyfit(o, m, 1), xs), color=C[j], lw=1.6)
        axes[1].scatter(m, E, s=10, color=C[j], alpha=0.55, label=f"{label}: slope {d['dE_dm']:.2f}")
    axes[0].axhline(0, color=GREY, lw=0.6)
    axes[0].set_xlabel('true offset o inside the hundreds bin (tens-digit units, 0-10)')
    axes[0].set_ylabel('first-digit adjacent difference m at q0 (nats)')
    axes[0].set_title('m follows the offset on average but scatters widely', fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel('first-digit adjacent difference m at q0 (nats)')
    axes[1].set_ylabel('expected tens digit E[t] at q1')
    axes[1].set_title('the tens digit follows m', fontsize=9)
    axes[1].legend(fontsize=7, frameon=False)
    save(fig, figs / 'adjdiff.png')


# ----------------------------------------------------------------------------- relay routing

def relay_summary(run: Path) -> tuple[dict, dict]:
    """Share of the steering effect on E[t] at q1 that survives each restoration of q0 states."""
    recs = read_jsonl(run / 'relay.jsonl')
    if not recs:
        return {}, {}
    names = recs[0]['rows']
    dE = {n: [] for n in names}
    dm = []
    for r in recs:
        k = r['k']
        lp0, lp1 = np.array(r['logp0']), np.array(r['logp1'])
        E = expect(lp1)
        for j, n in enumerate(names):
            dE[n].append(E[j] - E[0])
        m = lp0[:, k + 1] - lp0[:, k - 1]
        dm.append([m[1] - m[0], m[2] - m[0]])
    dE = {n: np.array(v) for n, v in dE.items()}
    dm = np.array(dm)
    out = {'n_units': len(recs), 'q0_adjdiff_change': {'+': float(dm[:, 0].mean()), '-': float(dm[:, 1].mean())}}
    shares = {}
    for edit in ('conv', 'rec', 'kv', 'all'):
        num = np.concatenate([dE[f'steer+-{edit}'] * dE['steer+'], dE[f'steer--{edit}'] * dE['steer-']])
        den = np.concatenate([dE['steer+'] ** 2, dE['steer-'] ** 2])
        shares[f'remaining_after_restoring_{edit}'] = ratio_ci(num, den)
    for edit in ('conv', 'rec', 'kv'):
        num = np.concatenate([dE[f'base+{edit}+'] * dE['steer+'], dE[f'base+{edit}-'] * dE['steer-']])
        den = np.concatenate([dE['steer+'] ** 2, dE['steer-'] ** 2])
        shares[f'transferred_by_{edit}_alone'] = ratio_ci(num, den)
    out['shares'] = shares
    out['mean_dE'] = {n: float(dE[n].mean()) for n in names[1:]}
    out['mean_abs_dE_steer'] = float(np.abs(np.concatenate([dE['steer+'], dE['steer-']])).mean())
    return out, {f'relay_dE_{n}': v for n, v in dE.items()}


# ----------------------------------------------------------------------------- units digit

def units_summary(run: Path) -> tuple[dict, dict]:
    """Units-digit readout at q2 with the true k and every forced t; tens-digit steering at q1."""
    recs = read_jsonl(run / 'units.jsonl')
    if not recs:
        return {}, {}
    R, AR = {}, {}
    for fam in ('sweep_h', 'sweep_v', 'refcoco', 'textconv'):
        rr = [r for r in recs if r['family'] == fam]
        if not rr:
            continue
        E_nat, o_true, p_true, ent, mass, adj, p_true_t, Ef = [], [], [], [], [], [], [], []
        clamp_o, clamp_E = [], []
        for r in rr:
            lp1, lp2 = np.array(r['logp1']), np.array(r['logp2'])
            k, t, u, x = r['k'], r['t'], r['u'], r['truth']
            P2 = softmax(lp2[t])
            E_nat.append(float((P2 * np.arange(10)).sum()))
            o_true.append(x - (100 * k + 10 * t))                             # continuous offset inside the tens bin
            p_true.append(float(P2[u]))
            ent.append(float(-(P2 * np.log2(P2 + 1e-12)).sum()))
            mass.append(float(np.exp(lp2[t]).sum()))
            Ef.append(P2)
            p_true_t.append(float(softmax(lp1[0])[t]))
            if 1 <= t <= 8:
                adj.append((float(lp1[0, t + 1] - lp1[0, t - 1]), float(P2 @ np.arange(10)), x - (100 * k + 10 * t)))
            for tf in range(10):                                              # clamp: forced t vs true value
                clamp_o.append(x - (100 * k + 10 * tf))
                clamp_E.append(float(expect(lp2[tf])))
        E_nat, o_true, Ef = np.array(E_nat), np.array(o_true), np.array(Ef)
        co, cE = np.array(clamp_o), np.array(clamp_E)
        inb = (o_true >= -0.5) & (o_true < 9.5)
        sl, r2o = lin_fit(o_true[inb], E_nat[inb])
        adj = np.array(adj)
        res = {'n': len(rr), 'p_true_units': float(np.mean(p_true)), 'p_true_tens_q1': float(np.mean(p_true_t)),
               'entropy_bits': float(np.mean(ent)), 'digit_mass_q2': float(np.mean(mass)),
               'units_marginal': Ef.mean(0).round(3).tolist(), 'mean_max_prob': float(Ef.max(1).mean()),
               'argmax_counts': np.bincount(Ef.argmax(1), minlength=10).tolist(),
               'argmax_zero_share': float((Ef.argmax(1) == 0).mean()), 'slope_E_vs_offset': sl, 'r2_offset': r2o,
               'E_low_clamp': float(cE[(co >= -30) & (co < -0.5)].mean()) if np.any((co >= -30) & (co < -0.5)) else None,
               'E_high_clamp': float(cE[(co >= 9.5) & (co < 40)].mean()) if np.any((co >= 9.5) & (co < 40)) else None}
        if len(adj) > 5:
            res['r2_tens_adjdiff'] = lin_fit(adj[:, 0], adj[:, 1])[1]
            res['r2_units_offset_same_rows'] = lin_fit(adj[:, 2], adj[:, 1])[1]
            res['n_adj'] = int(len(adj))
        R[fam] = res
        AR[f'units_{fam}_E'], AR[f'units_{fam}_o'] = E_nat, o_true
        AR[f'units_{fam}_clamp_o'], AR[f'units_{fam}_clamp_E'] = co, cE
        if len(adj):
            AR[f'units_{fam}_adj'] = adj
    # Steering the tens-digit readout at q1 and reading the units digit at q2.
    st = [r for r in recs if r['family'] in ('sweep_h', 'sweep_v') and len(r['rows']) > 10]
    if st:
        names = st[0]['rows']
        dd, dEu = {n: [] for n in names[10:]}, {n: [] for n in names[10:]}
        for r in st:
            t = r['t']
            lp1, lp2 = np.array(r['logp1']), np.array(r['logp2'])
            d = lp1[:, t + 1] - lp1[:, t - 1]
            Eu = expect(lp2)
            for j, n in enumerate(names[10:], start=10):
                dd[n].append(d[j] - d[0])
                dEu[n].append(Eu[j] - Eu[t])                                  # row t is the unsteered true-t row
        steer = {}
        for kind in ('adj', 'rand'):
            ns = [n for n in names[10:] if n.startswith(kind)]
            x = np.concatenate([dd[n] for n in ns])
            y = np.concatenate([dEu[n] for n in ns])
            steer[kind] = {'slope': float(np.polyfit(x, y, 1)[0]) if np.abs(x).sum() > 0 else None,
                           'mean_abs_d_adjdiff': float(np.abs(x).mean()), 'mean_abs_dE_units': float(np.abs(y).mean())}
            AR[f'units_steer_{kind}_dd'], AR[f'units_steer_{kind}_dE'] = x, y
        R['steer_q1'] = {'n_units': len(st), **steer}
    return R, AR


def patch3_summary(run: Path) -> dict:
    recs = read_jsonl(run / 'patch3.jsonl')
    if not recs:
        return {}
    conds = recs[0]['conds']
    a, b = conds.index('A'), conds.index('B')
    out = {'n_pairs': len(recs)}
    dnorm = []
    for c in conds:
        if c in ('A', 'B'):
            continue
        nums, dens = [], []
        for r in recs:
            lp = np.array(r['logp'])
            n_, d_ = recovery(lp[a], lp[b], lp[conds.index(c)])
            nums.append(n_)
            dens.append(d_)
        out[c] = ratio_ci(np.array(nums), np.array(dens))
    for r in recs:
        lp = np.array(r['logp'])
        d = (lp[b] - lp[a]) - (lp[b] - lp[a]).mean()
        dnorm.append(float(np.linalg.norm(d)))
    out['mean_norm_B_minus_A'] = float(np.mean(dnorm))
    out['mean_abs_dE_B_minus_A'] = float(np.mean([abs(expect(np.array(r['logp'])[b]) - expect(np.array(r['logp'])[a]))
                                                  for r in recs]))
    return out


# ----------------------------------------------------------------------------- figures of new data

def fig_relay(R: dict, figs: Path) -> None:
    sh = R['relay']['shares']
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    items = [('steered, q0 conv slot (L25-32) restored', 'remaining_after_restoring_conv', C[0]),
             ('steered, recurrent states restored', 'remaining_after_restoring_rec', C[0]),
             ('steered, q0 key/value (L28, L32) restored', 'remaining_after_restoring_kv', C[0]),
             ('steered, all three restored', 'remaining_after_restoring_all', C[0]),
             ('unsteered + steered conv slot only', 'transferred_by_conv_alone', C[1]),
             ('unsteered + steered recurrent states only', 'transferred_by_rec_alone', C[1]),
             ('unsteered + steered key/value only', 'transferred_by_kv_alone', C[1])]
    for i, (lab, key, col) in enumerate(items):
        est, lo, hi = sh[key]
        ax.barh(i, est, color=col, height=0.6)
        ax.errorbar(est, i, xerr=[[est - lo], [hi - est]], color=INK, lw=1, capsize=3)
        ax.text(max(est, 0) + 0.03, i, f'{est:.2f}', va='center', fontsize=8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1, color=GREY, ls=':', lw=1)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_xlabel('share of the steering effect on E[tens digit] at q1')
    ax.set_title('which q0 state carries the first-digit adjacent difference to q1', fontsize=9)
    save(fig, figs / 'relay.png')


def fig_units(R: dict, AR: dict, figs: Path) -> None:
    fams = [f for f in ('sweep_h', 'sweep_v', 'refcoco', 'textconv') if f'units_{f}_E' in AR]
    labels = {'sweep_h': 'disc x', 'sweep_v': 'disc y', 'refcoco': 'RefCOCO x', 'textconv': 'text-given value'}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax = axes[0]
    for j, f in enumerate(fams):
        co, cE = AR[f'units_{f}_clamp_o'], AR[f'units_{f}_clamp_E']
        bins = np.arange(-30, 41, 2.5)
        idx = np.digitize(co, bins)
        xs = [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins)) if np.any(idx == i)]
        ys = [cE[idx == i].mean() for i in range(1, len(bins)) if np.any(idx == i)]
        ax.plot(xs, ys, color=C[j], lw=1.6, marker='o', ms=3, label=labels[f])
    xs = np.linspace(-30, 40, 300)
    ax.plot(xs, np.clip(np.floor(xs + 0.5), 0, 9), color=GREY, ls='--', lw=1, label='exact units digit')
    ax.axvspan(-0.5, 9.5, color=GREY, alpha=0.12)
    ax.set_xlabel('true value − (100k + 10t) for the forced (k, t), units')
    ax.set_ylabel('expected units digit at q2')
    ax.set_title('units digit vs offset inside the forced tens bin (binned means)', fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax = axes[1]
    w = 0.8 / len(fams)
    for j, f in enumerate(fams):
        ax.bar(np.arange(10) + (j - (len(fams) - 1) / 2) * w, R['units'][f]['units_marginal'], width=w, color=C[j],
               label=labels[f])
    ax.axhline(0.1, color=GREY, ls=':', lw=1, label='uniform')
    ax.set_xticks(range(10))
    ax.set_xlabel('units digit')
    ax.set_ylabel('mean probability (true k and t forced)')
    ax.set_title('units-digit distribution', fontsize=9)
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, loc='lower right')
    ax = axes[2]
    if 'units_steer_adj_dd' in AR:
        ax.scatter(AR['units_steer_rand_dd'], AR['units_steer_rand_dE'], s=6, color=GREY, alpha=0.6, label='random direction')
        ax.scatter(AR['units_steer_adj_dd'], AR['units_steer_adj_dE'], s=6, color=C[0], alpha=0.6,
                   label=f"tens adjacent-difference direction (slope {R['units']['steer_q1']['adj']['slope']:.2f})")
    ax.set_xlabel('change of the tens-digit adjacent difference at q1 (nats)')
    ax.set_ylabel('change of the expected units digit at q2')
    ax.set_title('steering q1 (layer 24) and reading q2', fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    save(fig, figs / 'units.png')


def fig_chain(figs: Path) -> None:
    """Diagram (not data): positions along the answer and the measured information routes."""
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis('off')
    toks = [('image\ntokens', 0.4), ('… "d"', 2.0), ('"\\":"', 3.2), ('" ["\n= q0', 4.4), ('"3"\n= q1', 6.0),
            ('"1"\n= q2', 7.6), ('"2"', 9.2)]
    for name, x in toks:
        ax.add_patch(plt.Rectangle((x - 0.5, 0.0), 1.0, 0.8, fill=False, ec=INK, lw=1))
        ax.text(x, 0.4, name, ha='center', va='center', fontsize=8.5)
    def arrow(x0, x1, y, text, col, rad=0.0):
        ax.annotate('', xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle='->', color=col, lw=1.6,
                                                                    connectionstyle=f'arc3,rad={rad}'))
        ax.text((x0 + x1) / 2, y + (0.28 if rad <= 0 else -0.4), text, ha='center', fontsize=7.5, color=col)
    arrow(0.6, 2.0, 0.85, 'image read by full attention\n(mostly L20; tail scan)', C[0], rad=-0.3)
    arrow(2.3, 3.0, 0.85, 'conv relay', C[1], rad=-0.4)
    arrow(3.5, 4.2, 0.85, 'conv relay', C[1], rad=-0.4)
    arrow(4.7, 5.8, 0.85, 'conv relay L25-31:\nfirst-digit adjacent difference', C[1], rad=-0.4)
    arrow(6.3, 7.4, 0.85, 'conv relay from q1:\ntens-digit adjacent difference', C[2], rad=-0.4)
    ax.text(4.4, -0.35, 'q0 writes digit k\n(arc readout)', ha='center', fontsize=7.5)
    ax.text(6.0, -0.35, 'q1 writes tens digit t\n(+ embedding of k, prior neurons)', ha='center', fontsize=7.5)
    ax.text(7.6, -0.35, 'q2 writes units digit u', ha='center', fontsize=7.5)
    ax.set_xlim(-0.3, 10)
    ax.set_ylim(-0.8, 1.9)
    save(fig, figs / 'chain.png')


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--r2-run', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--figures', required=True)
    args = p.parse_args()
    run, r2, figs = Path(args.run), Path(args.r2_run), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    z = np.load(r2 / 'arrays.npz', allow_pickle=True)
    A2 = {k: z[k] for k in z.files}
    R2res = json.loads((r2 / 'results.json').read_text())
    R, AR = {'angle': angle_stats(A2), 'adjdiff': adjdiff_decomposition(A2)}, {}
    fig_adjdiff(A2, R['adjdiff'], figs)
    fig_cones(figs)
    fig_svd_angle(A2, R2res, figs)
    fig_chain(figs)
    rel, a = relay_summary(run)
    R['relay'], AR = rel, {**AR, **a}
    un, a = units_summary(run)
    R['units'], AR = un, {**AR, **a}
    R['patch3'] = patch3_summary(run)
    if R['relay']:
        fig_relay(R, figs)
    if R['units']:
        fig_units(R, AR, figs)
    (run / 'results_r3.json').write_text(json.dumps(to_jsonable(R), indent=1))
    np.savez(run / 'arrays_r3.npz', **AR)
    print(json.dumps(to_jsonable(R), indent=1)[:6000])


if __name__ == '__main__':
    main()
