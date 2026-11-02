"""Experiment 3.12 round 2: CPU analysis (no model is loaded).

Stages (--stage):
  select    choose the centre-prior neurons of MLP 27-31 from the q1 captures and write the ablation plan
            (neuron order, mean activations, random control set) used by the GPU stage "ablate";
  analysis  all sections below, written to results.json and arrays.npz in the run directory.

Sections of results.json:
  geometry   effective dimensionality of the coordinate-dependent final-layer state at q0 / q1, the
             digit-logit trajectory (rank, angle, digit phases) and the definition of z;
  q1_model   fits of the second-digit logits with a shared code z(x) plus a per-k offset;
  patch      source patching at q0 / q1 (recovery of the B - A digit-logit difference per condition);
  tail       recovery as a function of the first answer position that sees run B's image;
  attention  attention mass at q0 / q1 on token groups (pre-image text, image, post-image text, q0, self);
  prior      input-independence of the centre prior, its writers, the blank-as-centre test and ablations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.position import digit_failure as df
from analysis_tools.position import digit_failure_analysis as A
from analysis_tools.position.coordinate_readout_run import read_jsonl

PRIOR_MLP_LAYERS = (27, 28, 29, 30, 31)
N_VALUES = (16, 64, 256)                     # numbers of prior neurons ablated
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- helpers

def recovery(lpA: np.ndarray, lpB: np.ndarray, lpX: np.ndarray) -> tuple[float, float]:
    """Projection of the centred digit-logit change (X - A) on (B - A); returns (numerator, denominator)."""
    c = lambda v: v - v.mean(-1, keepdims=True)
    d, dB = c(lpX - lpA), c(lpB - lpA)
    return float((d * dB).sum()), float((dB * dB).sum())


def ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = 2000) -> tuple[float, float, float]:
    """Ratio of sums with a pair-level bootstrap 95% interval."""
    est = num.sum() / den.sum()
    idx = RNG.integers(0, len(num), (n_boot, len(num)))
    bs = num[idx].sum(1) / den[idx].sum(1)
    return float(est), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def ranks(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def normed(h: np.ndarray, gain: np.ndarray, eps: float) -> np.ndarray:
    """Final RMSNorm output n = gain * h / rms(h) for residuals h (..., D)."""
    return gain * h / np.sqrt((h ** 2).mean(-1, keepdims=True) + eps)


def compact(run: Path, round1: Path) -> dict:
    """Round-1 arrays needed here, gathered once from the feature shards and cached in the run directory."""
    path = run / 'round1_compact.npz'
    U = A.load(round1)
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {'U': U, **{k: z[k] for k in z.files}}
    shards = A.load_shard_arrays(round1 / 'features', ('resid', 'dla_mlp', 'dla_head', 'dla_mixer', 'logit'))
    out = {}
    for fam in ('sweep_h', 'sweep_v'):
        ids = [i for i in A.family(U, fam) if 100 <= U[i]['truth'] < 1000]
        out[f'{fam}_ids'] = np.array(ids)
        out[f'{fam}_resid32'] = A.gather(U, ids, shards, 'resid')[:, :, :, -1]           # (I, 9, 2, D)
        out[f'{fam}_resid'] = A.gather(U, ids, shards, 'resid')[:, 0, 0]                 # (I, 7, D) q0, all layers
        out[f'{fam}_resid_q1'] = np.stack([A.gather(U, [i], shards, 'resid')[0, U[i]['ks'].index(int(U[i]['truth'] // 100)), 1]
                                           for i in ids])                                 # (I, 7, D) q1 at true k
        out[f'{fam}_logit'] = A.gather(U, ids, shards, 'logit')                           # (I, 9, 2, 10)
    for fam in ('sweep_h', 'sweep_v', 'blank', 'refcoco', 'textconv', 'count', 'grid'):
        ids = A.family(U, fam)
        if fam == 'count':
            ids = [i for i in ids if U[i]['ks'] == [1, 2, 3, 4, 5]]
        out[f'{fam}_dla_ids'] = np.array(ids)
        out[f'{fam}_dla_mlp'] = A.gather(U, ids, shards, 'dla_mlp')[:, :, 1].astype(np.float32)     # (I, K, 32, 10)
        out[f'{fam}_dla_head'] = A.gather(U, ids, shards, 'dla_head')[:, :, 1].astype(np.float32)   # (I, K, 8, 16, 10)
        out[f'{fam}_dla_mixer'] = A.gather(U, ids, shards, 'dla_mixer')[:, :, 1].astype(np.float32)
    np.savez(path, **out)
    return {'U': U, **out}


# ----------------------------------------------------------------------------- geometry (Q1)

def trajectory(L: np.ndarray) -> dict:
    """SVD of a digit-logit trajectory L (N, 10) after centring over digits and over samples."""
    L = L - L.mean(-1, keepdims=True)
    mu = L.mean(0)
    u, s, vt = np.linalg.svd(L - mu, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    agree = {}
    for r in (1, 2, 3):
        rec = (u[:, :r] * s[:r]) @ vt[:r] + mu
        agree[r] = float((rec.argmax(1) == L.argmax(1)).mean())
    return {'u': u, 's': s, 'vt': vt, 'mu': mu, 'ev': ev, 'agree': agree, 'var_total': float((s ** 2).sum() / len(L))}


def runs_per_digit(d: np.ndarray) -> float:
    """Mean number of contiguous runs per emitted digit along an ordered sequence of arg-max digits."""
    runs = {}
    prev = None
    for v in d:
        if v != prev:
            runs[v] = runs.get(v, 0) + 1
        prev = v
    return float(np.mean(list(runs.values())))


def geometry(C: dict, W: dict) -> tuple[dict, dict]:
    R, AR = {}, {}
    g, eps, WU = W['norm_gain'], float(W['norm_eps']), W['W_U']
    for fam in ('sweep_h', 'sweep_v'):
        ids = list(C[f'{fam}_ids'])
        x = np.array([C['U'][i]['truth'] for i in ids])
        h0 = C[f'{fam}_resid32'][:, 0, 0].astype(np.float32)                  # (I, D) q0, layer 32
        n0 = normed(h0, g, eps)
        L0 = n0 @ WU.T
        rec_err = float(np.abs(L0 - C[f'{fam}_logit'][:, 0, 0]).max())
        tr = trajectory(L0)
        u, s, vt = tr['u'], tr['s'], tr['vt']
        z1, z2 = u[:, 0] * s[0], u[:, 1] * s[1]
        theta = np.unwrap(np.arctan2(z2, z1))
        if spearman(theta, x) < 0:                                           # orient the angle along +x
            theta, z2, vt[1] = -theta, -z2, -vt[1]
        theta = theta - theta[0]
        slope = np.polyfit(x, theta, 1)[0]
        radius = np.hypot(z1, z2)
        phase = np.degrees(np.arctan2(vt[1] * s[1], vt[0] * s[0]))            # digit phases in the (z1, z2) plane
        amp = np.hypot(vt[0] * s[0], vt[1] * s[1])
        # Residual-space dimensionality of the coordinate-dependent variation, every stored layer.
        layers = {}
        for li, l in enumerate(df.RESID_LAYERS):
            for role, key in (('q0', f'{fam}_resid'), ('q1', f'{fam}_resid_q1')):
                h = C[key][:, li].astype(np.float32)
                hc = h - h.mean(0)
                sv = np.linalg.svd(hc, compute_uv=False)
                ev = sv ** 2 / (sv ** 2).sum()
                layers[f'{role}_L{l}'] = {'pr': float(1 / (ev ** 2).sum()), 'n90': int(np.searchsorted(np.cumsum(ev), 0.9) + 1),
                                          'ev5': ev[:5].round(3).tolist()}
        # Residual readout direction of z1 / z2: r_j = W_U^T v_j (normalised residual space).
        r1, r2 = WU.T @ vt[0], WU.T @ vt[1]
        # q1: per forced k, rank of the image-driven second-digit trajectory.
        q1 = {}
        for k in range(1, 10):
            L1 = C[f'{fam}_logit'][:, k - 1, 1]
            t1 = trajectory(L1)
            q1[k] = {'ev': t1['ev'][:3].round(3).tolist(), 'var_total': round(t1['var_total'], 3),
                     'agree1': t1['agree'][1]}
        R[fam] = {'q0_logit_recon_err': rec_err, 'q0_ev': tr['ev'][:5].round(3).tolist(),
                  'q0_var_total': tr['var_total'], 'q0_argmax_agree': tr['agree'],
                  'angle_spearman': spearman(theta, x), 'angle_range_deg': float(np.degrees(theta.max() - theta.min())),
                  'deg_per_100_units': float(np.degrees(slope) * 100), 'period_units': float(2 * np.pi / slope),
                  'radius_cv': float(radius.std() / radius.mean()),
                  'z1_spearman': spearman(z1, x), 'digit_phase_deg': phase.round(1).tolist(), 'digit_amp': amp.round(2).tolist(),
                  'runs_per_digit_q0': runs_per_digit(L0.argmax(1)), 'r1_r2_cos': float(r1 @ r2 / np.linalg.norm(r1) / np.linalg.norm(r2)),
                  'layers': layers, 'q1': q1}
        AR[f'{fam}_x'], AR[f'{fam}_theta'], AR[f'{fam}_z1'], AR[f'{fam}_z2'] = x, theta, z1, z2
        AR[f'{fam}_L0'], AR[f'{fam}_vt'], AR[f'{fam}_s'], AR[f'{fam}_mu'] = L0, vt[:3], s[:3], tr['mu']
    return R, AR


def q1_model(C: dict) -> tuple[dict, dict]:
    """Second-digit logits within the valid interval: shared 1-D code plus per-k offset.

    For each image i with true first digit k_i, the forced-k rows k in {k_i - 1, k_i, k_i + 1} are used.
    Model S (shared): l(i, k) = a z_i + b_k (one z per image, one slope vector a, one offset per k).
    Model K (per-k): l(i, k) = a_k z_ik + b_k (independent rank-1 fit per k).
    The 'two-stage' reading predicts b_k - b_{k'} ~ -a * (z at 100 (k - k')) when z is linear in x.
    """
    R, AR = {}, {}
    for fam in ('sweep_h', 'sweep_v'):
        ids = list(C[f'{fam}_ids'])
        x = np.array([C['U'][i]['truth'] for i in ids])
        L = C[f'{fam}_logit'][:, :, 1]                                        # (I, 9, 10)
        L = L - L.mean(-1, keepdims=True)
        b = L.mean(0)                                                         # per-k offsets b_k (9, 10)
        res = L - b
        u, s, vt = np.linalg.svd(res.reshape(len(ids), -1), full_matrices=False)
        # Rank-1 in the (image) x (k, digit) unfolding = z_i times a (k, digit) pattern: allows per-k slopes.
        ev_unfold = s ** 2 / (s ** 2).sum()
        z = u[:, 0] * s[0]
        if spearman(z, x) < 0:
            z, vt[0] = -z, -vt[0]
        A_k = vt[0].reshape(9, 10)                                            # per-k slope vectors
        cos = np.array([[A_k[i] @ A_k[j] / np.linalg.norm(A_k[i]) / np.linalg.norm(A_k[j]) for j in range(9)] for i in range(9)])
        tot = float((res ** 2).sum())
        # Shared slope (same a for every k): project the per-k slopes on their mean direction.
        a = A_k.mean(0)
        pred_shared = z[:, None, None] * a[None, None, :]
        r2_shared = 1 - float(((res - pred_shared) ** 2).sum()) / tot
        r2_perk = float(ev_unfold[0])
        # Within-interval second-digit readout from z: expected digit slope against the true offset.
        # Shift test of the two-stage reading: if the logits depend on (x, k) only through x - 100 k, then
        # l(x, k) = l(x + 100, k + 1). Compare that distance with moving only the image or only k.
        dsh, dim, dk, centre = [], [], [], []
        for i in range(len(x)):
            j = int(np.argmin(np.abs(x - (x[i] + 100))))
            if abs(x[j] - x[i] - 100) > 4:
                continue
            for k in range(1, 9):
                o = x[i] - 100 * k
                if not (-50 <= o < 150):
                    continue
                a0, a1 = L[i, k - 1], L[j, k]
                dsh.append(((a0 - a1) ** 2).sum())
                dim.append(((a0 - L[j, k - 1]) ** 2).sum())
                dk.append(((a0 - L[i, k]) ** 2).sum())
                centre.append(k in (3, 4, 5))                                  # pairs touching k = 4 or 5
        shift = {'n': len(dsh), 'd_shift': float(np.mean(dsh)), 'd_image_only': float(np.mean(dim)),
                 'd_k_only': float(np.mean(dk)), 'shift_ratio': float(np.mean(dsh) / np.mean(dim))}
        c = np.array(centre)
        for name, sel in (('centre', c), ('other', ~c)):
            shift[f'shift_ratio_{name}'] = float(np.array(dsh)[sel].mean() / np.array(dim)[sel].mean())
            shift[f'd_shift_{name}'] = float(np.array(dsh)[sel].mean())
        R[fam] = {'shift_test': shift, 'unfold_ev': ev_unfold[:3].round(3).tolist(), 'r2_shared_slope': r2_shared, 'r2_perk_slope': r2_perk,
                  'slope_cos_min': float(cos.min()), 'slope_cos_mean': float(cos[np.triu_indices(9, 1)].mean()),
                  'z_spearman_x': spearman(z, x)}
        AR[f'{fam}_q1_z'], AR[f'{fam}_q1_Ak'], AR[f'{fam}_q1_b'] = z, A_k, b
    return R, AR


# ----------------------------------------------------------------------------- patching

def patch_summary(run: Path) -> tuple[dict, dict]:
    recs = read_jsonl(run / 'patch.jsonl')
    R, AR = {}, {}
    for role in ('q0', 'q1'):
        rr = [r for r in recs if r['role'] == role]
        if not rr:
            continue
        conds = rr[0]['conds']
        tab = {}
        for c in conds:
            num, den = [], []
            for r in rr:
                lp = np.array(r['logp'])
                n, d = recovery(lp[r['conds'].index('A')], lp[r['conds'].index('B')], lp[r['conds'].index(c)])
                num.append(n)
                den.append(d)
            est, lo, hi = ratio_ci(np.array(num), np.array(den))
            tab[c] = {'r': round(est, 3), 'ci': [round(lo, 3), round(hi, 3)]}
        norm = [float(np.linalg.norm((lambda v: v - v.mean())(np.array(r['logp'])[1] - np.array(r['logp'])[0]))) for r in rr]
        R[role] = {'n_pairs': len(rr), 'mean_norm_BA': float(np.mean(norm)), 'conditions': tab}
        AR[f'patch_{role}_conds'] = np.array(conds)
        AR[f'patch_{role}_r'] = np.array([[tab[c]['r'], *tab[c]['ci']] for c in conds])
    return R, AR


def tail_summary(run: Path) -> tuple[dict, dict]:
    path = run / 'tail.jsonl'
    if not path.exists():
        return {}, {}
    recs = read_jsonl(path)
    R, AR = {}, {}
    for role, axis in (('q0', 0), ('q0', 1), ('q1', 0), ('q1', 1)):
        rr = [r for r in recs if r['role'] == role and r['axis'] == axis]
        role = f'{role}_{"xy"[axis]}'
        if not rr:
            continue
        # Align by start offset relative to q0 (the y prefix is longer, so offsets differ in range).
        offs = sorted({s for r in rr for s in r['starts']})
        tab = {}
        for o in offs:
            num, den = [], []
            for r in rr:
                if o not in r['starts']:
                    continue
                lp = np.array(r['logp'])[r['starts'].index(o)]
                n, d = recovery(lp[0], lp[1], lp[2])
                num.append(n)
                den.append(d)
            est, lo, hi = ratio_ci(np.array(num), np.array(den))
            tab[o] = {'r': round(est, 3), 'ci': [round(lo, 3), round(hi, 3)], 'n': len(num)}
        R[role] = {'n_pairs': len(rr), 'by_start': tab}
        AR[f'tail_{role}'] = np.array([[o, tab[o]['r'], *tab[o]['ci']] for o in offs])
    return R, AR


def attention_summary(run: Path) -> tuple[dict, dict]:
    recs = read_jsonl(run / 'attn.jsonl')
    R, AR = {}, {}
    groups = ('pre', 'image', 'post', 'q0', 'q1')
    for fam in sorted({r['family'] for r in recs}):
        rr = [r for r in recs if r['family'] == fam]
        out = {'n': len(rr), 'check_max': float(max(max(r['check']) for r in rr))}
        for qi, q in enumerate(('q0', 'q1')):
            m = {g: np.array([np.array(r[g])[:, :, qi] for r in rr]) for g in groups}    # (N, 8 layers, 16 heads)
            out[q] = {g: m[g].mean((0, 2)).round(3).tolist() for g in groups}           # head mean per layer
            out[q]['L20H11'] = {g: round(float(m[g][:, df.FULL_LAYERS.index(20), 11].mean()), 3) for g in groups}
            out[q]['L32H8'] = {g: round(float(m[g][:, df.FULL_LAYERS.index(32), 8].mean()), 3) for g in groups}
            out[q]['L32H10'] = {g: round(float(m[g][:, df.FULL_LAYERS.index(32), 10].mean()), 3) for g in groups}
            for g in groups:
                AR[f'attn_{fam}_{q}_{g}'] = m[g].mean((0, 2))
        R[fam] = out
    return R, AR


# ----------------------------------------------------------------------------- prior (Q4)

def load_captures(run: Path) -> tuple[list[dict], dict]:
    meta = read_jsonl(run / 'capture.jsonl')
    feats = {}
    for m in meta:
        z = np.load(run / 'features' / 'capture' / f"{m['unit']}.npz")
        feats[m['unit']] = {k: z[k] for k in z.files}
    return meta, feats


def neuron_scores(meta: list[dict], feats: dict, W: dict) -> dict:
    """Per-neuron direct contribution to the centre prior at q1 under natural (true first digit) inputs.

    For neuron j of layer l the direct contribution to the digit logits at q1 is act_j / rms * P_l[j], with
    P_l = W_down^T (gain * W_U^T) of shape (12288, 10), centred over digits. Reference state: rows whose
    forced first digit equals the true one ("natural rows"), averaged over every disc image. Prior score:
    [mean over natural rows with k = 4 (x in 400-499) - mean over all natural rows] times P_l[j, 9], plus the
    same for k = 5 (x in 500-599) and digit 0. The blank-canvas score uses the blank rows k = 4 / 5 relative to
    the blank mean over k. Ablation replaces an activation by its mean over all natural rows.
    """
    rows = [m for m in meta if m['family'] in ('sweep_h', 'sweep_v') and 100 <= m['truth'] < 1000]
    blank = [m for m in meta if m['family'] == 'blank']
    out = {}
    for l in PRIOR_MLP_LAYERS:
        P = W[f'mlp{l}_proj']
        P = P - P.mean(1, keepdims=True)                                     # centred over digits
        nat, kk, nat_raw = [], [], []
        for m in rows:
            k = int(m['truth'] // 100)
            j = m['ks'].index(k)
            f = feats[m['unit']]
            nat.append(f[f'mlp{l}'][j].astype(np.float32) / f['rms'][j])
            nat_raw.append(f[f'mlp{l}'][j].astype(np.float32))
            kk.append(k)
        nat, kk = np.stack(nat), np.array(kk)
        ref = nat.mean(0)
        score = (nat[kk == 4].mean(0) - ref) * P[:, 9] + (nat[kk == 5].mean(0) - ref) * P[:, 0]
        bscore = np.zeros_like(score)
        for m in blank:
            f = feats[m['unit']]
            a = f[f'mlp{l}'].astype(np.float32) / f['rms'][:, None]          # (9, N)
            dev = a - a.mean(0, keepdims=True)
            bscore += (dev[3] * P[:, 9] + dev[4] * P[:, 0]) / len(blank)
        out[l] = {'score': score, 'blank_score': bscore, 'mean_act': np.stack(nat_raw).mean(0)}
    return out


def select_prior_neurons(run: Path) -> None:
    W = dict(np.load(run / 'weights.npz'))
    meta, feats = load_captures(run)
    sc = neuron_scores(meta, feats, W)
    allsc = np.concatenate([[(l, j, sc[l]['score'][j]) for j in range(len(sc[l]['score']))] for l in PRIOR_MLP_LAYERS])
    order = allsc[np.argsort(-allsc[:, 2])]
    plan = {'order': order[:, :2].astype(np.int64), 'order_score': order[:, 2], 'n_values': np.array(N_VALUES)}
    rows = [m for m in meta if m['family'] in ('sweep_h', 'sweep_v') and 100 <= m['truth'] < 1000]
    for l in PRIOR_MLP_LAYERS:
        plan[f'mean_mlp{l}'] = sc[l]['mean_act'].astype(np.float32)
        plan[f'blank_score{l}'] = sc[l]['blank_score'].astype(np.float32)
    # Random control: as many neurons per layer as in the largest selected set, drawn from the rest.
    top = plan['order'][:max(N_VALUES)]
    for l in PRIOR_MLP_LAYERS:
        n_l = int((top[:, 0] == l).sum())
        pool = np.setdiff1d(np.arange(len(sc[l]['score'])), top[top[:, 0] == l, 1])
        plan[f'rand_mlp{l}'] = np.sort(RNG.choice(pool, n_l, replace=False)).astype(np.int64)
    L32 = np.stack([feats[m['unit']]['L32'][m['ks'].index(int(m['truth'] // 100))].astype(np.float32)
                    for m in rows]).mean(0)                                   # (4096,) natural-row mean
    plan['mean_L32'] = L32.reshape(df.N_HEADS, 256)
    np.savez(run / 'ablation_plan.npz', **plan)
    cum = np.cumsum(order[:, 2])
    bs = np.concatenate([sc[l]['blank_score'] for l in PRIOR_MLP_LAYERS])
    flat = np.concatenate([sc[l]['score'] for l in PRIOR_MLP_LAYERS])
    top_b = set(np.argsort(-bs)[:256].tolist())
    top_n = set(np.argsort(-flat)[:256].tolist())
    print(json.dumps({'overlap_top256_natural_vs_blank': len(top_b & top_n),
                      'corr_natural_blank_scores': float(np.corrcoef(bs, flat)[0, 1]),'total_positive_score': float(order[order[:, 2] > 0, 2].sum()),
                      'top_share': {n: float(cum[n - 1] / order[order[:, 2] > 0, 2].sum()) for n in N_VALUES},
                      'layers_top256': {int(l): int((top[:, 0] == l).sum()) for l in PRIOR_MLP_LAYERS}}, indent=1))


def prior_dla(C: dict) -> tuple[dict, dict]:
    """(a) Input-independence of the centre prior at q1, from the round-1 direct logit attribution.

    For each family and image, the 'prior term' of k = 4 is the direct contribution of MLP 27-31 and L32 H8
    to logit(9) - mean digit logit at k = 4, minus the same quantity averaged over k != 4 (k = 5 / digit 0
    likewise). Reported: its mean and spread across images and families, and its correlation with x.
    """
    R, AR = {}, {}
    for fam in ('sweep_h', 'sweep_v', 'blank', 'refcoco', 'grid', 'textconv', 'count'):
        mlp = C.get(f'{fam}_dla_mlp')
        if mlp is None or not len(mlp):
            continue
        head = C[f'{fam}_dla_head']
        ids = list(C[f'{fam}_dla_ids'])
        ks = C['U'][ids[0]]['ks']
        x = np.array([C['U'][i]['truth'] for i in ids])
        mlp = mlp - mlp.mean(-1, keepdims=True)
        head = head - head.mean(-1, keepdims=True)
        out = {'n': len(ids)}
        for k, d in ((4, 9), (5, 0)):
            if k not in ks:
                continue
            j = ks.index(k)
            oth = [jj for jj in range(len(ks)) if jj != j]
            comp = {}
            for l in PRIOR_MLP_LAYERS:
                comp[f'mlp{l}'] = mlp[:, j, l - 1, d] - mlp[:, oth, l - 1, d].mean(1)
            for (l, h) in ((32, 8), (32, 10)):
                li = df.FULL_LAYERS.index(l)
                comp[f'L{l}H{h}'] = head[:, j, li, h, d] - head[:, oth, li, h, d].mean(1)
            tot = sum(v for n, v in comp.items() if n != 'L32H10')
            out[f'k{k}_d{d}'] = {'mean': float(tot.mean()), 'sd_images': float(tot.std()),
                                 'corr_x': float(np.corrcoef(tot, x)[0, 1]) if np.isfinite(x).all() and len(x) > 2 else None,
                                 'components_mean': {n: round(float(v.mean()), 3) for n, v in comp.items()},
                                 'components_sd': {n: round(float(v.std()), 3) for n, v in comp.items()}}
            AR[f'prior_{fam}_k{k}'] = tot
            AR[f'prior_{fam}_x'] = x
        R[fam] = out
    return R, AR


def prior_behaviour(C: dict) -> dict:
    """P(9 | k = 4) and P(0 | k = 5) along the sweeps, and the blank-as-centre test.

    Blank-as-centre: for every disc image x, the mean over k of the JS divergence between the blank canvas's
    q1 distribution and the image's q1 distribution at the same k; the minimum locates the coordinate that
    the blank canvas behaves like.
    """
    U = C['U']
    out = {}
    for fam, blank in (('sweep_h', 'blank_x'), ('sweep_v', 'blank_y')):
        ids = A.family(U, fam)
        x = np.array([U[i]['truth'] for i in ids])
        P = np.stack([np.exp(U[i]['logp1']) for i in ids])                  # (I, 9, 10)
        Pb = np.exp(U[blank]['logp1'])                                       # (9, 10)
        jsd = A.js(P, Pb[None]).mean(1)
        jsd0 = A.js(np.stack([A.softmax_digits(U[i]['logp0']) for i in ids]), A.softmax_digits(U[blank]['logp0'])[None])
        bins = {f'{lo}-{lo + 100}': {'p9_k4': float(P[(x >= lo) & (x < lo + 100), 3, 9].mean()),
                                     'p0_k5': float(P[(x >= lo) & (x < lo + 100), 4, 0].mean())}
                for lo in range(0, 1000, 100) if ((x >= lo) & (x < lo + 100)).any()}
        out[fam] = {'blank_like_x_q1': float(x[jsd.argmin()]), 'js_min_q1': float(jsd.min()), 'js_median_q1': float(np.median(jsd)),
                    'blank_like_x_q0': float(x[jsd0.argmin()]), 'js_min_q0': float(jsd0.min()),
                    'p_prior_by_bin': bins, 'blank_p9_k4': float(Pb[3, 9]), 'blank_p0_k5': float(Pb[4, 0])}
    return out


def ablation_summary(run: Path, C: dict) -> tuple[dict, dict]:
    """Effects of the q1 ablations on the centre prior, the second-digit accuracy and other families.

    Decoding uses the round-1 q0 distribution (q0 is not edited) and the ablated q1 distribution; plateau
    share = fraction of disc truths in [400, 610) whose emulated greedy value falls in [480, 520].
    """
    path = run / 'ablate.jsonl'
    if not path.exists():
        return {}, {}
    U = C['U']
    recs = {r['unit']: r for r in read_jsonl(path)}
    conds = next(iter(recs.values()))['conds']
    R, AR = {'conditions': conds}, {}
    for fam in ('sweep_h', 'sweep_v', 'blank', 'textconv', 'refcoco', 'count', 'grid'):
        ids = [i for i in recs if recs[i]['family'] == fam]
        if not ids:
            continue
        out = {}
        for ci, c in enumerate(conds):
            lp1 = {i: np.array(recs[i]['logp1'])[ci] for i in ids}
            P1 = {i: A.softmax_digits(lp1[i]) for i in ids}
            m = {}
            if fam == 'blank':
                for i in ids:
                    m[f'{i}_p9_k4'] = float(P1[i][3, 9])
                    m[f'{i}_p0_k5'] = float(P1[i][4, 0])
            else:
                w = 10 if fam == 'count' else 100
                ks = recs[ids[0]]['ks']
                ptrue, within1, val, xs, first_ok = [], [], [], [], []
                for i in ids:
                    t = U[i]['truth']
                    if not (w <= t < 10 * w):
                        continue
                    k_true, d2 = A.digits_of(t, w)
                    if k_true not in ks:
                        continue
                    p = P1[i][ks.index(k_true)]
                    ptrue.append(p[d2])
                    diff = abs(int(p.argmax()) - d2)
                    within1.append(min(diff, 10 - diff) <= 1)
                    if fam in ('sweep_h', 'sweep_v') and t >= A.DECODE_MIN:
                        p0 = A.softmax_digits(U[i]['logp0'])
                        k0 = int(p0[1:].argmax()) + 1
                        t2 = int(P1[i][ks.index(k0)].argmax())
                        val.append(100 * k0 + 10 * t2 + 5)
                        xs.append(t)
                m['p_true'] = float(np.mean(ptrue))
                m['within1'] = float(np.mean(within1))
                m['n'] = len(ptrue)
                if val:
                    v, xx = np.array(val), np.array(xs)
                    sel = (xx >= 400) & (xx < 610)
                    m['plateau'] = float(((v[sel] >= 480) & (v[sel] <= 520)).mean())
                    m['mae'] = float(np.abs(v - xx).mean())
                    m['n_decode'] = len(v)
                    AR[f'abl_{fam}_{c}_val'], AR[f'abl_{fam}_x'] = v, xx
                    # Expected second digit vs in-bin offset (gain of the offset readout).
                    E, o = [], []
                    for i in ids:
                        t = U[i]['truth']
                        if 100 <= t < 1000:
                            k_true = int(t // 100)
                            E.append((P1[i][ks.index(k_true)] * np.arange(10)).sum())
                            o.append((t - 100 * k_true) / 10)
                    m['inbin_slope'] = float(np.polyfit(o, E, 1)[0])
            out[c] = m
        R[fam] = out
    return R, AR


# ----------------------------------------------------------------------------- margin readout and steering

def lin_r2(X: np.ndarray, y: np.ndarray) -> float:
    X = np.column_stack([X, np.ones(len(y))])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - (y - X @ b).var() / y.var())


def margin_readout(C: dict) -> tuple[dict, dict]:
    """What the second digit reads: the true in-bin offset or the first-digit neighbour margins at q0.

    For natural rows (forced k = true first digit, truths 200-899) the expected second digit E[t] at q1 is
    regressed on (i) the true offset o = (x - 100 k) / 10, (ii) the q0 neighbour margins
    m_up = log P0(k+1) - log P0(k) and m_dn = log P0(k-1) - log P0(k), (iii) both. Reported: R^2 and the
    5-fold cross-validated R^2 of each regression.
    """
    U = C['U']
    R, AR = {}, {}
    for fam in ('sweep_h', 'sweep_v', 'refcoco', 'grid', 'textconv'):
        E, o, mu, md = [], [], [], []
        for i in A.family(U, fam):
            u = U[i]
            t = u['truth']
            if not (200 <= t < 900):
                continue
            k = int(round(t) // 100)
            if k not in u['ks']:
                continue
            p1 = np.exp(u['logp1'][u['ks'].index(k)])
            E.append(float((p1 * np.arange(10)).sum()))
            o.append((t - 100 * k) / 10)
            mu.append(u['logp0'][k + 1] - u['logp0'][k])
            md.append(u['logp0'][k - 1] - u['logp0'][k])
        E, o, mu, md = map(np.array, (E, o, mu, md))
        folds = np.arange(len(E)) % 5

        def cv(X):
            pred = np.zeros_like(E)
            for f in range(5):
                tr, te = folds != f, folds == f
                Xt = np.column_stack([X[tr], np.ones(tr.sum())])
                b = np.linalg.lstsq(Xt, E[tr], rcond=None)[0]
                pred[te] = np.column_stack([X[te], np.ones(te.sum())]) @ b
            return float(1 - ((E - pred) ** 2).mean() / E.var())
        M = np.column_stack([mu, md])
        R[fam] = {'n': len(E), 'r2_offset': lin_r2(o[:, None], E), 'r2_margins': lin_r2(M, E),
                  'r2_both': lin_r2(np.column_stack([o, M]), E), 'cv_offset': cv(o[:, None]), 'cv_margins': cv(M),
                  'cv_both': cv(np.column_stack([o, M])), 'corr_offset_margin': float(np.corrcoef(o, mu - md)[0, 1])}
        AR[f'margin_{fam}_E'], AR[f'margin_{fam}_o'], AR[f'margin_{fam}_d'] = E, o, mu - md
    return R, AR


def steer_summary(run: Path) -> tuple[dict, dict]:
    """Causal test of the margin readout: steering q0 along the neighbour-margin direction.

    Per row: change of the q0 margin d = (m_up - m_dn) and of E[t] at q1 relative to the unsteered row.
    Reported per (layer, direction): mean changes per scale and the slope dE/dd pooled over scales, compared
    with the natural slope of E on d across images (from margin_readout).
    """
    path = run / 'steer.jsonl'
    if not path.exists():
        return {}, {}
    recs = read_jsonl(path)
    names = recs[0]['rows']
    R, AR = {}, {}
    for fam in ('sweep_h', 'sweep_v', 'all'):
        rr = [r for r in recs if fam == 'all' or r['family'] == fam]
        dd = {n: [] for n in names}
        dE = {n: [] for n in names}
        for r in rr:
            k = r['k']
            lp0, lp1 = np.array(r['logp0']), np.array(r['logp1'])
            P1 = A.softmax_digits(lp1)
            d = lp0[:, k + 1] - lp0[:, k - 1]
            E = (P1 * np.arange(10)).sum(-1)
            for j, n in enumerate(names):
                dd[n].append(d[j] - d[0])
                dE[n].append(E[j] - E[0])
        out = {}
        for n in names[1:]:
            out[n] = {'d_margin': float(np.mean(dd[n])), 'd_E': float(np.mean(dE[n]))}
        groups = {}
        for n in names[1:]:
            key = n.rsplit('+', 1)[0].rsplit('-', 1)[0]
            groups.setdefault(key, []).append(n)
        slopes = {}
        for key, ns in groups.items():
            x = np.concatenate([dd[n] for n in ns])
            y = np.concatenate([dE[n] for n in ns])
            slopes[key] = {'slope': float(np.polyfit(x, y, 1)[0]), 'mean_abs_d_margin': float(np.abs(x).mean()),
                           'mean_abs_d_E': float(np.abs(y).mean())}
            AR[f'steer_{fam}_{key}_dd'], AR[f'steer_{fam}_{key}_dE'] = x, y
        R[fam] = {'n_units': len(rr), 'rows': out, 'slopes': slopes}
    return R, AR


# ----------------------------------------------------------------------------- main

def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--stage', required=True, choices=('select', 'analysis'))
    args = p.parse_args()
    run, round1 = Path(args.run), Path(args.round1)
    if args.stage == 'select':
        select_prior_neurons(run)
        return
    W = dict(np.load(run / 'weights.npz'))
    C = compact(run, round1)
    results, arrays = {}, {}
    for name, fn in (('geometry', lambda: geometry(C, W)), ('q1_model', lambda: q1_model(C)),
                     ('patch', lambda: patch_summary(run)), ('tail', lambda: tail_summary(run)),
                     ('attention', lambda: attention_summary(run)), ('prior_dla', lambda: prior_dla(C)),
                     ('ablation', lambda: ablation_summary(run, C)), ('margin', lambda: margin_readout(C)),
                     ('steer', lambda: steer_summary(run))):
        r, a = fn()
        results[name] = r
        arrays.update(a)
        print(f'{name} done', flush=True)
    results['prior_behaviour'] = prior_behaviour(C)
    (run / 'results.json').write_text(json.dumps(to_jsonable(results), indent=1))
    np.savez(run / 'arrays.npz', **arrays)
    print('analysis written', flush=True)


if __name__ == '__main__':
    main()
