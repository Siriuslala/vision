"""Metrics of experiment 3.7 round 2 from the stage artifacts; writes `results.json` into the run directory.

Sections: ladder (displacement weights per stimulus and condition, ViT-level probe readouts), edits (whole-image
position edits on RefCOCO), subspace (slot-mean spectra, shared subspaces, spatial frequencies, content-only
probes), erase (grounding after LEACE erasure), l20 (positional share of layer-20 keys / values / logits),
smalldisc, layout (Qwen3-VL), pe_checkpoints, coord35 (3.5 RoPE swaps vs displaced-region texture).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image

from analysis_tools.position.coordinate_readout_analysis import bootstrap_mean, prop, read_jsonl
from analysis_tools.position import vit_layout_r2 as v2
from analysis_tools.position.vit_layout_analysis import slope

TAGS = ('q35', 'q3vl')
LAMBDAS = (1e1, 1e2, 1e3, 1e4, 1e5)


def ridge(xtr, ytr, xva, yva, xte, lambdas=LAMBDAS):
    """Ridge regression on standardized features with lambda chosen by validation MSE.

    Solved through one eigendecomposition of the (D x D) Gram matrix in float64 torch (GPU when available),
    so all lambdas share the same factorization. Returns (test predictions, lambda).
    """
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=dev)
    xtr, ytr, xva, yva, xte = t(xtr), t(ytr), t(xva), t(yva), t(xte)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    ztr, zva, zte = (xtr - mu) / sd, (xva - mu) / sd, (xte - mu) / sd
    ym = ytr.mean(0)
    if ztr.shape[0] >= ztr.shape[1]:
        e, v = torch.linalg.eigh(ztr.T @ ztr)
        b = v.T @ (ztr.T @ (ytr - ym))
        weights = lambda lam: v @ (b / (e + lam)[:, None])
    else:                                            # dual form when there are fewer samples than features
        e, u = torch.linalg.eigh(ztr @ ztr.T)
        b = u.T @ (ytr - ym)
        weights = lambda lam: ztr.T @ (u @ (b / (e + lam)[:, None]))
    best = None
    for lam in lambdas:
        w = weights(lam)
        mse = float(((zva @ w + ym - yva) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, lam, w)
    return (zte @ best[2] + ym).cpu().numpy(), best[1]


def centers(grid) -> np.ndarray:
    H, W = grid
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return np.stack([(c.ravel() + .5) / W * 1000, (r.ravel() + .5) / H * 1000], 1)


def center_of(row: dict):
    s = row['score']
    return s['center'] if s.get('lenient_valid') else None


def paired_diff(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    return bootstrap_mean([a[k] - b[k] for k in keys if a[k] is not None and b[k] is not None])


# ----------------------------------------------------------------------------- ladder

def ladder_w(recs: list[dict]) -> dict:
    """Per stimulus and condition: case -> w (and axis-split w), base hit / validity."""
    w = defaultdict(dict)
    meta = defaultdict(lambda: {'base_hit': 0, 'base_n': 0, 'base_strict_valid': 0, 'invalid': defaultdict(int),
                                'n': defaultdict(int)})
    for rec in recs:
        by = defaultdict(dict)
        for r in rec['rows']:
            by[r['stimulus']][r['condition']] = r
        for s, conds in by.items():
            base = conds['base']
            b = center_of(base)
            m = meta[s]
            m['base_n'] += 1
            m['base_hit'] += bool(base['score'].get('hit'))
            m['base_strict_valid'] += bool(base['score'].get('valid'))
            for cname, r in conds.items():
                if cname == 'base':
                    continue
                m['n'][cname] += 1
                c = center_of(r)
                if c is None or b is None:
                    m['invalid'][cname] += 1
                    w[(s, cname)][rec['id']] = None
                    continue
                w[(s, cname)][rec['id']] = v2.w_shift(c, b, r['d'])
    return w, meta


def ladder_metrics(run: Path, tag: str) -> dict:
    path = run / 'ladder' / f'{tag}_rows.jsonl'
    if not path.exists():
        return {}
    recs = read_jsonl(path)
    w, meta = ladder_w(recs)
    out = {'n_cases': len(recs), 'stimuli': {}}
    for s in v2.STIMULI:
        if s not in meta:
            continue
        m = meta[s]
        conds = {}
        for (s2, c), vals in w.items():
            if s2 != s:
                continue
            conds[c] = {**bootstrap_mean(list(vals.values())), 'invalid': m['invalid'][c], 'n_rows': m['n'][c]}
        out['stimuli'][s] = {'base_hit': prop(m['base_hit'], m['base_n']),
                             'base_strict_valid': prop(m['base_strict_valid'], m['base_n']), 'conditions': conds}
    # Paired contrasts.
    def W(s, c):
        return w.get((s, c), {})
    contrasts = {}
    for s in v2.STIMULI:
        if (s, 'vit_swap') not in w:
            continue
        add = {k: (W(s, 'vit_swap_target')[k] + W(s, 'vit_swap_context')[k])
               for k in W(s, 'vit_swap_target') if W(s, 'vit_swap_target')[k] is not None
               and W(s, 'vit_swap_context').get(k) is not None}
        contrasts[f'{s}|target+context-vit'] = paired_diff(add, W(s, 'vit_swap'))
        contrasts[f'{s}|target-context'] = paired_diff(W(s, 'vit_swap_target'), W(s, 'vit_swap_context'))
        if (s, 'move') in w:
            tot = {k: W(s, 'vit_swap')[k] + W(s, 'content_swap')[k] for k in W(s, 'vit_swap')
                   if W(s, 'vit_swap')[k] is not None and W(s, 'content_swap').get(k) is not None}
            contrasts[f'{s}|vit+content-move'] = paired_diff(tot, W(s, 'move'))
    for c in ('rope_swap', 'vit_swap', 'content_swap'):
        contrasts[f'natural|{c}_homog-textured'] = paired_diff(W('natural', f'{c}_homog'), W('natural', f'{c}_textured'))
        for s in v2.STIMULI:
            if s != 'natural' and (s, c) in w:
                contrasts[f'{s}-natural|{c}'] = paired_diff(W(s, c), W('natural', c))
    out['contrasts'] = contrasts
    out['variants'] = variant_info(recs)
    out['probe'] = ladder_probe(run, tag, recs)
    return out


def variant_info(recs: list[dict]) -> dict:
    """Mean displacement length and texture of the homogeneity-selected regions (natural)."""
    d = defaultdict(list)
    for rec in recs:
        for r in rec['rows']:
            if r['stimulus'] == 'natural' and r['condition'] in ('rope_swap', 'rope_swap_homog', 'rope_swap_textured'):
                d[r['variant']].append(float(np.linalg.norm(r['d'])))
    return {k: bootstrap_mean(v) for k, v in d.items()}


def ladder_probe(run: Path, tag: str, recs: list[dict]) -> dict:
    """ViT-level readouts: (a) token position probe trained on base tokens, applied to the mean token of T;
    (b) context probe from the mean of the remaining tokens to the target center. 5-fold CV over cases.
    Returns per stimulus and condition the probe displacement weight w_probe / w_ctx."""
    feats = {}
    for rec in recs:
        f = run / 'ladder' / 'feats' / f"{tag}_{rec['id']}.pt"
        feats[rec['id']] = torch.load(f, weights_only=False)
    ids = [r['id'] for r in recs]
    folds = np.arange(len(ids)) % 5
    grid = {r['id']: r['grid'] for r in recs}
    rows = {r['id']: {(x['stimulus'], x['condition']): x for x in r['rows']} for r in recs}
    out = {}
    for s in v2.STIMULI:
        have = [i for i in ids if (s, 'base') in rows[i]]
        if len(have) < 20:
            continue
        idx = {i: k for k, i in enumerate(ids)}
        conds = sorted({c for i in have for (s2, c) in rows[i] if s2 == s and c != 'base'})
        pred_T = defaultdict(dict)
        pred_ctx = defaultdict(dict)
        # token probe
        for f in range(5):
            tr = [i for i in have if folds[idx[i]] != f]
            te = [i for i in have if folds[idx[i]] == f]
            X = np.concatenate([feats[i]['base_tokens'][s]['tokens'].float().numpy() for i in tr])
            Y = np.concatenate([centers(grid[i])[feats[i]['base_tokens'][s]['index'].numpy()] for i in tr])
            nva = len(X) // 6
            Xte = np.stack([feats[i]['feats'][f'{s}|{c}'][0].astype(np.float32) for i in te for c in ['base'] + conds
                            if f'{s}|{c}' in feats[i]['feats']])
            keys = [(i, c) for i in te for c in ['base'] + conds if f'{s}|{c}' in feats[i]['feats']]
            p, _ = ridge(X[nva:], Y[nva:], X[:nva], Y[:nva], Xte)
            for (i, c), pp in zip(keys, p):
                pred_T[c][i] = pp
            # context probe: base context mean -> target center
            Xc = np.stack([feats[i]['feats'][f'{s}|base'][2].astype(np.float32) for i in tr])
            Yc = np.stack([1000 * np.asarray(rows[i][(s, 'base')]['truth']['center']) / rows[i][(s, 'base')]['truth']['canvas'] for i in tr])
            nv = max(len(Xc) // 6, 4)
            Xct = np.stack([feats[i]['feats'][f'{s}|{c}'][2].astype(np.float32) for (i, c) in keys])
            p, _ = ridge(Xc[nv:], Yc[nv:], Xc[:nv], Yc[:nv], Xct)
            for (i, c), pp in zip(keys, p):
                pred_ctx[c][i] = pp
        res = {}
        # accuracy of the probes on base rows
        tb = {i: 1000 * np.asarray(rows[i][(s, 'base')]['truth']['center']) / rows[i][(s, 'base')]['truth']['canvas'] for i in have}
        res['ctx_base_mae'] = float(np.mean([np.abs(pred_ctx['base'][i] - tb[i]).mean() for i in have]))
        res['ctx_base_r2_x'] = float(1 - np.sum([(pred_ctx['base'][i][0] - tb[i][0]) ** 2 for i in have]) /
                                     np.sum([(tb[i][0] - np.mean([tb[j][0] for j in have])) ** 2 for i in have]))
        res['T_base_mae'] = float(np.mean([np.abs(pred_T['base'][i] - tb[i]).mean() for i in have]))
        for c in conds:
            wT, wC = [], []
            for i in have:
                r = rows[i].get((s, c))
                if r is None or r.get('d') is None or i not in pred_T[c]:
                    continue
                wT.append(v2.w_shift(pred_T[c][i].tolist(), pred_T['base'][i].tolist(), r['d']))
                wC.append(v2.w_shift(pred_ctx[c][i].tolist(), pred_ctx['base'][i].tolist(), r['d']))
            res[c] = {'w_probe_T': bootstrap_mean(wT), 'w_ctx': bootstrap_mean(wC)}
        out[s] = res
        print('ladder probe', tag, s, flush=True)
    return out


# ----------------------------------------------------------------------------- edits

BBOX_RE = re.compile(r'"bbox_2d":\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')


def rescore_bbox(text: str, truth_box: list[float], canvas: list[int]) -> dict:
    """Truncation-tolerant bbox score: the first complete "bbox_2d": [x0, y0, x1, y1] in the answer.

    The edits stage decoded at most 40 new tokens; the four coordinates precede the label, so an answer cut
    inside the label still contains the complete box. Answers without a complete box count as invalid.
    """
    from analysis_tools.position.vit_layout_r2_run import score_bbox
    m = BBOX_RE.search(text)
    if not m:
        return {'valid': False, 'correct': False, 'hit': False}
    box = '[{"bbox_2d": [' + ', '.join(m.groups()) + ']}]'
    return score_bbox({'canvas': canvas, 'box': truth_box,
                       'center': [(truth_box[0] + truth_box[2]) / 2, (truth_box[1] + truth_box[3]) / 2]}, box)


def edits_metrics(run: Path, tag: str) -> dict:
    path = run / 'edits' / f'{tag}_rows.jsonl'
    if not path.exists():
        return {}
    recs = read_jsonl(path)
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    by = defaultdict(dict)
    strict = defaultdict(int)
    for rec in recs:
        c = cases[rec['id']]
        for r in rec['rows']:
            by[(r['stimulus'], r['condition'])][rec['id']] = rescore_bbox(r['text'], c['box'], c['canvas'])
            strict[(r['stimulus'], r['condition'])] += bool(r['score'].get('valid'))
    out = defaultdict(dict)
    for (s, c), sc in by.items():
        n = len(sc)
        base = by[(s, 'base')]
        iou = {k: v.get('iou', 0.0) for k, v in sc.items()}
        biou = {k: v.get('iou', 0.0) for k, v in base.items()}
        err = {k: v.get('error') for k, v in sc.items()}
        lost = sum(base[k].get('correct', False) and not sc[k].get('correct', False) for k in sc)
        gained = sum(sc[k].get('correct', False) and not base[k].get('correct', False) for k in sc)
        out[s][c] = {'correct': prop(sum(v.get('correct', False) for v in sc.values()), n),
                     'valid': prop(sum(v.get('valid', False) for v in sc.values()), n),
                     'center_hit': prop(sum(v.get('hit', False) for v in sc.values()), n),
                     'mean_iou': bootstrap_mean(list(iou.values())), 'delta_iou': paired_diff(iou, biou),
                     'error': bootstrap_mean(list(err.values())), 'lost': lost, 'gained': gained,
                     'strict_valid': strict[(s, c)]}
    return dict(out)


# ----------------------------------------------------------------------------- subspace

def slot_stats(x: torch.Tensor) -> dict:
    """x: (n, Nv, D). Between-slot variance share and the spectrum of the slot means."""
    x = x.float()
    n = len(x)
    mu = x.mean((0, 1))
    total = float(((x - mu) ** 2).sum(-1).mean())
    means = x.mean(0)
    between = float(((means - mu) ** 2).sum(-1).mean())
    between = (between - total / n) / (1 - 1 / n)
    c = means - means.mean(0)
    u, s, vt = torch.linalg.svd(c, full_matrices=False)
    ev = s ** 2
    return {'pos_share': between / total, 'spectrum': (ev / ev.sum())[:40].tolist(),
            'pr': float(ev.sum() ** 2 / (ev ** 2).sum()), 'basis': vt, 'pcs': u * s, 'means': means,
            'slot_dev_norm': (means - means.mean(0)).norm(dim=1)}


def overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    """Mean squared cosine of the principal angles between two k-dim row-space bases (1 = same subspace)."""
    qa, qb = a[:k], b[:k]
    s = torch.linalg.svdvals(qa @ qb.T)
    return float((s ** 2).mean())


def rope_frequencies(head_dim: int = 72, theta: float = 10000.) -> np.ndarray:
    """Axial 2D RoPE of the Qwen ViT: per-axis frequencies (radians per patch), head_dim // 4 rotation pairs per axis."""
    dim = head_dim // 2
    return 1.0 / theta ** (np.arange(0, dim, 2) / dim)


def spatial_spectrum(pcs: torch.Tensor, side: int, k: int = 10) -> dict:
    """Power of the top-k slot-mean PCs over spatial frequency (cycles per merged token) on the token grid."""
    f = np.fft.fftfreq(side)
    fr = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    bins = np.unique(np.round(fr, 4))
    out = []
    for j in range(k):
        m = pcs[:, j].reshape(side, side).numpy()
        p = np.abs(np.fft.fft2(m - m.mean())) ** 2
        out.append([float(p[np.isclose(np.round(fr, 4), b)].sum() / p.sum()) for b in bins])
    return {'freq': bins.tolist(), 'power': out}


def ridge_xy(xtr, ytr, xte, lambdas=LAMBDAS):
    nva = len(xtr) // 6
    p, _ = ridge(xtr[nva:], ytr[nva:], xtr[:nva], ytr[:nva], xte, lambdas)
    return p


def subspace_metrics(run: Path, tag: str, seed: int = 0) -> dict:
    cap = run / 'capture'
    files = {f.stem[len(tag) + 1:]: f for f in cap.glob(f'{tag}_*.pt') if f.stem.endswith(('_base', '_both_zero'))}
    if not files:
        return {}
    data = {k: torch.load(f) for k, f in files.items()}
    side = int(round(data['nat_train_base'].shape[1] ** 0.5))
    out = {'families': {}}
    stats = {}
    for k, x in data.items():
        st = slot_stats(x)
        stats[k] = st
        dev = st['slot_dev_norm']
        out['families'][k] = {'n': len(x), 'pos_share': st['pos_share'], 'spectrum': st['spectrum'], 'pr': st['pr'],
                              'outlier_slots': int((dev > 3 * dev.median()).sum())}
    # split-half reliability and cross-family overlap of the top-k slot-mean subspaces
    nt = data['nat_train_base']
    half = len(nt) // 2
    sa, sb = slot_stats(nt[:half]), slot_stats(nt[half:])
    ov = {}
    for k in (2, 5, 10, 20):
        ov[f'k{k}'] = {'nat_halves': overlap(sa['basis'], sb['basis'], k),
                       'nat_train|nat_test': overlap(stats['nat_train_base']['basis'], stats['nat_test_base']['basis'], k),
                       'nat_train|syn_layout': overlap(stats['nat_train_base']['basis'], stats['syn_layout_base']['basis'], k),
                       'nat_train|syn_disc': overlap(stats['nat_train_base']['basis'], stats['syn_disc_base']['basis'], k),
                       'nat_train|nat_vflip': overlap(stats['nat_train_base']['basis'], stats['nat_vflip_base']['basis'], k),
                       'nat_train|nat_shuffled': overlap(stats['nat_train_base']['basis'], stats['nat_shuffled_base']['basis'], k),
                       'random': k / nt.shape[-1]}
    out['overlap'] = ov
    # same after removing the high-norm outlier slots
    dev = stats['nat_train_base']['slot_dev_norm']
    keep = (dev <= 3 * dev.median()).numpy()
    def clean(x):
        return slot_stats(x[:, torch.from_numpy(keep)])
    cn, cs = clean(data['nat_train_base']), clean(data['syn_layout_base'])
    ch_a, ch_b = clean(nt[:half]), clean(nt[half:])
    out['overlap_clean'] = {f'k{k}': {'nat_halves': overlap(ch_a['basis'], ch_b['basis'], k),
                                      'nat_train|syn_layout': overlap(cn['basis'], cs['basis'], k)} for k in (2, 5, 10, 20)}
    out['clean_spectrum'] = {'nat_train': cn['spectrum'], 'syn_layout': cs['spectrum'], 'n_slots': int(keep.sum())}
    # spatial frequency of the slot-mean PCs (all slots and the clean set re-embedded with zeros)
    out['spatial'] = {'nat_train': spatial_spectrum(stats['nat_train_base']['pcs'], side),
                      'syn_layout': spatial_spectrum(stats['syn_layout_base']['pcs'], side)}
    pcs_clean = torch.zeros(side * side, 10)
    pcs_clean[torch.from_numpy(keep)] = cn['pcs'][:, :10]
    out['spatial']['nat_train_clean'] = spatial_spectrum(pcs_clean, side)
    om = rope_frequencies()
    out['rope'] = {'rad_per_patch': om.tolist(), 'cycles_per_token': (2 * om / (2 * np.pi)).tolist()}
    # position probes: train on one family, test on others (token level, subsample)
    rng = np.random.default_rng(seed)
    xy = centers((side, side))
    def sample(x, n_tok=48):
        X, Y = [], []
        for img in x:
            j = rng.choice(len(img), n_tok, replace=False)
            X.append(img[j].float().numpy()); Y.append(xy[j])
        return np.concatenate(X), np.concatenate(Y)
    probes = {}
    for train in ('nat_train_base', 'syn_layout_base', 'nat_train_both_zero'):
        Xtr, Ytr = sample(data[train])
        for test in ('nat_test_base', 'nat_vflip_base', 'nat_shuffled_base', 'syn_disc_base', 'syn_layout_base',
                     'nat_test_both_zero', 'nat_vflip_both_zero'):
            if (train.endswith('both_zero')) != (test.endswith('both_zero')):
                continue
            if test == train:
                continue
            x = data[test].float()
            P = ridge_xy(Xtr, Ytr, x.reshape(-1, x.shape[-1]).numpy()).reshape(len(x), -1, 2)
            Y = np.broadcast_to(xy, P.shape)
            probes[f'{train}->{test}'] = {
                'mae_x': float(np.abs(P[..., 0] - Y[..., 0]).mean()), 'mae_y': float(np.abs(P[..., 1] - Y[..., 1]).mean()),
                'corr_x': float(np.corrcoef(P[..., 0].ravel(), Y[..., 0].ravel())[0, 1]),
                'corr_y': float(np.corrcoef(P[..., 1].ravel(), Y[..., 1].ravel())[0, 1]),
                'slope_y': slope(Y[..., 1].ravel(), P[..., 1].ravel()), 'slope_x': slope(Y[..., 0].ravel(), P[..., 0].ravel()),
                'mean_map': P.mean(0).tolist()}
    out['probes'] = probes
    # constant baseline MAE on a uniform grid
    out['const_mae'] = float(np.abs(xy - xy.mean(0)).mean())
    torch.save({k: {'basis': v['basis'][:20], 'pcs': v['pcs'][:, :20], 'means_norm': v['slot_dev_norm']}
                for k, v in stats.items()}, run / 'capture' / f'{tag}_slot_stats.pt')
    return out


# ----------------------------------------------------------------------------- erase

def erase_metrics(run: Path, tag: str) -> dict:
    path = run / 'erase' / f'{tag}_rows.jsonl'
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    by = defaultdict(dict)
    for r in rows:
        by[(r['kind'], r['condition'])][r['id']] = r
    out = defaultdict(dict)
    for (kind, cond), rs in by.items():
        base = by[(kind, 'none')]
        err = {k: (r['score'].get('error') if r['score'].get('lenient_valid') else None) for k, r in rs.items()}
        berr = {k: (r['score'].get('error') if r['score'].get('lenient_valid') else None) for k, r in base.items()}
        res = {'hit': prop(sum(r['score'].get('hit', False) for r in rs.values()), len(rs)),
               'valid': prop(sum(r['score'].get('valid', False) for r in rs.values()), len(rs)),
               'error': bootstrap_mean(list(err.values())), 'delta_error': paired_diff(err, berr)}
        out[kind][cond] = res
    return dict(out)


def erase_slopes(run: Path, tag: str, items: list[dict]) -> dict:
    path = run / 'erase' / f'{tag}_rows.jsonl'
    if not path.exists():
        return {}
    tr = {it['id']: it['truth'] for it in items}
    by = defaultdict(lambda: ([], []))
    for r in read_jsonl(path):
        c = center_of(r)
        if c is None:
            continue
        t = tr[r['id']]
        by[(r['kind'], r['condition'])][0].append(1000 * np.asarray(t['center']) / t['canvas'])
        by[(r['kind'], r['condition'])][1].append(c)
    out = defaultdict(dict)
    for (k, c), (t, p) in by.items():
        t, p = np.array(t), np.array(p)
        out[k][c] = {'slope_x': slope(t[:, 0], p[:, 0]), 'slope_y': slope(t[:, 1], p[:, 1])}
    return dict(out)


# ----------------------------------------------------------------------------- l20

def target_tokens(truth: dict, grid) -> list[int]:
    from analysis_tools.position.coordinate_readout import tokens_overlapping
    return tokens_overlapping(truth['box'], tuple(grid))


def l20_metrics(run: Path, tag: str, head: int = 11) -> dict:
    out = {}
    for cond in ('none', 'onehot_mix'):
        f = run / 'l20' / f'{tag}_{cond}.pt'
        if not f.exists():
            continue
        recs = [r for r in torch.load(f, weights_only=False) if r['status'] == 'ok']
        res = {}
        for kind in ('refcoco', 'disc'):
            rs = [r for r in recs if r['kind'] == kind]
            if len(rs) < 10:
                continue
            K = torch.stack([r['k'] for r in rs]).float()            # (n, kvh, Nv, hd)
            V = torch.stack([r['v'] for r in rs]).float()
            Q = torch.stack([r['q'] for r in rs]).float()            # (n, heads, hd)
            A = torch.stack([r['attn_img'] for r in rs]).float()     # (n, heads, Nv)
            G = torch.stack([r['gate'] for r in rs]).float()         # (n, heads*hd) -> reshape
            n, kvh, nv, hd = K.shape
            heads = Q.shape[1]
            groups = heads // kvh
            G = G.view(n, heads, hd)
            # leave-one-out slot means
            def loo(X):
                s = X.sum(0, keepdim=True)
                return (s - X) / (len(X) - 1)
            Kp, Vp = loo(K), loo(V)
            Kc, Vc = K - Kp, V - Vp
            def share(X, Xp):
                mu = X.mean((0, 2), keepdim=True)
                tot = ((X - mu) ** 2).sum(-1).mean((0, 2))
                bet = ((Xp - mu) ** 2).sum(-1).mean((0, 2))
                return (bet / tot).tolist()
            truth_xy = np.array([1000 * np.asarray(r['truth']['center']) / r['truth']['canvas'] for r in rs])
            grid = rs[0]['grid']
            xy = centers(grid)
            per_head = {}
            for h in range(heads):
                kh = h // groups
                scale = hd ** -0.5
                s_full = torch.einsum('nd,nvd->nv', Q[:, h], K[:, kh]) * scale
                s_pos = torch.einsum('nd,nvd->nv', Q[:, h], Kp[:, kh]) * scale
                s_con = torch.einsum('nd,nvd->nv', Q[:, h], Kc[:, kh]) * scale
                vf = s_full.var(1)
                a = A[:, h]
                mass = a.sum(1)
                enr_full, enr_pos, enr_con, cen = [], [], [], []
                for i, r in enumerate(rs):
                    T = target_tokens(r['truth'], grid)
                    area = len(T) / nv
                    for lst, s in ((enr_full, s_full[i]), (enr_pos, s_pos[i]), (enr_con, s_con[i])):
                        p = torch.softmax(s, 0)
                        lst.append(float(p[T].sum() / area))
                    cen.append((a[i] / a[i].sum()).numpy() @ xy)
                cen = np.array(cen)
                o_pos = torch.einsum('nv,nvd->nd', a, Vp[:, kh]) * G[:, h]
                o_con = torch.einsum('nv,nvd->nd', a, Vc[:, kh]) * G[:, h]
                o_all = o_pos + o_con
                def cv_r2(X):
                    X = X.numpy()
                    folds = np.arange(len(X)) % 5
                    P = np.zeros_like(truth_xy)
                    for f in range(5):
                        tr, te = folds != f, folds == f
                        P[te] = ridge_xy(X[tr], truth_xy[tr], X[te], (1e-1, 1, 1e1, 1e2, 1e3, 1e4))
                    return [float(1 - ((P[:, j] - truth_xy[:, j]) ** 2).sum() / ((truth_xy[:, j] - truth_xy[:, j].mean()) ** 2).sum())
                            for j in range(2)]
                per_head[h] = {
                    'logit_var_share_pos': float((s_pos.var(1) / vf).mean()), 'logit_var_share_con': float((s_con.var(1) / vf).mean()),
                    'enrich_full': float(np.mean(enr_full)), 'enrich_pos': float(np.mean(enr_pos)), 'enrich_con': float(np.mean(enr_con)),
                    'img_mass': float(mass.mean()),
                    'centroid_corr_x': float(np.corrcoef(cen[:, 0], truth_xy[:, 0])[0, 1]),
                    'centroid_corr_y': float(np.corrcoef(cen[:, 1], truth_xy[:, 1])[0, 1]),
                    **({'r2_o_pos': cv_r2(o_pos), 'r2_o_con': cv_r2(o_con), 'r2_o_all': cv_r2(o_all)} if h == head else {}),
                }
            res[kind] = {'n': n, 'k_pos_share': share(K, Kp), 'v_pos_share': share(V, Vp), 'heads': per_head}
        out[cond] = res
        print('l20', tag, cond, flush=True)
    return out


# ----------------------------------------------------------------------------- small discs

def smalldisc_metrics(run: Path, tag: str) -> dict:
    path = run / 'smalldisc' / f'{tag}_rows.jsonl'
    if not path.exists():
        return {}
    by = defaultdict(list)
    for r in read_jsonl(path):
        by[(r['case']['diameter'], r['case']['background'], r['condition'])].append(r)
    out = {}
    for (d, bg, c), rs in sorted(by.items()):
        t = np.array([1000 * np.asarray(r['case']['center']) / 512 for r in rs if r['score'].get('lenient_valid')])
        p = np.array([r['score']['center'] for r in rs if r['score'].get('lenient_valid')])
        out[f'{d}|{bg}|{c}'] = {'hit': prop(sum(r['score'].get('hit', False) for r in rs), len(rs)),
                                'error': bootstrap_mean([r['score'].get('error') for r in rs]),
                                'slope_x': slope(t[:, 0], p[:, 0]) if len(p) > 2 else None,
                                'slope_y': slope(t[:, 1], p[:, 1]) if len(p) > 2 else None}
    return out


# ----------------------------------------------------------------------------- 3.5 reanalysis and PE tables

def spearman(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Spearman rank correlation and a two-sided permutation p-value."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n_perm)])
    return rho, float((np.abs(null) >= abs(rho)).mean())


def coord35_texture(coord_run: Path) -> dict:
    """3.5 RefCOCO RoPE / both / content swaps vs the gray-level texture of the displaced region T'."""
    root = coord_run / 'refcoco'
    cases = {c['id']: c for c in json.loads((root / 'cases.json').read_text())}
    rows = []
    for rec in read_jsonl(root / 'interventions.jsonl'):
        conds = {c['condition']: c for c in rec['conditions']}
        base = conds['standard']['center']
        case = cases[rec['id']]
        tex = v2.token_texture(Image.open(case['image_path']).convert('RGB'))
        T = rec['target_tokens']
        for reg in rec['regions']:
            if not reg['valid']:
                continue
            a = reg['axis']
            r = {'id': rec['id'], 'axis': a, 'tex_Tp': float(tex[reg['region']].mean()), 'tex_T': float(tex[T].mean()),
                 'dist': float(np.linalg.norm(reg['d']))}
            for name, key in (('rope', f's1swap_rope_{a}'), ('both', f's1swap_both_{a}'), ('content', f'c3content_{a}')):
                c = conds.get(key, {}).get('center')
                r[name] = v2.w_shift(c, base, reg['d']) if c and base else None
            rows.append(r)
    out = {'n': len(rows)}
    for name in ('rope', 'both', 'content'):
        ok = [r for r in rows if r[name] is not None]
        t = np.array([r['tex_Tp'] for r in ok])
        w = np.array([r[name] for r in ok])
        q = np.quantile(t, [1 / 3, 2 / 3])
        rho, pval = spearman(t, w)
        out[name] = {'spearman_tex_Tp': rho, 'p_perm': pval, 'n': len(ok),
                     'spearman_dist': spearman(np.array([r['dist'] for r in ok]), w)[0],
                     'tertiles': [bootstrap_mean(w[t <= q[0]]), bootstrap_mean(w[(t > q[0]) & (t <= q[1])]),
                                  bootstrap_mean(w[t > q[1]])], 'tertile_edges': q.tolist()}
    return out


def pe_checkpoints() -> dict:
    """Mean per-entry norm of the learned ViT PE table of each cached Qwen checkpoint and its cosine to the
    same-size Qwen3-VL table (read directly from the safetensors files)."""
    import glob
    from safetensors import safe_open
    from utils.project_paths import load_paths
    hub = load_paths()['HF_HUB_CACHE']
    tabs = {}
    for m in ('Qwen3-VL-4B-Instruct', 'Qwen3-VL-8B-Instruct', 'Qwen3.5-4B', 'Qwen3.5-9B', 'Qwen3.5-27B', 'Qwen3.8-27B'):
        snaps = sorted(glob.glob(f'{hub}/models--Qwen--{m}/snapshots/*'))
        if not snaps:
            continue
        for f in sorted(glob.glob(snaps[0] + '/*.safetensors')):
            with safe_open(f, 'pt') as h:
                k = [k for k in h.keys() if k.endswith('visual.pos_embed.weight')]
                if k:
                    tabs[m] = h.get_tensor(k[0]).float()
                    break
    out = {}
    for m, t in tabs.items():
        ref = tabs['Qwen3-VL-4B-Instruct' if t.shape[1] == 1024 else 'Qwen3-VL-8B-Instruct']
        out[m] = {'shape': list(t.shape), 'mean_norm': float(t.norm(dim=1).mean()),
                  'ratio_to_qwen3vl': float(t.norm(dim=1).mean() / ref.norm(dim=1).mean()),
                  'cos_to_qwen3vl': float(torch.nn.functional.cosine_similarity(t, ref, 1).mean())}
    return out


def layout_metrics_fast(run: Path, tag: str) -> dict:
    """Round-1 layout readouts (same splits and targets as vit_layout_analysis.layout_metrics) with the
    Gram-based ridge solver of this module."""
    from analysis_tools.position.vit_layout import layout_labels
    root = run / 'layout'
    cases = json.loads((root / 'cases.json').read_text())
    labels = [layout_labels(c) for c in cases]
    n = len(cases)
    tr, va, te = slice(0, int(.6 * n)), slice(int(.6 * n), int(.75 * n)), slice(int(.75 * n), n)
    reg = ('red_x', 'red_y', 'blue_x', 'blue_y')
    cls = ('red_left', 'red_above', 'red_bigger')
    y = np.array([[l[k] for k in reg] for l in labels])
    yc = np.array([[2 * l[k] - 1 for k in cls] for l in labels])
    lambdas = (1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5)
    out = {}
    for path in sorted(root.glob(f'{tag}_*_features.npz')):
        cond = path.name[len(tag) + 1:-len('_features.npz')]
        data = np.load(path)
        res = {}
        for key in data.files:
            x = data[key].astype(np.float32)
            for ri, readout in enumerate(('red', 'blue', 'background', 'bg_single', 'pooled')):
                xr = x[:, ri]
                row = {}
                p, _ = ridge(xr[tr], y[tr], xr[va], y[va], xr[te], lambdas)
                for j, k in enumerate(reg):
                    sst = ((y[te, j] - y[te, j].mean()) ** 2).sum()
                    row[k] = {'r2': float(1 - ((p[:, j] - y[te, j]) ** 2).sum() / sst),
                              'mae': float(np.abs(p[:, j] - y[te, j]).mean())}
                p, _ = ridge(xr[tr], yc[tr], xr[va], yc[va], xr[te], lambdas)
                for j, k in enumerate(cls):
                    row[k] = prop(int(((p[:, j] > 0) == (yc[te, j] > 0)).sum()), len(p))
                res[f'{key}|{readout}'] = row
        out[cond] = res
        print('layout', tag, cond, flush=True)
    return out


def cached(run: Path, name: str, fn):
    """Compute one results section once; sections are stored under run/results_parts/."""
    d = run / 'results_parts'
    d.mkdir(exist_ok=True)
    f = d / f'{name}.json'
    if f.exists():
        return json.loads(f.read_text())
    v = fn()
    f.write_text(json.dumps(v, default=float) + '\n')
    return json.loads(f.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--round1', type=Path, required=True)
    p.add_argument('--coord-run', type=Path, required=True)
    args = p.parse_args()
    run = args.run
    res = {'pe_checkpoints': cached(run, 'pe_checkpoints', pe_checkpoints),
           'coord35_texture': cached(run, 'coord35_texture', lambda: coord35_texture(args.coord_run))}
    sections = {'ladder': ladder_metrics, 'edits': edits_metrics, 'subspace': subspace_metrics, 'erase': erase_metrics,
                'l20': l20_metrics, 'smalldisc': smalldisc_metrics}
    for tag in TAGS:
        r = {name: cached(run, f'{tag}_{name}', lambda fn=fn: fn(run, tag)) for name, fn in sections.items()}
        spot = run / 'ladder' / f'{tag}_spot.jsonl'
        if spot.exists():
            rows = [x for x in read_jsonl(spot) if 'equivalence' not in x]
            eq = [x for x in read_jsonl(spot) if 'equivalence' in x]
            r['spot'] = {'text_equal': prop(sum(x['text_equal'] for x in rows), len(rows)),
                         'center_diff': bootstrap_mean([x['center_diff'] for x in rows]),
                         'equivalence': eq[0]['equivalence'] if eq else []}
        if (run / 'layout' / 'cases.json').exists() and list((run / 'layout').glob(f'{tag}_*_features.npz')):
            r['layout'] = cached(run, f'{tag}_layout', lambda: layout_metrics_fast(run, tag))
        res[tag] = r
    from analysis_tools.position.vit_layout_r2_run import erase_items
    if (run / 'erase' / 'q35_rows.jsonl').exists():
        a = argparse.Namespace(output=run, natural=args.round1.parent / '2026-10-14_probe_refinement', seed=20261016,
                               erase_n=300, coord_cases=None, refcoco=None)
        res['q35']['erase_slopes'] = erase_slopes(run, 'q35', erase_items(a))
    (run / 'results.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    print('wrote', run / 'results.json')


if __name__ == '__main__':
    main()
