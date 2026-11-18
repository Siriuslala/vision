"""Experiment 3.15 round 3: sources of position information (model-free parts).

Results: docs/analysis/position/results_2026-11-11_position_sources.md.

Contents
* jgrid stimuli: grids of identical black figures with one red target whose entity rows / columns are placed at
  random spacings inside a random sub-rectangle of the canvas, so that a token's own position, the number of entity
  rows / entities that precede it, and the target coordinate are decorrelated across images;
* per-token variables of an image-token grid (own token row / column, entity counts, target flags);
* Gated DeltaNet algebra on captured projections: the exact decomposition of a head output into a weighted sum of
  earlier value vectors, o_n = sum_m A[n, m] v_m (see `delta_weights`);
* raster-order manipulations (cyclic roll of token rows / of tokens inside every row) and the coordinate the
  sequence order alone would imply;
* statistics: closed-form ridge with grouped cross-validation, variance partitioning of a representation over
  feature groups, following ratios and bootstrap intervals.

Coordinates are 0-1000 normalised; token rows / columns are 0-based in merged-token units (32 pixels).
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from analysis_tools.common import stimuli as st
from analysis_tools.position import coordinate_mechanism_natural as cmn

TOKEN = st.TOKEN_PIXELS
CANVAS = 640                                  # 20 x 20 merged tokens
FIG = 48                                      # rendered figure frame (pixels)
SPACINGS = (64, 80, 96, 112, 128)             # centre-to-centre spacing choices (pixels)
TEMPLATES = cmn.TEMPLATES


def rng_for(*parts) -> np.random.Generator:
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], 'little'))


# ----------------------------------------------------------------------------- jgrid stimuli

def jgrid_cases(n_items: int, seed: int = 1111) -> list[dict]:
    """Random layouts: R, C in 3..7, per-gap spacing drawn from SPACINGS, random origin, +-6 px jitter.

    Every layout keeps all figures inside the canvas with a 8-pixel margin. The target is any cell.
    """
    rng = rng_for(seed, 'jgrid')
    cases = []
    while len(cases) < n_items:
        R, C = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        dy = rng.choice(SPACINGS, R - 1) if R > 1 else np.zeros(0, int)
        dx = rng.choice(SPACINGS, C - 1) if C > 1 else np.zeros(0, int)
        span_y, span_x = int(dy.sum()) + FIG, int(dx.sum()) + FIG
        if span_y > CANVAS - 16 or span_x > CANVAS - 16:
            continue
        oy = int(rng.integers(8, CANVAS - 8 - span_y + 1))
        ox = int(rng.integers(8, CANVAS - 8 - span_x + 1))
        ys = oy + np.concatenate([[0], np.cumsum(dy)])
        xs = ox + np.concatenate([[0], np.cumsum(dx)])
        jit = rng.integers(-6, 7, (R, C, 2))
        k = len(cases)
        cases.append({'id': f'jgrid_{k:03d}', 'family': 'jgrid', 'rows': R, 'cols': C,
                      'ys': ys.tolist(), 'xs': xs.tolist(), 'jitter': jit.tolist(),
                      'target': [int(rng.integers(R)), int(rng.integers(C))],
                      'template': TEMPLATES[k % len(TEMPLATES)]})
    return cases


def jgrid_image(case: dict) -> tuple[Image.Image, dict]:
    """Render a jgrid case; truth has the target box/centre and every entity box (pixels, half-open)."""
    black = st.render_instance(case['template'], st.NONE, FIG)
    red = st.render_instance(case['template'], cmn.RED_CHANGE, FIG)
    canvas = np.full((CANVAS, CANVAS, 3), 255, np.uint8)
    boxes, tbox = [], None
    for r in range(case['rows']):
        for c in range(case['cols']):
            x = int(np.clip(case['xs'][c] + case['jitter'][r][c][0], 0, CANVAS - FIG))
            y = int(np.clip(case['ys'][r] + case['jitter'][r][c][1], 0, CANVAS - FIG))
            tile = red if [r, c] == case['target'] else black
            cmn.paste(canvas, tile, x, y)
            b = cmn.tight_box(tile, x, y)
            boxes.append({'row': r, 'col': c, 'box': b})
            if [r, c] == case['target']:
                tbox = b
    truth = cmn.truth_from_box(tbox, (CANVAS, CANVAS), extra={'row': case['target'][0] + 1,
                                                              'col': case['target'][1] + 1,
                                                              'rows': case['rows'], 'cols': case['cols'],
                                                              'entities': boxes})
    return Image.fromarray(canvas), truth


# ----------------------------------------------------------------------------- per-token variables

def token_table(truth: dict, H: int, W: int, family: str) -> dict[str, np.ndarray]:
    """Variables of every image token (raster order, length H*W).

    i, j            own token row / column; y_tok, x_tok their centres in 0-1000 units; n the raster index;
    tgt             1 if the token overlaps the target box; ent: overlaps any entity box (jgrid / single);
    row_cnt         jgrid: number of entity rows whose top edge lies above the token centre (rows started);
    col_cnt         jgrid: number of entities that overlap the token's row band and whose left edge lies left of
                    the token centre (entities already passed in this token row); 0 in empty token rows;
    after_tgt_row   1 if the token lies in a token row of the target and to the right of the target centre;
    ty, tx          target centre (units), constant per image.
    """
    ii, jj = np.divmod(np.arange(H * W), W)
    yc, xc = (ii + 0.5) * TOKEN, (jj + 0.5) * TOKEN
    out = {'i': ii, 'j': jj, 'n': np.arange(H * W), 'y_tok': 1000 * (ii + 0.5) / H, 'x_tok': 1000 * (jj + 0.5) / W}

    def overlaps(b):
        return (xc + TOKEN / 2 > b[0]) & (xc - TOKEN / 2 < b[2]) & (yc + TOKEN / 2 > b[1]) & (yc - TOKEN / 2 < b[3])

    tb = truth['box']
    out['tgt'] = overlaps(tb).astype(np.int8)
    trows = np.unique(ii[out['tgt'] > 0])
    out['after_tgt_row'] = (np.isin(ii, trows) & (xc > truth['center'][0])).astype(np.int8)
    out['ty'] = np.full(H * W, truth['center_units'][1])
    out['tx'] = np.full(H * W, truth['center_units'][0])
    ents = truth.get('entities')
    if family == 'jgrid':
        ent = np.zeros(H * W, bool)
        for e in ents:
            ent |= overlaps(e['box'])
        out['ent'] = ent.astype(np.int8)
        R = truth['rows']
        tops = np.array([min(e['box'][1] for e in ents if e['row'] == r) for r in range(R)])
        out['row_cnt'] = (tops[None, :] < yc[:, None]).sum(1)
        cc = np.zeros(H * W, int)
        for e in ents:
            b = e['box']
            band = (yc + TOKEN / 2 > b[1]) & (yc - TOKEN / 2 < b[3])
            cc += band & (b[0] < xc)
        out['col_cnt'] = cc
    elif family == 'single':
        out['ent'] = out['tgt'].copy()
    return out


def sample_tokens(table: dict, k: int, rng: np.random.Generator) -> np.ndarray:
    """k random image tokens plus every target token and every after-target-row token (sorted, unique)."""
    N = len(table['i'])
    must = np.where((table['tgt'] > 0) | (table['after_tgt_row'] > 0))[0]
    rest = np.setdiff1d(np.arange(N), must)
    pick = rng.choice(rest, min(k, len(rest)), replace=False)
    return np.unique(np.concatenate([must, pick]))


# ----------------------------------------------------------------------------- Gated DeltaNet algebra

def delta_weights(q, k, g, beta, queries) -> np.ndarray | object:
    """Exact read-out weights of a Gated DeltaNet head: o_n = sum_{m <= n} A[n, m] v_m.

    The recurrence of the transformers implementation is
        S_t = M_t S_{t-1} + beta_t k_t v_t^T,   M_t = alpha_t (I - beta_t k_t k_t^T),   o_t = S_t^T q_t,
    with alpha_t = exp(g_t), q and k L2-normalised and q scaled by d_k^{-1/2}. Unrolling gives
        o_n = sum_m v_m * beta_m * k_m^T (M_{m+1} ... M_n) q_n,
    so A[n, m] = beta_m k_m . u_{n,m} with u_{n,m} = M_{m+1} ... M_n q_n (M_n applied first).
    Works on torch tensors: q, k (T, Hh, d) already normalised/scaled, g, beta (T, Hh); queries (Q,) positions.
    Returns A (Q, Hh, T) with zeros for m > n.
    """
    import torch
    T, Hh, d = k.shape
    Q = len(queries)
    qpos = torch.as_tensor(queries, device=k.device)
    U = q[qpos].clone()                                     # (Q, Hh, d)
    A = torch.zeros(Q, Hh, T, device=k.device, dtype=k.dtype)
    alpha = g.exp()
    for r in range(int(qpos.max()), -1, -1):
        act = (qpos >= r)                                   # queries whose window contains r
        if not bool(act.any()):
            continue
        kr = k[r]                                           # (Hh, d)
        dot = (U * kr).sum(-1)                              # (Q, Hh)
        A[:, :, r] = torch.where(act[:, None], beta[r] * dot, torch.zeros_like(dot))
        # apply M_r = alpha_r (I - beta_r k_r k_r^T) for the next (earlier) source
        newU = alpha[r][None, :, None] * (U - beta[r][None, :, None] * dot[..., None] * kr[None])
        U = torch.where(act[:, None, None], newU, U)
    return A


# ----------------------------------------------------------------------------- raster-order manipulations

def roll_shift(box_px: list[float], H: int, W: int, axis: int, rng: np.random.Generator, min_shift: int = 3
               ) -> int | None:
    """A cyclic roll k (token units) along `axis` (0 rows, 1 columns) that keeps the target box contiguous.

    Slot row (column) R holds token row (column) (R - k) mod H; the target is not allowed to straddle the seam,
    and the sequence-implied target position moves by at least `min_shift` tokens. Returns None if impossible.
    """
    N = H if axis == 0 else W
    lo = int(box_px[1 if axis == 0 else 0] // TOKEN)
    hi = int((box_px[3 if axis == 0 else 2] - 1e-6) // TOKEN)
    ok = []
    for k in range(1, N):
        a, b = (lo + k) % N, (hi + k) % N
        if a <= b and min(k, N - k) >= min_shift:
            ok.append(k)
    return int(rng.choice(ok)) if ok else None


def seq_coordinate(c_units: float, k: int, N: int) -> float:
    """Coordinate implied by the slot a token at c_units occupies after a cyclic roll by k tokens (N per axis)."""
    t = c_units / 1000 * N
    return 1000 * (((t + k) % N) / N)


# ----------------------------------------------------------------------------- statistics

def ridge_fit(X, y, lam: float):
    """Closed-form ridge with intercept (torch, float64). X (n, p), y (n,) or (n, q). Returns (w, b)."""
    import torch
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    xm, ym = X.mean(0), y.mean(0)
    Xc, yc = X - xm, y - ym
    p = X.shape[1]
    if X.shape[0] >= p:
        w = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(p, dtype=X.dtype, device=X.device), Xc.T @ yc)
    else:
        G = Xc @ Xc.T
        w = Xc.T @ torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=X.dtype, device=X.device), yc)
    return w, ym - xm @ w


def ridge_cv_r2(X, y, groups: np.ndarray, lams=(1e-1, 1e1, 1e3, 1e5), folds: int = 5, test_mask=None) -> dict:
    """Grouped K-fold (by image) R^2 of a ridge probe; the penalty is chosen on the first fold's inner split.

    If test_mask (bool, n) is given, the probe is fit on ~test_mask and evaluated once on test_mask (held-out
    condition); otherwise K-fold over groups. Returns {'r2', 'lam', 'pred'}.
    """
    import torch
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)

    def r2(yt, yp):
        return float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())

    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    perm = rng.permutation(ug)
    fold_of = {g: i % folds for i, g in enumerate(perm)}
    fid = np.array([fold_of[g] for g in groups])
    # penalty selection on fold 0's training part (inner split by group parity)
    tr0 = np.where(fid != 0)[0]
    inner = fid[tr0] % 2 == 0
    best = None
    for lam in lams:
        w, b = ridge_fit(X[tr0[inner]], y[tr0[inner]], lam)
        s = r2(y[tr0[~inner]], X[tr0[~inner]] @ w + b)
        if best is None or s > best[1]:
            best = (lam, s)
    lam = best[0]
    pred = torch.zeros_like(y)
    if test_mask is not None:
        tm = torch.as_tensor(test_mask)
        w, b = ridge_fit(X[~tm], y[~tm], lam)
        pred = X @ w + b
        return {'r2': r2(y[tm], pred[tm]), 'lam': lam, 'pred': pred.numpy()}
    for f in range(folds):
        te = torch.as_tensor(fid == f)
        w, b = ridge_fit(X[~te], y[~te], lam)
        pred[te] = X[te] @ w + b
    return {'r2': r2(y, pred), 'lam': lam, 'pred': pred.numpy()}


def one_hot(v: np.ndarray, n: int | None = None) -> np.ndarray:
    v = np.asarray(v, int)
    n = int(v.max()) + 1 if n is None else n
    out = np.zeros((len(v), n), np.float32)
    out[np.arange(len(v)), np.clip(v, 0, n - 1)] = 1
    return out


def variance_partition(Y: np.ndarray, groups: dict[str, np.ndarray], img: np.ndarray, lam: float = 1.0,
                       folds: int = 5) -> dict:
    """Cross-validated share of the variance of Y (n, q) explained by feature groups, total and unique.

    total R^2 = 1 - SSE(all groups) / SST (pooled over the q columns, grouped CV by image);
    unique[g] = R^2(all) - R^2(all groups except g): the variance that only group g explains.
    """
    import torch
    names = list(groups)
    Yt = torch.as_tensor(Y, dtype=torch.float64)

    def cv_r2(cols):
        X = torch.as_tensor(np.concatenate([groups[c] for c in cols], 1), dtype=torch.float64)
        ug = np.unique(img)
        fold_of = {g: i % folds for i, g in enumerate(np.random.default_rng(0).permutation(ug))}
        fid = np.array([fold_of[g] for g in img])
        sse = 0.0
        for f in range(folds):
            te = torch.as_tensor(fid == f)
            w, b = ridge_fit(X[~te], Yt[~te], lam)
            sse += float(((Yt[te] - X[te] @ w - b) ** 2).sum())
        sst = float(((Yt - Yt.mean(0)) ** 2).sum())
        return 1 - sse / sst

    full = cv_r2(names)
    out = {'total': full, 'unique': {}, 'alone': {}}
    for g in names:
        rest = [c for c in names if c != g]
        out['unique'][g] = full - cv_r2(rest) if rest else full
        out['alone'][g] = cv_r2([g])
    return out


def following_ratio(ans_id: np.ndarray, ans_roll: np.ndarray, true: np.ndarray, seq: np.ndarray) -> dict:
    """Share of the sequence-implied displacement that the answer follows: slope through the origin of
    (ans_roll - ans_id) on (seq - true), with a bootstrap 95% interval over images."""
    d_ans, d_seq = np.asarray(ans_roll) - np.asarray(ans_id), np.asarray(seq) - np.asarray(true)
    ok = np.isfinite(d_ans) & np.isfinite(d_seq)
    d_ans, d_seq = d_ans[ok], d_seq[ok]
    if len(d_ans) < 3:
        return {'f': float('nan'), 'ci': [float('nan')] * 2, 'n': int(len(d_ans))}
    f = float((d_ans * d_seq).sum() / (d_seq ** 2).sum())
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(d_ans), len(d_ans))
        bs.append((d_ans[i] * d_seq[i]).sum() / (d_seq[i] ** 2).sum())
    return {'f': f, 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], 'n': int(len(d_ans))}


def boot_mean_ci(v: np.ndarray, n_boot: int = 1000) -> list[float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return [float('nan')] * 2
    rng = np.random.default_rng(0)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
