"""Experiment 3.4: MLP, relative-position and synthetic-internal probes, and evaluation.

Inputs are the artifacts written by ``position_probe_refinement.py`` (cases, sampled features,
ridge predictions). Every probe is trained on training images, selected on validation images,
and evaluated on test images; images never cross splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from analysis_tools.position import position_probe_refinement as pr

MLP_WIDTHS = (64, 256, 1024)
MLP_LRS = (3e-4, 1e-3, 3e-3)
MLP_WDS = (0.01, 0.1, 1.0)
MAX_EPOCHS = 25
PATIENCE = 4
BATCH = 1024
PAIRS_PER_IMAGE = 256
LENGTH_BINS = 8
PCA_DIM = 512
FOURIER_K = 8
BILINEAR_RANK = 16
CANDIDATE_STEP = 5
BOOTSTRAP = 2000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ----------------------------------------------------------------------------- data

def load_context(out: Path) -> dict:
    cases = json.loads((out / 'cases.json').read_text())
    meta = json.loads((out / 'extract_trainval.json').read_text())
    natural = cases['natural']
    split = np.array([c['split'] for c in natural])
    return dict(out=out, cases=cases, meta=meta, natural=natural, split=split,
                ids={s: np.flatnonzero(split == s) for s in ('train', 'validation', 'test')})


def sampled_labels(case: dict, vision: bool, key: str = 'patch_hw', id_offset: int = 0) -> dict:
    """Labels of the 64 sampled tokens of one image: image coords, grid coords, grid size (W, H)."""
    hw = tuple(case[key])
    mid, pid = pr.sample_tokens(hw, case['id'] + id_offset)
    grid, image = pr.token_labels(hw, vision)
    idx = pid if vision else mid
    size = np.array([hw[1], hw[0]]) // (1 if vision else 2)
    return dict(image=image[idx], grid=grid[idx], size=np.broadcast_to(size, (len(idx), 2)), index=idx)


def full_labels(rows: list[dict], vision: bool, key: str = 'patch_hw') -> dict:
    parts = [pr.token_labels(tuple(c[key]), vision) for c in rows]
    sizes = [np.broadcast_to(np.array([c[key][1], c[key][0]]) // (1 if vision else 2), (len(p[0]), 2)) for c, p in zip(rows, parts)]
    owner = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(parts)])
    return dict(grid=np.concatenate([p[0] for p in parts]), image=np.concatenate([p[1] for p in parts]),
                size=np.concatenate(sizes), owner=owner)


def stack_sampled(ctx: dict, ids: np.ndarray, vision: bool, key='patch_hw') -> dict:
    labs = [sampled_labels(ctx['natural'][i], vision, key) for i in ids]
    return {k: np.concatenate([l[k] for l in labs]) for k in ('image', 'grid', 'size')} | \
        dict(owner=np.repeat(np.arange(len(ids)), pr.TOKENS_PER_IMAGE))


# ----------------------------------------------------------------------------- metrics

def metric_block(pred: np.ndarray, target: np.ndarray, size: np.ndarray, owner: np.ndarray, train_mean: np.ndarray,
                 seed: int = 0) -> dict:
    """Coordinate metrics in 0-1000 units plus token-unit error and grid accuracy from image coordinates."""
    err = np.abs(pred - target)
    tok_err = err * size / 1000
    pred_cell = np.clip(np.floor(pred * size / 1000), 0, size - 1)
    true_cell = np.floor(target * size / 1000 + 1e-6)
    exact = np.all(pred_cell == true_cell, 1)
    within = np.all(np.abs(pred_cell - true_cell) <= 1, 1)
    sse = np.square(pred - target).sum()
    sst = np.square(target - train_mean).sum()
    n_img = owner.max() + 1
    per_img = np.bincount(owner, err.mean(1), n_img) / np.bincount(owner, minlength=n_img)
    per_img_tok = np.bincount(owner, tok_err.mean(1), n_img) / np.bincount(owner, minlength=n_img)
    per_img_exact = np.bincount(owner, exact, n_img) / np.bincount(owner, minlength=n_img)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_img, (BOOTSTRAP, n_img))
    ci = lambda v: np.quantile(v[pick].mean(1), [.025, .975]).tolist()
    return dict(mae=float(err.mean()), mae_ci95=ci(per_img), x_mae=float(err[:, 0].mean()), y_mae=float(err[:, 1].mean()),
                r2=float(1 - sse / sst), token_mae=float(tok_err.mean()), token_mae_ci95=ci(per_img_tok),
                token_x_mae=float(tok_err[:, 0].mean()), token_y_mae=float(tok_err[:, 1].mean()),
                exact_cell=float(exact.mean()), exact_cell_ci95=ci(per_img_exact), within_one_cell=float(within.mean()),
                n_tokens=int(len(pred)), n_images=int(n_img))


# ----------------------------------------------------------------------------- MLP

class MLP(torch.nn.Module):
    def __init__(self, dim: int, width: int, out: int = 2):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, width), torch.nn.GELU(), torch.nn.Linear(width, out))

    def forward(self, x):
        return self.net(x)


def standardizer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    mu = xf.mean(0)
    sd = xf.std(0)
    return mu, sd.clamp_min(1e-6 * float(sd.max()))


def train_net(make, batches_fn, xtr_n: int, val_fn, lr: float, wd: float, seed: int) -> tuple[torch.nn.Module, float, int]:
    """Generic AdamW loop with per-epoch validation MSE and early stopping; returns the best state."""
    torch.manual_seed(seed)
    net = make().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    gen = torch.Generator(device='cpu').manual_seed(seed)
    for epoch in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(xtr_n, generator=gen)
        for i in range(0, xtr_n, BATCH):
            loss = batches_fn(net, perm[i:i + BATCH])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(val_fn(net))
        if not np.isfinite(v):
            break
        if v < best - 1e-7:
            best, best_epoch, bad = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError('Training diverged before the first validation step.')
    net.load_state_dict(best_state)
    return net, best, best_epoch


def fit_mlp(x: torch.Tensor, y: torch.Tensor, xv: torch.Tensor, yv: torch.Tensor, width: int, lr: float, wd: float,
            seed: int = 0) -> tuple[torch.nn.Module, float, int]:
    """x, xv: standardized-on-the-fly float16 GPU features; y in 0-1000 units (scaled to 0-1 inside)."""
    mu, sd = standardizer(x)
    yt, yvt = y / 1000, yv / 1000
    norm = lambda a: (a.float() - mu) / sd

    def step(net, idx):
        idx = idx.to(x.device)
        return torch.nn.functional.mse_loss(net(norm(x[idx])), yt[idx])

    def val(net):
        return sum(torch.square(net(norm(xv[i:i + 8192])) - yvt[i:i + 8192]).sum() for i in range(0, len(xv), 8192)) / yvt.numel()

    net, best, epoch = train_net(lambda: MLP(x.shape[1], width), step, len(x), val, lr, wd, seed)
    net.mu, net.sd = mu, sd
    return net, best, epoch


@torch.no_grad()
def predict(net, x) -> np.ndarray:
    chunk = lambda a: a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))
    out = [net((chunk(x[i:i + 8192]).to(DEVICE).float() - net.mu) / net.sd) for i in range(0, len(x), 8192)]
    return (torch.cat(out) * 1000).cpu().numpy()


def to_gpu(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), device=DEVICE)


def run_mlp(ctx: dict) -> None:
    out = ctx['out']
    folder = out / 'probes' / 'mlp'
    folder.mkdir(parents=True, exist_ok=True)
    tr, va = ctx['ids']['train'], ctx['ids']['validation']
    for stage in ctx['meta']['feature_stages']:
        target = folder / f'{stage}.npz'
        if target.exists():
            continue
        t0 = datetime.now(timezone.utc)
        vision = pr.is_vision(stage)
        feats = np.load(out / 'features' / f'natural_{stage}.npy', mmap_mode='r')
        x = to_gpu(feats[tr].reshape(-1, feats.shape[-1]))
        xv = to_gpu(feats[va].reshape(-1, feats.shape[-1]))
        ltr, lva = stack_sampled(ctx, tr, vision), stack_sampled(ctx, va, vision)
        y, yv = to_gpu(ltr['image']).float(), to_gpu(lva['image']).float()
        log, saved = {}, {}
        test_x = feats[ctx['ids']['test']].reshape(-1, feats.shape[-1])
        x768 = np.load(out / 'features' / f'test768_{stage}.npy', mmap_mode='r').reshape(-1, feats.shape[-1])
        xsyn = np.load(out / 'features' / f'synthetic_{stage}.npy', mmap_mode='r')
        best_hp = None
        # MLP256 searches the full 3x3 (lr, weight decay) grid; MLP64 and MLP1024 search the learning rate
        # at the weight decay selected for MLP256 (compute budget; see the results document).
        for width in (256, 64, 1024):
            runs = []
            grid = [(lr, wd) for lr in MLP_LRS for wd in MLP_WDS] if width == 256 else [(lr, best_hp[1]) for lr in MLP_LRS]
            for lr, wd in grid:
                net, v, ep = fit_mlp(x, y, xv, yv, width, lr, wd)
                runs.append((v, lr, wd, ep, net))
            v, lr, wd, ep, net = min(runs, key=lambda r: r[0])
            log[f'mlp{width}'] = dict(val_mse_scaled=v, lr=lr, wd=wd, best_epoch=ep, params=sum(p.numel() for p in net.parameters()),
                                       grid=[dict(lr=r[1], wd=r[2], val=r[0], epoch=r[3]) for r in runs])
            saved[f'mlp{width}__test512'] = predict(net, test_x)
            saved[f'mlp{width}__test768'] = predict(net, x768)
            saved[f'mlp{width}__synthetic'] = predict(net, xsyn)
            if width == 256:
                best_hp = (lr, wd)
            del runs, net
        # Shuffled-label control for the MLP family: permute training labels across all training tokens.
        perm = torch.as_tensor(np.random.default_rng(pr.SHUFFLE_SEED).permutation(len(y)), device=DEVICE)
        net, v, ep = fit_mlp(x, y[perm], xv, yv, 256, *best_hp)
        log['mlp256_shuffled'] = dict(val_mse_scaled=v, lr=best_hp[0], wd=best_hp[1], best_epoch=ep)
        saved['mlp256_shuffled__test512'] = predict(net, test_x)
        np.savez(target, **saved)
        log['seconds'] = (datetime.now(timezone.utc) - t0).total_seconds()
        (folder / f'{stage}.json').write_text(json.dumps(log, indent=1) + '\n')
        print(f'MLP {stage}: ' + ', '.join(f'{k}={log[k]["val_mse_scaled"]:.5f}' for k in log if k.startswith('mlp'))
              + f' ({log["seconds"]:.0f}s)', flush=True)
        del x, xv
        torch.cuda.empty_cache()


def run_mlp_learning_curve(ctx: dict) -> None:
    out = ctx['out']
    folder = out / 'probes' / 'mlp_lc'
    folder.mkdir(parents=True, exist_ok=True)
    tr, va, te = ctx['ids']['train'], ctx['ids']['validation'], ctx['ids']['test']
    groups = ctx['meta']['lc_groups']
    orders = [np.arange(groups)] + [np.random.default_rng(50261014 + k).permutation(groups) for k in range(1, pr.LC_ORDERS)]
    sizes = [s for s in pr.LC_SIZES if s <= groups * pr.LC_GROUP_IMAGES]
    for stage in ctx['meta']['lc_stages']:
        target = folder / f'{stage}.npz'
        if target.exists():
            continue
        hp = json.loads((out / 'probes' / 'mlp' / f'{stage}.json').read_text())['mlp256']
        feats = np.load(out / 'features' / f'natural_{stage}.npy', mmap_mode='r')
        xall = to_gpu(feats[tr].reshape(-1, feats.shape[-1]))
        yall = to_gpu(stack_sampled(ctx, tr, False)['image']).float()
        xv = to_gpu(feats[va].reshape(-1, feats.shape[-1]))
        yv = to_gpu(stack_sampled(ctx, va, False)['image']).float()
        test_x = feats[te].reshape(-1, feats.shape[-1])
        saved, log = {}, {}
        for o, order in enumerate(orders):
            for size in sizes:
                if size == sizes[-1] and o:
                    continue
                imgs = np.concatenate([np.arange(g * pr.LC_GROUP_IMAGES, (g + 1) * pr.LC_GROUP_IMAGES) for g in order[:size // pr.LC_GROUP_IMAGES]])
                tok = torch.as_tensor((imgs[:, None] * pr.TOKENS_PER_IMAGE + np.arange(pr.TOKENS_PER_IMAGE)).ravel(), device=DEVICE)
                net, v, ep = fit_mlp(xall[tok], yall[tok], xv, yv, 256, hp['lr'], hp['wd'], seed=o)
                saved[f'n{size}_o{o}'] = predict(net, test_x)
                log[f'n{size}_o{o}'] = dict(val_mse_scaled=v, best_epoch=ep)
        np.savez(target, **saved)
        (folder / f'{stage}.json').write_text(json.dumps(log, indent=1) + '\n')
        print(f'MLP learning curve {stage} done', flush=True)
        del xall, xv
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- relative position

def stratified_pairs(image_xy: np.ndarray, seed: int, count: int = PAIRS_PER_IMAGE, bins: int = LENGTH_BINS) -> np.ndarray:
    """Ordered token pairs (a, b), a != b, balanced over equal-width displacement-length bins."""
    n = len(image_xy)
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    keep = a != b
    a, b = a[keep], b[keep]
    length = np.linalg.norm(image_xy[b] - image_xy[a], axis=1)
    edges = np.linspace(0, length.max() + 1e-9, bins + 1)
    which = np.clip(np.digitize(length, edges) - 1, 0, bins - 1)
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(which == k)) for k in range(bins)]
    quota = [min(len(p), count // bins) for p in pools]
    left = count - sum(quota)
    while left > 0:                      # redistribute unused quota to bins that still have pairs
        open_bins = [k for k in range(bins) if len(pools[k]) > quota[k]]
        if not open_bins:
            raise ValueError('Not enough token pairs for stratified sampling.')
        for k in open_bins:
            if left == 0:
                break
            quota[k] += 1
            left -= 1
    chosen = np.concatenate([pools[k][:quota[k]] for k in range(bins)])
    return np.stack([a[chosen], b[chosen]], 1)


def pair_table(ctx: dict, ids: np.ndarray, vision: bool) -> dict:
    """Pairs indexed into the flattened (image, 64 token) feature array of `ids`."""
    pa, pb, delta, owner = [], [], [], []
    for j, i in enumerate(ids):
        lab = sampled_labels(ctx['natural'][i], vision)
        pairs = stratified_pairs(lab['image'], 60261014 + int(i))
        pa.append(j * pr.TOKENS_PER_IMAGE + pairs[:, 0])
        pb.append(j * pr.TOKENS_PER_IMAGE + pairs[:, 1])
        delta.append(lab['image'][pairs[:, 1]] - lab['image'][pairs[:, 0]])
        owner.append(np.full(len(pairs), j))
    return dict(a=np.concatenate(pa), b=np.concatenate(pb), delta=np.concatenate(delta), owner=np.concatenate(owner))


def fourier(delta: torch.Tensor) -> torch.Tensor:
    """phi(Delta): [cos, sin] of omega_k Delta_x and Delta_y, omega_k = pi k / 1000, k = 1..K."""
    w = torch.arange(1, FOURIER_K + 1, device=delta.device, dtype=torch.float32) * np.pi / 1000
    ang = delta[..., None].float() * w           # (n, 2, K)
    return torch.cat([torch.cos(ang[:, 0]), torch.sin(ang[:, 0]), torch.cos(ang[:, 1]), torch.sin(ang[:, 1])], 1)


def decode_fourier(phi: torch.Tensor) -> torch.Tensor:
    cand = torch.arange(-1000, 1000 + CANDIDATE_STEP, CANDIDATE_STEP, device=phi.device, dtype=torch.float32)
    w = torch.arange(1, FOURIER_K + 1, device=phi.device, dtype=torch.float32) * np.pi / 1000
    cos, sin = torch.cos(cand[:, None] * w), torch.sin(cand[:, None] * w)      # (C, K)
    k = FOURIER_K
    sx = phi[:, :k] @ cos.T + phi[:, k:2 * k] @ sin.T
    sy = phi[:, 2 * k:3 * k] @ cos.T + phi[:, 3 * k:] @ sin.T
    return torch.stack([cand[sx.argmax(1)], cand[sy.argmax(1)]], 1)


class Bilinear(torch.nn.Module):
    """phi_j = (A_j x_a)^T (G_j x_b), A_j, G_j in R^{rho x D}, j = 1..4K."""

    def __init__(self, dim: int, outputs: int = 4 * FOURIER_K, rank: int = BILINEAR_RANK):
        super().__init__()
        self.outputs, self.rank = outputs, rank
        self.a = torch.nn.Linear(dim, outputs * rank, bias=False)
        self.g = torch.nn.Linear(dim, outputs * rank, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(outputs))
        # Small initialization keeps the initial bilinear outputs near zero; the default Linear
        # initialization makes rank-16 products start far above the unit target scale.
        for lin in (self.a, self.g):
            torch.nn.init.normal_(lin.weight, std=.3 / np.sqrt(dim))

    def forward(self, xa, xb):
        pa = self.a(xa).view(-1, self.outputs, self.rank)
        pb = self.g(xb).view(-1, self.outputs, self.rank)
        return (pa * pb).sum(-1) + self.bias


class PairMLP(torch.nn.Module):
    def __init__(self, dim: int, width: int = 512):
        super().__init__()
        self.net = MLP(3 * dim, width)

    def forward(self, za, zb):
        return self.net(torch.cat([za, zb, za * zb], 1))


def run_relative(ctx: dict) -> None:
    out = ctx['out']
    folder = out / 'probes' / 'relative'
    folder.mkdir(parents=True, exist_ok=True)
    tr, va, te = ctx['ids']['train'], ctx['ids']['validation'], ctx['ids']['test']
    for stage in ctx['meta']['feature_stages']:
        target = folder / f'{stage}.npz'
        if target.exists():
            continue
        t0 = datetime.now(timezone.utc)
        vision = pr.is_vision(stage)
        feats = np.load(out / 'features' / f'natural_{stage}.npy', mmap_mode='r')
        d = feats.shape[-1]
        x = {s: to_gpu(feats[ids].reshape(-1, d)) for s, ids in (('train', tr), ('validation', va), ('test', te))}
        pairs = {s: pair_table(ctx, ids, vision) for s, ids in (('train', tr), ('validation', va), ('test', te))}
        mu, sd = standardizer(x['train'])
        z = {s: ((v.float() - mu) / sd) for s, v in x.items()}
        del x
        dl = {s: torch.as_tensor(p['delta'], device=DEVICE).float() for s, p in pairs.items()}
        pa = {s: torch.as_tensor(p['a'], device=DEVICE) for s, p in pairs.items()}
        pb = {s: torch.as_tensor(p['b'], device=DEVICE) for s, p in pairs.items()}
        saved, log = {}, {}
        # R1: ridge on standardized feature differences, no intercept (antisymmetric by construction).
        gram = torch.zeros(d, d, dtype=torch.float64, device=DEVICE)
        cross = torch.zeros(d, 2, dtype=torch.float64, device=DEVICE)
        for i in range(0, len(pa['train']), 32768):
            diff = (z['train'][pb['train'][i:i + 32768]] - z['train'][pa['train'][i:i + 32768]]).double()
            gram.addmm_(diff.T, diff)
            cross.addmm_(diff.T, dl['train'][i:i + 32768].double())
        evals, q = torch.linalg.eigh(gram)
        qc = q.T @ cross
        diff_of = lambda s: (z[s][pb[s]] - z[s][pa[s]]).double()
        best = None
        for lam in pr.LAMBDAS:
            w = q @ (qc / (evals.clamp_min(0) + lam)[:, None])
            v = float(torch.square(diff_of('validation') @ w - dl['validation'].double()).mean())
            if best is None or v < best[0]:
                best = (v, lam, w)
        log['r1'] = dict(lambda_=float(best[1]), val_mse=best[0], at_boundary=best[1] in (pr.LAMBDAS[0], pr.LAMBDAS[-1]))
        saved['r1'] = (diff_of('test') @ best[2]).float().cpu().numpy()
        del gram, evals, q
        # R2: whitened PCA projection P (512 x D), pair MLP on [P x_a; P x_b; P x_a * P x_b].
        cov = (z['train'].T @ z['train']) / len(z['train'])
        ev, vec = torch.linalg.eigh(cov.double())
        top = torch.argsort(ev, descending=True)[:PCA_DIM]
        proj = (vec[:, top] / ev[top].clamp_min(1e-8).sqrt()).float()
        p = {s: v @ proj for s, v in z.items()}
        log['r2_pca_variance_kept'] = float(ev[top].sum() / ev.sum())
        del cov, vec

        def r2_step(net, idx, s='train'):
            idx = idx.to(DEVICE)
            return torch.nn.functional.mse_loss(net(p[s][pa[s][idx]], p[s][pb[s][idx]]), dl[s][idx] / 1000)

        def r2_pred(net, s):
            return torch.cat([net(p[s][pa[s][i:i + 65536]], p[s][pb[s][i:i + 65536]]) for i in range(0, len(pa[s]), 65536)]) * 1000

        runs = []
        for lr in (3e-4, 1e-3):
            net, v, ep = train_net(lambda: PairMLP(PCA_DIM), r2_step, len(pa['train']),
                                   lambda n: torch.square(r2_pred(n, 'validation') / 1000 - dl['validation'] / 1000).mean(), lr, 0.1, 0)
            runs.append((v, lr, ep, net))
        v, lr, ep, net = min(runs, key=lambda r: r[0])
        log['r2'] = dict(val_mse_scaled=v, lr=lr, wd=0.1, best_epoch=ep)
        with torch.no_grad():
            saved['r2'] = r2_pred(net, 'test').cpu().numpy()
        del runs, net
        # R3: low-rank bilinear readout of Fourier features of the displacement.
        phi = {s: fourier(v) for s, v in dl.items()}

        # R3 acts on the same whitened 512-d PCA inputs as R2: A_j = A'_j P keeps the low-rank bilinear
        # form in feature space while cutting parameters from 2*32*16*D to 2*32*16*512.
        def r3_step(net, idx, s='train'):
            idx = idx.to(DEVICE)
            return torch.nn.functional.mse_loss(net(p[s][pa[s][idx]], p[s][pb[s][idx]]), phi[s][idx])

        def r3_raw(net, s):
            return torch.cat([net(p[s][pa[s][i:i + 32768]], p[s][pb[s][i:i + 32768]]) for i in range(0, len(pa[s]), 32768)])

        runs = []
        for lr in (1e-3, 3e-3):
            net, v, ep = train_net(lambda: Bilinear(PCA_DIM), r3_step, len(pa['train']),
                                   lambda n: torch.square(r3_raw(n, 'validation') - phi['validation']).mean(), lr, 0.1, 0)
            runs.append((v, lr, ep, net))
        v, lr, ep, net = min(runs, key=lambda r: r[0])
        with torch.no_grad():
            raw = r3_raw(net, 'test')
            saved['r3'] = decode_fourier(raw).cpu().numpy()
            # Per-component test R^2 of phi (order: cos x, sin x, cos y, sin y; k = 1..K within each block).
            t = phi['test']
            saved['r3_phi_r2'] = (1 - torch.square(raw - t).sum(0) / torch.square(t - t.mean(0)).sum(0)).cpu().numpy()
            val_dec = decode_fourier(r3_raw(net, 'validation'))
            log['r3'] = dict(val_phi_mse=v, lr=lr, wd=0.1, best_epoch=ep,
                             val_mae=float((val_dec - dl['validation']).abs().mean()))
        saved['delta'] = pairs['test']['delta']
        saved['owner'] = pairs['test']['owner']
        saved['a'], saved['b'] = pairs['test']['a'], pairs['test']['b']
        np.savez(target, **saved)
        log['seconds'] = (datetime.now(timezone.utc) - t0).total_seconds()
        log['test_mae'] = {k: float(np.abs(saved[k] - saved['delta']).mean()) for k in ('r1', 'r2', 'r3')}
        (folder / f'{stage}.json').write_text(json.dumps(log, indent=1) + '\n')
        print(f'Relative {stage}: ' + ', '.join(f'{k}={v:.1f}' for k, v in log['test_mae'].items()) + f' ({log["seconds"]:.0f}s)', flush=True)
        del z, p, phi, net
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- synthetic internal

def synthetic_splits(rows: list[dict]) -> np.ndarray:
    """Per (template, grid) stratum of development-template patterns: 10 train, 3 validation, 3 test images."""
    split = np.array(['heldout' if r['held_out'] else ('uniform' if r['kind'] == 'uniform' else '') for r in rows], dtype=object)
    rng = np.random.default_rng(70261014)
    strata = {}
    for i, r in enumerate(rows):
        if r['kind'] == 'pattern' and not r['held_out']:
            strata.setdefault((r['template'], r['grid']), []).append(i)
    for members in strata.values():
        perm = rng.permutation(members)
        split[perm[:10]], split[perm[10:13]], split[perm[13:]] = 'train', 'validation', 'test'
    return split


def direct_ridge(x: torch.Tensor, y: torch.Tensor, xv: torch.Tensor, yv: torch.Tensor):
    """Moment ridge on explicit features; lambda selected on validation tokens."""
    m = pr.Moments(x.shape[1], 2, DEVICE)
    mv = pr.Moments(x.shape[1], 2, DEVICE)
    for i in range(0, len(x), 16384):
        m.add(x[i:i + 16384].double(), y[i:i + 16384].double())
    for i in range(0, len(xv), 16384):
        mv.add(xv[i:i + 16384].double(), yv[i:i + 16384].double())
    path = pr.ridge_path(m.__dict__, mv.__dict__, [0, 1])
    best = int(torch.argmax(-path['val_mse'].mean(1)))
    return path['w'][best].float(), path['b'][best].float(), float(pr.LAMBDAS[best])


def run_synthetic(ctx: dict) -> None:
    out = ctx['out']
    folder = out / 'probes' / 'synthetic'
    folder.mkdir(parents=True, exist_ok=True)
    rows = ctx['cases']['synthetic']
    split = synthetic_splits(rows)
    grids = np.array([r['grid'] for r in rows])
    for stage in ctx['meta']['feature_stages']:
        target = folder / f'{stage}.npz'
        if target.exists():
            continue
        vision = pr.is_vision(stage)
        lab = full_labels(rows, vision)
        feats = to_gpu(np.load(out / 'features' / f'synthetic_{stage}.npy', mmap_mode='r'))
        img_split = split[lab['owner']]
        img_grid = grids[lab['owner']]
        y = to_gpu(lab['image']).float()
        sel = lambda mask: torch.as_tensor(np.flatnonzero(mask), device=DEVICE)
        saved, log = {}, {}
        # Internal split: train on development templates, validate, test in distribution and on held-out templates.
        tr, va = sel(img_split == 'train'), sel(img_split == 'validation')
        w, b, lam = direct_ridge(feats[tr], y[tr], feats[va], y[va])
        saved['internal_ridge'] = (feats.float() @ w + b).cpu().numpy()
        log['internal_ridge_lambda'] = lam
        hp = json.loads((out / 'probes' / 'mlp' / f'{stage}.json').read_text())['mlp256']
        net, v, ep = fit_mlp(feats[tr], y[tr], feats[va], y[va], 256, hp['lr'], hp['wd'])
        saved['internal_mlp256'] = predict(net, feats)
        log['internal_mlp256'] = dict(val_mse_scaled=v, best_epoch=ep)
        # Leave-one-grid-size-out on development templates (ridge).
        dev = np.isin(img_split, ['train', 'validation', 'test'])
        loo = np.zeros_like(lab['image'])
        for g in np.unique(grids[[r['kind'] == 'pattern' for r in rows]]):
            tr_m = dev & (img_grid != g) & (img_split != 'validation')
            va_m = dev & (img_grid != g) & (img_split == 'validation')
            w, b, lam = direct_ridge(feats[sel(tr_m)], y[sel(tr_m)], feats[sel(va_m)], y[sel(va_m)])
            mask = img_grid == g
            loo[mask] = (feats[sel(mask)].float() @ w + b).cpu().numpy()
            log[f'grid_out_{g}_lambda'] = lam
        saved['grid_out_ridge'] = loo
        # Uniform canvases: leave-one-grid-size-out within uniform images (ridge, lambda chosen on the other uniform group).
        uni = img_split == 'uniform'
        uloo = np.full_like(lab['image'], np.nan)
        gs = list(np.unique(grids[[r['kind'] == 'uniform' for r in rows]]))
        for k, g in enumerate(gs):
            others = [h for h in gs if h != g]
            tr_m, va_m = uni & (img_grid == others[0]), uni & (img_grid == others[1])
            w, b, lam = direct_ridge(feats[sel(tr_m)], y[sel(tr_m)], feats[sel(va_m)], y[sel(va_m)])
            mask = uni & (img_grid == g)
            uloo[mask] = (feats[sel(mask)].float() @ w + b).cpu().numpy()
            log[f'uniform_out_{g}_lambda'] = lam
        saved['uniform_out_ridge'] = uloo
        np.savez(target, **saved)
        (folder / f'{stage}.json').write_text(json.dumps(log, indent=1) + '\n')
        print(f'Synthetic internal {stage} done', flush=True)
        del feats
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- evaluation

def evaluate(ctx: dict) -> None:
    out = ctx['out']
    meta = ctx['meta']
    natural = ctx['natural']
    te = ctx['ids']['test']
    test_rows = [natural[i] for i in te]
    columns = json.loads((out / 'predictions' / 'columns.json').read_text())
    results = dict(ridge={}, mlp={}, learning_curve={}, relative={}, synthetic={}, baselines={})
    train_means = {}
    for vision in (True, False):
        tm = np.concatenate([pr.token_labels(tuple(natural[i]['patch_hw']), vision)[1] for i in ctx['ids']['train']])
        train_means[vision] = tm.mean(0)
    # ---- ridge on all tokens (test512, test768) and on synthetic images
    syn_rows = ctx['cases']['synthetic']
    syn_kind = np.array([('uniform' if r['kind'] == 'uniform' else ('heldout' if r['held_out'] else 'pattern')) for r in syn_rows])
    syn_grid = np.array([r['grid'] for r in syn_rows])
    syn_split = synthetic_splits(syn_rows)
    conditions = dict(test512=(test_rows, 'patch_hw'), test768=(test_rows, 'patch_hw_768'), synthetic=(syn_rows, 'patch_hw'))
    for cond, (rows, key) in conditions.items():
        preds = np.load(out / 'predictions' / f'{cond}.npz')
        labs = {v: full_labels(rows, v, key) for v in (True, False)}
        res = {}
        for stage in meta['stages']:
            vision = pr.is_vision(stage)
            lab = labs[vision]
            p = preds[stage]
            col = {c: i for i, c in enumerate(columns[stage])}
            entry = {}
            subsets = {'all': np.ones(len(lab['owner']), bool)}
            if cond == 'synthetic':
                subsets = {f'{k}_{g}': (syn_kind[lab['owner']] == k) & (syn_grid[lab['owner']] == g)
                           for k in ('uniform', 'pattern', 'heldout') for g in np.unique(syn_grid)}
                subsets |= {k: syn_kind[lab['owner']] == k for k in ('uniform', 'pattern', 'heldout')}
            for probe in ('rp256', 'rp1024', 'full'):
                for sub, mask in subsets.items():
                    if not mask.any():          # only possible for reduced smoke-test stimulus sets
                        continue
                    owner = np.unique(lab['owner'][mask], return_inverse=True)[1]
                    pi = p[mask][:, [col[f'{probe}_image_x'], col[f'{probe}_image_y']]]
                    m = metric_block(pi, lab['image'][mask], lab['size'][mask], owner, train_means[vision])
                    if cond != 'synthetic':
                        ps = p[mask][:, [col[f'{probe}_image_shuf_x'], col[f'{probe}_image_shuf_y']]]
                        m['shuffled_mae'] = float(np.abs(ps - lab['image'][mask]).mean())
                        m['constant_mae'] = float(np.abs(train_means[vision] - lab['image'][mask]).mean())
                        pg = p[mask][:, [col[f'{probe}_grid_x'], col[f'{probe}_grid_y']]]
                        m['grid_probe_token_mae'] = float(np.abs(pg - lab['grid'][mask]).mean())
                        m['grid_probe_exact'] = float(np.all(np.rint(pg) == lab['grid'][mask], 1).mean())
                        m['grid_probe_within_one'] = float(np.all(np.abs(np.rint(pg) - lab['grid'][mask]) <= 1, 1).mean())
                    entry.setdefault(probe, {})[sub] = m
            if cond == 'test512' and stage in meta['lc_stages']:
                lc = {}
                for name in col:
                    if name.startswith('lc_') and name.endswith('_x'):
                        base = name[:-2]
                        pl = p[:, [col[base + '_x'], col[base + '_y']]]
                        lc[base] = float(np.abs(pl - lab['image']).mean())
                results['learning_curve'].setdefault(stage, {})['ridge'] = lc
            res[stage] = entry
        results['ridge'][cond] = res
        print(f'Evaluated ridge {cond}', flush=True)
    # ---- ridge restricted to the 64 sampled test tokens (same tokens as MLP), test512
    preds = np.load(out / 'predictions' / 'test512.npz')
    for stage in meta['feature_stages']:
        vision = pr.is_vision(stage)
        offsets = preds['offsets_vision' if vision else 'offsets_merged']
        rows_idx = np.concatenate([offsets[j] + sampled_labels(c, vision)['index'] for j, c in enumerate(test_rows)])
        lab = stack_sampled(ctx, te, vision)
        col = {c: i for i, c in enumerate(columns[stage])}
        pi = preds[stage][rows_idx][:, [col['full_image_x'], col['full_image_y']]]
        results['mlp'].setdefault('test512', {}).setdefault(stage, {})['full_ridge_sampled'] = metric_block(
            pi, lab['image'], lab['size'], lab['owner'], train_means[vision])
    # ---- MLP family
    for stage in meta['feature_stages']:
        vision = pr.is_vision(stage)
        saved = np.load(out / 'probes' / 'mlp' / f'{stage}.npz')
        log = json.loads((out / 'probes' / 'mlp' / f'{stage}.json').read_text())
        lab512 = stack_sampled(ctx, te, vision)
        lab768 = stack_sampled(ctx, te, vision, 'patch_hw_768')
        labsyn = full_labels(syn_rows, vision)
        for width in MLP_WIDTHS:
            name = f'mlp{width}'
            r = results['mlp']
            r.setdefault('test512', {}).setdefault(stage, {})[name] = metric_block(
                saved[f'{name}__test512'], lab512['image'], lab512['size'], lab512['owner'], train_means[vision])
            r['test512'][stage][name]['params'] = log[name]['params']
            r['test512'][stage][name]['hp'] = dict(lr=log[name]['lr'], wd=log[name]['wd'], epoch=log[name]['best_epoch'])
            r.setdefault('test768', {}).setdefault(stage, {})[name] = metric_block(
                saved[f'{name}__test768'], lab768['image'], lab768['size'], lab768['owner'], train_means[vision])
            ps = saved[f'{name}__synthetic']
            for k in ('uniform', 'pattern', 'heldout'):
                mask = syn_kind[labsyn['owner']] == k
                if not mask.any():
                    continue
                owner = np.unique(labsyn['owner'][mask], return_inverse=True)[1]
                r.setdefault('synthetic', {}).setdefault(stage, {}).setdefault(name, {})[k] = metric_block(
                    ps[mask], labsyn['image'][mask], labsyn['size'][mask], owner, train_means[vision])
        results['mlp']['test512'][stage]['mlp256_shuffled_mae'] = float(np.abs(saved['mlp256_shuffled__test512'] - lab512['image']).mean())
        results['mlp']['test512'][stage]['constant_mae'] = float(np.abs(train_means[vision] - lab512['image']).mean())
        # ---- relative position
        rel = np.load(out / 'probes' / 'relative' / f'{stage}.npz')
        delta = rel['delta']
        entry = {}
        for k in ('r1', 'r2', 'r3'):
            err = np.abs(rel[k] - delta).mean(1)
            n_img = rel['owner'].max() + 1
            per = np.bincount(rel['owner'], err, n_img) / np.bincount(rel['owner'], minlength=n_img)
            pick = np.random.default_rng(1).integers(0, n_img, (BOOTSTRAP, n_img))
            entry[k] = dict(mae=float(err.mean()), mae_ci95=np.quantile(per[pick].mean(1), [.025, .975]).tolist(),
                            per_image=per.tolist())
        per = {k: np.array(entry[k].pop('per_image')) for k in entry}
        best_rel = 'r2' if entry['r2']['mae'] <= entry['r3']['mae'] else 'r3'
        diff = per['r1'] - per[best_rel]
        pick = np.random.default_rng(2).integers(0, len(diff), (BOOTSTRAP, len(diff)))
        entry['gamma'] = float(entry['r1']['mae'] - min(entry['r2']['mae'], entry['r3']['mae']))
        entry['gamma_ci95'] = np.quantile(diff[pick].mean(1), [.025, .975]).tolist()
        # Displacement implied by the absolute full-ridge and best MLP predictions on the same pairs.
        col = {c: i for i, c in enumerate(columns[stage])}
        offsets = preds['offsets_vision' if vision else 'offsets_merged']
        rows_idx = np.concatenate([offsets[j] + sampled_labels(c, vision)['index'] for j, c in enumerate(test_rows)])
        full_abs = preds[stage][rows_idx][:, [col['full_image_x'], col['full_image_y']]]
        entry['abs_full_ridge'] = dict(mae=float(np.abs(full_abs[rel['b']] - full_abs[rel['a']] - delta).mean()))
        best_mlp = min(MLP_WIDTHS, key=lambda w: results['mlp']['test512'][stage][f'mlp{w}']['mae'])
        mp = saved[f'mlp{best_mlp}__test512']
        entry['abs_best_mlp'] = dict(mae=float(np.abs(mp[rel['b']] - mp[rel['a']] - delta).mean()), width=best_mlp)
        entry['mean_abs_delta'] = float(np.abs(delta).mean())
        entry['r3_phi_r2'] = rel['r3_phi_r2'].tolist()
        results['relative'][stage] = entry
        # ---- synthetic internal
        syn = np.load(out / 'probes' / 'synthetic' / f'{stage}.npz')
        sres = {}
        for key, masks in (('internal_ridge', {'in_distribution': syn_split == 'test', 'heldout': syn_kind == 'heldout'}),
                           ('internal_mlp256', {'in_distribution': syn_split == 'test', 'heldout': syn_kind == 'heldout'}),
                           ('grid_out_ridge', {g: (syn_kind == 'pattern') & (syn_grid == g) for g in np.unique(syn_grid)}),
                           ('uniform_out_ridge', {g: (syn_kind == 'uniform') & (syn_grid == g) for g in np.unique(syn_grid)})):
            for sub, img_mask in masks.items():
                mask = img_mask[labsyn['owner']]
                if not mask.any():
                    continue
                owner = np.unique(labsyn['owner'][mask], return_inverse=True)[1]
                sres.setdefault(key, {})[sub] = metric_block(syn[key][mask], labsyn['image'][mask], labsyn['size'][mask], owner,
                                                              labsyn['image'].mean(0))
        results['synthetic'][stage] = sres
        print(f'Evaluated {stage}', flush=True)
    # ---- MLP learning curves
    for stage in meta['lc_stages']:
        saved = np.load(out / 'probes' / 'mlp_lc' / f'{stage}.npz')
        lab = stack_sampled(ctx, te, False)
        results['learning_curve'].setdefault(stage, {})['mlp256'] = {k: float(np.abs(saved[k] - lab['image']).mean()) for k in saved.files}
    results['meta'] = dict(evaluated_utc=datetime.now(timezone.utc).isoformat(), bootstrap=BOOTSTRAP, bootstrap_unit='image',
                           code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           ridge_summary=json.loads((out / 'ridge_summary.json').read_text()),
                           mlp=dict(widths=MLP_WIDTHS, lrs=MLP_LRS, wds=MLP_WDS, max_epochs=MAX_EPOCHS, patience=PATIENCE, batch=BATCH),
                           relative=dict(pairs_per_image=PAIRS_PER_IMAGE, bins=LENGTH_BINS, pca=PCA_DIM, fourier_k=FOURIER_K,
                                         rank=BILINEAR_RANK, candidate_step=CANDIDATE_STEP))
    (out / 'results.json').write_text(json.dumps(results, indent=1) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=['mlp', 'mlp_lc', 'relative', 'synthetic', 'evaluate', 'all'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', help='restrict feature stages (smoke tests); default: all extracted')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    ctx = load_context(args.output)
    if args.stages:
        meta = ctx['meta']
        meta['feature_stages'] = [s for s in meta['feature_stages'] if s in args.stages]
        meta['lc_stages'] = [s for s in meta['lc_stages'] if s in args.stages]
    steps = ['mlp', 'mlp_lc', 'relative', 'synthetic', 'evaluate'] if args.command == 'all' else [args.command]
    for step in steps:
        dict(mlp=run_mlp, mlp_lc=run_mlp_learning_curve, relative=run_relative, synthetic=run_synthetic, evaluate=evaluate)[step](ctx)


if __name__ == '__main__':
    main()
