"""Experiment 3.14 toy models: which token mixer places row information after the target in raster order?

Each sample is a raster-ordered token grid (H x W tokens, cells of k x k tokens). Every cell row and cell
column is either a figure line or a blank line; cells on two figure lines hold a figure and one figure is
the target. Token input = content embedding (blank / figure / target) + learned absolute row and column
embeddings (the analogue of the ViT position code). Three mixer layers process the image tokens; a
readout block lets a learned query (ROW or COLUMN) attend with softmax attention over the final image
states and classifies the target's ordinal (number of figure lines up to and including the target's).

Mixers (all with a residual MLP; d = 64, 4 heads):
  causal_softmax, bidir_softmax: softmax attention with a causal / no mask;
  causal_delta:  gated linear attention S_t = alpha_t S_{t-1} + beta_t v_t k_t^T, o_t = S_t q_t, with data-dependent
                 alpha_t, beta_t in (0, 1) and a width-4 causal depthwise convolution (Gated DeltaNet without the
                 delta correction, which keeps its decay / write structure and allows a parallel CPU evaluation);
  bidir_delta:   the same recurrence run forward and backward (separate parameters), outputs summed;
  hybrid:        causal_delta, causal_delta, causal_softmax (the 3:1 pattern of Qwen3.5).

Carrier measurement (the 3.11 splice design): run A has the target at cell (r, c); run B moves it one cell
down (row query) or right (column query). The splice input equals run A except at the tokens D of the two
cells, which take run B's tokens. Replacing the readout keys / values of an image-token group by run B's
values gives the recovered fraction r = (LD_e - LD_A) / (LD_B - LD_A) of the logit difference between
B's and A's ordinal. Everything here is a simulation with small trained models (CPU).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis_tools.position.coordinate_flow import token_groups

ARCHS = ('causal_softmax', 'bidir_softmax', 'causal_delta', 'bidir_delta', 'hybrid')
GROUPS = ('obj', 'row_left', 'row_right', 'above', 'below', 'below_cols', 'all_but_obj', 'row_same')


# ----------------------------------------------------------------------------- data

def sample_grids(n: int, hc: int, wc: int, rng: np.random.Generator, p_blank: float = 0.25):
    """Cell-level layouts: content (n, hc, wc) in {0 blank, 1 figure, 2 target}, target cell, ordinals.

    Every layout keeps at least two figure rows and two figure columns, and the target lies on figure lines.
    """
    content = np.zeros((n, hc, wc), np.int64)
    tgt = np.zeros((n, 2), np.int64)
    ords = np.zeros((n, 2), np.int64)
    for i in range(n):
        while True:
            rows = rng.random(hc) > p_blank
            cols = rng.random(wc) > p_blank
            if rows.sum() >= 2 and cols.sum() >= 2:
                break
        content[i] = np.outer(rows, cols).astype(np.int64)
        r = rng.choice(np.flatnonzero(rows))
        c = rng.choice(np.flatnonzero(cols))
        content[i, r, c] = 2
        tgt[i] = (r, c)
        ords[i] = (rows[:r + 1].sum(), cols[:c + 1].sum())
    return content, tgt, ords


def to_tokens(content: np.ndarray, k: int) -> np.ndarray:
    """(n, hc*k * wc*k) raster token contents from cell contents (each cell covers k x k tokens)."""
    n, hc, wc = content.shape
    return np.repeat(np.repeat(content, k, 1), k, 2).reshape(n, -1)


# ----------------------------------------------------------------------------- model

class SoftmaxMixer(nn.Module):
    def __init__(self, d: int, heads: int, causal: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.causal = causal

    def forward(self, x):
        L = x.shape[1]
        mask = torch.ones(L, L, dtype=torch.bool).triu(1) if self.causal else None
        return self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]


class DeltaScan(nn.Module):
    """One direction of gated linear attention with data-dependent decay alpha_t and write strength beta_t.

    Recurrent form: S_t = alpha_t S_{t-1} + beta_t v_t k_t^T, o_t = S_t q_t (the decay / write structure of Gated
    DeltaNet without the delta correction). It is evaluated in the equivalent parallel form
    o_t = sum_{s <= t} exp(G_t - G_s) beta_s (q_t . k_s) v_s with G_t = sum_{u <= t} log alpha_u.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.conv = nn.Conv1d(3 * d, 3 * d, 4, groups=3 * d, padding=3)
        self.ab = nn.Linear(d, 2 * heads)
        self.out = nn.Linear(d, d)

    def forward(self, x, return_gates: bool = False):
        B, L, d = x.shape
        u = self.conv(self.qkv(x).transpose(1, 2))[..., :L].transpose(1, 2)
        q, k, v = F.silu(u).chunk(3, -1)
        q = F.normalize(q.view(B, L, self.h, self.dk), dim=-1).transpose(1, 2)     # (B, h, L, dk)
        k = F.normalize(k.view(B, L, self.h, self.dk), dim=-1).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        a, b = self.ab(x).chunk(2, -1)
        log_alpha = F.logsigmoid(a + 2.0).transpose(1, 2)                          # (B, h, L)
        beta = torch.sigmoid(b).transpose(1, 2)
        G = log_alpha.cumsum(-1)
        decay = G[..., :, None] - G[..., None, :]                                  # (B, h, t, s)
        causal = torch.ones(L, L, dtype=torch.bool).tril()
        w = torch.exp(decay.masked_fill(~causal, float('-inf'))) * beta[..., None, :]
        o = ((q @ k.transpose(-1, -2)) * w) @ v                                    # (B, h, L, dk)
        o = self.out(o.transpose(1, 2).reshape(B, L, d))
        if return_gates:
            return o, log_alpha.exp().transpose(1, 2), beta.transpose(1, 2)
        return o


class DeltaMixer(nn.Module):
    def __init__(self, d: int, heads: int, bidir: bool):
        super().__init__()
        self.fwd = DeltaScan(d, heads)
        self.bwd = DeltaScan(d, heads) if bidir else None

    def forward(self, x):
        o = self.fwd(x)
        if self.bwd is not None:
            o = o + self.bwd(x.flip(1)).flip(1)
        return o


class Block(nn.Module):
    def __init__(self, d: int, mixer: nn.Module):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mixer = mixer
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.mixer(self.n1(x))
        return x + self.mlp(self.n2(x))


class ToyModel(nn.Module):
    def __init__(self, arch: str, H: int, W: int, n_cls: int, d: int = 64, heads: int = 4, n_layers: int = 3):
        super().__init__()
        self.H, self.W = H, W
        self.content = nn.Embedding(3, d)
        self.row = nn.Embedding(H, d)
        self.col = nn.Embedding(W, d)
        kinds = {'causal_softmax': ['cs'] * n_layers, 'bidir_softmax': ['bs'] * n_layers,
                 'causal_delta': ['cd'] * n_layers, 'bidir_delta': ['bd'] * n_layers,
                 'hybrid': ['cd'] * (n_layers - 1) + ['cs']}[arch]
        make = {'cs': lambda: SoftmaxMixer(d, heads, True), 'bs': lambda: SoftmaxMixer(d, heads, False),
                'cd': lambda: DeltaMixer(d, heads, False), 'bd': lambda: DeltaMixer(d, heads, True)}
        self.blocks = nn.ModuleList([Block(d, make[kk]()) for kk in kinds])
        self.norm = nn.LayerNorm(d)
        self.query = nn.Embedding(2, d)                                    # 0 = ROW, 1 = COLUMN
        self.read = nn.MultiheadAttention(d, heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, n_cls))

    def image_states(self, tokens: torch.Tensor) -> torch.Tensor:
        L = tokens.shape[1]
        idx = torch.arange(L)
        x = self.content(tokens) + self.row(idx // self.W) + self.col(idx % self.W)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def readout(self, states: torch.Tensor, qtype: torch.Tensor) -> torch.Tensor:
        q = self.query(qtype)[:, None]
        o = self.read(q, states, states, need_weights=False)[0][:, 0]
        return self.head(q[:, 0] + o)

    def forward(self, tokens, qtype):
        return self.readout(self.image_states(tokens), qtype)


# ----------------------------------------------------------------------------- training and measurement

def train(arch: str, hc: int, wc: int, k: int, steps: int, seed: int, batch: int = 256, lr: float = 2e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    H, W = hc * k, wc * k
    model = ToyModel(arch, H, W, n_cls=max(hc, wc) + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for step in range(steps):
        content, _, ords = sample_grids(batch, hc, wc, rng)
        tok = torch.as_tensor(to_tokens(content, k))
        qt = torch.as_tensor(rng.integers(0, 2, batch))
        y = torch.as_tensor(ords[np.arange(batch), qt.numpy()])
        loss = F.cross_entropy(model(tok, qt), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def evaluate(model: ToyModel, hc: int, wc: int, k: int, n: int, seed: int) -> dict:
    rng = np.random.default_rng(10_000 + seed)
    content, _, ords = sample_grids(n, hc, wc, rng)
    tok = torch.as_tensor(to_tokens(content, k))
    out = {}
    for qt, name in ((0, 'row'), (1, 'col')):
        pred = model(tok, torch.full((n,), qt)).argmax(-1).numpy()
        out[name] = float((pred == ords[:, qt]).mean())
    return out


@torch.no_grad()
def carriers(model: ToyModel, hc: int, wc: int, k: int, n: int, seed: int) -> dict:
    """Mean recovered fraction per token group for row (move down) and column (move right) queries."""
    rng = np.random.default_rng(20_000 + seed)
    H, W = hc * k, wc * k
    res = {'row': {g: [] for g in GROUPS}, 'col': {g: [] for g in GROUPS}}
    count = {'row': 0, 'col': 0}
    tries = 0
    while min(count.values()) < n and tries < 50 * n:
        tries += 1
        content, tgt, _ = sample_grids(1, hc, wc, rng)
        r, c = tgt[0]
        for axis, (dr, dc), qt in (('row', (1, 0), 0), ('col', (0, 1), 1)):
            if count[axis] >= n:
                continue
            # destination: the next figure cell along the axis (crossing blank lines), so the ordinal changes by 1
            rr, cc = r + dr, c + dc
            while rr < hc and cc < wc and content[0, rr, cc] == 0:
                rr, cc = rr + dr, cc + dc
            if rr >= hc or cc >= wc:
                continue
            cb = content.copy()
            cb[0, r, c], cb[0, rr, cc] = 1, 2
            ta, tb = to_tokens(content, k)[0], to_tokens(cb, k)[0]
            D = np.flatnonzero(ta != tb)
            tA = np.flatnonzero(ta == 2).tolist()
            tB = np.flatnonzero(tb == 2).tolist()
            xa = torch.as_tensor(ta)[None]
            xs = xa.clone()
            xs[0, D] = torch.as_tensor(tb[D])
            sa, sb = model.image_states(xa), model.image_states(xs)
            q = torch.tensor([qt])
            la, lb = model.readout(sa, q)[0], model.readout(sb, q)[0]
            dA, dB = int(la.argmax()), int(lb.argmax())
            if dA == dB or float((lb[dB] - lb[dA]) - (la[dB] - la[dA])) < 1.0:
                continue
            groups = token_groups((H, W), tA, tB)
            lda, ldb = float(la[dB] - la[dA]), float(lb[dB] - lb[dA])
            for g in GROUPS:
                idx = groups[g]
                if not idx:
                    res[axis][g].append(0.0)
                    continue
                se = sa.clone()
                se[0, idx] = sb[0, idx]
                le = model.readout(se, q)[0]
                res[axis][g].append((float(le[dB] - le[dA]) - lda) / (ldb - lda))
            count[axis] += 1
    return {a: {g: float(np.mean(v)) if v else None for g, v in d.items()} | {'n': count[a]} for a, d in res.items()}


@torch.no_grad()
def delta_gates(model: ToyModel, hc: int, wc: int, k: int, n: int, seed: int) -> dict | None:
    """Mean alpha of the first causal delta layer at row-start tokens vs other tokens (reset signature)."""
    blk = next((b for b in model.blocks if isinstance(b.mixer, DeltaMixer)), None)
    if blk is None:
        return None
    rng = np.random.default_rng(30_000 + seed)
    content, _, _ = sample_grids(n, hc, wc, rng)
    tok = torch.as_tensor(to_tokens(content, k))
    L = tok.shape[1]
    idx = torch.arange(L)
    x = model.content(tok) + model.row(idx // model.W) + model.col(idx % model.W)
    _, alpha, _ = blk.mixer.fwd(blk.n1(x), return_gates=True)              # first block's input
    a = alpha.mean(-1).numpy()                                             # (n, L)
    start = (np.arange(L) % model.W) == 0
    return {'alpha_row_start': float(a[:, start].mean()), 'alpha_other': float(a[:, ~start].mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--archs', nargs='+', default=list(ARCHS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--layouts', nargs='+', default=['6x8k1', '4x5k2'], help='<cell rows>x<cell cols>k<tokens per cell>')
    p.add_argument('--n-eval', type=int, default=2000)
    p.add_argument('--n-carrier', type=int, default=150)
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'results.jsonl'
    done = set()
    if path.exists():
        done = {(r['arch'], r['layout'], r['seed']) for r in map(json.loads, path.read_text().splitlines())}
    for layout in args.layouts:
        hc, rest = layout.split('x')
        wc, k = rest.split('k')
        hc, wc, k = int(hc), int(wc), int(k)
        for seed in args.seeds:
            for arch in args.archs:
                if (arch, layout, seed) in done:
                    continue
                t0 = time.time()
                model = train(arch, hc, wc, k, args.steps, seed)
                rec = {'arch': arch, 'layout': layout, 'seed': seed, 'steps': args.steps,
                       'acc': evaluate(model, hc, wc, k, args.n_eval, seed),
                       'carriers': carriers(model, hc, wc, k, args.n_carrier, seed),
                       'gates': delta_gates(model, hc, wc, k, 256, seed), 'seconds': round(time.time() - t0, 1)}
                with path.open('a') as f:
                    f.write(json.dumps(rec) + '\n')
                print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
