"""Experiment 7.3 round 2: formation of ViT detectors for feature-absent targets (model-free part).

Results doc: docs/analysis/application/results_2026-11-11_vit_detector_formation.md.

Contents
* Trainable ViT updates besides LoRA (``located_finetune`` / ``targeted_finetune.LoRA``):
    - ``DenseDelta``: full fine-tuning of selected linear layers, implemented as a float32 additive update
      W + dW, b + db (zero initialised) so that the frozen bf16 weights stay untouched and only dW, db are saved;
    - ``WidthExtension``: m new neurons appended to the MLP of selected ViT blocks,
      out += W_out act(W_in x + b), x = the MLP input (post norm2); W_out starts at zero (model unchanged at step
      0), W_in rows are random with the median row norm of the block's linear_fc1, b starts at the median fc1 bias,
      so a new neuron starts as sparse as an existing one and can become a rectified detector.
  All three share the interface used by the stage drivers: attach / detach / active / state / load / n_params.
* Readout measures of a signal written into the residual stream:
    - ``neuron_contributions``: exact split of a target-minus-distractor projection of an MLP output change into
      per-neuron terms; ``effective_count`` (number of neurons carrying the split) and ``top_share``;
    - ``window_scores`` / ``spearman``: block scores to 3-block window scores and rank agreement.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


# ----------------------------------------------------------------------------- trainable updates

class DenseDelta(nn.Module):
    """Full fine-tuning of nn.Linear modules as a float32 additive update applied by forward hooks."""

    kind = 'full'

    def __init__(self, modules: dict[str, nn.Linear]):
        super().__init__()
        self.names = sorted(modules)
        self._mods = modules
        self.W = nn.ParameterDict()
        self.b = nn.ParameterDict()
        for n in self.names:
            m = modules[n]
            self.W[self._key(n)] = nn.Parameter(torch.zeros(m.out_features, m.in_features, device=m.weight.device))
            self.b[self._key(n)] = nn.Parameter(torch.zeros(m.out_features, device=m.weight.device))
        self.active: set[str] | None = None
        self._handles: list = []

    @staticmethod
    def _key(name: str) -> str:
        return name.replace('.', '__')

    def attach(self) -> None:
        if self._handles:
            raise RuntimeError('already attached')
        for n in self.names:
            W, b = self.W[self._key(n)], self.b[self._key(n)]

            def hook(module, inputs, output, W=W, b=b, n=n):
                if self.active is not None and n not in self.active:
                    return output
                return output + (inputs[0].float() @ W.t() + b).to(output.dtype)
            self._handles.append(self._mods[n].register_forward_hook(hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state(self) -> dict:
        return {'kind': self.kind, 'names': self.names,
                'W': {n: self.W[self._key(n)].detach().cpu() for n in self.names},
                'b': {n: self.b[self._key(n)].detach().cpu() for n in self.names}}

    def load(self, state: dict) -> None:
        if state['names'] != self.names:
            raise ValueError('state does not match the module list')
        with torch.no_grad():
            for n in self.names:
                self.W[self._key(n)].copy_(state['W'][n])
                self.b[self._key(n)].copy_(state['b'][n])

    def delta_norms(self) -> dict[str, float]:
        return {n: float(self.W[self._key(n)].norm()) for n in self.names}


class WidthExtension(nn.Module):
    """m new neurons per ViT block MLP: out += W_out act(W_in x + b) with x the MLP input.

    mlps: {name: Qwen3_5VisionMLP}; the activation is the block's own act_fn (same nonlinearity as the existing
    neurons).  record=True stores the new neurons' activations of the last call per block (analysis).
    """

    kind = 'width'

    def __init__(self, mlps: dict[str, nn.Module], m: int, seed: int = 0):
        super().__init__()
        self.names, self.m, self._mods = sorted(mlps), m, mlps
        self.W_in, self.b_in, self.W_out = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        g = torch.Generator().manual_seed(seed)
        for n in self.names:
            fc1 = mlps[n].linear_fc1
            d = fc1.in_features
            row_norm = float(fc1.weight.float().norm(dim=1).median())
            w = torch.randn(m, d, generator=g)
            w = w / w.norm(dim=1, keepdim=True) * row_norm
            bias = float(fc1.bias.float().median())
            dev = fc1.weight.device
            self.W_in[self._key(n)] = nn.Parameter(w.to(dev))
            self.b_in[self._key(n)] = nn.Parameter(torch.full((m,), bias, device=dev))
            self.W_out[self._key(n)] = nn.Parameter(torch.zeros(mlps[n].linear_fc2.out_features, m, device=dev))
        self.active: set[str] | None = None
        self.record = False
        self.acts: dict[str, torch.Tensor] = {}
        self._handles: list = []

    @staticmethod
    def _key(name: str) -> str:
        return name.replace('.', '__')

    def attach(self) -> None:
        if self._handles:
            raise RuntimeError('already attached')
        for n in self.names:
            k = self._key(n)
            act = self._mods[n].act_fn

            def hook(module, inputs, output, k=k, n=n, act=act):
                if self.active is not None and n not in self.active:
                    return output
                a = act(inputs[0].float() @ self.W_in[k].t() + self.b_in[k])
                if self.record:
                    self.acts[n] = a.detach()
                return output + (a @ self.W_out[k].t()).to(output.dtype)
            self._handles.append(self._mods[n].register_forward_hook(hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state(self) -> dict:
        return {'kind': self.kind, 'names': self.names, 'm': self.m,
                **{f: {n: getattr(self, f)[self._key(n)].detach().cpu() for n in self.names}
                   for f in ('W_in', 'b_in', 'W_out')}}

    def load(self, state: dict) -> None:
        if state['names'] != self.names or state['m'] != self.m:
            raise ValueError('state does not match the module list or width')
        with torch.no_grad():
            for f in ('W_in', 'b_in', 'W_out'):
                for n in self.names:
                    getattr(self, f)[self._key(n)].copy_(state[f][n])

    def delta_norms(self) -> dict[str, float]:
        return {n: float(self.W_out[self._key(n)].norm()) for n in self.names}


def width_params(m: int, n_blocks: int, d: int = 1152) -> int:
    """Parameters of WidthExtension: per block m (d + 1) for W_in, b and d m for W_out."""
    return n_blocks * m * (2 * d + 1)


def full_params(n_blocks: int, d: int = 1152, h: int = 4304) -> int:
    """Parameters of DenseDelta on fc1 and fc2 of n_blocks MLPs."""
    return n_blocks * (2 * d * h + h + d)


# ----------------------------------------------------------------------------- readout measures

def neuron_contributions(out_dir: np.ndarray, act_t: np.ndarray, act_d: np.ndarray) -> np.ndarray:
    """Exact split of v^T (dO_target - mean dO_distractor) into per-neuron terms.

    out_dir: (n_neurons,) projection v^T W[:, j] of each neuron's output weight column on the written direction v;
    act_t: (n_neurons,) activation change (or activation) of each neuron at the target;
    act_d: (n_neurons,) the same averaged over distractors.
    Returns c_j = out_dir_j * (act_t_j - act_d_j); sum_j c_j equals the projection difference because the MLP output
    is linear in the activations (biases cancel in the target-minus-distractor difference).
    """
    return out_dir * (act_t - act_d)


def effective_count(c: np.ndarray) -> float:
    """Effective number of contributing neurons N_eff = (sum |c_j|)^2 / sum c_j^2 (1 = one neuron, n = uniform)."""
    c = np.abs(np.asarray(c, dtype=np.float64))
    s2 = float((c ** 2).sum())
    return float(c.sum() ** 2 / s2) if s2 > 0 else 0.0


def top_share(c: np.ndarray, k: int) -> float:
    """Share of the signed total sum_j c_j carried by the k neurons with the largest c_j."""
    c = np.asarray(c, dtype=np.float64)
    tot = c.sum()
    return float(np.sort(c)[::-1][:k].sum() / tot) if tot != 0 else 0.0


def window_scores(block_scores: dict[int, float], windows: list[tuple[int, ...]]) -> dict[tuple[int, ...], float]:
    """Mean block score over the blocks of each window."""
    return {w: float(np.mean([block_scores[b] for b in w])) for w in windows}


def spearman(x, y) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    def rank(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind='mergesort')
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in np.unique(v):
            idx = v == val
            r[idx] = r[idx].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float('nan')


WINDOWS = [tuple(range(b, b + 3)) for b in range(1, 28, 3)]       # (1,2,3) ... (25,26,27)
