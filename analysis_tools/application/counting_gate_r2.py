"""Experiment 7.3 Part C round 2: gate modifications of the Gated DeltaNet counting layers (training and
inference-time edits).

Results doc: docs/analysis/application/results_2026-11-11_counting_gate_r2.md.

Gated DeltaNet layer l, value head h, token t (counting_gate.py):
    alpha_t = exp(-exp(A_log_h) * softplus(a_t + dt_bias_h)),   a_t = W_a x_t        (decay / forgetting)
    beta_t  = sigmoid(b_t),                                     b_t = W_b x_t        (write strength)
A `GateMod` changes these gate inputs at selected token positions (image tokens or all tokens) by
    a_t -> a'_t with softplus(a'_t + dt_bias_h) = exp(dA_h) * softplus(a_t + dW_a x_t + dd_h + dt_bias_h)
    b_t -> b_t + dW_b x_t + db_h
i.e. a dense or low-rank update of W_a / W_b (dW_a, dW_b: "full" fine-tuning or LoRA of in_proj_a / in_proj_b), a
shift of dt_bias (dd), a shift of A_log (dA; multiplies -log alpha by exp(dA)) and a per-head bias of b (db).
Fixed edits use dA only: dA = log(1 - s) scales -log alpha by (1 - s), so s = 1 removes forgetting (alpha = 1) and
s = 0.5 halves the per-token decay in log space.  The modified a'_t is written back into the in_proj_a output, so the
model's own Gated DeltaNet kernel runs unchanged.

Spec grammar (one string, parts joined by ';', then '@img' or '@all' for the positions):
    full:<layers>               dense dW_a, dW_b and per-head dd, dA, db (trainable)
    lora<r>:<layers>            rank-r LoRA of W_a and W_b (trainable)
    bias:<layers>               per-head dd, dA, db only (trainable)
    decay:<layers>              per-head dA only (trainable)
    fix:<layers>:<heads>:<s>    fixed log-alpha scale (1 - s) on heads 'all', 'nonint' (non-integrator heads of
                                experiment 3.8) or a '/'-separated list
<layers> is a comma list of DeltaNet layers (1-based) and/or ranges 'a-b' (full-attention layers in a range are
skipped; listing one explicitly is an error).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from analysis_tools.application import counting_gate as cg

SCALAR_MULT = 50.0     # trainable per-head scalars are stored as p with value SCALAR_MULT * p (larger effective lr)
LORA_ALPHA_OVER_R = 4.0
NEG_BIG = -1e4


def parse_layers(spec: str) -> list[int]:
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = (int(x) for x in part.split('-'))
            out += [l for l in range(a, b + 1) if l in cg.DN_LAYERS]
        else:
            l = int(part)
            if l not in cg.DN_LAYERS:
                raise ValueError(f'layer {l} is not a Gated DeltaNet layer')
            out.append(l)
    if len(set(out)) != len(out) or not out:
        raise ValueError(f'bad layer list {spec!r}')
    return sorted(out)


def parse_spec(spec: str) -> dict:
    body, sep, pos = spec.rpartition('@')
    if not sep or pos not in ('img', 'all'):
        raise ValueError(f'{spec!r}: must end with @img or @all')
    parts = []
    for p in body.split(';'):
        kind, _, rest = p.partition(':')
        if kind == 'fix':
            layers, heads, s = rest.split(':')
            s = float(s)
            if not 0.0 < s <= 1.0:
                raise ValueError(f'fixed scale s must be in (0, 1], got {s}')
            parts.append({'kind': 'fix', 'layers': parse_layers(layers), 'heads': heads, 's': s})
        elif kind in ('full', 'bias', 'decay') or (kind.startswith('lora') and kind[4:].isdigit()):
            parts.append({'kind': 'lora' if kind.startswith('lora') else kind, 'rank': int(kind[4:]) if kind.startswith('lora') else 0,
                          'layers': parse_layers(rest)})
        else:
            raise ValueError(f'unknown part {p!r}')
    return {'parts': parts, 'positions': pos}


def n_trainable(spec: str) -> int:
    n = 0
    H, D = cg.N_HEADS, cg.HIDDEN
    for p in parse_spec(spec)['parts']:
        per = {'full': 2 * H * D + 3 * H, 'bias': 3 * H, 'decay': H, 'fix': 0,
               'lora': 2 * p.get('rank', 0) * (D + H)}[p['kind']]
        n += per * len(p['layers'])
    return n


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus for y > 0: log(exp(y) - 1) = y + log(1 - exp(-y))."""
    y = y.clamp_min(1e-30)
    return y + torch.log(-torch.expm1(-y))


class GateMod(nn.Module):
    """Modification of the gate inputs (in_proj_a / in_proj_b outputs) of selected DeltaNet layers.

    Exposes the attributes used by targeted_finetune.attach_image_mask (pos_mask, _extra_handles), so image-token
    restriction works exactly as for the round-1 LoRA adapters."""

    def __init__(self, rt_layers, spec: str, integrators: dict[int, list[int]] | None = None):
        super().__init__()
        self.spec = spec
        cfg = parse_spec(spec)
        self.positions = cfg['positions']
        self.mods = {l: rt_layers[l - 1].linear_attn for p in cfg['parts'] for l in p['layers']}
        dev = next(iter(self.mods.values())).in_proj_a.weight.device
        H, D = cg.N_HEADS, cg.HIDDEN
        self.P = nn.ParameterDict()
        self.fixed_dA: dict[int, torch.Tensor] = {}
        for p in cfg['parts']:
            for l in p['layers']:
                k = f'L{l}'
                if p['kind'] == 'full':
                    self.P[f'{k}_Wa'] = nn.Parameter(torch.zeros(H, D, device=dev))
                    self.P[f'{k}_Wb'] = nn.Parameter(torch.zeros(H, D, device=dev))
                if p['kind'] == 'lora':
                    for g in ('a', 'b'):
                        A = torch.empty(p['rank'], D, device=dev)
                        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                        self.P[f'{k}_A{g}'] = nn.Parameter(A)
                        self.P[f'{k}_B{g}'] = nn.Parameter(torch.zeros(H, p['rank'], device=dev))
                if p['kind'] in ('full', 'bias'):
                    for s in ('dd', 'db'):
                        self.P[f'{k}_{s}'] = nn.Parameter(torch.zeros(H, device=dev))
                if p['kind'] in ('full', 'bias', 'decay'):
                    self.P[f'{k}_dA'] = nn.Parameter(torch.zeros(H, device=dev))
                if p['kind'] == 'fix':
                    if p['heads'] == 'all':
                        heads = list(range(H))
                    elif p['heads'] == 'nonint':
                        if integrators is None:
                            raise ValueError('nonint heads need the integrator head sets of experiment 3.8')
                        heads = [h for h in range(H) if h not in integrators.get(l, [])]
                    else:
                        heads = [int(h) for h in p['heads'].split('/')]
                    dA = self.fixed_dA.get(l, torch.zeros(H, device=dev)).clone()
                    dA[heads] += math.log1p(-p['s']) if p['s'] < 1 else -math.inf
                    self.fixed_dA[l] = dA
        self.pos_mask: torch.Tensor | None = None
        self._handles: list = []
        self._extra_handles: list = []

    # ----------------------------------------------------------------- per-layer pieces
    def _get(self, name):
        return self.P[name] if name in self.P else None

    def _delta(self, l: int, g: str, x: torch.Tensor) -> torch.Tensor | None:
        k = f'L{l}'
        W = self._get(f'{k}_W{g}')
        out = None
        if W is not None:
            out = x.float() @ W.t()
        A, B = self._get(f'{k}_A{g}'), self._get(f'{k}_B{g}')
        if A is not None:
            lo = LORA_ALPHA_OVER_R * ((x.float() @ A.t()) @ B.t())
            out = lo if out is None else out + lo
        return out

    def _mask(self, out: torch.Tensor) -> torch.Tensor | None:
        if self.positions == 'all':
            return None
        if self.pos_mask is None:
            raise RuntimeError('image-token GateMod used without a position mask (attach_image_mask not installed)')
        m = self.pos_mask.to(out.device)
        if m.shape != out.shape[:2]:
            raise ValueError(f'pos_mask {tuple(m.shape)} does not match {tuple(out.shape[:2])}')
        return m

    def hook_a(self, l: int):
        mod = self.mods[l]

        def hook(module, inputs, out):
            x = inputs[0]
            a = out.float()
            d = self._delta(l, 'a', x)
            u = a if d is None else a + d
            dd = self._get(f'L{l}_dd')
            if dd is not None:
                u = u + SCALAR_MULT * dd
            dA = self._get(f'L{l}_dA')
            dA = None if dA is None else SCALAR_MULT * dA
            if l in self.fixed_dA:
                dA = self.fixed_dA[l] if dA is None else dA + self.fixed_dA[l]
            dt = mod.dt_bias.float()
            if dA is None:
                new = u
            else:
                none_left = torch.isneginf(dA)
                fin = torch.where(none_left, torch.zeros_like(dA), dA)
                new = inv_softplus(torch.exp(fin) * nn.functional.softplus(u + dt)) - dt
                new = torch.where(none_left, torch.full_like(new, NEG_BIG), new)
            m = self._mask(out)
            if m is not None:
                new = torch.where(m[..., None] > 0, new, a)
            return new.to(out.dtype)
        return hook

    def hook_b(self, l: int):
        def hook(module, inputs, out):
            d = self._delta(l, 'b', inputs[0])
            db = self._get(f'L{l}_db')
            if d is None and db is None:
                return out
            b = out.float()
            new = b + (0 if d is None else d) + (0 if db is None else SCALAR_MULT * db)
            m = self._mask(out)
            if m is not None:
                new = torch.where(m[..., None] > 0, new, b)
            return new.to(out.dtype)
        return hook

    # ----------------------------------------------------------------- interface shared with targeted_finetune.LoRA
    def attach(self) -> None:
        if self._handles:
            raise RuntimeError('GateMod already attached')
        for l, mod in self.mods.items():
            self._handles.append(mod.in_proj_a.register_forward_hook(self.hook_a(l)))
            self._handles.append(mod.in_proj_b.register_forward_hook(self.hook_b(l)))

    def detach(self) -> None:
        for h in self._handles + self._extra_handles:
            h.remove()
        self._handles, self._extra_handles = [], []
        self.pos_mask = None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def delta_norms(self) -> dict:
        return {k: float(v.detach().float().norm()) * (SCALAR_MULT if k.split('_')[-1] in ('dd', 'db', 'dA') else 1.0)
                for k, v in self.P.items()}

    def state(self) -> dict:
        return {'spec': self.spec, 'P': {k: v.detach().cpu() for k, v in self.P.items()}}

    def load(self, state: dict) -> None:
        if state['spec'] != self.spec or set(state['P']) != set(self.P):
            raise ValueError('GateMod state does not match the spec')
        with torch.no_grad():
            for k, v in state['P'].items():
                self.P[k].copy_(v.to(self.P[k].device))
