"""Experiment 7.3 Part C: located fine-tuning of the Gated DeltaNet gates that implement row/column counting.

Results doc: docs/analysis/application/results_2026-11-06_counting_gate_finetune.md.

Gated DeltaNet layer l (1-based decoder index, l % 4 != 0) computes per value head h and token t
    alpha_t = exp(-exp(A_log_h) * softplus(a_t + dt_bias_h)),   a_t = in_proj_a(x_t)[h]   (decay / forgetting)
    beta_t  = sigmoid(b_t),                                     b_t = in_proj_b(x_t)[h]   (write strength)
    S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T.
Experiment 3.10 located the row and column counters in these recurrences (detection heads forget at a new figure
line, long-memory "integrator" heads accumulate).  The adapters here are LoRA updates of in_proj_a and in_proj_b
only (the per-token gate inputs), restricted to image-token positions, so text-only computation is unchanged.

This module is model-free: training stimuli, scopes, parameter counts and scoring.
"""
from __future__ import annotations

import numpy as np

from analysis_tools.behavior import ordinal_mechanism as om

DN_LAYERS = tuple(i for i in range(1, 33) if i % 4)       # the 24 Gated DeltaNet layers (1-based)
HIDDEN, N_HEADS, KEY_DIM, VALUE_DIM = 4096, 32, 16 * 128, 32 * 128
TRAIN_TEMPLATES = ('star', 'house')                       # held out from every 1.5 / 3.8 / 3.10 evaluation image
TRAIN_CELLS = (32, 64)
MIN_PRESENT, MAX_BLANKS = 10, 3


def parse_scope(scope: str) -> dict:
    """'gate:9,10,13' (in_proj_a + in_proj_b of the listed DeltaNet layers) or 'dnall' (every DeltaNet projection:
    in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj of all 24 layers)."""
    if scope == 'dnall':
        return {'kind': 'dnall', 'layers': list(DN_LAYERS)}
    kind, _, spec = scope.partition(':')
    if kind != 'gate' or not spec:
        raise ValueError(f'bad scope {scope!r}')
    layers = sorted(int(x) for x in spec.split(','))
    if any(l not in DN_LAYERS for l in layers) or len(set(layers)) != len(layers):
        raise ValueError(f'{scope!r}: every layer must be a distinct Gated DeltaNet layer')
    return {'kind': 'gate', 'layers': layers}


def lora_params(scope: str, rank: int) -> int:
    s = parse_scope(scope)
    gate = 2 * rank * (HIDDEN + N_HEADS)
    if s['kind'] == 'gate':
        return len(s['layers']) * gate
    per = rank * (HIDDEN + 2 * KEY_DIM + VALUE_DIM) + rank * (HIDDEN + VALUE_DIM) + gate + rank * (VALUE_DIM + HIDDEN)
    return len(s['layers']) * per


def dn_modules(rt_layers, scope: str) -> dict:
    s = parse_scope(scope)
    out = {}
    names = ('in_proj_a', 'in_proj_b') if s['kind'] == 'gate' else ('in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj')
    for l in s['layers']:
        mod = rt_layers[l - 1].linear_attn
        for n in names:
            out[f'L{l}.linear_attn.{n}'] = getattr(mod, n)
    return out


def random_slot_grid(rng: np.random.Generator) -> tuple[om.SlotGrid, int, int]:
    """A 1.5-style slot grid with a random set of present long-axis lines (10-21, up to 3 interior blanks).

    Returns (spec, long-axis count ordinal of the target (1-based), short-axis ordinal (1-based)).
    """
    axis = ('H', 'V')[int(rng.integers(2))]
    cell = int(rng.choice(TRAIN_CELLS))
    template = TRAIN_TEMPLATES[int(rng.integers(len(TRAIN_TEMPLATES)))]
    n_present = int(rng.integers(MIN_PRESENT, om.SLOTS + 1))
    start = int(rng.integers(0, om.SLOTS - n_present + 1))
    lines = list(range(start, start + n_present))
    n_blank = min(int(rng.integers(0, MAX_BLANKS + 1)), max(0, n_present - 3))
    for b in rng.choice(np.arange(1, n_present - 1), size=n_blank, replace=False) if n_blank else []:
        lines[int(b)] = -1
    present = tuple(sorted(x for x in lines if x >= 0))
    slot = int(rng.choice(present))
    other = int(rng.integers(0, om.SHORT))
    spec = om.SlotGrid(template, axis, cell, present, (slot, other))
    return spec, present.index(slot) + 1, other + 1


def rowcol_answer(spec: om.SlotGrid, count: int, other: int) -> tuple[int, int]:
    """(row, column) answer: the long-axis ordinal counts present lines only."""
    return (other, count) if spec.axis == 'H' else (count, other)
