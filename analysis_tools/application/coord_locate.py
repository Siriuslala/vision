"""Experiment 7.2: interpretability-located fine-tuning of the coordinate readout (helpers without a GPU).

Results: docs/analysis/application/results_2026-11-05_located_finetune.md. The component choice is pre-registered
in docs/analysis/position/results_2026-11-05_readout_prior_localization.md (experiment 3.16, base model only).

Contents:
* training conditions of 7.2, registered into analysis_tools.position.targeted_finetune.CONDITIONS at import time
  (the 3.13 LoRA implementation and trainer are reused unchanged);
* token-position masks: all image tokens, image tokens without the first one, the first image token only;
* nested training subsets (data efficiency) stratified by training family;
* BabyVision "Find the different" evaluation cases on the 16 annotated grids (official question, row/column JSON,
  point of the odd instance, point and row/column of a recoloured instance).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from analysis_tools.position import targeted_finetune as tf

MLP_NAMES = tf.MLP_NAMES
DN_IN = tf.DELTANET_IN
FA = tf.FA_NAMES
POSITION_KINDS = ('image', 'image_nofirst', 'first', 'text')

# 7.2 conditions. 'positions' uses POSITION_KINDS (applied through attach_positions, not the 3.13 image hook).
# The located conditions follow the pre-registered prediction of 3.16; controls have (close to) equal parameter counts.
L1_4 = (1, 2, 3, 4)
CONDITIONS_72: dict[str, dict] = {
    # Pre-registered main prediction: MLPs of layers 1-4, update only at image tokens (1.57 M parameters).
    'mlp1_4_img': {'layers': L1_4, 'modules': MLP_NAMES, 'positions': 'image'},
    # Position tests of the same modules: without the first image token; the first image token only; text tokens only.
    'mlp1_4_nofirst': {'layers': L1_4, 'modules': MLP_NAMES, 'positions': 'image_nofirst'},
    'mlp1_4_first': {'layers': L1_4, 'modules': MLP_NAMES, 'positions': 'first'},
    'mlp1_4_text': {'layers': L1_4, 'modules': MLP_NAMES, 'positions': 'text'},
    # Equal-parameter location controls at image tokens: layers 13-16 (largest image-token gradient leverage of the
    # MLPs, but deep), layers 25-28 (after the answer-time reads of layers 20 / 24).
    'mlp13_16_img': {'layers': (13, 14, 15, 16), 'modules': MLP_NAMES, 'positions': 'image'},
    'mlp25_28_img': {'layers': (25, 26, 27, 28), 'modules': MLP_NAMES, 'positions': 'image'},
    # Second training seed of the main condition (data order and LoRA initialisation).
    'mlp1_4_img_s2': {'layers': L1_4, 'modules': MLP_NAMES, 'positions': 'image', 'seed': 2},
    # Format-drift repair through data (train_fix.jsonl: + row/column answers + self-distilled instructions).
    'mlp1_4_img_fix': {'layers': L1_4, 'modules': MLP_NAMES, 'positions': 'image', 'train_file': 'train_fix.jsonl'},
    # Follow-up suggested by the first-token result (not pre-registered): a rank-1 down_proj update of layers 1-4
    # applied at the first image token only (65 k parameters), i.e. a learned image-segment initialisation.
    'down1_4_first_r1': {'layers': L1_4, 'modules': ('mlp.down_proj',), 'positions': 'first', 'rank': 1},
    # Data efficiency: same step count (200) on nested subsets of 202 and 800 samples.
    'mlp1_4_img_n202': {'layers': L1_4, 'modules': MLP_NAMES, 'positions': 'image', 'train_file': 'train_n200.jsonl', 'epochs': 16},
    'mlp1_4_img_n800': {'layers': L1_4, 'modules': MLP_NAMES, 'positions': 'image', 'train_file': 'train_n800.jsonl', 'epochs': 4},
    'all_linear_n202': {'layers': tuple(range(1, 33)), 'modules': tf.MIXER_NAMES + MLP_NAMES, 'train_file': 'train_n200.jsonl', 'epochs': 16},
    'all_linear_n800': {'layers': tuple(range(1, 33)), 'modules': tf.MIXER_NAMES + MLP_NAMES, 'train_file': 'train_n800.jsonl', 'epochs': 4},
    'early1_4_img_n202': {'layers': L1_4, 'modules': tf.MIXER_NAMES + MLP_NAMES, 'positions': 'image', 'train_file': 'train_n200.jsonl', 'epochs': 16},
    'early1_4_img_n800': {'layers': L1_4, 'modules': tf.MIXER_NAMES + MLP_NAMES, 'positions': 'image', 'train_file': 'train_n800.jsonl', 'epochs': 4},
}


def register(extra: dict[str, dict] | None = None) -> None:
    """Add the 7.2 conditions (and optional extra ones) to the 3.13 condition table (no existing key is changed)."""
    for k, v in {**CONDITIONS_72, **(extra or {})}.items():
        if k in tf.CONDITIONS and tf.CONDITIONS[k] != v:
            raise ValueError(f'Condition {k} already exists with another definition')
        tf.CONDITIONS[k] = v


def condition_positions(cond: str) -> str | None:
    if cond == 'base':
        return None
    return tf.CONDITIONS[cond].get('positions')


def ids_mask(ids: torch.Tensor, image_token_id: int, kind: str) -> torch.Tensor:
    """(B, L) float mask from input_ids for one position kind."""
    img = ids == image_token_id
    first = img & ~torch.cat([torch.zeros_like(img[:, :1]), img[:, :-1]], 1)
    if kind == 'image':
        m = img
    elif kind == 'image_nofirst':
        m = img & ~first
    elif kind == 'first':
        m = first
    elif kind == 'text':
        m = ~img
    else:
        raise ValueError(kind)
    return m.float()


def span_mask(kind: str, span: tuple[int, int], total: int) -> np.ndarray:
    """(total,) float mask for an inputs_embeds row whose image tokens occupy [start, end)."""
    s, e = span
    m = np.zeros(total, np.float32)
    if kind == 'image':
        m[s:e] = 1
    elif kind == 'image_nofirst':
        m[s + 1:e] = 1
    elif kind == 'first':
        m[s] = 1
    elif kind == 'text':
        m[:] = 1
        m[s:e] = 0
    else:
        raise ValueError(kind)
    return m


def attach_positions(model, lora: tf.LoRA, image_token_id: int, kind: str):
    """Restrict a LoRA's update to one position kind whenever the model is called with input_ids (training and
    generation); during cached decoding steps no image token is present, so the base weights are used. Calls with
    inputs_embeds must set lora.pos_mask themselves (span_mask)."""
    def pre(module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is not None:
            lora.pos_mask = ids_mask(ids, image_token_id, kind)
    handle = model.model.register_forward_pre_hook(pre, with_kwargs=True)
    lora._extra_handles.append(handle)
    return handle


def nested_subsets(families: list[str], sizes: tuple[int, ...], seed: int) -> dict[int, list[int]]:
    """Nested index subsets (smaller subsets are contained in larger ones), stratified by family: within every family
    a fixed random order is cut at the family's proportional share of each size."""
    fam = np.asarray(families)
    rng = np.random.default_rng(seed)
    order = {f: rng.permutation(np.flatnonzero(fam == f)) for f in sorted(set(families))}
    N = len(fam)
    out = {}
    for n in sizes:
        if n > N:
            raise ValueError(f'subset size {n} exceeds {N}')
        idx = []
        for f, o in order.items():
            idx.extend(o[:int(round(n * len(o) / N))].tolist())
        out[n] = sorted(idx)
    return out


# ----------------------------------------------------------------------------- BabyVision cases

def point_hit(coords: list[float] | None, box: list[float], canvas: tuple[int, int]) -> dict:
    """Point answer (0-1000) against a pixel box on a canvas (W, H): hit = inside the box; error = distance (0-1000
    units) to the box centre."""
    if coords is None or len(coords) != 2:
        return {'valid': False, 'hit': False}
    W, H = canvas
    px = (coords[0] * W / 1000, coords[1] * H / 1000)
    c = ((box[0] + box[2]) / 2 * 1000 / W, (box[1] + box[3]) / 2 * 1000 / H)
    return {'valid': True, 'hit': bool(box[0] <= px[0] <= box[2] and box[1] <= px[1] <= box[3]),
            'error': float(np.hypot(coords[0] - c[0], coords[1] - c[1]))}


def lenient_rowcol(text: str) -> tuple[int, int] | None:
    """Row and column (1-based) from the final answer of an official BabyVision response: the last line starting
    with 'Final answer' if present, else the last line; the first two integers in it."""
    import re
    final = text.split('</think>')[-1].strip()
    lines = [l for l in final.splitlines() if l.strip()]
    if not lines:
        return None
    cand = [l for l in lines if re.match(r'\s*final answer', l, re.I)]
    nums = re.findall(r'\d+', (cand[-1] if cand else lines[-1]))
    return (int(nums[0]), int(nums[1])) if len(nums) >= 2 else None


ORDINAL_ERROR_KINDS = ('correct', 'off_by_1', 'off_by_2_3', 'coord_prefix', 'other', 'unparsed')


def ordinal_error_kind(pred: int | None, truth: int, coord: float) -> str:
    """Class of one row or column answer on a grid. `coord` is the 0-1000 coordinate of the target centre on the
    same axis. 'coord_prefix' marks answers within 1.5 of coord / 10, i.e. the first two digits of the coordinate
    the model would write for the same target (checked only when coord >= 100, where the prefix has two digits and
    lies far outside the 1-20 range of row and column indices); it is tested after the near-miss classes."""
    if pred is None:
        return 'unparsed'
    d = abs(pred - truth)
    if d == 0:
        return 'correct'
    if d == 1:
        return 'off_by_1'
    if d <= 3:
        return 'off_by_2_3'
    if coord >= 100 and abs(pred - coord / 10) <= 1.5:
        return 'coord_prefix'
    return 'other'


def dumps(obj) -> str:
    return json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
