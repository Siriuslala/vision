"""Experiment 3.12 round 3: supplementary tests of the digit-readout mechanism of Qwen3.5-9B (GPU stages).

Results: docs/analysis/position/results_2026-10-21_digit_failure_r2.md (rewritten in round 3, section on
the units digit and on the relay pathway). Round 2 (`digit_failure_r2.py`) provides the cache-editing
helpers; round 1 (`digit_failure.py`, `digit_failure_run.py`) provides units, prompts and stimuli.

Positions along the answer: q0 = last answer-prefix token (predicts the hundreds digit k), q1 = the token k
(predicts the tens digit t), q2 = the token t (predicts the units digit u). Every stage uses the
prefill-then-step scheme of round 2.

Stages (comma list via --stages):
  units   units-digit readout. Prefill through q0, step with the true k, then step with every t in 0..9
          (rows 0-9) and read the units-digit distribution at q2. Rows 10-15 force the true t and add a
          vector at q1 (layer 24 output) along the tens-digit adjacent-difference direction
          g * (W_U[t+1] - W_U[t-1]) (scales +-0.05, +-0.1) or a random orthogonal direction (+-0.1).
  relay   causal routing of the first-digit adjacent difference from q0 to q1. Rows base / steer+ / steer-
          (vector along g * (W_U[k+1] - W_U[k-1]) added at q0, layer 24 output, scale +-0.1 of the residual
          norm) are prefilled; at the q1 step selected q0 states of a steered row are restored from the base
          row (or the reverse): the q0 slot of the Gated DeltaNet conv state (layers 25-32), the DeltaNet
          recurrent states, the q0 key/value of the full-attention layers, or all three.
  patch3  source patching at q2 on disc pairs (A, B) with the same forced k and t (A's digits) and
          x_B - x_A in [15, 35] units: run B's image KV, conv-state slots, recurrent states are copied into
          run A at the q2 step.
Layer numbers are 1-based.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import digit_failure as df
from analysis_tools.position import digit_failure_r2 as R2
from analysis_tools.position import digit_failure_run as dr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

STEER_LAYER = 24
UNIT_SCALES = (-0.1, -0.05, 0.05, 0.1)
RELAY_SCALE = 0.1
LATE = tuple(range(25, 33))
N_REFCOCO_UNITS = 60


def digits_of(v: float) -> tuple[int, int, int]:
    """Hundreds, tens and units digit of round(v) for 100 <= round(v) <= 999."""
    n = int(round(v))
    if not 100 <= n <= 999:
        raise ValueError(f'Value {v} is not a three-digit number after rounding')
    return n // 100, (n // 10) % 10, n % 10


# ----------------------------------------------------------------------------- inputs and hooks

def unit_inputs3(rt, u: df.Unit) -> dict:
    """Prompt + answer prefix through q0; position ids through q2 (the k and t slots are replaced per row)."""
    pre = df.prefix_ids(rt, u.prefix)
    img = dr.unit_image(u)
    if img is None:
        x0, p0, L = df.text_prepare(rt, u.prompt)
        span = None
    else:
        prep = cr.prepare(rt, img, u.prompt)
        x0, p0, L = cr.embeds(rt, prep), prep.pos, len(prep.ids)
        span = (prep.span['start'], prep.span['end'])
    d1 = df.digit_ids(rt)[1]
    x, p = df.append_tokens(rt, x0, p0, pre + [d1, d1])
    q0 = L + len(pre) - 1
    if q0 < 1:
        raise ValueError(f'Unit {u.id} has no token before q0')
    return {'x': x[:, :q0 + 1], 'pos': p, 'q0': q0, 'span': span, 'L': L}


class ScaledAdd:
    """Add coef[r] * ||h[0, position]|| * dirs[r] to row r of a decoder layer output at one position.

    Row 0 must be an unedited row: its residual norm at that layer sets the scale for every row, so the
    scale is relative to the unsteered residual of the same forward pass.
    """

    def __init__(self, rt, layer: int, coefs: torch.Tensor, dirs: torch.Tensor, position: int):
        if float(coefs[0]) != 0.0:
            raise ValueError('Row 0 must be unsteered (coefficient 0)')

        def hook(module, a, o):
            h = o[0] if isinstance(o, tuple) else o
            h = h.clone()
            n = h[0, position].float().norm()
            h[:, position] += (coefs.to(h.device)[:, None] * n * dirs.to(h.device)).to(h.dtype)
            return (h, *o[1:]) if isinstance(o, tuple) else h
        self.handle = rt.layers[layer - 1].register_forward_hook(hook)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def unit_direction(W_dir: torch.Tensor, d: int) -> torch.Tensor:
    """Unit vector g * (W_U[d+1] - W_U[d-1]) raising the logit of digit d+1 against digit d-1."""
    if not 1 <= d <= 8:
        raise ValueError(f'Adjacent-difference direction needs 1 <= d <= 8, got {d}')
    v = W_dir[d + 1] - W_dir[d - 1]
    return v / v.norm()


def orthogonal_random(v: torch.Tensor, rand: torch.Tensor) -> torch.Tensor:
    r = rand - (rand @ v) * v
    return r / r.norm()


def copy_conv_slots(lin: dict, dst: int, src: int, slots: tuple[int, ...], layers) -> None:
    """Copy selected conv-state slots (-1 = the token right before the step) of row src into row dst."""
    for l, lay in lin.items():
        if l not in layers:
            continue
        for i in lay.conv_states:
            if lay.is_conv_states_initialized[i]:
                for s in slots:
                    lay.conv_states[i][dst, :, s] = lay.conv_states[i][src, :, s]


def step_full(rt, cache, emb: torch.Tensor, pos_t: torch.Tensor) -> torch.Tensor:
    """One cached decoding step returning full-vocabulary log-probabilities restricted to digits (R, 10)."""
    return R2.step(rt, cache, emb, pos_t)


# ----------------------------------------------------------------------------- stage: units digit

def units_units(args) -> list[df.Unit]:
    units = dr.build_units(args)
    out = [u for u in units if u.family in ('sweep_h', 'sweep_v', 'textconv') and 99.5 <= u.truth < 999.5]
    rc = [u for u in units if u.family == 'refcoco' and 99.5 <= u.truth < 999.5][:N_REFCOCO_UNITS]
    return out + rc


@torch.inference_mode()
def run_units(rt, u: df.Unit, W_dir: torch.Tensor, rand: torch.Tensor) -> dict:
    k, t, uu = digits_of(u.truth)
    inp = unit_inputs3(rt, u)
    q0 = inp['q0']
    steer = 1 <= t <= 8
    names = [f't{d}' for d in range(10)]
    coefs = [0.0] * 10
    dirs = [torch.zeros(W_dir.shape[1], device=W_dir.device)] * 10
    if steer:
        v = unit_direction(W_dir, t)
        r = orthogonal_random(v, rand)
        for c in UNIT_SCALES:
            names.append(f'adj{c:+.2f}')
            coefs.append(c)
            dirs.append(v)
        for c in (-0.1, 0.1):
            names.append(f'rand{c:+.2f}')
            coefs.append(c)
            dirs.append(r)
    R = len(names)
    cache = R2.prefill(rt, inp['x'], inp['pos'], [0] * R)
    with ScaledAdd(rt, STEER_LAYER, torch.tensor(coefs), torch.stack(dirs), 0):
        lp1 = step_full(rt, cache, R2.digit_embeds(rt, [k] * R), inp['pos'][:, :, q0 + 1:q0 + 2])
    ts = list(range(10)) + [t] * (R - 10)
    lp2 = step_full(rt, cache, R2.digit_embeds(rt, ts), inp['pos'][:, :, q0 + 2:q0 + 3])
    return {'rows': names, 'k': k, 't': t, 'u': uu, 'logp1': lp1.cpu().numpy(), 'logp2': lp2.cpu().numpy()}


def load_dirs(rt, out: Path, r2_run: Path) -> tuple[torch.Tensor, torch.Tensor]:
    w = np.load(r2_run / 'weights.npz')
    W_dir = torch.tensor(w['W_U'] * w['norm_gain'][None], dtype=torch.float32, device=rt.model.device)
    gen = torch.Generator().manual_seed(0)
    rand = torch.randn(W_dir.shape[1], generator=gen).to(rt.model.device)
    return W_dir, rand


def stage_units(rt, out: Path, args) -> None:
    path = out / 'units.jsonl'
    done = {r['unit'] for r in read_jsonl(path)}
    W_dir, rand = load_dirs(rt, out, Path(args.r2_run))
    units = units_units(args)
    units = units[:args.limit] if args.limit else units
    t0 = time.time()
    for j, u in enumerate(units):
        if u.id in done:
            continue
        res = run_units(rt, u, W_dir, rand)
        append_jsonl(path, [{'unit': u.id, 'family': u.family, 'axis': u.axis, 'truth': u.truth, 'k': res['k'],
                             't': res['t'], 'u': res['u'], 'rows': res['rows'],
                             'logp1': res['logp1'].round(5).tolist(), 'logp2': res['logp2'].round(5).tolist()}])
        if j % 20 == 0:
            print(f'units {j}/{len(units)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- stage: relay routing

RELAY_EDITS = ('conv', 'rec', 'kv', 'all')


def relay_rows() -> list[tuple[str, int, str, int]]:
    """(name, source row, edit, edit source row); source rows: 0 base, 1 steer+, 2 steer-.

    'steer+' = steered row unedited; 'steer+-conv' = steered row whose conv slot of q0 is restored from base;
    'base+conv+' = base row receiving the steered row's conv slot of q0.
    """
    rows = [('base', 0, '', 0)]
    for sgn, src in (('+', 1), ('-', 2)):
        rows.append((f'steer{sgn}', src, '', 0))
        rows += [(f'steer{sgn}-{e}', src, e, 0) for e in RELAY_EDITS]
        rows += [(f'base+{e}{sgn}', 0, e, src) for e in ('conv', 'rec', 'kv')]
    return rows


def apply_edit(full: dict, lin: dict, dst: int, src: int, edit: str, q0: int) -> None:
    if edit in ('conv', 'all'):
        copy_conv_slots(lin, dst, src, (-1,), LATE)
    if edit in ('rec', 'all'):
        R2.copy_delta(lin, dst, src, conv=False, recurrent=True)
    if edit in ('kv', 'all'):
        R2.copy_kv(full, dst, src, [q0], [l for l in R2.FULL_LAYERS if l in LATE])
    if edit not in ('', 'conv', 'rec', 'kv', 'all'):
        raise ValueError(f'Unknown relay edit {edit!r}')


@torch.inference_mode()
def run_relay(rt, u: df.Unit, W_dir: torch.Tensor) -> dict:
    k = df.hundreds(u.truth)
    inp = unit_inputs3(rt, u)
    q0 = inp['q0']
    v = unit_direction(W_dir, k)
    X = inp['x'].expand(3, -1, -1)
    with ScaledAdd(rt, STEER_LAYER, torch.tensor([0.0, RELAY_SCALE, -RELAY_SCALE]), v[None].expand(3, -1), q0):
        o = rt.model(inputs_embeds=X, position_ids=inp['pos'][:, :, :q0 + 1].expand(-1, 3, -1).contiguous(),
                     use_cache=True, logits_to_keep=1)
    dig = torch.tensor(df.digit_ids(rt), device=o.logits.device)
    lp0 = o.logits[:, -1].float().log_softmax(-1)[:, dig]
    rows = relay_rows()
    cache = o.past_key_values
    cache.reorder_cache(torch.tensor([r[1] for r in rows], device=X.device))
    full, lin = R2.cache_layers(rt, cache)
    for i, (_, _, edit, esrc) in enumerate(rows):
        if edit:
            # The edit source row index in the reordered cache: rows 0..2 of the prefill sit at the first
            # reordered row with that source and no edit (base -> 0, steer+ -> 1, steer- -> 7).
            src_row = next(j for j, r in enumerate(rows) if r[1] == esrc and not r[2])
            apply_edit(full, lin, i, src_row, edit, q0)
    lp1 = step_full(rt, cache, R2.digit_embeds(rt, [k] * len(rows)), inp['pos'][:, :, q0 + 1:q0 + 2])
    return {'rows': [r[0] for r in rows], 'k': k, 'logp0': lp0.cpu().numpy(), 'logp1': lp1.cpu().numpy()}


def stage_relay(rt, out: Path, args) -> None:
    path = out / 'relay.jsonl'
    done = {r['unit'] for r in read_jsonl(path)}
    W_dir, _ = load_dirs(rt, out, Path(args.r2_run))
    units = [u for u in dr.build_units(args) if u.family in ('sweep_h', 'sweep_v') and 150 <= u.truth < 850]
    units = units[:args.limit] if args.limit else units
    t0 = time.time()
    for j, u in enumerate(units):
        if u.id in done:
            continue
        res = run_relay(rt, u, W_dir)
        append_jsonl(path, [{'unit': u.id, 'family': u.family, 'axis': u.axis, 'truth': u.truth, 'k': res['k'],
                             'rows': res['rows'], 'logp0': res['logp0'].round(5).tolist(),
                             'logp1': res['logp1'].round(5).tolist()}])
        if j % 20 == 0:
            print(f'relay {j}/{len(units)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- stage: patching at q2

PATCH3_CONDS = ('A', 'B', 'img', 'slot1', 'slot1@25-32', 'slot12', 'rec', 'delta', 'img+delta')


def patch3_pairs(units: list[df.Unit], per_axis: int = 48) -> list[dict]:
    """Directed pairs (A, B) on each disc sweep with 15 <= x_B - x_A <= 35, plus the reversed pairs.

    Both runs are forced to A's hundreds and tens digits (k, t) of round(x_A).
    """
    pairs = []
    for fam in ('sweep_h', 'sweep_v'):
        us = sorted([u for u in units if u.family == fam and 99.5 <= u.truth < 965], key=lambda u: u.truth)
        cand = []
        for a in us:
            bs = [b for b in us if 15 <= b.truth - a.truth <= 35]
            if bs:
                b = min(bs, key=lambda b: abs(b.truth - a.truth - 25))
                cand.append((a, b))
        sel = [cand[i] for i in np.linspace(0, len(cand) - 1, min(per_axis // 2, len(cand))).round().astype(int)]
        for a, b in sel:
            for x, y in ((a, b), (b, a)):
                k, t, _ = digits_of(x.truth)
                pairs.append({'A': x.id, 'B': y.id, 'k': k, 't': t})
    return pairs


@torch.inference_mode()
def run_patch3(rt, ua: df.Unit, ub: df.Unit, k: int, t: int) -> dict:
    ia, ib = unit_inputs3(rt, ua), unit_inputs3(rt, ub)
    if ia['x'].shape != ib['x'].shape or ia['span'] != ib['span']:
        raise ValueError(f'Pair {ua.id}/{ub.id} differs in length or image span')
    q0, (s, e) = ia['q0'], ia['span']
    emb_k = R2.digit_embeds(rt, [k, k]).to(ia['x'].dtype)
    X = torch.cat([torch.cat([ia['x'], ib['x']], 0), emb_k], 1)                  # through q1
    conds = list(PATCH3_CONDS)
    cache = R2.prefill(rt, X, ia['pos'], [1 if c == 'B' else 0 for c in conds])
    full, lin = R2.cache_layers(rt, cache)
    b = conds.index('B')
    for r, c in enumerate(conds):
        for part in c.split('+'):
            if part == 'img':
                R2.copy_kv(full, r, b, list(range(s, e)), R2.FULL_LAYERS)
            elif part == 'slot1':
                copy_conv_slots(lin, r, b, (-1,), range(1, 33))
            elif part == 'slot1@25-32':
                copy_conv_slots(lin, r, b, (-1,), LATE)
            elif part == 'slot12':
                copy_conv_slots(lin, r, b, (-1, -2), range(1, 33))
            elif part == 'rec':
                R2.copy_delta(lin, r, b, conv=False)
            elif part == 'delta':
                R2.copy_delta(lin, r, b)
            elif part not in ('A', 'B'):
                raise ValueError(f'Unknown patch3 condition part {part!r}')
    lp = step_full(rt, cache, R2.digit_embeds(rt, [t] * len(conds)), ia['pos'][:, :, q0 + 2:q0 + 3])
    return {'conds': conds, 'logp': lp.cpu().numpy()}


def stage_patch3(rt, out: Path, args) -> None:
    path = out / 'patch3.jsonl'
    done = {(r['A'], r['B']) for r in read_jsonl(path)}
    units = {u.id: u for u in dr.build_units(args) if u.family in ('sweep_h', 'sweep_v')}
    pairs = patch3_pairs(list(units.values()))
    pairs = pairs[:args.limit] if args.limit else pairs
    t0 = time.time()
    for j, pr in enumerate(pairs):
        if (pr['A'], pr['B']) in done:
            continue
        ua, ub = units[pr['A']], units[pr['B']]
        res = run_patch3(rt, ua, ub, pr['k'], pr['t'])
        append_jsonl(path, [{**pr, 'xA': ua.truth, 'xB': ub.truth, 'axis': ua.axis, 'conds': res['conds'],
                             'logp': res['logp'].round(5).tolist()}])
        if j % 10 == 0:
            print(f'patch3 {j}/{len(pairs)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- main

STAGES = {'units': stage_units, 'relay': stage_relay, 'patch3': stage_patch3}


def code_digest() -> dict:
    root = Path(__file__).parent
    names = ('digit_failure_r3.py', 'digit_failure_r2.py', 'digit_failure.py', 'digit_failure_run.py')
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()[:16] for n in names}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--round1', required=True, help='experiment 3.6 round-1 run (sweep cases and records)')
    p.add_argument('--r2-run', required=True, help='round-2 run directory (weights.npz)')
    p.add_argument('--refcoco-cases', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--limit', type=int, default=0, help='smoke test: process only this many units / pairs per stage')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages,
                         'runtime': qr.describe(rt), 'code': code_digest(), 'limit': args.limit,
                         'steer_layer': STEER_LAYER, 'unit_scales': UNIT_SCALES, 'relay_scale': RELAY_SCALE,
                         'scheme': 'prefill-then-step, batch = rows of one unit or pair'})
    meta.write_text(json.dumps(info, indent=1))
    for s in args.stages.split(','):
        t0 = time.time()
        STAGES[s](rt, out, args)
        print(f'stage {s} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
