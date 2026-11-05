"""Experiment 3.13 round 2 stage drivers (results: docs/analysis/position/results_2026-11-03_targeted_finetune_r2.md).

Stages (comma list via --stages):
  link        (CPU) set up the round-2 run directory: links to the round-1 samples, adapters and evaluations, so the
              round-1 stages (train / coord / general / factorial / spotcheck) run unchanged on new conditions;
  splice      teacher-forced digit readout on the disc sweeps for every splice configuration of one adapter
              (--condition): adapter active on a layer subset / component type / token-position subset;
  splicegen   greedy coordinate answers on the disc sweeps and 3.12 grids for the depth and type splices;
  capture     residual stream after every layer (0-32) at q0, q1 and the mean of the target image tokens, on dense
              disc sweeps (x and y), text-given coordinates (3.12 'textconv') and RefCOCO test points, for one condition;
  jacobian    Jacobians of the ten digit logits at q0 / q1 with respect to the residual after layers 16 / 20 / 24 / 27
              at q0 / q1 (pairs q0->q0, q1->q1, q0->q1; x sweep), for one condition;
  clock       causal tests of the period-100 clock: project its plane out of the tuned residual (q0 / image tokens)
              or add the tuned clock to the base residual; teacher-forced digit readout on the disc sweeps;
  steer       is the early change a constant shift? add the mean layer-4 image-token change of the layers-1-4
              hybrid to the base model's image tokens and measure the digit readout;
  scaledprep  (CPU) round-2 scaled training set: round-1 data + more RefCOCO + GQA scene-graph expressions, and the
              prompts of the self-distilled instruction samples;
  distill     base-model answers to the instruction prompts; writes train_scaled.jsonl;
  train, coord, general, factorial, spotcheck: the round-1 stages (targeted_finetune_run) on the round-2 directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import coordinate_value as cv
from analysis_tools.position import coordinate_value_r2 as c2
from analysis_tools.position import digit_failure as df
from analysis_tools.position import digit_failure_run as dfr
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_r2 as t2
from analysis_tools.position import targeted_finetune_run as tfr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

SEED = 20261103
CAPTURE_LAYERS = tuple(range(0, 33))
JACOBIAN_LAYERS = (16, 20, 24, 27)
ROUND1_CONDITIONS = ('base', 'all_linear', 'mlp28_32', 'mlp28_32_r64', 'late25_32', 'mixer16_24', 'mlp10_14')


# ----------------------------------------------------------------------------- link

def stage_link(out: Path, args) -> None:
    r1 = Path(args.round1_run)
    for name in ('train.jsonl', 'test.jsonl'):
        dst = out / name
        if not dst.exists():
            dst.symlink_to(r1 / name)
    (out / 'adapters').mkdir(exist_ok=True)
    (out / 'eval').mkdir(exist_ok=True)
    for cond in ROUND1_CONDITIONS:
        if cond != 'base':
            for suffix in ('.pt', '_log.jsonl'):
                dst = out / 'adapters' / f'{cond}{suffix}'
                if not dst.exists():
                    dst.symlink_to(r1 / 'adapters' / f'{cond}{suffix}')
        dst = out / 'eval' / cond
        if not dst.exists():
            dst.symlink_to(r1 / 'eval' / cond)
    print('linked round-1 samples, adapters and evaluations', flush=True)


# ----------------------------------------------------------------------------- rows

def disc_units(args, every: int, families: tuple[str, ...]) -> list[df.Unit]:
    """Disc-sweep units (x: 'sweep_h', y: 'sweep_v') and text-given coordinates ('textconv', x only) with the first
    digit forced to the true digit; values outside 100-999 are dropped (their first digit has another width)."""
    r1 = Path(args.round1)
    recs = read_jsonl(r1 / 'gen' / 'records.jsonl')
    cases = {c['id']: c for c in json.loads((r1 / 'cases.json').read_text())}
    units = []
    if 'sweep_h' in families:
        for r in dfr.sweep(recs, 'sweep_h', every):
            units.append(df.Unit(r['id'], 'sweep_h', 0, r['truth_units'][0], cv.POINT_PROMPT, df.point_prefix(0, None),
                                 source={'case': r['id'], 'center': cases[r['id']]['center']}))
    if 'sweep_v' in families:
        for r in dfr.sweep(recs, 'sweep_v', every):
            units.append(df.Unit(r['id'], 'sweep_v', 1, r['truth_units'][1], cv.POINT_PROMPT, df.point_prefix(1, 500),
                                 ctx=500, source={'case': r['id'], 'center': cases[r['id']]['center']}))
    if 'textconv' in families:
        for r in dfr.sweep(recs, 'sweep_h', every):
            px, py = (int(round(v)) for v in cases[r['id']]['center'])
            units.append(df.Unit(f"tc_{r['id']}", 'textconv', 0, 1000 * px / c2.CANVAS, c2.TEXT_QUERY.format(px=px, py=py),
                                 df.point_prefix(0, None), source={'case': r['id'], 'center': cases[r['id']]['center']}))
    if 'refcoco' in families:
        # RefCOCO point questions of the round-1 test split (x studied); the target tokens are those overlapping the box.
        for smp in tf.read_samples(Path(args.output) / 'test.jsonl'):
            if smp.family == 'refcoco' and smp.fmt == 'point':
                W = smp.truth['canvas'][0]
                units.append(df.Unit(f'rc_{smp.id}', 'refcoco', 0, 1000 * smp.truth['center'][0] / W, smp.prompt,
                                     df.point_prefix(0, None), source={'image': smp.image, 'box': smp.truth['box']}))
    keep = []
    for u in units:
        if 100 <= round(u.truth) < 1000:
            u.ks = (int(round(u.truth)) // 100,)
            keep.append(u)
    return keep


def build_rows(rt: qr.Runtime, units: list[df.Unit]) -> list[dict]:
    """Per unit: inputs_embeds / position ids with the answer prefix and the true first digit appended, the
    positions q0 (token before the first digit) and q1 (the first digit), the image span, the prompt length and
    the raster positions of the image tokens overlapping the disc."""
    digit_ids = df.digit_ids(rt)
    rows = []
    for u in units:
        if 'image' in u.source:
            img, truth = Image.open(u.source['image']).convert('RGB'), {'box': u.source['box']}
        else:
            img, truth = cr.disc_image(tuple(u.source['center']), cv.DIAMETER, cv.CANVAS)
        prep = cr.prepare(rt, img, u.prompt)
        pre = df.prefix_ids(rt, u.prefix)
        x, p = df.append_tokens(rt, cr.embeds(rt, prep), prep.pos, pre + [digit_ids[u.ks[0]]])
        L = len(prep.ids)
        s = prep.span['start']
        target = [s + i for i in cr.tokens_overlapping(truth['box'], prep.grid)]
        rows.append({'unit': u, 'x': x, 'p': p, 'q0': L + len(pre) - 1, 'q1': L + len(pre), 'span': (s, prep.span['end']),
                     'prompt_len': L, 'length': x.shape[1], 'target': target})
    return rows


def batches(rows: list[dict], size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def set_mask(lora, chunk: list[dict], positions: str | None) -> None:
    """pos_mask of a right-padded batch for the given position kind (None / 'all' = no mask)."""
    if lora is None:
        return
    if positions in (None, 'all'):
        lora.pos_mask = None
        return
    Ltot = max(r['length'] for r in chunk)
    lora.pos_mask = torch.tensor(np.stack([t2.position_mask(positions, r['length'], r['span'], r['prompt_len'], Ltot)
                                           for r in chunk]))


def intrinsic_positions(cond: str) -> str | None:
    """Token positions an adapter was trained on ('image' for image-position conditions, else None)."""
    return None if cond == 'base' else tf.CONDITIONS[cond].get('positions')


def light_forward(rt: qr.Runtime, chunk: list[dict], layers: tuple[int, ...] = (), grad: bool = False,
                  deltas: dict | None = None, edits: list | None = None) -> dict:
    """Right-padded batched prefill. Returns digit log-probabilities at q0 / q1 ('logp' (B, 2, 10), full-vocabulary
    softmax restricted to the digit tokens) and, for every layer in `layers`, the residual after that layer at q0, q1
    and the mean over the target tokens ('resid' layer -> (B, 3, D) float32).

    deltas: optional {(layer, role): (B, D) tensor} added to the residual after `layer` at q0 (role 0) or q1 (role 1);
    with grad=True the returned digit logits ('logit' (B, 2, 10)) keep the graph for Jacobians.
    edits: optional list of (layer, fn) with fn(h (B, L, D), chunk) -> h applied to the residual after `layer`.
    """
    X, P = c2.pad_batch([r['x'] for r in chunk], [r['p'] for r in chunk])
    B, Ltot = X.shape[:2]
    dev = X.device
    b = torch.arange(B, device=dev)
    q = torch.tensor([[r['q0'], r['q1']] for r in chunk], device=dev)
    Wt = torch.zeros(B, Ltot, device=dev)
    for i, r in enumerate(chunk):
        Wt[i, r['target']] = 1.0 / len(r['target'])
    lm = rt.model.model.language_model
    store: dict = {}
    handles = []

    def grab(l, h):
        h = h[0] if isinstance(h, tuple) else h
        sites = h[b[:, None].to(h.device), q.to(h.device)].float()
        tgt = torch.einsum('bl,bld->bd', Wt.to(h.device), h.float())
        store[l] = torch.cat([sites, tgt[:, None]], 1).detach().cpu()

    if 0 in layers:
        handles.append(lm.layers[0].register_forward_pre_hook(
            lambda m, a, kw: grab(0, kw.get('hidden_states', a[0] if a else None)), with_kwargs=True))
    for l in layers:
        if l > 0:
            handles.append(lm.layers[l - 1].register_forward_hook(lambda m, a, o, l=l: grab(l, o)))
    for l, fn in (edits or []):
        def edit(m, a, o, fn=fn):
            h = o[0] if isinstance(o, tuple) else o
            h = fn(h, chunk)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        handles.append(lm.layers[l - 1].register_forward_hook(edit))
    if deltas:
        by_layer: dict = {}
        for (l, role), d in deltas.items():
            by_layer.setdefault(l, []).append((role, d))

        def add(m, a, o, items):
            h = o[0] if isinstance(o, tuple) else o
            h = h.clone()
            for role, d in items:
                idx = q[:, role].to(h.device)
                h[b.to(h.device), idx] = h[b.to(h.device), idx] + d.to(h.dtype)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        for l, items in by_layer.items():
            handles.append(lm.layers[l - 1].register_forward_hook(lambda m, a, o, items=items: add(m, a, o, items)))
    normed = {}
    handles.append(lm.norm.register_forward_hook(
        lambda m, a, o: normed.__setitem__('h', o[b[:, None].to(o.device), q.to(o.device)])))
    ctx = torch.enable_grad() if grad else torch.inference_mode()
    try:
        with ctx:
            rt.model(inputs_embeds=X, position_ids=P, logits_to_keep=1)
            head = rt.model.lm_head
            dig = torch.tensor(df.digit_ids(rt), device=head.weight.device)
            h = normed['h'].to(head.weight.device)
            if grad:
                logit = (h.float() @ head.weight[dig].float().T)                  # (B, 2, 10)
                return {'logit': logit, 'resid': store}
            full = head(h).float().log_softmax(-1)                                  # (B, 2, V)
            return {'logp': full[..., dig].cpu().numpy(), 'resid': store}
    finally:
        for hd in handles:
            hd.remove()


# ----------------------------------------------------------------------------- splice

def load_lora(rt: qr.Runtime, out: Path, cond: str) -> tf.LoRA:
    """Adapter for the analysis passes (inputs_embeds; masks of image-position adapters are set per batch)."""
    return tf.load_adapter(rt.layers, out / 'adapters' / f'{cond}.pt', cond)


def stage_splice(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    lora = load_lora(rt, out, cond)
    configs = t2.splice_configs(lora.names, full=cond in ('all_linear', 'scaled_all_linear'))
    d = out / 'splice'
    d.mkdir(exist_ok=True)
    path = d / f'{cond}.jsonl'
    done = {r['config'] for r in read_jsonl(path)} if path.exists() else set()
    rows = build_rows(rt, disc_units(args, 2, ('sweep_h', 'sweep_v')))
    t0 = time.time()
    try:
        for cfg in configs:
            if cfg.name in done:
                continue
            lora.active = set(cfg.active)
            res = []
            own = intrinsic_positions(cond)
            if own is not None and cfg.positions not in ('all', own):
                continue                       # an image-position adapter has no update elsewhere
            for chunk in batches(rows, args.splice_batch):
                set_mask(lora, chunk, own if cfg.positions == 'all' else cfg.positions)
                o = light_forward(rt, chunk)
                for r, lp in zip(chunk, o['logp']):
                    u = r['unit']
                    res.append({'config': cfg.name, 'group': cfg.group, 'positions': cfg.positions,
                                'n_active': len(cfg.active), 'unit': u.id, 'family': u.family, 'truth': u.truth,
                                'k': u.ks[0], 'logp0': lp[0].tolist(), 'logp1': lp[1].tolist()})
            append_jsonl(path, res)
            m = t2.digit_metrics(np.array([r['truth'] for r in res]), np.array([r['logp0'] for r in res]),
                                 np.array([r['logp1'] for r in res]))
            print(f"splice {cond} {cfg.name}: gain {m['gain']:.2f} p1 {m['p_second']:.2f} p0 {m['p_first']:.2f} "
                  f"err {m['value_err']:.1f} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        lora.active, lora.pos_mask = None, None
        lora.detach()


SPLICEGEN_CONFIGS = ('base', 'full', 'fwd4', 'fwd8', 'fwd16', 'fwd20', 'fwd24', 'rev4', 'rev8', 'rev16', 'rev24', 'band5_8',
                     'mlp', 'mixer', 'fa', 'dn', 'full@image', 'fwd24@image')


def stage_splicegen(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    lora = load_lora(rt, out, cond)
    configs = {c.name: c for c in t2.splice_configs(lora.names)}
    d = out / 'splice'
    d.mkdir(exist_ok=True)
    path = d / f'{cond}_gen.jsonl'
    done = {r['config'] for r in read_jsonl(path)} if path.exists() else set()
    samples = [s for s in tf.read_samples(out / 'test.jsonl') if s.family in ('sweep_h', 'sweep_v', 'grid06', 'grid10', 'grid20')]
    t0 = time.time()
    try:
        for name in SPLICEGEN_CONFIGS:
            if name in done:
                continue
            cfg = configs[name]
            if cfg.positions not in ('all', 'image'):
                raise ValueError(f'splicegen supports image-position masks only: {name}')
            lora.active = set(cfg.active)
            handle = tf.attach_image_mask(rt.model, lora, rt.model.config.image_token_id) if cfg.positions == 'image' else None
            outs = tfr.generate_samples(rt, samples, args.token_budget, args.max_new_tokens)
            if handle is not None:
                handle.remove()
                lora._extra_handles.remove(handle)
                lora.pos_mask = None
            append_jsonl(path, [{'config': name, 'id': s.id, 'family': s.family, 'text': o['text'],
                                 'batch_size': o['batch_size'], **tf.score(s, o['text'])} for s, o in zip(samples, outs)])
            print(f'splicegen {cond} {name} ({time.time() - t0:.0f}s)', flush=True)
    finally:
        lora.active = None
        lora.detach()


# ----------------------------------------------------------------------------- capture and Jacobians

def stage_capture(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    lora = None if cond == 'base' else load_lora(rt, out, cond)
    cfg = None
    if args.splice_config:
        cfg = {c.name: c for c in t2.splice_configs(lora.names)}[args.splice_config]
        lora.active = set(cfg.active)
    positions = intrinsic_positions(cond) if cfg is None or cfg.positions == 'all' else cfg.positions
    d = Path(args.features) / 'capture'
    d.mkdir(parents=True, exist_ok=True)
    path = d / (f'{cond}.npz' if cfg is None else f'{cond}+{cfg.name}.npz')
    if path.exists():
        print(f'{path} exists', flush=True)
        return
    rows = build_rows(rt, disc_units(args, 1, ('sweep_h', 'sweep_v')) + disc_units(args, 2, ('textconv', 'refcoco')))
    resid, logp = [], []
    t0 = time.time()
    for chunk in batches(rows, args.splice_batch):
        set_mask(lora, chunk, positions)
        o = light_forward(rt, chunk, CAPTURE_LAYERS)
        resid.append(np.stack([o['resid'][l].numpy() for l in CAPTURE_LAYERS], 2).astype(np.float16))   # (B, 3, 33, D)
        logp.append(o['logp'])
    np.savez(path, resid=np.concatenate(resid), logp=np.concatenate(logp),
             truth=np.array([r['unit'].truth for r in rows]), family=np.array([r['unit'].family for r in rows]),
             unit=np.array([r['unit'].id for r in rows]), layers=np.array(CAPTURE_LAYERS))
    link = out / 'capture'
    if not link.exists():
        link.symlink_to(d)
    if lora is not None:
        lora.detach()
    print(f'capture {cond}: {len(rows)} rows ({time.time() - t0:.0f}s)', flush=True)


def stage_jacobian(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    lora = None if cond == 'base' else load_lora(rt, out, cond)
    d = Path(args.features) / 'jacobian'
    d.mkdir(parents=True, exist_ok=True)
    path = d / f'{cond}.npz'
    if path.exists():
        print(f'{path} exists', flush=True)
        return
    rows = build_rows(rt, disc_units(args, 2, ('sweep_h',)))
    D = rows[0]['x'].shape[-1]
    # Pairs (logit position, residual position): (q0, q0), (q1, q1) and the cross pair (q1 logits, q0 residual), which
    # carries the 3.12 round-2 relay of the q0 state to the second digit through the late DeltaNet convolutions.
    pairs = ((0, 0), (1, 1), (1, 0))
    J = np.zeros((len(rows), len(JACOBIAN_LAYERS), len(pairs), 10, D), np.float32)
    t0 = time.time()
    i0 = 0
    for chunk in batches(rows, args.jacobian_batch):
        B = len(chunk)
        set_mask(lora, chunk, intrinsic_positions(cond))
        deltas = {(l, r): torch.zeros(B, D, device=rt.device, requires_grad=True) for l in JACOBIAN_LAYERS for r in (0, 1)}
        o = light_forward(rt, chunk, grad=True, deltas=deltas)
        logit = o['logit']
        for pi, (out_role, in_role) in enumerate(pairs):
            for dg in range(10):
                grads = torch.autograd.grad(logit[:, out_role, dg].sum(), [deltas[(l, in_role)] for l in JACOBIAN_LAYERS],
                                            retain_graph=True)
                for li, g in enumerate(grads):
                    J[i0:i0 + B, li, pi, dg] = g.float().cpu().numpy()
        del o, logit, deltas
        i0 += B
    np.savez(path, J=J, pairs=np.array(pairs), truth=np.array([r['unit'].truth for r in rows]),
             unit=np.array([r['unit'].id for r in rows]), layers=np.array(JACOBIAN_LAYERS))
    link = out / 'jacobian'
    if not link.exists():
        link.symlink_to(d)
    if lora is not None:
        lora.detach()
    print(f'jacobian {cond}: {len(rows)} rows ({time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- clock ablation / injection

CLOCK_LAYERS_IMAGE = (20, 22, 24)


def clock_plane(H: np.ndarray, x: np.ndarray, period: float = 100.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clock component of residuals H (N, D) for coordinates x: least squares of H on [Legendre(x), cos, sin, 1];
    returns the orthonormal plane basis (2, D) spanned by the cos / sin coefficient vectors and the two coefficient
    vectors themselves (c_cos, c_sin) (D,), so that the fitted clock is c_cos cos(2 pi x / T) + c_sin sin(2 pi x / T)."""
    X1 = np.concatenate([t2.smooth_basis(x, 5), t2.periodic_basis(x, period), np.ones((len(x), 1))], 1)
    coef, *_ = np.linalg.lstsq(X1, H, rcond=None)
    c_cos, c_sin = coef[5], coef[6]
    return t2.orthonormal(np.stack([c_cos, c_sin])), c_cos, c_sin


def stage_clock(rt: qr.Runtime, out: Path, args) -> None:
    """Causal tests of the period-100 clock (planes estimated from the all-linear capture, per axis).

    tuned model (all_linear adapter): project the clock plane out of the residual (h <- h - (h - mu) P^T P, mu = the
    mean residual of the capture at that site) at q0 after layer 24, at all image tokens after layers 20 / 22 / 24, or
    both; a random plane of the same dimension is the control. base model: add the tuned model's fitted clock
    c_cos cos(2 pi x / 100) + c_sin sin(2 pi x / 100) at q0 after layer 24 (and at the target tokens after 20 / 22 /
    24). Teacher-forced digit readout on the disc sweeps as in 'splice'."""
    C = np.load(out / 'capture' / 'all_linear.npz')
    li = {l: list(C['layers']).index(l) for l in (24,) + CLOCK_LAYERS_IMAGE}
    planes = {}
    rng = np.random.default_rng(SEED)
    for fam in ('sweep_h', 'sweep_v'):
        m = C['family'] == fam
        x = C['truth'][m].astype(float)
        R = C['resid'][m].astype(np.float32)
        P, cc, cs = clock_plane(R[:, 0, li[24]], x)
        planes[(fam, 'q0', 24)] = (P, R[:, 0, li[24]].mean(0), cc, cs)
        for l in CLOCK_LAYERS_IMAGE:
            P, cc, cs = clock_plane(R[:, 2, li[l]], x)
            planes[(fam, 'img', l)] = (P, R[:, 2, li[l]].mean(0), cc, cs)
        planes[(fam, 'rand', 0)] = (t2.orthonormal(rng.normal(size=(2, R.shape[-1]))), None, None, None)
    dev = rt.device
    T = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)

    def project_q0(fam_key):
        def fn(h, chunk):
            h = h.clone()
            for i, r in enumerate(chunk):
                P, mu, _, _ = planes[(r['unit'].family, fam_key, 24)] if fam_key != 'rand' else planes[(r['unit'].family, 'rand', 0)]
                v = h[i, r['q0']].float()
                c = v - (T(mu) if mu is not None else 0)
                h[i, r['q0']] = (v - (c @ T(P).T) @ T(P)).to(h.dtype)
            return h
        return fn

    def project_img(l, rand=False):
        def fn(h, chunk):
            h = h.clone()
            for i, r in enumerate(chunk):
                P, mu, _, _ = planes[(r['unit'].family, 'rand', 0)] if rand else planes[(r['unit'].family, 'img', l)]
                s_, e_ = r['span']
                v = h[i, s_:e_].float()
                c = v - (T(mu) if mu is not None else 0)
                h[i, s_:e_] = (v - (c @ T(P).T) @ T(P)).to(h.dtype)
            return h
        return fn

    def inject(site, l):
        def fn(h, chunk):
            h = h.clone()
            for i, r in enumerate(chunk):
                u = r['unit']
                _, _, cc, cs = planes[(u.family, site, l)]
                a = 2 * np.pi * u.truth / 100
                add = T(cc * np.cos(a) + cs * np.sin(a))
                if site == 'q0':
                    h[i, r['q0']] = (h[i, r['q0']].float() + add).to(h.dtype)
                else:
                    h[i, r['target']] = (h[i, r['target']].float() + add).to(h.dtype)
            return h
        return fn

    tests = {
        'all_linear': {'none': [], 'ablate_q0_L24': [(24, project_q0('q0'))], 'rand_q0_L24': [(24, project_q0('rand'))],
                       'ablate_img_L20_24': [(l, project_img(l)) for l in CLOCK_LAYERS_IMAGE],
                       'rand_img_L20_24': [(l, project_img(l, True)) for l in CLOCK_LAYERS_IMAGE],
                       'ablate_both': [(l, project_img(l)) for l in CLOCK_LAYERS_IMAGE] + [(24, project_q0('q0'))]},
        'base': {'none': [], 'inject_q0_L24': [(24, inject('q0', 24))],
                 'inject_img_L20_24': [(l, inject('img', l)) for l in CLOCK_LAYERS_IMAGE],
                 'inject_both': [(l, inject('img', l)) for l in CLOCK_LAYERS_IMAGE] + [(24, inject('q0', 24))]},
    }
    rows = build_rows(rt, disc_units(args, 2, ('sweep_h', 'sweep_v')))
    path = out / 'splice' / 'clock.jsonl'
    path.parent.mkdir(exist_ok=True)
    done = {(r['model'], r['test']) for r in read_jsonl(path)} if path.exists() else set()
    for model_name, tt in tests.items():
        lora = load_lora(rt, out, 'all_linear') if model_name == 'all_linear' else None
        for name, edits in tt.items():
            if (model_name, name) in done:
                continue
            res = []
            for chunk in batches(rows, args.splice_batch):
                o = light_forward(rt, chunk, edits=edits)
                for r, lp in zip(chunk, o['logp']):
                    u = r['unit']
                    res.append({'model': model_name, 'test': name, 'unit': u.id, 'family': u.family, 'truth': u.truth,
                                'logp0': lp[0].tolist(), 'logp1': lp[1].tolist()})
            append_jsonl(path, res)
            mt = t2.digit_metrics(np.array([r['truth'] for r in res]), np.array([r['logp0'] for r in res]),
                                  np.array([r['logp1'] for r in res]))
            print(f"clock {model_name} {name}: gain {mt['gain']:.2f} p1 {mt['p_second']:.2f} p0 {mt['p_first']:.2f} "
                  f"err {mt['value_err']:.1f}", flush=True)
        if lora is not None:
            lora.detach()


# ----------------------------------------------------------------------------- constant-shift (steering) test

STEER_LAYER = 4


def stage_steer(rt: qr.Runtime, out: Path, args) -> None:
    """Is the early (layers 1-4) change at image tokens a constant shift?

    v = mean over image tokens and over images of (h_hybrid - h_base) after layer STEER_LAYER, where the hybrid uses
    the all-linear adapter in layers 1-4 only. Estimated separately on 40 disc images ('disc') and 40 RefCOCO test
    images ('refcoco'), on the target tokens only ('disc_target'), plus a random vector of the same norm. Each vector
    is added to every image token of the BASE model after layer STEER_LAYER (scales 0.5 / 1 / 2), and the digit
    readout is measured teacher-forced on the disc sweeps and RefCOCO points."""
    lora = load_lora(rt, out, 'all_linear')
    early = {c.name: c for c in t2.splice_configs(lora.names)}['band1_4']
    disc = build_rows(rt, disc_units(args, 2, ('sweep_h', 'sweep_v')))
    rc = build_rows(rt, disc_units(args, 2, ('refcoco',)))
    rng = np.random.default_rng(SEED)

    def mean_shift(rows, target_only=False):
        acc, n = None, 0
        for chunk in batches(rows, args.splice_batch):
            store = {}
            for tag, active in (('base', set()), ('hyb', set(early.active))):
                lora.active = active
                X, P = c2.pad_batch([r['x'] for r in chunk], [r['p'] for r in chunk])
                lm = rt.model.model.language_model
                h = lm.layers[STEER_LAYER - 1].register_forward_hook(
                    lambda m, a, o, tag=tag: store.__setitem__(tag, (o[0] if isinstance(o, tuple) else o).float()))
                with torch.inference_mode():
                    rt.model(inputs_embeds=X, position_ids=P, logits_to_keep=1)
                h.remove()
            d = store['hyb'] - store['base']
            for i, r in enumerate(chunk):
                idx = r['target'] if target_only else list(range(*r['span']))
                v = d[i, idx].sum(0)
                acc = v if acc is None else acc + v
                n += len(idx)
        lora.active = None
        return (acc / n).cpu().numpy()

    pick = lambda rows, k: [rows[int(i)] for i in rng.choice(len(rows), min(k, len(rows)), replace=False)]
    vecs = {'disc': mean_shift(pick(disc, 40)), 'disc_target': mean_shift(pick(disc, 40), True),
            'refcoco': mean_shift(pick(rc, 40))}
    r = rng.normal(size=vecs['disc'].shape)
    vecs['random'] = r / np.linalg.norm(r) * np.linalg.norm(vecs['disc'])
    lora.detach()
    np.savez(out / 'splice' / 'steer_vectors.npz', **vecs)
    path = out / 'splice' / 'steer.jsonl'
    done = {(r_['vector'], r_['scale']) for r_ in read_jsonl(path)} if path.exists() else set()
    dev = rt.device
    tests = [('none', 0.0)] + [(v, a) for v in ('disc', 'refcoco', 'disc_target') for a in (0.5, 1.0, 2.0)] + [('random', 1.0)]
    for name, a in tests:
        if (name, a) in done:
            continue
        vec = torch.tensor(vecs[name] if name != 'none' else np.zeros_like(vecs['disc']), device=dev, dtype=torch.float32)

        def fn(h, chunk, vec=vec, a=a):
            h = h.clone()
            for i, r_ in enumerate(chunk):
                s_, e_ = r_['span']
                h[i, s_:e_] = (h[i, s_:e_].float() + a * vec).to(h.dtype)
            return h
        res = []
        for rows in (disc, rc):
            for chunk in batches(rows, args.splice_batch):
                o = light_forward(rt, chunk, edits=[(STEER_LAYER, fn)] if name != 'none' else None)
                for r_, lp in zip(chunk, o['logp']):
                    u = r_['unit']
                    res.append({'vector': name, 'scale': a, 'unit': u.id, 'family': u.family, 'truth': u.truth,
                                'logp0': lp[0].tolist(), 'logp1': lp[1].tolist()})
        append_jsonl(path, res)
        for fams in (('sweep_h', 'sweep_v'), ('refcoco',)):
            rr = [x for x in res if x['family'] in fams]
            mt = t2.digit_metrics(np.array([x['truth'] for x in rr]), np.array([x['logp0'] for x in rr]),
                                  np.array([x['logp1'] for x in rr]))
            print(f"steer {name} x{a} {fams[0]}: gain {mt['gain']:.2f} p1 {mt['p_second']:.2f} p0 {mt['p_first']:.2f} "
                  f"err {mt['value_err']:.1f}", flush=True)


# ----------------------------------------------------------------------------- scaled training data

N_SCALED = {'refcoco_extra': 1600, 'gqa': 1600, 'textvqa_q': 500, 'gqa_q': 500, 'describe': 300, 'exist': 400}


def _resized(src: Path, root: Path, rel: str) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Resize a natural image to the 512 x 512 pixel budget with sides that are multiples of 32 (no upscaling
    beyond the processor minimum); returns (path, source size, new size)."""
    from analysis_tools.reread import cached_reread as crr
    im = Image.open(src).convert('RGB')
    new = tfr.min_area(crr.resize_budget(im, 512 * 512))
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        new.save(p)
    return str(p), im.size, new.size


def stage_scaledprep(out: Path, args) -> None:
    import pyarrow.parquet as pq
    from analysis_tools.reread import cached_reread_run as crun
    rng = np.random.default_rng(SEED)
    base_train = tf.read_samples(out / 'train.jsonl')
    used = {s.id.rsplit('_', 1)[0] for s in base_train if s.family == 'refcoco'}
    # More RefCOCO: unused cases of the 3.9 image-level training split (test images never enter training).
    r1 = Path(args.readout_tool_run)
    excl = {str(c['question_id']) for c in json.loads(Path(args.refcoco_cases).read_text())}
    pool = [c for c in read_jsonl(r1 / 'cases.jsonl') if c['set'] == 'refcoco' and c['split'] == 'train'
            and c['id'] not in used and c['id'].split('_')[1] not in excl]
    extra = []
    for j, i in enumerate(rng.choice(len(pool), N_SCALED['refcoco_extra'], replace=False)):
        c = pool[int(i)]
        fmt = 'point' if j % 2 == 0 else 'bbox'
        t = {k: c['truth'][k] for k in ('canvas', 'box', 'center')}
        extra.append(tf.Sample(f"{c['id']}_{fmt}", 'refcoco', 'train', str(r1 / 'images' / c['image']), fmt, t,
                               tf.question(c['target'], fmt), c['target'], {'digest': c['digest'], 'scaled': True}))
    # GQA scene-graph expressions on GQA training images (images of the GQA evaluation subset are excluded).
    gqa_root = Path(args.data_dir) / 'GQA'
    eval_imgs = {r['image'].split('.')[0] for r in read_jsonl(Path(args.vcot_gqa))}
    graphs = json.loads((gqa_root / 'scene' / 'train_sceneGraphs.json').read_text())
    gids = sorted(g for g in graphs if g not in eval_imgs)
    order = rng.permutation(len(gids))
    gqa, kinds = [], {'name': 0, 'attribute': 0}
    quota = {'name': N_SCALED['gqa'] // 2, 'attribute': N_SCALED['gqa'] - N_SCALED['gqa'] // 2}
    used_imgs = set()
    for i in order:
        if len(gqa) >= N_SCALED['gqa']:
            break
        g = gids[int(i)]
        ex = t2.gqa_expressions(graphs[g])
        ex = [e for e in ex if kinds[e['kind']] < quota[e['kind']]]
        if not ex:
            continue
        e = ex[int(rng.integers(len(ex)))]
        path, src, new = _resized(gqa_root / 'images' / 'images' / f'{g}.jpg', out, f'images/gqa/{g}.png')
        box = t2.scale_box(e['box'], src, new)
        truth = {'canvas': list(new), 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]}
        fmt = 'point' if len(gqa) % 2 == 0 else 'bbox'
        gqa.append(tf.Sample(f"gqa_{g}_{e['object_id']}_{fmt}", 'gqa', 'train', path, fmt, truth,
                             tf.question(e['expression'], fmt), e['expression'], {'kind': e['kind']}))
        kinds[e['kind']] += 1
        used_imgs.add(g)
    # Instruction prompts (answers are produced by the base model in the 'distill' stage).
    prompts = []
    tv = pq.read_table(Path(args.data_dir) / 'TextVQA/data/train-00000-of-00020.parquet').to_pylist()
    tv += pq.read_table(Path(args.data_dir) / 'TextVQA/data/train-00001-of-00020.parquet').to_pylist()
    for j, i in enumerate(rng.choice(len(tv), N_SCALED['textvqa_q'], replace=False)):
        r = tv[int(i)]
        src = out / 'images' / 'textvqa_src' / f"{r['image_id']}.jpg"
        src.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            src.write_bytes(r['image']['bytes'])
        path, _, _ = _resized(src, out, f"images/textvqa/{r['image_id']}.png")
        prompts.append({'id': f"ins_textvqa_{r['question_id']}", 'kind': 'textvqa', 'image': path,
                        'prompt': r['question'] + crun.SHORT})
    qs = json.loads((gqa_root / 'questions' / 'train_balanced_questions.json').read_text())
    qkeys = [k for k, v in qs.items() if v['imageId'] not in eval_imgs]
    for i in rng.choice(len(qkeys), N_SCALED['gqa_q'], replace=False):
        v = qs[qkeys[int(i)]]
        path, _, _ = _resized(gqa_root / 'images' / 'images' / f"{v['imageId']}.jpg", out, f"images/gqa/{v['imageId']}.png")
        prompts.append({'id': f'ins_gqa_{qkeys[int(i)]}', 'kind': 'gqa_q', 'image': path, 'prompt': v['question'] + crun.SHORT})
    del qs
    vocab = sorted({o['name'] for g in gids[:5000] for o in graphs[g]['objects'].values()} - t2.GQA_SKIP_NAMES)
    free = [g for g in gids if g not in used_imgs]
    pick = rng.choice(len(free), N_SCALED['describe'] + N_SCALED['exist'], replace=False)
    for j, i in enumerate(pick):
        g = free[int(i)]
        path, _, _ = _resized(gqa_root / 'images' / 'images' / f'{g}.jpg', out, f'images/gqa/{g}.png')
        if j < N_SCALED['describe']:
            prompts.append({'id': f'ins_describe_{g}', 'kind': 'describe', 'image': path,
                            'prompt': 'Describe this image in one or two sentences.'})
        else:
            q, _ = t2.existence_question([o['name'] for o in graphs[g]['objects'].values()], vocab, rng)
            prompts.append({'id': f'ins_exist_{g}', 'kind': 'exist', 'image': path, 'prompt': q + ' ' + crun.POPE_FMT})
    del graphs
    tf.dump_samples(out / 'scaled_grounding.jsonl', base_train + extra + gqa)
    (out / 'instruct_prompts.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in prompts))
    from collections import Counter
    info = {'grounding': dict(Counter(s.family for s in base_train + extra + gqa)), 'gqa_kinds': kinds,
            'instruct': dict(Counter(p['kind'] for p in prompts)), 'seed': SEED}
    (out / 'scaledprep.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def stage_distill(rt: qr.Runtime, out: Path, args) -> None:
    """Self-distillation: the base model's own greedy answers become the targets of the instruction samples, so the
    instruction part of the mixture keeps the base behaviour (answer format included) instead of importing a new one.
    Truncated answers (finish_reason 'length') are excluded and counted."""
    if args.condition != 'base':
        raise ValueError('distill uses the base model')
    prompts = [json.loads(l) for l in (out / 'instruct_prompts.jsonl').read_text().splitlines()]
    path = out / 'instruct_answers.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    todo = [p for p in prompts if p['id'] not in done]
    t0 = time.time()
    for i in range(0, len(todo), args.general_batch):
        chunk = todo[i:i + args.general_batch]
        outs = qr.generate_batch(rt, [([Image.open(p['image']).convert('RGB')], p['prompt']) for p in chunk],
                                 args.general_max_new)
        append_jsonl(path, [{**p, 'text': o['text'], 'finish_reason': o['finish_reason']} for p, o in zip(chunk, outs)])
        if i % (8 * args.general_batch) == 0:
            print(f'distill {i + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s)', flush=True)
    answers = read_jsonl(path)
    ok = [a for a in answers if a['finish_reason'] == 'eos' and a['text'].strip()]
    ins = []
    for a in ok:
        W, H = Image.open(a['image']).size
        ins.append(tf.Sample(a['id'], f"ins_{a['kind']}", 'train', a['image'], 'text',
                             {'canvas': [W, H], 'box': [0, 0, W, H], 'center': [W / 2, H / 2]}, a['prompt'], '',
                             {'answer_text': a['text']}))
    samples = tf.read_samples(out / 'scaled_grounding.jsonl') + ins
    tf.dump_samples(out / 'train_scaled.jsonl', samples)
    info = json.loads((out / 'scaledprep.json').read_text())
    info['distill'] = {'answers': len(answers), 'kept': len(ok), 'truncated': sum(a['finish_reason'] != 'eos' for a in answers),
                       'n_train_scaled': len(samples)}
    (out / 'scaledprep.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info['distill']), flush=True)


# ----------------------------------------------------------------------------- main

GPU_STAGES = {'splice': stage_splice, 'splicegen': stage_splicegen, 'capture': stage_capture, 'clock': stage_clock,
              'steer': stage_steer,
              'jacobian': stage_jacobian, 'distill': stage_distill}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--round1-run', default='')
    p.add_argument('--stages', required=True)
    p.add_argument('--condition', default='base', choices=['base', *tf.CONDITIONS])
    p.add_argument('--readout-tool-run', default='')
    p.add_argument('--refcoco-cases', default='')
    p.add_argument('--round1', default='')
    p.add_argument('--features', default='')
    p.add_argument('--data-dir', default='')
    p.add_argument('--vcot-gqa', default='')
    p.add_argument('--mmvp-csv', default='')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=32.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--step-samples', type=int, default=16)
    p.add_argument('--micro-tokens', type=int, default=4096)
    p.add_argument('--token-budget', type=int, default=16000)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--general-n', type=int, default=200)
    p.add_argument('--general-seed', type=int, default=20261020)
    p.add_argument('--general-batch', type=int, default=16)
    p.add_argument('--general-max-new', type=int, default=96)
    p.add_argument('--splice-batch', type=int, default=24)
    p.add_argument('--splice-config', default='', help='capture with only this splice configuration of the adapter')
    p.add_argument('--jacobian-batch', type=int, default=4)
    p.add_argument('--max-memory', default='')
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(',')
    cpu = {'link': stage_link, 'scaledprep': stage_scaledprep}
    if all(s in cpu for s in stages):
        for s in stages:
            cpu[s](out, args)
        return
    if any(s in cpu for s in stages):
        raise ValueError('CPU stages (link, scaledprep) must run on their own')
    rt = tfr.load_model(args)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages, 'condition': args.condition,
                         'runtime': qr.describe(rt), 'args': {k: v for k, v in vars(args).items()},
                         'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]})
    meta.write_text(json.dumps(info, indent=1))
    lora = None
    for s in stages:
        t0 = time.time()
        if s in GPU_STAGES:
            if lora is not None:          # these stages attach their own adapter
                lora.detach()
                lora = None
            GPU_STAGES[s](rt, out, args)
        else:
            if s != 'train' and lora is None:
                lora = tfr.attach(rt, args)
            tfr.STAGES[s](rt, out, args)
        print(f'stage {s} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
