"""Experiment 3.13 round 2 CPU analysis (results: docs/analysis/position/results_2026-11-03_targeted_finetune_r2.md).

Sections of results_r2.json:
  training / coord / general / mechanism: the round-1 measurements (targeted_finetune_analysis) on every condition;
  splice      teacher-forced digit metrics of every splice configuration (adapter on a layer / type / position subset);
  splicegen   greedy coordinate answers of the depth and type splices;
  geometry    periodic spectra of the residual stream versus the coordinate (every layer, sites q0 / q1 / target tokens),
              the period-100 'clock' fit and 3-D PCA coordinates for the manifold figures;
  alignment   principal angles between the coordinate-code subspace and the digit readout's input subspace (from the
              Jacobians) and the readout gain along the code; weight-based angles to the input weights of the
              digit-writing neurons of MLP 28-32.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_analysis as tfa
from analysis_tools.position import targeted_finetune_r2 as t2
from analysis_tools.position.coordinate_readout_run import read_jsonl

PERIODS = (10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0)
SITES = ('q0', 'q1', 'target')
MANIFOLD_LAYERS = (12, 16, 20, 24, 28, 32)
BOOT = 1000


# ----------------------------------------------------------------------------- splice

def _metrics(rows: list[dict]) -> dict:
    return t2.digit_metrics(np.array([r['truth'] for r in rows]), np.array([r['logp0'] for r in rows]),
                            np.array([r['logp1'] for r in rows]))


def _boot(rows: list[dict], key: str, rng) -> list[float]:
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(len(rows), size=len(rows))
        vals.append(_metrics([rows[i] for i in pick])[key])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def splice(run: Path, cond: str) -> dict:
    path = run / 'splice' / f'{cond}.jsonl'
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    by: dict = {}
    for r in rows:
        by.setdefault(r['config'], []).append(r)
    rng = np.random.default_rng(0)
    out = {}
    for name, rs in by.items():
        o = {'group': rs[0]['group'], 'positions': rs[0]['positions'], 'n_active': rs[0]['n_active']}
        for fam in ('sweep_h', 'sweep_v'):
            o[fam] = _metrics([r for r in rs if r['family'] == fam])
        o['pooled'] = _metrics(rs)
        o['pooled']['gain_ci'] = _boot(rs, 'gain', rng)
        for fam in ('sweep_h', 'sweep_v'):
            rf = [r for r in rs if r['family'] == fam]
            o[fam]['margin'] = t2.margin_scatter(np.array([r['truth'] for r in rf]), np.array([r['logp0'] for r in rf]))
        out[name] = o
    for name, o in out.items():
        o['recovery'] = {k: t2.recovery(o['pooled'][k], out['base']['pooled'][k], out['full']['pooled'][k])
                         for k in ('gain', 'p_second', 'p_first', 'value_err')}
    return out


def splicegen(run: Path, cond: str) -> dict:
    path = run / 'splice' / f'{cond}_gen.jsonl'
    if not path.exists():
        return {}
    samples = {s.id: s for s in tf.read_samples(run / 'test.jsonl')}
    by: dict = {}
    for r in read_jsonl(path):
        by.setdefault(r['config'], {})[r['id']] = r
    out = {}
    for name, rows in by.items():
        ss = [samples[i] for i in rows]
        out[name] = {g: tfa.group_metrics(ss, rows, g) for g in ('disc_x', 'disc_y', 'grid06', 'grid10', 'grid20')}
        for g in out[name].values():
            g.pop('curve', None)
    return out


def grouped(path: Path, keys: tuple[str, str], families: tuple[str, ...] = ('sweep_h', 'sweep_v')) -> dict:
    """Teacher-forced digit metrics of the clock / steer tests, grouped by two keys ('a/b' names)."""
    if not path.exists():
        return {}
    by: dict = {}
    for r in read_jsonl(path):
        if r['family'] in families:
            by.setdefault(f'{r[keys[0]]}/{r[keys[1]]}', []).append(r)
    return {k: _metrics(v) for k, v in by.items()}


# ----------------------------------------------------------------------------- geometry

def load_capture(run: Path, cond: str) -> dict | None:
    p = run / 'capture' / f'{cond}.npz'
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def clock_fit(H: np.ndarray, x: np.ndarray, period: float = 100.0, n_pc: int = 16) -> dict:
    """Project H onto the plane of its period-`period` component and measure how clock-like it is.

    The top n_pc principal components Y of H are regressed on [Legendre(x), cos, sin]; the two coefficient rows of
    cos / sin span the clock plane. Returns the projection (N, 2) onto an orthonormal basis of that plane, the
    unwrapped angle slope (degrees per `period` units; +-360 for one turn per period), the linearity of angle versus
    x (Pearson r) and the radius coefficient of variation (0 = perfect circle)."""
    Y = t2.pca_scores(H, min(n_pc, len(H) - 1))
    S = t2.smooth_basis(x, 3)
    X1 = np.concatenate([S, t2.periodic_basis(x, period), np.ones((len(x), 1))], 1)
    coef, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    Q, _ = np.linalg.qr(coef[3:5].T)
    resid = Y - np.concatenate([S, np.ones((len(x), 1))], 1) @ np.linalg.lstsq(np.concatenate([S, np.ones((len(x), 1))], 1), Y, rcond=None)[0]
    P = resid @ Q
    order = np.argsort(x)
    ang = np.unwrap(np.arctan2(P[order, 1], P[order, 0]))
    rad = np.linalg.norm(P, axis=1)
    return {'proj': P, 'slope_deg_per_period': float(np.polyfit(x[order], np.degrees(ang), 1)[0] * period),
            'angle_r': float(abs(np.corrcoef(x[order], ang)[0, 1])), 'radius_cv': float(rad.std() / rad.mean())}


def geometry(run: Path, conds: list[str]) -> tuple[dict, dict]:
    R, A = {}, {}
    for cond in conds:
        C = load_capture(run, cond)
        if C is None:
            continue
        o = {}
        layers = list(C['layers'])
        for fam in ('sweep_h', 'sweep_v', 'textconv', 'refcoco'):
            m = C['family'] == fam
            if not m.any():
                continue
            x = C['truth'][m].astype(float)
            H = C['resid'][m].astype(np.float32)               # (N, 3, 33, D)
            f = {}
            for si, site in enumerate(SITES):
                spec = [t2.periodic_spectrum(H[:, si, li], x, PERIODS) for li in range(len(layers))]
                f[site] = {'smooth': [s['smooth'] for s in spec],
                           'extra': {str(int(T)): [s['extra'][T] for s in spec] for T in PERIODS},
                           'chance': spec[0]['chance']}
                if fam in ('sweep_h', 'refcoco'):          # held-out decoding (slow): x sweep and natural images only
                    dec = [t2.periodic_decodability(H[:, si, li], x) for li in range(len(layers))]
                    f[site]['decode100'] = {k: [d[k] for d in dec] for k in ('periodic', 'x')}
                if fam != 'sweep_h':
                    continue
                cf = [clock_fit(H[:, si, li], x) for li in range(len(layers))]
                f[site]['clock'] = {k: [c[k] for c in cf] for k in ('slope_deg_per_period', 'angle_r', 'radius_cv')}
                for l in MANIFOLD_LAYERS:
                    li = layers.index(l)
                    P, _, var = t2.pca(H[:, si, li], 3)
                    A[f'{cond}/{fam}/{site}/L{l}/pca3'] = P
                    A[f'{cond}/{fam}/{site}/L{l}/var3'] = var
                    A[f'{cond}/{fam}/{site}/L{l}/clock'] = cf[li]['proj']
            A[f'{cond}/{fam}/x'] = x
            o[fam] = f
            # Digit outputs at the true first digit (teacher forced) for the same inputs.
            lp = C['logp'][m]
            o[fam]['digits'] = t2.digit_metrics(x, lp[:, 0], lp[:, 1])
            o[fam]['margin'] = t2.margin_scatter(x, lp[:, 0])
        R[cond] = o
    return R, A


# ----------------------------------------------------------------------------- alignment (Jacobian based)

def alignment(run: Path, conds: list[str], k_code: int = 3, k_read: int = 9) -> dict:
    out = {}
    J = {c: np.load(run / 'jacobian' / f'{c}.npz') for c in conds if (run / 'jacobian' / f'{c}.npz').exists()}
    caps = {c: load_capture(run, c) for c in J}
    for cond, z in J.items():
        Jc = z['J'].astype(np.float32)                         # (N, layers, pairs, 10, D)
        pairs = [tuple(p) for p in z['pairs']]
        o = {}
        for li, l in enumerate(z['layers']):
            for pi, (outr, inr) in enumerate(pairs):
                R, share = t2.readout_subspace(Jc[:, li, pi], k_read)
                tag = f"L{int(l)}/{('q0', 'q1')[outr]}<-{('q0', 'q1')[inr]}"
                o[tag] = {'readout_share': share.tolist()}
                for code_cond, C in caps.items():
                    if C is None:
                        continue
                    m = C['family'] == 'sweep_h'
                    x = C['truth'][m].astype(float)
                    li_c = list(C['layers']).index(int(l))
                    H = C['resid'][m][:, inr, li_c].astype(np.float32)
                    V, deriv = t2.code_subspace(H, x, k_code)
                    ang = t2.principal_angles(V, R)
                    # Readout gain along the code derivative, with the readout of `cond` and the code of `code_cond`.
                    Jmean = Jc[:, li, pi].mean(0, keepdims=True)
                    g = t2.readout_gain(Jmean, deriv)[0]
                    o[tag][f'code_{code_cond}'] = {'angles': ang.tolist(), 'gain_spread': float(g.std()),
                                                   'deriv_norm': float(np.linalg.norm(deriv)),
                                                   'deriv_in_readout': float(np.linalg.norm(t2.orthonormal(R) @ deriv)
                                                                             / max(np.linalg.norm(deriv), 1e-12))}
        out[cond] = o
    return out


# ----------------------------------------------------------------------------- alignment (weights of MLP 28-32)

def weight_alignment(run: Path, hf_cache: str, model: str, conds: list[str], top: int = 64, k_read: int = 16) -> dict:
    """Input subspace of the digit-writing neurons of MLP 28-32 versus the coordinate code entering that layer.

    For layer l, neuron i writes W_down[:, i]; its digit-writing score is the norm of the centred digit-unembedding
    projection W_U,c W_down[:, i] (W_U,c: (10, D) digit rows of the LM head minus their mean). The `top` neurons by
    score are the digit writers; their gate and up rows, scaled by the pre-MLP RMSNorm gain (1 + w), are the input
    directions through which the residual reaches them. The span of the top-k_read right singular vectors of these
    2 * top rows is compared with the code subspace of the residual after layer l - 1 (q0 and q1). For adapted
    conditions the LoRA update of the same matrices is added."""
    from safetensors import safe_open
    from eva.inference import resolve_model
    ckpt = Path(resolve_model(model, hf_cache))
    index = json.loads((ckpt / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = next(k.split('layers.')[0] for k in index if '.layers.' in k and 'mlp.down_proj' in k and 'visual' not in k)

    def get(name):
        with safe_open(ckpt / index[name], 'pt') as f:
            return f.get_tensor(name).float()
    head_name = next(k for k in index if k.endswith('lm_head.weight'))
    W_head = get(head_name)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    dig = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    WU = W_head[dig]
    WU = WU - WU.mean(0, keepdim=True)
    del W_head
    out = {}
    for cond in conds:
        C = load_capture(run, cond)
        if C is None:
            continue
        lora = None
        if cond != 'base':
            st = torch.load(run / 'adapters' / f'{cond}.pt', map_location='cpu', weights_only=False)
            lora = st
        m = C['family'] == 'sweep_h'
        x = C['truth'][m].astype(float)
        o = {}
        for l in range(28, 33):
            W = {}
            for nm in ('gate_proj', 'up_proj', 'down_proj'):
                w = get(f'{prefix}layers.{l - 1}.mlp.{nm}.weight')
                key = f'L{l}.mlp.{nm}'
                if lora is not None and key in lora.get('A', {}):
                    w = w + (lora['alpha'] / lora['rank']) * (lora['B'][key].float() @ lora['A'][key].float())
                W[nm] = w
            ln = 1.0 + get(f'{prefix}layers.{l - 1}.post_attention_layernorm.weight')
            score = torch.linalg.norm(WU @ W['down_proj'], dim=0)              # (F,)
            idx = torch.argsort(score, descending=True)[:top]
            rows = torch.cat([W['gate_proj'][idx] * ln, W['up_proj'][idx] * ln], 0).numpy()
            _, S, Vt = np.linalg.svd(rows, full_matrices=False)
            Rw = Vt[:k_read]
            li = list(C['layers']).index(l - 1)
            for si, site in enumerate(('q0', 'q1')):
                V, deriv = t2.code_subspace(C['resid'][m][:, si, li].astype(np.float32), x, 3)
                o[f'L{l}/{site}'] = {'angles': t2.principal_angles(V, Rw).tolist(),
                                     'deriv_in_readout': float(np.linalg.norm(Rw @ deriv) / max(np.linalg.norm(deriv), 1e-12)),
                                     'top_score': float(score[idx[0]]), 'sv_share': (S[:k_read] ** 2 / (S ** 2).sum()).tolist()}
        out[cond] = o
    # Chance level: angles between a random 3-d subspace and a k_read-d subspace in D dimensions.
    D = WU.shape[1]
    rng = np.random.default_rng(0)
    out['chance_angles'] = t2.principal_angles(rng.normal(size=(3, D)), rng.normal(size=(k_read, D))).tolist()
    return out


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    p.add_argument('--conditions', nargs='+', required=True)
    p.add_argument('--figures', required=True)
    p.add_argument('--hf-cache', default='')
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--skip', nargs='*', default=[])
    args = p.parse_args()
    run = Path(args.run)
    conds = [c for c in args.conditions if (run / 'eval' / c / 'coord.jsonl').exists()]
    print('conditions with coordinate results:', conds, flush=True)
    res = {'conditions': conds}
    res['training'] = tfa.training(run, conds)
    res['coord'] = tfa.coord(run, conds)
    res['general'] = tfa.general(run, [c for c in conds if (run / 'eval' / c / 'general.jsonl').exists()])
    arrays = {}
    if 'mechanism' not in args.skip:
        mc = [c for c in conds if (run / 'eval' / c / 'fact' / 'factorial' / 'rows.jsonl').exists()]
        res['mechanism'], marr = tfa.mechanism(run, mc)
        res['change'] = tfa.change(marr, mc)
        arrays.update({k: v for k, v in marr.items() if not k.endswith('_resid')})
    res['splice'] = {c: splice(run, c) for c in args.conditions if (run / 'splice' / f'{c}.jsonl').exists()}
    res['splicegen'] = {c: splicegen(run, c) for c in args.conditions if (run / 'splice' / f'{c}_gen.jsonl').exists()}
    res['clock'], res['steer'] = grouped(run / 'splice' / 'clock.jsonl', ('model', 'test')), \
        grouped(run / 'splice' / 'steer.jsonl', ('vector', 'scale'), ('sweep_h', 'sweep_v'))
    res['steer_refcoco'] = grouped(run / 'splice' / 'steer.jsonl', ('vector', 'scale'), ('refcoco',))
    all_caps = [c for c in args.conditions if (run / 'capture' / f'{c}.npz').exists()]
    all_caps += [p.stem for p in sorted((run / 'capture').glob('*+*.npz'))]
    res['geometry'], garr = geometry(run, all_caps)
    arrays.update(garr)
    res['alignment'] = alignment(run, [c for c in args.conditions if (run / 'jacobian' / f'{c}.npz').exists()])
    if 'weights' not in args.skip:
        res['weight_alignment'] = weight_alignment(run, args.hf_cache, args.model,
                                                   [c for c in ('base', 'all_linear', 'all_but_25_32', 'early1_4_img')
                                                    if c in all_caps])
    (run / 'results_r2.json').write_text(json.dumps(res, indent=1, default=float))
    np.savez(run / 'analysis_r2.npz', **arrays)
    print('analysis written', flush=True)
    from analysis_tools.position import targeted_finetune_r2_figures as F
    F.main_figures(run, Path(args.figures))


if __name__ == '__main__':
    main()
