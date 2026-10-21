"""Experiment 3.6 stage drivers: coordinate-value representation and conversion in Qwen3.5-9B.

Stages (comma list via --stages):
  tokenizer  audit how integers 0-1000 are tokenized in several contexts (no model needed).
  gen        greedy point answers for the dense disc sweeps and the experiment-1.3 T6 point cases.
  spotcheck  answer-level agreement of the batched manual decoder with the reference generate().
  capture    teacher-forced pass recording residual streams, block outputs and head outputs at the
             answer query positions, plus image-token residuals of a subset of grid images.
  force      continue x (or y) after forcing the true leading one or two digits.
  interp     interpolate the post-image residual stream between two discs at one layer and read x.
  steer      add a position offset to one decoder layer at the states predicting x's digits, along a
             linear encoding direction or along the empirical sweep manifold, and read the emitted x.
Design: docs/analysis/position/design_coordinate_value.md.
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
from analysis_tools.position import coordinate_value as cv
from analysis_tools.position.coordinate_readout_run import append_jsonl, done_ids, read_jsonl, score_t6, t6_cases, t6_input

VIS_LAYERS = (4, 8, 12, 15, 16, 20, 24, 28)
VIS_EVERY = 8                       # every 8th grid2d image stores its image-token residuals
SHARD = 32
STEER_LAYERS = (8, 12, 16, 20, 24, 28)
STEER_DELTAS = (-200, -120, -80, -40, -20, 0, 20, 40, 80, 120, 200)
STEER_BASES = 25
MANIFOLD_SIGMA = 10.0               # Gaussian kernel width (0-1000 units) for the sweep manifold


def code_digest() -> dict:
    files = ['coordinate_value.py', 'coordinate_value_run.py', 'coordinate_readout.py']
    root = Path(__file__).parent
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] for f in files}


def all_cases(t6_source: Path) -> list[dict]:
    t6 = [c for c in t6_cases(t6_source) if c['format'] == 'point']
    return cv.disc_cases() + t6


def case_input(case: dict):
    return t6_input(case) if case['set'] == 't6' else cv.disc_input(case)


def score(case: dict, truth: dict, text: str) -> dict:
    return score_t6(case, text) if case['set'] == 't6' else cr.score_disc(truth, 'point', text)


def group_key(case: dict):
    return case['set'] == 't6' and case['grid']


# ----------------------------------------------------------------------------- tokenizer

def stage_tokenizer(rt, out: Path, args) -> None:
    audit = cv.audit_tokenizer(rt.tokenizer)
    (out / 'tokenizer_audit.json').write_text(json.dumps(audit, indent=1))
    for k, v in audit.items():
        print(f"tokenizer {k}: {v['n_not_one_digit_per_token']}/{v['n_values']} violations; 357 -> {v['tokens_357']}")


# ----------------------------------------------------------------------------- gen

def stage_gen(rt, out: Path, args) -> None:
    path = out / 'gen' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = all_cases(Path(args.t6_source))
    (out / 'cases.json').write_text(json.dumps(cases, indent=1))
    finished = done_ids(path)
    groups: dict = {}
    for c in cases:
        if c['id'] not in finished:
            groups.setdefault(group_key(c), []).append(c)
    for key, group in groups.items():
        for i in range(0, len(group), args.batch_size):
            chunk = group[i:i + args.batch_size]
            preps, truths = [], []
            for c in chunk:
                image, truth, text = case_input(c)
                preps.append(cr.prepare(rt, image, text))
                truths.append(truth)
            if len({len(p.ids) for p in preps}) != 1:
                raise ValueError(f'Unequal prompt lengths in group {key}')
            t0 = time.time()
            res = cr.run_conditions(rt, [cr.embeds(rt, p) for p in preps], [p.pos for p in preps],
                                    args.max_new_tokens, batch_size=args.batch_size)
            rows = []
            for c, p, truth, r in zip(chunk, preps, truths, res):
                spans = cv.digit_spans(rt, r['ids'])
                rows.append({'id': c['id'], 'set': c['set'], 'text': r['text'], 'ids': r['ids'],
                             'finish_reason': r['finish_reason'], 'truth': truth, 'truth_units': cv.truth_units(truth),
                             'score': score(c, truth, r['text']), 'spans': spans, 'prompt_len': len(p.ids),
                             'batch_size': len(chunk)})
            append_jsonl(path, rows)
            print(f'gen {key} {i + len(chunk)}/{len(group)} {time.time() - t0:.1f}s', flush=True)


def stage_spotcheck(rt, out: Path, args) -> None:
    path = out / 'gen' / 'spotcheck.jsonl'
    finished = done_ids(path)
    records = {r['id']: r for r in read_jsonl(out / 'gen' / 'records.jsonl')}
    cases = [c for c in json.loads((out / 'cases.json').read_text()) if c['id'] in records]
    rng = np.random.default_rng(args.seed)
    for j in sorted(rng.choice(len(cases), size=min(args.n_spot, len(cases)), replace=False)):
        c = cases[j]
        if c['id'] in finished:
            continue
        image, truth, text = case_input(c)
        ref = qr.generate(rt, qr.build(rt, [image], text), args.max_new_tokens)
        rec = records[c['id']]
        append_jsonl(path, [{'id': c['id'], 'reference': ref['text'], 'manual': rec['text'],
                             'text_equal': ref['text'].strip() == rec['text'].strip(),
                             'answer_equal': cr.parse_coords(ref['text'], 'point') == cr.parse_coords(rec['text'], 'point')}])


# ----------------------------------------------------------------------------- capture

def stage_capture(rt, out: Path, args) -> None:
    feat = Path(args.features)
    feat.mkdir(parents=True, exist_ok=True)
    records = [r for r in read_jsonl(out / 'gen' / 'records.jsonl') if r['spans']]
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    grid_ids = [r['id'] for r in records if r['set'] == 'grid2d']
    vis_ids = set(grid_ids[::VIS_EVERY])
    index = []
    for s in range(0, len(records), SHARD):
        shard = feat / f'shard_{s // SHARD:03d}.npz'
        chunk = records[s:s + SHARD]
        index.append({'file': shard.name, 'ids': [r['id'] for r in chunk]})
        if shard.exists():
            continue
        t0 = time.time()
        caps, vis_rows, roles_all = [], {}, []
        for r in chunk:
            image, _, text = case_input(cases[r['id']])
            prep = cr.prepare(rt, image, text)
            roles = cv.query_positions(len(prep.ids), r['spans'])
            cap, vis = cv.capture(rt, prep, r['ids'], roles, VIS_LAYERS if r['id'] in vis_ids else ())
            caps.append(cap)
            roles_all.append(roles)
            if vis:
                vis_rows[r['id']] = vis
        np.savez(shard, ids=np.array([r['id'] for r in chunk]),
                 resid=np.stack([c.resid for c in caps]), attn=np.stack([c.attn for c in caps]),
                 mlp=np.stack([c.mlp for c in caps]), heads=np.stack([c.heads for c in caps]),
                 logp=np.stack([c.logp for c in caps]), next_logp=np.stack([c.next_logp for c in caps]),
                 roles=json.dumps(roles_all))
        for rid, vis in vis_rows.items():
            np.savez(feat / f'visual_{rid}.npz', **{f'L{l}': v for l, v in vis.items()})
        print(f'capture shard {s // SHARD} ({len(chunk)}) {time.time() - t0:.1f}s', flush=True)
    (feat / 'index.json').write_text(json.dumps({'roles': cv.ROLES, 'component_roles': cv.COMPONENT_ROLES,
                                                 'full_layers': cv.FULL_LAYERS, 'visual_layers': VIS_LAYERS,
                                                 'shards': index}, indent=1))


def load_resid(feat: Path, roles: tuple[str, ...], layers: tuple[int, ...]) -> tuple[list[str], np.ndarray]:
    """(N, len(layers), len(roles), D) float32 residuals of all captured samples."""
    index = json.loads((feat / 'index.json').read_text())
    ri = [cv.ROLES.index(r) for r in roles]
    ids, arrs = [], []
    for sh in index['shards']:
        z = np.load(feat / sh['file'])
        ids += z['ids'].tolist()
        arrs.append(z['resid'][:, list(layers)][:, :, ri].astype(np.float32))
    return ids, np.concatenate(arrs)


# ----------------------------------------------------------------------------- force

def true_value(v: float) -> int:
    return int(round(v))


def stage_force(rt, out: Path, args) -> None:
    path = out / 'force' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    records = [r for r in read_jsonl(out / 'gen' / 'records.jsonl') if r['spans'] and r['id'] not in finished]
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    digit_ids = cv.digit_token_ids(rt)
    t0 = time.time()
    for n, r in enumerate(records):
        image, _, text = case_input(cases[r['id']])
        prep = cr.prepare(rt, image, text)
        rows = []
        conds = []
        for a, axis in enumerate('xy'):
            prefix = r['ids'][:r['spans'][a][0]]
            truth = true_value(r['truth_units'][a])
            for k in (0, 1, 2):
                if k >= len(str(truth)):
                    continue                     # forcing every digit leaves nothing to predict
                forced = cv.leading_digits(truth, k)
                conds.append((axis, k, prefix + [digit_ids[d] for d in forced], forced, truth))
        groups: dict = {}
        for c in conds:
            groups.setdefault(len(c[2]), []).append(c)
        out_row = {'id': r['id'], 'set': r['set'], 'truth_units': r['truth_units'], 'conditions': []}
        for _, group in groups.items():
            xs, ps = zip(*[cv.prefix_inputs(rt, prep, c[2]) for c in group])
            res = cv.continue_number(rt, torch.cat(xs), torch.cat(ps, 1))
            for c, rr in zip(group, res):
                axis, k, _, forced, truth = c
                digits = ''.join(map(str, forced)) + rr['digits']
                out_row['conditions'].append({'axis': axis, 'n_forced': k, 'forced': forced, 'true_value': truth,
                                              'value': int(digits) if digits else None, 'stop': rr['stop']})
        rows.append(out_row)
        append_jsonl(path, rows)
        if n % 50 == 0:
            print(f'force {n}/{len(records)} {time.time() - t0:.1f}s', flush=True)


# ----------------------------------------------------------------------------- steer

def steering_vectors(out: Path, feat: Path) -> dict:
    """Linear encoding directions (grid2d) and the kernel-smoothed sweep manifold (sweep_h) per layer and role."""
    records = {r['id']: r for r in read_jsonl(out / 'gen' / 'records.jsonl')}
    roles = ('x0', 'x1', 'x2')
    ids, H = load_resid(feat, roles, STEER_LAYERS)            # (N, Ls, 3, D)
    tx = np.array([records[i]['truth_units'][0] for i in ids])
    sets = np.array([records[i]['set'] for i in ids])
    beta = np.zeros((len(STEER_LAYERS), 3, H.shape[-1]), np.float32)
    g = sets == 'grid2d'
    for li in range(len(STEER_LAYERS)):
        for k in range(3):
            ok = g & ~np.isnan(H[:, li, k, 0])
            X, t = H[ok, li, k], tx[ok]
            beta[li, k] = ((X - X.mean(0)) * (t - t.mean())[:, None]).sum(0) / ((t - t.mean()) ** 2).sum()
    sw = sets == 'sweep_h'
    return {'beta': beta, 'sweep_ids': [i for i, s in zip(ids, sw) if s], 'sweep_x': tx[sw], 'sweep_H': H[sw]}


def manifold_mean(sweep_x: np.ndarray, sweep_H: np.ndarray, at: float, exclude: int) -> np.ndarray:
    """Gaussian-kernel mean of sweep states (Ls, 3, D) at true x = at, leaving one sample out; NaN rows ignored."""
    w = np.exp(-0.5 * ((sweep_x - at) / MANIFOLD_SIGMA) ** 2)
    w[exclude] = 0
    out = np.zeros(sweep_H.shape[1:], np.float32)
    for k in range(sweep_H.shape[2]):
        ok = ~np.isnan(sweep_H[:, 0, k, 0])
        ww = w * ok
        out[:, k] = np.einsum('n,nld->ld', ww, np.nan_to_num(sweep_H[:, :, k])) / ww.sum()
    return out


def stage_steer(rt, out: Path, args) -> None:
    path = out / 'steer' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    sv = steering_vectors(out, Path(args.features))
    records = {r['id']: r for r in read_jsonl(out / 'gen' / 'records.jsonl')}
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    picks = np.linspace(0, len(sv['sweep_ids']) - 1, STEER_BASES).round().astype(int)
    np.savez(out / 'steer' / 'directions.npz', beta=sv['beta'], layers=np.array(STEER_LAYERS))
    t0 = time.time()
    for n, j in enumerate(picks):
        rid = sv['sweep_ids'][j]
        if rid in finished or not records[rid]['spans']:
            continue
        r = records[rid]
        x0 = sv['sweep_x'][j]
        image, _, text = case_input(cases[rid])
        prep = cr.prepare(rt, image, text)
        prefix = r['ids'][:r['spans'][0][0]]
        x, pos = cv.prefix_inputs(rt, prep, prefix)
        here = manifold_mean(sv['sweep_x'], sv['sweep_H'], x0, j)
        conds = []
        for li, layer in enumerate(STEER_LAYERS):
            for d in STEER_DELTAS:
                if not 40 <= x0 + d <= 960:
                    continue
                conds.append((layer, 'linear', d, torch.from_numpy(d * sv['beta'][li])))
                there = manifold_mean(sv['sweep_x'], sv['sweep_H'], x0 + d, j)
                conds.append((layer, 'manifold', d, torch.from_numpy(there[li] - here[li])))
        results = []
        for layer in STEER_LAYERS:
            group = [c for c in conds if c[0] == layer]
            steer = cv.Steerer(rt, layer)
            try:
                for i in range(0, len(group), args.batch_size):
                    g = group[i:i + args.batch_size]
                    deltas = torch.stack([c[3] for c in g]).to(rt.device)
                    res = cv.continue_number(rt, x.expand(len(g), -1, -1).contiguous(), pos.expand(-1, len(g), -1).contiguous(),
                                             steer, deltas)
                    for c, rr in zip(g, res):
                        results.append({'layer': c[0], 'method': c[1], 'delta': c[2],
                                        'value': int(rr['digits']) if rr['digits'] else None, 'stop': rr['stop'],
                                        'norm': float(c[3].norm())})
            finally:
                steer.close()
        append_jsonl(path, [{'id': rid, 'truth_x': float(x0), 'base_text': r['text'], 'results': results}])
        print(f'steer {n + 1}/{len(picks)} {time.time() - t0:.1f}s', flush=True)


INTERP_LAYERS = (12, 16, 20, 24, 28)
INTERP_ALPHAS = tuple(np.round(np.linspace(0, 1, 11), 2))
INTERP_OFFSETS = (64, 125)          # donor truth minus base truth (0-1000 units)
INTERP_BASES = 20


def stage_interp(rt, out: Path, args) -> None:
    """Interpolate the post-image residual stream between a base and a donor disc at one layer.

    Base and donor are sweep_h images whose true x differ by a fixed offset. The donor is run on the
    base's own token sequence (prompt + answer prefix up to x), so every post-image position has a
    counterpart. At layer l the positions from the end of the image span to the last prefix token are
    set to (1 - a) * base + a * donor; x is then decoded greedily without further patching.
    """
    path = out / 'interp' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    records = {r['id']: r for r in read_jsonl(out / 'gen' / 'records.jsonl')}
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    sweep = sorted([r for r in records.values() if r['set'] == 'sweep_h' and r['spans']], key=lambda r: r['truth_units'][0])
    tx = np.array([r['truth_units'][0] for r in sweep])
    t0 = time.time()
    for off in INTERP_OFFSETS:
        bases = [i for i in np.linspace(0, len(sweep) - 1, INTERP_BASES).round().astype(int) if tx[i] + off <= tx[-1]]
        for n, i in enumerate(bases):
            base = sweep[i]
            j = int(np.argmin(np.abs(tx - (tx[i] + off))))
            donor = sweep[j]
            key = f"{base['id']}__{donor['id']}"
            if key in finished:
                continue
            prefix = base['ids'][:base['spans'][0][0]]
            preps = [cr.prepare(rt, *cv.disc_input(cases[r['id']])[::2]) for r in (base, donor)]
            start = preps[0].span['end']
            xs = [cv.prefix_inputs(rt, p, prefix) for p in preps]
            hb = cv.layer_states(rt, *xs[0], INTERP_LAYERS, start)
            hd = cv.layer_states(rt, *xs[1], INTERP_LAYERS, start)
            results = []
            for l in INTERP_LAYERS:
                vals = torch.stack([(1 - a) * hb[l].float() + a * hd[l].float() for a in INTERP_ALPHAS])
                for k in range(0, len(INTERP_ALPHAS), args.batch_size):
                    v = vals[k:k + args.batch_size]
                    patch = cv.SpanPatcher(rt, l, start, v)
                    try:
                        B = len(v)
                        res = cv.continue_number(rt, xs[0][0].expand(B, -1, -1).contiguous(),
                                                 xs[0][1].expand(-1, B, -1).contiguous())
                    finally:
                        patch.close()
                    for a, rr in zip(INTERP_ALPHAS[k:k + args.batch_size], res):
                        results.append({'layer': l, 'alpha': float(a), 'value': int(rr['digits']) if rr['digits'] else None,
                                        'stop': rr['stop'], 'first_logp': rr['first_logp']})
            append_jsonl(path, [{'id': key, 'offset': off, 'base_truth': tx[i], 'donor_truth': tx[j],
                                 'base_emitted': base['score']['coords'][0], 'donor_emitted': donor['score']['coords'][0],
                                 'n_patched_positions': int(hb[INTERP_LAYERS[0]].shape[0]), 'results': results}])
            print(f'interp off={off} {n + 1}/{len(bases)} {time.time() - t0:.1f}s', flush=True)


STAGES = {'tokenizer': stage_tokenizer, 'gen': stage_gen, 'spotcheck': stage_spotcheck, 'capture': stage_capture,
          'force': stage_force, 'steer': stage_steer,
          'interp': stage_interp}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--features', required=True, help='Directory for large feature shards.')
    p.add_argument('--t6-source', required=True, help='Experiment-1.3 decomposition run directory.')
    p.add_argument('--stages', default='tokenizer,gen,spotcheck')
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--n-spot', type=int, default=20)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model, device='cuda:0')
    meta_path = args.output / 'experiment.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {'runs': []}
    meta['model'] = qr.describe(rt)
    meta['runs'].append({'started_utc': datetime.now(timezone.utc).isoformat(), 'stages': args.stages,
                         'args': {k: str(v) for k, v in vars(args).items() if k not in ('output', 't6_source', 'features')},
                         'code_sha256': code_digest(), 'thinking': 'off', 'decoder': 'manual greedy, unpadded batches'})
    meta_path.write_text(json.dumps(meta, indent=2))
    for stage in args.stages.split(','):
        STAGES[stage](rt, args.output, args)


if __name__ == '__main__':
    main()
