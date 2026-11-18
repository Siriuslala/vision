"""CPU analysis of experiment 3.11 round 5 (edge_heads_r3.py) -> summary.json.

Sections:
  avg     slopes of the moved value (pixels per pixel of edge movement) under decoding-time head transplants;
          first-digit logit recovery r of the groups; head-write comparison between point and box answers
          (W_O^{(l,h)} (o_B - o_A) in the residual stream); MLP 24-30 neuron attribution (Qwen3.5).
  vlscan  Qwen3-VL: causal heads (single-head patching) and the edge band each of them reads.
  mix     per image kind: r of head groups, region splices, pattern transplants, single-head scan, attention regions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.position import edge_heads_r2 as e2
from analysis_tools.position import edge_heads_r2_analysis as a2
from analysis_tools.position import edge_heads_r3 as e3

read_jsonl = e3.read_jsonl
EDGE_INDEX = {'left': 0, 'top': 1, 'right': 2, 'bottom': 3}
TAGS = ('q35', 'vl')
MODEL_NAME = {'q35': 'Qwen3.5-9B', 'vl': 'Qwen3-VL-8B'}


def boot_mean(x, rng, n: int = 1000, stat=np.mean) -> list[float]:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return [float('nan')] * 3 + [0]
    bs = stat(rng.choice(x, (n, len(x))), axis=1)
    return [round(float(stat(x)), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(len(x))]


def r_of(ld: dict, i: int, j0: int = 0, min_gap: float = 2.0) -> float | None:
    """Recovery of row i relative to row j0 (the unedited run A) for one value record of e2._ld."""
    if ld['dA'] is None or ld['dA'] == ld['dB']:
        return None
    base = ld['ld'][j0]
    den = ld['LD_B'] - base
    if abs(den) < min_gap:
        return None
    return (ld['ld'][i] - base) / den


def r_diff(ld: dict, i_b: int, i_a: int, min_gap: float = 2.0) -> float | None:
    """Recovery of the difference between two edited rows (pattern transplants: version B minus version A)."""
    if ld['dA'] is None or ld['dA'] == ld['dB']:
        return None
    den = ld['LD_B'] - ld['ld'][0]
    if abs(den) < min_gap:
        return None
    return (ld['ld'][i_b] - ld['ld'][i_a]) / den


# ----------------------------------------------------------------------------- avg

def avg_records(run: Path, tag: str) -> list[dict]:
    p = run / f'avg_{tag}.jsonl'
    return read_jsonl(p) if p.exists() else []


def moved_slopes(rec: dict) -> dict | None:
    """Slope (value change / edge change, pixels) of the moved value per decoding condition."""
    side, f = rec['side'], rec['format']
    key = e3.SIDE_KEY[(f, side)]
    ei = EDGE_INDEX[side]
    d_edge = rec['box_b'][ei] - rec['box_a'][ei]
    vals = {g['cond']: g['values'] for g in rec['gen']}
    if vals.get('A') is None or abs(d_edge) < 1:
        return None
    conv = a2.convention(vals['A'], rec['box_a'], rec['canvas'], f)
    base = a2.to_pixels(vals['A'], rec['canvas'], conv)[key]
    out = {}
    for c, v in vals.items():
        if c == 'A':
            continue
        out[c] = None if (v is None or key not in v) else (a2.to_pixels(v, rec['canvas'], conv)[key] - base) / d_edge
    out['_conv'] = conv
    return out


def avg_slopes(recs: list[dict], rng) -> dict:
    by: dict[tuple, dict[str, list]] = {}
    for r in recs:
        s = moved_slopes(r)
        if s is None:
            continue
        for setname in (r['set'], 'all'):
            for side in (r['side'], 'any'):
                d = by.setdefault((setname, r['format'], side), {})
                for c, v in s.items():
                    if not c.startswith('_'):
                        d.setdefault(c, []).append(v)
    # slopes: median (a few box answers switch coordinate convention between rows), bootstrap CI of the median
    return {f'{k[0]}:{k[1]}:{k[2]}': {c: boot_mean(v, rng, stat=np.median) for c, v in d.items()} for k, d in by.items()}


def avg_recovery(recs: list[dict], rng) -> dict:
    by: dict[tuple, dict[str, list]] = {}
    for r in recs:
        if 'ld' not in r:
            continue
        ld = r['ld']
        for setname in (r['set'], 'all'):
            d = by.setdefault((setname, r['format']), {})
            for i, c in enumerate(r['conds']):
                if i:
                    d.setdefault(c, []).append(r_of(ld, i))
            if 'topk_ld' in r:
                for i, c in enumerate(r['topk_conds']):
                    v = r_diff(r['topk_ld'] | {'ld': [ld['ld'][0]] + r['topk_ld']['ld']}, i + 1, 0)
                    d.setdefault(c, []).append(v)
    return {f'{k[0]}:{k[1]}': {c: boot_mean(v, rng) for c, v in d.items()} for k, d in by.items()}


def head_writes(run: Path, tag: str, recs: list[dict], o_proj: dict, FL: tuple, H: int) -> dict:
    """For each (case, side) with both formats: residual-stream write change of each head group,
    dW = sum_{(l,h) in group} W_O^{(l,h)} (o_B - o_A), at the query of the moved value's first digit.
    Reports the gain of the point write along the box write, rho = <dW_point, dW_box> / ||dW_box||^2, the cosine,
    and the norm of each, per group; plus per-head norms of the write change."""
    pairs: dict[tuple, dict] = {}
    for r in recs:
        if 'ld' not in r:
            continue
        pairs.setdefault((r['case'], r['side']), {})[r['format']] = r
    out = {'groups': {}, 'heads': {}}
    per_head = {f: [] for f in ('point', 'bbox')}
    for (case, side), d in pairs.items():
        if set(d) != {'point', 'bbox'}:
            continue
        arrs = {f: np.load(run / f'avg_{tag}' / f"{d[f]['id'].replace(':', '__')}.npz") for f in d}
        dW = {}
        for f, z in arrs.items():
            do = z['heads_b'].astype(np.float32) - z['heads_a'].astype(np.float32)      # (nL, H, hd)
            dW[f] = {(l, h): o_proj[l][:, h * do.shape[-1]:(h + 1) * do.shape[-1]] @ do[i, h]
                     for i, l in enumerate(FL) if l in o_proj for h in range(H)}
            per_head[f].append(np.array([[np.linalg.norm(dW[f][(l, h)]) for h in range(H)] for l in FL if l in o_proj]))
        groups = {g: [tuple(int(v) for v in s[1:].split('H')) for s in hs] for g, hs in d['point']['groups'].items()}
        groups['all16_32'] = [(l, h) for l in FL if l in o_proj for h in range(H)]
        for g, hs in groups.items():
            hs = [h for h in hs if h in dW['point']]
            if not hs:
                continue
            wp = sum(dW['point'][h] for h in hs)
            wb = sum(dW['bbox'][h] for h in hs)
            nb = float(wb @ wb)
            rec = out['groups'].setdefault(f'{side}:{g}', {'rho': [], 'cos': [], 'norm_point': [], 'norm_bbox': []})
            rec['rho'].append(float(wp @ wb) / nb if nb > 0 else None)
            rec['cos'].append(float(wp @ wb) / (np.linalg.norm(wp) * np.sqrt(nb) + 1e-9))
            rec['norm_point'].append(float(np.linalg.norm(wp)))
            rec['norm_bbox'].append(float(np.sqrt(nb)))
    rng = np.random.default_rng(0)
    out['groups'] = {k: {m: boot_mean(v, rng) for m, v in d.items()} for k, d in out['groups'].items()}
    out['heads'] = {f: np.mean(v, 0).round(3).tolist() for f, v in per_head.items() if v}
    return out


def neuron_analysis(run: Path, tag: str, recs: list[dict]) -> dict:
    """MLP 24-30 attribution patching: concentration (share of the summed attribution reached by the top-k
    neurons ranked by |attr|), agreement between point and box answers of the same pair (cosine of the attribution
    vectors), agreement across pairs, and the neurons that recur among the top-32 of many pairs."""
    A = {}
    for r in recs:
        p = run / f'avg_{tag}' / f"{r['id'].replace(':', '__')}.npz"
        if 'attr' not in np.load(p).files if p.exists() else True:
            continue
        A[r['id']] = (r, np.load(p)['attr'].astype(np.float32))
    if not A:
        return {}
    ks = (8, 32, 128, 512, 2048, 8192, 32768)
    conc = {f: [] for f in ('point', 'bbox')}
    vecs = {f: [] for f in ('point', 'bbox')}
    same_pair = []
    top_count: dict[tuple[str, int, int], int] = {}
    by_pair: dict[tuple, dict] = {}
    for rid, (r, a) in A.items():
        flat = a.ravel()
        tot = flat.sum()
        if abs(tot) < 1e-6:
            continue
        order = np.argsort(-np.abs(flat))
        conc[r['format']].append([float(flat[order[:k]].sum() / tot) for k in ks])
        vecs[r['format']].append(flat / (np.linalg.norm(flat) + 1e-9))
        by_pair.setdefault((r['case'], r['side']), {})[r['format']] = flat
        for i in order[:32]:
            key = (r['format'], int(i // a.shape[1]), int(i % a.shape[1]))
            top_count[key] = top_count.get(key, 0) + 1
    for d in by_pair.values():
        if set(d) == {'point', 'bbox'}:
            same_pair.append(float(d['point'] @ d['bbox'] / (np.linalg.norm(d['point']) * np.linalg.norm(d['bbox']) + 1e-9)))
    cross = {}
    for f, v in vecs.items():
        if len(v) > 1:
            M = np.stack(v)
            C = M @ M.T
            cross[f] = float(C[np.triu_indices(len(v), 1)].mean())
    rec = sorted(top_count.items(), key=lambda kv: -kv[1])[:40]
    n = {f: len(v) for f, v in conc.items()}
    return {'k': list(ks), 'concentration': {f: np.mean(v, 0).round(3).tolist() for f, v in conc.items() if v},
            'n': n, 'cos_point_vs_bbox_same_pair': [round(float(np.mean(same_pair)), 3), len(same_pair)] if same_pair else None,
            'cos_across_pairs': cross,
            'recurring_top32': [{'format': f, 'layer': e3.MLP_LAYERS[li], 'neuron': ni, 'pairs': c} for (f, li, ni), c in rec]}


# ----------------------------------------------------------------------------- mix

def mix_tables(run: Path, tag: str, rng) -> dict:
    p = run / f'mix_{tag}.jsonl'
    if not p.exists():
        return {}
    recs = [r for r in read_jsonl(p) if 'lds' in r]
    out = {}
    kinds = sorted({r['kind'] for r in recs})
    for kind in kinds + ['_all']:
        rs = [r for r in recs if r['kind'] == kind or kind == '_all']
        tab: dict[str, dict[str, list]] = {}
        for r in rs:
            names = r['conds']
            for ld in r['lds']:
                ax = 'x' if ld['key'].startswith('x') else 'y'
                d = tab.setdefault(ax, {})
                for i, c in enumerate(names):
                    if i == 0:
                        continue
                    if c.endswith('_B') and c[:-2] + '_A' in names:
                        d.setdefault(c[:-2], []).append(r_diff(ld, i, names.index(c[:-2] + '_A')))
                    elif c.endswith('_A'):
                        continue
                    else:
                        d.setdefault(c, []).append(r_of(ld, i))
        res = {}
        for ax, d in tab.items():
            res[ax] = {c: boot_mean(v, rng) for c, v in d.items() if not (c.startswith('L') and 'H' in c)}
            singles = {c: np.nanmean([x for x in v if x is not None]) if any(x is not None for x in v) else np.nan
                       for c, v in d.items() if c.startswith('L') and 'H' in c}
            if singles:
                order = sorted(singles, key=lambda c: -np.nan_to_num(singles[c], nan=-9))
                pos = np.array([max(0.0, np.nan_to_num(singles[c])) for c in order])
                res[ax]['top_singles'] = [[c, round(float(singles[c]), 4)] for c in order[:12]]
                res[ax]['n_heads_r_ge_0.05'] = int((pos >= 0.05).sum())
                res[ax]['top1_share_of_positive'] = round(float(pos[0] / max(pos.sum(), 1e-9)), 4)
                res[ax]['H11_share_of_positive'] = round(float(max(0, np.nan_to_num(singles.get('L20H11', np.nan))) / max(pos.sum(), 1e-9)), 4)
        res['n'] = len(rs)
        res['hit'] = mix_hits(rs)
        out[kind] = res
    out['attention'] = mix_attention(run, tag, recs)
    return out


def mix_hits(rs: list[dict]) -> list:
    hits = []
    for r in rs:
        v = {d['key']: d['value'] for d in r['values']}
        h, _ = a2.point_hit(v, r['box_a'], r['canvas'])
        hits.append(h)
    k = int(sum(hits))
    return [k, len(hits), a2.wilson(k, len(hits))]


MIX_ATT_HEADS = ((20, 11), (20, 0), (24, 2), (24, 7), (24, 12), (20, 10), (24, 5), (24, 9), (28, 0), (16, 1), (16, 2))


def mix_attention(run: Path, tag: str, recs: list[dict]) -> dict:
    """Mean attention mass on each token region of run A (regions of the run-A target only) per kind, head and
    axis, plus the regions' token shares."""
    if tag != 'q35':
        return {}
    layers = [l for l in e3.cv.cr.FULL_LAYERS if l >= 12]
    names = e3.REGIONS[:-1]
    acc: dict = {}
    for r in recs:
        z = np.load(run / f'mix_{tag}' / f"{r['id']}.npz")
        att = z['att_a'].astype(np.float32)                                     # (nL, H, 2, Nv)
        reg = e3.token_regions(r['box_a'], r['box_a'], tuple(r['grid']))
        Nv = att.shape[-1]
        lab = np.full(Nv, -1)
        for i, n in enumerate(names):
            lab[reg[n]] = i
        tot = att.sum(-1, keepdims=True)
        frac = np.stack([(att * (lab == i)).sum(-1) for i in range(len(names))], -1) / np.maximum(tot, 1e-9)
        share = np.array([(lab == i).mean() for i in range(len(names))])
        for (l, h) in MIX_ATT_HEADS:
            li = layers.index(l)
            for qi, ax in enumerate(('x', 'y')):
                acc.setdefault((r['kind'], hname(l, h), ax), []).append(frac[li, h, qi])
                acc.setdefault((r['kind'], '_token_share', ax), []).append(share)
                acc.setdefault((r['kind'], hname(l, h) + ':image', ax), []).append(np.array([tot[li, h, qi, 0]]))
    return {'regions': list(names), 'table': {f'{k[0]}|{k[1]}|{k[2]}': np.mean(v, 0).round(4).tolist() for k, v in acc.items()}}


def hname(l, h):
    return f'L{l}H{h}'


def mix2_tables(run: Path, tag: str, rng) -> dict:
    """Per-head pattern transplants (stage mix2): recovery of the B-minus-A difference of each transplant row."""
    p = run / f'mix2_{tag}.jsonl'
    if not p.exists():
        return {}
    recs = [r for r in read_jsonl(p) if 'lds' in r]
    out = {}
    for kind in sorted({r['kind'] for r in recs}) + ['_all']:
        tab: dict = {}
        for r in recs:
            if kind != '_all' and r['kind'] != kind:
                continue
            names = r['conds']
            for ld in r['lds']:
                ax = 'x' if ld['key'].startswith('x') else 'y'
                for i, c in enumerate(names):
                    if c.endswith('_B'):
                        tab.setdefault(ax, {}).setdefault(c[:-2], []).append(r_diff(ld, i, names.index(c[:-2] + '_A')))
        out[kind] = {ax: {c: boot_mean(v, rng) for c, v in d.items()} for ax, d in tab.items()}
        out[kind]['n'] = sum(1 for r in recs if kind == '_all' or r['kind'] == kind)
    return out


# ----------------------------------------------------------------------------- vlscan

def vlscan(run: Path, tag: str, rng, min_r: float = 0.02) -> dict:
    p = run / f'scan_{tag}.jsonl'
    if not p.exists():
        return {}
    gens = {g['id']: g for g in read_jsonl(run / f'nat_gen_{tag}.jsonl')}
    recs = read_jsonl(p)
    step = lambda f, k: a2.step_of(f, k)
    rs: dict[str, dict[str, list]] = {}
    for r in recs:
        for ld in r['lds']:
            st = step(r['format'], ld['key'])
            d = rs.setdefault(st, {})
            for i, c in enumerate(r['conds'][1:], 1):
                d.setdefault(c, []).append(r_of(ld, i))
    causal = {}
    for st, d in rs.items():
        tab = {c: boot_mean(v, rng) for c, v in d.items()}
        causal[st] = {'all_heads': tab.get('all_heads'),
                      'heads': {c: v for c, v in sorted(tab.items(), key=lambda kv: -np.nan_to_num(kv[1][0]))
                                if c.startswith('L') and v[0] >= min_r and v[1] > 0}}
    # band classification from the attention region masses (round-4 labels: e2.REGIONS)
    bands = {}
    regions = list(e2.REGIONS)
    acc: dict = {}
    for gid, g in gens.items():
        f = run / f'scan_{tag}_regions' / f"{gid.replace(':', '__')}.npz"
        if not f.exists() or g['values'] is None:
            continue
        if not e2.box_span_ok(g['box'], tuple(g['grid'])):
            continue
        rm = np.load(f)['regions'].astype(np.float32)                         # (nL, H, nq, nreg)
        for qi, v in enumerate(g['values']):
            st = step(g['format'], v['key'])
            acc.setdefault(st, []).append(rm[:, :, qi])
    for st, lst in acc.items():
        M = np.mean(lst, 0)                                                    # (nL, H, nreg)
        bands[st] = M
    out = {'causal': causal, 'regions': regions}
    # per causal head: in-box shares of left / mid / right (x steps) or top / mid / bottom (y steps)
    lab = {k: i for i, k in enumerate(regions)}
    def side_share(m, ax):
        inbox = sum(m[lab[k]] for k in ('interior', 'left', 'right', 'top', 'bottom', 'top_left', 'top_right', 'bottom_left', 'bottom_right'))
        if ax == 'x':
            lo = m[lab['left']] + m[lab['top_left']] + m[lab['bottom_left']]
            hi = m[lab['right']] + m[lab['top_right']] + m[lab['bottom_right']]
        else:
            lo = m[lab['top']] + m[lab['top_left']] + m[lab['top_right']]
            hi = m[lab['bottom']] + m[lab['bottom_left']] + m[lab['bottom_right']]
        return [round(float(inbox), 3), round(float(lo / max(inbox, 1e-9)), 3), round(float(hi / max(inbox, 1e-9)), 3)]
    FL = list(range(1, 37))
    table = {}
    for st, d in causal.items():
        if st not in bands:
            continue
        ax = 'x' if st in ('pt_x', 'bb_x1', 'bb_x2') else 'y'
        table[st] = {c: {'r': v, 'inbox_lo_hi': side_share(bands[st][int(c[1:].split('H')[0]) - 1, int(c.split('H')[1])], ax)}
                     for c, v in d['heads'].items()}
    out['heads'] = table
    return out


def vl_groups(run: Path, min_inbox: float = 0.3, min_side: float = 0.5) -> dict[str, str]:
    """Qwen3-VL reader groups at point answers from the vlscan stage: causal heads (single-head r >= 0.02 with a
    positive CI lower bound) whose in-box attention share is >= min_inbox and whose share on the low (left / top)
    or high (right / bottom) edge band of the box is >= min_side."""
    v = vlscan(run, 'vl', np.random.default_rng(0))
    out = {}
    for st, ax in (('pt_x', 'x'), ('pt_y', 'y')):
        lo, hi = [], []
        for c, d in v.get('heads', {}).get(st, {}).items():
            inbox, pl, ph = d['inbox_lo_hi']
            if inbox < min_inbox:
                continue
            spec = c[1:].replace('H', '.')
            if pl >= min_side:
                lo.append(spec)
            elif ph >= min_side:
                hi.append(spec)
        out[f'min{ax}'], out[f'max{ax}'] = ','.join(lo), ','.join(hi)
    return out


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--round4', type=Path, required=True)
    p.add_argument('--vl-groups', action='store_true', help='print shell exports of the Qwen3-VL reader groups and exit')
    args = p.parse_args()
    if args.vl_groups:
        for k, v in vl_groups(args.run).items():
            print(f'export VL_{k.upper()}="{v}"')
        return
    rng = np.random.default_rng(0)
    out = {}
    o_proj = a2.o_proj_weights((16, 20, 24, 28, 32))
    for tag in TAGS:
        recs = avg_records(args.run, tag)
        res = {}
        if recs:
            res['slopes'] = avg_slopes(recs, rng)
            res['recovery'] = avg_recovery(recs, rng)
            if tag == 'q35':
                res['writes'] = head_writes(args.run, tag, recs, o_proj, e3.cv.cr.FULL_LAYERS, 16)
                res['neurons'] = neuron_analysis(args.run, tag, recs)
            sp = args.run / f'spot_{tag}.jsonl'
            if sp.exists():
                s = read_jsonl(sp)
                res['spotcheck'] = [sum(x['answer_equal'] for x in s), len(s)]
        res['mix'] = mix_tables(args.run, tag, rng)
        res['mix2'] = mix2_tables(args.run, tag, rng)
        if tag == 'vl':
            res['vlscan'] = vlscan(args.run, tag, rng)
        out[tag] = res
    (args.run / 'summary.json').write_text(json.dumps(out, indent=1, default=float))
    print('wrote', args.run / 'summary.json')


if __name__ == '__main__':
    main()
