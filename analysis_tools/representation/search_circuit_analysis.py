"""Experiment 2.7 round 3 CPU analysis: tables, figures and summary.json.

Inputs (under --output): items.jsonl, results_behave.jsonl, trace/<set>.npz, knock.jsonl, patch.jsonl.
Feature classes are fixed a priori from the round-2 count of exclusive ViT detectors (block 9, zeta > 5):
a target is 'detector' if its direction has >= 5 exclusive detectors, otherwise 'comparison'.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.representation import odd_entity as oe

# round-2 exclusive-detector counts (results_2026-11-03_search_asymmetry_r2.md, section 4.2): pair -> (std, swap)
DETECTORS = {'qo': (30, 0), 'closure': (39, 0), 'dot': (90, 2), 'term': (8, 0), 'plusx': (107, 29),
             'plus': (2, 0), 'tilt': (52, 10), 'curve': (64, 22), 'circsq': (32, 39), 'sat': (0, 0), 'size': (0, 1)}
READERS = ((16, 6), (16, 11), (16, 2), (20, 2), (20, 11), (20, 10))
FULL = (4, 8, 12, 16, 20)
DN_LAYERS = [l for l in range(1, 33) if l % 4]
PAIR_LABEL = {'qo': 'Q/O', 'closure': 'open/closed', 'tilt': 'tilt', 'curve': 'curve', 'sat': 'saturation',
              'size': 'size', 'term': 'end bar', 'plus': '4/3-arm', 'dot': 'dot', 'circsq': 'rsquare/circle',
              'plusx': 'rot. cross'}
C_DET, C_CMP, C_GREY = '#2a6fbb', '#d9822b', '#888888'


def feature_class(pair: str, variant: str) -> str:
    n = DETECTORS[pair][0 if variant == 'std' else 1]
    return 'detector' if n >= 5 else 'comparison'


def comp_names(layer: int) -> list[str]:
    out = ['input']
    for m in range(1, layer):
        out += [('fa' if m % 4 == 0 else 'dn') + str(m), f'mlp{m}']
    return out


def wil(k, n):
    return oe.wilson(int(k), int(n)) if n else (float('nan'),) * 3


def load(output: Path):
    items = {it['id']: it for it in read_jsonl(output / 'items.jsonl')}
    beh = {r['id']: r for r in read_jsonl(output / 'results_behave.jsonl')} if (output / 'results_behave.jsonl').exists() else {}
    trace = {}
    for p in sorted((output / 'trace').glob('*.npz')):
        z = np.load(p)
        trace[p.stem] = {k: z[k] for k in z.files}
    knock = read_jsonl(output / 'knock.jsonl') if (output / 'knock.jsonl').exists() else []
    patch = read_jsonl(output / 'patch.jsonl') if (output / 'patch.jsonl').exists() else []
    return items, beh, trace, knock, patch


def tcell(it):
    return it['target'][0] * it['cols'] + it['target'][1]


def index_trace(trace):
    """id -> (set, row index) over all trace files."""
    out = {}
    for s, z in trace.items():
        for i, k in enumerate(z['id']):
            out[str(k)] = (s, i)
    return out


# ----------------------------------------------------------------------------- behaviour

def lenient(it, r) -> tuple[bool, bool]:
    """(row and column correct, column correct) from the last row/column pair in the text, whether or not the
    generation reached EOS (a fenced JSON list sometimes exceeds the 24-token budget)."""
    rc = oe.parse_answer(r['text'])
    if rc is None:
        return False, False
    return (rc[0] - 1 == it['target'][0] and rc[1] - 1 == it['target'][1]), rc[1] - 1 == it['target'][1]


def behaviour(items, beh) -> dict:
    res = {'atlas': {}, 'prev': {}, 'prev_col': {}, 'mag': {}}
    cells = defaultdict(list)
    for i, r in beh.items():
        it = items[i]
        ok, col = lenient(it, r)
        if it['set'] == 'atlas':
            cells[('atlas', it['pair_name'], it['level'], it['variant'])].append(ok)
        elif it['set'] == 'prev':
            cells[('prev', it['pair_name'], it['level'], it['variant'], it['n_prev'])].append(ok)
            cells[('prev_col', it['pair_name'], it['level'], it['variant'], it['n_prev'])].append(col)
        elif it['set'] == 'mag':
            b = 0 if tcell(it) < 6 else 1 if tcell(it) < 12 else 2
            cells[('mag', it['pair_name'], it['param'], it['variant'], b)].append(ok)
    for key, v in sorted(cells.items(), key=lambda kv: str(kv[0])):
        k, n = int(sum(v)), len(v)
        p, lo, hi = wil(k, n)
        res[key[0]]['|'.join(str(x) for x in key[1:])] = {'k': k, 'n': n, 'p': p, 'lo': lo, 'hi': hi}
    return res


# ----------------------------------------------------------------------------- localisation readout

def reader_localisation(items, trace) -> dict:
    """Model-internal localisation, independent of how the coordinate is spelled: the summed attention of the six
    reader heads (row query) on the target cell, its share relative to the corrupted image, and whether the target
    is the cell with the largest summed reader attention (top-1)."""
    idx = index_trace(trace)
    cells = defaultdict(list)
    for s in ('atlas', 'prev', 'mag'):
        if s not in trace:
            continue
        z = trace[s]
        A_ = z['ans_attn'].astype(np.float32)
        for i, k in enumerate(z['id']):
            it = items[str(k)]
            if it['role'] != 'odd':
                continue
            s2, j = idx[it['partner']]
            A0 = trace[s2]['ans_attn'].astype(np.float32)[j]
            m1 = sum(A_[i, FULL.index(l), h, 0] for l, h in READERS)
            m0 = sum(A0[FULL.index(l), h, 0] for l, h in READERS)
            t = tcell(it)
            top1 = bool(np.argmax(m1) == t)
            if s == 'atlas':
                key = (s, it['pair_name'], it['level'], it['variant'], 'early' if t < 6 else 'late')
            elif s == 'prev':
                key = (s, it['pair_name'], it['level'], it['variant'], it['n_prev'])
            else:
                key = (s, it['pair_name'], it['param'], it['variant'], 0 if t < 6 else 1 if t < 12 else 2)
            cells[key].append((top1, m1[t] / 6, np.log(m1[t] + 1e-6) - np.log(m0[t] + 1e-6)))
    out = {}
    for key, v in sorted(cells.items(), key=lambda kv: str(kv[0])):
        k = sum(x[0] for x in v)
        p, lo, hi = wil(k, len(v))
        out['|'.join(str(x) for x in key)] = {'n': len(v), 'top1': p, 'lo': lo, 'hi': hi,
                                              'mass': float(np.mean([x[1] for x in v])),
                                              'dlog': float(np.mean([x[2] for x in v]))}
    return out


# ----------------------------------------------------------------------------- reader prior

def reader_prior(items, trace) -> dict:
    """Mean attention of the six reader heads (row query) on each cell of homogeneous images."""
    out = {}
    sets = {}
    for s in ('atlas', 'mag', 'shift', 'prev'):
        if s not in trace:
            continue
        z = trace[s]
        A = z['ans_attn'].astype(np.float32)          # (n, 5 layers, H, 2 queries, cells)
        rows = []
        for i, k in enumerate(z['id']):
            it = items[str(k)]
            if it['role'] != 'corrupted':
                continue
            m = sum(A[i, FULL.index(l), h, 0] for l, h in READERS) / len(READERS)
            rows.append((it, m))
        sets[s] = rows
    hom = [m for s in ('atlas', 'mag') for it, m in sets.get(s, [])]
    if hom:
        H = np.mean(hom, 0)
        out['hom_map'] = (H / H.sum() * 36).reshape(6, 6).round(3).tolist()
        out['hom_n'] = len(hom)
        out['hom_first_row_share'] = float(H[:6].sum() / H.sum())
    if 'shift' in sets:
        for kb in sorted({it['k_blank'] for it, _ in sets['shift']}):
            ms = [m for it, m in sets['shift'] if it['k_blank'] == kb]
            M = np.mean(ms, 0)
            rel = M / M[kb:].sum() * (36 - kb)
            out[f'shift_k{kb}'] = {'n': len(ms), 'map': rel.reshape(6, 6).round(3).tolist(),
                                   'first_filled_row_rel': float(rel[kb:kb + 6].mean()),
                                   'blank_mass': float(M[:kb].sum() / M.sum()) if kb else 0.0}
    return out


# ----------------------------------------------------------------------------- logit attribution

def attribution(items, trace, sets=('atlas',)) -> dict:
    """Double-difference attribution of the reader advantage for the target, summed over the six reader heads
    (row query): D(c) = [z_t - mean_e z_e]_odd - [z_t - mean_e z_e]_corrupted, separately for components of the
    layer-16 heads and the layer-20 heads."""
    idx = index_trace(trace)
    per = defaultdict(lambda: {16: [], 20: []})
    tot = defaultdict(lambda: {16: [], 20: []})
    for s in sets:
        if s not in trace:
            continue
        z = trace[s]
        for i, k in enumerate(z['id']):
            it = items[str(k)]
            if it['role'] != 'odd' or it['partner'] not in idx:
                continue
            s2, j = idx[it['partner']]
            t = tcell(it)
            filled = [e for e in range(36) if not it['meta']['instances'][e].get('blank')]
            others = [e for e in filled if e != t]
            C1 = trace[s]['logit_contrib'][i].astype(np.float32)      # (6, 2, n_c, cells)
            C0 = trace[s2]['logit_contrib'][j].astype(np.float32)
            d = (C1[:, 0, :, t] - C1[:, 0][:, :, others].mean(-1)) - (C0[:, 0, :, t] - C0[:, 0][:, :, others].mean(-1))
            key = (it['pair_name'], it['level'], it['variant'])
            per[key][16].append(np.nansum(d[:3], 0)[:31])
            per[key][20].append(np.nansum(d[3:], 0))
            L1, L0 = trace[s]['logit'][i], trace[s2]['logit'][j]
            dt = (L1[:, 0, t] - L1[:, 0][:, others].mean(-1)) - (L0[:, 0, t] - L0[:, 0][:, others].mean(-1))
            tot[key][16].append(dt[:3].sum())
            tot[key][20].append(dt[3:].sum())
    out = {}
    for key, v in per.items():
        out['|'.join(key)] = {'class': feature_class(key[0], key[2]), 'n': len(v[16]),
                              'L16': np.mean(v[16], 0).tolist(), 'L20': np.nanmean(v[20], 0).tolist(),
                              'total16': float(np.mean(tot[key][16])), 'total20': float(np.mean(tot[key][20]))}
    return out


def group_attr(vec: np.ndarray, layer: int) -> dict:
    names = comp_names(layer)
    g = defaultdict(float)
    for n, x in zip(names, vec):
        if n == 'input':
            g['input'] += x
        elif n.startswith('dn'):
            g['dn_early' if int(n[2:]) <= 7 else 'dn_mid'] += x
        elif n.startswith('fa'):
            g['fa'] += x
        else:
            g['mlp_early' if int(n[3:]) <= 7 else 'mlp_mid'] += x
    return dict(g)


# ----------------------------------------------------------------------------- DeltaNet prediction error

def dn_errors(items, trace) -> dict:
    """Relative prediction error rho of the target cell (odd minus the same cell in the corrupted image), per DN layer
    and head; start effect in homogeneous images; source shares at the target."""
    idx = index_trace(trace)
    contrast = defaultdict(list)
    for s in ('atlas', 'mag', 'prev'):
        if s not in trace:
            continue
        z = trace[s]
        for i, k in enumerate(z['id']):
            it = items[str(k)]
            if it['role'] != 'odd':
                continue
            s2, j = idx[it['partner']]
            t = tcell(it)
            d = z['dn_rel_err'][i][:, :, t].astype(np.float32) - trace[s2]['dn_rel_err'][j][:, :, t].astype(np.float32)
            contrast[(s, feature_class(it['pair_name'], it['variant']))].append(d)
    out = {'contrast_mean': {}, 'top_heads': {}}
    for key, v in contrast.items():
        M = np.mean(v, 0)                                            # (24, 32)
        out['contrast_mean']['|'.join(key)] = {'n': len(v), 'by_layer': M.mean(1).round(4).tolist()}
        order = np.dstack(np.unravel_index(np.argsort(-M.ravel()), M.shape))[0][:8]
        out['top_heads']['|'.join(key)] = [(DN_LAYERS[a], int(b), round(float(M[a, b]), 4)) for a, b in order]
    # start effect: rho by raster index in homogeneous (corrupted) atlas/mag images without blanks
    hom = []
    for s in ('atlas', 'mag'):
        if s not in trace:
            continue
        z = trace[s]
        for i, k in enumerate(z['id']):
            if items[str(k)]['role'] == 'corrupted':
                hom.append(z['dn_rel_err'][i].astype(np.float32))
    if hom:
        H = np.mean(hom, 0)                                          # (24, 32, 36)
        out['hom_rho_by_index_layer'] = H.mean(1).round(4).tolist()
    return out


# ----------------------------------------------------------------------------- baseline and mark by raster position

def rbin(t: int) -> int:
    return 0 if t < 6 else 1 if t < 12 else 2 if t < 24 else 3


RBIN_LABEL = ['cells 1-6', 'cells 7-12', 'cells 13-24', 'cells 25-36']
GROUPS = ('input', 'dn_early', 'dn_mid', 'fa', 'mlp_early', 'mlp_mid')


def baseline_mark(items, trace, beh) -> dict:
    """Reader logit of the target cell relative to the other cells, summed over the three layer-16 reader heads
    (row query).  baseline = value in the corrupted image (position prior at that cell); mark = odd minus corrupted
    (what the target's pixel difference adds), with its additive split over component groups."""
    idx = index_trace(trace)
    res = defaultdict(list)
    for s in ('atlas', 'mag'):
        if s not in trace:
            continue
        z = trace[s]
        for i, k in enumerate(z['id']):
            it = items[str(k)]
            if it['role'] != 'odd':
                continue
            s2, j = idx[it['partner']]
            t = tcell(it)
            oth = [e for e in range(36) if e != t]
            L1, L0 = z['logit'][i][:3, 0].sum(0), trace[s2]['logit'][j][:3, 0].sum(0)
            base = L0[t] - L0[oth].mean()
            mark = (L1[t] - L1[oth].mean()) - base
            C1 = z['logit_contrib'][i][:3, 0].astype(float).sum(0)[:31]
            C0 = trace[s2]['logit_contrib'][j][:3, 0].astype(float).sum(0)[:31]
            g = group_attr((C1[:, t] - C1[:, oth].mean(1)) - (C0[:, t] - C0[:, oth].mean(1)), 16)
            ok = lenient(it, beh[it['id']])[0] if it['id'] in beh else np.nan
            res[(feature_class(it['pair_name'], it['variant']), rbin(t))].append([base, mark, ok] + [g[x] for x in GROUPS])
    out = {}
    for key, v in sorted(res.items()):
        v = np.asarray(v, float)
        se = v.std(0, ddof=1) / np.sqrt(len(v))
        out[f'{key[0]}|{key[1]}'] = {'n': len(v), 'baseline': v[:, 0].mean(), 'baseline_se': se[0], 'mark': v[:, 1].mean(),
                                     'mark_se': se[1], 'acc': float(np.nanmean(v[:, 2])),
                                     'groups': dict(zip(GROUPS, v[:, 3:].mean(0).round(3).tolist()))}
    return out


def mag_dose(items, trace, beh) -> dict:
    """mag set: accuracy and reader mark by difference size, target variant and raster band."""
    idx = index_trace(trace)
    res = defaultdict(list)
    if 'mag' not in trace:
        return {}
    z = trace['mag']
    for i, k in enumerate(z['id']):
        it = items[str(k)]
        if it['role'] != 'odd':
            continue
        s2, j = idx[it['partner']]
        t = tcell(it)
        oth = [e for e in range(36) if e != t]
        L1, L0 = z['logit'][i][:3, 0].sum(0), trace[s2]['logit'][j][:3, 0].sum(0)
        mark = (L1[t] - L1[oth].mean()) - (L0[t] - L0[oth].mean())
        ok = lenient(it, beh[it['id']])[0]
        res[(it['pair_name'], it['param'], it['variant'], 0 if t < 6 else 1)].append((ok, mark))
    out = {}
    for key, v in sorted(res.items()):
        k_ = sum(x[0] for x in v)
        p, lo, hi = wil(k_, len(v))
        out['|'.join(str(x) for x in key)] = {'n': len(v), 'acc': p, 'lo': lo, 'hi': hi,
                                              'mark': float(np.mean([x[1] for x in v]))}
    return out


def start_sources(items, trace, r2_capture: Path | None) -> dict:
    """Two causal-processing sources of the start effect in homogeneous images: (a) full-attention layer-8 attention
    of each entity's tokens on itself and on earlier entities (round-2 capture, recomputed softmax probabilities);
    (b) Gated DeltaNet relative prediction error by raster index (this experiment's trace)."""
    out = {}
    if r2_capture is not None and r2_capture.exists():
        E, O = [], []
        for f in sorted(r2_capture.glob('*.npz')):
            z = np.load(f, allow_pickle=True)
            for i, s in enumerate(z['ids']):
                if str(s).endswith(('homA', 'homB')):
                    a = z['dec_attn'][i, 1].astype(float)              # layer 8: (H, 36 query cells, 36 key cells)
                    E.append([a[:, e, :e].sum(-1).mean() for e in range(36)])
                    O.append([a[:, e, e].mean() for e in range(36)])
        out['fa8_earlier'] = np.mean(E, 0).round(4).tolist()
        out['fa8_own'] = np.mean(O, 0).round(4).tolist()
        out['fa8_n'] = len(E)
    hom = []
    for s in ('atlas', 'mag'):
        z = trace.get(s)
        if z is None:
            continue
        for i, k in enumerate(z['id']):
            if items[str(k)]['role'] == 'corrupted':
                hom.append(z['dn_rel_err'][i].astype(np.float32))
    H = np.mean(hom, 0)                                                   # (24, 32, 36)
    out['dn_rho_layer_mean'] = H.mean(1).round(4).tolist()
    out['dn_rho_n'] = len(hom)
    return out


def dn_sources(items, trace, heads) -> dict:
    """Composition of the delta-rule prediction at the target tokens: own cell, earlier cells of the same entity
    row, the previous entity row, older rows, text prefix (targets in entity rows 3-6)."""
    acc = defaultdict(list)
    for s in ('atlas', 'mag'):
        z = trace.get(s)
        if z is None:
            continue
        for i, k in enumerate(z['id']):
            it = items[str(k)]
            t = tcell(it)
            r = t // 6
            if r < 2:
                continue
            C = z['dn_contrib'][i].astype(float)
            parts = [C[:, :, t], C[:, :, r * 6:t].sum(-1), C[:, :, (r - 1) * 6:r * 6].sum(-1),
                     C[:, :, :(r - 1) * 6].sum(-1), C[:, :, 36]]
            acc[it['role']].append(np.stack(parts, -1))
    out = {}
    for role, v in acc.items():
        M = np.mean(v, 0)
        out[role] = {'layer_mean': {l: M[li].mean(0).round(3).tolist() for li, l in enumerate(DN_LAYERS)},
                     'heads': {f'{l}:{h}': M[DN_LAYERS.index(l), h].round(3).tolist() for l, h in heads}}
    return out


def dn_top_heads(items, trace, cls: str, k: int = 6) -> list[tuple[int, int]]:
    idx = index_trace(trace)
    acc = []
    z = trace['atlas']
    for i, kk in enumerate(z['id']):
        it = items[str(kk)]
        if it['role'] != 'odd' or feature_class(it['pair_name'], it['variant']) != cls:
            continue
        s2, j = idx[it['partner']]
        t = tcell(it)
        acc.append(z['dn_rel_err'][i][:, :, t].astype(float) - trace[s2]['dn_rel_err'][j][:, :, t])
    M = np.mean(acc, 0)
    order = np.argsort(-M.ravel())[:k]
    return [(DN_LAYERS[a], int(b)) for a, b in (np.unravel_index(o, M.shape) for o in order)]


# ----------------------------------------------------------------------------- knock / patch

def knock_summary(items, knock) -> dict:
    cells = defaultdict(list)
    for r in knock:
        it = items[r['id']]
        if it['role'] != 'odd':
            continue
        t = tcell(it)
        hit = r['arg_row'] == it['target'][0] + 1 and r['arg_col'] == it['target'][1] + 1
        rd = np.asarray(r['reader'])
        if it['set'] in ('atlas', 'mag'):
            band = 'early' if t < 6 else 'late'
            key = (it['set'], feature_class(it['pair_name'], it['variant']), band, r['cond'])
        else:
            key = ('prev', feature_class(it['pair_name'], it['variant']), f'n{it["n_prev"]}', r['cond'])
        cells[key].append((hit, r['s'], rd[:, t].mean(), float(np.argmax(rd.sum(0)) == t)))
    out = {}
    for key, v in sorted(cells.items()):
        h = [x[0] for x in v]
        p, lo, hi = wil(sum(h), len(h))
        out['|'.join(key)] = {'n': len(v), 'hit': p, 'lo': lo, 'hi': hi, 'S': float(np.mean([x[1] for x in v])),
                              'reader_target': float(np.mean([x[2] for x in v])),
                              'reader_target_se': float(np.std([x[2] for x in v], ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                              'top1': float(np.mean([x[3] for x in v]))}
    return out


def patch_summary(items, patch) -> dict:
    base = {}
    for r in patch:
        if r['cond'] == 'none':
            base[r['id']] = r
    cells = defaultdict(list)
    for r in patch:
        it = items[r['id']]
        b = base.get(r['id'])
        if b is None:
            continue
        hit = r['arg_row'] == it['target'][0] + 1 and r['arg_col'] == it['target'][1] + 1
        cells[(feature_class(it['pair_name'], it['variant']), r['cond'])].append(
            (hit, r['s'] - b['s'], float(np.mean(r['reader_target'])) - float(np.mean(b['reader_target']))))
    out = {}
    for key, v in sorted(cells.items()):
        out['|'.join(key)] = {'n': len(v), 'hit': float(np.mean([x[0] for x in v])),
                              'dS': float(np.mean([x[1] for x in v])), 'd_reader': float(np.mean([x[2] for x in v]))}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args(argv)
    items, beh, trace, knock, patch = load(args.output)
    r2cap = args.output.parent / '2026-11-03_search_asymmetry_r2' / 'capture'
    cmp_heads, det_heads = dn_top_heads(items, trace, 'comparison'), dn_top_heads(items, trace, 'detector')
    summary = {'behaviour': behaviour(items, beh) if beh else {}, 'reader_prior': reader_prior(items, trace),
               'baseline_mark': baseline_mark(items, trace, beh), 'mag_dose': mag_dose(items, trace, beh),
               'start_sources': start_sources(items, trace, r2cap),
               'dn_heads': {'comparison': cmp_heads, 'detector': det_heads},
               'dn_sources': dn_sources(items, trace, cmp_heads[:4] + det_heads[:2]),
               'localisation': reader_localisation(items, trace),
               'attribution': attribution(items, trace), 'attribution_prev': attribution(items, trace, ('prev',)),
               'dn': dn_errors(items, trace), 'knock': knock_summary(items, knock),
               'patch': patch_summary(items, patch)}
    summary['circuit'] = circuit_summary(items, args.output)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    from analysis_tools.representation import search_circuit_figures as fig
    args.figures.mkdir(parents=True, exist_ok=True)
    fig.make_all(items, beh, trace, knock, patch, summary, args.figures, args.output)
    print(f'[analyze] wrote {args.output / "summary.json"}', flush=True)



# ----------------------------------------------------------------------------- circuit stage (heads and neurons)

def circuit_summary(items, output: Path) -> dict:
    """Head- and neuron-level results of search_circuit_heads (circuit/rows.jsonl, records.npz, selection.json)."""
    d = output / 'circuit'
    if not (d / 'selection.json').exists():
        return {}
    sel = json.loads((d / 'selection.json').read_text())
    rows = read_jsonl(d / 'rows.jsonl')
    keep = set(sel['items'])
    rows = [r for r in rows if r['id'] in keep]
    out = {'selection': {k: sel[k] for k in ('fa8_heads', 'dn_heads')}, 'n': {}}
    nec = [r for r in rows if r['phase'] == 'necessity']
    base = {r['id']: r['reader'] for r in nec if r['cond'] == 'none'}
    for cls in ('comparison', 'detector'):
        out['n'][cls] = sum(1 for i in base if A_cls(items, i) == cls)
        cell = defaultdict(list)
        for r in nec:
            if r['cls'] == cls and r['cond'] != 'none':
                cell[r['cond']].append(base[r['id']] - r['reader'])
        out[f'drop_{cls}'] = {c: [float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))] for c, v in cell.items()}
        out[f'base_{cls}'] = float(np.mean([v for i, v in base.items() if A_cls(items, i) == cls]))
    sec = defaultdict(list)
    for r in rows:
        if r['phase'] == 'second':
            sec[(r['cls'], r['cond'])].append((r['reader'], r['top1']))
    out['second'] = {f'{c}|{k}': {'n': len(v), 'reader': float(np.mean([x[0] for x in v])),
                                  'reader_se': float(np.std([x[0] for x in v], ddof=1) / np.sqrt(len(v))),
                                  'top1': float(np.mean([x[1] for x in v]))} for (c, k), v in sorted(sec.items())}
    z = np.load(d / 'records.npz')
    cls_arr = z['cls']
    attr = z['attr'].astype(np.float32).sum(1)                               # (n, 8, I): L16 + L20 readers
    ids = z['id']
    # feature-agnosticity of neuron attribution: per condition mean vector, top-50 overlap and correlation
    cond = np.array([f"{items[str(i)]['pair_name']}|{items[str(i)]['level']}|{items[str(i)]['variant']}" for i in ids])
    vecs, labs, clss = [], [], []
    for c in sorted(set(cond)):
        m = cond == c
        if m.sum() < 3:
            continue
        vecs.append(attr[m].mean(0).ravel()); labs.append(c); clss.append(str(cls_arr[m][0]))
    V = np.array(vecs)
    top_sets = [set(np.argsort(-v)[:50].tolist()) for v in V]
    glob_top = set(np.argsort(-attr[cls_arr == 'comparison'].mean(0).ravel())[:50].tolist())
    out['neuron_conditions'] = labs
    out['neuron_condition_class'] = clss
    out['neuron_top50_overlap_with_global'] = [len(s & glob_top) / 50 for s in top_sets]
    Vc = V - V.mean(1, keepdims=True)
    C = (Vc @ Vc.T) / (np.linalg.norm(Vc, axis=1)[:, None] * np.linalg.norm(Vc, axis=1)[None] + 1e-9)
    out['neuron_corr'] = C.round(3).tolist()
    # concentration: share of total positive direct attribution carried by the top-k neurons (comparison mean)
    mc = attr[cls_arr == 'comparison'].mean(0).ravel()
    md = attr[cls_arr == 'detector'].mean(0).ravel()
    tot = mc.sum()
    out['neuron_share'] = {k: float(np.sort(mc)[::-1][:k].sum() / tot) for k in (10, 50, 200, 1000)}
    out['neuron_share_detector_with_cmp_top'] = {k: float(md[np.argsort(-mc)[:k]].sum() / md.sum()) for k in (10, 50, 200)}
    out['neuron_top10'] = [(int(8 + i // attr.shape[2]), int(i % attr.shape[2]), float(mc[i]), float(md[i]))
                           for i in np.argsort(-mc)[:10]]
    out['neuron_layer_share'] = (np.clip(mc.reshape(8, -1), 0, None).sum(1) / np.clip(mc, 0, None).sum()).round(3).tolist()
    # reader QK: q . dk per head, consistency of dk and q directions across items
    q, dk = z['q'].astype(np.float32), z['dk'].astype(np.float32)          # (n, 2, 3, 256)
    rq = {}
    for li, layer in enumerate((16, 20)):
        for hi in range(3):
            h = [hh for l, hh in READERS if l == layer][hi]
            qq, kk = q[:, li, hi], dk[:, li, hi]
            un = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
            res = {}
            for cl in ('comparison', 'detector'):
                m = cls_arr == cl
                res[cl] = {'dlogit': float(((qq[m] * kk[m]).sum(1) / 16).mean()),
                           'cos_q_dk': float((un(qq[m]) * un(kk[m])).sum(1).mean()),
                           'dk_norm': float(np.linalg.norm(kk[m], axis=1).mean())}
            Kn = un(kk); Qn = un(qq)
            mc_ = cls_arr == 'comparison'
            md_ = cls_arr == 'detector'
            res['dk_cos_within_cmp'] = float((Kn[mc_] @ Kn[mc_].T)[np.triu_indices(mc_.sum(), 1)].mean())
            res['dk_cos_within_det'] = float((Kn[md_] @ Kn[md_].T)[np.triu_indices(md_.sum(), 1)].mean())
            res['dk_cos_cmp_det'] = float((Kn[mc_] @ Kn[md_].T).mean())
            res['q_cos_across_items'] = float((Qn @ Qn.T)[np.triu_indices(len(Qn), 1)].mean())
            rq[f'L{layer}H{h}'] = res
    out['reader_qk'] = rq
    # residual mark direction at L16 input: consistency across items and classes
    dr = z['dres16'].astype(np.float32)
    dn_ = dr / np.linalg.norm(dr, axis=1, keepdims=True)
    mc_, md_ = cls_arr == 'comparison', cls_arr == 'detector'
    out['dres16_cos'] = {'cmp': float((dn_[mc_] @ dn_[mc_].T)[np.triu_indices(mc_.sum(), 1)].mean()),
                         'det': float((dn_[md_] @ dn_[md_].T)[np.triu_indices(md_.sum(), 1)].mean()),
                         'cmp_det': float((dn_[mc_] @ dn_[md_].T).mean())}
    # layer-8 QK account: (n, 2 images, 16 heads, 4 stats)
    f8 = z['fa8'].astype(np.float32)
    out['fa8_qk'] = {}
    for cl in ('comparison', 'detector'):
        m = cls_arr == cl
        o, c = f8[m, 0].mean(0), f8[m, 1].mean(0)
        out['fa8_qk'][cl] = {'odd': o.round(3).tolist(), 'corr': c.round(3).tolist()}
    return out


def A_cls(items, iid):
    it = items[iid]
    return feature_class(it['pair_name'], it['variant'])


if __name__ == '__main__':
    main()
