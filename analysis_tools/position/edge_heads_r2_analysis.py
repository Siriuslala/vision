"""CPU analysis of experiment 3.11 round 4 (edge_heads_r2.py); writes summary.json and arrays.npz.

Definitions (doc section 3):
  value step   one emitted coordinate value (point x, y; box x_min, y_min, x_max, y_max). Query q0 predicts its first
               digit; slot -1 is the token before q0, slots 1 and 2 predict the second and third digit.
  attention    a head's softmax row at a query, restricted to the image tokens and renormalized to sum to 1.
  edge bands   inside the target box, the left band is the two token columns starting at the column that holds
               x_min (boundary token plus first inner column); right, top, bottom likewise. Only boxes spanning
               >= 5 tokens on both axes are used, so opposite bands never overlap (edge_heads_r2.region_labels).
  across-edge profile  for an x step: attention inside the box split into left-band columns, middle columns and
               right-band columns (shares pL, pM, pR); e* are the token-count shares (uniform attention). For y
               steps: top / middle / bottom rows.
  along-edge profile   within the band that holds the most attention relative to its token share, the split into the
               two end segments (corner tokens) and the middle segment; corner enrichment = corner share / corner
               token share (1 = no corner preference).
  causal r     first-digit logit recovery of a component patched from run B (3.11 round 1): r = (LD_e - LD_A) /
               (LD_B - LD_A), LD = z[d_B] - z[d_A]; only values with d_A != d_B and |LD_B - LD_A| >= 2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.position import edge_heads_r2 as e2
from analysis_tools.representation import sees_right as sr

FULL_LAYERS = e2.FULL_LAYERS
LI = {l: i for i, l in enumerate(FULL_LAYERS)}
R3_STEPS = [('point', 0, 'pt_x', 'x'), ('point', 1, 'pt_y', 'y'), ('bbox', 0, 'bb_x1', 'x'), ('bbox', 1, 'bb_y1', 'y'),
            ('bbox', 2, 'bb_x2', 'x'), ('bbox', 3, 'bb_y2', 'y')]
KEY_TO_STEP = {('point', 'x'): 'pt_x', ('point', 'y'): 'pt_y', ('bbox', 'x_min'): 'bb_x1', ('bbox', 'y_min'): 'bb_y1',
               ('bbox', 'x_max'): 'bb_x2', ('bbox', 'y_max'): 'bb_y2'}
STEP_NAMES = [s[2] for s in R3_STEPS]
STEP_AXIS = {s[2]: s[3] for s in R3_STEPS}
BANDS = ('min', 'mid', 'max')
MIN_LD = 2.0


def hname(l: int, h: int) -> str:
    return f'L{l}H{h}'


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float('nan')] * 3
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - w, c + w]


def mcnemar_p(b: int, c: int) -> float:
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def boot_ci(x: np.ndarray, rng, n: int = 1000, stat=np.mean) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float(stat(x)) if len(x) else float('nan')] * 3
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ----------------------------------------------------------------------------- profiles

def band_masks(box, grid, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Per token: across-edge band index (0 min band, 1 middle, 2 max band, -1 outside the box) along `axis`, and
    along-edge segment index (0 first end band, 1 middle, 2 last end band, -1 outside) along the other axis."""
    H, W = grid
    c0, c1, r0, r1 = e2.box_cells(box, grid)
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    r, c = r.ravel(), c.ravel()
    inside = (c >= c0) & (c <= c1) & (r >= r0) & (r <= r1)

    def tri(v, lo, hi):
        out = np.where(v <= lo + e2.BAND - 1, 0, np.where(v >= hi - e2.BAND + 1, 2, 1))
        return np.where(inside, out, -1)
    if axis == 'x':
        return tri(c, c0, c1), tri(r, r0, r1)
    return tri(r, r0, r1), tri(c, c0, c1)


def profiles(p: np.ndarray, box, grid, axis: str) -> dict:
    """Across-edge shares (3,), their token shares (3,), along-edge shares within each band (3, 3) and token shares.

    p: (..., Nv) image attention. Returns arrays with leading dims of p."""
    across, along = band_masks(box, grid, axis)
    inside = across >= 0
    tot = p[..., inside].sum(-1, keepdims=True) + 1e-12
    acr = np.stack([p[..., across == k].sum(-1) for k in range(3)], -1) / tot
    acr_e = np.array([(across == k).sum() for k in range(3)], float) / inside.sum()
    alo = np.zeros(p.shape[:-1] + (3, 3))
    alo_e = np.zeros((3, 3))
    for k in range(3):
        sel = across == k
        sub = p[..., sel].sum(-1, keepdims=True) + 1e-12
        for j in range(3):
            alo[..., k, j] = p[..., sel & (along == j)].sum(-1) / sub[..., 0]
            alo_e[k, j] = (sel & (along == j)).sum() / max(sel.sum(), 1)
    return {'across': acr, 'across_e': acr_e, 'along': alo, 'along_e': alo_e, 'inbox': tot[..., 0]}


# ----------------------------------------------------------------------------- round-3 attention (first digit)

def load_r3(round3: Path) -> list[dict]:
    """Round-3 answers whose box spans >= 5 tokens on both axes: attention (nv, 8, 16, Nv) at first-digit queries."""
    cases = {c['id']: c for c in json.loads((round3 / 'cases.json').read_text())}
    items = []
    for g in sr.read_jsonl(round3 / 'gen.jsonl'):
        if not g['score']['valid']:
            continue
        c = cases[g['case']]
        grid = tuple(g['grid'])
        box = c['truth']['box']
        if not e2.box_span_ok(box, grid):
            continue
        a = np.load(round3 / 'attn' / f"{g['case']}_{g['format']}.npz")['attn'].astype(np.float32)
        a = a / a.sum(-1, keepdims=True)
        items.append({'case': g['case'], 'format': g['format'], 'group': c['group'], 'family': c['family'],
                      'grid': grid, 'box': box, 'canvas': c['truth']['canvas'], 'att': a,
                      'keys': [v['key'] for v in g['values']], 'correct': g['score']['correct']})
    return items


def r3_tables(items: list[dict]) -> dict:
    """Head x value-step region statistics over round-3 answers (all 128 heads, first-digit queries)."""
    out = {}
    for fmt, vi, step, axis in R3_STEPS:
        its = [it for it in items if it['format'] == fmt]
        reg = np.zeros((len(its), 8, 16, len(e2.REGIONS)))
        reg_e = np.zeros((len(its), len(e2.REGIONS)))
        acr = np.zeros((len(its), 8, 16, 3))
        acr_e = np.zeros((len(its), 3))
        alo = np.zeros((len(its), 8, 16, 3, 3))
        alo_e = np.zeros((len(its), 3, 3))
        centre = np.zeros(len(its))
        for i, it in enumerate(its):
            p = it['att'][vi]                                   # (8, 16, Nv)
            lab = e2.region_labels(it['box'], it['grid'])
            reg[i] = e2.region_mass(p, lab)
            reg_e[i] = np.bincount(lab, minlength=len(e2.REGIONS)) / len(lab)
            pr = profiles(p, it['box'], it['grid'], axis)
            acr[i], acr_e[i], alo[i], alo_e[i] = pr['across'], pr['across_e'], pr['along'], pr['along_e']
            b = it['box']
            centre[i] = ((b[0] + b[2]) / 2 / it['canvas'][0]) if axis == 'x' else ((b[1] + b[3]) / 2 / it['canvas'][1])
        out[step] = {'n': len(its), 'region': reg, 'region_e': reg_e, 'across': acr, 'across_e': acr_e,
                     'along': alo, 'along_e': alo_e, 'centre': centre}
    return out


def classify(acr_mean: np.ndarray, acr_e_mean: np.ndarray, target: float, min_target: float = 0.3) -> str:
    """Read label of one head at one step from its mean across-edge shares.

    'min' / 'max' when that band holds >= 50% of the in-box attention and >= 2x its token share; 'centre' when the
    middle holds >= 50% and >= 1.2x its token share; 'mixed' otherwise; 'off-target' when less than min_target
    of the image attention is inside the box."""
    if target < min_target:
        return 'off-target'
    enr = acr_mean / acr_e_mean
    for k, name in ((0, 'min'), (2, 'max')):
        if acr_mean[k] >= 0.5 and enr[k] >= 2:
            return name
    if acr_mean[1] >= 0.5 and enr[1] >= 1.2:
        return 'centre'
    return 'mixed'


def r3_summary(tab: dict) -> dict:
    """Per step and head: mean shares, label, corner enrichment within the read band, side-location correlation."""
    res = {}
    inner = [e2.REGIONS.index(r) for r in e2.REGIONS[:9]]
    for step, t in tab.items():
        rows = {}
        reg = t['region']
        inbox = reg[..., inner].sum(-1)                         # (n, 8, 16)
        for li, l in enumerate(FULL_LAYERS):
            for h in range(16):
                a = t['across'][:, li, h]
                am, ae = a.mean(0), t['across_e'].mean(0)
                tgt = float(inbox[:, li, h].mean())
                lab = classify(am, ae, tgt)
                k = int(np.argmax(am / ae))
                al = t['along'][:, li, h, k]
                corner = (al[:, 0] + al[:, 2]).mean()
                corner_e = (t['along_e'][:, k, 0] + t['along_e'][:, k, 2]).mean()
                side = (a[:, 2] - a[:, 0]) / (a[:, 2] + a[:, 0] + 1e-9)
                r_loc = float(np.corrcoef(side, t['centre'])[0, 1]) if side.std() > 0 else float('nan')
                rows[hname(l, h)] = {
                    'inbox': round(tgt, 4), 'ring': round(float(reg[:, li, h, e2.REGIONS.index('out_ring')].mean()), 4),
                    'col0': round(float(reg[:, li, h, e2.REGIONS.index('out_col0')].mean()), 4),
                    'across': [round(float(v), 4) for v in am], 'across_e': [round(float(v), 4) for v in ae],
                    'label': lab, 'band': BANDS[k], 'corner_share': round(float(corner), 4),
                    'corner_token_share': round(float(corner_e), 4),
                    'corner_enrichment': round(float(corner / corner_e), 3),
                    'side_vs_location_r': round(r_loc, 3),
                    'region_mean': [round(float(v), 4) for v in reg[:, li, h].mean(0)]}
        res[step] = {'n': t['n'], 'region_token_share': [round(float(v), 4) for v in t['region_e'].mean(0)],
                     'heads': rows}
    return res


# ----------------------------------------------------------------------------- probe records

def load_probe(run: Path) -> list[dict]:
    recs = [r for r in sr.read_jsonl(run / 'probe.jsonl') if 'skipped' not in r]
    return recs


def probe_arrays(run: Path, rec: dict) -> dict:
    return dict(np.load(run / 'probe' / f"{rec['id'].replace(':', '__')}.npz"))


def r_values(rec: dict, field: str = 'lds') -> dict[str, np.ndarray]:
    """{value key: r per row} for values with d_A != d_B and |LD_B - LD_A| >= MIN_LD.

    LD_A is the 'none' row (row 0) of the condition table; scan rows use the same LD_A."""
    out = {}
    for k, d in enumerate(rec[field]):
        ld_a = rec['lds'][k]['ld'][0]
        den = d['LD_B'] - ld_a
        if d['dA'] is None or d['dA'] == d['dB'] or abs(den) < MIN_LD:
            continue
        out[d['key']] = (np.array(d['ld']) - ld_a) / den
    return out


def step_of(fmt: str, key: str) -> str:
    return KEY_TO_STEP[(fmt, key)]


def summarize_r(vals: list[np.ndarray], names: list[str], rng) -> dict:
    """Mean r with bootstrap CI over items for every row name."""
    if not vals:
        return {'n': 0}
    m = np.stack(vals)
    return {'n': int(len(m)), **{n: [round(v, 4) for v in boot_ci(m[:, i], rng, 500)] for i, n in enumerate(names)}}


# ----------------------------------------------------------------------------- A: causal head scans

def scan_tables(recs: list[dict], rng) -> dict:
    """Single-head patching r per head and value, grouped by (set, kind of pair, value step or moved side)."""
    groups: dict[str, list[np.ndarray]] = {}
    full: dict[str, list[float]] = {}
    names = None
    for rec in recs:
        if 'scan' not in rec:
            continue
        names = rec['scan']
        rv = r_values(rec, 'scan_lds')
        rc = r_values(rec)
        for key, r in rv.items():
            if rec['set'] in ('nat', 'comp'):
                tag = f"{rec['set']}:{step_of(rec['format'], key)}"
            else:
                side = rec['variant'][0]
                if key != e2.SIDES[side][0]:
                    continue                      # only the value on the moved axis changes
                tag = f"{rec['set']}:{side}"
            groups.setdefault(tag, []).append(r)
            full.setdefault(tag, []).append(rc[key][rec['conds'].index('full_B')])
    out = {}
    for tag, vals in groups.items():
        m = np.stack(vals)
        mean = m.mean(0)
        ci = np.array([boot_ci(m[:, i], rng, 300)[1:] for i in range(m.shape[1])])
        relevant = [names[i][1:] for i in np.argsort(-mean) if mean[i] >= 0.02 and ci[i, 0] > 0]
        out[tag] = {'n': int(len(m)), 'mean': {names[i][1:]: round(float(mean[i]), 4) for i in range(len(names))},
                    'ci_lo': {names[i][1:]: round(float(ci[i, 0]), 4) for i in range(len(names))},
                    'relevant': relevant, 'sum_single': round(float(mean.sum()), 4),
                    'sum_relevant': round(float(sum(mean[names.index('h' + h)] for h in relevant)), 4),
                    'full_B': [round(v, 4) for v in boot_ci(np.array(full[tag]), rng, 500)]}
    return out


# ----------------------------------------------------------------------------- B: attention profiles in probe run A

def probe_profiles(run: Path, recs: list[dict], sets=('nat',)) -> dict:
    """Across/along-edge profiles of run A at every digit slot (nat and large comp boxes), per step and head."""
    acc: dict = {}
    for rec in recs:
        if rec['set'] not in sets or rec['kind'] != 'where':
            continue
        grid = tuple(rec['grid'])
        if not e2.box_span_ok(rec['box_a'], grid):
            continue
        arr = probe_arrays(run, rec)
        att = arr['att_a'].astype(np.float32)                         # (nq, 6, 16, Nv)
        lab = e2.region_labels(rec['box_a'], grid)
        for qi, (key, slot) in enumerate(rec['digit_queries']):
            step = step_of(rec['format'], key)
            if slot >= 1 and len([1 for k, s_ in rec['digit_queries'] if k == key]) <= slot + 1:
                continue
            axis = STEP_AXIS[step]
            pr = profiles(att[qi], rec['box_a'], grid, axis)
            d = acc.setdefault((rec['set'], step, slot), {'across': [], 'across_e': [], 'along': [], 'along_e': [],
                                                           'region': [], 'region_e': [], 'inbox': []})
            d['across'].append(pr['across']), d['across_e'].append(pr['across_e'])
            d['along'].append(pr['along']), d['along_e'].append(pr['along_e'])
            d['region'].append(e2.region_mass(att[qi], lab))
            d['region_e'].append(np.bincount(lab, minlength=len(e2.REGIONS)) / len(lab))
            d['inbox'].append(pr['inbox'])
    out = {}
    for (st_set, step, slot), d in acc.items():
        A = np.stack(d['across'])                                   # (n, 6, 16, 3)
        Ae = np.stack(d['across_e']).mean(0)
        Al = np.stack(d['along'])
        Ale = np.stack(d['along_e']).mean(0)
        inbox = np.stack(d['inbox'])
        reg = np.stack(d['region']).mean(0)
        heads = {}
        for li, l in enumerate(e2.ATT_LAYERS):
            for h in range(16):
                am = A[:, li, h].mean(0)
                k = int(np.argmax(am / Ae))
                corner = (Al[:, li, h, k, 0] + Al[:, li, h, k, 2]).mean()
                ce = Ale[k, 0] + Ale[k, 2]
                heads[hname(l, h)] = {'across': [round(float(v), 4) for v in am], 'inbox': round(float(inbox[:, li, h].mean()), 4),
                                      'label': classify(am, Ae, float(inbox[:, li, h].mean())), 'band': BANDS[k],
                                      'corner_enrichment': round(float(corner / ce), 3),
                                      'col0': round(float(reg[li, h, e2.REGIONS.index('out_col0')]), 4),
                                      'region_mean': [round(float(v), 4) for v in reg[li, h]]}
        out[f'{st_set}:{step}:{slot}'] = {'n': int(len(A)), 'across_e': [round(float(v), 4) for v in Ae], 'heads': heads}
    return out


# ----------------------------------------------------------------------------- C: group and mediation conditions

def condition_tables(recs: list[dict], rng) -> dict:
    """r of every condition row, grouped by set / pair kind / value (or moved side for one-edge moves)."""
    acc: dict = {}
    for rec in recs:
        rv = r_values(rec)
        for key, r in rv.items():
            if rec['set'] in ('str', 'rect'):
                side = rec['variant'][0]
                if key != e2.SIDES[side][0]:
                    continue
                tag = f"{rec['set']}:{side}:{key}"
            elif rec['set'] == 'comp':
                tag = f"comp:{rec['kind']}:{rec['format']}:{key}"
            else:
                tag = f"nat:{rec['format']}:{key}"
            acc.setdefault(tag, {'names': rec['conds'], 'vals': []})['vals'].append(r)
    return {tag: summarize_r(d['vals'], d['names'], rng) for tag, d in acc.items()}


# ----------------------------------------------------------------------------- D: behaviour under one-edge moves

def to_pixels(values: dict, canvas, conv: str) -> dict:
    W, H = canvas
    if conv == 'pixel':
        return dict(values)
    return {k: v * (W if k.startswith('x') else H) / 1000 for k, v in values.items()}


def convention(values: dict, box, canvas, fmt: str) -> str:
    """Coordinate convention ('norm' 0-1000 or 'pixel') under which run A's answer is nearer the truth."""
    errs = {}
    for conv in ('norm', 'pixel'):
        v = to_pixels(values, canvas, conv)
        if fmt == 'point':
            errs[conv] = abs(v['x'] - (box[0] + box[2]) / 2) + abs(v['y'] - (box[1] + box[3]) / 2)
        else:
            errs[conv] = sum(abs(v[k] - box[i]) for i, k in enumerate(('x_min', 'y_min', 'x_max', 'y_max')))
    return min(errs, key=errs.get)


def behaviour(run: Path, rng) -> dict:
    """Slope of every answer value (pixels) on the moved-edge displacement delta, per set, format, side."""
    pts: dict = {}
    conv_count: dict = {}
    for rec in sr.read_jsonl(run / 'behav.jsonl'):
        base = rec['rows'][0]
        if base['values'] is None:
            continue
        conv = convention(base['values'], base['box'], rec['canvas'], rec['format'])
        conv_count[(rec['set'], conv)] = conv_count.get((rec['set'], conv), 0) + 1
        v0 = to_pixels(base['values'], rec['canvas'], conv)
        for row in rec['rows'][1:]:
            if row['values'] is None:
                continue
            side, delta = row['variant']
            v = to_pixels(row['values'], rec['canvas'], conv)
            # signed displacement of the moved edge along its axis (positive = towards larger coordinates)
            disp = delta if e2.SIDES[side][1] == 'max' else -delta
            for k in v:
                pts.setdefault((rec['set'], rec['format'], side, k), []).append((rec['case'], disp, v[k] - v0[k]))
    out = {'convention_counts': {f'{s}:{c}': n for (s, c), n in conv_count.items()}}
    for (st, fmt, side, k), lst in pts.items():
        case_ids = sorted({c for c, _, _ in lst})
        arr = np.array([(d, y) for _, d, y in lst])
        slope = float((arr[:, 0] * arr[:, 1]).sum() / (arr[:, 0] ** 2).sum())   # through the origin

        def boot():
            idx = rng.integers(0, len(case_ids), len(case_ids))
            sel = np.concatenate([[i for i, (c, _, _) in enumerate(lst) if c == case_ids[j]] for j in idx])
            a = arr[sel.astype(int)]
            return (a[:, 0] * a[:, 1]).sum() / (a[:, 0] ** 2).sum()
        bs = [boot() for _ in range(300)]
        by_delta = {}
        for d in sorted({d for _, d, _ in lst}):
            ys = arr[arr[:, 0] == d, 1]
            by_delta[str(int(d))] = [round(float(np.median(ys)), 2), int(len(ys))]
        out[f'{st}:{fmt}:{side}:{k}'] = {'slope': round(slope, 4), 'ci': [round(float(np.percentile(bs, 2.5)), 4),
                                         round(float(np.percentile(bs, 97.5)), 4)], 'n_items': len(case_ids),
                                         'median_shift_by_delta': by_delta}
    return out


# ----------------------------------------------------------------------------- E: composites (L20 H11)

COND_LI = {l: i for i, l in enumerate(e2.COND_LAYERS)}
READERS = ['L24H7', 'L24H2', 'L20H0', 'L20H10', 'L24H5', 'L24H9', 'L24H6', 'L28H0']


def comp_cell(c: dict) -> str:
    return f"{c['objtype']}|{c['bg']}|{c['size']}|{c['ndistr']}"


def point_hit(values: dict | None, box, canvas) -> tuple[bool, float]:
    """Hit under either coordinate convention and the centre error (pixels) of the better convention."""
    if values is None:
        return False, float('nan')
    best = (False, float('inf'))
    for conv in ('norm', 'pixel'):
        v = to_pixels(values, canvas, conv)
        hit = box[0] <= v['x'] <= box[2] and box[1] <= v['y'] <= box[3]
        err = float(np.hypot(v['x'] - (box[0] + box[2]) / 2, v['y'] - (box[1] + box[3]) / 2))
        if (hit, -err) > (best[0], -best[1]):
            best = (hit, err)
    return best


def comp_ablation(run: Path, cases: dict) -> dict:
    rows: dict = {}
    for rec in sr.read_jsonl(run / 'ablate.jsonl'):
        c = cases[rec['id']]
        base = None
        for r in rec['results']:
            hit, err = point_hit(r['values'], c['box'], c['canvas'])
            if r['set'] == 'none':
                base = (hit, err)
            rows.setdefault(r['set'], []).append((comp_cell(c), hit, err, base))
    out = {}
    cells = sorted({cell for v in rows.values() for cell, *_ in v})
    for name, lst in rows.items():
        for cell in cells + ['all']:
            sel = [x for x in lst if cell == 'all' or x[0] == cell]
            hits = np.array([x[1] for x in sel])
            base = np.array([x[3][0] for x in sel])
            err = np.array([x[2] for x in sel])
            b = int((base & ~hits).sum())
            cc = int((~base & hits).sum())
            out.setdefault(cell, {})[name] = {'hit': [round(v, 4) for v in wilson(int(hits.sum()), len(hits))],
                                              'n': len(sel), 'lost': b, 'gained': cc, 'p': mcnemar_p(b, cc),
                                              'median_err': round(float(np.nanmedian(err)), 2)}
    return out


def mass_in(p: np.ndarray, box, grid) -> np.ndarray:
    return e2.box_mass(p, box, grid)


def comp_probe(run: Path, cases: dict, recs: list[dict], rng) -> dict:
    """H11 precision, sufficiency, relay to readers, query patching and distractor attention, by design cell."""
    acc: dict = {}
    for rec in recs:
        if rec['set'] != 'comp' or rec['format'] != 'point':
            continue
        c = cases[rec['case']]
        grid = tuple(rec['grid'])
        arr = probe_arrays(run, rec)
        names = rec['conds']
        ca = arr['cond_att'].astype(np.float32)                       # (R, nv, 3, 16, Nv)
        att_a = arr['att_a'].astype(np.float32)
        q0 = [i for i, (k, s_) in enumerate(rec['digit_queries']) if s_ == 0]
        cell = comp_cell(c)
        d = acc.setdefault((rec['kind'], cell), {})
        boxA, boxB = rec['box_a'], rec['box_b']
        for vi in range(len(q0)):
            h11 = att_a[q0[vi], e2.ATT_LAYERS.index(20), 11]
            d.setdefault('h11_mass_target', []).append(float(mass_in(h11, boxA, grid)))
            pk = e2.peak_xy(h11, grid)
            d.setdefault('h11_peak_in_box', []).append(float(boxA[0] - 16 <= pk[0] <= boxA[2] + 16 and boxA[1] - 16 <= pk[1] <= boxA[3] + 16))
            for hn in READERS + ['L16H1', 'L16H2', 'L20H11']:
                l, h = map(int, hn[1:].split('H'))
                pa = att_a[q0[vi], e2.ATT_LAYERS.index(l), h]
                d.setdefault(f'mass_target:{hn}', []).append(float(mass_in(pa, boxA, grid)))
            # relay: readers' target mass with H11 mean-ablated (row) vs none (row 0)
            ri = names.index('H11_mean')
            for hn in READERS:
                l, h = map(int, hn[1:].split('H'))
                if l not in COND_LI:
                    continue
                m0 = mass_in(ca[0, vi, COND_LI[l], h], boxA, grid)
                m1 = mass_in(ca[ri, vi, COND_LI[l], h], boxA, grid)
                d.setdefault(f'relay_drop:{hn}', []).append(float(m0 - m1))
            # query patching of H11: attention on A's target vs B's target box after q11_B
            qi = names.index('q11_B')
            pq = ca[qi, vi, COND_LI[20], 11]
            d.setdefault('q11_mass_A', []).append(float(mass_in(pq, boxA, grid)))
            d.setdefault('q11_mass_B', []).append(float(mass_in(pq, boxB, grid)))
            d.setdefault('none_mass_B', []).append(float(mass_in(ca[0, vi, COND_LI[20], 11], boxB, grid)))
            if c['ndistr']:
                for o in c['objects'][1:]:
                    d.setdefault(f"distr_mass:{o['role']}", []).append(float(mass_in(h11, o['box'], grid)))
        for key, r in r_values(rec).items():
            for n in ('H11_B', 'L20_B', 'full_B', 'dn_B', 'q11_B', 'H11_mean') + (('L16H1H2_B', 'dn_pre20_B') if rec['kind'] == 'what' else ()):
                if n in names:
                    d.setdefault(f'r:{n}', []).append(float(r[names.index(n)]))
                    d.setdefault(f'r_{key}:{n}', []).append(float(r[names.index(n)]))
    out = {}
    for (kind, cell), d in acc.items():
        out[f'{kind}:{cell}'] = {k: [round(v, 4) for v in boot_ci(np.array(v), rng, 300)] + [len(v)] for k, v in d.items()}
    return out


# ----------------------------------------------------------------------------- F: what H11 writes (OV) and reads (V)

def ridge_cv(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, lam: float = 10.0, folds: int = 5) -> np.ndarray:
    """Cross-validated R^2 (per output column) of ridge regression, folds split by group id."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    ug = np.unique(groups)
    fold = {g: i % folds for i, g in enumerate(np.random.default_rng(0).permutation(ug))}
    f = np.array([fold[g] for g in groups])
    pred = np.zeros_like(Y, dtype=float)
    for k in range(folds):
        tr, te = f != k, f == k
        mu = Y[tr].mean(0)
        A = X[tr].T @ X[tr] + lam * len(X[tr]) * np.eye(X.shape[1]) / 100
        W = np.linalg.solve(A, X[tr].T @ (Y[tr] - mu))
        pred[te] = X[te] @ W + mu
    ss = ((Y - pred) ** 2).sum(0)
    st = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss / st


def class_cv(X: np.ndarray, labels: list[str], groups: np.ndarray) -> float:
    cats = sorted(set(labels))
    Y = np.array([[l == c for c in cats] for l in labels], float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    ug = np.unique(groups)
    fold = {g: i % 5 for i, g in enumerate(np.random.default_rng(0).permutation(ug))}
    f = np.array([fold[g] for g in groups])
    correct = 0
    for k in range(5):
        tr, te = f != k, f == k
        A = Xs[tr].T @ Xs[tr] + 10 * np.eye(X.shape[1])
        W = np.linalg.solve(A, Xs[tr].T @ (Y[tr] - Y[tr].mean(0)))
        correct += int((np.argmax(Xs[te] @ W, 1) == np.argmax(Y[te], 1)).sum())
    return correct / len(labels)


OV_HEADS = ['L20H11', 'L20H10', 'L20H0', 'L24H5', 'L24H7', 'L16H1', 'L16H2', 'L20H3']


def ov_decoding(run: Path, cases: dict, recs: list[dict]) -> dict:
    """Decode target properties from each head's output (at the x and y first-digit queries of run A) and the
    image-token position from L20 H11's value / key vectors."""
    res = {}
    for st in ('comp', 'nat'):
        feats, ys, cats, grp, vk = {h: [] for h in OV_HEADS}, [], {'color': [], 'shape': [], 'objtype': []}, [], []
        for rec in recs:
            if rec['set'] != st or rec['format'] != 'point' or rec['kind'] != 'where':
                continue
            c = cases[rec['case']]
            arr = probe_arrays(run, rec)
            H = arr['heads'].astype(np.float32)                        # (nv, 5, 16, 256)
            b = rec['box_a']
            for hn in OV_HEADS:
                l, h = map(int, hn[1:].split('H'))
                feats[hn].append(np.concatenate([H[0, e2.OUT_LAYERS.index(l), h], H[1, e2.OUT_LAYERS.index(l), h]]))
            ys.append([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[0], b[2], b[1], b[3], np.log(max(b[2] - b[0], b[3] - b[1]))])
            if st == 'comp':
                o = c['objects'][0]
                cats['color'].append(o.get('color', 'none')), cats['shape'].append(o.get('shape', o.get('name')))
                cats['objtype'].append(c['objtype'])
            grp.append(rec['case'])
            if len(vk) < 120:
                vk.append((arr['v11'].astype(np.float32), arr['k11'].astype(np.float32), tuple(rec['grid']), rec['case']))
        if len(set(grp)) < 20:
            res[st] = {'n': len(ys), 'skipped': 'fewer than 20 items'}
            continue
        Y = np.array(ys)
        g = np.array(grp)
        out = {'n': len(Y), 'targets': ['cx', 'cy', 'x_min', 'x_max', 'y_min', 'y_max', 'log_size']}
        for hn in OV_HEADS:
            X = np.stack(feats[hn])
            d = {'r2': [round(float(v), 3) for v in ridge_cv(X, Y, g)]}
            if st == 'comp':
                shapes_only = np.array([t == 'shape' for t in cats['objtype']])
                d['color_acc'] = round(class_cv(X[shapes_only], [cats['color'][i] for i in np.nonzero(shapes_only)[0]],
                                                g[shapes_only]), 3)
                d['shape_acc'] = round(class_cv(X[shapes_only], [cats['shape'][i] for i in np.nonzero(shapes_only)[0]],
                                                g[shapes_only]), 3)
            out[hn] = d
        # V / K of the image tokens: decode each token's own column and row
        Xv, Xk, P, G = [], [], [], []
        for v, k, grid, cid in vk:
            Hh, Ww = grid
            r, cc = np.meshgrid(np.arange(Hh), np.arange(Ww), indexing='ij')
            Xv.append(v)
            Xk.append(k)
            P.append(np.stack([cc.ravel(), r.ravel()], 1))
            G += [cid] * len(v)
        Xv, Xk, P, G = np.concatenate(Xv), np.concatenate(Xk), np.concatenate(P).astype(float), np.array(G)
        out['v11_token_pos_r2'] = [round(float(x), 3) for x in ridge_cv(Xv, P, G)]
        out['k11_token_pos_r2'] = [round(float(x), 3) for x in ridge_cv(Xk, P, G)]
        res[st] = out
    return res


# ----------------------------------------------------------------------------- F2: where the attention goes (target neighbourhood)

NEIGH = ('inbox', 'ring', 'row_after', 'row_before', 'same_cols', 'col0', 'other')


def neighbourhood(run: Path, cases: dict, recs: list[dict], heads=('L20H11', 'L24H5', 'L16H1')) -> dict:
    """Attention of run A at the point first-digit queries of comp where pairs, split into: the target box, the
    one-token ring around it, the target's token rows outside the box (after / before the box in raster order),
    the target's token columns outside the box, the first image column, and the rest. Mean per object type,
    background, value and head."""
    acc: dict = {}
    for rec in recs:
        if rec['set'] != 'comp' or rec['format'] != 'point' or rec['kind'] != 'where':
            continue
        c = cases[rec['case']]
        H, W = rec['grid']
        c0, c1, r0, r1 = e2.box_cells(rec['box_a'], (H, W))
        r, cc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        r, cc = r.ravel(), cc.ravel()
        inb = (cc >= c0) & (cc <= c1) & (r >= r0) & (r <= r1)
        ring = ~inb & (cc >= c0 - 1) & (cc <= c1 + 1) & (r >= r0 - 1) & (r <= r1 + 1)
        rows = ~inb & ~ring & (r >= r0) & (r <= r1)
        masks = {'inbox': inb, 'ring': ring, 'row_after': rows & (cc > c1), 'row_before': rows & (cc < c0),
                 'same_cols': ~inb & ~ring & (cc >= c0) & (cc <= c1), 'col0': ~inb & ~ring & ~rows & (cc == 0)}
        masks['other'] = ~np.any(np.stack(list(masks.values())), 0)
        att = probe_arrays(run, rec)['att_a'].astype(np.float32)
        q0 = [i for i, (k, s_) in enumerate(rec['digit_queries']) if s_ == 0]
        for vi, ax in enumerate(('x', 'y')):
            for hn in heads:
                l, h = map(int, hn[1:].split('H'))
                p = att[q0[vi], e2.ATT_LAYERS.index(l), h]
                acc.setdefault(f"{c['objtype']}|{c['bg']}|{ax}|{hn}", []).append([float(p[masks[k]].sum()) for k in NEIGH])
    return {'regions': list(NEIGH), **{k: [round(float(v), 4) for v in np.mean(x, 0)] + [len(x)] for k, x in acc.items()}}


# ----------------------------------------------------------------------------- G: direct writes into the residual

def o_proj_weights(layers) -> dict[int, np.ndarray]:
    """Output-projection weights (D, 16 * hd) of the given full-attention layers, read on CPU from the checkpoint."""
    from safetensors import safe_open
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    ck = Path(resolve_model('Qwen3.5-9B', load_paths()['HF_HUB_CACHE']))
    wmap = json.loads((ck / 'model.safetensors.index.json').read_text())['weight_map']
    out = {}
    for l in layers:
        key = f'model.language_model.layers.{l - 1}.self_attn.o_proj.weight'
        with safe_open(str(ck / wmap[key]), framework='pt') as f:
            out[l] = f.get_tensor(key).float().numpy()
    return out


def direct_writes(run: Path, recs: list[dict], lam: float = 1e3) -> dict:
    """Linear read-out direction of the coordinate in the residual and each head's direct write along it.

    For every natural point answer of run A (nat set) and each axis a: a ridge direction w_a is fitted on
    the decoder-layer-28 output at the axis' first-digit query (target: box centre / canvas size; 5-fold CV R^2
    reported). Each head's direct write is W_O[:, head] o_head (its contribution to the residual); its projection
    on w_a is regressed across items on (edge_min, edge_max) / canvas. Additivity check: the summed projections of
    all L16-L28 heads against the centre."""
    W = o_proj_weights(e2.OUT_LAYERS)
    res = {}
    for ax_i, ax in enumerate(('x', 'y')):
        R, Hs, Y, E, G = [], [], [], [], []
        for rec in recs:
            if rec['format'] != 'point' or rec['kind'] != 'where' or rec['set'] != 'nat':
                continue
            arr = probe_arrays(run, rec)
            b = rec['box_a']
            Wc, Hc = rec['grid'][1] * e2.TP, rec['grid'][0] * e2.TP
            n = Wc if ax == 'x' else Hc
            lo, hi = (b[0], b[2]) if ax == 'x' else (b[1], b[3])
            R.append(arr['resid'][ax_i, e2.RES_LAYERS.index(28)].astype(np.float32))
            Hs.append(arr['heads'][ax_i].astype(np.float32))                        # (5, 16, hd)
            Y.append((lo + hi) / 2 / n)
            E.append([lo / n, hi / n])
            G.append(rec['set'])
        if len(Y) < 40:
            res[ax] = {'n': len(Y), 'skipped': 'fewer than 40 answers'}
            continue
        R, Hs, Y, E = np.stack(R), np.stack(Hs), np.array(Y), np.array(E)
        mu, sd = R.mean(0), R.std(0) + 1e-6
        Z = (R - mu) / sd
        r2 = float(ridge_cv(Z, Y[:, None], np.arange(len(Y)), lam=lam)[0])
        A = Z.T @ Z + lam * len(Z) / 100 * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (Y - Y.mean())) / sd                         # raw-residual direction
        heads = {}
        X = np.c_[E, np.ones(len(E))]
        total = np.zeros(len(Y))
        for li, l in enumerate(e2.OUT_LAYERS):
            if l == 32:
                continue
            hd = Hs.shape[-1]
            proj = W[l].T @ w                                                    # (16 * hd,)
            for h in range(16):
                c = Hs[:, li, h] @ proj[h * hd:(h + 1) * hd]
                total += c
                beta, *_ = np.linalg.lstsq(X, c, rcond=None)
                pred = X @ beta
                r2h = 1 - ((c - pred) ** 2).sum() / max(((c - c.mean()) ** 2).sum(), 1e-12)
                heads[hname(l, h)] = {'b_min': round(float(beta[0]), 4), 'b_max': round(float(beta[1]), 4),
                                      'r2': round(float(r2h), 3), 'sd': round(float(c.std()), 4)}
        beta, *_ = np.linalg.lstsq(X, total, rcond=None)
        res[ax] = {'n': len(Y), 'probe_cv_r2': round(r2, 3), 'heads': heads,
                   'sum_L16_28': {'b_min': round(float(beta[0]), 4), 'b_max': round(float(beta[1]), 4),
                                  'r_with_centre': round(float(np.corrcoef(total, Y)[0, 1]), 3)},
                   'probe_on_centre_slope': round(float(np.polyfit(Y, R @ w, 1)[0]), 4)}
    return res


def mediation(run: Path, rng) -> dict:
    """Per-layer MLP freezing (mediate.jsonl): r of the edge groups patched from B with single MLP layers frozen to
    run A, or with every MLP of layers 20-32 frozen except one. Grouped by set and moved side (nat: translation)."""
    acc: dict = {}
    for rec in sr.read_jsonl(run / 'mediate.jsonl'):
        names = rec['conds']
        for key, r in r_values(rec).items():
            if rec['set'] in ('str', 'rect'):
                side = rec['variant'][0]
                if key != e2.SIDES[side][0]:
                    continue
                tag = f"{rec['set']}:{side}:{key}"
            else:
                tag = f'nat:{key}'
            sel = [i for i, n in enumerate(names) if n == 'none' or n.startswith(f'edges{key}_')]
            acc.setdefault(tag, {'names': [names[i] for i in sel], 'vals': []})['vals'].append(r[sel])
    return {tag: summarize_r(d['vals'], d['names'], rng) for tag, d in acc.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--round3', type=Path, required=True)
    p.add_argument('--parts', nargs='+', default=['r3', 'probe', 'behav', 'comp', 'direct', 'mediate'])
    args = p.parse_args()
    rng = np.random.default_rng(20261105)
    summ_path = args.run / 'summary.json'
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    if 'r3' in args.parts:
        items = load_r3(args.round3)
        tab = r3_tables(items)
        summary['r3_regions'] = r3_summary(tab)
        np.savez_compressed(args.run / 'r3_profiles.npz',
                            **{f'{s}_{k}': v for s, t in tab.items() for k, v in t.items() if isinstance(v, np.ndarray)})
        print('r3 done', {s: t['n'] for s, t in tab.items()})
    cases = e2.load_cases(args.run)
    if 'probe' in args.parts and (args.run / 'probe.jsonl').exists():
        recs = load_probe(args.run)
        summary['scan'] = scan_tables(recs, rng)
        summary['profiles'] = probe_profiles(args.run, recs, ('nat', 'comp'))
        summary['conditions'] = condition_tables(recs, rng)
        summary['probe_n'] = {s: sum(r['set'] == s for r in recs) for s in ('nat', 'str', 'rect', 'comp')}
        summary['attn_check_max'] = max(r['attn_check'] for r in recs)
        print('probe done', summary['probe_n'])
    if 'mediate' in args.parts and (args.run / 'mediate.jsonl').exists():
        summary['mediation'] = mediation(args.run, rng)
    if 'behav' in args.parts and (args.run / 'behav.jsonl').exists():
        summary['behaviour'] = behaviour(args.run, rng)
    if 'comp' in args.parts:
        if (args.run / 'ablate.jsonl').exists():
            summary['comp_ablation'] = comp_ablation(args.run, cases)
        if (args.run / 'probe.jsonl').exists():
            recs = load_probe(args.run)
            summary['comp_probe'] = comp_probe(args.run, cases, recs, rng)
            summary['ov'] = ov_decoding(args.run, cases, recs)
            summary['neighbourhood'] = neighbourhood(args.run, cases, recs)
    if 'direct' in args.parts and (args.run / 'probe.jsonl').exists():
        summary['direct'] = direct_writes(args.run, load_probe(args.run))
    gen = sr.read_jsonl(args.run / 'gen.jsonl') if (args.run / 'gen.jsonl').exists() else []
    if gen:
        summary['gen'] = {}
        for st in ('nat', 'str', 'rect', 'comp'):
            for fmt in ('point', 'bbox'):
                gs = [g for g in gen if g['set'] == st and g['format'] == fmt]
                if gs:
                    ok = sum(g['score']['correct'] for g in gs)
                    summary['gen'][f'{st}:{fmt}'] = {'n': len(gs), 'valid': sum(g['score']['valid'] for g in gs),
                                                    'hit_norm': wilson(ok, len(gs))}
    summ_path.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
