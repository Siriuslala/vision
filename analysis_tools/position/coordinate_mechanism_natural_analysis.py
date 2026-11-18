"""Experiment 3.15 round 2: statistics and figures (CPU only).

Reads gen.jsonl, flow.jsonl, timeline.jsonl and think.jsonl of the run directory, the 3.15 round-1 RefCOCO CoT texts
and the 3.11 round-3 natural-image answers, writes summary.json and the figures of
docs/analysis/position/results_2026-11-05_coordinate_mechanism_natural.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.position import coordinate_mechanism_natural as cmn

C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
     'green': '#008300', 'gray': '#8c8c8c', 'dark': '#3a3a3a'}
FAM_LABEL = {'grid10': '10x10 grid', 'grid8': '8x8 grid', 'scatter': 'scatter', 'single': 'single figure',
             'thumb_same': 'thumb grid (identical)', 'thumb_mixed': 'thumb grid (mixed)', 'refcoco': 'RefCOCO'}
FAM_ORDER = ('grid10', 'grid8', 'thumb_same', 'thumb_mixed', 'scatter', 'single', 'refcoco')
FULL_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
R1 = 'position/2026-11-03_ordinal_vs_coordinate'
EH = 'position/2026-11-03_edge_heads'


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open()] if p.exists() else []


def boot_mean(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e6e6e6', lw=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- gen: value structure and count tests

def gen_index(rows: list[dict]) -> dict:
    return {(r['family'], r['case'], r['model'], r['fmt'], r['s'], r['variant']): r for r in rows}


def point_of(r, convert: bool = True):
    """Parsed point in 0-1000 units. Some answers use absolute pixel coordinates (plan.md section 6, item 9);
    when the pixel reading is clearly closer to the truth (error < half of the normalised reading's error), the
    answer is converted to 0-1000 units and counted as a pixel-convention answer."""
    if r is None:
        return None
    p = cmn.parse_point(r['text'])
    if p is None or not convert or not r.get('truth'):
        return p
    return to_units(p, r['truth'])[0]


def to_units(p, truth) -> tuple[list[float], bool]:
    W, H = truth['canvas']
    c = truth['center_units']
    q = [1000 * p[0] / W, 1000 * p[1] / H]
    e_n = np.hypot(p[0] - c[0], p[1] - c[1])
    e_p = np.hypot(q[0] - c[0], q[1] - c[1])
    return (q, True) if e_p < 0.5 * e_n else (p, False)


def lattice_share(G: dict, model: str = 'base', s_: float = 1.0) -> dict:
    """Share of answer values within 3 units of the half-token lattice 1000 k / (2 N_tok) (N_tok = merged tokens
    along the axis), and the chance level 7 / step for uniformly spread values."""
    out = {}
    for fam in FAM_ORDER:
        d = []
        step = None
        for k, r in G.items():
            if k[0] != fam or k[2] != model or k[3] != 'point' or k[4] != s_ or k[5] != 'clean':
                continue
            p = point_of(r)
            if p is None:
                continue
            W, H = r['truth']['canvas']
            for v, n in zip(p, (W // 32, H // 32)):
                step = 1000 / (2 * n)
                d.append(abs(((v / step) + 0.5) % 1 - 0.5) * step <= 3)
        if d:
            out[fam] = {'n_values': len(d), 'share': float(np.mean(d)), 'chance': float(min(1.0, 7 / step))}
    return out


def hit_rates(G: dict, model: str = 'base') -> dict:
    out = {}
    for fam in FAM_ORDER:
        for s_ in (1.0, 0.0):
            rs = [r for k, r in G.items() if k[0] == fam and k[2] == model and k[3] == 'point' and k[4] == s_ and k[5] == 'clean']
            h = 0
            for r in rs:
                p = point_of(r)
                if p is None:
                    continue
                b = r['truth'].get('cell_box_units', r['truth']['box_units'])
                h += b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
            if rs:
                out[f'{fam}|s{int(s_)}'] = {'n': len(rs), 'hit': h, 'ci': cmn.wilson(h, len(rs))}
    return out


def pixel_share(G: dict, fmt: str = 'point') -> dict:
    out = {}
    for fam in FAM_ORDER:
        rs = [r for k, r in G.items() if k[0] == fam and k[2] == 'base' and k[3] == fmt and k[4] == 1.0 and k[5] == 'clean']
        flags = [to_units(cmn.parse_point(r['text']), r['truth'])[1] for r in rs if cmn.parse_point(r['text'])]
        if flags:
            out[fam] = {'n': len(flags), 'pixel': int(sum(flags))}
    return out


def value_structure(G: dict, model: str = 'base') -> dict:
    out = {}
    for fam in FAM_ORDER:
        recs = [r for k, r in G.items() if k[0] == fam and k[2] == model and k[3] == 'point' and k[4] == 1.0
                and k[5] == 'clean']
        if not recs:
            continue
        fb, fc, hit, n_valid = [], [], 0, 0
        for r in recs:
            p = point_of(r)
            if p is None:
                continue
            n_valid += 1
            t = r['truth']
            bu = t['box_units']
            fb.append([cmn.relative_position(p[a], bu[a], bu[a + 2]) for a in (0, 1)])
            if 'cell_box_units' in t:
                cb = t['cell_box_units']
                fc.append([cmn.relative_position(p[a], cb[a], cb[a + 2]) for a in (0, 1)])
                hit += cb[0] <= p[0] <= cb[2] and cb[1] <= p[1] <= cb[3]
            else:
                hit += bu[0] <= p[0] <= bu[2] and bu[1] <= p[1] <= bu[3]
        fb, fc = np.array(fb), np.array(fc)
        d = {'n': len(recs), 'valid': n_valid, 'hit': hit, 'hit_ci': cmn.wilson(hit, n_valid),
             'box_rel_median': np.median(fb, 0).round(3).tolist() if len(fb) else None, 'box_rel': fb.round(3).tolist()}
        if len(fc):
            d['cell_rel'] = fc.round(3).tolist()
            for a, ax in enumerate(('x', 'y')):
                cls = [cmn.position_class(f) for f in fc[:, a]]
                d[f'cell_class_{ax}'] = {c: cls.count(c) / len(cls) for c in ('centre', 'upper_edge', 'lower_edge', 'other')}
        else:
            for a, ax in enumerate(('x', 'y')):
                cls = [cmn.position_class(f) for f in fb[:, a]]
                d[f'box_class_{ax}'] = {c: cls.count(c) / len(cls) for c in ('centre', 'upper_edge', 'lower_edge', 'other')}
        errs = [np.hypot(point_of(r)[0] - r['truth']['center_units'][0], point_of(r)[1] - r['truth']['center_units'][1])
                for r in recs if point_of(r)]
        d['median_error'] = float(np.median(errs)) if errs else None
        vals = [v for r in recs if point_of(r) for v in point_of(r)]
        d['round10'] = float(np.mean([v % 10 == 0 for v in vals])) if vals else None
        out[fam] = d
    return out


def paired_shift(G: dict, model: str, fmt: str, cond_b: tuple, cond_a=(1.0, 'clean')) -> dict:
    """Per family: answer shift (B condition minus clean) in units and in cell / box units."""
    out = {}
    for fam in FAM_ORDER:
        dx, dy, dxn, dyn, n = [], [], [], [], 0
        for k, r in G.items():
            if k[0] != fam or k[2] != model or k[3] != fmt or (k[4], k[5]) != cond_a:
                continue
            rb = G.get((fam, k[1], model, fmt) + cond_b)
            if rb is None:
                continue
            n += 1
            pa, pb = point_of(r), point_of(rb)
            if pa is None or pb is None:
                continue
            t = r['truth']
            scale = ([t['cell_box_units'][2] - t['cell_box_units'][0], t['cell_box_units'][3] - t['cell_box_units'][1]]
                     if 'cell_box_units' in t else
                     [t['box_units'][2] - t['box_units'][0], t['box_units'][3] - t['box_units'][1]])
            dx.append(pb[0] - pa[0])
            dy.append(pb[1] - pa[1])
            dxn.append((pb[0] - pa[0]) / scale[0])
            dyn.append((pb[1] - pa[1]) / scale[1])
        if n == 0:
            continue
        out[fam] = {'n': n, 'valid': len(dx), 'dx': boot_mean(dx), 'dy': boot_mean(dy), 'dx_norm': boot_mean(dxn),
                    'dy_norm': boot_mean(dyn), 'abs_dx': boot_mean(np.abs(dx)), 'abs_dy': boot_mean(np.abs(dy)),
                    'moved_gt_half': float(np.mean([abs(a) > 0.5 or abs(b) > 0.5 for a, b in zip(dxn, dyn)]))
                    if dx else None}
    return out


def bbox_structure(G: dict) -> dict:
    out = {}
    for fam in FAM_ORDER:
        rows = [(r, G.get((fam, k[1], 'base', 'point', 1.0, 'clean'))) for k, r in G.items()
                if k[0] == fam and k[2] == 'base' and k[3] == 'bbox']
        e_tight, e_cell, d_point = [], [], []
        for r, rp in rows:
            b = cmn.parse_bbox(r['text'])
            if b is None:
                continue
            t = r['truth']
            _, pix = to_units([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], t)
            if pix:                                         # pixel-convention box -> 0-1000 units
                W, H = t['canvas']
                b = [1000 * b[0] / W, 1000 * b[1] / H, 1000 * b[2] / W, 1000 * b[3] / H]
            e_tight.append(np.array(b) - np.array(t['box_units']))
            if 'cell_box_units' in t:
                e_cell.append(np.array(b) - np.array(t['cell_box_units']))
            p = point_of(rp)
            if p is not None:
                d_point.append([p[0] - (b[0] + b[2]) / 2, p[1] - (b[1] + b[3]) / 2, p[0] - b[2]])
        if not rows:
            continue
        d = {'n': len(rows), 'valid': len(e_tight),
             'tight_abs_err_median': np.median(np.abs(e_tight), 0).round(1).tolist() if e_tight else None,
             'tight_err_median': np.median(e_tight, 0).round(1).tolist() if e_tight else None}
        if e_cell:
            d['cell_abs_err_median'] = np.median(np.abs(e_cell), 0).round(1).tolist()
        if d_point:
            dp = np.array(d_point)
            d['point_minus_bbox_centre_median'] = np.median(dp[:, :2], 0).round(1).tolist()
            d['point_minus_bbox_xmax_median'] = float(np.median(dp[:, 2]).round(1))
            d['point_minus_bbox_centre_abs_median'] = np.median(np.abs(dp[:, :2]), 0).round(1).tolist()
        out[fam] = d
    return out


def rowcol(G: dict) -> dict:
    out = {}
    for model in ('base', 'early1_4_img'):
        for s in (1.0, 0.0):
            rows = [r for k, r in G.items() if k[0] == 'grid10' and k[2] == model and k[3] == 'rowcol' and k[4] == s]
            ok, err = 0, []
            for r in rows:
                m = re.search(r'"row"\s*:\s*(\d+)\s*,\s*"column"\s*:\s*(\d+)', r['text'])
                if not m:
                    continue
                rr = int(m.group(1))
                ok += rr == r['truth']['row']
                err.append(rr - r['truth']['row'])
            out[f'{model}_s{int(s)}'] = {'n': len(rows), 'row_correct': ok, 'row_err_mean': boot_mean(err)}
    # point-row shift vs ordinal-row shift per image under s = 0
    for model in ('base', 'early1_4_img'):
        d_ord, d_pt = [], []
        for k, r in G.items():
            if k[0] != 'grid10' or k[2] != model or k[3] != 'rowcol' or k[4] != 1.0:
                continue
            r0 = G.get(('grid10', k[1], model, 'rowcol', 0.0, 'clean'))
            p1, p0 = G.get(('grid10', k[1], model, 'point', 1.0, 'clean')), G.get(('grid10', k[1], model, 'point', 0.0, 'clean'))
            m1 = re.search(r'"row"\s*:\s*(\d+)', r['text'])
            m0 = re.search(r'"row"\s*:\s*(\d+)', r0['text']) if r0 else None
            if not (m1 and m0 and point_of(p1) and point_of(p0)):
                continue
            d_ord.append(int(m0.group(1)) - int(m1.group(1)))
            d_pt.append((point_of(p0)[1] - point_of(p1)[1]) / 100)
        out[f'{model}_rowshift'] = {'n': len(d_ord), 'ordinal': boot_mean(d_ord), 'point_row': boot_mean(d_pt),
                                    'point_row_when_ordinal_same': boot_mean([p for o, p in zip(d_ord, d_pt) if o == 0]),
                                    'n_ordinal_same': int(sum(o == 0 for o in d_ord))}
    return out


def text_mapping(rows: list[dict]) -> dict:
    out = {}
    for n in (8, 10):
        rs = [r for r in rows if r['family'] == f'text{n}']
        cls = []
        for r in rs:
            p = cmn.parse_point(r['text'])
            r_, c_ = (int(v) for v in re.match(r'text\d+_r(\d+)c(\d+)', r['case']).groups())
            if p is not None and max(p) <= 1.0:            # answers written on a 0-1 scale
                p = [1000 * v for v in p]
            if p is None:
                cls.append('invalid')
                continue
            w = 1000 / n
            fx = (p[0] - w * (c_ - 1)) / w
            fy = (p[1] - w * (r_ - 1)) / w
            cls.append(cmn.position_class(fx) + '|' + cmn.position_class(fy))
        out[f'text{n}'] = {'n': len(rs), 'classes': {c: cls.count(c) for c in sorted(set(cls))}, 'x_centre': sum(c.split('|')[0] == 'centre' for c in cls),
                           'x_upper_edge': sum(c.split('|')[0] == 'upper_edge' for c in cls),
                           'x_lower_edge': sum(c.split('|')[0] == 'lower_edge' for c in cls),
                           'examples': [r['text'][:60] for r in rs[:3]]}
    return out


# ----------------------------------------------------------------------------- CoT taxonomy

AVG_RE = re.compile(r'\(\s*\d{1,4}\s*\+\s*\d{1,4}\s*\)\s*/\s*2')
EDGE_RE = re.compile(r'(x_min|x_max|y_min|y_max|left|right|top|bottom|leftmost|rightmost|topmost|bottommost)'
                     r'[^\n]{0,80}?\b\d{1,4}\b', re.I)


def cot_style(text: str) -> str:
    """Rule coding of a RefCOCO CoT: edges_average (writes box edges then (a + b) / 2), edges_only (edges, no
    averaging), centre_direct (writes a centre estimate only), none (no coordinate number before the answer)."""
    body = text[:text.rfind('point_2d')] if 'point_2d' in text else text
    nums = [int(m) for m in re.findall(r'(?<![\d.])(\d{2,4})(?![\d.])', body) if 10 <= int(m) <= 1000]
    if not nums:
        return 'none'
    edges = len(EDGE_RE.findall(body))
    if AVG_RE.search(body):
        return 'edges_average'
    if edges >= 3:
        return 'edges_only'
    return 'centre_direct'


def cot_summary(run: Path, work: Path) -> dict:
    out = {}
    r1 = [r for r in read_jsonl(work / R1 / 'refcoco_gen.jsonl') if r['mode'] == 'think']
    for name, rows in (('r1_refcoco_160', r1), ('r2_crop_pairs', read_jsonl(run / 'think.jsonl'))):
        styles = [cot_style(r['text']) for r in rows]
        copies, n_pairs = 0, 0
        for r in rows:
            t = r['text']
            fin = cmn.parse_point(t[t.rfind('point_2d') - 5:]) if 'point_2d' in t else None
            body = t[:t.rfind('point_2d')] if 'point_2d' in t else t
            pairs = re.findall(r'[\[(]\s*(\d{1,4})\s*,\s*(\d{1,4})\s*[\])]', body)
            if fin and pairs:
                n_pairs += 1
                copies += [float(pairs[-1][0]), float(pairs[-1][1])] == fin
        out[name] = {'n': len(rows), 'styles': {s: styles.count(s) for s in ('edges_average', 'edges_only',
                                                                              'centre_direct', 'none')},
                     'answer_equals_last_written_pair': copies, 'n_with_written_pair': n_pairs}
    return out


# ----------------------------------------------------------------------------- flow

def flow_groups(rows: list[dict]) -> dict:
    """(family, role group, digit) -> {'step': [(lp list per pair)], 'win': ..., 'tail': ...}."""
    G: dict = {}
    for r in rows:
        fam = r['family']
        for v in r['values']:
            role = v['role']
            grp = {'x': 'value', 'y': 'value', 'x_min': 'value', 'x_max': 'value', 'cot_first': 'cot_first',
                   'final_x': 'final_x'}[role]
            G.setdefault((fam, grp, v['digit']), []).append(v)
    return G


def recov_table(vals: list[dict], part: str) -> dict:
    conds = vals[0][part]['conds']
    L = np.array([v[part]['lp'] for v in vals])           # (pairs, conds, 10)
    a, b = conds.index('A'), conds.index('B')
    out = {}
    for j, c in enumerate(conds):
        if c in ('A', 'B'):
            continue
        out[c] = {'r': cmn.recovery(L[:, a], L[:, b], L[:, j]), 'ci': cmn.recovery_ci(L[:, a], L[:, b], L[:, j])}
    cen = L - L.mean(-1, keepdims=True)
    out['_norm_BA'] = float(np.linalg.norm(cen[:, b] - cen[:, a], axis=-1).mean())
    out['_n'] = len(vals)
    return out


def tail_table(vals: list[dict]) -> dict:
    by: dict = {}
    for v in vals:
        for s, lp in zip(v['tail']['starts'], v['tail']['lp']):
            by.setdefault(s, []).append(lp)
    out = {}
    for s, L in sorted(by.items()):
        L = np.array(L)
        out[int(s)] = {'r': cmn.recovery(L[:, 0], L[:, 1], L[:, 2]), 'ci': cmn.recovery_ci(L[:, 0], L[:, 1], L[:, 2]),
                       'n': len(L)}
    return out


def flow_summary(rows: list[dict]) -> dict:
    out = {}
    for (fam, grp, dgt), vals in sorted(flow_groups(rows).items()):
        out[f'{fam}|{grp}|{dgt}'] = {'step': recov_table(vals, 'step'), 'win': recov_table(vals, 'win'),
                                     'tail': tail_table(vals)}
    return out


# ----------------------------------------------------------------------------- timeline

def timeline_summary(rows: list[dict]) -> dict:
    """Per (family/fmt, step class): mean image attention share per layer, KL of single-step / cumulative image
    hiding, and argmax change rate. Step classes are offsets relative to each value's first-digit step."""
    acc: dict = {}
    for r in rows:
        key = r['family'] if r['fmt'] == 'point' else f"{r['family']}_{r['fmt']}"
        vals = r['values']
        starts = [v['start'] for v in vals]
        for st in r['steps']:
            g = st['g']
            rel = None
            for s in starts:
                if -4 <= g - s <= 3:
                    rel = g - s
                    break
            cls = 'other' if rel is None else f'{rel:+d}'
            a = acc.setdefault((key, cls), {'share': [], 'kl': [], 'flip': [], 'lp_drop': []})
            a['share'].append(np.array(st['share']).mean(1))          # (8,) head-mean per layer
            a['kl'].append(st['kl'])
            a['flip'].append([int(x != st['argmax'][0]) for x in st['argmax']])
            a['lp_drop'].append([st['logp'][0] - x for x in st['logp']])
    out = {}
    for (key, cls), a in acc.items():
        out.setdefault(key, {})[cls] = {'n': len(a['share']), 'share': np.mean(a['share'], 0).round(3).tolist(),
                                        'kl': np.mean(a['kl'], 0).round(4).tolist(),
                                        'flip': np.mean(a['flip'], 0).round(3).tolist(),
                                        'lp_drop': np.mean(a['lp_drop'], 0).round(3).tolist()}
    return out


def edge_maps(rows: list[dict]) -> dict:
    """Attention of chosen heads at the first-digit step of x / x_min / x_max: mass inside the target box and the
    box-relative position of the attention peak along the value's axis."""
    heads = {'L20H11': (20, 11), 'L24H5': (24, 5), 'L24H7': (24, 7), 'L20H10': (20, 10), 'L24H9': (24, 9),
             'L16H1': (16, 1), 'L28H0': (28, 0)}
    out: dict = {}
    for r in rows:
        key = r['family'] if r['fmt'] == 'point' else f"{r['family']}_{r['fmt']}"
        W, H = r['truth']['canvas']
        gw, gh = W // 32, H // 32
        bu = r['truth']['box_units']
        bx = [bu[0] * gw / 1000, bu[1] * gh / 1000, bu[2] * gw / 1000, bu[3] * gh / 1000]
        for v in r['values']:
            if v['role'] not in ('x', 'y', 'x_min', 'x_max'):
                continue
            st = next((s for s in r['steps'] if s['g'] == v['start']), None)
            if st is None or 'maps' not in st:
                continue
            for name, (l, h) in heads.items():
                m = np.array(st['maps'][str(l)] if str(l) in st['maps'] else st['maps'][l])[h].reshape(gh, gw)
                tot = m.sum()
                if tot <= 0:
                    continue
                yy, xx = np.mgrid[0:gh, 0:gw] + 0.5
                inside = ((xx >= bx[0] - 0.5) & (xx <= bx[2] + 0.5) & (yy >= bx[1] - 0.5) & (yy <= bx[3] + 0.5))
                py, px = np.unravel_index(m.argmax(), m.shape)
                a = 0 if v['role'] in ('x', 'x_min', 'x_max') else 1
                lo, hi = (bx[0], bx[2]) if a == 0 else (bx[1], bx[3])
                f = ((px + 0.5) - lo) / max(hi - lo, 1e-6) if a == 0 else ((py + 0.5) - lo) / max(hi - lo, 1e-6)
                d = out.setdefault(key, {}).setdefault(f"{v['role']}|{name}", {'inside': [], 'peak_rel': [], 'share': []})
                d['inside'].append(float(m[inside].sum() / tot))
                d['peak_rel'].append(float(f))
                d['share'].append(float(tot))
    return {k: {kk: {'n': len(vv['inside']), 'inside': float(np.mean(vv['inside'])),
                     'peak_rel_median': float(np.median(vv['peak_rel'])), 'image_share': float(np.mean(vv['share']))}
                for kk, vv in d.items()} for k, d in out.items()}


# ----------------------------------------------------------------------------- figures

def fig_value_structure(vs: dict, vs_t: dict, path: Path) -> None:
    """Small multiples: x position of direct point answers inside the target cell (grid families) or the target
    box (RefCOCO), base model (blue, filled) vs the located LoRA of 3.13 r2 (orange line)."""
    panels = [f for f in ('grid10', 'grid8', 'thumb_same', 'thumb_mixed', 'refcoco') if f in vs]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.1 * len(panels), 5.4), sharey='row', squeeze=False)
    bins = np.linspace(-0.25, 1.25, 25)
    for j, f in enumerate(panels):
        for i, (a, lab) in enumerate(((0, 'x'), (1, 'y'))):
            ax = axes[i][j]
            key = 'cell_rel' if 'cell_rel' in vs[f] else 'box_rel'
            v = np.clip(np.array(vs[f][key])[:, a], -0.25, 1.25)
            ax.hist(v, bins, weights=np.ones(len(v)) / len(v), color=C['blue'], alpha=0.85, label=f'base (n={len(v)})')
            if f in vs_t and key in vs_t[f]:
                vt = np.clip(np.array(vs_t[f][key])[:, a], -0.25, 1.25)
                h, _ = np.histogram(vt, bins)
                ax.step(bins[:-1], h / len(vt), where='post', color=C['orange'], lw=1.8, label=f'located LoRA (n={len(vt)})')
            for xv in (0, 0.5, 1):
                ax.axvline(xv, color=C['gray'], lw=0.7, ls=':')
            ax.set_xlabel(f"{lab} inside target {'cell' if key == 'cell_rel' else 'box'}", fontsize=8)
            if i == 0:
                ax.set_title(FAM_LABEL[f], fontsize=9, loc='left')
            style(ax)
        axes[0][j].legend(fontsize=7, frameon=False, loc='upper left')
    axes[0][0].set_ylabel('share of answers')
    axes[1][0].set_ylabel('share of answers')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_count_tests(gate: dict, dell: dict, gate_t: dict, path: Path, hits: dict | None = None) -> None:
    """(a) y and (b) x shift of the point answer under alpha -> alpha^0 at image tokens of DeltaNet layers 9-16
    (0-1000 units; base blue, located LoRA orange); (c) x shift after erasing the raster-preceding entity."""
    fams = [f for f in FAM_ORDER if f in gate]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    x = np.arange(len(fams))
    for ax, key, title in ((axes[0], 'dy', '(a) decay alpha^0 at image tokens, L9-16: y shift'),
                           (axes[1], 'dx', '(b) decay alpha^0: x shift')):
        for off, src, col, lab in ((-0.18, gate, C['blue'], 'base'), (0.18, gate_t, C['orange'], 'located LoRA')):
            fs = [f for f in fams if f in src]
            xi = np.array([fams.index(f) for f in fs])
            m = np.array([src[f][key][0] for f in fs])
            lo = m - np.array([src[f][key][1] for f in fs])
            hi = np.array([src[f][key][2] for f in fs]) - m
            ax.bar(xi + off, m, 0.34, color=col, yerr=[lo, hi], capsize=3, ecolor=C['dark'], label=lab)
        ax.axhline(0, color=C['dark'], lw=0.8)
        ax.set_xticks(x, [FAM_LABEL[f] for f in fams], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('mean shift (0-1000 units)')
        ax.set_title(title, fontsize=10, loc='left')
        style(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fd = [f for f in FAM_ORDER if f in dell]
    xd = np.arange(len(fd))
    m = np.array([dell[f]['dx_norm'][0] for f in fd])
    lo = m - np.array([dell[f]['dx_norm'][1] for f in fd])
    hi = np.array([dell[f]['dx_norm'][2] for f in fd]) - m
    axes[2].bar(xd, m, 0.6, color=C['blue'], yerr=[lo, hi], capsize=3, ecolor=C['dark'])
    axes[2].axhline(0, color=C['dark'], lw=0.8)
    axes[2].axhline(-1, color=C['gray'], lw=0.8, ls=':')
    axes[2].text(len(fd) - 0.5, -0.97, 'count prediction (-1 cell)', fontsize=7, ha='right', va='bottom', color=C['dark'])
    axes[2].set_xticks(xd, [FAM_LABEL[f] for f in fd], rotation=30, ha='right', fontsize=8)
    axes[2].set_ylabel('mean x shift (cell widths; box widths for scatter)')
    axes[2].set_title('(c) erase the raster-preceding entity (base)', fontsize=10, loc='left')
    style(axes[2])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


FLOW_FAMS = (('disc', 'value', 'disc (single object)'), ('grid10', 'value', '10x10 grid'),
             ('refcoco_point', 'value', 'RefCOCO point'), ('refcoco_bbox', 'value', 'RefCOCO bbox x_min/x_max'),
             ('refcoco_cot', 'cot_first', 'RefCOCO CoT: first written number'),
             ('refcoco_cot', 'final_x', 'RefCOCO CoT: final answer x'))
STEP_SHOW = (('img', 'image KV at this step'), ('img@20', 'image KV, L20 only'), ('conv1', 'prev. token conv slot'),
             ('conv1@25-32', 'prev. token conv slot, L25-32'), ('rec', 'DeltaNet recurrent state'),
             ('ans_kv', 'earlier answer-token KV'), ('prev_all', 'all earlier-token states'),
             ('img+prev_all', 'image KV + earlier-token states'))


def fig_flow(fs: dict, path: Path) -> None:
    fams = [f for f in FLOW_FAMS if any(k.startswith(f'{f[0]}|{f[1]}|') for k in fs)]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.3 * len(fams), 4.2), sharey=True, squeeze=False)
    cols = [C['blue'], C['orange'], C['aqua']]
    for ax, (fam, grp, title) in zip(axes[0], fams):
        y = np.arange(len(STEP_SHOW))
        for d in range(3):
            k = f'{fam}|{grp}|{d}'
            if k not in fs:
                continue
            t = fs[k]['step']
            m = [t[c]['r'] if c in t else np.nan for c, _ in STEP_SHOW]
            ax.barh(y + (d - 1) * 0.27, m, 0.25, color=cols[d], label=['hundreds', 'tens', 'units'][d])
        ax.axvline(0, color=C['dark'], lw=0.6)
        ax.axvline(1, color=C['gray'], lw=0.6, ls=':')
        ax.set_yticks(y, [l for _, l in STEP_SHOW], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-0.2, 1.2)
        ax.set_title(title, fontsize=9, loc='left')
        ax.set_xlabel('recovery ratio r')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_tail(fs: dict, path: Path) -> None:
    fams = [f for f in FLOW_FAMS if any(k.startswith(f'{f[0]}|{f[1]}|') for k in fs)]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.4), sharey=True, squeeze=False)
    cols = [C['blue'], C['orange'], C['aqua']]
    for ax, (fam, grp, title) in zip(axes[0], fams):
        for d in range(3):
            k = f'{fam}|{grp}|{d}'
            if k not in fs:
                continue
            t = fs[k]['tail']
            s = sorted(int(v) for v in t)
            r = [t[v]['r'] if v in t else t[str(v)]['r'] for v in s]
            lo = [(t[v] if v in t else t[str(v)])['ci'][0] for v in s]
            hi = [(t[v] if v in t else t[str(v)])['ci'][1] for v in s]
            ax.plot(s, r, '-o', color=cols[d], ms=5, lw=1.8, label=['hundreds', 'tens', 'units'][d])
            ax.fill_between(s, lo, hi, color=cols[d], alpha=0.15, lw=0)
        ax.axhline(0, color=C['dark'], lw=0.6)
        ax.axhline(1, color=C['gray'], lw=0.6, ls=':')
        ax.set_xlabel('first answer position that sees image B\n(0 = step predicting the hundreds digit)', fontsize=8)
        ax.set_title(title, fontsize=9, loc='left')
        style(ax)
    axes[0][0].set_ylabel('recovery ratio r')
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


WIN_SHOW = (('img', 'image KV (window)'), ('img@20', 'image KV, L20'), ('img@24', 'image KV, L24'),
            ('H11', 'L20 H11 output'), ('L16', 'all L16 heads'), ('L20', 'all L20 heads'), ('L24', 'all L24 heads'),
            ('L28', 'all L28 heads'), ('edgeL', 'left/top edge readers'), ('edgeR', 'right/bottom edge readers'),
            ('edgeLR', 'both edge groups'), ('allheads', 'all 128 heads'))


def fig_heads(fs: dict, path: Path) -> None:
    fams = [f for f in FLOW_FAMS[:5] if f'{f[0]}|{f[1]}|0' in fs]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    M = np.array([[fs[f'{f}|{g}|0']['win'].get(c, {'r': np.nan})['r'] for c, _ in WIN_SHOW] for f, g, _ in fams])
    im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(WIN_SHOW)), [l for _, l in WIN_SHOW], rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(fams)), [t for _, _, t in fams], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if M[i, j] < 0.55 else 'black')
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label('recovery ratio r of the hundreds digit')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_timeline(ts: dict, path: Path) -> None:
    keys = [k for k in ('grid10', 'single', 'refcoco', 'refcoco_bbox') if k in ts]
    classes = ['-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3']
    labels = ['-4', '-3', '-2', '-1', 'hundreds', 'tens', 'units', 'separator']
    fig, axes = plt.subplots(2, len(keys), figsize=(3.6 * len(keys), 6.2), squeeze=False)
    for j, k in enumerate(keys):
        M = np.array([ts[k][c]['share'] if c in ts[k] else [np.nan] * 8 for c in classes]).T
        im = axes[0][j].imshow(M, cmap='viridis', vmin=0, vmax=0.8, aspect='auto')
        axes[0][j].set_xticks(range(len(classes)), labels, rotation=45, ha='right', fontsize=7)
        axes[0][j].set_yticks(range(8), [f'L{l}' for l in FULL_LAYERS], fontsize=7)
        axes[0][j].set_title(f'{k}: image attention share', fontsize=9, loc='left')
        fig.colorbar(im, ax=axes[0][j], fraction=0.05).set_label('share of attention on image tokens', fontsize=7)
        ax = axes[1][j]
        for ri, (name, col) in enumerate((('m_all', C['blue']), ('m_L20', C['orange']), ('m_L16-24', C['aqua']),
                                          ('m_cum', C['magenta']))):
            v = [ts[k][c]['kl'][ri + 1] if c in ts[k] else np.nan for c in classes]
            ax.plot(range(len(classes)), v, '-o', color=col, ms=4, lw=1.6, label=name)
        ax.set_yscale('symlog', linthresh=0.01)
        ax.set_xticks(range(len(classes)), labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('KL(clean || image hidden), nats', fontsize=8)
        style(ax)
    axes[1][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_schematic(fs: dict, path: Path, fam: str = 'refcoco_point') -> None:
    """Information-flow diagram for one three-digit value (schematic; arrow labels are measured recovery ratios of
    the flow stage for `fam`)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    g = lambda d, part, c: fs.get(f'{fam}|value|{d}', {}).get(part, {}).get(c, {}).get('r', float('nan'))
    # tail tables are keyed by int offsets in memory and by str offsets after a JSON round trip
    tail = lambda d, o: next((t[k]['r'] for t in [fs.get(f'{fam}|value|{d}', {}).get('tail', {})]
                              for k in (o, str(o)) if k in t), float('nan'))
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.9)
    ax.axis('off')

    def box(x, y, w, h, text, col, fs_=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', fc=col, ec=C['dark'], lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs_)

    def arrow(a, b, text='', col=C['dark'], rad=0.0, lw=1.4, off=(0, 0.12)):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11, color=col, lw=lw,
                                     connectionstyle=f'arc3,rad={rad}'))
        if text:
            ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, fontsize=7.5, ha='center', color=col)
    box(0.2, 0.5, 2.3, 4.8, 'image tokens\n(ViT position + content;\nDeltaNet L1-16 mixing\nalong the raster)', '#dbe8f8')
    box(2.8, 0.5, 1.4, 4.8, 'question\ntokens', '#eeeeee')
    names = ['token\n-3', 'token\n-2', 'token\n-1', 'hundreds\nstep', 'tens\nstep', 'units\nstep']
    xs = [4.7 + 1.35 * i for i in range(6)]
    for x, n in zip(xs, names):
        box(x, 0.5, 1.1, 1.0, n, '#fbe3d6' if 'step' in n else '#f4f4f4')
        box(x, 2.1, 1.1, 1.2, 'L16-24\nread heads', '#fff2cc')
        box(x, 3.9, 1.1, 1.0, 'L25-32\nMLP / conv', '#e2f0e9')
    for i, x in enumerate(xs):
        arrow((x + 0.55, 1.5), (x + 0.55, 2.1))
        arrow((x + 0.55, 3.3), (x + 0.55, 3.9))
        if i < 5:
            lab = ''
            if i == 2:
                lab = f'conv slot {g(0, "step", "conv1"):.2f}'
            if i == 3:
                lab = f'conv slot {g(1, "step", "conv1"):.2f}'
            if i == 4:
                lab = f'conv slot {g(2, "step", "conv1"):.2f}'
            arrow((x + 1.1, 4.4), (xs[i + 1], 4.4), lab, C['green'], off=(0, 0.62))
    for i, x in enumerate(xs[:4]):
        arrow((2.5, 2.9 - 0.05 * i), (x, 2.7), '', C['blue'], rad=-0.15 - 0.03 * i, lw=1.0 + 0.4 * (i == 3))
    for i, x in enumerate(xs[4:]):
        arrow((2.5, 2.4), (x, 2.5), '', C['gray'], rad=-0.25, lw=0.6)
    ax.text(3.6, 6.6, f'hundreds digit: image KV at its own step {g(0, "step", "img"):.2f}; visible from token -3 '
                       f'{tail(0, -3):.2f}; from token -6 {tail(0, -6):.2f}', fontsize=8.5, color=C['blue'])
    ax.text(3.6, 6.25, f'tens digit: image KV at its own step {g(1, "step", "img"):.2f}; previous-token conv slot '
                      f'{g(1, "step", "conv1"):.2f}; all earlier-token states {g(1, "step", "prev_all"):.2f}',
            fontsize=8.5, color=C['green'])
    ax.text(3.6, 5.9, f'units digit: image KV at its own step {g(2, "step", "img"):.2f}; previous-token conv slot '
                       f'{g(2, "step", "conv1"):.2f}', fontsize=8.5, color=C['green'])
    ax.text(0.2, 0.1, 'blue arrows: causal image reads (thick = studied step); grey: attention to the image without '
                      'causal effect; green: relay through the Gated DeltaNet short convolution of layers 25-32',
            fontsize=7.5, color=C['dark'])
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_examples(G: dict, run: Path, path: Path) -> None:
    """One input per family with the target box (green), the base answer (blue +) and the located-LoRA answer
    (orange x); answers are model outputs."""
    import argparse as _ap
    import os
    from analysis_tools.position import coordinate_mechanism_natural_run as cr_run
    args = _ap.Namespace(work=run.parent.parent, data_dir=os.environ['DATA_DIR'], n_items=64, n_refcoco=160)
    cases = cr_run.Cases(args)
    fams = [f for f in FAM_ORDER]
    fig, axs = plt.subplots(2, 4, figsize=(14, 7.6))
    axes = axs.ravel()
    axes[-1].axis('off')
    for ax, f in zip(axes, fams):
        c = cases.family(f, 64)[1]
        img, t = cases.render(c)
        W, H = img.size
        ax.imshow(img)
        b = t['box']
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=C['green'], lw=1.5))
        for model, mk, col in (('base', '+', C['blue']), ('early1_4_img', 'x', C['orange'])):
            p = point_of(G.get((f, c['id'], model, 'point', 1.0, 'clean')))
            if p is not None:
                ax.plot(p[0] * W / 1000, p[1] * H / 1000, mk, color=col, ms=9, mew=1.8, alpha=0.9)
        ax.set_title(f"{FAM_LABEL[f]}\n'{cmn.target_text(c)[:28]}'", fontsize=8)
        ax.axis('off')
    lg = axes[-1]
    lg.plot([], [], '+', color=C['blue'], ms=10, mew=2, label='base model answer')
    lg.plot([], [], 'x', color=C['orange'], ms=10, mew=2, label='located LoRA answer')
    lg.plot([], [], 's', mfc='none', mec=C['green'], ms=10, label='target box (truth)')
    lg.legend(fontsize=10, loc='center', frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--figures', type=Path, required=True)
    args = ap.parse_args()
    run, figs = args.run_dir, args.figures
    figs.mkdir(parents=True, exist_ok=True)
    work = run.parent.parent
    S: dict = {}
    gen = read_jsonl(run / 'gen.jsonl')
    if gen:
        G = gen_index([r for r in gen if not r['family'].startswith('text')])
        S['value_structure'] = {m: value_structure(G, m) for m in ('base', 'early1_4_img')}
        S['gate'] = {m: paired_shift(G, m, 'point', (0.0, 'clean')) for m in ('base', 'early1_4_img')}
        S['delL'] = {m: paired_shift(G, m, 'point', (1.0, 'delL')) for m in ('base', 'early1_4_img')}
        S['bbox'] = bbox_structure(G)
        S['pixel_convention'] = pixel_share(G)
        S['lattice'] = {m: lattice_share(G, m) for m in ('base', 'early1_4_img')}
        S['hits'] = {m: hit_rates(G, m) for m in ('base', 'early1_4_img')}
        S['rowcol'] = rowcol(G)
        S['text'] = text_mapping([r for r in gen if r['family'].startswith('text')])
        fig_value_structure(S['value_structure']['base'], S['value_structure']['early1_4_img'],
                            figs / 'value_structure.png')
        fig_examples(G, run, figs / 'examples.png')
        fig_count_tests(S['gate']['base'], S['delL']['base'], S['gate']['early1_4_img'], figs / 'count_tests.png')
    S['cot'] = cot_summary(run, work)
    flow = read_jsonl(run / 'flow.jsonl')
    if flow:
        S['flow'] = flow_summary(flow)
        fig_flow(S['flow'], figs / 'flow_steps.png')
        fig_tail(S['flow'], figs / 'flow_tail.png')
        fig_heads(S['flow'], figs / 'flow_heads.png')
        fig_schematic(S['flow'], figs / 'flow_schematic.png')
    tl = read_jsonl(run / 'timeline.jsonl')
    if tl:
        S['timeline'] = timeline_summary(tl)
        S['edge_maps'] = edge_maps(tl)
        fig_timeline(S['timeline'], figs / 'timeline.png')
    spot = run / 'spotcheck.json'
    if spot.exists():
        S['spotcheck'] = {k: v for k, v in json.loads(spot.read_text()).items() if k != 'rows'}
    slim = json.loads(json.dumps(S, default=float))
    for m in slim.get('value_structure', {}).values():
        for d in m.values():
            d.pop('box_rel', None)
            d.pop('cell_rel', None)
    (run / 'summary.json').write_text(json.dumps(slim, indent=1))
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in slim.items()}, indent=0)[:2000])


if __name__ == '__main__':
    main()
