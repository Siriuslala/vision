"""CPU analysis and figures of experiment 5.1 (local re-read pilot).

Reads the stage outputs of reread_pilot_run.py, scores every answer, and writes
  <run>/scores_<stage>.jsonl, <run>/results.json, and figures into --figures.
Statistics: Wilson 95 % intervals for proportions, image-level bootstrap intervals for AUROC and
paired AUROC differences, exact McNemar tests for paired binary outcomes with Holm correction over
the pre-specified comparisons of each task cell.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from analysis_tools.reread import reread_pilot as rp

BLUE, ORANGE, AQUA, YELLOW, VIOLET, GREY, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#52514e', '#e34948'
COND_LABEL = {'a': 'a full', 'b': 'b full 2x', 'c1': 'c1 crop 1x', 'c2': 'c2 crop 2x', 'c4': 'c4 crop 4x',
              'd': 'd cache orig-ID', 'e': 'e cache new-ID', 'fc': 'fc rand crop 2x', 'fd': 'fd rand cache',
              'h': 'h cache+crop 2x', 'g3': 'g3', 'g9': 'g9', 'g17': 'g17', 'g25': 'g25', 'gcat': 'gcat'}
PRIMARY = [('c2|1', 'a'), ('d|1', 'c1|1'), ('d|1', 'e|1'), ('d|1', 'fd|1'), ('c2|1', 'fc|1'), ('h|1', 'c2|1')]
B = 1000


def _np(o):
    if hasattr(o, 'item'):
        return o.item()
    raise TypeError(type(o))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def result_rows(path: Path) -> list[dict]:
    """Stage results with duplicates removed: two workers may resume the same stage concurrently, so the
    first record of each (case, condition, format, style, k) is kept; the number dropped is reported."""
    seen, out = set(), []
    rows = read_jsonl(path)
    for r in rows:
        key = (r['case_id'], r['cond'], r.get('format'), r.get('style'), r.get('k'))
        if key not in seen:
            seen.add(key)
            out.append(r)
    if len(out) != len(rows):
        print(f'{path}: dropped {len(rows) - len(out)} duplicate records')
    return out


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def prop(k: int, n: int) -> dict:
    return {'k': int(k), 'n': int(n), 'p': k / n if n else float('nan'), 'ci': rp.wilson(int(k), int(n))}


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for k successes in n trials at p = 0.5."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(min(1.0, sum(q for q in pmf if q <= pmf[k] * (1 + 1e-9))))


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar on paired booleans; b10 = a right & b wrong."""
    b10, b01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binom_two_sided(b10, b10 + b01)
    return {'a_only': b10, 'b_only': b01, 'p': float(p)}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def cost(rows: list[dict]) -> dict:
    return {'L': float(np.mean([r['meta']['L'] for r in rows])),
            'extra_visual': float(np.mean([r['meta']['n_extra_visual'] for r in rows])),
            'visual': float(np.mean([r['meta']['n_visual'] for r in rows])),
            'vit_new': float(np.mean([r['meta']['n_vit_patches_new'] for r in rows])),
            'vit_full': float(np.mean([r['meta']['n_vit_patches_full'] for r in rows]))}


# ----------------------------------------------------------------------------- detail task

def analyze_det(run: Path) -> dict:
    cases = {c['id']: c for c in json.loads((run / 'det/cases.json').read_text())}
    rows = result_rows(run / 'det/results.jsonl')
    for r in rows:
        r['y'] = cases[r['case_id']]['answer_yes']
        r['correct'] = (r['s'] > 0) == r['y']
    out = {}
    rng = np.random.default_rng(0)
    for n in sorted({c['grid_n'] for c in cases.values()}):
        for kind, level in sorted({(c['kind'], c['level']) for c in cases.values()}):
            ids = sorted(i for i, c in cases.items() if c['grid_n'] == n and c['kind'] == kind and c['level'] == level)
            if not ids:
                continue
            y = np.array([cases[i]['answer_yes'] for i in ids])
            boot = [rng.integers(0, len(ids), len(ids)) for _ in range(B)]
            base_rows = {r['case_id']: r for r in rows if r['cond'] == 'a' and r['case_id'] in set(ids)}
            for k in (1, 3):
                cell = {}
                sel = [r for r in rows if r['case_id'] in set(ids) and (r['k'] == k or r['cond'] in ('a', 'b'))]
                conds = sorted({r['cond'] for r in sel}, key=lambda c: list(COND_LABEL).index(c.split('|')[0]) * 2 + int('|1' in c))
                svec = {}
                for cond in conds:
                    by = {r['case_id']: r for r in sel if r['cond'] == cond}
                    if set(by) != set(ids):
                        raise ValueError(f'det {n} {kind}{level} k={k} {cond}: {len(by)}/{len(ids)} rows')
                    s = np.array([by[i]['s'] for i in ids])
                    svec[cond] = s
                    cor = (s > 0) == y
                    cell[cond] = {'auroc': auroc(s, y), 'acc': prop(cor.sum(), len(ids)),
                                  'yes_rate': float((s > 0).mean()), 'cost': cost(list(by.values()))}
                base = svec['a']
                for cond in conds:
                    vals = [auroc(svec[cond][b], y[b]) for b in boot]
                    dvals = [auroc(svec[cond][b], y[b]) - auroc(base[b], y[b]) for b in boot]
                    cell[cond]['auroc_ci'] = [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
                    cell[cond]['d_auroc'] = cell[cond]['auroc'] - cell['a']['auroc']
                    cell[cond]['d_auroc_ci'] = [float(np.nanpercentile(dvals, 2.5)), float(np.nanpercentile(dvals, 97.5))]
                comps = []
                for c1, c2 in PRIMARY:
                    if c1 in svec and c2 in svec:
                        d = [auroc(svec[c1][b], y[b]) - auroc(svec[c2][b], y[b]) for b in boot]
                        comps.append({'pair': [c1, c2], 'd_auroc': cell[c1]['auroc'] - cell[c2]['auroc'],
                                      'ci': [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                                      'p_boot': float(min(1.0, 2 * min(np.mean(np.array(d) <= 0), np.mean(np.array(d) >= 0))))})
                for c, ph in zip(comps, holm([c['p_boot'] for c in comps])):
                    c['p_holm'] = ph
                out[f'{n}|{kind}{level}|k{k}'] = {'conds': cell, 'comparisons': comps, 'n': len(ids)}
    return out


# ----------------------------------------------------------------------------- localization

def score_loc_rows(run: Path, stage: str) -> list[dict]:
    cases = {c['id']: c for c in json.loads((run / stage / 'cases.json').read_text())}
    rows = result_rows(run / stage / 'results.jsonl')
    for r in rows:
        c = cases[r['case_id']]
        n = c['grid_n']
        grid = (2 * n, 2 * n)
        win = rp.Window(*r['window'])
        if r['format'] == 'point':
            sc = rp.score_point(r['text'], r['finish_reason'], n, tuple(c['target']), win, grid)
            sc['correct'] = sc['hit']
        else:
            sc = rp.score_rowcol(r['text'], r['finish_reason'], n, tuple(c['target']), win, grid,
                                 last=r['style'] == 'think')
        r.update(task=c['task'], grid_n=n, score=sc)
    return rows


def summarize_binary(rows: list[dict], keys: list[str], cond_key: str = 'cond', require_base: bool = True) -> dict:
    """Group rows by keys + cond; report correct / in_window / frame_error / valid proportions and paired tests."""
    out = {}
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    for g, rs in groups.items():
        conds = sorted({r[cond_key] for r in rs}, key=lambda c: list(COND_LABEL).index(c.split('|')[0]) * 2 + int('|1' in c))
        by = {c: {r['case_id']: r for r in rs if r[cond_key] == c} for c in conds}
        ids = sorted(by['a']) if require_base else sorted({r['case_id'] for r in rs})
        cell = {}
        for c in conds:
            sub = [by[c][i] for i in ids if i in by[c]]
            m = len(sub)
            cell[c] = {'correct': prop(sum(r['score']['correct'] for r in sub), m),
                       'in_window': prop(sum(r['score'].get('in_window', False) for r in sub), m),
                       'frame_error': prop(sum(r['score'].get('frame_error', False) for r in sub), m),
                       'valid': prop(sum(r['score']['valid'] for r in sub), m),
                       'cost': cost(sub)}
            for key in ('hit_any', 'hit_tol', 'in_window_tol', 'frame_error_tol', 'valid_tol'):
                if key in sub[0]['score']:
                    cell[c][key] = prop(sum(r['score'][key] for r in sub), m)
            if 'center_in' in sub[0]['score']:
                cell[c]['center_in'] = prop(sum(r['score']['center_in'] for r in sub), m)
        comps = []
        for c1, c2 in PRIMARY + [('d|1', 'a'), ('e|1', 'a'), ('h|1', 'a'), ('b', 'a')]:
            if c1 in by and c2 in by:
                common = [i for i in ids if i in by[c1] and i in by[c2]]
                a = np.array([by[c1][i]['score']['correct'] for i in common])
                b = np.array([by[c2][i]['score']['correct'] for i in common])
                comps.append({'pair': [c1, c2], 'n': len(common), 'diff': float(a.mean() - b.mean()), **mcnemar(a, b)})
        primary = [i for i, c in enumerate(comps) if tuple(c['pair']) in PRIMARY]
        for i, ph in zip(primary, holm([comps[i]['p'] for i in primary])):
            comps[i]['p_holm'] = ph
        out['|'.join(map(str, g))] = {'conds': cell, 'comparisons': comps, 'n': len(ids)}
    return out


def score_ref_rows(run: Path) -> list[dict]:
    rows = result_rows(run / 'refcoco/results.jsonl')
    for r in rows:
        grid = tuple(r['grid'])
        r['score'] = rp.score_bbox(r['text'], r['finish_reason'], r['truth_norm'], rp.Window(*r['window']), grid)
        r['task'] = 'refcoco'
    return rows


def score_bv_rows(run: Path) -> list[dict]:
    rows = result_rows(run / 'babyvision/results.jsonl')
    for r in rows:
        sc = {'valid': False, 'correct': False, 'frame_error': False, 'in_window': False}
        if r['finish_reason'] == 'eos':
            rc = rp.parse_rowcol(r['text'])
            if rc is not None:
                pr, pc = rc[0] - 1, rc[1] - 1
                tr, tc = r['target_rc']
                r0, c0, k = r['cells']
                sc.update(valid=True, pred=[pr, pc], correct=(pr, pc) == (tr, tc),
                          in_window=r0 <= pr < r0 + k and c0 <= pc < c0 + k)
                sc['frame_error'] = not sc['correct'] and (pr, pc) == (tr - r0, tc - c0)
        r['score'] = sc
        r['task'] = 'babyvision'
    return rows


def trivial_ref(run: Path) -> dict:
    cases = json.loads((run / 'refcoco/cases.json').read_text())
    hits = [rp.box_iou(rp.trivial_box(rp.Window(*c['window']), tuple(c['grid'])), c['truth_norm']) >= 0.5 for c in cases]
    return prop(sum(hits), len(hits))


# ----------------------------------------------------------------------------- figures

def order_key(c):
    return list(COND_LABEL).index(c.split('|')[0]) * 2 + int('|1' in c)


def dot_panel(ax, cell: dict, metric: str, title: str, base_line: float | None = None, ylabel: str = ''):
    names = sorted({c.split('|')[0] for c in cell}, key=lambda c: list(COND_LABEL).index(c))
    xs = {n: i for i, n in enumerate(names)}
    for c, v in cell.items():
        name, _, h = c.partition('|')
        val = v[metric]['p'] if isinstance(v[metric], dict) else v[metric]
        ci = v[metric]['ci'] if isinstance(v[metric], dict) else v.get(f'{metric}_ci')
        off = {'': 0, '0': -0.15, '1': 0.15}[h]
        col = {'': GREY, '0': BLUE, '1': ORANGE}[h]
        ax.errorbar(xs[name] + off, val, yerr=None if ci is None else [[val - ci[0]], [ci[1] - val]],
                    fmt='o', color=col, ms=6, capsize=2, lw=1.2)
    if base_line is not None:
        ax.axhline(base_line, color=GREY, lw=0.8, ls='--')
    ax.set_xticks(range(len(names)), [COND_LABEL[n] for n in names], rotation=55, ha='right', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color='#e5e5e5', lw=0.6)
    ax.set_axisbelow(True)


def legend_hint(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', color=c, ls='', label=l) for c, l in
         ((GREY, 'no extra input / full image only'), (BLUE, 'extra input, no region hint'), (ORANGE, 'extra input + region hint'))]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=9)


def fig_det(det: dict, figs: Path):
    keys = [k for k in det if k.endswith('k3')]
    grids = sorted({int(k.split('|')[0]) for k in keys})
    kinds = sorted({k.split('|')[1] for k in keys}, key=lambda s: ('gap' not in s, s))
    names = {'gap1': 'gap 2 px', 'color2': 'colour ΔE 10', 'color3': 'colour ΔE 20'}
    for k in (3, 1):
        fig, axes = plt.subplots(len(grids), len(kinds), figsize=(4.6 * len(kinds), 3.8 * len(grids)), squeeze=False)
        for i, n in enumerate(grids):
            for j, kd in enumerate(kinds):
                cell = det[f'{n}|{kd}|k{k}']['conds']
                if k == 1:
                    cell = {c: v for c, v in cell.items() if not c.startswith('g')}
                dot_panel(axes[i, j], cell, 'auroc', f'{n}x{n} grid, {names.get(kd, kd)}, k={k}',
                          cell['a']['auroc'], 'AUROC of s = z_Yes - z_No')
                axes[i, j].set_ylim(0.3, 1.02)
        legend_hint(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(figs / f'det_auroc_k{k}.png', dpi=130)
        plt.close(fig)


def fig_scan(det: dict, figs: Path):
    keys = sorted(k for k in det if k.endswith('k3'))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    order = ['g3|1', 'g9|1', 'g17|1', 'g25|1', 'd|1', 'gcat|1', 'c1|1']
    labels = ['ViT 3', 'ViT 9', 'ViT 17', 'ViT 25', 'ViT 27 (d)', 'gcat', 'c1 re-encode']
    cols = [BLUE, ORANGE, AQUA, VIOLET, RED, YELLOW]
    for j, key in enumerate(keys):
        cell = det[key]['conds']
        ax.plot(range(len(order)), [cell[c]['auroc'] for c in order], marker='o', color=cols[j % len(cols)],
                lw=1.5, label='{0}x{0}, {1}'.format(key.split('|')[0], {'gap1': 'gap 2 px', 'color2': 'colour ΔE 10',
                                                                     'color3': 'colour ΔE 20'}[key.split('|')[1]]))
    ax.set_xticks(range(len(order)), labels, fontsize=9)
    ax.set_ylabel('AUROC (k=3, region hint)')
    ax.set_title('Feature source of the re-read window tokens (merger projection of ViT block)', fontsize=10)
    ax.grid(axis='y', color='#e5e5e5', lw=0.6)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figs / 'det_feature_scan.png', dpi=130)
    plt.close(fig)


def fig_loc(loc: dict, figs: Path, name: str, metric_label: str, extra_line: dict | None = None):
    keys = sorted(loc)
    ncol = min(4, len(keys))
    nrow = math.ceil(len(keys) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        cell = loc[key]['conds']
        metric = 'hit_tol' if 'point' in key else 'correct'      # point: format-tolerant parse, 0-1000 only
        dot_panel(ax, cell, metric, key.replace('|', ' / ') + (' (format-tolerant)' if metric == 'hit_tol' else ''),
                  cell['a'][metric]['p'], metric_label)
        if extra_line and key in extra_line:
            ax.axhline(extra_line[key], color=AQUA, lw=1, ls=':')
        ax.set_ylim(-0.03, 1.03)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis('off')
    legend_hint(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figs / f'{name}.png', dpi=130)
    plt.close(fig)


def fig_cost(summary: dict, figs: Path):
    """Accuracy gain over the full-image baseline vs extra visual tokens and new ViT patches."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    markers = {'c1': 's', 'c2': 'D', 'c4': 'P', 'd': 'o', 'e': 'v', 'h': '*', 'b': 'X'}
    colors = {}
    family = {'det': (BLUE, 'detail judgement (AUROC)'), 'loc': (ORANGE, 'synthetic localization (accuracy)'),
              'refcoco': (AQUA, 'RefCOCO (IoU >= 0.5 rate)')}
    for task, cell in summary.items():
        col, lab = family[task.split()[0]]
        colors[lab] = col
        base = cell['a']
        for c, v in cell.items():
            name = c.split('|')[0]
            if name not in markers or (c.endswith('|0')):
                continue
            gain = v['metric'] - base['metric']
            axes[0].scatter(v['cost']['extra_visual'] + (v['cost']['visual'] - base['cost']['visual'] if name == 'b' else 0),
                            gain, marker=markers[name], color=col, s=50, edgecolor='white', lw=0.5)
            axes[1].scatter(v['cost']['vit_new'], gain, marker=markers[name], color=col, s=50,
                            edgecolor='white', lw=0.5)
    for ax, xl in zip(axes, ('extra visual tokens given to the LLM (mean per prompt)', 'newly ViT-encoded patches (mean)')):
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xscale('symlog', linthresh=10)
        ax.set_xlim(left=-1)
        ax.set_xlabel(xl)
        ax.set_ylabel('gain over full-image baseline (a)')
        ax.grid(color='#eeeeee', lw=0.6)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=m, color=GREY, ls='', label=COND_LABEL[n]) for n, m in markers.items()]
    h += [Line2D([], [], marker='o', color=c, ls='', label=t) for t, c in colors.items()]
    fig.legend(handles=h, loc='center right', fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(figs / 'cost_gain.png', dpi=130)
    plt.close(fig)


def draw_window(ax, win: rp.Window, color: str, ls='-', scale: float = 1.0, label=None):
    x0, y0, x1, y1 = [v * scale for v in win.pixel_box()]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.8, ls=ls, label=label))


def fig_loc_cases(run: Path, rows: list[dict], figs: Path, name: str, picks: list[tuple[str, str]]):
    """Original image with oracle / random windows and point predictions of a, c2|1, d|1, e|1."""
    cases = {c['id']: c for c in json.loads((run / 'loc/cases.json').read_text())}
    show = [('a', GREY, 'o'), ('c2|1', ORANGE, 'D'), ('d|1', BLUE, 's'), ('e|1', VIOLET, 'v')]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.4 * len(picks), 4.8), squeeze=False)
    for ax, (cid, why) in zip(axes[0], picks):
        c = cases[cid]
        image, meta = rp.synth_image(c)
        ax.imshow(image)
        rs = {r['cond']: r for r in rows if r['case_id'] == cid and r['format'] == 'point'}
        win = rp.Window(*rs['a']['window'])
        draw_window(ax, win, BLUE, label='oracle window')
        draw_window(ax, rp.Window(*rs['a']['rwindow']), ORANGE, '--', label='random window')
        inst = meta['instances'][c['target'][0] * c['grid_n'] + c['target'][1]]
        x0, y0, x1, y1 = inst['cell_box']
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=AQUA, lw=2, label='target cell'))
        W = image.width
        for cond, col, m in show:
            sc = rs[cond]['score']
            if sc.get('pred'):
                ax.scatter(sc['pred'][0] / 1000 * W, sc['pred'][1] / 1000 * W, marker=m, s=70, color=col,
                           edgecolor='black', lw=0.6, zorder=5, label=f"{COND_LABEL[cond.split('|')[0]]}" +
                           (' +hint' if cond.endswith('|1') else '') + (' (hit)' if sc['hit'] else ''))
        ax.set_title(f"{c['task']} {c['grid_n']}x{c['grid_n']}: {why}", fontsize=9)
        ax.axis('off')
        ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / f'{name}.png', dpi=130)
    plt.close(fig)


def fig_ref_cases(refcoco: Path, rows: list[dict], figs: Path, picks: list[tuple[str, str]]):
    show = [('a', GREY), ('c2|1', ORANGE), ('d|1', BLUE), ('e|1', VIOLET)]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.6 * len(picks), 4.9), squeeze=False)
    for ax, (cid, why) in zip(axes[0], picks):
        rs = {r['cond']: r for r in rows if r['case_id'] == cid}
        r0 = rs['a']
        image = rp.resize32(Image.open(refcoco / 'refcoco_images' / r0['image']).convert('RGB'))
        ax.imshow(image)
        W, H = image.size
        draw_window(ax, rp.Window(*r0['window']), BLUE, label='oracle window')
        t = r0['truth_norm']
        ax.add_patch(Rectangle((t[0] * W / 1000, t[1] * H / 1000), (t[2] - t[0]) * W / 1000, (t[3] - t[1]) * H / 1000,
                               fill=False, ec=AQUA, lw=2.2, label='ground truth'))
        for cond, col in show:
            sc = rs[cond]['score']
            if sc.get('pred'):
                p = sc['pred']
                ax.add_patch(Rectangle((p[0] * W / 1000, p[1] * H / 1000), (p[2] - p[0]) * W / 1000,
                                       (p[3] - p[1]) * H / 1000, fill=False, ec=col, lw=1.4, ls='--',
                                       label=f"{COND_LABEL[cond.split('|')[0]]}{' +hint' if cond.endswith('|1') else ''}"
                                             f" IoU {sc['iou']:.2f}"))
        ax.set_title(f'"{r0["expression"][:40]}" ({why})', fontsize=8.5)
        ax.axis('off')
        ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / 'refcoco_cases.png', dpi=130)
    plt.close(fig)


def fig_inputs(run: Path, refcoco: Path, figs: Path):
    """Input example: original image with windows, the 2x pixel crop, and the cached-token window."""
    det_cases = json.loads((run / 'det/cases.json').read_text())
    det_rows = result_rows(run / 'det/results.jsonl')
    c = next(c for c in det_cases if c['grid_n'] == 20 and c['kind'] == 'gap' and c['answer_yes'])
    r = next(r for r in det_rows if r['case_id'] == c['id'] and r['k'] == 3 and r['cond'] == 'd|1')
    image, _ = rp.synth_image(c)
    win, rwin = rp.Window(*r['window']), rp.Window(*r['rwindow'])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1.6, 1, 1]})
    axes[0].imshow(image)
    draw_window(axes[0], win, BLUE, label='oracle window (k=3)')
    draw_window(axes[0], rwin, ORANGE, '--', label='random window')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_title(f"20x20 grid, gap 2 px; question asks about row {c['asked'][0] + 1}, column {c['asked'][1] + 1}",
                      fontsize=9)
    axes[1].imshow(rp.crop(image, win, 2))
    axes[1].set_title('c2: pixel crop, 2x bicubic, re-encoded\n(12x12 = 144 visual tokens)', fontsize=9)
    axes[2].imshow(rp.crop(image, win, 1))
    for i in range(1, win.h):
        axes[2].axhline(i * 32 - 0.5, color=BLUE, lw=0.5)
        axes[2].axvline(i * 32 - 0.5, color=BLUE, lw=0.5)
    axes[2].set_title('d / e: the 6x6 = 36 cached tokens of the window\n(blue lines: merged-token cells, 32 px)',
                      fontsize=9)
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(figs / 'inputs.png', dpi=130)
    plt.close(fig)


def fig_bv(bv_run: Path, rows: list[dict], figs: Path):
    """BabyVision: image with oracle window, and the row/column answer of each condition."""
    ids = sorted({r['taskId'] for r in rows})
    conds = sorted({r['cond'] for r in rows}, key=order_key)
    mat = np.full((len(ids), len(conds)), np.nan)
    for r in rows:
        mat[ids.index(r['taskId']), conds.index(r['cond'])] = 1.0 if r['score']['correct'] else (
            0.5 if r['score']['in_window'] else 0.0)
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(conds), 0.5 + 0.32 * len(ids)))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#e0e0e0', '#86b6ef', '#1c5cab'])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(conds)), [COND_LABEL[c.split('|')[0]] + (' +hint' if c.endswith('|1') else '') for c in conds],
                  rotation=55, ha='right', fontsize=8)
    ax.set_yticks(range(len(ids)), [f'task {i}' for i in ids], fontsize=8)
    cb = fig.colorbar(im, ax=ax, ticks=[1 / 6, 0.5, 5 / 6])
    cb.ax.set_yticklabels(['wrong, outside window', 'wrong, inside window', 'correct'], fontsize=8)
    ax.set_title('BabyVision Find-the-different (row/column answer)', fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / 'babyvision_matrix.png', dpi=130)
    plt.close(fig)


def fig_bv_cases(bv_run: Path, rows: list[dict], figs: Path, picks: list[int]):
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4.6), squeeze=False)
    for ax, tid in zip(axes[0], picks):
        rs = {r['cond']: r for r in rows if r['taskId'] == tid}
        r0 = rs['a']
        image = Image.open(bv_run / r0['image']).convert('RGB')
        ax.imshow(image)
        draw_window(ax, rp.Window(*r0['window']), BLUE, label='oracle window')
        if r0['rwindow']:
            draw_window(ax, rp.Window(*r0['rwindow']), ORANGE, '--', label='random window')
        lines = [f"truth r{r0['target_rc'][0] + 1} c{r0['target_rc'][1] + 1}"]
        for cond in ('a', 'b', 'c2|1', 'd|1', 'e|1', 'h|1'):
            if cond in rs:
                p = rs[cond]['score'].get('pred')
                lines.append(f"{COND_LABEL[cond.split('|')[0]]}{' +hint' if cond.endswith('|1') else ''}: "
                             + (f'r{p[0] + 1} c{p[1] + 1}' if p else 'invalid') + (' ✓' if rs[cond]['score']['correct'] else ''))
        ax.set_title(f'task {tid}', fontsize=9)
        ax.text(1.01, 0.5, '\n'.join(lines), transform=ax.transAxes, fontsize=7.5, va='center')
        ax.axis('off')
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / 'babyvision_cases.png', dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--bv-run', type=Path, required=True)
    p.add_argument('--refcoco', type=Path, required=True)
    args = p.parse_args()
    run, figs = args.run, args.figures
    figs.mkdir(parents=True, exist_ok=True)
    results = {}
    if (run / 'det/results.jsonl').exists():
        results['det'] = analyze_det(run)
        fig_det(results['det'], figs)
        fig_scan(results['det'], figs)
        fig_inputs(run, args.refcoco, figs)
    if (run / 'loc/results.jsonl').exists():
        loc_rows = score_loc_rows(run, 'loc')
        results['loc'] = summarize_binary(loc_rows, ['task', 'grid_n', 'format'])
        fig_loc(results['loc'], figs, 'loc_accuracy', 'accuracy (hit / exact row-col)')
        with (run / 'scores_loc.jsonl').open('w') as f:
            for r in loc_rows:
                f.write(json.dumps({k: r[k] for k in ('case_id', 'cond', 'format', 'task', 'grid_n', 'score', 'text')}, default=_np) + '\n')
        results['loc_rows'] = loc_rows
        fig_loc_panels(run, loc_rows, figs, 'loc_cases', [('loc_T5_20_00', 'pixel crop answers window-local ordinals'),
                                                           ('loc_T6_20_28', 'only the 2x full image fixes the point')])
    if (run / 'think/results.jsonl').exists():
        think_rows = score_loc_rows(run, 'think')
        results['think'] = summarize_binary(think_rows, ['task', 'grid_n', 'format'])
    if (run / 'think_long/results.jsonl').exists():
        long_rows = score_loc_rows(run, 'think_long')
        results['think_long'] = summarize_binary(long_rows, ['task', 'grid_n', 'format'], require_base=False)
    if (run / 'refcoco/results.jsonl').exists():
        ref_rows = score_ref_rows(run)
        results['refcoco'] = summarize_binary(ref_rows, ['task'])
        results['refcoco_trivial'] = trivial_ref(run)
        fig_loc(results['refcoco'], figs, 'refcoco_accuracy', 'IoU >= 0.5 rate',
                {'refcoco': results['refcoco_trivial']['p']})
        results['ref_rows'] = ref_rows
        sc = {}
        for r in ref_rows:
            sc.setdefault(r['case_id'], {})[r['cond']] = r['score']
        used: set = set()
        def first(pred):
            cid = next((cid for cid, d in sc.items() if cid not in used and pred(d)
                        and all(d[c]['valid'] for c in ('a', 'c2|1', 'd|1', 'e|1'))), None)
            used.add(cid)
            return cid
        picks = [(first(lambda d: d['d|1']['correct'] and not d['a']['correct']), 'cache fixes'),
                 (first(lambda d: d['c2|1']['correct'] and not d['a']['correct'] and not d['d|1']['correct']), 'crop fixes'),
                 (first(lambda d: d['a']['correct'] and not d['c2|1']['correct']), 'crop breaks'),
                 (first(lambda d: not any(d[c]['correct'] for c in ('a', 'b', 'c2|1', 'd|1', 'e|1'))), 'all fail')]
        fig_ref_cases(args.refcoco, ref_rows, figs, [p for p in picks if p[0] is not None])
        results['refcoco_case_picks'] = picks
    if (run / 'babyvision/results.jsonl').exists():
        bv_rows = score_bv_rows(run)
        results['babyvision'] = summarize_binary(bv_rows, ['task'])
        fig_bv(args.bv_run, bv_rows, figs)
        bsc = {}
        for r in bv_rows:
            bsc.setdefault(r['taskId'], {})[r['cond']] = r['score']['correct']
        bpicks = [next((t for t, d in sorted(bsc.items()) if d['b'] and not d['a']), None),
                  next((t for t, d in sorted(bsc.items()) if d.get('c2|0') and not d['a']), None),
                  next((t for t, d in sorted(bsc.items()) if not any(d.values())), None)]
        fig_bv_cases(args.bv_run, bv_rows, figs, [t for t in bpicks if t is not None])
        results['bv_rows'] = bv_rows
    # cost-gain summary over tasks (hinted conditions)
    summary = {}
    for key, cell in results.get('det', {}).items():
        if key.endswith('k3'):
            summary[f'det {key[:-3]}'] = {c: {'metric': v['auroc'], 'cost': v['cost']} for c, v in cell['conds'].items()}
    for stage in ('loc', 'refcoco'):
        for key, cell in results.get(stage, {}).items():
            m = 'hit_tol' if 'point' in key else 'correct'
            summary[f'{stage} {key}'] = {c: {'metric': v[m]['p'], 'cost': v['cost']} for c, v in cell['conds'].items()}
    if summary:
        fig_cost(summary, figs)
    for k in ('loc_rows', 'ref_rows', 'bv_rows'):
        results.pop(k, None)
    (run / 'results.json').write_text(json.dumps(results, indent=1, default=_np))
    print('wrote', run / 'results.json')



SHOW = [('a', GREY, 'o'), ('b', YELLOW, 'X'), ('c2|1', ORANGE, 'D'), ('d|1', BLUE, 's'), ('e|1', VIOLET, 'v')]


def fig_loc_panels(run: Path, rows: list[dict], figs: Path, name: str, picks: list[tuple[str, str]]):
    """Per case: original image with windows and point answers, the 2x crop, and the row/column answers."""
    cases = {c['id']: c for c in json.loads((run / 'loc/cases.json').read_text())}
    fig, axes = plt.subplots(len(picks), 3, figsize=(13, 4.6 * len(picks)), squeeze=False,
                             gridspec_kw={'width_ratios': [1.5, 0.9, 0.9]})
    for (ax0, ax1, ax2), (cid, why) in zip(axes, picks):
        c = cases[cid]
        image, meta = rp.synth_image(c)
        rs = {(r['cond'], r['format']): r for r in rows if r['case_id'] == cid}
        win = rp.Window(*rs[('a', 'point')]['window'])
        ax0.imshow(image)
        draw_window(ax0, win, BLUE, label='oracle window')
        inst = meta['instances'][c['target'][0] * c['grid_n'] + c['target'][1]]
        x0, y0, x1, y1 = inst['cell_box']
        ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=AQUA, lw=2.2, label='target cell'))
        lines = [f"truth: row {c['target'][0] + 1}, column {c['target'][1] + 1}",
                 f"(row {c['target'][0] - win.r0 // 2 + 1}, column {c['target'][1] - win.c0 // 2 + 1} inside the window)", '']
        for cond, col, m in SHOW:
            sp = rs[(cond, 'point')]['score']
            xy = sp.get('pred') or (None if not sp.get('valid_tol') else None)
            text = rs[(cond, 'point')]['text']
            vals = rp.parse_numbers(text, 'point_2d', 2, diagnostic=True)
            lab = COND_LABEL[cond.split('|')[0]] + (' +hint' if cond.endswith('|1') else '')
            if vals:
                ax0.scatter(vals[0] / 1000 * image.width, vals[1] / 1000 * image.height, marker=m, s=80, color=col,
                            edgecolor='black', lw=0.6, zorder=5, label=lab + (' (hit)' if sp['hit_tol'] else ''))
            sr = rs[(cond, 'rowcol')]['score']
            ans = f"row {sr['pred'][0] + 1}, col {sr['pred'][1] + 1}" if sr.get('pred') else 'invalid'
            tag = 'correct' if sr['correct'] else ('window-local ordinal' if sr['frame_error'] else 'wrong')
            lines.append(f'{lab}: {ans} ({tag})')
        ax0.set_title(f"{c['task']} {c['grid_n']}x{c['grid_n']}: {why}", fontsize=9.5)
        ax0.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False)
        ax1.imshow(rp.crop(image, win, 2))
        ax1.set_title('c2 input: window crop, 2x', fontsize=9)
        ax2.text(0.0, 0.95, 'row/column answers\n\n' + '\n'.join(lines), va='top', fontsize=8.5, family='monospace')
        for ax in (ax0, ax1, ax2):
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(figs / f'{name}.png', dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()
