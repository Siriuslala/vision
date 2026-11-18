"""Experiment 3.14 round 3: statistics and figures of the row-attention analysis (CPU stage).

Results: docs/analysis/position/results_2026-11-05_row_attention.md. Inputs are written by
`analysis_tools.position.row_attention` (maps_<tag>/, order_<tag>/, edits_<tag>.jsonl) and the reading-head
ranking of round 2 (summary.json). Token groups (raster indices on the merged token grid, relative to the
target tokens T):

  T       target tokens
  Ra      row-start tokens: first token column of the target's token rows (T excluded)
  Re      row-end tokens: last token column of the target's token rows (T excluded)
  Rl, Rr  other tokens of the target's token rows left / right of T
  C       target's token columns outside the target rows
  other   all remaining image tokens
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import row_attention as ra

GROUPS = ('T', 'Ra', 'Rl', 'Rr', 'Re', 'C', 'other')
GROUP_NAMES = {'T': '目标', 'Ra': '行首', 'Rl': '行内左侧', 'Rr': '行内右侧', 'Re': '行尾', 'C': '目标所在列',
               'other': '其余'}
COLORS = {'T': '#c0392b', 'Ra': '#2e86c1', 'Rl': '#85c1e9', 'Rr': '#a569bd', 'Re': '#6c3483', 'C': '#e67e22',
          'other': '#bdbdbd'}
MODEL = {'vl': 'Qwen3-VL-8B', 'q35': 'Qwen3.5-9B'}
R2_KEY = {'vl': 'vl_ds_on', 'q35': 'q35_none'}
FMT_OF = {'t6g10_rowcol': 'rowcol', 't6g10_colrow': 'colrow', 't6g10_point': 'point', 'disc_point': 'disc',
          'rc_point': 'refcoco', 'sweep_rowcol': 'sweep_rowcol', 'sweep_point': 'sweep_point'}
AXIS_OF = {'row': 'v', 'y': 'v', 'column': 'h', 'x': 'h'}
plt.rcParams['axes.unicode_minus'] = False


def use_cjk_font() -> None:
    """Register the CJK font named by TOKEN_MAP_FONT (.env) for the Chinese figure labels."""
    import os
    from matplotlib import font_manager
    path = os.environ.get('TOKEN_MAP_FONT')
    if not path or not Path(path).exists():
        raise ValueError('Set TOKEN_MAP_FONT to a font supporting Chinese, e.g. Noto Sans CJK.')
    font_manager.fontManager.addfont(path)
    plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=path).get_name(), 'DejaVu Sans']


# ----------------------------------------------------------------------------- helpers

def boot(x, n: int = 2000, seed: int = 0) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return {'mean': None, 'lo': None, 'hi': None, 'n': 0}
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return {'mean': float(x.mean()), 'lo': float(np.quantile(bs, 0.025)), 'hi': float(np.quantile(bs, 0.975)),
            'n': int(len(x))}


def groups(grid, target) -> dict[str, list[int]]:
    """Disjoint token groups of the module docstring."""
    H, W = grid
    g = ra.token_groups(tuple(grid), list(target))
    Re = [i * W + W - 1 for i in sorted({t // W for t in target}) if i * W + W - 1 not in set(target)]
    out = {'T': g['T'], 'Ra': g['Ra'], 'Rl': g['Rl'], 'Rr': [t for t in g['Rr'] if t not in Re], 'Re': Re, 'C': g['C']}
    used = set().union(*map(set, out.values()))
    out['other'] = [t for t in range(H * W) if t not in used]
    return out


def load_maps(run: Path, tag: str, sub: str = 'maps') -> dict[str, dict]:
    out = {}
    for f in sorted((run / f'{sub}_{tag}').glob('*.npz')):
        z = np.load(f)
        out[f.stem] = {k: z[k] for k in z.files}
    return out


def head_index(rec: dict, head: tuple[int, int]) -> int | None:
    hs = [tuple(int(v) for v in h) for h in rec['heads']]
    return hs.index(head) if head in hs else None


def query_index(rec: dict, axis: str) -> int | None:
    keys = [str(k) for k in rec['keys']]
    for i, k in enumerate(keys):
        if AXIS_OF.get(k) == axis:
            return i
    return None


def att_row(rec, head, axis) -> np.ndarray | None:
    """Image attention of one head at one axis query, normalised to sum 1 over the image tokens."""
    i, q = head_index(rec, head), query_index(rec, axis)
    if i is None or q is None:
        return None
    a = rec['att'][i, q].astype(np.float64)
    return a / a.sum()


def cond_of(case_id: str) -> str:
    if case_id.startswith('sweep_'):
        return 'sweep_' + case_id.rsplit('_', 1)[1]
    if case_id.startswith('rc'):
        return 'rc_point'
    head = case_id.split('_')[0]
    return f"{head}_{case_id.rsplit('_', 1)[1]}"


def r2_heads(r2_run: Path) -> dict:
    """{tag: {'rowcol|v': [(layer, head, r), ...], ...}} from the round-2 summary."""
    s = json.loads((r2_run / 'summary.json').read_text())['models']
    out = {}
    for tag, key in R2_KEY.items():
        out[tag] = {}
        for k, v in s[key]['keys'].items():
            fmt = k.split('|')[0].split('_')[1]
            out[tag][f"{fmt}|{k.split('|')[1]}"] = [
                (int(h['head'][1:].split('_')[0]), int(h['head'][1:].split('_')[1]), float(h['r']))
                for h in v['reading_heads'][:5]]
    return out


# ----------------------------------------------------------------------------- statistics

def group_shares(maps: dict, heads: list[tuple[int, int]], conds: list[str]) -> dict:
    """{cond: {axis: {head: {group: boot}}}} attention shares at each axis query."""
    out = {}
    for cond in conds:
        recs = [r for cid, r in maps.items() if cond_of(cid) == cond]
        out[cond] = {}
        for axis in ('v', 'h'):
            out[cond][axis] = {}
            for hd in heads:
                vals = defaultdict(list)
                for r in recs:
                    a = att_row(r, hd, axis)
                    if a is None:
                        continue
                    for g, idx in groups(r['grid'], r['target']).items():
                        vals[g].append(a[idx].sum() if idx else None)
                if vals:
                    out[cond][axis][f'L{hd[0]}H{hd[1]}'] = {g: boot(v) for g, v in vals.items()}
    return out


def weighted_shares(maps: dict, ranking: list[tuple[int, int, float]], cond: str, axis: str) -> dict:
    """r-weighted mean share over the top round-2 reading heads, per group (bootstrap over answers)."""
    recs = [r for cid, r in maps.items() if cond_of(cid) == cond]
    per = defaultdict(list)
    for r in recs:
        g = groups(r['grid'], r['target'])
        acc, wsum = defaultdict(float), 0.0
        for l, h, w in ranking:
            a = att_row(r, (l, h), axis)
            if a is None or w <= 0:
                continue
            wsum += w
            for name, idx in g.items():
                acc[name] += w * (a[idx].sum() if idx else 0.0)
        if wsum:
            for name in g:
                per[name].append(acc[name] / wsum)
    return {name: boot(v) for name, v in per.items()}


def decomposition(maps: dict, heads, conds) -> dict:
    """Per head and axis query: how much the MRoPE rotation shapes the attention over image tokens.

    s_m: full score of image token m; c_m: the same score with key m rotated as if it sat at the target's
    first token (content part); p_m = s_m - c_m. Reported: var(p) / var(s) over image tokens, the total
    variation distance between softmax(s) and softmax(c) over image tokens, and the per-section share of the
    content-score advantage of each group over `other`.
    """
    out = {}
    for hd in heads:
        for axis in ('v', 'h'):
            vr, tv, adv = [], [], defaultdict(list)
            for cid, r in maps.items():
                if cond_of(cid) not in conds:
                    continue
                i, q = head_index(r, hd), query_index(r, axis)
                if i is None or q is None:
                    continue
                c = r['comp'][i, q].astype(np.float64)                          # (7, Nv)
                s = c[:4].sum(0)
                con = c[4:7].sum(0) + c[3]
                p = s - con
                vr.append(p.var() / s.var())
                ps, pc = np.exp(s - s.max()), np.exp(con - con.max())
                tv.append(0.5 * np.abs(ps / ps.sum() - pc / pc.sum()).sum())
                g = groups(r['grid'], r['target'])
                for name in ('T', 'Ra', 'Rl', 'Rr', 'C'):
                    if g[name]:
                        adv[name].append([c[k][g[name]].mean() - c[k][g['other']].mean() for k in range(4)])
            if vr:
                out[f'L{hd[0]}H{hd[1]}|{axis}'] = {
                    'pos_var_frac': boot(vr), 'tv_softmax': boot(tv),
                    'section_adv': {n: dict(zip(ra.SECTIONS, np.mean(v, 0).round(3).tolist())) for n, v in adv.items()}}
    return out


def tuning(maps: dict, head, axis: str, conds) -> dict:
    """Row-start (axis v: column 0 per token row) or column-start (axis h: row 0 per token column) profile.

    Offsets are relative to the target's first token row (v) or first token column (h). Returns attention
    share and content score (relative to the mean content score of all image tokens) per offset.
    """
    att, sc = defaultdict(list), defaultdict(list)
    for cid, r in maps.items():
        if cond_of(cid) not in conds:
            continue
        i, q = head_index(r, head), query_index(r, axis)
        if i is None or q is None:
            continue
        H, W = (int(v) for v in r['grid'])
        a = r['att'][i, q].astype(np.float64)
        a = a / a.sum()
        c = r['comp'][i, q].astype(np.float64)
        con = c[4:7].sum(0) + c[3]
        con = con - con.mean()
        tgt = [int(t) for t in r['target']]
        if axis == 'v':
            t0 = min(t // W for t in tgt)
            line = [(k, k * W) for k in range(H)]
        else:
            t0 = min(t % W for t in tgt)
            line = [(k, k) for k in range(W)]
        for k, tok in line:
            if tok in tgt:
                continue
            att[k - t0].append(a[tok])
            sc[k - t0].append(con[tok])
    offs = sorted(att)
    return {'offset': offs, 'att': [float(np.mean(att[o])) for o in offs],
            'score': [float(np.mean(sc[o])) for o in offs], 'n': [len(att[o]) for o in offs]}


ROW_BINS = {'1-3': (0, 3), '4-6': (3, 6), '7-10': (6, 10)}


def anchor_by_row(maps: dict, head, conds) -> dict:
    """Row-start and target shares at the y query, binned by the target's cell row (1-based)."""
    acc = defaultdict(lambda: defaultdict(list))
    for cid, r in maps.items():
        if cond_of(cid) not in conds:
            continue
        a = att_row(r, head, 'v')
        g = groups(r['grid'], r['target'])
        if a is None or not g['Ra']:
            continue
        W = int(r['grid'][1])
        cell = min(int(t) // W for t in r['target']) // 2
        for b, (lo, hi) in ROW_BINS.items():
            if lo <= cell < hi:
                acc[b]['Ra'].append(a[g['Ra']].sum())
                acc[b]['T'].append(a[g['T']].sum())
    return {b: {k: boot(v) for k, v in d.items()} for b, d in acc.items()}


def edits_by_row(run: Path, tag: str, rows=('vT', 'vRa', 'vR')) -> dict:
    """Recovered fraction of the vertical-move value edits, binned by the target's cell row (grid row/col formats)."""
    path = run / f'edits_{tag}.jsonl'
    acc = defaultdict(lambda: defaultdict(list))
    for rec in cf.sr.read_jsonl(path):
        if rec['cond'] not in ('t6g10_rowcol', 't6g10_colrow'):
            continue
        W = rec['grid'][1]
        cell = min(rec['target']) // W // 2
        b = next(k for k, (lo, hi) in ROW_BINS.items() if lo <= cell < hi)
        for mv in rec['moves']:
            if mv['axis'] != 'v':
                continue
            for v in mv['values']:
                ld = v['ld']
                den = ld[0] - ld[1]
                if v['axis'] != 'v' or v['dA'] == v['dB'] or abs(den) < 2:
                    continue
                for name in rows:
                    i = rec['rows'].index(name)
                    if mv['sizes'][name]:
                        acc[b][name].append((ld[1 + i] - ld[1]) / den)
    return {b: {k: boot(v) for k, v in d.items()} for b, d in acc.items()}


def row_profile(maps: dict, head, axis: str, conds) -> dict:
    """Content score c_m (minus the image mean) along the target's first token row (axis v) or first token column
    (axis h), and along a control line 8 token rows (columns) away that contains no target token."""
    tgt_line, ctl_line = [], []
    for cid, r in maps.items():
        if cond_of(cid) not in conds:
            continue
        i, q = head_index(r, head), query_index(r, axis)
        if i is None or q is None:
            continue
        H, W = (int(v) for v in r['grid'])
        c = r['comp'][i, q].astype(np.float64)
        con = c[4:7].sum(0) + c[3]
        con = con - con.mean()
        t = [int(x) for x in r['target']]
        if axis == 'v':
            k0, used, n, tok = min(x // W for x in t), {x // W for x in t}, H, (lambda a, b: a * W + b)
            across = {x % W for x in t}
            size = W
        else:
            k0, used, n, tok = min(x % W for x in t), {x % W for x in t}, W, (lambda a, b: b * W + a)
            across = {x // W for x in t}
            size = H
        ctl = next(k for k in ((k0 + 8) % n, (k0 + 10) % n, (k0 + 12) % n) if k not in used)
        tgt_line.append([con[tok(k0, j)] if j not in across else np.nan for j in range(size)])
        ctl_line.append([con[tok(ctl, j)] for j in range(size)])
    return {'target_line': np.nanmean(tgt_line, 0).tolist(), 'control_line': np.mean(ctl_line, 0).tolist(),
            'n': len(tgt_line)}


def fig_profiles(profiles: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(profiles), figsize=(3.4 * len(profiles), 3.0), squeeze=False)
    for ax, (label, pr) in zip(axes[0], profiles.items()):
        x = np.arange(len(pr['target_line']))
        ax.plot(x, pr['target_line'], 'o-', ms=3, color='#c0392b', label='目标所在线（不含目标）')
        ax.plot(x, pr['control_line'], 's-', ms=3, color='#7f8c8d', label='相隔 8 个 token 的对照线')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel('沿线位置（列号或行号）', fontsize=8)
    axes[0, 0].set_ylabel('内容分数（减去图像均值）', fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cross_stats(run: Path, tag: str, heads) -> dict:
    """Run-A query against run-B keys (target moved two cells along the axis of the move).

    For the y query and a vertical move (x query, horizontal move): shares on the row-start tokens of run A's
    target rows (old anchors), of run B's target rows (new anchors), on run A's target tokens (old target) and
    run B's (new target), per condition AA / AB / A_anchorB / BB.
    """
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    d = run / f'cross_{tag}'
    if not d.exists():
        return {}
    for f in sorted(d.glob('*.npz')):
        z = np.load(f)
        case, axis = f.stem.split('__')
        H, W = (int(v) for v in z['grid'])
        ta, tb = [int(t) for t in z['target']], [int(t) for t in z['moved']]
        def anchors(t):
            return [i * W for i in sorted({x // W for x in t}) if i * W not in t]
        if axis == 'v':
            grp = {'old_target': ta, 'new_target': tb, 'old_anchor': anchors(ta), 'new_anchor': anchors(tb)}
        else:
            grp = {'old_target': ta, 'new_target': tb}
        hs = [tuple(int(v) for v in h) for h in z['heads']]
        keys = [str(k) for k in z['keys']]
        q = next((i for i, k in enumerate(keys) if AXIS_OF.get(k) == axis), None)
        if q is None:
            continue
        for hd in heads:
            if hd not in hs:
                continue
            i = hs.index(hd)
            for c_i, cname in enumerate([str(c) for c in z['conds']]):
                a = z['att'][i, c_i, q].astype(np.float64)
                a = a / a.sum()
                for g, idx in grp.items():
                    out[f'L{hd[0]}H{hd[1]}|{axis}|{cond_of(case)}'][cname][g].append(
                        a[[t for t in idx if t not in (ta if g.startswith('new') else tb)]].sum() if idx else None)
    return {k: {c: {g: boot(v) for g, v in dd.items()} for c, dd in d2.items()} for k, d2 in out.items()}


def fig_cross(cross: dict, path: Path) -> None:
    """Attention of run-A queries on run-B keys (rowcol format)."""
    panels = [('vl', 'L24H30|v|t6g10_rowcol', 'Qwen3-VL L24 H30 纵坐标'), ('q35', 'L20H11|v|t6g10_rowcol', 'Qwen3.5 L20 H11 纵坐标'),
              ('vl', 'L25H13|h|t6g10_rowcol', 'Qwen3-VL L25 H13 横坐标')]
    names = {'AA': 'A 的 query\nA 的 key', 'A_anchorB': 'A 的 query\n仅行首 key 取自 B', 'AB': 'A 的 query\nB 的 key',
             'BB': 'B 的 query\nB 的 key'}
    gname = {'old_target': '原目标位置', 'new_target': '新目标位置', 'old_anchor': '原目标行的行首', 'new_anchor': '新目标行的行首'}
    gcol = {'old_target': '#c0392b', 'new_target': '#f1948a', 'old_anchor': '#2e86c1', 'new_anchor': '#85c1e9'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, (tag, key, title) in zip(axes, panels):
        d = cross.get(tag, {}).get(key, {})
        conds = [c for c in ('AA', 'A_anchorB', 'AB', 'BB') if c in d]
        gs = [g for g in gname if conds and g in d[conds[0]]]
        w = 0.8 / max(len(gs), 1)
        for gi, g in enumerate(gs):
            m = np.array([d[c][g]['mean'] for c in conds])
            lo = np.array([d[c][g]['lo'] for c in conds])
            hi = np.array([d[c][g]['hi'] for c in conds])
            x = np.arange(len(conds)) + (gi - (len(gs) - 1) / 2) * w
            ax.bar(x, m, w, color=gcol[g], label=gname[g])
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt='none', ecolor='k', lw=0.7)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([names[c] for c in conds], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('图像内 attention 份额')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def edit_stats(run: Path, tag: str) -> dict:
    """{cond: {axis: {row: {'r': boot, 'dlogp': boot}}}} on the axis of the move."""
    path = run / f'edits_{tag}.jsonl'
    if not path.exists():
        return {}
    acc = defaultdict(lambda: defaultdict(list))
    excluded = defaultdict(int)
    for rec in cf.sr.read_jsonl(path):
        rows = rec['rows']
        for mv in rec['moves']:
            for v in mv['values']:
                if v['axis'] != mv['axis']:
                    continue
                ld = v['ld']
                den = ld[0] - ld[1]
                key = (rec['cond'], mv['axis'])
                if v['dA'] == v['dB'] or abs(den) < 2:
                    excluded[key] += 1
                    continue
                for i, name in enumerate(rows):
                    if name != 'none' and mv['sizes'][name] == 0:
                        continue
                    acc[key][name + '|r'].append((ld[1 + i] - ld[1]) / den)
                    acc[key][name + '|dlogp'].append(v['logp_dA'][1 + i] - v['logp_dA'][1])
    out = {}
    for (cond, axis), d in acc.items():
        out.setdefault(cond, {})[axis] = {k: boot(v) for k, v in d.items()}
        out[cond][axis]['excluded'] = excluded[(cond, axis)]
    return out


def order_stats(run: Path, tag: str, heads) -> dict:
    """Attention shares under the three image-token orders, per condition / head / axis.

    Groups: T; row-start tokens of the target rows (col 0); row-end tokens (last col); column-start tokens of
    the target columns (row 0) and column-end tokens (last row). The answer digit's log-probability
    (teacher-forced, run-A answer) is reported per order.
    """
    maps = load_maps(run, tag, 'order')
    out = {}
    for cid, r in maps.items():
        case, order = cid.split('__')
        cond = cond_of(case)
        H, W = (int(v) for v in r['grid'])
        tgt = [int(t) for t in r['target']]
        rows = sorted({t // W for t in tgt})
        cols = sorted({t % W for t in tgt})
        grp = {'T': tgt, 'row_start': [i * W for i in rows], 'row_end': [i * W + W - 1 for i in rows],
               'col_start': [j for j in cols], 'col_end': [(H - 1) * W + j for j in cols]}
        grp = {k: [t for t in v if t not in tgt] if k != 'T' else v for k, v in grp.items()}
        for hd in heads:
            for axis in ('v', 'h'):
                a = att_row(r, hd, axis)
                if a is None:
                    continue
                d = out.setdefault(cond, {}).setdefault(f'L{hd[0]}H{hd[1]}|{axis}', {}).setdefault(order, defaultdict(list))
                for k, idx in grp.items():
                    d[k].append(a[idx].sum() if idx else None)
        for axis in ('v', 'h'):
            q = query_index(r, axis)
            if q is None:
                continue
            d = out.setdefault(cond, {}).setdefault(f'logp|{axis}', {}).setdefault(order, defaultdict(list))
            d['top_is_raster'].append(None)
            d['logp'].append(float(r['digit_logp'][q].max()))
            d['argmax'].append(int(r['digit_logp'][q].argmax()))
    res = {}
    for cond, dd in out.items():
        for key, by_order in dd.items():
            for order, d in by_order.items():
                if key.startswith('logp'):
                    continue
                res.setdefault(cond, {}).setdefault(key, {})[order] = {k: boot(v) for k, v in d.items()}
    # agreement of the argmax digit with the raster order (same case, same axis)
    agree = defaultdict(lambda: defaultdict(list))
    for cid, r in maps.items():
        case, order = cid.split('__')
        base = maps.get(f'{case}__raster')
        if base is None:
            continue
        for axis in ('v', 'h'):
            q = query_index(r, axis)
            if q is None:
                continue
            agree[(cond_of(case), axis)][order].append(
                float(r['digit_logp'][q].argmax() == base['digit_logp'][q].argmax()))
    for (cond, axis), d in agree.items():
        res.setdefault(cond, {})[f'argmax_agree|{axis}'] = {o: boot(v) for o, v in d.items()}
    return res


# ----------------------------------------------------------------------------- figures

def case_image(cases: dict, run: Path, cid: str) -> Image.Image:
    if cid.startswith('sweep_'):
        return Image.open(run / 'sweep_images' / f"{cid.rsplit('_', 1)[0]}.png").convert('RGB')
    return cf.sr.load_image(cases[cid])


def overlay(ax, img: Image.Image, att: np.ndarray, grid, target, vmax: float | None = None, gamma: float = 1.0):
    """Image with the per-token attention share drawn as an opaque-ish viridis layer; returns the mappable."""
    H, W = (int(v) for v in grid)
    ax.imshow(img.resize((W * 32, H * 32)), extent=(0, W, H, 0))
    from matplotlib.colors import PowerNorm
    m = ax.imshow(att.reshape(H, W), cmap='viridis', alpha=0.62, extent=(0, W, H, 0), interpolation='nearest',
                  norm=PowerNorm(gamma, vmin=0, vmax=vmax if vmax else max(att.max(), 1e-6)))
    tg = [int(t) for t in target]
    r0, r1 = min(t // W for t in tg), max(t // W for t in tg)
    c0, c1 = min(t % W for t in tg), max(t % W for t in tg)
    ax.add_patch(Rectangle((c0, r0), c1 - c0 + 1, r1 - r0 + 1, fill=False, ec='red', lw=1.6))
    ax.set_xticks([])
    ax.set_yticks([])
    return m


def fig_overlays(maps: dict, cases: dict, run: Path, ids: list[str], panels: list[tuple[str, tuple[int, int], str]],
                 path: Path, title: str) -> None:
    """Rows = images, columns = (label, head, axis) panels; one colorbar per panel."""
    ids = [i for i in ids if i in maps]
    fig, axes = plt.subplots(len(ids), len(panels), figsize=(2.9 * len(panels), 2.75 * len(ids)), squeeze=False)
    for r_i, cid in enumerate(ids):
        rec = maps[cid]
        img = case_image(cases, run, cid)
        for c_i, (label, hd, axis) in enumerate(panels):
            ax = axes[r_i, c_i]
            a = att_row(rec, hd, axis)
            if a is None:
                ax.axis('off')
                continue
            m = overlay(ax, img, a, rec['grid'], rec['target'])
            cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            if c_i == len(panels) - 1:
                cb.set_label('图像内 attention 份额', fontsize=6)
            if r_i == 0:
                ax.set_title(label, fontsize=8)
            if c_i == 0:
                ax.set_ylabel(cid.replace('_', ' '), fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_sweep(maps: dict, run: Path, tpl: str, fmt: str, panels, path: Path, title: str) -> None:
    """Target swept down column 4 (rows 1-10) and along row 4 (columns 1-10): one heatmap per position."""
    rows = [f'sweep_{tpl}_{r}4_{fmt}' for r in range(10)]
    cols = [f'sweep_{tpl}_4{c}_{fmt}' for c in range(10)]
    fig, axes = plt.subplots(2 * len(panels), 10, figsize=(16, 3.3 * len(panels)), squeeze=False)
    for p_i, (label, hd, axis) in enumerate(panels):
        for line_i, ids in enumerate((rows, cols)):
            for k, cid in enumerate(ids):
                ax = axes[2 * p_i + line_i, k]
                rec = maps.get(cid)
                a = None if rec is None else att_row(rec, hd, axis)
                if a is None:
                    ax.axis('off')
                    continue
                m = overlay(ax, case_image({}, run, cid), a, rec['grid'], rec['target'], vmax=1.0, gamma=0.5)
                if k == 0:
                    ax.set_ylabel(f"{label}\n{'目标沿列移动' if line_i == 0 else '目标沿行移动'}", fontsize=7)
    fig.subplots_adjust(right=0.93, hspace=0.08, wspace=0.04)
    cax = fig.add_axes([0.945, 0.15, 0.01, 0.7])
    fig.colorbar(m, cax=cax, label='图像内 attention 份额（每个 token，平方根色标）')
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_group_bars(shares: dict, path: Path, title: str) -> None:
    """Stacked shares per (model, format, axis): weighted top-5 reading heads."""
    labels, data = [], []
    for key, d in shares.items():
        labels.append(key)
        data.append([d[g]['mean'] or 0.0 for g in GROUPS])
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(labels) + 1.6))
    left = np.zeros(len(labels))
    for gi, g in enumerate(GROUPS):
        ax.barh(range(len(labels)), data[:, gi], left=left, color=COLORS[g], label=GROUP_NAMES[g])
        left += data[:, gi]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('图像内 attention 份额（前 5 个读取 head 按恢复比例加权）')
    ax.legend(ncol=7, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.12 - 0.6 / len(labels)))
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_edits(stats: dict, path: Path) -> None:
    """Recovered fraction r of each value edit, both models, y (move down/up) and x (move right/left)."""
    rows = ('vT', 'vRa', 'vRl', 'vRr', 'vC')
    names = {'vT': '目标（取自 run B）', 'vRa': '行首（平移 2 格）', 'vRl': '行内左侧（平移）', 'vRr': '行内右侧（平移）',
             'vC': '目标所在列（平移）'}
    conds = ('t6g10_rowcol', 't6g10_colrow', 't6g10_point', 'disc_point')
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharey=True)
    for m_i, tag in enumerate(('vl', 'q35')):
        for a_i, axis in enumerate(('v', 'h')):
            ax = axes[m_i, a_i]
            width = 0.16
            for r_i, row in enumerate(rows):
                means, los, his = [], [], []
                for cond in conds:
                    b = stats.get(tag, {}).get(cond, {}).get(axis, {}).get(row + '|r')
                    means.append(b['mean'] if b and b['mean'] is not None else np.nan)
                    los.append(b['lo'] if b and b['lo'] is not None else np.nan)
                    his.append(b['hi'] if b and b['hi'] is not None else np.nan)
                x = np.arange(len(conds)) + (r_i - 2) * width
                means, los, his = map(np.array, (means, los, his))
                ax.bar(x, means, width, color=list(COLORS.values())[[0, 1, 2, 3, 5][r_i]], label=names[row])
                ax.errorbar(x, means, yerr=[means - los, his - means], fmt='none', ecolor='k', lw=0.8)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels([FMT_OF[c] for c in conds], fontsize=8)
            ax.set_title(f"{MODEL[tag]}：{'纵坐标（目标上下移动 2 格）' if axis == 'v' else '横坐标（目标左右移动 2 格）'}",
                         fontsize=9)
            if a_i == 0:
                ax.set_ylabel('恢复比例 r')
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_tuning(curves: dict, path: Path) -> None:
    """Row-start / column-start profiles: attention share and content score versus offset from the target."""
    fig, axes = plt.subplots(2, len(curves), figsize=(3.3 * len(curves), 5.2), squeeze=False)
    for i, (label, cv_) in enumerate(curves.items()):
        o = np.array(cv_['offset'])
        axes[0, i].bar(o, cv_['att'], color='#2e86c1')
        axes[0, i].set_title(label, fontsize=8)
        axes[1, i].plot(o, cv_['score'], 'o-', color='#c0392b', ms=3)
        axes[1, i].axhline(0, color='k', lw=0.5)
        for ax in axes[:, i]:
            ax.axvline(0, color='grey', ls=':', lw=0.8)
            ax.set_xlim(-10, 10)
        axes[1, i].set_xlabel('相对目标首行（或首列）的 token 偏移')
    axes[0, 0].set_ylabel('attention 份额（每个 token）')
    axes[1, 0].set_ylabel('内容分数（减去图像均值）')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_decomp(dec: dict, path: Path) -> None:
    """Positional share of the score variance and the softmax change when rotation differences are removed."""
    keys = list(dec)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(keys) + 1.5))
    for ax, metric, xl in ((axes[0], 'pos_var_frac', 'var(位置部分) / var(总分数)'),
                           (axes[1], 'tv_softmax', '去掉旋转差异后 attention 的总变差距离')):
        m = [dec[k][metric]['mean'] for k in keys]
        lo = [dec[k][metric]['lo'] for k in keys]
        hi = [dec[k][metric]['hi'] for k in keys]
        y = np.arange(len(keys))
        ax.barh(y, m, color=['#2e86c1' if k.startswith('VL') else '#8e44ad' for k in keys])
        ax.errorbar(m, y, xerr=[np.array(m) - lo, np.array(hi) - m], fmt='none', ecolor='k', lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(keys, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(xl, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_order(maps_by_order: dict, cases: dict, panels, ids, path: Path, title: str) -> None:
    """Rows = (image, order), columns = head panels; attention mapped back to raster positions."""
    orders = ra.ORDERS
    names = {'raster': '光栅顺序', 'rev': '行内从右到左', 'colmajor': '按列（从上到下）'}
    n = len(ids) * len(orders)
    fig, axes = plt.subplots(n, len(panels), figsize=(2.9 * len(panels), 2.7 * n), squeeze=False)
    for i, cid in enumerate(ids):
        img = cf.sr.load_image(cases[cid])
        for j, order in enumerate(orders):
            rec = maps_by_order.get(f'{cid}__{order}')
            for k, (label, hd, axis) in enumerate(panels):
                ax = axes[i * len(orders) + j, k]
                a = None if rec is None else att_row(rec, hd, axis)
                if a is None:
                    ax.axis('off')
                    continue
                m = overlay(ax, img, a, rec['grid'], rec['target'])
                cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=6)
                if i == 0 and j == 0:
                    ax.set_title(label, fontsize=8)
                if k == 0:
                    ax.set_ylabel(f'{cid}\n{names[order]}', fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--flow-run', type=Path, required=True)
    p.add_argument('--r2-run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args()
    use_cjk_font()
    args.figures.mkdir(parents=True, exist_ok=True)
    cases = cf.load_cases(args.flow_run)
    rank = r2_heads(args.r2_run)
    summary = {'shares': {}, 'weighted': {}, 'decomposition': {}, 'tuning': {}, 'edits': {}, 'order': {}}
    key_heads = {'vl': [('L24 H30', (24, 30)), ('L22 H11', (22, 11)), ('L25 H13', (25, 13)), ('L27 H20', (27, 20))],
                 'q35': [('L20 H11', (20, 11)), ('L24 H0', (24, 0)), ('L32 H8', (32, 8))]}
    conds = ['t6g10_rowcol', 't6g10_colrow', 't6g10_point', 'disc_point', 'rc_point', 'sweep_rowcol', 'sweep_point']
    weighted_for_fig = {}
    all_maps = {}
    for tag in ('vl', 'q35'):
        maps = load_maps(args.run, tag)
        all_maps[tag] = maps
        heads = sorted({tuple(int(v) for v in h) for r in maps.values() for h in r['heads']})
        summary['shares'][tag] = group_shares(maps, heads, conds)
        for cond in conds:
            fmt = 'rowcol' if cond in ('t6g10_rowcol', 't6g10_colrow', 'sweep_rowcol') else 'point'
            for axis in ('v', 'h'):
                w = weighted_shares(maps, rank[tag][f'{fmt}|{axis}'], cond, axis)
                summary['weighted'].setdefault(tag, {}).setdefault(cond, {})[axis] = w
                if cond in ('t6g10_rowcol', 't6g10_colrow', 't6g10_point', 'disc_point', 'rc_point') and w:
                    weighted_for_fig[f"{MODEL[tag]} {FMT_OF[cond]} {'y' if axis == 'v' else 'x'}"] = w
        dec = decomposition(maps, heads, {'t6g10_rowcol', 't6g10_colrow', 't6g10_point', 'disc_point', 'rc_point'})
        summary['decomposition'][tag] = dec
        for label, hd in key_heads[tag]:
            for axis in ('v', 'h'):
                for cset_name, cset in (('grid_rowcol', {'t6g10_rowcol', 't6g10_colrow', 'sweep_rowcol'}),
                                        ('grid_point', {'t6g10_point', 'sweep_point'}), ('disc', {'disc_point'}),
                                        ('refcoco', {'rc_point'})):
                    summary['tuning'].setdefault(tag, {})[f'{label}|{axis}|{cset_name}'] = tuning(maps, hd, axis, cset)
        summary['edits'][tag] = edit_stats(args.run, tag)
        y_head = (24, 30) if tag == 'vl' else (20, 11)
        summary.setdefault('by_row', {})[tag] = {
            'attention': anchor_by_row(maps, y_head, {'t6g10_rowcol', 't6g10_colrow', 'sweep_rowcol'}),
            'edits': edits_by_row(args.run, tag)}
        summary.setdefault('cross', {})[tag] = cross_stats(args.run, tag, [hd for _, hd in key_heads[tag]])
        if (args.run / f'order_{tag}').exists():
            summary['order'][tag] = order_stats(args.run, tag, [hd for _, hd in key_heads[tag]])
    (args.run / 'summary.json').write_text(json.dumps(summary, indent=1))

    # ---- figures
    fig_group_bars(weighted_for_fig, args.figures / 'group_shares.png',
                   '读取 head 的 attention 落点（按 3.14 第二轮的单 head 恢复比例加权）')
    fig_edits(summary['edits'], args.figures / 'value_edits.png')
    fig_cross(summary['cross'], args.figures / 'cross_keys.png')
    dec_fig = {}
    for tag in ('vl', 'q35'):
        for k, v in summary['decomposition'][tag].items():
            dec_fig[f"{'VL' if tag == 'vl' else 'Q3.5'} {k.replace('|v', ' y').replace('|h', ' x')}"] = v
    fig_decomp(dec_fig, args.figures / 'mrope_decomposition.png')
    curves = {}
    for tag, label, axis in (('vl', 'L24 H30', 'v'), ('vl', 'L22 H11', 'v'), ('vl', 'L25 H13', 'h'),
                             ('q35', 'L20 H11', 'v'), ('q35', 'L24 H0', 'v'), ('q35', 'L20 H11', 'h')):
        curves[f"{MODEL[tag]} {label} {'y：第 0 列各行' if axis == 'v' else 'x：第 0 行各列'}"] = \
            summary['tuning'][tag][f'{label}|{axis}|grid_rowcol']
    fig_tuning(curves, args.figures / 'anchor_tuning.png')
    profiles = {}
    for tag, label, hd, axis in (('vl', 'L24 H30', (24, 30), 'v'), ('vl', 'L22 H11', (22, 11), 'v'),
                                 ('vl', 'L25 H13', (25, 13), 'h'), ('q35', 'L20 H11', (20, 11), 'v'),
                                 ('q35', 'L20 H11', (20, 11), 'h')):
        pr = row_profile(all_maps[tag], hd, axis, {'t6g10_rowcol', 't6g10_colrow'})
        summary.setdefault('profiles', {})[f'{tag} {label} {axis}'] = pr
        profiles[f"{MODEL[tag]} {label} {'y：沿目标所在行' if axis == 'v' else 'x：沿目标所在列'}"] = pr
    fig_profiles(profiles, args.figures / 'line_profiles.png')
    (args.run / 'summary.json').write_text(json.dumps(summary, indent=1))
    grid_ids = ['t6g10_001_rowcol', 't6g10_003_rowcol', 't6g10_004_rowcol', 't6g10_007_rowcol', 't6g10_013_rowcol',
                't6g10_022_rowcol']
    point_ids = [i.replace('rowcol', 'point') for i in grid_ids[:4]]
    disc_ids = sorted(i for i in all_maps['vl'] if i.startswith('disc'))[:4]
    rc_ids = sorted(i for i in all_maps['vl'] if i.startswith('rc') and i in all_maps['q35'])[:6]
    vl_panels = [('VL L25 H13 横坐标', (25, 13), 'h'), ('VL L24 H30 纵坐标', (24, 30), 'v'),
                 ('VL L22 H11 纵坐标', (22, 11), 'v'), ('VL L27 H20 纵坐标', (27, 20), 'v')]
    q_panels = [('Q3.5 L20 H11 横坐标', (20, 11), 'h'), ('Q3.5 L20 H11 纵坐标', (20, 11), 'v'),
                ('Q3.5 L24 H0 纵坐标', (24, 0), 'v'), ('Q3.5 L32 H8 纵坐标', (32, 8), 'v')]
    for name, ids in (('grid_rowcol', grid_ids), ('grid_point', point_ids), ('disc', disc_ids), ('refcoco', rc_ids)):
        for tag, panels in (('vl', vl_panels), ('q35', q_panels)):
            fig_overlays(all_maps[tag], cases, args.run, ids, panels, args.figures / f'overlay_{name}_{tag}.png',
                         f'{MODEL[tag]}：首位数字 query 处的图像 attention（{name}，红框为目标）')
    for tag, panels in (('vl', [('VL L24 H30 y', (24, 30), 'v'), ('VL L25 H13 x', (25, 13), 'h')]),
                        ('q35', [('Q3.5 L20 H11 y', (20, 11), 'v'), ('Q3.5 L20 H11 x', (20, 11), 'h')])):
        fig_sweep(all_maps[tag], args.run, 'person', 'rowcol', panels, args.figures / f'sweep_rowcol_{tag}.png',
                  f'{MODEL[tag]}：目标沿列（上）与沿行（下）移动时读取 head 的 attention')
        pts = [('VL L27 H20 y', (27, 20), 'v'), ('VL L25 H13 x', (25, 13), 'h')] if tag == 'vl' else panels
        fig_sweep(all_maps[tag], args.run, 'person', 'point', pts, args.figures / f'sweep_point_{tag}.png',
                  f'{MODEL[tag]}：point 格式，目标沿列与沿行移动')
    for tag, panels in (('vl', [('VL L24 H30 y', (24, 30), 'v'), ('VL L22 H11 y', (22, 11), 'v'),
                                ('VL L25 H13 x', (25, 13), 'h')]),
                        ('q35', [('Q3.5 L20 H11 y', (20, 11), 'v'), ('Q3.5 L24 H0 y', (24, 0), 'v'),
                                 ('Q3.5 L20 H11 x', (20, 11), 'h')])):
        if (args.run / f'order_{tag}').exists():
            fig_order(load_maps(args.run, tag, 'order'), cases, panels, ['t6g10_003_rowcol', 't6g10_007_rowcol'],
                      args.figures / f'order_{tag}.png', f'{MODEL[tag]}：改变图像 token 的序列顺序')
    print(json.dumps({t: summary['weighted'][t]['t6g10_rowcol'] for t in summary['weighted']}, indent=0)[:1500])


if __name__ == '__main__':
    main()
