"""Experiment 5.3 analysis: accuracy with CIs, paired tests, per-question cost, timing, and figures.

Reads WORK_DIR/analysis/reread/2026-10-21_cache_only/{lean_<bench>,native/<bench>,timing}/ and, for the native
protocol comparison, experiment 5.2's results; writes summary.json next to the run and figures to
docs/analysis/reread/figures/cache_only/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from analysis_tools.reread import cached_reread_analysis as ca

NAT = ['textvqa', 'gqa', 'mmvp', 'pope']
SYN = ['syn_color', 'syn_gap', 'syn_loc']
TITLES = ca.TITLES
ORDER = ['base', 'crop1', 'crop2', 'c27', 'c26', 'c25', 'c27n', 's27', 's26', 's25', 's27n', 'u27', 'b27',
         'full2x', 'h27', 'o_crop2', 'o_c27', 'o_s27']
FAMILY = {'base': 'none', 'full2x': 'ref', 'h27': 'ref', 'crop1': 'crop', 'crop2': 'crop', 'o_crop2': 'crop'}
COLORS = {'none': '#8c8b86', 'crop': '#eb6834', 'cache': '#2a78d6', 'ref': '#c9c8c2'}
# Pre-specified comparisons (pooled over the four natural benchmarks; Holm within each benchmark).
PRIMARY = [('c27', 'crop1'), ('s27', 'crop2'), ('u27', 'crop2'), ('b27', 'crop2'), ('s27', 'crop1'),
           ('c26', 'crop1'), ('c25', 'crop1'), ('s26', 'crop2'), ('s25', 'crop2'), ('s27', 'c27'),
           ('c27', 'c27n'), ('s27', 's27n'), ('crop2', 'base'), ('s27', 'base'), ('c27', 'base'),
           ('o_s27', 'o_crop2'), ('o_c27', 'o_crop2')]
APPENDIX = [('s27', 'full2x'), ('s27', 'h27'), ('h27', 'crop2')]


def fam(c: str) -> str:
    return FAMILY.get(c, 'cache')


def dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        k = (r['case_id'], r['cond'])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def bench_summary(rows: list[dict], bench: str, rng) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r['cond'], {})[r['case_id']] = r
    out = {'conds': {}, 'pairs': {}}
    for c, d in by.items():
        x = np.array([r['correct'] for r in d.values()], float)
        e = {'n': len(x), 'acc': float(x.mean()), 'bad_call': sum(r.get('finish_reason') == 'bad_call' for r in d.values()),
             'length_cut': sum(r.get('finish_reason') == 'length' for r in d.values()),
             'unparsed': sum(not r.get('parsed', True) for r in d.values()),
             'extra_tokens': float(np.mean([r.get('n_extra_visual', 0) for r in d.values()])),
             'L': float(np.mean([r.get('L', 0) for r in d.values()])),
             'vit_gflops': float(np.mean([r.get('vit_flops', 0) for r in d.values()])) / 1e9,
             'merger_gflops': float(np.mean([r.get('merger_flops', 0) for r in d.values()])) / 1e9,
             'vit_patches': float(np.mean([r.get('vit_patches', 0) for r in d.values()])),
             'capped': sum(bool(r.get('capped')) for r in d.values())}
        e['ci'] = ca.boot_mean(x, rng) if bench == 'textvqa' else ca.wilson(x.sum(), len(x))
        cov = [r['coverage'] for r in d.values() if r.get('coverage') is not None]
        if cov:
            e['coverage_full'] = float(np.mean(np.array(cov) >= 0.999))
        if bench in ('syn_color', 'syn_gap'):
            ok = [r for r in d.values() if r.get('s') is not None]
            s, y = np.array([r['s'] for r in ok]), np.array([r['label'] for r in ok])
            e['auroc'] = ca.auroc(s, y)
            idx = rng.integers(0, len(s), (1000, len(s)))
            bs = [ca.auroc(s[i], y[i]) for i in idx]
            e['auroc_ci'] = [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))]
        if bench == 'syn_loc':
            e['frame_error'] = sum(bool(r.get('frame_error')) for r in d.values())
        if bench == 'mmvp':
            pairs = {}
            for r in d.values():
                pairs.setdefault((int(r['case_id'].split('_')[1]) - 1) // 2, []).append(r['correct'])
            e['pair_acc'] = float(np.mean([all(v) for v in pairs.values() if len(v) == 2]))
        out['conds'][c] = e
    tests = []
    for a, b in PRIMARY + APPENDIX:
        if a in by and b in by:
            ids = sorted(set(by[a]) & set(by[b]))
            pa = np.array([by[a][i]['correct'] for i in ids], float)
            pb = np.array([by[b][i]['correct'] for i in ids], float)
            tests.append((f'{a}-{b}', ca.paired(pa, pb, rng)))
    hp = ca.holm([t[1]['p'] for t in tests]) if tests else []
    for (k, t), h in zip(tests, hp):
        out['pairs'][k] = {**t, 'p_holm': h}
    return out


def pooled(run_rows: dict, benches: list[str], rng) -> dict:
    out = {}
    for a, b in PRIMARY + APPENDIX:
        pa, pb = [], []
        for bench in benches:
            by = {}
            for r in run_rows.get(bench, []):
                by.setdefault(r['cond'], {})[r['case_id']] = r['correct']
            if a in by and b in by:
                ids = sorted(set(by[a]) & set(by[b]))
                pa += [by[a][i] for i in ids]
                pb += [by[b][i] for i in ids]
        if pa:
            out[f'{a}-{b}'] = {**ca.paired(np.array(pa, float), np.array(pb, float), rng, B=4000), 'n': len(pa)}
    return out


def timing_summary(rows: list[dict]) -> dict:
    out = {}
    for bench in sorted({r['bench'] for r in rows}):
        for c in sorted({r['cond'] for r in rows}):
            xs = [r for r in rows if r['bench'] == bench and r['cond'] == c]
            if not xs:
                continue
            out.setdefault(bench, {})[c] = {
                'n': len(xs), 'seconds': float(np.median([r['seconds'] for r in xs])),
                **{k: float(np.median([r[k] for r in xs])) for k in ('t_vit', 't_call', 't_tool', 't_answer')},
                'peak_mib': float(np.median([r['peak_extra_bytes'] for r in xs])) / 2 ** 20,
                'kv_mib': float(np.median([r['kv_bytes'] for r in xs])) / 2 ** 20,
                'L': float(np.median([r['L_final'] for r in xs]))}
    # paired per-question time differences against the crop tools (same case, same session)
    diffs = {}
    for a, b in [('c27', 'crop1'), ('s27', 'crop2'), ('u27', 'crop2'), ('s25', 'crop2'), ('c25', 'crop1')]:
        for key in ('seconds', 't_tool'):
            da = {(r['bench'], r['case_id']): r[key] for r in rows if r['cond'] == a}
            db = {(r['bench'], r['case_id']): r[key] for r in rows if r['cond'] == b}
            ks = sorted(set(da) & set(db))
            if ks:
                d = np.array([da[k] - db[k] for k in ks])
                diffs[f'{a}-{b}:{key}'] = {'n': len(ks), 'median': float(np.median(d)), 'mean': float(d.mean()),
                                           'frac_faster': float((d < 0).mean())}
    out['paired'] = diffs
    return out


def window_size_effect(rows: list[dict], rng, pairs=(('crop2', 's27'), ('crop2', 'c27'), ('crop2', 'base'),
                                                       ('s27', 'base'), ('h27', 's27'))) -> dict:
    """TextVQA: paired score differences binned by the window's share of the image (model-chosen window)."""
    by = {}
    for r in rows:
        by.setdefault(r['cond'], {})[r['case_id']] = r
    share = {i: r['window'][2] * r['window'][3] / (r['grid'][0] * r['grid'][1]) for i, r in by['crop2'].items()
             if r.get('window')}
    edges = np.quantile(list(share.values()), [0, 1 / 3, 2 / 3, 1])
    out = {'edges': [float(e) for e in edges], 'bins': []}
    for k in range(3):
        ids = [i for i, a in share.items() if (edges[k] <= a <= edges[k + 1]) if k == 0 or a > edges[k]]
        e = {'n': len(ids), 'share_median': float(np.median([share[i] for i in ids]))}
        for a, b in pairs:
            if a in by and b in by:
                d = np.array([by[a][i]['correct'] - by[b][i]['correct'] for i in ids if i in by[a] and i in by[b]])
                e[f'{a}-{b}'] = {'diff': float(d.mean()), 'ci': ca.boot_mean(d, rng)}
        out['bins'].append(e)
    return out


def native_compare(run: Path, ref: Path, rng) -> dict:
    """New cache-only conditions under 5.2's native protocol vs 5.2's crop1 / crop2 / c27 on the same questions."""
    out, pool = {}, {}
    for bench in NAT:
        new = ca.read_jsonl(run / 'native' / bench / 'results.jsonl')
        old = ca.read_jsonl(ref / bench / 'results.jsonl')
        if not new:
            continue
        rows = dedup(old + new)
        out[bench] = bench_summary(rows, bench, rng)
        pool[bench] = rows
    if pool:
        out['pooled'] = pooled(pool, list(pool), rng)
    return out


def summarize(run: Path, ref: Path) -> dict:
    rng = np.random.default_rng(0)
    rows = {b: dedup(ca.read_jsonl(run / f'lean_{b}' / 'results.jsonl')) for b in NAT + SYN}
    rows = {b: r for b, r in rows.items() if r}
    S = {'lean': {b: bench_summary(r, b, rng) for b, r in rows.items()}}
    S['lean_pooled'] = pooled(rows, [b for b in NAT if b in rows], rng)
    if 'textvqa' in rows:
        S['textvqa_window_size'] = window_size_effect(rows['textvqa'], rng)
    S['lean_pooled_acc'] = {}
    for c in ORDER:
        xs = [r['correct'] for b in NAT if b in rows for r in rows[b] if r['cond'] == c]
        if xs:
            S['lean_pooled_acc'][c] = {'n': len(xs), 'acc': float(np.mean(xs))}
    t = ca.read_jsonl(run / 'timing' / 'timing.jsonl')
    if t:
        S['timing'] = timing_summary(t)
    # closed-call protocol (lean2): calls decoded only up to the bbox's ']' and closed with the fixed tail
    rows2 = {b: dedup(ca.read_jsonl(run / f'lean2_{b}' / 'results.jsonl')) for b in NAT}
    rows2 = {b: r for b, r in rows2.items() if r}
    if rows2:
        S['lean2'] = {b: bench_summary(r, b, rng) for b, r in rows2.items()}
        S['lean2_pooled'] = pooled(rows2, list(rows2), rng)
        S['lean2_pooled_acc'] = {c: float(np.mean([r['correct'] for b in rows2 for r in rows2[b] if r['cond'] == c]))
                                 for c in ORDER if any(r['cond'] == c for b in rows2 for r in rows2[b])}
    t2 = ca.read_jsonl(run / 'timing2' / 'timing.jsonl')
    if t2:
        S['timing2'] = timing_summary(t2)
    S['native'] = native_compare(run, ref, rng)
    v = run / 'verify' / 'verify.json'
    if v.exists():
        d = json.loads(v.read_text())
        S['verify'] = {k: d[k] for k in ('grid_vs_vis_maxabs', 'shift_even_vs_grid_maxabs', 'batch_agree',
                                         'batch_answer_agree', 'n_items')}
    return S


# ----------------------------------------------------------------------------- figures

def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)


def metric(e: dict, bench: str) -> tuple[float, list[float]]:
    if bench in ('syn_color', 'syn_gap'):
        return e['auroc'], e['auroc_ci']
    return e['acc'], e['ci']


def fig_accuracy(S: dict, fig_dir: Path) -> None:
    benches = [b for b in NAT + SYN if b in S['lean']]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6))
    for ax, bench in zip(axes.flat, benches):
        conds = [c for c in ORDER if c in S['lean'][bench]['conds']]
        for i, c in enumerate(conds):
            m, ci = metric(S['lean'][bench]['conds'][c], bench)
            col = COLORS[fam(c)]
            hollow = c.startswith('o_')
            ax.errorbar(i, m, yerr=[[m - ci[0]], [ci[1] - m]], fmt='o', ms=7, color=col,
                        mfc='white' if hollow else col, mew=1.6, elinewidth=1.2, capsize=0)
        bm, _ = metric(S['lean'][bench]['conds']['base'], bench)
        ax.axhline(bm, color='#8c8b86', ls='--', lw=1)
        ax.set_xticks(range(len(conds)), conds, rotation=60, fontsize=8.5)
        ax.set_title(TITLES[bench], fontsize=11)
        ax.set_ylabel('AUROC' if bench in ('syn_color', 'syn_gap') else 'accuracy')
        _style(ax)
    for ax in axes.flat[len(benches):]:
        ax.axis('off')
    h = [plt.Line2D([], [], marker='o', ls='', color=COLORS[k], ms=8, label=l) for k, l in
         [('none', 'no tool (base)'), ('crop', 'crop + re-encode (native tool)'), ('cache', 'cache-only read (no new ViT pass)'),
          ('ref', 'reference: extra ViT pass on the 2x image (full2x, h27)')]]
    h.append(plt.Line2D([], [], marker='o', ls='', color='#555', mfc='white', ms=8, label='oracle window'))
    axes.flat[-1].legend(handles=h, loc='center', frameon=False, fontsize=10)
    fig.suptitle('Experiment 5.3, lean tool protocol: accuracy by condition (Qwen3.5-9B, thinking off)', fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / 'accuracy_lean.png', dpi=150)
    plt.close(fig)


def fig_cost(S: dict, fig_dir: Path) -> None:
    """Pooled natural accuracy vs ViT GFLOPs per question and vs median wall-clock time."""
    acc = S['lean_pooled_acc']
    conds = [c for c in ORDER if c in acc and not c.startswith('o_')]
    gfl = {c: np.mean([S['lean'][b]['conds'][c]['vit_gflops'] for b in NAT if c in S['lean'].get(b, {}).get('conds', {})])
           for c in conds}
    tm = S.get('timing', {})
    sec = {c: np.mean([tm[b][c]['seconds'] for b in NAT if b in tm and c in tm[b]]) for c in conds
           if any(b in tm and c in tm[b] for b in NAT)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xs, xl in [(axes[0], gfl, 'ViT GFLOPs per question (analytic, mean over questions)'),
                       (axes[1], sec, 'median wall-clock seconds per question (batch size 1)')]:
        for c, x in xs.items():
            ax.scatter(x, acc[c]['acc'], s=70, color=COLORS[fam(c)], edgecolor='white', lw=1.5, zorder=3)
            ax.annotate(c, (x, acc[c]['acc']), xytext=(5, 4), textcoords='offset points', fontsize=9)
        ax.set_xlabel(xl)
        ax.set_ylabel('accuracy, 4 natural benchmarks pooled (800 q)')
        _style(ax)
    h = [plt.Line2D([], [], marker='o', ls='', color=COLORS[k], ms=8, label=l) for k, l in
         [('none', 'no tool'), ('crop', 'crop + re-encode'), ('cache', 'cache-only'), ('ref', 'extra 2x ViT pass (reference)')]]
    axes[0].legend(handles=h, frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(fig_dir / 'cost_accuracy.png', dpi=150)
    plt.close(fig)


def fig_timing(S: dict, fig_dir: Path, key: str = 'timing', name: str = 'timing_breakdown.png') -> None:
    tm = S.get(key)
    if not tm:
        return
    benches = [b for b in ['textvqa', 'gqa', 'mmvp', 'pope', 'syn_loc'] if b in tm]
    conds = [c for c in ['base', 'crop1', 'c27', 'c25', 'crop2', 's27', 's25', 'u27', 'full2x', 'h27'] if c in tm[benches[0]]]
    if not conds:
        return
    parts = [('t_vit', 'ViT pass(es) on the full image', '#8c8b86'), ('t_call', 'first prefill + call decoding', '#c9c8c2'),
             ('t_tool', 'tool execution (crop re-encode / cache read)', '#eb6834'),
             ('t_answer', 'continuation prefill + answer decoding', '#2a78d6')]
    fig, axes = plt.subplots(1, len(benches), figsize=(4 * len(benches), 4.6), sharey=False)
    for ax, bench in zip(np.atleast_1d(axes), benches):
        bottom = np.zeros(len(conds))
        for k, lab, col in parts:
            v = np.array([tm[bench][c][k] * 1000 for c in conds])
            ax.bar(range(len(conds)), v, bottom=bottom, color=col, edgecolor='white', lw=1, label=lab)
            bottom += v
        ax.set_xticks(range(len(conds)), conds, rotation=60, fontsize=9)
        ax.set_title(TITLES[bench], fontsize=10)
        ax.set_ylabel('median ms per question')
        _style(ax)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=150)
    plt.close(fig)


def fig_window_size(S: dict, fig_dir: Path) -> None:
    W = S.get('textvqa_window_size')
    if not W:
        return
    keys = [('crop2-base', 'crop2 − base', '#eb6834'), ('s27-base', 's27 − base', '#2a78d6'),
            ('crop2-s27', 'crop2 − s27', '#8c8b86'), ('h27-s27', 'h27 − s27 (reference)', '#c9c8c2')]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for j, (k, lab, col) in enumerate(keys):
        for b, e in enumerate(W['bins']):
            if k not in e:
                continue
            x = b + (j - 1.5) * 0.18
            m, ci = e[k]['diff'], e[k]['ci']
            ax.bar(x, m, 0.16, color=col, label=lab if b == 0 else None)
            ax.plot([x, x], ci, color='#333', lw=1)
    ax.axhline(0, color='#555', lw=0.8)
    ax.set_xticks(range(3), [f"window share ≤ {W['edges'][1]:.2f}\n(n={W['bins'][0]['n']})",
                             f"{W['edges'][1]:.2f}–{W['edges'][2]:.2f}\n(n={W['bins'][1]['n']})",
                             f"> {W['edges'][2]:.2f}\n(n={W['bins'][2]['n']})"])
    ax.set_ylabel('paired VQA-accuracy difference')
    ax.set_title('TextVQA, lean protocol: differences by the window\'s share of the image (terciles)', fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / 'textvqa_window_size.png', dpi=150)
    plt.close(fig)


def fig_position_ids(fig_dir: Path) -> None:
    """Schematic (no model data): MRoPE (t, h, w) ids of the first image and of an appended window of tokens."""
    H, W, f = 4, 6, 10                      # toy image grid and its first sequence position
    r0, c0, h, w = 1, 2, 2, 3               # window in merged tokens
    t2 = 40                                 # sequence position where the appended segment starts
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), gridspec_kw={'width_ratios': [6, 3.4, 3.4, 5]})

    def grid(ax, rows, cols, lab, hi=None, title=''):
        for i in range(rows):
            for j in range(cols):
                fc = '#dbe8f7' if hi and hi(i, j) else 'white'
                ax.add_patch(Rectangle((j, -i - 1), 1, 1, fc=fc, ec='#555', lw=0.8))
                ax.text(j + 0.5, -i - 0.5, lab(i, j), ha='center', va='center', fontsize=7.6)
        ax.set_xlim(-0.2, cols + 0.2)
        ax.set_ylim(-rows - 0.2, 0.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10)

    inwin = lambda i, j: r0 <= i < r0 + h and c0 <= j < c0 + w
    grid(axes[0], H, W, lambda i, j: f'({f},{f + i},{f + j})', inwin,
         f'first image, {H}x{W} tokens, starts at position f = {f}\nids (t, h, w) = (f, f + row, f + col); window shaded')
    axes[0].add_patch(Rectangle((c0, -r0 - h), w, h, fill=False, ec='#2a78d6', lw=2.5))
    grid(axes[1], h, w, lambda i, j: f'({t2},{f + r0 + i},{f + c0 + j})', None,
         f'option A (orig)\nappended at t2 = {t2}\n(h, w) = (f + r0 + i, f + c0 + j)')
    grid(axes[2], h, w, lambda i, j: f'({t2},{t2 + i},{t2 + j})', None,
         f'option B (new)\nfresh image at t2 = {t2}\n(h, w) = (t2 + i, t2 + j)')
    # 4x-token reader: half-step ids
    grid(axes[3], 2 * h - 1, 2 * w - 1, lambda i, j: f'{f + r0 + i / 2:g},{f + c0 + j / 2:g}', None,
         'option A for the shifted-merge reader s27:\n(2h-1)x(2w-1) tokens, (h, w) ids in steps of 1/2 (t = t2)')
    fig.suptitle('Schematic of MRoPE position ids for appended window tokens (illustration, not measured data)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / 'position_ids.png', dpi=150)
    plt.close(fig)


def fig_merges(fig_dir: Path) -> None:
    """Schematic (no model data): which 16-px patches each cache-only reader feeds to the 2x2 merger."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    n = 4
    specs = [('grid (c27): standard 2x2 groups\n4 patches -> 1 token, h x w tokens', [(0, 0), (0, 2), (2, 0), (2, 2)], 2),
             ('shift (s27): 2x2 groups at stride 1\n(2h-1) x (2w-1) tokens', [(0, 0), (0, 1), (1, 0), (1, 1)], 2),
             ('nearest (u27): one patch fills\nall 4 merger slots, 2h x 2w tokens', [(0, 0), (0, 1), (1, 0), (1, 1)], 1),
             ('bilinear (b27): 2x bilinear upsampling\nof patches, then standard groups', None, 0)]
    cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
    for ax, (title, starts, size) in zip(axes, specs):
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((j, -i - 1), 1, 1, fc='#f4f3ef', ec='#999', lw=0.8))
        if starts:
            for k, (i, j) in enumerate(starts):
                d = 0.06 * k if size == 2 else 0.04 * k
                ax.add_patch(Rectangle((j + d, -i - size + d), size - 2 * d, size - 2 * d, fill=False, ec=cols[k], lw=2.2))
        else:
            for i in range(2 * n):
                for j in range(2 * n):
                    ax.add_patch(Rectangle((j / 2, -(i + 1) / 2), 0.5, 0.5, fill=False, ec='#bbb', lw=0.4))
            for k, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                ax.add_patch(Rectangle((j, -i - 1), 1, 1, fill=False, ec=cols[k], lw=2.2))
        ax.set_xlim(-0.2, n + 0.2)
        ax.set_ylim(-n - 0.2, 0.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10)
    fig.suptitle('Schematic of the cache-only readers on a 4x4 patch window (grey cells: 16x16-pixel ViT patches; '
                 'coloured squares: groups sent to the merger; illustration, not data)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(fig_dir / 'merge_variants.png', dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--figures', type=Path, required=True)
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    fig_position_ids(args.figures)
    fig_merges(args.figures)
    S = summarize(args.run, args.ref)
    (args.run / 'summary.json').write_text(json.dumps(S, indent=1))
    if S['lean']:
        fig_accuracy(S, args.figures)
        fig_cost(S, args.figures)
        fig_window_size(S, args.figures)
    fig_timing(S, args.figures)
    fig_timing(S, args.figures, 'timing2', 'timing_breakdown_closed.png')
    print(json.dumps({k: v for k, v in S.get('lean_pooled', {}).items()}, indent=0)[:3000])


if __name__ == '__main__':
    main()
