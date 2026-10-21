"""Experiment 5.2 analysis: accuracy with CIs, paired tests, cost accounting, window choice, and figures.

Reads WORK_DIR/analysis/reread/2026-10-20_cached_reread/<bench>/results.jsonl and timing/timing.jsonl, writes
summary.json next to them and figures to docs/analysis/reread/figures/cached_reread/.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.reread import reread_pilot as rp

BENCHES = ['textvqa', 'gqa', 'mmvp', 'pope', 'babyvision', 'syn_color', 'syn_gap', 'syn_loc']
NATURAL = ['textvqa', 'gqa', 'mmvp', 'pope', 'babyvision']
TITLES = {'textvqa': 'TextVQA val', 'gqa': 'GQA val (small evidence)', 'mmvp': 'MMVP', 'pope': 'POPE adversarial',
          'babyvision': 'BabyVision FGD', 'syn_color': 'Synthetic ΔE 10 (20×20)', 'syn_gap': 'Synthetic gap 2 px (20×20)',
          'syn_loc': 'Synthetic T5 row/col (20×20)'}
ORDER = ['base', 'full2x', 'crop1', 'crop2', 'crop4', 'c27', 'c9', 'h27', 'h9', 'h9n', 'h9r', 'o_crop2', 'o_crop4', 'o_c9',
         'o_h27', 'o_h9', 'o_h9r']
FAMILY = {'base': 'none', 'full2x': 'none', 'crop1': 'crop', 'crop2': 'crop', 'crop4': 'crop', 'c27': 'cache',
          'c9': 'cache', 'h27': 'cache', 'h9': 'cache', 'h9n': 'cache', 'o_crop2': 'oracle_crop', 'o_crop4': 'oracle_crop',
          'o_c9': 'oracle_cache', 'o_h9': 'oracle_cache', 'h9r': 'cache', 'o_h27': 'oracle_cache',
          'o_h9r': 'oracle_cache'}
BENCH_COLORS = {'textvqa': '#2a78d6', 'gqa': '#eb6834', 'mmvp': '#1baf7a', 'pope': '#eda100', 'babyvision': '#e87ba4',
                'syn_color': '#008300', 'syn_gap': '#6250d6', 'syn_loc': '#e34948'}
COLORS = {'none': '#8c8b86', 'crop': '#eb6834', 'cache': '#2a78d6', 'oracle_crop': '#eb6834', 'oracle_cache': '#2a78d6'}
# Pre-specified paired comparisons (Holm-corrected within each benchmark).
PRIMARY = [('h27', 'crop2'), ('h9', 'crop2'), ('c27', 'crop1'), ('c9', 'crop1'), ('h27', 'full2x'), ('crop2', 'base'),
           ('h27', 'base'), ('h9', 'h27'), ('h9', 'h9n'), ('o_h27', 'o_crop2'), ('o_h9', 'o_crop2')]
KV_PER_TOKEN = 8 * 2 * 4 * 256 * 2          # Qwen3.5-9B: 8 full-attention layers, 4 KV heads, head_dim 256, bf16


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value against p = 0.5 (the exact McNemar test)."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(min(1.0, sum(q for q in pmf if q <= pmf[k] * (1 + 1e-9))))


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def wilson(k: float, n: int) -> list[float]:
    return rp.wilson(int(round(k)), n)


def boot_mean(x: np.ndarray, rng, B: int = 2000) -> list[float]:
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def paired(a: np.ndarray, b: np.ndarray, rng, B: int = 2000) -> dict:
    d = a - b
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    binary = set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
    if binary:
        n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        p = binom_two_sided(n01, n01 + n10)
        extra = {'a_only': n01, 'b_only': n10, 'test': 'exact McNemar'}
    else:
        signs = rng.choice([-1.0, 1.0], (B, len(d)))
        null = np.abs((signs * d).mean(1))
        p = float((null >= abs(d.mean()) - 1e-12).mean())
        extra = {'test': 'sign-flip permutation'}
    return {'diff': float(d.mean()), 'ci': [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
            'p': float(p), **extra}


def summarize(run: Path) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for bench in BENCHES:
        rows = read_jsonl(run / bench / 'results.jsonl')
        if not rows:
            continue
        by = {}
        for r in rows:
            by.setdefault(r['cond'], {}).setdefault(r['case_id'], r)      # duplicates: keep the first row
        conds = [c for c in ORDER if c in by]
        ids = sorted(set.intersection(*[set(by[c]) for c in conds]))
        res = {'n_cases': len(ids), 'conditions': {}}
        for c in conds:
            rs = [by[c][i] for i in ids]
            x = np.array([float(r['correct']) for r in rs])
            binary = set(np.unique(x)) <= {0.0, 1.0}
            d = {'n': len(x), 'acc': float(x.mean()),
                 'ci': wilson(x.sum(), len(x)) if binary else boot_mean(x, rng),
                 'L': float(np.mean([r['L'] for r in rs if r.get('L')])) if any(r.get('L') for r in rs) else None,
                 'extra_visual': float(np.mean([r.get('n_extra_visual', 0) for r in rs])),
                 'vit_patches': float(np.mean([r.get('vit_patches', 0) for r in rs])),
                 'truncated': int(sum(r.get('finish_reason') == 'length' for r in rs)),
                 'no_call': int(sum(bool(r.get('no_call')) for r in rs))}
            d['kv_mib'] = d['L'] * KV_PER_TOKEN / 2 ** 20 if d['L'] else None
            if bench == 'gqa':
                # Lenient GQA score: the normalized reference occurs as a word sequence inside the prediction.
                from analysis_tools.reread import cached_reread as cr
                gold = {c_['id']: c_['answer'] for c_ in json.loads((run / bench / 'cases.json').read_text())}
                xl = np.array([float(f" {cr.vqa_normalize(gold[r['case_id']])} " in f" {cr.vqa_normalize(str(r.get('pred') or ''))} ")
                               for r in rs])
                d['acc_lenient'], d['ci_lenient'] = float(xl.mean()), wilson(xl.sum(), len(xl))
            if bench == 'babyvision':
                xd = np.array([float(r['correct_diag']) for r in rs])
                d['acc_diag'], d['ci_diag'] = float(xd.mean()), wilson(xd.sum(), len(xd))
            if bench in ('syn_color', 'syn_gap'):
                s = np.array([r['s'] if r.get('s') is not None else 0.0 for r in rs])
                y = np.array([r['label'] for r in rs])
                bs = []
                for _ in range(1000):
                    k = rng.integers(0, len(s), len(s))
                    bs.append(auroc(s[k], y[k]))
                d['auroc'] = auroc(s, y)
                d['auroc_ci'] = [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))]
                d['yes_rate'] = float(np.mean([r.get('pred') == 'yes' for r in rs]))
            if bench == 'syn_loc':
                d['frame_error'] = int(sum(bool(r.get('frame_error')) for r in rs))
            cov = [r['coverage'] for r in rs if r.get('coverage') is not None]
            if cov:
                d['coverage_mean'] = float(np.mean(cov))
                d['coverage_full'] = float(np.mean([v >= 0.99 for v in cov]))
            win = [r for r in rs if r.get('window')]
            if win:
                d['window_area_frac'] = float(np.mean([w['window'][2] * w['window'][3] / (w['grid'][0] * w['grid'][1])
                                                       for w in win]))
            if bench == 'mmvp':
                pairs = {}
                for r in rs:
                    pairs.setdefault((int(r['case_id'].split('_')[1]) - 1) // 2, []).append(r['correct'])
                d['pair_acc'] = float(np.mean([all(v) for v in pairs.values()]))
            res['conditions'][c] = d
        comps = []
        for a, b in PRIMARY:
            if a in by and b in by:
                comps.append({'a': a, 'b': b, **paired(np.array([float(by[a][i]['correct']) for i in ids]),
                                                     np.array([float(by[b][i]['correct']) for i in ids]), rng)})
        for c, p in zip(comps, holm([c['p'] for c in comps])):
            c['p_holm'] = p
        res['paired'] = comps
        out[bench] = res
    # Pooled paired comparisons over the natural benchmarks (TextVQA, GQA, MMVP, POPE): per-question score
    # differences are concatenated; the test is a sign-flip permutation test on the mean difference.
    pooled = []
    for a, b in [('c27', 'crop1'), ('h27', 'crop2'), ('h27', 'full2x'), ('crop2', 'base'), ('h27', 'base'),
                 ('c9', 'crop1'), ('h9', 'crop2'), ('h9', 'h9n')]:
        xa, xb = [], []
        for bench in ['textvqa', 'gqa', 'mmvp', 'pope']:
            rows = read_jsonl(run / bench / 'results.jsonl')
            ra = {r['case_id']: float(r['correct']) for r in rows if r['cond'] == a}
            rb = {r['case_id']: float(r['correct']) for r in rows if r['cond'] == b}
            for i in sorted(set(ra) & set(rb)):
                xa.append(ra[i])
                xb.append(rb[i])
        if xa:
            r = paired(np.array(xa), np.array(xb), rng)
            r.pop('a_only', None), r.pop('b_only', None)
            pooled.append({'a': a, 'b': b, 'n': len(xa), **r, 'test': 'sign-flip permutation'})
    out['pooled_natural'] = pooled
    timing = read_jsonl(run / 'timing' / 'timing.jsonl')
    if timing:
        t = {}
        for bench in sorted({r['bench'] for r in timing}):
            t[bench] = {}
            for c in ORDER:
                rs = [r for r in timing if r['bench'] == bench and r['cond'] == c]
                if rs:
                    t[bench][c] = {'n': len(rs), 'sec_median': float(np.median([r['seconds'] for r in rs])),
                                   'sec_mean': float(np.mean([r['seconds'] for r in rs])),
                                   'peak_extra_gib': float(np.mean([r['peak_extra_bytes'] for r in rs]) / 2 ** 30),
                                   'L_final': float(np.mean([r['L_final'] for r in rs])),
                                   'kv_mib': float(np.mean([r['kv_bytes'] for r in rs]) / 2 ** 20)}
        out['timing'] = t
    return out


# ----------------------------------------------------------------------------- figures

def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)


def fig_accuracy(S: dict, fig_dir: Path) -> None:
    benches = [b for b in BENCHES if b in S]
    ncol = 4
    nrow = math.ceil(len(benches) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow), squeeze=False)
    for ax, b in zip(axes.flat, benches):
        cs = S[b]['conditions']
        names = [c for c in ORDER if c in cs]
        key, cik = ('auroc', 'auroc_ci') if b in ('syn_color', 'syn_gap') else ('acc', 'ci')
        for i, c in enumerate(names):
            v, ci = cs[c][key], cs[c][cik]
            fam = FAMILY[c]
            ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt='o', ms=6, color=COLORS[fam], lw=1.5,
                        mfc='white' if fam.startswith('oracle') else COLORS[fam], mew=1.5)
        if 'base' in cs:
            ax.axhline(cs['base'][key], color='#8c8b86', ls='--', lw=1)
        ax.set_xticks(range(len(names)), names, rotation=60, fontsize=8)
        ax.set_title(f"{TITLES[b]} (n={S[b]['n_cases']})", fontsize=10)
        ax.set_ylabel('AUROC of s' if key == 'auroc' else ('VQA accuracy' if b == 'textvqa' else 'accuracy'), fontsize=9)
        _style(ax)
    for ax in axes.flat[len(benches):]:
        ax.axis('off')
    handles = [plt.Line2D([], [], marker='o', ls='', color=COLORS['none'], label='no tool (base, full 2×)'),
               plt.Line2D([], [], marker='o', ls='', color=COLORS['crop'], label='native pixel crop'),
               plt.Line2D([], [], marker='o', ls='', color=COLORS['cache'], label='cached re-read (ours)'),
               plt.Line2D([], [], marker='o', ls='', color='#52514e', mfc='white', label='oracle window (open marker)'),
               plt.Line2D([], [], ls='--', color='#8c8b86', label='base')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(fig_dir / 'accuracy_by_condition.png', dpi=150)
    plt.close(fig)


def fig_cost(S: dict, fig_dir: Path) -> None:
    """Mean accuracy change over the natural benchmarks vs mean prompt tokens and vs wall-clock time."""
    nat = [b for b in ['textvqa', 'gqa', 'mmvp', 'pope'] if b in S]      # BabyVision is at floor (see doc)
    conds = [c for c in ORDER if not c.startswith('o_') and all(c in S[b]['conditions'] for b in nat)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for c in conds:
        dacc = np.mean([S[b]['conditions'][c]['acc'] - S[b]['conditions']['base']['acc'] for b in nat])
        L = np.mean([S[b]['conditions'][c]['L'] for b in nat])
        fam = FAMILY[c]
        axes[0].scatter(L, dacc, s=50, color=COLORS[fam], zorder=3, edgecolor='white', lw=1.5)
        axes[0].annotate(c, (L, dacc), xytext=(4, 4), textcoords='offset points', fontsize=8, color='#0b0b0b')
        if 'timing' in S:
            tb = [b for b in S['timing'] if b in nat and c in S['timing'][b]]
            if tb:
                sec = np.mean([S['timing'][b][c]['sec_median'] for b in tb])
                axes[1].scatter(sec, dacc, s=50, color=COLORS[fam], zorder=3, edgecolor='white', lw=1.5)
                axes[1].annotate(c, (sec, dacc), xytext=(4, 4), textcoords='offset points', fontsize=8)
    axes[0].set_xlabel('mean prompt length at answer time (tokens)')
    axes[1].set_xlabel('median end-to-end time per question, batch size 1 (s)')
    for ax in axes:
        ax.axhline(0, color='#8c8b86', ls='--', lw=1)
        ax.set_ylabel('accuracy − base, mean of TextVQA/GQA/MMVP/POPE')
        _style(ax)
    handles = [plt.Line2D([], [], marker='o', ls='', color=COLORS[k], label=l) for k, l in
               [('none', 'no tool'), ('crop', 'native pixel crop'), ('cache', 'cached re-read (ours)')]]
    axes[0].legend(handles=handles, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'cost_accuracy.png', dpi=150)
    plt.close(fig)


def fig_windows(run: Path, S: dict, fig_dir: Path) -> None:
    """Window choice: area fraction of the chosen window and evidence coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(0, 1, 21)
    for b in [x for x in BENCHES if x in S]:
        uniq = {}
        for r in read_jsonl(run / b / 'results.jsonl'):
            if r['cond'] == 'crop1' and r.get('window'):
                uniq.setdefault(r['case_id'], r)                  # duplicates: keep the first row
        rows = list(uniq.values())
        col = BENCH_COLORS[b]
        if not rows:
            continue
        af = [r['window'][2] * r['window'][3] / (r['grid'][0] * r['grid'][1]) for r in rows]
        axes[0].hist(af, bins=bins, histtype='step', lw=1.8, label=TITLES[b], color=col)
        cov = [r['coverage'] for r in rows if r.get('coverage') is not None]
        if cov:
            axes[1].hist(cov, bins=bins, histtype='step', lw=1.8, label=TITLES[b], color=col)
    axes[0].set_xlabel('chosen window area / image area')
    axes[1].set_xlabel('fraction of evidence box inside chosen window')
    for ax in axes:
        ax.set_ylabel('questions')
        ax.legend(frameon=False, fontsize=7)
        _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / 'window_choice.png', dpi=150)
    plt.close(fig)


def fig_cases(run: Path, fig_dir: Path, picks: list[tuple[str, str]]) -> None:
    """Input examples: the image (as fed to the model) with the first window of crop2 and h9, the evidence box,
    and the final answers of several conditions."""
    from PIL import Image
    from analysis_tools.reread import cached_reread as cr
    conds = ['base', 'full2x', 'crop1', 'crop2', 'c27', 'c9', 'h27', 'h9', 'o_crop2', 'o_h27']
    picks = [p for p in picks if (run / p[0] / 'results.jsonl').exists()]
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 3.8 * len(picks)), squeeze=False,
                             gridspec_kw={'width_ratios': [1.1, 1]})
    for (bench, cid), (a0, a1) in zip(picks, axes):
        rows = {r['cond']: r for r in read_jsonl(run / bench / 'results.jsonl') if r['case_id'] == cid}
        meta = {c['id']: c for c in json.loads((run / bench / 'cases.json').read_text())}[cid]
        im = cr.resize_budget(Image.open(meta['image_path']).convert('RGB'), 512 * 512)
        a0.imshow(im)
        H, W = rows['base']['grid']
        sx, sy = im.width / 1000, im.height / 1000
        for c, col, ls in [('crop2', COLORS['crop'], '-'), ('h9', COLORS['cache'], '--')]:
            r = rows.get(c)
            if r and r.get('windows'):
                for w in r['windows']:
                    a0.add_patch(plt.Rectangle((w[1] * 32, w[0] * 32), w[3] * 32, w[2] * 32, fill=False, ec=col,
                                               lw=2, ls=ls))
        if meta.get('oracle_box'):
            b = meta['oracle_box']
            a0.add_patch(plt.Rectangle((b[0] * sx, b[1] * sy), (b[2] - b[0]) * sx, (b[3] - b[1]) * sy, fill=False,
                                       ec='#1baf7a', lw=2))
        a0.set_axis_off()
        q = meta['question'].split('\n')[0]
        a0.set_title(f'{TITLES[bench]}: {q[:70]}', fontsize=9)
        ans = meta['answer'] if isinstance(meta['answer'], str) else (meta['answer'][0] if meta['answer'] else '')
        lines = [f'reference: {ans}']
        for c in conds:
            r = rows.get(c)
            if r:
                lines.append(f"{c:7s} [{'✓' if r['correct'] >= 0.5 else '✗'}] calls={r.get('n_calls', 0)}  "
                             f"{str(r.get('pred'))[:38]}")
        a1.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=9)
        a1.set_axis_off()
    handles = [plt.Line2D([], [], color=COLORS['crop'], lw=2, label='windows requested under crop2'),
               plt.Line2D([], [], color=COLORS['cache'], lw=2, ls='--', label='windows requested under h9'),
               plt.Line2D([], [], color='#1baf7a', lw=2, label='evidence box (GQA)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(fig_dir / 'cases.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--figures', type=Path, required=True)
    ap.add_argument('--cases', nargs='*', default=[], help='bench:case_id pairs for the example figure')
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    S = summarize(args.run)
    (args.run / 'summary.json').write_text(json.dumps(S, indent=1))
    fig_accuracy(S, args.figures)
    fig_cost(S, args.figures)
    fig_windows(args.run, S, args.figures)
    fig_cases(args.run, args.figures, [tuple(c.split(':', 1)) for c in args.cases])
    for b in BENCHES:
        if b not in S:
            continue
        cs = S[b]['conditions']
        key = 'auroc' if b in ('syn_color', 'syn_gap') else 'acc'
        print(b, S[b]['n_cases'], ' '.join(f"{c}={cs[c][key]:.3f}" for c in ORDER if c in cs))


if __name__ == '__main__':
    main()
