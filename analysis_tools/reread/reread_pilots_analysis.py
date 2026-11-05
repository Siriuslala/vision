"""Experiment 5.4 analysis: statistics and figures (CPU only).

Reads the run directory written by reread_pilots_run and writes summary.json plus figures.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GRAY, ORANGE, BLUE, AQUA, VIOLET, MAGENTA = '#8a8984', '#eb6834', '#2a78d6', '#1baf7a', '#6250d6', '#e87ba4'
INK, INK2 = '#0b0b0b', '#52514e'
RNG = np.random.default_rng(0)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open()] if p.exists() else []


def boot_mean(x: np.ndarray, n: int = 5000) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return float('nan'), float('nan'), float('nan')
    idx = RNG.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def perm_p(d: np.ndarray, n: int = 20000) -> float:
    """Two-sided sign-flip permutation p value for a paired mean difference."""
    d = np.asarray(d, float)
    if len(d) == 0 or not np.any(d):
        return 1.0
    s = RNG.choice([-1.0, 1.0], (n, len(d)))
    null = np.abs((s * d).mean(1))
    return float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


def by_cond(rows: list[dict]) -> dict:
    out = collections.defaultdict(dict)
    for r in rows:
        out[r['cond']][r['case_id']] = r
    return out


def paired(bc: dict, a: str, b: str, key: str = 'correct', subset=None) -> dict:
    ids = sorted(set(bc[a]) & set(bc[b]))
    if subset is not None:
        ids = [i for i in ids if i in subset]
    d = np.array([bc[a][i][key] - bc[b][i][key] for i in ids])
    m, lo, hi = boot_mean(d)
    return {'diff': m, 'lo': lo, 'hi': hi, 'p': perm_p(d), 'n': len(ids)}


# ----------------------------------------------------------------------------- option 1

def analyse_train(run: Path, figs: Path, out: dict) -> None:
    rows = read_jsonl(run / 'train' / 'loss.jsonl')
    if not rows:
        return
    keys = [k for k in rows[0] if k.startswith('rel_')]
    win = 25
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {'rel_u27': GRAY, 'rel_l27': BLUE, 'rel_l27_0': ORANGE, 'rel_l27_9': AQUA}
    labels = {'rel_u27': 'u27 (nearest, no training)', 'rel_l27': 'learned, block 27',
              'rel_l27_0': 'learned, block 27 + block 0', 'rel_l27_9': 'learned, block 27 + block 9'}
    for k in keys:
        y = np.array([r[k] for r in rows])
        ys = np.convolve(y, np.ones(win) / win, mode='valid')
        ax.plot(np.arange(len(ys)) + win - 1, ys, color=colors.get(k, INK), lw=2, label=labels.get(k, k))
    ax.set_xlabel('training step (batch of 16 images)')
    ax.set_ylabel('relative squared error vs 2x-cache tokens')
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Learned 2x module: training error (moving average of 25 steps)', fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / 'train_loss.png', dpi=150)
    plt.close(fig)
    last = rows[-100:]
    out['train_last100'] = {k: float(np.mean([r[k] for r in last])) for k in keys}
    meta = run / 'train' / 'meta_train.json'
    if meta.exists():
        m = json.loads(meta.read_text())
        out['train_meta'] = {k: m[k] for k in ('n_files', 'steps', 'seconds', 'n_params', 'hparams')}


def analyse_fidelity(run: Path, figs: Path, out: dict) -> None:
    rows = read_jsonl(run / 'fidelity' / 'fidelity.jsonl')
    if not rows:
        return
    res = {}
    for tgt in ('cache2x', 'crop2'):
        readers = sorted({k[len(tgt) + 1:-3] for k in rows[0] if k.startswith(tgt + '_') and k.endswith('_se')})
        ss = np.array([r[f'{tgt}_ss'] for r in rows])
        base = np.array([r[f'{tgt}_u27_se'] for r in rows])
        res[tgt] = {}
        for rd in readers:
            se = np.array([r[f'{tgt}_{rd}_se'] for r in rows])
            # relative error: summed squared error / summed squared deviation of the target tokens from the
            # window mean; gap closure: share of u27's squared error removed (0 = u27, 1 = exact)
            closure = 1 - se / base
            res[tgt][rd] = {'rel_err': float(se.sum() / ss.sum()), 'gap_closure': float(1 - se.sum() / base.sum()),
                            'gap_closure_ci': boot_mean(closure)[1:],
                            'cos': float(np.mean([r[f'{tgt}_{rd}_cos'] for r in rows]))}
    out['fidelity'] = res
    out['fidelity_n'] = len(rows)
    order = [k for k in ('u27', 'b27', 'rep27', 'l27', 'l27_9', 'l27_0', 'h27') if k in res['crop2']]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    col = {'u27': GRAY, 'b27': '#b5b4ae', 'rep27': '#d9d8d4', 'l27': BLUE, 'l27_0': ORANGE, 'l27_9': AQUA, 'h27': VIOLET}
    for ax, tgt, title in ((axs[0], 'cache2x', 'target: 2x-cache tokens (training target)'),
                           (axs[1], 'crop2', 'target: 2x crop re-encode tokens')):
        names = [k for k in order if k in res[tgt]]
        vals = [res[tgt][k]['rel_err'] for k in names]
        ax.bar(range(len(names)), vals, color=[col[k] for k in names], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=8, color=INK)
        ax.set_xticks(range(len(names)), names)
        ax.set_ylabel('relative squared error')
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'Feature fidelity on {len(rows)} held-out TextVQA val windows (lower is better)', fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / 'fidelity.png', dpi=150)
    plt.close(fig)


def analyse_opt1(run: Path, figs: Path, out: dict) -> None:
    rows = read_jsonl(run / 'opt1_textvqa' / 'results.jsonl')
    if not rows:
        return
    bc = by_cond(rows)
    acc = {c: boot_mean([r['correct'] for r in v.values()]) + (len(v),) for c, v in bc.items()}
    ref = bc.get('crop1') or next(iter(bc.values()))
    small = {i for i, r in bc['c27'].items() if r.get('window_area') is not None and r['window_area'] <= 0.15}
    comps = [(a, b) for a, b in [('crop2', 'crop1'), ('crop2', 'c27'), ('h27', 'c27'), ('u27', 'c27'), ('l27', 'c27'),
                                 ('l27_0', 'c27'), ('l27_9', 'c27'), ('l27', 'crop2'), ('l27_0', 'crop2'),
                                 ('l27_9', 'crop2'), ('c27', 'crop1'), ('h27', 'crop2')] if a in bc and b in bc]
    out['opt1'] = {'acc': acc, 'n_small': len(small),
                   'paired_all': {f'{a}-{b}': paired(bc, a, b) for a, b in comps},
                   'paired_small': {f'{a}-{b}': paired(bc, a, b, subset=small) for a, b in comps},
                   'acc_small': {c: boot_mean([v[i]['correct'] for i in small if i in v]) for c, v in bc.items()}}
    order = [c for c in ('base', 'crop1', 'c27', 'u27', 'l27', 'l27_9', 'l27_0', 'h27', 'crop2') if c in bc]
    col = {'base': GRAY, 'crop1': ORANGE, 'crop2': ORANGE, 'c27': BLUE, 'u27': BLUE, 'h27': VIOLET,
           'l27': AQUA, 'l27_0': AQUA, 'l27_9': AQUA}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, sub, title in ((axs[0], None, f'all {len(ref)} questions'),
                           (axs[1], small, f'small windows (area <= 0.15, n = {len(small)})')):
        for i, c in enumerate(order):
            vals = [r['correct'] for k, r in bc[c].items() if sub is None or k in sub]
            m, lo, hi = boot_mean(vals)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt='o', color=col[c], ms=8, capsize=3, lw=1.5,
                        mfc='white' if c in ('crop1', 'u27') else col[c])
        ax.set_xticks(range(len(order)), order, rotation=30)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    axs[0].set_ylabel('TextVQA VQA accuracy')
    fig.suptitle('Option 1: learned 2x readers vs crop re-encode (closed-call lean protocol, 512^2 budget)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / 'opt1_textvqa.png', dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- option 2

def analyse_opt2(run: Path, figs: Path, out: dict, tag: str = '', budget: str = '1024^2') -> None:
    rows = read_jsonl(run / f'opt2_doc{tag}' / 'results.jsonl')
    if not rows:
        return
    bc = by_cond(rows)
    acc = {c: {'anls': boot_mean([r['correct'] for r in v.values()]), 'em': boot_mean([r['em'] for r in v.values()]),
               'n': len(v), 'vit_gflops': float(np.mean([r['vit_flops'] for r in v.values()]) / 1e9),
               'L': float(np.mean([r['L'] for r in v.values()])),
               'coverage': float(np.nanmean([r['coverage'] if r.get('coverage') is not None else np.nan
                                             for r in v.values()])) if any('coverage' in r for r in v.values()) else None,
               'capped': int(sum(bool(r.get('capped')) for r in v.values()))}
           for c, v in bc.items()}
    comps = [(a, b) for a, b in [('base_hi', 'base_lo'), ('lo_c27', 'base_lo'), ('lo_crop2', 'base_lo'),
                                 ('lo_crop_hi', 'base_lo'), ('lo_cache_hi', 'base_lo'), ('lo_cache_hi', 'lo_crop_hi'),
                                 ('lo_crop_hi', 'base_hi'), ('lo_cache_hi', 'base_hi'), ('hi_c27', 'hi_crop1'),
                                 ('hi_c27', 'base_hi'), ('hi_crop1', 'base_hi'), ('lo_c27', 'lo_crop2'),
                                 ('lo_l27_9', 'lo_crop2'), ('lo_l27_9', 'lo_c27'), ('lo_crop2', 'lo_crop_hi')] if a in bc and b in bc]
    out['opt2' + tag] = {'acc': acc, 'paired': {f'{a}-{b}': paired(bc, a, b) for a, b in comps}}
    tim = read_jsonl(run / f'opt2_timing{tag}' / 'timing.jsonl')
    order = [c for c in ('base_lo', 'lo_c27', 'lo_crop2', 'lo_crop_hi', 'lo_cache_hi', 'base_hi', 'hi_c27', 'hi_crop1',
                         'lo_l27', 'lo_l27_0', 'lo_l27_9') if c in bc]
    if tim:
        tc = collections.defaultdict(list)
        for r in tim:
            tc[r['cond']].append(r)
        out['opt2_timing' + tag] = {c: {k: float(np.median([r[k] for r in v])) for k in
                                  ('seconds', 't_vit', 't_call', 't_tool', 't_answer', 'peak_extra_bytes', 'kv_bytes')}
                              | {'n': len(v)} for c, v in tc.items()}
        # paired timing differences
        tid = {c: {r['case_id']: r['seconds'] for r in v} for c, v in tc.items()}
        pt = {}
        for a, b in [('lo_cache_hi', 'lo_crop_hi'), ('hi_c27', 'hi_crop1'), ('lo_crop_hi', 'base_hi'),
                     ('lo_cache_hi', 'base_hi'), ('lo_c27', 'lo_crop2')]:
            if a in tid and b in tid:
                ids = sorted(set(tid[a]) & set(tid[b]))
                d = np.array([tid[a][i] - tid[b][i] for i in ids])
                pt[f'{a}-{b}'] = {'median_diff': float(np.median(d)), 'frac_faster': float(np.mean(d < 0)), 'n': len(ids)}
        out['opt2_timing_paired' + tag] = pt
    # figure: accuracy vs end-to-end time
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1.1, 1]})
    col = {'base_lo': GRAY, 'base_hi': '#3d3d3a', 'lo_c27': BLUE, 'lo_crop2': ORANGE, 'lo_crop_hi': ORANGE,
           'lo_cache_hi': BLUE, 'hi_c27': BLUE, 'hi_crop1': ORANGE, 'lo_l27': AQUA, 'lo_l27_0': AQUA, 'lo_l27_9': AQUA}
    hollow = {'lo_crop2', 'lo_c27', 'hi_crop1', 'hi_c27'}
    ax = axs[0]
    for i, c in enumerate(order):
        m, lo, hi = acc[c]['anls']
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt='o', color=col[c], ms=8, capsize=3,
                    mfc='white' if c in hollow else col[c])
    ax.set_xticks(range(len(order)), order, rotation=30)
    ax.set_ylabel('DocVQA ANLS')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'accuracy (n = {acc[order[0]]["n"]}, 95% bootstrap CI)', fontsize=9)
    ax = axs[1]
    if tim:
        stages = [('t_vit', '#b5b4ae', 'ViT passes (+ module)'), ('t_call', '#d9d8d4', 'prefill + call decoding'),
                  ('t_tool', ORANGE, 'tool (crop ViT or cache slice)'), ('t_answer', BLUE, 'continuation + answer')]
        T = out['opt2_timing' + tag]
        tcs = [c for c in order if c in T]
        bottom = np.zeros(len(tcs))
        for k, colr, lab in stages:
            v = np.array([T[c][k] for c in tcs])
            ax.bar(range(len(tcs)), v, bottom=bottom, color=colr, label=lab, width=0.6, edgecolor='white', lw=1)
            bottom += v
        ax.set_xticks(range(len(tcs)), tcs, rotation=30)
        ax.set_ylabel('median seconds per question (batch size 1)')
        ax.legend(frameon=False, fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title('end-to-end time by stage (stage medians, stacked)', fontsize=9)
    fig.suptitle(f'Option 2: DocVQA, lo = {budget} budget view, hi = 2x lo sides from original pixels', fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / f'opt2_doc{tag}.png', dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--figures', type=Path, required=True)
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    analyse_train(args.run, args.figures, out)
    analyse_fidelity(args.run, args.figures, out)
    analyse_opt1(args.run, args.figures, out)
    analyse_opt2(args.run, args.figures, out)
    analyse_opt2(args.run, args.figures, out, '_512', '512^2')
    (args.run / 'summary.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:6000])


if __name__ == '__main__':
    main()
