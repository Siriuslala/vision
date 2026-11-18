"""Experiment 7.5 CPU analysis: premask table, training tables, mechanism checks, answer-format pilot, figures.

Results doc: docs/analysis/application/results_2026-11-11_new_optimizations.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.application import new_opt as no

BINS = [b[0] for b in no.RASTER_BINS]
BENCHES = ('pope', 'textvqa', 'mmvp', 'gqa')
SYNTH_COORD = ('flat', 'clutter', 'chart_ui', 'sweep_h', 'sweep_v', 'grid06', 'grid10', 'grid20')
ADAPTERS = ('fa8qk_s1', 'fa8qk_s2', 'fa12qk_s1', 'fa24qk_s1', 'mlp8_15_s1')
LABELS = {'base': 'original model', 'fa8qk_s1': 'L8 q/k, seed 1 (registered)', 'fa8qk_s2': 'L8 q/k, seed 2',
          'fa12qk_s1': 'L12 q/k (location control)', 'fa24qk_s1': 'L24 q/k (location control)',
          'mlp8_15_s1': 'MLP 8-15 (component control)', 'late_n1200_s1': 'ViT blocks 25-27 MLP (7.3)',
          'vitall_n1200_s1': 'all ViT, 4M params (7.3)'}
VIT_REFERENCE = {'late_n1200_s1': 'ViT block 25-27 MLP (7.3)', 'vitall_n1200_s1': 'all ViT (7.3)'}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def prop(rows, key='correct') -> dict:
    k, n = sum(bool(r.get(key)) for r in rows), len(rows)
    p, lo, hi = no.wilson(k, n)
    return {'k': k, 'n': n, 'p': p, 'lo': lo, 'hi': hi}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired booleans (b better than a counts as 'up')."""
    from math import comb
    up = sum(1 for x, y in zip(a, b) if not x and y)
    down = sum(1 for x, y in zip(a, b) if x and not y)
    n = up + down
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(up, down) + 1)) / 2 ** n)
    return {'up': up, 'down': down, 'p': p}


def by_bin(rows, variant='swap', split=None) -> dict:
    sel = [r for r in rows if r['variant'] == variant and (split is None or r['split'] == split)]
    return {b: prop([r for r in sel if r['bin'] == b]) for b in BINS} | {'all': prop(sel)}


# ----------------------------------------------------------------------------- premask

def premask_summary(run: Path) -> dict:
    out = {}
    d = run / 'premask'
    for p in sorted(d.glob('*.jsonl')):
        model, cond = p.stem.split('__')
        rows = read_jsonl(p)
        out.setdefault(model, {})[cond] = {'swap': by_bin(rows, 'swap'), 'std': by_bin(rows, 'std'),
                                           'rows': {r['id']: r['correct'] for r in rows}}
    for model, conds in out.items():
        base = conds.get('none')
        if base is None:
            continue
        for cond, s in conds.items():
            if cond == 'none':
                continue
            s['mcnemar'] = {}
            rows = read_jsonl(d / f'{model}__{cond}.jsonl')
            brows = {r['id']: r for r in read_jsonl(d / f'{model}__none.jsonl')}
            for v in ('swap', 'std'):
                for b in BINS:
                    ids = [r['id'] for r in rows if r['variant'] == v and r['bin'] == b]
                    s['mcnemar'][f'{v}|{b}'] = mcnemar([brows[i]['correct'] for i in ids],
                                                      [s['rows'][i] for i in ids])
    for conds in out.values():
        for s in conds.values():
            s.pop('rows')
    return out


def own_mass_profile(rows: list[dict], heads=no.L8_TEMPLATE_HEADS) -> dict:
    """Mean layer-8 own-entity and earlier-entity attention mass per entity raster index (homogeneous summary over
    all probe images; every image contributes all 36 entities)."""
    own = np.array([r['own8'] for r in rows if 'own8' in r])        # (N, H, 36)
    prev = np.array([r['prev8'] for r in rows if 'prev8' in r])
    if own.size == 0:
        return {}
    h = list(heads)
    return {'own_template': own[:, h].mean((0, 1)).round(4).tolist(),
            'prev_template': prev[:, h].mean((0, 1)).round(4).tolist(),
            'own_per_head': own.mean(0).round(4).tolist()}


def reader_by_bin(rows: list[dict], variant='swap') -> dict:
    out = {}
    for b in BINS:
        v = [np.mean(r['reader_target']) for r in rows if r['variant'] == variant and r['bin'] == b and
             'reader_target' in r]
        out[b] = no.bootstrap_mean_ci(np.array(v))
    return out


# ----------------------------------------------------------------------------- adapters

def eval_summary(d: Path) -> dict:
    s = {}
    probe = read_jsonl(d / 'probe.jsonl')
    if probe:
        for r in probe:
            r['bin'] = no.raster_bin(r['target'])
        s['probe_swap'] = by_bin(probe, 'swap')
        s['probe_swap_train'] = by_bin(probe, 'swap', 'train_feature')
        s['probe_std'] = by_bin(probe, 'std')
        s['probe_swap_split'] = {sp: prop([r for r in probe if r['variant'] == 'swap' and r['split'] == sp])
                                 for sp in sorted({r['split'] for r in probe})}
    gen = read_jsonl(d / 'general.jsonl')
    if gen:
        s['general'] = {b: prop([r for r in gen if r['bench'] == b]) for b in BENCHES}
    txt = read_jsonl(d / 'text.jsonl')
    if txt:
        s['text'] = prop(txt)
    coord = read_jsonl(d / 'coord.jsonl')
    if coord:
        s['coord_synth'] = prop([r for r in coord if r['family'] in SYNTH_COORD], 'hit')
        s['coord_refcoco'] = prop([r for r in coord if r['family'] == 'refcoco'], 'hit')
    cnt = read_jsonl(d / 'count.jsonl')
    if cnt:
        s['count'] = prop(cnt)
    bv = read_jsonl(d / 'babyvision.jsonl')
    if bv:
        s['babyvision'] = {'k': sum(r['score']['correct'] for r in bv), 'n': len(bv)}
    return s


def paired_probe(base_d: Path, d: Path, variant='swap') -> dict:
    b = {r['id']: r for r in read_jsonl(base_d / 'probe.jsonl')}
    a = read_jsonl(d / 'probe.jsonl')
    out = {}
    for bn in BINS + ['all']:
        ids = [r['id'] for r in a if r['variant'] == variant and (bn == 'all' or no.raster_bin(r['target']) == bn)]
        am = {r['id']: r['correct'] for r in a}
        out[bn] = mcnemar([b[i]['correct'] for i in ids], [am[i] for i in ids])
    return out


# ----------------------------------------------------------------------------- answer formats

def format_summary(run: Path) -> dict:
    d = run / 'format'
    styles = {s: {r['id']: r for r in read_jsonl(d / f'{s}.jsonl')} for s in ('list', 'key', 'bare')}
    if not all(styles.values()):
        return {}
    out = {}
    for group, fams in (('synthetic', SYNTH_COORD), ('refcoco', ('refcoco',))):
        for fmt in ('point', 'bbox'):
            ids = [i for i, r in styles['list'].items() if r['family'] in fams and r['fmt'] == fmt]
            if not ids:
                continue
            g = {}
            for s, rows in styles.items():
                sel = [rows[i] for i in ids]
                g[s] = {'valid': prop(sel, 'valid'),
                        'err': [no.bootstrap_mean_ci(np.array([r['abs_err'][j] for r in sel if r['valid']]))
                                for j in range(2 if fmt == 'point' else 4)]}
                if fmt == 'point':
                    g[s]['hit'] = prop([r for r in sel if r['valid']], 'hit')
                else:
                    g[s]['iou'] = no.bootstrap_mean_ci(np.array([r['iou'] for r in sel if r['valid']]))
            # paired per-value error differences against the native list style (both valid)
            for s in ('key', 'bare'):
                both = [i for i in ids if styles['list'][i]['valid'] and styles[s][i]['valid']]
                g[f'{s}-list'] = {'n': len(both), 'diff': [no.bootstrap_mean_ci(np.array(
                    [styles[s][i]['abs_err'][j] - styles['list'][i]['abs_err'][j] for i in both]))
                    for j in range(2 if fmt == 'point' else 4)]}
            out[f'{group}|{fmt}'] = g
    return out


# ----------------------------------------------------------------------------- figures

def figures(res: dict, run: Path, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10})
    colors = {'none': '#4a4a4a', 'own_l8_h691': '#d1495b', 'own_l8_all': '#edae49', 'own_l4_all': '#00798c',
              'own_l12_all': '#66a182'}
    # 1. premask accuracy by raster bin (swap and std)
    pm = res['premask'].get('Qwen3.5-9B', {})
    if pm:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, v in zip(axes, ('swap', 'std')):
            for ci, (cond, s) in enumerate(pm.items()):
                x = np.arange(len(BINS)) + (ci - len(pm) / 2) * 0.12
                p = np.array([s[v][b]['p'] for b in BINS])
                lo = np.array([s[v][b]['lo'] for b in BINS])
                hi = np.array([s[v][b]['hi'] for b in BINS])
                ax.errorbar(x, p, yerr=[p - lo, hi - p], fmt='o', ms=4, capsize=2,
                            color=colors.get(cond, None), label=cond)
            ax.set_xticks(range(len(BINS)), ['cell 1', 'cells 2-6', 'cells 7-12', 'cells 13-36'])
            ax.set_title({'swap': 'target lacks the part (comparison class)',
                          'std': 'target has the part (detector class)'}[v])
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel('teacher-forced accuracy (row and column)')
        axes[1].legend(fontsize=8, loc='lower right')
        fig.tight_layout()
        fig.savefig(fig_dir / 'premask_accuracy.png', dpi=160)
        plt.close(fig)
    # 2. layer-8 own-entity mass by entity index (base vs adapters)
    prof = res.get('own_profile', {})
    if prof:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for name, p in prof.items():
            if not p:
                continue
            if name not in ('base', 'fa8qk_s1', 'fa8qk_s2'):
                continue
            lab = LABELS.get(name, name)
            ax.plot(range(1, 37), p['own_template'], marker='o', ms=3, label=f'{lab}: own entity')
            ax.plot(range(1, 37), p['prev_template'], ls='--', lw=1, label=f'{lab}: earlier entities')
        ax.set_xlabel('entity raster index in the 6 x 6 probe')
        ax.set_ylabel('attention mass (mean of L8 H6/H9/H1)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / 'l8_own_mass.png', dpi=160)
        plt.close(fig)
    # 3. adapter accuracy by bin
    ad = res.get('adapters', {})
    if ad:
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        names = [n for n in ad if 'probe_swap' in ad[n]]
        for ci, n in enumerate(names):
            x = np.arange(len(BINS)) + (ci - len(names) / 2) * 0.1
            s = ad[n]['probe_swap']
            p = np.array([s[b]['p'] for b in BINS])
            lo = np.array([s[b]['lo'] for b in BINS])
            hi = np.array([s[b]['hi'] for b in BINS])
            ax.errorbar(x, p, yerr=[p - lo, hi - p], fmt='o', ms=4, capsize=2, label=LABELS.get(n, n))
        ax.set_xticks(range(len(BINS)), ['cell 1', 'cells 2-6', 'cells 7-12', 'cells 13-36'])
        ax.set_ylabel('accuracy, target lacks the part')
        ax.set_title('6 x 6 probe, greedy generation; bars = Wilson 95% CI')
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / 'adapter_probe_bins.png', dpi=160)
        plt.close(fig)
    # 4. answer formats: per-value error
    fm = res.get('format', {})
    if fm:
        keys = list(fm)
        fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.4))
        axes = np.atleast_1d(axes)
        sc = {'list': '#4a4a4a', 'key': '#00798c', 'bare': '#d1495b'}
        for ax, k in zip(axes, keys):
            g = fm[k]
            nval = len(g['list']['err'])
            for si, s in enumerate(('bare', 'list', 'key')):
                e = g[s]['err']
                x = np.arange(nval) + (si - 1) * 0.22
                m = np.array([v[0] for v in e])
                lo = np.array([v[1] for v in e])
                hi = np.array([v[2] for v in e])
                ax.bar(x, m, 0.22, yerr=[m - lo, hi - m], color=sc[s], capsize=2, label=s)
            ax.set_xticks(range(nval), ['x', 'y'] if nval == 2 else ['x_min', 'y_min', 'x_max', 'y_max'])
            ax.set_title(k.replace('|', ', '))
            ax.set_ylabel('mean |error| (0-1000 units)')
            ax.grid(alpha=0.3, axis='y')
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / 'format_errors.png', dpi=160)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--vit-run', required=True)
    p.add_argument('--figures', required=True)
    args = p.parse_args()
    run, vit = Path(args.run), Path(args.vit_run)
    res = {'premask': premask_summary(run)}
    base_pm = read_jsonl(run / 'premask' / 'Qwen3.5-9B__none.jsonl')
    res['own_profile'] = {'base': own_mass_profile(base_pm)}
    res['reader_bins'] = {'base': reader_by_bin(base_pm)}
    res['adapters'] = {'base': eval_summary(run / 'eval' / 'base')}
    res['paired'] = {}
    for n in ADAPTERS:
        d = run / 'eval' / n
        if not d.exists():
            continue
        res['adapters'][n] = eval_summary(d)
        res['paired'][n] = paired_probe(run / 'eval' / 'base', d)
        at = read_jsonl(d / 'attn.jsonl')
        if at:
            res['own_profile'][n] = own_mass_profile(at)
            res['reader_bins'][n] = reader_by_bin(at)
    for n in VIT_REFERENCE:
        d = vit / 'eval' / n
        if d.exists():
            res['adapters'][n] = eval_summary(d)
            res['paired'][n] = paired_probe(run / 'eval' / 'base', d)
    for n in list(res['adapters']):
        a = run / 'adapters' / f'{n}.pt'
        if a.exists():
            import torch
            st = torch.load(a, map_location='cpu', weights_only=False)
            res['adapters'][n]['n_params'] = st['n_params']
            res['adapters'][n]['train_seconds'] = st['train_seconds']
    res['decomp'] = {}
    for n in ADAPTERS:
        rows = read_jsonl(run / 'eval' / n / 'decomp.jsonl')
        if not rows:
            continue
        g = json.loads((run / 'eval' / n / 'decomp_gates.json').read_text())
        res['decomp'][n] = {'gates': g, 'parts': {}}
        for part in ('base', 'full', 'gate', 'query_key'):
            pr = [r for r in rows if r['part'] == part]
            res['decomp'][n]['parts'][part] = {'swap': by_bin(pr, 'swap'), 'std': by_bin(pr, 'std'),
                                               'reader': reader_by_bin(pr)}
    res['format'] = format_summary(run)
    (run / 'results.json').write_text(json.dumps(res, indent=1))
    figures(res, run, Path(args.figures))
    print(json.dumps({k: list(v) if isinstance(v, dict) else v for k, v in res.items()}, indent=0)[:2000])


if __name__ == '__main__':
    main()
