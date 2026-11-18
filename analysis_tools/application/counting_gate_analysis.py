"""Experiment 7.3 Part C CPU analysis: layer sweep, condition tables, count probe, gate statistics, figures.

Results doc: docs/analysis/application/results_2026-11-06_counting_gate_finetune.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.application import counting_gate_run as cgr
from analysis_tools.application import interp_opt as io_
from analysis_tools.application import located_finetune_analysis as lfa

K_BINS = ((1, 5), (6, 11), (12, 20))
BENCHES = ('pope', 'textvqa', 'mmvp', 'gqa')


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def prop(rows, key='correct'):
    k, n = sum(bool(r[key]) for r in rows), len(rows)
    p, lo, hi = io_.wilson(k, n)
    return {'k': k, 'n': n, 'p': p, 'lo': lo, 'hi': hi}


def ridge_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float = 10.0, n_pc: int = 128) -> np.ndarray:
    """5-fold (by group) ridge predictions on the top principal components of the training folds."""
    pred = np.zeros_like(y, dtype=float)
    ug = np.unique(groups)
    folds = np.array_split(np.random.default_rng(0).permutation(ug), 5)
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        mu = X[tr].mean(0)
        _, _, vt = np.linalg.svd(X[tr] - mu, full_matrices=False)
        P = vt[:n_pc].T
        Z = (X[tr] - mu) @ P
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y[tr] - y[tr].mean()))
        pred[te] = ((X[te] - mu) @ P) @ w + y[tr].mean()
    return pred


def probe_summary(path: Path) -> dict:
    z = np.load(path)
    out = {}
    for axis in ('H', 'V'):
        sel = np.array([c.startswith(axis) for c in z['cond']])
        base = sel & np.isin(z['variant'], ['base', 'shift', 'delR'])          # no interior blank before the target
        gapv = sel & np.isin(z['variant'], ['insL', 'delL'])
        groups = np.array([i.split('_b')[1][:3] + c for i, c in zip(z['ids'], z['cond'])])
        for l in cgr.PROBE_LAYERS:
            X = z[f'L{l}'].astype(np.float64)
            k = z['k'].astype(float)
            pred = np.full(len(k), np.nan)
            pred[base] = ridge_cv(X[base], k[base], groups[base])
            # recount weight on images with an interior blank: fit on all no-blank images, apply
            mu = X[base].mean(0)
            _, _, vt = np.linalg.svd(X[base] - mu, full_matrices=False)
            P = vt[:128].T
            Zb = (X[base] - mu) @ P
            w = np.linalg.solve(Zb.T @ Zb + 10 * np.eye(128), Zb.T @ (k[base] - k[base].mean()))
            pg = ((X[gapv] - mu) @ P) @ w + k[base].mean()
            kg, gg = k[gapv], z['gap'][gapv].astype(float)
            m = kg != gg
            A = np.stack([np.ones(m.sum()), kg[m], gg[m]], 1)
            coef = np.linalg.lstsq(A, pg[m], rcond=None)[0]
            res = pred[base] - k[base]
            kb = k[base]
            out[f'{axis}_L{l}'] = {'r2': float(1 - np.var(res) / np.var(kb)),
                                   'sd_k_le5': float(res[kb <= 5].std()), 'sd_k_ge12': float(res[kb >= 12].std()),
                                   'recount_weight': float(coef[2] / (coef[1] + coef[2]))}
    return out


def gate_summary(path: Path, layers: list[int]) -> dict:
    rows = json.loads(path.read_text())
    out = {}
    for cond in ('H64', 'V64'):
        for l in layers:
            r = {(x['variant'], x['cls']): x for x in rows if x['condition'] == cond and x['layer'] == l}
            if ('base', 1) not in r:
                continue
            a1, a2 = np.array(r[('base', 1)]['alpha']), np.array(r[('base', 2)]['alpha'])
            b1, b2 = np.array(r[('base', 1)]['beta']), np.array(r[('base', 2)]['beta'])
            out[f'{cond}_L{l}'] = {'alpha_onset': a1.tolist(), 'alpha_other': a2.tolist(),
                                   'beta_onset': b1.tolist(), 'beta_other': b2.tolist()}
    return out


def update_meta(run: Path) -> dict:
    """adapters/meta.json: scope, parameter count and training time of every trained adapter (kept after uncited
    adapter files are deleted)."""
    path = run / 'adapters' / 'meta.json'
    meta = json.loads(path.read_text()) if path.exists() else {}
    if (run / 'adapters').exists():
        import torch
        for f in (run / 'adapters').glob('*.pt'):
            if f.stem not in meta:
                st = torch.load(f, map_location='cpu', weights_only=False)
                meta[f.stem] = {k: st[k] for k in ('scope', 'seed', 'n_params', 'train_seconds', 'rank', 'gpu') if k in st}
        path.write_text(json.dumps(meta, indent=1))
    return meta


def analyse(run: Path) -> dict:
    res: dict = {}
    sweep = read_jsonl(run / 'sweep' / 'sweep.jsonl')
    if sweep:
        res['sweep'] = {'rows': [{k: v for k, v in r.items() if k != 'answers'} for r in sweep],
                        'choice': cgr.choose_layers(sweep)}
    meta = update_meta(run)
    names = ['base'] + sorted(n for n in meta if (run / 'eval' / n).exists())
    for n in names:
        d = run / 'eval' / n
        r: dict = {'adapter': meta.get(n)}
        ordr = read_jsonl(d / 'ord.jsonl')
        if ordr:
            o = {}
            for cond in ('H32', 'H64', 'V32', 'V64'):
                for v in ('base', 'shift', 'insL', 'delL', 'delR'):
                    sel = [x for x in ordr if x['condition'] == cond and x['variant'] == v]
                    o[f'{cond}_{v}'] = prop(sel)
                    if v in ('insL', 'delL'):
                        o[f'{cond}_{v}_gap'] = prop([dict(x, g=bool(x['match_gap'])) for x in sel], 'g')
                sel = [x for x in ordr if x['condition'] == cond and x['variant'] == 'base']
                for a, b in K_BINS:
                    o[f'{cond}_base_k{a}-{b}'] = prop([x for x in sel if a <= x['c'] <= b])
                o[f'{cond}_base_plus1'] = prop([dict(x, p1=x['err'] == 1) for x in sel], 'p1')
            o['all'] = prop(ordr)
            o['other_correct'] = prop([dict(x, oc=bool(x['other_correct'])) for x in ordr], 'oc')
            r['ord'] = o
        th = read_jsonl(d / 'think.jsonl')
        if th:
            r['think'] = {f'{c}_{v}': prop([x for x in th if x['condition'] == c and x['variant'] == v])
                          for c in cgr.THINK_CONDITIONS for v in cgr.THINK_VARIANTS}
            r['think']['truncated'] = sum(x['finish_reason'] == 'length' for x in th)
        g8 = read_jsonl(d / 'g8.jsonl')
        if g8:
            r['g8'] = {'row': prop(g8, 'row_ok'), 'col': prop(g8, 'col_ok'), 'both': prop(g8)}
        cnt = read_jsonl(d / 'count.jsonl')
        if cnt:
            r['count'] = prop(cnt)
        co = read_jsonl(d / 'coord.jsonl')
        if co:
            syn = [x for x in co if x['family'] != 'refcoco']
            r['coord'] = {'synthetic': prop(syn, 'hit'), 'refcoco': prop([x for x in co if x['family'] == 'refcoco'], 'hit'),
                          'invalid': sum(not x['valid'] for x in co)}
        gen = read_jsonl(d / 'general.jsonl')
        if gen:
            r['general'] = {b: float(np.mean([x['correct'] for x in gen if x['bench'] == b])) for b in BENCHES}
        tx = read_jsonl(d / 'text.jsonl')
        if tx:
            r['text'] = prop(tx)
        bv = read_jsonl(d / 'babyvision.jsonl')
        if bv:
            r['babyvision'] = prop([dict(x, correct=x['score']['correct']) for x in bv])
        if (d / 'probe.npz').exists():
            r['probe'] = probe_summary(d / 'probe.npz')
        res[n] = r
    # paired comparisons vs base
    comps = {}
    for n in names[1:]:
        cm = {}
        for name, key, sel in (('ord_all', 'correct', lambda x: True),
                               ('ord_base_variant', 'correct', lambda x: x['variant'] == 'base'),
                               ('ord_gap_variants', 'correct', lambda x: x['variant'] in ('insL', 'delL'))):
            a, b = read_jsonl(run / 'eval' / 'base' / 'ord.jsonl'), read_jsonl(run / 'eval' / n / 'ord.jsonl')
            if a and b:
                cm[name] = lfa.paired(a, b, sel, key)
        a, b = read_jsonl(run / 'eval' / 'base' / 'think.jsonl'), read_jsonl(run / 'eval' / n / 'think.jsonl')
        if a and b:
            cm['think'] = lfa.paired(a, b, lambda x: True)
        a, b = read_jsonl(run / 'eval' / 'base' / 'g8.jsonl'), read_jsonl(run / 'eval' / n / 'g8.jsonl')
        if a and b:
            cm['g8'] = lfa.paired(a, b, lambda x: True)
        a, b = read_jsonl(run / 'eval' / 'base' / 'coord.jsonl'), read_jsonl(run / 'eval' / n / 'coord.jsonl')
        if a and b:
            cm['coord_synthetic'] = lfa.paired(a, b, lambda x: x['family'] != 'refcoco', 'hit')
        a, b = read_jsonl(run / 'eval' / 'base' / 'general.jsonl'), read_jsonl(run / 'eval' / n / 'general.jsonl')
        if a and b:
            for bench in BENCHES:
                A = {x['id']: x['correct'] for x in a if x['bench'] == bench}
                B = {x['id']: x['correct'] for x in b if x['bench'] == bench}
                ids = sorted(set(A) & set(B))
                dd = np.array([float(B[i]) - float(A[i]) for i in ids])
                rng = np.random.default_rng(0)
                boots = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
                cm[f'general_{bench}'] = {'delta': float(dd.mean()), 'ci': [float(np.percentile(boots, 2.5)),
                                                                             float(np.percentile(boots, 97.5))]}
        comps[n] = cm
    res['_paired_vs_base'] = comps
    return res


def figures(res: dict, figs: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if 'sweep' in res:
        rows = res['sweep']['rows']
        base = next(r for r in rows if r['layer'] == 0)['acc']
        layers = sorted({r['layer'] for r in rows if r['layer']})
        fig, ax = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        cols = {'H64': '#ff7f0e', 'V64': '#1f77b4', 'g8': '#2ca02c'}
        for a, e in zip(ax, cgr.SWEEP_EDITS):
            for s, c in cols.items():
                y = [base[s] - next(r for r in rows if r['layer'] == l and r['edit'] == e)['acc'][s] for l in layers]
                a.plot(range(len(layers)), y, marker='o', ms=3, color=c, label=s)
            for l in res['sweep']['choice']['chosen']:
                a.axvspan(layers.index(l) - 0.4, layers.index(l) + 0.4, color='#dddddd', zorder=0)
            a.axhline(cgr.EFFECT_MIN, color='k', ls=':', lw=1)
            a.set_xticks(range(len(layers)))
            a.set_xticklabels(layers, fontsize=8)
            a.set_xlabel('Gated DeltaNet layer edited (image tokens only)')
            a.set_title({'nodecay': 'no forgetting in non-integrator heads (alpha = 1)', 'nowrite': 'no writing in any head (beta = 0)'}[e])
        ax[0].set_ylabel('accuracy drop of the long-axis ordinal')
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / 'layer_sweep.png', dpi=150)
        plt.close(fig)
    names = [n for n in res if not n.startswith('_') and n != 'sweep' and 'ord' in res[n]]
    if len(names) > 1:
        fig, ax = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
        for a, cond in zip(ax, ('H32', 'H64', 'V32', 'V64')):
            for i, n in enumerate(names):
                ps = [res[n]['ord'][f'{cond}_base_k{lo}-{hi}'] for lo, hi in K_BINS]
                a.errorbar(np.arange(3) + (i - len(names) / 2) * 0.07, [p['p'] for p in ps],
                           yerr=[[max(0, p['p'] - p['lo']) for p in ps], [max(0, p['hi'] - p['p']) for p in ps]], marker='o', capsize=2, label=n)
            a.set_xticks(range(3))
            a.set_xticklabels([f'{lo}-{hi}' for lo, hi in K_BINS])
            a.set_xlabel('true count ordinal of the target')
            a.set_title(f'{cond}, base variant (n=64)')
        ax[0].set_ylabel('long-axis ordinal accuracy')
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / 'ordinal_by_count.png', dpi=150)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--figures', required=True)
    args = p.parse_args()
    run, figs = Path(args.run), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    res = analyse(run)
    (run / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    figures(res, figs)
    if 'sweep' in res:
        print(json.dumps(res['sweep']['choice']))


if __name__ == '__main__':
    main()
