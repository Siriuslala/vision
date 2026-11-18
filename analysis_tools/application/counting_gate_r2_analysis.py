"""CPU analysis and figures of experiment 7.3 Part C round 2 (gate modifications and inference-time edits).

Results doc: docs/analysis/application/results_2026-11-11_counting_gate_r2.md.

Count readout used throughout: a ridge probe fitted on the BASE model's activations only (target-cell tokens or the
answer position, 128 principal components, lambda = 10) and applied unchanged to every modified model, so that a
change of the represented count's scale shows up as a shift (a probe refitted per model would absorb it).  Base
predictions are 5-fold cross-validated by base image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.application import located_finetune_analysis as lfa

CONDS = ('H32', 'H64', 'H96', 'V32', 'V64')
BENCHES = ('pope', 'textvqa', 'mmvp', 'gqa')
K_BIG = 12
_FITS: dict = {}          # cached base-model probe fits (per base file, site, layer, axis, fold)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float('nan')] * 2
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


class Cond:
    """File lookup for one condition: files of this run first, then (for r1:<name> or base) round-1 files."""

    def __init__(self, run: Path, r1: Path, name: str):
        self.name, self.dirs = name, [run / 'eval' / name]
        if name.startswith('r1:'):
            self.dirs.append(r1 / 'eval' / name[3:])
        elif name == 'base':
            self.dirs.append(r1 / 'eval' / 'base')

    def path(self, fn: str) -> Path | None:
        for d in self.dirs:
            if (d / fn).exists():
                return d / fn
        return None

    def rows(self, fn: str) -> list[dict]:
        p = self.path(fn)
        return read_jsonl(p) if p else []


def probe_fn(X: np.ndarray, y: np.ndarray, lam: float = 10.0, npc: int = 128):
    """Ridge regression of y on the top-npc principal components of X; returns (predict function, direction beta in
    residual space) with prediction (x - mu) . beta + mean(y)."""
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:npc].T
    Z = (X - mu) @ V
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(V.shape[1]), Z.T @ (y - y.mean()))
    beta = V @ w
    ym = float(y.mean())
    return (lambda X2: (X2 - mu) @ beta + ym), beta


def ord_summary(rows: list[dict]) -> dict:
    o = {'all': [sum(r['correct'] for r in rows), len(rows)]}
    for c in CONDS:
        sel = [r for r in rows if r['condition'] == c and r['variant'] == 'base']
        o[f'{c}_base'] = [sum(r['correct'] for r in sel), len(sel)]
        o[f'{c}_base_plus1'] = sum(r['err'] == 1 for r in sel)
        o[f'{c}_base_minus1'] = sum(r['err'] == -1 for r in sel)
        big = [r for r in sel if r['c'] >= K_BIG]
        o[f'{c}_base_kbig'] = [sum(r['correct'] for r in big), len(big)]
        small = [r for r in sel if r['c'] < 6]
        o[f'{c}_base_ksmall'] = [sum(r['correct'] for r in small), len(small)]
        cond_all = [r for r in rows if r['condition'] == c]
        o[f'{c}_all'] = [sum(r['correct'] for r in cond_all), len(cond_all)]
        gap = [r for r in rows if r['condition'] == c and r['variant'] in ('insL', 'delL')]
        o[f'{c}_recount'] = sum(bool(r['match_gap']) and not r['correct'] for r in gap)
    o['other_correct'] = sum(bool(r['other_correct']) for r in rows)
    return o


def err_by_k(rows, cond, variant='base'):
    sel = [r for r in rows if r['condition'] == cond and r['variant'] == variant and r['err'] is not None]
    ks = sorted({r['c'] for r in sel})
    return {int(k): float(np.mean([r['err'] for r in sel if r['c'] == k])) for k in ks}


def ansprobe_summary(cond: Cond, base: Cond) -> dict | None:
    """Answer-position expected number and base-probe readouts (target tokens / answer position) per layer."""
    pc, pb = cond.path('ansprobe.npz'), base.path('ansprobe.npz')
    if pc is None or pb is None:
        return None
    C, B = np.load(pc), np.load(pb)
    if not (C['ids'] == B['ids']).all():
        raise ValueError('ansprobe ids differ between conditions')
    rows = json.loads(cond.path('ansprobe.json').read_text())
    k = B['k'].astype(float)
    cnd = B['cond']
    en, marg = [], []
    for r in rows:
        lp = {int(n): v for n, v in r['logp'].items()}
        ns = np.array(sorted(lp))
        p = np.exp(np.array([lp[n] for n in ns]) - max(lp.values()))
        p /= p.sum()
        en.append(float((ns * p).sum()))
        marg.append(lp[r['k']] - lp.get(r['k'] + 1, np.nan))
    en, marg = np.array(en), np.array(marg)
    out = {'E_n_minus_k': {}, 'margin_k_vs_k1': {}, 'probe_shift': {}}
    for c in CONDS:
        m = (cnd == c) & (k >= K_BIG)
        out['E_n_minus_k'][c] = float(np.mean(en[m] - k[m]))
        out['margin_k_vs_k1'][c] = float(np.nanmean(marg[m]))
    layers = [int(x) for x in B['layers']]
    groups = np.array([i.split('_')[1] for i in B['ids']])
    for site in ('tgt', 'ans'):
        for li, l in enumerate(layers):
            for axis in ('H', 'V'):
                sel = np.array([c.startswith(axis) for c in cnd])
                Xb, Xc, y = B[site][sel, li].astype(np.float32), C[site][sel, li].astype(np.float32), k[sel]
                g = groups[sel]
                ug = np.unique(g)
                folds = {u: i % 5 for i, u in enumerate(ug)}
                fb = np.array([folds[u] for u in g])
                pred_b, pred_c = np.zeros(len(y)), np.zeros(len(y))
                for f in range(5):
                    tr, te = fb != f, fb == f
                    key = (str(pb), site, l, axis, f)
                    if key not in _FITS:
                        _FITS[key] = probe_fn(Xb[tr], y[tr])[0]
                    fn = _FITS[key]
                    pred_b[te], pred_c[te] = fn(Xb[te]), fn(Xc[te])
                for c in CONDS:
                    if not c.startswith(axis):
                        continue
                    mm = (cnd[sel] == c) & (y >= K_BIG)
                    out['probe_shift'][f'{site}_L{l}_{c}'] = float(np.mean(pred_c[mm] - pred_b[mm]))
                big = y >= K_BIG
                out['probe_shift'][f'{site}_L{l}_{axis}_kbig'] = float(np.mean(pred_c[big] - pred_b[big]))
                out['probe_shift'][f'{site}_L{l}_{axis}_ksmall'] = float(np.mean(pred_c[y < 6] - pred_b[y < 6]))
                out.setdefault('probe_curve', {})[f'{site}_L{l}_{axis}'] = {
                    'k': y.tolist(), 'base': pred_b.tolist(), 'cond': pred_c.tolist()}
    out['E_n'] = en.tolist()
    out['k'] = k.tolist()
    out['cond'] = cnd.tolist()
    return out


def residual_delta_cos(run: Path, r1: Path, names: list[str], site='ans', axis='V') -> dict:
    """Cosine between conditions' mean residual change (condition - base) at the answer position, per layer, over
    the axis's k >= 12 base images."""
    base = np.load(Cond(run, r1, 'base').path('ansprobe.npz'))
    sel = np.array([c.startswith(axis) for c in base['cond']]) & (base['k'] >= K_BIG)
    deltas = {}
    for n in names:
        p = Cond(run, r1, n).path('ansprobe.npz')
        if p is None:
            continue
        C = np.load(p)
        deltas[n] = (C[site][sel].astype(np.float32) - base[site][sel].astype(np.float32)).mean(0)   # (layers, D)
    layers = [int(x) for x in base['layers']]
    out = {}
    for li, l in enumerate(layers):
        M = {}
        for a in deltas:
            for b in deltas:
                va, vb = deltas[a][li], deltas[b][li]
                M[f'{a}|{b}'] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        out[f'L{l}'] = M
        out[f'L{l}_norm'] = {a: float(np.linalg.norm(deltas[a][li])) for a in deltas}
    return out


def headsweep_summary(run: Path, r1: Path, ordinal_run: Path, layer: int = 15) -> dict | None:
    rows = read_jsonl(run / 'headsweep' / f'L{layer}.jsonl')
    if not rows:
        return None
    cases = {c['id']: c for c in read_jsonl(ordinal_run / 'cases.jsonl')}
    base = next(r for r in rows if r['head'] == -1)
    ids = base['ids']
    k = np.array([cases[i]['truth']['count'] for i in ids])
    cond = np.array([cases[i]['condition'] for i in ids])
    def stats(errs):
        e = np.array([np.nan if x is None else x for x in errs], float)
        return {f'{c}_kbig_err': float(np.nanmean(e[(cond == c) & (k >= K_BIG)])) for c in ('H64', 'V64')} | \
               {f'{c}_acc': float(np.mean(e[cond == c] == 0)) for c in ('H64', 'V64')}
    out = {'base': stats(base['err']), 'heads': {}}
    for r in rows:
        if r['head'] >= 0:
            out['heads'][int(r['head'])] = stats(r['err'])
    gs = json.loads((r1 / 'eval' / 'base' / 'gatestat.json').read_text())
    alpha = {}
    for g in gs:
        if g['layer'] == layer and g['variant'] == 'base':
            alpha.setdefault(g['cls'], []).append(np.array(g['alpha']))
    out['alpha_by_cls'] = {int(c): np.mean(v, 0).tolist() for c, v in alpha.items()}
    return out


def safety(cond: Cond, base: Cond) -> dict:
    out = {}
    for fn, key, sel in (('general.jsonl', 'correct', None), ('coord.jsonl', 'hit', None), ('think.jsonl', 'correct', None),
                         ('count.jsonl', 'correct', None), ('g8.jsonl', 'correct', None), ('gmulti.jsonl', 'correct', None),
                         ('text.jsonl', 'correct', None), ('babyvision.jsonl', None, None)):
        a, b = base.rows(fn), cond.rows(fn)
        if not b:
            continue
        if fn == 'general.jsonl':
            for bench in BENCHES:
                A = {x['id']: float(x['correct']) for x in a if x['bench'] == bench}
                B = {x['id']: float(x['correct']) for x in b if x['bench'] == bench}
                ids = sorted(set(A) & set(B))
                dd = np.array([B[i] - A[i] for i in ids])
                rng = np.random.default_rng(0)
                boots = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
                out[f'general_{bench}'] = {'base': float(np.mean([A[i] for i in ids])), 'delta': float(dd.mean()),
                                           'ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                           'n': len(ids)}
        elif fn == 'text.jsonl':
            same = sum(x['text'] == y['text'] for x, y in zip(a, b)) if a else None
            out['text'] = {'correct': sum(x['correct'] for x in b), 'n': len(b), 'identical_to_base': same}
        elif fn == 'babyvision.jsonl':
            out['babyvision'] = [sum(x['score']['correct'] for x in b), len(b)]
        elif fn == 'coord.jsonl':
            for fam, f in (('synthetic', lambda x: x['family'] != 'refcoco'), ('refcoco', lambda x: x['family'] == 'refcoco')):
                bb = [x for x in b if f(x)]
                out[f'coord_{fam}'] = [sum(x['hit'] for x in bb), len(bb)]
                if a:
                    out[f'coord_{fam}_paired'] = lfa.paired(a, b, f, 'hit')
            out['coord_invalid'] = sum(not x['valid'] for x in b)
        elif fn == 'gmulti.jsonl':
            for n in (6, 10, 12):
                bb = [x for x in b if x['n'] == n]
                out[f'gmulti_{n}'] = {'row': sum(x['row_ok'] for x in bb), 'col': sum(x['col_ok'] for x in bb), 'n': len(bb)}
        else:
            out[fn.split('.')[0]] = [sum(bool(x[key]) for x in b), len(b)]
            if a and fn in ('think.jsonl', 'g8.jsonl', 'count.jsonl'):
                out[fn.split('.')[0] + '_paired'] = lfa.paired(a, b, lambda x: True, key)
    return out


def analyse(run: Path, r1: Path, ordinal_run: Path) -> dict:
    names = sorted(p.name for p in (run / 'eval').iterdir() if p.is_dir())
    base = Cond(run, r1, 'base')
    base_ord = base.rows('ord.jsonl')
    res = {'conditions': {}}
    meta = {}
    for p in sorted((run / 'adapters').glob('*.pt')) if (run / 'adapters').exists() else []:
        log = read_jsonl(run / 'adapters' / f'{p.stem}_log.jsonl')
        meta[p.stem] = {'loss_odd_first10': float(np.mean([r['loss_odd'] for r in log[:10]])) if log else None,
                        'loss_odd_last10': float(np.mean([r['loss_odd'] for r in log[-10:]])) if log else None,
                        'loss_vqa_last10': float(np.mean([r['loss_vqa'] for r in log[-10:]])) if log else None}
    for n in names + ['r1:gate_sel_s1', 'r1:gate_wrong_s1', 'r1:gate_sel_s2', 'r1:dnall_s1']:
        c = Cond(run, r1, n)
        r = {'train': meta.get(n)}
        rows = c.rows('ord.jsonl')
        if rows:
            r['ord'] = ord_summary(rows)
            r['err_by_k'] = {cc: err_by_k(rows, cc) for cc in CONDS}
            if n != 'base' and base_ord:
                r['paired_all'] = lfa.paired(base_ord, rows, lambda x: True)
        a = ansprobe_summary(c, base) if n != 'base' and not n.startswith('vl_') else None   # vl_*: other model
        if a:
            r['ansprobe'] = {k: v for k, v in a.items() if k not in ('probe_curve', 'E_n', 'k', 'cond')}
        if n == 'base' and base.path('ansprobe.npz'):
            rows_b = json.loads(base.path('ansprobe.json').read_text())
            r['ansprobe_base_rows'] = len(rows_b)
            r['E_n_minus_k'] = {}
            for cc in CONDS:
                d = []
                for x in rows_b:
                    if x['condition'] != cc or x['k'] < K_BIG:
                        continue
                    lp = {int(n): v for n, v in x['logp'].items()}
                    ns = np.array(sorted(lp))
                    p = np.exp(np.array([lp[n] for n in ns]) - max(lp.values()))
                    d.append(float((ns * p / p.sum()).sum()) - x['k'])
                r['E_n_minus_k'][cc] = float(np.mean(d))
        r['safety'] = safety(c, base)
        res['conditions'][n] = r
    res['headsweep'] = headsweep_summary(run, r1, ordinal_run)
    res['delta_cos'] = residual_delta_cos(run, r1, [n for n in res['conditions'] if n != 'base' and not n.startswith('vl_')])
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--r1-run', required=True)
    p.add_argument('--ordinal-run', required=True)
    p.add_argument('--figures', required=True)
    a = p.parse_args()
    run, r1 = Path(a.run), Path(a.r1_run)
    res = analyse(run, r1, Path(a.ordinal_run))
    (run / 'results.json').write_text(json.dumps(res, indent=1))
    res = json.loads((run / 'results.json').read_text())          # string keys, as the figures expect
    figs = Path(a.figures)
    figs.mkdir(parents=True, exist_ok=True)
    fig_examples(Path(a.ordinal_run), figs)
    fig_err_by_k(res, [('base', 'base'), ('l15_nonint', 'L15 alpha=1, non-integrator heads'),
                       ('l15_s50', 'L15 -log alpha x0.5, all heads'), ('l14_nonint', 'L14 alpha=1'),
                       ('l13_nonint', 'L13 alpha=1')], figs, 'l15_err_by_k.png')
    fig_heads(res, figs)
    fig_gain(run, r1, [('l15_nonint', 'L15 alpha=1 (inference)'), ('r1:gate_sel_s1@9,10', '7.3C gate LoRA, only L9/10 active'),
                       ('r1:gate_wrong_s1@18,21', '7.3C gate LoRA, only L18/21 active'), ('full1e4_L15', 'full gate fine-tuning L15'),
                       ('bias_1_23', 'per-head gate scalars L1-23'), ('full1e4_L27', 'full gate fine-tuning L27')], figs)
    fig_location(res, [6, 10, 15, 18, 21, 27, 30], figs)
    fig_summary(res, figs)
    fig_flow(figs)
    print(json.dumps({n: {'all': r.get('ord', {}).get('all'), 'V64+1': r.get('ord', {}).get('V64_base_plus1')}
                      for n, r in res['conditions'].items()}))



# ----------------------------------------------------------------------------- figures

PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
GREY = '#8a8984'


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6})
    return plt


def fig_examples(ordinal_run: Path, figs: Path) -> None:
    from PIL import Image, ImageDraw
    plt = _plt()
    cases = [c for c in read_jsonl(ordinal_run / 'cases.jsonl') if c['mode'] == 'direct' and c['variant'] == 'base']
    pick = [next(c for c in cases if c['condition'] == 'H64' and c['truth']['count'] == 14),
            next(c for c in cases if c['condition'] == 'V64' and c['truth']['count'] == 15)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), gridspec_kw={'width_ratios': [3.2, 1]})
    for ax, c in zip(axs, pick):
        im = Image.open(ordinal_run / c['image']).convert('RGB')
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = c['truth']['target_cell_box']
        d.rectangle([x0, y0, x1, y1], outline=(235, 104, 52), width=5)
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{c['condition']} base: true long-axis count k = {c['truth']['count']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / 'examples.png', dpi=150)
    plt.close(fig)


def fig_err_by_k(res: dict, names: list[tuple[str, str]], figs: Path, fn: str) -> None:
    plt = _plt()
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, cond in zip(axs, ('H32', 'H64', 'V32', 'V64')):
        for i, (n, lab) in enumerate(names):
            e = res['conditions'].get(n, {}).get('err_by_k', {}).get(cond)
            if not e:
                continue
            ks = sorted(int(k) for k in e)
            ax.plot(ks, [e[str(k)] if str(k) in e else e[k] for k in ks], marker='o', ms=3, lw=1.6,
                    color=GREY if n == 'base' else PALETTE[i % 8], label=lab)
        ax.axhline(0, color='#52514e', lw=0.8)
        ax.set_title(f'{cond} base variant (64 images)')
        ax.set_xlabel('true long-axis count k')
    axs[0].set_ylabel('mean signed error of the answer (answer - k)')
    axs[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / fn, dpi=150)
    plt.close(fig)


def fig_heads(res: dict, figs: Path) -> None:
    h = res.get('headsweep')
    if not h:
        return
    plt = _plt()
    heads = sorted(int(x) for x in h['heads'])
    alpha = np.array(h['alpha_by_cls']['2'] if '2' in h['alpha_by_cls'] else h['alpha_by_cls'][2])
    fig, axs = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax, cond in zip(axs, ('V64', 'H64')):
        d = [h['heads'][str(x) if str(x) in h['heads'] else x][f'{cond}_kbig_err'] - h['base'][f'{cond}_kbig_err'] for x in heads]
        sc = ax.bar(heads, d, color=plt.cm.viridis(alpha[heads]), edgecolor='none')
        ax.axhline(0, color='#52514e', lw=0.8)
        ax.set_ylabel(f'{cond}: change of mean\nerror at k >= 12')
    axs[-1].set_xlabel('layer-15 value head (forgetting removed in this head only, image tokens)')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axs, fraction=0.025)
    cb.set_label('base mean alpha of the head at figure tokens (per-token retention)')
    fig.savefig(figs / 'l15_heads.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_gain(run: Path, r1: Path, names: list[tuple[str, str]], figs: Path, site='ans', layer=24) -> None:
    plt = _plt()
    base = Cond(run, r1, 'base')
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, axis in zip(axs, 'HV'):
        for i, (n, lab) in enumerate(names):
            a = ansprobe_summary(Cond(run, r1, n), base)
            if a is None:
                continue
            cur = a['probe_curve'][f'{site}_L{layer}_{axis}']
            k, pb, pc = np.array(cur['k']), np.array(cur['base']), np.array(cur['cond'])
            ks = np.unique(k)
            ax.plot(ks, [(pc[k == x] - pb[k == x]).mean() for x in ks], color=PALETTE[i % 8], lw=1.6, marker='o', ms=3,
                    label=lab)
        ax.axhline(0, color=GREY, lw=1.2, label='base (reference)' if axis == 'H' else None)
        ax.set_title(f'{"H" if axis == "H" else "V"} conditions, base images (n = {"192" if axis == "H" else "128"})')
        ax.set_xlabel('true long-axis count k')
    axs[0].set_ylabel(f'change of the base-fitted count readout\n(answer position, layer {layer}), instances')
    axs[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / f'gain_{site}_L{layer}.png', dpi=150)
    plt.close(fig)


def fig_location(res: dict, layers: list[int], figs: Path) -> None:
    plt = _plt()
    xs, acc, p1, shift = [], [], [], []
    for l in layers:
        c = res['conditions'].get(f'full1e4_L{l}')
        if not c or 'ord' not in c:
            continue
        xs.append(l)
        acc.append(c['ord']['all'][0])
        p1.append(c['ord']['V64_base_plus1'] + c['ord']['H64_base_plus1'])
        shift.append(c.get('ansprobe', {}).get('probe_shift', {}).get('ans_L24_V_kbig', np.nan))
    if not xs:
        return
    b = res['conditions']['base']['ord']
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, y, yb, lab in ((axs[0], acc, b['all'][0], 'correct long-axis answers (/1600)'),
                           (axs[1], p1, b['V64_base_plus1'] + b['H64_base_plus1'], '+1 answers, H64 + V64 base (/128)'),
                           (axs[2], shift, 0.0, 'answer-position probe shift, V, k >= 12')):
        ax.bar([str(x) for x in xs], y, color=PALETTE[0], width=0.6)
        ax.axhline(yb, color=GREY, lw=1.2, ls='--', label='base')
        ax.set_xlabel('DeltaNet layer of the single-layer gate fine-tuning')
        ax.set_title(lab, fontsize=9)
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / 'location.png', dpi=150)
    plt.close(fig)


def fig_flow(figs: Path) -> None:
    """Schematic of the counting pathway and of where each modification acts (drawn, not data)."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis('off')
    boxes = [
        (0.1, 'image tokens\nin raster order\n(32x32 px each)'),
        (2.2, 'DeltaNet L6-11\nforgetting heads detect\na new figure line;\nintegrators add +1'),
        (4.3, 'DeltaNet L13-15\nslow-forgetting heads\n(alpha 0.93-0.99) add a\ncomponent growing with k'),
        (6.4, 'count quantity on\nthe target cell and its\nrow tokens: c = g k\n(g slightly > 1)'),
        (8.5, 'answer position,\nL16 / L20 attention:\nreader heads copy c\ninto the number state'),
        (10.6, 'number tokens:\nround(c);\nk + 1 when\n(g - 1) k > 0.5'),
    ]
    for x, t in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.9), 1.9, 1.7, boxstyle='round,pad=0.05', fc='#eef4fc', ec=PALETTE[0], lw=1.2))
        ax.text(x + 0.95, 2.75, t, ha='center', va='center', fontsize=8.2)
    for x, _ in boxes[:-1]:
        ax.annotate('', xy=(x + 2.15, 2.75), xytext=(x + 1.95, 2.75), arrowprops=dict(arrowstyle='->', color='#52514e'))
    notes = [(3.15, 'image-token gate changes\nat DeltaNet L10-18 (LoRA,\nfull, per-head bias):\nlower g'),
             (5.35, 'L15 alpha = 1 at image\ntokens (inference only):\nremoves the k-growing\ncomponent, lowers g'),
             (9.45, 'gate changes at L21 / L27 / L30\n(after the L20 read):\nno effect')]
    for x, t in notes:
        ax.text(x, 0.85, t, ha='center', va='center', fontsize=7.8, color=PALETTE[1])
        ax.annotate('', xy=(x, 1.85), xytext=(x, 1.45), arrowprops=dict(arrowstyle='->', color=PALETTE[1]))
    fig.savefig(figs / 'flow.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_summary(res: dict, figs: Path) -> None:
    """Per condition: change of the answer-position count readout (V, k >= 12, layer 24) vs change of E[n] - k on
    V64 k >= 12 (both relative to the base model)."""
    plt = _plt()
    pts = []
    for n, c in res['conditions'].items():
        a = c.get('ansprobe')
        if not a:
            continue
        pts.append((n, a['probe_shift']['ans_L24_V_kbig'], a['E_n_minus_k']['V64'], c.get('ord', {}).get('all', [None])[0]))
    if not pts:
        return
    base_e = res['conditions']['base']['E_n_minus_k']['V64']
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    groups = {'inference edit': ('l1',), 'trained (gate)': ('full', 'bias', 'decay', 'lora', 'r1:')}
    for i, (g, pre) in enumerate(groups.items()):
        sel = [p for p in pts if p[0].startswith(pre)]
        ax.scatter([p[1] for p in sel], [p[2] - base_e for p in sel], s=36, color=PALETTE[i], label=g,
                   edgecolor='white', linewidth=0.8, zorder=3)
        for p in sel:
            if p[0] in ('l15_nonint', 'r1:gate_sel_s1@9,10', 'r1:gate_wrong_s1@18,21', 'full1e4_L15', 'bias_1_23',
                        'full1e4_L27', 'l13_nonint', 'full1e4_L6', 'lora32_sel'):
                ax.annotate(p[0].replace('r1:', ''), (p[1], p[2] - base_e), fontsize=6.5, xytext=(3, 3),
                            textcoords='offset points', color='#52514e')
    ax.axhline(0, color=GREY, lw=0.8); ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel('change of the base-fitted count readout at the answer\nposition (layer 24, V conditions, k >= 12), instances')
    ax.set_ylabel('change of E[n] - k on V64, k >= 12\n(from candidate log-probabilities)')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / 'summary.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
