"""Statistics and figures for experiment 4.4 (DeepStack follow-ups).

Part A and B read runs/<model>/<condition>.scores.jsonl of this experiment and, for the conditions
full / off / mean / blur (sigma 2) on the coordinate tasks, of experiment 4.3 (same cases, same code path).
Part C fits ridge readouts on the per-token states written by ``deepstack_todo.represent``.

Outputs: analysis/summary.json, analysis/tables.md and the figures copied to the docs directory.
Paired comparisons against Full use exact McNemar tests and item-level paired bootstrap intervals;
p values are uncorrected (the decisive contrasts are pre-specified in the results document).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.behavior.decomposition_analysis import GRID_INK, INK, MUTED, _style
from analysis_tools.deepstack.deepstack_ablation_analysis import (
    coordinate_calibration, rowcol_offsets, select, with_pixel_reading,
)
from analysis_tools.deepstack.deepstack_comparison import slug
from analysis_tools.deepstack.deepstack_comparison_analysis import (
    fast_auroc, fmt_p, mcnemar, paired_bootstrap, rate,
)
from analysis_tools.deepstack.deepstack_todo import REP_N, detail_targets, token_targets

ORDER = ('full', 'blur2', 'blur4', 'blur8', 'blur16', 'blank', 'mean', 'off', 'blankflip')
LABELS = {'full': 'Full', 'blur2': 'Blur σ=2', 'blur4': 'Blur σ=4', 'blur8': 'Blur σ=8', 'blur16': 'Blur σ=16',
          'blank': 'Blank', 'mean': 'Mean', 'off': 'Off', 'blankflip': 'Blank-flip'}
COLORS = {'full': '#52514e', 'blur2': '#f2a27c', 'blur4': '#eb6834', 'blur8': '#b8441a', 'blur16': '#7a2a0c',
          'blank': '#6250d6', 'mean': '#1baf7a', 'off': '#2a78d6', 'blankflip': '#c2408a'}
ABLATION_NAMES = {'full': 'full', 'off': 'off', 'mean': 'mean', 'blur': 'blur2'}
COORD_METRICS = {
    'T6 point': ('T6', 'point', (10, 20), 'hit', None),
    'T6 bbox': ('T6', 'bbox', (10, 20), 'correct', None),
    'T6 row/col': ('T6', 'row_column', (10, 20), 'correct', None),
    'T5 point': ('T5', 'point', (10, 20), 'hit', None),
    'T5 bbox': ('T5', 'bbox', (10, 20), 'correct', None),
    'T5 row/col': ('T5', 'row_column', (10, 20), 'correct', None),
    'RefCOCO a>=5%': ('RefCOCO', 'bbox', None, 'correct', ('5to10', 'ge10')),
    'RefCOCO 1-5%': ('RefCOCO', 'bbox', None, 'correct', ('1to5',)),
}
GENERAL_METRICS = ('POPE', 'TextVQA', 'BabyVision')
MODELS = ('Qwen3-VL-8B-Instruct', 'Qwen3-VL-4B-Instruct')


# ----------------------------------------------------------------------------- parts A and B


def load_runs(output: Path, ablation: Path, model: str) -> dict[str, dict[str, dict]]:
    """condition -> case id -> score row merged with its case (this experiment first, then 4.3)."""
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    runs: dict = {}
    for path in sorted((output / 'runs' / slug(model)).glob('*.scores.jsonl')):
        cond = path.name[:-len('.scores.jsonl')]
        runs[cond] = {r['id']: with_pixel_reading({**cases[r['id']], **r}) for r in read_jsonl(path)}
    for old, new in ABLATION_NAMES.items():
        path = ablation / 'runs' / slug(model) / f'{old}.scores.jsonl'
        if not path.exists():
            continue
        rows = {r['id']: with_pixel_reading({**cases[r['id']], **r}) for r in read_jsonl(path) if r['id'] in cases}
        runs.setdefault(new, {})
        for cid, row in rows.items():
            runs[new].setdefault(cid, row)       # this experiment's own run of a case takes precedence
    return runs


def compare(runs: dict, ids: list[str], cond: str, field: str, n_boot: int, seed: int) -> dict | None:
    if cond not in runs or 'full' not in runs or not ids:
        return None
    if any(i not in runs[cond] for i in ids):
        return None                            # never compare unequal item sets
    a = [bool(runs[cond][i].get(field)) for i in ids]
    b = [bool(runs['full'][i].get(field)) for i in ids]
    t = mcnemar(a, b)
    return {'n': len(ids), 'rate': rate(a), 'full': rate(b), 'delta': float(np.mean(a) - np.mean(b)),
            'delta_ci': paired_bootstrap(np.array(a, float), np.array(b, float), n_boot, seed),
            'only_cond': t['only_first'], 'only_full': t['only_second'], 'p': t['p']}


def general_ids(runs: dict, task: str) -> list[str]:
    return sorted(i for i, r in runs.get('full', {}).items() if r['task'] == task)


def pope_auroc(runs: dict, cond: str) -> float | None:
    ids = general_ids(runs, 'POPE')
    if cond not in runs or not ids or any(i not in runs[cond] for i in ids):
        return None
    s = np.array([runs[cond][i]['s'] for i in ids])
    y = np.array([runs[cond][i]['truth']['answer_yes'] for i in ids], bool)
    return float(fast_auroc(s, y))


def mirror_fit(runs: dict, cond: str) -> dict:
    """x and y calibration of valid T6/T5 point answers pooled over grids (canvas fractions)."""
    out = {}
    for task in ('T6', 'T5'):
        rows = [r for r in runs.get(cond, {}).values() if r['task'] == task and r['format'] == 'point'
                and r.get('valid')]
        if len(rows) < 3:
            continue
        size = np.array([r['truth']['canvas'] for r in rows], float)
        true = np.array([r['truth']['target_center'] for r in rows], float) / size
        pred = np.array([r['pred_center'] for r in rows], float) / size
        entry = {'n_valid': len(rows)}
        for i, axis in enumerate('xy'):
            slope, icpt = np.polyfit(true[:, i], pred[:, i], 1)
            entry[axis] = {'slope': float(slope), 'intercept': float(icpt),
                           'r': float(np.corrcoef(true[:, i], pred[:, i])[0, 1])}
        # Share of valid answers whose x lies on the mirrored side of the canvas centre.
        entry['x_mirrored_side'] = float(np.mean(np.sign(pred[:, 0] - .5) == -np.sign(true[:, 0] - .5)))
        out[task] = entry
    return out


def analyze_behavior(output: Path, ablation: Path, n_boot: int, seed: int) -> dict:
    summary: dict = {}
    for model in MODELS:
        runs = load_runs(output, ablation, model)
        if 'full' not in runs:
            continue
        m: dict = {'conditions': sorted(runs, key=ORDER.index), 'coord': {}, 'general': {}, 'calibration': {},
                   'mirror': {}, 'rowcol': {}}
        for name, spec in COORD_METRICS.items():
            ids = select(runs['full'], spec)
            if not ids:
                continue
            m['coord'][name] = {c: compare(runs, ids, c, spec[3], n_boot, seed) for c in ORDER}
            m['coord'][name]['full_rate'] = rate([bool(runs['full'][i].get(spec[3])) for i in ids])
        for task in GENERAL_METRICS:
            ids = general_ids(runs, task)
            if not ids:
                continue
            m['general'][task] = {c: compare(runs, ids, c, 'correct', n_boot, seed) for c in ORDER}
            m['general'][task]['full_rate'] = rate([bool(runs['full'][i].get('correct')) for i in ids])
            if task == 'TextVQA':
                m['general'][task]['vqa_acc'] = {c: float(np.mean([runs[c][i]['vqa_acc'] for i in ids]))
                                                 for c in ORDER if c in runs and all(i in runs[c] for i in ids)}
            if task == 'POPE':
                m['general'][task]['auroc'] = {c: pope_auroc(runs, c) for c in ORDER if c in runs}
        for c in ORDER:
            if c in runs:
                m['calibration'][c] = coordinate_calibration(runs, c)
                m['mirror'][c] = mirror_fit(runs, c)
                m['rowcol'][c] = {g: rowcol_offsets(runs, c, g) for g in (10, 20)}
        summary[model] = m
    return summary


# ----------------------------------------------------------------------------- part C


def ridge_fit(x_tr: np.ndarray, y_tr, alphas: list[float]):
    """Ridge on standardized features via one eigendecomposition.

    ``y_tr`` is an array or a dict of arrays (several targets share the decomposition); returns, per target,
    a dict alpha -> predictor (a bare dict alpha -> predictor when ``y_tr`` is an array).
    """
    import torch
    single = not isinstance(y_tr, dict)
    ys = {'_': y_tr} if single else y_tr
    device = os.environ.get('PROBE_DEVICE', 'cpu')     # a GPU only speeds up the eigendecomposition
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    X = torch.from_numpy((x_tr - mu) / sd).to(device, torch.float64)
    evals, evecs = torch.linalg.eigh(X.T @ X)
    out = {}
    for name, y in ys.items():
        ym = y.mean(0)
        XtY = evecs.T @ (X.T @ torch.from_numpy(y - ym).to(device, torch.float64))
        preds = {}
        for a in alphas:
            W = (evecs @ (XtY / (evals[:, None] + a * len(X)))).float().cpu().numpy()
            preds[a] = (lambda x, W=W, ym=ym: ((x - mu) / sd) @ W + ym)
        out[name] = preds
    return out['_'] if single else out


def r2(y: np.ndarray, p: np.ndarray, ref_mean: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - ref_mean) ** 2).sum())


def split_images(meta: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image-level 60/20/20 split stratified by source; reconstruction images are always in the test split."""
    rng = np.random.default_rng(seed)
    images, src = np.unique(meta['image'], return_index=True)
    sources = meta['source'][src]
    view = set(meta['view'].tolist())
    tr, va, te = [], [], []
    for s in np.unique(sources):
        pool = [i for i in images[sources == s] if i not in view]
        rng.shuffle(pool)
        n = len(pool) + sum(v in images[sources == s] for v in view)
        n_te = max(0, int(round(0.2 * n)) - sum(v in images[sources == s] for v in view))
        n_va = int(round(0.2 * n))
        te += pool[:n_te]
        va += pool[n_te:n_te + n_va]
        tr += pool[n_te + n_va:]
    te += sorted(view)
    return np.array(tr), np.array(va), np.array(te)


def metric_mask(meta: dict, seed: int, per_image: int = 64) -> np.ndarray:
    """Tokens used for fitting and scoring: all stored tokens, except a 64-token sample of each fully stored image."""
    rng = np.random.default_rng(seed + 7)
    keep = np.ones(len(meta['image']), bool)
    for v in meta['view']:
        idx = np.flatnonzero(meta['image'] == v)
        drop = rng.permutation(idx)[per_image:]
        keep[drop] = False
    return keep


SITE_ORDER = {
    'qwen3_vl_8b': ['vit9', 'vit17', 'vit25', 'vit27', 'D1', 'D2', 'D3', 'V', 'full_L1', 'full_L1pD1',
                    'full_L2', 'full_L2pD2', 'full_L3', 'full_L3pD3', 'full_L4', 'full_L8', 'full_L16',
                    'full_L24', 'full_L36', 'off_L2', 'off_L3', 'off_L4', 'off_L8', 'off_L16', 'off_L24', 'off_L36'],
    'qwen35_9b': ['vit9', 'vit17', 'vit25', 'vit27', 'V', 'full_L1', 'full_L2', 'full_L3', 'full_L4', 'full_L8',
                  'full_L16', 'full_L24', 'full_L32'],
}
RECON_SITES = ('vit9', 'D1', 'V', 'vit27', 'full_L4', 'off_L4', 'full_L16')
SHARE_K = 16
SHARE_SITES = ('vit9', 'vit17', 'vit25', 'vit27', 'D1', 'D2', 'D3', 'V')
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def stored_detail_targets(output: Path, meta: dict) -> dict:
    """detail_targets of every stored token, recomputed from the representation images."""
    rep = read_jsonl(output / 'rep_images.jsonl')
    out = {'structure': np.zeros((len(meta['image']), 64), np.float32),
           'hfenergy': np.zeros((len(meta['image']), 16), np.float32)}
    for i in np.unique(meta['image']):
        rows = np.flatnonzero(meta['image'] == i)
        t = detail_targets(Image.open(output / rep[i]['image']).convert('RGB'))
        for k in out:
            out[k][rows] = t[k][meta['token'][rows]]
    return out


def analyze_representation(output: Path, seed: int, n_boot: int) -> dict:
    out: dict = {}
    for model_slug, sites in SITE_ORDER.items():
        d = output / 'represent' / model_slug
        if not (d / 'meta.npz').exists():
            continue
        meta = dict(np.load(d / 'meta.npz'))
        keep = metric_mask(meta, seed)
        tr, va, te = split_images(meta, seed)
        img = meta['image']
        m_tr, m_va, m_te = (np.isin(img, s) & keep for s in (tr, va, te))
        dt_ = stored_detail_targets(output, meta)
        targets = {'position': meta['xy'].astype(np.float32), 'structure': dt_['structure'],
                   'hfenergy': dt_['hfenergy'], 'coarse': meta['t_coarse'].astype(np.float32)}
        content = meta['content'].astype(bool)
        res: dict = {'n_tokens': {'train': int(m_tr.sum()), 'val': int(m_va.sum()), 'test': int(m_te.sum())},
                     'n_images': {'train': len(tr), 'val': len(va), 'test': len(te)}, 'sites': {}}
        recon = {}
        for site in sites:
            path = d / f'site_{site}.npy'
            if not path.exists():
                continue
            X = np.load(path).astype(np.float32)
            entry = {}
            # Position is fitted on every token; detail and coarse colour on content tokens only.
            groups = {'all': ('position',), 'content': ('structure', 'hfenergy', 'coarse')}
            for gname, tnames in groups.items():
                cm = content if gname == 'content' else np.ones_like(content)
                fit, val, both = m_tr & cm, m_va & cm, (m_tr | m_va) & cm
                preds = ridge_fit(X[fit], {t: targets[t][fit] for t in tnames}, ALPHAS)
                extra = {'grey': meta['t_grey'].astype(np.float32)[both]} if gname == 'all' and site in RECON_SITES else {}
                best = {}
                for t in tnames:
                    ref = targets[t][fit].mean(0)
                    sc = {a: r2(targets[t][val], f(X[val]), ref) for a, f in preds[t].items()}
                    best[t] = max(sc, key=sc.get)
                refit = ridge_fit(X[both], {**{t: targets[t][both] for t in tnames}, **extra}, ALPHAS)
                for t in tnames:
                    f, ref, e = refit[t][best[t]], targets[t][both].mean(0), {'alpha': best[t]}
                    for si, sname in enumerate(meta['sources']):
                        mask = m_te & cm & (meta['source'] == si)
                        P, Y, ims = f(X[mask]), targets[t][mask], img[mask]
                        e[str(sname)] = r2(Y, P, ref)
                        uniq, inv = np.unique(ims, return_inverse=True)
                        per_e = np.bincount(inv, ((Y - P) ** 2).sum(1))
                        per_t = np.bincount(inv, ((Y - ref) ** 2).sum(1))
                        rng = np.random.default_rng(seed)
                        k = rng.integers(0, len(uniq), (n_boot, len(uniq)))
                        boot = 1 - per_e[k].sum(1) / per_t[k].sum(1)
                        e[f'{sname}_ci'] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
                    entry[t] = e
                if extra:
                    vmask = np.isin(img, meta['view'])
                    recon[site] = refit['grey'][1e-1](X[vmask]).astype(np.float16)
            res['sites'][site] = entry
            print(model_slug, site, {t: round(entry[t].get('synthetic', np.nan), 3) for t in entry}, flush=True)
        # Positional share: fraction of a site's token variance inside the top-K principal subspace of the same
        # site's features for uniform images (a content-free, position-only signal). Chance level is K / dim.
        share = {}
        for site in SHARE_SITES:
            a, b = d / f'site_{site}.npy', d / f'site_blank_{site}.npy'
            if not (a.exists() and b.exists()):
                continue
            S, B = np.load(a).astype(np.float32), np.load(b).astype(np.float32)
            Bc = B[keep] - B[keep].mean(0)
            _, _, vt = np.linalg.svd(Bc[::2], full_matrices=False)
            basis = vt[:SHARE_K]
            share[site] = {'chance': SHARE_K / S.shape[1]}
            for si, sname in enumerate(meta['sources']):
                mask = keep & (meta['source'] == si)
                sc = S[mask] - S[mask].mean(0)
                share[site][str(sname)] = float(((sc @ basis.T) ** 2).sum() / (sc ** 2).sum())
        res['positional_share'] = share
        np.savez(output / 'analysis' / f'recon_{model_slug}.npz', **recon,
                 image=img[np.isin(img, meta['view'])], token=meta['token'][np.isin(img, meta['view'])])
        out[model_slug] = res
    return out


# ----------------------------------------------------------------------------- tables and figures


def _fmt_rate(st):
    return '–' if not st else f"{st['acc']:.2f} [{st['ci'][0]:.2f}, {st['ci'][1]:.2f}]"


def write_tables(summary: dict, path: Path) -> None:
    lines = ['# Experiment 4.4 tables', '']
    for model, m in summary.get('behavior', {}).items():
        conds = [c for c in ORDER if c in m['conditions']]
        lines += [f'## {model}: accuracy per condition (Wilson 95% CI); delta vs Full, McNemar p', '']
        lines += ['| metric | n | ' + ' | '.join(LABELS[c] for c in conds) + ' |',
                  '|---|---|' + '---|' * len(conds)]
        for group in ('coord', 'general'):
            for name, row in m[group].items():
                cells = []
                for c in conds:
                    st = row.get(c)
                    cells.append('–' if not st else (f"{st['rate']['acc']:.2f}" if c == 'full' else
                                                     f"{st['rate']['acc']:.2f} ({st['delta']:+.2f}, p={fmt_p(st['p'])})"))
                lines.append(f"| {name} | {row['full_rate']['n']} | " + ' | '.join(cells) + ' |')
        lines.append('')
        if 'POPE' in m['general']:
            lines.append('POPE AUROC: ' + ', '.join(f'{LABELS[c]} {v:.3f}' for c, v in m['general']['POPE']['auroc'].items() if v))
        if 'TextVQA' in m['general']:
            lines.append('TextVQA VQA accuracy: ' + ', '.join(f'{LABELS[c]} {v:.3f}' for c, v in m['general']['TextVQA']['vqa_acc'].items()))
        lines += ['', '### Point calibration (pooled grids; slope, intercept, r) and mirrored-side share', '',
                  '| condition | task | n valid | x slope | x icpt | x r | y slope | y icpt | y r | x on mirrored side |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for c in conds:
            for task, e in m['mirror'].get(c, {}).items():
                lines.append(f"| {LABELS[c]} | {task} | {e['n_valid']} | {e['x']['slope']:.2f} | {e['x']['intercept']:.2f} | "
                             f"{e['x']['r']:.2f} | {e['y']['slope']:.2f} | {e['y']['intercept']:.2f} | {e['y']['r']:.2f} | "
                             f"{e['x_mirrored_side']:.2f} |")
        lines += ['', '### RefCOCO box-centre calibration', '', '| condition | n valid | x slope | x r | y slope | y r |',
                  '|---|---|---|---|---|---|']
        for c in conds:
            for e in m['calibration'].get(c, []):
                if e['task'] == 'RefCOCO':
                    lines.append(f"| {LABELS[c]} | {e['n_valid']} | {e['x']['slope']:.2f} | {e['x']['r']:.2f} | "
                                 f"{e['y']['slope']:.2f} | {e['y']['r']:.2f} |")
        lines += ['', '### Row/column answers (T5+T6): exact, one under, mean offsets', '',
                  '| condition | grid | n | invalid | exact | one under | mean d_row | mean d_col |', '|---|---|---|---|---|---|---|---|']
        for c in conds:
            for g, e in m['rowcol'].get(c, {}).items():
                if e['n']:
                    lines.append(f"| {LABELS[c]} | {g} | {e['n']} | {e['invalid']} | {e['exact']} | {e['one_under']} | "
                                 f"{e['mean_d_row']:.2f} | {e['mean_d_col']:.2f} |")
        lines.append('')
    for model_slug, res in summary.get('representation', {}).items():
        lines += [f'## {model_slug}: held-out R² per site (test images; 95% CI by image bootstrap)', '',
                  f"tokens {res['n_tokens']}, images {res['n_images']}", '',
                  '| site | target | synthetic | refcoco | babyvision | alpha |', '|---|---|---|---|---|---|']
        for site, e in res['sites'].items():
            for t, v in e.items():
                lines.append(f"| {site} | {t} | " + ' | '.join(
                    f"{v[s]:.3f} [{v[s + '_ci'][0]:.3f}, {v[s + '_ci'][1]:.3f}]" for s in REP_N) + f" | {v['alpha']} |")
        lines += ['', f'Positional share (token variance inside the top-{SHARE_K} principal subspace of uniform-image features):', '',
                  '| site | ' + ' | '.join(REP_N) + ' |', '|---|' + '---|' * len(REP_N)]
        for site, v in res['positional_share'].items():
            lines.append(f'| {site} | ' + ' | '.join(f'{v[s]:.3f}' for s in REP_N) + f" | chance {v['chance']:.4f} |")
        lines.append('')
    path.write_text('\n'.join(lines) + '\n')


def plot_sweep(summary: dict, out: Path) -> None:
    """Accuracy of the coordinate metrics across conditions ordered from intact to content-free injection."""
    plt = _style()
    panels = [('Qwen3-VL-8B-Instruct', ['T6 point', 'T5 point', 'RefCOCO a>=5%', 'RefCOCO 1-5%']),
              ('Qwen3-VL-4B-Instruct', ['T6 point', 'T6 bbox'])]
    panels = [(m, ms) for m, ms in panels if m in summary['behavior']]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 3.8), squeeze=False)
    marker = {'T6 point': 'o', 'T5 point': 's', 'RefCOCO a>=5%': 'D', 'RefCOCO 1-5%': '^', 'T6 bbox': 'v'}
    for ax, (model, metrics) in zip(axes[0], panels):
        m = summary['behavior'][model]
        conds = [c for c in ORDER if any(m['coord'].get(k, {}).get(c) for k in metrics)]
        xs = np.arange(len(conds))
        for k, name in enumerate(metrics):
            row = m['coord'].get(name)
            if not row:
                continue
            ys, lo, hi, xx = [], [], [], []
            for x, c in zip(xs, conds):
                st = row.get(c)
                if st:
                    xx.append(x + (k - 1.5) * 0.08)
                    ys.append(st['rate']['acc'])
                    lo.append(st['rate']['acc'] - st['rate']['ci'][0])
                    hi.append(st['rate']['ci'][1] - st['rate']['acc'])
            ax.errorbar(xx, ys, yerr=[lo, hi], marker=marker[name], ms=6, lw=1.5, capsize=0, color=INK if k == 0 else
                        ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6'][k % 4], label=f"{name} (n={row['full_rate']['n']})")
        ax.set_xticks(xs, [LABELS[c] for c in conds], rotation=35, ha='right')
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel('hit rate / IoU≥0.5 rate (Wilson 95% CI)')
        ax.set_title(model, loc='left')
        ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_mirror(output: Path, ablation: Path, out: Path) -> None:
    """Predicted vs true x (and y) of T6 point answers under Full, Blank, Off and Blank-flip."""
    plt = _style()
    conds = ('full', 'blank', 'off', 'blankflip')
    models = [m for m in MODELS if (output / 'runs' / slug(m)).exists()]
    fig, axes = plt.subplots(len(models), len(conds), figsize=(3.0 * len(conds), 3.0 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        runs = load_runs(output, ablation, model)
        for j, c in enumerate(conds):
            ax = axes[i][j]
            rows = [r for r in runs.get(c, {}).values() if r['task'] == 'T6' and r['format'] == 'point' and r.get('valid')]
            if rows:
                size = np.array([r['truth']['canvas'] for r in rows], float)
                true = np.array([r['truth']['target_center'] for r in rows], float) / size
                pred = np.array([r['pred_center'] for r in rows], float) / size
                ax.scatter(true[:, 0], pred[:, 0], s=16, color=COLORS[c], edgecolor='white', lw=0.5, label='x')
                ax.scatter(true[:, 1], pred[:, 1], s=16, facecolor='none', edgecolor=MUTED, lw=0.8, label='y')
            ax.plot([0, 1], [0, 1], ':', color=MUTED, lw=1)
            ax.plot([0, 1], [1, 0], '--', color=GRID_INK, lw=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f"{model.split('-Instruct')[0].replace('Qwen3-VL-', 'VL-')} · {LABELS[c]} (n={len(rows)})",
                         fontsize=9, loc='left')
            if j == 0:
                ax.set_ylabel('predicted centre / canvas')
            if i == len(models) - 1:
                ax.set_xlabel('true centre / canvas')
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_mirror_refcoco(output: Path, ablation: Path, out: Path) -> None:
    """RefCOCO: predicted box-centre x against the true box-centre x (image fractions), Qwen3-VL-8B."""
    plt = _style()
    conds = ('full', 'blank', 'off', 'blankflip')
    runs = load_runs(output, ablation, 'Qwen3-VL-8B-Instruct')
    fig, axes = plt.subplots(1, len(conds), figsize=(3.0 * len(conds), 3.2), squeeze=False)
    for ax, c in zip(axes[0], conds):
        rows = [r for r in runs.get(c, {}).values() if r['task'] == 'RefCOCO' and r.get('valid')]
        if rows:
            true = np.array([(r['bbox'][0] + r['bbox'][2]) / 2 / r['width'] for r in rows])
            pred = np.array([(r['pred_box'][0] + r['pred_box'][2]) / 2 / r['width'] for r in rows])
            ax.scatter(true, pred, s=7, color=COLORS[c], alpha=0.6, lw=0, label='x (box centre)')
        ax.plot([0, 1], [0, 1], ':', color=MUTED, lw=1)
        ax.plot([0, 1], [1, 0], '--', color=GRID_INK, lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'RefCOCO · {LABELS[c]} (n={len(rows)})', fontsize=9, loc='left')
        ax.set_xlabel('true centre x / width')
    axes[0][0].set_ylabel('predicted centre x / width')
    axes[0][0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_inputs(output: Path, out: Path) -> None:
    """Images from which the injected features are computed: a T6 crop and a RefCOCO image per condition."""
    from PIL import ImageFilter
    from analysis_tools.deepstack.deepstack_todo import blank_like
    plt = _style()
    cases = read_jsonl(output / 'cases.jsonl')
    t6 = next(c for c in cases if c['task'] == 'T6' and c['grid'] == 10)
    ref = next(c for c in cases if c['task'] == 'RefCOCO')
    images = [Image.open(output / t6['image']).convert('RGB').crop((0, 0, 320, 320)),
              Image.open(output / 'refcoco_images' / ref['image']).convert('RGB')]
    conds = [('original', None), ('Blur σ=2', 2), ('Blur σ=4', 4), ('Blur σ=8', 8), ('Blur σ=16', 16), ('Blank', 'blank')]
    fig, axes = plt.subplots(2, len(conds), figsize=(2.2 * len(conds), 4.6))
    for i, im in enumerate(images):
        for j, (name, c) in enumerate(conds):
            shown = im if c is None else (blank_like(im) if c == 'blank' else im.filter(ImageFilter.GaussianBlur(radius=c)))
            axes[i][j].imshow(shown)
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            if i == 0:
                axes[i][j].set_title(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_general(summary: dict, out: Path) -> None:
    """Difference to Full of the non-coordinate tasks and of RefCOCO, paired bootstrap 95% CI."""
    plt = _style()
    m = summary['behavior'].get('Qwen3-VL-8B-Instruct')
    if not m or not m['general']:
        return
    rows = [(t, m['general'][t]) for t in GENERAL_METRICS if t in m['general']]
    rows += [('RefCOCO a>=5% (coordinate)', m['coord']['RefCOCO a>=5%']), ('T6 point (coordinate)', m['coord']['T6 point'])]
    conds = [c for c in ORDER if c != 'full' and any(r.get(c) for _, r in rows)]
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(rows) * 1.6 + 1.2))
    ys = np.arange(len(rows))[::-1] * 1.6
    labelled = set()
    for k, c in enumerate(conds):
        off = (k - (len(conds) - 1) / 2) * 0.17
        for y, (name, r) in zip(ys, rows):
            st = r.get(c)
            if not st:
                continue
            ax.plot(st['delta_ci'], [y + off] * 2, color=COLORS[c], lw=1.3)
            ax.plot(st['delta'], y + off, 'o', ms=6, color=COLORS[c] if st['p'] < 0.05 else 'white',
                    mec=COLORS[c], mew=1.2, label=None if c in labelled else LABELS[c])
            labelled.add(c)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys, [f"{n} (n={r['full_rate']['n']}, Full {r['full_rate']['acc']:.2f})" for n, r in rows])
    ax.set_xlabel('accuracy difference to Full (paired bootstrap 95% CI); filled = McNemar p < 0.05')
    ax.grid(axis='y', visible=False)
    ax.legend(fontsize=8, ncol=1, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_readability(summary: dict, out: Path, target: str, ylabel: str) -> None:
    plt = _style()
    rep = summary['representation']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    src_names = {'synthetic': 'synthetic grids', 'refcoco': 'RefCOCO', 'babyvision': 'BabyVision'}
    for ax, src in zip(axes, REP_N):
        if 'qwen3_vl_8b' in rep:
            sites = rep['qwen3_vl_8b']['sites']
            main = [s for s in SITE_ORDER['qwen3_vl_8b'] if s in sites and not s.startswith('off_')]
            x = np.arange(len(main))
            v = [sites[s][target][src] for s in main]
            ci = np.array([sites[s][target][src + '_ci'] for s in main])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color='#2a78d6', alpha=0.15, lw=0)
            ax.plot(x, v, 'o-', color='#2a78d6', ms=5, lw=1.6, label='Qwen3-VL-8B Full')
            off = {s.replace('off_', 'full_'): s for s in sites if s.startswith('off_')}
            xo = [main.index(k) for k in off if k in main]
            ax.plot(xo, [sites[off[main[i]]][target][src] for i in xo], 's--', color='#eb6834', ms=5, lw=1.4,
                    label='Qwen3-VL-8B Off')
            ax.set_xticks(x, [s.replace('full_', '').replace('pD', '+D') for s in main], rotation=60, ha='right', fontsize=8)
        if 'qwen35_9b' in rep:
            s35 = rep['qwen35_9b']['sites']
            mapping = {s: s for s in ('vit9', 'vit17', 'vit25', 'vit27', 'V')}
            mapping.update({f'full_L{k}': f'full_L{k}' for k in (1, 2, 3, 4, 8, 16, 24)})
            mapping['full_L36'] = 'full_L32'
            xs, ys = [], []
            for k_vl, k35 in mapping.items():
                if k_vl in main and k35 in s35:
                    xs.append(main.index(k_vl))
                    ys.append(s35[k35][target][src])
            ax.plot(xs, ys, 'D:', color='#1baf7a', ms=5, lw=1.4, label='Qwen3.5-9B (last decoder layer at L36 slot = L32)')
        ax.set_title(src_names[src], loc='left')
        ax.axhline(0, color=MUTED, lw=0.8)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_share(summary: dict, out: Path) -> None:
    plt = _style()
    rep = summary['representation']
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    styles = {('qwen3_vl_8b', 'synthetic'): ('#2a78d6', 'o-'), ('qwen3_vl_8b', 'refcoco'): ('#2a78d6', 's--'),
              ('qwen3_vl_8b', 'babyvision'): ('#2a78d6', '^:'), ('qwen35_9b', 'synthetic'): ('#1baf7a', 'o-'),
              ('qwen35_9b', 'refcoco'): ('#1baf7a', 's--'), ('qwen35_9b', 'babyvision'): ('#1baf7a', '^:')}
    for (ms, src), (col, st) in styles.items():
        if ms not in rep:
            continue
        sh = rep[ms]['positional_share']
        xs = [SHARE_SITES.index(s) for s in sh]
        ax.plot(xs, [sh[s][src] for s in sh], st, color=col, ms=5, lw=1.4,
                label=f"{'Qwen3-VL-8B' if ms == 'qwen3_vl_8b' else 'Qwen3.5-9B'} · {src}")
    ax.set_xticks(range(len(SHARE_SITES)), SHARE_SITES)
    ax.set_ylabel(f'variance share in top-{SHARE_K} uniform-image subspace')
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_recon(output: Path, out: Path) -> None:
    """Grey-footprint reconstructions of held-out images from several sites (ridge readout per token)."""
    plt = _style()
    rows = []
    for ms, sites in (('qwen3_vl_8b', ('vit9', 'D1', 'vit27', 'V', 'full_L4', 'off_L4')), ('qwen35_9b', ('vit9', 'vit27', 'V', 'full_L4'))):
        p = output / 'analysis' / f'recon_{ms}.npz'
        if p.exists():
            rows.append((ms, sites, dict(np.load(p))))
    if not rows:
        return
    rep = read_jsonl(output / 'rep_images.jsonl')
    view_imgs = list(dict.fromkeys(rows[0][2]['image'].tolist()))
    chosen = []
    for src in REP_N:      # one image per source
        for v in view_imgs:
            if rep[v]['source'] == src:
                chosen.append(v)
                break
    ncol = 1 + sum(len(s) for _, s, _ in rows)
    fig, axes = plt.subplots(len(chosen), ncol, figsize=(1.9 * ncol, 2.0 * len(chosen)), squeeze=False)
    for i, v in enumerate(chosen):
        image = Image.open(output / rep[v]['image']).convert('L')
        tg = token_targets(image)
        rws, cls = tg['grid']
        crop = (slice(0, min(rws, 10) * 32), slice(0, min(cls, 10) * 32))   # top-left 10x10 tokens for legibility
        axes[i][0].imshow(np.asarray(image, float)[crop] / 255, cmap='gray', vmin=0, vmax=1)
        axes[i][0].set_title('input (grey)' if i == 0 else '', fontsize=8)
        col = 1
        for ms, sites, rec in rows:
            mask = rec['image'] == v
            toks = rec['token'][mask]
            for s in sites:
                canvas = np.zeros((rws * 32, cls * 32))
                vals = rec[s][mask].astype(float)
                for t, val in zip(toks, vals):
                    r, c = divmod(int(t), cls)
                    canvas[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = val.reshape(32, 32)
                im = axes[i][col].imshow(np.clip(canvas[crop], 0, 1), cmap='gray', vmin=0, vmax=1)
                name = ('VL ' if ms == 'qwen3_vl_8b' else '3.5 ') + s.replace('full_', 'Full ').replace('off_', 'Off ')
                axes[i][col].set_title(name if i == 0 else '', fontsize=8)
                col += 1
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label('grey level (0 = black, 1 = white)', fontsize=8)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ablation', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip-representation', action='store_true')
    a = p.parse_args()
    ana = a.output / 'analysis'
    ana.mkdir(exist_ok=True)
    summary = {'behavior': analyze_behavior(a.output, a.ablation, a.n_boot, a.seed)}
    cached = ana / 'representation.json'
    if a.skip_representation and cached.exists():
        summary['representation'] = json.loads(cached.read_text())
    else:
        summary['representation'] = analyze_representation(a.output, a.seed, a.n_boot)
        cached.write_text(json.dumps(summary['representation'], indent=1))
    (ana / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    write_tables(summary, ana / 'tables.md')
    figs = {'sweep.png': lambda o: plot_sweep(summary, o), 'mirror.png': lambda o: plot_mirror(a.output, a.ablation, o),
            'general.png': lambda o: plot_general(summary, o),
            'inputs.png': lambda o: plot_inputs(a.output, o),
            'mirror_refcoco.png': lambda o: plot_mirror_refcoco(a.output, a.ablation, o)}
    if summary['representation']:
        figs.update({'position_readability.png': lambda o: plot_readability(summary, o, 'position', 'position R² (held-out)'),
                     'structure_readability.png': lambda o: plot_readability(summary, o, 'structure', '4-px structure R² (held-out, content tokens)'),
                     'hfenergy_readability.png': lambda o: plot_readability(summary, o, 'hfenergy', 'high-frequency energy map R² (held-out, content tokens)'),
                     'coarse_readability.png': lambda o: plot_readability(summary, o, 'coarse', '4×4 colour R² (held-out, content tokens)'),
                     'reconstruction.png': lambda o: plot_recon(a.output, o)})
    a.figures.mkdir(parents=True, exist_ok=True)
    for name, fn in figs.items():
        fn(ana / name)
        if (ana / name).exists():
            shutil.copy(ana / name, a.figures / name)
    print('wrote', ana / 'tables.md')


if __name__ == '__main__':
    main()
