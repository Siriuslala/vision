"""CPU analysis and figures of experiment 3.14 (causal mask vs linear attention).

Inputs: the run directory written by `causal_vs_linear_run.py` / `causal_vs_linear_toy.py`, the 3.11 run
(cases and the Qwen3.5 no-edit KV records used as reference) and the 1.5 run (slot-grid truth and Qwen3.5
answers). Outputs: `summary.json` in the run directory and PNG figures.

Summaries:
  kv        mean recovered fraction r per token group, per model / mode / condition / pairing / axis, with a
            bootstrap 95% interval over values (3.11 definition, r undefined when d_A = d_B or |LD_B - LD_A| < 2);
  attention per model / mode / condition and query axis: mass of the head-mean attention on profile groups at
            the layers holding most of the image attention, and the share on token column 0;
  gen       answer validity and correctness per condition;
  ordinal   1.5 slot grids: accuracy, restart (answer = count since the last blank), +1 rates; running-count
            probe restart weight b (ridge on clean cells, base-grouped 5-fold);
  toy       accuracy and carrier recovery of the toy models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import rowcol_mechanism as rm
from analysis_tools.representation.sees_right import read_jsonl

GROUPS = ('obj', 'row_left', 'row_right', 'below_cols', 'below', 'above', 'ring1', 'ring5p', 'all_but_obj', 'all_img', 'text')
COLORS = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
          'green': '#008300', 'violet': '#6250d6', 'grey': '#8a8984'}


def wil(k: int, n: int) -> dict:
    lo, hi = rm.wilson(k, n)
    return {'k': int(k), 'n': int(n), 'p': (k / n) if n else float('nan'), 'lo': lo, 'hi': hi}


def boot(x, n: int = 2000, seed: int = 0) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'n': 0}
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return {'mean': float(x.mean()), 'lo': float(np.quantile(bs, 0.025)), 'hi': float(np.quantile(bs, 0.975)), 'n': int(len(x))}


def recovery(v: dict, cond: str) -> float | None:
    lda, ldb = v['LD'].get('none'), v['LD_B']
    if lda is None or cond not in v['LD'] or v['dA'] is None or v['dA'] == v['dB'] or abs(ldb - lda) < 2:
        return None
    return (v['LD'][cond] - lda) / (ldb - lda)


AXIS_OF_KEY = {'x': 'h', 'column': 'h', 'y': 'v', 'row': 'v'}


# ----------------------------------------------------------------------------- KV

def kv_summary(path: Path, extra_groups: tuple = ()) -> dict:
    """{cond|kind|axis: {group: boot}} restricted to values on the displacement axis."""
    recs = read_jsonl(path)
    acc: dict = {}
    for r in recs:
        for pr in r['pairs']:
            kind = 'splice' if pr['kind'] == 'splice' else ('swap' if pr['kind'] == 'swap' else 'move')
            for v in pr['values']:
                if AXIS_OF_KEY.get(v['key'], v['axis']) != pr['axis']:
                    continue
                key = f"{r['cond']}|{kind}|{pr['axis']}"
                d = acc.setdefault(key, {})
                names = [n for n in v['LD'] if n != 'none']
                for g in names:
                    d.setdefault(g, []).append(recovery(v, g))
    return {k: {g: boot(x) for g, x in d.items()} for k, d in acc.items()}


def layer_profile(summary: dict, key: str, group: str = 'all_img') -> tuple[list[str], list[float]]:
    d = summary.get(key, {})
    names = sorted([n for n in d if n.startswith(group + '@')], key=lambda n: int(n.split('@')[1].split('-')[0]))
    return [n.split('@')[1] for n in names], [d[n]['mean'] for n in names]


# ----------------------------------------------------------------------------- attention

def attention_summary(run: Path, tag: str, recs: list[dict]) -> dict:
    """Head-mean attention mass on profile groups at the first-digit query of each value, per axis.

    Layers: the full-attention layers whose head-mean image share at the digit queries is in the top quarter
    for this model and mode (reported), so that the comparison uses each model's own 'looking' layers.
    Also: share on token column 0 relative to its uniform expectation, and per layer image share.
    """
    out = {}
    for cond in sorted({r['cond'] for r in recs}):
        rows = {'h': [], 'v': []}
        img_share = []
        for r in recs:
            if r['cond'] != cond:
                continue
            f = run / f'kv_attn_{tag}' / f"{r['id']}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            mass, groups = z['mass'], list(z['groups'])            # (layers, heads, K, G)
            H, W = r['grid']
            for k, v in enumerate(r['values_meta']):
                ax = AXIS_OF_KEY.get(v['key'])
                if ax is None:
                    continue
                m = mass[:, :, k].mean(1)                            # (layers, G)
                rows[ax].append(m)
                img_share.append(1 - m[:, groups.index('text')])
        if not img_share:
            continue
        share = np.mean(img_share, 0)
        top = np.argsort(share)[::-1][:max(2, len(share) // 4)]
        res = {'image_share_by_layer': share.tolist(), 'top_layers_index': sorted(top.tolist()), 'groups': groups}
        for ax in ('h', 'v'):
            if not rows[ax]:
                continue
            m = np.stack(rows[ax])                                   # (n, layers, G)
            sel = m[:, top].mean(1)                                  # (n, G)
            img = 1 - sel[:, groups.index('text')]
            res[ax] = {g: boot(sel[:, i] / img) for i, g in enumerate(groups) if g != 'text'}
            res[ax]['n'] = int(len(m))
            res[ax]['per_layer'] = {g: (m[:, :, i] / (1 - m[:, :, groups.index('text')])).mean(0).tolist()
                                    for i, g in enumerate(groups) if g != 'text'}
        out[cond] = res
    return out


# ----------------------------------------------------------------------------- generation

def gen_summary(path: Path) -> dict:
    out = {}
    for r in read_jsonl(path):
        d = out.setdefault(r['cond'], {'n': 0, 'valid': 0, 'correct': 0, 'd_row': [], 'd_col': []})
        d['n'] += 1
        d['valid'] += bool(r['score']['valid'])
        d['correct'] += bool(r['score'].get('correct'))
        for key in ('d_row', 'd_col'):
            if key in r['score']:
                d[key].append(r['score'][key])
    for d in out.values():
        d['acc'] = wil(d['correct'], d['n'])
        for key in ('d_row', 'd_col'):
            x = d.pop(key)
            if x:
                d[key + '_hist'] = {str(k): int(v) for k, v in zip(*np.unique(x, return_counts=True))}
    return out


# ----------------------------------------------------------------------------- ordinal

def ordinal_truth(ordinal_run: Path, conds) -> dict:
    cases = rm.load_ordinal_cases(ordinal_run, tuple(conds))
    return {c['id']: c for c in cases}


def ordinal_behaviour(recs: list[dict], truth: dict) -> dict:
    out = {}
    for cond in sorted({r['condition'] for r in recs}):
        sub = [r for r in recs if r['condition'] == cond]
        gap = [r for r in sub if r['variant'] in ('insL', 'delL') and truth[r['id']]['truth']['gap'] != truth[r['id']]['truth']['count']]
        clean = [r for r in sub if r['variant'] in ('base', 'shift', 'delR')]
        t = lambda r, k: truth[r['id']]['truth'][k]
        e = {'restart': wil(sum(r['answer_long'] == t(r, 'gap') for r in gap), len(gap)),
             'count_gap': wil(sum(r['answer_long'] == t(r, 'count') for r in gap), len(gap)),
             'slot_gap': wil(sum(r['answer_long'] == t(r, 'slot') for r in gap), len(gap)),
             'acc_clean': wil(sum(r['answer_long'] == t(r, 'count') for r in clean), len(clean)),
             'plus1_clean': wil(sum(r['answer_long'] is not None and r['answer_long'] - t(r, 'count') == 1 for r in clean), len(clean)),
             'minus1_clean': wil(sum(r['answer_long'] is not None and r['answer_long'] - t(r, 'count') == -1 for r in clean), len(clean)),
             'short_acc': wil(sum(r['answer_short'] == truth[r['id']]['truth']['other'] for r in sub), len(sub)),
             'invalid': int(sum(r['answer_long'] is None for r in sub))}
        big = [r for r in clean if t(r, 'count') >= 8]
        e['plus1_k8'] = wil(sum(r['answer_long'] is not None and r['answer_long'] - t(r, 'count') == 1 for r in big), len(big))
        out[cond] = e
    return out


def paired_restart(recs_a: list[dict], recs_b: list[dict], truth: dict) -> dict:
    """McNemar on the restart category between two runs of the same images (a = reference)."""
    a = {r['id']: r for r in recs_a}
    out = {}
    for cond in sorted({r['condition'] for r in recs_b}):
        ids = [r['id'] for r in recs_b if r['condition'] == cond and r['variant'] in ('insL', 'delL') and r['id'] in a
               and truth[r['id']]['truth']['gap'] != truth[r['id']]['truth']['count']]
        b = {r['id']: r for r in recs_b}
        g = lambda r: r['answer_long'] == truth[r['id']]['truth']['gap']
        lost = sum(g(a[i]) and not g(b[i]) for i in ids)
        gained = sum(not g(a[i]) and g(b[i]) for i in ids)
        out[cond] = {'lost': int(lost), 'gained': int(gained), 'p': rm.mcnemar_exact(lost, gained), 'n': len(ids)}
    return out


def probe_restart(fdir: Path, recs: list[dict], truth: dict, n_pc: int = 128, lam=(1e0, 1e1, 1e2, 1e3)) -> dict:
    """Cached wrapper of _probe_restart (cache keyed by the record ids)."""
    if not recs:
        return {}
    key = f"{fdir.name}_{recs[0]['condition']}_{len(recs)}"
    cache = fdir.parent / 'probe_cache.json'
    store = json.loads(cache.read_text()) if cache.exists() else {}
    if key not in store:
        store[key] = _probe_restart(fdir, recs, truth, n_pc, lam)
        cache.write_text(json.dumps(store))
    return store[key]


def _probe_restart(fdir: Path, recs: list[dict], truth: dict, n_pc: int, lam) -> dict:
    """Running-count probe on the 21 target-line cells: restart weight b and clean R^2 per layer.

    Training cells: no interior blank before them (any variant); 5 folds by base. On cells after an interior
    blank with k - k_gap >= 1: pred - k = a + b (k_gap - k); b = 1 is a full restart, 0 keeps the running count.
    """
    X, rows = [], []
    for r in recs:
        c = truth[r['id']]
        geo = rm.geometry(c['truth'], c['axis'])
        f = fdir / r['file']
        if not f.exists():
            continue
        import torch
        feats = torch.load(f).float().numpy()                       # (layers, 21, D)
        for slot in range(21):
            if slot not in geo.present:
                continue
            holes = [i for i in range(geo.present[0], slot) if i not in geo.present]
            rows.append((c['base'], geo.running_count(slot), geo.count_since_gap(slot), bool(holes)))
            X.append(feats[:, slot])
    if not X:
        return {}
    X = np.stack(X, 1)                                              # (layers, N, D)
    base, k, kg, gap = (np.array(v) for v in zip(*rows))
    k, kg = k.astype(float), kg.astype(float)
    folds = base % 5
    out = []
    for li in range(X.shape[0]):
        Z = X[li] - X[li][~gap].mean(0)
        U, S, Vt = np.linalg.svd(Z[~gap], full_matrices=False)
        P = Z @ Vt[:n_pc].T
        pred = np.zeros(len(k))
        for f in range(5):
            tr, te = (folds != f) & ~gap, folds == f
            # inner choice of the ridge penalty on a held-out base fold inside the training folds
            best, bw = None, None
            inner = (folds[tr] == (f + 1) % 5)
            for l in lam:
                A = P[tr][~inner]
                w = np.linalg.solve(A.T @ A + l * np.eye(A.shape[1]), A.T @ (k[tr][~inner] - k[tr][~inner].mean()))
                err = np.mean((P[tr][inner] @ w + k[tr][~inner].mean() - k[tr][inner]) ** 2)
                if best is None or err < best:
                    best, bl = err, l
            A = P[tr]
            w = np.linalg.solve(A.T @ A + bl * np.eye(A.shape[1]), A.T @ (k[tr] - k[tr].mean()))
            pred[te] = P[te] @ w + k[tr].mean()
        clean, after = ~gap, gap & (k - kg >= 1)
        r2 = 1 - ((pred[clean] - k[clean]) ** 2).sum() / ((k[clean] - k[clean].mean()) ** 2).sum()
        A = np.stack([np.ones(after.sum()), kg[after] - k[after]], 1)
        coef = np.linalg.lstsq(A, pred[after] - k[after], rcond=None)[0] if after.sum() > 5 else [np.nan, np.nan]
        out.append({'r2_clean': float(r2), 'slot_follow': float(coef[0]), 'restart_weight': float(coef[1]),
                    'n_after': int(after.sum()), 'n_clean': int(clean.sum())})
    return {'per_layer': out}


# ----------------------------------------------------------------------------- toy

def toy_summary(root: Path) -> dict:
    """Per (arch, layout): mean accuracy and carrier recovery over seeds, plus per-seed off-target shares.

    row_share (row query): recovery carried by the target row outside the target (row_left + row_right);
    col_share (column query): recovery carried by the target column's neighbours above and below (above +
    below; with a one-cell move right the 'below' group is dominated by the column-mates, see below_cols).
    Duplicate records of the same (arch, layout, seed) keep the last one.
    """
    recs = {}
    for f in sorted(root.glob('*/results.jsonl')):
        for r in read_jsonl(f):
            recs[(r['arch'], r['layout'], r['seed'])] = r
    out = {}
    for (arch, layout, seed), r in sorted(recs.items()):
        d = out.setdefault(f'{arch}|{layout}', {'acc_row': [], 'acc_col': [], 'row': {}, 'col': {}, 'seeds': []})
        d['acc_row'].append(r['acc']['row'])
        d['acc_col'].append(r['acc']['col'])
        cr_, cc_ = r['carriers']['row'], r['carriers']['col']
        for ax, c in (('row', cr_), ('col', cc_)):
            for g, v in c.items():
                if g != 'n' and v is not None:
                    d[ax].setdefault(g, []).append(v)
        d['seeds'].append({'seed': seed, 'row_share': (cr_['row_left'] or 0) + (cr_['row_right'] or 0),
                           'row_col_share': (cr_['above'] or 0) + (cr_['below'] or 0),
                           'col_share': (cc_['above'] or 0) + (cc_['below'] or 0),
                           'col_row_share': (cc_['row_left'] or 0) + (cc_['row_right'] or 0),
                           'row_obj': cr_['obj'], 'col_obj': cc_['obj'], 'n_row': cr_['n'], 'n_col': cc_['n']})
    for d in out.values():
        for ax in ('row', 'col'):
            d[ax] = {g: float(np.mean(v)) for g, v in d[ax].items()}
        d['acc_row'], d['acc_col'] = float(np.mean(d['acc_row'])), float(np.mean(d['acc_col']))
        d['n_seeds'] = len(d['seeds'])
        d['row_offtarget_runs'] = int(sum(x['row_share'] > 0.2 for x in d['seeds']))
        d['col_column_runs'] = int(sum(x['col_share'] > 0.2 for x in d['seeds']))
        d['row_column_runs'] = int(sum(x['row_col_share'] > 0.2 for x in d['seeds']))
    return out


# ----------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def fig_kv(S: dict, out: Path) -> None:
    """Grouped bars: recovered fraction per token group, splice pairs, for x (h) and y (v), per model/mode."""
    plt = _plt()
    groups = ('obj', 'row_left', 'row_right', 'below_cols', 'below', 'above')
    labels = ['target', 'row\nbefore', 'row\nafter', 'column\nbelow', 'rows\nbelow', 'above']
    series = [('Qwen3.5 no edit', 'q35_none', COLORS['blue']), ('Qwen3.5 target no-write', 'q35_nowrite_target', COLORS['orange']),
              ('Qwen3.5 DeltaNet L9-16 frozen', 'q35_freeze_l9_16', COLORS['yellow']), ('Qwen3-VL DeepStack on', 'vl_ds_on', COLORS['aqua']),
              ('Qwen3-VL DeepStack off', 'vl_ds_off', COLORS['violet'])]
    conds = [('t6g10_rowcol', 'h', '10x10 grid, column'), ('t6g10_rowcol', 'v', '10x10 grid, row'),
             ('t6g10_point', 'h', '10x10 grid, point x'), ('t6g10_point', 'v', '10x10 grid, point y'),
             ('disc_point', 'h', 'single disc, x'), ('disc_point', 'v', 'single disc, y')]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharey=True)
    for ax, (cond, axis, title) in zip(axes.ravel(), conds):
        present = [(n, t, c) for n, t, c in series if f'{cond}|splice|{axis}' in S['kv'].get(t, {})]
        w = 0.8 / max(1, len(present))
        for i, (name, tag, col) in enumerate(present):
            d = S['kv'][tag][f'{cond}|splice|{axis}']
            m = [d.get(g, {}).get('mean', np.nan) for g in groups]
            lo = [d.get(g, {}).get('lo', np.nan) for g in groups]
            hi = [d.get(g, {}).get('hi', np.nan) for g in groups]
            x = np.arange(len(groups)) + (i - (len(present) - 1) / 2) * w
            ax.bar(x, m, w * 0.92, color=col, label=name)
            ax.errorbar(x, m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)], fmt='none', ecolor='#52514e', lw=0.8)
        ax.axhline(0, color='#52514e', lw=0.6)
        ax.set_xticks(range(len(groups)), labels)
        ax.set_title(title)
        ax.set_ylabel('recovered fraction r')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'kv_groups.png', dpi=150)
    plt.close(fig)


def fig_attention(S: dict, out: Path) -> None:
    """Head-mean attention enrichment (mass / uniform share) on groups at x vs y queries, 10x10 row/col."""
    plt = _plt()
    groups = ('target', 'row_before', 'row_after', 'col', 'col0', 'below', 'above')
    labels = ['target', 'row before', 'row after', 'target col', 'col 0', 'below', 'above']
    tags = [('Qwen3.5 no edit', 'q35_none'), ('Qwen3.5 frozen', 'q35_freeze'), ('Qwen3-VL DS on', 'vl_ds_on'), ('Qwen3-VL DS off', 'vl_ds_off')]
    tags = [t for t in tags if 't6g10_rowcol' in S['attention'].get(t[1], {})]
    fig, axes = plt.subplots(1, len(tags), figsize=(3.3 * len(tags), 3.4), sharey=True, squeeze=False)
    for ax, (name, tag) in zip(axes[0], tags):
        d = S['attention'][tag]['t6g10_rowcol']
        sizes = S['attention_sizes']['t6g10_rowcol']
        x = np.arange(len(groups))
        for j, (axis, col, lab) in enumerate((('h', COLORS['orange'], 'column query'), ('v', COLORS['blue'], 'row query'))):
            if axis not in d:
                continue
            m = [d[axis][g]['mean'] / sizes[g] for g in groups]
            ax.bar(x + (j - 0.5) * 0.4, m, 0.38, color=col, label=lab)
        ax.axhline(1, color='#52514e', lw=0.6, ls='--')
        ax.set_yscale('log')
        ax.set_xticks(x, labels, rotation=45)
        ax.set_title(name)
    axes[0, 0].set_ylabel('attention enrichment (x uniform)')
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'attention_profile.png', dpi=150)
    plt.close(fig)


def fig_layers(S: dict, out: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, axis, title in ((axes[0], 'h', 'column (10x10 row/col, true move)'), (axes[1], 'v', 'row (10x10 row/col, true move)')):
        for name, tag, col in (('Qwen3.5', 'q35_none', COLORS['blue']), ('Qwen3-VL DS on', 'vl_ds_on', COLORS['aqua']),
                               ('Qwen3-VL DS off', 'vl_ds_off', COLORS['violet'])):
            names, vals = layer_profile(S['kv'].get(tag, {}), f't6g10_rowcol|move|{axis}')
            if not names:
                continue
            centers = [np.mean([int(v) for v in n.split('-')]) / (32 if tag.startswith('q35') else 36) for n in names]
            ax.plot(centers, vals, '-o', color=col, lw=2, ms=4, label=name)
        ax.axhline(0, color='#52514e', lw=0.6)
        ax.set_xlabel('relative depth (layer / n layers)')
        ax.set_ylabel('image-KV recovery, single layer (bin)')
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'kv_layers.png', dpi=150)
    plt.close(fig)


def fig_ordinal(S: dict, out: Path) -> None:
    plt = _plt()
    conds = ['H32', 'H64', 'V32', 'V64']
    metrics = [('restart', 'restart after blank'), ('acc_clean', 'correct (no blank)'), ('plus1_k8', '+1 (count >= 8)')]
    series = [('Qwen3.5', 'q35', COLORS['blue']), ('Qwen3-VL DS on', 'vl_ds_on', COLORS['aqua']), ('Qwen3-VL DS off', 'vl_ds_off', COLORS['violet'])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        present = [s for s in series if s[1] in S['ordinal']]
        for i, (name, tag, col) in enumerate(present):
            d = S['ordinal'][tag]
            v = [d[c][m]['p'] if c in d else np.nan for c in conds]
            lo = [d[c][m]['lo'] if c in d else np.nan for c in conds]
            hi = [d[c][m]['hi'] if c in d else np.nan for c in conds]
            x = np.arange(len(conds)) + (i - (len(present) - 1) / 2) * 0.27
            ax.bar(x, v, 0.25, color=col, label=name)
            ax.errorbar(x, v, yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)], fmt='none', ecolor='#52514e', lw=0.8)
        ax.set_xticks(range(len(conds)), conds)
        ax.set_title(title)
    axes[0].set_ylabel('fraction')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'ordinal_compare.png', dpi=150)
    plt.close(fig)


def fig_knock(S: dict, out: Path) -> None:
    plt = _plt()
    K = S.get('knock', {})
    if not K:
        return
    names = list(K['behaviour'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for j, (cond, col) in enumerate((('H32', COLORS['orange']), ('H64', COLORS['blue']))):
        v = [K['behaviour'][n][cond]['restart']['p'] for n in names]
        lo = [K['behaviour'][n][cond]['restart']['lo'] for n in names]
        hi = [K['behaviour'][n][cond]['restart']['hi'] for n in names]
        x = np.arange(len(names)) + (j - 0.5) * 0.38
        axes[0].bar(x, v, 0.36, color=col, label=cond)
        axes[0].errorbar(x, v, yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)], fmt='none', ecolor='#52514e', lw=0.8)
        a = [K['behaviour'][n][cond]['acc_clean']['p'] for n in names]
        axes[1].bar(x, a, 0.36, color=col, label=cond)
    for ax, t in zip(axes, ('restart after blank (insL, delL)', 'long-axis accuracy, base images')):
        ax.set_xticks(range(len(names)), [n.replace('img:', 'KO ').replace('local:', 'local KO ') for n in names], rotation=30)
        ax.set_title(t)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'knockout_32px.png', dpi=150)
    plt.close(fig)


def fig_toy(S: dict, out: Path) -> None:
    plt = _plt()
    T = S.get('toy', {})
    if not T:
        return
    archs = ['causal_softmax', 'bidir_softmax', 'causal_delta', 'bidir_delta', 'hybrid']
    names = ['causal softmax', 'bidir softmax', 'causal linear', 'bidir linear', 'hybrid (2 linear + 1 softmax)']
    groups = ('obj', 'row_left', 'row_right', 'below', 'above')
    labels = ['target', 'row before', 'row after', 'below', 'above']
    cols = [COLORS['blue'], COLORS['aqua'], COLORS['orange'], COLORS['yellow'], COLORS['violet']]
    layouts = sorted({k.split('|')[1] for k in T})
    fig, axes = plt.subplots(len(layouts), 2, figsize=(11, 3.3 * len(layouts)), squeeze=False, sharey=True)
    for li, lay in enumerate(layouts):
        for j, (ax_name, title) in enumerate((('row', 'row query (target moved down)'), ('col', 'column query (target moved right)'))):
            ax = axes[li, j]
            present = [(a, n, c) for a, n, c in zip(archs, names, cols) if f'{a}|{lay}' in T]
            w = 0.8 / max(1, len(present))
            for i, (a, n, c) in enumerate(present):
                d = T[f'{a}|{lay}'][ax_name]
                m = [d.get(g, np.nan) for g in groups]
                ax.bar(np.arange(len(groups)) + (i - (len(present) - 1) / 2) * w, m, w * 0.92, color=c, label=n)
            ax.axhline(0, color='#52514e', lw=0.6)
            ax.set_xticks(range(len(groups)), labels)
            ax.set_title(f'{title}, layout {lay}')
            ax.set_ylabel('recovered fraction r')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / 'toy_carriers.png', dpi=150)
    plt.close(fig)
    # per-seed scatter: column-carrier share (column query) vs row-carrier share (row query)
    fig, axes = plt.subplots(1, len(layouts), figsize=(5.2 * len(layouts), 4.4), squeeze=False)
    marks = {'6x8k1': 'o', '4x5k2': 's'}
    for li, lay in enumerate(layouts):
        ax = axes[0, li]
        for a, n, c in zip(archs, names, cols):
            if f'{a}|{lay}' not in T:
                continue
            sd = T[f'{a}|{lay}']['seeds']
            ax.scatter([x['col_share'] for x in sd], [x['row_share'] for x in sd], s=60, color=c, edgecolor='white',
                       linewidth=1.5, marker=marks.get(lay, 'o'), label=n)
        ax.axvline(0.2, color='#8a8984', lw=0.6, ls='--')
        ax.axhline(0.2, color='#8a8984', lw=0.6, ls='--')
        ax.set_xlim(-0.05, 1)
        ax.set_ylim(-0.05, 1)
        ax.set_xlabel('column query: recovery in target column (above + below)')
        ax.set_ylabel('row query: recovery in target row outside target')
        ax.set_title(f'one point per trained model (seed), layout {lay}')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(out / 'toy_scatter.png', dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def fig_inputs(flow_run: Path, out: Path) -> None:
    """Input example (one 10x10 row/col case of 3.11) and the token groups used by the KV patching."""
    import matplotlib.patches as mp
    from PIL import Image
    plt = _plt()
    cases = json.loads((flow_run / 'cases.json').read_text())
    case = next(c for c in cases if c['cond'] == 't6g10_rowcol' and 3 <= c['truth']['target_rc'][1] <= 6
                and 3 <= c['truth']['target_rc'][0] <= 6)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.3))
    ax[0].imshow(Image.open(case['image_path']).convert('RGB'))
    x0, y0, x1, y1 = case['truth']['box']
    ax[0].add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='black', lw=1.2))
    ax[0].set_title('input: 10x10 grid (64 px cells)\n"Give its position ... {row, column}"', fontsize=9)
    ax[0].axis('off')
    H = W = 20
    colors = {'target / moved (D)': '#000000', 'row before': COLORS['aqua'], 'row after': COLORS['orange'],
              'column below': COLORS['yellow'], 'other rows below': COLORS['violet'], 'above': '#dddddd'}
    for k, (title, moved) in enumerate([('row query: target moved one cell down', (2, 0)),
                                        ('column query: target moved one cell right', (0, 2))]):
        a = ax[k + 1]
        r0, c0 = 8, 8
        tgt = [(r0 + i) * W + c0 + j for i in (0, 1) for j in (0, 1)]
        mov = [(r0 + moved[0] + i) * W + c0 + moved[1] + j for i in (0, 1) for j in (0, 1)]
        g = cf.token_groups((H, W), tgt, mov)
        img = np.ones((H * W, 3))
        def paint(idx, col):
            img[idx] = [int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        paint(g['above'], colors['above'])
        paint(g['below'], colors['other rows below'])
        paint(g['below_cols'], colors['column below'])
        paint(g['row_left'], colors['row before'])
        paint(g['row_right'], colors['row after'])
        paint(g['obj'], colors['target / moved (D)'])
        a.imshow(img.reshape(H, W, 3), interpolation='nearest')
        a.set_xticks(np.arange(-.5, W, 1), minor=True)
        a.set_yticks(np.arange(-.5, H, 1), minor=True)
        a.grid(which='minor', color='white', lw=0.4)
        a.tick_params(which='both', length=0)
        a.set_xticks([0, 10, 19])
        a.set_yticks([0, 10, 19])
        a.set_xlabel('token column')
        a.set_ylabel('token row')
        a.set_title(title + '\n(20x20 merged tokens, raster order left to right)', fontsize=9)
    ax[2].legend(handles=[mp.Patch(color=c, label=n) for n, c in colors.items()], loc='center left',
                 bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'inputs_groups.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--flow-run', type=Path, required=True)
    p.add_argument('--ordinal-run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    run = args.run
    S: dict = {'kv': {}, 'attention': {}, 'gen': {}, 'ordinal': {}, 'probe': {}}
    for f in sorted(run.glob('kv_*.jsonl')):
        tag = f.stem[3:]
        S['kv'][tag] = kv_summary(f)
        S['attention'][tag] = attention_summary(run, tag, read_jsonl(f))
    for f in sorted(run.glob('gen_*.jsonl')):
        S['gen'][f.stem[4:]] = gen_summary(f)
    # group sizes on the 10x10 grid (20x20 tokens, a 2x2-token target) for enrichment ratios
    from analysis_tools.position.causal_vs_linear import profile_masks
    sizes = {}
    for f in sorted(run.glob('kv_*.jsonl')):
        for r in read_jsonl(f):
            if r['cond'] == 't6g10_rowcol':
                H, W = r['grid']
                tg = r['pairs'][0]['target']
                m = profile_masks((H, W), tg)
                sizes.setdefault('t6g10_rowcol', []).append({g: v.sum() / (H * W) for g, v in m.items()})
    S['attention_sizes'] = {c: {g: float(np.mean([d[g] for d in v])) for g in v[0]} for c, v in sizes.items()}
    # Qwen3.5 no-edit reference from 3.11 when this run has none of its own
    if (args.flow_run / 'kv.jsonl').exists():
        S['kv']['q35_3.11'] = kv_summary(args.flow_run / 'kv.jsonl')
    # ordinal
    odir = run / 'ordinal'
    conds = ('H32', 'H64', 'V32', 'V64')
    truth = ordinal_truth(args.ordinal_run, conds)
    q35 = [{'id': c['id'], 'condition': c['condition'], 'variant': c['variant'], 'answer_long': c['answer_long'],
            'answer_short': c['answer_short']} for c in truth.values()]
    S['ordinal']['q35'] = ordinal_behaviour(q35, truth)
    for f in sorted(odir.glob('vl_*_none.jsonl')) if odir.exists() else []:
        tag = f.stem.rsplit('_none', 1)[0]
        recs = read_jsonl(f)
        S['ordinal'][tag] = ordinal_behaviour(recs, truth)
        S['probe'][tag] = {c: probe_restart(odir / f.stem, [r for r in recs if r['condition'] == c], truth) for c in conds}
    knock = sorted(odir.glob('q35_none_*.jsonl')) if odir.exists() else []
    if knock:
        K = {'behaviour': {}, 'paired': {}, 'probe': {}}
        recs_by = {f.stem[len('q35_none_'):]: read_jsonl(f) for f in knock}
        order = ['none'] + sorted([n for n in recs_by if n != 'none'], key=lambda n: (n.startswith('local'), len(n), n))
        for n in order:
            if n not in recs_by:
                continue
            K['behaviour'][n] = ordinal_behaviour(recs_by[n], truth)
            if n != 'none' and 'none' in recs_by:
                K['paired'][n] = paired_restart(recs_by['none'], recs_by[n], truth)
            K['probe'][n] = {c: probe_restart(odir / f'q35_none_{n}', [r for r in recs_by[n] if r['condition'] == c], truth)
                             for c in ('H32', 'H64')}
        S['knock'] = K
    if (run / 'toy').exists():
        S['toy'] = toy_summary(run / 'toy')
    (run / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    for fn in (fig_kv, fig_attention, fig_layers, fig_ordinal, fig_knock, fig_toy):
        try:
            fn(S, args.figures)
        except (KeyError, ValueError, IndexError) as exc:            # report, never hide, a missing input
            print(f'{fn.__name__} skipped: {exc!r}')
    fig_inputs(args.flow_run, args.figures)
    print('summary written')


if __name__ == '__main__':
    main()
