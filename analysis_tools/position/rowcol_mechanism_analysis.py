"""Analysis of experiment 3.10 (one mechanism for row and column ordinals). CPU only.

Commands:
  heads    boundary-reset heads from the gate stage: per Gated DeltaNet head, alpha at the boundary tokens
           (first figure tokens after an interior blank line) minus alpha at matched control tokens; writes
           boundary_heads.json, read by the gapgate stage;
  analyze  every statistic (results.json) and figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

from analysis_tools.position import rowcol_mechanism as rm
from analysis_tools.position import sequence_order_r2 as s2
from analysis_tools.position.sequence_order_analysis import BLUE, ORANGE, AQUA, YELLOW, VIOLET, INK, MUTED, GRID
from analysis_tools.position.sequence_order_run import read_jsonl

GREY = '#9a9893'
PINK = '#d6457f'
BOUNDARY_DROP = -0.2          # alpha(boundary) - alpha(control) below this marks a boundary-reset head
CMAP = 'viridis'


def wil(k, n):
    lo, hi = rm.wilson(k, n)
    return {'k': int(k), 'n': int(n), 'p': (k / n) if n else float('nan'), 'ci': [lo, hi]}


def boot_mean(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {'mean': float('nan'), 'ci': [float('nan')] * 2, 'n': 0}
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return {'mean': float(x.mean()), 'ci': [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))], 'n': int(len(x))}


def gate_cases(run: Path, ordinal_run: Path, conditions) -> list[dict]:
    rec = {r['id']: r for r in read_jsonl(run / 'gates' / 'records.jsonl')}
    cases = [c for c in rm.load_ordinal_cases(ordinal_run, tuple(conditions)) if c['id'] in rec]
    for c in cases:
        c['file'] = run / 'gates' / rec[c['id']]['file']
        c['geo'] = rm.geometry(c['truth'], c['axis'])
    return cases


# ----------------------------------------------------------------------------- gates at boundaries

def boundary_gate_stats(cases: list[dict]) -> dict:
    """Per condition: mean over images of alpha/beta at boundary, control, blank and row-start tokens, per head."""
    out = {}
    for cond in sorted({c['condition'] for c in cases}):
        acc = {k: [] for k in ('a_b', 'a_c', 'b_b', 'b_c', 'a_blank', 'b_blank', 'a_other', 'b_other', 'prof_a', 'prof_b')}
        for c in cases:
            if c['condition'] != cond or c['variant'] not in ('insL', 'delL'):
                continue
            g = c['geo']
            bt, ct = g.boundary_tokens(), g.matched_control_tokens()
            if len(ct) == 0:
                continue
            d = torch.load(c['file'])
            a, b = d['alpha'].float().numpy(), d['beta'].float().numpy()     # (24, Nv, 32)
            H, W = g.grid
            blank = np.array([t for p in g.interior_blanks() for o in range(6) for t in g.cell_tokens(p, o)])
            acc['a_b'].append(a[:, bt].mean(1)); acc['a_c'].append(a[:, ct].mean(1))
            acc['b_b'].append(b[:, bt].mean(1)); acc['b_c'].append(b[:, ct].mean(1))
            acc['a_blank'].append(a[:, blank].mean(1)); acc['b_blank'].append(b[:, blank].mean(1))
            other = np.setdiff1d(np.arange(H * W), np.concatenate([bt, ct, blank]))
            acc['a_other'].append(a[:, other].mean(1)); acc['b_other'].append(b[:, other].mean(1))
            # Profile along the raster sequence around each boundary token (offsets -6..+6, same token row for H).
            step = 1 if g.axis == 'H' else W
            offs = np.arange(-6, 7)
            idx = bt[:, None] + offs[None] * step
            ok = (idx >= 0) & (idx < H * W)
            idx = np.clip(idx, 0, H * W - 1)
            pa = np.where(ok[None, :, :, None], a[:, idx], np.nan)
            pb = np.where(ok[None, :, :, None], b[:, idx], np.nan)
            acc['prof_a'].append(np.nanmean(pa, 1)); acc['prof_b'].append(np.nanmean(pb, 1))   # (24, 13, 32)
        if not acc['a_b']:
            continue
        m = {k: np.mean(v, 0) for k, v in acc.items()}
        m['n_images'] = len(acc['a_b'])
        out[cond] = m
    return out


def select_heads(st: dict) -> dict:
    sel = {}
    for axis in ('H', 'V'):
        conds = [c for c in st if c.startswith(axis)]
        if not conds:
            continue
        d = np.mean([st[c]['a_b'] - st[c]['a_c'] for c in conds], 0)               # (24, 32)
        heads = {int(l): [int(h) for h in np.where(d[i] < BOUNDARY_DROP)[0]] for i, l in enumerate(rm.LINEAR_LAYERS)}
        sel[axis] = {'heads': heads, 'n': int(sum(len(v) for v in heads.values())), 'conditions': conds,
                     'd_alpha_min': float(d.min()), 'fallback': False}
    if 'V' in sel and sel['V']['n'] == 0 and 'H' in sel:
        sel['V'] = {**sel['H'], 'fallback': True, 'note': 'no V head passes the threshold; the H set is used'}
    return sel


def cmd_heads(args) -> None:
    run = Path(args.run)
    cases = gate_cases(run, Path(args.ordinal_run), ('H32', 'H64', 'V32', 'V64'))
    st = boundary_gate_stats(cases)
    sel = select_heads(st)
    classes = np.array(json.loads(Path(args.gate_stats).read_text())['classes'])
    for axis, v in sel.items():
        lab = [int(classes[rm.LINEAR_LAYERS.index(l), h]) for l, hs in v['heads'].items() for h in hs]
        v['class_counts'] = {rm.CLASS_NAMES[k]: lab.count(k) for k in rm.CLASS_NAMES}
    (run / 'boundary_heads.json').write_text(json.dumps(sel, indent=1))
    np.savez_compressed(run / 'boundary_gate_stats.npz', **{f'{c}__{k}': v for c, m in st.items() for k, v in m.items()
                                                         if isinstance(v, np.ndarray)})
    print(json.dumps({a: {k: v[k] for k in ('n', 'd_alpha_min', 'fallback', 'class_counts')} for a, v in sel.items()}))


def gate_summary(st: dict, sel: dict, classes: np.ndarray) -> dict:
    out = {}
    for cond, m in st.items():
        d = m['a_b'] - m['a_c']
        db = m['b_b'] - m['b_c']
        rr = classes == 2
        out[cond] = {'n_images': int(m['n_images']), 'n_drop_heads': int((d < BOUNDARY_DROP).sum()),
                     'min_d_alpha': float(d.min()),
                     'row_reset_heads_mean_d_alpha': float(d[rr].mean()),
                     'integrator_mean_d_alpha': float(d[classes == 1].mean()),
                     'blank_minus_other_alpha_all': float((m['a_blank'] - m['a_other']).mean()),
                     'blank_minus_other_beta_all': float((m['b_blank'] - m['b_other']).mean()),
                     'boundary_minus_control_beta_all': float(db.mean()),
                     'n_beta_up_heads': int((db > 0.2).sum())}
    hd = {a: np.mean([st[c]['a_b'] - st[c]['a_c'] for c in st if c.startswith(a)], 0) for a in ('H', 'V')}
    out['corr_H32_H64'] = float(np.corrcoef((st['H32']['a_b'] - st['H32']['a_c']).ravel(),
                                            (st['H64']['a_b'] - st['H64']['a_c']).ravel())[0, 1])
    out['corr_H_V'] = float(np.corrcoef(hd['H'].ravel(), hd['V'].ravel())[0, 1])
    # Overlap with the 3.8 row-reset heads.
    sh = sel['H']['heads']
    lab = [int(classes[rm.LINEAR_LAYERS.index(int(l)), h]) for l, hs in sh.items() for h in hs]
    out['H_heads_by_class'] = {rm.CLASS_NAMES[k]: lab.count(k) for k in rm.CLASS_NAMES}
    out['frac_row_reset_heads_dropping_H'] = float((hd['H'][classes == 2] < BOUNDARY_DROP).mean())
    return out


def fig_gates(st: dict, sel: dict, classes: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), gridspec_kw={'width_ratios': [1.2, 1.2, 1]})
    for i, axis in enumerate(('H', 'V')):
        d = np.mean([st[c]['a_b'] - st[c]['a_c'] for c in st if c.startswith(axis)], 0)
        db = np.mean([st[c]['b_b'] - st[c]['b_c'] for c in st if c.startswith(axis)], 0)
        for j, (mat, lab) in enumerate(((d, 'Δα'), (db, 'Δβ'))):
            ax = axes[i, j]
            im = ax.imshow(mat, cmap=CMAP, vmin=-0.6, vmax=0.3, aspect='auto')
            ax.grid(False)
            ax.set_yticks(range(0, 24, 3), [rm.LINEAR_LAYERS[k] for k in range(0, 24, 3)])
            ax.set_xlabel('head')
            ax.set_ylabel('DeltaNet layer')
            name = 'columns (H: blank column)' if axis == 'H' else 'rows (V: blank row)'
            ax.set_title(f'{lab} at boundary tokens − matched controls\n{name}', fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.04)
            cb.set_label(f'{lab} (gate units, 0–1 scale)')
        ax = axes[i, 2]
        conds = [c for c in st if c.startswith(axis)]
        prof = np.mean([st[c]['prof_a'] for c in conds], 0)                       # (24, 13, 32)
        hs = sel['H']['heads']
        groups = {'boundary-reset heads (H set)': [(rm.LINEAR_LAYERS.index(int(l)), h) for l, v in hs.items() for h in v],
                  'integrator heads (3.8)': list(zip(*np.where(classes == 1))),
                  'other heads': list(zip(*np.where(classes == 0)))}
        for (name, idx), col in zip(groups.items(), (ORANGE, BLUE, GREY)):
            if not idx:
                continue
            y = np.mean([prof[l, :, h] for l, h in idx], 0)
            ax.plot(np.arange(-6, 7), y, color=col, lw=2, marker='o', ms=4, label=name)
        ax.axvline(0, color=MUTED, ls=':', lw=1)
        ax.set_xlabel('offset from boundary token (tokens)' if axis == 'H' else 'offset from boundary token (token rows)')
        ax.set_ylabel('mean α')
        ax.set_ylim(0, 1.02)
        ax.set_title(f'α around the boundary ({"H" if axis == "H" else "V"})', fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_examples(cases: list[dict], ordinal_run: Path, path: Path) -> None:
    """Input examples: an H64 and a V64 image with an interior blank line, boundary and control tokens marked."""
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from PIL import Image
    picks = []
    for cond in ('H64', 'V64'):
        cand = [c for c in cases if c['condition'] == cond and c['variant'] == 'insL' and c['c'] >= 8]
        picks.append(cand[0])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={'width_ratios': [3.2, 1]})
    for ax, c in zip(axes, picks):
        g = c['geo']
        im = Image.open(ordinal_run / c['image']).convert('RGB')
        ax.imshow(im)
        H, W = g.grid
        for t, col, lw in [(t, ORANGE, 1.6) for t in g.boundary_tokens()] + [(t, AQUA, 1.2) for t in g.matched_control_tokens()]:
            r, q = divmod(int(t), W)
            ax.add_patch(Rectangle((q * 32, r * 32), 32, 32, fill=False, edgecolor=col, lw=lw))
        for t in g.cell_tokens(*g.target):
            r, q = divmod(int(t), W)
            ax.add_patch(Rectangle((q * 32, r * 32), 32, 32, fill=False, edgecolor=PINK, lw=2))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{c['id'].split('|')[0]}: count k = {c['truth']['count']}, restarted count = {c['truth']['gap']}, "
                     f"slot = {c['truth']['slot']}\nmodel answer (direct) = {c['answer_long']}", fontsize=9)
    handles = [Line2D([], [], color=ORANGE, lw=2, label='boundary tokens (first figure token after the blank line)'),
               Line2D([], [], color=AQUA, lw=2, label='matched control tokens (3 lines further, no blank)'),
               Line2D([], [], color=PINK, lw=2, label='target cell tokens (red figure)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- running-count probe

def probe_dataset(cases: list[dict], cond: str):
    """Cell-level features (7, N, D) and labels for one condition; blank cells are excluded."""
    X, rows = [], []
    for c in cases:
        if c['condition'] != cond:
            continue
        g = c['geo']
        res = torch.load(c['file'])['res'].float()                                  # (7, 42, D)
        o2 = (g.target[1] + 3) % 6
        for j in range(42):
            slot, o = j % 21, (g.target[1] if j < 21 else o2)
            if slot not in g.present:
                continue
            holes = [i for i in range(g.present[0], slot) if i not in g.present]
            rows.append({'base': c['base'], 'variant': c['variant'], 'slot': slot, 'k': g.running_count(slot),
                         'kg': g.count_since_gap(slot), 'gap_before': bool(holes), 'target': slot == g.target[0] and j < 21, 'line': j < 21,
                         'answer': c['answer_long'], 'c': c['c']})
            X.append(res[:, j])
    return torch.stack(X, 1), rows


def probe_condition(X: torch.Tensor, rows: list[dict], n_pc: int = 256) -> dict:
    """Train on cells without an interior blank before them; test on held-out bases (5 folds)."""
    k = np.array([r['k'] for r in rows], float)
    kg = np.array([r['kg'] for r in rows], float)
    slot = np.array([r['slot'] + 1 for r in rows], float)
    gap = np.array([r['gap_before'] for r in rows])
    base = np.array([r['base'] for r in rows])
    folds = base % 5
    out = {'layers': list(range(X.shape[0])), 'per_layer': []}
    preds_all = []
    for li in range(X.shape[0]):
        Z = X[li] - X[li].mean(0)
        torch.manual_seed(0)                      # pca_lowrank is randomized; fix it for reproducible numbers
        U, S, V = torch.pca_lowrank(Z, q=n_pc, center=False)
        P = (Z @ V).double()
        pred = np.zeros(len(rows))
        for f in range(5):
            tr = torch.as_tensor((folds != f) & ~gap)
            te = folds == f
            model, _ = s2.ridge_select(P[tr], torch.as_tensor(k[tr.numpy()]), base[tr.numpy()])
            pred[te] = s2.ridge_predict(model, P[torch.as_tensor(te)]).numpy()
        preds_all.append(pred)
        clean, after = ~gap, gap & (k - kg >= 1)
        resid = pred[clean] - k[clean]
        # pred - k = a + b (kg - k) on cells after an interior blank: a = slot follow (slot+1-k = 1), b = restart weight.
        A = np.stack([np.ones(after.sum()), kg[after] - k[after]], 1)
        coef = np.linalg.lstsq(A, pred[after] - k[after], rcond=None)[0] if after.sum() > 5 else [np.nan, np.nan]
        by_k = {}
        for kk in range(1, 21):
            m = clean & (k == kk)
            if m.sum() >= 5:
                e = pred[m] - kk
                by_k[kk] = {'bias': float(e.mean()), 'sd': float(e.std()), 'exact': float((np.abs(e) < 0.5).mean()),
                            'n': int(m.sum())}
        out['per_layer'].append({'r2_clean': float(1 - (resid ** 2).sum() / ((k[clean] - k[clean].mean()) ** 2).sum()),
                                 'slot_follow': float(coef[0]), 'restart_weight': float(coef[1]),
                                 'n_after': int(after.sum()), 'by_k': by_k})
    out['preds'] = np.stack(preds_all)
    out['k'], out['kg'], out['gap'] = k, kg, gap
    return out


def target_vs_answer(pr: dict, rows: list[dict], layer_index: int) -> dict:
    """Probe readout at the target cell for answers k+1 vs k, stratified by the true count k.

    Only targets without an interior blank before them and k >= 8. For every k, the readout is centred on the
    mean readout of all such targets with that k, so the comparison is not confounded by the count itself
    (ridge shrinkage makes the raw readout depend on k).
    """
    pred = pr['preds'][layer_index]
    sel = [(p, r) for p, r in zip(pred, rows)
           if r['target'] and not r['gap_before'] and r['answer'] is not None and r['k'] >= 8]
    mean_k = {}
    for p, r in sel:
        mean_k.setdefault(r['k'], []).append(p)
    mean_k = {k: float(np.mean(v)) for k, v in mean_k.items()}
    d = {'plus': [], 'exact': []}
    for p, r in sel:
        diff = r['answer'] - r['k']
        key = 'plus' if diff == 1 else 'exact' if diff == 0 else None
        if key:
            d[key].append(p - mean_k[r['k']])
    out = {k: boot_mean(v) for k, v in d.items()}
    out['plus_minus_exact'] = float(out['plus']['mean'] - out['exact']['mean']) if d['plus'] and d['exact'] else float('nan')
    rng = np.random.default_rng(1)
    if d['plus'] and d['exact']:
        bs = [rng.choice(d['plus'], len(d['plus'])).mean() - rng.choice(d['exact'], len(d['exact'])).mean() for _ in range(2000)]
        out['plus_minus_exact_ci'] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    return out


def fig_probe(prs: dict, rows_by: dict, examples: dict, layer_names, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cols = {'H64': ORANGE, 'H32': YELLOW, 'V64': BLUE, 'V32': AQUA}
    x = np.arange(len(layer_names))
    ax = axes[0, 0]
    for cond, pr in prs.items():
        ax.plot(x, [p['restart_weight'] for p in pr['per_layer']], color=cols[cond], marker='o', lw=2, label=cond)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls=':')
    ax.set_xticks(x, layer_names)
    ax.set_xlabel('layer')
    ax.set_ylabel('restart weight b')
    ax.set_title('(a) cells after a blank line: readout restarts at the blank?\n1 = count restarts, 0 = running count kept', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 1]
    for cond, pr in prs.items():
        ax.plot(x, [p['r2_clean'] for p in pr['per_layer']], color=cols[cond], marker='o', lw=2, label=cond)
    ax.set_xticks(x, layer_names)
    ax.set_xlabel('layer')
    ax.set_ylabel('held-out R² (cells without a blank before them)')
    ax.set_ylim(0.5, 1.0)
    ax.set_title('(b) running count is linearly readable at image tokens', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    li = layer_names.index('16')
    for j, (key, lab) in enumerate((('sd', 'readout SD (instances)'), ('bias', 'readout bias (instances)'))):
        ax = axes[0, 2] if j == 0 else axes[1, 2]
        for cond, pr in prs.items():
            bk = pr['per_layer'][li]['by_k']
            kk = sorted(bk)
            ax.plot(kk, [bk[q][key] for q in kk], color=cols[cond], marker='o', ms=4, lw=1.8, label=cond)
        ax.set_xlabel('running count k')
        ax.set_ylabel(lab)
        if key == 'bias':
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f'({"c" if j == 0 else "f"}) layer 16: ' + ('readout SD vs number of accumulated instances' if j == 0 else 'readout bias vs k (high-k negative part = ridge shrinkage)'), fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    for j, cond in enumerate(('H64', 'V64')):
        ax = axes[1, j]
        ex = examples.get(cond)
        if ex is None:
            ax.axis('off')
            continue
        ax.plot(ex['slot'], ex['k'], color=INK, lw=1.5, ls='-', marker='s', ms=4, label='running count k (truth)')
        ax.plot(ex['slot'], ex['kg'], color=GREY, lw=1.5, ls='--', marker='^', ms=4, label='count restarted at blank')
        ax.plot(ex['slot'], ex['pred'], color=cols[cond], lw=2, marker='o', ms=5, label='probe readout (layer 16)')
        for b in ex['blanks']:
            ax.axvspan(b + 0.5, b + 1.5, color=GRID, alpha=0.9, lw=0)
        ax.set_xlabel('long-axis slot (1-based); grey band = blank line')
        ax.set_ylabel('count')
        ax.set_title(f'({"d" if j == 0 else "e"}) {cond} example ({ex["id"]}): readout along the target line', fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- gate interventions at boundaries

def gapgate_stats(run: Path, stage: str = 'gapgate') -> dict:
    recs = read_jsonl(run / stage / 'records.jsonl')
    out = {}
    for cond in sorted({r['condition'] for r in recs}):
        for kind in ('gap', 'base'):
            sub = [r for r in recs if r['condition'] == cond and ((r['variant'] != 'base') if kind == 'gap'
                                                                  else (r['variant'] == 'base' and 'artificial_line' in r))]
            if not sub:
                continue
            names = sorted({n for r in sub for n in r['runs']})
            res = {}
            for n in names:
                rs = [(r, r['runs'][n]) for r in sub if n in r['runs']]
                if kind == 'gap':
                    rs2 = [(r, a) for r, a in rs if r['truth']['gap'] != r['truth']['count']]
                    res[n] = {'count': wil(sum(a['long'] == r['truth']['count'] for r, a in rs2), len(rs2)),
                              'restart': wil(sum(a['long'] == r['truth']['gap'] for r, a in rs2), len(rs2)),
                              'slot': wil(sum(a['long'] == r['truth']['slot'] for r, a in rs2), len(rs2)),
                              'mean_err': boot_mean([a['long'] - r['truth']['count'] for r, a in rs2 if a['long'] is not None]),
                              'short_correct': None}
                else:
                    res[n] = {'count': wil(sum(a['long'] == r['truth']['count'] for r, a in rs), len(rs)),
                              'restart': wil(sum(a['long'] == r['pred_restart'] for r, a in rs), len(rs)),
                              'lower': wil(sum(a['long'] is not None and a['long'] < r['truth']['count'] for r, a in rs), len(rs)),
                              'mean_err': boot_mean([a['long'] - r['truth']['count'] for r, a in rs if a['long'] is not None])}
            # Paired change of the restart category, no edit -> each edit (b: restart lost, c: restart gained).
            if kind == 'gap':
                rs2 = [r for r in sub if r['truth']['gap'] != r['truth']['count']]
                for n in names:
                    if n == 'none':
                        continue
                    b = sum(r['runs']['none']['long'] == r['truth']['gap'] and r['runs'][n]['long'] != r['truth']['gap'] for r in rs2)
                    c = sum(r['runs']['none']['long'] != r['truth']['gap'] and r['runs'][n]['long'] == r['truth']['gap'] for r in rs2)
                    res[f'mcnemar_restart|{n}'] = {'lost': b, 'gained': c, 'p': rm.mcnemar_exact(b, c)}
            out[f'{cond}|{kind}'] = res
    return out


def fig_gapgate(st: dict, st2: dict | None, path: Path, st3: dict | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    axes = axes.ravel()
    ax = axes[0]
    names = ['none', 'nodecay_boundary', 'nodecay_control', 'nodecay_random', 'reset_integrators']
    labels = ['no edit', 'α=1 at boundary\n(boundary heads)', 'α=1 at control\n(boundary heads)',
              'α=1 at boundary\n(random heads)', 'α=0 at boundary\n(integrators)']
    conds = [c for c in ('H32', 'H64', 'V32', 'V64') if f'{c}|gap' in st]
    w = 0.8 / len(names)
    colors = [GREY, ORANGE, YELLOW, AQUA, BLUE]
    for i, (n, lab, col) in enumerate(zip(names, labels, colors)):
        vals = [st[f'{c}|gap'].get(n, {}).get('restart', {'p': np.nan, 'ci': [np.nan] * 2}) for c in conds]
        p = np.array([v['p'] for v in vals])
        lo = p - np.array([v['ci'][0] for v in vals])
        hi = np.array([v['ci'][1] for v in vals]) - p
        ax.bar(np.arange(len(conds)) + (i - 2) * w, p, w * 0.92, color=col, yerr=np.clip([lo, hi], 0, None), capsize=2, label=lab.replace('\n', ' '))
    ax.set_xticks(range(len(conds)), conds)
    ax.set_ylabel('P(answer = count restarted at the blank)')
    ax.set_title('(a) insL / delL images: answers that restart counting at the blank line', fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax = axes[1]
    names = ['none', 'reset_boundary', 'reset_random']
    labels = ['no edit', 'α=0 at artificial boundary (boundary heads)', 'α=0 at artificial boundary (random heads)']
    conds = [c for c in ('H32', 'H64', 'V32', 'V64') if f'{c}|base' in st]
    for i, (n, lab, col) in enumerate(zip(names, labels, (GREY, ORANGE, AQUA))):
        vals = [st[f'{c}|base'].get(n, {}).get('count', {'p': np.nan, 'ci': [np.nan] * 2}) for c in conds]
        p = np.array([v['p'] for v in vals])
        ax.bar(np.arange(len(conds)) + (i - 1) * 0.27, p, 0.25, color=col,
               yerr=np.clip([p - np.array([v['ci'][0] for v in vals]), np.array([v['ci'][1] for v in vals]) - p], 0, None), capsize=2, label=lab)
    ax.set_xticks(range(len(conds)), conds)
    ax.set_ylabel('P(answer = true count k)')
    ax.set_ylim(0, 1)
    ax.set_title('(b) base images, c ≥ 6: artificial boundary 3 lines before the target\n(answers equal to the restarted count 4: 0 in every bar)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    if st2:
        names = ['none', 'nodecay_run|boundary_heads', 'nodecay_run|all_heads_mid', 'nodecay_run|all_heads_all_layers',
                 'nowrite_run|all_heads_all_layers']
        labels = ['no edit', 'α=1 on blank run + boundary (boundary heads)', 'α=1 on blank run + boundary (all heads, L9–16)',
                  'α=1 on blank run + boundary (all heads, all DeltaNet layers)', 'β=0 on blank run + boundary (all heads, all layers)']
        conds = [c for c in ('H32', 'H64') if f'{c}|gap' in st2]
        w = 0.8 / len(names)
        for i, (n, lab, col) in enumerate(zip(names, labels, (GREY, ORANGE, YELLOW, VIOLET, BLUE))):
            vals = [st2[f'{c}|gap'].get(n, {}).get('restart', {'p': np.nan, 'ci': [np.nan] * 2}) for c in conds]
            p = np.array([v['p'] for v in vals])
            ci = np.array([v['ci'] for v in vals])
            ax.bar(np.arange(len(conds)) + (i - 2) * w, p, w * 0.92, color=col,
                   yerr=np.clip([p - ci[:, 0], ci[:, 1] - p], 0, None), capsize=2, label=lab)
        ax.set_xticks(range(len(conds)), conds)
        ax.set_ylabel('P(answer = count restarted at the blank)')
        ax.set_title('(c) H grids: removing every DeltaNet reset over the blank run', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.axis('off')
    ax = axes[3]
    if st3:
        names = ['none', 'swap_boundary_cell', 'swap_control_cell']
        labels = ['no edit', 'boundary cell embeddings from the base image (figure on its left)',
                  'control cell embeddings from the base image']
        conds = [c for c in ('H32', 'H64') if f'{c}|gap' in st3]
        for metric, off, hatch in (('restart', 0, None), ('slot', 1, '//')):
            for i, (n, lab, col) in enumerate(zip(names, labels, (GREY, ORANGE, AQUA))):
                vals = [st3[f'{c}|gap'][n][metric] for c in conds]
                p = np.array([v['p'] for v in vals])
                ci = np.array([v['ci'] for v in vals])
                x = np.arange(len(conds)) * 2 + off + (i - 1) * 0.27
                ax.bar(x, p, 0.25, color=col, alpha=1 if hatch is None else 0.5, hatch=hatch,
                       yerr=np.clip([p - ci[:, 0], ci[:, 1] - p], 0, None), capsize=2,
                       label=f'{lab} – {"restart" if metric == "restart" else "slot (blank counted)"}')
        ax.set_xticks([0, 1, 2, 3], ['H32\nrestart', 'H32\nslot', 'H64\nrestart', 'H64\nslot'])
        ax.set_ylabel('fraction of answers')
        ax.set_title('(d) H grids: swapping the visual content of the cell after the blank', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- head classes (8x8)

def headclass_stats(run: Path) -> dict:
    recs = read_jsonl(run / 'headclass' / 'records.jsonl')
    names = list(recs[0]['runs'])
    out = {'n_heads': recs[0]['n_heads'], 'n_images': len(recs)}
    for n in names:
        rows = [(r['target'][0] + 1, r['target'][1] + 1, r['runs'][n]) for r in recs]
        err = [a['row'] - tr for tr, tc, a in rows if a['row'] is not None]
        by_row = {tr: float(np.mean([a['row'] - t for t, c, a in rows if t == tr and a['row'] is not None])) for tr in range(1, 9)}
        out[n] = {'row_acc': wil(sum(a['row'] == tr for tr, tc, a in rows), len(rows)),
                  'col_acc': wil(sum(a['col'] == tc for tr, tc, a in rows), len(rows)),
                  'row_err': boot_mean(err), 'row_err_by_row': by_row,
                  'under': int(sum(e < 0 for e in err)), 'over': int(sum(e > 0 for e in err))}
    return out


def fig_headclass(st: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    groups = ['integrator', 'row_reset', 'onset', 'other', 'all']
    lab = {'integrator': 'integrator', 'row_reset': 'row-reset', 'onset': 'cell-row onset', 'other': 'other', 'all': 'all heads'}
    for j, (kind, col) in enumerate((('nodecay', BLUE), ('nowrite', ORANGE))):
        for k, axis_key, hatch in ((0, 'row_acc', None), (1, 'col_acc', '//')):
            p = np.array([st[f'{kind}|{g}'][axis_key]['p'] for g in groups])
            ci = np.array([st[f'{kind}|{g}'][axis_key]['ci'] for g in groups])
            ax.bar(np.arange(len(groups)) + (2 * j + k - 1.5) * 0.2, p, 0.18, color=col, alpha=1 if k == 0 else 0.45,
                   hatch=hatch, yerr=np.clip([p - ci[:, 0], ci[:, 1] - p], 0, None), capsize=2,
                   label=f'{"α = 1" if kind == "nodecay" else "β = 0"}, {"row" if k == 0 else "column"} accuracy')
    ax.axhline(st['none']['row_acc']['p'], color=MUTED, ls=':', lw=1, label='no edit (row)')
    ax.set_xticks(range(len(groups)), [f"{lab[g]}\n({st['n_heads'].get(g, 192) if g != 'all' else 192} heads)" for g in groups])
    ax.set_ylabel('accuracy (128 images, 8×8 grids)')
    ax.set_title('(a) gate edits in layers 9–16 restricted to one head class', fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax = axes[1]
    for g, col in zip(groups, (BLUE, ORANGE, AQUA, GREY, INK)):
        y = [st[f'nodecay|{g}']['row_err_by_row'][r] for r in range(1, 9)]
        ax.plot(range(1, 9), y, color=col, marker='o', lw=2, label=f'α = 1, {lab[g]}')
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel('target row (1-based)')
    ax.set_ylabel('mean row error (answer − truth)')
    ax.set_title('(b) α = 1 restricted to one head class: mean row error by target row', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- implicit counting

def implicit_stats(run: Path) -> dict:
    recs = read_jsonl(run / 'implicit' / 'records.jsonl')
    out = {}
    for s in sorted({r['set'] for r in recs}):
        for mode in ('direct', 'think', 'thinking_on'):
            sub = [r for r in recs if r['set'] == s and r['mode'] == mode]
            if not sub:
                continue
            for n in sub[0]['runs']:
                rows = [r for r in sub if n in r['runs']]
                err = [r['runs'][n]['row'] - r['truth_row'] for r in rows if r['runs'][n]['row'] is not None]
                out[f'{s}|{mode}|{n}'] = {
                    'row_acc': wil(sum(r['runs'][n]['row'] == r['truth_row'] for r in rows), len(rows)),
                    'col_acc': wil(sum(r['runs'][n]['col'] == r['truth_col'] for r in rows), len(rows)),
                    'row_err': boot_mean(err), 'invalid': int(sum(r['runs'][n]['row'] is None for r in rows)),
                    'length_cut': int(sum(r['runs'][n]['finish_reason'] == 'length' for r in rows))}
        # Paired direct vs think within each intervention.
        d = {r['item']: r for r in recs if r['set'] == s and r['mode'] == 'direct'}
        t = {r['item']: r for r in recs if r['set'] == s and r['mode'] == 'think'}
        for n in ('none', 'nodecay|mid', 'nowrite|mid'):
            items = [i for i in d if i in t and n in d[i]['runs'] and n in t[i]['runs']]
            b = sum(d[i]['runs'][n]['row'] == d[i]['truth_row'] and t[i]['runs'][n]['row'] != t[i]['truth_row'] for i in items)
            c = sum(d[i]['runs'][n]['row'] != d[i]['truth_row'] and t[i]['runs'][n]['row'] == t[i]['truth_row'] for i in items)
            same = sum(d[i]['runs'][n]['row'] == t[i]['runs'][n]['row'] for i in items)
            out[f'{s}|paired|{n}'] = {'direct_only': b, 'think_only': c, 'p': rm.mcnemar_exact(b, c),
                                      'same_answer': wil(same, len(items))}
        # Thinking-on paired with direct on the same items.
        th = {r['item']: r for r in recs if r['set'] == s and r['mode'] == 'thinking_on'}
        for n in ('none', 'nodecay|mid'):
            items = [i for i in th if i in d and n in th[i]['runs']]
            if items:
                out[f'{s}|thinking_on_vs_direct|{n}'] = {
                    'n': len(items), 'thinking_on_correct': sum(th[i]['runs'][n]['row'] == th[i]['truth_row'] for i in items),
                    'direct_correct': sum(d[i]['runs'][n]['row'] == d[i]['truth_row'] for i in items),
                    'same_answer': sum(th[i]['runs'][n]['row'] == d[i]['runs'][n]['row'] for i in items)}
    return out


def think_text_stats(run: Path) -> dict:
    """How think-then-answer texts arrive at the row under no decay: enumeration of row labels and final row."""
    import re
    recs = [r for r in read_jsonl(run / 'implicit' / 'records.jsonl') if r['mode'] == 'think']
    out = {}
    for s in sorted({r['set'] for r in recs}):
        for n in ('none', 'nodecay|mid'):
            rows = [r for r in recs if r['set'] == s and n in r['runs']]
            enum = [bool(re.search(r'[Rr]ow 1\b.*[Rr]ow 2\b.*[Rr]ow 3\b', r['runs'][n]['text'], re.S)) for r in rows]
            acc_e = [r['runs'][n]['row'] == r['truth_row'] for r, e in zip(rows, enum) if e]
            acc_n = [r['runs'][n]['row'] == r['truth_row'] for r, e in zip(rows, enum) if not e]
            out[f'{s}|{n}'] = {'enumerate_rows': wil(sum(enum), len(enum)),
                               'acc_when_enumerating': wil(sum(acc_e), len(acc_e)),
                               'acc_otherwise': wil(sum(acc_n), len(acc_n))}
    return out


def fig_implicit(st: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={'width_ratios': [1, 1, 0.8]})
    for ax, s, title in ((axes[0], 'g8x8', '(a) 8×8 grids (128 images): row accuracy'),
                         (axes[1], 'V64', '(b) 1.5 V64 base images (64): row accuracy')):
        names = ['none', 'nodecay|mid', 'nowrite|mid']
        labels = ['no edit', 'α = 1, layers 9–16', 'β = 0, layers 9–16']
        for j, (mode, col) in enumerate((('direct', BLUE), ('think', ORANGE))):
            vals = [st.get(f'{s}|{mode}|{n}', {}).get('row_acc') for n in names]
            p = np.array([v['p'] if v else np.nan for v in vals])
            ci = np.array([v['ci'] if v else [np.nan, np.nan] for v in vals])
            ax.bar(np.arange(3) + (j - 0.5) * 0.38, p, 0.36, color=col, yerr=np.clip([p - ci[:, 0], ci[:, 1] - p], 0, None), capsize=2,
                   label='direct answer' if mode == 'direct' else 'think-then-answer (thinking off)')
        ax.set_xticks(range(3), labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('row accuracy')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    keys = [('none', 'no edit'), ('nodecay|mid', 'α = 1, L9–16')]
    for j, (mode, col, lab) in enumerate((('direct', BLUE, 'direct answer'), ('thinking_on', VIOLET, 'thinking on'))):
        p = [st.get(f'g8x8|thinking_on_vs_direct|{k}', {}) for k, _ in keys]
        y = [(v.get(f'{"direct" if mode == "direct" else "thinking_on"}_correct', np.nan) / v['n']) if v else np.nan for v in p]
        ax.bar(np.arange(2) + (j - 0.5) * 0.38, y, 0.36, color=col, label=lab)
    n = st.get('g8x8|thinking_on_vs_direct|none', {}).get('n', 0)
    ax.set_xticks(range(2), [l for _, l in keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f'row accuracy (n = {n} images, rows 4–8)')
    ax.set_title('(c) thinking-on subset (8×8)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- output order

def order_stats(run: Path) -> dict:
    recs = read_jsonl(run / 'order' / 'records.jsonl')
    out = {}
    for s in sorted({r['set'] for r in recs}):
        per = {}
        for order in ('direct', 'swapped'):
            sub = [r for r in recs if r['set'] == s and r['order'] == order]
            if not sub:
                continue
            e = {'n': len(sub), 'valid': int(sum(r['row'] is not None for r in sub))}
            for key, tk in (('row', 'truth_row'), ('col', 'truth_col')):
                diffs = [r[key] - r[tk] for r in sub if r[key] is not None]
                e[f'{key}_acc'] = wil(sum(d == 0 for d in diffs), len(sub))
                e[f'{key}_plus1'] = wil(sum(d == 1 for d in diffs), len(sub))
                e[f'{key}_minus1'] = wil(sum(d == -1 for d in diffs), len(sub))
            att = {f'{q}_{m}': [] for q in ('row', 'col') for m in ('on_target_H11', 'in_target_line_H11', 'mass_target_mean')}
            for r in sub:
                if r.get('status') != 'ok':
                    continue
                maps = np.load(run / 'order' / 'maps' / f"{r['item']}_{order}.npy").astype(np.float32)   # (2, 16, 2, Nv)
                H, W = r['grid']
                T = np.array(r['target_tokens'])
                trow, tcol = set((T // W).tolist()), set((T % W).tolist())
                for qi, q in enumerate(('row', 'col')):
                    a = maps[1, 11, qi]
                    pk = int(a.argmax())
                    att[f'{q}_on_target_H11'].append(pk in set(T.tolist()))
                    att[f'{q}_in_target_line_H11'].append((pk // W in trow) if q == 'row' else (pk % W in tcol))
                    att[f'{q}_mass_target_mean'].append(float(maps[1, :, qi][:, T].sum(1).mean()))
            for k, v in att.items():
                e[k] = wil(sum(v), len(v)) if 'mass' not in k else boot_mean(v)
            e['row_first_fraction'] = float(np.mean([r.get('row_first', False) for r in sub if r.get('status') == 'ok']))
            per[order] = e
        # Paired change of the row-query attention peak on target: row emitted first (direct) vs second (swapped).
        dd = {r['item']: r for r in recs if r['set'] == s and r['order'] == 'direct'}
        ss = {r['item']: r for r in recs if r['set'] == s and r['order'] == 'swapped'}
        items = [i for i in dd if i in ss]
        for key, tk in (('row', 'truth_row'), ('col', 'truth_col')):
            b = sum(dd[i][key] == dd[i][tk] and ss[i][key] != ss[i][tk] for i in items)
            c = sum(dd[i][key] != dd[i][tk] and ss[i][key] == ss[i][tk] for i in items)
            per[f'paired_{key}'] = {'direct_only': b, 'swapped_only': c, 'p': rm.mcnemar_exact(b, c),
                                    'same_answer': wil(sum(dd[i][key] == ss[i][key] for i in items), len(items))}
        out[s] = per
    return out


def fig_order(st: dict, path: Path) -> None:
    sets = [s for s in ('g8x8', 'H32', 'H64', 'V32', 'V64') if s in st]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    panels = (('acc', 'answer accuracy'), ('on_target_H11', 'L20 H11 peak on a target token'),
              ('in_target_line_H11', 'L20 H11 peak in the target token row (row query)\n/ token column (column query)'))
    for ax, (key, lab) in zip(axes, panels):
        bars = [('row', 'direct', BLUE, None), ('row', 'swapped', BLUE, '//'), ('col', 'direct', ORANGE, None),
                ('col', 'swapped', ORANGE, '//')]
        for j, (q, o, col, hatch) in enumerate(bars):
            k = f'{q}_{key}' if key != 'acc' else f'{q}_acc'
            vals = [st[s].get(o, {}).get(k) for s in sets]
            p = np.array([v['p'] if v else np.nan for v in vals])
            ci = np.array([v['ci'] if v else [np.nan] * 2 for v in vals])
            ax.bar(np.arange(len(sets)) + (j - 1.5) * 0.2, p, 0.18, color=col, alpha=1 if hatch is None else 0.5,
                   hatch=hatch, yerr=np.clip([p - ci[:, 0], ci[:, 1] - p], 0, None), capsize=2,
                   label=f'{"row" if q == "row" else "column"}, {"row first" if o == "direct" else "column first"}')
        ax.set_xticks(range(len(sets)), sets)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('fraction')
        ax.set_title(lab, fontsize=9)
        if key == 'on_target_H11':
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- synthesis schematic

def fig_synthesis(path: Path) -> None:
    fig = plt.figure(figsize=(15, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.2)
    ax.axis('off')
    boxes = [
        (0.3, 3.6, 2.6, 1.9, 'ViT + merger', 'each token: local content\n+ absolute position\n(edge tokens look like\n"row start" / "after blank")', GREY),
        (3.4, 4.1, 3.6, 1.6, 'Columns: within-token-row\naccumulator (DeltaNet)', 'α drops at row start and over\na blank run (heads in L3–29);\ncount partly restarts at the\nnearest visual boundary', ORANGE),
        (3.4, 1.9, 3.6, 1.6, 'Rows: cross-row chain\n(DeltaNet L9–16)', 'forgetting heads detect a new\nfigure row; integrators write\nand accumulate; ViT row recalibrates;\nblank rows add nothing', BLUE),
        (7.6, 4.1, 3.0, 1.6, 'carried by the target\nand 2–16 tokens after it', 'column estimate is local\n(only tokens after the\nboundary see it)', ORANGE),
        (7.6, 1.9, 3.0, 1.6, 'carried by every token\nof the target cell row', 'row estimate is shared\nby the whole cell row\n(read from target row)', BLUE),
        (11.2, 3.0, 3.5, 1.9, 'Readout (full attention L16 / L20)', 'answer format token attends:\ncolumn → target token\nrow → target ROW tokens\nthen digit = round(estimate)', VIOLET),
    ]
    for x, y, w, h, t, b, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', facecolor='white', edgecolor=col, lw=2))
        ax.text(x + w / 2, y + h - 0.12, t, ha='center', va='top', fontsize=9, weight='bold', color=INK)
        ax.text(x + w / 2, y + h / 2 - 0.25, b, ha='center', va='center', fontsize=8, color=MUTED)
    arrows = [((2.9, 4.6), (3.4, 4.9)), ((2.9, 4.4), (3.4, 2.7)), ((7.0, 4.9), (7.6, 4.9)), ((7.0, 2.7), (7.6, 2.7)),
              ((10.6, 4.9), (11.2, 4.2)), ((10.6, 2.7), (11.2, 3.6))]
    for a, b in arrows:
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.2))
    ax.text(7.5, 0.9, 'Shared behavioural law: both estimates are saturating sums of per-instance increments, so their noise '
            'grows with the number of accumulation steps;\nthe +1 over-estimate is not in the image-token count and is added at '
            'the readout (+1 after ~5–8 columns, ~9–13 rows).', ha='center', fontsize=9, color=INK)
    ax.text(7.5, 0.25, 'Image tokens precede the question, so both accumulators run in prefill before any text '
            '(implicit counting); the output order only changes the readout.', ha='center', fontsize=9, color=INK)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def cmd_analyze(args) -> None:
    run, figs = Path(args.run), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    classes = np.array(json.loads(Path(args.gate_stats).read_text())['classes'])
    results = {}
    if (run / 'gates' / 'records.jsonl').exists():
        cases = gate_cases(run, Path(args.ordinal_run), ('H32', 'H64', 'V32', 'V64'))
        st = boundary_gate_stats(cases)
        sel = json.loads((run / 'boundary_heads.json').read_text())
        results['gates'] = gate_summary(st, sel, classes)
        fig_gates(st, sel, classes, figs / 'boundary_gates.png')
        fig_examples(cases, Path(args.ordinal_run), figs / 'examples.png')
        prs, examples, names = {}, {}, ['4', '8', '12', '16', '20', '24', '28']
        results['probe'] = {}
        for cond in ('H64', 'V64', 'H32', 'V32'):
            X, rows = probe_dataset(cases, cond)
            pr = probe_condition(X, rows)
            prs[cond] = pr
            results['probe'][cond] = {'per_layer': pr['per_layer'],
                                      'target_vs_answer_L16': target_vs_answer(pr, rows, names.index('16')),
                                      'target_vs_answer_L24': target_vs_answer(pr, rows, names.index('24'))}
            # Example: the insL image with the largest k - kg on the target line, layer 16.
            pred = pr['preds'][names.index('16')]
            cand = [c for c in cases if c['condition'] == cond and c['variant'] in ('insL', 'delL')]
            best = max(cand, key=lambda c: c['truth']['count'] - c['truth']['gap'])
            idx = [i for i, r in enumerate(rows) if r['base'] == best['base'] and r['variant'] == best['variant'] and r['line']]
            examples[cond] = {'id': best['id'].split('|')[0], 'slot': [rows[i]['slot'] + 1 for i in idx],
                              'k': [rows[i]['k'] for i in idx], 'kg': [rows[i]['kg'] for i in idx],
                              'pred': [float(pred[i]) for i in idx], 'blanks': best['geo'].interior_blanks()}
        results['probe_examples'] = examples
        fig_probe(prs, None, examples, names, figs / 'running_count_probe.png')
    if (run / 'gapgate' / 'records.jsonl').exists():
        results['gapgate'] = gapgate_stats(run)
        if (run / 'gapgate2' / 'records.jsonl').exists():
            results['gapgate2'] = gapgate_stats(run, 'gapgate2')
        if (run / 'gapswap' / 'records.jsonl').exists():
            results['gapswap'] = gapgate_stats(run, 'gapswap')
        fig_gapgate(results['gapgate'], results.get('gapgate2'), figs / 'boundary_intervention.png', results.get('gapswap'))
    if (run / 'headclass' / 'records.jsonl').exists():
        results['headclass'] = headclass_stats(run)
        fig_headclass(results['headclass'], figs / 'head_classes.png')
    if (run / 'implicit' / 'records.jsonl').exists():
        results['implicit'] = implicit_stats(run)
        results['think_text'] = think_text_stats(run)
        fig_implicit(results['implicit'], figs / 'implicit_counting.png')
    if (run / 'order' / 'records.jsonl').exists():
        results['order'] = order_stats(run)
        fig_order(results['order'], figs / 'output_order.png')
    fig_synthesis(figs / 'synthesis.png')
    (run / 'results.json').write_text(json.dumps(results, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print('wrote', run / 'results.json')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--command', choices=('heads', 'analyze'), required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--ordinal-run', required=True)
    p.add_argument('--gate-stats', required=True)
    p.add_argument('--figures', required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    {'heads': cmd_heads, 'analyze': cmd_analyze}[args.command](args)


if __name__ == '__main__':
    main()
