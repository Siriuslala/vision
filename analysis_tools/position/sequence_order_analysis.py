"""Statistics and figures of experiment 3.8 (positional role of sequence order).

Reads the stage records under the run directory, writes `results.json` there and figures to --figures.
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

from analysis_tools.position import sequence_order as so

BLUE, ORANGE, AQUA, YELLOW, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'figure.dpi': 150})


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    z, p = 1.96, k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [mid - half, mid + half]


def boot(values, n: int = 2000, seed: int = 0) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return {'mean': math.nan, 'ci': [math.nan, math.nan], 'n': 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return {'mean': float(v.mean()), 'ci': [float(np.quantile(means, .025)), float(np.quantile(means, .975))], 'n': int(len(v))}


def value(a: dict, key: str):
    """Expected row/column when the digit distribution exists, else the parsed answer, else None."""
    if f'e_{key}' in a:
        return a[f'e_{key}']
    return a.get(key)


def follow(rec: dict, cond: dict, base: dict) -> dict:
    """Answer shift along the condition axis divided by the target's slot displacement (cells)."""
    ax = 0 if cond.get('axis') == 'v' else 1
    key = ('row', 'col')[ax]
    d = cond['slot_cell'][ax] - base['slot_cell'][ax]
    a, b = value(cond, key), value(base, key)
    out = {'d': d, 'valid': a is not None and b is not None}
    if out['valid'] and abs(d) > 1e-6:
        out['f'] = (a - b) / d
        other = ('col', 'row')[ax]
        ao, bo = value(cond, other), value(base, other)
        out['off_axis'] = None if ao is None or bo is None else ao - bo
    return out


# ----------------------------------------------------------------------------- behavior

def behavior_stats(recs: list[dict]) -> dict:
    res = {}
    for shape in sorted({f"{r['rows']}x{r['cols']}" for r in recs}):
        sub = [r for r in recs if f"{r['rows']}x{r['cols']}" == shape]
        base_ok = [r for r in sub if (c := r['conditions'][0]).get('row') == r['target'][0] and c.get('col') == r['target'][1]]
        out = {'n_images': len(sub), 'baseline_correct': len(base_ok), 'baseline_ci': wilson(len(base_ok), len(sub))}
        names = sorted({c['name'] for r in sub for c in r['conditions']})
        for name in names:
            fs, offs, dropped, total, changed, exact_slot = [], [], 0, 0, 0, 0
            row_e, col_e = [], []
            for r in sub:
                base = r['conditions'][0]
                cond = next((c for c in r['conditions'] if c['name'] == name), None)
                if cond is None:
                    continue
                total += 1
                if not cond.get('valid'):
                    dropped += 1
                    continue
                changed += (cond.get('row'), cond.get('col')) != (base.get('row'), base.get('col'))
                if cond.get('axis') in ('v', 'h'):
                    fo = follow(r, cond, base)
                    if cond['family'] in ('roll_rows', 'roll_within') and abs(abs(fo['d']) - 1) > 1e-6:
                        continue                 # the target wrapped around the image edge
                    if 'f' in fo:
                        fs.append(fo['f'])
                        offs.append(fo['off_axis'])
                if value(cond, 'row') is not None:
                    row_e.append(value(cond, 'row') - r['target'][0])
                    col_e.append(value(cond, 'col') - r['target'][1])
            out[name] = {'n': total, 'invalid': dropped, 'answer_changed': changed, 'changed_ci': wilson(changed, total - dropped),
                         'follow': boot(fs), 'off_axis': boot([o for o in offs if o is not None]),
                         'row_err': boot(row_e), 'col_err': boot(col_e),
                         'row_exact': sum(abs(e) < 0.5 for e in row_e), 'col_exact': sum(abs(e) < 0.5 for e in col_e)}
        res[shape] = out
    return res


def global_correlation(recs: list[dict], name: str) -> dict:
    """Pearson correlation of the answer row / column with the token coordinates and the slot coordinates."""
    rows = []
    for r in recs:
        cond = next((c for c in r['conditions'] if c['name'] == name), None)
        if cond is None or not cond.get('valid') or value(cond, 'row') is None:
            continue
        rows.append([value(cond, 'row'), value(cond, 'col'), r['target'][0], r['target'][1], *cond['slot_cell']])
    a = np.asarray(rows, float)
    out = {'n': len(a), 'row_exact': int(np.sum(np.abs(a[:, 0] - a[:, 2]) < 0.5)) if len(a) else 0,
           'col_exact': int(np.sum(np.abs(a[:, 1] - a[:, 3]) < 0.5)) if len(a) else 0,
           'out_of_range': int(np.sum((a[:, :2] > 7.5).any(1))) if len(a) else 0}
    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else math.nan
    if len(a) > 5:
        for i, key in ((0, 'row'), (1, 'col')):
            out[key] = {'token_r': corr(a[:, i], a[:, 2]), 'token_c': corr(a[:, i], a[:, 3]),
                        'slot_R': corr(a[:, i], a[:, 4]), 'slot_C': corr(a[:, i], a[:, 5])}
    return out


def edit_stats(recs: list[dict]) -> dict:
    """Instance-row / column duplication and deletion: answer shift against the counting prediction."""
    out = {}
    names = sorted({c['name'] for r in recs for c in r['conditions'] if c['name'] != 'identity'})
    for nm in names:
        shifts, n, invalid, match = [], 0, 0, 0
        for r in recs:
            base = r['conditions'][0]
            c = next((x for x in r['conditions'] if x['name'] == nm), None)
            if c is None:
                continue
            n += 1
            key = 'row' if c['axis'] == 'v' else 'col'
            if not c.get('valid') or value(c, key) is None:
                invalid += 1
                continue
            d = value(c, key) - value(base, key)
            shifts.append(d / c['count'])
            match += (c.get(key) - base.get(key)) == c['count']
        out[nm] = {'n': n, 'invalid': invalid, 'follow': boot(shifts), 'greedy_matches_count': match,
                   'match_ci': wilson(match, n - invalid)}
    return out


def fig_rows(recs: list[dict], fig_dir: Path) -> None:
    """Mean expected row answer against the true row for vertical slot conditions (8x8 grids)."""
    sq = [r for r in recs if r['rows'] == r['cols'] == 8]
    panels = [('Image roll', [('roll_rows+1', 'bottom cell row read first', BLUE), ('roll_rows-1', 'top cell row read last', ORANGE)]),
              ('Block move (2x2 target only)', [('block_down', 'down', BLUE), ('block_up', 'up', ORANGE)]),
              ('Exchange of two whole cell rows', [('carry_v+1_L16_R16', 'with the row below', BLUE),
                                                    ('carry_v-1_L16_R16', 'with the row above', ORANGE)])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (title, conds) in zip(axes, panels):
        ax.plot([0, 7], [0, 7], color=MUTED, lw=0.8, ls='--')
        for nm, lab, col in conds:
            xs, ms = [], []
            for row in range(8):
                v = [value(c, 'row') for r in sq if r['target'][0] == row
                     for c in r['conditions'] if c['name'] == nm and c.get('valid') and value(c, 'row') is not None]
                if v:
                    xs.append(row); ms.append(float(np.mean(v)))
            ax.plot(xs, ms, '-o', color=col, lw=2, ms=5, label=lab)
        ax.set_title(title, fontsize=10); ax.set_xlabel('True row of the target (0-based)')
        ax.legend(fontsize=8, frameon=False, loc='upper left')
    axes[0].set_ylabel('Mean expected row answer')
    fig.tight_layout(); fig.savefig(fig_dir / 'rows.png', bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- restore

def restore_stats(recs: list[dict], identity: dict) -> dict:
    """identity: case id -> answer under the standard order (from the behavior stage)."""
    out = {}
    for move in ('down', 'up', 'right', 'left'):
        groups = sorted({c['group'] for r in recs for c in r['conditions'] if c['move'] == move})
        res = {}
        for g in groups:
            fs, n, invalid = [], 0, 0
            for r in recs:
                base_list = [c for c in r['conditions'] if c['move'] == move and c['group'] == 'none']
                cond = next((c for c in r['conditions'] if c['move'] == move and c['group'] == g), None)
                if cond is None or not base_list:
                    continue
                n += 1
                ax = 0 if cond['axis'] == 'v' else 1
                key = ('row', 'col')[ax]
                base = identity.get(r['id'])
                if value(cond, key) is None or base is None or value(base, key) is None:
                    invalid += 1
                    continue
                fs.append((value(cond, key) - value(base, key)) / (cond['slot_cell'][ax] - base['slot_cell'][ax]))
            res[g] = {**boot(fs), 'n_total': n, 'invalid': invalid}
        for tpl in ('person', 'ring'):
            fs = []
            for r in recs:
                cond = next((c for c in r['conditions'] if c['move'] == move and c['group'] == 'none'), None)
                base = identity.get(r['id'])
                if r['template'] != tpl or cond is None or base is None:
                    continue
                ax = 0 if cond['axis'] == 'v' else 1
                key = ('row', 'col')[ax]
                if value(cond, key) is not None and value(base, key) is not None:
                    fs.append((value(cond, key) - value(base, key)) / (cond['slot_cell'][ax] - base['slot_cell'][ax]))
            res[f'none_{tpl}'] = boot(fs)
        out[move] = res
    return out


# ----------------------------------------------------------------------------- figures

def rearranged(image: np.ndarray, order: np.ndarray, H: int, W: int, extra: np.ndarray | None = None,
               at: int = 0) -> np.ndarray:
    """Image whose raster slot i shows the 32x32 patch of token order[i] (slot view of the sequence)."""
    t = so.TOKEN
    patches = [image[(j // W) * t:(j // W + 1) * t, (j % W) * t:(j % W + 1) * t] for j in order]
    if extra is not None:
        patches = patches[:at] + list(extra) + patches[at:]
    rows = math.ceil(len(patches) / W)
    out = np.full((rows * t, W * t, 3), 255, np.uint8)
    for i, p in enumerate(patches):
        out[(i // W) * t:(i // W + 1) * t, (i % W) * t:(i % W + 1) * t] = p
    return out


def fig_inputs(fig_dir: Path) -> None:
    image, _ = so.grid_image('person', 8, 8, (3, 2))
    img = np.asarray(image)
    H = W = 16
    T = so.cell_tokens((3, 2), (H, W))
    blank = np.full((so.TOKEN, so.TOKEN, 3), 235, np.uint8)
    panels = [('Standard order', so.order_identity(H, W), None),
              ('Block move (down)', so.order_block_move(H, W, T, so.cell_tokens((4, 2), (H, W))), None),
              ('Image roll (rows +2)', so.order_roll_rows(H, W, 2), None),
              ('Within-row roll (+2)', so.order_roll_within_rows(H, W, 2), None),
              ('Carry L8 R0 (down)', so.order_carry(H, W, (3, 2), 1, 8, 0), None),
              ('Shuffle others', so.order_shuffle_others(H, W, T, 0), None),
              ('Transpose', so.order_transpose(H, W), None),
              ('Insert 2 blank rows', so.order_identity(H, W), 'insert')]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, (title, order, kind) in zip(axes.ravel(), panels):
        if kind == 'insert':
            view = rearranged(img, order, H, W, extra=np.stack([blank] * (2 * W)), at=0)
        else:
            view = rearranged(img, order, H, W)
        ax.imshow(view)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle('Slot view: the patch shown at raster slot i is the token placed in slot i (gray = inserted blank token)',
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(fig_dir / 'inputs.png', bbox_inches='tight')
    plt.close(fig)


def fig_follow(stats: dict, fig_dir: Path) -> None:
    s = stats['8x8']
    items = [('Block move, vertical', ['block_down', 'block_up']), ('Block move, horizontal', ['block_right', 'block_left']),
             ('Image roll, +1 = bottom row first, -1 = top row last', ['roll_rows+1', 'roll_rows-1']),
             ('Within-row roll', ['roll_within+1', 'roll_within-1']),
             ('Carry full rows, vertical', ['carry_v+1_L16_R16', 'carry_v-1_L16_R16']),
             ('Carry 4 trailing, horizontal', ['carry_h+1_4', 'carry_h-1_4'])]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels, means, los, his, cols = [], [], [], [], []
    for label, names in items:
        for nm in names:
            if nm in s and s[nm]['follow']['n']:
                f = s[nm]['follow']
                labels.append(f"{label}: {nm.split('_')[-1] if 'carry' not in nm else ('+1' if '+1' in nm else '-1')}")
                means.append(f['mean']); los.append(f['mean'] - f['ci'][0]); his.append(f['ci'][1] - f['mean'])
                cols.append(BLUE if ('v' in nm.split('_')[1] or nm.startswith(('block_down', 'block_up', 'roll_rows'))) else ORANGE)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, xerr=[los, his], color=cols, height=0.6, error_kw={'ecolor': INK, 'lw': 1})
    for yi, m, h in zip(y, means, his):
        ax.text(max(m, 0) + h + 0.03, yi, f'{m:.2f}', va='center', fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.8); ax.axvline(1, color=MUTED, lw=0.8, ls='--')
    ax.set_xlabel('Follow ratio f (answer shift / slot shift)')
    ax.set_title('8x8 grids: blue = row answer under vertical slot shifts, orange = column under horizontal', fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / 'follow.png', bbox_inches='tight'); plt.close(fig)


def fig_edit(es: dict, fig_dir: Path) -> None:
    items = [('dup_row_before', 'duplicate a figure row just above the target'), ('delete_row_before', 'delete the figure row just above'),
             ('dup_row_start', 'duplicate a figure row above the top row'), ('delete_row0', 'delete the top figure row'),
             ('dup_col_before', 'duplicate the figure cell left of the target'), ('delete_col_before', 'delete the figure cell left of the target'),
             ('dup_col_start', 'duplicate a figure cell before the row start')]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(items))[::-1]
    m = [es[k]['follow']['mean'] for k, _ in items]
    lo = [mm - es[k]['follow']['ci'][0] for mm, (k, _) in zip(m, items)]
    hi = [es[k]['follow']['ci'][1] - mm for mm, (k, _) in zip(m, items)]
    ax.barh(y, m, xerr=[lo, hi], color=[BLUE] * 4 + [ORANGE] * 3, height=0.6, error_kw={'ecolor': INK, 'lw': 1})
    for yi, mm, h in zip(y, m, hi):
        ax.text(max(mm, 0) + h + 0.03, yi, f'{mm:.2f}', va='center', fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in items], fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.8); ax.axvline(1, color=MUTED, lw=0.8, ls='--')
    ax.set_xlabel('Answer shift / shift predicted by counting figures from the image start')
    ax.set_title('Editing figure rows (blue, row answer) and figure cells (orange, column answer), 8x8 grids', fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / 'edit.png', bbox_inches='tight'); plt.close(fig)


def fig_carry(stats: dict, fig_dir: Path) -> None:
    s = stats['8x8']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, dr in zip(axes, ('+1', '-1')):
        for lab, key, col in (('left (preceding) context only', lambda a: f'carry_v{dr}_L{a}_R0', BLUE),
                              ('right (following) context only', lambda a: f'carry_v{dr}_L0_R{a}', ORANGE),
                              ('both sides', lambda a: f'carry_v{dr}_L{a}_R{a}', AQUA)):
            xs, ms, lo, hi = [], [], [], []
            for a in (0, 2, 4, 8, 16):
                nm = f'block_{"down" if dr == "+1" else "up"}' if a == 0 else key(a)
                if nm in s and s[nm]['follow']['n']:
                    f = s[nm]['follow']
                    xs.append(a); ms.append(f['mean']); lo.append(f['ci'][0]); hi.append(f['ci'][1])
            ax.plot(xs, ms, '-o', color=col, lw=2, ms=5, label=lab)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
        ax.set_xticks([0, 2, 4, 8, 16]); ax.set_xticklabels(['0', '2', '4', '8', '16 (whole row)'])
        ax.set_xlabel('Context tokens carried with the target (per side)')
        ax.set_ylabel('Row follow ratio f'); ax.set_ylim(-0.1, 1.15)
        ax.set_title(f'Target moved {"down" if dr == "+1" else "up"} one cell', fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / 'carry.png', bbox_inches='tight'); plt.close(fig)


GROUP_LABELS = [('none', 'no restoration'), ('all', 'all 32 layers'), ('linear_all', 'all DeltaNet (24)'),
                ('full_all', 'all full attention (8)'), ('linear_1_12', 'DeltaNet 1-11'), ('linear_13_24', 'DeltaNet 13-23'),
                ('linear_25_32', 'DeltaNet 25-31'), ('full_1_12', 'full attn 4, 8, 12'), ('full_13_32', 'full attn 16-32'),
                ('layers_1_8', 'layers 1-8'), ('layers_9_16', 'layers 9-16'), ('layers_17_32', 'layers 17-32'),
                ('all_target_slots', 'all layers, target slots only'), ('all_other_slots', 'all layers, other slots only')]


def fig_restore(rs: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, moves, title in ((axes[0], ('down', 'up'), 'Row follow, vertical block moves'),
                             (axes[1], ('right', 'left'), 'Column follow, horizontal block moves')):
        y = np.arange(len(GROUP_LABELS))[::-1]
        for k, (mv, col) in enumerate(zip(moves, (BLUE, ORANGE))):
            m = [rs[mv].get(g, {}).get('mean', np.nan) for g, _ in GROUP_LABELS]
            lo = [m_ - rs[mv].get(g, {}).get('ci', [np.nan] * 2)[0] for m_, (g, _) in zip(m, GROUP_LABELS)]
            hi = [rs[mv].get(g, {}).get('ci', [np.nan] * 2)[1] - m_ for m_, (g, _) in zip(m, GROUP_LABELS)]
            ax.barh(y + (0.2 if k == 0 else -0.2), m, xerr=[lo, hi], height=0.38, color=col, label=mv,
                    error_kw={'ecolor': INK, 'lw': 0.8})
        ax.set_yticks(y); ax.set_yticklabels([l for _, l in GROUP_LABELS], fontsize=8)
        ax.axvline(0, color=MUTED, lw=0.8); ax.axvline(1, color=MUTED, lw=0.8, ls='--')
        ax.set_title(title, fontsize=10); ax.set_xlabel('Follow ratio f'); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / 'restore_groups.png', bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for mv, col, lab in (('down', BLUE, 'row follow, block down'), ('right', ORANGE, 'column follow, block right')):
        base = rs[mv]['none_person']['mean']
        xs = list(range(1, 33))
        m = [rs[mv].get(f'single{l}', {}).get('mean', np.nan) for l in xs]
        ax.plot(xs, m, '-o', color=col, lw=2, ms=4, label=lab)
        ax.axhline(base, color=col, lw=0.8, ls=':')
    for l in so.FULL_LAYERS:
        ax.axvspan(l - 0.4, l + 0.4, color=GRID, zorder=0)
    ax.set_xlabel('Single restored layer (shaded = full attention)'); ax.set_ylabel('Follow ratio f')
    ax.set_xticks(range(1, 33, 2)); ax.legend(fontsize=8, frameon=False)
    ax.set_title('Restoring one layer at a time (person template; dotted = no restoration)', fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / 'restore_single.png', bbox_inches='tight'); plt.close(fig)


def fig_probe(pr: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, name, title in ((axes[0], 'blank', 'Rolled white canvas'), (axes[1], 'grid', 'Rolled 8x8 grids')):
        if name not in pr:
            continue
        ls = sorted(int(k) for k in pr[name])
        for key, col, lab, st in (('slot_row', BLUE, 'slot row R', '-'), ('token_row', BLUE, 'token row r', '--'),
                                  ('slot_col', ORANGE, 'slot column C', '-'), ('token_col', ORANGE, 'token column c', '--')):
            ax.plot(ls, [max(pr[name][str(l)][key], -0.2) for l in ls], st, color=col, lw=2, label=lab)
        ax.set_ylim(-0.22, 1.02); ax.set_xlabel('Layer (0 = decoder input)'); ax.set_ylabel('Held-out R²')
        ax.set_title(title, fontsize=10); ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    for variant, st in (('identical_mrope', '-'), ('identical_flat', '--')):
        if variant not in pr:
            continue
        ls = sorted(int(k) for k in pr[variant])
        for key, col in (('log_index', VIOLET), ('row', BLUE), ('col', ORANGE)):
            ax.plot(ls, [max(pr[variant][str(l)][key], -0.2) for l in ls], st, color=col, lw=2,
                    label=f"{key} ({'MRoPE kept' if variant.endswith('mrope') else 'MRoPE flat'})")
    ax.set_ylim(-0.22, 1.02); ax.set_xlabel('Layer'); ax.set_title('256 identical visual tokens', fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / 'probe.png', bbox_inches='tight'); plt.close(fig)


def fail20_stats(recs: list[dict]) -> dict:
    out = {}
    names = []
    for r in recs:
        for c in r['conditions']:
            if c['name'] not in names:
                names.append(c['name'])
    for nm in names:
        hist, n, invalid, row_ok = {}, 0, 0, 0
        for r in recs:
            c = next((x for x in r['conditions'] if x['name'] == nm), None)
            if c is None:
                continue
            n += 1
            sc = c['score']
            if not sc.get('valid'):
                invalid += 1
                continue
            # Column error relative to where the slot puts the target (slot cell column), and to the token column.
            dcol = sc['d_col']
            key = str(dcol) if abs(dcol) <= 2 else 'other'
            hist[key] = hist.get(key, 0) + 1
            row_ok += sc['d_row'] == 0
        out[nm] = {'n': n, 'invalid': invalid, 'col_error_hist': hist, 'row_correct': row_ok,
                   'col_plus1': hist.get('1', 0), 'col_plus1_ci': wilson(hist.get('1', 0), n - invalid),
                   'col_correct': hist.get('0', 0), 'col_correct_ci': wilson(hist.get('0', 0), n - invalid)}
    return out


def fig_fail20(fs: dict, fig_dir: Path) -> None:
    names = [n for n in fs]
    cats = [('-2', '-2'), ('-1', '-1'), ('0', 'correct'), ('1', '+1'), ('2', '+2'), ('other', 'other / invalid')]
    cols = [VIOLET, AQUA, BLUE, ORANGE, YELLOW, MUTED]
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for (k, lab), col in zip(cats, cols):
        v = np.array([fs[n]['col_error_hist'].get(k, 0) + (fs[n]['invalid'] if k == 'other' else 0) for n in names], float)
        tot = np.array([fs[n]['n'] for n in names], float)
        ax.barh(y, v / tot, left=left, color=col, height=0.6, label=lab, edgecolor='white', linewidth=1)
        left += v / tot
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.set_xlabel('Fraction of 20x20 cases')
    ax.set_title('Column answer minus true (token) column, 20x20 grids', fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout(); fig.savefig(fig_dir / 'fail20.png', bbox_inches='tight'); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--figures', required=True)
    args = p.parse_args()
    run, fig_dir = Path(args.run), Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    beh = read_jsonl(run / 'behavior' / 'records.jsonl')
    if beh:
        results['behavior'] = behavior_stats(beh)
        sq = [r for r in beh if r['rows'] == r['cols'] == 8]
        results['global'] = {nm: global_correlation(sq, nm) for nm in ('identity', 'transpose', 'reverse_rows', 'mirror_rows', 'shuffle_others')}
        fig_rows(beh, fig_dir)
    edits = read_jsonl(run / 'behavior' / 'records_edit.jsonl')
    if edits:
        results['edit'] = edit_stats(edits)
        fig_edit(results['edit'], fig_dir)
        fig_follow(results['behavior'], fig_dir)
        fig_carry(results['behavior'], fig_dir)
    rest = read_jsonl(run / 'restore' / 'records.jsonl')
    if rest:
        results['restore'] = restore_stats(rest, {r['id']: r['conditions'][0] for r in beh})
        fig_restore(results['restore'], fig_dir)
    if (run / 'probe' / 'results.json').exists():
        results['probe'] = json.loads((run / 'probe' / 'results.json').read_text())
        fig_probe(results['probe'], fig_dir)
    f20 = read_jsonl(run / 'fail20' / 'records.jsonl')
    if f20:
        results['fail20'] = fail20_stats(f20)
        fig_fail20(results['fail20'], fig_dir)
    fig_inputs(fig_dir)
    (run / 'results.json').write_text(json.dumps(results, indent=1))
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in results.items()})[:500])


if __name__ == '__main__':
    main()
