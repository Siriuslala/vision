"""Summaries and figures of experiments 2.5b and 2.5c (see patching_extension.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis_tools.representation.patching_extension import (
    FAIL_CONDITIONS, FULL_ATTENTION_LAYERS, GEN_CONDITIONS, MIN_SCORE_GAP, PROBE_GROUPS, SITES, YESNO_CONDITIONS,
    bootstrap_summary, paired_probe, read_all, wilson,
)

# Fixed categorical order (validated reference palette, light mode).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
ROW_ORDER = ['target_clean', 'target_corrupt', 'targets', 'random', 'nontarget', 'all_visual', 'question', 'qmark',
             'q_end', 'think_end', 'last_prompt', 'prompt_text', 'answer_prefix', 'query']
ROW_LABEL = {'target_clean': 'target (clean cell)', 'target_corrupt': 'target (corrupt cell)',
             'targets': 'both target cells', 'random': 'random visual', 'nontarget': 'non-target visual',
             'all_visual': 'all visual', 'question': 'question text', 'qmark': '"?" token', 'q_end': 'question end',
             'think_end': '</think>', 'last_prompt': 'last prompt token', 'prompt_text': 'all post-image prompt',
             'answer_prefix': 'answer prefix', 'query': 'query (answer) token'}
REF_25A = {'target': 'target_clean', 'all_visual': 'all_visual', 'question': 'question', 'template': 'think_end',
           'last': 'last_prompt', 'text_all': 'prompt_text'}


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def qualified(base: dict, screen: dict) -> str:
    """Analysis subset of a scanned pair.

    'qualified': clean and corrupted inputs both answered correctly (sign of s) and the score gap
    is at least MIN_SCORE_GAP (the 2.5a criterion).  'informative': gap >= MIN_SCORE_GAP regardless
    of correctness.  'weak': smaller or negative gap.
    """
    gap = base['s_clean'] - base['s_corrupt']
    if gap < MIN_SCORE_GAP:
        return 'weak'
    if screen['kind'] == 'yesno':
        return 'qualified' if base['s_clean'] > 0 > base['s_corrupt'] else 'informative'
    if screen['condition'] == 't5_fail':
        return 'informative'
    return 'qualified' if base['s_clean'] > 0 > base['s_corrupt'] else 'informative'


def summarize(out: Path, n_boot: int, seed: int) -> dict:
    screen = {r['pair_id']: r for r in read_all(out, 'screen')}
    bases = {b['pair_id']: b for b in read_all(out, 'baselines')}
    subset = {pid: qualified(b, screen[pid]) for pid, b in bases.items()}
    by_key: dict[tuple, dict] = {}
    sanity = {'identity_max_abs': 0.0, 'all_tokens_max_dev': 0.0}
    seen = set()
    for rec in read_all(out, 'records'):
        ident = (rec['pair_id'], rec['direction'], rec['mode'], rec['site'], rec['group'])
        if ident in seen:          # a pair scanned twice by parallel processes counts once
            continue
        seen.add(ident)
        if rec['mode'] == 'sanity_identity':
            receiver = rec['s_corrupt'] if rec['direction'] == 'restore' else rec['s_clean']
            sanity['identity_max_abs'] = max(sanity['identity_max_abs'], abs(rec['s_patched'] - receiver))
            continue
        if rec['mode'] == 'sanity_all':
            sanity['all_tokens_max_dev'] = max(sanity['all_tokens_max_dev'], abs(rec['ratio'] - 1))
            continue
        key = (rec['condition'], rec['direction'], rec['mode'], rec['site'], rec['group'])
        d = by_key.setdefault(key, {})
        d[rec['pair_id']] = rec
    summary = []
    for (cond, direction, mode, site, group), d in by_key.items():
        for name, keep in (('qualified', {'qualified'}), ('informative', {'qualified', 'informative'})):
            rows = [r for pid, r in d.items() if subset[pid] in keep]
            if len(rows) < 3:
                continue
            gap = np.array([r['s_clean'] - r['s_corrupt'] for r in rows])
            if direction == 'restore':
                num = np.array([r['s_patched'] - r['s_corrupt'] for r in rows])
            else:
                num = np.array([r['s_clean'] - r['s_patched'] for r in rows])
            s = bootstrap_summary(num, gap, n_boot, seed)
            summary.append({'condition': cond, 'direction': direction, 'mode': mode, 'site': site, 'group': group,
                            'subset': name, **s})
    # Absolute shifts over all scanned pairs (logit units), used for weak-signal failure conditions.
    shift = []
    for (cond, direction, mode, site, group), d in by_key.items():
        if mode != 'std':
            continue
        rows = list(d.values())
        num = [r['s_patched'] - r['s_corrupt'] if direction == 'restore' else r['s_clean'] - r['s_patched'] for r in rows]
        shift.append({'condition': cond, 'direction': direction, 'site': site, 'group': group, 'n': len(rows),
                      'mean_shift': float(np.mean(num)),
                      'mean_gap': float(np.mean([r['s_clean'] - r['s_corrupt'] for r in rows]))})
    return {'summary': summary, 'shift': shift, 'sanity': sanity, 'subset': subset}


def screen_table(out: Path) -> dict:
    # Parallel processes may screen the same condition; screening is deterministic, keep one row per pair.
    screen = list({r['pair_id']: r for r in read_all(out, 'screen')}.values())
    bases = {b['pair_id']: b for b in read_all(out, 'baselines')}
    table = {}
    for r in screen:
        t = table.setdefault(r['condition'], {'n': 0, 'status': {}})
        t['n'] += 1
        t['status'][r['status']] = t['status'].get(r['status'], 0) + 1
    for cond, t in table.items():
        rows = [bases[r['pair_id']] for r in screen if r['condition'] == cond and r['pair_id'] in bases]
        if not rows:
            continue
        sc, sk = np.array([b['s_clean'] for b in rows]), np.array([b['s_corrupt'] for b in rows])
        gap = sc - sk
        n = len(rows)
        t.update(scanned=n, gap_median=float(np.median(gap)), gap_mean=float(gap.mean()),
                 gap_ge1=int((gap >= MIN_SCORE_GAP).sum()), gap_ge1_ci=wilson(int((gap >= MIN_SCORE_GAP).sum()), n),
                 gap_pos=int((gap > 0).sum()))
        if cond in YESNO_CONDITIONS or cond == 't3_fail':
            t.update(clean_yes=int((sc > 0).sum()), corrupt_no=int((sk < 0).sum()),
                     clean_yes_ci=wilson(int((sc > 0).sum()), n), corrupt_no_ci=wilson(int((sk < 0).sum()), n),
                     both_correct=int(((sc > 0) & (sk < 0)).sum()), auroc=auroc(sc, sk))
    return table


def probe_summary(out: Path, conditions: list[str], folds: int, seed: int, reference: str) -> dict:
    """Cross-validated paired probe per condition, site and group; plus transfer from `reference`."""
    screen = {r['pair_id']: r for r in read_all(out, 'screen')}
    bases = {b['pair_id']: b for b in read_all(out, 'baselines')}
    deltas = {}
    for cond in conditions:
        ids = [pid for pid, r in screen.items() if r['condition'] == cond and pid in bases
               and (out / 'probe_deltas' / f'{pid}.npy').exists()]
        if ids:
            deltas[cond] = (ids, np.stack([np.load(out / 'probe_deltas' / f'{pid}.npy').astype(np.float32) for pid in ids]))
    result = {}
    ref = deltas.get(reference)
    for cond, (ids, arr) in deltas.items():
        res = {'n': len(ids), 'within': {}, 'transfer': {}, 'pair_ids': ids}
        for gi, g in enumerate(PROBE_GROUPS):
            acc, dp, tr = [], [], []
            for si in range(len(SITES)):
                x = arr[:, si, gi]
                if not np.any(x):
                    acc.append(float('nan')); dp.append(float('nan')); tr.append(float('nan'))
                    continue
                p = paired_probe(x, min(8, len(ids)), seed)
                acc.append(p['acc']); dp.append(p['dprime'])
                if ref is not None and cond != reference:
                    w = ref[1][:, si, gi].mean(0)
                    tr.append(float(((x @ w) > 0).mean()) if np.linalg.norm(w) > 0 else float('nan'))
                else:
                    tr.append(p['acc'])
            res['within'][g] = {'acc': acc, 'dprime': dp}
            res['transfer'][g] = tr
            # Mean difference norm, to show whether the difference exists before any probe.
            res.setdefault('delta_norm', {})[g] = np.linalg.norm(arr[:, :, gi], axis=-1).mean(0).tolist()
        result[cond] = res
    return result


# ---------------------------------------------------------------------------- figures

def _lookup(summary: list[dict], **kw) -> dict:
    return {(r['site'], r['group']): r for r in summary if all(r[k] == v for k, v in kw.items())}


def _heat(ax, mat, rows, title, cmap, norm, xlabel=True):
    """Heatmap over decoder sites; unmeasured sites are dropped and measured ones span to the midpoints."""
    cols = [k for k in range(mat.shape[1]) if not np.all(np.isnan(mat[:, k]))]
    if not cols:
        cols = list(range(mat.shape[1]))
    x = np.asarray(cols, float)
    mids = (x[1:] + x[:-1]) / 2
    edges = np.concatenate([[x[0] - 0.5], mids, [x[-1] + 0.5]])
    im = ax.pcolormesh(edges, np.arange(len(rows) + 1) - 0.5, np.ma.masked_invalid(mat[:, cols]), cmap=cmap,
                       norm=norm, shading='flat')
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([ROW_LABEL.get(r, r) for r in rows], fontsize=8)
    ticks = list(range(0, 33, 4))
    ax.set_xticks(ticks)
    ax.set_xticklabels(['emb' if t == 0 else str(t) for t in ticks], fontsize=8)
    if len(cols) < mat.shape[1]:
        ax.plot(x, np.full(len(x), len(rows) - 0.5), '|', color='#555555', ms=5, clip_on=False)
    for t in FULL_ATTENTION_LAYERS:
        ax.axvline(t, color='#888888', lw=0.4, alpha=0.6)
    ax.set_title(title, fontsize=9)
    if xlabel:
        ax.set_xlabel('decoder site (emb = merger output; k = output of layer k; grey lines = full-attention layers)', fontsize=8)
    return im


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
    return cmap, TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.0)


def matrix(lookup: dict, rows: list[str], stat: str) -> np.ndarray:
    m = np.full((len(rows), len(SITES)), np.nan)
    for i, g in enumerate(rows):
        for s in SITES:
            r = lookup.get((s, g))
            if r is not None:
                m[i, s] = r[stat]
    return m


def prefill_reference(ref_summary: Path, condition: str = 'red_rowcol') -> tuple[np.ndarray, list[str]]:
    """2.5a prefill heatmap (restore median) mapped onto the row names used here."""
    data = json.load(open(ref_summary))
    rows = ['target_clean', 'all_visual', 'question', 'think_end', 'last_prompt', 'prompt_text']
    m = np.full((len(rows), len(SITES)), np.nan)
    for r in data:
        if r['condition'] != condition or r['direction'] != 'restore':
            continue
        site = 0 if r['site'] == 'merger' else (int(r['site'][3:]) if r['site'].startswith('dec') else None)
        if site is None or r['group'] not in REF_25A:
            continue
        name = REF_25A[r['group']]
        if name in rows:
            m[rows.index(name), site] = r['median']
    labels = [ROW_LABEL[g] if g != 'think_end' else 'chat template (2.5a)' for g in rows]
    return m, rows, labels


def fig_generation_heatmaps(summary, out_png: Path, conditions: list[str], ref_summary: Path | None,
                            prefill_cond: str | None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap, norm = _cmap()
    panels = []
    if ref_summary is not None and ref_summary.exists():
        panels.append(('ref', '2.5a prefill: red_rowcol, Yes/No first token'))
    if prefill_cond:
        panels.append((prefill_cond, f'prefill: {prefill_cond} (revised T3, Yes/No)'))
    panels += [(c, f'generation: {c} (first differing digit)') for c in conditions]
    heights = [6 if c == 'ref' else (8 if c == prefill_cond else 11) for c, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 1.0 + 0.28 * sum(heights) + 0.6 * len(panels)),
                             squeeze=False, gridspec_kw={'height_ratios': heights, 'hspace': 0.45})
    for k, (ax, (cond, title)) in enumerate(zip(axes[:, 0], panels)):
        if cond == 'ref':
            m, rows, labels = prefill_reference(ref_summary)
            im = _heat(ax, m, rows, title, cmap, norm, xlabel=k == len(panels) - 1)
            ax.set_yticklabels(labels, fontsize=8)
            continue
        look = _lookup(summary, condition=cond, direction='restore', mode='std', subset='qualified')
        if not look:
            look = _lookup(summary, condition=cond, direction='restore', mode='std', subset='informative')
        rows = [g for g in ROW_ORDER if any((s, g) in look for s in SITES)]
        n = next(iter(look.values()))['n'] if look else 0
        im = _heat(ax, matrix(look, rows, 'median'), rows, f'{title}, n = {n}', cmap, norm,
                   xlabel=k == len(panels) - 1)
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.6, label='recovery ratio r (pair median)')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_paths(summary, out_png: Path, conditions: list[str], group: str = 'all_visual'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = [('std', 'std: patch at site, downstream free'), ('direct', 'direct: answer tokens read patched visual'),
             ('via', 'via: prompt relay only'), ('direct1', 'direct1: read only in layer k+1')]
    ncol = 2
    nrow = int(np.ceil(len(conditions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.0 * nrow), squeeze=False, sharex=True, sharey=True)
    for ax, cond in zip(axes.flat, conditions):
        n = 0
        for (mode, label), color in zip(modes, SERIES):
            look = _lookup(summary, condition=cond, direction='restore', mode=mode, group=group, subset='qualified')
            if not look:
                continue
            xs = sorted(s for s, _ in look)
            med = [look[(s, group)]['median'] for s in xs]
            lo = [look[(s, group)]['median_lo'] for s in xs]
            hi = [look[(s, group)]['median_hi'] for s in xs]
            n = look[(xs[0], group)]['n']
            ax.plot(xs, med, color=color, lw=2, label=label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
        for t in FULL_ATTENTION_LAYERS:
            ax.axvline(t, color='#bbbbbb', lw=0.6, zorder=0)
        ax.axhline(0, color='#888888', lw=0.6)
        ax.set_title(f'{cond} ({ROW_LABEL[group]}), n = {n}', fontsize=9)
        ax.set_ylim(-0.2, 1.15)
        ax.grid(axis='y', color='#eeeeee')
    for ax in axes[-1]:
        ax.set_xlabel('decoder site k (0 = merger output)')
    for ax in axes[:, 0]:
        ax.set_ylabel('recovery ratio (median)')
    for ax in list(axes.flat)[len(conditions):]:
        ax.axis('off')
    axes[0, 0].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_failure_heatmaps(summary, shift, out_png: Path, conditions: list[str]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    rows = ['target_clean', 'nontarget', 'all_visual', 'question', 'qmark', 'q_end', 'think_end', 'last_prompt',
            'prompt_text']
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
    fig, axes = plt.subplots(len(conditions), 1, figsize=(10, 3.0 * len(conditions)), squeeze=False)
    for k, (ax, cond) in enumerate(zip(axes[:, 0], conditions)):
        look = {(r['site'], r['group']): r for r in shift if r['condition'] == cond and r['direction'] == 'restore'}
        if not look:
            ax.axis('off')
            continue
        m = matrix(look, rows, 'mean_shift')
        gap = next(iter(look.values()))['mean_gap']
        n = next(iter(look.values()))['n']
        vmax = max(abs(gap), 0.5)
        im = _heat(ax, m, rows, f'{cond}: mean logit shift s_patched - s_corrupt (all pairs, n = {n}; '
              f'mean total gap {gap:.2f})', cmap, TwoSlopeNorm(vmin=-vmax / 2, vcenter=0, vmax=vmax),
                   xlabel=k == len(conditions) - 1)
        fig.colorbar(im, ax=ax, shrink=0.8, label='logit')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_probe(probe: dict, table: dict, out_png: Path, conditions: list[str], key: str = 'within'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ['target_clean', 'qmark', 'think_end', 'last_prompt']
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 3.6), sharey=True)
    for ax, g in zip(axes, groups):
        for cond, color in zip(conditions, SERIES):
            if cond not in probe:
                continue
            y = probe[cond]['within'][g]['acc'] if key == 'within' else probe[cond]['transfer'][g]
            au = table.get(cond, {}).get('auroc')
            label = f"{cond} (n={probe[cond]['n']}" + (f', output AUROC {au:.2f})' if au is not None else ')')
            ax.plot(SITES, y, color=color, lw=2, label=label)
        for t in FULL_ATTENTION_LAYERS:
            ax.axvline(t, color='#bbbbbb', lw=0.6, zorder=0)
        ax.axhline(0.5, color='#888888', lw=0.8, ls='--')
        ax.set_title(ROW_LABEL[g], fontsize=9)
        ax.set_xlabel('decoder site k')
        ax.set_ylim(0.3, 1.02)
        ax.grid(axis='y', color='#eeeeee')
    axes[0].set_ylabel('held-out paired accuracy' + (' (direction from t3_gap16_g06)' if key != 'within' else ''))
    axes[-1].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_examples(out: Path, out_png: Path, pair_ids: list[str], args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from analysis_tools.representation.patching_extension import cell_box, pair_from_row, render_pair
    screen = {r['pair_id']: r for r in read_all(out, 'screen')}
    bases = {b['pair_id']: b for b in read_all(out, 'baselines')}
    fig, axes = plt.subplots(2, len(pair_ids), figsize=(3.0 * len(pair_ids), 6.4), squeeze=False)
    for j, pid in enumerate(pair_ids):
        row = screen[pid]
        pair = pair_from_row(row, args)
        ci, meta, ki, _ = render_pair(pair)
        for i, (img, name, text) in enumerate(((ci, 'clean', row.get('clean_text', '')),
                                              (ki, 'corrupt', row.get('corrupt_text', '')))):
            ax = axes[i, j]
            ax.imshow(img)
            for rc, color in ((pair.clean_target, '#eb6834'), (pair.corrupt_target, '#2a78d6')):
                if rc is not None:
                    x0, y0, x1, y1 = cell_box(meta, rc)
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.5, ls='--'))
            b = bases.get(pid, {})
            s = b.get('s_clean' if i == 0 else 's_corrupt')
            answer = ' '.join(text.replace('```json', '').replace('```', '').split())[:34]
            ax.set_title(f'{pid}\n{name}: s = {s:.2f}' + (f'\n{answer}' if answer else ''), fontsize=7)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_detail(out: Path, out_png: Path) -> dict:
    """Median recovery of single answer-token patches: rows = answer offset, columns = site."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read_all(out, 'detail')
    if not rows:
        return {}
    conds = [c for c in GEN_CONDITIONS if any(r['condition'] == c for r in rows)]
    sites = sorted({r['site'] for r in rows})
    cmap, norm = _cmap()
    fig, axes = plt.subplots(1, len(conds), figsize=(3.3 * len(conds), 4.2), squeeze=False)
    summary = {}
    for ax, cond in zip(axes[0], conds):
        sub = [r for r in rows if r['condition'] == cond]
        offsets = sorted({r['offset'] for r in sub})[-10:]
        m = np.full((len(offsets), len(sites)), np.nan)
        labels = []
        for i, o in enumerate(offsets):
            toks = [r['token'] for r in sub if r['offset'] == o]
            labels.append(f'{o}: {max(set(toks), key=toks.count)!r}')
            for j, site in enumerate(sites):
                v = [r['ratio'] for r in sub if r['offset'] == o and r['site'] == site]
                if v:
                    m[i, j] = float(np.median(v))
        summary[cond] = {'offsets': offsets, 'sites': sites, 'median': m.tolist(), 'labels': labels,
                         'n': len({r['pair_id'] for r in sub})}
        im = ax.imshow(m, aspect='auto', cmap=cmap, norm=norm)
        ax.set_yticks(range(len(offsets)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels(sites, fontsize=7)
        ax.set_xlabel('decoder site k', fontsize=8)
        ax.set_title(f"{cond}, n = {summary[cond]['n']}", fontsize=8)
    axes[0, 0].set_ylabel('answer token (offset from query, most common text)', fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label='recovery ratio (median)')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return summary


def analyze(out: Path, args) -> None:
    res = summarize(out, args.n_boot, args.seed)
    table = screen_table(out)
    yesno = [c for c in list(YESNO_CONDITIONS) + ['t3_fail'] if c in table]
    probe = probe_summary(out, yesno, 8, args.seed, 't3_gap16_g06')
    (out / 'summary.json').write_text(json.dumps(res['summary']))
    (out / 'shift.json').write_text(json.dumps(res['shift']))
    (out / 'screen_table.json').write_text(json.dumps(table, indent=1))
    (out / 'sanity.json').write_text(json.dumps(res['sanity'], indent=1))
    (out / 'probe.json').write_text(json.dumps({c: {k: v for k, v in p.items() if k != 'pair_ids'}
                                                for c, p in probe.items()}))
    figs = out / 'figures'
    figs.mkdir(exist_ok=True)
    ref = Path(args.output).parent / '2026-10-13_activation_patching' / 'summary_bf16.json'
    gen = [c for c in GEN_CONDITIONS if c in table]
    if gen:
        fig_generation_heatmaps(res['summary'], figs / 'heatmap_generation_t6.png',
                                [c for c in gen if c.startswith('t6')], ref, 't3_red_g06' if 't3_red_g06' in table else None)
        fig_generation_heatmaps(res['summary'], figs / 'heatmap_generation_t5.png',
                                [c for c in gen if c.startswith('t5')], None, 't3_gap16_g06' if 't3_gap16_g06' in table else None)
        fig_paths(res['summary'], figs / 'paths_all_visual.png', gen, 'all_visual')
        fig_paths(res['summary'], figs / 'paths_targets.png', gen, 'targets')
    detail = fig_detail(out, figs / 'answer_tokens.png')
    (out / 'detail_summary.json').write_text(json.dumps(detail))
    if yesno:
        fig_failure_heatmaps(res['summary'], res['shift'], figs / 'failure_shift.png', yesno)
        fig_probe(probe, table, figs / 'probe_within.png', yesno, 'within')
        fig_probe(probe, table, figs / 'probe_transfer.png', [c for c in yesno if c != 't3_gap16_g06'], 'transfer')
    if 't5_fail' in table:
        fig_failure_heatmaps(res['summary'], res['shift'], figs / 'failure_shift_t5.png', ['t5_fail'])
    print(json.dumps({'sanity': res['sanity'], 'screen': {c: t['status'] for c, t in table.items()}}, indent=1))
