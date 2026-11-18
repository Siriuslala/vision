"""Experiment 7.3 CPU analysis: localisation profiles, the pre-registered block choice, training-condition tables
and figures.  Results doc: docs/analysis/application/results_2026-11-05_located_vit_finetune.md.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analysis_tools.application import interp_opt as io_
from analysis_tools.application import located_finetune as lf

WINDOW = 3
ACHIEVABLE_MIN = 5.0          # a new unit must be able to reach the exclusive-detector threshold
ELIGIBLE_FRACTION = 0.75
ABSENT_MAX_DETECTORS = 2      # pairs counted as "absent direction lacks detectors" (base model, block 9)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def hit(r: dict, target) -> bool:
    return r['arg_row'] == target[0] + 1 and r['arg_col'] == target[1] + 1


def absent_pairs(det: dict) -> list[str]:
    """Probe pairs whose absent direction has at most ABSENT_MAX_DETECTORS exclusive detectors at ViT block 9."""
    return [p for p in lf.PROBE_PAIRS if det[p]['9']['B|A']['n_detectors'] <= ABSENT_MAX_DETECTORS]


def block_profiles(det: dict, causal: list[dict]) -> dict:
    """Per-block summaries used by the block choice (arrays indexed by block - 1)."""
    nb = len(det[lf.PROBE_PAIRS[0]])
    abs_p = absent_pairs(det)
    prof = {'absent_pairs': abs_p, 'blocks': list(range(1, nb + 1))}
    for key, direction in (('present', 'A|B'), ('absent', 'B|A')):
        pairs = lf.PROBE_PAIRS if key == 'present' else abs_p
        for stat in ('n_detectors', 'max_zeta', 'achievable'):
            prof[f'{key}_{stat}'] = np.array([[det[p][str(b)][direction][stat] for b in range(1, nb + 1)]
                                              for p in pairs])
    prof['eligible_fraction'] = (prof['absent_achievable'] >= ACHIEVABLE_MIN).mean(0)
    # causal: hit-rate change per block; rows of one quad follow its 'none' row (stage_locate writes them in order)
    base = defaultdict(list)
    by = defaultdict(lambda: defaultdict(list))
    ref = None
    for r in causal:
        if r['kind'] == 'none':
            ref = (hit(r, r['target']), r['s'])
            base[(r['exp'], r['pair'])].append(ref[0])
            continue
        by[(r['exp'], r['kind'], tuple(r['blocks']))][r['pair']].append((hit(r, r['target']) - ref[0], r['s'] - ref[1]))
    for exp, pairs in (('delete', lf.PROBE_PAIRS), ('transplant', abs_p)):
        dh = np.full((len(pairs), nb), np.nan)
        ds = np.full((len(pairs), nb), np.nan)
        for i, p in enumerate(pairs):
            for b in range(1, nb + 1):
                v = by[(exp, 'single', (b,))].get(p, [])
                if v:
                    dh[i, b - 1] = np.mean([x[0] for x in v])
                    ds[i, b - 1] = np.mean([x[1] for x in v])
        prof[f'{exp}_dhit'], prof[f'{exp}_dscore'] = dh, ds
        prof[f'{exp}_base_hit'] = np.array([np.mean(base[(exp, p)]) for p in pairs])
    win = sorted({tuple(r['blocks']) for r in causal if r['kind'] == 'window'})
    prof['delete_window'] = {f'{w[0]}-{w[-1]}': float(np.mean([x[0] for p in lf.PROBE_PAIRS
                                                               for x in by[('delete', 'window', w)].get(p, [])]))
                             for w in win}
    return prof


def choose_window(prof: dict) -> dict:
    """Pre-registered rule: the WINDOW consecutive blocks, all with eligible_fraction >= ELIGIBLE_FRACTION, that
    maximise the summed mean transplant hit gain on feature-absent targets (ties -> earlier window)."""
    gain = np.nanmean(prof['transplant_dhit'], 0)
    nb = len(gain)
    best, best_v = None, -np.inf
    for s in range(nb - WINDOW + 1):
        w = list(range(s, s + WINDOW))
        if (prof['eligible_fraction'][w] < ELIGIBLE_FRACTION).any():
            continue
        v = float(gain[w].sum())
        if v > best_v + 1e-9:
            best, best_v = [x + 1 for x in w], v
    return {'blocks': best, 'transplant_gain_sum': best_v, 'gain_by_block': gain.tolist()}


# ----------------------------------------------------------------------------- figures

def fig_locate(prof: dict, choice: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b = np.array(prof['blocks'])
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    shade = lambda a: [a.axvspan(choice['blocks'][0] - 0.5, choice['blocks'][-1] + 0.5, color='#dddddd', zorder=0)] \
        if choice['blocks'] else None
    a = ax[0]
    shade(a)
    a.plot(b, np.median(prof['present_n_detectors'], 0), color='#1f77b4', marker='o', ms=3, label='one A among Bs (present)')
    a.plot(b, np.median(prof['absent_n_detectors'], 0), color='#ff7f0e', marker='o', ms=3, label='one B among As (absent)')
    a.set_yscale('symlog')
    a.set_xlabel('ViT block')
    a.set_ylabel('exclusive detectors (zeta > 5), median over pairs')
    a.legend(fontsize=8)
    a.set_title('(a) existing detectors')
    a = ax[1]
    shade(a)
    for key, col, lab in (('absent_max_zeta', '#ff7f0e', 'best existing neuron (absent)'),
                          ('absent_achievable', '#d62728', 'best new linear unit (absent)'),
                          ('present_max_zeta', '#1f77b4', 'best existing neuron (present)')):
        m = np.median(prof[key], 0)
        q1, q3 = np.percentile(prof[key], [25, 75], 0)
        a.plot(b, m, color=col, marker='o', ms=3, label=lab)
        a.fill_between(b, q1, q3, color=col, alpha=0.15)
    a.axhline(ACHIEVABLE_MIN, color='k', ls=':', lw=1)
    a.set_yscale('symlog')
    a.set_xlabel('ViT block')
    a.set_ylabel('deviation zeta of the target (median, IQR)')
    a.legend(fontsize=8)
    a.set_title('(b) existing vs achievable detector')
    a = ax[2]
    shade(a)
    a.plot(b, np.nanmean(prof['delete_dhit'], 0), color='#1f77b4', marker='o', ms=3)
    a.axhline(0, color='k', lw=0.5)
    a.set_xlabel('ViT block (MLP output at the target replaced)')
    a.set_ylabel('change in hit rate, present targets')
    a.set_title('(c) deletion of the present-feature MLP output')
    a = ax[3]
    shade(a)
    a.plot(b, np.nanmean(prof['transplant_dhit'], 0), color='#ff7f0e', marker='o', ms=3)
    a.axhline(0, color='k', lw=0.5)
    a.set_xlabel('ViT block (foreign detector output added)')
    a.set_ylabel('change in hit rate, absent targets')
    a.set_title('(d) transplant of a detector output')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--interp-run', required=True)
    p.add_argument('--figures', required=True)
    p.add_argument('--only-locate', action='store_true')
    args = p.parse_args()
    import torch as _torch
    _torch.set_num_threads(16)            # CPU cap (common rules)
    run, figs = Path(args.run), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    det = json.loads((run / 'locate' / 'detectors.json').read_text())
    prof = block_profiles(det, read_jsonl(run / 'locate' / 'causal.jsonl'))
    choice = choose_window(prof)
    fig_locate(prof, choice, figs / 'locate_profiles.png')
    summary = {'choice': choice, 'absent_pairs': prof['absent_pairs'],
               'eligible_fraction': prof['eligible_fraction'].tolist(),
               'delete_window': prof['delete_window'],
               **{k: np.round(np.nanmedian(v, 0), 3).tolist() if k.endswith(('n_detectors', 'zeta', 'achievable'))
                  else np.round(np.nanmean(v, 0), 3).tolist()
                  for k, v in prof.items() if isinstance(v, np.ndarray) and v.ndim == 2},
               'delete_base_hit': prof['delete_base_hit'].tolist(), 'transplant_base_hit': prof['transplant_base_hit'].tolist()}
    (run / 'locate' / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(choice))
    if args.only_locate:
        return
    res = analyse(run)
    (run / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    figures(run, res, choice, figs)



def fig_examples(path: Path) -> None:
    """Every pair: A and B tiles (salient level for visibility of the part) and one 6x6 feature-absent image crop."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis_tools.representation import search_asymmetry as sa
    pairs = list(lf.PROBE_PAIRS)
    fig, ax = plt.subplots(2, len(pairs), figsize=(1.6 * len(pairs), 3.6))
    for j, p in enumerate(pairs):
        rng = np.random.default_rng(j)
        jit = sa.jitter_map(3, 3, rng)
        img, _ = lf.grid_image(p, lf.LEVELS[p]['subtle'], 3, 3, (1, 1), 0, jit, 'swap')
        tiles = np.concatenate([lf.render_entity(p, True, lf.LEVELS[p]['subtle'], 0, (0, 0)),
                                lf.render_entity(p, False, lf.LEVELS[p]['subtle'], 0, (0, 0))], 1)
        ax[0, j].imshow(tiles)
        ax[0, j].set_title(f"{p}\n{lf.split_of(p).replace('_', ' ')}", fontsize=8)
        ax[1, j].imshow(np.asarray(img))
        for a in ax[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    ax[0, 0].set_ylabel('A | B', fontsize=9)
    ax[1, 0].set_ylabel('3x3 absent\n(target centre)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- training conditions

BENCHES = ('pope', 'textvqa', 'mmvp', 'gqa')
SPLITS = ('train_feature', 'heldout_feature', 'heldout_family')
EARLIER_BINS = ((0, 5), (6, 11), (12, 23), (24, 35))


def _load(run: Path, cond: str, name: str) -> list[dict]:
    p = run / 'eval' / cond / f'{name}.jsonl'
    return read_jsonl(p) if p.exists() else []


def _prop(rows: list[dict], key: str = 'correct') -> dict:
    k, n = sum(bool(r[key]) for r in rows), len(rows)
    p_, lo, hi = io_.wilson(k, n)
    return {'k': k, 'n': n, 'p': p_, 'lo': lo, 'hi': hi}


def paired(a: list[dict], b: list[dict], select, key: str = 'correct') -> dict:
    A = {r['id']: bool(r[key]) for r in a if select(r)}
    B = {r['id']: bool(r[key]) for r in b if select(r)}
    ids = sorted(set(A) & set(B))
    b01 = sum((not A[i]) and B[i] for i in ids)
    b10 = sum(A[i] and not B[i] for i in ids)
    return {'n': len(ids), 'a_wrong_b_right': b01, 'a_right_b_wrong': b10, 'p': io_.mcnemar_exact(b01, b10)}


def conditions(run: Path) -> list[str]:
    """'base' plus every trained condition with evaluation results (adapter metadata in adapters/meta.json, written
    from the adapter states before the uncited adapter files were deleted)."""
    meta = json.loads((run / 'adapters' / 'meta.json').read_text())
    return ['base'] + sorted(n for n in meta if (run / 'eval' / n).exists())


def analyse(run: Path) -> dict:
    res: dict = {}
    for c in conditions(run):
        r: dict = {}
        if c != 'base':
            r['adapter'] = json.loads((run / 'adapters' / 'meta.json').read_text())[c]
        probe = _load(run, c, 'probe')
        if probe:
            pr = {}
            for sp in SPLITS:
                for v in ('swap', 'std'):
                    pr[f'{sp}_{v}'] = _prop([x for x in probe if x['split'] == sp and x['variant'] == v])
            for pair in lf.PROBE_PAIRS:
                for v in ('swap', 'std'):
                    pr[f'{pair}_{v}'] = _prop([x for x in probe if x['pair'] == pair and x['variant'] == v])
            for sp in SPLITS:
                sel = [x for x in probe if x['split'] == sp and x['variant'] == 'swap']
                for a, b in EARLIER_BINS:
                    pr[f'{sp}_swap_earlier_{a}-{b}'] = _prop([x for x in sel if a <= x['target'][0] * 6 + x['target'][1] <= b])
            pr['invalid'] = sum(not x['valid'] for x in probe)
            r['probe'] = pr
        count = _load(run, c, 'count')
        if count:
            r['count'] = _prop(count)
        coord = _load(run, c, 'coord')
        if coord:
            r['coord'] = {f: {**_prop([x for x in coord if x['family'] == f], 'hit'),
                              'mean_err': float(np.mean([x['error'] for x in coord if x['family'] == f and x['valid']]))}
                          for f in sorted({x['family'] for x in coord})}
        gen = _load(run, c, 'general')
        if gen:
            r['general'] = {b: float(np.mean([x['correct'] for x in gen if x['bench'] == b])) for b in BENCHES}
        text = _load(run, c, 'text')
        if text:
            r['text'] = _prop(text)
        bv = _load(run, c, 'babyvision')
        if bv:
            r['babyvision'] = {s: _prop([dict(x, correct=x['score']['correct']) for x in bv if x['subtype'] == s])
                               for s in sorted({x['subtype'] for x in bv})}
            r['babyvision']['all'] = _prop([dict(x, correct=x['score']['correct']) for x in bv])
        dp = run / 'eval' / c / 'detectors.json'
        if dp.exists():
            det = json.loads(dp.read_text())
            r['detectors'] = {p: {d: [det[p][str(b)][d]['n_detectors'] for b in range(1, 28)] for d in ('A|B', 'B|A')}
                              for p in det}
            r['detectors_maxzeta'] = {p: [det[p][str(b)]['B|A']['max_zeta'] for b in range(1, 28)] for p in det}
        pv = _load(run, f'{c}_preview', 'preview')
        if pv:
            out = {}
            for cond in sorted({x['condition'] for x in pv}):
                sel = [x for x in pv if x['condition'] == cond]
                out[cond] = {'all': _prop(sel), **{f'{a}-{b}': _prop([x for x in sel if a <= x['target'][0] * 6 + x['target'][1] <= b])
                                                   for a, b in EARLIER_BINS},
                             **{sp: _prop([x for x in sel if lf.split_of(x['pair']) == sp]) for sp in SPLITS}}
            r['preview'] = out
            base_single = [x for x in pv if x['condition'] == 'single']
            r['preview_paired'] = {cond: paired([dict(x, id=x['id']) for x in base_single],
                                                [x for x in pv if x['condition'] == cond], lambda x: True)
                                   for cond in out if cond != 'single'}
        res[c] = r
    # paired comparisons against base
    comps = {}
    pb, gb = _load(run, 'base', 'probe'), _load(run, 'base', 'general')
    for c in res:
        if c == 'base':
            continue
        pc = _load(run, c, 'probe')
        cm = {}
        if pb and pc:
            for sp in SPLITS:
                cm[f'{sp}_swap'] = paired(pb, pc, lambda x, sp=sp: x['split'] == sp and x['variant'] == 'swap')
                cm[f'{sp}_std'] = paired(pb, pc, lambda x, sp=sp: x['split'] == sp and x['variant'] == 'std')
        gc = _load(run, c, 'general')
        if gb and gc:
            for bench in BENCHES:
                A = {x['id']: x['correct'] for x in gb if x['bench'] == bench}
                B = {x['id']: x['correct'] for x in gc if x['bench'] == bench}
                ids = sorted(set(A) & set(B))
                d = np.array([float(B[i]) - float(A[i]) for i in ids])
                rng = np.random.default_rng(0)
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                cm[f'general_{bench}'] = {'n': len(ids), 'delta': float(d.mean()),
                                          'ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        cb, cc = _load(run, 'base', 'coord'), _load(run, c, 'coord')
        if cb and cc:
            cm['coord_refcoco'] = paired(cb, cc, lambda x: x['family'] == 'refcoco', 'hit')
        comps[c] = cm
    res['_paired_vs_base'] = comps
    return res


def figures(run: Path, res: dict, choice: dict, figs: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'loc': '#d62728', 'late': '#ff7f0e', 'vitall': '#1f77b4', 'modelall': '#7f7f7f'}
    labels = {'loc': 'MLP blocks 13-15 (predicted)', 'late': 'MLP blocks 25-27', 'vitall': 'full-ViT LoRA',
              'modelall': 'full-model LoRA'}
    # data-efficiency curves
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, sp in enumerate(SPLITS):
        a = ax[j]
        b = res['base']['probe'][f'{sp}_swap']
        a.axhline(b['p'], color='k', ls=':', lw=1, label='base')
        for fam, col in colors.items():
            pts = sorted((v['adapter']['n_odd'], v['probe'][f'{sp}_swap'], k) for k, v in res.items()
                         if k.startswith(fam + '_') and 'probe' in v and v['adapter'].get('seed') == 1)
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1]['p'] for p in pts]
            e = [[p[1]['p'] - p[1]['lo'] for p in pts], [p[1]['hi'] - p[1]['p'] for p in pts]]
            a.errorbar(x, y, yerr=e, color=col, marker='o', capsize=3, label=labels[fam],
                       ls='-' if len(pts) > 1 else 'none')
            seeds = [(v['adapter']['n_odd'], v['probe'][f'{sp}_swap']['p']) for k, v in res.items()
                     if k.startswith(fam + '_') and 'probe' in v and v['adapter'].get('seed', 1) != 1]
            if seeds:
                a.scatter([s[0] for s in seeds], [s[1] for s in seeds], color=col, marker='x')
        a.set_xscale('log')
        a.set_xlabel('odd-entity training items (log scale)')
        a.set_ylabel('accuracy, feature-absent targets')
        a.set_title(sp.replace('_', ' '))
        a.set_ylim(0, 1)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'data_efficiency.png', dpi=150)
    plt.close(fig)
    # detector formation
    tuned = [k for k in res if 'detectors' in res[k] and k != 'base']
    if 'detectors' in res.get('base', {}) and tuned:
        groups = (('train_feature', lf.TRAIN_PAIRS), ('heldout_feature', lf.HELDOUT_FEATURES),
                  ('heldout_family', lf.HELDOUT_FAMILY))
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        bl = np.arange(1, 28)
        for j, (name, pairs) in enumerate(groups):
            a = ax[j]
            if choice.get('blocks'):
                a.axvspan(choice['blocks'][0] - 0.5, choice['blocks'][-1] + 0.5, color='#dddddd', zorder=0)
            for k in ['base'] + tuned:
                if k != 'base' and k not in ('loc_n1200_s1', 'vitall_n1200_s1', 'late_n1200_s1'):
                    continue
                fam = k.split('_')[0]
                y = np.median([res[k]['detectors'][p]['B|A'] for p in pairs], 0)
                a.plot(bl, y, color='k' if k == 'base' else colors[fam], marker='o', ms=3,
                       label='base' if k == 'base' else labels[fam])
            a.set_yscale('symlog')
            a.set_xlabel('ViT block')
            a.set_ylabel('exclusive detectors for the absent target (median)')
            a.set_title(name.replace('_', ' '))
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / 'detector_formation.png', dpi=150)
        plt.close(fig)
    # window scan: 3-block MLP LoRA (rank 8, 1200 items) at every window
    wins = {1: 'early_n1200_s1', 4: 'w04_n1200_s1', 7: 'w07_n1200_s1', 10: 'w10_n1200_s1', 13: 'loc_n1200_s1',
            16: 'w16_n1200_s1', 19: 'w19_n1200_s1', 22: 'w22_n1200_s1', 25: 'late_n1200_s1'}
    wins = {k: v for k, v in wins.items() if v in res and 'probe' in res[v]}
    if len(wins) > 2:
        fig, a = plt.subplots(figsize=(8, 4.2))
        xs = np.array(sorted(wins)) + 1
        for sp, col in zip(SPLITS, ('#2ca02c', '#d62728', '#9467bd')):
            ps = [res[wins[x - 1]]['probe'][f'{sp}_swap'] for x in xs]
            a.errorbar(xs, [p['p'] for p in ps], yerr=[[p['p'] - p['lo'] for p in ps], [p['hi'] - p['p'] for p in ps]],
                       color=col, marker='o', capsize=3, label=sp.replace('_', ' '))
            a.axhline(res['base']['probe'][f'{sp}_swap']['p'], color=col, ls=':', lw=1)
        a.axvspan(12.5, 15.5, color='#dddddd', zorder=0)
        a.set_xticks(xs)
        a.set_xticklabels([f'{x - 1}-{x + 1}' for x in xs])
        a.set_xlabel('ViT blocks whose MLP carries the LoRA (0.26M parameters)')
        a.set_ylabel('accuracy, feature-absent targets')
        a.set_ylim(0, 1)
        a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / 'window_scan.png', dpi=150)
        plt.close(fig)
    # general ability deltas
    comps = res['_paired_vs_base']
    keys = [k for k in comps if any(f'general_{b}' in comps[k] for b in BENCHES)]
    if keys:
        fig, a = plt.subplots(figsize=(max(8, 0.9 * len(keys) * 1.2), 4))
        w = 0.8 / len(BENCHES)
        bc = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']
        for i, bch in enumerate(BENCHES):
            d = [comps[k][f'general_{bch}']['delta'] for k in keys]
            lo = [comps[k][f'general_{bch}']['delta'] - comps[k][f'general_{bch}']['ci'][0] for k in keys]
            hi = [comps[k][f'general_{bch}']['ci'][1] - comps[k][f'general_{bch}']['delta'] for k in keys]
            a.bar(np.arange(len(keys)) + i * w, d, w, yerr=[lo, hi], color=bc[i], label=bch, capsize=2)
        a.axhline(0, color='k', lw=0.5)
        a.set_xticks(np.arange(len(keys)) + 0.3)
        a.set_xticklabels(keys, rotation=40, ha='right', fontsize=8)
        a.set_ylabel('accuracy change vs base (paired)')
        a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / 'general_delta.png', dpi=150)
        plt.close(fig)
    # preview pilot
    pv = res.get('base', {}).get('preview')
    if pv:
        fig, a = plt.subplots(figsize=(7, 4))
        cs = {'single': 'k', 'preview_first': '#d62728', 'blank_first': '#7f7f7f', 'preview_after': '#1f77b4'}
        xb = np.arange(len(EARLIER_BINS))
        for i, (cond, col) in enumerate(cs.items()):
            if cond not in pv:
                continue
            ps = [pv[cond][f'{lo}-{hi}'] for lo, hi in EARLIER_BINS]
            a.errorbar(xb + (i - 1.5) * 0.08, [p['p'] for p in ps], yerr=[[p['p'] - p['lo'] for p in ps], [p['hi'] - p['p'] for p in ps]],
                       color=col, marker='o', capsize=3, label=cond.replace('_', ' '))
        a.set_xticks(xb)
        a.set_xticklabels([f'{lo}-{hi}' for lo, hi in EARLIER_BINS])
        a.set_xlabel('entities earlier in raster order than the target')
        a.set_ylabel('accuracy, feature-absent targets')
        a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / 'preview_pilot.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
