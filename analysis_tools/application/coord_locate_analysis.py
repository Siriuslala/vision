"""Experiment 7.2 analysis (CPU): coordinate accuracy, cost, data efficiency, general ability, BabyVision and the
representation check of the pre-registered prediction.

Every condition listed in --conditions must have complete outputs; missing files raise instead of being skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis_tools.application import coord_locate as cl
from analysis_tools.position import readout_prior as rp
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_analysis as tfa
from analysis_tools.position import targeted_finetune_r2 as t2
from analysis_tools.position.coordinate_readout_run import read_jsonl

C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
     'green': '#008300', 'gray': '#8a8a8a', 'black': '#222222'}


def wilson(k: int, n: int) -> list[float]:
    lo, hi = t2.wilson(k, n)
    return [k / n if n else float('nan'), lo, hi]


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    return tfa.mcnemar_p(int((a & ~b).sum()), int((~a & b).sum()))


def adapter_info(run: Path, cond: str) -> dict:
    if cond == 'base':
        return {'n_params': 0}
    st = torch.load(run / 'adapters' / f'{cond}.pt', map_location='cpu', weights_only=False)
    return {'n_params': int(st['n_params']), 'train_seconds': float(st.get('train_seconds', float('nan'))),
            'n_steps': int(st['n_steps']), 'epochs': int(st.get('epochs', 1)), 'n_train': st.get('n_train'),
            'positions': st.get('positions', tf.CONDITIONS[cond].get('positions')), 'gpu': st.get('gpu', 'unrecorded'),
            'lr': st.get('lr')}


def digits(run: Path, cond: str, name: str = 'digits.jsonl') -> dict:
    rows = read_jsonl(run / 'eval' / cond / name)
    out = {}
    for fam in ('sweep_h', 'sweep_v', 'refcoco'):
        rr = [r for r in rows if r['family'] == fam]
        t = np.array([r['truth'] for r in rr])
        l0, l1 = np.array([r['logp0'] for r in rr]), np.array([r['logp1'] for r in rr])
        out[fam] = {**t2.digit_metrics(t, l0, l1), 'margin': rp.margin_decomposition(t, l0)}
    return out


def babyvision(run: Path, cond: str, base: dict | None) -> dict:
    rows = {r['id']: r for r in read_jsonl(run / 'eval' / cond / 'babyvision.jsonl')}
    out = {}
    for task in ('official', 'odd_rowcol', 'odd_point', 'red_point', 'red_rowcol'):
        ids = sorted(i for i, r in rows.items() if r['task'] == task)
        key = 'hit' if task.endswith('point') else 'correct'
        v = np.array([bool(rows[i][key]) for i in ids])
        o = {'acc': wilson(int(v.sum()), len(v)), 'n': len(v),
             'truncated': int(sum(rows[i]['finish_reason'] == 'length' for i in ids))}
        if task.endswith('point'):
            e = np.array([rows[i].get('error', np.nan) for i in ids], float)
            o['median_error'] = float(np.nanmedian(e)) if np.isfinite(e).any() else float('nan')
        if task.endswith('rowcol'):
            # Lenient re-parse (tolerates the fine-tuned answer-template drift); strict = base JSON parser.
            from analysis_tools.application.coord_locate_run import lenient_rowcol_json
            cases = {json.loads(l)['id']: json.loads(l) for l in (run / 'babyvision' / 'cases.jsonl').read_text().splitlines()}
            len_ = [lenient_rowcol_json(rows[i]['text']) for i in ids]
            tru = [cases[i]['truth']['rc'] for i in ids]
            lc = np.array([l is not None and list(l) == t for l, t in zip(len_, tru)])
            o['lenient'] = wilson(int(lc.sum()), len(lc))
            o['strict_parse_fail'] = int(sum(rows[i]['rc'] is None for i in ids))
            o['row'] = float(np.mean([l is not None and l[0] == t[0] for l, t in zip(len_, tru)]))
            o['col'] = float(np.mean([l is not None and l[1] == t[1] for l, t in zip(len_, tru)]))
            o['row_plus1'] = float(np.mean([l is not None and l[0] == t[0] + 1 for l, t in zip(len_, tru)]))
            o['col_plus1'] = float(np.mean([l is not None and l[1] == t[1] + 1 for l, t in zip(len_, tru)]))
            if base is not None:
                bl = np.array([(lambda l, t: l is not None and list(l) == t)(lenient_rowcol_json(base[i]['text']), t)
                               for i, t in zip(ids, tru)])
                o['lenient_mcnemar_p'] = mcnemar(bl, lc)
        if task == 'official':
            o['cannot_see'] = int(sum(any(k in rows[i]['text'].lower() for k in ("can't see", 'cannot see', 'without seeing',
                                                                                   'unable to see', "don't see the image"))
                                      for i in ids))
        if task == 'official':
            o['strict'] = int(sum(rows[i]['strict'] for i in ids))
            o['unfinished'] = int(sum(rows[i]['finish_reason'] == 'length' for i in ids))
        if base is not None:
            b = np.array([bool(base[i][key]) for i in ids])
            o['mcnemar_p'] = mcnemar(b, v)
        out[task] = o
    return out


def ordinal(run: Path, cond: str, base: dict | None) -> dict:
    rows = {r['id']: r for r in read_jsonl(run / 'eval' / cond / 'ordinal.jsonl')}
    ids = sorted(rows)
    out = {}
    for fam in ('grid06', 'grid10', 'grid20', 'all'):
        ii = [i for i in ids if fam == 'all' or rows[i]['family'] == fam]
        c = np.array([rows[i]['correct'] for i in ii])
        o = {'lenient': wilson(int(c.sum()), len(c)),
             'strict': wilson(int(sum(rows[i]['strict_correct'] for i in ii)), len(ii)),
             'row': float(np.mean([rows[i]['row_err'] == 0 for i in ii])),
             'col': float(np.mean([rows[i]['col_err'] == 0 for i in ii])),
             'row_plus1': float(np.mean([rows[i]['row_err'] == 1 for i in ii])),
             'col_plus1': float(np.mean([rows[i]['col_err'] == 1 for i in ii])),
             'unparsed': int(sum(rows[i]['lenient'] is None for i in ii))}
        if base is not None:
            o['mcnemar_p'] = mcnemar(np.array([base[i]['correct'] for i in ii]), c)
        out[fam] = o
    return out


def ordinal_errors(run: Path, cond: str) -> dict:
    """Counts of cl.ordinal_error_kind over all row and column answers (144 = 72 images x 2 axes), overall and per
    grid family. The target centre in 0-1000 units comes from the coordinate truth of the same test image."""
    centre = {}
    for r in read_jsonl(run / 'test.jsonl'):
        if r['family'] in ('grid06', 'grid10', 'grid20'):
            (cx, cy), (w, h) = r['truth']['center'], r['truth']['canvas']
            centre[r['id']] = (cx / w * 1000, cy / h * 1000)
    out = {f: {k: 0 for k in cl.ORDINAL_ERROR_KINDS} for f in ('grid06', 'grid10', 'grid20', 'all')}
    for r in read_jsonl(run / 'eval' / cond / 'ordinal.jsonl'):
        if r['id'] not in centre:
            raise KeyError(f"ordinal id {r['id']} missing from test.jsonl")
        cx, cy = centre[r['id']]
        pred = r['lenient'] if r['lenient'] is not None else (None, None)
        for p, t, c in ((pred[0], r['truth'][0], cy), (pred[1], r['truth'][1], cx)):
            k = cl.ordinal_error_kind(p, t, c)
            out[r['family']][k] += 1
            out['all'][k] += 1
    return out


def clock_curves(feat: Path, cond: str) -> dict | None:
    path = feat / 'capture' / f'{cond}.npz'
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    R, fam, tr, layers = z['resid'], z['family'], z['truth'], list(z['layers'])
    out = {}
    for f in ('sweep_h', 'sweep_v', 'refcoco', 'textconv'):
        s = fam == f
        for si, site in enumerate(('q0', 'target')):
            out[f'{f}|{site}'] = {int(l): rp.clock_decodability(R[s, si, li].astype(np.float32), tr[s])
                                  for li, l in enumerate(layers)}
    return out


LABEL = {'base': 'base', 'all_linear': 'all linear (21.6M)', 'mlp_all': 'all MLPs (12.6M)',
         'early1_4_img': 'L1-4 all linear @img (2.7M)', 'mlp1_4_img': 'L1-4 MLP @img (1.57M)',
         'mlp1_4_nofirst': 'L1-4 MLP @img w/o first', 'mlp1_4_first': 'L1-4 MLP @first img token',
         'mlp1_4_text': 'L1-4 MLP @text', 'mlp13_16_img': 'L13-16 MLP @img', 'mlp25_28_img': 'L25-28 MLP @img',
         'mlp1_4_img_s2': 'L1-4 MLP @img, seed 2', 'down1_4_first_r1': 'L1-4 down rank 1 @first (65k)',
         'mlp1_4_img_fix': 'L1-4 MLP @img + mixed data'}


def fig_main(S: dict, figs: Path, conds: list[str]) -> None:
    cs = [c for c in LABEL if c in conds]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    y = np.arange(len(cs))
    pal = {'base': C['gray'], 'all_linear': C['blue'], 'mlp_all': C['blue'], 'early1_4_img': C['aqua'],
           'mlp1_4_img': C['orange']}
    cols = [pal.get(c, C['yellow']) for c in cs]
    vals = [[S['coord'][c]['disc_x']['error']['mean'] for c in cs],
            [S['coord'][c]['grid20']['hit']['p'] for c in cs],
            [S['digits'][c]['sweep_h']['gain'] if c in S['digits'] else np.nan for c in cs],
            [S['digits'][c]['sweep_h']['p_second'] if c in S['digits'] else np.nan for c in cs]]
    titles = ('disc x mean error (0-1000 units)', '20x20 grid point hit rate', 'second-digit gain (disc x)',
              'P(true second digit) (disc x)')
    for ax, v, t in zip(axes, vals, titles):
        ax.barh(y, v, color=cols, height=0.7)
        for yi, vi in zip(y, v):
            if np.isfinite(vi):
                ax.text(vi, yi, f' {vi:.2f}' if vi < 5 else f' {vi:.1f}', va='center', fontsize=7, color=C['black'])
        ax.set_title(t, fontsize=9)
        ax.grid(axis='x', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(y, [LABEL[c] for c in cs], fontsize=8)
    axes[0].invert_yaxis()
    fig.savefig(figs / 'main_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_efficiency(S: dict, figs: Path) -> None:
    fam = {'mlp1_4_img': (C['orange'], 'L1-4 MLP @img (1.57M)'), 'all_linear': (C['blue'], 'all linear (21.6M)'),
           'early1_4_img': (C['aqua'], 'L1-4 all linear @img (2.7M)')}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for base, (c, lab) in fam.items():
        pts = [(n, cond) for n, cond in ((202, f'{base}_n202'), (800, f'{base}_n800'), (3200, base)) if cond in S['coord']]
        if not pts:
            continue
        ns = [p[0] for p in pts]
        axes[0].plot(ns, [S['coord'][p[1]]['disc_x']['error']['mean'] for p in pts], 'o-', color=c, lw=2, label=lab)
        axes[1].plot(ns, [S['coord'][p[1]]['refcoco']['error']['mean'] for p in pts], 'o-', color=c, lw=2, label=lab)
    for ax, t in zip(axes, ('disc x mean error', 'RefCOCO mean centre error')):
        ax.set_xscale('log')
        ax.set_xticks([202, 800, 3200], ['202', '800', '3200'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel('training samples (200 steps in every condition)')
        ax.set_ylabel(t + ' (0-1000 units)')
        ax.axhline(S['coord']['base']['disc_x' if 'disc' in t else 'refcoco']['error']['mean'], color=C['gray'], ls='--', lw=1,
                   label='base')
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(figs / 'data_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_clock(S: dict, figs: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    panels = (('sweep_h|target', 'disc x: target image tokens'), ('sweep_h|q0', 'disc x: q0'), ('sweep_v|q0', 'disc y: q0'))
    for ax, (key, title) in zip(axes, panels):
        for c, col in (('base', C['gray']), ('all_linear', C['blue']), ('early1_4_img', C['aqua']), ('mlp1_4_img', C['orange']),
                       ('mlp1_4_first', C['yellow']), ('mlp1_4_img_fix', C['magenta'])):
            if c in S['clock']:
                d = S['clock'][c][key]
                ax.plot(list(d), [max(-1, v) for v in d.values()], 'o-', ms=3, lw=2, color=col, label=LABEL[c])
        ax.axhline(0, color=C['gray'], lw=1)
        ax.axvline(20, color=C['gray'], lw=1, ls=':')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('layer')
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('block-CV R^2 of cos/sin(2 pi x / 100)\n(clipped at -1)')
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(figs / 'clock_after_training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_ordinal(S: dict, figs: Path, conds: list[str]) -> None:
    """Stacked shares of the ordinal answer classes (row and column answers pooled over the 72 grid images)."""
    cs = [c for c in LABEL if c in conds and c in S['ordinal_errors']]
    kinds = cl.ORDINAL_ERROR_KINDS
    cols = {'correct': C['green'], 'off_by_1': C['aqua'], 'off_by_2_3': C['yellow'], 'coord_prefix': C['magenta'],
            'other': C['gray'], 'unparsed': C['black']}
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(cs) + 1.2))
    left = np.zeros(len(cs))
    for k in kinds:
        v = np.array([S['ordinal_errors'][c]['all'][k] / sum(S['ordinal_errors'][c]['all'].values()) for c in cs])
        ax.barh(np.arange(len(cs)), v, left=left, color=cols[k], height=0.7, label={'off_by_2_3': 'off by 2-3'}.get(k, k.replace('_', ' ')))
        left += v
    ax.set_yticks(np.arange(len(cs)), [LABEL[c] for c in cs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('share of row and column answers (72 grid images x 2 axes)')
    ax.legend(fontsize=7, frameon=False, ncol=6, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(figs / 'ordinal_errors.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--figures', required=True)
    p.add_argument('--conditions', nargs='+', required=True)
    p.add_argument('--skip', default='', help='comma list of sections not evaluated for every condition')
    args = p.parse_args()
    cl.register()
    run, feat, figs = Path(args.run), Path(args.features), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    conds = args.conditions
    S = {'adapters': {c: adapter_info(run, c) for c in conds}}
    S['coord'] = tfa.coord(run, conds)
    gen_conds = [c for c in conds if (run / 'eval' / c / 'general.jsonl').exists()]
    S['general'] = tfa.general(run, gen_conds)
    S['digits'] = {c: digits(run, c) for c in conds if (run / 'eval' / c / 'digits.jsonl').exists()}
    bv_base = {r['id']: r for r in read_jsonl(run / 'eval' / 'base' / 'babyvision.jsonl')} \
        if (run / 'eval' / 'base' / 'babyvision.jsonl').exists() else None
    S['babyvision'] = {c: babyvision(run, c, None if c == 'base' else bv_base) for c in conds
                       if (run / 'eval' / c / 'babyvision.jsonl').exists()}
    S['clock'] = {c: v for c in conds if (v := clock_curves(feat, c)) is not None}
    ob = {r['id']: r for r in read_jsonl(run / 'eval' / 'base' / 'ordinal.jsonl')} \
        if (run / 'eval' / 'base' / 'ordinal.jsonl').exists() else None
    S['ordinal'] = {c: ordinal(run, c, None if c == 'base' else ob) for c in conds
                    if (run / 'eval' / c / 'ordinal.jsonl').exists()}
    S['ordinal_errors'] = {c: ordinal_errors(run, c) for c in conds if (run / 'eval' / c / 'ordinal.jsonl').exists()}
    S['digits_splice'] = {f'{c}@{k}': digits(run, c, f'digits_{k}.jsonl') for c in conds for k in ('question', 'prefix')
                          if (run / 'eval' / c / f'digits_{k}.jsonl').exists()}
    (run / 'summary.json').write_text(json.dumps(S, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    print('wrote', run / 'summary.json', flush=True)
    fig_main(S, figs, conds)
    fig_efficiency(S, figs)
    fig_ordinal(S, figs, conds)
    if S['clock']:
        fig_clock(S, figs)


if __name__ == '__main__':
    main()
