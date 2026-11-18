"""Experiment 7.4 analysis (CPU): diagnosis of BabyVision row/column find-the-difference failures, condition
comparison with paired tests and Wilson intervals, error-type shifts, general ability, format drift and the
representation checks (7.2 periodic code, 7.3C gate statistics).

Conditions without a complete output file are listed under summary['missing'] (never silently dropped).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.application import bv_rowcol as bv
from analysis_tools.application import coord_locate_analysis as cla
from analysis_tools.position.coordinate_readout_run import read_jsonl

COND_ORDER = ('base', 'main', 'mlp_only', 'gate_only', 'vit_only', 'main_nocot', 'main_alltok', 'main_e3', 'main_alltok_e3', 'main_alltok_e3_s2', 'mlp_alltok_e3', 'full_lora')
LABEL = {'base': 'base', 'main': 'main (L1-4 MLP + gate L6,7,9,10 @img + ViT 25-27)', 'main_s2': 'main, seed 2',
         'mlp_only': 'L1-4 MLP @img', 'gate_only': 'gate L6,7,9,10 @img', 'vit_only': 'ViT MLP 25-27',
         'main_nocot': 'main, no reasoning samples', 'full_lora': 'all-linear LoRA (decoder)',
         'main_alltok': 'main modules, all tokens', 'main_alltok_s2': 'main modules, all tokens, seed 2',
         'mlp_alltok': 'L1-4 MLP, all tokens', 'main_e3': 'main, 3 epochs', 'main_alltok_e3': 'main modules, all tokens, 3 epochs',
         'main_alltok_e3_s2': 'main modules, all tokens, 3 epochs, seed 2', 'mlp_alltok_e3': 'L1-4 MLP, all tokens, 3 epochs'}
SHORT = {'base': 'base', 'main': 'main', 'main_s2': 'main s2', 'mlp_only': 'MLP', 'gate_only': 'gate',
         'vit_only': 'ViT', 'main_nocot': 'no-CoT', 'full_lora': 'full', 'main_alltok': 'all-tok',
         'main_alltok_s2': 'all-tok s2', 'mlp_alltok': 'MLP all-tok', 'main_e3': 'main e3', 'main_alltok_e3': 'all-tok e3',
         'main_alltok_e3_s2': 'all-tok e3 s2', 'mlp_alltok_e3': 'MLP all-tok e3'}
COL = {'correct': '#1baf7a', 'ordinal': '#eda100', 'detection': '#2a78d6', 'format': '#8a8a8a'}


def rate(xs) -> dict:
    xs = [bool(x) for x in xs]
    k, n = sum(xs), len(xs)
    return {'k': k, 'n': n, 'p': k / n if n else float('nan'), 'ci': bv.wilson(k, n)}


def paired(a: dict, b: dict) -> dict:
    """a, b: id -> bool. McNemar exact on the shared ids."""
    ids = sorted(set(a) & set(b))
    n10 = sum(a[i] and not b[i] for i in ids)
    n01 = sum(b[i] and not a[i] for i in ids)
    return {'n': len(ids), 'base_only': n10, 'cond_only': n01, 'p': bv.mcnemar_exact(n10, n01)}


def load_bv(run: Path, cond: str, name: str = 'babyvision.jsonl') -> list[dict] | None:
    p = run / 'eval' / cond / name
    return read_jsonl(p) if p.exists() else None


def load_rep(run: Path, cond: str, name: str = 'replica.jsonl') -> list[dict] | None:
    p = run / 'eval' / cond / name
    return read_jsonl(p) if p.exists() else None


# ----------------------------------------------------------------------------- diagnosis

def classify_rep(R: list[dict], odd_task: str, mark_task: str) -> dict[str, str]:
    by = {(r['rep'], r['task']): r for r in R}
    out = {}
    for (rep, task), r in by.items():
        if task != odd_task:
            continue
        m = by.get((rep, mark_task))
        p = by.get((rep, 'odd_pt'))
        cell = None if p is None or p.get('cell') is None else (p['cell'][0] - 1, p['cell'][1] - 1)
        out[rep] = bv.diagnose(r['rc'], r['format'], None if m is None else m['rc'], tuple(r['truth']), cell)
    return out


def classify_bv(B: list[dict], ann: dict, cases: dict, odd_task: str) -> dict[str, str]:
    """Per BabyVision item: error class of the official (or odd_rowcol) answer; the marked answer is the red_rowcol
    (or red_cot) answer on the recoloured TARGET cell (pick 0), the point evidence the odd_point answer."""
    by = {r['id']: r for r in B}
    out = {}
    mark_task = 'red_cot' if odd_task == 'official' else 'red_rowcol'
    for tid, it in ann.items():
        r = by.get(f'{tid}|{odd_task}')
        if r is None:
            continue
        m = by.get(f'{tid}|{mark_task}|0')
        p = by.get(f'{tid}|odd_point')
        cell = None
        if p is not None and p.get('coords') is not None:
            cell = bv.cell_of_point(p['coords'], it['rows'], it['cols'], it['tight_boxes'], tuple(cases[f'{tid}|odd_point']['canvas']))
        fmt = r.get('format')
        if fmt is None and 'format' not in r:     # 7.2-style record
            fmt = bv.format_class(r['text'], r['finish_reason'], r['rc'])
        truth = tuple(cases[f'{tid}|{odd_task}']['truth']['rc'])
        out[str(tid)] = bv.diagnose(r['rc'], fmt, None if m is None else m.get('rc'), truth, cell)
    return out


def signed_errors(R: list[dict], task: str) -> dict:
    """Row / column signed error of correct-format answers, by density bin."""
    out = {}
    for dens in ('<1', '1-2', '>=2'):
        rr = [r for r in R if r['task'] == task and r['density'] == dens and r['rc'] is not None]
        out[dens] = {'row': [r['rc'][0] - r['truth'][0] for r in rr], 'col': [r['rc'][1] - r['truth'][1] for r in rr]}
    return out


# ----------------------------------------------------------------------------- per-condition summary

def bv_summary(B: list[dict], base: list[dict] | None) -> dict:
    out = {}
    for task in ('official', 'official_dims', 'odd_rowcol', 'odd_point', 'red_point', 'red_rowcol', 'red_cot'):
        rr = [r for r in B if r['task'] == task]
        if not rr:
            continue
        key = 'hit' if task.endswith('point') else 'correct'
        s = rate(r[key] for r in rr)
        if not task.endswith('point'):
            s['format'] = {k: sum((r.get('format') or bv.format_class(r['text'], r['finish_reason'], r.get('rc'))) == k for r in rr)
                           for k in ('truncated', 'cannot_see', 'unparsed')}
            s['row'] = rate(r['row_ok'] for r in rr) if 'row_ok' in rr[0] else None
        if base is not None:
            bb = {r['id']: bool(r[key]) for r in base if r['task'] == task}
            s['paired'] = paired(bb, {r['id']: bool(r[key]) for r in rr})
        out[task] = s
    return out


def rep_summary(R: list[dict], base: list[dict] | None) -> dict:
    out = {}
    for task in bv.REP_TASKS if hasattr(bv, 'REP_TASKS') else ('odd_rc', 'mark_rc', 'odd_pt', 'odd_cot', 'mark_cot'):
        key = 'cell_hit' if task == 'odd_pt' else 'correct'
        rr = [r for r in R if r['task'] == task]
        if not rr:
            continue
        s = {'all': rate(r[key] for r in rr)}
        for fam in bv.FAMILIES:
            s[fam] = rate(r[key] for r in rr if r['family'] == fam)
        for dens in ('<1', '1-2', '>=2'):
            s['dens' + dens] = rate(r[key] for r in rr if r['density'] == dens)
        s['trained_families'] = rate(r[key] for r in rr if r['family'] != 'number')
        if task in ('mark_rc', 'mark_cot', 'odd_rc', 'odd_cot'):
            s['ord_le6'] = rate(r['correct'] for r in rr if max(r['truth']) <= 6)
            s['ord_ge7'] = rate(r['correct'] for r in rr if max(r['truth']) >= 7)
        if task.endswith('cot'):
            ac = [bv.arithmetic_consistency(r['text'], r['rows'], r['cols']) for r in rr]
            s['mediated'] = sum(a is not None for a in ac)
            s['consistent'] = sum(bool(a and a['consistent']) for a in ac)
            # Errors of the written numbers for mediated answers: target point in the target cell?
        if task != 'odd_pt':
            s['format'] = {k: sum(r['format'] == k for r in rr) for k in ('truncated', 'cannot_see', 'unparsed')}
            s['mean_tokens'] = float(np.mean([r['n_tokens'] for r in rr]))
        if base is not None:
            bb = {r['id']: bool(r[key]) for r in base if r['task'] == task}
            s['paired'] = paired(bb, {r['id']: bool(r[key]) for r in rr})
            s['paired_number'] = paired({k: v for k, v in bb.items() if '_number_' in k},
                                        {r['id']: bool(r[key]) for r in rr if r['family'] == 'number'})
        out[task] = s
    return out


REP_TASKS = ('odd_rc', 'mark_rc', 'odd_pt', 'odd_cot', 'mark_cot')


def coord_summary(run: Path, cond: str) -> dict | None:
    p = run / 'eval' / cond / 'coord.jsonl'
    if not p.exists():
        return None
    rr = read_jsonl(p)
    out = {}
    for fam in sorted({r['family'] for r in rr}):
        x = [r for r in rr if r['family'] == fam]
        errs = [r['error'] for r in x if r.get('error') is not None]
        out[fam] = {'n': len(x), 'hit': rate(r.get('hit', False) for r in x),
                    'mean_error': float(np.mean(errs)) if errs else None, 'invalid': sum(r.get('error') is None for r in x)}
    return out


def general_summary(run: Path, cond: str, base_dir: Path) -> dict | None:
    p, t = run / 'eval' / cond / 'general.jsonl', run / 'eval' / cond / 'text.jsonl'
    if not p.exists() or not t.exists():
        return None
    G = read_jsonl(p)
    Gb = {r['id']: r for r in read_jsonl(base_dir / 'general.jsonl')}
    out = {}
    for bench in ('pope', 'textvqa', 'mmvp', 'gqa'):
        rr = [r for r in G if r['bench'] == bench]
        s = rate(r['correct'] for r in rr)
        s['paired'] = paired({r['id']: bool(Gb[r['id']]['correct']) for r in rr if r['id'] in Gb},
                             {r['id']: bool(r['correct']) for r in rr})
        s['identical_text'] = sum(r['text'] == Gb[r['id']]['text'] for r in rr if r['id'] in Gb)
        out[bench] = s
    T = read_jsonl(t)
    Tb = {r['q']: r for r in read_jsonl(base_dir / 'text.jsonl')}
    out['text'] = {'correct': sum(r['correct'] for r in T), 'n': len(T), 'identical': sum(r['text'] == Tb[r['q']]['text'] for r in T)}
    return out


def ordinal_summary(run: Path, cond: str) -> dict:
    out = {}
    p = run / 'eval' / cond / 'ordinal.jsonl'
    if p.exists():
        rr = read_jsonl(p)
        out['direct'] = rate(r['correct'] for r in rr)
        out['direct_by'] = {f: rate(r['correct'] for r in rr if r['family'] == f) for f in ('grid06', 'grid10', 'grid20')}
    p = run / 'eval' / cond / 'ordinal_cot.jsonl'
    if p.exists():
        rr = read_jsonl(p)
        out['cot'] = rate(r['correct'] for r in rr)
        out['cot_by'] = {f: rate(r['correct'] for r in rr if r['family'] == f) for f in ('grid06', 'grid10', 'grid20')}
    return out


def gate_contrast(run: Path, cond: str) -> dict | None:
    p = run / 'eval' / cond / 'gatestat.json'
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    out = {}
    for l in sorted({r['layer'] for r in rows}):
        d = {r['cls']: r for r in rows if r['layer'] == l}
        if 1 in d and 3 in d:
            out[l] = {'alpha_row_onset_minus_other': float(np.mean(np.array(d[1]['alpha']) - np.array(d[3]['alpha']))),
                      'beta_row_onset_minus_other': float(np.mean(np.array(d[1]['beta']) - np.array(d[3]['beta']))),
                      'alpha_other': float(np.mean(d[3]['alpha'])), 'beta_other': float(np.mean(d[3]['beta']))}
    return out


# ----------------------------------------------------------------------------- figures

def fig_examples(run: Path, figs: Path) -> None:
    reps = [json.loads(l) for l in (run / 'replicas' / 'eval.jsonl').read_text().splitlines()]
    from PIL import Image
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for j, fam in enumerate(bv.FAMILIES):
        r = next(x for x in reps if x['family'] == fam)
        for i, key in enumerate(('image', 'marked_image')):
            ax = axes[i, j]
            ax.imshow(Image.open(r[key]))
            ax.set_xticks([]); ax.set_yticks([])
            t = r['target']
            ttl = f"{fam}: {r['rows']}x{r['cols']}, {r['content']}" if i == 0 else f"marked (target row {t[0]+1}, col {t[1]+1})"
            ax.set_title(ttl, fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / 'replica_examples.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def fig_diagnosis(S: dict, figs: Path) -> None:
    D = S['diagnosis']
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={'width_ratios': [1.6, 1, 1.2]})
    ax = axes[0]
    groups = [('BV official', D['bv_official']), ('BV direct', D['bv_direct'])]
    for fam in bv.FAMILIES:
        groups.append((f'{fam} direct', D['rep_direct_by_family'][fam]))
    for fam in bv.FAMILIES:
        groups.append((f'{fam} reasoning', D['rep_cot_by_family'][fam]))
    bottom = np.zeros(len(groups))
    for cls in bv.ERROR_CLASSES:
        v = np.array([g[1].get(cls, 0) / max(1, sum(g[1].values())) for g in groups])
        ax.bar(range(len(groups)), v, bottom=bottom, color=COL[cls], label=cls)
        bottom += v
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g[0]}\n(n={sum(g[1].values())})" for g in groups], rotation=60, fontsize=8, ha='right')
    ax.set_ylabel('fraction of answers')
    ax.set_title('Error classes of the base model (find the difference)')
    ax.legend(fontsize=8, loc='upper right')
    ax = axes[1]
    dens = ('<1', '1-2', '>=2')
    for k, (task, lab, c) in enumerate((('mark_rc', 'marked, direct', '#eb6834'), ('mark_cot', 'marked, reasoning', '#2a78d6'),
                                       ('odd_rc', 'odd, direct', '#1baf7a'))):
        v = [S['conds']['base']['rep'][task]['dens' + d]['p'] for d in dens]
        ci = [S['conds']['base']['rep'][task]['dens' + d]['ci'] for d in dens]
        x = np.arange(3) + (k - 1) * 0.25
        ax.bar(x, v, 0.25, color=c, label=lab, yerr=np.abs(np.array(ci).T - np.array(v)), capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels([f'{d} token' for d in dens])
    ax.set_xlabel('smaller cell side (merged tokens of 32 px)')
    ax.set_ylabel('row/column accuracy (replicas)')
    ax.set_title('Accuracy vs cell size (base)')
    ax.legend(fontsize=8)
    ax = axes[2]
    se = D['signed_mark_rc']
    bins = np.arange(-6.5, 7.5, 1)
    for d, c in zip(dens, ('#e87ba4', '#eda100', '#008300')):
        e = np.array(se[d]['row'] + se[d]['col'])
        e = e[e != 0]
        ax.hist(np.clip(e, -6, 6), bins=bins, histtype='step', lw=2, color=c, label=f'{d} token (n wrong axes={len(e)})')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('signed ordinal error (answer - truth), wrong axes only, clipped at ±6')
    ax.set_ylabel('count')
    ax.set_title('Marked target, direct answers (base)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'diagnosis.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_main(S: dict, figs: Path, conds: list[str]) -> None:
    panels = [('BV official (16)', 'bv', 'official'), ('BV official + stated grid size (16)', 'bv', 'official_dims'),
              ('BV marked, reasoning (96)', 'bv', 'red_cot'),
              ('BV marked, direct (96)', 'bv', 'red_rowcol'), ('BV odd, direct (16)', 'bv', 'odd_rowcol'),
              ('replica odd, reasoning (200)', 'rep', 'odd_cot'), ('replica marked, reasoning (200)', 'rep', 'mark_cot'),
              ('replica odd, direct (200)', 'rep', 'odd_rc'), ('replica marked, direct (200)', 'rep', 'mark_rc'),
              ('replica odd point, cell hit (200)', 'rep', 'odd_pt'), ('BV recoloured cell point hit (96)', 'bv', 'red_point')]
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(conds)))
    for ax, (title, part, task) in zip(axes.flat, panels):
        for i, c in enumerate(conds):
            d = S['conds'][c].get(part, {}).get(task)
            if d is None:
                continue
            s = d['all'] if part == 'rep' else d
            ax.bar(i, s['p'], color=cols[i], yerr=[[s['p'] - s['ci'][0]], [s['ci'][1] - s['p']]], capsize=3)
            pv = d.get('paired', {}).get('p')
            if c != 'base' and pv is not None and pv < 0.05:
                ax.text(i, s['ci'][1] + 0.02, '*', ha='center')
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([SHORT[c] for c in conds], rotation=30, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis('off')
    fig.text(0.5, -0.01, 'Bars: accuracy with Wilson 95% intervals; * = exact McNemar p < 0.05 vs base on the same items.', ha='center')
    fig.tight_layout()
    fig.savefig(figs / 'main_results.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_shift(S: dict, figs: Path, conds: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, key, title in ((axes[0], 'classes_cot', 'Replica odd, reasoning prompt'), (axes[1], 'classes_direct', 'Replica odd, direct prompt')):
        bottom = np.zeros(len(conds))
        for cls in bv.ERROR_CLASSES:
            v = np.array([S['conds'][c].get(key, {}).get(cls, 0) / max(1, sum(S['conds'][c].get(key, {}).values())) for c in conds])
            ax.bar(range(len(conds)), v, bottom=bottom, color=COL[cls], label=cls)
            bottom += v
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([SHORT[c] for c in conds], rotation=30)
        ax.set_ylabel('fraction of 200 answers')
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'error_shift.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_clock(S: dict, figs: Path, conds: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(conds)))
    for ax, key, title in ((axes[0], 'sweep_h|target', 'canvas x: target image tokens'), (axes[1], 'sweep_h|q0', 'canvas x: q0')):
        for i, c in enumerate(conds):
            if c not in S['clock']:
                continue
            d = S['clock'][c][key]
            ls = sorted(int(k) for k in d)
            ax.plot(ls, [max(-1, d[l] if l in d else d[str(l)]) for l in ls], 'o-', color=cols[i], label=SHORT[c], ms=3)
        ax.axvline(20, color='k', ls='--', lw=0.6)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_xlabel('decoder layer')
        ax.set_ylabel('periodic-code CV $R^2$ (clipped at -1)')
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'clock_after_training.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--run72', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--figures', required=True)
    a = p.parse_args()
    run, figs, feat = Path(a.run), Path(a.figures), Path(a.features)
    figs.mkdir(parents=True, exist_ok=True)
    ann = {it['taskId']: it for it in json.loads(Path(run / 'bv_annotation.json').read_text())['items']} \
        if (run / 'bv_annotation.json').exists() else None
    cases = {c['id']: c for c in (json.loads(l) for l in (run / 'bv_cases.jsonl').read_text().splitlines())}
    S = {'conds': {}, 'missing': []}
    Bb, Rb = load_bv(run, 'base'), load_rep(run, 'base')
    base_dir = run / 'eval' / 'base'
    conds = []
    for c in COND_ORDER:
        B, R = load_bv(run, c), load_rep(run, c)
        if B is None or R is None:
            S['missing'].append(c)
            continue
        conds.append(c)
        e = {'bv': bv_summary(B, None if c == 'base' else Bb), 'rep': rep_summary(R, None if c == 'base' else Rb),
             'coord': coord_summary(run, c), 'general': general_summary(run, c, base_dir), 'ordinal': ordinal_summary(run, c),
             'gate': gate_contrast(run, c)}
        e['classes_cot'] = _count(classify_rep(R, 'odd_cot', 'mark_cot'))
        e['classes_direct'] = _count(classify_rep(R, 'odd_rc', 'mark_rc'))
        if ann is not None:
            e['bv_mediated'] = {}
            for t in ('official', 'official_dims', 'red_cot'):
                rr = [r for r in B if r['task'] == t]
                ac = [bv.arithmetic_consistency(r['text'], ann[r['taskId']]['rows'], ann[r['taskId']]['cols']) for r in rr]
                pin = 0
                for r, a_ in zip(rr, ac):
                    if a_ is not None:
                        it = ann[r['taskId']]
                        cell = bv.cell_of_point(a_['point'], it['rows'], it['cols'], it['tight_boxes'], tuple(cases[r['id']]['canvas']))
                        pin += cell is not None and [cell[0] + 1, cell[1] + 1] == cases[r['id']]['truth']['rc']
                e['bv_mediated'][t] = {'n': len(rr), 'mediated': sum(a_ is not None for a_ in ac),
                                       'consistent': sum(bool(a_ and a_['consistent']) for a_ in ac), 'point_in_target': pin}
            e['bv_classes_official'] = classify_bv(B, ann, cases, 'official')
            e['bv_classes_direct'] = classify_bv(B, ann, cases, 'odd_rowcol')
        p_ad = run / 'adapters' / f'{c}.pt'
        if p_ad.exists():
            import torch
            st = torch.load(p_ad, map_location='cpu', weights_only=False)
            e['adapter'] = {'n_params': st['n_params'], 'train_seconds': st['train_seconds'], 'n_train': st['n_train'],
                            'n_steps': st['n_steps'], 'gpu': st['gpu']}
        S['conds'][c] = e
    # Diagnosis (base).
    if 'base' in S['conds']:
        cb = classify_rep(Rb, 'odd_cot', 'mark_cot')
        cd = classify_rep(Rb, 'odd_rc', 'mark_rc')
        fam_of = {r['rep']: r['family'] for r in Rb}
        dens_of = {r['rep']: r['density'] for r in Rb}
        S['diagnosis'] = {
            'rep_cot_by_family': {f: _count({k: v for k, v in cb.items() if fam_of[k] == f}) for f in bv.FAMILIES},
            'rep_direct_by_family': {f: _count({k: v for k, v in cd.items() if fam_of[k] == f}) for f in bv.FAMILIES},
            'rep_cot_by_density': {d: _count({k: v for k, v in cb.items() if dens_of[k] == d}) for d in ('<1', '1-2', '>=2')},
            'rep_direct_by_density': {d: _count({k: v for k, v in cd.items() if dens_of[k] == d}) for d in ('<1', '1-2', '>=2')},
            'signed_mark_rc': signed_errors(Rb, 'mark_rc'), 'signed_mark_cot': signed_errors(Rb, 'mark_cot'),
            'bv_official': _count(S['conds']['base'].get('bv_classes_official', {})),
            'bv_direct': _count(S['conds']['base'].get('bv_classes_direct', {})),
        }
        # Resolution check (2x upscaling, base only).
        Bu, Ru = load_bv(run, 'base', 'babyvision_up2.jsonl'), load_rep(run, 'base', 'replica_up2.jsonl')
        if Bu is not None and Ru is not None:
            S['diagnosis']['up2'] = {
                'bv': {t: {'x1': rate(r['correct'] for r in Bb if r['task'] == t), 'x2': rate(r['correct'] for r in Bu if r['task'] == t),
                           'paired': paired({r['id']: r['correct'] for r in Bb if r['task'] == t}, {r['id']: r['correct'] for r in Bu if r['task'] == t})}
                       for t in ('official', 'odd_rowcol', 'red_rowcol')},
                'rep': {t: {d: {'x1': rate(r['correct'] for r in Rb if r['task'] == t and r['density'] == d),
                                'x2': rate(r['correct'] for r in Ru if r['task'] == t and r['density'] == d)}
                            for d in ('<1', '1-2', '>=2', 'all')} for t in ('odd_rc', 'mark_rc')}}
            for t in ('odd_rc', 'mark_rc'):
                for d in ('all',):
                    S['diagnosis']['up2']['rep'][t][d] = {'x1': rate(r['correct'] for r in Rb if r['task'] == t),
                                                         'x2': rate(r['correct'] for r in Ru if r['task'] == t),
                                                         'paired': paired({r['id']: r['correct'] for r in Rb if r['task'] == t},
                                                                          {r['id']: r['correct'] for r in Ru if r['task'] == t})}
        sp = base_dir / 'spotcheck.json'
        if sp.exists():
            S['spotcheck'] = {k: v for k, v in json.loads(sp.read_text()).items() if k != 'items'}
    # Representation: periodic code (7.2 probe).
    S['clock'] = {c: v for c in conds if (v := cla.clock_curves(feat, c)) is not None}
    (run / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    fig_examples(run, figs)
    fig_flow(figs)
    if 'main_alltok_e3' in conds:
        fig_by_ordinal(run, figs)
    if 'diagnosis' in S:
        fig_diagnosis(S, figs)
    fig_conds = [c for c in ('base', 'main', 'main_e3', 'main_alltok_e3', 'main_alltok_e3_s2', 'mlp_alltok_e3', 'gate_only',
                             'vit_only', 'full_lora') if c in conds]
    if len(conds) > 1:
        fig_main(S, figs, fig_conds)
        fig_shift(S, figs, fig_conds)
    if S['clock']:
        fig_clock(S, figs, [c for c in conds if c in S['clock']])
    print('conditions', conds, 'missing', S['missing'], flush=True)


def _count(d: dict) -> dict:
    out = {k: 0 for k in bv.ERROR_CLASSES}
    for v in d.values():
        out[v] += 1
    return out



def fig_flow(figs: Path) -> None:
    """Schematic (not data): the base-model path from image to row/column answer and the path taught by 7.4."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(17, 6.2))
    ax.set_xlim(0, 17); ax.set_ylim(0, 6.2); ax.axis('off')

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=color, ec='#222222', lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9, wrap=True)

    def arrow(x0, y0, x1, y1, color='#222222'):
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', color=color, lw=1.4))
    box(0.1, 2.6, 2.5, 1.0, 'image\n(grid of entities;\n0.6-3 merged tokens per cell)', '#eeeeee')
    box(3.0, 2.6, 2.2, 1.0, 'ViT + merger\n(entity features per\n32x32-px token)', '#dbe9f6')
    box(5.8, 4.4, 3.2, 1.2, 'decoder L1-15 at image tokens:\nDeltaNet accumulators count\nentity rows / columns (L6-11)', '#fde7d3')
    box(5.8, 0.6, 3.2, 1.2, 'decoder L1-4 MLP at image tokens:\nposition code that L20-24 heads\nread as coordinate digits', '#d9f2e6')
    box(9.5, 4.4, 3.3, 1.2, 'base path: answer copies the count\n(exact for <= 6 entities,\nundercounts sub-token rows)', '#fde7d3')
    box(9.5, 0.6, 3.3, 1.2, 'taught path: write target point\nand grid extent (0-1000), then\n(x - X0) / ((X1 - X0) / C)', '#d9f2e6')
    box(13.6, 2.6, 3.0, 1.0, 'row, column\n("Final answer: (r,c)")', '#eeeeee')
    arrow(2.65, 3.1, 3.0, 3.1); arrow(5.25, 3.3, 5.8, 4.8); arrow(5.25, 2.9, 5.8, 1.2)
    arrow(9.05, 5.0, 9.5, 5.0, '#eb6834'); arrow(9.05, 1.2, 9.5, 1.2, '#1baf7a')
    arrow(12.85, 5.0, 13.6, 3.4, '#eb6834'); arrow(12.85, 1.2, 13.6, 2.8, '#1baf7a')
    ax.text(10.5, 3.1, 'finding the odd entity (detection) precedes both paths\nand remains the main failure', ha='center', fontsize=9, style='italic')
    fig.savefig(figs / 'flow.png', dpi=140, bbox_inches='tight')
    plt.close(fig)



def fig_by_ordinal(run: Path, figs: Path) -> None:
    """Marked-target accuracy by the larger of the true row / column ordinal (replicas), base vs the headline adapter."""
    bins = [(1, 3), (4, 6), (7, 10), (11, 15), (16, 28)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles = {('base', 'mark_rc'): ('#eb6834', '--', 'base, direct'), ('base', 'mark_cot'): ('#eb6834', '-', 'base, reasoning'),
              ('main_alltok_e3', 'mark_rc'): ('#1baf7a', '--', 'adapter, direct'),
              ('main_alltok_e3', 'mark_cot'): ('#1baf7a', '-', 'adapter, reasoning')}
    for (c, t), (col, ls, lab) in styles.items():
        R = load_rep(run, c)
        ys, lo, hi = [], [], []
        for a, b in bins:
            rr = [r['correct'] for r in R if r['task'] == t and a <= max(r['truth']) <= b]
            s_ = rate(rr)
            ys.append(s_['p']); lo.append(max(0, s_['p'] - s_['ci'][0])); hi.append(max(0, s_['ci'][1] - s_['p']))
        ax.errorbar(range(len(bins)), ys, yerr=[lo, hi], color=col, ls=ls, marker='o', capsize=3, label=lab)
    n = [sum(1 for r in load_rep(run, 'base') if r['task'] == 'mark_rc' and a <= max(r['truth']) <= b) for a, b in bins]
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f'{a}-{b}\n(n={k})' for (a, b), k in zip(bins, n)])
    ax.set_xlabel('larger of the true row and column ordinal of the marked entity')
    ax.set_ylabel('row/column accuracy (Wilson 95%)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'accuracy_by_ordinal.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
