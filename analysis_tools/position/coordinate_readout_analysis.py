"""Metrics of experiment 3.5 (coordinate readout path) computed from the stage artifacts.

Reads the run directory written by `coordinate_readout_run.py` and writes `results.json`
(all metrics with Wilson / bootstrap intervals) and `tables.md`. Figures are drawn by
`coordinate_readout_figures.py` from the same loaded data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

FULL_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
Z = 1.959964


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return [centre - half, centre + half]


def prop(k: int, n: int) -> dict:
    return {'k': int(k), 'n': int(n), 'p': k / n if n else float('nan'), 'ci': wilson(k, n)}


def bootstrap_mean(values, n_boot: int = 2000, seed: int = 0) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return {'mean': float('nan'), 'ci': [float('nan'), float('nan')], 'n': 0}
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return {'mean': float(v.mean()), 'ci': np.percentile(boots, [2.5, 97.5]).tolist(), 'n': int(len(v)),
            'median': float(np.median(v))}


def load(run: Path) -> dict:
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    return {'cases': cases, 'baseline': read_jsonl(run / 'baseline' / 'records.jsonl'),
            'spot': read_jsonl(run / 'baseline' / 'spotcheck.jsonl'),
            'attention': [r for r in read_jsonl(run / 'attention' / 'records.jsonl') if r.get('status') == 'ok'],
            'attention_failed': [r for r in read_jsonl(run / 'attention' / 'records.jsonl') if r.get('status') != 'ok'],
            'interventions': read_jsonl(run / 'interventions' / 'records.jsonl'),
            'failure': read_jsonl(run / 'failure' / 'records.jsonl'),
            'failure_slot': read_jsonl(run / 'failure' / 'records_slot.jsonl'),
            'ablation': read_jsonl(run / 'ablation' / 'records.jsonl')}


# ----------------------------------------------------------------------------- C0

def c0_metrics(d: dict) -> dict:
    groups = defaultdict(list)
    for r in d['baseline']:
        c = d['cases'][r['id']]
        level = {'grid': c.get('diameter'), 'trans': c.get('diameter'), 'size': c.get('canvas'), 't6': c.get('grid')}[c['set']]
        groups[(c['set'], level, r['format'])].append((c, r))
    out = {}
    for (st, level, fmt), rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        s = [r['score'] for _, r in rows]
        n = len(s)
        entry = {'set': st, 'level': level, 'format': fmt, 'n': n, 'valid': prop(sum(x['valid'] for x in s), n),
                 'truncated': sum(r['finish_reason'] != 'eos' for _, r in rows)}
        if st == 't6':
            entry['correct'] = prop(sum(x['correct'] for x in s), n)
            entry['prior_text_agree'] = prop(sum(r['prior_agree'] for _, r in rows), n)
            if fmt != 'rowcol':
                off = np.array([x['offset_cells'] for x in s if x['valid']])
                entry['mean_offset_cells'] = off.mean(0).tolist()
                tc = np.array([1000 * np.asarray(r['truth']['target_center']) / np.asarray(r['truth']['canvas'])
                               for _, r in rows if r['score']['valid']])
                pc = np.array([x['coords'][:2] if fmt == 'point' else
                               ((np.asarray(x['coords'][:2]) + np.asarray(x['coords'][2:])) / 2).tolist()
                               for x in s if x['valid']])
                entry['fit'] = [np.polyfit(tc[:, i], pc[:, i], 1).tolist() for i in (0, 1)]
            else:
                wrong = [x for x in s if x['valid'] and not x['correct']]
                entry['rowcol_errors'] = {'n': len(wrong), 'col_only': sum(x['d_row'] == 0 for x in wrong),
                                          'col_plus1': sum(x['d_row'] == 0 and x['d_col'] == 1 for x in wrong),
                                          'row_only': sum(x['d_col'] == 0 for x in wrong)}
        else:
            entry['hit'] = prop(sum(x.get('hit', False) for x in s), n)
            err = [x['error'] for x in s if x['valid']]
            entry['error'] = bootstrap_mean(err)
            if fmt == 'bbox':
                entry['iou50'] = prop(sum(x.get('correct', False) for x in s), n)
            tc = np.array([1000 * np.asarray(r['truth']['center']) / np.asarray(r['truth']['canvas'])
                           for _, r in rows if r['score']['valid']])
            pc = np.array([x['center'] for x in s if x['valid']])
            entry['fit'] = [np.polyfit(tc[:, i], pc[:, i], 1).tolist() for i in (0, 1)]
            entry['mean_offset'] = (pc - tc).mean(0).tolist()
            entry['error_pixels'] = bootstrap_mean([x['error'] * c['canvas'] / 1000 for (c, _), x in zip(rows, s) if x['valid']])
        out[f'{st}|{level}|{fmt}'] = entry
    spot = d['spot']
    out['spotcheck'] = {'n': len(spot), 'text_equal': sum(x['text_equal'] for x in spot),
                        'answer_equal': sum(x['answer_equal'] for x in spot)}
    return out


# ----------------------------------------------------------------------------- C1

def query_index(rec: dict, kind: str) -> list[int]:
    """Indices into the query list: 'first' = first digit of every coordinate value, 'v0'/'v1' etc."""
    names = rec['query_names']
    if kind == 'first':
        return [i for i, n in enumerate(names) if n.endswith('d0') and n.startswith('v')]
    return [names.index(kind)] if kind in names else []


def c1_groups(d: dict) -> dict:
    """Records grouped by task: disc point, disc bbox, T6 row-column, T6 point."""
    groups = defaultdict(list)
    for r in d['attention']:
        task = ('disc_' if r['set'] in ('grid', 'trans') else f"t6g{r['grid'][0] // 2:02d}_") + r['format']
        groups[task].append(r)
    return groups


def select_heads(records: list[dict], seed: int = 0, top: int = 10) -> dict:
    """Candidate pointing heads: rank by mean log E at first-digit queries on a selection half of the
    correct samples (split by image), keep heads whose argmax lies in the target in >= half the cases."""
    correct = [r for r in records if r['correct']]
    keys = sorted({r['id'].rsplit('_', 1)[0] for r in correct})
    rng = np.random.default_rng(seed)
    sel_keys = set(rng.permutation(keys)[:len(keys) // 2].tolist())
    sel = [r for r in correct if r['id'].rsplit('_', 1)[0] in sel_keys]
    val = [r for r in correct if r['id'].rsplit('_', 1)[0] not in sel_keys]

    def stats(rs):
        logE, hit = [], []
        for r in rs:
            q = query_index(r, 'first')
            E = np.asarray(r['E'])[:, :, q]
            logE.append(np.log(np.maximum(E, 1e-6)).mean(-1))
            hit.append(np.asarray(r['hit'])[:, :, q].mean(-1))
        return np.mean(logE, 0), np.mean(hit, 0)
    if not sel or not val:
        return {}
    s_logE, s_hit = stats(sel)
    v_logE, v_hit = stats(val)
    order = np.argsort(-s_logE.ravel())
    heads = []
    for flat in order:
        li, h = divmod(int(flat), s_logE.shape[1])
        if s_hit[li, h] >= 0.5:
            heads.append((li, h))
        if len(heads) == top:
            break
    return {'heads': [[FULL_LAYERS[li], h] for li, h in heads],
            'selection': [{'layer': FULL_LAYERS[li], 'head': h, 'logE': float(s_logE[li, h]), 'hit': float(s_hit[li, h])}
                          for li, h in heads],
            'validation': [{'layer': FULL_LAYERS[li], 'head': h, 'logE': float(v_logE[li, h]), 'hit': float(v_hit[li, h])}
                           for li, h in heads],
            'n_selection': len(sel), 'n_validation': len(val),
            'logE_map_selection': s_logE.tolist(), 'hit_map_selection': s_hit.tolist()}


def c1_metrics(d: dict) -> dict:
    out = {}
    pos = positions(d)
    for task, recs in sorted(c1_groups(d).items()):
        entry = {'n': len(recs), 'n_correct': sum(r['correct'] for r in recs)}
        cand = select_heads(recs)
        entry['candidates'] = cand
        # Layer profile: per query type, mean over samples of the max-over-heads log E (correct samples).
        prof = {}
        for kind in ('q_end', 'think_end', 'last', 'first'):
            vals, mass = [], []
            for r in recs:
                if not r['correct']:
                    continue
                q = query_index(r, kind)
                E = np.log(np.maximum(np.asarray(r['E'])[:, :, q], 1e-6)).mean(-1)      # (8, heads)
                vals.append(E.max(1))
                mass.append(np.asarray(r['vis_mass'])[:, :, q].mean(-1).mean(1))
            if vals:
                prof[kind] = {'max_logE': np.mean(vals, 0).tolist(), 'vis_mass': np.mean(mass, 0).tolist()}
        entry['layer_profile'] = prof
        # Attribution share of target tokens per layer (rho), correct vs incorrect, first coordinate value.
        for tag, sub in (('correct', [r for r in recs if r['correct']]), ('incorrect', [r for r in recs if not r['correct']])):
            sub = [r for r in sub if np.isfinite(np.asarray(r['rho'], float)).all()]
            if sub:
                rho = np.array([np.asarray(r['rho']).mean(0) for r in sub])
                share = np.array([np.asarray(r['grad_share_image']).mean(0) for r in sub])
                entry[f'rho_{tag}'] = {'mean': rho.mean(0).tolist(), 'n': len(sub)}
                entry[f'image_share_{tag}'] = share.mean(0).tolist()
        # Candidate-head centroid against truth and output (x for value 0, y for value 1; rowcol: column/row).
        if cand.get('heads'):
            entry['centroid'] = centroid_analysis(recs, cand['heads'], pos)
        out[task] = entry
    return out


def positions(d: dict) -> dict:
    """id -> (output, truth) positions in 0-1000 units (x, y) for every valid baseline record."""
    out = {}
    for r in d['baseline']:
        c, s = d['cases'][r['id']], r['score']
        if not s.get('valid'):
            continue
        t = r['truth']
        size = np.asarray(t['canvas'], float)
        if c['set'] != 't6':
            out[r['id']] = (np.asarray(s['center']), 1000 * np.asarray(t['center']) / size)
        elif c['format'] == 'rowcol':
            def cell_center(rc):
                return 1000 * (np.asarray(t['grid_origin']) + (np.asarray(rc[::-1]) + 0.5) * t['cell']) / size
            out[r['id']] = (cell_center(s['pred_rc']), cell_center(t['target_rc']))
        else:
            out[r['id']] = (1000 * np.asarray(s['center_px']) / size, 1000 * np.asarray(t['target_center']) / size)
    return out


def centroid_analysis(recs: list[dict], heads: list[list[int]], pos: dict) -> dict:
    """Mean candidate-head attention centroid at the first digit of x (value 0) and y (value 1)."""
    rows = []
    for r in recs:
        q = query_index(r, 'first')
        if len(q) < 2:
            continue
        cen = np.asarray(r['centroid'])                     # (8, heads, Q, 2)
        li = [(FULL_LAYERS.index(l), h) for l, h in heads]
        if r['format'] == 'rowcol':
            qx, qy = q[1], q[0]                              # column value is second, row value first
        else:
            qx, qy = q[0], q[1]
        cx = np.mean([cen[a, b, qx, 0] for a, b in li])
        cy = np.mean([cen[a, b, qy, 1] for a, b in li])
        if r['id'] not in pos:
            continue
        out, truth = pos[r['id']]
        rows.append({'id': r['id'], 'correct': r['correct'], 'centroid': [float(cx), float(cy)],
                     'output': out.tolist(), 'truth': truth.tolist()})
    res = {'rows': rows}
    for tag in ('correct', 'incorrect'):
        sub = [x for x in rows if x['correct'] == (tag == 'correct')]
        if len(sub) < 3:
            continue
        c = np.array([x['centroid'] for x in sub]); o = np.array([x['output'] for x in sub]); t = np.array([x['truth'] for x in sub])
        res[tag] = {'n': len(sub),
                    'corr_centroid_output': [float(np.corrcoef(c[:, i], o[:, i])[0, 1]) for i in (0, 1)],
                    'corr_centroid_truth': [float(np.corrcoef(c[:, i], t[:, i])[0, 1]) for i in (0, 1)],
                    'dist_centroid_output': float(np.linalg.norm(c - o, axis=1).mean()),
                    'dist_centroid_truth': float(np.linalg.norm(c - t, axis=1).mean())}
    return res


# ----------------------------------------------------------------------------- C2

FAMILIES = ('s2shift', 's1shift', 's2swap', 's1swap_pe', 's1swap_rope', 's1swap_both', 'conflict_both', 's4perm',
            'c3content', 'real')


def displacement_weight(dy: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    d = np.asarray(d, float)
    w = float(dy @ d / (d @ d))
    return w, float(np.linalg.norm(dy - w * d) / np.linalg.norm(d))


def c2_metrics(d: dict) -> dict:
    per_family = defaultdict(lambda: defaultdict(list))   # family -> subset -> list of (id, w, e)
    gap = {'n': 0, 'identical': 0, 'diff': [], 'logit_diff': []}
    s1_center_shift, rope_shift, rope_scale = [], [], defaultdict(list)
    recog = defaultdict(list)
    for rec in d['interventions']:
        conds = {c['condition']: c for c in rec['conditions']}
        std = conds['standard']
        if std['center'] is None:
            continue
        subset = 'correct' if rec['baseline_correct'] else 'incorrect'
        g = conds['gap']
        if g['center'] is not None:
            gap['n'] += 1
            gap['identical'] += g['coords'] == std['coords']
            gap['diff'].append(float(np.linalg.norm(np.asarray(g['center']) - np.asarray(std['center']))))
            if g['first_digit_logits'] and std['first_digit_logits']:
                gap['logit_diff'].append(float(np.abs(np.asarray(g['first_digit_logits'][0]) -
                                                      np.asarray(std['first_digit_logits'][0])).max()))
        if conds.get('s1center', {}).get('center') is not None:
            s1_center_shift.append(float(np.linalg.norm(np.asarray(conds['s1center']['center']) - np.asarray(std['center']))))
        if conds.get('s1rope_shift', {}).get('center') is not None:
            rope_shift.append(float(np.linalg.norm(np.asarray(conds['s1rope_shift']['center']) - np.asarray(std['center']))))
        for name, c in conds.items():
            if name.startswith('s1rope_scale') and c['center'] is not None:
                rope_scale[name].append((np.asarray(c['center']) - np.asarray(std['center'])).tolist())
            if c.get('d') is None or c['center'] is None:
                continue
            fam = next((f for f in FAMILIES if name.startswith(f + '_')), None)
            if fam is None:
                continue
            ref = conds[c.get('ref', 'standard')]
            if ref['center'] is None:
                continue
            dy = np.asarray(c['center']) - np.asarray(ref['center'])
            w, e = displacement_weight(dy, c['d'])
            key = name.split('_')[-1]
            for sub in (subset, 'all'):
                per_family[fam][sub].append({'id': rec['id'], 'cond': name, 'w': w, 'e': e, 'dist': float(np.linalg.norm(c['d'])),
                                             'diameter': rec['diameter'], 'key': key})
        for name, v in rec.get('recognition', {}).items():
            recog[name.split('_')[0]].append(v)
    by_axis = defaultdict(list)
    for rec in d['interventions']:
        conds = {c['condition']: c for c in rec['conditions']}
        for name, c in conds.items():
            fam = next((f for f in FAMILIES if name.startswith(f + '_')), None)
            if fam is None or c.get('d') is None or c['center'] is None or conds[c.get('ref', 'standard')]['center'] is None:
                continue
            ax = 0 if abs(c['d'][0]) > 0 else 1
            w = (c['center'][ax] - conds[c.get('ref', 'standard')]['center'][ax]) / c['d'][ax]
            by_axis[(fam, 'x' if ax == 0 else 'y')].append(w)
    out_axis = {f'{fam}|{ax}': bootstrap_mean(v) for (fam, ax), v in sorted(by_axis.items())}
    out = {'weights_by_axis': out_axis, 'gap_consistency': {'n': gap['n'], 'identical': prop(gap['identical'], gap['n']),
                               'center_diff': bootstrap_mean(gap['diff']),
                               'first_digit_logit_maxdiff': bootstrap_mean(gap['logit_diff'])},
           's1_center_vs_standard': bootstrap_mean(s1_center_shift), 'vit_rope_shift_vs_standard': bootstrap_mean(rope_shift),
           'vit_rope_scale_displacement': {k: np.mean(v, 0).tolist() for k, v in rope_scale.items()},
           'recognition': {k: bootstrap_mean(v) for k, v in recog.items()}, 'families': {}}
    for fam, subs in per_family.items():
        out['families'][fam] = {}
        for sub, rows in subs.items():
            # Average per image first, then bootstrap over images.
            by_img = defaultdict(list)
            for x in rows:
                by_img[x['id']].append(x)
            w_img = [np.mean([x['w'] for x in v]) for v in by_img.values()]
            e_img = [np.mean([x['e'] for x in v]) for v in by_img.values()]
            entry = {'w': bootstrap_mean(w_img), 'e_perp': bootstrap_mean(e_img), 'n_conditions': len(rows)}
            by_dist = defaultdict(list)
            for x in rows:
                by_dist[round(x['dist'])].append(x['w'])
            entry['w_by_distance'] = {str(k): bootstrap_mean(v) for k, v in sorted(by_dist.items())}
            by_d = defaultdict(list)
            for x in rows:
                by_d[x['diameter']].append(x['w'])
            entry['w_by_diameter'] = {str(k): bootstrap_mean(v) for k, v in sorted(by_d.items())}
            out['families'][fam][sub] = entry
    return out


# ----------------------------------------------------------------------------- failure analysis

def fail_metrics(d: dict) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in d['failure'] + d['failure_slot']:
        groups[(r['grid'], r['format'])].append(r)
    for (grid, fmt), recs in sorted(groups.items()):
        per = defaultdict(list)
        for r in recs:
            conds = {c['condition']: c for c in r['conditions']}
            std = conds['standard']['score']
            if not std.get('valid'):
                continue
            for name, c in conds.items():
                s = c['score']
                if not s.get('valid'):
                    per[name].append(None)
                    continue
                if fmt == 'rowcol':
                    delta = [s['pred_rc'][1] - std['pred_rc'][1], s['pred_rc'][0] - std['pred_rc'][0]]
                else:
                    delta = (np.asarray(s['center_px']) - np.asarray(std['center_px'])) / 64
                    delta = delta.tolist()
                per[name].append({'d': delta, 'cells': c.get('cells')})
        entry = {}
        for name, vals in per.items():
            ok = [v for v in vals if v is not None]
            dd = np.array([v['d'] for v in ok]) if ok else np.zeros((0, 2))
            e = {'n': len(vals), 'invalid': len(vals) - len(ok),
                 'mean_dx_cells': bootstrap_mean(dd[:, 0]) if len(dd) else None,
                 'mean_dy_cells': bootstrap_mean(dd[:, 1]) if len(dd) else None,
                 'changed': prop(int((np.abs(dd) > (0 if fmt == 'rowcol' else 0.5)).any(1).sum()), len(ok)) if len(dd) else None}
            cells = [v['cells'] for v in ok if v['cells'] is not None]
            if cells:
                axis = 0 if any(k in name for k in ('right', 'left')) else 1
                follow = [v['d'][axis] / v['cells'] for v in ok]
                e['follow_ratio'] = bootstrap_mean(follow)
            entry[name] = e
        res = {'n': len(recs), 'conditions': entry}
        if fmt == 'rowcol':
            res['candidates'] = candidate_summary(recs)
        out[f'g{grid}|{fmt}'] = res
    return out


def candidate_summary(recs: list[dict]) -> dict:
    """Log-prob margins between the correct column/row and its +1 neighbour, by baseline outcome."""
    out = {}
    for axis, key, dkey in (('col', 'col_logprob', 'd_col'), ('row', 'row_logprob', 'd_row')):
        rows = []
        for r in recs:
            lp = r.get(key)
            s = r['baseline_score']
            if not lp or not s.get('valid'):
                continue
            t = r['truth']['target_rc'][1 if axis == 'col' else 0] + 1
            if str(t) not in lp or str(t + 1) not in lp:
                continue
            rows.append({'margin_plus1': lp[str(t + 1)] - lp[str(t)],
                         'margin_minus1': (lp[str(t - 1)] - lp[str(t)]) if str(t - 1) in lp else None,
                         'd': s[dkey], 'truth': t})
        groups = {'correct': [x for x in rows if x['d'] == 0], 'plus1': [x for x in rows if x['d'] == 1],
                  'other_error': [x for x in rows if x['d'] not in (0, 1)]}
        out[axis] = {g: {'n': len(v), 'margin_plus1': bootstrap_mean([x['margin_plus1'] for x in v]),
                         'margin_minus1': bootstrap_mean([x['margin_minus1'] for x in v if x['margin_minus1'] is not None])}
                     for g, v in groups.items()}
        out[axis]['rows'] = rows
    return out


# ----------------------------------------------------------------------------- C4

def c4_metrics(d: dict) -> dict:
    """Mean ablation of candidate pointing heads versus random heads from the same layers."""
    recs = d['ablation']
    if not recs:
        return {}
    out = {}
    sets = [c['set'] for c in recs[0]['conditions'] if c['set'] != 'none']
    per_set = {}
    for name in sets:
        derr, drec, moved, hit = [], [], [], []
        for r in recs:
            cs = {c['set']: c for c in r['conditions']}
            b, c = cs['none'], cs[name]
            if not (b['score']['valid'] and c['score']['valid']):
                continue
            derr.append(c['score']['error'] - b['score']['error'])
            drec.append(c['recognition'] - b['recognition'])
            moved.append(float(np.linalg.norm(np.asarray(c['score']['center']) - np.asarray(b['score']['center']))))
            hit.append(c['score']['hit'])
        per_set[name] = {'d_error': bootstrap_mean(derr), 'd_recognition': bootstrap_mean(drec),
                         'displacement': bootstrap_mean(moved), 'hit_rate': float(np.mean(hit)), 'n': len(derr)}
    out['candidates'] = per_set['candidates']
    rnd = [v for k, v in per_set.items() if k.startswith('random')]
    out['random'] = {'d_error_means': [v['d_error']['mean'] for v in rnd],
                     'displacement_means': [v['displacement']['mean'] for v in rnd],
                     'd_recognition_means': [v['d_recognition']['mean'] for v in rnd],
                     'hit_rates': [v['hit_rate'] for v in rnd]}
    base = [r['conditions'][0] for r in recs if r['conditions'][0]['score']['valid']]
    out['baseline_hit_rate'] = float(np.mean([c['score']['hit'] for c in base]))
    out['baseline_error'] = bootstrap_mean([c['score']['error'] for c in base])
    out['n_images'] = len(recs)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    d = load(args.run)
    results = {'c0': c0_metrics(d), 'c1': c1_metrics(d), 'c2': c2_metrics(d), 'failure': fail_metrics(d), 'c4': c4_metrics(d),
               'attention_failed': [r['id'] for r in d['attention_failed']]}
    (args.run / 'results.json').write_text(json.dumps(results, indent=1, default=float))
    print('wrote', args.run / 'results.json')


if __name__ == '__main__':
    main()
