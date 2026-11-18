"""Experiment 2.7 round 4: CPU statistics and figures (summary.json, figures/search_pipeline/*.png).

Inputs under --output: zeta.npz, records/*.npz, rows.jsonl (Qwen3.5 main), probe.npz (Qwen3.5 probe),
vl/{readers.json, rows.jsonl, records.npz, selection.json} (Qwen3-VL); round-3 circuit/records.npz and
circuit/selection.json under --round3.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.representation import search_circuit_analysis as A
from analysis_tools.representation import search_pipeline as sp

CLS = ('comparison', 'detector')
# gate-decomposition component order of search_pipeline_run.neuron_decomposition
COMP33 = (['input'] + [('fa' if m % 4 == 0 else 'dn') + str(m) for m in range(1, 16)]
          + [f'mlp{m}' for m in range(1, 15)] + ['h6_own', 'h6_earlier', 'h6_rest'])
GROUPS = {
    'decoder input': ['input'],
    'DeltaNet 1-7': ['dn1', 'dn2', 'dn3', 'dn5', 'dn6', 'dn7'],
    'MLP 1-7': [f'mlp{m}' for m in range(1, 8)],
    'FA 4': ['fa4'],
    'L8 H6 own cell': ['h6_own'],
    'L8 H6 earlier cells': ['h6_earlier'],
    'L8 H6 text': ['h6_rest'],
    'L8 other heads': ['fa8'],
    'DeltaNet 9-15': ['dn9', 'dn10', 'dn11', 'dn13', 'dn14', 'dn15'],
    'FA 12': ['fa12'],
    'MLP 8-14': [f'mlp{m}' for m in range(8, 15)],
}
ROLES5 = ('odd', 'corr', 'same', 'blk', 'ref')


def ci(x):
    return sp.mean_ci(x)


# ----------------------------------------------------------------------------- Qwen3.5 interventions

def intervention_table(rows: list[dict]) -> dict:
    """cond -> class -> {reader, reader_ci, drop, drop_ci, top1, n, s}; drop is paired against the odd 'none'."""
    base = {r['id']: r for r in rows if r['cond'] == 'none'}
    out = defaultdict(dict)
    by = defaultdict(list)
    for r in rows:
        by[(r['cond'], r['cls'])].append(r)
    for (c, k), lst in by.items():
        rd = [r['reader'] for r in lst]
        dr = [base[r['id']]['reader'] - r['reader'] for r in lst]
        m, h = ci(rd)
        dm, dh = ci(dr)
        out[c][k] = {'reader': m, 'reader_ci': h, 'drop': dm, 'drop_ci': dh, 'n': len(lst),
                     'top1': float(np.mean([r['top1'] for r in lst])), 's': float(np.mean([r['s'] for r in lst])),
                     'reader_col': float(np.mean([r['reader_col'] for r in lst]))}
    return dict(out)


def redirect_table(rows: list[dict]) -> dict:
    """Fraction of row / column argmax digits equal to the target's or to the redirect cell's."""
    out = defaultdict(dict)
    by = defaultdict(list)
    for r in rows:
        if r['cond'] == 'none' or r['cond'].startswith(('redir', 'r6_', 'fa16', 'fa_all')):
            by[(r['cond'], r['cls'])].append({**r, 'ecell': r.get('ecell', sp.redirect_cell(r['tcell']))})
    for (c, k), lst in by.items():
        tr = np.array([r['arg_row'] == r['tcell'] // 6 + 1 for r in lst])
        er = np.array([r['arg_row'] == r['ecell'] // 6 + 1 for r in lst])
        tc = np.array([r['arg_col'] == r['tcell'] % 6 + 1 for r in lst])
        ec = np.array([r['arg_col'] == r['ecell'] % 6 + 1 for r in lst])
        out[c][k] = {'row_target': tr.mean(), 'row_redirect': er.mean(), 'col_target': tc.mean(),
                     'col_redirect': ec.mean(), 'n': len(lst),
                     'row_redirect_wilson': A.wil(er.sum(), len(lst)), 'col_redirect_wilson': A.wil(ec.sum(), len(lst))}
    return dict(out)


# ----------------------------------------------------------------------------- records

def load_records(output: Path) -> list[dict]:
    recs = []
    for p in sorted((output / 'records').glob('*.npz')):
        z = np.load(p)
        recs.append({k: z[k] for k in z.files})
    return recs


def h6_stats(recs, rows_by_id, cls_of) -> dict:
    """L8 H6 output at the target (after W_O, mean over the 4 tokens)."""
    res = defaultdict(lambda: defaultdict(list))
    for r in recs:
        i = str(r['id'])
        c = cls_of[i]
        h = r['h6'].astype(np.float64)                 # odd, corr, same, blk, ref
        pt = r['h6_parts'].astype(np.float64)          # (3 roles, own/earlier/rest/earlier_uniform/own_uniform, d)
        res[c]['cos_odd_corr'].append(sp.cosine(h[0], h[1]))
        res[c]['cos_odd_same'].append(sp.cosine(h[0], h[2]))
        res[c]['cos_odd_blk'].append(sp.cosine(h[0], h[3]))
        res[c]['rel_odd_corr'].append(np.linalg.norm(h[0] - h[1]) / np.linalg.norm(h[1]))
        res[c]['rel_odd_same'].append(np.linalg.norm(h[0] - h[2]) / np.linalg.norm(h[1]))
        res[c]['rel_odd_blk'].append(np.linalg.norm(h[0] - h[3]) / np.linalg.norm(h[1]))
        res[c]['cos_earlier_uniform'].append(sp.cosine(pt[0, 1], pt[0, 3]))
        res[c]['share_earlier'].append(np.linalg.norm(pt[0, 1]) / np.linalg.norm(h[0]))
        m = r['l8_mass'][:, 6]                          # (3 roles, 37)
        t = int(rows_by_id[i]['tcell'])
        res[c]['mass_own_odd'].append(m[0, t]); res[c]['mass_own_corr'].append(m[1, t]); res[c]['mass_own_same'].append(m[2, t])
        res[c]['mass_earlier_odd'].append(m[0, :t].sum()); res[c]['mass_earlier_corr'].append(m[1, :t].sum())
        res[c]['mass_earlier_same'].append(m[2, :t].sum())
        res[c]['mass_text_odd'].append(m[0, 36])
        e = m[0, :t] / max(m[0, :t].sum(), 1e-9)
        res[c]['entropy_ratio'].append(float(-(e * np.log(e + 1e-12)).sum() / np.log(max(t, 2))))
    out = {}
    for c, d in res.items():
        out[c] = {k: ci(v) for k, v in d.items()}
        out[c]['n'] = len(d['cos_odd_corr'])
    return out


def h6_distance_profile(recs, rows_by_id) -> dict:
    """Mean L8 H6 mass from the target query on an earlier cell as a function of raster distance (odd images)."""
    prof = defaultdict(list)
    for r in recs:
        t = int(rows_by_id[str(r['id'])]['tcell'])
        m = r['l8_mass'][0, 6]
        for e in range(t):
            prof[t - e].append(m[e])
    return {int(k): float(np.mean(v)) for k, v in sorted(prof.items())}


def h6_dimensionality(recs) -> dict:
    """Effective dimensionality of the L8 H6 output across corrupted images and how well the pair identity of the
    majority entity is decoded from it (nearest class mean, leave-one-out)."""
    X = np.stack([r['h6'][1].astype(np.float64) for r in recs])
    return {'participation_ratio_corr': sp.participation_ratio(X), 'n': len(X)}


def l8_key_stats(recs, rows_by_id, cls_of) -> dict:
    """Key-side decomposition of L8 H6 logits from the target query (odd images): mean contribution of each
    component group to the logit on earlier cells minus the logit on the own cell; RoPE effect."""
    names = ['input'] + [n for m in range(1, 8) for n in ((('fa' if m % 4 == 0 else 'dn') + str(m)), f'mlp{m}')]
    res = defaultdict(lambda: defaultdict(list))
    for r in recs:
        i = str(r['id'])
        t = int(rows_by_id[i]['tcell'])
        c = cls_of[i]
        for role, j in (('odd', 0), ('corr', 1), ('same', 2)):
            comp = r['l8k_comp'][j]                     # (15, 36)
            res[c][f'{role}_earlier'].append(comp[:, :t].mean(1))
            res[c][f'{role}_own'].append(comp[:, t])
            lg, nr = r['l8k_logit'][j], r['l8k_norope'][j]
            res[c][f'{role}_logit_earlier'].append(lg[:t].mean()); res[c][f'{role}_logit_own'].append(lg[t])
            res[c][f'{role}_logit_sd_earlier'].append(lg[:t].std())
            if t >= 3:
                res[c][f'{role}_rope_corr'].append(np.corrcoef(lg[:t], nr[:t])[0, 1])
        res[c]['q_odd'].append(r['l8_q'][0]); res[c]['q_corr'].append(r['l8_q'][1])
    out = {}
    for c, d in res.items():
        o = {'names': names}
        for k, v in d.items():
            if k.startswith('q_'):
                continue
            arr = np.asarray(v, np.float64)
            o[k] = arr.mean(0).tolist() if arr.ndim > 1 else ci(arr)
        q = np.asarray(d['q_odd'], np.float64)
        qn = q / np.linalg.norm(q, axis=1, keepdims=True)
        cs = qn @ qn.T
        o['q_cross_image_cos'] = float(cs[np.triu_indices(len(q), 1)].mean())
        out[c] = o
    return out


def neuron_stats(recs, cls_of, attr_sign: np.ndarray | None = None) -> dict:
    """Gate pre-activation of the round-3 top-200 MLP 8-15 neurons at the target tokens (mean over the 4 tokens).

    A_own = gate(odd) - gate(corrupted): effect of the target's own pixels with the majority reference;
    A_ref = gate(ref_all) - gate(odd): effect of replacing what the target reads from other cells by the same-type
    context (reference = target type); A_blk = gate(h6_blk) - gate(odd).  Subtraction predicts A_ref ~ -A_own."""
    res = defaultdict(lambda: defaultdict(list))
    grp_idx = {g: [COMP33.index(n) for n in ns] for g, ns in GROUPS.items()}
    for r in recs:
        c = cls_of[str(r['id'])]
        g = r['nrn_gate'].astype(np.float64).mean(1)      # (5 roles, 200)
        a = r['nrn_act'].astype(np.float64).mean(1)
        con = r['nrn_contrib'].astype(np.float64).mean(2)  # (5, 33, 200)
        res[c]['A_own'].append(g[0] - g[1])
        res[c]['A_ref'].append(g[4] - g[0])
        res[c]['A_blk'].append(g[3] - g[0])
        res[c]['act'].append(a)
        res[c]['dcon_own'].append(np.stack([con[0, ix].sum(0) - con[1, ix].sum(0) for ix in grp_idx.values()]))
        res[c]['dcon_ref'].append(np.stack([con[4, ix].sum(0) - con[0, ix].sum(0) for ix in grp_idx.values()]))
        res[c]['dcon_blk'].append(np.stack([con[3, ix].sum(0) - con[0, ix].sum(0) for ix in grp_idx.values()]))
        res[c]['con_corr'].append(np.stack([con[1, ix].sum(0) for ix in grp_idx.values()]))
        res[c]['check'].append(float(np.max(r['nrn_check'])))
    out = {}
    for c, d in res.items():
        Ao, Ar, Ab = (np.asarray(d[k]) for k in ('A_own', 'A_ref', 'A_blk'))   # (n, 200)
        act = np.asarray(d['act'])                                             # (n, 5, 200)
        mo, mr, mb = Ao.mean(0), Ar.mean(0), Ab.mean(0)
        slope = float(np.polyfit(mo, mr, 1)[0])
        out[c] = {'n': len(Ao), 'A_own_mean': float(mo.mean()), 'A_ref_mean': float(mr.mean()), 'A_blk_mean': float(mb.mean()),
                  'slope_ref_on_own': slope, 'corr_ref_own': float(np.corrcoef(mo, mr)[0, 1]),
                  'corr_blk_own': float(np.corrcoef(mo, mb)[0, 1]),
                  'cancel_ratio': float(-(mr.sum()) / mo.sum()), 'blk_ratio': float(-(mb.sum()) / mo.sum()),
                  'act_mean': act.mean((0, 2)).tolist(),
                  'frac_neurons_active_odd': float((act[:, 0] > 0.1).mean()),
                  'dcon_own': dict(zip(GROUPS, np.asarray(d['dcon_own']).mean((0, 2)).tolist())),
                  'dcon_ref': dict(zip(GROUPS, np.asarray(d['dcon_ref']).mean((0, 2)).tolist())),
                  'dcon_blk': dict(zip(GROUPS, np.asarray(d['dcon_blk']).mean((0, 2)).tolist())),
                  'max_reconstruction_error': float(np.max(d['check'])),
                  'per_neuron': {'A_own': mo.tolist(), 'A_ref': mr.tolist(), 'A_blk': mb.tolist()}}
    return out


def vit_stats(output: Path, recs, rows_by_id, cls_of) -> dict:
    z = np.load(output / 'zeta.npz', allow_pickle=True)['stats'].item()
    counts = {c: [int((s['zeta'][k] > 5).sum()) for k in range(27)] for c, s in z.items()}
    per_cls = defaultdict(list)
    for c, s in z.items():
        variant = c.split('|')[-1]
        pair = c.split('|')[1]
        per_cls[A.feature_class(pair, variant)].append(counts[c])
    out = {'excl_counts': counts, 'excl_by_class_mean': {k: np.mean(v, 0).tolist() for k, v in per_cls.items()}}
    # activation of exclusive neurons (block 9 set) on the target: odd vs corrupted vs same-type
    act = defaultdict(lambda: defaultdict(list))
    for r in recs:
        i = str(r['id'])
        cond = rows_by_id[i]['cond']
        zz = z[cond]['zeta']
        for k in (6, 9, 12, 18, 24):
            E = np.where(zz[k - 1] > 5)[0]
            if len(E):
                v = r['vit_tmax'][:, k - 1, E].astype(np.float64).mean(1)
                act[cls_of[i]][k].append(v)
    out['excl_act'] = {c: {k: np.mean(v, 0).tolist() for k, v in d.items()} for c, d in act.items()}
    return out


def x0_locality(recs, cls_of) -> dict:
    res = defaultdict(lambda: defaultdict(list))
    for r in recs:
        c = cls_of[str(r['id'])]
        d2 = r['x0_dnorm'].astype(np.float64) ** 2
        rel = r['x0_norm'].astype(np.float64)
        tot = d2.sum()
        top = np.argsort(-d2)
        res[c]['share_top1_token'].append(d2[top[0]] / tot)
        res[c]['share_top4_tokens'].append(d2[top[:4]].sum() / tot)
        res[c]['rel_change_top1'].append(np.sqrt(d2[top[0]]) / rel[top[0]])
        res[c]['rel_change_median_other'].append(np.median(np.sqrt(d2[top[4:]]) / rel[top[4:]]))
    return {c: {k: ci(v) for k, v in d.items()} for c, d in res.items()}


def ridge_cv_accuracy(X: np.ndarray, y: np.ndarray, folds: np.ndarray, alpha: float = 10.0) -> float:
    """Accuracy of a one-vs-rest ridge classifier (least squares on one-hot targets, standardised features),
    evaluated on held-out folds (folds: fold index per row)."""
    classes = np.unique(y)
    pred = np.zeros_like(y)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt, Xe = (X[tr] - mu) / sd, (X[te] - mu) / sd
        Y = (y[tr][:, None] == classes[None]).astype(np.float64) * 2 - 1
        Xa = np.hstack([Xt, np.ones((len(Xt), 1))])
        if Xa.shape[1] > Xa.shape[0]:            # dual form when features outnumber samples
            W = Xa.T @ np.linalg.solve(Xa @ Xa.T + alpha * np.eye(len(Xa)), Y)
        else:
            W = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(Xa.shape[1]), Xa.T @ Y)
        pred[te] = classes[np.argmax(np.hstack([Xe, np.ones((len(Xe), 1))]) @ W, 1)]
    return float((pred == y).mean())


# ----------------------------------------------------------------------------- readers

def value_probe(recs, rows_by_id) -> dict:
    """Ridge classifiers (5-fold over images) predicting a cell's row and column from the reader value vector of that
    cell (mean over its 4 tokens), per reader head."""
    V = np.stack([r['reader_v'].astype(np.float32) for r in recs])        # (n, 6, 36, 256)
    n = len(V)
    rows_, cols_ = np.tile(np.arange(36) // 6, n), np.tile(np.arange(36) % 6, n)
    grp = np.repeat(np.arange(n), 36)
    out = {}
    for h in range(V.shape[1]):
        X = V[:, h].reshape(-1, V.shape[-1]).astype(np.float64)
        out[str(h)] = {name: ridge_cv_accuracy(X, y, grp % 5) for name, y in (('row', rows_), ('col', cols_))}
    return out


def output_probe(output: Path) -> dict:
    """Ridge classifiers predicting the target row / column from the six reader outputs (W_O applied, summed over
    heads of the same layer) at the row and column query, 5-fold over atlas odd images."""
    p = output / 'probe.npz'
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=True)
    O = z['out'].astype(np.float32)                                       # (n, 6, 2, d)
    tg = np.asarray(z['target'])
    reader = np.asarray(z['reader'])                                      # (n, 6, 36)
    tc = tg[:, 0] * 6 + tg[:, 1]
    hit = reader.sum(1).argmax(-1) == tc
    out = {'n': int(len(O)), 'reader_top1_on_target': float(hit.mean())}
    for qi, qn in ((0, 'row_query'), (1, 'col_query')):
        for lname, hs in (('L16', [0, 1, 2]), ('L20', [3, 4, 5]), ('all', list(range(6)))):
            X = O[:, hs, qi].sum(1)
            for yname, y in (('row', tg[:, 0]), ('col', tg[:, 1])):
                for sub, msk in (('all', np.ones(len(X), bool)), ('reader_on_target', hit), ('reader_off_target', ~hit)):
                    if msk.sum() < 30:
                        continue
                    Xs, ys = X[msk].astype(np.float64), y[msk]
                    folds = np.random.default_rng(0).permutation(len(ys)) % 5
                    out[f'{qn}|{lname}|{yname}|{sub}'] = ridge_cv_accuracy(Xs, ys, folds)
    return out


# ----------------------------------------------------------------------------- round-3 neurons: shared vs specific

def neuron_sharing(round3: Path) -> dict:
    z = np.load(round3 / 'circuit' / 'records.npz')
    items = {}
    for line in open(round3 / 'items.jsonl'):
        it = json.loads(line)
        items[it['id']] = it
    attr = z['attr'].astype(np.float32).sum(1).reshape(len(z['id']), -1)    # (n, 8 * 12288)
    cond = np.array([items[str(i)]['pair_name'] + '|' + items[str(i)]['odd_variant'] for i in z['id']])
    cls = np.asarray(z['cls'])
    sel = json.loads((round3 / 'circuit' / 'selection.json').read_text())
    top = np.array([(l - 8) * 12288 + i for l, i in sel['neurons_top']['200']])
    conds = [c for c in sorted(set(cond)) if (cond == c).sum() >= 3]
    M = np.stack([attr[cond == c].mean(0) for c in conds])                   # (n_cond, N)
    ccls = [A.feature_class(c.split('|')[0], c.split('|')[1]) for c in conds]
    Mt = M[:, top]
    # a top neuron is "shared" if its mean attribution is positive in >= 80% of conditions of BOTH classes,
    # "comparison-specific" / "detector-specific" if positive in >= 80% of one class and <= 50% of the other
    cmp_rows = [j for j, k in enumerate(ccls) if k == 'comparison']
    det_rows = [j for j, k in enumerate(ccls) if k == 'detector']
    pc = (Mt[cmp_rows] > 0).mean(0)
    pd = (Mt[det_rows] > 0).mean(0)
    kind = np.where((pc >= .8) & (pd >= .8), 'shared', np.where((pc >= .8) & (pd <= .5), 'comparison',
                    np.where((pd >= .8) & (pc <= .5), 'detector', 'mixed')))
    share = {}
    for k in ('shared', 'comparison', 'detector', 'mixed'):
        m = kind == k
        share[k] = {'n': int(m.sum()),
                    'attr_share_cmp': float(Mt[cmp_rows][:, m].sum() / Mt[cmp_rows].sum()),
                    'attr_share_det': float(Mt[det_rows][:, m].sum() / Mt[det_rows].sum())}
    layer_of = np.array([l for l, _ in sel['neurons_top']['200']])
    share['layers'] = {k: np.bincount(layer_of[kind == k] - 8, minlength=8).tolist() for k in ('shared', 'comparison', 'detector', 'mixed')}
    # selection-independent view: top-200 by detector-class mean vs top-200 by comparison-class mean, and the top-50 of
    # every single condition, all over the full 98,304 neurons
    det_top = np.argsort(-attr[cls == 'detector'].mean(0))[:200]
    cmp_top = np.argsort(-attr[cls == 'comparison'].mean(0))[:200]
    share['overlap_top200_det_vs_cmp'] = int(len(set(det_top) & set(cmp_top)))
    per = {c: set(np.argsort(-M[j])[:50]) for j, c in enumerate(conds)}
    cnt = {}
    for c, st in per.items():
        others = [o for o in conds if o != c]
        cnt[c] = float(np.mean([len(st & set(np.argsort(-M[conds.index(o)])[:200])) / 50 for o in others]))
    share['top50_found_in_other_top200'] = cnt
    allc = defaultdict(int)
    for st in per.values():
        for n_ in st:
            allc[n_] += 1
    share['n_conditions'] = len(conds)
    share['neurons_in_top50_of_all_conditions'] = [[int(n_ // 12288 + 8), int(n_ % 12288)] for n_, k in allc.items() if k == len(conds)]
    share['neurons_in_top50_of_one_condition'] = int(sum(1 for k in allc.values() if k == 1))
    share['neurons_in_top50_union'] = len(allc)
    return {'conditions': conds, 'classes': ccls, 'kind': kind.tolist(), 'summary': share,
            'top': sel['neurons_top']['200']}


# ----------------------------------------------------------------------------- Qwen3-VL

def vl_stats(output: Path) -> dict:
    d = output / 'vl'
    if not (d / 'rows.jsonl').exists():
        return {}
    rows = read_jsonl(d / 'rows.jsonl')
    out = {'readers': json.loads((d / 'readers.json').read_text())['readers']}
    for ph in ('p2', 'p3', 'p4'):
        rr = [r for r in rows if r['phase'] == ph]
        if rr:
            out[ph] = intervention_table(rr)
    out['redirect'] = redirect_table([r for r in rows if r['phase'] == 'p2'])
    if (d / 'records.npz').exists():
        z = np.load(d / 'records.npz', allow_pickle=True)
        names = list(z['names'])
        adv = np.asarray(z['adv'], np.float64)                    # (n, 2, 6, C)
        M = adv[:, 0] - adv[:, 1]                                 # mark per head and component
        cls = np.asarray(z['cls'])
        out['mark'] = {}
        for c in CLS:
            m = M[cls == c].sum(1).mean(0)                        # sum over readers
            out['mark'][c] = {'total': float(m.sum()), 'by_comp': dict(zip(names, m.tolist())),
                              'n': int((cls == c).sum())}
        out['adv_err_max'] = float(np.max(z['adv_err']))
        out['x0_share_top1'] = {c: float(np.mean([np.sort(x ** 2)[-1] / (x ** 2).sum() for x, k in zip(z['x0_dnorm'], cls) if k == c]))
                                for c in CLS}
    if (d / 'selection.json').exists():
        out['selection'] = json.loads((d / 'selection.json').read_text())
    beh = read_jsonl(output / 'vl_behave.jsonl') if (output / 'vl_behave.jsonl').exists() else []
    out['n_behave'] = len(beh)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--round3', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args(argv)
    items = {r['id']: r for r in read_jsonl(args.output / 'items.jsonl')}
    cls_of = {i: r['cls'] for i, r in items.items()}
    summary = {}
    rows = read_jsonl(args.output / 'rows.jsonl') if (args.output / 'rows.jsonl').exists() else []
    if rows:
        summary['interventions'] = intervention_table(rows)
        summary['redirect'] = redirect_table(rows)
        if (args.output / 'redirect.jsonl').exists():
            summary['redirect_post'] = redirect_table(read_jsonl(args.output / 'redirect.jsonl'))
        summary['n_vit_excl'] = {c: ci([r['n_vit_excl'] for r in rows if r['cond'] == 'none' and r['cls'] == c]) for c in CLS}
    recs = load_records(args.output)
    if recs:
        summary['x0_locality'] = x0_locality(recs, cls_of)
        summary['h6'] = h6_stats(recs, items, cls_of)
        summary['h6_distance'] = h6_distance_profile(recs, items)
        summary['h6_dim'] = h6_dimensionality(recs)
        summary['l8_key'] = l8_key_stats(recs, items, cls_of)
        summary['neurons'] = neuron_stats(recs, cls_of)
        summary['vit'] = vit_stats(args.output, recs, items, cls_of)
        summary['value_probe'] = value_probe(recs, items)
    summary['output_probe'] = output_probe(args.output)
    summary['neuron_sharing'] = neuron_sharing(args.round3)
    summary['vl'] = vl_stats(args.output)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    args.figures.mkdir(parents=True, exist_ok=True)
    from analysis_tools.representation import search_pipeline_figures as F
    F.make_all(args.output, args.round3, args.figures, summary, recs, rows, items)


if __name__ == '__main__':
    main()
