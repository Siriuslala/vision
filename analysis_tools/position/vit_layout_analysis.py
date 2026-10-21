"""Metrics of experiment 3.7 (ViT position and layout) from the stage artifacts.

Writes `results.json` (all metrics with Wilson / bootstrap intervals) into the run directory.
Layout probes are fitted here from the saved features (ridge, lambda chosen on a validation split).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np

from analysis_tools.position.coordinate_readout_analysis import bootstrap_mean, displacement_weight, prop, read_jsonl
from analysis_tools.position.vit_layout import layout_labels

TAGS = ('q35', 'q3vl')


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie); here 'pos' are the identical pairs (answer Yes)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def slope(truth: np.ndarray, pred: np.ndarray) -> float:
    t = truth - truth.mean()
    return float(t @ (pred - pred.mean()) / (t @ t))


def lenient_center(r: dict) -> list[float] | None:
    """Diagnostic parse: the strict answer, or a list whose point_2d entries are all identical
    (Qwen3-VL often repeats the same point). Returns None otherwise."""
    if r['score']['valid']:
        return r['score']['center']
    pts = re.findall(r'"point_2d":\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', r['text'])
    if pts and len(set(pts)) == 1 and r['finish_reason'] == 'eos':
        return [float(v) for v in pts[0]]
    return None


def lenient_metrics(rows: list[dict], size: int) -> dict:
    hits, errs, truth, pred = 0, [], [], []
    for r in rows:
        c = lenient_center(r)
        if c is None:
            continue
        t = np.array([1000 * v / size for v in r['case']['center']])
        px = np.asarray(c) * size / 1000
        hits += float(np.linalg.norm(px - np.asarray(r['case']['center']))) <= r['case']['diameter'] / 2
        errs.append(float(np.linalg.norm(np.asarray(c) - t)))
        truth.append(t); pred.append(c)
    out = {'valid': prop(len(errs), len(rows)), 'hit': prop(int(hits), len(rows)), 'error': bootstrap_mean(errs)}
    if len(pred) >= 3:
        t, p = np.array(truth), np.array(pred)
        out.update(slope_x=slope(t[:, 0], p[:, 0]), slope_y=slope(t[:, 1], p[:, 1]))
    return out


def pixel_hit(r: dict) -> bool:
    """Hit when the 0-1000 answer is read as absolute pixel coordinates of the canvas."""
    x, y = r['score']['center']
    S = r['case']['canvas']
    cx, cy = r['case']['center']
    return bool(x < S and y < S and np.hypot(x - cx, y - cy) <= r['case']['diameter'] / 2)


def coord_metrics(recs: list[dict]) -> dict:
    """Per (canvas, condition): hit rate, center error, regression slopes and output spread."""
    groups = defaultdict(list)
    for r in recs:
        bg = r['case'].get('background', 'white')
        groups[(r['case']['canvas'] if bg == 'white' else f"{r['case']['canvas']}{bg}", r['condition'])].append(r)
    out = {}
    for (canvas, cond), rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        size = int(str(canvas).replace('noise', ''))
        valid = [r for r in rows if r['score']['valid']]
        truth = np.array([[1000 * v / size for v in r['case']['center']] for r in valid]) if valid else np.zeros((0, 2))
        pred = np.array([r['score']['center'] for r in valid]) if valid else np.zeros((0, 2))
        # Diagnostic (plan section 6.9): the same output read as absolute pixels.
        pix = [r['score']['valid'] and pixel_hit(r) for r in rows]
        r_disc = rows[0]['case']['diameter'] / 2
        m = {'n': len(rows), 'valid': prop(len(valid), len(rows)),
             'hit': prop(sum(r['score']['hit'] for r in rows), len(rows)),
             'hit_any': prop(sum(bool(r['score']['hit']) or p for r, p in zip(rows, pix)), len(rows)),
             'hit_pixel_only': prop(sum(p and not r['score']['hit'] for r, p in zip(rows, pix)), len(rows)),
             'chance_norm': float(np.pi * r_disc ** 2 / size ** 2),
             'chance_pixel': float(np.pi * r_disc ** 2 / 1e6),
             'error': bootstrap_mean([r['score']['error'] for r in valid])}
        m['lenient'] = lenient_metrics(rows, size)
        if len(valid) >= 3:
            m.update(slope_x=slope(truth[:, 0], pred[:, 0]), slope_y=slope(truth[:, 1], pred[:, 1]),
                     corr_x=float(np.corrcoef(truth[:, 0], pred[:, 0])[0, 1]) if pred[:, 0].std() > 0 else 0.0,
                     corr_y=float(np.corrcoef(truth[:, 1], pred[:, 1])[0, 1]) if pred[:, 1].std() > 0 else 0.0,
                     spread=float(pred.std(0).mean()))
        out[f'{canvas}|{cond}'] = m
    return out


def t2_metrics(recs: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in recs:
        groups[(r['case']['canvas'], r['condition'])].append(r)
    out = {}
    for (canvas, cond), rows in sorted(groups.items()):
        same = np.array([r['s'] for r in rows if r['case']['same']])
        diff = np.array([r['s'] for r in rows if not r['case']['same']])
        correct = sum((r['s'] > 0) == r['case']['same'] for r in rows)
        out[f'{canvas}|{cond}'] = {'n': len(rows), 'auroc': auroc(same, diff), 'accuracy': prop(correct, len(rows)),
                                   'mean_s_same': float(same.mean()), 'mean_s_diff': float(diff.mean()),
                                   'yes_rate': prop(int(sum(r['s'] > 0 for r in rows)), len(rows))}
    return out


def swap_metrics(recs: list[dict]) -> dict:
    """Displacement weight w per (canvas, variant, kind), relative to the variant's own base output."""
    vals = defaultdict(list)
    fails = defaultdict(lambda: [0, 0])
    for rec in recs:
        conds = {c['condition']: c for c in rec['conditions']}
        canvas = rec['case']['canvas']
        for name, c in conds.items():
            variant, kind = name.split('|')
            if kind == 'base':
                continue
            base = conds[f'{variant}|base']
            key = f"{canvas}|{variant}|{kind.rsplit('_', 1)[0]}"
            fails[key][1] += 1
            if c['center'] is None or base['center'] is None:
                fails[key][0] += 1
                continue
            w, _ = displacement_weight(np.asarray(c['center']) - np.asarray(base['center']), np.asarray(c['d']))
            vals[(key, rec['id'])].append(w)
    per = defaultdict(list)
    for (key, _), ws in vals.items():
        per[key].append(float(np.mean(ws)))
    out = {k: {'w': bootstrap_mean(v), 'n_images': len(v), 'invalid': fails[k][0], 'total': fails[k][1]}
           for k, v in sorted(per.items())}
    base_hits = defaultdict(lambda: [0, 0])
    for rec in recs:
        for c in rec['conditions']:
            if c['condition'].endswith('|base'):
                k = f"{rec['case']['canvas']}|{c['condition'].split('|')[0]}"
                base_hits[k][0] += int(c['score']['hit'])
                base_hits[k][1] += 1
    out['base_hits'] = {k: prop(*v) for k, v in base_hits.items()}
    return out


def ridge_fit_predict(xtr, ytr, xva, yva, xte, lambdas=(1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5)):
    """Ridge on standardized features; lambda chosen by validation MSE; returns test predictions."""
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    ztr, zva, zte = (xtr - mu) / sd, (xva - mu) / sd, (xte - mu) / sd
    ym = ytr.mean(0)
    u, s, vt = np.linalg.svd(ztr, full_matrices=False)
    uy = u.T @ (ytr - ym)
    best = None
    for lam in lambdas:
        w = vt.T @ ((s / (s ** 2 + lam))[:, None] * uy)
        mse = float(((zva @ w + ym - yva) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, lam, w)
    return zte @ best[2] + ym, best[1]


def layout_metrics(run: Path, tag: str) -> dict:
    root = run / 'layout'
    cases = json.loads((root / 'cases.json').read_text())
    labels = [layout_labels(c) for c in cases]
    n = len(cases)
    tr, va = slice(0, int(.6 * n)), slice(int(.6 * n), int(.75 * n))
    te = slice(int(.75 * n), n)
    reg = ('red_x', 'red_y', 'blue_x', 'blue_y')
    cls = ('red_left', 'red_above', 'red_bigger')
    out = {}
    for path in sorted(root.glob(f'{tag}_*_features.npz')):
        cond = path.name[len(tag) + 1:-len('_features.npz')]
        data = np.load(path)
        res = {}
        for key in data.files:
            x = data[key].astype(np.float32)                               # (n, 5, D)
            for ri, readout in enumerate(('red', 'blue', 'background', 'bg_single', 'pooled')):
                xr = x[:, ri]
                row = {}
                y = np.array([[l[k] for k in reg] for l in labels])
                p, lam = ridge_fit_predict(xr[tr], y[tr], xr[va], y[va], xr[te])
                for j, k in enumerate(reg):
                    sst = ((y[te, j] - y[te, j].mean()) ** 2).sum()
                    row[k] = {'r2': float(1 - ((p[:, j] - y[te, j]) ** 2).sum() / sst),
                              'mae': float(np.abs(p[:, j] - y[te, j]).mean())}
                yc = np.array([[2 * l[k] - 1 for k in cls] for l in labels])
                p, _ = ridge_fit_predict(xr[tr], yc[tr], xr[va], yc[va], xr[te])
                for j, k in enumerate(cls):
                    ok = int(((p[:, j] > 0) == (yc[te, j] > 0)).sum())
                    row[k] = prop(ok, len(p))
                res[f'{key}|{readout}'] = row
        out[cond] = res
        print('layout', tag, cond, 'done', flush=True)
    return out


def leace_metrics(run: Path, tag: str) -> dict:
    root = run / 'leace'
    if not (root / f'{tag}_coord.jsonl').exists():
        return {}
    coord = read_jsonl(root / f'{tag}_coord.jsonl')
    for r in coord:
        r['case']['canvas'] = 512
    out = {'coord': coord_metrics(coord), 't2': t2_metrics(read_jsonl(root / f'{tag}_t2.jsonl'))}
    # Paired change of center error relative to no erasure.
    by = defaultdict(dict)
    for r in coord:
        by[r['case']['id']][r['condition']] = r
    deltas = defaultdict(list)
    for cid, conds in by.items():
        base = conds.get('none')
        for name, r in conds.items():
            if name != 'none' and base and base['score']['valid'] and r['score']['valid']:
                deltas[name].append(r['score']['error'] - base['score']['error'])
    out['delta_error'] = {k: bootstrap_mean(v) for k, v in deltas.items()}
    p = root / f'{tag}_probe.json'
    if p.exists():
        out['probe'] = {k: {kk: vv for kk, vv in v.items() if kk != 'pred_mean_map'} for k, v in json.loads(p.read_text()).items()}
    return out


def reread_metrics(recs: list[dict]) -> dict:
    """w = fraction of the move from the crop-local answer toward the original-image answer."""
    out = defaultdict(list)
    valid = defaultdict(lambda: [0, 0])
    for rec in recs:
        conds = {c['condition']: c for c in rec['conditions']}
        local = np.asarray(rec['truth_local'])
        full = np.asarray(rec['truth_full'])
        d = full - local
        for name, c in conds.items():
            valid[name][1] += 1
            if c['center'] is None:
                continue
            valid[name][0] += 1
            y = np.asarray(c['center'])
            w, _ = displacement_weight(y - local, d)
            out[name].append({'w': w, 'err_local': float(np.linalg.norm(y - local)),
                              'err_full': float(np.linalg.norm(y - full))})
    return {k: {'valid': prop(*valid[k]), 'w': bootstrap_mean([r['w'] for r in out[k]]),
                'err_local': bootstrap_mean([r['err_local'] for r in out[k]]),
                'err_full': bootstrap_mean([r['err_full'] for r in out[k]])} for k in valid}


def refcoco_metrics(recs: list[dict]) -> dict:
    """IoU>=0.5 accuracy per condition, and paired changes against base (same expressions)."""
    by = defaultdict(dict)
    for r in recs:
        by[r['case_id']][r['condition']] = r['score']
    out = {}
    for cond in sorted({r['condition'] for r in recs}):
        rows = [v[cond] for v in by.values() if cond in v]
        m = {'n': len(rows), 'acc': prop(sum(x['correct'] for x in rows), len(rows)),
             'center_in': prop(sum(x['center_in'] for x in rows), len(rows)),
             'mean_iou': bootstrap_mean([x['iou'] for x in rows])}
        if cond != 'base':
            pairs = [(v['base'], v[cond]) for v in by.values() if 'base' in v and cond in v]
            m['gained'] = sum((not a['correct']) and b['correct'] for a, b in pairs)
            m['lost'] = sum(a['correct'] and not b['correct'] for a, b in pairs)
            m['delta_iou'] = bootstrap_mean([b['iou'] - a['iou'] for a, b in pairs])
        out[cond] = m
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = args.run
    res = {}
    for tag in TAGS:
        r = {}
        if (run / 'coord' / f'{tag}_records.jsonl').exists():
            r['coord'] = coord_metrics(read_jsonl(run / 'coord' / f'{tag}_records.jsonl'))
        if (run / 't2' / f'{tag}_records.jsonl').exists():
            r['t2'] = t2_metrics(read_jsonl(run / 't2' / f'{tag}_records.jsonl'))
        spot = run / 'coord' / f'{tag}_spotcheck.json'
        if spot.exists():
            rows = json.loads(spot.read_text())
            r['spotcheck'] = {'same_answer': prop(sum(x['same_answer'] for x in rows), len(rows)),
                              'same_text': prop(sum(x['same_text'] for x in rows), len(rows))}
        probes = {}
        for f in sorted((run / 'probe').glob(f'{tag}_*.json')):
            d = json.loads(f.read_text())
            probes[d['condition']] = {k: {kk: vv for kk, vv in v.items() if kk != 'mae_per_image'}
                                      for k, v in d['stages'].items()}
        if probes:
            r['probe'] = probes
        if (run / 'layout' / 'cases.json').exists() and list((run / 'layout').glob(f'{tag}_*_features.npz')):
            r['layout'] = layout_metrics(run, tag)
        r['leace'] = leace_metrics(run, tag)
        if (run / 'refcoco' / f'{tag}_records.jsonl').exists():
            r['refcoco'] = refcoco_metrics(read_jsonl(run / 'refcoco' / f'{tag}_records.jsonl'))
        if r:
            res[tag] = r
    if (run / 'swap' / 'records.jsonl').exists():
        res['q35']['swap'] = swap_metrics(read_jsonl(run / 'swap' / 'records.jsonl'))
    if (run / 'reread' / 'records.jsonl').exists():
        res['q35']['reread'] = reread_metrics(read_jsonl(run / 'reread' / 'records.jsonl'))
    (run / 'results.json').write_text(json.dumps(res, indent=1) + '\n')
    print('wrote', run / 'results.json')


if __name__ == '__main__':
    main()
