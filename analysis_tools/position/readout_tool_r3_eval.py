"""Experiment 3.9 round 3 evaluation: reduced-input readers against the model, input cost, and
RefCOCO failure modes.  Called from readout_tool_r3_analysis (stage evaluate)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position import readout_tool_r2_analysis as a2
from analysis_tools.position import readout_tool_r3 as r3
from analysis_tools.position.coordinate_readout_run import read_jsonl
from analysis_tools.position.readout_tool_analysis import boot_ci, mcnemar_exact, wilson

R2_NAMES = {'all_content_approx': 'cnn_slot_content_approx', 'k0_content_approx': 'cnn_content_approx',
            'all_noncontent': 'cnn_slot', 'approx_point': 'approx_point'}
N_VISUAL, SEQ_LEN, D_MODEL, D_HEAD = 1024, 1300, 4096, 256


def cost(config: tuple | None, n_params: int) -> dict:
    """Extra inference cost of a reader for one call at N_VISUAL visual tokens and SEQ_LEN tokens.

    attention rows: q.K for k heads at one query over the cached keys (+ softmax) -> 2 k D_HEAD SEQ_LEN;
    content: PCA projection of the visual-token states of each content layer -> 2 N_VISUAL D_MODEL 64;
    reader: about 2 x parameters x tokens for a convolutional reader applied at every token.
    Storage counts the per-call tensors the tool must read beyond the KV cache (fp16)."""
    if config is None:
        return {'heads': 0, 'gflops': 0.0, 'extra_storage_kb': 0.0}
    spec, layers, extent, approx = config
    k = len(r3.parse_heads(spec))
    attn = 2 * k * D_HEAD * SEQ_LEN
    cont = 2 * N_VISUAL * D_MODEL * 64 * len(layers)
    reader = 2 * n_params * N_VISUAL
    store = (k * N_VISUAL + 64 * len(layers) * N_VISUAL) * 2 / 1024
    return {'heads': k, 'content_layers': len(layers), 'attn_mflops': attn / 1e6, 'content_gflops': cont / 1e9,
            'reader_gflops': reader / 1e9, 'gflops': (attn + cont + reader) / 1e9, 'extra_storage_kb': store,
            'layers_touched': sorted({r3.ATTN_LAYERS[i // 16] for i in r3.parse_heads(spec)})}


def load_predictions(out: Path, r2run: Path) -> tuple[list[str], dict, dict]:
    ids = json.loads((r2run / 'tool_r2' / 'ids.json').read_text())
    preds, logs = {}, {}
    for f in sorted((r2run / 'tool_r2').glob('predictions_*.npz')):
        d = dict(np.load(f))
        for new, old in R2_NAMES.items():
            if old in d:
                preds[new] = d[old]
    for f in sorted((r2run / 'tool_r2').glob('readers_*.json')):
        d = json.loads(f.read_text())
        for new, old in R2_NAMES.items():
            if old in d and isinstance(d[old], dict):
                logs[new] = d[old]
    if json.loads((out / 'ids.json').read_text()) != ids:
        raise ValueError('round-3 reader ids differ from the round-2 capture order')
    for f in sorted(out.glob('predictions_P*.npz')):
        preds.update(dict(np.load(f)))
    for f in sorted(out.glob('readers_P*.json')):
        logs.update(json.loads(f.read_text()))
    return ids, preds, logs


def summaries(cases, ids, P, own, test_idx) -> dict:
    rows = a2.score_rows(cases, [ids[k] for k in test_idx], P[test_idx], own)
    return {s: a2.set_summary([r for r in rows if r['set'] == s]) for s in a2.EVAL_SETS if any(r['set'] == s for r in rows)}, rows


def bin_rate(values, flags, edges) -> list[dict]:
    values, flags = np.asarray(values, float), np.asarray(flags, bool)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (values >= lo) & (values < hi)
        k, n = int(flags[m].sum()), int(m.sum())
        out.append({'lo': lo, 'hi': hi, 'n': n, 'rate': k / n if n else float('nan'), 'ci': wilson(k, n) if n else [np.nan, np.nan]})
    return out


def failure_modes(cases, rows_by_reader: dict, ann: dict, zs_peak: dict) -> dict:
    """RefCOCO test failure modes per reader (tool point/box) and for the model's own point and box."""
    res = {}
    base = rows_by_reader[next(iter(rows_by_reader))]
    rc = [r for r in base if r['set'] == 'refcoco']
    area = [r['area_frac'] for r in rc]
    others = [ann[r['id']]['n_others'] for r in rc]
    occl = [ann[r['id']]['occlusion'] for r in rc]
    area_edges = [0, 0.02, 0.06, 0.15, 0.35, 1.01]
    other_edges = [0, 1, 2, 3, 5, 100]
    occ_edges = [0, 1e-6, 0.25, 0.5, 1.01]
    res['n'] = len(rc)
    res['model'] = {'area': bin_rate(area, [r['own_hit'] for r in rc], area_edges),
                    'others': bin_rate(others, [r['own_hit'] for r in rc], other_edges),
                    'occlusion': bin_rate(occl, [r['own_hit'] for r in rc], occ_edges)}
    for name, rows in rows_by_reader.items():
        rr = [r for r in rows if r['set'] == 'refcoco']
        if [r['id'] for r in rr] != [r['id'] for r in rc]:
            raise ValueError('row order differs between readers')
        hit = [r['tool_hit'] for r in rr]
        iou = np.asarray([r['tool_iou'] for r in rr], float)
        res[name] = {'area': bin_rate(area, hit, area_edges), 'others': bin_rate(others, hit, other_edges),
                     'occlusion': bin_rate(occl, hit, occ_edges),
                     'iou_by_area': [{'lo': lo, 'hi': hi, 'iou': float(np.nanmean(iou[(np.asarray(area) >= lo) & (np.asarray(area) < hi)]))}
                                     for lo, hi in zip(area_edges[:-1], area_edges[1:])]}
        # selection outcomes, tool versus model on the same case
        sel_t, sel_m = [], []
        for r in rr:
            a = ann[r['id']]
            tgt = np.asarray(cases[r['id']]['truth']['box'], float) * 1000 / np.asarray(cases[r['id']]['truth']['canvas'] * 2)
            oth = [np.asarray(o) for o in a['others']]
            sel_t.append(r3.selection_outcome(np.asarray(r['tool_point'], float), tgt, oth))
            sel_m.append(r3.selection_outcome(np.asarray(r['own_point'], float), tgt, oth) if r['own_point'] else 'invalid')
        tab = {}
        for t, m in zip(sel_t, sel_m):
            tab[f'{t}|{m}'] = tab.get(f'{t}|{m}', 0) + 1
        res[name]['selection'] = tab
        # box extent: log area ratio and IoU on cases whose point hits
        lr, ious = [], []
        for r in rr:
            if r['tool_box'] is None or not r['tool_hit']:
                continue
            t = np.asarray(cases[r['id']]['truth']['box'], float) * 1000 / np.asarray(cases[r['id']]['truth']['canvas'] * 2)
            b = np.asarray(r['tool_box'], float)
            lr.append(np.log(max((b[2] - b[0]) * (b[3] - b[1]), 1.0) / max((t[2] - t[0]) * (t[3] - t[1]), 1.0)))
            ious.append(r['tool_iou'])
        lr = np.asarray(lr)
        res[name]['extent'] = {'n_hit': len(lr), 'log_area_ratio_median': float(np.median(lr)) if len(lr) else np.nan,
                               'frac_too_small_half': float((lr < np.log(0.5)).mean()) if len(lr) else np.nan,
                               'frac_too_large_double': float((lr > np.log(2)).mean()) if len(lr) else np.nan,
                               'iou_given_hit': float(np.mean(ious)) if ious else np.nan}
    # model's own box extent for reference
    lr = []
    for r in rc:
        if r['own_box'] is None or not r['own_hit']:
            continue
        t = np.asarray(cases[r['id']]['truth']['box'], float) * 1000 / np.asarray(cases[r['id']]['truth']['canvas'] * 2)
        b = np.asarray(r['own_box'], float)
        lr.append(np.log(max((b[2] - b[0]) * (b[3] - b[1]), 1.0) / max((t[2] - t[0]) * (t[3] - t[1]), 1.0)))
    lr = np.asarray(lr)
    res['model']['extent'] = {'n_hit': len(lr), 'log_area_ratio_median': float(np.median(lr)),
                              'frac_too_small_half': float((lr < np.log(0.5)).mean()),
                              'frac_too_large_double': float((lr > np.log(2)).mean())}
    # salient-part bias: where points land inside the target box (normalized u, v) and on the mask
    sal = {}
    pts = {'H11 peak': {r['id']: zs_peak[r['id']] for r in rc}}
    for name, rows in rows_by_reader.items():
        pts[f'tool:{name}'] = {r['id']: r['tool_point'] for r in rows if r['set'] == 'refcoco'}
    pts['model point'] = {r['id']: r['own_point'] for r in rc if r['own_point']}
    pts['model box centre'] = {r['id']: rt_.answer_center(r['own_box']) for r in rc if r['own_box']}
    for name, d in pts.items():
        uv, on_mask, d_mask, d_box = [], [], [], []
        for cid, p in d.items():
            c = cases[cid]
            t = np.asarray(c['truth']['box'], float) * 1000 / np.asarray(c['truth']['canvas'] * 2)
            p = np.asarray(p, float)
            uv.append(((p[0] - t[0]) / max(t[2] - t[0], 1e-6), (p[1] - t[1]) / max(t[3] - t[1], 1e-6)))
            g = rt_.grid_of(c['truth']['canvas'])
            col = int(np.clip(p[0] / 1000 * g[1], 0, g[1] - 1))
            row = int(np.clip(p[1] / 1000 * g[0], 0, g[0] - 1))
            on_mask.append(ann[cid]['token_mask'][row * g[1] + col] >= 0.5)
            d_mask.append(float(np.linalg.norm(p - np.asarray(ann[cid]['mask_centroid']))))
            d_box.append(float(np.linalg.norm(p - r3.box_center(t))))
        uv = np.asarray(uv)
        people = np.asarray([ann[cid]['subset'] == 'testA' for cid in d])
        sal[name] = {'n': len(uv), 'u_median': float(np.median(uv[:, 0])), 'v_median': float(np.median(uv[:, 1])),
                     'v_median_testA': float(np.median(uv[people, 1])) if people.any() else np.nan,
                     'v_median_testB': float(np.median(uv[np.asarray([ann[c]['subset'] == 'testB' for c in d]), 1])),
                     'on_mask': float(np.mean(on_mask)), 'err_to_mask_centroid_median': float(np.median(d_mask)),
                     'err_to_box_centre_median': float(np.median(d_box)), 'uv': uv.round(3).tolist(), 'ids': list(d)}
    res['salient'] = sal
    return res


def stage_evaluate(r2run: Path, r1: Path, out: Path) -> None:
    cases = {c['id']: c for c in read_jsonl(r2run / 'cases.jsonl')}
    ids, preds, logs = load_predictions(out, r2run)
    own = a2.own_answers(r2run, r1)
    test_idx = [k for k, i in enumerate(ids) if cases[i]['split'] == 'test' and not i.endswith('_hp')]
    res = {'readers': {}, 'cost': {}}
    rows_by = {}
    from analysis_tools.position.readout_tool_r3_analysis import CONFIGS
    for name, P in preds.items():
        s, rows = summaries(cases, ids, P, own, test_idx)
        cfg = CONFIGS.get(name.split('@')[0])
        n_params = logs.get(name, {}).get('n_params', 360_000 if cfg else 0)
        res['readers'][name] = {'test': s, 'val': logs.get(name, {}).get('val'), 'config': cfg}
        res['cost'][name] = cost(cfg, n_params)
        rows_by[name] = rows
    # zero-training decoders (point only)
    zs = np.load(out / 'zeroshot_points.npz', allow_pickle=False)
    if zs['ids'].tolist() != ids:
        raise ValueError('zeroshot ids differ from the capture order')
    H11 = r3.head_index(20, 11)
    zero = {'zs_H11_peak': np.concatenate([zs['peak'][:, H11]] * 3, 1).astype(float),
            'zs_H11_peak_target_end': np.concatenate([zs['peak_te']] * 3, 1).astype(float)}
    for k in zs.files:
        if k.startswith('box_'):
            B = zs[k].astype(float)
            zero[f'zs_{k[4:]}'] = np.concatenate([(B[:, :2] + B[:, 2:]) / 2, B], 1)
    for name, P in zero.items():
        P = P.copy()
        if name in ('zs_H11_peak', 'zs_H11_peak_target_end'):
            P[:, 2:] = np.nan                                          # point-only decoders
        s, rows = summaries(cases, ids, P, own, test_idx)
        res['readers'][name] = {'test': s}
        rows_by[name] = rows
    ann = json.loads((out / 'refcoco_annotations.json').read_text())
    zs_peak = {ids[k]: zs['peak'][k, H11].tolist() for k in test_idx}
    focus = [n for n in ('all_content_approx', 'k4_content_approx', 'k1_H11_content_approx', 'k1_H11',
                         'zs_H11_peak', 'zs_L20H11_content') if n in rows_by]
    res['failure'] = failure_modes(cases, {n: rows_by[n] for n in focus}, ann, zs_peak)
    # paired comparisons of reduced readers with the full round-2 tool on RefCOCO and every set
    ref = {r['id']: r for r in rows_by['all_content_approx']}
    res['paired_vs_full'] = {}
    for name, rows in rows_by.items():
        out_s = {}
        for s in a2.EVAL_SETS:
            rr = [r for r in rows if r['set'] == s]
            if not rr:
                continue
            d = np.asarray([r['tool_error'] - ref[r['id']]['tool_error'] for r in rr])
            b = sum(r['tool_hit'] and not ref[r['id']]['tool_hit'] for r in rr)
            c = sum(ref[r['id']]['tool_hit'] and not r['tool_hit'] for r in rr)
            out_s[s] = {'err_diff_mean': float(d.mean()), 'err_diff_ci': boot_ci(d), 'only_reader': b, 'only_full': c,
                        'p_mcnemar': mcnemar_exact(b, c)}
        res['paired_vs_full'][name] = out_s
    # seed replicates: configs refit with training seeds 1 and 2 ('config@s1', 'config@s2')
    res['seeds'] = {}
    for base in sorted({n.split('@')[0] for n in res['readers'] if '@s' in n}):
        names = [base] + sorted(n for n in res['readers'] if n.startswith(base + '@s'))
        rr = [res['readers'][n] for n in names]
        res['seeds'][base] = {'names': names,
                              'val_mean': [r['val']['mean'] for r in rr if r.get('val')],
                              'rc_median': [r['test']['refcoco']['tool_err_median'] for r in rr],
                              'rc_iou': [r['test']['refcoco']['tool_iou'] for r in rr],
                              'rc_hit': [r['test']['refcoco']['tool_hit'] for r in rr],
                              'heldout_hit': [float(np.mean([r['test'][s]['tool_hit'] for s in ('disc', 'ring', 't6', 'layout')])) for r in rr],
                              't6_hit': [r['test']['t6']['tool_hit'] for r in rr], 'layout_hit': [r['test']['layout']['tool_hit'] for r in rr]}
    (out / 'rows_focus.json').write_text(json.dumps({n: rows_by[n] for n in focus}))
    (out / 'results_r3.json').write_text(json.dumps(res, indent=1, default=float))
    brief = {n: {s: (round(v['tool_hit'], 3), round(v['tool_err_median'], 1), round(v['tool_iou'], 3) if v['tool_iou'] == v['tool_iou'] else None)
                 for s, v in r['test'].items()} for n, r in res['readers'].items()}
    print(json.dumps(brief, indent=0)[:6000], flush=True)
