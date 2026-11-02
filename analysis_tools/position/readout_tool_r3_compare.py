"""Experiment 3.9 round 3: the readout tool against experiment 3.13 (targeted fine-tuning) on the
3.13 test set.

The tool's point and box for every (image, target) call are written as answer text in the model's
JSON format and scored with the 3.13 scorer (targeted_finetune.score) and group metrics
(targeted_finetune_analysis.group_metrics), so both methods use identical cases, hit rules and
aggregation.  Every fine-tuning condition with an `eval/<condition>/coord.jsonl` is compared.
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

from analysis_tools.position import readout_tool_r2 as r2
from analysis_tools.position import readout_tool_r2_analysis as a2
from analysis_tools.position import readout_tool_r3 as r3
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_analysis as tfa
from analysis_tools.position.coordinate_readout_run import read_jsonl
from analysis_tools.position.readout_tool_r3_analysis import make_items_r3
from analysis_tools.position.readout_tool_r3_run import tf_calls

READERS = {  # tool name -> (reader file relative to the round-3 or round-2 dir, config name)
    'tool_k1': ('r3', 'k1_H11_content_approx', 'k1_H11_content_approx'),
    'tool_k1_nocc': ('r3', 'k1_H11_content_approx@nocc', 'k1_H11_content_approx'),
    'tool_k4': ('r3', 'k4_content_approx', 'k4_content_approx'),
    'tool_k4_nocc': ('r3', 'k4_content_approx@nocc', 'k4_content_approx'),
    'tool_all128': ('r2', 'cnn_slot_content_approx', 'all_content_approx'),
    'tool_all128_nocc': ('r3', 'all_content_approx@nocc', 'all_content_approx'),
    'tool_k0': ('r2', 'cnn_content_approx', 'k0_content_approx'),
}
LABELS = {'base': 'model (no training)', 'all_linear': 'LoRA, all linear layers (3.13)',
          'mlp10_14': 'LoRA, MLP of layers 10-14 (3.13)', 'mixer16_24': 'LoRA, token mixers of layers 16-24 (3.13)', 'tool_k1': 'tool: L20 H11 + content + rough point',
          'tool_k1_nocc': 'tool (1 head), clutter/chart_ui not in training', 'tool_k4': 'tool: 4 heads',
          'tool_k4_nocc': 'tool (4 heads), clutter/chart_ui not in training', 'tool_all128': 'tool: 128 heads (round 2)',
          'tool_all128_nocc': 'tool (128 heads), clutter/chart_ui not in training', 'tool_k0': 'tool: content + rough point',
          'tool_peak': 'H11 attention peak (3.11 decoder)'}


def load_tf_capture(out: Path) -> dict:
    cap = {'ids': [], 'grid': [], 'cont': [], 'maps_slot': [], 'content_0': [], 'content_16': []}
    for f in sorted((out / 'tf_features').glob('chunk_*.npz')):
        d = np.load(f)
        cut = np.cumsum(np.concatenate([[0], d['map_len']]))
        cap['ids'] += d['ids'].tolist()
        cap['grid'] += [tuple(int(v) for v in g) for g in d['grid']]
        cap['cont'] += d['cont'].tolist()
        maps, c0, c16 = d['maps_slot'], d['content_0'], d['content_16']      # read each array once
        for a, b in zip(cut[:-1], cut[1:]):
            cap['maps_slot'].append(maps[:, a:b])
            cap['content_0'].append(c0[a:b])
            cap['content_16'].append(c16[a:b])
    return cap


def answer_texts(p: np.ndarray) -> dict:
    pt = [int(v) for v in np.clip(np.rint(p[:2]), 0, 1000)]
    out = {'point': json.dumps([{'point_2d': pt}])}
    if not np.isnan(p[2:]).any():
        b = np.clip(np.rint(p[2:]), 0, 1000).astype(int).tolist()
        out['bbox'] = json.dumps([{'bbox_2d': b}])
    return out


def predict_all(r2run: Path, out: Path, cap: dict, calls: dict) -> dict[str, np.ndarray]:
    cases = {cid: {'truth': calls[cid]['truth']} for cid in cap['ids']}
    preds = {}
    for name, (where, fname, cfg) in READERS.items():
        path = (out / f'{fname}.pt') if where == 'r3' else (r2run / 'tool_r2' / f'{fname}.pt')
        if not path.exists():
            print(f'skip {name}: {path.name} missing', flush=True)
            continue
        params = torch.load(path, weights_only=False)
        items = make_items_r3(cap, cases, cfg, {})
        m = r2.load_grid_pointer(params)
        preds[name] = r2.predict_grid_pointer(m, params['mean'], params['std'], items, 'cpu')
    h = r3.head_index(20, 11)
    P = np.full((len(cap['ids']), 6), np.nan)
    for i in range(len(cap['ids'])):
        P[i, :2] = r3.peak_centroid(cap['maps_slot'][i][h], cap['grid'][i])
    preds['tool_peak'] = P
    return preds


def stage_compare(r2run: Path, out: Path, tf_run: Path) -> None:
    samples = tf.read_samples(tf_run / 'test.jsonl')
    calls = {c['id']: c for c in tf_calls(tf_run / 'test.jsonl')}
    cap = load_tf_capture(out)
    missing = set(calls) - set(cap['ids'])
    if missing:
        raise ValueError(f'{len(missing)} 3.13 test calls lack captured features')
    preds = predict_all(r2run, out, cap, calls)
    item_call = {i: cid for cid, c in calls.items() for i in c['items']}
    pos = {cid: k for k, cid in enumerate(cap['ids'])}
    rows_by = {}
    for name, P in preds.items():
        rows = {}
        for s in samples:
            texts = answer_texts(P[pos[item_call[s.id]]])
            rows[s.id] = {'id': s.id, 'finish_reason': 'eos', **tf.score(s, texts.get(s.fmt, ''))}
        rows_by[name] = rows
        (out / 'tf_eval').mkdir(exist_ok=True)
        (out / 'tf_eval' / f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows.values()) + '\n')
    conds = sorted(p.parent.name for p in (tf_run / 'eval').glob('*/coord.jsonl'))
    for c in conds:
        rows = tfa.coord_rows(tf_run, c)
        if all(s.id in rows for s in samples):
            rows_by[c] = rows
        else:
            print(f'skip 3.13 condition {c}: incomplete ({sum(s.id in rows for s in samples)}/{len(samples)})', flush=True)
    res = {'conditions': {}, 'paired_vs_base': {}, 'n_calls': len(calls), 'n_items': len(samples)}
    for name, rows in rows_by.items():
        res['conditions'][name] = {g: tfa.group_metrics(samples, rows, g) for g in tfa.GROUPS}
        if name != 'base' and 'base' in rows_by:
            res['paired_vs_base'][name] = {g: tfa.paired(samples, rows_by['base'], rows, g) for g in tfa.GROUPS}
    trained = [c for c in rows_by if c != 'base' and not c.startswith('tool')]
    res['paired_tool_vs_trained'] = {t: {c: {g: tfa.paired(samples, rows_by[c], rows_by[t], g) for g in tfa.GROUPS}
                                         for c in trained} for t in rows_by if t.startswith('tool')}
    for name, d in res['conditions'].items():
        for g in d.values():
            g.pop('curve', None)
    (out / 'compare_tf.json').write_text(json.dumps(res, indent=1, default=float))
    for name, d in res['conditions'].items():
        print(f'{name:18s}', ' '.join(f'{g}:{_hit(v):.2f}' for g, v in d.items() if v), flush=True)


def _hit(v):
    h = v['hit']
    return h['p'] if isinstance(h, dict) else (h[0] if isinstance(h, (list, tuple)) else h)


def _hit_ci(v):
    h = v['hit']
    if isinstance(h, dict):
        return h.get('ci', [np.nan, np.nan])
    return [np.nan, np.nan]


def fig_training(out: Path, figs: Path) -> None:
    p = out / 'compare_tf.json'
    if not p.exists():
        return
    res = json.loads(p.read_text())
    groups = ['refcoco', 'flat', 'clutter', 'chart_ui', 'disc_x', 'disc_y', 'grid06', 'grid10', 'grid20']
    names = [n for n in ('base', 'tool_k1', 'tool_k1_nocc', 'tool_all128') if n in res['conditions']]
    # 3.13 conditions shown: all linear layers, and the two best targeted ones (the full table is in compare_tf.json)
    names += [n for n in ('all_linear', 'mlp10_14', 'mixer16_24') if n in res['conditions']]
    cols = {'base': '#eb6834', 'tool_k1': '#1baf7a', 'tool_k1_nocc': '#008300', 'tool_all128': '#2a78d6'}
    trained_cols = ['#e87ba4', '#7a5195', '#a0522d', '#555555']
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.6), gridspec_kw={'width_ratios': [3, 3, 1.6]})
    w = 0.8 / len(names)
    k_tr = 0
    for i, n in enumerate(names):
        c = cols.get(n)
        if c is None:
            c = trained_cols[k_tr % len(trained_cols)]
            k_tr += 1
        d = res['conditions'][n]
        x = np.arange(len(groups)) + (i - len(names) / 2 + 0.5) * w
        y = np.asarray([_hit(d[g]) for g in groups])
        ci = np.asarray([_hit_ci(d[g]) for g in groups], float)
        axes[0].bar(x, y, width=w * 0.92, color=c, label=LABELS.get(n, f'fine-tuned (3.13): {n}'))
        if not np.isnan(ci).all():
            axes[0].errorbar(x, y, yerr=np.clip([y - ci[:, 0], ci[:, 1] - y], 0, None), fmt='none', ecolor='black',
                             elinewidth=0.8, capsize=1.5)
        e = np.asarray([d[g]['error']['mean'] for g in groups])
        axes[1].bar(x, e, width=w * 0.92, color=c)
        ious = [d[g]['iou']['mean'] if isinstance(d[g].get('iou'), dict) else np.nan for g in groups[:4]]
        axes[2].bar(np.arange(4) + (i - len(names) / 2 + 0.5) * w, ious, width=w * 0.92, color=c)
    axes[0].set_xticks(range(len(groups)), groups, rotation=20)
    axes[0].set_ylabel('hit rate (Wilson 95% CI)')
    axes[0].set_ylim(0, 1.05)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=9, frameon=False, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.08))
    axes[1].set_yscale('log')
    axes[1].set_xticks(range(len(groups)), groups, rotation=20)
    axes[1].set_ylabel('mean centre error (0-1000 units, log scale)')
    axes[2].set_xticks(range(4), groups[:4], rotation=20)
    axes[2].set_ylabel('mean box IoU (bbox items)')
    axes[2].set_ylim(0, 1)
    fig.suptitle('Readout tool against targeted fine-tuning (experiment 3.13) on the 3.13 test set, scored with the 3.13 scorer')
    fig.savefig(figs / 'vs_training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', figs / 'vs_training.png', flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--r2', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tf-run', type=Path, required=True)
    a = p.parse_args()
    stage_compare(a.r2, a.out, a.tf_run)


if __name__ == '__main__':
    main()
