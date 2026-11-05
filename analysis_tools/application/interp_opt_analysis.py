"""Experiment 7.1 CPU analysis: tables (results.json) and figures for the results document.

Inputs: eval/<condition>/{probe,count,coord,general,text,mech}.jsonl written by interp_opt_run.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.application import interp_opt as io_
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position.coordinate_readout_run import read_jsonl

HELD_OUT = ('qo', 'closure')
TRAINED = ('tilt', 'sat')
BENCHES = ('pope', 'textvqa', 'mmvp', 'gqa')
COLORS = {'base': '#52514e', 'lora_causal': '#2a78d6', 'lora_bidir': '#eb6834', 'lora_contrast': '#1baf7a',
          'lora_vit': '#eda100', 'lora_vit_bidir': '#e87ba4', 'vcd_all': '#6250d6', 'base_bidir': '#a8a7a2',
          'lora_bidir_causaleval': '#e34948', 'lora_vit_bidireval': '#008300', 'vcd_digit': '#9085e9'}
PROBE_FIG = ('base', 'lora_causal', 'lora_bidir', 'lora_contrast', 'lora_vit', 'lora_vit_bidir')
GENERAL_FIG = ('base_bidir', 'lora_causal', 'lora_bidir', 'lora_contrast', 'lora_vit', 'lora_vit_bidir', 'vcd_all')
MECH_FIG = ('base', 'base_bidir', 'lora_causal', 'lora_bidir', 'lora_vit', 'lora_vit_bidir')


def _load(run: Path, cond: str, name: str) -> list[dict]:
    p = run / 'eval' / cond / f'{name}.jsonl'
    return read_jsonl(p) if p.exists() else []


def _prop(rows: list[dict], key: str = 'correct') -> dict:
    k, n = sum(bool(r[key]) for r in rows), len(rows)
    p, lo, hi = io_.wilson(k, n)
    return {'k': k, 'n': n, 'p': p, 'lo': lo, 'hi': hi}


def probe_cells(rows: list[dict]) -> dict:
    out = {}
    for grp, pairs in (('heldout', HELD_OUT), ('trained', TRAINED)):
        for v in ('std', 'swap'):
            sel = [r for r in rows if r['pair'] in pairs and r['variant'] == v]
            out[f'{grp}_{v}_row1'] = _prop([r for r in sel if r['target'][0] == 0])
            out[f'{grp}_{v}_rest'] = _prop([r for r in sel if r['target'][0] > 0])
            out[f'{grp}_{v}_all'] = _prop(sel)
        for pair in pairs:
            for v in ('std', 'swap'):
                sel = [r for r in rows if r['pair'] == pair and r['variant'] == v]
                out[f'{pair}_{v}_row1'] = _prop([r for r in sel if r['target'][0] == 0])
                out[f'{pair}_{v}_rest'] = _prop([r for r in sel if r['target'][0] > 0])
    for grp, pairs in (('heldout', HELD_OUT), ('trained', TRAINED)):
        sel = [r for r in rows if r['pair'] in pairs and r['variant'] == 'swap']
        # First-row targets outside the first column (a first-column target coincides with the top-left answer).
        out[f'{grp}_swap_row1_colgt1'] = _prop([r for r in sel if r['target'][0] == 0 and r['target'][1] > 0])
        # Share of answers naming the top-left entity, whatever the target (a default answer when nothing is found).
        out[f'{grp}_swap_pred_topleft'] = _prop([dict(r, tl=r['valid'] and r['pred'] == [0, 0]) for r in sel], 'tl')
    out['invalid'] = sum(not r['valid'] for r in rows)
    return out


def paired(a: list[dict], b: list[dict], select) -> dict:
    """McNemar on items present in both conditions: b01 = a wrong & b right, b10 = a right & b wrong."""
    A = {r['id']: bool(r['correct']) for r in a if select(r)}
    B = {r['id']: bool(r['correct']) for r in b if select(r)}
    ids = sorted(set(A) & set(B))
    b01 = sum((not A[i]) and B[i] for i in ids)
    b10 = sum(A[i] and not B[i] for i in ids)
    return {'n': len(ids), 'a_wrong_b_right': b01, 'a_right_b_wrong': b10, 'p': io_.mcnemar_exact(b01, b10)}


def analyse(run: Path, conditions: list[str]) -> dict:
    res: dict = {}
    for c in conditions:
        probe = _load(run, c, 'probe')
        if not probe and not _load(run, c, 'general') and not _load(run, c, 'vitsal') and not _load(run, c, 'oddz'):
            continue
        r: dict = {}
        if probe:
            r['probe'] = probe_cells(probe)
        count = _load(run, c, 'count')
        if count:
            r['count'] = {'correct': _prop(count), 'row_ok': _prop([x for x in count if x['valid']], 'row_ok'),
                          'col_ok': _prop([x for x in count if x['valid']], 'col_ok'),
                          'invalid': sum(not x['valid'] for x in count)}
        coord = _load(run, c, 'coord')
        if coord:
            fam = {}
            for f in sorted({x['family'] for x in coord}):
                sel = [x for x in coord if x['family'] == f]
                errs = [x['error'] for x in sel if x['valid']]
                fam[f] = {**_prop(sel, 'hit'), 'mean_err': float(np.mean(errs)) if errs else float('nan')}
                if f == 'sweep_h':          # 3.12 plateau definition on the x coordinate
                    ok = [x for x in sel if x['valid']]
                    share, n = tf.plateau_share(np.array([x['center'][0] for x in ok]), np.array([x['truth_units'][0] for x in ok]))
                    fam[f]['plateau_480_520'] = share
                    fam[f]['plateau_n'] = n
            r['coord'] = fam
        gen = _load(run, c, 'general')
        if gen:
            r['general'] = {b: {'mean': float(np.mean([x['correct'] for x in gen if x['bench'] == b])),
                                'n': sum(x['bench'] == b for x in gen),
                                'truncated': sum(x['bench'] == b and x['finish_reason'] == 'length' for x in gen)}
                            for b in BENCHES}
        text = _load(run, c, 'text')
        if text:
            r['text'] = _prop(text)
        oz = _load(run, c, 'oddz')
        if oz:
            agg = {}
            for l in ('8', '16'):
                for grp, f in (('row1', lambda t: t[0] == 0), ('rest', lambda t: t[0] > 0)):
                    sel = [x for x in oz if x['variant'] == 'swap' and x['target'] is not None and f(x['target'])]
                    tz = [x[f'z{l}'][x['target'][0] * 6 + x['target'][1]] for x in sel]
                    mo = [max(z for i, z in enumerate(x[f'z{l}']) if i != x['target'][0] * 6 + x['target'][1]) for x in sel]
                    agg[f'swap_{grp}_L{l}_target_z'] = float(np.mean(tz))
                    agg[f'swap_{grp}_L{l}_margin'] = float(np.mean(np.array(tz) - np.array(mo)))
                H = np.array([x[f'z{l}'] for x in oz if x['target'] is None])
                agg[f'hom_L{l}_z_by_index'] = H.mean(0).tolist()
                agg[f'hom_n'] = int(len(H))
            r['oddz'] = agg
        if probe:
            bins = [(0, 5), (6, 11), (12, 23), (24, 35)]
            sel = [x for x in probe if x['pair'] in HELD_OUT and x['variant'] == 'swap']
            r['probe_by_earlier'] = {f'{a}-{b}': _prop([x for x in sel if a <= x['target'][0] * 6 + x['target'][1] <= b])
                                     for a, b in bins}
        vs = _load(run, c, 'vitsal')
        if vs:
            agg = {}
            for v in ('std', 'swap'):
                for key in ('popout_b9', 'popout_b12', 'popout_b16', 'salience_merger'):
                    x = np.array([r[key] for r in vs if r['variant'] == v])
                    rng = np.random.default_rng(0)
                    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
                    agg[f'{v}_{key}'] = [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            r['vitsal'] = agg
        mech = _load(run, c, 'mech')
        if mech:
            m: dict = {}
            for v in ('std', 'swap'):
                for grp, f in (('row1', lambda x: x['target'][0] == 0), ('rest', lambda x: x['target'][0] > 0)):
                    sel = [x for x in mech if x['variant'] == v and f(x)]
                    for l in ('4', '8'):
                        m[f'{v}_{grp}_L{l}_later'] = float(np.mean([x['attn'][l]['later'] for x in sel]))
                        m[f'{v}_{grp}_L{l}_earlier'] = float(np.mean([x['attn'][l]['earlier'] for x in sel]))
                        m[f'{v}_{grp}_L{l}_below'] = float(np.mean([x['attn'][l].get('below', np.nan) for x in sel]))
                    if sel and 'contrast_z' in sel[0]:
                        m[f'{v}_{grp}_contrast_z'] = float(np.mean([x['contrast_z'] for x in sel]))
                    m[f'{v}_{grp}_n'] = len(sel)
            r['mech'] = m
        res[c] = r
    # Paired comparisons on the probe.
    comps = {}
    pairs = [('lora_causal_s3', 'lora_vit_s3'), ('lora_causal_s2', 'lora_vit_s2'), ('lora_causal_s2', 'lora_bidir_s2'), ('lora_causal', 'lora_bidir'), ('lora_vit', 'lora_vit_bidir'), ('lora_vit', 'lora_vit_bidireval'), ('lora_causal', 'lora_contrast'), ('lora_causal', 'lora_vit'), ('base', 'lora_vit'), ('base', 'lora_causal'),
             ('lora_bidir', 'lora_bidir_causaleval'), ('base', 'base_bidir'), ('base', 'vcd_digit'), ('base', 'vcd_all')]
    sels = {'heldout_swap_row1': lambda r: r['pair'] in HELD_OUT and r['variant'] == 'swap' and r['target'][0] == 0,
            'heldout_swap_rest': lambda r: r['pair'] in HELD_OUT and r['variant'] == 'swap' and r['target'][0] > 0,
            'heldout_swap_row1_colgt1': lambda r: r['pair'] in HELD_OUT and r['variant'] == 'swap' and r['target'][0] == 0 and r['target'][1] > 0,
            'heldout_std_all': lambda r: r['pair'] in HELD_OUT and r['variant'] == 'std',
            'trained_swap_row1': lambda r: r['pair'] in TRAINED and r['variant'] == 'swap' and r['target'][0] == 0,
            'trained_swap_rest': lambda r: r['pair'] in TRAINED and r['variant'] == 'swap' and r['target'][0] > 0}
    for a, b in pairs:
        pa, pb = _load(run, a, 'probe'), _load(run, b, 'probe')
        if pa and pb:
            comps[f'{a}_vs_{b}'] = {k: paired(pa, pb, f) for k, f in sels.items()}
        ca, cb = _load(run, a, 'coord'), _load(run, b, 'coord')
        if ca and cb:
            for fam in sorted({x['family'] for x in ca}):
                ra = [dict(x, correct=x['hit']) for x in ca]
                rb = [dict(x, correct=x['hit']) for x in cb]
                comps.setdefault(f'{a}_vs_{b}', {})[f'coord_{fam}'] = paired(ra, rb, lambda r, fam=fam: r['family'] == fam)
        ga, gb = _load(run, a, 'general'), _load(run, b, 'general')
        if ga and gb:
            for bench in BENCHES:
                A = {x['id']: x['correct'] for x in ga if x['bench'] == bench}
                B = {x['id']: x['correct'] for x in gb if x['bench'] == bench}
                ids = sorted(set(A) & set(B))
                d = np.array([B[i] - A[i] for i in ids])
                rng = np.random.default_rng(0)
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                comps.setdefault(f'{a}_vs_{b}', {})[f'general_{bench}'] = {
                    'n': len(ids), 'delta': float(d.mean()), 'ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    res['_paired'] = comps
    return res


# ----------------------------------------------------------------------------- figures

def figures(run: Path, res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    conds = [c for c in PROBE_FIG if c in res and 'probe' in res[c]]
    # 1. probe accuracy: held-out / trained pairs x std / swap x first row / other rows.
    panels = [('heldout', 'swap'), ('heldout', 'std'), ('trained', 'swap'), ('trained', 'std')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, (grp, v) in zip(axes, panels):
        w = 0.8 / len(conds)
        for j, c in enumerate(conds):
            for i, part in enumerate(('row1', 'rest')):
                s = res[c]['probe'][f'{grp}_{v}_{part}']
                ax.bar(i + (j - (len(conds) - 1) / 2) * w, s['p'], w * 0.9, color=COLORS[c],
                       label=c if i == 0 else None)
                ax.errorbar(i + (j - (len(conds) - 1) / 2) * w, s['p'], [[s['p'] - s['lo']], [s['hi'] - s['p']]],
                            color='#0b0b0b', lw=0.8, capsize=1.5)
        n1 = res[conds[0]]['probe'][f'{grp}_{v}_row1']['n']
        n2 = res[conds[0]]['probe'][f'{grp}_{v}_rest']['n']
        ax.set_xticks([0, 1], [f'first row (n={n1})', f'rows 2-6 (n={n2})'])
        ax.set_title(f"{'held-out pairs (Q/O, closure)' if grp == 'heldout' else 'trained pairs (tilt, sat)'}: {v}",
                     fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', color='#e6e5e1', lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('row-and-column accuracy')
    axes[-1].legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / 'probe_accuracy.png', dpi=150)
    plt.close(fig)
    # 2. general benchmarks: paired delta vs base with bootstrap CI.
    gconds = [c for c in GENERAL_FIG if 'general' in res.get(c, {})]
    if gconds and 'general' in res.get('base', {}):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        w = 0.8 / len(gconds)
        for j, c in enumerate(gconds):
            comp = res['_paired'].get(f'base_vs_{c}') or {}
            for i, b in enumerate(BENCHES):
                key = f'general_{b}'
                if key not in comp:
                    continue
                d, (lo, hi) = comp[key]['delta'], comp[key]['ci']
                x = i + (j - (len(gconds) - 1) / 2) * w
                ax.bar(x, d, w * 0.9, color=COLORS[c], label=c if i == 0 else None)
                ax.errorbar(x, d, [[d - lo], [hi - d]], color='#0b0b0b', lw=0.8, capsize=1.5)
        ax.axhline(0, color='#52514e', lw=0.8)
        ax.set_xticks(range(len(BENCHES)), ['POPE', 'TextVQA', 'MMVP', 'GQA'])
        ax.set_ylabel('accuracy minus base (n=200 each)')
        ax.legend(fontsize=7, frameon=False, ncol=3)
        ax.grid(axis='y', color='#e6e5e1', lw=0.6)
        fig.tight_layout()
        fig.savefig(out / 'general_delta.png', dpi=150)
        plt.close(fig)
    # 3. attention mass of target tokens on raster-later entities (L4 / L8), swap targets.
    mconds = [c for c in MECH_FIG if 'mech' in res.get(c, {})]
    if mconds:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
        for ax, part in zip(axes, ('row1', 'rest')):
            w = 0.8 / len(mconds)
            for j, c in enumerate(mconds):
                for i, l in enumerate(('4', '8')):
                    for k, (side, hatch) in enumerate((('earlier', ''), ('below', '//'))):
                        val = res[c]['mech'][f'swap_{part}_L{l}_{side}']
                        x = i * 2 + k + (j - (len(mconds) - 1) / 2) * w
                        ax.bar(x, val, w * 0.9, color=COLORS[c], hatch=hatch, edgecolor='white',
                               label=c if (i == 0 and k == 0) else None)
            ax.set_xticks([0, 1, 2, 3], ['L4 earlier\nentities', 'L4 rows\nbelow', 'L8 earlier\nentities', 'L8 rows\nbelow'])
            ax.set_title(f"swap targets, {'first row' if part == 'row1' else 'rows 2-6'}", fontsize=10)
            ax.grid(axis='y', color='#e6e5e1', lw=0.6)
            ax.set_axisbelow(True)
        axes[0].set_ylabel('attention probability mass\n(mean over heads and target tokens)')
        axes[1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / 'attention_mass.png', dpi=150)
        plt.close(fig)
    # 4. ViT pop-out of the odd entity (held-out pairs).
    vconds = [c for c in ('base', 'lora_vit', 'lora_vit_bidir') if 'vitsal' in res.get(c, {})]
    if vconds:
        keys = ['popout_b9', 'popout_b12', 'popout_b16', 'salience_merger']
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
        for ax, key in zip(axes, keys):
            w = 0.8 / len(vconds)
            for j, c in enumerate(vconds):
                for i, v in enumerate(('std', 'swap')):
                    m, lo, hi = res[c]['vitsal'][f'{v}_{key}']
                    x = i + (j - (len(vconds) - 1) / 2) * w
                    ax.bar(x, m, w * 0.9, color=COLORS[c], label=c if i == 0 else None)
                    ax.errorbar(x, m, [[m - lo], [hi - m]], color='#0b0b0b', lw=0.8, capsize=1.5)
            ax.set_xticks([0, 1], ['std (odd has feature)', 'swap (odd lacks it)'], fontsize=8)
            ax.set_title(key.replace('popout_b', 'ViT block ').replace('salience_merger', 'merger output'), fontsize=10)
            ax.grid(axis='y', color='#e6e5e1', lw=0.6)
            ax.set_axisbelow(True)
        axes[0].set_ylabel('mean of top-10 neuron z')
        axes[3].set_ylabel('Mahalanobis salience')
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / 'vit_popout.png', dpi=150)
        plt.close(fig)
    # 5. seed replicates and the dependence on the number of raster-earlier entities (held-out swap targets).
    fams = {'lora_causal': ('lora_causal', 'lora_causal_s2', 'lora_causal_s3'), 'lora_bidir': ('lora_bidir', 'lora_bidir_s2'),
            'lora_vit': ('lora_vit', 'lora_vit_s2', 'lora_vit_s3')}
    fams = {f: [c for c in cs if c in res and 'probe' in res[c]] for f, cs in fams.items()}
    if all(fams.values()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        ax = axes[0]
        for i, (f, cs) in enumerate(fams.items()):
            for j, c in enumerate(cs):
                for k, (part, mk) in enumerate((('row1', 'o'), ('rest', 's'))):
                    sres = res[c]['probe'][f'heldout_swap_{part}']
                    ax.errorbar(i * 3 + k + (j - 1) * 0.18, sres['p'], [[sres['p'] - sres['lo']], [sres['hi'] - sres['p']]],
                                fmt=mk, color=COLORS[f], ms=7, lw=0.8, capsize=2, label=f if (j == 0 and k == 0) else None)
        ax.set_xticks([0, 1, 3, 4, 6, 7], ['first row', 'rows 2-6'] * 3, fontsize=8)
        ax.set_ylabel('held-out swap accuracy')
        ax.set_title('training seeds (each marker = one seed; circle n=36, square n=60)', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis='y', color='#e6e5e1', lw=0.6)
        ax = axes[1]
        bins = ['0-5', '6-11', '12-23', '24-35']
        for f, cs in fams.items():
            ks = [sum(res[c]['probe_by_earlier'][b]['k'] for c in cs) for b in bins]
            ns = [sum(res[c]['probe_by_earlier'][b]['n'] for c in cs) for b in bins]
            w = [io_.wilson(k, n) for k, n in zip(ks, ns)]
            ax.errorbar(range(4), [x[0] for x in w], [[x[0] - x[1] for x in w], [x[2] - x[0] for x in w]], marker='o',
                        color=COLORS[f], lw=2, ms=7, capsize=2, label=f'{f} ({len(cs)} seeds pooled)')
        ax.set_xticks(range(4), bins)
        ax.set_xlabel('number of raster-earlier entities before the target')
        ax.set_ylabel('held-out swap accuracy')
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis='y', color='#e6e5e1', lw=0.6)
        fig.tight_layout()
        fig.savefig(out / 'seeds_earlier.png', dpi=150)
        plt.close(fig)
    # 6. start-of-sequence effect: homogeneous-image oddness z by raster index (L8 and L16).
    oconds = [c for c in ('base', 'base_bidir', 'lora_causal', 'lora_bidir', 'lora_vit', 'lora_vit_bidir') if 'oddz' in res.get(c, {})]
    if oconds:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
        for ax, l in zip(axes, ('8', '16')):
            for c in oconds:
                ax.plot(range(1, 37), res[c]['oddz'][f'hom_L{l}_z_by_index'], color=COLORS[c], lw=2, label=c)
            ax.set_title(f'homogeneous images, decoder layer {l}', fontsize=10)
            ax.set_xlabel('entity raster index (1 = top-left)')
            ax.grid(color='#e6e5e1', lw=0.6)
        axes[0].set_ylabel('within-image z on oddness direction u')
        axes[1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / 'start_effect.png', dpi=150)
        plt.close(fig)
    # 7. input examples: first-row held-out swap and std items.
    items = {it.id: it for it in io_.read_items(run / 'probe.jsonl')}
    ex = [it for it in items.values() if it.meta['pair'] in HELD_OUT and it.truth['target'][0] == 0][:4]
    fig, axes = plt.subplots(1, len(ex), figsize=(3.2 * len(ex), 3.4))
    for ax, it in zip(np.atleast_1d(axes), ex):
        ax.imshow(Image.open(it.image))
        ax.set_title(f"{it.meta['pair']} {it.meta['variant']}, target row {it.truth['target'][0] + 1} "
                     f"col {it.truth['target'][1] + 1}", fontsize=8)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out / 'probe_examples.png', dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--conditions', nargs='+', required=True)
    p.add_argument('--figures', required=True)
    a = p.parse_args()
    run = Path(a.run)
    res = analyse(run, a.conditions)
    (run / 'results.json').write_text(json.dumps(res, indent=1))
    figures(run, res, Path(a.figures))
    print(json.dumps({c: {k: v for k, v in r.items() if k in ('probe',)} for c, r in res.items() if c != '_paired'})[:3000])


if __name__ == '__main__':
    main()
