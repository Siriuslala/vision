"""Figures of experiment 3.15 round 3 (docs/analysis/position/results_2026-11-11_position_sources.md)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis_tools.position import position_sources as ps

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GREY = '#52514e'
NOF_LAYERS = (9, 10, 11, 13, 14, 15)
O_LAYERS = (2, 5, 7, 9, 10, 11, 13, 14, 15, 18, 22, 26, 30)
FAM_LABEL = {'jgrid': 'grids (jgrid)', 'single': 'single figure', 'refcoco': 'RefCOCO photos', 'grid10': '10x10 grid'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_examples(run: Path, out: Path) -> None:
    from analysis_tools.position.position_sources_run import family_cases, render
    ns = argparse.Namespace(work=str(run.parent.parent), data_dir='')
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    items = [ps.jgrid_cases(2)[0], ps.jgrid_cases(2)[1]]
    imgs = [ps.jgrid_image(c) for c in items]
    from analysis_tools.position import coordinate_mechanism_natural as cmn
    imgs.append(cmn.single_image(cmn.single_cases(1)[0]))
    try:
        rc = family_cases('refcoco', 1, ns)[0]
        im, t, _ = render(rc, ns)
        imgs.append((im, t))
        title4 = f'RefCOCO: "{rc["target"]}"'
    except Exception:                                   # data directory not configured for figures
        title4 = 'RefCOCO (not rendered)'
    titles = ['jgrid layout A', 'jgrid layout B', 'single figure', title4]
    for ax, (im, t), ti in zip(axs, imgs, titles):
        ax.imshow(im)
        W, H = im.size
        for x in range(0, W + 1, 32):
            ax.axvline(x - 0.5, color='#cccccc', lw=0.3)
        for y in range(0, H + 1, 32):
            ax.axhline(y - 0.5, color='#cccccc', lw=0.3)
        b = t['box']
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=C[2], lw=1.5))
        ax.set_title(ti, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, out / 'examples.png')


def fig_variance(summary: dict, out: Path) -> None:
    vp = summary['probes']['vp']
    fams = [f for f in ('jgrid', 'single', 'refcoco') if f in vp]
    groups = ['position', 'count', 'content', 'target_coord']
    names = [f'res0'] + [f'o{l}' for l in O_LAYERS] + ['res8', 'res16', 'res19']
    fig, axs = plt.subplots(1, len(fams), figsize=(4.6 * len(fams), 3.4), sharey=True)
    for ax, fam in zip(np.atleast_1d(axs), fams):
        xs = [n for n in names if n in vp[fam]]
        bottom = np.zeros(len(xs))
        for gi, g in enumerate(groups):
            v = np.array([vp[fam][n]['unique'].get(g, 0.0) for n in xs])
            if not np.any(v):
                continue
            ax.bar(range(len(xs)), np.clip(v, 0, None), bottom=bottom, color=C[gi], label=f'{g} (unique)', width=0.75,
                   edgecolor='white', linewidth=1)
            bottom += np.clip(v, 0, None)
        tot = np.array([vp[fam][n]['total'] for n in xs])
        ax.bar(range(len(xs)), np.clip(tot - bottom, 0, None), bottom=bottom, color='#d9d9d6', width=0.75,
               edgecolor='white', linewidth=1, label='shared between groups')
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([n.replace('o', 'DN').replace('res', 'R') for n in xs], rotation=70, fontsize=7)
        ax.set_title(FAM_LABEL[fam])
        ax.set_ylim(0, 1)
    np.atleast_1d(axs)[0].set_ylabel('cross-validated R$^2$ of top-64 PCs')
    np.atleast_1d(axs)[-1].legend(fontsize=7, loc='upper right', frameon=False)
    save(fig, out / 'variance_partition.png')


def fig_probes(summary: dict, out: Path) -> None:
    pr = summary['probes']['probe']
    names = ['res0'] + [f'o{l}' for l in O_LAYERS] + ['res8', 'res16', 'res19']
    targets = [('y_tok', 'own row (y)'), ('x_tok', 'own column (x)'), ('row_cnt|pos', 'row count beyond position'),
               ('col_cnt|pos', 'column count beyond position')]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    for ax, (t, lab) in zip(axs, targets):
        for fi, fam in enumerate(('jgrid', 'single', 'refcoco')):
            if fam not in pr:
                continue
            xs = [n for n in names if n in pr[fam] and t in pr[fam][n]]
            if not xs:
                continue
            ax.plot(range(len(xs)), [pr[fam][n][t] for n in xs], '-o', ms=4, lw=2, color=C[fi], label=FAM_LABEL[fam])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('o', 'DN').replace('res', 'R') for n in names], rotation=70, fontsize=7)
        ax.set_title(lab)
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_ylim(-0.1, 1.02)
    axs[0].set_ylabel('cross-validated R$^2$')
    axs[0].legend(frameon=False, fontsize=7)
    save(fig, out / 'probes.png')


def fig_heads(summary: dict, run: Path, out: Path) -> None:
    hs = summary['probes']['heads']
    cls = None
    p = run.parent.parent / 'position/2026-10-16_sequence_order_r2/gates/stats.json'
    if p.exists():
        cls = np.array(json.loads(p.read_text())['classes'])
    panels = [('jgrid', 'y_tok'), ('jgrid', 'x_tok'), ('jgrid', 'row_cnt|pos'), ('refcoco', 'y_tok'), ('refcoco', 'x_tok')]
    panels = [q for q in panels if q[0] in hs and q[1] in hs[q[0]]]
    fig, axs = plt.subplots(len(panels), 1, figsize=(9, 1.5 * len(panels) + 0.6))
    lin = [l for l in range(1, 33) if l % 4]
    for ax, (fam, t) in zip(np.atleast_1d(axs), panels):
        m = np.clip(np.array(hs[fam][t]), 0, 1)
        im = ax.imshow(m, aspect='auto', cmap='viridis', vmin=0, vmax=1 if t != 'row_cnt|pos' else 0.5)
        ax.set_yticks(range(len(NOF_LAYERS)))
        ax.set_yticklabels([f'L{l}' for l in NOF_LAYERS], fontsize=7)
        ax.set_xticks(range(0, 32, 4))
        ax.set_title(f'{FAM_LABEL[fam]}: {t}', fontsize=8)
        if cls is not None:
            for r, l in enumerate(NOF_LAYERS):
                for h in range(32):
                    c = int(cls[lin.index(l), h])
                    if c == 2:
                        ax.plot(h, r, 's', mfc='none', mec=C[1], ms=6, mew=1.2)
                    elif c == 3:
                        ax.plot(h, r, 'o', mfc='none', mec=C[4], ms=5, mew=1.0)
                    elif c == 1:
                        ax.plot(h, r, '.', color='white', ms=2)
        plt.colorbar(im, ax=ax, label='R$^2$', fraction=0.025, pad=0.01)
    np.atleast_1d(axs)[-1].set_xlabel('DeltaNet head')
    save(fig, out / 'head_probes.png')


def fig_amaps(run: Path, out: Path, summary: dict) -> None:
    """Read-out weights A[n, m] of three heads for one query token, drawn on the token grid."""
    import glob
    picks = []
    meta = [json.loads(l) for l in (run / 'capture' / 'meta.jsonl').open()]
    for fam in ('jgrid', 'refcoco'):
        for m in [m for m in meta if m['family'] == fam and m['cond'] == 'clean'][:4]:
            rec = torch.load(run / 'capture' / f"{m['case']}_clean.pt", weights_only=False)
            if 'amap' in rec:
                picks.append((fam, rec))
                break
    if not picks:
        return
    p = run.parent.parent / 'position/2026-10-16_sequence_order_r2/gates/stats.json'
    cls = np.array(json.loads(p.read_text())['classes']) if p.exists() else None
    lin = [l for l in range(1, 33) if l % 4]
    # one head of each class in layer 13 (or the first layer with that class)
    chosen = []
    for code, name in ((2, 'row-start reset'), (1, 'integrator'), (0, 'other')):
        for l in (13, 14, 11, 10, 9, 15):
            hs = np.where(cls[lin.index(l)] == code)[0] if cls is not None else [0]
            if len(hs):
                chosen.append((l, int(hs[0]), name))
                break
    fig, axs = plt.subplots(len(picks), len(chosen), figsize=(3.4 * len(chosen), 3.2 * len(picks)))
    axs = np.atleast_2d(axs)
    for r, (fam, rec) in enumerate(picks):
        tab = rec['table']
        H = int(tab['i'].max()) + 1
        W = int(tab['j'].max()) + 1
        q = rec['queries']
        # a query token in the lower half, inside the image interior
        qi = [k for k, n in enumerate(q) if tab['i'][n] >= H // 2 and 2 <= tab['j'][n] < W - 2]
        k = qi[0] if qi else len(q) // 2
        n = int(q[k])
        for c, (l, h, name) in enumerate(chosen):
            A = rec['amap'][l][k, h].float().numpy()
            m = np.abs(A).reshape(H, W)
            m = m / max(m.max(), 1e-12)
            ax = axs[r, c]
            im = ax.imshow(m, cmap='viridis', vmin=0, vmax=1)
            ax.plot(tab['j'][n], tab['i'][n], 'x', color=C[7], ms=8, mew=2)
            ax.set_title(f'{FAM_LABEL[fam]}\nL{l} H{h} ({name})', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='|A[n,m]| / max')
    save(fig, out / 'readout_weights.png')


def fig_nof(summary: dict, out: Path) -> None:
    nof = summary['probes']['nof']
    fams = [f for f in ('jgrid', 'single', 'refcoco') if f in nof]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, t in zip(axs, ('y_tok', 'x_tok')):
        for fi, fam in enumerate(fams):
            for name, ls in (('res19', '-'), ('o18', ':')):
                if name not in nof[fam]:
                    continue
                b = np.array(nof[fam][name][t]['by_bin'])
                ax.plot(b[:, 0], b[:, 1], ls, marker='o', ms=3, lw=2, color=C[fi],
                        label=f'{FAM_LABEL[fam]}, {"residual L19" if name == "res19" else "DeltaNet L18 output"}')
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_xlabel(f'true own {"row" if t == "y_tok" else "column"} position (0-1000)')
        ax.set_title(f'decoded {"y" if t == "y_tok" else "x"} shift under no forgetting')
    axs[0].set_ylabel('decoded shift (units)')
    axs[1].legend(frameon=False, fontsize=6)
    save(fig, out / 'nof_probe_shift.png')


def fig_roll(summary: dict, out: Path) -> None:
    roll = summary.get('roll', {})
    rows = []
    for tag, lab in (('q35', 'Qwen3.5'), ('vl', 'Qwen3-VL')):
        for fam in ('grid10', 'single', 'refcoco'):
            for cond in ('rowroll', 'colroll', 'rowroll_rDN', 'rowroll_rDN9', 'rowroll_rDN1_7', 'colroll_rDN',
                         'rowroll_dsoff'):
                k = f'{fam}:point:{cond}'
                if k in roll.get(tag, {}) and 'follow' in roll[tag][k]:
                    f = roll[tag][k]['follow']
                    rows.append((lab, fam, cond, f['f'], f['ci']))
    if not rows:
        return
    fams = ('grid10', 'single', 'refcoco')
    conds = ['rowroll', 'rowroll_rDN9', 'rowroll_rDN1_7', 'rowroll_rDN', 'colroll', 'colroll_rDN', 'rowroll_dsoff']
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
    for ax, lab in zip(axs, ('Qwen3.5', 'Qwen3-VL')):
        sub = [r for r in rows if r[0] == lab]
        cs = [c for c in conds if any(r[2] == c for r in sub)]
        w = 0.8 / max(len(cs), 1)
        for ci, c in enumerate(cs):
            for fi, fam in enumerate(fams):
                rr = [r for r in sub if r[1] == fam and r[2] == c]
                if not rr:
                    continue
                f, ci_ = rr[0][3], rr[0][4]
                ax.bar(fi + (ci - len(cs) / 2 + 0.5) * w, f, w * 0.9, color=C[conds.index(c)], label=c if fi == 0 else None)
                ax.errorbar(fi + (ci - len(cs) / 2 + 0.5) * w, f, yerr=[[f - ci_[0]], [ci_[1] - f]], color=GREY, lw=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels([FAM_LABEL[f] for f in fams])
        ax.axhline(0, color=GREY, lw=0.5)
        ax.axhline(1, color=GREY, lw=0.5, ls=':')
        ax.set_title(lab)
        ax.legend(frameon=False, fontsize=7)
    axs[0].set_ylabel('following ratio of the sequence-implied shift')
    save(fig, out / 'roll_following.png')


def fig_steer(summary: dict, out: Path) -> None:
    """Dose-response of answers to added directions (DeltaNet mixer outputs of layers 9-15, all image tokens)."""
    st = summary.get('steer', {})
    if not st:
        return
    scales = [-10, -5, -2, -1, 1, 2, 5, 10]
    panels = [('jgrid:rowcol', 'row', 'jgrid: row ordinal shift (rows)'),
              ('jgrid:point', 'y', 'jgrid: y shift (units)'),
              ('jgrid:point', 'x', 'jgrid: x shift (units)'),
              ('refcoco:point', 'y', 'RefCOCO: y shift (units)')]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, (key, comp, lab) in zip(axs, panels):
        for di, d in enumerate(('rowcnt', 'ypos', 'colcnt', 'xpos')):
            xs, ys, lo, hi = [], [], [], []
            for sc in scales:
                k = f'{key}:{d}{sc:+g}'
                if k in st and comp in st[k]['shift']:
                    v = st[k]['shift'][comp]
                    xs.append(sc); ys.append(v[0]); lo.append(v[0] - v[1]); hi.append(v[2] - v[0])
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], marker='o', ms=4, lw=2, color=C[di], label=d, capsize=2)
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_xlabel('steps added (multiples of the fitted unit step)')
        ax.set_title(lab, fontsize=9)
    axs[0].legend(frameon=False, fontsize=7)
    save(fig, out / 'steering.png')


def fig_vlflow(summary: dict, out: Path) -> None:
    vf = summary.get('vlflow', {})
    if not vf:
        return
    fams = list(vf)
    fig, axs = plt.subplots(len(fams), 2, figsize=(13, 2.8 * len(fams)))
    axs = np.atleast_2d(axs)
    for r, fam in enumerate(fams):
        d = vf[fam]
        for c, kind in enumerate(('step', 'win')):
            keys = [k for k in d.get(f'{kind}_d0', {}) if k not in ('n_values', 'diff_norm')]
            m = np.array([[d.get(f'{kind}_d{g}', {}).get(k, [np.nan])[0] for k in keys] for g in range(3)])
            ax = axs[r, c]
            im = ax.imshow(np.clip(m, -0.1, 1), cmap='viridis', vmin=-0.1, vmax=1, aspect='auto')
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if np.isfinite(m[i, j]):
                        ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center', fontsize=5.5,
                                color='white' if m[i, j] < 0.5 else 'black')
            ax.set_yticks(range(3))
            ax.set_yticklabels(['hundreds', 'tens', 'units'])
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=70, fontsize=6)
            ax.set_title(f'Qwen3-VL {fam}: {"studied step only" if kind == "step" else "window from 3 tokens before"}',
                         fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label='recovery r')
    save(fig, out / 'vl_flow.png')


def make_all(run: Path, out: Path, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in (lambda: fig_examples(run, out), lambda: fig_variance(summary, out), lambda: fig_probes(summary, out),
              lambda: fig_heads(summary, run, out), lambda: fig_amaps(run, out, summary),
              lambda: fig_nof(summary, out), lambda: fig_roll(summary, out), lambda: fig_steer(summary, out),
              lambda: fig_vlflow(summary, out), lambda: fig_schematic(out)):
        try:
            f()
        except KeyError as e:                       # a part whose data has not been produced yet
            print('figure skipped, missing', e)


def fig_schematic(out: Path) -> None:
    """Process schematic of coordinate readout in both models (a drawing summarising the measurements)."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.2))
    cols = {'tok': '#dbe8f8', 'dn': '#fde3d6', 'read': '#d6f1e7', 'out': '#fbeccc'}

    def box(ax, x, y, w, h, text, kind):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=cols[kind], ec=GREY, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=6.8)

    def arrow(ax, x0, y0, x1, y1, label=None):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=GREY, lw=1.2))
        if label:
            ax.text((x0 + x1) / 2 + 0.01, (y0 + y1) / 2, label, fontsize=6.5, color=GREY)

    ax = axs[0]
    ax.set_title('Qwen3.5-9B (Gated DeltaNet + full attention)', fontsize=10)
    box(ax, 0.05, 0.86, 0.9, 0.1, 'ViT + 2x2 merger: every image token carries its own row/column features\n'
        '(token-attached position; moves with the token under a raster roll)', 'tok')
    box(ax, 0.05, 0.62, 0.42, 0.18, 'DeltaNet L1-15, long-memory heads:\n'
        'o_n = sum_m A[n,m] v_m averages the\nvalues of tokens 1-4 rows above\n-> vertical code (sequence-dependent;\nin grids also the row count)', 'dn')
    box(ax, 0.53, 0.62, 0.42, 0.18, 'DeltaNet row-start reset heads:\nreset where the token itself\nlooks like a left edge, then\naverage the current row\n-> horizontal code anchored to content', 'dn')
    box(ax, 0.05, 0.40, 0.9, 0.14, 'image-token residual at L16-24 = token-attached position + DeltaNet position code\n'
        '(vertical code shared by grids, figures and photos; horizontal code image-type specific)', 'tok')
    box(ax, 0.05, 0.18, 0.42, 0.15, 'grids: L20 H11 reads the target tokens\n(count-like discrete readout)', 'read')
    box(ax, 0.53, 0.18, 0.42, 0.15, 'photos: edge heads (L20/L24) read the\nbox edges; two edges summed in residual', 'read')
    box(ax, 0.05, 0.02, 0.9, 0.1, 'format tokens -> hundreds digit (image read) -> tens/units via L25-32 short convolution', 'out')
    for x0, x1 in ((0.26, 0.26), (0.74, 0.74)):
        arrow(ax, x0, 0.86, x1, 0.80)
        arrow(ax, x0, 0.62, x1, 0.54)
        arrow(ax, x0, 0.40, x1, 0.33)
        arrow(ax, x0, 0.18, x1, 0.12)
    ax = axs[1]
    ax.set_title('Qwen3-VL-8B (full attention only, DeepStack)', fontsize=10)
    box(ax, 0.05, 0.86, 0.9, 0.1, 'ViT + merger, plus DeepStack: ViT intermediate features added to image tokens\n'
        'after decoder layers 1-3 (token-attached position, strengthened)', 'tok')
    box(ax, 0.05, 0.62, 0.9, 0.18, 'causal full attention among image tokens builds a sequence-dependent vertical code;\n'
        'used when DeepStack is removed (row-roll following 0.39-0.57),\nweakly with DeepStack (0.02-0.23);'
        ' no sequence-dependent horizontal code', 'dn')
    box(ax, 0.05, 0.40, 0.9, 0.14, 'image-token residual = mostly token-attached position', 'tok')
    box(ax, 0.05, 0.18, 0.9, 0.15, 'readers: heads of layers 25-28 (largest single head L25 H13); heads with the largest\n'
        'in-target attention (L22 H10, L24 H30) carry little causal signal on their own', 'read')
    box(ax, 0.05, 0.02, 0.9, 0.1, 'format tokens -> hundreds (half read at its own step) -> tens/units copied from earlier\n'
        'answer-token key/values (no short convolution)', 'out')
    for y0, y1 in ((0.86, 0.80), (0.62, 0.54), (0.40, 0.33), (0.18, 0.12)):
        arrow(ax, 0.5, y0, 0.5, y1)
    for a in axs:
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.axis('off')
    save(fig, out / 'process_schematic.png')
