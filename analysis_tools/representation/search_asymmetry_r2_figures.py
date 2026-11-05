"""Figures of experiment 2.7 round 2 (called from search_asymmetry_r2_analysis.analyze).

Colours: blue = feature-present target, orange = feature-absent target, grey = ordinary distractor (null);
heatmaps use viridis with a labelled colorbar.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry_analysis as sa1
from analysis_tools.representation import search_asymmetry_r2 as r2
from analysis_tools.representation import search_asymmetry_r2_analysis as A

BLUE, ORANGE, GREY, MUTED = A.BLUE, A.ORANGE, A.GREY, A.MUTED
ORDER = ['qo', 'term', 'plus', 'dot', 'length', 'closure', 'tilt', 'sat', 'size', 'curve', 'circsq', 'plusx']
REL_NAME = {'subset': 'B ⊂ A (pixels)', 'superset': 'A ⊂ B (pixels)', 'exclusive': 'neither'}


def _err(d):
    return [[d['p'] - d['lo']], [d['hi'] - d['p']]]


def fig_examples(out: Path, items: dict, figs: Path) -> None:
    pairs = list(r2.NEW_PAIRS)
    cols = [('std', 'subtle'), ('swap', 'subtle'), ('std', 'salient'), ('swap', 'salient')]
    by = {(it['pair_name'], it['level'], it['pair'], it['variant']): it for it in items.values()
          if it.get('set') == 'pairs'}
    fig, axes = plt.subplots(len(pairs), 4, figsize=(9.6, 2.5 * len(pairs)))
    for i, pair in enumerate(pairs):
        for j, (v, lvl) in enumerate(cols):
            it = by[(pair, lvl, 0, v)]
            ax = axes[i, j]
            ax.imshow(Image.open(out / it['path']))
            x0, y0, x1, y1 = it['meta']['instances'][it['target'][0] * 6 + it['target'][1]]['cell_box']
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ORANGE, ls='--', lw=1.5))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            name = 'feature-present' if v == 'std' else 'feature-absent'
            ax.set_title(f'{A.LABEL[pair]}\n{name}, {lvl}' if j == 0 else f'{name}, {lvl}', fontsize=8)
    fig.savefig(figs / 'examples.png'); plt.close(fig)


def fig_atlas(s: dict, figs: Path) -> None:
    at = s['atlas']
    pairs = [p for p in ORDER if f'{p}|subtle' in at]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(pairs))
    for off, v, c, lab in ((-0.17, 'present', BLUE, 'feature-present target (subtle)'),
                           (0.17, 'absent', ORANGE, 'feature-absent target (subtle)')):
        d = [at[f'{p}|subtle'][v] for p in pairs]
        ax.bar(x + off, [e['p'] for e in d], 0.32, color=c, label=lab,
               yerr=np.hstack([_err(e) for e in d]), capsize=2, error_kw={'lw': 1})
        sal = [at.get(f'{p}|salient', {}).get(v) for p in pairs]
        ax.scatter([xi + off for xi, e in zip(x, sal) if e], [e['p'] for e in sal if e], marker='D', s=26,
                   facecolor='white', edgecolor=c, lw=1.5, zorder=3,
                   label=lab.replace('subtle', 'salient level'))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{A.LABEL[p]}\n[{REL_NAME[at[p + "|subtle"]["relation"]]}]' + ('*' if at[p + '|subtle']['round'] == 2 else '')
                        for p in pairs], fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('strict accuracy (6×6)'); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    fig.savefig(figs / 'atlas.png'); plt.close(fig)


def fig_salience(s: dict, figs: Path) -> None:
    sal, at = s['salience'], s['atlas']
    names = [p for p in ORDER if p in sal] + [n for n in ('closure@salient', 'qo@salient', 'gap2', 'red') if n in sal]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [sal[n]['present'] for n in names], 0.38, color=BLUE, label='feature-present target')
    ax.bar(x + 0.2, [sal[n]['absent'] for n in names], 0.38, color=ORANGE, label='feature-absent target')
    ax.scatter(x, [sal[n]['null'] for n in names], marker='_', s=300, color='black', zorder=3,
               label='ordinary distractor (null)')
    ax.set_yscale('log'); ax.set_ylabel('local salience (ViT block 9, top-10 mean)')
    ax.set_xticks(x); ax.set_xticklabels([A.LABEL.get(n.split('@')[0], n) + (' (salient)' if '@' in n else '')
                                          for n in names], rotation=35, ha='right', fontsize=7)
    ax.legend(fontsize=8)
    ax = axes[1]
    for n in names:
        key = n.replace('@', '|') if '@' in n else f'{n}|subtle'
        if key not in at:
            continue
        for v, c in (('present', BLUE), ('absent', ORANGE)):
            ratio = sal[n][v] / sal[n]['null']
            ax.errorbar(ratio, at[key][v]['p'], yerr=_err(at[key][v]), fmt='o', color=c, ms=6, capsize=2)
            if v == 'absent':
                ax.annotate(A.LABEL.get(n.split('@')[0], n)[:14] + ('*' if '@' in n else ''), (ratio, at[key][v]['p']),
                            fontsize=6, xytext=(3, -8), textcoords='offset points', color=MUTED)
    ax.axvline(1, color=GREY, ls=':', lw=1)
    ax.set_xscale('log'); ax.set_xlabel('local salience / null salience'); ax.set_ylabel('strict accuracy')
    ax.scatter([], [], color=BLUE, label='feature-present target'); ax.scatter([], [], color=ORANGE, label='feature-absent target')
    ax.legend(fontsize=8, loc='lower right')
    fig.savefig(figs / 'salience.png'); plt.close(fig)


def fig_patch(s: dict, figs: Path) -> None:
    keys = [k for k in ('closure@salient|absent', 'r1cap:closure|absent', 'qo@salient|absent', 'r1cap:qo|absent',
                        'closure@salient|present', 'qo@salient|present') if k in s['patch']]
    if not keys:
        return
    top = 3
    fig, axes = plt.subplots(len(keys), 2 * top + 1, figsize=(2.0 * (2 * top + 1), 2.1 * len(keys)))
    axes = np.atleast_2d(axes)
    vmax = max(max(np.max(s['patch'][k]['map_target']), np.max(s['patch'][k]['map_distractor'])) for k in keys)
    for i, k in enumerate(keys):
        name, v = k.split('|')
        pair = name.split('@')[0].replace('r1cap:', '')
        level = 'salient' if '@' in name else 'subtle'
        d = A.pixel_difference_map(pair, level)
        ax = axes[i, 0]
        ax.imshow(d, cmap='Greys', vmin=0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f'{A.LABEL[pair]}, {level}\n|A−B| pixels per patch', fontsize=7)
        ax.set_ylabel('feature-absent' if v == 'absent' else 'feature-present', fontsize=8)
        p = s['patch'][k]
        for j in range(top):
            for m, (what, title) in enumerate((('map_target', 'target'), ('map_distractor', 'distractors'))):
                ax = axes[i, 1 + 2 * j + m]
                im = ax.imshow(np.array(p[what][j]), cmap='viridis', vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                ax.set_title(f'#{p["neurons"][j]} {title}\n(dev {p["dev"][j]:.0f})' if m == 0 else f'#{p["neurons"][j]} {title}',
                             fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.6, label=f'MLP activation after GELU (ViT block {s["patch"][keys[0]]["block"]})')
    fig.savefig(figs / 'patch_neurons.png'); plt.close(fig)


def fig_direction(s: dict, figs: Path) -> None:
    od = s['oddness']
    names = list(A.R1_STEMS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, site in zip(axes[:2], ('dec0', 'dec16')):
        im = ax.imshow(np.array(od['cos'][site]), cmap='viridis', vmin=-1, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([A.LABEL[n] for n in names], rotation=60, ha='right', fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([A.LABEL[n] for n in names], fontsize=7)
        ax.grid(False); ax.set_title(f'cosine of pair oddness directions, {site}', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8, label='cosine')
    ax = axes[2]
    sites = list(A.SITES)
    for n, c in (('qo', BLUE), ('closure', ORANGE), ('curve', A.AQUA), ('dot', A.VIOLET), ('term', GREY)):
        if n not in od['target_z']:
            continue
        tz = od['target_z'][n]
        ax.plot(sites, [tz.get(f'{x}|present') for x in sites], '-o', color=c, ms=4, label=f'{A.LABEL[n]}: present')
        ax.plot(sites, [tz.get(f'{x}|absent') for x in sites], '--s', color=c, ms=4, label=f'{A.LABEL[n]}: absent')
    ax.axhline(0, color=GREY, lw=1)
    ax.set_ylabel('target z on oddness direction'); ax.legend(fontsize=6, ncol=2)
    fig.savefig(figs / 'direction.png'); plt.close(fig)


def fig_novelty(s: dict, figs: Path) -> None:
    nm = s['oddness']['null_map']
    sites = [x for x in ('dec0', 'dec8', 'dec16') if x in nm]
    fig, axes = plt.subplots(1, len(sites) + 1, figsize=(4.2 * (len(sites) + 1), 3.8))
    vmax = max(np.abs(nm[x]).max() for x in sites)
    for ax, site in zip(axes, sites):
        im = ax.imshow(np.array(nm[site]).reshape(6, 6), cmap='viridis', vmin=-vmax, vmax=vmax)
        ax.set_title(f'homogeneous images, {site}', fontsize=9)
        ax.set_xticks(range(6)); ax.set_xticklabels(range(1, 7)); ax.set_yticks(range(6)); ax.set_yticklabels(range(1, 7))
        ax.set_xlabel('column'); ax.set_ylabel('row'); ax.grid(False)
    fig.colorbar(im, ax=axes[:len(sites)], shrink=0.8, label='entity z on oddness direction')
    ax = axes[-1]
    for site, c in zip(sites, (GREY, ORANGE, BLUE)):
        ax.plot(range(1, 37), nm[site], '-o', ms=3, color=c, label=site)
    ax.set_xlabel('raster index of entity'); ax.set_ylabel('entity z (no odd entity)'); ax.legend(fontsize=8)
    fig.savefig(figs / 'novelty.png'); plt.close(fig)


def fig_index(s: dict, figs: Path) -> None:
    idx = s['index']
    pairs = [p for p in ('qo', 'closure', 'tilt') if f'g6|{p}' in idx]
    fig, axes = plt.subplots(2, len(pairs) + 1, figsize=(4.6 * (len(pairs) + 1), 8.2), constrained_layout=True)
    for j, p in enumerate(pairs):
        d = idx[f'g6|{p}']
        acc = np.array(d['k']) / np.maximum(1, np.array(d['n']))
        ax = axes[0, j]
        im = ax.imshow(acc.reshape(6, 6), cmap='viridis', vmin=0, vmax=1)
        for i in range(36):
            ax.text(i % 6, i // 6, f'{d["k"][i]}/{d["n"][i]}', ha='center', va='center', fontsize=6,
                    color='white' if acc[i] < 0.6 else 'black')
        ax.set_title(f'{A.LABEL[p]}: feature-absent accuracy', fontsize=9); ax.grid(False)
        ax.set_xticks(range(6)); ax.set_xticklabels(range(1, 7)); ax.set_yticks(range(6)); ax.set_yticklabels(range(1, 7))
        ax = axes[1, j]
        h = np.array(d['wrong_answer_hist'])
        im2 = ax.imshow((h / max(1, h.sum())).reshape(6, 6), cmap='viridis', vmin=0)
        ax.set_title(f'{A.LABEL[p]}: location of wrong answers', fontsize=9); ax.grid(False)
        ax.set_xticks(range(6)); ax.set_xticklabels(range(1, 7)); ax.set_yticks(range(6)); ax.set_yticklabels(range(1, 7))
        fig.colorbar(im2, ax=ax, shrink=0.75, label='share of wrong answers')
    for j in range(len(pairs)):
        fig.colorbar(im, ax=axes[0, j], shrink=0.75, label='strict accuracy')
    # layouts: accuracy against the number of raster-earlier entities, pooled pairs
    ax = axes[0, -1]
    bins = [(0, 2), (3, 5), (6, 11), (12, 17), (18, 23), (24, 35)]
    for lay, c, mk in (('g6', GREY, 'o'), ('3x12', BLUE, 's'), ('12x3', ORANGE, '^')):
        ks, ns = np.zeros(len(bins)), np.zeros(len(bins))
        for p in pairs:
            key = f'g6|{p}' if lay == 'g6' else f'{lay}|{p}|swap'
            if key not in idx:
                continue
            for i, (k, n) in enumerate(zip(idx[key]['k'], idx[key]['n'])):
                b = next(bi for bi, (lo, hi) in enumerate(bins) if lo <= i <= hi)
                ks[b] += k; ns[b] += n
        ok = ns > 0
        w = [oe.wilson(int(k), int(n)) for k, n in zip(ks[ok], ns[ok])]
        xs = np.arange(len(bins))[ok]
        ax.errorbar(xs, [a[0] for a in w], yerr=[[a[0] - a[1] for a in w], [a[2] - a[0] for a in w]], fmt='-' + mk,
                    color=c, capsize=2, label={'g6': '6×6', '3x12': '3×12 (3 rows)', '12x3': '12×3 (12 rows)'}[lay])
    ax.set_xticks(range(len(bins))); ax.set_xticklabels([f'{a}-{b}' if a != b else str(a) for a, b in bins], fontsize=7)
    ax.set_xlabel('number of raster-earlier entities'); ax.set_ylabel('feature-absent accuracy (3 pairs pooled)')
    ax.legend(fontsize=8)
    ax = axes[1, -1]
    for lay, c, mk in (('g6', GREY, 'o'), ('3x12', BLUE, 's'), ('12x3', ORANGE, '^')):
        rows = 6 if lay == 'g6' else int(lay.split('x')[0])
        cols = 6 if lay == 'g6' else int(lay.split('x')[1])
        ks, ns = np.zeros(rows), np.zeros(rows)
        for p in pairs:
            key = f'g6|{p}' if lay == 'g6' else f'{lay}|{p}|swap'
            if key not in idx:
                continue
            for i, (k, n) in enumerate(zip(idx[key]['k'], idx[key]['n'])):
                ks[i // cols] += k; ns[i // cols] += n
        w = [oe.wilson(int(k), int(n)) for k, n in zip(ks, ns)]
        ax.errorbar(np.arange(1, rows + 1), [a[0] for a in w], yerr=[[a[0] - a[1] for a in w], [a[2] - a[0] for a in w]],
                    fmt='-' + mk, color=c, capsize=2, label={'g6': '6×6', '3x12': '3×12', '12x3': '12×3'}[lay])
    ax.set_xlabel('target row'); ax.set_ylabel('feature-absent accuracy (3 pairs pooled)'); ax.legend(fontsize=8)
    fig.savefig(figs / 'index.png'); plt.close(fig)


def fig_attention(s: dict, figs: Path) -> None:
    att = s['attention']
    names = [n for n in ('r1cap:qo', 'r1cap:closure', 'r1cap:curve', 'dot', 'term') if n in att]
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = dict(zip(names, (BLUE, ORANGE, A.AQUA, A.VIOLET, GREY)))
    for n in names:
        a = att[n]
        lab = A.LABEL[n.replace('r1cap:', '')]
        axes[0].plot(a['vit_blocks'], a['vit_self_ratio|present'], '-o', color=cols[n], ms=4, label=f'{lab}: present')
        axes[0].plot(a['vit_blocks'], a['vit_self_ratio|absent'], '--s', color=cols[n], ms=4, label=f'{lab}: absent')
        axes[1].plot(a['full_layers'], a['dec_earlier|present'], '-o', color=cols[n], ms=4)
        axes[1].plot(a['full_layers'], a['dec_earlier|absent'], '--s', color=cols[n], ms=4)
        axes[2].plot(a['full_layers'], a['ans_target|present'], '-o', color=cols[n], ms=4)
        axes[2].plot(a['full_layers'], a['ans_target|absent'], '--s', color=cols[n], ms=4)
        if a.get('ans_target_found|absent'):
            axes[2].plot(a['full_layers'], a['ans_target_found|absent'], ':', color=cols[n], lw=1.2)
    axes[0].axhline(1, color=GREY, lw=1)
    axes[0].set_xlabel('ViT block'); axes[0].set_ylabel('target self-attention / distractor self-attention')
    axes[1].set_xlabel('decoder full-attention layer'); axes[1].set_ylabel('target tokens → earlier entities (mass)')
    axes[2].set_xlabel('decoder full-attention layer'); axes[2].set_ylabel('row-digit query → target (mass)')
    axes[0].legend(fontsize=6, ncol=2)
    fig.savefig(figs / 'attention.png'); plt.close(fig)


def fig_attention_maps(out: Path, r1: Path, figs: Path, files: dict) -> None:
    """One feature-present and one feature-absent example (Q vs O): input, per-entity self-attention in the
    ViT (block 12), and the row-digit query's attention over entities (layer 16 head mean)."""
    if 'r1cap:qo' not in files:
        return
    z = np.load(files['r1cap:qo'])
    ids = [str(x) for x in z['ids']]
    VA = z['vit_attn'].astype(np.float32).mean(2)
    QA = z['ans_attn'].astype(np.float32).mean(2)
    bl = list(z['vit_attn_blocks']).index(12)
    li = list(z['full_layers']).index(16)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    for r, off in enumerate((0, 1)):
        i = off
        path = r1 / (ids[i].replace('r1|', '').replace(':', '/') + '.png')
        axes[r, 0].imshow(Image.open(path)); axes[r, 0].grid(False); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_title('feature-present (Q among O)' if off == 0 else 'feature-absent (O among Q)', fontsize=9)
        selfm = np.diag(VA[i, bl])
        im1 = axes[r, 1].imshow(selfm.reshape(6, 6), cmap='viridis'); axes[r, 1].grid(False)
        axes[r, 1].set_title('ViT block 12: self-attention of each entity\n(mass of its patches on its own patches)', fontsize=9)
        fig.colorbar(im1, ax=axes[r, 1], shrink=0.8, label='probability mass (head mean)')
        im2 = axes[r, 2].imshow(QA[i, li].reshape(6, 6), cmap='viridis'); axes[r, 2].grid(False)
        axes[r, 2].set_title('decoder layer 16: row-digit query\nattention per entity', fontsize=9)
        fig.colorbar(im2, ax=axes[r, 2], shrink=0.8, label='probability mass (head mean)')
        t = z['targets'][i]
        for ax in axes[r, 1:]:
            ax.add_patch(plt.Rectangle((t[1] - 0.5, t[0] - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2, ls='--'))
            ax.set_xticks(range(6)); ax.set_xticklabels(range(1, 7)); ax.set_yticks(range(6)); ax.set_yticklabels(range(1, 7))
    fig.savefig(figs / 'attention_maps.png'); plt.close(fig)


def fig_steer(s: dict, figs: Path, sites=('dec0', 'dec8', 'dec16')) -> None:
    st, st2 = s.get('steer', {}), s.get('steer2', {})
    if not st.get('add'):
        return
    add, add2 = st['add'], st2.get('add', {})
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
    alphas = sorted({float(c.split('|')[2]) for c in add if '|u|' in c})
    for site, c in zip(sites, (GREY, ORANGE, BLUE)):
        d = [add[f'{site}|u|{a:g}'] for a in alphas]
        axes[0].errorbar(alphas, [e['p'] for e in d], yerr=np.hstack([_err(e) for e in d]), fmt='-o', color=c,
                         capsize=2, label=f'oddness direction u, {site}')
        dr = [add2.get(f'{site}|rand|{a:g}') for a in alphas]
        if all(dr):
            axes[0].errorbar(alphas, [e['p'] for e in dr], yerr=np.hstack([_err(e) for e in dr]), fmt='--x', color=c,
                             capsize=2, label=f'random direction, {site}')
        axes[1].plot(alphas, [e['att_L16_max_delta'] for e in d], '-o', color=c, label=f'{site}, layer 16')
        axes[1].plot(alphas, [e['att_L20_max_delta'] for e in d], '--s', color=c, label=f'{site}, layer 20')
    axes[0].axhline(add['none']['p'], color='black', lw=1, ls=':', label='no edit')
    axes[0].set_xlabel('alpha (multiples of m)'); axes[0].set_ylabel('answer points to the edited entity')
    axes[0].legend(fontsize=6)
    axes[1].set_xlabel('alpha'); axes[1].set_ylabel('Δ attention, row-digit query → edited entity (best head)')
    axes[1].legend(fontsize=7)
    rem, rem2 = st.get('remove', {}), st2.get('remove', {})
    labels = [f'{x}: u' for x in sites] + ['all 5 sites: u', 'all 5 sites: random']
    x = np.arange(len(labels))
    for off, v, c, nm in ((-0.2, 'std', BLUE, 'feature-present'), (0.2, 'swap', ORANGE, 'feature-absent')):
        vals = [rem.get(f'{site}|remove_u|{v}', {}) for site in sites] + \
               [rem2.get(f'project_u_all|{v}', {}), rem2.get(f'project_rand_all|{v}', {})]
        axes[2].bar(x + off, [e.get('R', np.nan) for e in vals], 0.38, color=c, label=f'{nm} target',
                    yerr=[[e.get('R', 0) - e.get('lo', 0) for e in vals], [e.get('hi', 0) - e.get('R', 0) for e in vals]],
                    capsize=2)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=25, ha='right', fontsize=7)
    axes[2].set_ylabel('retained fraction R'); axes[2].axhline(1, color=GREY, lw=1, ls=':'); axes[2].legend(fontsize=7)
    axes[2].set_title('remove the component along u at the target', fontsize=9)
    labs = ['edit only', '+ reader heads ablated', '+ control heads ablated']
    vals = [add.get('dec0|u|1'), add2.get('dec0|u|1+ablate_reader'), add2.get('dec0|u|1+ablate_control')]
    ok = [e for e in vals if e]
    axes[3].bar(range(len(ok)), [e['p'] for e in ok], 0.5, color=[GREY, ORANGE, A.AQUA][:len(ok)],
                yerr=np.hstack([_err(e) for e in ok]), capsize=2)
    axes[3].set_xticks(range(len(ok))); axes[3].set_xticklabels(labs[:len(ok)], fontsize=7, rotation=15)
    axes[3].set_ylabel('answer points to the edited entity (dec0, alpha = 1)')
    for k, v in (('ablate_reader', 'reader'), ('ablate_control', 'control')):
        txt = ', '.join(f'{nm} R={rem2[f"{k}|{var}"]["R"]:.2f}' for var, nm in (('std', 'present'), ('swap', 'absent'))
                        if f'{k}|{var}' in rem2)
        if txt:
            axes[3].text(0.02, 0.97 - (0.07 if v == 'control' else 0), f'odd images, {v} heads ablated: {txt}',
                         transform=axes[3].transAxes, fontsize=7, va='top', color=MUTED)
    fig.savefig(figs / 'steer.png'); plt.close(fig)

def fig_index_states(s: dict, figs: Path, site: str = 'dec16') -> None:
    """Left: decoder oddness z of feature-absent targets by raster index (6x6 index set); right: accuracy against
    the margin between the target's z and the largest z among the other entities."""
    st = s.get('index_states', {})
    keys = [k for k in st if k.startswith('index_')]
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    cols = {'qo': BLUE, 'closure': ORANGE, 'tilt': A.AQUA}
    margins, correct = [], []
    for k in keys:
        d = st[k]
        pair = k.split('_')[1]
        idx = np.array(d['index']); zt = np.array(d[f'{site}|target_z']); zo = np.array(d[f'{site}|max_other_z'])
        m = [zt[idx == i].mean() if (idx == i).any() else np.nan for i in range(36)]
        axes[0].plot(range(1, 37), m, '-o', ms=3, color=cols.get(pair, GREY), label=f'{A.LABEL[pair]}: target')
        mo = [zo[idx == i].mean() if (idx == i).any() else np.nan for i in range(36)]
        axes[0].plot(range(1, 37), mo, ':', color=cols.get(pair, GREY), lw=1.2, label=f'{A.LABEL[pair]}: max other entity')
        margins.extend((zt - zo).tolist()); correct.extend([c is True for c in d['correct']])
    axes[0].set_xlabel('raster index of the feature-absent target'); axes[0].set_ylabel(f'z on oddness direction ({site})')
    axes[0].legend(fontsize=7, ncol=2)
    margins, correct = np.array(margins), np.array(correct)
    edges = np.quantile(margins, np.linspace(0, 1, 9))
    xs, ys, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (margins >= a) & (margins <= b)
        w = oe.wilson(int(correct[sel].sum()), int(sel.sum()))
        xs.append(margins[sel].mean()); ys.append(w[0]); lo.append(w[0] - w[1]); hi.append(w[2] - w[0])
    axes[1].errorbar(xs, ys, yerr=[lo, hi], fmt='-o', color=ORANGE, capsize=2)
    axes[1].axvline(0, color=GREY, lw=1, ls=':')
    axes[1].set_xlabel('target z − largest z of the other entities'); axes[1].set_ylabel('accuracy (3 pairs pooled, octile bins)')
    fig.savefig(figs / 'index_states.png'); plt.close(fig)


def fig_oddness_behaviour(s: dict, figs: Path) -> None:
    """Accuracy against the target's z on the oddness direction at the decoder input (dec0) and at decoder layer 16."""
    od, at = s['oddness']['target_z'], s['atlas']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, site in zip(axes, ('dec0', 'dec16')):
        for n, tz in od.items():
            key = n.replace('@', '|') if '@' in n else f'{n}|subtle'
            if key not in at:
                continue
            for v, c in (('present', BLUE), ('absent', ORANGE)):
                ax.errorbar(tz[f'{site}|{v}'], at[key][v]['p'], yerr=_err(at[key][v]), fmt='o', color=c, ms=6, capsize=2)
                if v == 'absent':
                    ax.annotate(A.LABEL[n.split('@')[0]][:16] + (' (sal.)' if '@' in n else ''), (tz[f'{site}|{v}'], at[key][v]['p']),
                                fontsize=6, color=MUTED, xytext=(3, -8), textcoords='offset points')
        ax.set_xlabel(f'target z on oddness direction at {site}')
        ax.axvline(0, color=GREY, lw=1, ls=':')
    axes[0].set_ylabel('strict accuracy (6×6)')
    axes[0].scatter([], [], color=BLUE, label='feature-present target'); axes[0].scatter([], [], color=ORANGE, label='feature-absent target')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.savefig(figs / 'oddness_behaviour.png'); plt.close(fig)


def make_all(s: dict, out: Path, r1: Path, figs: Path, items: dict) -> None:
    fig_examples(out, items, figs)
    fig_atlas(s, figs)
    fig_salience(s, figs)
    fig_patch(s, figs)
    fig_direction(s, figs)
    fig_novelty(s, figs)
    fig_index(s, figs)
    fig_attention(s, figs)
    fig_attention_maps(out, r1, figs, A.capture_files(out, r1))
    fig_steer(s, figs)
    fig_index_states(s, figs)
    fig_oddness_behaviour(s, figs)
