"""Experiment 3.14 round 2 (CPU): reading heads, attention mass on target / target row / target column, figures.

Inputs: heads_<tag>.jsonl and maps_<tag>/<id>.npz written by causal_vs_linear_heads (tags q35_none, vl_ds_on).

* head recovery: for every single head (and every whole layer, all heads) the recovered fraction
  r = (LD_e - LD_A) / (LD_B - LD_A) of the first-digit logit difference, on values of the displacement axis of
  the decoder-only splice pair (x values for horizontal moves, y values for vertical moves);
* reading heads: the TOP_K heads with the largest mean r per model, condition and axis;
* attention mass: for each answer and head, the share of the head's image attention (at the query predicting
  the value's first digit) that falls on the target tokens, on the target's token rows (target excluded), on the
  target's token columns (target excluded), on the target-row tokens after / before the target in raster order and
  on the first token column; the reading-head summary weights the TOP_K heads by their mean r (negative r -> 0);
* figures: head recovery ranking, target-centred mean attention maps, example overlays, mass bars, and two
  schematic figures for the document (splice protocol; simulated Gated DeltaNet accumulators).

Layers are 1-based.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_tools.position import causal_vs_linear as cv
from analysis_tools.position import coordinate_flow as cf
from analysis_tools.representation.sees_right import read_jsonl

COLORS = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
          'green': '#008300', 'violet': '#6250d6', 'grey': '#8a8984'}
AXIS_OF_KEY = {'x': 'h', 'column': 'h', 'y': 'v', 'row': 'v'}
TAGS = {'q35_none': 'Qwen3.5-9B', 'vl_ds_on': 'Qwen3-VL-8B'}
MODEL_COLOR = {'q35_none': COLORS['blue'], 'vl_ds_on': COLORS['green']}
GROUPS = ('target', 'row', 'col', 'row_after', 'row_before', 'col0', 'row_start', 'row_end')
HALF = 9                       # target-centred window: offsets -HALF .. +HALF tokens around the target's top-left token


def boot(x, n: int = 2000, seed: int = 0) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'n': 0}
    bs = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return {'mean': float(x.mean()), 'lo': float(np.quantile(bs, 0.025)), 'hi': float(np.quantile(bs, 0.975)),
            'n': int(len(x))}


def recovery(v: dict, cond: str) -> float | None:
    """r of one value record; None when the pair does not change the first digit or LD_B - LD_A is below 2."""
    lda, ldb = v['LD'].get('none'), v['LD_B']
    if lda is None or cond not in v['LD'] or v['dA'] is None or v['dA'] == v['dB'] or abs(ldb - lda) < 2:
        return None
    return (v['LD'][cond] - lda) / (ldb - lda)


def axis_values(rec: dict) -> list[tuple[str, int, dict]]:
    """(axis, value index, value record) for the splice pairs, restricted to values of the displacement axis."""
    keys = [m['key'] for m in rec['values_meta']]
    out = []
    for pr in rec['pairs']:
        for v in pr['values']:
            if AXIS_OF_KEY.get(v['key'], v['axis']) == pr['axis']:
                out.append((pr['axis'], keys.index(v['key']), v))
    return out


def head_name(l: int, h: int) -> str:
    return f'h{l}_{h}'


def parse_head(name: str) -> tuple[int, int]:
    l, h = name[1:].split('_')
    return int(l), int(h)


def group_masks(grid: tuple[int, int], target: list[int]) -> dict[str, np.ndarray]:
    """Profile groups of causal_vs_linear plus row_start / row_end: first / last token column of the target rows."""
    m = cv.profile_masks(grid, target)
    H, W = grid
    cc = np.arange(H * W) % W
    rows = m['row'] | m['target']
    m['row_start'] = rows & (cc == 0) & ~m['target']
    m['row_end'] = rows & (cc == W - 1) & ~m['target']
    return {g: m[g] for g in GROUPS}


def head_summary(recs: list[dict]) -> dict:
    """{cond|axis: {condition name: boot of r}} over all patching rows."""
    acc: dict = {}
    for r in recs:
        for axis, _, v in axis_values(r):
            d = acc.setdefault(f"{r['cond']}|{axis}", {})
            for name in v['LD']:
                if name != 'none':
                    d.setdefault(name, []).append(recovery(v, name))
    return {k: {n: boot(x) for n, x in d.items()} for k, d in acc.items()}


def reading_heads(hs: dict, key: str, top_k: int) -> list[tuple[str, float]]:
    d = hs[key]
    singles = sorted(((n, b['mean']) for n, b in d.items() if n.startswith('h') and np.isfinite(b['mean'])),
                     key=lambda t: -t[1])
    return singles[:top_k]


def mass_records(run: Path, tag: str, recs: list[dict], heads_by_key: dict, layer_index: dict[int, int]) -> dict:
    """Per cond|axis: per-answer group shares for every reading head and the r-weighted reading-head mean."""
    out: dict = {}
    for r in recs:
        z = np.load(run / f'maps_{tag}' / f"{r['id']}.npz")
        maps = z['maps'].astype(np.float32)                        # (layers, heads, Q, Nv)
        masks = group_masks(tuple(r['grid']), list(r['target']))
        seen = set()
        for axis, k, _ in axis_values(r):
            key = f"{r['cond']}|{axis}"
            if key not in heads_by_key or (key, k) in seen:
                continue
            seen.add((key, k))
            heads = heads_by_key[key]
            w = np.array([max(rv, 0.0) for _, rv in heads])
            shares = []
            for name, _ in heads:
                l, h = parse_head(name)
                a = maps[layer_index[l], h, k]
                tot = a.sum()
                shares.append({g: float(a[m].sum() / tot) for g, m in masks.items()} | {'image': float(tot)})
            mix = {g: float(np.dot(w, [s[g] for s in shares]) / w.sum()) if w.sum() > 0 else float('nan')
                   for g in list(GROUPS) + ['image']}
            out.setdefault(key, []).append({'id': r['id'], 'heads': shares, 'mix': mix,
                                            'sizes': {g: int(m.sum()) for g, m in masks.items()},
                                            'n_img': int(len(next(iter(masks.values()))))})
    return out


def centred_map(a: np.ndarray, grid: tuple[int, int], target: list[int]) -> np.ndarray:
    """(2*HALF+1, 2*HALF+1) window of a (Nv,) map (normalised to image mass 1) around the target's top-left token."""
    H, W = grid
    a = (a / a.sum()).reshape(H, W)
    r0, c0 = min(t // W for t in target), min(t % W for t in target)
    out = np.full((2 * HALF + 1, 2 * HALF + 1), np.nan)
    for i in range(-HALF, HALF + 1):
        for j in range(-HALF, HALF + 1):
            rr, cc = r0 + i, c0 + j
            if 0 <= rr < H and 0 <= cc < W:
                out[i + HALF, j + HALF] = a[rr, cc]
    return out


def summarise(run: Path, top_k: int) -> dict:
    S: dict = {'top_k': top_k, 'models': {}}
    for tag in TAGS:
        path = run / f'heads_{tag}.jsonl'
        if not path.exists():
            continue
        recs = read_jsonl(path)
        hs = head_summary(recs)
        any_rec = recs[0]
        layers = sorted({parse_head(n)[0] for k in hs for n in hs[k] if n.startswith('h')} |
                        {int(n[1:]) for k in hs for n in hs[k] if n.startswith('L')})
        layer_index = {l: i for i, l in enumerate(layers)}
        heads_by_key = {k: reading_heads(hs, k, top_k) for k in hs}
        mass = mass_records(run, tag, recs, heads_by_key, layer_index)
        M: dict = {'n_answers': len(recs), 'correct': int(sum(r['correct'] for r in recs)), 'keys': {}}
        for key in hs:
            d = hs[key]
            layer_r = {n: d[n]['mean'] for n in d if n.startswith('L')}
            rows = mass.get(key, [])
            entry = {'n_values': d.get('full_all', {}).get('n', 0), 'full_all': d.get('full_all'),
                     'reading_heads': [{'head': n, 'r': rv, 'r_ci': [d[n]['lo'], d[n]['hi']]} for n, rv in heads_by_key[key]],
                     'sum_top_k': float(sum(rv for _, rv in heads_by_key[key])),
                     'best_layers': sorted(layer_r.items(), key=lambda t: -t[1])[:4],
                     'mix_share': {g: boot([x['mix'][g] for x in rows]) for g in list(GROUPS) + ['image']},
                     'top1_share': {g: boot([x['heads'][0][g] for x in rows]) for g in list(GROUPS) + ['image']},
                     'per_head_share': [{g: float(np.mean([x['heads'][i][g] for x in rows])) for g in GROUPS}
                                        for i in range(len(heads_by_key[key]))] if rows else [],
                     'token_fraction': ({g: float(np.mean([x['sizes'][g] / x['n_img'] for x in rows])) for g in GROUPS}
                                        if rows else {})}
            M['keys'][key] = entry
        S['models'][tag] = M
        S['models'][tag]['_layer_index'] = layer_index
    return S


# ----------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def fig_head_ranking(S: dict, out: Path, cond: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, axis, title in ((axes[0], 'h', 'x / column value'), (axes[1], 'v', 'y / row value')):
        off = 0
        ticks, labels = [], []
        for tag, M in S['models'].items():
            e = M['keys'].get(f'{cond}|{axis}')
            if e is None:
                continue
            for i, hd in enumerate(e['reading_heads']):
                x = off + i
                lo, hi = hd['r_ci']
                ax.bar(x, hd['r'], color=MODEL_COLOR[tag], width=0.7)
                ax.errorbar(x, hd['r'], yerr=[[hd['r'] - lo], [hi - hd['r']]], color='#333333', lw=1, capsize=2)
                l, h = parse_head(hd['head'])
                ticks.append(x)
                labels.append(f'L{l} H{h}')
            fa = e['full_all']['mean']
            ax.hlines(fa, off - 0.4, off + len(e['reading_heads']) - 0.6, color=MODEL_COLOR[tag], ls='--', lw=1.2)
            ax.text(off + len(e['reading_heads']) - 0.6, fa, f' all heads {fa:.2f}', fontsize=7, va='bottom', ha='right')
            off += len(e['reading_heads']) + 1
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.axhline(0, color='#999999', lw=0.6)
        ax.set_title(title)
    axes[0].set_ylabel('recovered fraction r (single head)')
    import matplotlib.patches as mp
    axes[1].legend(handles=[mp.Patch(color=MODEL_COLOR[t], label=TAGS[t]) for t in S['models']], frameon=False,
                   loc='upper right')
    fig.tight_layout()
    fig.savefig(out / f'head_ranking_{cond}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _centred_mean(run: Path, tag: str, recs: list[dict], key: str, heads: list[tuple[str, float]],
                  layer_index: dict, weighted: bool) -> np.ndarray:
    w = np.array([max(rv, 0.0) for _, rv in heads]) if weighted else np.eye(len(heads))[0]
    acc = []
    for r in recs:
        if r['cond'] != key.split('|')[0]:
            continue
        z = np.load(run / f'maps_{tag}' / f"{r['id']}.npz")
        maps = z['maps'].astype(np.float32)
        for axis, k, _ in axis_values(r):
            if axis != key.split('|')[1]:
                continue
            m = sum(wi * centred_map(maps[layer_index[parse_head(n)[0]], parse_head(n)[1], k], tuple(r['grid']),
                                     list(r['target'])) for wi, (n, _) in zip(w, heads) if wi > 0) / w.sum()
            acc.append(m)
            break
    return np.nanmean(np.stack(acc), 0)


def fig_centred_maps(S: dict, run: Path, out: Path, cond: str, weighted: bool) -> None:
    import matplotlib.patches as mp
    plt = _plt()
    tags = list(S['models'])
    fig, axes = plt.subplots(len(tags), 2, figsize=(8.2, 3.9 * len(tags)), squeeze=False)
    for i, tag in enumerate(tags):
        recs = read_jsonl(run / f'heads_{tag}.jsonl')
        li = S['models'][tag]['_layer_index']
        for j, (axis, qname) in enumerate((('h', 'x / column query'), ('v', 'y / row query'))):
            key = f'{cond}|{axis}'
            e = S['models'][tag]['keys'][key]
            heads = [(h['head'], h['r']) for h in e['reading_heads']]
            m = _centred_mean(run, tag, recs, key, heads, li, weighted)
            ax = axes[i, j]
            im = ax.imshow(m, cmap='viridis', vmin=0, extent=(-HALF - .5, HALF + .5, HALF + .5, -HALF - .5))
            ax.add_patch(mp.Rectangle((-0.5, -0.5), 2, 2, fill=False, ec='#ff3b30', lw=1.6))
            ax.set_xlabel('token column offset from target')
            ax.set_ylabel('token row offset from target')
            l, h = parse_head(heads[0][0])
            who = f'top-{len(heads)} reading heads, r-weighted' if weighted else f'top reading head L{l} H{h}'
            ax.set_title(f'{TAGS[tag]}, {qname}\n{who}', fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label('share of image attention per token')
    fig.tight_layout()
    fig.savefig(out / f'centred_maps_{cond}_{"weighted" if weighted else "top1"}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_examples(S: dict, run: Path, flow_run: Path, out: Path, cond: str, n_examples: int = 2) -> None:
    """Overlays of the top reading head's map on the input image for a few answers (both models, both queries)."""
    import matplotlib.patches as mp
    from PIL import Image
    plt = _plt()
    cases = {c['id']: c for c in json.loads((flow_run / 'cases.json').read_text())}
    tags = list(S['models'])
    recs = {t: {r['id']: r for r in read_jsonl(run / f'heads_{t}.jsonl')} for t in tags}
    def interior(r):                     # targets away from the left/right image border show the row band best
        W = r['grid'][1]
        return 4 <= min(t % W for t in r['target']) <= W - 6
    ids = [i for i in recs[tags[0]] if all(i in recs[t] for t in tags) and recs[tags[0]][i]['cond'] == cond
           and interior(recs[tags[0]][i])]
    ids = ids[:n_examples]
    fig, axes = plt.subplots(len(ids), 2 * len(tags), figsize=(4.0 * 2 * len(tags), 3.9 * len(ids)), squeeze=False)
    for i, cid in enumerate(ids):
        img = np.asarray(Image.open(cases[cid]['image_path']).convert('RGB'))
        for t_i, tag in enumerate(tags):
            r = recs[tag][cid]
            li = S['models'][tag]['_layer_index']
            z = np.load(run / f'maps_{tag}' / f'{cid}.npz')
            maps = z['maps'].astype(np.float32)
            H, W = r['grid']
            keys = [m['key'] for m in r['values_meta']]
            for a_i, (axis, qname) in enumerate((('h', 'x query'), ('v', 'y query'))):
                ax = axes[i, 2 * t_i + a_i]
                head = S['models'][tag]['keys'][f'{cond}|{axis}']['reading_heads'][0]['head']
                l, h = parse_head(head)
                k = next(j for j, kk in enumerate(keys) if AXIS_OF_KEY.get(kk) == axis)
                a = maps[li[l], h, k]
                a = (a / a.sum()).reshape(H, W)
                ax.imshow(img, extent=(0, W, H, 0))
                im = ax.imshow(a, cmap='viridis', alpha=0.6, extent=(0, W, H, 0), vmin=0)
                x0, y0, x1, y1 = cases[cid]['truth']['box']
                ax.add_patch(mp.Rectangle((x0 / 32, y0 / 32), (x1 - x0) / 32, (y1 - y0) / 32, fill=False,
                                          ec='#ff3b30', lw=1.4))
                ax.set_title(f'{TAGS[tag]} L{l} H{h}, {qname}', fontsize=8.5)
                ax.axis('off')
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label('share of image attention', fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f'examples_{cond}.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_mass(S: dict, out: Path, conds: list[str]) -> None:
    plt = _plt()
    groups = (('target', 'target'), ('row', 'target rows'), ('col', 'target columns'), ('row_after', 'row, after target'))
    fig, axes = plt.subplots(len(conds), 2, figsize=(10, 3.2 * len(conds)), squeeze=False, sharey=True)
    for ci, cond in enumerate(conds):
        for j, (axis, qname) in enumerate((('h', 'x / column query'), ('v', 'y / row query'))):
            ax = axes[ci, j]
            for t_i, (tag, M) in enumerate(S['models'].items()):
                e = M['keys'].get(f'{cond}|{axis}')
                if e is None:
                    continue
                for g_i, (g, _) in enumerate(groups):
                    b = e['mix_share'][g]
                    x = g_i + (t_i - 0.5) * 0.36
                    ax.bar(x, b['mean'], width=0.34, color=MODEL_COLOR[tag], label=TAGS[tag] if g_i == 0 else None)
                    ax.errorbar(x, b['mean'], yerr=[[b['mean'] - b['lo']], [b['hi'] - b['mean']]], color='#333333',
                                lw=1, capsize=2)
                    frac = e['token_fraction'][g]
                    ax.hlines(frac, x - 0.17, x + 0.17, color='#111111', lw=1.2, ls=':')
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([n for _, n in groups])
            ax.set_title(f'{cond}: {qname}')
            if j == 0:
                ax.set_ylabel('share of image attention\n(top-5 reading heads, r-weighted)')
    axes[0, 1].legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(out / 'mass_groups.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_protocol(out: Path) -> None:
    """Schematic of the decoder-only splice and of r (illustration, not data)."""
    import matplotlib.patches as mp
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw={'width_ratios': [1, 1, 1.35]})
    H = W = 8
    tgtA = [(3, 3), (3, 4), (4, 3), (4, 4)]
    tgtB = [(5, 3), (5, 4), (6, 3), (6, 4)]

    def grid(ax, red, title, mark=None):
        for r in range(H):
            for c in range(W):
                fc = '#ff3b30' if (r, c) in red else '#f2f2f2'
                ax.add_patch(mp.Rectangle((c, r), 0.92, 0.92, fc=fc, ec='none'))
        if mark:
            for (r, c), col in mark.items():
                ax.add_patch(mp.Rectangle((c, r), 0.92, 0.92, fc='none', ec=col, lw=1.6))
        ax.set_xlim(-0.2, W + 0.1)
        ax.set_ylim(H + 0.1, -0.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=9)
    grid(axes[0], tgtA, 'run A: original image tokens\n(answer: row of red target)')
    D = {p: '#111111' for p in tgtA + tgtB}
    grid(axes[1], tgtB, 'run B (splice): only tokens in D\ntake run-B embeddings; rest = run A', D)
    a = axes[2]
    a.axis('off')
    txt = ('1. Prefill run A and run B; teacher-force run A\'s answer in both.\n'
           '2. Patched run: run A, but the keys/values of one token group G\n'
           '    in the full-attention layers are copied from run B.\n'
           '3. At the query predicting the first digit, LD = z[d_B] - z[d_A].\n'
           '4. r_G = (LD_patched - LD_A) / (LD_B - LD_A):\n'
           '    r = 1  -> group G carries the whole run-B / run-A difference;\n'
           '    r = 0  -> group G carries none of it.\n\n'
           'Black frames: D = target tokens of run A and run B.\n'
           'Tokens before the first token of D are identical in both runs\n'
           '(causal decoder), so r = 0 there by construction.')
    a.text(0.0, 0.95, txt, va='top', ha='left', fontsize=9, family='monospace')
    fig.tight_layout()
    fig.savefig(out / 'splice_protocol.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def simulate_accumulators(H: int = 6, W: int = 8, target: tuple[int, int] = (3, 4), beta: float = 0.5,
                          alpha_reset: float = 0.0) -> dict[str, np.ndarray]:
    """Scalar simulation of three idealised Gated DeltaNet heads on a raster sequence of H x W tokens.

    column accumulator: alpha = alpha_reset at the first token of each token row, 1 elsewhere; every token writes
      value 1 with strength beta (state m_t = (1 - beta) * alpha_t * m_{t-1} + beta), i.e. a saturating count of
      tokens since the row start;
    row accumulator: alpha = 1 everywhere; only row-start tokens write value 1 (count of row starts so far);
    target-in-row flag: alpha = alpha_reset at row starts, 1 elsewhere; only target tokens write value 1, so the
      flag is non-zero exactly on the target and on the tokens after it in the same token row.
    Returns (H, W) maps of the three states after each token. Illustration only: real heads have vector states.
    """
    col = np.zeros((H, W))
    row = np.zeros((H, W))
    flag = np.zeros((H, W))
    mc = mr = mf = 0.0
    for r in range(H):
        for c in range(W):
            a = alpha_reset if c == 0 else 1.0
            mc = (1 - beta) * a * mc + beta
            if c == 0:
                mr = mr + 1.0
            mf = a * mf
            if (r, c) == target:
                mf = (1 - beta) * mf + beta
            col[r, c], row[r, c], flag[r, c] = mc, mr, mf
    return {'column accumulator': col, 'row accumulator': row, 'target-in-row flag': flag}


def fig_accumulators(out: Path) -> None:
    import matplotlib.patches as mp
    plt = _plt()
    sim = simulate_accumulators()
    H, W = 6, 8
    tr, tc = 3, 4
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    titles = {'column accumulator': '(a) column accumulator\nreset at each row start',
              'row accumulator': '(b) row accumulator\n+1 at each row start, never reset',
              'target-in-row flag': '(c) "target seen in this row"\nwritten by target, reset at row start'}
    for ax, (name, m) in zip(axes[:3], sim.items()):
        im = ax.imshow(m, cmap='viridis')
        ax.add_patch(mp.Rectangle((tc - .5, tr - .5), 1, 1, fill=False, ec='#ff3b30', lw=2))
        ax.set_title(titles[name], fontsize=9)
        ax.set_xlabel('token column')
        ax.set_ylabel('token row')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label('state value (simulated)')
    conj = (sim['target-in-row flag'] > 0).astype(float) * sim['row accumulator']
    im = axes[3].imshow(np.where(conj > 0, conj, np.nan), cmap='viridis', vmin=0, vmax=H)
    axes[3].add_patch(mp.Rectangle((tc - .5, tr - .5), 1, 1, fill=False, ec='#ff3b30', lw=2))
    axes[3].set_title('(d) tokens holding both "target in my row"\nand the row index (value = row index)', fontsize=9)
    axes[3].set_xlabel('token column')
    axes[3].set_ylabel('token row')
    axes[3].set_xlim(-.5, W - .5)
    axes[3].set_ylim(H - .5, -.5)
    cb = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.03)
    cb.set_label('row index carried (simulated)')
    fig.tight_layout()
    fig.savefig(out / 'accumulators_sim.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--round1', type=Path, required=True)
    p.add_argument('--flow-run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--top-k', type=int, default=5)
    args = p.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    fig_protocol(args.figures)
    fig_accumulators(args.figures)
    S = summarise(args.run, args.top_k)
    if len(S['models']) < 2:
        raise FileNotFoundError(f'Need heads_q35_none.jsonl and heads_vl_ds_on.jsonl in {args.run}')
    conds = sorted({k.split('|')[0] for M in S['models'].values() for k in M['keys']})
    for cond in conds:
        fig_head_ranking(S, args.figures, cond)
        fig_centred_maps(S, args.run, args.figures, cond, weighted=False)
        fig_centred_maps(S, args.run, args.figures, cond, weighted=True)
        fig_examples(S, args.run, args.flow_run, args.figures, cond)
    fig_mass(S, args.figures, conds)
    for M in S['models'].values():
        M.pop('_layer_index', None)
    (args.run / 'summary.json').write_text(json.dumps(S, indent=1))
    print(json.dumps({t: {k: {'heads': [(h['head'], round(h['r'], 2)) for h in e['reading_heads']],
                              'all': round(e['full_all']['mean'], 2),
                              'mix': {g: round(e['mix_share'][g]['mean'], 3) for g in ('target', 'row', 'col', 'row_after')}}
                          for k, e in M['keys'].items()} for t, M in S['models'].items()}, indent=1))


if __name__ == '__main__':
    main()
