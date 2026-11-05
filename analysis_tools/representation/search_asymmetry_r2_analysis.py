"""Experiment 2.7 round 2 CPU analysis.

Stages:
  direction  oddness directions u (leave-one-pair-out) and their typical size m at the steering sites, from the
             round-1 captures; writes directions.npz for the GPU steering stage
  analyze    every table (summary.json) and figure

Colours (repeated in the document captions): blue = feature-present target, orange = feature-absent target,
grey = ordinary distractor (null reference); heatmaps use viridis with a labelled colorbar.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.representation import search_asymmetry_analysis as sa1
from analysis_tools.representation import search_asymmetry_r2 as r2

BLUE, ORANGE, GREY, AQUA, VIOLET, INK, MUTED = sa1.BLUE, sa1.ORANGE, '#8a8984', sa1.AQUA, sa1.VIOLET, sa1.INK, sa1.MUTED
R1_STEMS = {'qo': 'sa_qo', 'closure': 'sa_closure', 'tilt': 'sa_tilt', 'curve': 'sa_curve', 'sat': 'sa_sat',
            'size': 'sa_size', 'length': 'sa_length', 'gap2': 'oe_gap2', 'red': 'oe_red'}
LABEL = {**sa1.PAIR_LABEL, 'term': 'line with vs without end bar', 'plus': '4-arm vs 3-arm cross',
         'dot': 'square with vs without dot', 'circsq': 'rounded square vs circle', 'plusx': 'rotated vs upright cross'}
SITES = ('vit12', 'vit24', 'dec0', 'dec4', 'dec8', 'dec12', 'dec16')


def wrow(k: int, n: int) -> dict:
    return sa1.wrow(k, n)


def contrasts(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """X: (n_img, n_ent, d), t: (n_img,) target index.  Target minus mean of the other entities, (n_img, d)."""
    out = np.empty((len(X), X.shape[-1]), np.float32)
    for i in range(len(X)):
        out[i] = oe.contrast(X[i], int(t[i]))
    return out


def load_states(path: Path, sites=SITES) -> dict:
    """Entity states (float32) of one capture file ordered (present, absent, homA, homB) per quadruple."""
    z = np.load(path)
    tg = z['targets']
    out = {'t': (tg[:, 0] * 6 + tg[:, 1]).astype(int), 'ids': [str(x) for x in z['ids']]}
    for s in sites:
        if f'state_{s}' in z.files:
            out[s] = z[f'state_{s}'].astype(np.float32)
    return out


def pair_odd_dirs(states: dict, sites=SITES) -> dict:
    """Per site: the pair's oddness direction (unit) and the per-image target contrasts."""
    out = {}
    for s in sites:
        if s not in states:
            continue
        c = contrasts(states[s], states['t'])
        out[s] = {'u': r2.oddness_direction(c[0::4], c[1::4]), 'c_present': c[0::4], 'c_absent': c[1::4]}
    return out


def r1_directions(round1: Path, sites=SITES) -> tuple[dict, dict]:
    states = {p: load_states(round1 / 'capture' / f'{stem}.npz', sites) for p, stem in R1_STEMS.items()}
    return states, {p: pair_odd_dirs(st, sites) for p, st in states.items()}


def loo_direction(dirs: dict, site: str, exclude: str | None) -> np.ndarray:
    """Unit mean of the round-1 pairs' oddness directions, leaving out `exclude`."""
    u = np.mean([d[site]['u'] for p, d in dirs.items() if p != exclude], 0)
    return u / np.linalg.norm(u)


def stage_direction(output: Path, round1: Path, sites: list[str], extra_pairs: list[str]) -> None:
    """directions.npz: u|site|pair (leave-one-pair-out unit direction) and m|site|pair (median projection of a
    feature-present target's contrast on u over all round-1 pairs, the typical size of the oddness signal)."""
    _, dirs = r1_directions(round1, sites)
    out = {}
    for s in sites:
        proj = []
        for p, d in dirs.items():
            u = loo_direction(dirs, s, p)
            proj.extend((d[s]['c_present'] @ u).tolist())
        m = float(np.median(proj))
        for p in list(dirs) + extra_pairs:
            out[f'u|{s}|{p}'] = loo_direction(dirs, s, p if p in dirs else None).astype(np.float32)
            out[f'm|{s}|{p}'] = np.float32(m)
    np.savez(output / 'directions.npz', **out)
    print(f'[direction] {len(out) // 2} directions; m = ' +
          ', '.join(f'{s}: {float(out[f"m|{s}|qo"]):.2f}' for s in sites), flush=True)


# ----------------------------------------------------------------------------- behaviour

def behaviour_cells(recs: dict, items: dict, key_fn) -> dict:
    cells = defaultdict(list)
    for iid, r in recs.items():
        it = items.get(iid)
        if it is None:
            continue
        k = key_fn(it)
        if k is not None:
            cells[k].append((it, r))
    return cells


def atlas(output: Path, round1: Path) -> dict:
    """Accuracy of feature-present and feature-absent targets on 6 x 6 grids, both levels, with McNemar tests
    on matched items (same target cell, orientation and jitter)."""
    out = {}
    for root, pairs in ((round1, list(sa.PAIRS)), (output, list(r2.NEW_PAIRS) + ['qo', 'closure'])):
        if not (root / 'results_behave.jsonl').exists():
            continue
        items = {it['id']: it for it in read_jsonl(root / 'items.jsonl')}
        recs = {r['item']: r for r in read_jsonl(root / 'results_behave.jsonl')}
        cells = behaviour_cells(recs, items, lambda it: (it['pair_name'], it['level'], it['variant'])
                                if it.get('group') == 'pair' and it['grid'] == 6 and it.get('set', 'pairs') == 'pairs'
                                else None)
        for pair in pairs:
            for level in ('subtle', 'salient'):
                a = {it['pair']: r['correct'] for it, r in cells.get((pair, level, 'std'), [])}
                b = {it['pair']: r['correct'] for it, r in cells.get((pair, level, 'swap'), [])}
                if not a or not b:
                    continue
                common = sorted(set(a) & set(b))
                b_only = sum(a[p] and not b[p] for p in common)
                c_only = sum(b[p] and not a[p] for p in common)
                out[f'{pair}|{level}'] = {'present': wrow(sum(a.values()), len(a)), 'absent': wrow(sum(b.values()), len(b)),
                                          'present_only': b_only, 'absent_only': c_only,
                                          'p': sa.mcnemar_exact(b_only, c_only),
                                          'relation': r2.NEW_PAIRS.get(pair, {}).get('relation', r2.OLD_RELATION.get(pair)),
                                          'round': 2 if pair in r2.NEW_PAIRS else 1}
    return out


def index_profile(output: Path, round1: Path) -> dict:
    """Feature-absent accuracy by target raster index on 6 x 6 grids (round-1 subtle items + round-2 index set),
    and by layout (3 x 12, 12 x 3) with the number of raster-earlier entities."""
    res = {}
    r1_items = {it['id']: it for it in read_jsonl(round1 / 'items.jsonl')}
    r1_recs = {r['item']: r for r in read_jsonl(round1 / 'results_behave.jsonl')}
    items = {it['id']: it for it in read_jsonl(output / 'items.jsonl')}
    p = output / 'results_behave.jsonl'
    recs = {r['item']: r for r in read_jsonl(p)} if p.exists() else {}
    grid = defaultdict(lambda: defaultdict(list))
    pred_hist = defaultdict(lambda: np.zeros(36))
    for its, rs, sel in ((r1_items, r1_recs, lambda it: it.get('group') == 'pair' and it['grid'] == 6
                          and it['level'] == 'subtle' and it['variant'] == 'swap'),
                         (items, recs, lambda it: it.get('set') == 'index')):
        for iid, r in rs.items():
            it = its[iid]
            if not sel(it):
                continue
            idx = it['target'][0] * 6 + it['target'][1]
            grid[it['pair_name']][idx].append(bool(r['correct']))
            if r['valid'] and not r['correct'] and 0 <= r['pred'][0] < 6 and 0 <= r['pred'][1] < 6:
                pred_hist[it['pair_name']][r['pred'][0] * 6 + r['pred'][1]] += 1
    for pair, d in grid.items():
        res[f'g6|{pair}'] = {'k': [int(sum(d[i])) for i in range(36)], 'n': [len(d[i]) for i in range(36)],
                             'wrong_answer_hist': pred_hist[pair].astype(int).tolist()}
    lay = defaultdict(list)
    for iid, r in recs.items():
        it = items[iid]
        if it.get('set') != 'layout':
            continue
        lay[(it['pair_name'], f'{it["rows"]}x{it["cols"]}', it['variant'])].append((it, r))
    for (pair, l, v), lst in lay.items():
        rows, cols = (int(x) for x in l.split('x'))
        n_e = rows * cols
        k = np.zeros(n_e, int)
        n = np.zeros(n_e, int)
        for it, r in lst:
            i = it['target'][0] * cols + it['target'][1]
            k[i] += r['correct']
            n[i] += 1
        res[f'{l}|{pair}|{v}'] = {'rows': rows, 'cols': cols, 'k': k.tolist(), 'n': n.tolist(), 'total': wrow(int(k.sum()), int(n.sum()))}
    return res


def logistic_index(prof: dict, pairs=('qo', 'closure', 'tilt'), early: int = 6) -> dict:
    """Pool 6 x 6 and both layouts (feature-absent targets) and fit three logistic regressions of success on
    pair indicators plus position terms (Newton-Raphson; Wald 95% intervals; log-likelihood for comparison):
      joint:     fewer than `early` raster-earlier entities + first row + last row + first column + last column
      early:     the same without first row
      first_row: the same without the early indicator
    If the dependence is on the number of earlier entities, `early` survives in the joint model and the early
    model fits at least as well as the first-row model."""
    rows_x = []
    for key, d in prof.items():
        parts = key.split('|')
        if parts[0] == 'g6':
            rows, cols, pair = 6, 6, parts[1]
        else:
            if parts[2] != 'swap':
                continue
            rows, cols, pair = d['rows'], d['cols'], parts[1]
        if pair not in pairs:
            continue
        for i, (k, n) in enumerate(zip(d['k'], d['n'])):
            r, c = divmod(i, cols)
            pos = {'early': float(i < early), 'first_row': float(r == 0), 'last_row': float(r == rows - 1),
                   'first_col': float(c == 0), 'last_col': float(c == cols - 1)}
            rows_x.append((pair, pos, k, n))
    out = {}
    for model, terms in (('joint', ['early', 'first_row', 'last_row', 'first_col', 'last_col']),
                         ('early', ['early', 'last_row', 'first_col', 'last_col']),
                         ('first_row', ['first_row', 'last_row', 'first_col', 'last_col'])):
        names = ['const'] + [f'pair_{p}' for p in pairs[1:]] + terms
        X, y = [], []
        for pair, pos, k, n in rows_x:
            f = [1.0] + [float(pair == p) for p in pairs[1:]] + [pos[t] for t in terms]
            X.extend([f] * n)
            y.extend([1.0] * k + [0.0] * (n - k))
        X, y = np.array(X), np.array(y)
        w = np.zeros(X.shape[1])
        for _ in range(60):
            p = 1 / (1 + np.exp(-X @ w))
            w = w + np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w)), X.T @ (y - p))
        p = 1 / (1 + np.exp(-X @ w))
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w)))))
        out[model] = {nm: {'coef': float(b), 'lo': float(b - 1.96 * e), 'hi': float(b + 1.96 * e)}
                      for nm, b, e in zip(names, w, se)}
        out[model]['loglik'] = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        out[model]['n'] = int(len(y))
    return out

# ----------------------------------------------------------------------------- ViT neurons

def capture_files(output: Path, round1: Path) -> dict[str, Path]:
    """name -> capture file.  Round-1 files keep their activations at blocks 3-18; round-2 files add attention."""
    files = {p: round1 / 'capture' / f'{stem}.npz' for p, stem in R1_STEMS.items()}
    for path in sorted((output / 'capture').glob('*.npz')) if (output / 'capture').exists() else []:
        parts = path.stem.split('_')
        source, pair, level = parts[0], parts[-2], parts[-1]
        if source == 'r2':
            files[pair if level == 'subtle' else f'{pair}@{level}'] = path
        else:
            files[f'r1cap:{pair}'] = path
    return files


def local_salience(act: np.ndarray, e: int, k: int = 10, eps: float = 0.01) -> float:
    """Local salience of entity e: mean of the k largest standardised positive deviations over all neurons,
    (z_e - mean_{e' != e} z_e') / (std_{e' != e} z_e' + eps).  act: (n_ent, n_neurons)."""
    return sa1.popout_stat(act, e, k)[0]


def salience_table(files: dict[str, Path], block: int = 9) -> dict:
    """Per capture: local salience of the feature-present target, the feature-absent target and an ordinary
    distractor (the entity 7 cells after the target in raster order, in the same images), and the number of
    exclusive detectors in each direction (from the homogeneous images)."""
    out = {}
    for name, path in files.items():
        if name.startswith('r1cap:'):
            continue
        z = np.load(path)
        key = f'act_vit{block}'
        if key not in z.files:
            continue
        a = z[key].astype(np.float32)
        tg = z['targets']
        n = len(a) // 4
        pres, absn, null = [], [], []
        for q in range(n):
            t = int(tg[4 * q][0]) * 6 + int(tg[4 * q][1])
            c = (t + 7) % 36
            pres.append(local_salience(a[4 * q], t))
            absn.append(local_salience(a[4 * q + 1], t))
            null.append(0.5 * (local_salience(a[4 * q], c) + local_salience(a[4 * q + 1], c)))
        homa, homb = a[2::4].reshape(-1, a.shape[-1]), a[3::4].reshape(-1, a.shape[-1])
        z_ab = (homa.mean(0) - homb.mean(0)) / (homb.std(0) + 0.01)      # a lone A among B, per neuron
        z_ba = (homb.mean(0) - homa.mean(0)) / (homa.std(0) + 0.01)      # a lone B among A
        out[name] = {'present': float(np.mean(pres)), 'absent': float(np.mean(absn)), 'null': float(np.mean(null)),
                     'present_sem': float(np.std(pres) / math.sqrt(n)), 'absent_sem': float(np.std(absn) / math.sqrt(n)),
                     'n': n, 'excl_A': int((z_ab > 5).sum()), 'excl_B': int((z_ba > 5).sum()),
                     'top_zA': float(np.sort(z_ab)[-10:].mean()), 'top_zB': float(np.sort(z_ba)[-10:].mean())}
    return out


def rotate_to_canonical(m: np.ndarray, orient: int) -> np.ndarray:
    """Rotate a (4, 4) within-entity patch map so that the orientation-dependent part (tail, gap, bar) always
    points to the same corner (the orient-0 direction, down-right).  Orientation o is o quarter turns clockwise
    from orient 0, so the map is turned o quarter turns counter-clockwise."""
    return np.rot90(m, k=orient)


def patch_neurons(path: Path, items: dict, variant_index: int, top: int = 4) -> dict:
    """For one capture file with per-patch activations: the neurons with the largest mean positive deviation of
    the target (variant_index 0 = feature-present, 1 = feature-absent), and their within-entity (4 x 4) patch maps
    on the target and on the distractors, rotated to the canonical orientation."""
    z = np.load(path)
    P = z['patch_acts'].astype(np.float32)                                   # (n_img, 36, 16, 4304)
    ids = [str(x) for x in z['ids']][:len(P)]
    tg = z['targets'][:len(P)]
    block = int(z['patch_block'])
    a = z[f'act_vit{block}'].astype(np.float32)[:len(P)] if f'act_vit{block}' in z.files else P.max(2)
    sel = list(range(variant_index, len(P), 4))
    dev = []
    for i in sel:
        t = int(tg[i][0]) * 6 + int(tg[i][1])
        o = np.delete(a[i], t, 0)
        dev.append((a[i, t] - o.mean(0)) / (o.std(0) + 0.01))
    dev = np.mean(dev, 0)
    neurons = [int(j) for j in np.argsort(dev)[::-1][:top]]
    maps_t, maps_d = np.zeros((top, 4, 4)), np.zeros((top, 4, 4))
    for i in sel:
        it = items[ids[i]] if ids[i] in items else items['r1|' + ids[i]]
        t = int(tg[i][0]) * 6 + int(tg[i][1])
        for r, j in enumerate(neurons):
            # 16 patches of an entity: merged tokens in row-major (2 x 2), each holding a 2 x 2 patch block
            m = P[i, :, :, j].reshape(36, 2, 2, 2, 2).transpose(0, 1, 3, 2, 4).reshape(36, 4, 4)
            maps_t[r] += rotate_to_canonical(m[t], it['orient'])
            maps_d[r] += np.mean([rotate_to_canonical(m[e], it['orient']) for e in range(36) if e != t], 0)
    return {'block': block, 'neurons': neurons, 'dev': [float(dev[j]) for j in neurons],
            'map_target': (maps_t / len(sel)).tolist(), 'map_distractor': (maps_d / len(sel)).tolist()}


def pixel_difference_map(pair: str, level: str) -> np.ndarray:
    """(4, 4) mean absolute pixel difference between A and B tiles per 16-pixel patch, orient 0, no jitter."""
    p = r2.ALL_PAIRS[pair]['levels'][level]
    d = np.abs(r2.render_entity(pair, True, p, 0, (0, 0)).astype(float) -
               r2.render_entity(pair, False, p, 0, (0, 0)).astype(float)).mean(-1)
    return d.reshape(4, 16, 4, 16).mean((1, 3))


# ----------------------------------------------------------------------------- oddness direction and novelty

def oddness_summary(files: dict[str, Path], dirs1: dict, behave: dict) -> dict:
    """Projection of every entity on the leave-one-pair-out oddness direction, per site.

    For each capture and site: s_e = <h_e - mean_e' h_e', u>; reported as the within-image z-score of the target
    (feature-present / feature-absent images, split by whether the model answered correctly) and as a 6 x 6
    map of z-scores in the homogeneous images (the null map)."""
    out = {'cos': {}, 'target_z': {}, 'null_map': {}}
    names = list(R1_STEMS)
    for s in SITES:
        M = np.array([[float(dirs1[a][s]['u'] @ dirs1[b][s]['u']) for b in names] for a in names])
        out['cos'][s] = M.tolist()
    null = defaultdict(list)
    for name, path in files.items():
        if name.startswith('r1cap:'):
            continue
        st = load_states(path)
        pair = name.split('@')[0]
        res = {}
        for s in SITES:
            if s not in st:
                continue
            u = loo_direction(dirs1, s, pair if pair in dirs1 else None)
            X = st[s]
            sc = np.einsum('ned,d->ne', X - X.mean(1, keepdims=True), u)
            zs = np.stack([r2.entity_zscores(v) for v in sc])
            t = st['t']
            for v, off in (('present', 0), ('absent', 1)):
                idx = list(range(off, len(X), 4))
                ok = [behave.get(st['ids'][i]) for i in idx]
                zt = np.array([zs[i, t[i]] for i in idx])
                res[f'{s}|{v}'] = float(zt.mean())
                for flag, nm in ((True, 'found'), (False, 'missed')):
                    m = np.array([o is flag for o in ok])
                    res[f'{s}|{v}|{nm}'] = float(zt[m].mean()) if m.any() else None
                    res[f'{s}|{v}|{nm}_n'] = int(m.sum())
            hom = np.concatenate([zs[2::4], zs[3::4]])
            null[s].append(hom.mean(0))
        out['target_z'][name] = res
    out['null_map'] = {s: np.mean(v, 0).tolist() for s, v in null.items()}
    return out


def surprise_test(files: dict[str, Path], dirs1: dict, sites=('dec8', 'dec16'), source_site: str = 'dec0') -> dict:
    """Does the decoder's oddness score follow a causal (raster-template) or a global comparison?

    For every homogeneous and feature-absent image: Spearman correlation over entities 2..36 between the decoder
    oddness score s_e (projection on u at `sites`) and (a) the causal surprise ||x_e - mean_{e'<e} x_e'|| or
    (b) the global surprise ||x_e - mean_{e'!=e} x_e'||, with x the entity states at `source_site` (the decoder
    input, before any decoder mixing).  Because both the causal surprise and the decoder score fall with the
    raster index on average, the 'detrended' variant first subtracts, per pair and image kind, the mean over
    images at each raster index from every quantity, so that only image-specific variation is correlated."""
    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    out = {}
    for s in sites:
        acc = {(k, d): {'causal': [], 'global': []} for k in ('hom', 'absent') for d in ('raw', 'detrended')}
        for name, path in files.items():
            if name.startswith('r1cap:') or '@' in name:
                continue
            st = load_states(path, (source_site, s))
            if s not in st or source_site not in st:
                continue
            u = loo_direction(dirs1, s, name if name in dirs1 else None)
            X, Y = st[source_site], st[s]
            for kind, offs in (('hom', (2, 3)), ('absent', (1,))):
                idx = [i for i in range(len(X)) if i % 4 in offs]
                sc = np.stack([(Y[i] - Y[i].mean(0)) @ u for i in idx])[:, 1:]
                cs = np.stack([r2.causal_surprise(X[i]) for i in idx])[:, 1:]
                gs = np.stack([r2.global_surprise(X[i]) for i in idx])[:, 1:]
                for d in ('raw', 'detrended'):
                    a, b, c = (sc, cs, gs) if d == 'raw' else (sc - sc.mean(0), cs - cs.mean(0), gs - gs.mean(0))
                    for j in range(len(idx)):
                        acc[(kind, d)]['causal'].append(spearman(a[j], b[j]))
                        acc[(kind, d)]['global'].append(spearman(a[j], c[j]))
        out[s] = {f'{k}|{d}': {m: float(np.mean(v)) for m, v in val.items()} |
                  {f'{m}_sem': float(np.std(v) / math.sqrt(len(v))) for m, v in val.items()} | {'n': len(val['causal'])}
                  for (k, d), val in acc.items()}
    return out


def index_states(output: Path, dirs1: dict, flags: dict, sites=('dec0', 'dec8', 'dec16')) -> dict:
    """Decoder oddness of feature-absent targets by raster index (states/ files of the index and layout sets).

    Per file and site: for every image, the within-image z-score of the target on the leave-one-pair-out
    oddness direction, the largest z among the other entities, and whether the model answered correctly."""
    out = {}
    d = output / 'states'
    if not d.exists():
        return out
    for path in sorted(d.glob('*.npz')):
        z = np.load(path)
        set_name, pair, lay = path.stem.split('_')
        ids = [str(x) for x in z['ids']]
        var = [str(x) for x in z['variants']]
        t = z['targets'].astype(int)
        res = {'index': t.tolist(), 'variant': var, 'correct': [flags.get(i) for i in ids]}
        for site in sites:
            if f'state_{site}' not in z.files:
                continue
            X = z[f'state_{site}'].astype(np.float32)
            u = loo_direction(dirs1, site, pair if pair in dirs1 else None)
            sc = np.einsum('ned,d->ne', X - X.mean(1, keepdims=True), u)
            zs = np.stack([r2.entity_zscores(v) for v in sc])
            res[f'{site}|target_z'] = [float(zs[i, t[i]]) for i in range(len(X))]
            res[f'{site}|max_other_z'] = [float(np.delete(zs[i], t[i]).max()) for i in range(len(X))]
            res[f'{site}|argmax'] = [int(np.argmax(zs[i])) for i in range(len(X))]
        out[path.stem] = res
    return out


def template_observer(files: dict[str, Path], site: str = 'dec0', ns=(1, 2, 3, 5, 8, 12, 20, 30), draws: int = 20,
                      seed: int = 0) -> dict:
    """A simulated comparison observer on the model's own entity states (feature-absent images).

    For n earlier entities drawn at random from the distractors of an image, the template is their mean state at
    `site`; the observer is correct when the target is farther from the template than another random distractor.
    Returns the win rate per n: how much a running-average comparison gains from more samples."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('qo', 'closure', 'tilt', 'curve', 'dot', 'plus', 'term', 'qo@salient', 'closure@salient'):
        if name not in files:
            continue
        st = load_states(files[name], (site,))
        X, t = st[site], st['t']
        res = {}
        for n in ns:
            wins = []
            for i in range(1, len(X), 4):
                o = [e for e in range(X.shape[1]) if e != t[i]]
                for _ in range(draws):
                    sel = rng.choice(o, n + 1, replace=False)
                    tmpl = X[i, sel[:n]].mean(0)
                    wins.append(np.linalg.norm(X[i, t[i]] - tmpl) > np.linalg.norm(X[i, sel[n]] - tmpl))
            res[str(n)] = float(np.mean(wins))
        out[name] = res
    return out


# ----------------------------------------------------------------------------- attention

def attention_summary(files: dict[str, Path], behave: dict) -> dict:
    """ViT and decoder attention aggregated to entities (head mean).

    ViT, per block: the target's self-mass (mean over its 16 query patches of the probability on its own 16
    patches) and the attention it receives (mean over the other 35 entities of their mass on the target), each
    minus the same quantity for the average distractor.  Decoder, per full-attention layer: the mass that the
    query predicting the row digit puts on the target, and the mass from the target's tokens onto raster-earlier
    entities."""
    out = {}
    for name, path in files.items():
        z = np.load(path)
        if 'vit_attn' not in z.files:
            continue
        VA = z['vit_attn'].astype(np.float32).mean(2)        # (n_img, nb, 36, 36)
        QA = z['ans_attn'].astype(np.float32)                 # (n_img, nl, 16, 36)
        DA = z['dec_attn'].astype(np.float32).mean(2)        # (n_img, nl, 36, 36)
        tg = z['targets']
        ids = [str(x) for x in z['ids']]
        t = (tg[:, 0] * 6 + tg[:, 1]).astype(int)
        res = {'vit_blocks': z['vit_attn_blocks'].tolist(), 'full_layers': z['full_layers'].tolist()}
        for v, off in (('present', 0), ('absent', 1), ('homA', 2), ('homB', 3)):
            idx = list(range(off, len(VA), 4))
            diag = np.einsum('nbee->nbe', VA[idx])                       # (n, nb, 36)
            recv = (VA[idx].sum(2) - diag) / 35                         # attention received from the others
            tt = t[idx]
            self_t = np.array([diag[i, :, tt[i]] for i in range(len(idx))])
            self_d = np.array([np.delete(diag[i], tt[i], 1).mean(1) for i in range(len(idx))])
            recv_t = np.array([recv[i, :, tt[i]] for i in range(len(idx))])
            recv_d = np.array([np.delete(recv[i], tt[i], 1).mean(1) for i in range(len(idx))])
            res[f'vit_self|{v}'] = (self_t - self_d).mean(0).tolist()
            res[f'vit_self_ratio|{v}'] = (self_t / self_d).mean(0).tolist()
            res[f'vit_recv|{v}'] = (recv_t - recv_d).mean(0).tolist()
            res[f'vit_recv_ratio|{v}'] = (recv_t / recv_d).mean(0).tolist()
            ans = QA[idx].mean(2)                                        # (n, nl, 36)
            res[f'ans_target|{v}'] = np.array([ans[i, :, tt[i]] for i in range(len(idx))]).mean(0).tolist()
            ok = np.array([behave.get(ids[i]) is True for i in idx])
            res[f'ans_target_found|{v}'] = (np.array([ans[i, :, tt[i]] for i in range(len(idx))])[ok].mean(0).tolist()
                                            if ok.any() else None)
            res[f'ans_target_missed|{v}'] = (np.array([ans[i, :, tt[i]] for i in range(len(idx))])[~ok].mean(0).tolist()
                                             if (~ok).any() else None)
            earlier = np.array([DA[i, :, tt[k], :tt[k]].sum(-1) for k, i in enumerate(idx)])   # (n, nl)
            res[f'dec_earlier|{v}'] = earlier.mean(0).tolist()
            res[f'n|{v}'] = len(idx)
        out[name] = res
    return out


# ----------------------------------------------------------------------------- steering

def steering_summary(output: Path) -> dict:
    path = output / 'steer.jsonl'
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    add = [r for r in rows if r['test'] == 'add']
    rem = [r for r in rows if r['test'] == 'remove']
    out = {'add': {}, 'remove': {}, 'add_by_pair': {}}
    cells = defaultdict(list)
    for r in add:
        cells[r['cond']].append(r)
    base = {r['item']: r for r in add if r['cond'] == 'none'}
    for c, lst in sorted(cells.items()):
        k = sum(r['hit'] for r in lst)
        d = {**wrow(k, len(lst)), 'dS': float(np.mean([r['s'] - base[r['item']]['s'] for r in lst]))}
        for key in [x for x in lst[0] if x.startswith('att_L') and not x.endswith('heads')]:
            d[key] = float(np.mean([r[key] for r in lst]))
            d[key + '_delta'] = float(np.mean([r[key] - base[r['item']][key] for r in lst]))
        out['add'][c] = d
        bp = defaultdict(list)
        for r in lst:
            bp[r['pair']].append(r['hit'])
        out['add_by_pair'][c] = {p: wrow(sum(v), len(v)) for p, v in bp.items()}
    # strongest heads: mean increase of the answer query's mass on the edited entity
    heads = {}
    for c in cells:
        if c == 'none':
            continue
        for key in [x for x in cells[c][0] if x.endswith('heads')]:
            delta = np.mean([np.array(r[key]) - np.array(base[r['item']][key]) for r in cells[c]], 0)
            heads[f'{c}|{key}'] = delta.tolist()
    out['head_delta'] = heads
    rc = defaultdict(list)
    for r in rem:
        rc[(r['cond'], r['variant'])].append(r)
    for (c, v), lst in sorted(rc.items()):
        num = np.array([r['s'] - r['s_base'] for r in lst])
        den = np.array([r['s_clean'] - r['s_base'] for r in lst])
        keep = den > 1.0
        R, lo, hi = sa1.bootstrap_ratio(num[keep], den[keep]) if keep.sum() > 1 else (float('nan'),) * 3
        out['remove'][f'{c}|{v}'] = {'R': R, 'lo': lo, 'hi': hi, 'n': int(keep.sum()), 'n_dropped': int((~keep).sum())}
    return out


def steering2_summary(output: Path) -> dict:
    """Controls of steer2: random directions at every alpha, reader-head ablation, projection ablation."""
    path = output / 'steer2.jsonl'
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    out = {'add': {}, 'remove': {}}
    cells = defaultdict(list)
    for r in rows:
        if r['test'] == 'add':
            cells[r['cond']].append(r['hit'])
    out['add'] = {c: wrow(sum(v), len(v)) for c, v in sorted(cells.items())}
    rc = defaultdict(list)
    for r in rows:
        if r['test'] == 'remove':
            rc[(r['cond'], r['variant'])].append(r)
    for (c, v), lst in sorted(rc.items()):
        num = np.array([r['s'] - r['s_base'] for r in lst])
        den = np.array([r['s_clean'] - r['s_base'] for r in lst])
        keep = den > 1.0
        R, lo, hi = sa1.bootstrap_ratio(num[keep], den[keep])
        acc = float(np.mean([r['arg_row'] - 1 == json_target(r)[0] and r['arg_col'] - 1 == json_target(r)[1]
                             for r in lst])) if 'target' in lst[0] else None
        out['remove'][f'{c}|{v}'] = {'R': R, 'lo': lo, 'hi': hi, 'n': int(keep.sum()), 'n_dropped': int((~keep).sum()),
                                     'top1': acc}
    heads = output / 'steer2_heads.json'
    if heads.exists():
        out['heads'] = json.loads(heads.read_text())
    return out


def json_target(r: dict) -> tuple[int, int]:
    return tuple(r['target'])


# ----------------------------------------------------------------------------- entry

def behave_flags(output: Path, round1: Path) -> dict:
    """item id (as stored in capture files) -> correct, for both rounds and the round-2 'r1|' prefix."""
    flags = {}
    for root in (round1, output):
        p = root / 'results_behave.jsonl'
        if p.exists():
            for r in read_jsonl(p):
                flags[r['item']] = bool(r['correct'])
                flags['r1|' + r['item']] = bool(r['correct'])
    oe_dir = round1.parent / '2026-10-20_odd_entity'
    if (oe_dir / 'results_behave.jsonl').exists():
        for r in read_jsonl(oe_dir / 'results_behave.jsonl'):
            if r['mode'] == 'direct':
                flags[r['item']] = bool(r['correct'])
    return flags


def analyze(args) -> dict:
    from analysis_tools.representation import search_asymmetry_r2_figures as figs
    out, r1 = args.output, args.round1
    args.figures.mkdir(parents=True, exist_ok=True)
    flags = behave_flags(out, r1)
    summary = {'atlas': atlas(out, r1), 'index': index_profile(out, r1)}
    summary['index_logit'] = logistic_index(summary['index'])
    files = capture_files(out, r1)
    _, dirs1 = r1_directions(r1)
    summary['salience'] = salience_table(files)
    summary['oddness'] = oddness_summary(files, dirs1, flags)
    summary['surprise'] = surprise_test(files, dirs1)
    summary['template_observer'] = template_observer(files)
    summary['attention'] = attention_summary(files, flags)
    summary['index_states'] = index_states(out, dirs1, flags)
    items = {it['id']: it for it in read_jsonl(out / 'items.jsonl')}
    items.update({'r1|' + it['id']: it for it in read_jsonl(r1 / 'items.jsonl')})
    summary['patch'] = {}
    for name in ('closure@salient', 'r1cap:closure', 'qo@salient', 'r1cap:qo', 'dot', 'term', 'plus'):
        if name in files:
            for v, off in (('present', 0), ('absent', 1)):
                summary['patch'][f'{name}|{v}'] = patch_neurons(files[name], items, off)
    summary['steer'] = steering_summary(out)
    summary['steer2'] = steering2_summary(out)
    spot = out / 'spotcheck.json'
    if spot.exists():
        s = json.loads(spot.read_text())
        summary['spotcheck'] = {'n': s['n'], 'same_answer': s['same_answer'], 'same_text': s['same_text']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    figs.make_all(summary, out, r1, args.figures, items)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['direction', 'analyze'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--round1', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--steer-sites', nargs='+', default=['dec0', 'dec8', 'dec16'])
    p.add_argument('--extra-pairs', nargs='+', default=list(r2.NEW_PAIRS))
    args = p.parse_args(argv)
    if args.stage == 'direction':
        stage_direction(args.output, args.round1, args.steer_sites, args.extra_pairs)
    else:
        analyze(args)


if __name__ == '__main__':
    main()
