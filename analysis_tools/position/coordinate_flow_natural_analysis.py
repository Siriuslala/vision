"""CPU analysis of experiment 3.11 round 2 (coordinate readout on natural images).

Inputs: the round-2 run directory (coordinate_flow_natural.py: cases.json with RefCOCO masks, gen.jsonl,
patch.jsonl, attn/, ablate.jsonl) and the 2.5e run directory (readout maps of the 300 RefCOCO answers).
Output: summary.json and salient_rows.json in the run directory.

  salient  where the attention of the reading heads lands on RefCOCO objects relative to the bounding box,
           its centre and the COCO segmentation mask; dependence on object size.
  patch    recovered fraction r of every head / layer set / DeltaNet group for the what pair (expression
           swapped to another object in the same image) and the where pair (target content moved), per group.
  select   referent selectivity of every head's attention: how much attention moves from object A to object B
           when only the expression changes (what pair).
  ablate   accuracy, wrong-object rate and centre error under mean ablation of head sets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from analysis_tools.position import coordinate_readout as cr
from analysis_tools.representation import sees_right as sr
from analysis_tools.representation.sees_right_analysis import boot_mean, prop

FULL_LAYERS = cr.FULL_LAYERS
LI = {l: i for i, l in enumerate(FULL_LAYERS)}
MIN_GAP = 2.0
TP = cr.TOKEN_PIXELS
PERSON_WORDS = {'man', 'men', 'woman', 'women', 'guy', 'girl', 'boy', 'lady', 'kid', 'child', 'person', 'player',
                'baby', 'people', 'dude', 'catcher', 'batter', 'umpire', 'skier', 'surfer', 'mom', 'dad', 'he',
                'she', 'her', 'him', 'chick', 'gal', 'adult', 'toddler', 'skateboarder', 'snowboarder', 'groom',
                'bride', 'teen', 'officer', 'cop', 'pitcher', 'old', 'shirt', 'jacket', 'hat', 'kids', 'guys'}
ANIMAL_WORDS = {'dog', 'cat', 'horse', 'elephant', 'giraffe', 'zebra', 'cow', 'sheep', 'bear', 'bird', 'puppy',
                'kitten', 'calf', 'lamb', 'animal', 'pony', 'duck', 'goat', 'teddy'}


def read_jsonl(path: Path) -> list[dict]:
    return sr.read_jsonl(path) if path.exists() else []


def read_unique(path: Path) -> list[dict]:
    """Records keyed by id, last one kept (two GPU processes may both finish the same case)."""
    return list({r['id']: r for r in read_jsonl(path)}.values())


def category(expr: str) -> str:
    """Keyword heuristic on the referring expression: person, animal or other (stated as a heuristic)."""
    words = set(re.findall(r'[a-z]+', expr.lower()))
    if words & ANIMAL_WORDS:
        return 'animal'
    if words & PERSON_WORDS:
        return 'person'
    return 'other'


def coverage(poly: list[list[float]] | None, box: list[float] | None, canvas, grid) -> np.ndarray:
    """(Nv,) fraction of each merged token's 32x32 pixel cell covered by the polygon (or box)."""
    W, H = canvas
    im = Image.new('L', (int(W), int(H)), 0)
    d = ImageDraw.Draw(im)
    if poly is not None:
        d.polygon([tuple(p) for p in poly], fill=1)
    else:
        d.rectangle(box, fill=1)
    a = np.asarray(im, np.float32)
    h, w = grid
    a = a[:h * TP, :w * TP].reshape(h, TP, w, TP).mean((1, 3))
    return a.ravel()


def mask_centroid(poly: list[list[float]], canvas) -> np.ndarray:
    W, H = canvas
    im = Image.new('L', (int(W), int(H)), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=1)
    ys, xs = np.nonzero(np.asarray(im))
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5])


def local_peak(p: np.ndarray, grid) -> tuple[int, np.ndarray]:
    """Peak token and the attention-weighted centre (pixels) of its 3x3 neighbourhood (round-1 decoder)."""
    H, W = grid
    j = int(np.argmax(p))
    r, c = divmod(j, W)
    idx = [rr * W + cc for rr in range(max(0, r - 1), min(H, r + 2)) for cc in range(max(0, c - 1), min(W, c + 2))]
    cen = np.stack([(np.array(idx) % W + 0.5) * TP, (np.array(idx) // W + 0.5) * TP], 1)
    w = p[idx]
    return j, (w[:, None] * cen).sum(0) / w.sum()


def centroid(p: np.ndarray, grid) -> np.ndarray:
    H, W = grid
    j = np.arange(H * W)
    cen = np.stack([(j % W + 0.5) * TP, (j // W + 0.5) * TP], 1)
    return (p[:, None] * cen).sum(0) / p.sum()


def box_uv(pt: np.ndarray, box) -> np.ndarray:
    """Point in box-normalized coordinates: (0, 0) top-left corner, (1, 1) bottom-right corner."""
    return np.array([(pt[0] - box[0]) / (box[2] - box[0]), (pt[1] - box[1]) / (box[3] - box[1])])


def to_units(pt: np.ndarray, canvas) -> np.ndarray:
    return pt * 1000 / np.asarray(canvas, float)


# ----------------------------------------------------------------------------- salient part (Q2)

READERS = {'H11': (20, 11), 'L20H2': (20, 2), 'L24mean': None, 'L20mean': None}


def salient_rows(run: Path, source: Path) -> list[dict]:
    cases = {c['image_key']: c for c in json.loads((run / 'cases.json').read_text()) if c['group'] == 'rc'}
    src_cases = {c['id']: c for c in json.loads((source / 'cases.json').read_text())}
    gen = {g['id']: g for g in sr.read_jsonl(source / 'gen.jsonl')}
    rows = []
    for r in sr.read_jsonl(source / 'readout' / 'records.jsonl'):
        if not r['id'].startswith('rc_point') or r.get('status') != 'ok':
            continue
        sc = src_cases[r['id']]
        c = cases[sc['image_key']]
        g = gen[r['id']]
        grid = tuple(r['grid'])
        canvas = c['truth']['canvas']
        box = c['truth']['box']
        a = np.load(source / 'readout' / 'maps' / f"{r['id']}.npz")['attn'].astype(np.float32)[:, :, :2]
        p = a / a.sum(-1, keepdims=True)                                      # (8, 16, 2, Nv), image-normalized
        mcov = coverage(c['mask'], None, canvas, grid)
        bcov = coverage(None, box, canvas, grid)
        center = np.array(c['truth']['center'])
        mcen = mask_centroid(c['mask'], canvas)
        maps = {'H11': p[LI[20], 11], 'L20H2': p[LI[20], 2], 'L20mean': p[LI[20]].mean(0),
                'L24mean': p[LI[24]].mean(0)}
        model = np.array(g['score']['out_units']) if g['score'].get('out_units') is not None else None
        row = {'id': r['id'], 'expression': c['expression'], 'category': category(c['expression']),
               'canvas': canvas, 'grid': list(grid), 'box': box, 'center_u': to_units(center, canvas).tolist(),
               'mask_centroid_u': to_units(mcen, canvas).tolist(),
               'area_frac': (box[2] - box[0]) * (box[3] - box[1]) / (canvas[0] * canvas[1]),
               'mask_box_frac': float(mcov.sum() / max(bcov.sum(), 1e-6)),
               'model_u': None if model is None else model.tolist(), 'model_ok': bool(g['score']['correct']),
               'maps': {}}
        for name, m in maps.items():
            ent = {}
            for k, q in enumerate(('x', 'y')):
                j, pk = local_peak(m[k], grid)
                ent[q] = {'peak_px': pk.tolist(), 'peak_u': to_units(pk, canvas).tolist(),
                          'peak_uv': box_uv(pk, box).tolist(),
                          'centroid_u': to_units(centroid(m[k], grid), canvas).tolist(),
                          'peak_in_mask_token': float(mcov[j]), 'peak_in_box_token': float(bcov[j]),
                          'mass_mask': float((m[k] * mcov).sum()), 'mass_box_bg': float((m[k] * (bcov - mcov).clip(0)).sum()),
                          'mass_out': float((m[k] * (1 - bcov)).sum())}
            # round-1 decoded point: x from the x-query map, y from the y-query map
            ent['decoded_u'] = [ent['x']['peak_u'][0], ent['y']['peak_u'][1]]
            row['maps'][name] = ent
        row['mask_token_frac_in_box'] = float(mcov.sum() / max(bcov.sum(), 1e-6))
        rows.append(row)
    return rows


def salient_summary(rows: list[dict], rng) -> dict:
    out: dict = {'n': len(rows)}
    cats = np.array([r['category'] for r in rows])
    area = np.array([r['area_frac'] for r in rows])
    cen = np.array([r['center_u'] for r in rows])
    mcen = np.array([r['mask_centroid_u'] for r in rows])
    model = np.array([r['model_u'] if r['model_u'] is not None else [np.nan, np.nan] for r in rows])
    out['categories'] = {k: int((cats == k).sum()) for k in ('person', 'animal', 'other')}
    out['model'] = {'err_box_center_median': float(np.nanmedian(np.linalg.norm(model - cen, axis=1))),
                    'err_mask_centroid_median': float(np.nanmedian(np.linalg.norm(model - mcen, axis=1))),
                    'closer_to_box_center': prop(np.linalg.norm(model - cen, axis=1) < np.linalg.norm(model - mcen, axis=1)),
                    'hit': prop(np.array([r['model_ok'] for r in rows]))}
    for name in READERS:
        ent = {}
        dec = np.array([r['maps'][name]['decoded_u'] for r in rows])
        boxes = [r['box'] for r in rows]
        hit = np.array([b[0] <= d[0] * r['canvas'][0] / 1000 <= b[2] and b[1] <= d[1] * r['canvas'][1] / 1000 <= b[3]
                        for d, b, r in zip(dec, boxes, rows)])
        ent['decoded_hit'] = prop(hit)
        ent['decoded_err_median'] = float(np.median(np.linalg.norm(dec - cen, axis=1)))
        for q in ('x', 'y'):
            pk = np.array([r['maps'][name][q]['peak_u'] for r in rows])
            ce = np.array([r['maps'][name][q]['centroid_u'] for r in rows])
            uv = np.array([r['maps'][name][q]['peak_uv'] for r in rows])
            inb = (uv >= 0).all(1) & (uv <= 1).all(1)
            d = np.linalg.norm(pk - cen, axis=1)
            sub = {'peak_err_median': float(np.median(d)), 'centroid_err_median': float(np.median(np.linalg.norm(ce - cen, axis=1))),
                   'peak_in_box': prop(inb),
                   'peak_in_mask_token': prop(np.array([r['maps'][name][q]['peak_in_mask_token'] >= 0.5 for r in rows])),
                   'mass_mask': boot_mean(np.array([r['maps'][name][q]['mass_mask'] for r in rows]), rng),
                   'mass_box_bg': boot_mean(np.array([r['maps'][name][q]['mass_box_bg'] for r in rows]), rng),
                   'mass_out': boot_mean(np.array([r['maps'][name][q]['mass_out'] for r in rows]), rng),
                   'mask_box_frac': boot_mean(np.array([r['mask_box_frac'] for r in rows]), rng)}
            # vertical position of the peak inside the box, by category (0 = top edge, 1 = bottom edge)
            sub['v_by_cat'] = {}
            for k in ('person', 'animal', 'other'):
                sel = (cats == k) & inb
                v = uv[sel, 1]
                sub['v_by_cat'][k] = {'n': int(sel.sum()), 'median': float(np.median(v)) if sel.any() else float('nan'),
                                      'upper_third': prop(v < 1 / 3) if sel.any() else None,
                                      'u_median': float(np.median(uv[sel, 0])) if sel.any() else float('nan')}
            # size dependence
            sub['by_area'] = []
            qb = np.quantile(area, [0, 0.25, 0.5, 0.75, 1.0])      # area quartiles (RefCOCO objects are large)
            qb[-1] += 1e-9
            for lo, hi in zip(qb[:-1], qb[1:]):
                sel = (area >= lo) & (area < hi)
                diag = np.array([np.hypot((r['box'][2] - r['box'][0]) * 1000 / r['canvas'][0],
                                          (r['box'][3] - r['box'][1]) * 1000 / r['canvas'][1]) for r in rows])
                me = np.linalg.norm(model - cen, axis=1)
                sub['by_area'].append({'lo': float(lo), 'hi': float(hi), 'n': int(sel.sum()),
                                       'peak_err_median': float(np.median(d[sel])) if sel.any() else float('nan'),
                                       'peak_err_rel_median': float(np.median(d[sel] / diag[sel])) if sel.any() else float('nan'),
                                       'model_err_median': float(np.nanmedian(me[sel])) if sel.any() else float('nan'),
                                       'peak_in_box': prop(inb[sel]) if sel.any() else None,
                                       'model_hit': prop(np.array([r['model_ok'] for r in rows])[sel]) if sel.any() else None})
            ent[q] = sub
        out[name] = ent
    # correlation of peak offset with model offset (does the model follow the peak?)
    pk = np.array([r['maps']['H11']['decoded_u'] for r in rows])
    ok = np.isfinite(model).all(1)
    out['model_vs_peak_offset_r'] = [float(np.corrcoef((pk - cen)[ok, a], (model - cen)[ok, a])[0, 1]) for a in (0, 1)]
    return out


# ----------------------------------------------------------------------------- edge-reading heads

def attn_items_2p5e(run: Path, source: Path) -> list[dict]:
    """RefCOCO first-digit attention of the 2.5e readout: [{'attn' (8,16,2,Nv), 'grid', 'box', 'category', 'id'}]."""
    cases = {c['image_key']: c for c in json.loads((run / 'cases.json').read_text()) if c['group'] == 'rc'}
    src_cases = {c['id']: c for c in json.loads((source / 'cases.json').read_text())}
    out = []
    for r in sr.read_jsonl(source / 'readout' / 'records.jsonl'):
        if not r['id'].startswith('rc_point') or r.get('status') != 'ok':
            continue
        c = cases[src_cases[r['id']]['image_key']]
        a = np.load(source / 'readout' / 'maps' / f"{r['id']}.npz")['attn'].astype(np.float32)[:, :, :2]
        out.append({'id': c['id'], 'attn': a, 'grid': tuple(r['grid']), 'box': c['truth']['box'],
                    'category': category(c['expression']), 'area_frac': None})
    return out


def edge_profile(items: list[dict]) -> dict:
    """Where on the object each head looks, at the x query (horizontal) and the y query (vertical).

    For every head and image: the attention restricted to tokens whose centre lies inside the target box
    and renormalized gives the box-normalized centroid (u at the x query, v at the y query; 0 = left/top
    edge, 1 = right/bottom edge). Across images, the peak coordinate (3x3 local centre, pixels) is
    regressed on the two box edges, peak = b_min * edge_min + b_max * edge_max + c (least squares): a
    centre reader has b_min = b_max = 0.5, a min-edge reader b_min = 1, b_max = 0.
    Images whose box contains fewer than 4 token centres are skipped.
    """
    U, PK, ED, MB = [], [], [], []
    for it in items:
        H, W = it['grid']
        j = np.arange(H * W)
        cen = np.stack([(j % W + 0.5) * TP, (j // W + 0.5) * TP], 1)
        b = it['box']
        uv = np.stack([(cen[:, 0] - b[0]) / (b[2] - b[0]), (cen[:, 1] - b[1]) / (b[3] - b[1])], 1)
        inb = ((uv >= 0) & (uv <= 1)).all(1)
        if inb.sum() < 4:
            continue
        p = it['attn'] / it['attn'].sum(-1, keepdims=True)              # (8, 16, 2, Nv)
        q = p[..., inb]
        MB.append(q.sum(-1))
        q = q / q.sum(-1, keepdims=True)
        U.append(np.stack([q[:, :, 0] @ uv[inb, 0], q[:, :, 1] @ uv[inb, 1]], -1))   # (8, 16, 2)
        pk = np.zeros((8, 16, 2))
        for li in range(8):
            for h in range(16):
                for k in range(2):
                    pk[li, h, k] = local_peak(p[li, h, k], (H, W))[1][k]
        PK.append(pk)
        ED.append([[b[0], b[2]], [b[1], b[3]]])
    U, PK, ED, MB = np.stack(U), np.stack(PK), np.array(ED), np.stack(MB)
    beta = np.zeros((8, 16, 2, 2))
    for k in range(2):
        X = np.column_stack([ED[:, k, 0], ED[:, k, 1], np.ones(len(ED))])
        for li in range(8):
            for h in range(16):
                coef, *_ = np.linalg.lstsq(X, PK[:, li, h, k], rcond=None)
                beta[li, h, k] = coef[:2]
    return {'n': int(len(U)), 'uv_median': np.median(U, 0).tolist(), 'beta': beta.tolist(),
            'mass_in_box': MB.mean(0).tolist()}


def edge_classes(prof: dict, lo: float = 0.35, hi: float = 0.65) -> dict:
    """Heads whose median box-normalized centroid is near the min edge (u < lo at the x query and v < 0.4 at the
    y query) or near the max edge (u > hi and v > hi); layers 16-32."""
    uv = np.array(prof['uv_median'])
    mn, mx = [], []
    for li, l in enumerate(FULL_LAYERS):
        if l < 16:
            continue
        for h in range(16):
            u, v = uv[li, h]
            if u < lo and v < 0.4:
                mn.append(f'{l}.{h}')
            if u > hi and v > hi:
                mx.append(f'{l}.{h}')
    return {'min_edge': mn, 'max_edge': mx}


def edge_pair_decode(items: list[dict], cases: dict) -> dict:
    """Cross-fitted centre decoding from edge heads: classes chosen on one image parity, evaluated on the other.

    Decoded x = mean of the peak x of the averaged min-edge-head map and of the averaged max-edge-head map at
    the x query; y likewise at the y query. Compared with the L20 H11 peak decoder and the model's answer.
    """
    idx = np.arange(len(items))
    dec = np.full((len(items), 2), np.nan)
    for half in (0, 1):
        train = [items[i] for i in idx if i % 2 != half]
        cls = edge_classes(edge_profile(train))
        mn, mx = [tuple(map(int, t.split('.'))) for t in cls['min_edge']], [tuple(map(int, t.split('.'))) for t in cls['max_edge']]
        for i in idx[idx % 2 == half]:
            it = items[i]
            p = it['attn'] / it['attn'].sum(-1, keepdims=True)
            for k in range(2):
                a = np.mean([p[LI[l], h, k] for l, h in mn], 0)
                b = np.mean([p[LI[l], h, k] for l, h in mx], 0)
                dec[i, k] = 0.5 * (local_peak(a, it['grid'])[1][k] + local_peak(b, it['grid'])[1][k])
    err, hit = [], []
    for it, d in zip(items, dec):
        c = cases[it['id']]
        W, H = c['truth']['canvas']
        cen = np.array(c['truth']['center'])
        err.append(float(np.linalg.norm((d - cen) * 1000 / np.array([W, H]))))
        b = c['truth']['box']
        hit.append(bool(b[0] <= d[0] <= b[2] and b[1] <= d[1] <= b[3]))
    return {'n': len(items), 'hit': prop(np.array(hit)), 'err_median': float(np.median(err))}


FLIP_BINS = (-0.21, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.21)
FLIP_HEADS = ('20.11', '24.11', '20.0', '24.7', '24.2', '20.10', '24.5', '24.9')


def inbox_centroid(attn: np.ndarray, grid, box) -> np.ndarray | None:
    """(8, 16, 2) box-normalized in-box attention centroid (u at the x query, v at the y query), or None."""
    H, W = grid
    j = np.arange(H * W)
    cen = np.stack([(j % W + 0.5) * TP, (j // W + 0.5) * TP], 1)
    uv = np.stack([(cen[:, 0] - box[0]) / (box[2] - box[0]), (cen[:, 1] - box[1]) / (box[3] - box[1])], 1)
    inb = ((uv >= 0) & (uv <= 1)).all(1)
    if inb.sum() < 4:
        return None
    q = attn[..., inb].astype(np.float32)
    q = q / q.sum(-1, keepdims=True)
    return np.stack([q[:, :, 0] @ uv[inb, 0], q[:, :, 1] @ uv[inb, 1]], -1)


def flip_summary(run: Path, source: Path, rng) -> dict:
    """Upright (2.5e readout) versus vertically flipped (flip stage) in-box centroids, RefCOCO persons.

    In the flipped image a person's head lies at the bottom of the box (v near 1). A head that reads the
    semantic part keeps following the head (v rises toward 1); a head that reads the top edge stays near v = 0.
    Only images whose upright and flipped answers both hit the target box are used.
    """
    from analysis_tools.position.coordinate_flow_natural import flipped_case
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    gen = {g['id']: g for g in sr.read_jsonl(source / 'gen.jsonl')}
    recs = read_jsonl(run / 'flip.jsonl')
    up, fl, flip_ok = {}, {}, []
    peak_v = {k: {'upright': [], 'flipped': []} for k in ('person', 'animal', 'other')}
    for r in recs:
        if r['group'] != 'rc':
            continue
        flip_ok.append(bool(r['score']['correct']))
        c = cases[r['id']]
        if not r['score']['correct'] or 'attn_check' not in r:
            continue
        sid = f"rc_point_{int(c['image_key'][2:]):04d}"
        mp = source / 'readout' / 'maps' / f'{sid}.npz'
        if not gen[sid]['score']['correct'] or not mp.exists():
            continue
        grid = tuple(r['grid'])
        a_up = np.load(mp)['attn'][:, :, :2]
        a_fl = np.load(run / 'flip_attn' / f"{r['id']}.npz")['attn']
        cat = category(c['expression'])
        fbox = flipped_case(c)['truth']['box']
        peak_v[cat]['upright'].append(float(box_uv(local_peak(a_up[LI[20], 11, 1].astype(np.float32), grid)[1], c['truth']['box'])[1]))
        peak_v[cat]['flipped'].append(float(box_uv(local_peak(a_fl[LI[20], 11, 1].astype(np.float32), grid)[1], fbox)[1]))
        if cat != 'person':
            continue
        u = inbox_centroid(a_up, grid, c['truth']['box'])
        f = inbox_centroid(a_fl, grid, fbox)
        if u is None or f is None:
            continue
        up[r['id']], fl[r['id']] = u, f
    U, F = np.stack(list(up.values())), np.stack(list(fl.values()))
    out = {'n_person': len(U), 'flip_hit': prop(np.array(flip_ok)), 'heads': list(FLIP_HEADS), 'upright': {}, 'flipped': {},
           'h11_peak_v': {}}
    for cat, d in peak_v.items():
        e = {}
        for cond, v in d.items():
            v = np.array(v)
            e[cond] = {'n': len(v), 'median': float(np.median(v)), 'top_fifth': prop(v < 0.2), 'bottom_two_fifths': prop(v > 0.6),
                       'hist': np.histogram(np.clip(v, -0.2, 1.2), bins=FLIP_BINS)[0].tolist()}
        out['h11_peak_v'][cat] = e
    out['bins'] = list(FLIP_BINS)
    for hd in FLIP_HEADS:
        l, h = (int(t) for t in hd.split('.'))
        for name, X in (('upright', U), ('flipped', F)):
            out[name][hd] = {'x': boot_mean(X[:, LI[l], h, 0], rng), 'y': boot_mean(X[:, LI[l], h, 1], rng)}
    return out


# ----------------------------------------------------------------------------- patching (Q1)

def patch_shares(run: Path) -> dict:
    """Per group and pair kind: r of every condition over values with d_A != d_B and |LD_B - LD_A| >= MIN_GAP.

    For the where pair only the value on the displaced axis is used; the what pair uses both values.
    """
    res: dict = {}
    for rec in read_unique(run / 'patch.jsonl'):
        if rec.get('status') != 'ok':
            continue
        for pr in rec['pairs']:
            for v in pr['values']:
                if pr['kind'] == 'where' and v['axis'] != pr['axis']:
                    continue
                ldA = v['LD']['none']
                gap = v['LD_B'] - ldA
                if v['dA'] == v['dB'] or abs(gap) < MIN_GAP:
                    continue
                key = (rec['group'], pr['kind'])
                d = res.setdefault(key, {'r': {}, 'ids': [], 'axis': []})
                d['ids'].append(rec['id'])
                d['axis'].append(v['axis'])
                for name, ld in v['LD'].items():
                    d['r'].setdefault(name, []).append((ld - ldA) / gap)
    return res


def patch_summary(sh: dict, rng) -> dict:
    out = {}
    for (grp, kind), d in sh.items():
        r = {k: np.array(v) for k, v in d['r'].items()}
        heads = np.array([[r[f'h{l}_{h}'].mean() for h in range(16)] for l in FULL_LAYERS])
        ent = {'n_values': len(d['ids']), 'n_images': len(set(d['ids'])), 'heads': heads.tolist(),
               'sum_single': float(heads.sum())}
        for name in ('full_all', 'L20', 'L24', 'L16', 'L32', 'L28', 'L20_L24', 'full_minus_L20', 'dn_all', 'dn_1_8',
                     'dn_9_16', 'dn_17_24', 'dn_25_32', 'all_mixers', 'h20_11', 'h32_8', 'h32_10'):
            if name in r:
                ent[name] = boot_mean(r[name], rng)
        loo = {}
        for l in (20, 24):
            for h in range(16):
                if f'loo{l}_{h}' in r:
                    loo[f'{l}.{h}'] = float(r['full_all'].mean() - r[f'loo{l}_{h}'].mean())
        ent['loo_drop'] = loo
        order = np.argsort(heads.ravel())[::-1]
        ent['top_heads'] = [[f'{FULL_LAYERS[i // 16]}.{i % 16}', float(heads.ravel()[i])] for i in order[:10]]
        cum = np.cumsum(np.sort(heads.ravel())[::-1])
        ent['k_for_half_sum'] = int(np.searchsorted(cum, 0.5 * cum[-1]) + 1) if cum[-1] > 0 else None
        out[f'{grp}:{kind}'] = ent
    return out


# ----------------------------------------------------------------------------- attention selectivity (Q1)

def selectivity(run: Path, rng) -> dict:
    """Referent selectivity S of each head at the first-digit queries (mean of the x and y queries).

    m_X(Y): attention mass (fraction of the head's image attention) on the tokens of object Y's box in
    run X. S = 0.5 * [(m_A(A) - m_A(B)) + (m_B(B) - m_B(A))]; S > 0 when the head follows the referent named by
    the expression. Also returned: the image attention share of objects A and B summed over runs (salience).
    """
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    out: dict = {}
    for rec in read_unique(run / 'patch.jsonl'):
        if rec.get('status') != 'ok':
            continue
        c = cases[rec['id']]
        f = run / 'attn' / f"{rec['id']}.npz"
        if 'truth_b' not in c or not f.exists():
            continue
        z = np.load(f)
        if 'what' not in z:
            continue
        grid = tuple(rec['grid'])
        ta = np.zeros(grid[0] * grid[1])
        tb = np.zeros_like(ta)
        ta[cr.tokens_overlapping(c['truth']['box'], grid)] = 1
        tb[cr.tokens_overlapping(c['truth_b']['box'], grid)] = 1
        both = (ta > 0) & (tb > 0)
        ta[both] = 0
        tb[both] = 0
        A = z['a'].astype(np.float32)
        B = z['what'].astype(np.float32)
        A = A / A.sum(-1, keepdims=True)
        B = B / B.sum(-1, keepdims=True)
        mAA, mAB, mBB, mBA = (A @ ta), (A @ tb), (B @ tb), (B @ ta)          # (8, 16, 2)
        S = 0.5 * ((mAA - mAB) + (mBB - mBA))
        sal = 0.25 * (mAA + mAB + mBB + mBA)
        d = out.setdefault(rec['group'], {'S': [], 'sal': [], 'pA': []})
        d['S'].append(S.mean(-1))
        d['sal'].append(sal.mean(-1))
        d['pA'].append((mAA.mean(-1)))
    res = {}
    for g, d in out.items():
        S = np.stack(d['S'])
        res[g] = {'n': len(S), 'S': S.mean(0).tolist(), 'salience': np.stack(d['sal']).mean(0).tolist(),
                  'S_H11': boot_mean(S[:, LI[20], 11], rng)}
    return res


# ----------------------------------------------------------------------------- generation and ablation

def gen_summary(run: Path) -> dict:
    out: dict = {}
    for g in read_jsonl(run / 'gen.jsonl'):
        d = out.setdefault(f"{g['group']}:{g['which']}", {'n': 0, 'valid': 0, 'correct': 0, 'wrong_object': 0})
        d['n'] += 1
        d['valid'] += int(g['score']['valid'])
        d['correct'] += int(g['score']['correct'])
        d['wrong_object'] += int(g['score'].get('wrong_object', False))
    sp = read_jsonl(run / 'spotcheck.jsonl')
    out['spotcheck'] = {'n': len(sp), 'answer_equal': sum(s['answer_equal'] for s in sp),
                        'text_equal': sum(s['text_equal'] for s in sp)}
    return out


def ablate_summary(run: Path, rng) -> dict:
    cases = {c['id']: c for c in json.loads((run / 'cases.json').read_text())}
    rows = read_unique(run / 'ablate.jsonl')
    out: dict = {}
    for rec in rows:
        c = cases[rec['case']]
        W, H = c['truth']['canvas']
        cen = np.array(c['truth']['center']) * 1000 / np.array([W, H])
        base = {r['set']: r for r in rec['results']}
        for name, r in base.items():
            d = out.setdefault(f"{rec['group']}@{rec['from']}", {}).setdefault(
                name, {'correct': [], 'wrong_object': [], 'err': [], 'valid': [], 'changed': [], 'off': []})
            sc = r['score']
            b = c['truth']['box']
            size = np.array([b[2] - b[0], b[3] - b[1]]) * 1000 / np.array([W, H])
            d['off'].append(((np.array(sc['out_units']) - cen) / size).tolist() if sc.get('out_units') else [np.nan, np.nan])
            d['valid'].append(sc['valid'])
            d['correct'].append(bool(sc['correct']))
            d['wrong_object'].append(bool(sc.get('wrong_object', False)))
            d['err'].append(float(np.linalg.norm(np.array(sc['out_units']) - cen)) if sc.get('out_units') else np.nan)
            d['changed'].append(r['text'] != base['none']['text'])
    res: dict = {}
    for gk, sets in out.items():
        res[gk] = {}
        base_ok = np.array(sets['none']['correct'])
        for name, d in sets.items():
            ok = np.array(d['correct'])
            b, cc = int(np.sum(ok & ~base_ok)), int(np.sum(~ok & base_ok))
            from math import comb
            nn = b + cc
            p = float(min(1.0, 2 * sum(comb(nn, i) for i in range(min(b, cc) + 1)) / 2 ** nn)) if nn else 1.0
            off = np.array(d['off'], float) - np.array(sets['none']['off'], float)
            res[gk][name] = {'shift_x': boot_mean(off[:, 0], rng), 'shift_y': boot_mean(off[:, 1], rng),
                             'offset_x': boot_mean(np.array(d['off'], float)[:, 0], rng),
                             'offset_y': boot_mean(np.array(d['off'], float)[:, 1], rng),
                             'correct': prop(ok), 'wrong_object': prop(np.array(d['wrong_object'])),
                             'valid': prop(np.array(d['valid'])), 'err_median': float(np.nanmedian(d['err'])),
                             'changed': prop(np.array(d['changed'])), 'mcnemar_p_vs_none': p}
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261021)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = salient_rows(args.run, args.source)
    (args.run / 'salient_rows.json').write_text(json.dumps(rows))
    summary = {'salient': salient_summary(rows, rng), 'gen': gen_summary(args.run)}
    prof = edge_profile(attn_items_2p5e(args.run, args.source))
    cases_all = {c['id']: c for c in json.loads((args.run / 'cases.json').read_text())}
    for grp in ('ms_large', 'ms_small'):
        its = []
        for rec in read_unique(args.run / 'patch.jsonl'):
            f = args.run / 'attn' / f"{rec['id']}.npz"
            if rec.get('status') == 'ok' and rec['group'] == grp and f.exists():
                its.append({'id': rec['id'], 'attn': np.load(f)['a'].astype(np.float32), 'grid': tuple(rec['grid']),
                            'box': cases_all[rec['id']]['truth']['box']})
        if len(its) >= 8:
            summary.setdefault('edges_ms', {})[grp] = edge_profile(its)
    items = attn_items_2p5e(args.run, args.source)
    summary['edges'] = {'rc_2p5e': prof, 'classes': edge_classes(prof),
                        'pair_decode': edge_pair_decode(items, {c['id']: c for c in json.loads((args.run / 'cases.json').read_text())})}
    sh = patch_shares(args.run)
    if sh:
        summary['patch'] = patch_summary(sh, rng)
        summary['select'] = selectivity(args.run, rng)
    if (args.run / 'flip.jsonl').exists():
        summary['flip'] = flip_summary(args.run, args.source, rng)
    if (args.run / 'ablate.jsonl').exists():
        summary['ablate'] = ablate_summary(args.run, rng)
    (args.run / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: (v if k != 'salient' else {kk: vv for kk, vv in v.items() if kk in ('n', 'model', 'categories')})
                      for k, v in summary.items() if k in ('salient', 'gen')}, indent=1)[:3000])


if __name__ == '__main__':
    main()
