"""Stage drivers of experiment 3.5 (coordinate readout path).

Stages (selected with --stages):
  c0      baselines: disc grid / translation / canvas-size series (point, bbox) and re-run of the
          experiment-1.3 T6 red-target grids (row-column, point, bbox) with the manual decoder.
  c1      attention readout at full-attention layers and gradient x activation attribution.
  c2      causal interventions on the position sources S1 (ViT), S2 (MRoPE ids), S4 (sequence order),
          the 2x2 conflict design, and C3 content swap with a recognition check.
  fail    targeted failure analysis of the T6 grids (MRoPE shifts/scaling, candidate-answer scoring).

Every stage writes JSONL records under the run directory and is resumable: finished record ids
are skipped on restart.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr

DIAMETERS = (32, 64, 128)
DIGITS = [str(d) for d in range(10)]


# ----------------------------------------------------------------------------- io helpers

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')


def done_ids(path: Path) -> set[str]:
    return {r['id'] for r in read_jsonl(path)}


def code_digest() -> dict:
    here = Path(__file__).parent
    names = ('coordinate_readout.py', 'coordinate_readout_run.py')
    return {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in names}


# ----------------------------------------------------------------------------- cases

def disc_cases(seed: int) -> list[dict]:
    """C0 single-target cases: 6x6 jittered grid x 3 diameters, translation series, canvas-size series."""
    rng = np.random.default_rng(seed)
    cases = []
    for d in DIAMETERS:
        for i, (cx, cy) in enumerate(cr.grid_centers(6)):
            jx, jy = rng.integers(-16, 17, 2).tolist()
            for fmt in ('point', 'bbox'):
                cases.append({'id': f'grid_d{d}_p{i:02d}_{fmt}', 'set': 'grid', 'diameter': d, 'center': [cx + jx, cy + jy],
                              'canvas': cr.CANVAS, 'format': fmt, 'image_key': f'grid_d{d}_p{i:02d}'})
    for d in DIAMETERS:
        r = d / 2
        for axis in ('h', 'v'):
            for k in range(-7, 8):
                off = 32 * k
                if abs(off) > cr.CANVAS / 2 - r:
                    continue                      # the disc must lie fully inside the canvas
                c = [cr.CANVAS / 2 + off, cr.CANVAS / 2] if axis == 'h' else [cr.CANVAS / 2, cr.CANVAS / 2 + off]
                cases.append({'id': f'trans_d{d}_{axis}{k:+d}_point', 'set': 'trans', 'diameter': d, 'center': c,
                              'canvas': cr.CANVAS, 'format': 'point', 'axis': axis, 'offset': off,
                              'image_key': f'trans_d{d}_{axis}{k:+d}'})
    # Canvas-size series: the same relative disc position on canvases of 256-1280 pixels (8-40 merged tokens).
    for size in (256, 384, 512, 768, 1024, 1280):
        for i, (u, v) in enumerate([(0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8), (0.5, 0.5), (0.35, 0.65), (0.65, 0.35)]):
            d = max(32, size // 8)
            cases.append({'id': f'size{size}_q{i}_point', 'set': 'size', 'diameter': d, 'center': [u * size, v * size],
                          'canvas': size, 'format': 'point', 'image_key': f'size{size}_q{i}'})
    return cases


def disc_input(case: dict) -> tuple[Image.Image, dict, str]:
    image, truth = cr.disc_image(tuple(case['center']), case['diameter'], case['canvas'])
    return image, truth, cr.DISC_QUERY + cr.SUFFIX[case['format']]


def t6_cases(source: Path, grids=(6, 10, 20)) -> list[dict]:
    """Experiment-1.3 T6 (red target) cases, primary wording, with their stored responses."""
    cases = [c for c in read_jsonl(source / 'cases.jsonl') if c['task'] == 'T6' and c['stage'] == 'main'
             and c['wording'] == 'primary' and c['grid'] in grids]
    results = {r['id']: r for r in read_jsonl(source / 'results_main.jsonl')}
    out = []
    for c in cases:
        fmt = {'row_column': 'rowcol'}.get(c['format'], c['format'])
        out.append({'id': f"t6_g{c['grid']:02d}_{c['record_id']}_{fmt}", 'set': 't6', 'grid': c['grid'], 'format': fmt,
                    'image_path': str(source / c['image']), 'image_sha256': c['image_sha256'], 'prompt': c['prompt'],
                    'truth': c['truth'], 'prior_response': results[c['id']]['response'], 'source_id': c['id'],
                    'image_key': f"t6_g{c['grid']:02d}_{c['record_id']}"})
    return out


def t6_input(case: dict) -> tuple[Image.Image, dict, str]:
    image = Image.open(case['image_path']).convert('RGB')
    if hashlib.sha256(Path(case['image_path']).read_bytes()).hexdigest() != case['image_sha256']:
        raise ValueError(f"Image hash mismatch for {case['image_path']}")
    return image, case['truth'], case['prompt']


def score_t6(case: dict, text: str) -> dict:
    """Score T6 answers: row-column exact match, point/bbox center cell and offsets in cell units."""
    t = case['truth']
    tr, tc = t['target_rc']
    if case['format'] == 'rowcol':
        rc = cr.parse_rowcol(text)
        if rc is None:
            return {'valid': False, 'correct': False}
        return {'valid': True, 'correct': (rc[0] - 1, rc[1] - 1) == (tr, tc), 'pred_rc': [rc[0] - 1, rc[1] - 1],
                'd_row': rc[0] - 1 - tr, 'd_col': rc[1] - 1 - tc}
    coords = cr.parse_coords(text, case['format'])
    if coords is None:
        return {'valid': False, 'correct': False}
    size = np.asarray(t['canvas'], float)
    center = cr.answer_center(coords) * size / 1000
    ox, oy = t['grid_origin']
    col, row = int((center[0] - ox) // t['cell']), int((center[1] - oy) // t['cell'])
    off = (center - np.asarray(t['target_center'])) / t['cell']
    out = {'valid': True, 'center_px': center.tolist(), 'pred_rc': [row, col], 'hit': (row, col) == (tr, tc),
           'offset_cells': off.tolist(), 'coords': coords}
    if case['format'] == 'bbox':
        box = (np.asarray(coords) * np.tile(size, 2) / 1000).tolist()
        out['iou'] = cr.box_iou(box, t['target_tight_box'])
    out['correct'] = out['hit']
    return out


def case_input(case: dict):
    return t6_input(case) if case['set'] == 't6' else disc_input(case)


def score(case: dict, truth: dict, text: str) -> dict:
    return score_t6(case, text) if case['set'] == 't6' else cr.score_disc(truth, case['format'], text)


# ----------------------------------------------------------------------------- C0

def stage_c0(rt, out: Path, args) -> None:
    path = out / 'baseline' / 'records.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = disc_cases(args.seed) + t6_cases(Path(args.t6_source))
    (out / 'cases.json').write_text(json.dumps(cases, indent=1))
    finished = done_ids(path)
    todo = [c for c in cases if c['id'] not in finished]
    # Group by (prompt, canvas) so that each decoding batch has equal sequence length without padding.
    groups: dict[tuple, list[dict]] = {}
    for c in todo:
        groups.setdefault((c['set'] == 't6' and c['grid'], c.get('canvas'), c['format']), []).append(c)
    for key, group in groups.items():
        for i in range(0, len(group), args.batch_size):
            chunk = group[i:i + args.batch_size]
            preps, truths = [], []
            for c in chunk:
                image, truth, text = case_input(c)
                preps.append(cr.prepare(rt, image, text))
                truths.append(truth)
            lengths = {len(p.ids) for p in preps}
            if len(lengths) != 1:
                raise ValueError(f'Unequal prompt lengths in group {key}: {lengths}')
            t0 = time.time()
            res = cr.run_conditions(rt, [cr.embeds(rt, p) for p in preps], [p.pos for p in preps],
                                    args.max_new_tokens, batch_size=args.batch_size)
            rows = []
            for c, truth, r in zip(chunk, truths, res):
                row = {'id': c['id'], 'set': c['set'], 'format': c['format'], 'text': r['text'], 'ids': r['ids'],
                       'finish_reason': r['finish_reason'], 'truth': truth, 'score': score(c, truth, r['text']),
                       'batch_size': len(chunk)}
                if c['set'] == 't6':
                    row['prior_response'] = c['prior_response']
                    row['prior_agree'] = c['prior_response'].strip() == r['text'].strip()
                rows.append(row)
            append_jsonl(path, rows)
            print(f'c0 {key} {i + len(chunk)}/{len(group)} {time.time() - t0:.1f}s', flush=True)


def stage_spotcheck(rt, out: Path, args) -> None:
    """Answer-level agreement of the manual batched decoder with the reference generate()."""
    path = out / 'baseline' / 'spotcheck.jsonl'
    finished = done_ids(path)
    records = {r['id']: r for r in read_jsonl(out / 'baseline' / 'records.jsonl')}
    cases = [c for c in json.loads((out / 'cases.json').read_text()) if c['id'] in records]
    rng = np.random.default_rng(args.seed)
    pick = [cases[i] for i in sorted(rng.choice(len(cases), size=min(args.n_spot, len(cases)), replace=False))]
    for c in pick:
        if c['id'] in finished:
            continue
        image, truth, text = case_input(c)
        ref = qr.generate(rt, qr.build(rt, [image], text), args.max_new_tokens)
        rec = records[c['id']]
        append_jsonl(path, [{'id': c['id'], 'reference': ref['text'], 'manual': rec['text'],
                             'text_equal': ref['text'].strip() == rec['text'].strip(),
                             'answer_equal': score(c, truth, ref['text']).get('coords', score(c, truth, ref['text']).get('pred_rc'))
                             == rec['score'].get('coords', rec['score'].get('pred_rc'))}])


# ----------------------------------------------------------------------------- C1

def special_positions(rt, prep: cr.Prepared) -> dict:
    """Prompt positions used as queries: question end, </think>, last prompt token."""
    ids = prep.ids.tolist()
    im_end = rt.tokenizer.convert_tokens_to_ids('<|im_end|>')
    think_end = rt.tokenizer.convert_tokens_to_ids('</think>')
    return {'q_end': ids.index(im_end) - 1, 'think_end': len(ids) - 1 - ids[::-1].index(think_end),
            'last': len(ids) - 1}


def target_tokens(case: dict, truth: dict, grid) -> list[int]:
    box = truth['target_tight_box'] if case['set'] == 't6' else truth['box']
    return cr.tokens_overlapping(box, grid)


def head_metrics(a: np.ndarray, c: np.ndarray, target: list[int], centers: np.ndarray) -> dict:
    """a, c: (layers, heads, Q, Nv) attention and effective contribution over visual tokens."""
    tmask = np.zeros(a.shape[-1], bool)
    tmask[target] = True
    frac = tmask.mean()
    def enrich(w):
        tot = w.sum(-1)
        return (w[..., tmask].sum(-1) / np.maximum(tot, 1e-30)) / frac
    out = {'vis_mass': a.sum(-1), 'E': enrich(a), 'E_eff': enrich(c),
           'hit': tmask[a.argmax(-1)], 'hit_eff': tmask[c.argmax(-1)]}
    wn = a / np.maximum(a.sum(-1, keepdims=True), 1e-30)
    out['centroid'] = wn @ centers                                         # (layers, heads, Q, 2)
    return out


def stage_c1(rt, out: Path, args) -> None:
    root = out / 'attention'
    (root / 'maps').mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    base = read_jsonl(out / 'baseline' / 'records.jsonl')
    sets = set(args.c1_sets.split(','))
    digit_ids = [rt.tokenizer.convert_tokens_to_ids(d) for d in DIGITS]
    for rec in base:
        case = cases[rec['id']]
        if rec['id'] in finished or case['set'] not in sets or case['format'] not in ('point', 'bbox', 'rowcol'):
            continue
        if case['set'] == 't6' and case['format'] == 'bbox':
            continue
        t0 = time.time()
        image, truth, text = case_input(case)
        prep = cr.prepare(rt, image, text)
        coord = cr.coordinate_tokens(rt, rec['ids'], case['format'])
        if not coord:
            append_jsonl(path, [{'id': rec['id'], 'status': 'no_coordinate_tokens'}])
            continue
        x, pos = cr.teacher_forced(rt, prep, rec['ids'][:-1])
        L = len(prep.ids)
        sp = special_positions(rt, prep)
        # Query position predicting generated token i is L + i - 1.
        digit_q = [[L + i - 1 for i in toks] for toks in coord]
        queries = [sp['q_end'], sp['think_end'], sp['last']] + [q for qs in digit_q for q in qs]
        names = ['q_end', 'think_end', 'last'] + [f'v{k}d{j}' for k, qs in enumerate(digit_q) for j in range(len(qs))]
        s, e = prep.span['start'], prep.span['end']
        ro = cr.attention_readout(rt, x, pos, queries, (s, e))
        a = np.stack([ro.attn[l][:, :, s:e].numpy() for l in cr.FULL_LAYERS])          # (8, heads, Q, Nv)
        c = np.stack([ro.contrib[l].numpy() for l in cr.FULL_LAYERS])
        # Attention mass on prompt segments for the relay analysis.
        seg = {'image': (s, e), 'pre': (0, s), 'question': (e, sp['q_end'] + 1), 'template': (sp['q_end'] + 1, L),
               'answer': (L, x.shape[1])}
        seg_mass = {k: np.stack([ro.attn[l][:, :, u:v].sum(-1).numpy() for l in cr.FULL_LAYERS]).tolist()
                    for k, (u, v) in seg.items()}
        target = target_tokens(case, truth, prep.grid)
        m = head_metrics(a, c, target, cr.token_centers(prep.grid))
        key = [0, 1, 2] + [names.index(f'v{k}d0') for k in range(len(coord))]
        # Gradient x activation for the first digit of each coordinate value.
        targets = [(digit_q[k][0], rec['ids'][coord[k][0]]) for k in range(len(coord))]
        try:
            A = cr.grad_attribution(rt, x, pos, targets, digit_ids)['A'].numpy()      # (n, 32, Lx)
        except torch.OutOfMemoryError:
            # Long prompts (20x20 grids) may not fit a full backward pass on a shared GPU; the
            # attention readout is kept and the attribution is recorded as missing, never imputed.
            torch.cuda.empty_cache()
            A = np.full((len(targets), len(rt.layers), x.shape[1]), np.nan, np.float32)
        tm = np.zeros(e - s, bool)
        tm[target] = True
        absA = np.abs(A)
        rho = absA[:, :, s:e][:, :, tm].sum(-1) / np.maximum(absA[:, :, s:e].sum(-1), 1e-30)
        seg_grad = {k: absA[:, :, u:v].sum(-1).tolist() for k, (u, v) in seg.items()}
        np.savez_compressed(root / 'maps' / f"{rec['id']}.npz", attn=a[:, :, key].astype(np.float16),
                            contrib=c[:, :, key].astype(np.float16), grad_vis=A[:, :, s:e].astype(np.float16))
        append_jsonl(path, [{
            'id': rec['id'], 'status': 'ok', 'set': case['set'], 'format': case['format'], 'grid': list(prep.grid),
            'correct': bool(rec['score'].get('correct')), 'score': rec['score'], 'query_names': names,
            'query_positions': queries, 'key_queries': key, 'special': sp, 'span': [s, e], 'target_tokens': target,
            'check': ro.check, 'seg_mass': seg_mass, 'rho': rho.tolist(),
            'grad_share_image': (absA[:, :, s:e].sum(-1) / absA.sum(-1)).tolist(), 'grad_segments': seg_grad,
            **{k: v.tolist() for k, v in m.items()}, 'seconds': round(time.time() - t0, 2)}])
        print(f"c1 {rec['id']} {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- C2 / C3

SHIFTS = (-8, -4, -2, 2, 4, 8)            # merged units for MRoPE shifts (S2)
PE_MARGIN = 16                            # virtual-canvas margin in ViT patches for S1 shifts
SWAP_DISTANCES = (2, 4, 6)                # merged units between T and T'
ROPE_SCALES = (0.5, 0.8, 1.25)             # ViT RoPE id scaling about the image center


def swap_regions(target: list[int], grid, center_px, canvas) -> list[dict]:
    """T' = T translated by k merged units horizontally or vertically toward the larger free side."""
    H, W = grid
    rows = sorted({t // W for t in target})
    cols = sorted({t % W for t in target})
    out = []
    for axis in ('h', 'v'):
        toward = 1 if (center_px[0] if axis == 'h' else center_px[1]) < canvas / 2 else -1
        for k in SWAP_DISTANCES:
            dr, dc = (0, toward * k) if axis == 'h' else (toward * k, 0)
            moved = [(t // W + dr) * W + (t % W + dc) for t in target]
            ok = all(0 <= r + dr < H for r in rows) and all(0 <= c + dc < W for c in cols) and not set(moved) & set(target)
            out.append({'axis': axis, 'k': k, 'dr': dr, 'dc': dc, 'valid': ok, 'region': moved if ok else None,
                        'd': [1000 * dc / W, 1000 * dr / H]})
    return out


def stage_c2(rt, out: Path, args) -> None:
    """Position-source interventions on single-disc point (and bbox) cases."""
    root = out / 'interventions'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    digit_ids = [rt.tokenizer.convert_tokens_to_ids(d) for d in DIGITS]
    yes, no = qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No')
    fmts = set(args.c2_formats.split(','))
    base = [r for r in read_jsonl(out / 'baseline' / 'records.jsonl')
            if cases[r['id']]['set'] == 'grid' and r['format'] in fmts]
    for rec in base:
        if rec['id'] in finished:
            continue
        t0 = time.time()
        case = cases[rec['id']]
        image, truth, text = case_input(case)
        prep = cr.prepare(rt, image, text)
        H, W = prep.grid
        target = target_tokens(case, truth, prep.grid)
        regions = swap_regions(target, prep.grid, truth['center'], case['canvas'])
        conds = []                                        # (name, x, pos, d, meta)
        x0 = cr.embeds(rt, prep)
        conds.append(('standard', x0, prep.pos, None, {}))
        gp = cr.gap_pos(prep, args.gap)
        conds.append(('gap', x0, gp, None, {}))
        for dl in SHIFTS:
            conds.append((f's2shift_h{dl:+d}', x0, cr.shift_image_pos(prep, gp, dl, 0), [0, 1000 * dl / H],
                          {'ref': 'gap', 'axis': 'v', 'delta': dl}))
            conds.append((f's2shift_w{dl:+d}', x0, cr.shift_image_pos(prep, gp, 0, dl), [1000 * dl / W, 0],
                          {'ref': 'gap', 'axis': 'h', 'delta': dl}))
        pe0 = cr.shifted_pe(prep, PE_MARGIN, 0, 0)
        conds.append(('s1center', cr.embeds(rt, prep, cr.vision_embeds(rt, prep, interp=pe0)), prep.pos, None, {}))
        for dl in SHIFTS:                                   # same pixel displacement as S2: 2 patches per unit
            for axis, (dr, dc) in (('v', (2 * dl, 0)), ('h', (0, 2 * dl))):
                if abs(2 * dl) > PE_MARGIN:
                    continue
                v = cr.vision_embeds(rt, prep, interp=cr.shifted_pe(prep, PE_MARGIN, dr, dc))
                # A window shifted by +dc patches carries the PE of patches further right, so content
                # would be read as lying further right: predicted displacement +dc patches.
                conds.append((f's1shift_{axis}{dl:+d}', cr.embeds(rt, prep, v), prep.pos,
                              [1000 * dc / (2 * W), 1000 * dr / (2 * H)], {'ref': 's1center', 'axis': axis, 'delta': dl}))
        # ViT RoPE: a global shift must leave the output unchanged (relative encoding; implementation check);
        # scaling the ids about the image center changes all relative offsets between patches.
        conds.append(('s1rope_shift', cr.embeds(rt, prep, cr.vision_embeds(rt, prep, vit_pos=prep.vit_pos + 7)),
                      prep.pos, None, {'ref': 'standard'}))
        center = (prep.vit_pos.float().amax(0) + prep.vit_pos.float().amin(0)) / 2
        for a in ROPE_SCALES:
            vp = center + a * (prep.vit_pos.float() - center)
            conds.append((f's1rope_scale{a}', cr.embeds(rt, prep, cr.vision_embeds(rt, prep, vit_pos=vp)), prep.pos,
                          None, {'ref': 'standard', 'scale': a}))
        for reg in regions:
            if not reg['valid']:
                continue
            tag = f"{reg['axis']}{reg['k']}"
            b = reg['region']
            pos_sw = cr.swap_image_pos(prep, prep.pos, target, b)
            conds.append((f's2swap_{tag}', x0, pos_sw, reg['d'], {'ref': 'standard'}))
            for name, pe, rope in (('pe', True, False), ('rope', False, True), ('both', True, True)):
                vp, ip = cr.swap_vit(prep, target, b, pe, rope)
                v = cr.vision_embeds(rt, prep, vit_pos=vp, interp=ip)
                conds.append((f's1swap_{name}_{tag}', cr.embeds(rt, prep, v), prep.pos, reg['d'], {'ref': 'standard'}))
                if name == 'both':
                    conds.append((f'conflict_both_{tag}', cr.embeds(rt, prep, v), pos_sw, reg['d'], {'ref': 'standard'}))
            order = cr.swap_index(H * W, target, b)
            # Sequence permutation: slot i holds token order[i] with that token's own embedding and ids.
            conds.append((f's4perm_{tag}', cr.embeds(rt, prep, order=order), cr.permute_pos(prep, prep.pos, order),
                          reg['d'], {'ref': 'standard'}))
            # Content swap (C3): embeddings exchanged, ids and slots unchanged.
            conds.append((f'c3content_{tag}', cr.embeds(rt, prep, order=order), prep.pos, reg['d'], {'ref': 'standard'}))
            # Natural control: the disc actually drawn at the T' location.
            moved = [truth['center'][0] + 32 * reg['dc'], truth['center'][1] + 32 * reg['dr']]
            img2, truth2 = cr.disc_image(tuple(moved), case['diameter'], case['canvas'])
            prep2 = cr.prepare(rt, img2, text)
            conds.append((f'real_{tag}', cr.embeds(rt, prep2), prep2.pos, reg['d'], {'ref': 'standard'}))
        res = cr.run_conditions(rt, [c[1] for c in conds], [c[2] for c in conds], args.c2_max_new_tokens,
                                batch_size=args.batch_size, track_ids=digit_ids)
        rows = []
        for (name, _, _, d, meta), r in zip(conds, res):
            coords = cr.parse_coords(r['text'], case['format'])
            ct = cr.coordinate_tokens(rt, r['ids'], case['format']) if coords else []
            first_digit_logits = [r['tracked'][toks[0]] for toks in ct] if ct else None
            rows.append({'condition': name, 'd': d, **meta, 'text': r['text'], 'finish_reason': r['finish_reason'],
                         'coords': coords, 'center': None if coords is None else cr.answer_center(coords).tolist(),
                         'first_digit_logits': first_digit_logits})
        # Recognition check for the content swaps (Yes - No logit at the first answer token).
        rprep = cr.prepare(rt, image, cr.RECOGNITION_QUESTION)
        recog = {}
        with torch.inference_mode():
            for name, order in [('standard', None)] + [(f"c3content_{g['axis']}{g['k']}", cr.swap_index(H * W, target, g['region']))
                                                     for g in regions if g['valid']]:
                lg = rt.model(inputs_embeds=cr.embeds(rt, rprep, order=order), position_ids=rprep.pos,
                              logits_to_keep=1).logits[0, -1].float()
                recog[name] = float(lg[yes] - lg[no])
        append_jsonl(path, [{'id': rec['id'], 'format': case['format'], 'diameter': case['diameter'], 'grid': [H, W],
                             'truth_center': (1000 * np.asarray(truth['center']) / case['canvas']).tolist(),
                             'baseline_correct': bool(rec['score'].get('correct')), 'target_tokens': target,
                             'regions': regions, 'conditions': rows, 'recognition': recog,
                             'batch_size': args.batch_size, 'seconds': round(time.time() - t0, 1)}])
        print(f"c2 {rec['id']} {len(conds)} conditions {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- targeted failure analysis

FAIL_SHIFTS = (-4, -2, 2, 4)              # merged units; one 64-pixel grid cell = 2 units
FAIL_SCALES = (0.9, 0.95, 1.05, 1.1)


def candidate_logprobs(rt, prep, ids: list[int], toks: list[int], values: list[int]) -> dict:
    """Teacher-forced log p of alternative values for one generated number.

    The digits at `toks` are replaced by each candidate's digits; the score sums log-probabilities of
    the candidate digits and of the token that followed the number, so that 1 and 12 are distinguished.
    """
    out = {}
    end = toks[-1] + 1
    for v in values:
        if v < 1:
            continue
        digits = [rt.tokenizer.convert_tokens_to_ids(ch) for ch in str(v)]
        seq = ids[:toks[0]] + digits + ids[end:end + 1]
        x, pos = cr.teacher_forced(rt, prep, seq[:-1])
        L = len(prep.ids)
        with torch.inference_mode():
            logits = rt.model(inputs_embeds=x, position_ids=pos).logits[0].float().log_softmax(-1)
        span = range(toks[0], toks[0] + len(digits) + 1)
        out[v] = float(sum(logits[L + i - 1, seq[i]] for i in span))
    return out


def stage_fail(rt, out: Path, args) -> None:
    root = out / 'failure'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    digit_ids = [rt.tokenizer.convert_tokens_to_ids(d) for d in DIGITS]
    # Priority order of (grid, format) blocks, e.g. "20:rowcol,10:point"; later blocks run only if time allows.
    order = [(int(g), f) for g, f in (item.split(':') for item in args.fail_order.split(','))]
    by_key = {}
    for r in read_jsonl(out / 'baseline' / 'records.jsonl'):
        c = cases[r['id']]
        if c['set'] == 't6':
            by_key.setdefault((c['grid'], r['format']), []).append(r)
    base = [r for key in order for r in by_key.get(key, [])]
    for rec in base:
        if rec['id'] in finished:
            continue
        t0 = time.time()
        case = cases[rec['id']]
        image, truth, text = case_input(case)
        prep = cr.prepare(rt, image, text)
        H, W = prep.grid
        x0 = cr.embeds(rt, prep)
        gp = cr.gap_pos(prep, args.gap)
        conds = [('standard', x0, prep.pos, {}), ('gap', x0, gp, {})]
        for dl in FAIL_SHIFTS:
            conds.append((f's2shift_h{dl:+d}', x0, cr.shift_image_pos(prep, gp, dl, 0), {'axis': 'v', 'delta': dl}))
            conds.append((f's2shift_w{dl:+d}', x0, cr.shift_image_pos(prep, gp, 0, dl), {'axis': 'h', 'delta': dl}))
        for a in FAIL_SCALES:
            conds.append((f's2scale_h{a}', x0, cr.scale_image_pos(prep, prep.pos, a, 1.0), {'axis': 'v', 'scale': a}))
            conds.append((f's2scale_w{a}', x0, cr.scale_image_pos(prep, prep.pos, 1.0, a), {'axis': 'h', 'scale': a}))
        pe0 = cr.shifted_pe(prep, PE_MARGIN, 0, 0)
        conds.append(('s1center', cr.embeds(rt, prep, cr.vision_embeds(rt, prep, interp=pe0)), prep.pos, {}))
        for dl in (-4, 4):                                   # patches = one 64-pixel cell
            for axis, (dr, dc) in (('v', (dl, 0)), ('h', (0, dl))):
                v = cr.vision_embeds(rt, prep, interp=cr.shifted_pe(prep, PE_MARGIN, dr, dc))
                conds.append((f's1shift_{axis}{dl:+d}', cr.embeds(rt, prep, v), prep.pos, {'axis': axis, 'delta_patches': dl}))
        # Region swaps between the target cell tokens and the tokens one cell (2 merged units) away.
        target = target_tokens(case, truth, prep.grid)
        tr, tc = truth['target_rc']
        for tag, (dr, dc) in (('right', (0, 2)), ('left', (0, -2)), ('down', (2, 0)), ('up', (-2, 0))):
            moved = [(t // W + dr) * W + (t % W + dc) for t in target]
            if not all(0 <= t // W + dr < H and 0 <= t % W + dc < W for t in target) or set(moved) & set(target):
                continue
            meta = {'axis': 'h' if dc else 'v', 'cells': (dc or dr) // 2}
            vp, ip = cr.swap_vit(prep, target, moved, False, True)
            conds.append((f's1rope_swap_{tag}', cr.embeds(rt, prep, cr.vision_embeds(rt, prep, vit_pos=vp, interp=ip)),
                          prep.pos, meta))
            conds.append((f's2swap_{tag}', x0, cr.swap_image_pos(prep, prep.pos, target, moved), meta))
            conds.append((f'c3content_{tag}', cr.embeds(rt, prep, order=cr.swap_index(H * W, target, moved)), prep.pos, meta))
        center = (prep.vit_pos.float().amax(0) + prep.vit_pos.float().amin(0)) / 2
        for a in FAIL_SCALES:
            for axis, col in (('v', 0), ('h', 1)):
                vp = prep.vit_pos.float().clone()
                vp[:, col] = center[col] + a * (vp[:, col] - center[col])
                conds.append((f's1rope_scale_{axis}{a}', cr.embeds(rt, prep, cr.vision_embeds(rt, prep, vit_pos=vp)),
                              prep.pos, {'axis': axis, 'scale': a}))
        res = cr.run_conditions(rt, [c[1] for c in conds], [c[2] for c in conds], args.max_new_tokens,
                                batch_size=args.batch_size, track_ids=digit_ids)
        rows = []
        for (name, _, _, meta), r in zip(conds, res):
            rows.append({'condition': name, **meta, 'text': r['text'], 'finish_reason': r['finish_reason'],
                         'score': score_t6(case, r['text'])})
        extra = {}
        if case['format'] == 'rowcol' and rec['score'].get('valid'):
            ct = cr.coordinate_tokens(rt, rec['ids'], 'rowcol')
            tr, tc = truth['target_rc']
            if ct:
                extra['row_logprob'] = candidate_logprobs(rt, prep, rec['ids'], ct[0], [tr, tr + 1, tr + 2])
                extra['col_logprob'] = candidate_logprobs(rt, prep, rec['ids'], ct[1], [tc, tc + 1, tc + 2])
        append_jsonl(path, [{'id': rec['id'], 'grid': case['grid'], 'format': case['format'], 'merged_grid': [H, W],
                             'truth': truth, 'baseline_score': rec['score'], 'conditions': rows, **extra,
                             'batch_size': args.batch_size, 'seconds': round(time.time() - t0, 1)}])
        print(f"fail {rec['id']} {time.time() - t0:.1f}s", flush=True)


def stage_fail2(rt, out: Path, args) -> None:
    """Supplementary slot interventions on T6 grids: sequence permutation and 3-cell swaps."""
    root = out / 'failure'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records_slot.jsonl'
    finished = done_ids(path)
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    order = [(int(g), f) for g, f in (item.split(':') for item in args.fail_order.split(','))]
    by_key = {}
    for r in read_jsonl(out / 'baseline' / 'records.jsonl'):
        c = cases[r['id']]
        if c['set'] == 't6':
            by_key.setdefault((c['grid'], r['format']), []).append(r)
    for rec in [r for key in order for r in by_key.get(key, [])]:
        if rec['id'] in finished:
            continue
        t0 = time.time()
        case = cases[rec['id']]
        image, truth, text = case_input(case)
        prep = cr.prepare(rt, image, text)
        H, W = prep.grid
        x0 = cr.embeds(rt, prep)
        target = target_tokens(case, truth, prep.grid)
        conds = [('standard', x0, prep.pos, {})]
        for cells in (1, 3):
            for tag, (dr, dc) in (('right', (0, 2)), ('left', (0, -2)), ('down', (2, 0)), ('up', (-2, 0))):
                dr, dc = dr * cells, dc * cells
                moved = [(t // W + dr) * W + (t % W + dc) for t in target]
                if not all(0 <= t // W + dr < H and 0 <= t % W + dc < W for t in target) or set(moved) & set(target):
                    continue
                meta = {'axis': 'h' if dc else 'v', 'cells': (dc or dr) // 2}
                perm = cr.swap_index(H * W, target, moved)
                conds.append((f's4perm_{tag}{cells}', cr.embeds(rt, prep, order=perm), cr.permute_pos(prep, prep.pos, perm), meta))
                conds.append((f'c3content_{tag}{cells}', cr.embeds(rt, prep, order=perm), prep.pos, meta))
                vp, ip = cr.swap_vit(prep, target, moved, True, True)
                conds.append((f's1swap_both_{tag}{cells}', cr.embeds(rt, prep, cr.vision_embeds(rt, prep, vit_pos=vp, interp=ip)),
                              prep.pos, meta))
        res = cr.run_conditions(rt, [c[1] for c in conds], [c[2] for c in conds], args.max_new_tokens,
                                batch_size=args.batch_size)
        rows = [{'condition': name, **meta, 'text': r['text'], 'finish_reason': r['finish_reason'],
                 'score': score_t6(case, r['text'])} for (name, _, _, meta), r in zip(conds, res)]
        append_jsonl(path, [{'id': rec['id'], 'grid': case['grid'], 'format': case['format'], 'merged_grid': [H, W],
                             'truth': truth, 'baseline_score': rec['score'], 'conditions': rows,
                             'batch_size': args.batch_size, 'seconds': round(time.time() - t0, 1)}])
        print(f"fail2 {rec['id']} {time.time() - t0:.1f}s", flush=True)


def stage_c1grad(rt, out: Path, args) -> None:
    """Recompute missing gradient x activation attributions (records whose rho is NaN) in place."""
    root = out / 'attention'
    path = root / 'records.jsonl'
    records = read_jsonl(path)
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    base = {r['id']: r for r in read_jsonl(out / 'baseline' / 'records.jsonl')}
    digit_ids = [rt.tokenizer.convert_tokens_to_ids(d) for d in DIGITS]
    changed = 0
    for rec in records:
        if rec.get('status') != 'ok' or np.isfinite(np.asarray(rec['rho'], float)).all():
            continue
        b = base[rec['id']]
        case = cases[rec['id']]
        image, truth, text = case_input(case)
        prep = cr.prepare(rt, image, text)
        coord = cr.coordinate_tokens(rt, b['ids'], case['format'])
        x, pos = cr.teacher_forced(rt, prep, b['ids'][:-1])
        L = len(prep.ids)
        targets = [(L + toks[0] - 1, b['ids'][toks[0]]) for toks in coord]
        A = cr.grad_attribution(rt, x, pos, targets, digit_ids)['A'].numpy()
        s, e = rec['span']
        tm = np.zeros(e - s, bool)
        tm[rec['target_tokens']] = True
        absA = np.abs(A)
        rec['rho'] = (absA[:, :, s:e][:, :, tm].sum(-1) / np.maximum(absA[:, :, s:e].sum(-1), 1e-30)).tolist()
        rec['grad_share_image'] = (absA[:, :, s:e].sum(-1) / absA.sum(-1)).tolist()
        maps = dict(np.load(root / 'maps' / f"{rec['id']}.npz"))
        maps['grad_vis'] = A[:, :, s:e].astype(np.float16)
        np.savez_compressed(root / 'maps' / f"{rec['id']}.npz", **maps)
        changed += 1
        print(f"c1grad {rec['id']}", flush=True)
        if changed % 8 == 0:
            path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))


# ----------------------------------------------------------------------------- C4

class HeadAblation:
    """Replace the gated outputs of selected heads (o_proj input slices) from a sequence position onward.

    heads: list of (layer, head); means: (layer, head) -> (head_dim,) replacement vector.
    Positions before `start` (the prompt up to its last token) are untouched; every decoding step is ablated.
    """

    def __init__(self, rt, heads, means, start: int):
        self.handles = []
        hd = rt.layers[heads[0][0] - 1].self_attn.head_dim if heads else 0
        by_layer = {}
        for l, h in heads:
            by_layer.setdefault(l, []).append(h)
        for l, hs in by_layer.items():
            def pre(module, args, hs=hs, l=l):
                x = args[0]
                lo = start if x.shape[1] > 1 else 0
                x = x.clone()
                for h in hs:
                    x[:, lo:, h * hd:(h + 1) * hd] = means[(l, h)].to(x.dtype)
                return (x,)
            self.handles.append(rt.layers[l - 1].self_attn.o_proj.register_forward_pre_hook(pre))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def head_means(rt, out: Path, cases: dict, records: list[dict], n: int = 32) -> dict:
    """Mean gated output of every full-attention head over teacher-forced answer positions."""
    sums, count = {}, 0
    handles = []
    hd = rt.layers[3].self_attn.head_dim
    state = {}
    for l in cr.FULL_LAYERS:
        def pre(module, args, l=l):
            x = args[0][0, state['lo']:].float()                      # (T, heads*hd)
            sums[l] = sums.get(l, 0) + x.view(x.shape[0], -1, hd).sum(0)
        handles.append(rt.layers[l - 1].self_attn.o_proj.register_forward_pre_hook(pre))
    try:
        for rec in records[:n]:
            image, truth, text = case_input(cases[rec['id']])
            prep = cr.prepare(rt, image, text)
            x, pos = cr.teacher_forced(rt, prep, rec['ids'][:-1])
            state['lo'] = len(prep.ids) - 1
            with torch.inference_mode():
                rt.model(inputs_embeds=x, position_ids=pos, logits_to_keep=1)
            count += x.shape[1] - state['lo']
    finally:
        for h in handles:
            h.remove()
    return {(l, h): sums[l][h] / count for l in cr.FULL_LAYERS for h in range(sums[l].shape[0])}


def stage_c4(rt, out: Path, args) -> None:
    root = out / 'ablation'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    res = json.loads((out / 'results.json').read_text())
    cand = [tuple(h) for h in res['c1']['disc_point']['candidates']['heads']]
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    base = [r for r in read_jsonl(out / 'baseline' / 'records.jsonl') if cases[r['id']]['set'] == 'grid'
            and r['format'] == 'point']
    means = head_means(rt, out, cases, base)
    rng = np.random.default_rng(args.seed)
    n_heads = rt.layers[3].self_attn.config.num_attention_heads
    layers = [l for l, _ in cand]
    controls = []
    for _ in range(args.n_random):
        pick = []
        for l in layers:
            options = [h for h in range(n_heads) if (l, h) not in cand and (l, h) not in pick]
            pick.append((l, int(rng.choice(options))))
        controls.append(pick)
    (root / 'head_sets.json').write_text(json.dumps({'candidates': cand, 'controls': controls}))
    yes, no = qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No')
    for rec in base:
        if rec['id'] in finished:
            continue
        t0 = time.time()
        case = cases[rec['id']]
        image, truth, text = case_input(case)
        prep = cr.prepare(rt, image, text)
        rprep = cr.prepare(rt, image, cr.RECOGNITION_QUESTION)
        x0 = cr.embeds(rt, prep)
        rx = cr.embeds(rt, rprep)
        rows = []
        for name, heads in [('none', []), ('candidates', cand)] + [(f'random{i}', h) for i, h in enumerate(controls)]:
            with HeadAblation(rt, heads, means, len(prep.ids) - 1):
                r = cr.greedy(rt, x0, prep.pos, args.c2_max_new_tokens)[0]
            with HeadAblation(rt, heads, means, len(rprep.ids) - 1), torch.inference_mode():
                lg = rt.model(inputs_embeds=rx, position_ids=rprep.pos, logits_to_keep=1).logits[0, -1].float()
            sc = cr.score_disc(truth, 'point', r['text'])
            rows.append({'set': name, 'text': r['text'], 'score': sc, 'recognition': float(lg[yes] - lg[no])})
        append_jsonl(path, [{'id': rec['id'], 'diameter': case['diameter'], 'baseline_correct': bool(rec['score'].get('correct')),
                             'conditions': rows, 'seconds': round(time.time() - t0, 1)}])
        print(f"c4 {rec['id']} {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- main

STAGES = {'c0': stage_c0, 'spotcheck': stage_spotcheck, 'c1': stage_c1, 'c2': stage_c2, 'fail': stage_fail, 'fail2': stage_fail2, 'c1grad': stage_c1grad, 'c4': stage_c4}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--t6-source', required=True, help='Experiment-1.3 decomposition run directory.')
    p.add_argument('--stages', default='c0')
    p.add_argument('--seed', type=int, default=20261014)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--n-spot', type=int, default=20)
    p.add_argument('--gap', type=int, default=8)
    p.add_argument('--c1-sets', default='grid,trans,t6')
    p.add_argument('--c2-formats', default='point')
    p.add_argument('--n-random', type=int, default=20)
    p.add_argument('--fail-order', default='20:rowcol,10:point,20:point,10:rowcol,6:rowcol,6:point')
    p.add_argument('--c2-max-new-tokens', type=int, default=48)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model, device='cuda:0')
    meta_path = args.output / 'experiment.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {'runs': []}
    meta['model'] = qr.describe(rt)
    meta['runs'].append({'started_utc': datetime.now(timezone.utc).isoformat(), 'stages': args.stages,
                         'args': {k: str(v) for k, v in vars(args).items() if k not in ('output', 't6_source')},
                         'code_sha256': code_digest(), 'thinking': 'off', 'decoder': 'manual greedy, unpadded batches'})
    meta_path.write_text(json.dumps(meta, indent=2))
    for stage in args.stages.split(','):
        STAGES[stage](rt, args.output, args)


if __name__ == '__main__':
    main()
