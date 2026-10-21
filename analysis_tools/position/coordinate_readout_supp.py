"""Supplementary stages of experiment 3.5 (coordinate readout path).

Stages (selected with --stages of `coordinate_readout_run.py`):
  c4patch   activation patching of token-mixer outputs at the answer positions: run A (disc at T) receives
            the outputs of selected heads / linear-attention layers from run B (disc actually drawn at T');
            single heads, the five candidate pointing heads, top-k heads by pointing score, all full-attention
            heads, Gated DeltaNet layers, and random head sets.
  c4redund  mean ablation of progressively larger head sets ranked by pointing score, random sets of the
            same sizes, and Gated DeltaNet layer outputs.
  rc0       RefCOCO baseline (point and bbox) on the experiment-3.4 test split (512-pixel copies).
  rc1       C1 attention readout and attribution on the RefCOCO answers.
  rc2       RefCOCO position-source interventions: global MRoPE / learned-PE shifts (design 6.6) plus
            region swaps of ViT RoPE ids, MRoPE ids, sequence slots and content for referents small enough
            to have a disjoint translated copy inside the image.
  c5        row / column insertion on the T6 grids: blank pixel bands and position-id insertion.

All stages are resumable through record ids, as in `coordinate_readout_run.py`.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import coordinate_readout_run as run

PI_HEADS = [(20, 10), (16, 11), (20, 14), (16, 15), (16, 6)]   # candidate pointing heads of the disc point task
TOPK = (10, 20, 40)
REDUND_K = (5, 10, 20, 40, 64)
N_RANDOM_SETS = 3
LIN_MID = (13, 14, 15, 17, 18, 19, 21, 22, 23)                 # Gated DeltaNet layers where rho peaks (19-23) and before
SMALL_SETS = ('none', 'pi5', 'top20', 'full_all', 'mixers_all')  # sets also run at swap distances 2 and 6


# ----------------------------------------------------------------------------- shared helpers

def linear_layers(rt) -> list[int]:
    return [i + 1 for i, b in enumerate(rt.layers) if b.block_type == 'linear_attention']


def pointing_rank(out: Path) -> list[tuple[int, int]]:
    """All full-attention heads ranked by mean log E (selection half, disc point task, experiment C1)."""
    res = json.loads((out / 'results.json').read_text())
    logE = np.asarray(res['c1']['disc_point']['candidates']['logE_map_selection'])
    n = logE.shape[1]
    return [(cr.FULL_LAYERS[i // n], i % n) for i in np.argsort(-logE.ravel(), kind='stable')]


def head_sets(rt, out: Path, seed: int) -> dict:
    """Named head sets; entries are (layer, head) with head = -1 for a whole Gated DeltaNet layer."""
    rank = pointing_rank(out)
    res = json.loads((out / 'results.json').read_text())
    cand7 = [tuple(h) for h in res['c1']['disc_point']['candidates']['heads']]
    lin = linear_layers(rt)
    sets = {'none': [], 'pi5': PI_HEADS, 'cand7': cand7, 'full_all': rank,
            'lin_mid': [(l, -1) for l in LIN_MID], 'lin_all': [(l, -1) for l in lin]}
    sets['mixers_all'] = sets['full_all'] + sets['lin_all']
    sets['lin_early'] = [(l, -1) for l in lin if l < LIN_MID[0]]
    sets['lin_late'] = [(l, -1) for l in lin if l > LIN_MID[-1]]
    for fl in cr.FULL_LAYERS:
        sets[f'full_L{fl}'] = [(fl, h) for h in range(16)]
    sets['full_L16L20'] = sets['full_L16'] + sets['full_L20']
    sets['top10_L16'] = [x for x in rank[:10] if x[0] == 16]
    sets['top10_L20'] = [x for x in rank[:10] if x[0] == 20]
    for l, h in PI_HEADS:
        sets[f'h{l}_{h}'] = [(l, h)]
    for k in TOPK + REDUND_K:
        sets[f'top{k}'] = rank[:k]
    rng = np.random.default_rng(seed)
    exclude = set(PI_HEADS) | set(cand7)
    for i in range(N_RANDOM_SETS):
        # Patching control: random heads from the same layers as the five candidate heads.
        pick = []
        for l, _ in PI_HEADS:
            options = [h for h in range(16) if (l, h) not in exclude and (l, h) not in pick]
            pick.append((l, int(rng.choice(options))))
        sets[f'rand5L_{i}'] = pick
        # Redundancy control: k random full-attention heads outside the top-40 by pointing score.
        pool = rank[40:]
        for k in REDUND_K:
            if k <= len(pool):
                sets[f'rand{k}_{i}'] = [pool[j] for j in sorted(rng.choice(len(pool), k, replace=False))]
    return {k: [tuple(map(int, h)) for h in v] for k, v in sets.items()}


def run_units(rt, units: list[dict], max_rows: int, start: int, means: dict | None = None,
              max_new_tokens: int = 48) -> list[list[dict]]:
    """Decode groups of rows that must share a batch (patching pairs) under row-wise mixer edits.

    unit: {'x': [(1, L, D)], 'pos': [(4, 1, L)], 'follow': {local follower: local leader},
           'ops': [(local row, layer, head, local source | 'mean')]}. Returns per-unit row results.
    """
    out, i = [], 0
    while i < len(units):
        chunk, rows = [], 0
        while i < len(units) and (not chunk or rows + len(units[i]['x']) <= max_rows):
            chunk.append(units[i])
            rows += len(units[i]['x'])
            i += 1
        xs, ps, follow, ops, offs, off = [], [], {}, [], [], 0
        for u in chunk:
            offs.append(off)
            xs += u['x']
            ps += u['pos']
            follow.update({f + off: l + off for f, l in u['follow'].items()})
            ops += [(r + off, l, h, s if s == 'mean' else s + off) for r, l, h, s in u['ops']]
            off += len(u['x'])
        x, p = torch.cat(xs, 0), torch.cat(ps, 1)
        with cr.MixerEdit(rt, ops, start, means):
            res = cr.greedy_follow(rt, x, p, follow, max_new_tokens)
        for u, o in zip(chunk, offs):
            out.append(res[o:o + len(u['x'])])
    return out


@torch.inference_mode()
def mixer_means(rt, cases: dict, records: list[dict], n: int = 32) -> dict:
    """Mean token-mixer output over teacher-forced answer positions (last prompt token onward).

    Keys (layer, head) for full-attention heads and (layer, -1) for whole Gated DeltaNet outputs.
    The first n disc point baselines are used, as in the original C4 ablation (`run.head_means`).
    """
    sums, count, state, handles = {}, 0, {}, []
    for l in range(1, len(rt.layers) + 1):
        module, hd = cr.mixer_projection(rt, l)

        def pre(module, args, l=l):
            sums[l] = sums.get(l, 0) + args[0][0, state['lo']:].float().sum(0)
        handles.append(module.register_forward_pre_hook(pre))
    try:
        for rec in records[:n]:
            image, truth, text = run.case_input(cases[rec['id']])
            prep = cr.prepare(rt, image, text)
            x, pos = cr.teacher_forced(rt, prep, rec['ids'][:-1])
            state['lo'] = len(prep.ids) - 1
            rt.model(inputs_embeds=x, position_ids=pos, logits_to_keep=1)
            count += x.shape[1] - state['lo']
    finally:
        for h in handles:
            h.remove()
    means = {}
    for l in range(1, len(rt.layers) + 1):
        _, hd = cr.mixer_projection(rt, l)
        v = sums[l] / count
        if rt.layers[l - 1].block_type == 'full_attention':
            means.update({(l, h): v[h * hd:(h + 1) * hd] for h in range(v.shape[0] // hd)})
        else:
            means[(l, -1)] = v
    return means


def load_means(rt, out: Path, cases: dict, base: list[dict]) -> dict:
    path = out / 'ablation' / 'mixer_means.pt'
    if path.exists():
        saved = torch.load(path)
        return {tuple(map(int, k.split(','))): v.to(rt.device) for k, v in saved.items()}
    means = mixer_means(rt, cases, base)
    torch.save({f'{l},{h}': v.cpu() for (l, h), v in means.items()}, path)
    return means


def disc_point_base(out: Path, cases: dict) -> list[dict]:
    return [r for r in run.read_jsonl(out / 'baseline' / 'records.jsonl')
            if cases[r['id']]['set'] == 'grid' and r['format'] == 'point']


def center_of(text: str, fmt: str) -> list[float] | None:
    coords = cr.parse_coords(text, fmt)
    return None if coords is None else cr.answer_center(coords).tolist()


# ----------------------------------------------------------------------------- C4 patching

def stage_c4patch(rt, out: Path, args, layer_sweep: int | None = None) -> None:
    """Output patching; with layer_sweep = l, every single head of layer l and every leave-one-out set of
    layer l are patched instead (pair distance 4 only), written to patching/records_L{l}.jsonl."""
    root = out / 'patching'
    root.mkdir(parents=True, exist_ok=True)
    path = root / ('records.jsonl' if layer_sweep is None else f'records_L{layer_sweep}.jsonl')
    finished = run.done_ids(path)
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    sets = head_sets(rt, out, args.seed)
    if layer_sweep is None:
        (root / 'head_sets.json').write_text(json.dumps({k: v for k, v in sets.items()}))
        full_names = (['none'] + [f'h{l}_{h}' for l, h in PI_HEADS] + ['pi5', 'cand7'] + [f'top{k}' for k in TOPK]
                      + ['full_all'] + [f'full_L{l}' for l in cr.FULL_LAYERS] + [f'rand5L_{i}' for i in range(N_RANDOM_SETS)]
                      + ['lin_early', 'lin_mid', 'lin_late', 'lin_all', 'mixers_all'])
    else:
        heads = [(layer_sweep, h) for h in range(16)]
        sets = {'none': [], f'full_L{layer_sweep}': heads}
        for h in range(16):
            sets[f'only_{h}'] = [(layer_sweep, h)]
            sets[f'drop_{h}'] = [x for x in heads if x[1] != h]
        full_names = list(sets)
    c2 = {r['id']: r for r in run.read_jsonl(out / 'interventions' / 'records.jsonl')}
    for rec in disc_point_base(out, cases):
        if rec['id'] in finished or rec['id'] not in c2:
            continue
        t0 = time.time()
        case = cases[rec['id']]
        image, truth, text = run.case_input(case)
        prep_a = cr.prepare(rt, image, text)
        xa = cr.embeds(rt, prep_a)
        units, meta = [], []
        for reg in c2[rec['id']]['regions']:
            if not reg['valid']:
                continue
            moved = [truth['center'][0] + 32 * reg['dc'], truth['center'][1] + 32 * reg['dr']]
            img_b, truth_b = cr.disc_image(tuple(moved), case['diameter'], case['canvas'])
            prep_b = cr.prepare(rt, img_b, text)
            if len(prep_b.ids) != len(prep_a.ids):
                raise ValueError(f"Run B length differs for {rec['id']}")
            xb = cr.embeds(rt, prep_b)
            tag = f"{reg['axis']}{reg['k']}"
            units.append({'x': [xb], 'pos': [prep_b.pos], 'follow': {}, 'ops': []})
            meta.append((tag, 'B_free', reg))
            if layer_sweep is not None and reg['k'] != 4:
                continue
            names = full_names if reg['k'] == 4 else SMALL_SETS
            for name in names:
                # Row 0: source run B, fed the tokens chosen by row 1; row 1: run A with patched outputs.
                ops = [(1, l, h, 0) for l, h in sets[name]]
                units.append({'x': [xb, xa], 'pos': [prep_b.pos, prep_a.pos], 'follow': {0: 1}, 'ops': ops})
                meta.append((tag, name, reg))
        res = run_units(rt, units, args.max_rows, len(prep_a.ids) - 1, max_new_tokens=args.c2_max_new_tokens)
        rows = []
        for (tag, name, reg), r in zip(meta, res):
            r = r[-1]
            rows.append({'pair': tag, 'set': name, 'd': reg['d'], 'k': reg['k'], 'axis': reg['axis'], 'text': r['text'],
                         'center': center_of(r['text'], 'point'), 'score': cr.score_disc(truth, 'point', r['text'])})
        run.append_jsonl(path, [{'id': rec['id'], 'diameter': case['diameter'], 'baseline_correct': bool(rec['score'].get('correct')),
                                 'truth_center': (1000 * np.asarray(truth['center']) / case['canvas']).tolist(),
                                 'conditions': rows, 'max_rows': args.max_rows, 'seconds': round(time.time() - t0, 1)}])
        print(f"c4patch {rec['id']} {len(rows)} conditions {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- C4 redundancy (mean ablation)

def stage_c4redund(rt, out: Path, args) -> None:
    root = out / 'ablation'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records_redundancy.jsonl'
    finished = run.done_ids(path)
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    base = disc_point_base(out, cases)
    means = load_means(rt, out, cases, base)
    sets = head_sets(rt, out, args.seed)
    names = (['none', 'pi5'] + [f'top{k}' for k in REDUND_K] + ['full_all']
             + [f'rand{k}_{i}' for k in REDUND_K for i in range(N_RANDOM_SETS) if f'rand{k}_{i}' in sets]
             + [f'full_L{l}' for l in cr.FULL_LAYERS] + ['lin_early', 'lin_mid', 'lin_late', 'lin_all', 'mixers_all'])
    (root / 'head_sets_redundancy.json').write_text(json.dumps({k: sets[k] for k in names}))
    yes, no = qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No')
    for rec in base:
        if rec['id'] in finished:
            continue
        t0 = time.time()
        case = cases[rec['id']]
        image, truth, text = run.case_input(case)
        prep = cr.prepare(rt, image, text)
        x0 = cr.embeds(rt, prep)
        units = [{'x': [x0], 'pos': [prep.pos], 'follow': {}, 'ops': [(0, l, h, 'mean') for l, h in sets[n]]}
                 for n in names]
        res = run_units(rt, units, args.max_rows, len(prep.ids) - 1, means, args.c2_max_new_tokens)
        # Recognition check: Yes - No logit with the same edits at the answer position.
        rprep = cr.prepare(rt, image, cr.RECOGNITION_QUESTION)
        rx = cr.embeds(rt, rprep)
        recog = []
        for i in range(0, len(names), args.max_rows):
            chunk = names[i:i + args.max_rows]
            ops = [(j, l, h, 'mean') for j, n in enumerate(chunk) for l, h in sets[n]]
            with cr.MixerEdit(rt, ops, len(rprep.ids) - 1, means), torch.inference_mode():
                lg = rt.model(inputs_embeds=rx.expand(len(chunk), -1, -1), position_ids=rprep.pos.expand(-1, len(chunk), -1),
                              logits_to_keep=1).logits[:, -1].float()
            recog += (lg[:, yes] - lg[:, no]).tolist()
        rows = [{'set': n, 'n_heads': len(sets[n]), 'text': r[0]['text'], 'score': cr.score_disc(truth, 'point', r[0]['text']),
                 'recognition': g} for n, r, g in zip(names, res, recog)]
        run.append_jsonl(path, [{'id': rec['id'], 'diameter': case['diameter'], 'baseline_correct': bool(rec['score'].get('correct')),
                                 'conditions': rows, 'max_rows': args.max_rows, 'seconds': round(time.time() - t0, 1)}])
        print(f"c4redund {rec['id']} {time.time() - t0:.1f}s", flush=True)


def stage_c4post(rt, out: Path, args) -> None:
    """Mean ablation applied from the first token after the image (question, template and answer) instead
    of from the last prompt token, to test whether coordinate information reaches the answer through text
    positions that read the image before generation starts."""
    root = out / 'ablation'
    path = root / 'records_post_image.jsonl'
    finished = run.done_ids(path)
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    base = disc_point_base(out, cases)
    means = load_means(rt, out, cases, base)
    sets = head_sets(rt, out, args.seed)
    names = ['none', 'top10', 'top10_L16', 'top10_L20', 'full_L16', 'full_L20', 'full_L16L20', 'full_all']
    for rec in base:
        if rec['id'] in finished:
            continue
        t0 = time.time()
        case = cases[rec['id']]
        image, truth, text = run.case_input(case)
        prep = cr.prepare(rt, image, text)
        x0 = cr.embeds(rt, prep)
        rows = []
        for where, start in (('answer', len(prep.ids) - 1), ('post_image', prep.span['end'])):
            units = [{'x': [x0], 'pos': [prep.pos], 'follow': {}, 'ops': [(0, l, h, 'mean') for l, h in sets[n]]}
                     for n in names]
            res = run_units(rt, units, args.max_rows, start, means, args.c2_max_new_tokens)
            rows += [{'set': n, 'from': where, 'text': r[0]['text'], 'score': cr.score_disc(truth, 'point', r[0]['text'])}
                     for n, r in zip(names, res)]
        run.append_jsonl(path, [{'id': rec['id'], 'diameter': case['diameter'], 'truth_center': (1000 * np.asarray(truth['center']) / case['canvas']).tolist(),
                                 'conditions': rows, 'seconds': round(time.time() - t0, 1)}])
        print(f"c4post {rec['id']} {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- RefCOCO

REFCOCO_QUERY = 'Locate the object described as: {expr}.'


def point_conventions(point: list[float], box: list[float], size: tuple[int, int]) -> dict:
    """Strict (0-1000) and diagnostic (absolute pixel) hit of a point answer against a pixel box.

    Thin wrapper over the shared helper of plan.md section 6 item 9
    (`analysis_tools.behavior.coordinate_convention`): h_norm, h_pix, h_any, and the chance rates of a
    uniform random point on [0, 1000]^2 under each convention and their union.
    """
    from analysis_tools.behavior.coordinate_convention import chance_rates, convention_hits
    hits = convention_hits(tuple(point), size, box)
    chance = chance_rates(size, box)
    return {'h_norm': hits['h_norm'], 'h_pix': hits['h_pix'], 'h_any': hits['h_any'],
            'chance_norm': chance['p_norm'], 'chance_pix': chance['p_pix'], 'chance_any': chance['p_any']}


def score_refcoco(truth: dict, fmt: str, text: str) -> dict:
    coords = cr.parse_coords(text, fmt)
    if coords is None:
        return {'valid': False, 'correct': False, 'hit': False}
    W, H = truth['canvas']
    box = truth['box']
    scale = np.asarray([W, H], float) / 1000
    center = cr.answer_center(coords)
    true_center = 1000 * np.asarray(truth['center']) / np.asarray([W, H], float)
    out = {'valid': True, 'coords': coords, 'center': center.tolist(), 'error': float(np.linalg.norm(center - true_center))}
    out.update(point_conventions(center.tolist(), box, (W, H)))
    out['hit'] = out['h_norm']
    if fmt == 'bbox':
        out['iou'] = cr.box_iou((np.asarray(coords) * np.tile(scale, 2)).tolist(), box)
        out['iou_pix'] = cr.box_iou(coords, box)
        out['correct'] = out['iou'] >= 0.5
    else:
        out['correct'] = out['hit']
    return out


def refcoco_cases(args) -> list[dict]:
    """Experiment-3.4 test images (512-pixel copies) with the first referring expression of their RefCOCO row."""
    import pyarrow.parquet as pq
    src = Path(args.refcoco_source)
    meta = json.loads((src / 'cases.json').read_text())['natural']
    test = [c for c in meta if c['split'] == 'test']
    tables = {}
    out = []
    for c in test:
        if c['shard'] not in tables:
            tables[c['shard']] = pq.read_table(Path(args.refcoco) / 'data' / c['shard'], columns=['answer', 'bbox', 'file_name', 'question_id'])
        row = tables[c['shard']].slice(c['row'], 1).to_pylist()[0]
        if row['file_name'] != c['file_name']:
            raise ValueError(f"RefCOCO row mismatch for case {c['id']}")
        sw, sh = c['source_size']
        iw, ih = c['input_size']
        x, y, w, h = row['bbox']
        box = [x * iw / sw, y * ih / sh, (x + w) * iw / sw, (y + h) * ih / sh]
        image_path = src / 'images' / 'natural512' / f"{c['id']:04d}.png"
        for fmt in ('point', 'bbox'):
            out.append({'id': f"rc{c['id']:04d}_{fmt}", 'set': 'refcoco', 'format': fmt, 'image_key': f"rc{c['id']:04d}",
                        'image_path': str(image_path), 'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
                        'expression': row['answer'][0], 'question_id': row['question_id'], 'source_case': c['id'],
                        'prompt': REFCOCO_QUERY.format(expr=row['answer'][0]) + cr.SUFFIX[fmt],
                        'truth': {'canvas': [iw, ih], 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]}})
    return out


def refcoco_input(case: dict):
    p = Path(case['image_path'])
    if hashlib.sha256(p.read_bytes()).hexdigest() != case['image_sha256']:
        raise ValueError(f'Image hash mismatch for {p}')
    return Image.open(p).convert('RGB'), case['truth'], case['prompt']


def stage_rc0(rt, out: Path, args) -> None:
    root = out / 'refcoco'
    root.mkdir(parents=True, exist_ok=True)
    cpath = root / 'cases.json'
    if not cpath.exists():
        cpath.write_text(json.dumps(refcoco_cases(args), indent=1))
    cases = json.loads(cpath.read_text())
    path = root / 'baseline.jsonl'
    finished = run.done_ids(path)
    for c in cases:
        if c['id'] in finished:
            continue
        t0 = time.time()
        image, truth, text = refcoco_input(c)
        prep = cr.prepare(rt, image, text)
        r = cr.greedy(rt, cr.embeds(rt, prep), prep.pos, args.max_new_tokens)[0]
        run.append_jsonl(path, [{'id': c['id'], 'set': 'refcoco', 'format': c['format'], 'text': r['text'], 'ids': r['ids'],
                                 'finish_reason': r['finish_reason'], 'truth': truth, 'grid': list(prep.grid),
                                 'score': score_refcoco(truth, c['format'], r['text']), 'batch_size': 1}])
        print(f"rc0 {c['id']} {time.time() - t0:.1f}s", flush=True)
    # Spot check against the reference generate() path.
    spath = root / 'spotcheck.jsonl'
    sdone = run.done_ids(spath)
    recs = {r['id']: r for r in run.read_jsonl(path)}
    rng = np.random.default_rng(args.seed)
    for i in sorted(rng.choice(len(cases), size=min(args.n_spot, len(cases)), replace=False)):
        c = cases[int(i)]
        if c['id'] in sdone:
            continue
        image, truth, text = refcoco_input(c)
        ref = qr.generate(rt, qr.build(rt, [image], text), args.max_new_tokens)
        run.append_jsonl(spath, [{'id': c['id'], 'reference': ref['text'], 'manual': recs[c['id']]['text'],
                                  'text_equal': ref['text'].strip() == recs[c['id']]['text'].strip(),
                                  'answer_equal': cr.parse_coords(ref['text'], c['format']) == cr.parse_coords(recs[c['id']]['text'], c['format'])}])


def stage_rc1(rt, out: Path, args) -> None:
    root = out / 'refcoco' / 'attention'
    (root / 'maps').mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = run.done_ids(path)
    cases = {c['id']: c for c in json.loads((out / 'refcoco' / 'cases.json').read_text())}
    digit_ids = [rt.tokenizer.convert_tokens_to_ids(d) for d in run.DIGITS]
    for rec in run.read_jsonl(out / 'refcoco' / 'baseline.jsonl'):
        if rec['id'] in finished:
            continue
        case = cases[rec['id']]
        image, truth, text = refcoco_input(case)
        try:
            row = run.attention_record(rt, root, rec, case, image, truth, text, digit_ids)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {'id': rec['id'], 'status': 'oom'}
        run.append_jsonl(path, [row])
        print(f"rc1 {rec['id']} {row['status']}", flush=True)


def refcoco_swaps(target: list[int], grid) -> list[dict]:
    """T' = T translated by its own extent (plus one token) along x or y toward the larger free side.

    Only referents whose translated token box lies inside the image are kept; T and T' are disjoint.
    """
    H, W = grid
    rows = sorted({t // W for t in target})
    cols = sorted({t % W for t in target})
    out = []
    for axis in ('h', 'v'):
        lo, hi, n = (cols[0], cols[-1], W) if axis == 'h' else (rows[0], rows[-1], H)
        k = hi - lo + 2
        toward = 1 if (n - 1 - hi) >= lo else -1
        dr, dc = (0, toward * k) if axis == 'h' else (toward * k, 0)
        moved = [(t // W + dr) * W + (t % W + dc) for t in target]
        ok = all(0 <= r + dr < H for r in rows) and all(0 <= c + dc < W for c in cols)
        out.append({'axis': axis, 'k': k, 'dr': dr, 'dc': dc, 'valid': ok, 'region': moved if ok else None,
                    'd': [1000 * dc / W, 1000 * dr / H]})
    return out


def stage_rc2(rt, out: Path, args) -> None:
    root = out / 'refcoco'
    path = root / 'interventions.jsonl'
    finished = run.done_ids(path)
    cases = {c['id']: c for c in json.loads((root / 'cases.json').read_text())}
    digit_ids = [rt.tokenizer.convert_tokens_to_ids(d) for d in run.DIGITS]
    for rec in run.read_jsonl(root / 'baseline.jsonl'):
        if rec['id'] in finished or rec['format'] != 'point':
            continue
        t0 = time.time()
        case = cases[rec['id']]
        image, truth, text = refcoco_input(case)
        prep = cr.prepare(rt, image, text)
        H, W = prep.grid
        x0 = cr.embeds(rt, prep)
        gp = cr.gap_pos(prep, args.gap)
        conds = [('standard', x0, prep.pos, None, {}), ('gap', x0, gp, None, {})]
        for dl in run.SHIFTS:
            conds.append((f's2shift_h{dl:+d}', x0, cr.shift_image_pos(prep, gp, dl, 0), [0, 1000 * dl / H], {'ref': 'gap', 'axis': 'v', 'delta': dl}))
            conds.append((f's2shift_w{dl:+d}', x0, cr.shift_image_pos(prep, gp, 0, dl), [1000 * dl / W, 0], {'ref': 'gap', 'axis': 'h', 'delta': dl}))
        pe0 = cr.shifted_pe(prep, run.PE_MARGIN, 0, 0)
        conds.append(('s1center', cr.embeds(rt, prep, cr.vision_embeds(rt, prep, interp=pe0)), prep.pos, None, {}))
        for dl in run.SHIFTS:
            for axis, (dr, dc) in (('v', (2 * dl, 0)), ('h', (0, 2 * dl))):
                if abs(2 * dl) > run.PE_MARGIN:
                    continue
                v = cr.vision_embeds(rt, prep, interp=cr.shifted_pe(prep, run.PE_MARGIN, dr, dc))
                conds.append((f's1shift_{axis}{dl:+d}', cr.embeds(rt, prep, v), prep.pos,
                              [1000 * dc / (2 * W), 1000 * dr / (2 * H)], {'ref': 's1center', 'axis': axis, 'delta': dl}))
        conds.append(('s1rope_shift', cr.embeds(rt, prep, cr.vision_embeds(rt, prep, vit_pos=prep.vit_pos + 7)), prep.pos, None, {'ref': 'standard'}))
        target = run.target_tokens(case, truth, prep.grid)
        regions = refcoco_swaps(target, prep.grid)
        for reg in regions:
            if not reg['valid']:
                continue
            tag, b = reg['axis'], reg['region']
            conds.append((f's2swap_{tag}', x0, cr.swap_image_pos(prep, prep.pos, target, b), reg['d'], {'ref': 'standard'}))
            for name, pe, rope in (('rope', False, True), ('both', True, True)):
                vp, ip = cr.swap_vit(prep, target, b, pe, rope)
                conds.append((f's1swap_{name}_{tag}', cr.embeds(rt, prep, cr.vision_embeds(rt, prep, vit_pos=vp, interp=ip)),
                              prep.pos, reg['d'], {'ref': 'standard'}))
            order = cr.swap_index(H * W, target, b)
            conds.append((f's4perm_{tag}', cr.embeds(rt, prep, order=order), cr.permute_pos(prep, prep.pos, order), reg['d'], {'ref': 'standard'}))
            conds.append((f'c3content_{tag}', cr.embeds(rt, prep, order=order), prep.pos, reg['d'], {'ref': 'standard'}))
        res = cr.run_conditions(rt, [c[1] for c in conds], [c[2] for c in conds], args.c2_max_new_tokens,
                                batch_size=args.batch_size, track_ids=digit_ids)
        rows = []
        for (name, _, _, d, meta), r in zip(conds, res):
            rows.append({'condition': name, 'd': d, **meta, 'text': r['text'], 'finish_reason': r['finish_reason'],
                         'center': center_of(r['text'], 'point'), 'score': score_refcoco(truth, 'point', r['text'])})
        run.append_jsonl(path, [{'id': rec['id'], 'grid': [H, W], 'baseline_correct': bool(rec['score'].get('correct')),
                                 'target_tokens': target, 'regions': regions, 'conditions': rows,
                                 'batch_size': args.batch_size, 'seconds': round(time.time() - t0, 1)}])
        print(f"rc2 {rec['id']} {len(conds)} conditions {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- C5 row / column insertion

def stage_c5(rt, out: Path, args) -> None:
    root = out / 'insertion'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = run.done_ids(path)
    cases = {c['id']: c for c in json.loads((out / 'cases.json').read_text())}
    base = [r for r in run.read_jsonl(out / 'baseline' / 'records.jsonl')
            if cases[r['id']]['set'] == 't6' and r['format'] == 'rowcol']
    for rec in base:
        if rec['id'] in finished:
            continue
        t0 = time.time()
        case = cases[rec['id']]
        image, truth, text = run.case_input(case)
        cell = truth['cell']
        ox, oy = truth['grid_origin']
        tr, tc = truth['target_rc']
        Wc, Hc = truth['canvas']
        prep = cr.prepare(rt, image, text)
        unit = cell // cr.TOKEN_PIXELS
        # Pixel bands: above the target row / at the top edge / at the bottom edge (control), and the
        # column analogues. Row and column ordinals of the target are unchanged by every insertion.
        bands = [('pix_v_target', 'v', oy + tr * cell), ('pix_v_top', 'v', 0), ('pix_v_bottom', 'v', Hc),
                 ('pix_h_target', 'h', ox + tc * cell), ('pix_h_left', 'h', 0), ('pix_h_right', 'h', Wc)]
        xs, ps, names = [cr.embeds(rt, prep)], [prep.pos], ['standard']
        for name, axis, at in (('id_v_target', 'v', (oy + tr * cell) // cr.TOKEN_PIXELS),
                               ('id_h_target', 'h', (ox + tc * cell) // cr.TOKEN_PIXELS)):
            xs.append(xs[0])
            ps.append(cr.insert_ids(prep, axis, at, unit))
            names.append(name)
        res = cr.run_conditions(rt, xs, ps, args.max_new_tokens, batch_size=args.batch_size)
        bx, bp, bn = [], [], []
        for name, axis, at in bands:
            p2 = cr.prepare(rt, cr.insert_band(image, axis, at, cell), text)
            bx.append(cr.embeds(rt, p2))
            bp.append(p2.pos)
            bn.append(name)
        if len({x.shape[1] for x in bx}) != 1:
            raise ValueError(f"Unequal band lengths for {rec['id']}")
        res += cr.run_conditions(rt, bx, bp, args.max_new_tokens, batch_size=args.batch_size)
        names += bn
        rows = [{'condition': n, 'text': r['text'], 'finish_reason': r['finish_reason'], 'score': run.score_t6(case, r['text'])
                 if n.startswith(('standard', 'id_')) else rowcol_score(truth, r['text'])} for n, r in zip(names, res)]
        run.append_jsonl(path, [{'id': rec['id'], 'grid': case['grid'], 'truth_rc': truth['target_rc'],
                                 'baseline_score': rec['score'], 'conditions': rows, 'batch_size': args.batch_size,
                                 'seconds': round(time.time() - t0, 1)}])
        print(f"c5 {rec['id']} {time.time() - t0:.1f}s", flush=True)


def rowcol_score(truth: dict, text: str) -> dict:
    """Row-column score against the unchanged ordinal truth (pixel geometry is not needed)."""
    rc = cr.parse_rowcol(text)
    if rc is None:
        return {'valid': False, 'correct': False}
    tr, tc = truth['target_rc']
    return {'valid': True, 'correct': (rc[0] - 1, rc[1] - 1) == (tr, tc), 'pred_rc': [rc[0] - 1, rc[1] - 1],
            'd_row': rc[0] - 1 - tr, 'd_col': rc[1] - 1 - tc}


def stage_c4l20(rt, out: Path, args) -> None:
    stage_c4patch(rt, out, args, layer_sweep=20)


STAGES = {'c4patch': stage_c4patch, 'c4l20': stage_c4l20, 'c4post': stage_c4post, 'c4redund': stage_c4redund, 'rc0': stage_rc0, 'rc1': stage_rc1,
          'rc2': stage_rc2, 'c5': stage_c5}
