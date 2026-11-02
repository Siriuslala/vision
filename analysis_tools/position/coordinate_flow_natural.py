"""Experiment 3.11 round 2: coordinate readout on natural images (case building and GPU stages).

Results: docs/analysis/position/results_2026-10-21_coordinate_flow_natural.md. Builds on round 1
(coordinate_flow.py), whose first-digit recovered fraction r and pairing conventions are reused:

    LD = z[d_B] - z[d_A],   r_e = (LD_e - LD_A) / (LD_B - LD_A),

with z the next-token logits at the query that predicts the first digit of a coordinate value, d_A the
first digit of run A's greedy answer and d_B the argmax digit of run B at the same answer position. Both
runs are teacher-forced on run A's answer, so they share every answer token.

Two pair types separate selecting the referent ("what") from reading its position ("where"):
  what   run B keeps the image and replaces the referring expression by the expression of another object
         in the same image (RefCOCO: another annotated referent of the same COCO image; synthetic: another
         shape). Every image token's state is identical in both runs (the image precedes the text), so any
         difference at the answer comes from the text.
  where  run B keeps the expression and moves the referent: the merged visual embeddings of the target
         tokens are exchanged with an equally sized region (the round-1 / 3.5 content swap).

Groups: 'rc' (the 300 RefCOCO point questions of 2.5e), 'ms_small' / 'ms_large' (synthetic images with four
coloured shapes, 64 or 160 pixels wide, same prompt template as RefCOCO), and the round-1 single-target
cases 'disc' / 't6g10' (ablation only).

Stages (--stages; GPU stages run in order inside one process and resume by record id):
  cases     CPU: case list, RefCOCO masks and referent boxes, synthetic images.
  gen       greedy answers to the run-A prompt and to the what-pair prompt (equal-length batches).
  spotcheck answer-level agreement of the batched decoder with qwen_runtime.generate.
  patch     per pair: answer-position output patching of each full-attention head, layer sets, leave-one-out
            sets and Gated DeltaNet mixer groups (run B captured in its own forward, so prompts may differ in
            length); attention rows of all 128 heads at both first-digit queries in run A and run B.
  means     mean head outputs at answer positions per group (for mean ablation).
  ablate    greedy answers with mean-ablated head sets at answer positions (or from the prompt text on).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position.coordinate_readout_supp import refcoco_swaps
from analysis_tools.representation import sees_right as sr

FULL_LAYERS = cr.FULL_LAYERS
N_LAYERS = 32
DN_LAYERS = tuple(l for l in range(1, N_LAYERS + 1) if l not in FULL_LAYERS)


def point_prompt(expr: str) -> str:
    """The 2.5e / 3.5 RefCOCO point prompt (asks for the centre of the bounding box)."""
    return sr.REFCOCO_QUERY.format(expr=expr) + cf.FORMAT_SUFFIX['point']


COLORS = {'red': (220, 30, 30), 'blue': (30, 70, 220), 'green': (20, 150, 50), 'orange': (245, 140, 0)}
SHAPES = ('circle', 'square', 'triangle', 'diamond')
MS_SIZES = {'ms_small': 64, 'ms_large': 160}
read_jsonl, append_jsonl, done_ids = sr.read_jsonl, sr.append_jsonl, sr.done_ids


# ----------------------------------------------------------------------------- synthetic multi-shape images

def draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple[float, float, float, float], color) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if shape == 'circle':
        draw.ellipse(box, fill=color)
    elif shape == 'square':
        draw.rectangle(box, fill=color)
    elif shape == 'triangle':
        draw.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=color)
    elif shape == 'diamond':
        draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=color)
    else:
        raise ValueError(f'Unknown shape {shape}')


def multishape_image(rng: np.random.Generator, size: int, n_obj: int = 4, canvas: int = 512,
                     gap: int = 16) -> tuple[Image.Image, list[dict]]:
    """White canvas with n_obj shapes; every object has a unique colour and a unique shape.

    Objects are axis-aligned squares of side `size` placed by rejection sampling so that any two are at
    least `gap` pixels apart. Returns the image and [{'name', 'color', 'shape', 'box', 'center'}].
    """
    colors = rng.permutation(list(COLORS))[:n_obj]
    shapes = rng.permutation(list(SHAPES))[:n_obj]
    boxes: list[list[float]] = []
    for _ in range(10000):
        if len(boxes) == n_obj:
            break
        x0, y0 = rng.uniform(gap, canvas - size - gap, 2)
        b = [float(x0), float(y0), float(x0 + size), float(y0 + size)]
        if all(b[0] >= o[2] + gap or o[0] >= b[2] + gap or b[1] >= o[3] + gap or o[1] >= b[3] + gap for o in boxes):
            boxes.append(b)
    if len(boxes) != n_obj:
        raise RuntimeError('Could not place all objects; lower the size or the count.')
    img = Image.new('RGB', (canvas, canvas), (255, 255, 255))
    d = ImageDraw.Draw(img)
    objs = []
    for c, s, b in zip(colors, shapes, boxes):
        draw_shape(d, s, b, COLORS[c])
        objs.append({'name': f'{c} {s}', 'color': str(c), 'shape': str(s), 'box': b,
                     'center': [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]})
    return img, objs


# ----------------------------------------------------------------------------- RefCOCO metadata

def coco_image_id(file_name: str) -> str:
    """'COCO_train2014_000000384745_7.jpg' -> '000000384745' (the HF rows are one referent each)."""
    return file_name.rsplit('_', 1)[0].rsplit('_', 1)[1]


def polygon_to_input(seg: list[float], src_size, in_size) -> list[list[float]]:
    """Flat COCO polygon [x0, y0, x1, y1, ...] in source pixels -> [[x, y], ...] in input pixels."""
    sx, sy = in_size[0] / src_size[0], in_size[1] / src_size[1]
    pts = np.asarray(seg, float).reshape(-1, 2) * [sx, sy]
    return pts.tolist()


def box_xywh_to_input(b, src_size, in_size) -> list[float]:
    sx, sy = in_size[0] / src_size[0], in_size[1] / src_size[1]
    return [b[0] * sx, b[1] * sy, (b[0] + b[2]) * sx, (b[1] + b[3]) * sy]


def box_iou(a, b) -> float:
    return cr.box_iou(a, b)


def pick_distractor(target_box, others: list[dict], canvas) -> dict | None:
    """The referent of the same image farthest (in 0-1000 units) from the target, ignoring near-duplicates.

    A Chebyshev distance of at least 100 units guarantees a different first digit on at least one axis."""
    W, H = canvas
    tc = np.array([(target_box[0] + target_box[2]) / 2 * 1000 / W, (target_box[1] + target_box[3]) / 2 * 1000 / H])
    best, best_d = None, 0.0
    for o in others:
        if box_iou(o['box'], target_box) > 0.3:
            continue
        b = o['box']
        oc = np.array([(b[0] + b[2]) / 2 * 1000 / W, (b[1] + b[3]) / 2 * 1000 / H])
        d = float(np.abs(oc - tc).max())
        if d > best_d:
            best, best_d = o, d
    return best if best_d >= 100 else None


def refcoco_index(refcoco_dir: Path) -> dict[str, list[dict]]:
    """COCO image id -> referents [{'shard', 'row', 'bbox' (source xywh), 'expressions'}] over all splits."""
    import pyarrow.parquet as pq
    out: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(refcoco_dir / 'data' / '*.parquet'))):
        t = pq.read_table(f, columns=['answer', 'bbox', 'file_name']).to_pydict()
        for i, (ans, bb, fn) in enumerate(zip(t['answer'], t['bbox'], t['file_name'])):
            out.setdefault(coco_image_id(fn), []).append({'shard': Path(f).name, 'row': i, 'bbox': bb,
                                                          'expressions': list(ans)})
    return out


def segmentation(refcoco_dir: Path, shard: str, row: int) -> list[float]:
    import pyarrow.parquet as pq
    t = pq.read_table(refcoco_dir / 'data' / shard, columns=['segmentation'])
    return t.column('segmentation')[row].as_py()


# ----------------------------------------------------------------------------- cases

def stage_cases(out: Path, args) -> None:
    src = {c['id']: c for c in json.loads((Path(args.source) / 'cases.json').read_text())}
    natural = {c['id']: c for c in json.loads(Path(args.probe_cases).read_text())['natural']}
    index = refcoco_index(Path(args.refcoco_dir))
    seg_cache: dict[str, object] = {}
    cases = []
    for cid, c in src.items():
        if c['cond'] != 'rc_point':
            continue
        nat = natural[int(c['image_key'][2:])]
        src_size, in_size = nat['source_size'], nat['input_size']
        if list(in_size) != list(c['truth']['canvas']):
            raise ValueError(f'Canvas mismatch for {cid}')
        img_id = coco_image_id(nat['file_name'])
        if nat['shard'] not in seg_cache:
            import pyarrow.parquet as pq
            seg_cache[nat['shard']] = pq.read_table(Path(args.refcoco_dir) / 'data' / nat['shard'],
                                                    columns=['segmentation', 'bbox']).to_pydict()
        tab = seg_cache[nat['shard']]
        tbox = box_xywh_to_input(tab['bbox'][nat['row']], src_size, in_size)
        if max(abs(a - b) for a, b in zip(tbox, c['truth']['box'])) > 1.0:
            raise ValueError(f'RefCOCO source row does not match the case box for {cid}')
        others, seen = [], [tbox]
        for ref in index[img_id]:
            b = box_xywh_to_input(ref['bbox'], src_size, in_size)
            if any(box_iou(b, s) > 0.9 for s in seen):
                continue
            seen.append(b)
            others.append({'box': b, 'expression': ref['expressions'][0], 'shard': ref['shard'], 'row': ref['row']})
        dis = pick_distractor(tbox, others, in_size)
        case = {'id': f"{c['image_key']}", 'group': 'rc', 'set': 'refcoco', 'cond': 'rc', 'format': 'point',
                'image_key': c['image_key'], 'image_path': c['image_path'], 'image_sha256': c['image_sha256'],
                'expression': c['expression'], 'prompt': c['prompt'],
                'truth': c['truth'], 'mask': polygon_to_input(tab['segmentation'][nat['row']], src_size, in_size),
                'others': [o['box'] for o in others], 'other_expressions': [o['expression'] for o in others],
                'source': {'shard': nat['shard'], 'row': nat['row'], 'file_name': nat['file_name']}}
        if case['prompt'] != point_prompt(c['expression']):
            raise ValueError(f'Unexpected RefCOCO prompt for {cid}')
        if dis is not None:
            case['expression_b'] = dis['expression']
            case['prompt_b'] = point_prompt(dis['expression'])
            b = dis['box']
            case['truth_b'] = {'box': b, 'center': [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]}
        cases.append(case)
    img_dir = out / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for group, size in MS_SIZES.items():
        for i in range(args.n_ms):
            img, objs = multishape_image(rng, size, n_obj=4 if size <= 64 else 3)
            p = img_dir / f'{group}_{i:03d}.png'
            img.save(p)
            t = int(rng.integers(len(objs)))
            tb = objs[t]['box']
            rest = [o for j, o in enumerate(objs) if j != t]
            dis = pick_distractor(tb, rest, (512, 512))
            case = {'id': f'{group}_{i:03d}', 'group': group, 'set': 'ms', 'cond': group, 'format': 'point',
                    'image_key': f'{group}_{i:03d}', 'image_path': str(p), 'image_sha256': sr.sha256(p),
                    'expression': objs[t]['name'], 'prompt': point_prompt(objs[t]['name']),
                    'truth': {'canvas': [512, 512], 'box': tb, 'center': objs[t]['center']},
                    'objects': objs, 'others': [o['box'] for o in rest],
                    'other_expressions': [o['name'] for o in rest]}
            if dis is not None:
                case['expression_b'] = dis['name']
                case['prompt_b'] = point_prompt(dis['name'])
                case['truth_b'] = {'box': dis['box'], 'center': dis['center']}
            cases.append(case)
    # Round-1 single-target synthetic cases (ablation only): point format, answers regenerated here.
    r1 = {c['id']: c for c in json.loads((Path(args.round1) / 'cases.json').read_text())}
    for cid, c in r1.items():
        if c['cond'] in ('disc_point', 't6g10_point'):
            cases.append({**c, 'group': c['cond'].split('_')[0], 'cond': c['cond'].split('_')[0], 'others': []})
    (out / 'cases.json').write_text(json.dumps(cases, indent=1))
    counts: dict[str, list[int]] = {}
    for c in cases:
        counts.setdefault(c['group'], [0, 0])
        counts[c['group']][0] += 1
        counts[c['group']][1] += int('prompt_b' in c)
    print(json.dumps({k: {'cases': v[0], 'with_what_pair': v[1]} for k, v in counts.items()}, indent=1))


def load_cases(out: Path) -> dict:
    return {c['id']: c for c in json.loads((out / 'cases.json').read_text())}


def load_image(case: dict) -> Image.Image:
    return sr.load_image(case)


def score_point(case: dict, values: list[dict] | None) -> dict:
    """Point answers: inside the target box (or disc/cell for round-1 cases); wrong_object = inside another box only."""
    if case['set'] in ('disc', 't6'):
        return cf.score_values(case, values)
    if values is None:
        return {'valid': False, 'correct': False, 'wrong_object': False}
    v = {d['key']: d['value'] for d in values}
    W, H = case['truth']['canvas']
    px = np.array([v['x'] * W / 1000, v['y'] * H / 1000])
    inside = lambda b: bool(b[0] <= px[0] <= b[2] and b[1] <= px[1] <= b[3])
    correct = inside(case['truth']['box'])
    return {'valid': True, 'correct': correct, 'out_units': [v['x'], v['y']],
            'wrong_object': (not correct) and any(inside(b) for b in case.get('others', []))}


# ----------------------------------------------------------------------------- generation

def stage_gen(rt, out: Path, args) -> None:
    """Greedy answers to the run-A prompt ('a') and the what-pair prompt ('b') of every case."""
    path = out / 'gen.jsonl'
    finished = done_ids(path)
    items = []
    for c in load_cases(out).values():
        if c['group'] not in args.gen_groups:
            continue
        for which in ('a', 'b'):
            key = 'prompt' if which == 'a' else 'prompt_b'
            if key in c and f"{c['id']}:{which}" not in finished:
                items.append((c, which, c[key]))
    for i in range(0, len(items), args.prep_chunk):
        chunk = items[i:i + args.prep_chunk]
        preps = [cr.prepare(rt, load_image(c), p) for c, _, p in chunk]
        by_len: dict[int, list] = {}
        for it, p in zip(chunk, preps):
            by_len.setdefault(len(p.ids), []).append((it, p))
        for L, group in by_len.items():
            for j in range(0, len(group), args.batch_size):
                part = group[j:j + args.batch_size]
                res = cr.run_conditions(rt, [cr.embeds(rt, p) for _, p in part], [p.pos for _, p in part],
                                        args.max_new_tokens, batch_size=len(part))
                rows = []
                for ((c, which, _), _), r in zip(part, res):
                    vals = cf.answer_values(rt, r['ids'], 'point')
                    tcase = c if which == 'a' else {**c, 'truth': {**c['truth'], **c['truth_b']},
                                                    'others': [c['truth']['box']]}
                    rows.append({'id': f"{c['id']}:{which}", 'case': c['id'], 'which': which, 'group': c['group'],
                                 'text': r['text'], 'ids': r['ids'], 'prompt_len': L, 'batch_size': len(part),
                                 'values': vals, 'score': score_point(tcase, vals)})
                append_jsonl(path, rows)
        print(f'gen {i + len(chunk)}/{len(items)}', flush=True)
        del preps
        torch.cuda.empty_cache()


def stage_spotcheck(rt, out: Path, args) -> None:
    path = out / 'spotcheck.jsonl'
    finished = done_ids(path)
    cases = load_cases(out)
    gen = read_jsonl(out / 'gen.jsonl')
    rng = np.random.default_rng(args.seed)
    for k in sorted(rng.choice(len(gen), min(args.spotcheck, len(gen)), replace=False)):
        g = gen[int(k)]
        if g['id'] in finished:
            continue
        c = cases[g['case']]
        prompt = c['prompt'] if g['which'] == 'a' else c['prompt_b']
        ref = qr.generate(rt, qr.build(rt, [load_image(c)], prompt), args.max_new_tokens)
        vr = cf.answer_values(rt, rt.tokenizer.encode(ref['text'], add_special_tokens=False), 'point')
        val = lambda v: None if v is None else [(d['key'], d['value']) for d in v]
        append_jsonl(path, [{'id': g['id'], 'reference': ref['text'], 'manual': g['text'], 'batch_size': g['batch_size'],
                             'text_equal': ref['text'].strip() == g['text'].strip(),
                             'answer_equal': val(vr) == val(g['values'])}])


# ----------------------------------------------------------------------------- patching

def patch_conditions() -> list[tuple[str, list[tuple[int, int]]]]:
    """(name, [(layer, head)]) with head = -1 for a whole Gated DeltaNet mixer output."""
    full = [(l, h) for l in FULL_LAYERS for h in range(16)]
    dn = [(l, -1) for l in DN_LAYERS]
    conds = [('none', [])]
    conds += [(f'h{l}_{h}', [(l, h)]) for l, h in full]
    conds += [(f'L{l}', [(l, h) for h in range(16)]) for l in FULL_LAYERS]
    conds += [('full_all', full), ('full_minus_L20', [x for x in full if x[0] != 20]),
              ('L20_L24', [x for x in full if x[0] in (20, 24)])]
    conds += [(f'loo{l}_{h}', [x for x in full if x != (l, h)]) for l in (20, 24) for h in range(16)]
    conds += [('dn_all', dn)]
    conds += [(f'dn_{a}_{a + 7}', [x for x in dn if a <= x[0] <= a + 7]) for a in (1, 9, 17, 25)]
    conds += [('all_mixers', full + dn)]
    return conds


class TailCapture:
    """Capture token-mixer outputs (output-projection inputs) of every decoder layer at positions >= start."""

    def __init__(self, rt, start: int):
        self.store: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer in range(1, N_LAYERS + 1):
            module, _ = cr.mixer_projection(rt, layer)

            def pre(module, args, layer=layer):
                self.store[layer] = args[0][0, start:].detach().clone()
            self.handles.append(module.register_forward_pre_hook(pre))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


class TailPatch:
    """Write donor mixer outputs (from TailCapture of another run) into rows of a batched forward.

    ops: (row, layer, head); head = -1 replaces the whole mixer output of the layer. Positions >= start of
    the patched forward receive donor positions 0.. in order (both runs share the teacher-forced tail).
    """

    def __init__(self, rt, donor: dict[int, torch.Tensor], ops: list[tuple[int, int, int]], start: int):
        self.handles = []
        by_layer: dict[int, list] = {}
        for op in ops:
            by_layer.setdefault(op[1], []).append(op)
        for layer, lops in by_layer.items():
            module, hd = cr.mixer_projection(rt, layer)
            src = donor[layer]

            def pre(module, args, lops=lops, hd=hd, src=src):
                x = args[0]
                if x.shape[1] - start != src.shape[0]:
                    raise ValueError('Donor tail length differs from the patched tail.')
                y = x.clone()
                s = src.to(x.device, x.dtype)
                for row, _, h in lops:
                    sl = slice(None) if h < 0 else slice(h * hd, (h + 1) * hd)
                    y[row, start:, sl] = s[:, sl]
                return (y,)
            self.handles.append(module.register_forward_pre_hook(pre))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def value_lds(logits_b: torch.Tensor, logits: torch.Tensor, ans: list[int], values: list[dict],
              digits: list[int]) -> list[dict]:
    """Per value: d_A, d_B (run-B argmax digit), LD_B and LD of every row.

    logits_b: (n_tail, V) of run B; logits: (R, n_tail, V) of the patched rows; tail index i predicts ans[i].
    """
    dig = torch.tensor(digits, device=logits_b.device)
    out = []
    for v in values:
        i = v['toks'][0]
        dA = ans[i]
        dB = digits[int(logits_b[i, dig].argmax())]
        ldb = float(logits_b[i, dB] - logits_b[i, dA])
        ld = (logits[:, i, dB].float() - logits[:, i, dA].float()).cpu().tolist()
        out.append({'key': v['key'], 'axis': v['axis'], 'dA': digits.index(dA) if dA in digits else None,
                    'dB': digits.index(dB), 'LD_B': round(ldb, 4), 'ld': ld})
    return out


def pairs_for(rt, c: dict, prep) -> list[dict]:
    """Run-B prompt and inputs for the what pair and every feasible where (content-swap) pair."""
    out = []
    if 'prompt_b' in c:
        pb = cr.prepare(rt, load_image(c), c['prompt_b'])
        out.append({'tag': 'what', 'kind': 'what', 'axis': 'both', 'prep': pb, 'xb': cr.embeds(rt, pb)})
    target = cr.tokens_overlapping(c['truth']['box'], prep.grid)
    for reg in refcoco_swaps(target, prep.grid):
        if not reg['valid']:
            continue
        order = cr.swap_index(prep.grid[0] * prep.grid[1], target, reg['region'])
        out.append({'tag': f"{reg['axis']}swap", 'kind': 'where', 'axis': reg['axis'], 'prep': prep,
                    'xb': cr.embeds(rt, prep, order=order), 'region': reg['region']})
    return out


def image_attention(rt, x, pos, prep, values, L) -> tuple[np.ndarray, dict]:
    """(8, 16, 2, Nv) attention of every full-attention head at the two first-digit queries, plus check."""
    q = [L - 1 + v['toks'][0] for v in values]
    ro = cf.attention_rows(rt, x, pos, q)
    s, e = prep.span['start'], prep.span['end']
    a = np.stack([ro.attn[l][:, :, s:e].numpy() for l in FULL_LAYERS])
    return a.astype(np.float16), ro.check


@torch.inference_mode()
def stage_patch(rt, out: Path, args) -> None:
    path = out / 'patch.jsonl'
    (out / 'attn').mkdir(exist_ok=True)
    cases = load_cases(out)
    digits = cf.digit_ids(rt)
    conds = patch_conditions()
    gen = {g['id']: g for g in read_jsonl(out / 'gen.jsonl')}
    todo = []
    for grp in args.patch_groups:
        # cases with a what pair first, so that a smaller n_patch keeps every what pair
        ids = sorted([cid for cid, c in cases.items() if c['group'] == grp and f'{cid}:a' in gen],
                     key=lambda cid: 'prompt_b' not in cases[cid])[:args.n_patch]
        todo += ids
    if args.reverse:
        todo = todo[::-1]
    for cid in todo:
        if cid in done_ids(path):                       # re-read: a second process may share the list
            continue
        c, g = cases[cid], gen[f'{cid}:a']
        if not g['score']['valid']:
            append_jsonl(path, [{'id': cid, 'status': 'invalid_answer'}])
            continue
        t0 = time.time()
        prep = cr.prepare(rt, load_image(c), c['prompt'])
        L = len(prep.ids)
        ans = g['ids']
        xa_tf, pos_a = cr.teacher_forced(rt, prep, ans[:-1])
        attn = {'a': image_attention(rt, xa_tf, pos_a, prep, g['values'], L)}
        recs = []
        for pr in pairs_for(rt, c, prep):
            pb = pr['prep']
            Lb = len(pb.ids)
            xb_tf, pos_b = cr.teacher_forced(rt, pb, ans[:-1], x=pr['xb'])
            with TailCapture(rt, Lb - 1) as cap:
                lb = rt.model(inputs_embeds=xb_tf, position_ids=pos_b, use_cache=False,
                              logits_to_keep=slice(Lb - 1, None)).logits[0].float()
            donor = cap.store
            attn[pr['tag']] = image_attention(rt, xb_tf, pos_b, pb, g['values'], Lb)
            rows: dict[str, list] = {}
            meta = None
            for i in range(0, len(conds), args.patch_rows):
                part = conds[i:i + args.patch_rows]
                x = xa_tf.expand(len(part), -1, -1).contiguous()
                p = pos_a.expand(-1, len(part), -1).contiguous()
                ops = [(j, l, h) for j, (_, heads) in enumerate(part) for l, h in heads]
                with TailPatch(rt, donor, ops, L - 1):
                    lg = rt.model(inputs_embeds=x, position_ids=p, use_cache=False,
                                  logits_to_keep=slice(L - 1, None)).logits.float()
                lds = value_lds(lb.to(lg.device), lg, ans, g['values'], digits)
                if meta is None:
                    meta = [{k: d[k] for k in ('key', 'axis', 'dA', 'dB', 'LD_B')} for d in lds]
                for k, d in enumerate(lds):
                    for j, (name, _) in enumerate(part):
                        rows.setdefault(name, [None] * len(lds))[k] = round(d['ld'][j], 4)
            values = [{**m, 'LD': {name: rows[name][k] for name, _ in conds}} for k, m in enumerate(meta)]
            recs.append({'tag': pr['tag'], 'kind': pr['kind'], 'axis': pr['axis'], 'prompt_len_b': Lb,
                         'region': pr.get('region'), 'values': values})
            del donor
        np.savez_compressed(out / 'attn' / f'{cid}.npz', **{k: v[0] for k, v in attn.items()})
        append_jsonl(path, [{'id': cid, 'status': 'ok', 'group': c['group'], 'correct': g['score']['correct'],
                             'grid': list(prep.grid), 'prompt_len': L, 'pairs': recs,
                             'attn_check': {k: v[1] for k, v in attn.items()}, 'seconds': round(time.time() - t0, 1)}])
        print(f"patch {cid} {len(recs)} pairs {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- mean ablation

@torch.inference_mode()
def stage_means(rt, out: Path, args) -> None:
    """Mean full-attention head outputs over answer positions (teacher-forced own answers), per group."""
    cases = load_cases(out)
    gen = [g for g in read_jsonl(out / 'gen.jsonl') if g['which'] == 'a' and g['score']['valid']]
    res = {}
    for grp in args.ablate_groups:
        gs = [g for g in gen if g['group'] == grp][:args.n_means]
        if not gs:
            raise ValueError(f'No valid answers for group {grp}')
        acc = {l: None for l in FULL_LAYERS}
        n = 0
        for g in gs:
            c = cases[g['case']]
            prep = cr.prepare(rt, load_image(c), c['prompt'])
            L = len(prep.ids)
            x, pos = cr.teacher_forced(rt, prep, g['ids'][:-1])
            with TailCapture(rt, L - 1) as cap:
                rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
            for l in FULL_LAYERS:
                s = cap.store[l].float().sum(0).cpu()
                acc[l] = s if acc[l] is None else acc[l] + s
            n += cap.store[FULL_LAYERS[0]].shape[0]
        res[grp] = {str(l): (acc[l] / n).numpy() for l in FULL_LAYERS}
        print(f'means {grp}: {len(gs)} answers, {n} positions', flush=True)
    np.savez_compressed(out / 'means.npz', **{f'{g}_{l}': v for g, d in res.items() for l, v in d.items()})


def parse_sets(spec: str) -> dict[str, list[tuple[int, int]]]:
    """'H11=20.11;readers=20.11,24.3' -> {'H11': [(20, 11)], 'readers': [(20, 11), (24, 3)]}."""
    out: dict[str, list[tuple[int, int]]] = {'none': []}
    for item in filter(None, (s.strip() for s in spec.split(';'))):
        name, heads = item.split('=')
        hs = []
        for t in heads.split(','):
            l, h = t.split('.')
            l, h = int(l), int(h)
            if l not in FULL_LAYERS or not 0 <= h < 16:
                raise ValueError(f'Invalid head {t} in ablation set {name}')
            hs.append((l, h))
        out[name.strip()] = hs
    return out


@torch.inference_mode()
def stage_ablate(rt, out: Path, args) -> None:
    path = out / 'ablate.jsonl'
    finished = done_ids(path)
    cases = load_cases(out)
    sets = parse_sets(args.ablate_sets)
    names = list(sets)
    mz = np.load(out / 'means.npz')
    for grp in args.ablate_groups:
        means = {}
        for l in FULL_LAYERS:
            module, hd = cr.mixer_projection(rt, l)
            m = torch.from_numpy(mz[f'{grp}_{l}']).to(module.weight.device)
            for h in range(16):
                means[(l, h)] = m[h * hd:(h + 1) * hd]
        ids = [cid for cid, c in cases.items() if c['group'] == grp][:args.n_ablate]
        for cid in ids:
            for start_at in args.ablate_from:
                rid = f'{cid}@{start_at}'
                if rid in finished:
                    continue
                c = cases[cid]
                prep = cr.prepare(rt, load_image(c), c['prompt'])
                L = len(prep.ids)
                start = L - 1 if start_at == 'answer' else prep.span['end']
                x = cr.embeds(rt, prep)
                ops = [(j, l, h, 'mean') for j, n in enumerate(names) for l, h in sets[n]]
                res = []
                for i in range(0, len(names), args.batch_size):
                    sub = names[i:i + args.batch_size]
                    sops = [(j - i, l, h, s) for j, l, h, s in ops if i <= j < i + len(sub)]
                    with cr.MixerEdit(rt, sops, start, means=means):
                        res += cr.greedy(rt, x.expand(len(sub), -1, -1).contiguous(),
                                         prep.pos.expand(-1, len(sub), -1).contiguous(), args.max_new_tokens)
                rows = []
                for n, r in zip(names, res):
                    vals = cf.answer_values(rt, r['ids'], 'point')
                    rows.append({'set': n, 'text': r['text'], 'values': vals, 'score': score_point(c, vals)})
                append_jsonl(path, [{'id': rid, 'case': cid, 'group': grp, 'from': start_at, 'sets': sets,
                                     'results': rows}])
        print(f'ablate {grp} done', flush=True)


# ----------------------------------------------------------------------------- vertical flip

def flipped_case(c: dict) -> dict:
    """The case with its image flipped upside down; target box and mask mirrored about the horizontal midline."""
    W, H = c['truth']['canvas']
    b = c['truth']['box']
    fb = [b[0], H - b[3], b[2], H - b[1]]
    out = {**c, 'truth': {**c['truth'], 'box': fb, 'center': [(fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2]},
           'others': [[o[0], H - o[3], o[2], H - o[1]] for o in c.get('others', [])]}
    if 'mask' in c:
        out['mask'] = [[x, H - y] for x, y in c['mask']]
    return out


@torch.inference_mode()
def stage_flip(rt, out: Path, args) -> None:
    """Answers and first-digit attention rows (all full-attention heads) on vertically flipped images.

    Separates a semantic part (e.g. a person's head, which moves to the bottom of the box) from a geometric
    edge (the top edge stays on top) as the target of each head's attention.
    """
    path = out / 'flip.jsonl'
    (out / 'flip_attn').mkdir(exist_ok=True)
    finished = done_ids(path)
    cases = [c for c in load_cases(out).values() if c['group'] in args.flip_groups and c['id'] not in finished]
    for i in range(0, len(cases), args.prep_chunk):
        chunk = cases[i:i + args.prep_chunk]
        imgs = [load_image(c).transpose(Image.FLIP_TOP_BOTTOM) for c in chunk]
        preps = [cr.prepare(rt, im, c['prompt']) for c, im in zip(chunk, imgs)]
        by_len: dict[int, list] = {}
        for c, p in zip(chunk, preps):
            by_len.setdefault(len(p.ids), []).append((c, p))
        for L, group in by_len.items():
            for j in range(0, len(group), args.batch_size):
                part = group[j:j + args.batch_size]
                res = cr.run_conditions(rt, [cr.embeds(rt, p) for _, p in part], [p.pos for _, p in part],
                                        args.max_new_tokens, batch_size=len(part))
                rows = []
                for (c, prep), r in zip(part, res):
                    fc = flipped_case(c)
                    vals = cf.answer_values(rt, r['ids'], 'point')
                    rec = {'id': c['id'], 'group': c['group'], 'text': r['text'], 'ids': r['ids'], 'values': vals,
                           'score': score_point(fc, vals), 'grid': list(prep.grid), 'batch_size': len(part)}
                    if vals is not None and len(vals) == 2:
                        x, pos = cr.teacher_forced(rt, prep, r['ids'][:-1])
                        a, chk = image_attention(rt, x, pos, prep, vals, L)
                        np.savez_compressed(out / 'flip_attn' / f"{c['id']}.npz", attn=a)
                        rec['attn_check'] = chk
                    rows.append(rec)
                append_jsonl(path, rows)
        print(f'flip {i + len(chunk)}/{len(cases)}', flush=True)
        del preps
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- main

STAGES = {'gen': stage_gen, 'spotcheck': stage_spotcheck, 'patch': stage_patch, 'means': stage_means,
          'ablate': stage_ablate, 'flip': stage_flip}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stages', nargs='+', required=True, choices=['cases', *STAGES])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True, help='2.5e run directory (RefCOCO cases and answers)')
    p.add_argument('--round1', type=Path, required=True, help='round-1 3.11 run directory (single-target cases)')
    p.add_argument('--probe-cases', type=Path, required=True, help='3.4 cases.json with the RefCOCO source rows')
    p.add_argument('--refcoco-dir', type=Path, required=True)
    p.add_argument('--n-ms', type=int, default=48)
    p.add_argument('--gen-groups', nargs='+', default=['rc', 'ms_small', 'ms_large', 'disc', 't6g10'])
    p.add_argument('--patch-groups', nargs='+', default=['rc', 'ms_small', 'ms_large'])
    p.add_argument('--n-patch', type=int, default=160)
    p.add_argument('--patch-rows', type=int, default=48)
    p.add_argument('--ablate-groups', nargs='+', default=['rc', 'ms_small', 'ms_large', 'disc', 't6g10'])
    p.add_argument('--ablate-sets', default='')
    p.add_argument('--ablate-from', nargs='+', default=['answer'], choices=['answer', 'text'])
    p.add_argument('--n-ablate', type=int, default=300)
    p.add_argument('--n-means', type=int, default=64)
    p.add_argument('--flip-groups', nargs='+', default=['rc', 'ms_large'])
    p.add_argument('--reverse', action='store_true')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--prep-chunk', type=int, default=64)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--max-memory', default='')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = list(args.stages)
    if 'cases' in stages:
        stage_cases(args.output, args)
        stages.remove('cases')
    if not stages:
        return
    rt = cf.load_runtime(args.model, args.max_memory)
    for st in stages:
        meta = {'stage': st, 'args': {k: str(v) for k, v in vars(args).items()}, 'runtime': qr.describe(rt),
                'code_sha256': sr.sha256(Path(__file__)), 'started': datetime.now(timezone.utc).isoformat(),
                'device': torch.cuda.get_device_name(0)}
        append_jsonl(args.output / 'run_log.jsonl', [meta])
        STAGES[st](rt, args.output, args)


if __name__ == '__main__':
    main()
