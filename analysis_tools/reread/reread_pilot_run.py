"""GPU stage drivers of experiment 5.1 (local re-read pilot).

Stages (selected with --stages; every stage is resumable at image granularity):
  verify      engine checks: custom-embedding path vs the standard processor path for one and two images
              (first-token logits), and batched vs single greedy decoding on 20 prompts.
  det         detail task (revised T3, Yes-No logit) on 10x10 / 20x20 grids, k = 1 and 3 windows,
              all conditions including the feature-source scan.
  loc         T5 / T6 localization with point and row/column direct answers, k = 5 windows.
  think       think-then-answer row/column subset of loc (20x20 grids).
  think_long  the same prompts for a smaller subset and three conditions with a 2048-token budget.
  refcoco     RefCOCO val small objects (area 1-5 %, experiment 4.2 cases), bbox answers.
  babyvision  BabyVision Find-the-different (16 items, experiment 1.3 annotation), row/column answers.

Conditions (see the results document, section 2):
  a full image; b full image upscaled 2x; c1/c2/c4 full image + pixel crop re-encoded at 1x/2x/4x;
  d full image + cached window tokens with original row/col ids; e same tokens with new-image ids;
  fc / fd random-window controls of c2 / d; h = d + c2 (cached window followed by a 2x crop);
  g<k> cached window taken from the merger projection of ViT block k; gcat = g9 + g17 + d.
Each condition with extra visual input runs without (hint 0) and with (hint 1) the textual region hint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.reread import reread_pilot as rp

CODE_FILES = ('reread_pilot.py', 'reread_pilot_run.py')
DET_CONDS = ['a', 'b'] + [f'{c}|{h}' for c in ('c1', 'c2', 'c4', 'd', 'e', 'fc', 'fd', 'h') for h in (0, 1)]
DET_SCAN = ['g3|1', 'g9|1', 'g17|1', 'g25|1', 'gcat|1']
DET_BLOCKS = (3, 9, 17, 25)
LOC_CONDS = ['a', 'b'] + [f'{c}|{h}' for c in ('c2', 'd', 'e', 'fc', 'fd', 'h') for h in (0, 1)]
THINK_CONDS = ['a', 'c2|1', 'd|1', 'e|1', 'h|1']
THINK_LONG_CONDS = ['c2|1', 'd|1', 'e|1']      # budget check of the truncated think-then-answer runs
REF_CONDS = LOC_CONDS
BV_CONDS = ['a', 'b', 'c2|0', 'c2|1', 'd|0', 'd|1', 'e|1', 'fc|1', 'fd|1', 'h|1']


# ----------------------------------------------------------------------------- io

def code_digest() -> dict:
    root = Path(__file__).parent
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] for f in CODE_FILES}


def write_meta(out: Path, rt, args, stage: str, extra: dict | None = None) -> None:
    meta = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **qr.describe(rt), 'code': code_digest(),
            'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, **(extra or {})}
    (out / f'{stage}_meta.json').write_text(json.dumps(meta, indent=1) + '\n')


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


# ----------------------------------------------------------------------------- conditions

def parse_cond(cond: str) -> tuple[str, int]:
    name, _, hint = cond.partition('|')
    return name, int(hint or 0)


def crops_needed(conds: list[str]) -> set[tuple[str, int]]:
    need = set()
    for c in conds:
        name, _ = parse_cond(c)
        if name in ('c1', 'c2', 'c4'):
            need.add(('W', int(name[1])))
        if name == 'h':
            need.add(('W', 2))
        if name == 'fc':
            need.add(('R', 2))
    return need


@torch.inference_mode()
def build_inputs(rt, image: Image.Image, win: rp.Window, rwin: rp.Window | None, conds: list[str],
                 prompt: str, blocks: tuple[int, ...] = ()) -> list[dict]:
    """Encode the image (and crops) once and assemble every condition. Returns dicts with x, pos, meta."""
    need = sorted(crops_needed(conds))
    crops = [rp.crop(image, win if w == 'W' else rwin, s) for w, s in need]
    enc = rp.encode(rt, [image] + crops, blocks)
    full, crop_enc = enc[0], dict(zip(need, enc[1:]))
    big = rp.encode(rt, [image.resize((image.width * 2, image.height * 2), Image.Resampling.BICUBIC)])[0] \
        if 'b' in conds else None
    grid = full['grid']
    s0 = rp.Segment(full['vis'], grid)

    def cached(w: rp.Window, key: str = 'vis', mode: str = 'orig') -> rp.Segment:
        idx = torch.tensor(w.indices(grid[1]), device=full['vis'].device)
        return rp.Segment(full[key][idx], (w.h, w.w), mode, w)

    def fresh(e: dict) -> rp.Segment:
        return rp.Segment(e['vis'], e['grid'])

    out = []
    for cond in conds:
        name, hint = parse_cond(cond)
        used = win
        vit_new = 0
        if name == 'a':
            segs = [s0]
        elif name == 'b':
            segs, vit_new = [fresh(big)], big['vis'].shape[0] * 4
        elif name in ('c1', 'c2', 'c4'):
            e = crop_enc[('W', int(name[1]))]
            segs, vit_new = [s0, fresh(e)], e['vis'].shape[0] * 4
        elif name == 'd':
            segs = [s0, cached(win)]
        elif name == 'e':
            segs = [s0, cached(win, mode='new')]
        elif name == 'fc':
            e = crop_enc[('R', 2)]
            segs, vit_new, used = [s0, fresh(e)], e['vis'].shape[0] * 4, rwin
        elif name == 'fd':
            segs, used = [s0, cached(rwin)], rwin
        elif name == 'h':
            e = crop_enc[('W', 2)]
            segs, vit_new = [s0, cached(win), fresh(e)], e['vis'].shape[0] * 4
        elif name.startswith('g') and name[1:].isdigit():
            segs = [s0, cached(win, f'block_{name[1:]}')]
        elif name == 'gcat':
            segs = [s0, cached(win, 'block_9'), cached(win, 'block_17'), cached(win)]
        else:
            raise ValueError(f'Unknown condition {cond}')
        text = (rp.hint_text(len(segs) - 1, used.norm_box(grid)) if hint else '') + prompt
        x, pos, meta = rp.assemble(rt, segs, text)
        out.append({'cond': cond, 'x': x, 'pos': pos, 'meta': {**meta, 'n_vit_patches_new': vit_new,
                                                               'n_vit_patches_full': grid[0] * grid[1] * 4},
                    'window_used': used.as_list(), 'prompt': text})
    return out


@torch.inference_mode()
def run_group(rt, items: list[dict], mode: str, batch_size: int, max_new_tokens: int) -> list[dict]:
    """Run items grouped by sequence length (no padding). mode 'yesno' (prefill) or 'gen' (greedy)."""
    yes, no = qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No')
    by_len: dict[int, list[int]] = {}
    for i, it in enumerate(items):
        by_len.setdefault(it['x'].shape[1], []).append(i)
    res: list[dict | None] = [None] * len(items)
    for L, idx in by_len.items():
        bs = max(1, min(batch_size, (batch_size * 2048) // max(L, 2048)))      # long prompts get smaller batches
        for j in range(0, len(idx), bs):
            chunk = idx[j:j + bs]
            x = torch.cat([items[i]['x'] for i in chunk], 0)
            p = torch.cat([items[i]['pos'] for i in chunk], 1)
            if mode == 'yesno':
                logits = rt.model(inputs_embeds=x, position_ids=p, logits_to_keep=1).logits[:, -1].float()
                for b, i in enumerate(chunk):
                    top = int(logits[b].argmax())
                    res[i] = {'s': float(logits[b, yes] - logits[b, no]), 'top_token': rt.tokenizer.decode([top]),
                              'batch_size': len(chunk)}
            else:
                outs = cr.greedy(rt, x, p, max_new_tokens)
                for b, i in enumerate(chunk):
                    o = outs[b]
                    res[i] = {'text': o['text'], 'n_tokens': len(o['ids']), 'finish_reason': o['finish_reason'],
                              'batch_size': len(chunk)}
    return res


def strip(item: dict) -> dict:
    return {k: v for k, v in item.items() if k not in ('x', 'pos')}


# ----------------------------------------------------------------------------- synthetic stages

def synth_windows(case: dict, k: int, rng: np.random.Generator, centred: bool) -> tuple[rp.Window, rp.Window]:
    n = case['grid_n']
    key = tuple(case['asked'] if centred else case['target'])
    cells = rp.centered_cells(key, k, n, n) if centred else rp.random_cells_containing(key, k, n, n, rng)
    win = rp.cell_window(cells[0], cells[1], k)
    tgt = rp.cell_window(case['target'][0], case['target'][1], 1)
    asked = rp.cell_window(case['asked'][0], case['asked'][1], 1) if 'asked' in case else tgt
    grid = (2 * n, 2 * n)
    try:
        rwin = rp.random_window(grid, (win.h, win.w), [win, tgt, asked], rng, step=2)
    except ValueError:
        # Large windows on small grids: no disjoint window exists. Take the cell-aligned window with the
        # least overlap with the oracle window among those excluding the target and asked cells.
        best, best_ov = None, None
        for r0 in range(0, grid[0] - win.h + 1, 2):
            for c0 in range(0, grid[1] - win.w + 1, 2):
                cand = rp.Window(r0, c0, win.h, win.w)
                if cand.overlaps(tgt) or cand.overlaps(asked):
                    continue
                ov = max(0, min(r0 + win.h, win.r0 + win.h) - max(r0, win.r0)) * \
                    max(0, min(c0 + win.w, win.c0 + win.w) - max(c0, win.c0))
                if best_ov is None or ov < best_ov or (ov == best_ov and rng.random() < 0.5):
                    best, best_ov = cand, ov
        if best is None:
            raise
        rwin = best
    return win, rwin


def stage_det(rt, out: Path, args) -> None:
    root = out / 'det'
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    for n in args.grids:
        for kind, level in (('gap', 1), ('color', 2), ('color', 3)):
            cases += rp.det_cases(n, kind, level, args.det_n, args.seed + 1000 * n + 10 * level + (kind == 'gap'))
    (root / 'cases.json').write_text(json.dumps(cases, indent=0))
    path = root / 'results.jsonl'
    done = {r['case_id'] for r in read_jsonl(path)}
    t0 = time.time()
    pending = [c for c in cases if c['id'] not in done]
    for start in range(0, len(pending), args.image_chunk):
        chunk_cases = pending[start:start + args.image_chunk]
        items, keys = [], []
        for case in chunk_cases:
            image, _ = rp.synth_image(case)
            prompt = rp.det_prompt(tuple(case['asked']))
            for k in (1, 3):
                rng = np.random.default_rng(case['seed'] * 7 + k)
                win, rwin = synth_windows(case, k, rng, centred=True)
                conds = DET_CONDS + (DET_SCAN if k == 3 else [])
                if k == 3:
                    conds = [c for c in conds if c not in ('a', 'b')]        # a and b do not depend on k
                for it in build_inputs(rt, image, win, rwin, conds, prompt, DET_BLOCKS if k == 3 else ()):
                    items.append(it)
                    keys.append({'case_id': case['id'], 'k': k, 'window': win.as_list(), 'rwindow': rwin.as_list()})
        res = run_group(rt, items, 'yesno', args.batch_size, 1)
        append_jsonl(path, [{**k, **strip(it), **r} for k, it, r in zip(keys, items, res)])
        del items
        print(f'det {start + len(chunk_cases)}/{len(pending)} {time.time() - t0:.0f}s', flush=True)
    write_meta(root, rt, args, 'det', {'conditions': DET_CONDS + DET_SCAN, 'blocks': DET_BLOCKS})


def stage_loc(rt, out: Path, args, think: bool = False, long: bool = False) -> None:
    root = out / ('think_long' if long else 'think' if think else 'loc')
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    for task in ('T6', 'T5'):
        for n in args.grids:
            cases += rp.loc_cases(n, task, args.loc_n, args.seed + 5000 + 100 * n + (task == 'T5'))
    if think:
        cases = [c for c in cases if c['grid_n'] == max(args.grids)]
        cases = [c for c in cases if int(c['id'].rsplit('_', 1)[1]) < (args.think_long_n if long else args.think_n)]
    (root / 'cases.json').write_text(json.dumps(cases, indent=0))
    path = root / 'results.jsonl'
    done = {r['case_id'] for r in read_jsonl(path)}
    fmts = [('rowcol', 'think')] if think else [('point', 'direct'), ('rowcol', 'direct')]
    conds = THINK_LONG_CONDS if long else THINK_CONDS if think else LOC_CONDS
    t0 = time.time()
    pending = [c for c in cases if c['id'] not in done]
    for start in range(0, len(pending), args.image_chunk):
        chunk_cases = pending[start:start + args.image_chunk]
        items, keys = [], []
        for case in chunk_cases:
            image, _ = rp.synth_image(case)
            rng = np.random.default_rng(case['seed'] * 7 + 5)
            win, rwin = synth_windows(case, args.loc_k, rng, centred=False)
            for fmt, style in fmts:
                built = build_inputs(rt, image, win, rwin, conds, rp.loc_prompt(case['task'], fmt, style))
                for it in built:
                    items.append(it)
                    keys.append({'case_id': case['id'], 'format': fmt, 'style': style, 'k': args.loc_k,
                                 'window': win.as_list(), 'rwindow': rwin.as_list()})
        budget = args.max_think_long if long else args.max_think if think else args.max_new_tokens
        res = run_group(rt, items, 'gen', args.batch_size, budget)
        append_jsonl(path, [{**k, **strip(it), **r} for k, it, r in zip(keys, items, res)])
        del items
        print(f'{root.name} {start + len(chunk_cases)}/{len(pending)} {time.time() - t0:.0f}s', flush=True)
    write_meta(root, rt, args, root.name, {'conditions': conds})


# ----------------------------------------------------------------------------- natural images

def refcoco_cases(args) -> list[dict]:
    rows = read_jsonl(args.refcoco / 'refcoco_cases.jsonl')
    rows = [r for r in rows if r['stratum'] == '1to5']
    return rows[:args.ref_n]


def refcoco_todo(args, cases: list[dict]) -> list[int]:
    """Indices processed by this worker: all, or those from --ref-skip on (second worker)."""
    return list(range(args.ref_skip, len(cases)))


def ref_random_window(grid, win: rp.Window, center_tok: tuple[float, float], rng) -> tuple[rp.Window | None, float]:
    """Same-size window that does not contain the target centre; least overlap with the oracle window among
    200 lattice samples. Returns (window or None, overlap fraction of the oracle window area)."""
    H, W = grid
    best, best_ov = None, 2.0
    for _ in range(200):
        r0, c0 = int(rng.integers(0, H - win.h + 1)), int(rng.integers(0, W - win.w + 1))
        cand = rp.Window(r0, c0, win.h, win.w)
        if r0 <= center_tok[0] < r0 + win.h and c0 <= center_tok[1] < c0 + win.w:
            continue
        ov = max(0, min(r0 + win.h, win.r0 + win.h) - max(r0, win.r0)) * \
            max(0, min(c0 + win.w, win.c0 + win.w) - max(c0, win.c0)) / (win.h * win.w)
        if ov < best_ov:
            best, best_ov = cand, ov
    return best, (best_ov if best is not None else float('nan'))


def stage_refcoco(rt, out: Path, args) -> None:
    root = out / 'refcoco'
    root.mkdir(parents=True, exist_ok=True)
    cases = refcoco_cases(args)
    path = root / 'results.jsonl'
    done = {r['case_id'] for r in read_jsonl(path)}
    meta_rows, buf = [], []
    t0 = time.time()

    def flush():
        items = [it for _, it in buf]
        res = run_group(rt, items, 'gen', args.batch_size, args.max_new_tokens)
        append_jsonl(path, [{**info, **strip(it), **r} for (info, it), r in zip(buf, res)])
        buf.clear()

    todo = set(refcoco_todo(args, cases))
    for ci, case in enumerate(cases):
        src = Image.open(args.refcoco / 'refcoco_images' / case['image']).convert('RGB')
        image = rp.resize32(src)
        sx, sy = image.width / case['width'], image.height / case['height']
        x0, y0, x1, y1 = case['bbox']               # experiment 4.2 stores [x_min, y_min, x_max, y_max] pixels
        box_px = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
        truth = [1000 * x0 / case['width'], 1000 * y0 / case['height'], 1000 * x1 / case['width'],
                 1000 * y1 / case['height']]
        grid = (image.height // rp.TOKEN, image.width // rp.TOKEN)
        rng = np.random.default_rng(args.seed + ci)
        win = rp.pixel_to_window(box_px, grid, rng)
        center = ((box_px[1] + box_px[3]) / 2 / rp.TOKEN, (box_px[0] + box_px[2]) / 2 / rp.TOKEN)
        rwin, ov = ref_random_window(grid, win, center, rng)
        info = {'case_id': case['id'], 'truth_norm': truth, 'grid': list(grid), 'window': win.as_list(),
                'rwindow': rwin.as_list() if rwin else None, 'rwindow_overlap': ov, 'expression': case['expression'],
                'image': case['image'], 'image_size': [image.width, image.height]}
        meta_rows.append(info)
        if case['id'] in done or ci not in todo:
            continue
        conds = REF_CONDS if rwin is not None else [c for c in REF_CONDS if not c.startswith('f')]
        buf.extend((info, it) for it in build_inputs(rt, image, win, rwin, conds, case['prompt']))
        if len({b[0]['case_id'] for b in buf}) >= args.image_chunk:
            flush()
            print(f'refcoco {ci + 1}/{len(cases)} {time.time() - t0:.0f}s', flush=True)
    if buf:
        flush()
    (root / 'cases.json').write_text(json.dumps(meta_rows, indent=0))   # all cases; identical for every worker
    write_meta(root, rt, args, 'refcoco' if args.ref_skip == 0 else f'refcoco_from{args.ref_skip}', {'conditions': REF_CONDS})


def stage_babyvision(rt, out: Path, args) -> None:
    root = out / 'babyvision'
    root.mkdir(parents=True, exist_ok=True)
    ann = {it['taskId']: it for it in json.loads(args.bv_annotation.read_text())['items']}
    cases = [c for c in read_jsonl(args.bv_run / 'babyvision_cases.jsonl') if c['task'] == 'T5']
    path = root / 'results.jsonl'
    done = {r['case_id'] for r in read_jsonl(path)}
    meta_rows = []
    for case in cases:
        item = ann[case['taskId']]
        image = Image.open(args.bv_run / case['image']).convert('RGB')
        grid = (image.height // rp.TOKEN, image.width // rp.TOKEN)
        nr, nc = item['rows'], item['cols']
        rng = np.random.default_rng(args.seed + case['taskId'])
        k = min(3, nr, nc)
        r0, c0 = rp.random_cells_containing(tuple(item['answer_rc']), k, nr, nc, rng)
        boxes = [item['cell_boxes'][r * nc + c] for r in range(r0, r0 + k) for c in range(c0, c0 + k)]
        union = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
        win = rp.bbox_window(union, grid)
        tb = item['cell_boxes'][item['answer_rc'][0] * nc + item['answer_rc'][1]]
        center = ((tb[1] + tb[3]) / 2 / rp.TOKEN, (tb[0] + tb[2]) / 2 / rp.TOKEN)
        rwin, ov = ref_random_window(grid, win, center, rng)
        info = {'case_id': case['id'], 'taskId': case['taskId'], 'rows': nr, 'cols': nc,
                'target_rc': item['answer_rc'], 'cells': [r0, c0, k], 'window': win.as_list(),
                'rwindow': rwin.as_list() if rwin else None, 'rwindow_overlap': ov, 'grid': list(grid),
                'image': case['image']}
        meta_rows.append(info)
        if case['id'] in done:
            continue
        conds = BV_CONDS if rwin is not None else [c for c in BV_CONDS if not c.startswith('f')]
        items = build_inputs(rt, image, win, rwin, conds, case['prompt'])
        res = run_group(rt, items, 'gen', args.batch_size, args.max_new_tokens)
        append_jsonl(path, [{**info, **strip(it), **r} for it, r in zip(items, res)])
        del items
    (root / 'cases.json').write_text(json.dumps(meta_rows, indent=0))
    write_meta(root, rt, args, 'babyvision', {'conditions': BV_CONDS})


# ----------------------------------------------------------------------------- verification

@torch.inference_mode()
def stage_verify(rt, out: Path, args) -> None:
    """Custom path vs standard processor path, and batched vs single greedy decoding."""
    root = out / 'verify'
    root.mkdir(parents=True, exist_ok=True)
    report = {'logits': [], 'batch_agreement': None}
    case = rp.loc_cases(10, 'T6', 1, args.seed)[0]
    image, _ = rp.synth_image(case)
    prompt = rp.loc_prompt('T6', 'point')
    win = rp.cell_window(2, 3, 3)
    crop2 = rp.crop(image, win, 2)
    for label, imgs, conds in (('one_image', [image], 'a'), ('two_images', [image, crop2], 'c2|0')):
        batch = qr.build(rt, imgs, prompt)
        ref = rt.model(**batch).logits[0, -1].float()
        it = build_inputs(rt, image, win, None, [conds], prompt)[0]
        mine = rt.model(inputs_embeds=it['x'], position_ids=it['pos']).logits[0, -1].float()
        mrope, _ = rt.model.model.get_rope_index(batch['input_ids'], mm_token_type_ids=batch['mm_token_type_ids'],
                                                 image_grid_thw=batch['image_grid_thw'])
        report['logits'].append({'case': label, 'max_abs_diff': float((ref - mine).abs().max()),
                                 'ref_scale': float(ref.abs().max()), 'same_top': int(ref.argmax()) == int(mine.argmax()),
                                 'same_length': batch['input_ids'].shape[1] == it['x'].shape[1],
                                 'same_mrope': bool(torch.equal(mrope.cpu(), it['pos'][1:].cpu()))})
    # batched vs single decoding on 20 prompts of equal length (5 cases x 4 conditions)
    items = []
    for c in rp.loc_cases(20, 'T6', 5, args.seed + 1):
        image, _ = rp.synth_image(c)
        rng = np.random.default_rng(c['seed'])
        w, rw = synth_windows(c, 5, rng, centred=False)
        items += build_inputs(rt, image, w, rw, ['a', 'c2|0', 'd|0', 'e|0'], rp.loc_prompt('T6', 'point'))
    batched = run_group(rt, items, 'gen', args.batch_size, args.max_new_tokens)
    single = run_group(rt, items, 'gen', 1, args.max_new_tokens)
    agree = [b['text'] == s['text'] for b, s in zip(batched, single)]
    report['batch_agreement'] = {'n': len(agree), 'identical_text': int(sum(agree)),
                                 'pairs': [(b['text'], s['text']) for b, s in zip(batched, single)]}
    (root / 'verify.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v if k != 'batch_agreement' else {kk: vv for kk, vv in v.items() if kk != 'pairs'}
                      for k, v in report.items()}, indent=1), flush=True)
    write_meta(root, rt, args, 'verify')


STAGES = {'verify': stage_verify, 'det': stage_det, 'loc': stage_loc,
          'think': lambda rt, out, args: stage_loc(rt, out, args, think=True),
          'think_long': lambda rt, out, args: stage_loc(rt, out, args, think=True, long=True),
          'refcoco': stage_refcoco, 'babyvision': stage_babyvision}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--stages', required=True)
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--grids', type=int, nargs='+', default=[10, 20])
    p.add_argument('--det-n', type=int, default=48)
    p.add_argument('--loc-n', type=int, default=32)
    p.add_argument('--loc-k', type=int, default=5)
    p.add_argument('--think-n', type=int, default=16)
    p.add_argument('--ref-n', type=int, default=151)
    p.add_argument('--ref-skip', type=int, default=0, help='second worker: start at this RefCOCO case index')
    p.add_argument('--refcoco', type=Path, required=True, help='experiment 4.2 RefCOCO case directory')
    p.add_argument('--bv-run', type=Path, required=True, help='experiment 1.3 run directory (BabyVision cases)')
    p.add_argument('--bv-annotation', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--image-chunk', type=int, default=8)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--max-think', type=int, default=1024)
    p.add_argument('--max-think-long', type=int, default=2048)
    p.add_argument('--think-long-n', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    for stage in args.stages.split(','):
        print(f'== stage {stage}', flush=True)
        STAGES[stage](rt, args.output, args)


if __name__ == '__main__':
    main()
