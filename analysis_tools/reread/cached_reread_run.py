"""Experiment 5.2 runner: native zoom tool vs cached re-read on natural and synthetic benchmarks.

Stages (comma list):
  verify   custom embedding path vs the processor path, and padded batch vs batch size 1;
  <bench>  accuracy run of one benchmark over all conditions (textvqa, gqa, mmvp, pope, babyvision,
           syn_color, syn_gap, syn_loc); results.jsonl rows are keyed by (case_id, condition) and resumable;
  timing   batch-size-1 end-to-end pipeline with a real KV-cache continuation after the tool call, on a
           subset of every benchmark, recording wall-clock time, peak GPU memory and token counts.

See docs/analysis/reread/design_cached_reread.md.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.reread import cached_reread as cr
from analysis_tools.reread import reread_pilot as rp
from eva.benchmark import score as bv_strict
from eva.diagnostic_scoring import diagnostic_score

# Every condition uses the same answer convention: free text is allowed, the last line carries
# "Final answer: ...". After a tool turn the model answers in prose, so a bare-answer format cannot be enforced.
FINAL = 'Write the final answer on its own last line as "Final answer: <answer>".'
SHORT_FMT = 'Answer the question using a single word or phrase. ' + FINAL
SHORT = '\n' + SHORT_FMT
MMVP_FMT = "Answer with the option's letter, (a) or (b). Write the final answer on its own last line as " \
           '"Final answer: <letter>".'
POPE_FMT = 'Answer yes or no. Write the final answer on its own last line as "Final answer: yes" or "Final answer: no".'
BV_CHOICE_FMT = 'Answer with the option letter. Write the final answer on its own last line as "Final answer: X".'
BV_BLANK_FMT = ('Follow the question\'s answer format. Write the final answer on its own last line as '
                '"Final answer: <answer>".')
DET_FMT = 'Answer Yes or No. Write the final answer on its own last line as "Final answer: Yes" or "Final answer: No".'
LOC_FMT = ('Write the final answer on its own last line as "Final answer: {"row": r, "column": c}", counting rows '
           'from top to bottom and columns from left to right, starting at 1.')
NATURAL = ('textvqa', 'gqa', 'mmvp', 'pope', 'babyvision')
SYNTH = ('syn_color', 'syn_gap', 'syn_loc')
ORACLE_BENCHES = ('gqa',) + SYNTH
BASE_CONDS = ['base', 'full2x', 'crop1', 'crop2', 'crop4', 'c27', 'c9', 'h9', 'h9n']
ORACLE_CONDS = ['o_crop2', 'o_h9']
TIMING_CONDS = ['base', 'full2x', 'crop1', 'crop2', 'crop4', 'c27', 'h27', 'h9']


# ----------------------------------------------------------------------------- io helpers

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(cr.dumps(r) + '\n')


def code_digest() -> dict:
    files = ['analysis_tools/reread/cached_reread.py', 'analysis_tools/reread/cached_reread_run.py',
             'analysis_tools/reread/reread_pilot.py', 'analysis_tools/common/qwen_runtime.py']
    return {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16] for f in files}


def write_meta(root: Path, rt, args, stage: str, extra: dict) -> None:
    meta = {'stage': stage, 'time': datetime.now(timezone.utc).isoformat(), 'runtime': qr.describe(rt),
            'args': {k: str(v) for k, v in vars(args).items()}, 'code': code_digest(),
            'gpu': torch.cuda.get_device_name(0), **extra}
    (root / f'meta_{stage}.json').write_text(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------- benchmarks

def load_cases(bench: str, args) -> list[cr.Case]:
    rng = np.random.default_rng(args.seed + sum(map(ord, bench)))
    n = args.n[bench] if isinstance(args.n, dict) else args.n
    if bench == 'textvqa':
        rows = read_jsonl(args.data_dir / 'TextVQA/data/val_data.jsonl')
        idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
        return [cr.Case(f"textvqa_{rows[i]['question_id']}", bench, rows[i]['question'] + SHORT, rows[i]['answers'],
                        SHORT, str(args.data_dir / 'TextVQA/data/val_images' / rows[i]['image'])) for i in sorted(idx)]
    if bench == 'gqa':
        rows = read_jsonl(args.vcot_gqa)
        keep = []
        for r in rows:
            b = r['bboxs']
            ub = [min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)]
            ratio = (ub[2] - ub[0]) * (ub[3] - ub[1]) / (r['width'] * r['height'])
            if 0 < ratio <= args.gqa_max_area:
                keep.append((r, ub, ratio))
        idx = rng.choice(len(keep), size=min(n, len(keep)), replace=False)
        out = []
        for i in sorted(idx):
            r, ub, ratio = keep[i]
            box = [1000 * ub[0] / r['width'], 1000 * ub[1] / r['height'], 1000 * ub[2] / r['width'],
                   1000 * ub[3] / r['height']]
            out.append(cr.Case(f'gqa_{i}', bench, r['question'] + SHORT, r['answer'], SHORT,
                               str(args.data_dir / 'GQA/images/images' / r['image']), box,
                               {'area_ratio': ratio}))
        return out
    if bench == 'mmvp':
        with open(args.mmvp_csv) as f:
            rows = list(csv.DictReader(f))
        out = []
        for r in rows[:n]:
            q = f"{r['Question']}\nOptions: {r['Options']}\n{MMVP_FMT}"
            out.append(cr.Case(f"mmvp_{r['Index']}", bench, q, r['Correct Answer'].strip('() ').lower(), MMVP_FMT,
                               str(args.data_dir / 'MMVP Images' / f"{r['Index']}.jpg"),
                               meta={'pair': (int(r['Index']) - 1) // 2}))
        return out
    if bench == 'pope':
        rows = read_jsonl(args.data_dir / 'POPE/adversarial.jsonl')
        yes = [r for r in rows if r['answer'] == 'yes']
        no = [r for r in rows if r['answer'] == 'no']
        pick = [yes[i] for i in rng.choice(len(yes), n // 2, replace=False)] + \
               [no[i] for i in rng.choice(len(no), n - n // 2, replace=False)]
        return [cr.Case(f"pope_{r['id']}", bench, r['question'] + ' ' + POPE_FMT, r['answer'], POPE_FMT,
                        str(args.data_dir / 'POPE/images/adversarial' / r['image'])) for r in pick]
    if bench == 'babyvision':
        rows = read_jsonl(args.data_dir / 'BabyVision/train.jsonl')
        rows = [r for r in rows if r['type'] == 'Fine-grained Discrimination'][:n]
        out = []
        for r in rows:
            q = r['question'].strip()
            if r['ansType'] == 'choice':
                q += '\nOptions:\n' + '\n'.join(f'{chr(65 + i)}. {o}' for i, o in enumerate(r['options']))
                fmt = BV_CHOICE_FMT
            else:
                fmt = BV_BLANK_FMT
            q += '\n' + fmt
            out.append(cr.Case(f"bv_{r['taskId']}", bench, q, None, fmt, str(args.data_dir / 'BabyVision' / r['image']),
                               meta={'row': {k: r[k] for k in ('taskId', 'question', 'ansType', 'options',
                                                               'choiceAns', 'blankAns', 'subtype')}}))
        return out
    if bench in SYNTH:
        kind = {'syn_color': ('color', 2), 'syn_gap': ('gap', 1)}.get(bench)
        if kind:
            cases = rp.det_cases(args.grid_n, kind[0], kind[1], n, args.seed + (1 if kind[0] == 'color' else 2))
        else:
            cases = rp.loc_cases(args.grid_n, 'T5', n, args.seed + 3)
        out = []
        for c in cases:
            g = args.grid_n * rp.CELL // rp.TOKEN
            if bench == 'syn_loc':
                r0, c0 = rp.random_cells_containing(tuple(c['target']), args.loc_k, args.grid_n, args.grid_n,
                                                    np.random.default_rng(c['seed']))
                win = rp.cell_window(r0, c0, args.loc_k)
                q = rp.T5_QUERY + ' Give its position. ' + LOC_FMT
                ev = c['target']
            else:
                r0, c0 = rp.centered_cells(tuple(c['asked']), 3, args.grid_n, args.grid_n)
                win = rp.cell_window(r0, c0, 3)
                q = rp.T3_REVISED.format(r=c['asked'][0] + 1, c=c['asked'][1] + 1) + ' ' + rp.ORDINAL_NOTE + ' ' + DET_FMT
                ev = c['asked']
            ct = rp.CELL // rp.TOKEN
            evbox = rp.Window(ev[0] * ct, ev[1] * ct, ct, ct).norm_box((g, g))
            fmt = LOC_FMT if bench == 'syn_loc' else DET_FMT
            out.append(cr.Case(c['id'], bench, q, c.get('answer_yes', c['target']), fmt, '', win.norm_box((g, g)),
                               {'synth': c, 'evidence_box': evbox}))
        return out
    raise ValueError(f'Unknown benchmark {bench!r}.')


def load_image(case: cr.Case, args) -> Image.Image:
    if case.bench in SYNTH:
        return rp.synth_image(case.meta['synth'])[0]
    im = Image.open(case.image_path).convert('RGB')
    budget = args.bv_max_pixels if case.bench == 'babyvision' else args.max_pixels
    return cr.resize_budget(im, budget)


def evidence_box(case: cr.Case) -> list[float] | None:
    """Unpadded evidence box (0-1000) used for the window-coverage metric."""
    return case.meta.get('evidence_box') or case.oracle_box


def oracle_window_box(case: cr.Case) -> list[float] | None:
    """Oracle window box: synthetic windows are given; GQA evidence boxes are doubled about their centre."""
    if case.oracle_box is None:
        return None
    if case.bench in SYNTH:
        return case.oracle_box
    x1, y1, x2, y2 = case.oracle_box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
    return [cx - w, cy - h, cx + w, cy + h]


# ----------------------------------------------------------------------------- scoring

def score(case: cr.Case, text: str, finish: str, s: float | None) -> dict:
    b = case.bench
    ans, parsed = cr.final_answer(text)
    if b == 'textvqa':
        return {'correct': cr.vqa_accuracy(ans, case.answer), 'pred': ans, 'parsed': parsed}
    if b == 'gqa':
        return {'correct': float(cr.exact_match(ans, case.answer)), 'pred': ans, 'parsed': parsed}
    if b == 'mmvp':
        p = cr.option_letter(ans)
        return {'correct': float(p == case.answer), 'pred': p, 'parsed': parsed}
    if b == 'pope':
        p = cr.yes_no_answer(ans)
        return {'correct': float(p == case.answer), 'pred': p, 'parsed': parsed}
    if b == 'babyvision':
        row = case.meta['row']
        st = bv_strict(row, text, 'babyvision')
        dg = diagnostic_score(row, text, 'length' if finish == 'length' else 'stop')
        return {'correct': float(st['correct']), 'correct_diag': float(dg['correct']), 'pred': st['prediction']}
    if b in ('syn_color', 'syn_gap'):
        p = cr.yes_no_answer(ans)
        return {'correct': float(p == ('yes' if case.answer else 'no')), 'pred': p, 's': s, 'parsed': parsed,
                'label': int(bool(case.answer))}
    if b == 'syn_loc':
        rc = rp.parse_rowcol(ans)
        out = {'correct': float(rc is not None and (rc[0] - 1, rc[1] - 1) == tuple(case.answer)),
               'pred': list(rc) if rc else None, 'parsed': parsed}
        return out
    raise ValueError(b)


def coverage(win: rp.Window, grid: tuple[int, int], box: list[float] | None) -> float | None:
    """Fraction of the evidence box area inside the window (0-1000 coordinates)."""
    if box is None:
        return None
    w = win.norm_box(grid)
    ix = max(0.0, min(w[2], box[2]) - max(w[0], box[0]))
    iy = max(0.0, min(w[3], box[3]) - max(w[1], box[1]))
    a = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy / a if a > 0 else None


def frame_error(case: cr.Case, pred, win: rp.Window, grid) -> bool:
    """syn_loc: a wrong answer equal to the target's row / column inside the window (window-local ordinal)."""
    if case.bench != 'syn_loc' or pred is None:
        return False
    ct = rp.CELL // rp.TOKEN
    local = (case.answer[0] - win.r0 // ct + 1, case.answer[1] - win.c0 // ct + 1)
    return tuple(pred) == local and tuple(pred) != (case.answer[0] + 1, case.answer[1] + 1)


# ----------------------------------------------------------------------------- per-case features

def needs(conds: list[str]) -> dict:
    cs = [cr.CONDITIONS[c] for c in conds]
    return {'block9': any(c.site == 'block_9' for c in cs if c.kind == 'cache'),
            'hr': any(c.kind == 'cache' and c.scale == 2 for c in cs)}


def features(rt, image: Image.Image, need: dict) -> dict:
    blocks = (9,) if need['block9'] else ()
    f1 = rp.encode(rt, [image], blocks)[0]
    out = {'f1': f1}
    if need['hr']:
        out['f2'] = rp.encode(rt, [cr.upscale(image, 2)], blocks)[0]
    return out


def second_segment(rt, cond: cr.Condition, image: Image.Image, feats: dict, win: rp.Window) -> tuple[cr.VSeg, dict]:
    """The tool's visual output for one window under one condition, with its cost accounting."""
    if cond.kind == 'crop':
        k = cr.crop_scale(win, cond.scale)
        im = cr.upscale(rp.crop(image, win), k)
        e = rp.encode(rt, [im])[0]
        return cr.VSeg(e['vis'], e['grid'], 'new'), {'crop_scale': k, 'vit_patches': 4 * e['grid'][0] * e['grid'][1]}
    f = feats['f1'] if cond.scale == 1 else feats['f2']
    s = cond.scale
    wh = rp.Window(win.r0 * s, win.c0 * s, win.h * s, win.w * s)
    gw = f['grid'][1]
    idx = torch.tensor(wh.indices(gw), device=f[cond.site].device)
    perm = None
    if cond.order == 'reverse':
        perm = torch.arange(len(idx) - 1, -1, -1)
        idx = idx[perm.to(idx.device)]
    seg = cr.VSeg(f[cond.site][idx], (wh.h, wh.w), cond.ids, (float(win.r0), float(win.c0)), 1.0 / s, perm)
    return seg, {'crop_scale': None, 'vit_patches': 0}


# ----------------------------------------------------------------------------- accuracy stage

def run_bench(rt, bench: str, out: Path, args) -> None:
    """Accuracy run of one benchmark. Round 1 (the first tool call) is generated once per case and shared by all
    tool conditions; later rounds may issue further calls (up to args.max_calls), executed by the condition's own
    tool, so later windows can depend on what the condition returned."""
    root = out / bench
    root.mkdir(parents=True, exist_ok=True)
    cases = load_cases(bench, args)
    conds = BASE_CONDS + (ORACLE_CONDS if bench in ORACLE_BENCHES else [])
    if args.conds:
        # an explicit list may also name supplementary conditions (e.g. h9r) outside the default set
        conds = [c for c in args.conds if c in cr.CONDITIONS and (bench in ORACLE_BENCHES or not cr.CONDITIONS[c].oracle)]
    path = root / 'results.jsonl'
    done = {(r['case_id'], r['cond']) for r in read_jsonl(path)}
    (root / 'cases.json').write_text(cr.dumps([{'id': c.id, 'question': c.question, 'answer': c.answer,
                                                'image_path': c.image_path, 'oracle_box': c.oracle_box,
                                                'meta': {k: v for k, v in c.meta.items() if k != 'row'}}
                                               for c in cases]))
    # Yes/No score s = z(' Yes') - z(' No') at the token after the last "Final answer:" marker (detail tasks).
    yn = (qr.token_id(rt, ' Yes'), qr.token_id(rt, ' No')) if bench in ('syn_color', 'syn_gap') else None
    stop_tool = cr.stop_ids(rt, True)
    max_new = args.max_new_bv if bench == 'babyvision' else args.max_new
    need = needs(conds)
    t_start = time.time()
    for ci in range(0, len(cases), args.chunk):
        chunk = [c for c in cases[ci:ci + args.chunk] if any((c.id, k) not in done for k in conds)]
        if not chunk:
            continue
        images = [load_image(c, args) for c in chunk]
        feats = [features(rt, im, need) for im in images]
        grids = [f['f1']['grid'] for f in feats]
        tool_conds = [k for k in conds if cr.CONDITIONS[k].kind in ('crop', 'cache') and not cr.CONDITIONS[k].oracle]
        s1 = [cr.stage1_ids(rt, c.question) for c in chunk]
        calls = [None] * len(chunk)
        if tool_conds:
            items1 = [cr.assemble(rt, ids, [cr.VSeg(f['f1']['vis'], f['f1']['grid'])]) for ids, f in zip(s1, feats)]
            calls = cr.run_batched(rt, items1, args.max_call, stop_tool, args.token_budget, args.max_batch)
            del items1
        states = []          # one per (case, condition) still to run
        for j, (c, im, f, g) in enumerate(zip(chunk, images, feats, grids)):
            for k in conds:
                if (c.id, k) in done:
                    continue
                cond = cr.CONDITIONS[k]
                st = {'case': c, 'j': j, 'cond': cond, 'row': {'case_id': c.id, 'bench': bench, 'grid': list(g),
                                                              'cond': k}, 'windows': [], 'calls': [], 'gen': [],
                      'vit_patches': 4 * g[0] * g[1], 'done': False}
                if cond.kind == 'base':
                    st['ids'], st['segs'] = cr.plain_ids(rt, c.question), [cr.VSeg(f['f1']['vis'], g)]
                elif cond.kind == 'full2x':
                    e = rp.encode(rt, [cr.upscale(im, 2)])[0]
                    st['ids'], st['segs'] = cr.plain_ids(rt, c.question), [cr.VSeg(e['vis'], e['grid'])]
                    st['vit_patches'] = 4 * e['grid'][0] * e['grid'][1]
                else:
                    if cond.kind == 'cache' and cond.scale == 2:
                        st['vit_patches'] += 16 * g[0] * g[1]
                    st['ids'], st['segs'] = list(s1[j]), [cr.VSeg(f['f1']['vis'], g)]
                    if cond.oracle:
                        box = oracle_window_box(c)
                        if box is None:
                            continue
                        call_ids = rt.tokenizer(
                            '<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n'
                            f'[{", ".join(str(round(v)) for v in box)}]\n</parameter>\n</function>\n</tool_call>',
                            add_special_tokens=False)['input_ids']
                        st['calls'].append({'text': None, 'bbox': box, 'oracle': True})
                    else:
                        call = calls[j]
                        box = cr.parse_call(call['text'])
                        call_ids = call['ids']
                        st['calls'].append({'text': call['text'], 'bbox': box, 'n_tokens': call['n_tokens'],
                                            'finish': call['finish_reason']})
                        if box is None:
                            # No valid first call: the round-1 output is this condition's final answer.
                            st['gen'] = call_ids
                            st.update(done=True, text=call['text'], finish=call['finish_reason'], no_call=True,
                                      item=cr.assemble(rt, st['ids'], st['segs']))
                            states.append(st)
                            continue
                    add_window(rt, st, im, f, box, call_ids)
                states.append(st)
        # rounds: answer, or call the tool again (up to max_calls calls in total)
        for rnd in range(args.max_calls):
            active = [st for st in states if not st['done']]
            if not active:
                break
            for st in active:
                st['item'] = cr.assemble(rt, st['ids'], st['segs'])
            res = cr.run_batched(rt, [st['item'] for st in active], max_new, stop_tool, args.token_budget,
                                 args.max_batch)
            for st, r in zip(active, res):
                st['batch_size'] = r['batch_size']
                box = cr.parse_call(r['text']) if r['finish_reason'] == 'stop' else None
                is_call = '<tool_call>' in r['text']
                if (is_call and box is not None and st['cond'].kind in ('crop', 'cache')
                        and len(st['calls']) < args.max_calls):
                    st['calls'].append({'text': r['text'], 'bbox': box, 'n_tokens': r['n_tokens'],
                                        'finish': r['finish_reason']})
                    add_window(rt, st, images[st['j']], feats[st['j']], box, r['ids'])
                else:
                    # 'call_limit': a complete call beyond the allowed number; a call cut by the token budget
                    # keeps finish_reason 'length'.
                    st.update(done=True, text=r['text'], gen=r['ids'],
                              finish='call_limit' if (is_call and box is not None) else r['finish_reason'])
        rows = {}
        marker_jobs = []
        for st in states:
            c = st['case']
            if not st['done']:                 # still calling after the last allowed round
                st.update(text='', finish='call_limit', gen=[])
            row = st['row']
            g = tuple(row['grid'])
            it = st['item']
            row.update(text=st['text'], finish_reason=st['finish'], n_calls=len(st['calls']), calls=st['calls'],
                       windows=[w.as_list() for w in st['windows']], no_call=bool(st.get('no_call')),
                       L=it['L'] + len(st['gen']), n_visual=it['n_visual'], n_extra_visual=it['n_extra_visual'],
                       vit_patches=st['vit_patches'], batch_size=st.get('batch_size'),
                       **score(c, st['text'], st['finish'], None))
            if st['windows']:
                w0 = st['windows'][0]
                row.update(window=w0.as_list(), window_box=w0.norm_box(g), coverage=coverage(w0, g, evidence_box(c)),
                           coverage_any=max(coverage(w, g, evidence_box(c)) or 0 for w in st['windows'])
                           if evidence_box(c) else None)
                row['frame_error'] = any(frame_error(c, row.get('pred'), w, g) for w in st['windows'])
            rows[(c.id, row['cond'])] = row
            marker_jobs.append(((c.id, row['cond']), it, st['gen']))
        if yn is not None and marker_jobs:
            ext, keys = [], []
            for key, it, gen in marker_jobs:
                cut = cr.marker_cut(rt, gen) if gen else None
                rows[key]['s'] = None
                if cut is not None:
                    ext.append(cr.extend_item(rt, it, gen[:cut]))
                    keys.append(key)
            if ext:
                for key, r in zip(keys, cr.run_batched(rt, ext, 1, stop_tool, args.token_budget, args.max_batch, yn)):
                    rows[key]['s'] = r['s']
        append_jsonl(path, list(rows.values()))
        del feats, states, marker_jobs
        torch.cuda.empty_cache()
        print(f'[{bench}] {min(ci + args.chunk, len(cases))}/{len(cases)} cases, {time.time() - t_start:.0f}s',
              flush=True)
    write_meta(root, rt, args, bench, {'conditions': conds, 'n_cases': len(cases), 'max_calls': args.max_calls,
                                       'token_budget': args.token_budget, 'max_batch': args.max_batch})


def add_window(rt, st: dict, image: Image.Image, feats: dict, box: list[float], call_ids: list[int]) -> None:
    """Execute one tool call for a state: append the call, the tool response and the condition's visual output."""
    g = feats['f1']['grid']
    win = cr.box_to_window(box, g)
    seg, cost = second_segment(rt, st['cond'], image, feats, win)
    st['ids'] = st['ids'] + list(call_ids) + cr.suffix_ids(rt, win.norm_box(g), st['case'].fmt)
    st['segs'].append(seg)
    st['windows'].append(win)
    st['vit_patches'] += cost['vit_patches']
    st.setdefault('crop_scales', []).append(cost['crop_scale'])


# ----------------------------------------------------------------------------- timing stage (batch size 1)

@torch.inference_mode()
def greedy_cached(rt, x: torch.Tensor, pos: torch.Tensor, cache, max_new: int, stops: set[int]):
    """Batch-size-1 greedy decoding that starts from an optional cache; returns (ids, cache)."""
    out = rt.model(inputs_embeds=x[None], position_ids=pos[:, None].to(rt.device), past_key_values=cache,
                   use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    nxt = pos.amax(-1).to(rt.device) + 1
    embed = rt.model.model.get_input_embeddings()
    ids = []
    logits = out.logits[:, -1]
    for step in range(max_new):
        tok = int(logits.argmax(-1))
        ids.append(tok)
        if tok in stops or step == max_new - 1:
            break
        out = rt.model(inputs_embeds=embed(torch.tensor([[tok]], device=rt.device)),
                       position_ids=(nxt + step)[:, None, None], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1]
    return ids, cache


def timed_case(rt, case: cr.Case, image: Image.Image, cond: cr.Condition, args) -> dict:
    """End-to-end batch-size-1 pipeline with KV-cache continuation across tool rounds (same rules as run_bench)."""
    stop_tool = cr.stop_ids(rt, True)
    max_new = args.max_new_bv if case.bench == 'babyvision' else args.max_new
    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    rec = {'case_id': case.id, 'bench': case.bench, 'cond': cond.name, 'n_calls': 0}
    if cond.kind in ('base', 'full2x'):
        e = rp.encode(rt, [cr.upscale(image, cond.scale if cond.kind == 'full2x' else 1)])[0]
        it = cr.assemble(rt, cr.plain_ids(rt, case.question), [cr.VSeg(e['vis'], e['grid'])])
        ids, cache = greedy_cached(rt, it['x'], it['pos'], None, max_new, stop_tool)
        L, extra = it['L'] + len(ids), 0
    else:
        f = features(rt, image, needs([cond.name]))
        g = f['f1']['grid']
        ids_all = cr.stage1_ids(rt, case.question)
        segs = [cr.VSeg(f['f1']['vis'], g)]
        it = cr.assemble(rt, ids_all, segs)
        gen, cache = greedy_cached(rt, it['x'], it['pos'], None, args.max_call, stop_tool)
        fed = it['L'] + len(gen) - 1                 # tokens already in the cache (the last generated one is not)
        extra = 0
        while True:
            box = cr.parse_call(rt.tokenizer.decode(gen))
            if box is None or rec['n_calls'] >= args.max_calls:
                L = it['L'] + len(gen)
                break
            rec['n_calls'] += 1
            win = cr.box_to_window(box, g)
            seg, _ = second_segment(rt, cond, image, f, win)
            ids_all = ids_all + gen + cr.suffix_ids(rt, win.norm_box(g), case.fmt)
            segs.append(seg)
            it = cr.assemble(rt, ids_all, segs)
            extra = it['n_extra_visual']
            gen, cache = greedy_cached(rt, it['x'][fed:], it['pos'][:, fed:], cache, max_new, stop_tool)
            fed = it['L'] + len(gen) - 1
    torch.cuda.synchronize()
    rec.update(seconds=time.perf_counter() - t0, peak_extra_bytes=torch.cuda.max_memory_allocated() - base_mem,
               L_final=L, n_extra_visual=extra, **cr.kv_bytes(rt.model.config, L))
    del cache
    return rec


def run_timing(rt, out: Path, args) -> None:
    root = out / 'timing'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'timing.jsonl'
    done = {(r['case_id'], r['cond']) for r in read_jsonl(path)}
    conds = TIMING_CONDS
    warm = True
    for bench in args.timing_benches:
        cases = load_cases(bench, args)[:args.timing_n]
        for c in cases:
            im = load_image(c, args)
            if warm:                                  # one untimed pass for kernel warm-up
                for k in conds:
                    timed_case(rt, c, im, cr.CONDITIONS[k], args)
                warm = False
            rows = []
            for k in conds:
                if (c.id, k) not in done:
                    rows.append(timed_case(rt, c, im, cr.CONDITIONS[k], args))
            append_jsonl(path, rows)
            torch.cuda.empty_cache()
        print(f'[timing] {bench} done', flush=True)
    write_meta(root, rt, args, 'timing', {'conditions': conds})


# ----------------------------------------------------------------------------- verification

@torch.inference_mode()
def run_verify(rt, out: Path, args) -> None:
    """(1) first-token logits of the custom path vs the processor path; (2) padded batch vs batch size 1."""
    root = out / 'verify'
    root.mkdir(parents=True, exist_ok=True)
    cases = load_cases('textvqa', args)[:args.verify_n] + load_cases('syn_loc', args)[:4]
    rep = {'logit_maxabs': [], 'batch_agree': [], 'batch_pairs': []}
    items = []
    for c in cases:
        im = load_image(c, args)
        batch = qr.build(rt, [im], c.question)
        ref = rt.model(**batch).logits[0, -1].float()
        e = rp.encode(rt, [im], (9,))[0]
        it = cr.assemble(rt, cr.plain_ids(rt, c.question), [cr.VSeg(e['vis'], e['grid'])])
        mine = rt.model(inputs_embeds=it['x'][None], position_ids=it['pos'][:, None].to(rt.device)).logits[0, -1].float()
        rep['logit_maxabs'].append(float((ref - mine).abs().max()))
        win = cr.box_to_window([300, 300, 600, 600], e['grid'])
        for k in ('crop2', 'h9', 'c9'):
            cond = cr.CONDITIONS[k]
            f = features(rt, im, needs([k]))
            seg, _ = second_segment(rt, cond, im, f, win)
            ids = cr.stage1_ids(rt, c.question) + rt.tokenizer(
                '<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n[300, 300, 600, 600]\n</parameter>\n'
                '</function>\n</tool_call>', add_special_tokens=False)['input_ids'] + cr.suffix_ids(rt, win.norm_box(e['grid']), c.fmt)
            items.append(cr.assemble(rt, ids, [cr.VSeg(f['f1']['vis'], f['f1']['grid']), seg]))
    stops = cr.stop_ids(rt, False)
    single = [cr.greedy_padded(rt, [it], args.max_new, stops)[0]['text'] for it in items]
    batched = [r['text'] for r in cr.run_batched(rt, items, args.max_new, stops, args.token_budget, args.max_batch)]
    rep['batch_agree'] = sum(a == b for a, b in zip(single, batched)) / len(items)
    rep['batch_pairs'] = list(zip(single, batched))
    rep['n_items'] = len(items)
    (root / 'verify.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ('logit_maxabs', 'batch_agree', 'n_items')}), flush=True)


# ----------------------------------------------------------------------------- main

def load_runtime(args) -> qr.Runtime:
    """Single-GPU load through the shared runtime, or a layer split over the visible GPUs when --max-memory
    gives per-GPU weight budgets in GiB (used when no single shared GPU has room)."""
    if not args.max_memory:
        return qr.load(args.model)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    ckpt = resolve_model(args.model, load_paths()['HF_HUB_CACHE'])
    mem = {i: f'{g}GiB' for i, g in enumerate(args.max_memory)}
    model = AutoModelForImageTextToText.from_pretrained(ckpt, local_files_only=True, dtype=torch.bfloat16,
                                                        device_map='auto', max_memory=mem,
                                                        attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    return qr.Runtime(args.model, ckpt, model, AutoProcessor.from_pretrained(ckpt, local_files_only=True), 'cuda:0')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', default='Qwen3.5-9B')
    ap.add_argument('--stages', required=True)
    ap.add_argument('--seed', type=int, default=20261020)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--vcot-gqa', type=Path, required=True)
    ap.add_argument('--mmvp-csv', type=Path, required=True)
    ap.add_argument('--n-natural', type=int, default=300)
    ap.add_argument('--n-bv', type=int, default=163)
    ap.add_argument('--n-synth', type=int, default=48)
    ap.add_argument('--grid-n', type=int, default=20)
    ap.add_argument('--loc-k', type=int, default=5)
    ap.add_argument('--gqa-max-area', type=float, default=0.02)
    ap.add_argument('--max-pixels', type=int, default=512 * 512)
    ap.add_argument('--bv-max-pixels', type=int, default=1024 * 1024)
    ap.add_argument('--max-call', type=int, default=256)
    ap.add_argument('--max-calls', type=int, default=3)
    ap.add_argument('--max-new', type=int, default=128)
    ap.add_argument('--max-new-bv', type=int, default=160)
    ap.add_argument('--chunk', type=int, default=16)
    ap.add_argument('--token-budget', type=int, default=24000)
    ap.add_argument('--max-batch', type=int, default=16)
    ap.add_argument('--conds', nargs='*', default=None)
    ap.add_argument('--timing-n', type=int, default=20)
    ap.add_argument('--timing-benches', nargs='*', default=['textvqa', 'gqa', 'syn_loc'])
    ap.add_argument('--verify-n', type=int, default=8)
    ap.add_argument('--max-memory', type=float, nargs='*', default=None)
    args = ap.parse_args()
    args.n = {b: args.n_natural for b in NATURAL} | {'babyvision': args.n_bv} | {b: args.n_synth for b in SYNTH}
    args.output.mkdir(parents=True, exist_ok=True)
    rt = load_runtime(args)
    for stage in args.stages.split(','):
        t = time.time()
        if stage == 'verify':
            run_verify(rt, args.output, args)
        elif stage == 'timing':
            run_timing(rt, args.output, args)
        else:
            run_bench(rt, stage, args.output, args)
        print(f'stage {stage} finished in {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
