"""Experiment 3.15 GPU stages (Qwen3.5-9B, bf16, SDPA, thinking off).

Stages
  answer    grid point items of experiment 1.7 (10x10, 64 images): forced answers after controlled assistant
            prefixes (no CoT, template index sentence, the model's own CoT up to its index statement, the full
            CoT), with index edits, image swaps and an answer-time image mask; records the answer, digit
            distributions and the image attention share of every full-attention head at the answer steps.
  gate      the same 64 images: graded weakening of the Gated DeltaNet decay at image tokens in layers 9-16
            (alpha -> alpha^s, s in GATE_S) for three prompts: direct row/column, direct point, think point.
  refcoco   RefCOCO test cases: direct and think point answers, then forced answers after no CoT / the full
            CoT, each with and without the answer-time image mask.
  spotcheck batched vs single-sequence greedy decoding on 20 prompts.

Forced answers run at batch size 1 with an explicit greedy loop: the prompt is prefilled up to (but not
including) the last assistant-prefix token; that token, the answer cue and the generated answer tokens are
then fed one step at a time, so that the answer-time image mask and the attention probe of experiment 1.7
(cot_mechanism_engine.Probe) act on every token from the end of the prefix onward. The mask adds -inf to the
image key positions in all 8 full-attention layers; the DeltaNet states are left untouched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.behavior import cot_mechanism as cm
from analysis_tools.behavior import cot_mechanism_engine as eng
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import ordinal_vs_coordinate as ov

COT_RUN = 'behavior/2026-10-21_cot_mechanism'        # experiment 1.7: point items and their think CoTs
R1_IMAGES = 'position/2026-10-16_readout_tool/images'  # RefCOCO test images used by 3.9 r2
R2_CASES = 'position/2026-10-20_readout_tool_r2/cases.jsonl'
N_ITEMS = 64
GATE_LAYERS = tuple(range(9, 17))
GATE_S = (1.0, 0.5, 0.25, 0.0)
CUE = cm.FORCE_CUE['point']                          # '\n\nFinal answer: [{"point_2d": ['
ANSWER_NEW = 14
RC_THINK = ('Locate the object described as: {target}. First describe step by step how you find this object and '
            'estimate the center of its bounding box. Then give the final answer on the last line as '
            '[{{"point_2d": [x, y]}}].' + cm.COORD_NOTE)
RC_DIRECT = ('Locate the object described as: {target}. Give the center of its bounding box. Return only '
             '[{{"point_2d": [x, y]}}].' + cm.COORD_NOTE + cm.JSON_ONLY)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def code_digest() -> dict:
    here = Path(__file__).parent
    return {p: hashlib.sha256((here / p).read_bytes()).hexdigest()[:16]
            for p in ('ordinal_vs_coordinate.py', 'ordinal_vs_coordinate_run.py')}


# ----------------------------------------------------------------------------- forced answers

def digit_ids(rt) -> list[int]:
    return [qr.token_id(rt, str(d)) for d in range(10)]


@torch.inference_mode()
def forced_answer(rt, image, question: str, prefix: str, mask_image: bool, max_new: int = ANSWER_NEW) -> dict:
    """Greedy answer after `prefix + CUE` with optional answer-time image masking (batch size 1).

    Returns the generated text, per-step digit log-probabilities (steps after the cue) and per-step image
    attention shares (layers x heads) for the last 3 cue steps and the generated steps.
    """
    full = eng.build_batch(rt, [image], [question], prefixes=[prefix + CUE])
    head = eng.build_batch(rt, [image], [question], prefixes=[prefix]) if prefix else \
        eng.build_batch(rt, [image], [question])
    ids, hid = full['input_ids'], head['input_ids']
    L = int(hid.shape[1])
    if not torch.equal(ids[0, :L - 1], hid[0, :L - 1]):
        common = int((ids[0, :L] == hid[0, :L]).cumprod(0).sum())
        L = common
    start = L - 1                                     # first fed token = last prefix token
    sp = eng.spans(rt, full)
    sp = dict(sp, P=start)                            # positions >= start count as 'generated' in the probe
    kw = {k: full[k] for k in ('pixel_values', 'image_grid_thw') if k in full}
    if 'mm_token_type_ids' in full:
        kw['mm_token_type_ids'] = full['mm_token_type_ids'][:, :start]
    out = rt.model(input_ids=ids[:, :start], use_cache=True, logits_to_keep=1, **kw)
    cache, delta = out.past_key_values, rt.model.model.rope_deltas.clone()
    forced = ids[0, start:].tolist()                  # last prefix token + cue tokens
    probe = eng.Probe(sp, record=True, mask_image=mask_image)
    dig = digit_ids(rt)
    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    gen, logp_digits = [], []
    cur = torch.tensor([[forced[0]]], device=ids.device)
    n_forced = len(forced)
    with eng.probing(probe):
        for t in range(n_forced - 1 + max_new):
            pos = torch.full((3, 1, 1), start + t, device=ids.device, dtype=torch.long) + delta.view(1, 1, 1)
            probe.new_step()
            logits = rt.model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True,
                              logits_to_keep=1).logits[0, -1].float()
            if t + 1 < n_forced:
                nxt = forced[t + 1]
            else:
                lp = logits.log_softmax(-1)
                logp_digits.append(lp[dig].tolist())
                nxt = int(logits.argmax())
                gen.append(nxt)
                if nxt in eos:
                    break
            cur = torch.tensor([[nxt]], device=ids.device)
        probe.finish()
    keep = probe.steps[max(0, n_forced - 4):]          # last 3 cue steps + generated steps
    att = [[[float(x) for x in st[l]['img'][0]] for l in eng.FULL_LAYERS] for st in keep]
    text = rt.tokenizer.decode(gen, skip_special_tokens=True)
    del cache
    return {'text': text, 'tokens': [rt.tokenizer.decode([g]) for g in gen], 'logp_digits': logp_digits,
            'img_att': att, 'n_cue_steps_kept': min(3, n_forced - 1), 'prompt_len': start}


def parse_xy(text: str):
    """x, y from the text generated after the cue '[{"point_2d": ['."""
    import re
    m = re.match(r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)', text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def grid_items(args) -> list[dict]:
    """1.7 point items with their partner B, truth and think CoT."""
    gen = {r['item']: r for r in read_jsonl(args.work / COT_RUN / 'gen/records.jsonl')
           if r['task'] == 'point' and r['mode'] == 'think'}
    out = []
    for it in cm.build_items('point', N_ITEMS):
        b = cm.pair_item(it)
        out.append({'item': it, 'partner': b, 'cot': gen[it.id]})
    return out


def answer_conditions(rec: dict) -> list[tuple[str, str, str]]:
    """(condition, image 'A'/'B', assistant prefix) for one grid item."""
    it, b, cot = rec['item'], rec['partner'], rec['cot']
    rA, cA = it.target[0] + 1, it.target[1] + 1
    rB, cB = b.target[0] + 1, b.target[1] + 1
    tA = ov.template_prefix(it.rows, it.cols, rA, cA)
    tB = ov.template_prefix(it.rows, it.cols, rB, cB)
    conds = [('f0', 'A', ''), ('f0_mask', 'A', ''), ('f0_imgB', 'B', ''),
             ('tplA', 'A', tA), ('tplA_mask', 'A', tA), ('tplB', 'A', tB), ('tplB_mask', 'A', tB), ('tplA_imgB', 'B', tA)]
    info = ov.index_prefix(cot['text'])
    if info is not None:
        conds += [('own', 'A', info['prefix']), ('own_mask', 'A', info['prefix']),
                  ('ownB', 'A', ov.edit_indices(info, rB, cB)), ('own_imgB', 'B', info['prefix'])]
        pre = ov.pre_index_prefix(cot['text'])
        if pre:
            conds += [('pre', 'A', pre.rstrip())]
    body = ov.full_body(cot['text'])
    if body is not None and cot['finish_reason'] == 'eos':
        conds += [('full', 'A', body), ('full_mask', 'A', body)]
    return conds


def stage_answer(rt, args) -> None:
    path = args.output / 'answer.jsonl'
    done = {(r['item'], r['condition']) for r in read_jsonl(path)}
    question = cm.prompt('point', 'think')
    t0 = time.time()
    for n, rec in enumerate(grid_items(args)):
        it, b = rec['item'], rec['partner']
        imgs = {'A': it.render()[0], 'B': b.render()[0]}
        rows = []
        for cond, which, prefix in answer_conditions(rec):
            if (it.id, cond) in done:
                continue
            r = forced_answer(rt, imgs[which], question, prefix, mask_image=cond.endswith('_mask'))
            rows.append(dict(r, item=it.id, partner=b.id, condition=cond, image=which, prefix=prefix,
                             xy=parse_xy(r['text']), truthA=cm.truth_of(it), truthB=cm.truth_of(b)))
        append_jsonl(path, rows)
        print(f'answer {n + 1}/{N_ITEMS} {time.time() - t0:.0f}s', flush=True)


CONT_CONDITIONS = ('tplA', 'tplB', 'tplA_imgB', 'ownB', 'own_imgB')
CONT_MAX_NEW = 400


def stage_cont(rt, args) -> None:
    """Free continuation (no answer cue) after the index prefixes: does the CoT compute with the written
    indices or re-read the image? Left-padded batches of the 1.7 engine (continue_batch)."""
    path = args.output / 'cont.jsonl'
    done = {(r['item'], r['condition']) for r in read_jsonl(path)}
    question = cm.prompt('point', 'think')
    jobs = []
    for rec in grid_items(args):
        it, b = rec['item'], rec['partner']
        for cond, which, prefix in answer_conditions(rec):
            if cond in CONT_CONDITIONS and (it.id, cond) not in done:
                jobs.append((it, b, cond, which, prefix))
    for k in range(0, len(jobs), args.batch_think):
        chunk = jobs[k:k + args.batch_think]
        t0 = time.time()
        items = [((it if which == 'A' else b).render()[0], question, prefix) for it, b, _, which, prefix in chunk]
        res = eng.continue_batch(rt, items, CONT_MAX_NEW)
        append_jsonl(path, [dict(r, item=it.id, partner=b.id, condition=cond, image=which, prefix=prefix,
                                 truthA=cm.truth_of(it), truthB=cm.truth_of(b), batch_size=len(chunk))
                            for (it, b, cond, which, prefix), r in zip(chunk, res)])
        print(f'cont {k + len(chunk)}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- graded decay

@contextmanager
def graded_gates(rt, s: float, positions: torch.Tensor, prompt_len: int):
    """alpha -> alpha^s at `positions` in the Gated DeltaNet layers of GATE_LAYERS (prefill only, all heads)."""
    handles = []
    if s != 1.0:
        for layer in GATE_LAYERS:
            block = rt.layers[layer - 1]
            if block.block_type != 'linear_attention':
                continue
            mod = block.linear_attn

            def hook(m, inp, out, mod=mod):
                if out.shape[1] != prompt_len:
                    return out
                out = out.clone()
                p = positions.to(out.device)
                out[:, p] = ov.graded_a(out[:, p], mod.dt_bias, s)
                return out
            handles.append(mod.in_proj_a.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


GATE_PROMPTS = {'rowcol_direct': (cm.prompt('rowcol', 'direct'), 24),
                'point_direct': (cm.prompt('point', 'direct'), 24),
                'point_think': (cm.prompt('point', 'think'), 1024)}


def stage_gate(rt, args) -> None:
    path = args.output / 'gate.jsonl'
    done = {(r['item'], r['prompt'], r['s']) for r in read_jsonl(path)}
    items = [r['item'] for r in grid_items(args)]
    images = {it.id: it.render()[0] for it in items}
    for pname, (text, max_new) in GATE_PROMPTS.items():
        probe_batch = qr.build(rt, [images[items[0].id]], text)
        L = int(probe_batch['input_ids'].shape[1])
        img = (probe_batch['input_ids'][0] == rt.model.config.image_token_id).nonzero().flatten()
        for s in GATE_S:
            todo = [it for it in items if (it.id, pname, s) not in done]
            bs = args.batch_think if pname.endswith('think') else args.batch_direct
            for k in range(0, len(todo), bs):
                chunk = todo[k:k + bs]
                t0 = time.time()
                with graded_gates(rt, s, img, L):
                    res = qr.generate_batch(rt, [([images[it.id]], text) for it in chunk], max_new)
                append_jsonl(path, [dict(r, item=it.id, prompt=pname, s=s, batch_size=len(chunk), truth=cm.truth_of(it))
                                    for it, r in zip(chunk, res)])
                print(f'gate {pname} s={s} {k + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- RefCOCO

def refcoco_cases(args) -> list[dict]:
    cases = [c for c in read_jsonl(args.work / R2_CASES)
             if c['set'] == 'refcoco' and c['split'] == 'test' and c['phrasing'] == 0]
    rng = np.random.default_rng(20261103)
    pick = sorted(rng.choice(len(cases), min(args.n_refcoco, len(cases)), replace=False))
    return [cases[i] for i in pick]


def rc_image(c: dict, args):
    from PIL import Image
    return Image.open(args.work / R1_IMAGES / c['image']).convert('RGB')


def stage_refcoco(rt, args) -> None:
    gpath, apath = args.output / 'refcoco_gen.jsonl', args.output / 'refcoco_answer.jsonl'
    cases = refcoco_cases(args)
    done = {(r['id'], r['mode']) for r in read_jsonl(gpath)}
    for mode, tpl, max_new, bs in (('direct', RC_DIRECT, 40, args.batch_direct), ('think', RC_THINK, 768, args.batch_think)):
        todo = [c for c in cases if (c['id'], mode) not in done]
        todo.sort(key=lambda c: tuple(c['truth']['canvas']))
        for k in range(0, len(todo), bs):
            chunk = todo[k:k + bs]
            t0 = time.time()
            res = qr.generate_batch(rt, [([rc_image(c, args)], tpl.format(target=c['target'])) for c in chunk], max_new)
            append_jsonl(gpath, [dict(r, id=c['id'], mode=mode, batch_size=len(chunk)) for c, r in zip(chunk, res)])
            print(f'refcoco {mode} {k + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    gen = {(r['id'], r['mode']): r for r in read_jsonl(gpath)}
    done = {(r['id'], r['condition']) for r in read_jsonl(apath)}
    for n, c in enumerate(cases):
        cot = gen[(c['id'], 'think')]
        body = ov.full_body(cot['text']) if cot['finish_reason'] == 'eos' else None
        conds = [('f0', ''), ('f0_mask', '')] + ([('full', body), ('full_mask', body)] if body else [])
        image, q = rc_image(c, args), RC_THINK.format(target=c['target'])
        rows = []
        for cond, prefix in conds:
            if (c['id'], cond) in done:
                continue
            r = forced_answer(rt, image, q, prefix, mask_image=cond.endswith('_mask'))
            rows.append(dict(r, id=c['id'], condition=cond, xy=parse_xy(r['text'])))
        append_jsonl(apath, rows)
        if n % 20 == 0:
            print(f'refcoco answer {n + 1}/{len(cases)}', flush=True)


# ----------------------------------------------------------------------------- spot check

def stage_spotcheck(rt, args) -> None:
    """Batched vs single greedy decoding: 10 grid point (direct) + 10 RefCOCO direct prompts."""
    items = [r['item'] for r in grid_items(args)][:10]
    cases = refcoco_cases(args)[:10]
    jobs = [([it.render()[0]], cm.prompt('point', 'direct')) for it in items] + \
           [([rc_image(c, args)], RC_DIRECT.format(target=c['target'])) for c in cases]
    batched = qr.generate_batch(rt, jobs[:10], 24) + qr.generate_batch(rt, jobs[10:], 40)
    single = [qr.generate(rt, qr.build(rt, ims, t), 40) for ims, t in jobs]
    rows = [{'batched': b['text'], 'single': s['text'],
             'same_answer': parse_xy(b['text'].split('[', 2)[-1]) == parse_xy(s['text'].split('[', 2)[-1])}
            for b, s in zip(batched, single)]
    (args.output / 'spotcheck.json').write_text(json.dumps({'n': len(rows), 'agree': sum(r['same_answer'] for r in rows),
                                                             'rows': rows}, indent=1))
    print('spotcheck agree', sum(r['same_answer'] for r in rows), '/', len(rows), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True, help='WORK_DIR/analysis')
    ap.add_argument('--model', default='Qwen3.5-9B')
    ap.add_argument('--n-refcoco', type=int, default=160)
    ap.add_argument('--batch-direct', type=int, default=16)
    ap.add_argument('--batch-think', type=int, default=16)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    meta = dict(qr.describe(rt), code=code_digest(), stages=args.stages, n_refcoco=args.n_refcoco,
                batch_direct=args.batch_direct, batch_think=args.batch_think, gate_s=GATE_S, gate_layers=GATE_LAYERS,
                started=time.strftime('%Y-%m-%d %H:%M:%S'))
    (args.output / f'run_{args.stages.replace(",", "_")}.json').write_text(json.dumps(meta, indent=1))
    for stage in args.stages.split(','):
        {'answer': stage_answer, 'cont': stage_cont, 'gate': stage_gate, 'refcoco': stage_refcoco, 'spotcheck': stage_spotcheck}[stage](rt, args)


if __name__ == '__main__':
    main()
