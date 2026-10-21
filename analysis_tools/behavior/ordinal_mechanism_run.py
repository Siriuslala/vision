"""GPU stages of experiment 1.5 (ordinal mechanism): generation, spot-check, feature capture and steering.

  gen      -- batched greedy answers for the direct or the think-then-answer cases (resumable, shardable)
  spot     -- re-run a random subset at batch size 1 with qwen_runtime.generate
  capture  -- teacher-forced residual streams at the answer roles (see ROLES) on even layers 0..32
  steer    -- add encoding directions (coordinate, count, random, 3.6 coordinate direction) to the state that
              predicts the long-axis ordinal and continue the number greedily

Teacher forcing re-tokenizes the model's own response; generation used the same tokenizer, so the
prefix reproduces the generated context up to tokenizer round-trip differences (checked per case).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.behavior import ordinal_mechanism as om
from analysis_tools.behavior.decomposition import read_jsonl, sha256_file, write_jsonl
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import coordinate_value as cv

LAYERS = tuple(range(0, 33, 2))            # 0 = input embedding, l = output of decoder layer l (1-based)
ROLES = ('last', 'row0', 'col0', 'vis')    # last prompt token, states predicting the first row / column digit,
                                           # mean over the target cell's merged visual tokens
STEER_LAYERS = (16, 20, 24, 28)
MAX_TOKENS = {'direct': 64, 'think': 1536}


def load_image(case: dict, output: Path) -> Image.Image:
    path = output / case['image']
    if sha256_file(path) != case['image_sha256']:
        raise ValueError(f'Image hash mismatch for {path}')
    return Image.open(path).convert('RGB')


def results(output: Path, mode: str) -> dict[str, dict]:
    out = {}
    for p in sorted((output / 'gen').glob(f'results_{mode}*.jsonl')):
        for r in read_jsonl(p):
            if r['id'] in out:
                raise ValueError(f'Duplicate result {r["id"]}')
            out[r['id']] = r
    return out


# ----------------------------------------------------------------------------- generation

def stage_gen(rt, output: Path, mode: str, batch_size: int, shard: tuple[int, int], limit: int | None) -> None:
    cases = [c for c in read_jsonl(output / 'cases.jsonl') if c['mode'] == mode]
    done = set(results(output, mode))
    i0, k = shard
    pending = [c for j, c in enumerate(cases) if j % k == i0 and c['id'] not in done][:limit]
    print(f'gen {mode} shard {i0}/{k}: {len(done)} done, {len(pending)} pending', flush=True)
    pending.sort(key=lambda c: (tuple(c['truth']['canvas']), c['id']))
    (output / 'gen').mkdir(exist_ok=True)
    sink_path = output / 'gen' / f'results_{mode}{"" if k == 1 else f"_shard{i0}of{k}"}.jsonl'
    start, n_done, i = dt.datetime.now().isoformat(timespec='seconds'), 0, 0
    with sink_path.open('a') as sink:
        while i < len(pending):
            size = tuple(pending[i]['truth']['canvas'])
            chunk = [pending[i]]
            while len(chunk) < batch_size and i + len(chunk) < len(pending) \
                    and tuple(pending[i + len(chunk)]['truth']['canvas']) == size:
                chunk.append(pending[i + len(chunk)])
            tic = time.perf_counter()
            outs = qr.generate_batch(rt, [([load_image(c, output)], c['prompt']) for c in chunk], MAX_TOKENS[mode])
            sec = (time.perf_counter() - tic) / len(chunk)
            for c, o in zip(chunk, outs):
                sink.write(json.dumps({'id': c['id'], 'response': o['text'], 'n_tokens': o['n_tokens'],
                                       'finish_reason': o['finish_reason'], 'batch_size': len(chunk),
                                       'seconds': round(sec, 3)}) + '\n')
            sink.flush()
            i += len(chunk)
            n_done += len(chunk)
            if n_done // 160 != (n_done - len(chunk)) // 160:
                print(f'{mode} {n_done}/{len(pending)} {sec:.2f}s/case', flush=True)
    with (output / 'gen' / 'run_log.jsonl').open('a') as log:
        log.write(json.dumps({'stage': 'gen', 'mode': mode, 'shard': list(shard), 'batch_size': batch_size,
                              'max_new_tokens': MAX_TOKENS[mode], 'start': start, 'n_run': n_done,
                              'end': dt.datetime.now().isoformat(timespec='seconds'),
                              'provenance': qr.describe(rt)}) + '\n')


def stage_spot(rt, output: Path, n: int) -> None:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    rows = []
    for mode, m in (('direct', n), ('think', max(n // 5, 1))):
        res = results(output, mode)
        ids = sorted(res)
        if not ids:
            continue
        for cid in [ids[j] for j in np.random.default_rng(0).choice(len(ids), min(m, len(ids)), replace=False)]:
            case = cases[cid]
            o = qr.generate(rt, qr.build(rt, [load_image(case, output)], case['prompt']), MAX_TOKENS[mode])
            rows.append({'id': cid, 'mode': mode, 'batched': res[cid]['response'], 'single': o['text'],
                         'answer_batched': om.parse_answer(res[cid]['response']),
                         'answer_single': om.parse_answer(o['text'])})
    write_jsonl(output / 'gen' / 'spotcheck.jsonl', rows)
    for mode in sorted({r['mode'] for r in rows}):
        sub = [r for r in rows if r['mode'] == mode]
        print(f"spot {mode}: answer agreement {sum(r['answer_batched'] == r['answer_single'] for r in sub)}/{len(sub)}, "
              f"identical text {sum(r['batched'].strip() == r['single'].strip() for r in sub)}/{len(sub)}")


# ----------------------------------------------------------------------------- answer token positions

def rowcol_digit_tokens(rt, ids: list[int]) -> tuple[list[int], list[int]] | None:
    """Answer-token indices of the row and column digits of the LAST {"row": r, "column": c} in the answer."""
    text = rt.tokenizer.decode(ids, skip_special_tokens=False)
    found = list(om.ROWCOL.finditer(text))
    if not found:
        return None
    m = found[-1]
    ends = []
    # Cumulative decoding keeps character offsets exact for byte-level BPE pieces.
    for i in range(len(ids)):
        ends.append(len(rt.tokenizer.decode(ids[:i + 1], skip_special_tokens=False)))
    starts = [0] + ends[:-1]
    out = []
    for g in (1, 2):
        a, b = m.span(g)
        toks = [i for i in range(len(ids)) if starts[i] >= a and ends[i] <= b]
        if not toks or ''.join(rt.tokenizer.decode([ids[i]]) for i in toks) != text[a:b]:
            return None
        out.append(toks)
    return out[0], out[1]


# ----------------------------------------------------------------------------- capture

@torch.inference_mode()
def capture_one(rt, prep: cr.Prepared, answer_ids: list[int], index: list[int], n_vis: int) -> np.ndarray:
    """(len(LAYERS), len(ROLES), D) float16 states; the last role averages the n_vis trailing positions."""
    x, pos = cr.teacher_forced(rt, prep, answer_ids)
    idx = torch.tensor(index, device=x.device)
    store = {}
    handles = []

    def put(layer, h):
        v = h[0, idx].float()
        store[layer] = torch.cat([v[:len(ROLES) - 1], v[len(ROLES) - 1:].mean(0, keepdim=True)], 0).cpu()

    if 0 in LAYERS:
        handles.append(rt.layers[0].register_forward_pre_hook(
            lambda m, a, kw: put(0, kw.get('hidden_states', a[0] if a else None)), with_kwargs=True))
    for l in LAYERS:
        if l > 0:
            handles.append(rt.layers[l - 1].register_forward_hook(
                lambda m, a, o, l=l: put(l, o[0] if isinstance(o, tuple) else o)))
    try:
        rt.model(inputs_embeds=x, position_ids=pos, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    if n_vis < 1:
        raise ValueError('target covers no visual token')
    return torch.stack([store[l] for l in LAYERS]).half().numpy()


def stage_capture(rt, output: Path, mode: str) -> None:
    cases = [c for c in read_jsonl(output / 'cases.jsonl') if c['mode'] == mode]
    res = results(output, mode)
    root = output / 'features'
    root.mkdir(exist_ok=True)
    for cond in sorted({c['condition'] for c in cases}):
        path = root / f'{cond}_{mode}.npz'
        if path.exists():
            print('exists', path.name)
            continue
        sub = [c for c in cases if c['condition'] == cond and c['id'] in res]
        ids, feats, status, tic = [], [], [], time.time()
        for c in sub:
            r = res[c['id']]
            answer_ids = rt.tokenizer.encode(r['response'], add_special_tokens=False)
            spans = rowcol_digit_tokens(rt, answer_ids) if r['finish_reason'] == 'eos' else None
            if spans is None:
                status.append({'id': c['id'], 'status': 'no_answer_tokens'})
                continue
            prep = cr.prepare(rt, load_image(c, output), c['prompt'])
            L = len(prep.ids)
            vis = [prep.span['start'] + t for t in cr.tokens_overlapping(c['truth']['target_cell_box'], prep.grid)]
            index = [L - 1, L + spans[0][0] - 1, L + spans[1][0] - 1] + vis
            feats.append(capture_one(rt, prep, answer_ids, index, len(vis)))
            ids.append(c['id'])
            status.append({'id': c['id'], 'status': 'ok', 'n_vis': len(vis), 'answer_tokens': len(answer_ids)})
        np.savez(path, ids=np.array(ids), H=np.stack(feats), layers=np.array(LAYERS), roles=np.array(ROLES))
        write_jsonl(root / f'{cond}_{mode}_status.jsonl', status)
        print(f'capture {cond} {mode}: {len(ids)}/{len(sub)} in {time.time() - tic:.0f}s', flush=True)


# ----------------------------------------------------------------------------- steering

def stage_steer(rt, output: Path, mode: str, batch_size: int, every: int = 1) -> None:
    """Continue the long-axis ordinal from its prefix with one steering vector added at one layer."""
    dirs = np.load(output / 'steer' / f'directions_{mode}.npz', allow_pickle=True)
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    res = results(output, mode)
    path = output / 'steer' / f'records_{mode}.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    targets = [str(i) for i in dirs['ids']]
    # every k-th base of each condition (base index divisible by k); the direction file keeps all bases
    chosen = [cid for cid in targets if cases[cid]['base'] % every == 0]
    print(f'steer {mode}: {len(chosen)} of {len(targets)} cases, {len(done)} done', flush=True)
    tic = time.time()
    with path.open('a') as sink:
        for n, cid in enumerate(chosen):
            if cid in done:
                continue
            case, r = cases[cid], res[cid]
            answer_ids = rt.tokenizer.encode(r['response'], add_special_tokens=False)
            spans = rowcol_digit_tokens(rt, answer_ids)
            long_span = spans[1] if case['axis'] == 'H' else spans[0]
            prep = cr.prepare(rt, load_image(case, output), case['prompt'])
            x, pos = cv.prefix_inputs(rt, prep, answer_ids[:long_span[0]])
            j = targets.index(cid)
            out = {'id': cid, 'condition': case['condition'], 'c': case['c'], 'base_answer': None, 'runs': []}
            for li, layer in enumerate(STEER_LAYERS):
                vecs = dirs['vectors'][j, li]                     # (K, D)
                labels = [str(s) for s in dirs['labels']]
                steer = cv.Steerer(rt, layer)
                try:
                    for i in range(0, len(labels), batch_size):
                        d = torch.from_numpy(vecs[i:i + batch_size]).float().to(rt.device)[:, None]
                        B = d.shape[0]
                        rr = cv.continue_number(rt, x.expand(B, -1, -1).contiguous(),
                                                pos.expand(-1, B, -1).contiguous(), steer, d)
                        for lab, o in zip(labels[i:i + batch_size], rr):
                            out['runs'].append({'layer': layer, 'label': lab,
                                                'value': int(o['digits']) if o['digits'] else None,
                                                'stop': o['stop']})
                finally:
                    steer.close()
            sink.write(json.dumps(out) + '\n')
            sink.flush()
            if n % 20 == 0:
                print(f'steer {mode} {n}/{len(chosen)} {time.time() - tic:.0f}s', flush=True)


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('command', choices=['gen', 'spot', 'capture', 'steer'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', default='Qwen3.5-9B')
    ap.add_argument('--mode', choices=om.MODES, default='direct')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--spot-n', type=int, default=20)
    ap.add_argument('--every', type=int, default=1, help='steer: use bases whose index is divisible by this')
    a = ap.parse_args()
    rt = qr.load(a.model, device='cuda:0')
    shard = tuple(int(v) for v in a.shard.split('/'))
    if a.command == 'gen':
        stage_gen(rt, a.output, a.mode, a.batch_size, shard, a.limit)
    elif a.command == 'spot':
        stage_spot(rt, a.output, a.spot_n)
    elif a.command == 'capture':
        stage_capture(rt, a.output, a.mode)
    else:
        stage_steer(rt, a.output, a.mode, a.batch_size, a.every)


if __name__ == '__main__':
    main()
