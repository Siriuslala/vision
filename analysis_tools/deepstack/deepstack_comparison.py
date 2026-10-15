"""Experiment 4.2: end-to-end comparison of DeepStack (Qwen3-VL) and non-DeepStack (Qwen3.5) models.

Design: docs/analysis/deepstack/plan.md, section 5.  Commands:

  decomp_prepare  -- per-model directory that links the experiment-1.3 stimuli and cases
  decomp_run      -- batched greedy answers to the experiment-1.3 main cases (resumable)
  spotcheck       -- re-answer a stratified subset at batch size 1 and compare with the batched run
  refcoco_build   -- area-stratified RefCOCO val sample (one expression per image)
  refcoco_run     -- batched greedy bbox answers on the RefCOCO sample (resumable)
  refcoco_score   -- IoU scoring of the RefCOCO answers

The 1.3 per-model directory holds exactly the files that ``decomposition.score`` and
``decomposition_analysis`` read, so scoring and per-model analysis reuse the 1.3 code unchanged.
BabyVision runs through ``eva.evaluation`` (see scripts/analysis/deepstack/deepstack_comparison.sh).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import (
    last_json_object, max_tokens, read_jsonl, sha256_file, write_jsonl,
)

MODELS = ('Qwen3-VL-8B-Instruct', 'Qwen3.5-9B', 'Qwen3-VL-4B-Instruct', 'Qwen3.5-4B')
LINKED = ('cases.jsonl', 'images', 'images.jsonl', 'config.json')
# Area strata of the RefCOCO target box, as a fraction of the image area (half-open intervals).
STRATA = (('lt1', 0.0, 0.01), ('1to5', 0.01, 0.05), ('5to10', 0.05, 0.10), ('ge10', 0.10, math.inf))
REFCOCO_PROMPT = ('Locate the object described as: "{expr}". Give its tight bounding box. '
                  'Return only [{{"bbox_2d": [x_min, y_min, x_max, y_max]}}]. All coordinates use 0-1000 '
                  'normalization; x increases rightward and y increases downward. '
                  'Output only the JSON, with no explanation.')


def slug(model: str) -> str:
    return model.lower().replace('.', '').replace('-instruct', '').replace('-', '_')


# ----------------------------------------------------------------------------- batched runner


def _first_step_processor():
    from transformers import LogitsProcessor

    class FirstStep(LogitsProcessor):
        """Keeps a float32 copy of the first-step logits of every row without changing them."""

        def __init__(self):
            self.first = None

        def __call__(self, input_ids, scores):
            if self.first is None:
                self.first = scores.float().clone()
            return scores
    return FirstStep()


def answer_batch(rt, images: list[Image.Image], prompts: list[str], budget: int,
                 yes_no_ids: tuple[int, int] | None) -> list[dict]:
    """Greedy, left-padded batched decoding that also returns first-token logits per row.

    Equivalent in output fields to the batch-size-1 runner of experiment 1.3; the geometry
    check guarantees that the processor kept the native image size.
    """
    import torch
    from transformers import LogitsProcessorList
    from analysis_tools.common import qwen_runtime as qr

    texts = [qr.prompt_text(rt, 1, p, False) for p in prompts]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist()):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError(f'Processor resized a {image.width}x{image.height} image to {w * 16}x{h * 16}.')
    first = _first_step_processor()
    tic = time.perf_counter()
    with torch.inference_mode():
        out = rt.model.generate(**batch, max_new_tokens=budget, do_sample=False, temperature=None, top_p=None,
                                top_k=None, logits_processor=LogitsProcessorList([first]))
    elapsed = time.perf_counter() - tic
    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = rt.tokenizer.pad_token_id
    prompt_lengths = batch['attention_mask'].sum(1).tolist()
    records = []
    for i, row in enumerate(out[:, batch['input_ids'].shape[1]:].tolist()):
        end = next((j + 1 for j, t in enumerate(row) if t in eos), None)
        new = row[:end] if end is not None else [t for t in row if t != pad]
        logits = first.first[i]
        rec = {'response': rt.tokenizer.decode(new, skip_special_tokens=True),
               'response_raw': rt.tokenizer.decode(new, skip_special_tokens=False),
               'n_tokens': len(new), 'finish_reason': 'eos' if end is not None else 'length',
               'max_new_tokens': budget, 'prompt_tokens': int(prompt_lengths[i]),
               'seconds': round(elapsed / len(images), 3), 'batch_size': len(images),
               'first_top_token': rt.tokenizer.decode([int(logits.argmax())])}
        if yes_no_ids is not None:
            z_yes, z_no = float(logits[yes_no_ids[0]]), float(logits[yes_no_ids[1]])
            rec.update(z_yes=z_yes, z_no=z_no, s=z_yes - z_no)
        records.append(rec)
    return records


def _bucketed(cases: list[dict], batch_size: int, cfg: dict, key) -> list[list[dict]]:
    """Batches of equal decoding budget and equal image size, so padding stays small."""
    groups: dict = {}
    for case in cases:
        groups.setdefault(key(case), []).append(case)
    batches = []
    for k in sorted(groups):
        rows = groups[k]
        batches += [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    return batches


def _check_provenance(path: Path, payload: dict) -> None:
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != payload:
            raise ValueError(f'Run configuration differs from {path}: {stored} vs {payload}')
    else:
        path.write_text(json.dumps(payload, indent=2) + '\n')


# ----------------------------------------------------------------------------- experiment 1.3 subtasks


def decomp_prepare(source: Path, target: Path) -> None:
    """Link the formal 1.3 stimuli and cases into a fresh per-model directory."""
    target.mkdir(parents=True, exist_ok=True)
    for name in LINKED:
        link, real = target / name, (source / name).resolve()
        if not real.exists():
            raise FileNotFoundError(real)
        if link.is_symlink() or link.exists():
            if link.resolve() != real:
                raise ValueError(f'{link} points to {link.resolve()}, expected {real}')
            continue
        link.symlink_to(real)
    print(f'linked {LINKED} into {target}')


def decomp_run(output: Path, cfg: dict, batch_size: int, limit: int | None) -> None:
    from analysis_tools.common import qwen_runtime as qr

    cases = [c for c in read_jsonl(output / 'cases.jsonl') if c['stage'] == 'main']
    results_path = output / 'results_main.jsonl'
    done = {r['id'] for r in read_jsonl(results_path)} if results_path.exists() else set()
    pending = [c for c in cases if c['id'] not in done]
    if limit is not None:
        pending = pending[:limit]
    print(f'main: {len(done)} done, {len(pending)} to run', flush=True)
    if not pending:
        return
    rt = qr.load(cfg['model'], device='cuda:0')
    run_cfg = {k: cfg[k] for k in ('model', 'max_new_tokens_binary', 'max_new_tokens_locate',
                                   'max_new_tokens_explicit', 'max_new_tokens_thinking')}
    _check_provenance(output / 'run_main.json', {'run': run_cfg, 'model': qr.describe(rt),
                                                 'runner': 'deepstack_comparison.decomp_run (batched greedy)',
                                                 'batch_size': 'per record and per attempt in run_log.jsonl'})
    ids = (qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No'))
    batches = _bucketed(pending, batch_size, cfg, key=lambda c: (max_tokens(c, cfg), c['grid'] if
                                                                   c['task'] not in ('T1', 'T2') else 0))
    start = dt.datetime.now().isoformat(timespec='seconds')
    n_done = 0
    with results_path.open('a') as sink:
        for batch in batches:
            images = []
            for case in batch:
                path = output / case['image']
                if sha256_file(path) != case['image_sha256']:
                    raise ValueError(f'Image hash mismatch for {case["image"]}')
                images.append(Image.open(path).convert('RGB'))
            budget = max_tokens(batch[0], cfg)
            binary = batch[0]['format'] == 'yes_no'
            records = answer_batch(rt, images, [c['prompt'] for c in batch], budget, ids if binary else None)
            for case, rec in zip(batch, records):
                sink.write(json.dumps({'id': case['id'], **rec}) + '\n')
            sink.flush()
            n_done += len(batch)
            if n_done // 400 != (n_done - len(batch)) // 400:
                print(f'main: {n_done}/{len(pending)} {dt.datetime.now():%H:%M:%S}', flush=True)
    with (output / 'run_log.jsonl').open('a') as log:
        log.write(json.dumps({'stage': 'main', 'start': start, 'n_run': n_done, 'batch_size': batch_size,
                              'end': dt.datetime.now().isoformat(timespec='seconds')}) + '\n')


def spotcheck(output: Path, cfg: dict, n: int, seed: int, kind: str) -> None:
    """Re-answer n results at batch size 1 (qwen_runtime.build + the same first-step capture) and compare.

    kind='decomp' compares against results_main.jsonl; kind='refcoco' against refcoco_results.jsonl.
    """
    import torch
    from transformers import LogitsProcessorList
    from analysis_tools.common import qwen_runtime as qr

    if kind == 'decomp':
        cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl') if c['stage'] == 'main'}
        results = read_jsonl(output / 'results_main.jsonl')
    else:
        cases = {c['id']: c for c in read_jsonl(output / 'refcoco_cases.jsonl')}
        results = read_jsonl(output / f'refcoco_results_{slug(cfg["model"])}.jsonl')
    by_stratum: dict = {}
    for r in results:
        c = cases[r['id']]
        by_stratum.setdefault((c.get('task'), c.get('format'), c.get('grid'), c.get('stratum')), []).append(r)
    rng = random.Random(seed)
    keys = sorted(by_stratum, key=str)
    picked = []
    while len(picked) < min(n, len(results)):
        for k in keys:
            pool = [r for r in by_stratum[k] if r not in picked]
            if pool and len(picked) < n:
                picked.append(rng.choice(pool))
    rt = qr.load(cfg['model'], device='cuda:0')
    ids = (qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No'))
    rows = []
    for r in picked:
        case = cases[r['id']]
        image = (Image.open(output / case['image']) if kind == 'decomp'
                 else Image.open(Path(cfg['refcoco_images']) / case['image'])).convert('RGB')
        batch = qr.build(rt, [image], case['prompt']) if kind == 'decomp' else _build_any(rt, image, case['prompt'])
        first = _first_step_processor()
        with torch.inference_mode():
            out = rt.model.generate(**batch, max_new_tokens=r['max_new_tokens'], do_sample=False, temperature=None,
                                    top_p=None, top_k=None, logits_processor=LogitsProcessorList([first]))
        text = rt.tokenizer.decode(out[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        row = {'id': r['id'], 'batched': r['response'], 'single': text, 'text_equal': text == r['response']}
        if 's' in r:
            logits = first.first[0]
            s = float(logits[ids[0]] - logits[ids[1]])
            row.update(s_batched=r['s'], s_single=s, sign_equal=(s > 0) == (r['s'] > 0))
        rows.append(row)
    name = f'spotcheck_{kind}_{slug(cfg["model"])}.jsonl'
    write_jsonl(output / name, rows)
    text_eq = sum(r['text_equal'] for r in rows)
    signs = [r['sign_equal'] for r in rows if 'sign_equal' in r]
    print(f'{name}: identical text {text_eq}/{len(rows)}; binary decision equal {sum(signs)}/{len(signs)}')


def t3c5_run(output: Path, supplement: Path, formal: Path, cfg: dict, batch_size: int) -> None:
    """Revised T3 wording C5 (1.3 supplement) on all 9216 formal T3 images; prefill only.

    The C5 question asks whether the designated figure differs, so Yes means "different";
    scoring uses s_same = -s as in the supplement.
    """
    from analysis_tools.common import qwen_runtime as qr

    cases = read_jsonl(supplement / 'cases_t3full.jsonl')
    if any(c['yes_means_same'] or c['image_root'] != 'formal' for c in cases):
        raise ValueError('Unexpected C5 case layout.')
    path = output / 'results_t3c5.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    pending = [c for c in cases if c['id'] not in done]
    print(f't3c5: {len(done)} done, {len(pending)} to run', flush=True)
    if not pending:
        return
    rt = qr.load(cfg['model'], device='cuda:0')
    _check_provenance(output / 'run_t3c5.json', {'model': qr.describe(rt), 'source': 'cases_t3full.jsonl',
                                                 'decision': 'prefill first-token logits; s_same = -(z_Yes - z_No)'})
    ids = (qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No'))
    with path.open('a') as sink:
        for batch in _bucketed(pending, batch_size, cfg, key=lambda c: c['grid']):
            images = []
            for c in batch:
                p = formal / c['image']
                if sha256_file(p) != c['image_sha256']:
                    raise ValueError(f'Image hash mismatch for {p}')
                images.append(Image.open(p).convert('RGB'))
            for c, rec in zip(batch, answer_batch(rt, images, [c['prompt'] for c in batch], 1, ids)):
                same = -rec['s']
                sink.write(json.dumps({'id': c['id'], 's': rec['s'], 'z_yes': rec['z_yes'], 'z_no': rec['z_no'],
                                       's_same': same, 'correct': (same > 0) == c['truth']['same'],
                                       'top_token': rec['first_top_token'], 'batch_size': len(batch)}) + '\n')
            sink.flush()


def batch_validate(source: Path, output: Path, cfg: dict, batch_size: int, per_cell: int, seed: int) -> None:
    """Answer a stratified subset of cases with the batched runner and compare with an existing
    batch-size-1 run of the same model (experiment 1.3 for Qwen3.5-9B).

    Strata are task x format x grid; agreement is reported as identical text and, for binary
    cases, an identical sign of s.
    """
    from analysis_tools.common import qwen_runtime as qr

    cases = {c['id']: c for c in read_jsonl(source / 'cases.jsonl') if c['stage'] == 'main'}
    reference = {r['id']: r for r in read_jsonl(source / 'results_main.jsonl')}
    strata: dict = {}
    for cid in sorted(reference):
        c = cases[cid]
        strata.setdefault((c['task'], c['format'], c['grid']), []).append(c)
    rng = random.Random(seed)
    picked = [c for k in sorted(strata) for c in rng.sample(strata[k], min(per_cell, len(strata[k])))]
    rt = qr.load(cfg['model'], device='cuda:0')
    ids = (qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No'))
    rows = []
    for batch in _bucketed(picked, batch_size, cfg, key=lambda c: (max_tokens(c, cfg), c['task'] in ('T1', 'T2'),
                                                                     c['grid'])):
        images = [Image.open(source / c['image']).convert('RGB') for c in batch]
        binary = batch[0]['format'] == 'yes_no'
        recs = answer_batch(rt, images, [c['prompt'] for c in batch], max_tokens(batch[0], cfg),
                            ids if binary else None)
        for c, rec in zip(batch, recs):
            ref = reference[c['id']]
            row = {'id': c['id'], 'task': c['task'], 'format': c['format'], 'grid': c['grid'],
                   'batched': rec['response'], 'single': ref['response'], 'text_equal': rec['response'] == ref['response']}
            if binary:
                row.update(s_batched=rec['s'], s_single=ref['s'], sign_equal=(rec['s'] > 0) == (ref['s'] > 0))
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    name = f'batch_validation_{slug(cfg["model"])}.jsonl'
    write_jsonl(output / name, rows)
    signs = [r['sign_equal'] for r in rows if 'sign_equal' in r]
    print(f'{name}: identical text {sum(r["text_equal"] for r in rows)}/{len(rows)}; '
          f'binary decision equal {sum(signs)}/{len(signs)}; batch size {batch_size}')


def bv_order(reference: Path, tail_from: int | None = None, rank_range: tuple[int, int] | None = None) -> None:
    """Print BabyVision taskIds sorted by the response length of a reference run (longest first).

    eva.evaluation keeps the given case order, so batches then group responses of similar length;
    a batch lasts as long as its longest response, and this ordering only changes throughput.
    """
    records = read_jsonl(reference)
    if len(records) != 388:
        raise ValueError(f'Reference run {reference} is incomplete.')
    ranked = sorted(records, key=lambda r: (-r['generated_tokens'], r['taskId']))
    if tail_from is not None:
        # Items ranked tail_from.. (0-based) in ascending length, for a second worker that starts
        # from the short end; the main worker is stopped once it reaches rank tail_from.
        ranked = ranked[tail_from:][::-1]
    elif rank_range is not None:
        # Extra worker on ranks [start, stop) in descending length, for a further split of a long run.
        ranked = ranked[rank_range[0]:rank_range[1]]
    print(' '.join(str(r['taskId']) for r in ranked))


def bv_merge(main: Path) -> None:
    """Merge the head worker (main file) and the tail worker (parts/<stem>.tail*.jsonl) of one model.

    The head files are moved to parts/<stem>.head.*; the merged predictions, a meta file listing
    both sources and a fresh summary are written at the main path, so each model keeps one run file.
    """
    from eva.benchmark import summarize

    parts = main.parent / 'parts'
    head = parts / f'{main.stem}.head.jsonl'
    if head.exists():
        raise FileExistsError(f'{head} exists; merge was already done.')
    others = sorted(p for p in parts.glob(f'{main.stem}.*.jsonl') if '.head.' not in p.name)
    if not others:
        raise ValueError(f'No parts for {main.name}')
    meta = json.loads(Path(str(main) + '.meta.json').read_text())
    for src in others:   # checked before any file is moved; GPU placement is not part of the run definition
        other = json.loads(Path(str(src) + '.meta.json').read_text())
        for key in set(meta['config']) | set(other['config']):
            if key not in ('selected_ids', 'gpu_ids') and meta['config'].get(key) != other['config'].get(key):
                raise ValueError(f'{src.name} differs in configuration field {key}.')
    for suffix in ('', '.meta.json', '.summary.json'):
        Path(str(main) + suffix).rename(Path(str(head) + suffix))
    records, counts = {}, {}
    for src in [head] + others:
        rows = read_jsonl(src)
        counts[src.name] = len(rows)
        for r in rows:
            if r['taskId'] in records:
                raise ValueError(f'taskId {r["taskId"]} appears in two parts.')
            records[r['taskId']] = r
    if len(records) != 388:
        raise ValueError(f'Merged run has {len(records)}/388 items.')
    merged = [records[k] for k in sorted(records)]
    write_jsonl(main, merged)
    meta['merged_from'] = counts
    Path(str(main) + '.meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    Path(str(main) + '.summary.json').write_text(json.dumps(summarize(merged), indent=2, ensure_ascii=False) + '\n')
    print(f'merged {counts} into {main.name}')


def bv_truncated(run: Path, n: int, seed: int, rank_range: tuple[int, int] | None = None) -> None:
    """Print a random sample of taskIds whose response hit the output budget (budget-extension check).

    ``rank_range`` selects a slice [a, b) of the sorted sample so that the check can be split over workers.
    """
    ids = sorted(r['taskId'] for r in read_jsonl(run) if r['finish_reason'] == 'length')
    picked = sorted(random.Random(seed).sample(ids, min(n, len(ids))))
    if rank_range is not None:
        picked = picked[rank_range[0]:rank_range[1]]
    print(' '.join(str(i) for i in picked))


def _build_any(rt, image: Image.Image, prompt: str):
    """Batch-size-1 inputs for natural images, which the processor may resize (no geometry check)."""
    from analysis_tools.common import qwen_runtime as qr
    return rt.processor(text=[qr.prompt_text(rt, 1, prompt, False)], images=[image],
                        return_tensors='pt').to(rt.device)


# ----------------------------------------------------------------------------- RefCOCO


def stratum_of(area: float) -> str:
    for name, lo, hi in STRATA:
        if lo <= area < hi:
            return name
    raise ValueError(f'Area fraction {area} outside all strata.')


def refcoco_build(refcoco: Path, output: Path, per_stratum: int, seed: int) -> None:
    """One expression per COCO image, sampled per area stratum from RefCOCO val.

    Images are written once to output/refcoco_images so that every model reads identical bytes.
    The first referring expression of each annotation is used, as in comparison_grounding.
    """
    import pyarrow.parquet as pq

    shards = sorted((refcoco / 'data').glob('val-*.parquet'))
    if not shards:
        raise FileNotFoundError('Missing RefCOCO val shards.')
    rows = []
    for shard in shards:
        table = pq.read_table(shard, columns=['question_id', 'answer', 'bbox', 'file_name', 'iscrowd', 'image'])
        for r in table.to_pylist():
            image = Image.open(io.BytesIO(r['image']['bytes']))
            width, height = image.size
            x, y, w, h = r['bbox']
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
                raise ValueError(f'Invalid RefCOCO bbox {r["bbox"]} in {r["file_name"]}')
            coco_image = r['file_name'].rsplit('_', 1)[0]
            rows.append({'question_id': r['question_id'], 'coco_image': coco_image, 'file_name': r['file_name'],
                         'expression': r['answer'][0], 'bbox': [x, y, x + w, y + h], 'width': width,
                         'height': height, 'area': w * h / (width * height), 'iscrowd': r['iscrowd'],
                         'bytes': r['image']['bytes']})
    population = {name: 0 for name, *_ in STRATA}
    for r in rows:
        r['stratum'] = stratum_of(r['area'])
        population[r['stratum']] += 1
    rng = random.Random(seed)
    rng.shuffle(rows)
    used, cases = set(), []
    image_dir = output / 'refcoco_images'
    image_dir.mkdir(parents=True, exist_ok=True)
    for name, *_ in STRATA:      # smallest stratum first, so scarce small objects keep their image
        taken = 0
        for r in (r for r in rows if r['stratum'] == name):
            if taken >= per_stratum or r['coco_image'] in used:
                continue
            used.add(r['coco_image'])
            rel = f'{r["question_id"]}.jpg'
            (image_dir / rel).write_bytes(r['bytes'])
            cases.append({'id': f'refcoco|{r["question_id"]}', 'image': rel,
                          'image_sha256': hashlib.sha256(r['bytes']).hexdigest(),
                          'prompt': REFCOCO_PROMPT.format(expr=r['expression']),
                          **{k: r[k] for k in ('question_id', 'coco_image', 'file_name', 'expression', 'bbox',
                                               'width', 'height', 'area', 'stratum')}})
            taken += 1
    write_jsonl(output / 'refcoco_cases.jsonl', cases)
    counts = {name: sum(c['stratum'] == name for c in cases) for name, *_ in STRATA}
    meta = {'source': 'lmms-lab RefCOCO val parquet', 'n_annotations': len(rows), 'population': population,
            'sampled': counts, 'per_stratum': per_stratum, 'seed': seed,
            'strata': [[n, lo, None if hi == math.inf else hi] for n, lo, hi in STRATA],
            'rule': 'one expression per COCO image; first referring expression; smallest stratum sampled first'}
    (output / 'refcoco_build.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta))


def refcoco_run(output: Path, cfg: dict, batch_size: int) -> None:
    from analysis_tools.common import qwen_runtime as qr

    cases = read_jsonl(output / 'refcoco_cases.jsonl')
    path = output / f'refcoco_results_{slug(cfg["model"])}.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    pending = [c for c in cases if c['id'] not in done]
    print(f'refcoco: {len(done)} done, {len(pending)} to run', flush=True)
    if not pending:
        return
    rt = qr.load(cfg['model'], device='cuda:0')
    _check_provenance(output / f'refcoco_run_{slug(cfg["model"])}.json',
                      {'model': qr.describe(rt), 'max_new_tokens': cfg['max_new_tokens_locate'],
                       'batch_size': batch_size, 'decoding': 'greedy', 'thinking': False,
                       'image_processor': {k: str(v) for k, v in rt.processor.image_processor.to_dict().items()
                                           if k in ('size', 'patch_size', 'merge_size')}})
    pending.sort(key=lambda c: (c['height'], c['width']))
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    with path.open('a') as sink:
        for batch in batches:
            images = []
            for c in batch:
                p = output / 'refcoco_images' / c['image']
                if sha256_file(p) != c['image_sha256']:
                    raise ValueError(f'Image hash mismatch for {p}')
                images.append(Image.open(p).convert('RGB'))
            for c, rec in zip(batch, _answer_natural(rt, images, [c['prompt'] for c in batch],
                                                       cfg['max_new_tokens_locate'])):
                sink.write(json.dumps({'id': c['id'], **rec}) + '\n')
            sink.flush()


def _answer_natural(rt, images, prompts, budget):
    """answer_batch without the native-size check: natural photos are resized by the processor."""
    import torch
    from analysis_tools.common import qwen_runtime as qr

    texts = [qr.prompt_text(rt, 1, p, False) for p in prompts]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    grids = batch['image_grid_thw'].tolist()
    with torch.inference_mode():
        out = rt.model.generate(**batch, max_new_tokens=budget, do_sample=False, temperature=None, top_p=None,
                                top_k=None)
    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = rt.tokenizer.pad_token_id
    records = []
    for i, row in enumerate(out[:, batch['input_ids'].shape[1]:].tolist()):
        end = next((j + 1 for j, t in enumerate(row) if t in eos), None)
        new = row[:end] if end is not None else [t for t in row if t != pad]
        records.append({'response': rt.tokenizer.decode(new, skip_special_tokens=True), 'n_tokens': len(new),
                        'finish_reason': 'eos' if end is not None else 'length', 'max_new_tokens': budget,
                        'batch_size': len(images), 'processed_hw': [grids[i][1] * 16, grids[i][2] * 16]})
    return records


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse_bbox(response: str) -> list[float] | None:
    """Last bbox_2d of a [{"bbox_2d": [...]}] answer; a bare dict or a bare 4-number list is also accepted."""
    try:
        value = last_json_object(response)
    except ValueError:
        bare = re.findall(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
                          r'\s*(-?\d+(?:\.\d+)?)\s*\]', response)
        return [float(v) for v in bare[-1]] if bare else None
    if isinstance(value, list) and value and isinstance(value[0], dict):
        value = value[0]
    if isinstance(value, dict):
        value = value.get('bbox_2d')
    if (isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                            for v in value)):
        return [float(v) for v in value]
    return None


def score_refcoco_case(case: dict, record: dict) -> dict:
    box = parse_bbox(record['response']) if record['finish_reason'] == 'eos' else None
    if box is None or box[2] <= box[0] or box[3] <= box[1]:
        return {'id': case['id'], 'valid': False, 'iou': 0.0, 'correct': False, 'center_in': False}
    w, h = case['width'], case['height']
    pred = [box[0] / 1000 * w, box[1] / 1000 * h, box[2] / 1000 * w, box[3] / 1000 * h]
    iou = box_iou(pred, case['bbox'])
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    g = case['bbox']
    return {'id': case['id'], 'valid': True, 'pred_box': pred, 'iou': iou, 'correct': iou >= 0.5,
            'center_in': g[0] <= cx <= g[2] and g[1] <= cy <= g[3]}


def refcoco_score(output: Path) -> None:
    cases = {c['id']: c for c in read_jsonl(output / 'refcoco_cases.jsonl')}
    for model in MODELS:
        path = output / f'refcoco_results_{slug(model)}.jsonl'
        if not path.exists():
            continue
        records = read_jsonl(path)
        if len({r['id'] for r in records}) != len(records):
            raise ValueError(f'Duplicate results in {path}')
        rows = [score_refcoco_case(cases[r['id']], r) for r in records]
        write_jsonl(output / f'refcoco_scores_{slug(model)}.jsonl', rows)
        print(f'{model}: {len(rows)}/{len(cases)} scored, IoU>=0.5 {np.mean([r["correct"] for r in rows]):.3f}')


# ----------------------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('bv_order', 'bv_merge', 'bv_truncated', 'decomp_prepare', 'decomp_run', 't3c5_run', 'spotcheck', 'batch_validate', 'refcoco_build',
                                       'refcoco_run', 'refcoco_score'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, help='Experiment 1.3 output directory (decomp_prepare).')
    p.add_argument('--refcoco', type=Path)
    p.add_argument('--supplement', type=Path, help='Experiment 1.3 supplement directory (t3c5_run).')
    p.add_argument('--model', default='Qwen3-VL-8B-Instruct', choices=MODELS)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--spotcheck-n', type=int, default=20)
    p.add_argument('--spotcheck-kind', choices=('decomp', 'refcoco'), default='decomp')
    p.add_argument('--per-stratum', type=int, default=300)
    p.add_argument('--rank-range', type=int, nargs=2, default=None, help='bv_order: ranks [start, stop).')
    p.add_argument('--tail-from', type=int, default=None, help='bv_order: second-worker ranks from this index.')
    p.add_argument('--per-cell', type=int, default=8, help='Cases per task x format x grid (batch_validate).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-new-tokens-binary', type=int, default=8)
    p.add_argument('--max-new-tokens-locate', type=int, default=256)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch size must be positive')
    # Reasoning-stage budgets are unused here but keep the 1.3 run-configuration schema.
    cfg = {'model': args.model, 'max_new_tokens_binary': args.max_new_tokens_binary,
           'max_new_tokens_locate': args.max_new_tokens_locate, 'max_new_tokens_explicit': 2048,
           'max_new_tokens_thinking': 8192, 'refcoco_images': str(args.output / 'refcoco_images')}
    if args.command == 'bv_truncated':
        bv_truncated(args.source, args.spotcheck_n, args.seed, tuple(args.rank_range) if args.rank_range else None)
    elif args.command == 'bv_merge':
        bv_merge(args.source)
    elif args.command == 'bv_order':
        bv_order(args.source, args.tail_from, tuple(args.rank_range) if args.rank_range else None)
    elif args.command == 'decomp_prepare':
        decomp_prepare(args.source, args.output)
    elif args.command == 'decomp_run':
        decomp_run(args.output, cfg, args.batch_size, args.limit)
    elif args.command == 'spotcheck':
        spotcheck(args.output, cfg, args.spotcheck_n, args.seed, args.spotcheck_kind)
    elif args.command == 't3c5_run':
        t3c5_run(args.output, args.supplement, args.source, cfg, args.batch_size)
    elif args.command == 'batch_validate':
        batch_validate(args.source, args.output, cfg, args.batch_size, args.per_cell, args.seed)
    elif args.command == 'refcoco_build':
        refcoco_build(args.refcoco, args.output, args.per_stratum, args.seed)
    elif args.command == 'refcoco_run':
        refcoco_run(args.output, cfg, args.batch_size)
    else:
        refcoco_score(args.output)


if __name__ == '__main__':
    main()
