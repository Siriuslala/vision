"""Experiment 4.3 (light): DeepStack ablation inside frozen Qwen3-VL models.

Design: docs/analysis/deepstack/plan.md, section 6.  Only the DeepStack features added to the
outputs of decoder layers 1-3 at visual-token positions are changed; the main-path visual tokens,
all weights, prompts, and greedy decoding stay identical across conditions.

Conditions (``CONDITIONS``):
  full        D_k unchanged (baseline)
  off         every D_k replaced by zeros
  off1..off3  only D_k of extraction layer k replaced by zeros
  mean        every D_k replaced by its reference-set mean token, broadcast to all visual tokens
  blur        every D_k recomputed from a Gaussian-blurred copy of the same image (sigma 2 pixels)

Commands:
  build      -- stratified case subset (experiment-1.3 stimuli and the 4.2 RefCOCO sample) and the
                reference image list used by the mean condition
  means      -- per-model reference means of the DeepStack features
  verify     -- confirms on one input that each condition changes decoder layers 1-3 as intended
  run        -- batched greedy answers for one model and one condition (resumable)
  spotcheck  -- re-answers a subset at batch size 1 and compares with the batched run
  score      -- experiment-1.3 diagnostic scoring and RefCOCO IoU scoring of every run
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageFilter

from analysis_tools.behavior.decomposition import (
    max_tokens, read_jsonl, score_case, sha256_file, write_jsonl,
)
from analysis_tools.deepstack.deepstack_comparison import (
    _answer_natural, _bucketed, _check_provenance, answer_batch, score_refcoco_case, slug,
)

CONDITIONS = ('full', 'off', 'blur', 'mean', 'off1', 'off2', 'off3')
BLUR_SIGMA = 2.0                      # pixels at the input resolution, as in plan section 6.1
# Case filters of the light run (plan section 6.3, updated from the 4.2 results).
BINARY_LEVELS = (2, 3)                # colour Delta E 10 and 20
LOCATE_GRIDS = (10, 20)
LOCATE_FORMATS = ('point', 'bbox', 'row_column')
T5_RECORDS_PER_CELL = 8               # per grid x kind x level; the same images are asked in all three formats


# ----------------------------------------------------------------------------- DeepStack override


class DeepstackOverride:
    """Replaces the DeepStack features passed from Qwen3VLModel to its language model.

    The wrapper intercepts the ``deepstack_visual_embeds`` argument of
    ``model.model.language_model.forward`` (a list with one tensor of shape (n_visual_tokens, d) per
    extraction layer, concatenated over the images of the batch).  The argument is only present in
    the prefill pass, so cached decoding steps are untouched.  ``calls`` counts the replaced prefill
    passes so that callers can assert the override actually fired.
    """

    def __init__(self, rt, condition: str, means: list | None = None):
        if condition not in CONDITIONS:
            raise ValueError(f'Unknown condition {condition!r}')
        if condition == 'mean' and means is None:
            raise ValueError('The mean condition needs reference means.')
        self.rt, self.condition, self.means = rt, condition, means
        self.blur_features = None    # set per batch for the blur condition
        self.calls = 0

    def replace(self, embeds: list):
        import torch
        n_layers = len(self.rt.model.model.visual.deepstack_visual_indexes)
        if len(embeds) != n_layers:
            raise ValueError(f'Expected {n_layers} DeepStack tensors, got {len(embeds)}')
        self.calls += 1
        c = self.condition
        if c == 'full':
            return embeds
        if c == 'off':
            return [torch.zeros_like(e) for e in embeds]
        if c.startswith('off'):
            k = int(c[3:]) - 1
            return [torch.zeros_like(e) if i == k else e for i, e in enumerate(embeds)]
        if c == 'mean':
            return [m.to(e.device, e.dtype).expand_as(e).clone() for m, e in zip(self.means, embeds)]
        if self.blur_features is None:
            raise ValueError('Blur features were not prepared for this batch.')
        out = [b.to(e.device, e.dtype) for b, e in zip(self.blur_features, embeds)]
        if any(b.shape != e.shape for b, e in zip(out, embeds)):
            raise ValueError('Blurred DeepStack features do not match the visual tokens of the batch.')
        self.blur_features = None    # one prefill per prepared batch
        return out


@contextmanager
def deepstack_override(rt, override: DeepstackOverride):
    lm = rt.model.model.language_model
    original = lm.forward

    def forward(*args, **kwargs):
        if kwargs.get('deepstack_visual_embeds') is not None:
            kwargs['deepstack_visual_embeds'] = override.replace(kwargs['deepstack_visual_embeds'])
        return original(*args, **kwargs)

    lm.forward = forward
    try:
        yield override
    finally:
        del lm.forward               # restore the class method


def blurred(image: Image.Image, sigma: float = BLUR_SIGMA) -> Image.Image:
    # Pillow's GaussianBlur radius is the standard deviation of the kernel.
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def deepstack_features(rt, images: list[Image.Image]) -> list:
    """DeepStack features of the images (list over extraction layers of (n_tokens, d)) and the grids."""
    import torch
    proc = rt.processor.image_processor(images=images, return_tensors='pt')
    with torch.inference_mode():
        out = rt.model.model.get_image_features(proc['pixel_values'].to(rt.device),
                                                proc['image_grid_thw'].to(rt.device), return_dict=True)
    return list(out.deepstack_features), proc['image_grid_thw'].tolist()


def prepare_blur(rt, override: DeepstackOverride, images: list[Image.Image]) -> None:
    if override.condition != 'blur':
        return
    feats, grids = deepstack_features(rt, [blurred(im) for im in images])
    ref = rt.processor.image_processor(images=images, return_tensors='pt')['image_grid_thw'].tolist()
    if grids != ref:
        raise ValueError('Blurring changed the processed image grid.')
    override.blur_features = feats


# ----------------------------------------------------------------------------- build


def _select_decomp(cases: list[dict], seed: int) -> list[dict]:
    main = [c for c in cases if c['stage'] == 'main' and c['wording'] == 'primary']
    chosen = [c for c in main if c['task'] in ('T1', 'T2') and c['kind'] == 'color' and c['level'] in BINARY_LEVELS]
    chosen += [c for c in main if c['task'] == 'T6' and c['grid'] in LOCATE_GRIDS and c['format'] in LOCATE_FORMATS]
    t5 = [c for c in main if c['task'] == 'T5' and c['grid'] in LOCATE_GRIDS and c['format'] in LOCATE_FORMATS]
    rng = random.Random(seed)
    records: dict = {}
    for c in t5:
        records.setdefault((c['grid'], c['kind'], c['level']), set()).add(c['record_id'])
    keep = set()
    for key in sorted(records):
        pool = sorted(records[key])
        if len(pool) < T5_RECORDS_PER_CELL:
            raise ValueError(f'Only {len(pool)} T5 records in cell {key}')
        keep |= set(rng.sample(pool, T5_RECORDS_PER_CELL))
    chosen += [c for c in t5 if c['record_id'] in keep]
    for c in chosen:
        c['suite'] = 'decomp'
    return chosen


def build(output: Path, decomp_source: Path, refcoco_source: Path, refcoco_dir: Path, n_ref_synth: int,
          n_ref_natural: int, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, real in (('images', decomp_source / 'images'), ('refcoco_images', refcoco_source / 'refcoco_images')):
        link, real = output / name, real.resolve()
        if not real.exists():
            raise FileNotFoundError(real)
        if link.is_symlink() or link.exists():
            if link.resolve() != real:
                raise ValueError(f'{link} points to {link.resolve()}, expected {real}')
        else:
            link.symlink_to(real)
    decomp = _select_decomp(read_jsonl(decomp_source / 'cases.jsonl'), seed)
    refcoco = read_jsonl(refcoco_source / 'refcoco_cases.jsonl')
    for c in refcoco:
        c.update(suite='refcoco', task='RefCOCO', format='bbox')
    cases = decomp + refcoco
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case ids')
    cases_path = output / 'cases.jsonl'
    if cases_path.exists() and read_jsonl(cases_path) != cases:
        raise ValueError(f'{cases_path} exists with a different selection')
    write_jsonl(cases_path, cases)

    # Reference set for the mean condition, disjoint from every evaluated image.
    used = {c['image'] for c in decomp}
    images = read_jsonl(decomp_source / 'images.jsonl')
    def kind_of(im):
        change = (im.get('spec') or {}).get('change')
        return change['kind'] if change else None
    # Small grids of every kind, plus single and pair images of the non-colour kinds.
    synth = sorted({im['path'] for im in images if im['path'] not in used
                    and ((im['role'] == 'grid' and im.get('rows') in (3, 6))
                         or (im['role'].startswith(('t1_', 't2_')) and kind_of(im) in ('gap', 'rotate')))})
    rng = random.Random(seed + 1)
    if len(synth) < n_ref_synth:
        raise ValueError(f'Only {len(synth)} synthetic reference candidates')
    synth = rng.sample(synth, n_ref_synth)
    import pyarrow.parquet as pq
    table = pq.read_table(refcoco_dir / 'data' / 'testA-00000-of-00001.parquet', columns=['file_name', 'image'])
    by_image: dict = {}
    for r in table.to_pylist():
        by_image.setdefault(r['file_name'].rsplit('_', 1)[0], r['image']['bytes'])
    eval_images = {c['coco_image'] for c in refcoco}
    pool = sorted(k for k in by_image if k not in eval_images)
    natural = rng.sample(pool, n_ref_natural)
    ref_dir = output / 'reference_images'
    ref_dir.mkdir(exist_ok=True)
    for name in natural:
        path = ref_dir / f'{name}.jpg'
        if not path.exists():
            path.write_bytes(by_image[name])
    reference = [{'source': 'synthetic', 'image': s, 'sha256': sha256_file(output / s)} for s in synth]
    reference += [{'source': 'refcoco_testA', 'image': f'reference_images/{n}.jpg',
                   'sha256': sha256_file(ref_dir / f'{n}.jpg')} for n in natural]
    write_jsonl(output / 'reference.jsonl', reference)
    summary: dict = {}
    for c in cases:
        key = f"{c['task']}|{c.get('format')}|{c.get('grid', '-')}"
        summary[key] = summary.get(key, 0) + 1
    _check_provenance(output / 'build.json', {
        'seed': seed, 'decomp_source': decomp_source.name, 'refcoco_source': refcoco_source.name,
        'binary_levels': list(BINARY_LEVELS), 'locate_grids': list(LOCATE_GRIDS),
        't5_records_per_cell': T5_RECORDS_PER_CELL, 'n_cases': len(cases), 'cells': dict(sorted(summary.items())),
        'reference': {'n_synthetic': n_ref_synth, 'n_refcoco_testA': n_ref_natural,
                      'sha256': hashlib.sha256(json.dumps(reference).encode()).hexdigest()}})
    print(f'{len(cases)} cases; reference {len(reference)} images')
    for k, v in sorted(summary.items()):
        print(f'  {k}: {v}')


# ----------------------------------------------------------------------------- means / verify


def compute_means(output: Path, model: str) -> None:
    import torch
    from analysis_tools.common import qwen_runtime as qr
    reference = read_jsonl(output / 'reference.jsonl')
    rt = qr.load(model, device='cuda:0')
    sums, count = None, 0
    for item in reference:
        path = output / item['image']
        if sha256_file(path) != item['sha256']:
            raise ValueError(f'Reference image hash mismatch: {path}')
        feats, _ = deepstack_features(rt, [Image.open(path).convert('RGB')])
        s = [f.double().sum(0).cpu() for f in feats]
        sums = s if sums is None else [a + b for a, b in zip(sums, s)]
        count += feats[0].shape[0]
    means = [(x / count).float() for x in sums]
    target = output / f'means_{slug(model)}.pt'
    torch.save({'means': means, 'n_tokens': count, 'n_images': len(reference),
                'reference_sha256': hashlib.sha256((output / 'reference.jsonl').read_bytes()).hexdigest(),
                'model': {k: str(v) for k, v in qr.describe(rt).items()}}, target)
    print(f'{model}: means over {count} tokens of {len(reference)} images; norms',
          [round(float(m.norm()), 2) for m in means])


def load_means(output: Path, model: str):
    import torch
    path = output / f'means_{slug(model)}.pt'
    if not path.exists():
        raise FileNotFoundError(f'{path}; run COMMAND=means first')
    payload = torch.load(path)
    if payload['reference_sha256'] != hashlib.sha256((output / 'reference.jsonl').read_bytes()).hexdigest():
        raise ValueError('Reference set changed after the means were computed.')
    return payload['means']


def verify(output: Path, model: str) -> None:
    """Relative change of decoder layer 1-3 outputs (visual positions) and of the first-token logits."""
    import torch
    from analysis_tools.common import qwen_runtime as qr
    cases = read_jsonl(output / 'cases.jsonl')
    case = next(c for c in cases if c['task'] == 'T6' and c['format'] == 'point')
    rt = qr.load(model, device='cuda:0')
    means = load_means(output, model)
    image = Image.open(output / case['image']).convert('RGB')
    batch = qr.build(rt, [image], case['prompt'])
    span = qr.image_spans(rt, batch)[0]
    positions = torch.arange(span['start'], span['end'])
    store = {}
    for cond in CONDITIONS:
        override = DeepstackOverride(rt, cond, means)
        prepare_blur(rt, override, [image])
        with deepstack_override(rt, override), qr.capture(rt, [1, 2, 3, 4], positions) as cap, torch.inference_mode():
            logits = rt.model(**batch).logits[0, -1].float().cpu()
        if override.calls != 1:
            raise RuntimeError(f'{cond}: override fired {override.calls} times')
        store[cond] = ({k: v.clone() for k, v in cap.items()}, logits)
    rows = []
    for cond in CONDITIONS:
        hs, logits = store[cond]
        base_h, base_l = store['full']
        row = {'condition': cond}
        for layer in (1, 2, 3, 4):
            row[f'layer{layer}'] = float((hs[layer] - base_h[layer]).norm() / base_h[layer].norm())
        row['logits_max_abs'] = float((logits - base_l).abs().max())
        rows.append(row)
        print(json.dumps(row))
    (output / 'checks').mkdir(exist_ok=True)
    write_jsonl(output / 'checks' / f'verify_{slug(model)}.jsonl', rows)


# ----------------------------------------------------------------------------- run


def run(output: Path, model: str, condition: str, tasks: list[str], batch_size: int, cfg: dict) -> None:
    from analysis_tools.common import qwen_runtime as qr
    cases = [c for c in read_jsonl(output / 'cases.jsonl') if c['task'] in tasks]
    run_dir = output / 'runs' / slug(model)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f'{condition}.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    pending = [c for c in cases if c['id'] not in done]
    print(f'{model} {condition}: {len(done)} done, {len(pending)} to run', flush=True)
    if not pending:
        return
    rt = qr.load(model, device='cuda:0')
    means = load_means(output, model) if condition == 'mean' else None
    _check_provenance(run_dir / f'{condition}.json', {
        'model': qr.describe(rt), 'condition': condition, 'blur_sigma': BLUR_SIGMA, 'batch_size': 'per record and per attempt in run_log.jsonl',
        'max_new_tokens_binary': cfg['max_new_tokens_binary'], 'max_new_tokens_locate': cfg['max_new_tokens_locate'],
        'decoding': 'greedy', 'thinking': False,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]})
    override = DeepstackOverride(rt, condition, means)
    ids = (qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No'))
    decomp = [c for c in pending if c['suite'] == 'decomp']
    natural = sorted([c for c in pending if c['suite'] == 'refcoco'], key=lambda c: (c['height'], c['width']))
    batches = _bucketed(decomp, batch_size, cfg, key=lambda c: (max_tokens(c, cfg), c['task'],
                                                                 c['grid'] if c['task'] not in ('T1', 'T2') else 0))
    batches += [natural[i:i + batch_size] for i in range(0, len(natural), batch_size)]
    start, n_done = dt.datetime.now().isoformat(timespec='seconds'), 0
    with path.open('a') as sink, deepstack_override(rt, override):
        for batch in batches:
            images = []
            for c in batch:
                p = output / (c['image'] if c['suite'] == 'decomp' else f"refcoco_images/{c['image']}")
                if sha256_file(p) != c['image_sha256']:
                    raise ValueError(f'Image hash mismatch for {p}')
                images.append(Image.open(p).convert('RGB'))
            prepare_blur(rt, override, images)
            calls = override.calls
            if batch[0]['suite'] == 'refcoco':
                records = _answer_natural(rt, images, [c['prompt'] for c in batch], cfg['max_new_tokens_locate'])
            else:
                binary = batch[0]['format'] == 'yes_no'
                records = answer_batch(rt, images, [c['prompt'] for c in batch], max_tokens(batch[0], cfg),
                                       ids if binary else None)
            if override.calls != calls + 1:
                raise RuntimeError(f'DeepStack override fired {override.calls - calls} times in one batch')
            for c, rec in zip(batch, records):
                sink.write(json.dumps({'id': c['id'], 'condition': condition, **rec}) + '\n')
            sink.flush()
            n_done += len(batch)
            if n_done // 400 != (n_done - len(batch)) // 400:
                print(f'{condition}: {n_done}/{len(pending)} {dt.datetime.now():%H:%M:%S}', flush=True)
    with (run_dir / 'run_log.jsonl').open('a') as log:
        log.write(json.dumps({'condition': condition, 'start': start, 'n_run': n_done, 'batch_size': batch_size,
                              'end': dt.datetime.now().isoformat(timespec='seconds')}) + '\n')


def spotcheck(output: Path, model: str, conditions: list[str], n: int, seed: int, cfg: dict) -> None:
    """Batch-size-1 answers (qwen_runtime.build + generate) under the same override, compared by score."""
    import torch
    from analysis_tools.common import qwen_runtime as qr
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    rt = qr.load(model, device='cuda:0')
    rows = []
    for cond in conditions:
        path = output / 'runs' / slug(model) / f'{cond}.jsonl'
        records = [r for r in read_jsonl(path) if cases[r['id']]['suite'] == 'decomp']
        sample = random.Random(seed).sample(records, n)
        override = DeepstackOverride(rt, cond, load_means(output, model) if cond == 'mean' else None)
        with deepstack_override(rt, override):
            for rec in sample:
                case = cases[rec['id']]
                image = Image.open(output / case['image']).convert('RGB')
                prepare_blur(rt, override, [image])
                batch = qr.build(rt, [image], case['prompt'])
                if case['format'] == 'yes_no':
                    with torch.inference_mode():
                        logits = rt.model(**batch).logits[0, -1].float()
                    s = float(logits[qr.token_id(rt, 'Yes')] - logits[qr.token_id(rt, 'No')])
                    single = {'response': rec['response'], 'finish_reason': 'eos', 's': s}
                else:
                    g = qr.generate(rt, batch, max_tokens(case, cfg))
                    single = {'response': g['text'], 'finish_reason': g['finish_reason']}
                a, b = score_case(case, rec, True), score_case(case, single, True)
                key = 'hit' if case['format'] == 'point' else 'correct'
                rows.append({'condition': cond, 'id': case['id'], 'text_equal': single['response'] == rec['response'],
                             'answer_equal': a.get(key) == b.get(key), 'batched': rec['response'],
                             'single': single['response'], 's_batched': rec.get('s'), 's_single': single.get('s')})
    (output / 'checks').mkdir(exist_ok=True)
    write_jsonl(output / 'checks' / f'spotcheck_{slug(model)}.jsonl', rows)
    for cond in conditions:
        sub = [r for r in rows if r['condition'] == cond]
        print(f'{cond}: text equal {sum(r["text_equal"] for r in sub)}/{len(sub)}, '
              f'answer equal {sum(r["answer_equal"] for r in sub)}/{len(sub)}')


# ----------------------------------------------------------------------------- score


def score(output: Path) -> None:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    for path in sorted((output / 'runs').glob('*/*.jsonl')):
        if path.name == 'run_log.jsonl' or path.name.endswith('.scores.jsonl'):
            continue
        records = read_jsonl(path)
        if len({r['id'] for r in records}) != len(records):
            raise ValueError(f'Duplicate results in {path}')
        rows = [score_refcoco_case(cases[r['id']], r) if cases[r['id']]['suite'] == 'refcoco'
                else {'id': r['id'], **score_case(cases[r['id']], r, True)} for r in records]
        write_jsonl(path.parent / f'{path.stem}.scores.jsonl', rows)
        print(f'{path.parent.name}/{path.stem}: {len(rows)} scored')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('build', 'means', 'verify', 'run', 'spotcheck', 'score'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen3-VL-8B-Instruct')
    p.add_argument('--condition', default='full', choices=CONDITIONS)
    p.add_argument('--conditions', nargs='+', default=['full', 'off'])
    p.add_argument('--tasks', nargs='+', default=['T1', 'T2', 'T5', 'T6', 'RefCOCO'])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-new-tokens-binary', type=int, default=8)
    p.add_argument('--max-new-tokens-locate', type=int, default=256)
    p.add_argument('--decomp-source', type=Path)
    p.add_argument('--refcoco-source', type=Path)
    p.add_argument('--refcoco-dir', type=Path)
    p.add_argument('--n-ref-synthetic', type=int, default=250)
    p.add_argument('--n-ref-natural', type=int, default=250)
    p.add_argument('--spotcheck-n', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    cfg = {'max_new_tokens_binary': a.max_new_tokens_binary, 'max_new_tokens_locate': a.max_new_tokens_locate,
           'max_new_tokens_explicit': None, 'max_new_tokens_thinking': None}
    if a.command == 'build':
        build(a.output, a.decomp_source, a.refcoco_source, a.refcoco_dir, a.n_ref_synthetic, a.n_ref_natural, a.seed)
    elif a.command == 'means':
        compute_means(a.output, a.model)
    elif a.command == 'verify':
        verify(a.output, a.model)
    elif a.command == 'run':
        run(a.output, a.model, a.condition, a.tasks, a.batch_size, cfg)
    elif a.command == 'spotcheck':
        spotcheck(a.output, a.model, a.conditions, a.spotcheck_n, a.seed, cfg)
    else:
        score(a.output)


if __name__ == '__main__':
    main()
