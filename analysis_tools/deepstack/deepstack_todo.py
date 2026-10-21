"""Experiment 4.4: DeepStack follow-ups (plan section 10) inside frozen Qwen3-VL models.

Design: docs/analysis/deepstack/results_2026-10-20_todo.md, section 2.  Three questions:

  A. Blur sweep and content-free injection -- at which spatial frequency does the coordinate damage of
     removing DeepStack start, and does a content-free injection (DeepStack features of a uniform image)
     already suffice?  ``blankflip`` mirrors that content-free injection left-right at the token level, a
     causal test of whether the decoder reads absolute position from the injected features.
  B. Non-coordinate control -- POPE (object existence, yes/no), TextVQA (OCR, fine detail) and BabyVision
     (direct answer) under the same conditions.
  C. Representation -- per-token linear readability of position and local pixel detail in ViT blocks,
     DeepStack features, main-path tokens and decoder layers of Qwen3-VL-8B (Full and Off) and Qwen3.5-9B.

Conditions (every one changes only the DeepStack features added after decoder layers 1-3):
  full, off, mean          as in experiment 4.3 (``deepstack_ablation``)
  blur<s>                  D_k from a Gaussian-blurred copy (standard deviation s pixels)
  blank                    D_k from a uniform image of the same size filled with the image's mean colour
  blankflip                ``blank`` with the token grid of every D_k mirrored left-right

Commands: build | verify | run | represent | score
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageFilter

from analysis_tools.behavior.decomposition import max_tokens, read_jsonl, score_case, sha256_file, write_jsonl
from analysis_tools.deepstack.deepstack_ablation import deepstack_features, deepstack_override, load_means
from analysis_tools.deepstack.deepstack_comparison import _check_provenance, score_refcoco_case, slug

BLUR_SIGMAS = (2, 4, 8, 16)
CONDITIONS = ('full', 'off', 'mean', *(f'blur{s}' for s in BLUR_SIGMAS), 'blank', 'blankflip')
COORD_TASKS = ('T5', 'T6', 'RefCOCO')
GENERAL_TASKS = ('POPE', 'TextVQA', 'BabyVision')
BUDGET = {'POPE': 8, 'TextVQA': 16, 'BabyVision': 48}
POPE_N, TEXTVQA_N = 500, 300
REP_N = {'synthetic': 90, 'refcoco': 90, 'babyvision': 90}
REP_LONG_SIDE = 640            # natural representation images are resized so both sides are multiples of 32
REP_TOKENS = 64                # stored tokens per image (48 content tokens when available + 16 others)
REP_VIEW = 4                   # images per source whose every token is stored, for reconstruction figures
VIT_BLOCKS = (9, 17, 25, 27)   # 1-based; 9/17/25 are the DeepStack extraction blocks 8/16/24 (0-based)
DEC_LAYERS = {'qwen3_vl': (1, 2, 3, 4, 8, 16, 24, 36), 'qwen3_5': (1, 2, 3, 4, 8, 16, 24, 32)}
REP_PROMPT = 'Describe the image.'
HIGHPASS_SIGMA = 2.0
TOKEN_PIXELS = 32


# ----------------------------------------------------------------------------- conditions


def blur_sigma(condition: str) -> float | None:
    return float(condition[4:]) if condition.startswith('blur') else None


def blank_like(image: Image.Image) -> Image.Image:
    mean = tuple(int(round(v)) for v in np.asarray(image.convert('RGB'), dtype=np.float64).mean((0, 1)))
    return Image.new('RGB', image.size, mean)


def flip_tokens(features, grids: list[list[int]]):
    """Mirror each image's merged token grid left-right; ``features`` is (sum of tokens, d) in raster order."""
    import torch
    if sum((h // 2) * (w // 2) for _, h, w in grids) != features.shape[0]:
        raise ValueError('Token count does not match the image grids.')
    parts, start = [], 0
    for _, h, w in grids:
        rows, cols = h // 2, w // 2
        block = features[start:start + rows * cols]
        parts.append(block.reshape(rows, cols, -1).flip(1).reshape(rows * cols, -1))
        start += rows * cols
    return torch.cat(parts, 0)


class TodoOverride:
    """Replacement of the DeepStack features for one batch, following the condition table above.

    For image-derived conditions ``prepare`` must be called once per batch; ``replace`` consumes the
    prepared features so a stale batch can never be reused.
    """

    def __init__(self, rt, condition: str, means: list | None = None):
        if condition not in CONDITIONS:
            raise ValueError(f'Unknown condition {condition!r}')
        if condition == 'mean' and means is None:
            raise ValueError('The mean condition needs reference means.')
        self.rt, self.condition, self.means = rt, condition, means
        self.prepared = None
        self.calls = 0

    @property
    def image_derived(self) -> bool:
        return self.condition.startswith(('blur', 'blank'))

    def prepare(self, images: list[Image.Image]) -> None:
        if not self.image_derived:
            return
        c = self.condition
        sources = ([im.filter(ImageFilter.GaussianBlur(radius=blur_sigma(c))) for im in images]
                   if c.startswith('blur') else [blank_like(im) for im in images])
        feats, grids = deepstack_features(self.rt, sources)
        ref = self.rt.processor.image_processor(images=images, return_tensors='pt')['image_grid_thw'].tolist()
        if grids != ref:
            raise ValueError('The replacement images changed the processed grid.')
        if c == 'blankflip':
            feats = [flip_tokens(f, grids) for f in feats]
        self.prepared = feats

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
        if c == 'mean':
            return [m.to(e.device, e.dtype).expand_as(e).clone() for m, e in zip(self.means, embeds)]
        if self.prepared is None:
            raise ValueError('Replacement features were not prepared for this batch.')
        out = [p.to(e.device, e.dtype) for p, e in zip(self.prepared, embeds)]
        if any(p.shape != e.shape for p, e in zip(out, embeds)):
            raise ValueError('Prepared DeepStack features do not match the visual tokens of the batch.')
        self.prepared = None
        return out


# ----------------------------------------------------------------------------- build


def _resize32(image: Image.Image, long_side: int) -> Image.Image:
    scale = long_side / max(image.size)
    w = max(64, int(round(image.width * scale / 32)) * 32)
    h = max(64, int(round(image.height * scale / 32)) * 32)
    return image.convert('RGB').resize((w, h), Image.BICUBIC)


def build(output: Path, ablation: Path, data_dir: Path, seed: int) -> None:
    """Case lists for parts A and B and the representation image set of part C."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'refcoco_images'):
        link, real = output / name, (ablation / name).resolve()
        if not real.exists():
            raise FileNotFoundError(real)
        if not link.is_symlink():
            link.symlink_to(real)
        elif link.resolve() != real:
            raise ValueError(f'{link} points elsewhere')
    rng = random.Random(seed)
    cases = [dict(c, part='A') for c in read_jsonl(ablation / 'cases.jsonl') if c['task'] in COORD_TASKS]

    # POPE adversarial, balanced yes/no.
    pope = [json.loads(line) for line in (data_dir / 'POPE' / 'adversarial.jsonl').open()]
    for answer in ('yes', 'no'):
        pool = [r for r in pope if r['answer'] == answer]
        for r in rng.sample(pool, POPE_N // 2):
            path = data_dir / 'POPE' / 'images' / 'adversarial' / r['image']
            cases.append({'id': f"pope|{r['id']}", 'part': 'B', 'suite': 'general', 'task': 'POPE', 'format': 'yes_no',
                          'image': f"pope_images/{r['image']}", 'source_image': r['image_source'],
                          'prompt': f"{r['question']} Answer with only Yes or No.",
                          'truth': {'answer_yes': answer == 'yes'}, '_src': path})
    # TextVQA validation.
    import pyarrow.parquet as pq
    rows = []
    for f in sorted((data_dir / 'TextVQA' / 'data').glob('validation-*.parquet')):
        rows += pq.read_table(f, columns=['question_id', 'question', 'answers', 'image_id']).to_pylist()
    picked = rng.sample(rows, TEXTVQA_N)
    wanted = {r['question_id'] for r in picked}
    images = {}
    for f in sorted((data_dir / 'TextVQA' / 'data').glob('validation-*.parquet')):
        for r in pq.read_table(f, columns=['question_id', 'image']).to_pylist():
            if r['question_id'] in wanted:
                images[r['question_id']] = r['image']['bytes']
    (output / 'textvqa_images').mkdir(exist_ok=True)
    for r in picked:
        path = output / 'textvqa_images' / f"{r['question_id']}.jpg"
        if not path.exists():
            Image.open(io.BytesIO(images[r['question_id']])).convert('RGB').save(path, quality=95)
        cases.append({'id': f"textvqa|{r['question_id']}", 'part': 'B', 'suite': 'general', 'task': 'TextVQA',
                      'format': 'short', 'image': f"textvqa_images/{r['question_id']}.jpg",
                      'prompt': f"{r['question']}\nAnswer the question using a single word or phrase.",
                      'truth': {'answers': r['answers']}})
    # BabyVision, every item, direct-answer prompt.
    from eva.benchmark import prompt_for
    for r in (json.loads(line) for line in (data_dir / 'BabyVision' / 'train.jsonl').open()):
        prompt = prompt_for(r, 'babyvision') + '\nDo not explain. Output only the final answer line.'
        cases.append({'id': f"babyvision|{r['taskId']}", 'part': 'B', 'suite': 'general', 'task': 'BabyVision',
                      'format': 'babyvision', 'image': f"babyvision_images/{Path(r['image']).name}",
                      'prompt': prompt, 'truth': {k: r.get(k) for k in ('ansType', 'options', 'choiceAns', 'blankAns',
                                                                          'question', 'type', 'subtype')},
                      '_src': data_dir / 'BabyVision' / r['image']})
    for sub in ('pope_images', 'babyvision_images'):
        (output / sub).mkdir(exist_ok=True)
    for c in cases:
        src = c.pop('_src', None)
        if src is not None:
            dst = output / c['image']
            if not dst.exists():
                Image.open(src).convert('RGB').save(dst)
        c['image_sha256'] = c.get('image_sha256') or sha256_file(output / c['image'] if c['task'] != 'RefCOCO'
                                                                   else output / 'refcoco_images' / c['image'])
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case ids')
    target = output / 'cases.jsonl'
    if target.exists() and read_jsonl(target) != json.loads(json.dumps(cases)):
        raise ValueError(f'{target} exists with a different selection')
    write_jsonl(target, cases)

    # Representation images (part C).
    rep_dir = output / 'rep_images'
    rep_dir.mkdir(exist_ok=True)
    synth = sorted({c['image'] for c in cases if c['task'] in ('T5', 'T6')})
    synth = rng.sample(synth, REP_N['synthetic'])
    rep = [{'source': 'synthetic', 'image': s, 'sha256': sha256_file(output / s)} for s in synth]
    ref = sorted({c['image'] for c in cases if c['task'] == 'RefCOCO'})
    bv = sorted({c['image'] for c in cases if c['task'] == 'BabyVision'})
    for source, pool, base in (('refcoco', ref, output / 'refcoco_images'), ('babyvision', bv, output)):
        for name in rng.sample(pool, REP_N[source]):
            dst = rep_dir / f'{source}_{Path(name).stem}.png'
            if not dst.exists():
                _resize32(Image.open(base / name), REP_LONG_SIDE).save(dst)
            rep.append({'source': source, 'image': f'rep_images/{dst.name}', 'sha256': sha256_file(dst)})
    write_jsonl(output / 'rep_images.jsonl', rep)
    counts: dict = {}
    for c in cases:
        counts[c['task']] = counts.get(c['task'], 0) + 1
    _check_provenance(output / 'build.json', {'seed': seed, 'ablation_source': ablation.name, 'counts': counts,
                                              'pope_n': POPE_N, 'textvqa_n': TEXTVQA_N, 'rep_n': REP_N,
                                              'rep_long_side': REP_LONG_SIDE})
    print(counts, 'rep images', len(rep))


# ----------------------------------------------------------------------------- answering


def answer_any(rt, images: list[Image.Image], prompts: list[str], budget: int, yes_no_ids=None) -> list[dict]:
    """Left-padded greedy decoding with first-step logits; the processor may resize natural images."""
    import torch
    from transformers import LogitsProcessorList
    from analysis_tools.common import qwen_runtime as qr
    from analysis_tools.deepstack.deepstack_comparison import _first_step_processor
    texts = [qr.prompt_text(rt, 1, p, False) for p in prompts]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    first = _first_step_processor()
    with torch.inference_mode():
        out = rt.model.generate(**batch, max_new_tokens=budget, do_sample=False, temperature=None, top_p=None,
                                top_k=None, logits_processor=LogitsProcessorList([first]))
    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = rt.tokenizer.pad_token_id
    records = []
    for i, row in enumerate(out[:, batch['input_ids'].shape[1]:].tolist()):
        end = next((j + 1 for j, t in enumerate(row) if t in eos), None)
        new = row[:end] if end is not None else [t for t in row if t != pad]
        rec = {'response': rt.tokenizer.decode(new, skip_special_tokens=True), 'n_tokens': len(new),
               'finish_reason': 'eos' if end is not None else 'length', 'max_new_tokens': budget,
               'batch_size': len(images)}
        if yes_no_ids is not None:
            z = first.first[i]
            rec.update(s=float(z[yes_no_ids[0]] - z[yes_no_ids[1]]))
        records.append(rec)
    return records


def _image_path(output: Path, case: dict) -> Path:
    return output / ('refcoco_images/' + case['image'] if case['task'] == 'RefCOCO' else case['image'])


def _budget(case: dict, cfg: dict) -> int:
    if case['suite'] == 'decomp':
        return max_tokens(case, cfg)
    if case['task'] == 'RefCOCO':
        return cfg['max_new_tokens_locate']
    return BUDGET[case['task']]


def run(output: Path, ablation: Path, model: str, condition: str, tasks: list[str], batch_size: int,
        cfg: dict) -> None:
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
    means = load_means(ablation, model) if condition == 'mean' else None
    _check_provenance(run_dir / f'{condition}.json', {
        'model': qr.describe(rt), 'condition': condition, 'decoding': 'greedy', 'thinking': False,
        'budgets': {**BUDGET, 'locate': cfg['max_new_tokens_locate'], 'binary': cfg['max_new_tokens_binary']},
        'batch_size': 'per record', 'means_source': ablation.name if condition == 'mean' else None})
    override = TodoOverride(rt, condition, means)
    ids = (qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No'))
    groups: dict = {}
    for c in pending:
        key = (_budget(c, cfg), c['task'], c.get('grid', 0) if c['suite'] == 'decomp' else 0)
        groups.setdefault(key, []).append(c)
    batches = []
    for key in sorted(groups):
        rows = groups[key]
        if rows[0]['suite'] != 'decomp':       # similar natural image sizes share a batch
            rows = sorted(rows, key=lambda c: Image.open(_image_path(output, c)).size)
        batches += [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    n_done = 0
    with path.open('a') as sink, deepstack_override(rt, override):
        for batch in batches:
            images = []
            for c in batch:
                p = _image_path(output, c)
                if sha256_file(p) != c['image_sha256']:
                    raise ValueError(f'Image hash mismatch for {p}')
                images.append(Image.open(p).convert('RGB'))
            override.prepare(images)
            calls = override.calls
            binary = batch[0]['format'] == 'yes_no'
            records = answer_any(rt, images, [c['prompt'] for c in batch], _budget(batch[0], cfg),
                                 ids if binary else None)
            if override.calls != calls + 1:
                raise RuntimeError(f'DeepStack override fired {override.calls - calls} times in one batch')
            for c, rec in zip(batch, records):
                sink.write(json.dumps({'id': c['id'], 'condition': condition, **rec}) + '\n')
            sink.flush()
            n_done += len(batch)
            if n_done // 300 != (n_done - len(batch)) // 300:
                print(f'{condition}: {n_done}/{len(pending)} {dt.datetime.now():%H:%M:%S}', flush=True)
    with (run_dir / 'run_log.jsonl').open('a') as log:
        log.write(json.dumps({'condition': condition, 'n_run': n_done, 'batch_size': batch_size,
                              'end': dt.datetime.now().isoformat(timespec='seconds')}) + '\n')


def verify(output: Path, ablation: Path, model: str) -> None:
    """Checks on one T6 image: each condition changes decoder outputs only from layer 2 on, and the flip
    helper inverts an image mirror (the content peak of the mirrored image maps back to the same token)."""
    import torch
    from analysis_tools.common import qwen_runtime as qr
    case = next(c for c in read_jsonl(output / 'cases.jsonl') if c['task'] == 'T6' and c['format'] == 'point')
    rt = qr.load(model, device='cuda:0')
    image = Image.open(_image_path(output, case)).convert('RGB')
    # Asymmetric probe image: white canvas with one black square in the upper-left area.
    probe = Image.new('RGB', (640, 384), (255, 255, 255))
    probe.paste((0, 0, 0), (64, 64, 160, 160))
    feats, grids = deepstack_features(rt, [probe])
    blank, _ = deepstack_features(rt, [blank_like(probe)])
    mirrored, _ = deepstack_features(rt, [probe.transpose(Image.FLIP_LEFT_RIGHT)])
    # Content map of every token (distance from the uniform-image features).  If flip_tokens mirrors the
    # raster token grid, the mirrored image's content map mapped back by flip_tokens matches the original map.
    dev = (feats[0] - blank[0]).float().norm(dim=1)
    dev_m = (mirrored[0] - blank[0]).float().norm(dim=1)
    back = flip_tokens(dev_m[:, None], grids)[:, 0]
    corr_back = float(torch.corrcoef(torch.stack([dev, back]))[0, 1])
    corr_raw = float(torch.corrcoef(torch.stack([dev, dev_m]))[0, 1])
    print(f'flip check: corr(map, flip(mirrored map)) = {corr_back:.3f}; without flip {corr_raw:.3f}')
    if corr_back < 0.8 or corr_back < corr_raw + 0.2:
        raise RuntimeError('flip_tokens does not invert an image mirror')
    batch = qr.build(rt, [image], case['prompt'])
    span = qr.image_spans(rt, batch)[0]
    positions = torch.arange(span['start'], span['end'])
    means = load_means(ablation, model)
    store = {}
    for cond in CONDITIONS:
        ov = TodoOverride(rt, cond, means)
        ov.prepare([image])
        with deepstack_override(rt, ov), qr.capture(rt, [1, 2, 3, 4], positions) as cap, torch.inference_mode():
            logits = rt.model(**batch).logits[0, -1].float().cpu()
        if ov.calls != 1:
            raise RuntimeError(f'{cond}: override fired {ov.calls} times')
        store[cond] = ({k: v.clone() for k, v in cap.items()}, logits)
    rows = []
    for cond in CONDITIONS:
        hs, logits = store[cond]
        base_h, base_l = store['full']
        row = {'condition': cond, **{f'layer{k}': float((hs[k] - base_h[k]).norm() / base_h[k].norm()) for k in (1, 2, 3, 4)},
               'logits_max_abs': float((logits - base_l).abs().max())}
        rows.append(row)
        print(json.dumps(row))
    (output / 'checks').mkdir(exist_ok=True)
    write_jsonl(output / 'checks' / f'verify_{slug(model)}.jsonl',
                rows + [{'flip_check': {'corr_back': corr_back, 'corr_raw': corr_raw, 'grid': grids[0]}}])


def spotcheck(output: Path, ablation: Path, model: str, conditions: list[str], n: int, seed: int, cfg: dict) -> None:
    """Batch-size-1 answers under the same override for a random subset of each batched run, compared by score."""
    import torch
    from analysis_tools.common import qwen_runtime as qr
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    rt = qr.load(model, device='cuda:0')
    ids = (qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No'))
    rows = []
    for cond in conditions:
        records = read_jsonl(output / 'runs' / slug(model) / f'{cond}.jsonl')
        sample = random.Random(seed).sample(records, n)
        ov = TodoOverride(rt, cond, load_means(ablation, model) if cond == 'mean' else None)
        with deepstack_override(rt, ov):
            for rec in sample:
                case = cases[rec['id']]
                image = Image.open(_image_path(output, case)).convert('RGB')
                ov.prepare([image])
                single = answer_any(rt, [image], [case['prompt']], _budget(case, cfg),
                                    ids if case['format'] == 'yes_no' else None)[0]
                score_of = (lambda r: score_refcoco_case(case, r)) if case['suite'] == 'refcoco' else \
                    (lambda r: score_case(case, r, True)) if case['suite'] == 'decomp' else (lambda r: score_general(case, r))
                a, b = score_of(rec), score_of(single)
                key = 'hit' if case.get('format') == 'point' else 'correct'
                rows.append({'condition': cond, 'id': case['id'], 'text_equal': single['response'] == rec['response'],
                             'answer_equal': a.get(key) == b.get(key)})
    (output / 'checks').mkdir(exist_ok=True)
    write_jsonl(output / 'checks' / f'spotcheck_{slug(model)}.jsonl', rows)
    for cond in conditions:
        sub = [r for r in rows if r['condition'] == cond]
        print(f'{cond}: answer equal {sum(r["answer_equal"] for r in sub)}/{len(sub)}, '
              f'text equal {sum(r["text_equal"] for r in sub)}/{len(sub)}')


# ----------------------------------------------------------------------------- representation (part C)


def token_targets(image: Image.Image) -> dict:
    """Per merged token (raster order): normalized centre, 32x32 grey high-pass footprint, 4x4 RGB means.

    The high-pass footprint is the grey image minus its Gaussian blur (standard deviation 2 pixels), i.e. the
    detail that a sigma-2 blur removes; it is the quantity whose readability part C compares across sites.
    """
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    grey = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    low = np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(radius=HIGHPASS_SIGMA)),
                     dtype=np.float32) / 255.0
    high = grey - low
    h, w = grey.shape
    rows, cols = h // TOKEN_PIXELS, w // TOKEN_PIXELS
    t = TOKEN_PIXELS

    def tiles(a):
        a = a[:rows * t, :cols * t]
        return a.reshape(rows, t, cols, t, *a.shape[2:]).swapaxes(1, 2).reshape(rows * cols, t, t, *a.shape[2:])
    hp = tiles(high).reshape(rows * cols, -1)
    gr = tiles(grey).reshape(rows * cols, -1)
    coarse = tiles(rgb).reshape(rows * cols, 4, 8, 4, 8, 3).mean((2, 4)).reshape(rows * cols, -1)
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    xy = np.stack([(c.ravel() + .5) / cols, (r.ravel() + .5) / rows], 1).astype(np.float32)
    content = gr.std(1) > 0.01
    return {'xy': xy, 'highpass': hp, 'grey': gr, 'coarse': coarse, 'content': content, 'grid': (rows, cols)}


def detail_targets(image: Image.Image) -> dict:
    """Phase-tolerant detail targets per merged token (raster order).

    structure -- 8x8 grey thumbnail of the 32x32 footprint (4-pixel cells) minus the footprint mean: the
                 shape of edges and strokes at 4-pixel resolution, independent of overall brightness.
    hfenergy  -- log mean squared sigma-2 high-pass response in each 8x8-pixel sub-block (4x4 values): where
                 fine texture and thin lines are, without requiring their exact pixel phase.
    """
    tg = token_targets(image)
    n = tg['grey'].shape[0]
    grey = tg['grey'].reshape(n, 8, 4, 8, 4).mean((2, 4)).reshape(n, 64)
    structure = grey - grey.mean(1, keepdims=True)
    hp = tg['highpass'].reshape(n, 4, 8, 4, 8)
    hfenergy = np.log(1e-4 + (hp ** 2).mean((2, 4))).reshape(n, 16)
    return {'structure': structure.astype(np.float32), 'hfenergy': hfenergy.astype(np.float32)}


def select_tokens(content: np.ndarray, rng: np.random.Generator, n: int = REP_TOKENS) -> np.ndarray:
    idx_c, idx_o = np.flatnonzero(content), np.flatnonzero(~content)
    n_c = min(len(idx_c), n * 3 // 4)
    n_o = min(len(idx_o), n - n_c)
    n_c = min(len(idx_c), n - n_o)
    return np.sort(np.concatenate([rng.choice(idx_c, n_c, replace=False), rng.choice(idx_o, n_o, replace=False)]))


def _vision_states(rt, image: Image.Image) -> dict:
    """Merged-token (raster order) states of one image: ViT blocks (2x2-averaged), DeepStack features, V."""
    import torch
    from analysis_tools.position.vit_layout import pixel_inputs
    visual = rt.model.model.visual
    store, handles = {}, []
    for k in VIT_BLOCKS:
        handles.append(visual.blocks[k - 1].register_forward_hook(
            lambda m, i, o, k=k: store.__setitem__(f'vit{k}', o[0] if isinstance(o, tuple) else o)))
    for j, merger in enumerate(getattr(visual, 'deepstack_merger_list', []) or []):
        handles.append(merger.register_forward_hook(lambda m, i, o, j=j: store.__setitem__(f'D{j + 1}', o)))
    handles.append(visual.merger.register_forward_hook(lambda m, i, o: store.__setitem__('V', o)))
    pixel, grid = pixel_inputs(rt, [image])
    try:
        with torch.inference_mode():
            visual(pixel, grid_thw=grid)
    finally:
        for h in handles:
            h.remove()
    out = {}
    for key, value in store.items():
        value = value.float()
        if key.startswith('vit'):          # block order: 4 consecutive patches form one merged token
            value = value.reshape(-1, 4, value.shape[-1]).mean(1)
        out[key] = value.cpu()
    return out


def represent(output: Path, model: str, seed: int) -> None:
    import torch
    from analysis_tools.common import qwen_runtime as qr
    rep = read_jsonl(output / 'rep_images.jsonl')
    rt = qr.load(model, device='cuda:0')
    family = rt.family
    layers = DEC_LAYERS[family]
    rng = np.random.default_rng(seed)
    view = {s: set([i for i, r in enumerate(rep) if r['source'] == s][:REP_VIEW]) for s in REP_N}
    view_ids = set().union(*view.values())
    sites: dict = {}
    meta = {'image': [], 'token': [], 'xy': [], 'content': [], 'source': []}
    targets = {'highpass': [], 'coarse': [], 'grey': []}
    full_view: dict = {}
    for i, item in enumerate(rep):
        path = output / item['image']
        if sha256_file(path) != item['sha256']:
            raise ValueError(f'hash mismatch {path}')
        image = Image.open(path).convert('RGB')
        tg = token_targets(image)
        states = _vision_states(rt, image)
        blank_states = _vision_states(rt, blank_like(image))
        n_tok = tg['xy'].shape[0]
        if states['V'].shape[0] != n_tok:
            raise ValueError(f'{path}: {states["V"].shape[0]} tokens, expected {n_tok}')
        batch = qr.build(rt, [image], REP_PROMPT)
        span = qr.image_spans(rt, batch)[0]
        positions = torch.arange(span['start'], span['end'])
        dec = {}
        conds = ('full', 'off') if family == 'qwen3_vl' else ('full',)
        for cond in conds:
            ov = TodoOverride(rt, cond)
            with deepstack_override(rt, ov) if family == 'qwen3_vl' else _null(), \
                    qr.capture(rt, list(layers), positions) as cap, torch.inference_mode():
                rt.model(**batch)
            for layer in layers:
                if cond == 'off' and layer == 1:
                    continue                    # layer-1 output precedes every injection
                dec[f'{cond}_L{layer}'] = cap[layer]
        if family == 'qwen3_vl':
            for j in (1, 2, 3):
                dec[f'full_L{j}+D{j}'] = dec[f'full_L{j}'] + states[f'D{j}']
        all_sites = {**states, **{f'blank_{k}': v for k, v in blank_states.items()}, **dec}
        keep = np.arange(n_tok) if i in view_ids else select_tokens(tg['content'], rng)
        for key, value in all_sites.items():
            sites.setdefault(key, []).append(value[keep].numpy().astype(np.float16))
        meta['image'].append(np.full(len(keep), i))
        meta['token'].append(keep)
        meta['xy'].append(tg['xy'][keep])
        meta['content'].append(tg['content'][keep])
        meta['source'].append(np.full(len(keep), list(REP_N).index(item['source'])))
        for k in targets:
            targets[k].append(tg[k][keep].astype(np.float16))
        if i % 30 == 0:
            print(f'{model}: {i + 1}/{len(rep)} {dt.datetime.now():%H:%M:%S}', flush=True)
    target_dir = output / 'represent' / slug(model)
    target_dir.mkdir(parents=True, exist_ok=True)
    for key, chunks in sites.items():
        np.save(target_dir / f'site_{key.replace("+", "p")}.npy', np.concatenate(chunks))
    np.savez(target_dir / 'meta.npz', **{k: np.concatenate(v) for k, v in meta.items()},
             **{f't_{k}': np.concatenate(v) for k, v in targets.items()},
             view=np.array(sorted(view_ids)), sources=np.array(list(REP_N)))
    _check_provenance(target_dir / 'represent.json', {'model': qr.describe(rt), 'vit_blocks_1based': list(VIT_BLOCKS),
                                                       'decoder_layers': list(layers), 'prompt': REP_PROMPT,
                                                       'tokens_per_image': REP_TOKENS, 'seed': seed,
                                                       'sites': sorted(sites)})
    print(f'{model}: saved {len(sites)} sites to {target_dir}')


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------------------- scoring


def _norm_vqa(text: str) -> str:
    text = text.lower().strip().rstrip('.')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def score_general(case: dict, record: dict) -> dict:
    task = case['task']
    if task == 'POPE':
        s = record['s']
        return {'id': case['id'], 's': s, 'pred_yes': s > 0, 'correct': (s > 0) == case['truth']['answer_yes']}
    if task == 'TextVQA':
        pred = _norm_vqa(record['response'].split('\n')[0])
        hits = sum(_norm_vqa(a) == pred for a in case['truth']['answers'])
        return {'id': case['id'], 'pred': pred, 'vqa_acc': min(1.0, hits / 3), 'correct': hits >= 3}
    from eva.diagnostic_scoring import diagnostic_score
    sc = diagnostic_score(case['truth'], record['response'], record['finish_reason'])
    return {'id': case['id'], 'correct': bool(sc['correct']), 'strict_correct': bool(sc['strict_correct']),
            'prediction': sc['prediction']}


def score(output: Path) -> None:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    for path in sorted((output / 'runs').glob('*/*.jsonl')):
        if path.name == 'run_log.jsonl' or path.name.endswith('.scores.jsonl'):
            continue
        records = read_jsonl(path)
        if len({r['id'] for r in records}) != len(records):
            raise ValueError(f'Duplicate results in {path}')
        rows = []
        for r in records:
            c = cases[r['id']]
            if c['suite'] == 'refcoco':
                rows.append(score_refcoco_case(c, r))
            elif c['suite'] == 'decomp':
                rows.append({'id': r['id'], **score_case(c, r, True)})
            else:
                rows.append(score_general(c, r))
        write_jsonl(path.parent / f'{path.stem}.scores.jsonl', rows)
        print(f'{path.parent.name}/{path.stem}: {len(rows)} scored')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('build', 'verify', 'run', 'represent', 'score', 'spotcheck'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ablation', type=Path)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--model', default='Qwen3-VL-8B-Instruct')
    p.add_argument('--condition', default='full', choices=CONDITIONS)
    p.add_argument('--tasks', nargs='+', default=list(COORD_TASKS))
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-new-tokens-binary', type=int, default=8)
    p.add_argument('--max-new-tokens-locate', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--conditions', nargs='+', default=['blank', 'blankflip'])
    p.add_argument('--spotcheck-n', type=int, default=20)
    a = p.parse_args()
    cfg = {'max_new_tokens_binary': a.max_new_tokens_binary, 'max_new_tokens_locate': a.max_new_tokens_locate,
           'max_new_tokens_explicit': None, 'max_new_tokens_thinking': None}
    if a.command == 'build':
        build(a.output, a.ablation, a.data_dir, a.seed)
    elif a.command == 'verify':
        verify(a.output, a.ablation, a.model)
    elif a.command == 'run':
        run(a.output, a.ablation, a.model, a.condition, a.tasks, a.batch_size, cfg)
    elif a.command == 'spotcheck':
        spotcheck(a.output, a.ablation, a.model, a.conditions, a.spotcheck_n, a.seed, cfg)
    elif a.command == 'represent':
        represent(a.output, a.model, a.seed)
    else:
        score(a.output)


if __name__ == '__main__':
    main()
