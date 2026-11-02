"""Experiment 4.5: DeepStack ablations on natural fine-grained tasks (Qwen3-VL), compared with Qwen3.5.

Design: docs/analysis/deepstack/results_2026-10-21_natural.md, section 2.  Question: on natural images
that need fine detail (scene text, documents, small objects, fine-grained bird attributes, MMVP), does the
content of the DeepStack injection reach the answer, or only its position code (experiment 4.4)?

Global conditions (each changes only the DeepStack features added after decoder layers 1-3; see
``deepstack_todo``): full, off, mean, blank, blankflip, blur2, blur8, and
  meanpos     position-only injection: Blank features minus their per-image token average, plus the Mean vector
Region conditions (only items with an evidence box, i.e. the question-relevant region from Visual-CoT):
  blank_in    Full features everywhere except tokens overlapping the evidence box, which get Blank features
  blank_out   Blank features everywhere except the evidence tokens, which keep Full features
  blank_ctrl  Blank features at a control box of the same token shape that does not overlap the evidence box

Every image is resized beforehand to sides that are multiples of 32 pixels within a pixel budget, so the
processor keeps the size and each merged token covers a known 32x32 pixel cell.

Commands: build | run | score
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import random
import re
import subprocess

import numpy as np
from PIL import Image, ImageFilter

from analysis_tools.behavior.decomposition import read_jsonl, sha256_file, write_jsonl
from analysis_tools.deepstack.deepstack_ablation import deepstack_features, deepstack_override, load_means
from analysis_tools.deepstack.deepstack_comparison import _check_provenance, _first_step_processor, slug
from analysis_tools.deepstack.deepstack_todo import blank_like, blur_sigma, flip_tokens

TASKS = ('textvqa', 'docvqa', 'gqa', 'cub', 'mmvp')
REGION_TASKS = ('textvqa', 'docvqa', 'gqa', 'cub')
GLOBAL_CONDITIONS = ('full', 'off', 'mean', 'blank', 'meanpos', 'blankflip', 'blur2', 'blur8')
REGION_CONDITIONS = ('blank_in', 'blank_out', 'blank_ctrl')
CONDITIONS = GLOBAL_CONDITIONS + REGION_CONDITIONS
N_ITEMS = {'textvqa': 400, 'docvqa': 400, 'gqa': 400, 'cub': 400, 'mmvp': 300}
MAX_AREA = {'textvqa': 0.005, 'gqa': 0.015}      # evidence-box area / image area (small text, small objects)
PIXEL_BUDGET = {'docvqa': 1600 * 1024}           # others: 1024 * 1024
DEFAULT_BUDGET = 1024 * 1024
MIN_PIXELS = 65536                               # processor minimum (preprocessor_config shortest_edge)
BUDGET = {'textvqa': 24, 'docvqa': 32, 'gqa': 16, 'cub': 8, 'mmvp': 8}
TOKEN_PIXELS = 32
# Tasks with a teacher-forced gold log-probability. DocVQA is excluded after its Full run for compute reasons
# (its prompts are about 1600 tokens long); ANLS serves as its graded metric.
TF_TASKS = ('textvqa', 'gqa')
SHORT_SUFFIX = '\nAnswer the question using a single word or phrase.'
YES_NO_SUFFIX = ' Answer with only Yes or No.'
MMVP_SUFFIX = '\nAnswer with only the letter of the correct option, a or b.'
VISCOT_ARCHIVE = 'Visual-CoT/cot_images_tar_split'


# ----------------------------------------------------------------------------- geometry


def resize32(image: Image.Image, max_pixels: int) -> tuple[Image.Image, float, float]:
    """Resize to sides that are multiples of 32 with MIN_PIXELS <= area <= max_pixels.

    Returns the image and the horizontal and vertical scale factors from original to resized pixels.
    """
    w, h = image.size
    s = min(1.0, math.sqrt(max_pixels / (w * h)))
    nw = max(64, int(w * s / TOKEN_PIXELS) * TOKEN_PIXELS)
    nh = max(64, int(h * s / TOKEN_PIXELS) * TOKEN_PIXELS)
    if nw * nh < MIN_PIXELS:
        # The processor upscales images below its minimum area; upscale here so the size stays known.
        s = math.sqrt(MIN_PIXELS / (w * h))
        nw = int(math.ceil(w * s / TOKEN_PIXELS)) * TOKEN_PIXELS
        nh = int(math.ceil(h * s / TOKEN_PIXELS)) * TOKEN_PIXELS
    return image.convert('RGB').resize((nw, nh), Image.BICUBIC), nw / w, nh / h


def token_mask(boxes: list[list[float]], width: int, height: int) -> np.ndarray:
    """Boolean (rows, cols) mask of merged tokens whose 32x32 cell overlaps any box (resized pixels)."""
    rows, cols = height // TOKEN_PIXELS, width // TOKEN_PIXELS
    mask = np.zeros((rows, cols), bool)
    for x0, y0, x1, y1 in boxes:
        c0 = max(0, int(math.floor(x0 / TOKEN_PIXELS)))
        c1 = min(cols, int(math.ceil(x1 / TOKEN_PIXELS)))
        r0 = max(0, int(math.floor(y0 / TOKEN_PIXELS)))
        r1 = min(rows, int(math.ceil(y1 / TOKEN_PIXELS)))
        mask[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)] = True
    return mask


def control_mask(mask: np.ndarray, rng: random.Random, tries: int = 200) -> np.ndarray | None:
    """Same-shape copy of the bounding rectangle of ``mask`` moved to a location that does not overlap it.

    The control replaces the same pattern of tokens (the evidence mask shifted as a whole), so the number of
    altered tokens is identical. Returns None when no non-overlapping placement exists.
    """
    rr, cc = np.nonzero(mask)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    patch = mask[r0:r1, c0:c1]
    rows, cols = mask.shape
    spots = [(r, c) for r in range(rows - (r1 - r0) + 1) for c in range(cols - (c1 - c0) + 1)]
    rng.shuffle(spots)
    for r, c in spots[:tries] if len(spots) > tries else spots:
        out = np.zeros_like(mask)
        out[r:r + r1 - r0, c:c + c1 - c0] = patch
        if not (out & mask).any():
            return out
    return None


def mask_to_list(mask: np.ndarray) -> list[int]:
    return np.flatnonzero(mask.reshape(-1)).tolist()


# ----------------------------------------------------------------------------- build


def _viscot(data_public: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (data_public / 'Visual-CoT' / 'metadata' / f'{name}_cot_train.jsonl').open()]


def _area(r: dict) -> float:
    return max((b[2] - b[0]) * (b[3] - b[1]) for b in r['bboxs']) / (r['width'] * r['height'])


def _one_per_image(rows: list[dict], rng: random.Random, n: int, keep=lambda r: True) -> list[dict]:
    rows = [r for r in rows if keep(r)]
    rng.shuffle(rows)
    seen, out = set(), []
    for r in rows:
        if r['image'] in seen:
            continue
        seen.add(r['image'])
        out.append(r)
        if len(out) == n:
            break
    if len(out) < n:
        raise ValueError(f'Only {len(out)} items available, {n} requested')
    return out


def select(data_dir: Path, data_public: Path, mmvp_csv: Path, seed: int) -> list[dict]:
    """Item lists per task; image sources are resolved later by ``build``."""
    rng = random.Random(seed)
    short = lambda r: len(str(r['answer']).split()) <= 4
    items = []
    for r in _one_per_image(_viscot(data_public, 'textvqa'), rng, N_ITEMS['textvqa'],
                            lambda r: _area(r) < MAX_AREA['textvqa'] and short(r)):
        items.append({'task': 'textvqa', 'question': r['question'], 'answers': [r['answer']],
                      'source': f"TextVQA/data/images/{r['image']}", 'boxes': r['bboxs'], 'area': _area(r)})
    for r in _one_per_image(_viscot(data_public, 'docvqa'), rng, N_ITEMS['docvqa'], short):
        items.append({'task': 'docvqa', 'question': r['question'], 'answers': r.get('possible_answers') or [r['answer']],
                      'source': f"viscot/docvqa/{r['image']}", 'boxes': r['bboxs'], 'area': _area(r)})
    for r in _one_per_image(_viscot(data_public, 'gqa'), rng, N_ITEMS['gqa'],
                            lambda r: _area(r) < MAX_AREA['gqa'] and len(r['answer'].split()) == 1):
        items.append({'task': 'gqa', 'question': r['question'], 'answers': [r['answer']],
                      'source': f"GQA/images/images/{r['image']}", 'boxes': r['bboxs'], 'area': _area(r)})
    cub = _viscot(data_public, 'cub')
    for answer in ('Yes', 'No'):
        for r in _one_per_image(cub, rng, N_ITEMS['cub'] // 2, lambda r, a=answer: r['answer'] == a):
            items.append({'task': 'cub', 'question': r['question'], 'answers': [answer],
                          'source': f"viscot/cub/{r['image']}", 'boxes': r['bboxs'], 'area': _area(r)})
    # CUB images may repeat across the two answer halves; that is harmless for paired comparisons.
    with mmvp_csv.open() as f:
        for row in csv.DictReader(f):
            items.append({'task': 'mmvp', 'question': f"{row['Question']}\nOptions: {row['Options']}",
                          'answers': [row['Correct Answer'].strip('() ').lower()], 'pair': (int(row['Index']) - 1) // 2,
                          'source': f"MMVP/{row['Index']}.jpg", 'boxes': None, 'area': None})
    for i, it in enumerate(items):
        it['id'] = f"{it['task']}|{i:04d}"
    return items


def extract_viscot(items: list[dict], data_public: Path, target: Path) -> None:
    """Extract the needed Visual-CoT images (CUB, DocVQA) from the split tar archive by streaming it once."""
    wanted = sorted({'cot_image_data/' + it['source'][len('viscot/'):] for it in items
                     if it['source'].startswith('viscot/')})
    missing = [w for w in wanted if not (target / w).exists()]
    if not missing:
        return
    target.mkdir(parents=True, exist_ok=True)
    listing = target / 'wanted.txt'
    listing.write_text('\n'.join(missing) + '\n')
    parts = sorted((data_public / VISCOT_ARCHIVE).glob('cot_images_*'))
    # GNU tar stops reading once every listed member was found (--occurrence), and the archive stores CUB and
    # DocVQA first, so only the head of the 130 GB stream is read.
    cat = subprocess.Popen(['cat', *map(str, parts)], stdout=subprocess.PIPE)
    tar = subprocess.run(['tar', '-x', '-C', str(target), '--occurrence=1', '-T', str(listing)], stdin=cat.stdout)
    cat.stdout.close()
    cat.kill()
    still = [w for w in missing if not (target / w).exists()]
    if still:
        raise FileNotFoundError(f'{len(still)} Visual-CoT images were not extracted (tar exit {tar.returncode}), '
                                f'e.g. {still[:3]}')


def build(output: Path, data_dir: Path, data_public: Path, mmvp_csv: Path, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    items = select(data_dir, data_public, mmvp_csv, seed)
    extract_viscot(items, data_public, output / 'viscot_raw')
    rng = random.Random(seed + 1)
    (output / 'images').mkdir(exist_ok=True)
    cases = []
    for it in items:
        src = it['source']
        if src.startswith('viscot/'):
            path = output / 'viscot_raw' / 'cot_image_data' / src[len('viscot/'):]
        elif src.startswith('MMVP/'):
            path = mmvp_csv.parent / src[len('MMVP/'):]
            if not path.exists():
                path = data_dir / 'MMVP Images' / src[len('MMVP/'):]
        else:
            path = data_dir / src
        raw = Image.open(path)
        image, sx, sy = resize32(raw, PIXEL_BUDGET.get(it['task'], DEFAULT_BUDGET))
        name = f"images/{it['id'].replace('|', '_')}.png"
        if not (output / name).exists():
            image.save(output / name)
        case = {'id': it['id'], 'task': it['task'], 'image': name, 'image_sha256': sha256_file(output / name),
                'width': image.width, 'height': image.height, 'source': src, 'answers': it['answers'],
                'question': it['question'], 'area': it['area']}
        if it['task'] == 'mmvp':
            case['pair'] = it['pair']
            case['prompt'] = it['question'] + MMVP_SUFFIX
        elif it['task'] == 'cub':
            case['prompt'] = it['question'] + YES_NO_SUFFIX
        else:
            case['prompt'] = it['question'] + SHORT_SUFFIX
        if it['boxes'] is not None:
            # Visual-CoT boxes are given in the original image pixels; the width/height fields of the metadata
            # match the stored image, which ``raw.size`` confirms.
            boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in it['boxes']]
            mask = token_mask(boxes, image.width, image.height)
            ctrl = control_mask(mask, rng)
            case.update(boxes=boxes, region=mask_to_list(mask), control=None if ctrl is None else mask_to_list(ctrl),
                        token_frac=float(mask.mean()))
        cases.append(case)
    target = output / 'cases.jsonl'
    if target.exists() and read_jsonl(target) != json.loads(json.dumps(cases)):
        raise ValueError(f'{target} exists with a different selection')
    write_jsonl(target, cases)
    counts: dict = {}
    for c in cases:
        counts[c['task']] = counts.get(c['task'], 0) + 1
    _check_provenance(output / 'build.json', {'seed': seed, 'counts': counts, 'max_area': MAX_AREA,
                                              'pixel_budget': PIXEL_BUDGET, 'default_budget': DEFAULT_BUDGET,
                                              'mmvp_sha256': sha256_file(mmvp_csv)})
    print(counts, 'no control box:', sum(1 for c in cases if c.get('region') is not None and c['control'] is None))


# ----------------------------------------------------------------------------- conditions


def position_only(blank, grids: list[list[int]], mean):
    """Position-only injection: the token-wise deviation of the Blank features from their per-image token average,
    added to the reference mean vector of the Mean condition.

    The per-image average of D_k(uniform image) carries the uniform colour (content), the deviation carries what
    varies with token position; the reference mean replaces the colour content by the average natural-image content.
    """
    import torch
    parts, start = [], 0
    for _, h, w in grids:
        n = (h // 2) * (w // 2)
        block = blank[start:start + n]
        parts.append(block - block.mean(0, keepdim=True) + mean.to(block.device, block.dtype).reshape(1, -1))
        start += n
    if start != blank.shape[0]:
        raise ValueError('Token count does not match the image grids.')
    return torch.cat(parts, 0)


class NaturalOverride:
    """DeepStack replacement for one batch (global and region conditions).

    ``prepare(images, cases)`` must be called before every forward that carries DeepStack features; ``replace``
    consumes the prepared tensors, so a stale batch can never be reused.
    """

    def __init__(self, rt, condition: str, means: list | None = None):
        if condition not in CONDITIONS:
            raise ValueError(f'Unknown condition {condition!r}')
        if condition in ('mean', 'meanpos') and means is None:
            raise ValueError('The mean condition needs reference means.')
        self.rt, self.condition, self.means = rt, condition, means
        self.prepared = None
        self.calls = 0

    @property
    def image_derived(self) -> bool:
        return self.condition not in ('full', 'off', 'mean')

    @staticmethod
    def combine(full: list, blank: list, masks: list[np.ndarray]) -> list:
        """Per token: ``blank`` where the concatenated mask is True, ``full`` elsewhere."""
        import torch
        sel = torch.from_numpy(np.concatenate([m.reshape(-1) for m in masks]))
        out = []
        for f, b in zip(full, blank):
            if f.shape != b.shape or f.shape[0] != sel.numel():
                raise ValueError('Region mask does not match the DeepStack features.')
            s = sel.to(f.device)[:, None]
            out.append(torch.where(s, b.to(f.dtype), f))
        return out

    def region_masks(self, cases: list[dict]) -> list[np.ndarray]:
        masks = []
        for c in cases:
            n = (c['height'] // TOKEN_PIXELS) * (c['width'] // TOKEN_PIXELS)
            key = 'control' if self.condition == 'blank_ctrl' else 'region'
            if c.get(key) is None:
                raise ValueError(f"{c['id']} has no {key} tokens for {self.condition}")
            m = np.zeros(n, bool)
            m[c[key]] = True
            masks.append(~m if self.condition == 'blank_out' else m)
        return masks

    def prepare(self, images: list[Image.Image], cases: list[dict]) -> None:
        if not self.image_derived:
            return
        c = self.condition
        if c.startswith('blur'):
            sources = [im.filter(ImageFilter.GaussianBlur(radius=blur_sigma(c))) for im in images]
        else:
            sources = [blank_like(im) for im in images]
        feats, grids = deepstack_features(self.rt, sources)
        ref = [[1, cs['height'] // 16, cs['width'] // 16] for cs in cases]
        if grids != ref:
            raise ValueError(f'Processed grids {grids} differ from the stored image sizes {ref}.')
        if c == 'blankflip':
            feats = [flip_tokens(f, grids) for f in feats]
        elif c == 'meanpos':
            feats = [position_only(f, grids, m) for f, m in zip(feats, self.means)]
        elif c in REGION_CONDITIONS:
            full, _ = deepstack_features(self.rt, images)
            feats = self.combine(full, feats, self.region_masks(cases))
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


# ----------------------------------------------------------------------------- answering


def _gold(case: dict) -> str:
    return case['answers'][0]


def answer_batch(rt, images: list[Image.Image], cases: list[dict], budget: int, ids: dict, override) -> list[dict]:
    """Greedy answers plus first-step logits and the teacher-forced log-probability of the gold answer.

    The gold log-probability is computed in a second forward pass over prompt + gold answer (the model's chat
    template answer prefix is part of the prompt), summed over the gold answer tokens.
    """
    import torch
    from transformers import LogitsProcessorList
    from analysis_tools.common import qwen_runtime as qr
    texts = [qr.prompt_text(rt, 1, c['prompt'], False) for c in cases]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    for c, (_, h, w) in zip(cases, batch['image_grid_thw'].tolist()):
        if (h * 16, w * 16) != (c['height'], c['width']):
            raise ValueError(f"Processor resized {c['id']}")
    first = _first_step_processor()
    if override is not None:
        override.prepare(images, cases)
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
        z = first.first[i]
        rec = {'response': rt.tokenizer.decode(new, skip_special_tokens=True), 'n_tokens': len(new),
               'finish_reason': 'eos' if end is not None else 'length', 'max_new_tokens': budget,
               'batch_size': len(images)}
        if cases[i]['task'] == 'cub':
            rec['s'] = float(z[ids['Yes']] - z[ids['No']])
        elif cases[i]['task'] == 'mmvp':
            rec['s'] = float(z[ids['a']] - z[ids['b']])
        records.append(rec)
    # Teacher-forced gold answer for the short-answer tasks.
    if cases[0]['task'] in TF_TASKS:
        golds = [rt.tokenizer(_gold(c), add_special_tokens=False)['input_ids'] for c in cases]
        full_texts = [t + _gold(c) for t, c in zip(texts, cases)]
        tf = rt.processor(text=full_texts, images=images, padding=True, return_tensors='pt').to(rt.device)
        if override is not None:
            override.prepare(images, cases)
        keep = max(len(g) for g in golds) + 1         # only the answer positions need vocabulary logits
        with torch.inference_mode():
            logits = rt.model(**tf, logits_to_keep=keep).logits.float()
        logp = torch.log_softmax(logits[:, :-1], -1)
        target = tf['input_ids'][:, -keep + 1:]
        tok_lp = logp.gather(-1, target[..., None])[..., 0]
        for i, g in enumerate(golds):
            n = len(g)
            if tf['input_ids'][i, -n:].tolist() != g:
                # Tokens merged across the prompt/answer boundary; the value would be ill-defined, so it is
                # recorded as missing and counted in the analysis.
                records[i]['gold_logp'] = None
                continue
            records[i]['gold_logp'] = float(tok_lp[i, -n:].sum())
            records[i]['gold_ntok'] = n
    return records


def load_split(model: str, split: str):
    """Load the model across the visible GPUs with per-GPU caps, e.g. split='12,7' (GiB), when no single GPU has
    room. Mirrors ``qwen_runtime.load`` except for the device map; inputs go to cuda:0."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from analysis_tools.common import qwen_runtime as qr
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    checkpoint = resolve_model(model, load_paths()['HF_HUB_CACHE'])
    caps = {i: f'{float(g)}GiB' for i, g in enumerate(split.split(','))}
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    net = AutoModelForImageTextToText.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16,
                                                      device_map='auto', max_memory=caps,
                                                      attn_implementation='sdpa').eval()
    net.requires_grad_(False)
    return qr.Runtime(model, checkpoint, net, processor, 'cuda:0')


def run(output: Path, ablation: Path, model: str, condition: str, tasks: list[str], batch_size: int,
        docvqa_n: int | None = None, split: str | None = None) -> None:
    """``docvqa_n`` limits DocVQA to its first n items (by id) in every condition except Full (compute budget)."""
    from analysis_tools.common import qwen_runtime as qr
    cases = [c for c in read_jsonl(output / 'cases.jsonl') if c['task'] in tasks]
    if docvqa_n is not None and condition != 'full':
        keep = set(sorted(c['id'] for c in cases if c['task'] == 'docvqa')[:docvqa_n])
        cases = [c for c in cases if c['task'] != 'docvqa' or c['id'] in keep]
    if condition in REGION_CONDITIONS:
        cases = [c for c in cases if c['task'] in REGION_TASKS and
                 c.get('control' if condition == 'blank_ctrl' else 'region') is not None]
    run_dir = output / 'runs' / slug(model)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f'{condition}.jsonl'
    done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
    pending = [c for c in cases if c['id'] not in done]
    print(f'{model} {condition}: {len(done)} done, {len(pending)} to run', flush=True)
    if not pending:
        return
    rt = load_split(model, split) if split else qr.load(model, device='cuda:0')
    if rt.family != 'qwen3_vl' and condition != 'full':
        raise ValueError(f'{model} has no DeepStack; only the full condition applies.')
    means = load_means(ablation, model) if condition in ('mean', 'meanpos') else None
    _check_provenance(run_dir / f'{condition}.json', {
        'model': qr.describe(rt), 'condition': condition, 'decoding': 'greedy', 'thinking': False,
        'budgets': BUDGET, 'means_source': ablation.name if condition in ('mean', 'meanpos') else None})
    override = NaturalOverride(rt, condition, means) if rt.family == 'qwen3_vl' else None
    ids = {w: qr.token_id(rt, w) for w in ('Yes', 'No', 'a', 'b')}
    batches = []
    for task in TASKS:
        rows = sorted([c for c in pending if c['task'] == task], key=lambda c: (c['height'], c['width']))
        batches += [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    n_done = 0
    ctx = deepstack_override(rt, override) if override is not None else _null()
    with path.open('a') as sink, ctx:
        for batch in batches:
            images = []
            for c in batch:
                p = output / c['image']
                if sha256_file(p) != c['image_sha256']:
                    raise ValueError(f'Image hash mismatch for {p}')
                images.append(Image.open(p).convert('RGB'))
            calls = override.calls if override is not None else 0
            records = answer_batch(rt, images, batch, BUDGET[batch[0]['task']], ids, override)
            expected = 2 if batch[0]['task'] in TF_TASKS else 1
            if override is not None and override.calls != calls + expected:
                raise RuntimeError(f'DeepStack override fired {override.calls - calls} times, expected {expected}')
            for c, rec in zip(batch, records):
                sink.write(json.dumps({'id': c['id'], 'condition': condition, **rec}) + '\n')
            sink.flush()
            n_done += len(batch)
            if n_done // 200 != (n_done - len(batch)) // 200:
                print(f'{condition}: {n_done}/{len(pending)} {dt.datetime.now():%H:%M:%S}', flush=True)
    with (run_dir / 'run_log.jsonl').open('a') as log:
        log.write(json.dumps({'condition': condition, 'n_run': n_done, 'batch_size': batch_size,
                              'end': dt.datetime.now().isoformat(timespec='seconds')}) + '\n')


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------------------- scoring


def norm_answer(text: str) -> str:
    text = text.strip().split('\n')[0].lower().strip().rstrip('.')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def anls(pred: str, golds: list[str], tau: float = 0.5) -> float:
    """Average normalized Levenshtein similarity (DocVQA metric) against the best gold answer."""
    def lev(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]
    best = 0.0
    p = pred.strip().split('\n')[0].lower().strip()
    for g in golds:
        g = g.lower().strip()
        d = lev(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - d if d < tau else 0.0)
    return best


def parse_letter(text: str) -> str | None:
    m = re.search(r'\(?\b([ab])\b\)?', text.strip().lower())
    return m.group(1) if m else None


def score_case(case: dict, record: dict) -> dict:
    task, resp = case['task'], record['response']
    row = {'id': case['id'], 'task': task, 'valid': record['finish_reason'] == 'eos'}
    if task == 'cub':
        row.update(s=record['s'], correct=(record['s'] > 0) == (case['answers'][0] == 'Yes'))
    elif task == 'mmvp':
        letter = parse_letter(resp)
        row.update(pred=letter, s=record['s'], correct=letter == case['answers'][0])
    else:
        pred = norm_answer(resp)
        row.update(pred=pred, correct=any(norm_answer(a) == pred for a in case['answers']),
                   anls=anls(resp, case['answers']), gold_logp=record.get('gold_logp'))
    return row


def score(output: Path) -> None:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    for path in sorted((output / 'runs').glob('*/*.jsonl')):
        if path.name == 'run_log.jsonl' or path.name.endswith('.scores.jsonl'):
            continue
        records = read_jsonl(path)
        if len({r['id'] for r in records}) != len(records):
            raise ValueError(f'Duplicate results in {path}')
        write_jsonl(path.parent / f'{path.stem}.scores.jsonl', [score_case(cases[r['id']], r) for r in records])
        print(f'{path.parent.name}/{path.stem}: {len(records)} scored')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('build', 'run', 'score'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ablation', type=Path)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--data-public', type=Path)
    p.add_argument('--mmvp-csv', type=Path)
    p.add_argument('--model', default='Qwen3-VL-8B-Instruct')
    p.add_argument('--condition', default='full', choices=CONDITIONS)
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--docvqa-n', type=int, default=None)
    p.add_argument('--split', default=None, help='per-GPU GiB caps for a multi-GPU load, e.g. 12,7')
    a = p.parse_args()
    if a.command == 'build':
        build(a.output, a.data_dir, a.data_public, a.mmvp_csv, a.seed)
    elif a.command == 'run':
        run(a.output, a.ablation, a.model, a.condition, a.tasks, a.batch_size, a.docvqa_n, a.split)
    else:
        score(a.output)


if __name__ == '__main__':
    main()
