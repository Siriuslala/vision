"""Experiment 1.6: ViT PE scaling on grids, BabyVision and RefCOCO, plus T0 counting with image variation.

Part A tests whether scaling the learned ViT position embedding of Qwen3.5 (experiment 3.7) changes
behaviour beyond blank canvases:
  grid        stratified subset of the 1.3 formal cases (T4/T5/T6 row/column and point at 6, 10, 20;
              T1/T2 as recognition controls), direct answers, greedy
  explicit    1.3 think-then-answer cases (T4/T5/T6 row/column at 6 and 20, thinking off), greedy
  babyvision  Visual Tracking subtypes and Find the different, same sampling settings as the existing
              Qwen3.5-9B thinking-off run, identical batch composition and seeds across PE conditions
  refcoco     RefCOCO test-split subset (bbox and point prompts), greedy

Part B builds a T0 (row/column counting) dataset with genuine per-image variation under DATA_DIR and
re-runs T0 under the same PE conditions (stages t0 and t0_explicit).

Commands: build-grid | build-t0 | build-refcoco | build-babyvision | run | spotcheck
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import (
    COORD_NOTE, JSON_ONLY, T0_QUERY, T0_SUFFIX, read_jsonl, sha256_file, write_jsonl,
)
from analysis_tools.common.stimuli import (
    DEVELOPMENT_TEMPLATES, GENERATOR_VERSION, TOKEN_PIXELS, WHITE, StimulusSpec, canvas_layout, generate,
    lab_to_srgb, render_instance,
)

PE_RATIO = 2.756   # Qwen3-VL-8B / Qwen3.5-9B PE norm ratio (experiment 3.7, vit_layout_run.PE_RATIO)
PE_CONDITIONS = {'x1': 1.0, 'x2': 2.0, 'x2.76': PE_RATIO}
CODE_FILES = ('pe_scaling_t0.py', 'pe_scaling_t0_analysis.py')

# ----------------------------------------------------------------------------- shared helpers


def code_digest() -> dict:
    here = Path(__file__).parent
    return {f: hashlib.sha256((here / f).read_bytes()).hexdigest() for f in CODE_FILES if (here / f).exists()}


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('a') as sink:
        for r in rows:
            sink.write(json.dumps(r) + '\n')


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = [r['id'] for r in read_jsonl(path)]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate records in {path}')
    return set(ids)


# ----------------------------------------------------------------------------- Part A: grid subset

GRID_TASKS = ('T4', 'T5', 'T6')
GRID_FORMATS = ('row_column', 'point')
GRID_SIZES = (6, 10, 20)


def select_grid_cases(formal_cases: list[dict], per_combo: int, control_per_combo: int, seed: int) -> list[dict]:
    """Stratified subset of the 1.3 main and explicit cases (primary wording only).

    T4/T5: per (task, format, grid), `per_combo` cases per (kind, level) combination.
    T6: all cases (48 per format and grid).  T1/T2: `control_per_combo` cases per (task, kind, level, variant),
    pooled over grids (T1/T2 images are single 256x256 crops whose content does not depend on the grid).
    Explicit: all 1.3 explicit T4/T5/T6 row/column cases (think-then-answer, thinking off).
    """
    rng = random.Random(seed)
    groups: dict = {}
    out = []
    for c in formal_cases:
        if c['wording'] != 'primary' or c['thinking']:
            continue
        if c['stage'] == 'explicit' and c['task'] in GRID_TASKS:
            out.append({**c, 'subset': 'explicit'})
            continue
        if c['stage'] != 'main':
            continue
        if c['task'] in GRID_TASKS and c['format'] in GRID_FORMATS and c['grid'] in GRID_SIZES:
            if c['task'] == 'T6':
                out.append({**c, 'subset': 'grid'})
            else:
                groups.setdefault(('loc', c['task'], c['format'], c['grid'], c['kind'], c['level']), []).append(c)
        elif c['task'] in ('T1', 'T2'):
            groups.setdefault(('ctl', c['task'], c['kind'], c['level'], c['variant']), []).append(c)
    for key in sorted(groups, key=str):
        pool = sorted(groups[key], key=lambda c: c['id'])
        n = per_combo if key[0] == 'loc' else control_per_combo
        if len(pool) < n:
            raise ValueError(f'Stratum {key} has {len(pool)} cases, fewer than {n}.')
        out.extend({**c, 'subset': 'grid'} for c in rng.sample(pool, n))
    return out


def build_grid(output: Path, formal: Path, per_combo: int, control_per_combo: int, seed: int) -> None:
    cases = select_grid_cases(read_jsonl(formal / 'cases.jsonl'), per_combo, control_per_combo, seed)
    for c in cases:
        c['image'] = str((formal / c['image']).relative_to(formal))
    root = output / 'grid'
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / 'cases.jsonl', cases)
    counts = Counter((c['subset'], c['task'], c['format'], c['grid']) for c in cases)
    meta = {'formal': str(formal.name), 'per_combo': per_combo, 'control_per_combo': control_per_combo, 'seed': seed,
            'n_cases': len(cases), 'counts': {'|'.join(map(str, k)): v for k, v in sorted(counts.items(), key=str)}}
    (root / 'build.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'n_cases': len(cases)}))


# ----------------------------------------------------------------------------- Part B: T0 dataset

T0_DATASET_VERSION = 't0-count-variation-v1'
# (rows, cols, cell); the 1.3 square grids, the non-square grids of the 1.3 supplement, and two 20-instance
# axes crossed with 6 rows, plus the fixed-canvas 24x24 grid (cell 32) where 1.3 supplement found 25x25 answers.
T0_LAYOUTS = ((3, 3, 64), (6, 6, 64), (10, 10, 64), (20, 20, 64), (4, 8, 64), (8, 4, 64), (6, 20, 64),
              (20, 6, 64), (10, 20, 64), (20, 10, 64), (24, 24, 32))
# V0 reproduces the 1.3 T0 images (one image per template); V1-V3 add variation cumulatively.
T0_VARIATIONS = {
    'V0_fixed': dict(jitter=False, color=False, mixed=False),
    # Control: the canvas and grid origin of the jittered layout (32-pixel margin, 16-pixel origin shift) with
    # zero per-instance offsets; separates the effect of jitter from the layout change that jitter requires.
    'V0s_shifted': dict(jitter=False, color=False, mixed=False),
    'V1_jitter': dict(jitter=True, color=False, mixed=False),
    'V2_jitter_color': dict(jitter=True, color=True, mixed=False),
    'V3_jitter_color_mixed': dict(jitter=True, color=True, mixed=True),
}
COLOR_L = (15.0, 45.0)        # CIELAB lightness range of the per-image stroke colour
COLOR_C = (25.0, 60.0)        # CIELAB chroma range


def layout_name(rows: int, cols: int, cell: int) -> str:
    return f'{rows}x{cols}_c{cell}'


def sample_stroke_color(rng: np.random.Generator) -> tuple[int, int, int]:
    """Dark saturated colour, uniform hue; resampled until inside the sRGB gamut."""
    for _ in range(1000):
        L, C, h = rng.uniform(*COLOR_L), rng.uniform(*COLOR_C), rng.uniform(0, 2 * math.pi)
        try:
            return lab_to_srgb((L, C * math.cos(h), C * math.sin(h)))
        except ValueError:
            continue
    raise RuntimeError('No in-gamut colour found.')


def render_t0(rows: int, cols: int, cell: int, jitter: int, color: tuple[int, int, int],
              templates: list[str], seed: int, offsets: bool = True) -> tuple[Image.Image, dict]:
    """Normal (anomaly-free) grid with per-instance template, jitter and one stroke colour.

    The canvas and grid origin follow stimuli.canvas_layout for a jittered spec, so the canvas stays aligned to the
    32-pixel token grid and every jittered instance stays inside the canvas.  Jitter offsets are drawn per instance
    from the integers in [-jitter, jitter]; offsets=False keeps the jittered layout with all offsets zero.
    """
    spec = StimulusSpec(templates[0], rows, cols, cell=cell, jitter=jitter, seed=seed)
    width, height, ox, oy = canvas_layout(spec)
    canvas = np.full((height, width, 3), WHITE, np.uint8)
    rng = np.random.default_rng(seed)
    tiles: dict[str, np.ndarray] = {}
    instances = []
    for r in range(rows):
        for c in range(cols):
            t = templates[r * cols + c]
            if t not in tiles:
                tile = render_instance(t, cell=cell)
                mask = (tile < WHITE).any(-1)
                tile = np.full_like(tile, WHITE)
                tile[mask] = color
                tiles[t] = tile
            tile = tiles[t]
            jx, jy = rng.integers(-jitter, jitter + 1, 2).tolist() if jitter and offsets else (0, 0)
            x0, y0 = ox + c * cell + jx, oy + r * cell + jy
            region = canvas[y0:y0 + cell, x0:x0 + cell]
            np.minimum(region, tile, out=region)
            dark = np.argwhere((tile < WHITE).any(-1))
            (ty0, tx0), (ty1, tx1) = dark.min(0), dark.max(0) + 1
            box = [float(x0 + tx0), float(y0 + ty0), float(x0 + tx1), float(y0 + ty1)]
            instances.append({'row': r, 'col': c, 'template': t, 'jitter': [jx, jy], 'tight_box': box,
                              'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]})
    image = Image.fromarray(canvas)
    meta = {'canvas': [width, height], 'grid_origin': [ox, oy], 'cell': cell, 'rows': rows, 'cols': cols,
            'jitter': jitter, 'stroke_rgb': list(color), 'instances': instances}
    return image, meta


def t0_seed(base: int, *parts) -> int:
    digest = hashlib.sha256('|'.join(map(str, (base, *parts))).encode()).digest()
    return int.from_bytes(digest[:4], 'big')


def build_t0(dataset: Path, n_per_variation: int, seed: int) -> None:
    """Write images, metadata.jsonl and manifest.json under the dataset directory (refuses to overwrite)."""
    if (dataset / 'manifest.json').exists():
        raise FileExistsError(f'{dataset} already holds a dataset; choose a new version directory.')
    (dataset / 'images').mkdir(parents=True, exist_ok=True)
    records = []
    for rows, cols, cell in T0_LAYOUTS:
        lname = layout_name(rows, cols, cell)
        for vname, v in T0_VARIATIONS.items():
            if vname == 'V0_fixed':
                for template in DEVELOPMENT_TEMPLATES:   # stimuli-v1 normal grid, identical to the 1.3 T0 images
                    image, m = generate(StimulusSpec(template, rows, cols, cell=cell))
                    meta = {'canvas': m['canvas'], 'grid_origin': m['grid_origin'], 'cell': cell, 'rows': rows,
                            'cols': cols, 'jitter': 0, 'stroke_rgb': [0, 0, 0],
                            'instances': [{'row': i['row'], 'col': i['col'], 'template': template, 'jitter': [0, 0],
                                           'tight_box': i['tight_box'], 'center': i['center']} for i in m['instances']]}
                    records.append(_save_t0(dataset, image, meta, lname, vname, template, None))
                continue
            if vname == 'V0s_shifted':
                for template in DEVELOPMENT_TEMPLATES:
                    image, meta = render_t0(rows, cols, cell, cell // 8, (0, 0, 0), [template] * (rows * cols), 0,
                                            offsets=False)
                    meta['jitter'] = 0
                    records.append(_save_t0(dataset, image, meta, lname, vname, template, None))
                continue
            for k in range(n_per_variation):
                s = t0_seed(seed, lname, vname, k)
                rng = np.random.default_rng(s)
                if v['mixed']:
                    templates = [str(t) for t in rng.choice(DEVELOPMENT_TEMPLATES, rows * cols)]
                else:
                    templates = [DEVELOPMENT_TEMPLATES[k % len(DEVELOPMENT_TEMPLATES)]] * (rows * cols)
                color = sample_stroke_color(rng) if v['color'] else (0, 0, 0)
                jitter = cell // 8 if v['jitter'] else 0
                image, meta = render_t0(rows, cols, cell, jitter, color, templates, s + 1)
                records.append(_save_t0(dataset, image, meta, lname, vname,
                                        'mixed' if v['mixed'] else templates[0], s))
    write_jsonl(dataset / 'metadata.jsonl', records)
    manifest = {
        'name': T0_DATASET_VERSION, 'generator': GENERATOR_VERSION, 'experiment': '1.6',
        'description': 'Anomaly-free grids for row/column counting (T0) with per-image variation.',
        'seed': seed, 'n_per_variation': n_per_variation, 'layouts': [list(x) for x in T0_LAYOUTS],
        'variations': T0_VARIATIONS, 'control_rule': 'V0s_shifted: layout of the jittered spec, zero offsets',
        'jitter_rule': 'per instance, uniform integers in [-cell/8, cell/8] pixels',
        'color_rule': {'space': 'CIELAB D65', 'L': COLOR_L, 'chroma': COLOR_C, 'hue': 'uniform'},
        'templates': list(DEVELOPMENT_TEMPLATES), 'n_images': len(records),
        'metadata_sha256': sha256_file(dataset / 'metadata.jsonl'),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (dataset / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    counts = Counter((r['layout'], r['variation']) for r in records)
    print(json.dumps({'n_images': len(records), 'distinct_sha256': len({r['sha256'] for r in records}),
                      'per_cell': sorted(set(counts.values()))}))


def _save_t0(dataset: Path, image: Image.Image, meta: dict, lname: str, vname: str, template: str,
             seed: int | None) -> dict:
    stem = f'{lname}_{vname.split("_")[0]}_{template}' if seed is None else f'{lname}_{vname.split("_")[0]}_{seed:010d}'
    rel = f'images/{stem}.png'
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    data = buffer.getvalue()
    (dataset / rel).write_bytes(data)
    return {'id': stem, 'image': rel, 'sha256': hashlib.sha256(data).hexdigest(), 'layout': lname,
            'variation': vname, 'template': template, 'seed': seed, **meta}


T0_EXPLICIT = (' First describe step by step how you count the rows and the columns. Then give the final answer on '
               'the last line as {"rows": R, "columns": C}.')


def t0_cases(dataset: Path, explicit_per_layout: int) -> list[dict]:
    """Direct T0 cases for every image; think-then-answer cases for the first images of V3 in each layout."""
    cases = []
    seen: Counter = Counter()
    for m in read_jsonl(dataset / 'metadata.jsonl'):
        truth = {'rows': m['rows'], 'cols': m['cols'], 'canvas': m['canvas']}
        base = {'image': m['image'], 'image_sha256': m['sha256'], 'layout': m['layout'], 'variation': m['variation'],
                'template': m['template'], 'grid': f"{m['rows']}x{m['cols']}", 'format': 'count', 'task': 'T0',
                'truth': truth, 'thinking': False}
        cases.append({**base, 'id': f"t0|{m['id']}", 'stage': 't0', 'prompt': T0_QUERY[0] + T0_SUFFIX + JSON_ONLY})
        if m['variation'] == 'V3_jitter_color_mixed' and seen[m['layout']] < explicit_per_layout:
            seen[m['layout']] += 1
            cases.append({**base, 'id': f"t0x|{m['id']}", 'stage': 'explicit', 'prompt': T0_QUERY[0] + T0_EXPLICIT})
    return cases


# ----------------------------------------------------------------------------- RefCOCO test subset

STRATA = (('lt1', 0.0, 0.01), ('1to5', 0.01, 0.05), ('5to10', 0.05, 0.10), ('ge10', 0.10, math.inf))
REFCOCO_BBOX = ('Locate the object described as: "{expr}". Give its tight bounding box. '
                'Return only [{{"bbox_2d": [x_min, y_min, x_max, y_max]}}].' + COORD_NOTE + JSON_ONLY)
REFCOCO_POINT = ('Locate the object described as: "{expr}". Give the center of its bounding box. '
                 'Return only [{{"point_2d": [x, y]}}].' + COORD_NOTE + JSON_ONLY)


def build_refcoco(refcoco: Path, output: Path, per_stratum: int, seed: int) -> None:
    """One expression per COCO image from the RefCOCO test split, sampled per box-area stratum.

    Same rule as experiment 4.2 (val split): first referring expression, smallest stratum sampled first.
    Each image gets a bbox and a point prompt.
    """
    import pyarrow.parquet as pq

    shards = sorted((refcoco / 'data').glob('test-*.parquet'))
    if not shards:
        raise FileNotFoundError('Missing RefCOCO test shards.')
    rows = []
    for shard in shards:
        table = pq.read_table(shard, columns=['question_id', 'answer', 'bbox', 'file_name', 'image'])
        for r in table.to_pylist():
            image = Image.open(io.BytesIO(r['image']['bytes']))
            width, height = image.size
            x, y, w, h = r['bbox']
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
                raise ValueError(f'Invalid RefCOCO bbox {r["bbox"]} in {r["file_name"]}')
            area = w * h / (width * height)
            stratum = next(n for n, lo, hi in STRATA if lo <= area < hi)
            rows.append({'question_id': r['question_id'], 'coco_image': r['file_name'].rsplit('_', 1)[0],
                         'file_name': r['file_name'], 'expression': r['answer'][0], 'bbox': [x, y, x + w, y + h],
                         'width': width, 'height': height, 'area': area, 'stratum': stratum,
                         'bytes': r['image']['bytes'], 'shard': shard.name})
    rng = random.Random(seed)
    rng.shuffle(rows)
    root = output / 'refcoco'
    (root / 'images').mkdir(parents=True, exist_ok=True)
    used, cases = set(), []
    for name, *_ in STRATA:
        taken = 0
        for r in (r for r in rows if r['stratum'] == name):
            if taken >= per_stratum or r['coco_image'] in used:
                continue
            used.add(r['coco_image'])
            rel = f'images/{r["question_id"]}.jpg'
            (root / rel).write_bytes(r['bytes'])
            info = {k: r[k] for k in ('question_id', 'coco_image', 'file_name', 'expression', 'bbox', 'width',
                                      'height', 'area', 'stratum', 'shard')}
            for fmt, template in (('bbox', REFCOCO_BBOX), ('point', REFCOCO_POINT)):
                cases.append({'id': f'refcoco|{r["question_id"]}|{fmt}', 'image': rel, 'format': fmt,
                              'image_sha256': hashlib.sha256(r['bytes']).hexdigest(),
                              'prompt': template.format(expr=r['expression']), **info})
            taken += 1
    write_jsonl(root / 'cases.jsonl', cases)
    meta = {'source': 'lmms-lab RefCOCO test parquet', 'n_annotations': len(rows), 'per_stratum': per_stratum,
            'seed': seed, 'sampled': dict(Counter(c['stratum'] for c in cases if c['format'] == 'bbox')),
            'rule': 'one expression per COCO image; first referring expression; smallest stratum first'}
    (root / 'build.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta))


# ----------------------------------------------------------------------------- BabyVision subset

def babyvision_selection(rows: list[dict]) -> list[dict]:
    """All Visual Tracking items (maze and line-tracing subtypes) plus Find the different."""
    return [r for r in rows if r['type'] == 'Visual Tracking' or r['subtype'] == 'Find the different']


def build_babyvision(data_root: Path, reference_run: Path, output: Path) -> None:
    from eva.benchmark import load_cases, prompt_for
    rows, manifest = load_cases(data_root, 'babyvision', 'train')
    selected = babyvision_selection(rows)
    reference = {r['taskId']: r for r in read_jsonl(reference_run)}
    cases = []
    for r in selected:
        ref = reference[r['taskId']]
        prompt = prompt_for(r, 'babyvision')
        if prompt != ref['prompt']:
            raise ValueError(f'Prompt of task {r["taskId"]} differs from the reference run.')
        cases.append({'id': f'bv|{r["taskId"]}', 'taskId': r['taskId'], 'type': r['type'], 'subtype': r['subtype'],
                      'ansType': r['ansType'], 'image_path_rel': r['image'], 'prompt': prompt,
                      'image_sha256': sha256_file(Path(r['image_path'])),
                      'reference_tokens': ref['generated_tokens'], 'reference_correct': ref['score']['correct']})
    # Batches group items of similar reference length to limit padding waste; composition is fixed here
    # and reused for every PE condition and seed, so the per-batch seeds pair the conditions.
    cases.sort(key=lambda c: (c['reference_tokens'], c['taskId']))
    root = output / 'babyvision'
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / 'cases.jsonl', cases)
    meta = {'manifest_sha256': sha256_file(manifest), 'reference_run': reference_run.name, 'n_cases': len(cases),
            'subtypes': dict(Counter(c['subtype'] for c in cases))}
    (root / 'build.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta))


# ----------------------------------------------------------------------------- inference

def generate_scored(rt, items: list[tuple[Image.Image, str]], max_new_tokens: int, *, resize_ok: bool,
                    sampling: dict | None = None, yes_no: bool = False) -> list[dict]:
    """Left-padded batched decoding of single-image prompts (thinking off).

    Greedy when `sampling` is None; otherwise `sampling` holds do_sample/temperature/top_k/top_p and the caller
    sets the RNG seed.  With yes_no=True, the first-step logits give s = z_Yes - z_No per item (the 1.3 decision
    score of direct yes/no answers).  resize_ok=False raises if the processor resized a synthetic stimulus.
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList
    from analysis_tools.common import qwen_runtime as qr

    class FirstStep(LogitsProcessor):
        def __init__(self):
            self.first = None

        def __call__(self, input_ids, scores):
            if self.first is None:
                self.first = scores.float().clone()
            return scores

    texts = [qr.prompt_text(rt, 1, p, False) for _, p in items]
    images = [im for im, _ in items]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    grids = batch['image_grid_thw'].tolist()
    if not resize_ok:
        for im, (_, h, w) in zip(images, grids):
            if (h * 16, w * 16) != (im.height, im.width):
                raise ValueError(f'Processor resized a {im.width}x{im.height} image to {w * 16}x{h * 16}.')
    first = FirstStep()
    kwargs = dict(do_sample=False, temperature=None, top_p=None, top_k=None) if sampling is None else dict(sampling)
    with torch.inference_mode():
        out = rt.model.generate(**batch, max_new_tokens=max_new_tokens, logits_processor=LogitsProcessorList([first]),
                                **kwargs)
    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = rt.tokenizer.pad_token_id
    yes_id, no_id = (qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No')) if yes_no else (None, None)
    results = []
    for i, row in enumerate(out[:, batch['input_ids'].shape[1]:].tolist()):
        end = next((j + 1 for j, t in enumerate(row) if t in eos), None)
        new = row[:end] if end is not None else [t for t in row if t != pad]
        rec = {'response': rt.tokenizer.decode(new, skip_special_tokens=True), 'n_tokens': len(new),
               'finish_reason': 'eos' if end is not None else 'length', 'max_new_tokens': max_new_tokens,
               'processed_hw': [grids[i][1] * 16, grids[i][2] * 16]}
        logits = first.first[i]
        rec['first_top_token'] = rt.tokenizer.decode([int(logits.argmax())])
        if yes_no:
            z_yes, z_no = float(logits[yes_id]), float(logits[no_id])
            rec.update(z_yes=z_yes, z_no=z_no, s=z_yes - z_no)
        results.append(rec)
    return results


def pe_context(rt, scale: float):
    if scale == 1.0:
        return nullcontext()
    from analysis_tools.position import vit_layout as vl
    return vl.vit_edit(rt, vl.ViTEdit(pe_scale=scale))


def load_image(path: Path, sha: str) -> Image.Image:
    if sha256_file(path) != sha:
        raise ValueError(f'Image hash mismatch for {path}')
    return Image.open(path).convert('RGB')


def stage_spec(stage: str, args) -> tuple[Path, list[dict], callable]:
    """(results path, cases, image resolver) of one stage."""
    out = args.output
    if stage in ('grid', 'grid_explicit'):
        cases = [c for c in read_jsonl(out / 'grid' / 'cases.jsonl')
                 if c['subset'] == ('explicit' if stage == 'grid_explicit' else 'grid')]
        return out / 'grid' / f'results_{stage}.jsonl', cases, lambda c: args.formal / c['image']
    if stage in ('t0', 't0_explicit'):
        cases = [c for c in t0_cases(args.t0_dataset, args.t0_explicit_per_layout)
                 if c['stage'] == ('explicit' if stage == 't0_explicit' else 't0')]
        return out / 't0' / f'results_{stage}.jsonl', cases, lambda c: args.t0_dataset / c['image']
    if stage == 'refcoco':
        return out / 'refcoco' / 'results.jsonl', read_jsonl(out / 'refcoco' / 'cases.jsonl'), \
            lambda c: out / 'refcoco' / c['image']
    raise ValueError(f'Unknown stage {stage!r}')


def budget(stage: str, case: dict, args) -> int:
    if stage.endswith('explicit'):
        return args.max_new_tokens_explicit
    return args.max_new_tokens_binary if case['format'] == 'yes_no' else args.max_new_tokens_locate


def run_greedy(rt, stage: str, args) -> None:
    path, cases, resolve = stage_spec(stage, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(path)
    shard_i, shard_k = map(int, args.shard.split('/'))
    natural = stage == 'refcoco'
    for cname in args.conditions:
        scale = PE_CONDITIONS[cname]
        todo = [c for c in cases if f"{c['id']}|{cname}" not in finished]
        todo = todo[shard_i::shard_k]
        # Group by token budget, then by image size, so batches pad little.
        todo.sort(key=lambda c: (budget(stage, c, args), c.get('format') == 'yes_no', c.get('height', 0),
                                 c.get('width', 0), str(c.get('grid')), c['id']))
        bs = args.batch_size_explicit if stage.endswith('explicit') else args.batch_size
        i = 0
        while i < len(todo):
            head = todo[i]
            key = (budget(stage, head, args), head['format'] == 'yes_no')
            chunk = []
            for c in todo[i:i + bs]:      # contiguous run with the same budget and decision type
                if (budget(stage, c, args), c['format'] == 'yes_no') != key:
                    break
                chunk.append(c)
            i += len(chunk)
            items = [(load_image(resolve(c), c['image_sha256']), c['prompt']) for c in chunk]
            with pe_context(rt, scale):
                recs = generate_scored(rt, items, budget(stage, head, args), resize_ok=natural,
                                       yes_no=head['format'] == 'yes_no')
            append_jsonl(path, [{'id': f"{c['id']}|{cname}", 'case_id': c['id'], 'condition': cname, 'scale': scale,
                                 'batch_size': len(chunk), **r} for c, r in zip(chunk, recs)])
        print(f'{stage} {cname}: done {len(todo)}', flush=True)


def run_babyvision(rt, args) -> None:
    from transformers import set_seed
    from eva.decoding import sampling_seed

    root = args.output / 'babyvision'
    cases = read_jsonl(root / 'cases.jsonl')
    path = root / 'results.jsonl'
    finished = done_ids(path)
    sampling = dict(do_sample=True, temperature=args.bv_temperature, top_k=args.bv_top_k, top_p=args.bv_top_p)
    batches = [cases[i:i + args.bv_batch_size] for i in range(0, len(cases), args.bv_batch_size)]
    shard_i, shard_k = map(int, args.shard.split('/'))
    jobs = [(seed, cname, b) for seed in args.bv_seeds for cname in args.conditions for b in range(len(batches))]
    for seed, cname, b in jobs[shard_i::shard_k]:
        batch = batches[b]
        if all(f"{c['id']}|{cname}|s{seed}" in finished for c in batch):
            continue
        if any(f"{c['id']}|{cname}|s{seed}" in finished for c in batch):
            raise ValueError(f'Batch {b} of {cname} seed {seed} is partially written; remove it and rerun.')
        items = [(load_image(args.data_root / 'BabyVision' / c['image_path_rel'], c['image_sha256']), c['prompt'])
                 for c in batch]
        batch_seed = sampling_seed(seed, b)
        set_seed(batch_seed)
        with pe_context(rt, PE_CONDITIONS[cname]):
            recs = generate_scored(rt, items, args.bv_max_new_tokens, resize_ok=True, sampling=sampling)
        append_jsonl(path, [{'id': f"{c['id']}|{cname}|s{seed}", 'case_id': c['id'], 'taskId': c['taskId'],
                             'condition': cname, 'scale': PE_CONDITIONS[cname], 'seed': seed, 'batch_index': b,
                             'batch_seed': batch_seed, 'batch_size': len(batch), **r} for c, r in zip(batch, recs)])
        print(f'babyvision {cname} seed {seed} batch {b}/{len(batches)}: '
              f'{max(r["n_tokens"] for r in recs)} tokens max', flush=True)


def write_run_meta(rt, args, stage: str) -> None:
    from analysis_tools.common import qwen_runtime as qr
    root = args.output / 'run_meta'
    root.mkdir(parents=True, exist_ok=True)
    meta = {'stage': stage, 'model': qr.describe(rt), 'conditions': {c: PE_CONDITIONS[c] for c in args.conditions},
            'batch_size': args.batch_size, 'batch_size_explicit': args.batch_size_explicit,
            'bv_batch_size': args.bv_batch_size, 'max_new_tokens': {
                'binary': args.max_new_tokens_binary, 'locate': args.max_new_tokens_locate,
                'explicit': args.max_new_tokens_explicit, 'babyvision': args.bv_max_new_tokens},
            'babyvision_sampling': {'temperature': args.bv_temperature, 'top_k': args.bv_top_k,
                                    'top_p': args.bv_top_p, 'seeds': args.bv_seeds,
                                    'seed_policy': 'eva.decoding.sampling_seed(seed, batch_index)'},
            'shard': args.shard, 'code_sha256': code_digest(), 'thinking': False}
    append_jsonl(root / f'{stage}.jsonl', [meta])


def run(args) -> None:
    from analysis_tools.common import qwen_runtime as qr
    rt = qr.load(args.model, attn=args.attn)
    for stage in args.stages:
        write_run_meta(rt, args, stage)
        if stage == 'babyvision':
            run_babyvision(rt, args)
        else:
            run_greedy(rt, stage, args)


def spotcheck(args) -> None:
    """Batched x1 answers of the grid stage against the 1.3 formal answers (batch size 1, unpadded)."""
    formal = {r['id']: r for r in read_jsonl(args.formal / 'results_main.jsonl')}
    formal.update({r['id']: r for r in read_jsonl(args.formal / 'results_explicit.jsonl')})
    rows = []
    for stage in ('grid', 'grid_explicit'):
        path = args.output / 'grid' / f'results_{stage}.jsonl'
        if not path.exists():
            continue
        for r in read_jsonl(path):
            if r['condition'] != 'x1':
                continue
            f = formal[r['case_id']]
            rows.append({'id': r['case_id'], 'stage': stage, 'same_text': r['response'].strip() == f['response'].strip(),
                         'batch_size': r['batch_size']})
    out = {'n': len(rows), 'same_text': sum(r['same_text'] for r in rows),
           'by_stage': {s: [sum(r['same_text'] for r in rows if r['stage'] == s), sum(r['stage'] == s for r in rows)]
                        for s in ('grid', 'grid_explicit')}}
    (args.output / 'spotcheck.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('build-grid', 'build-t0', 'build-refcoco', 'build-babyvision', 'run',
                                       'spotcheck'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--formal', type=Path, help='1.3 formal run directory (read only)')
    p.add_argument('--t0-dataset', type=Path)
    p.add_argument('--refcoco', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--babyvision-reference', type=Path)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--per-combo', type=int, default=12)
    p.add_argument('--control-per-combo', type=int, default=4)
    p.add_argument('--t0-per-variation', type=int, default=32)
    p.add_argument('--t0-explicit-per-layout', type=int, default=12)
    p.add_argument('--refcoco-per-stratum', type=int, default=100)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--attn', default='sdpa')
    p.add_argument('--stages', nargs='+', default=[])
    p.add_argument('--conditions', nargs='+', default=list(PE_CONDITIONS))
    p.add_argument('--shard', default='0/1')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--batch-size-explicit', type=int, default=16)
    p.add_argument('--max-new-tokens-binary', type=int, default=8)
    p.add_argument('--max-new-tokens-locate', type=int, default=256)
    p.add_argument('--max-new-tokens-explicit', type=int, default=2048)
    p.add_argument('--bv-batch-size', type=int, default=8)
    p.add_argument('--bv-seeds', type=int, nargs='+', default=[42])
    p.add_argument('--bv-max-new-tokens', type=int, default=32768)
    p.add_argument('--bv-temperature', type=float, default=1.0)
    p.add_argument('--bv-top-k', type=int, default=40)
    p.add_argument('--bv-top-p', type=float, default=1.0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'build-grid':
        build_grid(args.output, args.formal, args.per_combo, args.control_per_combo, args.seed)
    elif args.command == 'build-t0':
        build_t0(args.t0_dataset, args.t0_per_variation, args.seed)
    elif args.command == 'build-refcoco':
        build_refcoco(args.refcoco, args.output, args.refcoco_per_stratum, args.seed)
    elif args.command == 'build-babyvision':
        build_babyvision(args.data_root, args.babyvision_reference, args.output)
    elif args.command == 'run':
        run(args)
    else:
        spotcheck(args)


if __name__ == '__main__':
    main()
