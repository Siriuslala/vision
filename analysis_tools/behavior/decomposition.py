"""Experiment 1.3: decomposition of find-the-difference into sub-tasks T0-T6.

Design: docs/analysis/behavior/design_decomposition.md.  Commands:

  build  -- render every stimulus and write images.jsonl, cases.jsonl, config.json
  run    -- answer the cases of one stage, appending one JSON line per case (resumable)
  score  -- strict format scores and coordinate-content diagnostics, saved separately

Analysis and figures live in ``decomposition_analysis.py``.

All images of the main condition come from ``analysis_tools.common.stimuli``.  T1 and
T2 inputs are built from the same target instance as the grid image, rendered by
``render_instance`` and pasted at a random 32-pixel-aligned offset of a 256x256 canvas,
so that single-instance inputs are not all identical images.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from analysis_tools.common.stimuli import (
    DEVELOPMENT_TEMPLATES, GENERATOR_VERSION, LEVELS, MIN_CANVAS, NONE, TOKEN_PIXELS, Change, StimulusSpec,
    generate, render_instance, uniform_targets,
)

KINDS = ('gap', 'color', 'rotate')
GRIDS = (3, 6, 10, 20)
TEMPLATES_BY_KIND = {'gap': DEVELOPMENT_TEMPLATES, 'color': DEVELOPMENT_TEMPLATES, 'rotate': ('person', 'arrow')}
CELL = 64
FORMATS = ('row_column', 'point', 'bbox')
BINARY_TASKS = ('T1', 'T2', 'T3')
CODE_FILES = ('decomposition.py', 'decomposition_analysis.py', 'decomposition_reasoning.py')

# ----------------------------------------------------------------------------- prompts

ANSWER_YES_NO = ' Answer with only Yes or No.'
COORD_NOTE = ' All coordinates use 0-1000 normalization; x increases rightward and y increases downward.'
JSON_ONLY = ' Output only the JSON, with no explanation.'
LOCATE_SUFFIX = {
    'row_column': ' Give its position. Return only {"row": r, "column": c}, counting rows from top to bottom '
                  'and columns from left to right, starting at 1.',
    'point': ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].' + COORD_NOTE,
    'bbox': ' Give its tight bounding box. Return only [{"bbox_2d": [x_min, y_min, x_max, y_max]}].' + COORD_NOTE,
}
EXPLICIT_LOCATE = (' First describe step by step what you observe and how you find this figure, including how many '
                   'rows and columns of figures there are. Then give the final answer on the last line as '
                   '{"row": r, "column": c}, counting rows from top to bottom and columns from left to right, '
                   'starting at 1.')
EXPLICIT_BINARY = (' First describe step by step what you observe and how you reason. Then give the final answer '
                   'on the last line as "Answer: Yes" or "Answer: No".')
ORDINAL_NOTE = 'Rows are counted from top to bottom and columns from left to right, starting at 1.'

# T1 property questions; "Yes" is the correct answer for the unchanged template.
T1_QUESTION = {
    ('gap', None): ('Are all strokes of this figure connected without any gap?',
                    'Is every line of this figure continuous, with no break?'),
    ('color', None): ('Is this figure drawn entirely in black?', 'Are all lines of this figure pure black?'),
    ('rotate', 'person'): ('Is this figure upright?', 'Is this figure in its normal upright orientation, without any tilt?'),
    ('rotate', 'arrow'): ('Does this arrow point exactly to the right?',
                          'Is this arrow horizontal and pointing to the right?'),
}
# T4 attribute descriptions (primary, alternative).
T4_QUERY = {
    ('gap', None): ('One figure has a gap in its strokes.', 'Exactly one figure has a break in its lines.'),
    ('color', None): ('One figure is not drawn in black.', 'Exactly one figure has lines that are not pure black.'),
    ('rotate', 'person'): ('One figure is not upright; it is rotated.', 'Exactly one figure is tilted.'),
    ('rotate', 'arrow'): ('One arrow does not point exactly to the right; it is rotated.', 'Exactly one arrow is tilted.'),
}
T5_QUERY = ('One figure differs from all the others.', 'Exactly one figure is not the same as the rest.')
T6_QUERY = ('One figure is drawn in red.', 'Exactly one figure is colored red.')
T2_QUESTION = ('Are the two figures exactly identical?', 'Do the left and right figures look exactly the same?')
T3_BOX = ('Is the figure inside the blue box exactly identical to the other figures?',
          'Does the figure in the blue box look exactly the same as all other figures in the image?')
T3_ROWCOL = ('Is the figure in row {r}, column {c} exactly identical to the other figures? ' + ORDINAL_NOTE,
             'Does the figure at row {r}, column {c} look exactly the same as all other figures in the image? '
             + ORDINAL_NOTE)
T0_QUERY = ('How many rows and how many columns of figures are in this image?',
            'Count the figures in this image: how many rows and how many columns are there?')
T0_SUFFIX = ' Return only {"rows": R, "columns": C}.'


def _lookup(table: dict, kind: str, template: str):
    return table.get((kind, template), table.get((kind, None)))


# ----------------------------------------------------------------------------- utilities

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_digest() -> dict:
    here = Path(__file__).parent
    return {name: sha256_file(here / name) for name in CODE_FILES if (here / name).exists()}


def unit_seed(base: int, *parts) -> int:
    text = '|'.join(str(p) for p in (base, *parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def magnitude_label(kind: str, level: int) -> str:
    value = LEVELS[kind][level - 1]
    unit = {'gap': 'px', 'color': ' dE', 'rotate': ' deg'}[kind]
    return 'red' if value == 'red' else f'{value}{unit}'


def is_alt(rep: int, alt_period: int) -> bool:
    """Every alt_period-th image also receives the alternative wording (rep 0, 5, 10, ... for period 5)."""
    return alt_period > 0 and rep % alt_period == 0


# ----------------------------------------------------------------------------- stimuli

def place_instances(tiles: list[np.ndarray], rng: np.random.Generator) -> tuple[Image.Image, list[list[float]]]:
    """Paste 1 or 2 horizontally adjacent tiles at a random 32-aligned offset of a 256x256 canvas.

    Returns the image and the half-open tight box of each tile in canvas pixels.
    """
    width = CELL * len(tiles)
    xs = np.arange(0, MIN_CANVAS - width + 1, TOKEN_PIXELS)
    ys = np.arange(0, MIN_CANVAS - CELL + 1, TOKEN_PIXELS)
    x0, y0 = int(rng.choice(xs)), int(rng.choice(ys))
    canvas = np.full((MIN_CANVAS, MIN_CANVAS, 3), 255, np.uint8)
    boxes = []
    for i, tile in enumerate(tiles):
        x = x0 + i * CELL
        canvas[y0:y0 + CELL, x:x + CELL] = tile
        dark = np.argwhere((tile < 255).any(-1))
        (ty0, tx0), (ty1, tx1) = dark.min(0), dark.max(0) + 1
        boxes.append([float(x + tx0), float(y0 + ty0), float(x + tx1), float(y0 + ty1)])
    return Image.fromarray(canvas), boxes


def _grid_truth(meta: dict, spec: StimulusSpec) -> dict:
    target = next((i for i in meta['instances'] if i['is_target']), None)
    out = {'canvas': meta['canvas'], 'grid_origin': meta['grid_origin'], 'cell': spec.cell,
           'rows': spec.rows, 'cols': spec.cols}
    if target is not None:
        out.update(target_rc=[target['row'], target['col']], target_cell_box=target['cell_box'],
                   target_tight_box=target['tight_box'], target_center=target['center'])
    return out


class ImageStore:
    """Writes PNGs under output/images and verifies the hash of any existing file."""

    def __init__(self, output: Path):
        self.output = output
        self.records: list[dict] = []

    def save(self, image: Image.Image, rel: str, **record) -> dict:
        path = self.output / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if path.exists():
            if sha256_file(path) != digest:
                raise ValueError(f'Existing image {rel} differs from the regenerated stimulus.')
        else:
            path.write_bytes(data)
        row = {'image_id': rel.removesuffix('.png').replace('/', ':'), 'path': rel, 'sha256': digest, **record}
        self.records.append(row)
        return row


# ----------------------------------------------------------------------------- case construction

def _case(stage: str, task: str, image: dict, prompt: str, *, fmt: str, wording: str, variant: str,
          truth: dict, group: dict, thinking: bool = False) -> dict:
    cid = '|'.join([stage, task, image['image_id'], variant, fmt, wording])
    return {'id': cid, 'stage': stage, 'task': task, 'format': fmt, 'wording': wording, 'variant': variant,
            'image': image['path'], 'image_sha256': image['sha256'], 'image_id': image['image_id'],
            'prompt': prompt, 'thinking': thinking, 'truth': truth, **group}


def binary_prompt(question: str, style: str) -> str:
    return question + (EXPLICIT_BINARY if style == 'explicit' else ANSWER_YES_NO)


def locate_prompt(query: str, fmt: str, style: str) -> str:
    if style == 'explicit':
        if fmt != 'row_column':
            raise ValueError('Explicit reasoning uses the row/column format only.')
        return query + EXPLICIT_LOCATE
    return query + LOCATE_SUFFIX[fmt] + JSON_ONLY


def build_main(output: Path, cfg: dict) -> tuple[list[dict], list[dict], dict]:
    """Render all main-condition images and return (images, cases, reasoning_pool).

    reasoning_pool maps (kind, level, grid, rep) -> rendered records reused by the reasoning stages.
    """
    store = ImageStore(output)
    cases: list[dict] = []
    pool: dict = {}
    n, period = cfg['images_per_unit'], cfg['alt_period']
    for kind in KINDS:
        templates = TEMPLATES_BY_KIND[kind]
        for level in range(1, len(LEVELS[kind]) + 1):
            for grid in GRIDS:
                seed = unit_seed(cfg['seed'], kind, level, grid)
                targets = uniform_targets(grid, grid, n, seed)
                rng = np.random.default_rng(seed + 1)
                for rep in range(n):
                    template = templates[rep % len(templates)]
                    site = (rep // len(templates)) % 3 if kind == 'gap' else 0
                    change = Change(kind, level, site)
                    target = targets[rep]
                    flat = [i for i in range(grid * grid) if i != target[0] * grid + target[1]]
                    normal = divmod(int(rng.choice(flat)), grid)
                    stem = f'{kind}_L{level}_g{grid:02d}_r{rep:02d}'
                    group = {'kind': kind, 'level': level, 'magnitude': LEVELS[kind][level - 1], 'grid': grid,
                             'template': template, 'site': site, 'rep': rep, 'record_id': stem}
                    base = StimulusSpec(template, grid, grid, change, target=target, seed=seed + rep)
                    img, meta = generate(base)
                    truth = _grid_truth(meta, base)
                    rec = store.save(img, f'images/grid/{stem}.png', role='grid', spec=meta['spec'], **truth)
                    marks = {}
                    for which, rc in (('target', target), ('normal', normal)):
                        spec = StimulusSpec(template, grid, grid, change, target=target, marker=rc, seed=seed + rep)
                        mimg, mmeta = generate(spec)
                        marks[which] = store.save(mimg, f'images/marked/{stem}_{which}.png', role='marked',
                                                  spec=mmeta['spec'], marker_rc=list(rc), **truth)
                    # T1 / T2 inputs: the same target instance and the template, at random positions.
                    changed, plain = render_instance(template, change, CELL), render_instance(template, NONE, CELL)
                    small = {}
                    for name, tiles in (('t1_neg', [changed]), ('t1_pos', [plain])):
                        im, boxes = place_instances(tiles, rng)
                        small[name] = store.save(im, f'images/single/{stem}_{name}.png', role=name, boxes=boxes,
                                                 canvas=[MIN_CANVAS, MIN_CANVAS])
                    changed_left = bool(rng.integers(0, 2))
                    pair = [changed, plain] if changed_left else [plain, changed]
                    for name, tiles in (('t2_neg', pair), ('t2_pos', [plain, plain])):
                        im, boxes = place_instances(tiles, rng)
                        small[name] = store.save(im, f'images/pair/{stem}_{name}.png', role=name, boxes=boxes,
                                                 canvas=[MIN_CANVAS, MIN_CANVAS],
                                                 changed_side=('left' if changed_left else 'right') if name == 't2_neg' else None)
                    pool[(kind, level, grid, rep)] = {'grid': rec, 'group': group, 'truth': truth, 'normal': list(normal)}
                    wordings = ('primary', 'alt') if is_alt(rep, period) else ('primary',)
                    for w, word in enumerate(wordings):
                        q1 = _lookup(T1_QUESTION, kind, template)[w]
                        for name, yes in (('t1_neg', False), ('t1_pos', True)):
                            cases.append(_case('main', 'T1', small[name], binary_prompt(q1, 'direct'), fmt='yes_no',
                                               wording=word, variant=name[3:], truth={'answer_yes': yes}, group=group))
                        for name, yes in (('t2_neg', False), ('t2_pos', True)):
                            cases.append(_case('main', 'T2', small[name], binary_prompt(T2_QUESTION[w], 'direct'),
                                               fmt='yes_no', wording=word, variant=name[3:], truth={'answer_yes': yes},
                                               group=group))
                        for which, rc in (('target', target), ('normal', normal)):
                            yes = which == 'normal'
                            cases.append(_case('main', 'T3', marks[which], binary_prompt(T3_BOX[w], 'direct'),
                                               fmt='yes_no', wording=word, variant=f'box_{which}',
                                               truth={'answer_yes': yes, 'designated_rc': list(rc)}, group=group))
                            q = T3_ROWCOL[w].format(r=rc[0] + 1, c=rc[1] + 1)
                            cases.append(_case('main', 'T3', rec, binary_prompt(q, 'direct'), fmt='yes_no',
                                               wording=word, variant=f'rowcol_{which}',
                                               truth={'answer_yes': yes, 'designated_rc': list(rc)}, group=group))
                        locate = [('T4', _lookup(T4_QUERY, kind, template)[w]), ('T5', T5_QUERY[w])]
                        if kind == 'color' and LEVELS[kind][level - 1] == 'red':
                            locate.append(('T6', T6_QUERY[w]))
                        fmts = FORMATS if word == 'primary' else ('row_column',)
                        for task, query in locate:
                            for fmt in fmts:
                                cases.append(_case('main', task, rec, locate_prompt(query, fmt, 'direct'), fmt=fmt,
                                                   wording=word, variant='target', truth=truth, group=group))
    # T0: normal images without any anomalous instance.
    for grid in GRIDS:
        seed = unit_seed(cfg['seed'], 'count', grid)
        for rep in range(cfg['count_images_per_grid']):
            template = DEVELOPMENT_TEMPLATES[rep % len(DEVELOPMENT_TEMPLATES)]
            spec = StimulusSpec(template, grid, grid, seed=seed + rep)
            img, meta = generate(spec)
            stem = f'count_g{grid:02d}_r{rep:02d}'
            truth = _grid_truth(meta, spec)
            rec = store.save(img, f'images/count/{stem}.png', role='count', spec=meta['spec'], **truth)
            group = {'kind': 'none', 'level': 0, 'magnitude': None, 'grid': grid, 'template': template, 'site': 0,
                     'rep': rep, 'record_id': stem}
            for w, word in enumerate(('primary', 'alt') if is_alt(rep, period) else ('primary',)):
                cases.append(_case('main', 'T0', rec, T0_QUERY[w] + T0_SUFFIX + JSON_ONLY, fmt='count',
                                   wording=word, variant='normal', truth=truth, group=group))
    return store.records, cases, pool


def build_reasoning(pool: dict, cfg: dict) -> list[dict]:
    """Explicit-reasoning (thinking off) and thinking-on subsets drawn from the main images."""
    cases = []
    for stage, per_cell, levels in (('explicit', cfg['explicit_per_cell'], cfg['explicit_levels']),
                                    ('thinking', cfg['thinking_per_cell'], cfg['thinking_levels'])):
        style, thinking = ('explicit', False) if stage == 'explicit' else ('direct', True)
        for grid in cfg['reasoning_grids']:
            for kind in KINDS:
                for level in levels:
                    for rep in range(per_cell):
                        item = pool[(kind, level, grid, rep)]
                        rec, group, truth = item['grid'], item['group'], item['truth']
                        for which, rc in (('target', truth['target_rc']), ('normal', item['normal'])):
                            q = T3_ROWCOL[0].format(r=rc[0] + 1, c=rc[1] + 1)
                            cases.append(_case(stage, 'T3', rec, binary_prompt(q, style), fmt='yes_no',
                                               wording='primary', variant=f'rowcol_{which}', thinking=thinking,
                                               truth={'answer_yes': which == 'normal', 'designated_rc': list(rc),
                                                      **truth}, group=group))
                        for task, query in (('T4', _lookup(T4_QUERY, kind, group['template'])[0]), ('T5', T5_QUERY[0])):
                            cases.append(_case(stage, task, rec, locate_prompt(query, 'row_column', style),
                                               fmt='row_column', wording='primary', variant='target',
                                               thinking=thinking, truth=truth, group=group))
            red_level = LEVELS['color'].index('red') + 1
            for rep in range(per_cell):
                item = pool[('color', red_level, grid, rep)]
                cases.append(_case(stage, 'T6', item['grid'], locate_prompt(T6_QUERY[0], 'row_column', style),
                                   fmt='row_column', wording='primary', variant='target', thinking=thinking,
                                   truth=item['truth'], group=item['group']))
    return cases


BUILD_KEYS = ('seed', 'images_per_unit', 'count_images_per_grid', 'alt_period', 'reasoning_grids',
              'explicit_levels', 'explicit_per_cell', 'thinking_levels', 'thinking_per_cell')


def build(output: Path, cfg: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    build_cfg = {k: cfg[k] for k in BUILD_KEYS}
    if config_path.exists():
        stored = json.loads(config_path.read_text())['build']
        if stored != build_cfg:
            raise ValueError(f'Build configuration differs from {config_path}: {stored} vs {build_cfg}')
    images, cases, pool = build_main(output, cfg)
    for stage in ('explicit', 'thinking'):
        if cfg[f'{stage}_per_cell'] > cfg['images_per_unit']:
            raise ValueError(f'{stage}_per_cell exceeds images_per_unit')
    cases += build_reasoning(pool, cfg)
    # Interleave by replicate so that any prefix of a run is balanced across units.
    order = {'main': 0, 'explicit': 1, 'thinking': 2}
    cases.sort(key=lambda c: (order[c['stage']], c['rep'], c['grid'], c['kind'], c['level'], c['task'],
                              c['variant'], c['format'], c['wording']))
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case identifiers.')
    payload = {'build': build_cfg, 'generator': GENERATOR_VERSION, 'code_sha256': code_digest(),
               'n_images': len(images), 'n_cases': {s: sum(c['stage'] == s for c in cases) for s in order}}
    if (output / 'cases.jsonl').exists():
        old = read_jsonl(output / 'cases.jsonl')
        if [c['id'] for c in old] != ids or any(a != b for a, b in zip(old, cases)):
            raise ValueError('Existing cases.jsonl differs from the regenerated cases.')
    write_jsonl(output / 'images.jsonl', images)
    write_jsonl(output / 'cases.jsonl', cases)
    if not config_path.exists():
        config_path.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload['n_cases']), len(images), 'images')


# ----------------------------------------------------------------------------- inference

RUN_KEYS = ('model', 'max_new_tokens_binary', 'max_new_tokens_locate', 'max_new_tokens_explicit',
            'max_new_tokens_thinking')


def max_tokens(case: dict, cfg: dict) -> int:
    if case['thinking']:
        return cfg['max_new_tokens_thinking']
    if case['stage'] in ('explicit',):
        return cfg['max_new_tokens_explicit']
    return cfg['max_new_tokens_binary'] if case['format'] == 'yes_no' else cfg['max_new_tokens_locate']


def run(output: Path, cfg: dict, stage: str, limit: int | None, cases_file: str = 'cases.jsonl') -> None:
    """Answer pending cases of one stage; each result is appended and flushed immediately."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList
    from analysis_tools.common import qwen_runtime as qr

    cases = [c for c in read_jsonl(output / cases_file) if c['stage'] == stage]
    if not cases:
        raise ValueError(f'No cases for stage {stage!r} in {cases_file}.')
    run_cfg = {k: cfg[k] for k in RUN_KEYS}
    cfg_path = output / f'run_{stage}.json'
    if cfg_path.exists():
        stored = json.loads(cfg_path.read_text())
        if stored['run'] != run_cfg:
            raise ValueError(f'Run configuration differs from {cfg_path}: {stored["run"]} vs {run_cfg}')
    results_path = output / f'results_{stage}.jsonl'
    done = {r['id'] for r in read_jsonl(results_path)} if results_path.exists() else set()
    pending = [c for c in cases if c['id'] not in done]
    if limit is not None:
        pending = pending[:limit]
    print(f'{stage}: {len(done)} done, {len(pending)} to run', flush=True)
    if not pending:
        return
    rt = qr.load(cfg['model'], device='cuda:0')
    yes_id, no_id = qr.token_id(rt, 'Yes'), qr.token_id(rt, 'No')
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({'run': run_cfg, 'model': qr.describe(rt)}, indent=2) + '\n')
    elif json.loads(cfg_path.read_text())['model'] != qr.describe(rt):
        raise ValueError('Model provenance differs from the stored run configuration.')
    log_path = output / 'run_log.jsonl'
    start = dt.datetime.now().isoformat(timespec='seconds')

    class FirstStep(LogitsProcessor):
        """Keeps a float32 copy of the first-step logits without changing them (greedy decoding)."""
        def __init__(self):
            self.first = None

        def __call__(self, input_ids, scores):
            if self.first is None:
                self.first = scores[0].float().clone()
            return scores

    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    n_done = 0
    with results_path.open('a') as sink:
        for case in pending:
            path = output / case['image']
            if sha256_file(path) != case['image_sha256']:
                raise ValueError(f'Image hash mismatch for {case["image"]}')
            image = Image.open(path).convert('RGB')
            batch = qr.build(rt, [image], case['prompt'], thinking=case['thinking'])
            budget = max_tokens(case, cfg)
            first = FirstStep()
            tic = time.perf_counter()
            with torch.inference_mode():
                out = rt.model.generate(**batch, max_new_tokens=budget, do_sample=False, temperature=None,
                                        top_p=None, top_k=None, logits_processor=LogitsProcessorList([first]))
            elapsed = time.perf_counter() - tic
            new = out[0, batch['input_ids'].shape[1]:]
            finished = len(new) > 0 and int(new[-1]) in eos
            record = {'id': case['id'], 'response': rt.tokenizer.decode(new, skip_special_tokens=True),
                      'response_raw': rt.tokenizer.decode(new, skip_special_tokens=False),
                      'n_tokens': int(len(new)), 'finish_reason': 'eos' if finished else 'length',
                      'max_new_tokens': budget, 'prompt_tokens': int(batch['input_ids'].shape[1]),
                      'seconds': round(elapsed, 3)}
            logits = first.first
            record['first_top_token'] = rt.tokenizer.decode([int(logits.argmax())])
            if case['format'] == 'yes_no' and not case['thinking']:
                z_yes, z_no = float(logits[yes_id]), float(logits[no_id])
                record.update(z_yes=z_yes, z_no=z_no, s=z_yes - z_no)
            sink.write(json.dumps(record) + '\n')
            sink.flush()
            n_done += 1
            if n_done % 200 == 0:
                print(f'{stage}: {n_done}/{len(pending)}', flush=True)
    with log_path.open('a') as log:
        log.write(json.dumps({'stage': stage, 'cases_file': cases_file, 'start': start,
                              'end': dt.datetime.now().isoformat(timespec='seconds'), 'n_run': n_done,
                              'code_sha256': code_digest()}) + '\n')


# ----------------------------------------------------------------------------- parsing and scoring

YES_NO = re.compile(r'^\s*(yes|no)\s*[.!]?\s*$', re.I)


def answer_part(response: str) -> str:
    """Text after the thinking block; responses without thinking are returned unchanged."""
    return response.split('</think>')[-1].strip()


def parse_json(text: str):
    body = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    return json.loads(body)


def last_json_object(text: str):
    """Last {...} object in free text that parses as JSON (used for explicit-reasoning answers)."""
    for match in reversed(list(re.finditer(r'\{[^{}]*\}', text))):
        try:
            return json.loads(match.group(0))
        except ValueError:
            continue
    raise ValueError('No JSON object in text.')


def parse_binary(response: str, stage: str) -> bool | None:
    text = answer_part(response)
    if stage == 'explicit':
        found = re.findall(r'Answer\s*:\s*\**\s*(Yes|No)\b', text, re.I)
        return found[-1].lower() == 'yes' if found else None
    m = YES_NO.match(text)
    return m.group(1).lower() == 'yes' if m else None


def box_iou(a, b) -> float:
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.


def _is_int(v) -> bool:
    return type(v) is int


def _is_num(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def cell_of(truth: dict, pixel) -> tuple[int, int] | None:
    """Zero-based (row, col) of the grid cell containing a pixel, or None outside the grid."""
    ox, oy = truth['grid_origin']
    col = math.floor((pixel[0] - ox) / truth['cell'])
    row = math.floor((pixel[1] - oy) / truth['cell'])
    if 0 <= row < truth['rows'] and 0 <= col < truth['cols']:
        return row, col
    return None


def score_locate(case: dict, record: dict, *, diagnostic: bool = False) -> dict:
    """Score one T4-T6 answer.  Truncated or malformed answers are invalid and count as failures."""
    out = {'valid': False, 'correct': False}
    if record['finish_reason'] != 'eos':
        out['failure'] = 'truncated'
        return out
    truth, fmt = case['truth'], case['format']
    try:
        value = last_json_object(answer_part(record['response'])) if case['stage'] == 'explicit' \
            else parse_json(answer_part(record['response']))
    except ValueError:
        out['failure'] = 'unparsable'
        return out
    tr, tc = truth['target_rc']
    if diagnostic:
        # Coordinate-content diagnostic: accept unambiguous wrappers that differ from the requested schema.
        if fmt == 'row_column' and isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            value = value[0]
        if fmt in ('point', 'bbox') and isinstance(value, dict):
            value = [value]
    if fmt == 'row_column':
        if not isinstance(value, dict) or not _is_int(value.get('row')) or not _is_int(value.get('column')):
            out['failure'] = 'schema'
            return out
        row, col = value['row'], value['column']
        if not (1 <= row <= truth['rows'] and 1 <= col <= truth['cols']):
            out['failure'] = 'out_of_range'
            return out
        return {'valid': True, 'correct': (row - 1, col - 1) == (tr, tc), 'pred_rc': [row - 1, col - 1],
                'd_row': row - 1 - tr, 'd_col': col - 1 - tc}
    key, length = ('point_2d', 2) if fmt == 'point' else ('bbox_2d', 4)
    if diagnostic and isinstance(value, list) and len(value) == length and all(_is_num(v) for v in value):  # bare list
        value = [{key: value}]
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict)
            and isinstance(value[0].get(key), list) and len(value[0][key]) == length
            and all(_is_num(v) and 0 <= v <= 1000 for v in value[0][key])):
        out['failure'] = 'schema'
        return out
    coords = np.array(value[0][key], float)
    size = np.array(truth['canvas'], float)
    pixel = coords * np.tile(size, length // 2) / 1000
    if length == 4:
        if pixel[0] >= pixel[2] or pixel[1] >= pixel[3]:
            out['failure'] = 'degenerate_box'
            return out
        iou = box_iou(pixel.tolist(), truth['target_tight_box'])
        center = (pixel[:2] + pixel[2:]) / 2
    else:
        center = pixel
    distance = float(np.linalg.norm(center - np.array(truth['target_center']))) / truth['cell']
    cell = cell_of(truth, center)
    result = {'valid': True, 'pred_center': center.tolist(), 'cell_distance': distance,
              'pred_rc': list(cell) if cell else None, 'hit': cell == (tr, tc)}
    if length == 4:
        result.update(iou=iou, correct=iou >= 0.5, pred_box=pixel.tolist())
    else:
        result['correct'] = result['hit']
    return result


def score_count(case: dict, record: dict) -> dict:
    out = {'valid': False, 'correct': False}
    if record['finish_reason'] != 'eos':
        return {**out, 'failure': 'truncated'}
    try:
        value = parse_json(answer_part(record['response']))
    except ValueError:
        return {**out, 'failure': 'unparsable'}
    if not isinstance(value, dict) or not _is_int(value.get('rows')) or not _is_int(value.get('columns')):
        return {**out, 'failure': 'schema'}
    rows, cols = value['rows'], value['columns']
    t = case['truth']
    return {'valid': True, 'correct': rows == t['rows'] and cols == t['cols'], 'rows_correct': rows == t['rows'],
            'cols_correct': cols == t['cols'], 'pred_rows': rows, 'pred_cols': cols}


def score_binary(case: dict, record: dict) -> dict:
    """Main stage: decision from the sign of s; format compliance kept separately.

    Reasoning stages: decision from the parsed final answer; unparsable or truncated answers are failures.
    """
    yes = case['truth']['answer_yes']
    parsed = parse_binary(record['response'], case['stage']) if record['finish_reason'] == 'eos' else None
    out = {'answer_yes': yes, 'format_valid': parsed is not None, 'parsed_yes': parsed,
           'parsed_correct': parsed == yes if parsed is not None else False}
    # The runner stores s for every yes/no case, but in reasoning stages the first generated
    # token is reasoning text (or <think>), so s is only a decision score for direct answers.
    if 's' in record and case['stage'] not in ('explicit', 'thinking'):
        out.update(s=record['s'], correct=(record['s'] > 0) == yes, valid=True)
    else:
        out.update(correct=out['parsed_correct'], valid=parsed is not None)
    return out


def score_case(case: dict, record: dict, diagnostic: bool) -> dict:
    if case['format'] == 'yes_no':
        return score_binary(case, record)
    if case['format'] == 'count':
        return score_count(case, record)
    return score_locate(case, record, diagnostic=diagnostic)


def score(output: Path, cases_file: str = 'cases.jsonl', suffix: str = '') -> None:
    cases = {c['id']: c for c in read_jsonl(output / cases_file)}
    for stage in sorted({c['stage'] for c in cases.values()}):
        path = output / f'results_{stage}.jsonl'
        if not path.exists():
            continue
        records = read_jsonl(path)
        if len({r['id'] for r in records}) != len(records):
            raise ValueError(f'Duplicate results in {path}')
        for name, diagnostic in (('strict', False), ('diagnostic', True)):
            rows = []
            for record in records:
                if record['id'] not in cases:
                    raise ValueError(f'Result {record["id"]} has no case.')
                case = cases[record['id']]
                rows.append({'id': record['id'], 'finish_reason': record['finish_reason'],
                             **score_case(case, record, diagnostic)})
            write_jsonl(output / f'scores_{stage}_{name}{suffix}.jsonl', rows)
        n_case = sum(c['stage'] == stage for c in cases.values())
        print(f'{stage}: scored {len(records)}/{n_case}')


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('build', 'run', 'score'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('main', 'explicit', 'thinking', 'babyvision'), default='main')
    p.add_argument('--limit', type=int, default=None, help='Run at most this many pending cases.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--images-per-unit', type=int, default=48)
    p.add_argument('--count-images-per-grid', type=int, default=16)
    p.add_argument('--alt-period', type=int, default=5)
    p.add_argument('--reasoning-grids', type=int, nargs='+', default=[6, 20])
    p.add_argument('--explicit-levels', type=int, nargs='+', default=[1, 3])
    p.add_argument('--explicit-per-cell', type=int, default=4)
    p.add_argument('--thinking-levels', type=int, nargs='+', default=[3])
    p.add_argument('--thinking-per-cell', type=int, default=2)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--max-new-tokens-binary', type=int, default=8)
    p.add_argument('--max-new-tokens-locate', type=int, default=256)
    p.add_argument('--max-new-tokens-explicit', type=int, default=2048)
    p.add_argument('--max-new-tokens-thinking', type=int, default=8192)
    args = p.parse_args()
    cfg = {k.replace('-', '_'): v for k, v in vars(args).items()}
    for key in ('images_per_unit', 'count_images_per_grid', 'max_new_tokens_binary', 'max_new_tokens_locate'):
        if cfg[key] < 1:
            p.error(f'{key} must be positive')
    if args.command == 'build':
        build(args.output, cfg)
    elif args.command == 'run':
        cases_file = 'babyvision_cases.jsonl' if args.stage == 'babyvision' else 'cases.jsonl'
        run(args.output, cfg, args.stage, args.limit, cases_file)
    else:
        score(args.output)
        if (args.output / 'babyvision_cases.jsonl').exists():
            score(args.output, 'babyvision_cases.jsonl')


if __name__ == '__main__':
    main()
