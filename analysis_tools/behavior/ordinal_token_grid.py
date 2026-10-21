"""Experiment 1.4: ordinal offset and the visual token grid.

Question: on axes holding 20 instances of 64 pixels the row/column ordinal answers of Qwen3.5-9B are
systematically one too large, whereas 20 instances of 32 pixels show no such offset.  The factorial
grid below separates the tokens per instance (cell / 32), the axis length in pixels, the number of
instances on the axis and the phase of instance borders relative to the 32-pixel merged-token grid.

Layouts have 6 rows and n columns (the long axis is horizontal; experiment 1.3 showed that the offset
follows the axis holding many instances, not its orientation); conditions with an r<rows> suffix
use more rows to connect the 6-row strips with the square grids of experiment 1.3.  Commands:

  build      -- render stimuli, write images.jsonl, cases.jsonl, config.json
  run        -- batched greedy answers (qwen_runtime.generate_batch), resumable, shardable
  spotcheck  -- re-run a random subset at batch size 1 (qwen_runtime.generate)
  score      -- strict and content-diagnostic scores plus the coordinate-convention columns
  mech       -- teacher-forced log p of neighbouring column ordinals and pointing-head attention
                over token columns, for T6 row/column answers of selected conditions
  convention -- recompute the convention diagnostic for the formal 1.3 run and its supplement

Cells that are not multiples of 32 pixels (48, 80) are outside stimuli-v1's StimulusSpec, so this
module composes grids from `stimuli.render_instance` with the same placement rule as
`stimuli.generate`; for 32-multiple cells the images are byte-identical to stimuli-v1 (tested).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from analysis_tools.behavior import coordinate_convention as cc
from analysis_tools.behavior.decomposition import (
    JSON_ONLY, LOCATE_SUFFIX, T0_QUERY, T0_SUFFIX, T6_QUERY, ImageStore, read_jsonl, score_case, sha256_file,
    unit_seed, write_jsonl,
)
from analysis_tools.common.stimuli import (
    DEVELOPMENT_TEMPLATES, GENERATOR_VERSION, MIN_CANVAS, NONE, TOKEN_PIXELS, WHITE, Change, render_instance,
)

GENERATOR = GENERATOR_VERSION + '+free_cell'
RED = Change('color', 4)
ROWS = 6
PATCH = 16


# ----------------------------------------------------------------------------- stimuli

@dataclass(frozen=True)
class GridSpec:
    """Grid of identical instances with an optional red target.

    cell: instance cell edge in pixels (multiple of 16 so that every axis length stays a multiple
          of 32 when the instance count is even).
    phase: offset in pixels added to the grid origin on both axes (0 = borders on the token grid).
    target: zero-based (row, col) of the red instance, or None.
    """
    template: str
    rows: int
    cols: int
    cell: int
    phase: int = 0
    target: tuple[int, int] | None = None

    def __post_init__(self):
        if self.cell % PATCH or self.cell < 32:
            raise ValueError('cell must be a multiple of 16 pixels and at least 32')
        if not 0 <= self.phase < TOKEN_PIXELS:
            raise ValueError('phase must lie in [0, 32)')
        if (self.rows * self.cell) % TOKEN_PIXELS or (self.cols * self.cell) % TOKEN_PIXELS:
            raise ValueError('rows * cell and cols * cell must be multiples of 32 pixels')
        if self.target is not None and not (0 <= self.target[0] < self.rows and 0 <= self.target[1] < self.cols):
            raise ValueError(f'target {self.target} outside the grid')


def layout(spec: GridSpec) -> tuple[int, int, int, int]:
    """Canvas width/height and grid origin; same rule as stimuli.canvas_layout (no jitter)."""
    margin = TOKEN_PIXELS if spec.phase else 0
    def axis(n):
        content = n * spec.cell
        size = math.ceil(max(MIN_CANVAS, content + margin) / TOKEN_PIXELS) * TOKEN_PIXELS
        origin = ((size - content - margin) // 2 // TOKEN_PIXELS) * TOKEN_PIXELS
        return size, origin + spec.phase
    width, ox = axis(spec.cols)
    height, oy = axis(spec.rows)
    return width, height, ox, oy


def render(spec: GridSpec) -> tuple[Image.Image, dict]:
    width, height, ox, oy = layout(spec)
    canvas = np.full((height, width, 3), WHITE, np.uint8)
    tiles = {c: render_instance(spec.template, c, spec.cell) for c in (NONE, RED)}
    instances = []
    for r in range(spec.rows):
        for c in range(spec.cols):
            is_target = spec.target == (r, c)
            tile = tiles[RED if is_target else NONE]
            x0, y0 = ox + c * spec.cell, oy + r * spec.cell
            region = canvas[y0:y0 + spec.cell, x0:x0 + spec.cell]
            np.minimum(region, tile, out=region)
            dark = np.argwhere((tile < WHITE).any(-1))
            (ty0, tx0), (ty1, tx1) = dark.min(0), dark.max(0) + 1
            box = [float(x0 + tx0), float(y0 + ty0), float(x0 + tx1), float(y0 + ty1)]
            instances.append({'row': r, 'col': c, 'is_target': is_target,
                              'cell_box': [x0, y0, x0 + spec.cell, y0 + spec.cell], 'tight_box': box,
                              'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]})
    image = Image.fromarray(canvas)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    meta = {'generator': GENERATOR, 'spec': asdict(spec), 'canvas': [width, height], 'grid_origin': [ox, oy],
            'patch_grid': [height // PATCH, width // PATCH],
            'merged_grid': [height // TOKEN_PIXELS, width // TOKEN_PIXELS], 'instances': instances,
            'sha256': hashlib.sha256(buffer.getvalue()).hexdigest()}
    return image, meta


def truth_of(meta: dict, spec: GridSpec) -> dict:
    out = {'canvas': meta['canvas'], 'grid_origin': meta['grid_origin'], 'cell': spec.cell, 'rows': spec.rows,
           'cols': spec.cols, 'phase': spec.phase, 'merged_grid': meta['merged_grid']}
    target = next((i for i in meta['instances'] if i['is_target']), None)
    if target is not None:
        out.update(target_rc=[target['row'], target['col']], target_cell_box=target['cell_box'],
                   target_tight_box=target['tight_box'], target_center=target['center'])
    return out


def column_token_span(origin_x: int, cell: int, col: int) -> tuple[float, float]:
    """Left and right edge of column `col` in merged-token units (32 pixels)."""
    x0 = origin_x + col * cell
    return x0 / TOKEN_PIXELS, (x0 + cell) / TOKEN_PIXELS


def stratified_targets(rows: int, cols: int, per_col: int, seed: int) -> list[tuple[int, int]]:
    """per_col targets in every column (rows uniform), returned in a seeded random order."""
    rng = np.random.default_rng(seed)
    targets = [(int(rng.integers(0, rows)), c) for c in range(cols) for _ in range(per_col)]
    order = rng.permutation(len(targets))
    return [targets[i] for i in order]


# ----------------------------------------------------------------------------- conditions

CONDITION = re.compile(r'c(\d+)n(\d+)p(\d+)(?:r(\d+))?')


def condition_name(cell: int, n: int, phase: int, rows: int = ROWS) -> str:
    return f'c{cell}n{n}p{phase}' + ('' if rows == ROWS else f'r{rows}')


def parse_condition(name: str) -> tuple[int, int, int]:
    """(cell, n, phase) of a condition name c<cell>n<count>p<phase>[r<rows>]."""
    m = CONDITION.fullmatch(name)
    if not m:
        raise ValueError(f'Bad condition name {name!r}; expected c<cell>n<count>p<phase>[r<rows>]')
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def condition_rows(name: str) -> int:
    """Row count of a condition: 6 unless the name carries an r<rows> suffix."""
    parse_condition(name)
    m = CONDITION.fullmatch(name)
    return int(m.group(4)) if m.group(4) else ROWS


def build(output: Path, cfg: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    store = ImageStore(output)
    cases = []
    t6_prompt = {fmt: T6_QUERY[0] + LOCATE_SUFFIX[fmt] + JSON_ONLY for fmt in ('row_column', 'point')}
    t0_prompt = T0_QUERY[0] + T0_SUFFIX + JSON_ONLY
    for name in cfg['conditions']:
        cell, n, phase = parse_condition(name)
        rows = condition_rows(name)
        per_col = math.ceil(cfg['t6_per_condition'] / n)
        seed = unit_seed(cfg['seed'], 'ordinal', name)
        targets = stratified_targets(rows, n, per_col, seed)
        group = {'condition': name, 'cell': cell, 'n': n, 'phase': phase, 'rows': rows, 'cols': n,
                 'tokens_per_instance': cell / TOKEN_PIXELS, 'axis_pixels': n * cell}
        for rep, target in enumerate(targets):
            template = DEVELOPMENT_TEMPLATES[rep % len(DEVELOPMENT_TEMPLATES)]
            spec = GridSpec(template, rows, n, cell, phase, target)
            img, meta = render(spec)
            stem = f'{name}_T6_r{rep:03d}'
            rec = store.save(img, f'images/{name}/{stem}.png', role='T6', spec=meta['spec'])
            truth = truth_of(meta, spec)
            for fmt, prompt in t6_prompt.items():
                cases.append({'id': f'T6|{stem}|{fmt}', 'task': 'T6', 'format': fmt, 'stage': 'main',
                              'variant': 'target', 'image': rec['path'], 'image_sha256': rec['sha256'],
                              'image_id': rec['image_id'], 'prompt': prompt, 'thinking': False, 'truth': truth,
                              'template': template, 'rep': rep, **group})
        for rep in range(cfg['t0_per_condition']):
            template = DEVELOPMENT_TEMPLATES[rep % len(DEVELOPMENT_TEMPLATES)]
            spec = GridSpec(template, rows, n, cell, phase)
            img, meta = render(spec)
            stem = f'{name}_T0_r{rep:03d}'
            rec = store.save(img, f'images/{name}/{stem}.png', role='T0', spec=meta['spec'])
            cases.append({'id': f'T0|{stem}|count', 'task': 'T0', 'format': 'count', 'stage': 'main',
                          'variant': 'none', 'image': rec['path'], 'image_sha256': rec['sha256'],
                          'image_id': rec['image_id'], 'prompt': t0_prompt, 'thinking': False,
                          'truth': truth_of(meta, spec), 'template': template, 'rep': rep, **group})
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case identifiers.')
    path = output / 'cases.jsonl'
    if path.exists():
        new = {c['id']: c for c in cases}
        if any(new.get(c['id']) != c for c in read_jsonl(path)):
            raise ValueError('Existing cases.jsonl differs from the regenerated cases.')
    write_jsonl(path, cases)
    write_jsonl(output / 'images.jsonl', store.records)
    keys = ('seed', 'conditions', 't6_per_condition', 't0_per_condition')
    (output / 'config.json').write_text(json.dumps({'generator': GENERATOR, 'rows': ROWS, 'n_cases': len(cases),
                                                    'build': {k: cfg[k] for k in keys}}, indent=2) + '\n')
    print(len(cases), 'cases', len(store.records), 'images')


# ----------------------------------------------------------------------------- run

def load_image(case: dict, output: Path) -> Image.Image:
    path = output / case['image']
    if sha256_file(path) != case['image_sha256']:
        raise ValueError(f'Image hash mismatch for {path}')
    return Image.open(path).convert('RGB')


def result_files(output: Path) -> list[Path]:
    return sorted(output.glob('results*.jsonl'))


def run(output: Path, cfg: dict, shard: tuple[int, int], limit: int | None) -> None:
    from analysis_tools.common import qwen_runtime as qr
    cases = read_jsonl(output / 'cases.jsonl')
    done = {r['id'] for p in result_files(output) for r in read_jsonl(p)}
    index, count = shard
    pending = [c for i, c in enumerate(cases) if i % count == index and c['id'] not in done]
    if limit is not None:
        pending = pending[:limit]
    print(f'shard {index}/{count}: {len(done)} done overall, {len(pending)} to run', flush=True)
    if not pending:
        return
    rt = qr.load(cfg['model'], device='cuda:0')
    run_cfg = {'model': cfg['model'], 'max_new_tokens': cfg['max_new_tokens'], 'batch_size': cfg['batch_size'],
               'thinking': False, 'provenance': qr.describe(rt)}
    cfg_path = output / 'run.json'
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != run_cfg:
            raise ValueError(f'Run configuration differs from {cfg_path}')
    else:
        cfg_path.write_text(json.dumps(run_cfg, indent=2) + '\n')
    # A batch shares one image size so that every row has the same visual-token count.
    pending.sort(key=lambda c: (tuple(c['truth']['canvas']), c['id']))
    suffix = '' if count == 1 else f'_shard{index}of{count}'
    start = dt.datetime.now().isoformat(timespec='seconds')
    n_done, i, bs = 0, 0, cfg['batch_size']
    with (output / f'results{suffix}.jsonl').open('a') as sink:
        while i < len(pending):
            size = tuple(pending[i]['truth']['canvas'])
            chunk = [pending[i]]
            while len(chunk) < bs and i + len(chunk) < len(pending) \
                    and tuple(pending[i + len(chunk)]['truth']['canvas']) == size:
                chunk.append(pending[i + len(chunk)])
            tic = time.perf_counter()
            outs = qr.generate_batch(rt, [([load_image(c, output)], c['prompt']) for c in chunk], cfg['max_new_tokens'])
            sec = (time.perf_counter() - tic) / len(chunk)
            for c, o in zip(chunk, outs):
                sink.write(json.dumps({'id': c['id'], 'response': o['text'], 'n_tokens': o['n_tokens'],
                                       'finish_reason': o['finish_reason'], 'batch_size': len(chunk),
                                       'seconds': round(sec, 3)}) + '\n')
            sink.flush()
            i += len(chunk)
            n_done += len(chunk)
            if n_done // 400 != (n_done - len(chunk)) // 400:
                print(f'{n_done}/{len(pending)}', flush=True)
    with (output / 'run_log.jsonl').open('a') as log:
        log.write(json.dumps({'shard': list(shard), 'start': start, 'n_run': n_done,
                              'end': dt.datetime.now().isoformat(timespec='seconds')}) + '\n')


def spotcheck(output: Path, cfg: dict, n: int) -> None:
    from analysis_tools.common import qwen_runtime as qr
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    results = {r['id']: r for p in result_files(output) for r in read_jsonl(p)}
    ids = sorted(results)
    pick = [ids[i] for i in np.random.default_rng(0).choice(len(ids), min(n, len(ids)), replace=False)]
    rt = qr.load(cfg['model'], device='cuda:0')
    rows = []
    for cid in pick:
        case = cases[cid]
        out = qr.generate(rt, qr.build(rt, [load_image(case, output)], case['prompt']), cfg['max_new_tokens'])
        batched = {'response': results[cid]['response'], 'finish_reason': results[cid]['finish_reason']}
        single = {'response': out['text'], 'finish_reason': out['finish_reason']}
        rows.append({'id': cid, 'batched': batched['response'], 'single': single['response'],
                     'answer_batched': answer_key(case, batched), 'answer_single': answer_key(case, single)})
    write_jsonl(output / 'spotcheck.jsonl', rows)
    print(f"spotcheck: answer agreement {sum(r['answer_batched'] == r['answer_single'] for r in rows)}/{len(rows)}, "
          f"identical text {sum(r['batched'].strip() == r['single'].strip() for r in rows)}/{len(rows)}")


def answer_key(case: dict, record: dict):
    """Answer-level content used for the spot-check: (rows, cols), (row, col) or the hit cell."""
    s = score_case(case, record, diagnostic=True)
    if not s.get('valid'):
        return None
    if case['format'] == 'count':
        return [s['pred_rows'], s['pred_cols']]
    return s['pred_rc']


# ----------------------------------------------------------------------------- scoring

def score(output: Path) -> None:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    records = [r for p in result_files(output) for r in read_jsonl(p)]
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate results.')
    for name, diagnostic in (('strict', False), ('diagnostic', True)):
        rows = []
        for r in records:
            case = cases[r['id']]
            s = score_case(case, r, diagnostic)
            if case['format'] == 'point':
                s['convention'] = cc.score_point(r['response'], r['finish_reason'], case['truth']['canvas'],
                                                 case['truth']['target_cell_box'])
            rows.append({'id': r['id'], 'finish_reason': r['finish_reason'], **s})
        write_jsonl(output / f'scores_{name}.jsonl', rows)
    print(f'scored {len(records)}/{len(cases)}')


# ----------------------------------------------------------------------------- convention rescoring

def convention_rows(cases: dict, records: list[dict], box_key: str = 'target_cell_box') -> list[dict]:
    rows = []
    for r in records:
        case = cases[r['id']]
        if case['format'] != 'point':
            continue
        t = case['truth']
        rows.append({'id': r['id'], **cc.score_point(r['response'], r['finish_reason'], t['canvas'], t[box_key])})
    return rows


def convention(output: Path, formal: Path, supplement: Path) -> None:
    """Convention columns for earlier runs, written under this experiment's directory (sources read only)."""
    dest = output / 'convention'
    dest.mkdir(parents=True, exist_ok=True)
    formal_cases = {c['id']: c for c in read_jsonl(formal / 'cases.jsonl')}
    rows = convention_rows(formal_cases, read_jsonl(formal / 'results_main.jsonl'))
    write_jsonl(dest / 'formal_main.jsonl', rows)
    print('formal main point answers:', len(rows))
    for part in ('layouts', 'ring'):
        cases = {c['id']: c for c in read_jsonl(supplement / f'cases_{part}.jsonl')}
        records = [r for p in sorted(supplement.glob(f'results_{part}*.jsonl')) for r in read_jsonl(p)
                   if r['id'] in cases]
        rows = convention_rows(cases, records)
        write_jsonl(dest / f'supplement_{part}.jsonl', rows)
        print(f'supplement {part} point answers:', len(rows))


# ----------------------------------------------------------------------------- mechanism

POINTING_HEADS = ((16, 2), (16, 11), (16, 6), (20, 11), (12, 1), (16, 0), (16, 1), (24, 11), (16, 3), (16, 15))
"""Pointing heads selected on 6x6 T6 row/column answers in experiment 3.5 (results.json c1/t6g06_rowcol)."""


def column_prefix(response: str) -> tuple[str, int, str] | None:
    """Split a row/column answer into (text before the column digits, column value, closing brace).

    Accepts the bare object and the same object inside a ```json fence; the prefix keeps the fence so
    that teacher forcing reproduces the context in which the model generated the column ordinal.
    """
    m = re.fullmatch(r'((?:```json\s*)?\{\s*"row"\s*:\s*\d+\s*,\s*"column"\s*:\s*)(\d+)(\s*\})(?:\s*```)?\s*',
                     response.lstrip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3).strip()


def mech(output: Path, cfg: dict, shard: tuple[int, int]) -> None:
    """Teacher-forced column-ordinal log-probabilities and pointing-head attention per token column."""
    import torch
    from analysis_tools.common import qwen_runtime as qr
    from analysis_tools.position import coordinate_readout as cr
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    results = {r['id']: r for p in result_files(output) for r in read_jsonl(p)}
    root = output / 'mech'
    root.mkdir(exist_ok=True)
    index, count = shard
    path = root / f'records{"" if count == 1 else f"_shard{index}of{count}"}.jsonl'
    done = {r['id'] for p in root.glob('records*.jsonl') for r in read_jsonl(p)}
    todo = [cid for cid, c in sorted(cases.items()) if c['task'] == 'T6' and c['format'] == 'row_column'
            and c['condition'] in cfg['mech_conditions'] and cid in results]
    todo = [cid for i, cid in enumerate(todo) if i % count == index and cid not in done]
    print(f'mech shard {index}/{count}: {len(todo)} to run', flush=True)
    if not todo:
        return
    rt = qr.load(cfg['model'], device='cuda:0')
    layers = sorted({l for l, _ in POINTING_HEADS})
    with path.open('a') as sink:
        for cid in todo:
            tic = time.time()
            case, rec = cases[cid], results[cid]
            split = column_prefix(rec['response']) if rec['finish_reason'] == 'eos' else None
            if split is None:
                sink.write(json.dumps({'id': cid, 'status': 'unparsed', 'response': rec['response']}) + '\n')
                continue
            prefix, pred_col, closing = split
            tc = case['truth']['target_rc'][1] + 1
            prep = cr.prepare(rt, load_image(case, output), case['prompt'])
            prefix_ids = rt.tokenizer.encode(prefix, add_special_tokens=False)
            L = len(prep.ids)
            logp = {}
            for v in sorted({tc - 1, tc, tc + 1, tc + 2, pred_col}):
                if v < 1:
                    continue
                ids = rt.tokenizer.encode(prefix + str(v) + closing, add_special_tokens=False)
                if ids[:len(prefix_ids)] != prefix_ids:
                    raise ValueError(f'Prefix tokenization changed for {cid}')
                x, pos = cr.teacher_forced(rt, prep, ids[:-1])
                with torch.inference_mode():
                    lp = rt.model(inputs_embeds=x, position_ids=pos).logits[0].float().log_softmax(-1)
                logp[v] = float(sum(lp[L + i - 1, ids[i]] for i in range(len(prefix_ids), len(ids))))
            # Attention of the pointing heads at the query that emits the first column digit.
            ids = rt.tokenizer.encode(prefix + str(pred_col) + closing, add_special_tokens=False)
            x, pos = cr.teacher_forced(rt, prep, ids[:-1])
            q = L + len(prefix_ids) - 1
            ro = cr.attention_readout(rt, x, pos, [q], (prep.span['start'], prep.span['end']), layers=layers)
            H, W = prep.grid
            s, e = prep.span['start'], prep.span['end']
            profiles = {}
            for layer, head in POINTING_HEADS:
                a = ro.attn[layer][head, 0]
                vis = a[s:e].numpy().reshape(H, W)
                profiles[f'L{layer}H{head}'] = {'col_mass': vis.sum(0).tolist(), 'vis_mass': float(vis.sum())}
            sink.write(json.dumps({'id': cid, 'status': 'ok', 'condition': case['condition'], 'true_col': tc,
                                   'pred_col': pred_col, 'logp': {str(k): v for k, v in logp.items()},
                                   'merged_grid': [H, W], 'heads': profiles,
                                   'check': {str(k): v for k, v in ro.check.items()},
                                   'seconds': round(time.time() - tic, 2)}) + '\n')
            sink.flush()


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('build', 'run', 'spotcheck', 'score', 'mech', 'convention'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--formal', type=Path, help='Formal 1.3 run directory (read only; convention command).')
    p.add_argument('--supplement', type=Path, help='1.3 supplement run directory (read only; convention command).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--conditions', nargs='+', default=[])
    p.add_argument('--t6-per-condition', type=int, default=192)
    p.add_argument('--t0-per-condition', type=int, default=24)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--spotcheck-n', type=int, default=20)
    p.add_argument('--mech-conditions', nargs='+', default=[])
    args = p.parse_args()
    cfg = {k.replace('-', '_'): v for k, v in vars(args).items()}
    for name in args.conditions + args.mech_conditions:
        parse_condition(name)
    m = re.fullmatch(r'(\d+)/(\d+)', args.shard)
    if not m or not int(m.group(1)) < int(m.group(2)):
        p.error('--shard must be index/count with index < count')
    shard = (int(m.group(1)), int(m.group(2)))
    if args.command == 'build':
        if not args.conditions:
            p.error('build needs --conditions')
        build(args.output, cfg)
    elif args.command == 'run':
        run(args.output, cfg, shard, args.limit)
    elif args.command == 'spotcheck':
        spotcheck(args.output, cfg, args.spotcheck_n)
    elif args.command == 'score':
        score(args.output)
    elif args.command == 'mech':
        mech(args.output, cfg, shard)
    else:
        if args.formal is None or args.supplement is None:
            p.error('convention needs --formal and --supplement')
        convention(args.output, args.formal, args.supplement)


if __name__ == '__main__':
    main()
