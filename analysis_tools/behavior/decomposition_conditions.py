"""Supplementary conditions for experiment 1.3 (design_decomposition.md, section 2.2).

Parts (each with its own cases/results/scores files under the output directory):

  t3dev    -- T3 prompt candidates on a balanced subset of the formal T3 images
  t3full   -- the selected T3 prompt on every formal primary-wording T3 case
  layouts  -- non-square grids, fixed 768x768 canvas and 20x20 at cell 32 (T0, T3, T4, T5, T6)
  ring     -- instances on a circle (T0 count, T5, T6; point and clockwise index answers)

Commands: build, run, spotcheck, score.  Images of t3dev / t3full are read from the formal
run directory (never written); every other image is rendered into the output directory.
Yes/No cases are scored from first-token logits of a single prefill (batch size 1, which
reproduces the formal s values exactly); generation cases use left-padded batched greedy
decoding (qwen_runtime.generate_batch) and a spot-check against batch size 1.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import (
    ANSWER_YES_NO, COORD_NOTE, JSON_ONLY, LOCATE_SUFFIX, ORDINAL_NOTE, T0_QUERY, T0_SUFFIX, T4_QUERY, T5_QUERY,
    T6_QUERY, ImageStore, _grid_truth, _lookup, answer_part, parse_json, read_jsonl, score_case, sha256_file,
    unit_seed, write_jsonl,
)
from analysis_tools.common.stimuli import (
    DEVELOPMENT_TEMPLATES, GENERATOR_VERSION, LEVELS, Change, RingSpec, StimulusSpec, generate, generate_ring,
    uniform_targets,
)

PARTS = ('t3dev', 't3full', 'layouts', 'ring')

# ----------------------------------------------------------------------------- T3 prompt candidates
# Each candidate: (question template, yes_means_same).  {D} is the designation phrase; rows/columns
# designations append ORDINAL_NOTE.  C0 reproduces the formal primary wording exactly.
T3_CANDIDATES = {
    'C0_original': ('Is {D} exactly identical to the other figures?', True),
    'C1_differs': ('Is {D} different from the other figures in any way?', False),
    'C2_careful': ('Look carefully at {D} and compare it with the other figures. Is it exactly identical to '
                   'them? The answer may be Yes or No.', True),
    'C3_copy': ('Most figures in this image are identical copies of the same drawing. Is {D} one of these '
                'identical copies? Look carefully; the answer may be Yes or No.', True),
    'C4_reference': ('Is {D} exactly identical to the figure in row {r2}, column {c2}?', True),
    'C5_majority': ('Is there anything about {D} that differs from the majority of the figures?', False),
    'C6_most': ('Does {D} look the same as most of the figures in the image? Look carefully; the answer may be '
                'Yes or No.', True),
    'C7_odd': ('Is {D} the odd one out?', False),
}


def t3_prompt(candidate: str, mode: str, rc, ref_rc=None) -> tuple[str, bool]:
    """Prompt text and polarity for one candidate; mode is 'box' or 'rowcol'."""
    template, yes_same = T3_CANDIDATES[candidate]
    phrase = 'the figure inside the blue box' if mode == 'box' else f'the figure in row {rc[0] + 1}, column {rc[1] + 1}'
    needs_ref = '{r2}' in template
    if needs_ref and ref_rc is None:
        raise ValueError(f'{candidate} needs a reference instance')
    text = template.format(D=phrase, r2=ref_rc[0] + 1 if needs_ref else 0, c2=ref_rc[1] + 1 if needs_ref else 0)
    if mode == 'rowcol' or needs_ref:
        text += ' ' + ORDINAL_NOTE
    return text + ANSWER_YES_NO, yes_same


def reference_rc(grid_rows: int, grid_cols: int, exclude: list, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    flat = [i for i in range(grid_rows * grid_cols) if divmod(i, grid_cols) not in {tuple(e) for e in exclude}]
    return divmod(int(rng.choice(flat)), grid_cols)


def _t3_case(part, base: dict, candidate: str, mode: str, which: str, rc, ref, extra: dict) -> dict:
    prompt, yes_same = t3_prompt(candidate, mode, rc, ref)
    same = which == 'normal'
    return {'id': '|'.join([part, 'T3', base['image_id'], f'{mode}_{which}', candidate]), 'part': part, 'task': 'T3',
            'format': 'yes_no', 'candidate': candidate, 'mode': mode, 'variant': f'{mode}_{which}',
            'image_root': base['image_root'], 'image': base['image'], 'image_sha256': base['image_sha256'],
            'image_id': base['image_id'], 'prompt': prompt, 'thinking': False, 'yes_means_same': yes_same,
            'truth': {'same': same, 'answer_yes': same == yes_same, 'designated_rc': list(rc),
                      'reference_rc': list(ref) if ref is not None else None}, **extra}


def build_t3(formal: Path, part: str, candidates: list[str], dev_reps: int, dev_levels: list[int]) -> list[dict]:
    """T3 cases on formal images; t3dev uses a balanced subset, t3full every primary-wording case."""
    formal_cases = [c for c in read_jsonl(formal / 'cases.jsonl')
                    if c['stage'] == 'main' and c['task'] == 'T3' and c['wording'] == 'primary']
    targets = {(c['record_id'], c['variant']): c['truth']['designated_rc'] for c in formal_cases}
    cases = []
    for c in formal_cases:
        if part == 't3dev' and not (c['rep'] < dev_reps and c['level'] in dev_levels):
            continue
        mode, which = c['variant'].split('_')
        rc, g = c['truth']['designated_rc'], c['grid']
        target = targets[(c['record_id'], f'{mode}_target')]
        # Reference instance for C4: a normal instance different from the designated one.
        ref = reference_rc(g, g, [rc, target], unit_seed(0, 'ref', c['id']))
        base = {'image_root': 'formal', 'image': c['image'], 'image_sha256': c['image_sha256'], 'image_id': c['image_id']}
        extra = {k: c[k] for k in ('kind', 'level', 'grid', 'template', 'rep', 'record_id')}
        extra.update(formal_id=c['id'], layout=f'g{g}', rows=g, cols=g, cell=64, target_rc=list(target))
        for cand in candidates:
            cases.append(_t3_case(part, base, cand, mode, which, rc, ref, extra))
    return cases


# ----------------------------------------------------------------------------- layouts (grids)
# name -> (rows, cols, cell).  Rows are listed first: "4x8" has 4 rows and 8 columns.
LAYOUTS = {
    'g4x8': (4, 8, 64), 'g8x4': (8, 4, 64), 'g10x20': (10, 20, 64), 'g20x10': (20, 10, 64),
    'f6c128': (6, 6, 128), 'f12c64': (12, 12, 64), 'f24c32': (24, 24, 32), 'g20c32': (20, 20, 32),
}
SEARCH_CHANGES = (('gap', 4), ('rotate', 3))    # discriminable changes for T3-T5
RED = ('color', 4)
TEMPLATES_BY_KIND = {'gap': DEVELOPMENT_TEMPLATES, 'color': DEVELOPMENT_TEMPLATES, 'rotate': ('person', 'arrow')}


def _locate_case(part, task, rec, fmt, prompt, truth, group) -> dict:
    return {'id': '|'.join([part, task, rec['image_id'], fmt]), 'part': part, 'task': task, 'format': fmt,
            'variant': 'target', 'image_root': 'self', 'image': rec['path'], 'image_sha256': rec['sha256'],
            'image_id': rec['image_id'], 'prompt': prompt, 'thinking': False, 'truth': truth, 'stage': 'main', **group}


def _anomaly_spec_args(kind: str, level: int, rep: int):
    templates = TEMPLATES_BY_KIND[kind]
    template = templates[rep % len(templates)]
    site = (rep // len(templates)) % 3 if kind == 'gap' else 0
    return template, Change(kind, level, site), site


def build_layouts(output: Path, cfg: dict) -> tuple[list[dict], list[dict]]:
    store = ImageStore(output)
    cases = []
    n_search, n_red, n_count = cfg['search_per_change'], cfg['red_per_layout'], cfg['count_per_layout']
    for name in cfg['layouts']:
        rows, cols, cell = LAYOUTS[name]
        for kind, level in SEARCH_CHANGES + (RED,):
            n = n_red if (kind, level) == RED else n_search
            seed = unit_seed(cfg['seed'], 'supp', name, kind, level)
            targets = uniform_targets(rows, cols, n, seed)
            rng = np.random.default_rng(seed + 1)
            for rep in range(n):
                template, change, site = _anomaly_spec_args(kind, level, rep)
                target = targets[rep]
                spec = StimulusSpec(template, rows, cols, change, target=target, cell=cell, seed=seed + rep)
                img, meta = generate(spec)
                truth = _grid_truth(meta, spec)
                stem = f'{name}_{kind}_L{level}_r{rep:03d}'
                rec = store.save(img, f'images/{name}/{stem}.png', role='grid', spec=meta['spec'], **truth)
                group = {'layout': name, 'rows': rows, 'cols': cols, 'cell': cell, 'kind': kind, 'level': level,
                         'magnitude': LEVELS[kind][level - 1], 'template': template, 'site': site, 'rep': rep,
                         'record_id': stem, 'grid': max(rows, cols)}
                fmts = ('row_column', 'point')
                if (kind, level) == RED:
                    for fmt in fmts:
                        cases.append(_locate_case('layouts', 'T6', rec, fmt, T6_QUERY[0] + LOCATE_SUFFIX[fmt] + JSON_ONLY,
                                                  truth, group))
                    continue
                for task, query in (('T4', _lookup(T4_QUERY, kind, template)[0]), ('T5', T5_QUERY[0])):
                    for fmt in fmts:
                        cases.append(_locate_case('layouts', task, rec, fmt, query + LOCATE_SUFFIX[fmt] + JSON_ONLY,
                                                  truth, group))
                flat = [i for i in range(rows * cols) if i != target[0] * cols + target[1]]
                normal = divmod(int(rng.choice(flat)), cols)
                base = {'image_root': 'self', 'image': rec['path'], 'image_sha256': rec['sha256'],
                        'image_id': rec['image_id']}
                extra = {**group, 'target_rc': list(target), 'stage': 'main'}
                for which, rc in (('target', target), ('normal', normal)):
                    for cand in cfg['layout_t3_candidates']:
                        cases.append(_t3_case('layouts', base, cand, 'rowcol', which, rc, None, extra))
        seed = unit_seed(cfg['seed'], 'supp', name, 'count')
        for rep in range(n_count):
            template = DEVELOPMENT_TEMPLATES[rep % len(DEVELOPMENT_TEMPLATES)]
            spec = StimulusSpec(template, rows, cols, cell=cell, seed=seed + rep)
            img, meta = generate(spec)
            truth = _grid_truth(meta, spec)
            stem = f'{name}_count_r{rep:03d}'
            rec = store.save(img, f'images/{name}/{stem}.png', role='count', spec=meta['spec'], **truth)
            group = {'layout': name, 'rows': rows, 'cols': cols, 'cell': cell, 'kind': 'none', 'level': 0,
                     'magnitude': None, 'template': template, 'site': 0, 'rep': rep, 'record_id': stem,
                     'grid': max(rows, cols)}
            cases.append(_locate_case('layouts', 'T0', rec, 'count', T0_QUERY[0] + T0_SUFFIX + JSON_ONLY, truth, group))
    return store.records, cases


# ----------------------------------------------------------------------------- ring
# name -> (n, radius); all rings use a 768x768 canvas and 64-pixel cells.
RINGS = {'n8': (8, 352), 'n12': (12, 352), 'n16': (16, 352), 'n24': (24, 352), 'n8_r128': (8, 128),
         'n8_r256': (8, 256)}
RING_CANVAS = 768
RING_INDEX = (' The figures are arranged evenly on a circle. Number them clockwise, starting with the figure at '
              'the top (the 12 o\'clock position) as figure 1. Give the number of this figure. Return only '
              '{"index": k}.')
RING_COUNT = 'How many figures are in this image? Return only {"count": N}.'


def ring_truth(meta: dict) -> dict:
    target = next((i for i in meta['instances'] if i['is_target']), None)
    spec = meta['spec']
    out = {'canvas': meta['canvas'], 'ring_center': meta['ring_center'], 'radius': spec['radius'], 'n': spec['n'],
           'cell': spec['cell'], 'instance_centers': [i['center'] for i in meta['instances']],
           'instance_cell_boxes': [i['cell_box'] for i in meta['instances']]}
    if target is not None:
        out.update(target_index=target['index'], target_center=target['center'], target_cell_box=target['cell_box'],
                   target_tight_box=target['tight_box'], target_angle_deg=target['angle_deg'])
    return out


def build_ring(output: Path, cfg: dict) -> tuple[list[dict], list[dict]]:
    store = ImageStore(output)
    cases = []
    for name in cfg['rings']:
        n_inst, radius = RINGS[name]
        changes = (SEARCH_CHANGES + (RED,)) if name in cfg['ring_full'] else (RED,)
        for kind, level in changes:
            per = cfg['red_per_layout'] if (kind, level) == RED else cfg['search_per_change']
            seed = unit_seed(cfg['seed'], 'ring', name, kind, level)
            targets = np.random.default_rng(seed).integers(0, n_inst, per).tolist()
            for rep in range(per):
                template, change, site = _anomaly_spec_args(kind, level, rep)
                spec = RingSpec(template, n_inst, radius, RING_CANVAS, change, target=int(targets[rep]), seed=seed + rep)
                img, meta = generate_ring(spec)
                truth = ring_truth(meta)
                stem = f'ring_{name}_{kind}_L{level}_r{rep:03d}'
                rec = store.save(img, f'images/ring/{stem}.png', role='ring', spec=meta['spec'], **truth)
                group = {'layout': name, 'n': n_inst, 'radius': radius, 'kind': kind, 'level': level,
                         'magnitude': LEVELS[kind][level - 1], 'template': template, 'site': site, 'rep': rep,
                         'record_id': stem}
                task, query = ('T6', T6_QUERY[0]) if (kind, level) == RED else ('T5', T5_QUERY[0])
                cases.append(_locate_case('ring', task, rec, 'point', query + LOCATE_SUFFIX['point'] + JSON_ONLY,
                                          truth, group))
                cases.append(_locate_case('ring', task, rec, 'index', query + RING_INDEX + JSON_ONLY, truth, group))
        if name in cfg['ring_full']:
            seed = unit_seed(cfg['seed'], 'ring', name, 'count')
            for rep in range(cfg['count_per_layout']):
                template = DEVELOPMENT_TEMPLATES[rep % len(DEVELOPMENT_TEMPLATES)]
                img, meta = generate_ring(RingSpec(template, n_inst, radius, RING_CANVAS, seed=seed + rep))
                truth = ring_truth(meta)
                stem = f'ring_{name}_count_r{rep:03d}'
                rec = store.save(img, f'images/ring/{stem}.png', role='ring_count', spec=meta['spec'], **truth)
                group = {'layout': name, 'n': n_inst, 'radius': radius, 'kind': 'none', 'level': 0, 'magnitude': None,
                         'template': template, 'site': 0, 'rep': rep, 'record_id': stem}
                cases.append(_locate_case('ring', 'T0', rec, 'ring_count', RING_COUNT + JSON_ONLY, truth, group))
    return store.records, cases


# ----------------------------------------------------------------------------- build / run

BUILD_KEYS = ('seed', 'dev_reps', 'dev_levels', 't3_candidates', 't3_selected', 'layouts', 'layout_t3_candidates',
              'search_per_change', 'red_per_layout', 'count_per_layout', 'rings', 'ring_full')


def build(output: Path, formal: Path, cfg: dict, part: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if part == 't3dev':
        images, cases = [], build_t3(formal, part, cfg['t3_candidates'], cfg['dev_reps'], cfg['dev_levels'])
    elif part == 't3full':
        if not cfg['t3_selected']:
            raise ValueError('t3full requires --t3-selected (chosen on t3dev).')
        images, cases = [], build_t3(formal, part, [cfg['t3_selected']], 10 ** 9, list(range(1, 5)))
    elif part == 'layouts':
        images, cases = build_layouts(output, cfg)
    else:
        images, cases = build_ring(output, cfg)
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case identifiers.')
    path = output / f'cases_{part}.jsonl'
    if path.exists():
        # Rebuilding may add cases (e.g. the selected T3 prompt) but never alter or drop existing ones.
        new = {c['id']: c for c in cases}
        if any(new.get(c['id']) != c for c in read_jsonl(path)):
            raise ValueError(f'Existing {path.name} differs from the regenerated cases.')
    write_jsonl(path, cases)
    if images:
        write_jsonl(output / f'images_{part}.jsonl', images)
    cfg_path = output / 'config.json'
    stored = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    stored.setdefault('build', {})[part] = {k: cfg[k] for k in BUILD_KEYS}
    stored['generator'] = GENERATOR_VERSION
    stored.setdefault('n_cases', {})[part] = len(cases)
    cfg_path.write_text(json.dumps(stored, indent=2) + '\n')
    print(part, len(cases), 'cases', len(images), 'images')


def image_path(case: dict, output: Path, formal: Path) -> Path:
    return (formal if case['image_root'] == 'formal' else output) / case['image']


def load_image(case, output, formal) -> Image.Image:
    path = image_path(case, output, formal)
    if sha256_file(path) != case['image_sha256']:
        raise ValueError(f'Image hash mismatch for {path}')
    return Image.open(path).convert('RGB')


def results_files(output: Path, part: str) -> list[Path]:
    return sorted(output.glob(f'results_{part}*.jsonl'))


def run(output: Path, formal: Path, cfg: dict, part: str, shard: tuple[int, int], limit: int | None,
        task: str | None = None) -> None:
    from analysis_tools.common import qwen_runtime as qr
    cases = read_jsonl(output / f'cases_{part}.jsonl')
    done = {r['id'] for p in results_files(output, part) for r in read_jsonl(p)}
    index, count = shard
    pending = [c for i, c in enumerate(cases) if i % count == index and c['id'] not in done
               and (task is None or c['task'] == task)]
    if limit is not None:
        pending = pending[:limit]
    print(f'{part} shard {index}/{count}: {len(done)} done overall, {len(pending)} to run', flush=True)
    if not pending:
        return
    rt = qr.load(cfg['model'], device='cuda:0')
    run_cfg = {'model': cfg['model'], 'max_new_tokens': cfg['max_new_tokens'], 'batch_size': cfg['batch_size'],
               'binary_batch_size': 1, 'provenance': qr.describe(rt)}
    cfg_path = output / f'run_{part}.json'
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != run_cfg:
            raise ValueError(f'Run configuration differs from {cfg_path}')
    else:
        cfg_path.write_text(json.dumps(run_cfg, indent=2) + '\n')
    suffix = ('' if count == 1 else f'_shard{index}of{count}') + (f'_{task}' if task else '')
    start = dt.datetime.now().isoformat(timespec='seconds')
    binary = [c for c in pending if c['format'] == 'yes_no']
    gen = [c for c in pending if c['format'] != 'yes_no']
    # Group generation cases by image size so that a batch shares one visual-token count.
    gen.sort(key=lambda c: (tuple(c['truth']['canvas']), c['id']))
    n_done = 0
    with (output / f'results_{part}{suffix}.jsonl').open('a') as sink:
        for case in binary:
            tic = time.perf_counter()
            batch = qr.build(rt, [load_image(case, output, formal)], case['prompt'])
            out = qr.yes_no(rt, batch)
            sink.write(json.dumps({'id': case['id'], **out, 'prompt_tokens': int(batch['input_ids'].shape[1]),
                                   'seconds': round(time.perf_counter() - tic, 3)}) + '\n')
            sink.flush()
            n_done += 1
            if n_done % 200 == 0:
                print(f'{part}: {n_done}/{len(pending)}', flush=True)
        bs = cfg['batch_size']
        i = 0
        while i < len(gen):
            size = tuple(gen[i]['truth']['canvas'])
            chunk = [gen[i]]
            while len(chunk) < bs and i + len(chunk) < len(gen) and tuple(gen[i + len(chunk)]['truth']['canvas']) == size:
                chunk.append(gen[i + len(chunk)])
            tic = time.perf_counter()
            outs = qr.generate_batch(rt, [([load_image(c, output, formal)], c['prompt']) for c in chunk],
                                     cfg['max_new_tokens'])
            sec = (time.perf_counter() - tic) / len(chunk)
            for c, o in zip(chunk, outs):
                sink.write(json.dumps({'id': c['id'], 'response': o['text'], 'n_tokens': o['n_tokens'],
                                       'finish_reason': o['finish_reason'], 'batch_size': len(chunk),
                                       'seconds': round(sec, 3)}) + '\n')
            sink.flush()
            i += len(chunk)
            n_done += len(chunk)
            if n_done // 200 != (n_done - len(chunk)) // 200:
                print(f'{part}: {n_done}/{len(pending)}', flush=True)
    with (output / 'run_log.jsonl').open('a') as log:
        log.write(json.dumps({'part': part, 'shard': list(shard), 'start': start,
                              'end': dt.datetime.now().isoformat(timespec='seconds'), 'n_run': n_done}) + '\n')


def spotcheck(output: Path, formal: Path, cfg: dict, part: str, n: int) -> None:
    """Re-run n random generation cases at batch size 1 (qwen_runtime.generate) and store both answers."""
    from analysis_tools.common import qwen_runtime as qr
    cases = {c['id']: c for c in read_jsonl(output / f'cases_{part}.jsonl') if c['format'] != 'yes_no'}
    results = {r['id']: r for p in results_files(output, part) for r in read_jsonl(p) if r['id'] in cases}
    ids = sorted(results)
    pick = [ids[i] for i in np.random.default_rng(0).choice(len(ids), min(n, len(ids)), replace=False)]
    rt = qr.load(cfg['model'], device='cuda:0')
    rows = []
    for cid in pick:
        case = cases[cid]
        out = qr.generate(rt, qr.build(rt, [load_image(case, output, formal)], case['prompt']), cfg['max_new_tokens'])
        rows.append({'id': cid, 'batched': results[cid]['response'], 'single': out['text'],
                     'single_finish_reason': out['finish_reason']})
    write_jsonl(output / f'spotcheck_{part}.jsonl', rows)
    print(f'spotcheck {part}: {sum(r["batched"].strip() == r["single"].strip() for r in rows)}/{len(rows)} identical text')


# ----------------------------------------------------------------------------- scoring

def _circular(d: int, n: int) -> int:
    """Signed circular difference in (-n/2, n/2]."""
    d = (d + n) % n
    return d - n if d > n // 2 else d


def score_ring(case: dict, record: dict, diagnostic: bool) -> dict:
    out = {'valid': False, 'correct': False}
    if record['finish_reason'] != 'eos':
        return {**out, 'failure': 'truncated'}
    try:
        value = parse_json(answer_part(record['response']))
    except ValueError:
        return {**out, 'failure': 'unparsable'}
    t = case['truth']
    if diagnostic and isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict) \
            and case['format'] != 'point':
        value = value[0]
    if case['format'] == 'ring_count':
        if not isinstance(value, dict) or type(value.get('count')) is not int:
            return {**out, 'failure': 'schema'}
        return {'valid': True, 'correct': value['count'] == t['n'], 'pred_count': value['count']}
    if case['format'] == 'index':
        if not isinstance(value, dict) or type(value.get('index')) is not int:
            return {**out, 'failure': 'schema'}
        k = value['index']
        if not 1 <= k <= t['n']:
            return {**out, 'failure': 'out_of_range', 'pred_index': k - 1}
        return {'valid': True, 'correct': k - 1 == t['target_index'], 'pred_index': k - 1,
                'd_index': _circular(k - 1 - t['target_index'], t['n'])}
    if diagnostic and isinstance(value, dict):
        value = [value]
    if diagnostic and isinstance(value, list) and len(value) == 2 and all(type(v) in (int, float) for v in value):
        value = [{'point_2d': value}]
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict)
            and isinstance(value[0].get('point_2d'), list) and len(value[0]['point_2d']) == 2
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in value[0]['point_2d'])):
        return {**out, 'failure': 'schema'}
    p = np.array(value[0]['point_2d'], float) * np.array(t['canvas'], float) / 1000
    centers = np.array(t['instance_centers'])
    nearest = int(np.argmin(np.linalg.norm(centers - p, axis=1)))
    x0, y0, x1, y1 = t['target_cell_box']
    hit = bool(x0 <= p[0] < x1 and y0 <= p[1] < y1)
    c = np.array(t['ring_center'], float)
    tp = np.array(t['target_center'], float)
    pred_r, true_r = float(np.linalg.norm(p - c)), float(np.linalg.norm(tp - c))
    ang = lambda v: math.degrees(math.atan2(v[0], -v[1])) % 360    # clockwise from 12 o'clock
    d_ang = (ang(p - c) - ang(tp - c) + 180) % 360 - 180
    return {'valid': True, 'correct': hit, 'hit': hit, 'pred_center': p.tolist(),
            'cell_distance': float(np.linalg.norm(p - tp)) / t['cell'], 'nearest_index': nearest,
            'd_index_nearest': _circular(nearest - t['target_index'], t['n']),
            'radial_ratio': pred_r / true_r, 'pred_radius': pred_r, 'true_radius': true_r,
            'd_angle_deg': d_ang, 'd_angle_steps': d_ang / (360 / t['n'])}


def score_t3(case: dict, record: dict) -> dict:
    s_same = record['s'] if case['yes_means_same'] else -record['s']
    return {'valid': True, 's': record['s'], 's_same': s_same, 'pred_same': s_same > 0,
            'correct': (s_same > 0) == case['truth']['same'], 'top_token': record['top_token']}


def score(output: Path, part: str) -> None:
    cases = {c['id']: c for c in read_jsonl(output / f'cases_{part}.jsonl')}
    records = [r for p in results_files(output, part) for r in read_jsonl(p)]
    if len({r['id'] for r in records}) != len(records):
        raise ValueError(f'Duplicate results for {part}')
    for name, diagnostic in (('strict', False), ('diagnostic', True)):
        rows = []
        for r in records:
            case = cases[r['id']]
            if case['format'] == 'yes_no':
                s = score_t3(case, r)
            elif part == 'ring':
                s = score_ring(case, r, diagnostic)
            else:
                s = score_case(case, r, diagnostic)
            rows.append({'id': r['id'], **s})
        write_jsonl(output / f'scores_{part}_{name}.jsonl', rows)
    print(f'{part}: scored {len(records)}/{len(cases)}')


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('build', 'run', 'spotcheck', 'score'))
    p.add_argument('--part', choices=PARTS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--formal', type=Path, required=True, help='Formal experiment 1.3 output directory (read only).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dev-reps', type=int, default=4)
    p.add_argument('--dev-levels', type=int, nargs='+', default=[2, 4])
    p.add_argument('--t3-candidates', nargs='+', default=list(T3_CANDIDATES))
    p.add_argument('--t3-selected', default='')
    p.add_argument('--layouts', nargs='+', default=list(LAYOUTS))
    p.add_argument('--layout-t3-candidates', nargs='+', default=['C0_original'])
    p.add_argument('--search-per-change', type=int, default=96)
    p.add_argument('--red-per-layout', type=int, default=192)
    p.add_argument('--count-per-layout', type=int, default=48)
    p.add_argument('--rings', nargs='+', default=list(RINGS))
    p.add_argument('--ring-full', nargs='+', default=['n8', 'n12', 'n16', 'n24'])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--shard', default='0/1', help='index/count; cases with position % count == index')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--task', default=None, help='Run only cases of this sub-task (e.g. T3).')
    p.add_argument('--spotcheck-n', type=int, default=20)
    args = p.parse_args()
    cfg = {k.replace('-', '_'): v for k, v in vars(args).items()}
    for cand in args.t3_candidates + args.layout_t3_candidates + ([args.t3_selected] if args.t3_selected else []):
        if cand not in T3_CANDIDATES:
            p.error(f'unknown T3 candidate {cand}')
    for name in args.layouts:
        if name not in LAYOUTS:
            p.error(f'unknown layout {name}')
    for name in args.rings + args.ring_full:
        if name not in RINGS:
            p.error(f'unknown ring {name}')
    m = re.fullmatch(r'(\d+)/(\d+)', args.shard)
    if not m or not int(m.group(1)) < int(m.group(2)):
        p.error('--shard must be index/count with index < count')
    shard = (int(m.group(1)), int(m.group(2)))
    if args.command == 'build':
        build(args.output, args.formal, cfg, args.part)
    elif args.command == 'run':
        run(args.output, args.formal, cfg, args.part, shard, args.limit, args.task)
    elif args.command == 'spotcheck':
        spotcheck(args.output, args.formal, cfg, args.part, args.spotcheck_n)
    else:
        score(args.output, args.part)


if __name__ == '__main__':
    main()
