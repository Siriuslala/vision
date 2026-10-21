"""Experiment 2.6: how Qwen3.5 locates the odd entity among many (stimuli, cases, scoring, metrics).

Design (docs/analysis/representation/design_odd_entity.md).  Every core stimulus comes as a matched
quadruple that shares template, target cell and gap site:

  std   -- the target carries change X, the 35 other entities are the template T   (odd = X)
  swap  -- the target is T, the 35 other entities carry X                         (odd = T)
  homT  -- every entity is T (no odd entity)
  homX  -- every entity carries X (no odd entity)

Because the odd entity is X in `std` and T in `swap`, the within-image contrast of the target,
D = h_target - mean(h_others), splits into a feature part (sign flips between std and swap) and an
oddness part (sign kept).  Their relative size per layer says whether the entity tokens encode
"which feature" or "different from the rest".  `homT` / `homX` supply donor activations for the
context-replacement patching and the reference states of a "majority feature" readout.

Auxiliary groups: grid size (3x3, 10x10), heterogeneous majorities (random template per entity),
half/half boundary images (no odd entity; tests local contrast), and natural object crops (RefCOCO).

This module is model-free; GPU stages live in `odd_entity_run.py`, CPU analysis in
`odd_entity_analysis.py`.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from analysis_tools.behavior.decomposition import (
    EXPLICIT_LOCATE, JSON_ONLY, LOCATE_SUFFIX, T5_QUERY, read_jsonl, unit_seed, write_jsonl,
)
from analysis_tools.common.stimuli import (
    DEVELOPMENT_TEMPLATES, GENERATOR_VERSION, NONE, TOKEN_PIXELS, WHITE, Change, StimulusSpec, canvas_layout,
    render_instance,
)

GENERATOR = GENERATOR_VERSION + '+odd_entity'
# Change kinds: (kind, level); gap sites are drawn per pair.  Levels index stimuli.LEVELS (1-based).
KINDS = {
    'red': ('color', 4),        # pure red, salient colour
    'color20': ('color', 3),    # CIELAB dE 20 from black
    'color10': ('color', 2),    # CIELAB dE 10 (near chance in experiment 1.3)
    'gap16': ('gap', 4),        # 16-pixel gap, salient shape
    'gap2': ('gap', 1),         # 2-pixel gap, subtle shape
}
CORE_VARIANTS = ('std', 'swap', 'homT', 'homX')
ODD_VARIANTS = ('std', 'swap')

DIRECT_PROMPT = T5_QUERY[0] + LOCATE_SUFFIX['row_column'] + JSON_ONLY
THINK_PROMPT = T5_QUERY[0] + EXPLICIT_LOCATE
NATURAL_QUERY = 'One picture shows a different kind of object from all the others.'
NATURAL_DIRECT = NATURAL_QUERY + LOCATE_SUFFIX['row_column'] + JSON_ONLY
NATURAL_THINK = NATURAL_QUERY + EXPLICIT_LOCATE

ROWCOL = re.compile(r'"row"\s*:\s*(\d+)\s*,\s*"column"\s*:\s*(\d+)')

# Natural crops: RefCOCO testA holds people; testB holds other objects, labelled here by keyword.
NATURAL_CATEGORIES = {
    'person': None,             # every testA expression
    'car': ('car',), 'bus': ('bus',), 'truck': ('truck',), 'motorcycle': ('motorcycle', 'motorbike'),
    'bicycle': ('bike', 'bicycle'), 'dog': ('dog',), 'cat': ('cat',), 'horse': ('horse',),
    'elephant': ('elephant',), 'zebra': ('zebra',), 'giraffe': ('giraffe',), 'sheep': ('sheep',),
    'cow': ('cow',), 'bear': ('bear',), 'train': ('train',), 'boat': ('boat',), 'chair': ('chair',),
    'couch': ('couch', 'sofa'), 'bed': ('bed',), 'bowl': ('bowl',), 'cup': ('cup', 'mug'),
    'pizza': ('pizza',), 'cake': ('cake',), 'banana': ('banana',), 'donut': ('donut', 'doughnut'),
    'broccoli': ('broccoli',), 'sandwich': ('sandwich',), 'umbrella': ('umbrella',), 'bottle': ('bottle',),
    'laptop': ('laptop',), 'vase': ('vase',), 'clock': ('clock',),
}
NATURAL_CELL = 128
NATURAL_FIT = 112
NATURAL_MIN_SIDE = 48


# ----------------------------------------------------------------------------- composition

def change_for(kind: str, site: int) -> Change:
    base, level = KINDS[kind]
    return Change(base, level, site if base == 'gap' else 0)


def compose(rows: int, cols: int, cell: int, tiles: dict[tuple[int, int], np.ndarray]) -> tuple[Image.Image, dict]:
    """Place one (cell, cell, 3) tile per grid cell on a token-aligned white canvas (stimuli-v1 geometry)."""
    width, height, ox, oy = canvas_layout(StimulusSpec('person', rows, cols, cell=cell))
    canvas = np.full((height, width, 3), WHITE, np.uint8)
    instances = []
    for r in range(rows):
        for c in range(cols):
            tile = tiles[(r, c)]
            if tile.shape != (cell, cell, 3):
                raise ValueError(f'Tile at {(r, c)} has shape {tile.shape}, expected {(cell, cell, 3)}')
            x0, y0 = ox + c * cell, oy + r * cell
            canvas[y0:y0 + cell, x0:x0 + cell] = tile
            instances.append({'row': r, 'col': c, 'index': r * cols + c,
                              'cell_box': [x0, y0, x0 + cell, y0 + cell],
                              'center': [x0 + cell / 2, y0 + cell / 2]})
    meta = {'canvas': [width, height], 'merged_grid': [height // TOKEN_PIXELS, width // TOKEN_PIXELS],
            'grid_origin': [ox, oy], 'rows': rows, 'cols': cols, 'cell': cell, 'instances': instances}
    return Image.fromarray(canvas), meta


def entity_tokens(meta: dict) -> list[list[int]]:
    """Row-major merged-token indices (within the image) of every entity cell, in entity raster order."""
    width = meta['merged_grid'][1]
    out = []
    for inst in meta['instances']:
        x0, y0, x1, y1 = inst['cell_box']
        if any(v % TOKEN_PIXELS for v in (x0, y0, x1, y1)):
            raise ValueError('Entity cell is not aligned to the 32-pixel token grid')
        out.append([r * width + c for r in range(y0 // TOKEN_PIXELS, y1 // TOKEN_PIXELS)
                    for c in range(x0 // TOKEN_PIXELS, x1 // TOKEN_PIXELS)])
    return out


def neighbours(rows: int, cols: int, rc: tuple[int, int]) -> list[int]:
    """Entity indices of the up to eight cells around rc (Moore neighbourhood)."""
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = rc[0] + dr, rc[1] + dc
            if (dr or dc) and 0 <= r < rows and 0 <= c < cols:
                out.append(r * cols + c)
    return out


def context_groups(rows: int, cols: int, rc: tuple[int, int]) -> dict[str, list[int]]:
    """Non-target entity groups used by the context-replacement patching."""
    t = rc[0] * cols + rc[1]
    others = [i for i in range(rows * cols) if i != t]
    nb = neighbours(rows, cols, rc)
    return {'all': others, 'nbr': nb, 'far': [i for i in others if i not in nb],
            'before': [i for i in others if i < t], 'after': [i for i in others if i > t]}


# ----------------------------------------------------------------------------- stimulus sets

def _save(image: Image.Image, output: Path, rel: str) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    path = output / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Existing image {rel} differs from the regenerated stimulus')
    else:
        path.write_bytes(data)
    return digest


def balanced_targets(rows: int, cols: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Each cell repeated as evenly as possible, in random order (raster-order analysis needs all rows)."""
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    reps = math.ceil(count / len(cells))
    pool = [cells[i] for i in rng.permutation(len(cells) * reps) % len(cells)]
    return pool[:count]


def grid_quadruple(kind: str, template: str, rows: int, cols: int, target: tuple[int, int], site: int,
                   variants=CORE_VARIANTS, templates: dict | None = None) -> dict[str, tuple[Image.Image, dict]]:
    """Render the requested variants.  `templates` (rc -> template) gives heterogeneous majorities."""
    change = change_for(kind, site)
    cache: dict = {}

    def tile(rc, changed):
        key = ((templates or {}).get(rc, template), changed)
        if key not in cache:
            cache[key] = render_instance(key[0], change if changed else NONE, 64)
        return cache[key]

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rule = {'std': lambda rc: rc == target, 'swap': lambda rc: rc != target,
            'homT': lambda rc: False, 'homX': lambda rc: True}
    out = {}
    for v in variants:
        image, meta = compose(rows, cols, 64, {rc: tile(rc, rule[v](rc)) for rc in cells})
        for inst in meta['instances']:
            rc = (inst['row'], inst['col'])
            inst['changed'] = bool(rule[v](rc))
            inst['template'] = (templates or {}).get(rc, template)
        out[v] = (image, meta)
    return out


def boundary_image(kind: str, template: str, orient: str, x_first: bool, site: int) -> tuple[Image.Image, dict]:
    """6x6 image split in half (orient 'LR': left/right columns, 'TB': top/bottom rows) between X and T."""
    change = change_for(kind, site)
    tx, tt = render_instance(template, change, 64), render_instance(template, NONE, 64)
    tiles = {}
    for r in range(6):
        for c in range(6):
            first = (c < 3) if orient == 'LR' else (r < 3)
            tiles[(r, c)] = tx if first == x_first else tt
    image, meta = compose(6, 6, 64, tiles)
    for inst in meta['instances']:
        first = (inst['col'] < 3) if orient == 'LR' else (inst['row'] < 3)
        inst['changed'] = bool(first == x_first)
    return image, meta


def build(output: Path, cfg: dict) -> None:
    """Write every image and the item list (items.jsonl).  Natural crops need the RefCOCO crop bank."""
    output.mkdir(parents=True, exist_ok=True)
    items = []

    def add(image, meta, rel, **fields):
        digest = _save(image, output, rel)
        items.append({'id': rel.removesuffix('.png').replace('/', ':'), 'path': rel, 'sha256': digest,
                      'meta': meta, **fields})

    for group, grids, kinds, pairs, variants in (
            ('core', [6], cfg['core_kinds'], cfg['core_pairs'], CORE_VARIANTS),
            ('grid', cfg['grid_sizes'], cfg['aux_kinds'], cfg['aux_pairs'], ODD_VARIANTS),
            ('hetero', [6], cfg['aux_kinds'], cfg['aux_pairs'], ODD_VARIANTS)):
        for g in grids:
            for kind in kinds:
                rng = np.random.default_rng(unit_seed(cfg['seed'], 'odd_entity', group, g, kind))
                targets = balanced_targets(g, g, pairs, rng)
                for p, target in enumerate(targets):
                    template = DEVELOPMENT_TEMPLATES[p % len(DEVELOPMENT_TEMPLATES)]
                    site = int(rng.integers(0, 3))
                    templates = None
                    if group == 'hetero':
                        templates = {(r, c): DEVELOPMENT_TEMPLATES[int(rng.integers(0, 4))]
                                     for r in range(g) for c in range(g)}
                        template = templates[target]
                    quad = grid_quadruple(kind, template, g, g, target, site, variants, templates)
                    for v, (image, meta) in quad.items():
                        add(image, meta, f'images/{group}/g{g:02d}_{kind}_p{p:03d}_{v}.png', group=group, grid=g,
                            kind=kind, pair=p, variant=v, template=template, site=site, target=list(target),
                            n_entities=g * g)
    for kind in cfg['aux_kinds']:
        rng = np.random.default_rng(unit_seed(cfg['seed'], 'odd_entity', 'boundary', kind))
        for orient in ('LR', 'TB'):
            for p in range(cfg['boundary_per_orient']):
                template = DEVELOPMENT_TEMPLATES[p % len(DEVELOPMENT_TEMPLATES)]
                site, x_first = int(rng.integers(0, 3)), bool(p % 2 == 0)
                image, meta = boundary_image(kind, template, orient, x_first, site)
                add(image, meta, f'images/boundary/{kind}_{orient}_p{p:03d}.png', group='boundary', grid=6, kind=kind,
                    pair=p, variant=f'b{orient}', template=template, site=site, target=None, x_first=x_first,
                    n_entities=36)
    if cfg.get('natural_pairs', 0):
        items.extend(build_natural(output, cfg))
    write_jsonl(output / 'items.jsonl', items)
    (output / 'build_config.json').write_text(json.dumps({**cfg, 'generator': GENERATOR}, indent=1))


# ----------------------------------------------------------------------------- natural crops

def keyword_category(expressions: list[str]) -> str | None:
    """Single category whose keywords occur in the referring expressions, or None if zero or several."""
    text = ' ' + ' '.join(expressions).lower() + ' '
    hits = [name for name, keys in NATURAL_CATEGORIES.items()
            if keys and any(re.search(rf'\b{k}s?\b', text) for k in keys)]
    return hits[0] if len(hits) == 1 else None


def fit_crop(crop: Image.Image) -> np.ndarray:
    """Resize a crop to fit NATURAL_FIT x NATURAL_FIT, centred in a white NATURAL_CELL tile."""
    scale = NATURAL_FIT / max(crop.size)
    w, h = max(1, round(crop.width * scale)), max(1, round(crop.height * scale))
    small = crop.convert('RGB').resize((w, h), Image.BICUBIC)
    tile = Image.new('RGB', (NATURAL_CELL, NATURAL_CELL), (WHITE,) * 3)
    tile.paste(small, ((NATURAL_CELL - w) // 2, (NATURAL_CELL - h) // 2))
    return np.asarray(tile)


def crop_bank(refcoco_dir: Path, per_category_max: int = 64) -> dict[str, list[np.ndarray]]:
    """Distinct object crops per category from RefCOCO testA (person) and testB (keyword-labelled objects)."""
    import pyarrow.parquet as pq
    bank: dict[str, list[np.ndarray]] = {}
    seen = set()
    for split in ('testA', 'testB'):
        for f in sorted((refcoco_dir / 'data').glob(f'{split}-*.parquet')):
            table = pq.read_table(f, columns=['image', 'answer', 'bbox', 'file_name'])
            for row in table.to_pylist():
                x, y, w, h = row['bbox']
                key = (row['file_name'].rsplit('_', 1)[0], round(x), round(y), round(w), round(h))
                if key in seen or min(w, h) < NATURAL_MIN_SIDE:
                    continue
                cat = 'person' if split == 'testA' else keyword_category(row['answer'])
                if cat is None or len(bank.get(cat, [])) >= per_category_max:
                    continue
                seen.add(key)
                image = Image.open(io.BytesIO(row['image']['bytes']))
                bank.setdefault(cat, []).append(fit_crop(image.crop((x, y, x + w, y + h))))
    return bank


def build_natural(output: Path, cfg: dict) -> list[dict]:
    bank = crop_bank(Path(cfg['refcoco_dir']))
    need = 9
    cats = sorted(c for c, v in bank.items() if len(v) >= need)
    if len(cats) < 2:
        raise ValueError(f'Too few natural categories with >= {need} crops: {cats}')
    rng = np.random.default_rng(unit_seed(cfg['seed'], 'odd_entity', 'natural'))
    targets = balanced_targets(3, 3, cfg['natural_pairs'], rng)
    items = []
    for p, target in enumerate(targets):
        a, b = rng.choice(cats, 2, replace=False).tolist()
        ia = rng.choice(len(bank[a]), 9, replace=False)
        ib = rng.choice(len(bank[b]), 9, replace=False)
        for v in ODD_VARIANTS:
            maj, odd = (a, b) if v == 'std' else (b, a)
            mi, oi = (ia, ib) if v == 'std' else (ib, ia)
            tiles, k = {}, 0
            for r in range(3):
                for c in range(3):
                    if (r, c) == tuple(target):
                        tiles[(r, c)] = bank[odd][oi[0]]
                    else:
                        tiles[(r, c)] = bank[maj][mi[k]]
                        k += 1
            image, meta = compose(3, 3, NATURAL_CELL, tiles)
            for inst in meta['instances']:
                inst['category'] = odd if (inst['row'], inst['col']) == tuple(target) else maj
            rel = f'images/natural/p{p:03d}_{v}.png'
            digest = _save(image, output, rel)
            items.append({'id': rel.removesuffix('.png').replace('/', ':'), 'path': rel, 'sha256': digest,
                          'meta': meta, 'group': 'natural', 'grid': 3, 'kind': f'{a}|{b}', 'pair': p, 'variant': v,
                          'template': None, 'site': None, 'target': list(target), 'n_entities': 9,
                          'majority': maj, 'odd': odd})
    return items


# ----------------------------------------------------------------------------- cases and scoring

def prompt_for(item: dict, mode: str) -> str:
    natural = item['group'] == 'natural'
    if mode == 'direct':
        return NATURAL_DIRECT if natural else DIRECT_PROMPT
    if mode == 'think':
        return NATURAL_THINK if natural else THINK_PROMPT
    raise ValueError(f'Unknown mode {mode!r}')


def behaviour_cases(items: list[dict], think_pairs: int) -> list[dict]:
    """Direct cases for every odd image; think cases for the first `think_pairs` pairs of core kinds and natural."""
    cases = []
    for it in items:
        if it['variant'] not in ODD_VARIANTS:
            continue
        modes = ['direct']
        if it['group'] in ('core', 'natural') and it['pair'] < think_pairs:
            modes.append('think')
        for mode in modes:
            cases.append({'id': f'{it["id"]}|{mode}', 'item': it['id'], 'mode': mode, 'prompt': prompt_for(it, mode),
                          'max_new_tokens': 32 if mode == 'direct' else 1024})
    return cases


def parse_answer(text: str) -> tuple[int, int] | None:
    """Last {"row": r, "column": c} pair in the response (1-based), or None."""
    found = ROWCOL.findall(text)
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def score(item: dict, record: dict) -> dict:
    """Strict-content score: the last row/column pair must name the odd entity; truncation counts as failure."""
    rc = parse_answer(record['text']) if record['finish_reason'] == 'eos' else None
    t = item['target']
    out = {'valid': rc is not None, 'pred': None if rc is None else [rc[0] - 1, rc[1] - 1]}
    out['correct'] = bool(rc is not None and rc[0] - 1 == t[0] and rc[1] - 1 == t[1])
    if rc is not None:
        out['d_row'], out['d_col'] = rc[0] - 1 - t[0], rc[1] - 1 - t[1]
        out['adjacent'] = max(abs(out['d_row']), abs(out['d_col'])) == 1
    return out


def digit_spans(text: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Character spans of the row and column numbers of the last row/column pair in `text`."""
    matches = list(ROWCOL.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return m.span(1), m.span(2)


# ----------------------------------------------------------------------------- metrics

def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def contrast(h: np.ndarray, target: int) -> np.ndarray:
    """Within-image contrast of the target: h[target] - mean of the other entities.  h: (n_entities, d)."""
    others = np.delete(h, target, axis=0)
    return h[target] - others.mean(0)


def oddness_split(d_std: np.ndarray, d_swap: np.ndarray) -> dict:
    """Split mean target contrasts into an oddness part (shared sign) and a feature part (flipped sign).

    d_std, d_swap: (n_pairs, d) contrasts of matched std / swap images.  Returns the directions, their
    norms, the cosine between the mean contrasts and the oddness share rho = |u_odd|^2 / (|u_odd|^2 + |u_feat|^2).
    """
    m_std, m_swap = d_std.mean(0), d_swap.mean(0)
    u_odd, u_feat = (m_std + m_swap) / 2, (m_std - m_swap) / 2
    no, nf = float(np.linalg.norm(u_odd)), float(np.linalg.norm(u_feat))
    cos = float(m_std @ m_swap / (np.linalg.norm(m_std) * np.linalg.norm(m_swap) + 1e-12))
    return {'u_odd': u_odd, 'u_feat': u_feat, 'norm_odd': no, 'norm_feat': nf, 'cos': cos,
            'rho': no * no / (no * no + nf * nf + 1e-12)}


def top1(h: np.ndarray, w: np.ndarray, target: int) -> bool:
    """Does direction w rank the target first among the entities of one image?"""
    return int(np.argmax(h @ w)) == target


def outlier_top1(h: np.ndarray, target: int) -> bool:
    """Template-free detector: the entity farthest from the image mean is the target."""
    return int(np.argmax(np.linalg.norm(h - h.mean(0), axis=1))) == target


def margin(h: np.ndarray, w: np.ndarray, target: int) -> float:
    """Target score minus the mean of the others, in units of the others' standard deviation."""
    s = h @ w
    others = np.delete(s, target)
    return float((s[target] - others.mean()) / (others.std() + 1e-12))


def retained(s_patch: float, s_base: float, s_clean: float) -> float:
    """Fraction of the clean-vs-baseline answer score kept after a patch (1 = unchanged, 0 = baseline)."""
    gap = s_clean - s_base
    return float('nan') if abs(gap) < 1e-9 else (s_patch - s_base) / gap


def load_items(output: Path) -> dict[str, dict]:
    return {it['id']: it for it in read_jsonl(output / 'items.jsonl')}
