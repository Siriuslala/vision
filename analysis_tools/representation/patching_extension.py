"""Experiments 2.5b and 2.5c: activation patching on failure cases and during answer generation.

Design: docs/analysis/representation/design_causal.md (sections 2.5b and 2.5c).

Both experiments share one engine.  A pair consists of a clean and a corrupted image with the
same prompt.  The receiver sequence is the prompt followed by a teacher-forced answer prefix that
is identical for both inputs; the score at the final position is

    s = z_a - z_b,

where a is the token the clean input generates at the first position where the two answers
diverge (or 'Yes' for yes/no questions) and b the corresponding corrupted-input token (or 'No').
For 2.5b failure pairs of the find-the-different task, a is the correct digit and b the digit the
model actually produced.

Patch modes at decoder site l (0 = decoder input embeddings, k = output of decoder layer k):
  std      replace token group G at site l; every later computation is free.
  direct   G <- source and every other prompt position <- receiver at all sites >= l; only answer
           positions are recomputed, so the effect reaches the score only through answer tokens
           reading G during generation.
  direct1  G <- source at site l only, every prompt position <- receiver at sites > l; answer
           positions read the source values of G only in decoder layer l + 1.
  via      post-image prompt text <- its values from the std run of the same (l, G), visual
           tokens <- receiver, at all sites > l; the effect reaches the score only through the
           prompt-side relay.

Layer numbers are 1-based; site 0 is the merger output written into the decoder input.
Stages: screen (decode answers, qualify pairs) -> scan (patching) -> analyze (summaries, figures).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

from analysis_tools.common import stimuli
from analysis_tools.common.stimuli import (
    DEVELOPMENT_TEMPLATES, NONE, TOKEN_PIXELS, Change, StimulusSpec, generate,
)

N_LAYERS = 32
SITES = tuple(range(N_LAYERS + 1))          # 0 = decoder input embeddings, 1..32 = layer outputs
FULL_ATTENTION_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MIN_SCORE_GAP = 1.0

# Prompts reused verbatim from experiment 1.3 (analysis_tools/behavior/decomposition.py) and the
# revised T3 wording of the 1.3 supplement (C5_majority, plan.md section 6 item 8).
LOCATE_SUFFIX = {
    'rowcol': ' Give its position. Return only {"row": r, "column": c}, counting rows from top to bottom '
              'and columns from left to right, starting at 1.',
    'point': ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].'
             ' All coordinates use 0-1000 normalization; x increases rightward and y increases downward.',
}
JSON_ONLY = ' Output only the JSON, with no explanation.'
T5_QUERY = 'One figure differs from all the others.'
T6_QUERY = 'One figure is drawn in red.'
ORDINAL_NOTE = 'Rows are counted from top to bottom and columns from left to right, starting at 1.'
T3_REVISED = 'Is there anything about the figure in row {r}, column {c} that differs from the majority of the figures?'
ANSWER_YES_NO = ' Answer with only Yes or No.'


def locate_prompt(query: str, fmt: str) -> str:
    return query + LOCATE_SUFFIX[fmt] + JSON_ONLY


def t3_prompt(rc: tuple[int, int]) -> str:
    """Revised T3 question; 'Yes' means the designated figure differs (correct for the clean image)."""
    return T3_REVISED.format(r=rc[0] + 1, c=rc[1] + 1) + ' ' + ORDINAL_NOTE + ANSWER_YES_NO


# ---------------------------------------------------------------------------- conditions
# kind: 'gen' (answer generated, score at the first diverging answer token) or 'yesno'.
# pairing: 'move' (anomaly at a different cell in the corrupted image, along `axis`),
#          'remove' (corrupted image has no anomaly), 'fail' (records from experiment 1.3).
GEN_CONDITIONS = {
    't6_rowcol_col': dict(query=T6_QUERY, fmt='rowcol', change=('color', 4), grid=6, pairing='move', axis='col'),
    't6_rowcol_row': dict(query=T6_QUERY, fmt='rowcol', change=('color', 4), grid=6, pairing='move', axis='row'),
    't6_point_x': dict(query=T6_QUERY, fmt='point', change=('color', 4), grid=6, pairing='move', axis='col'),
    't6_point_y': dict(query=T6_QUERY, fmt='point', change=('color', 4), grid=6, pairing='move', axis='row'),
    't5_rowcol_col': dict(query=T5_QUERY, fmt='rowcol', change=('gap', 4), grid=6, pairing='move', axis='col'),
    't5_rowcol_row': dict(query=T5_QUERY, fmt='rowcol', change=('gap', 4), grid=6, pairing='move', axis='row'),
    't5_rowcol_remove': dict(query=T5_QUERY, fmt='rowcol', change=('gap', 4), grid=6, pairing='remove', axis=None),
}
YESNO_CONDITIONS = {
    't3_gap16_g06': dict(change=('gap', 4), grid=6),
    't3_gap16_g10': dict(change=('gap', 4), grid=10),
    't3_gap16_g20': dict(change=('gap', 4), grid=20),
    't3_red_g06': dict(change=('color', 4), grid=6),
    't3_color20_g06': dict(change=('color', 3), grid=6),
    't3_color10_g06': dict(change=('color', 2), grid=6),
}
FAIL_CONDITIONS = ('t5_fail', 't3_fail')
ALL_CONDITIONS = tuple(GEN_CONDITIONS) + tuple(YESNO_CONDITIONS) + FAIL_CONDITIONS

VISUAL_GROUPS = ('target_clean', 'target_corrupt', 'targets', 'random', 'nontarget', 'all_visual')
TEXT_GROUPS = ('question', 'qmark', 'q_end', 'think_end', 'last_prompt', 'prompt_text')
ANSWER_GROUPS = ('answer_prefix', 'query')
PATH_GROUPS = ('targets', 'all_visual')        # groups that also get direct / direct1 / via modes
PROBE_GROUPS = ('target_clean', 'qmark', 'q_end', 'think_end', 'last_prompt')


@dataclass(frozen=True)
class Pair:
    pair_id: str
    condition: str
    kind: str                         # 'gen' or 'yesno'
    fmt: str                          # 'rowcol', 'point' or 'yesno'
    prompt: str
    clean: StimulusSpec
    corrupt: StimulusSpec
    clean_target: tuple[int, int]
    corrupt_target: tuple[int, int] | None
    axis: str | None = None
    source: dict = field(default_factory=dict, compare=False, hash=False)


def spec_from_dict(d: dict) -> StimulusSpec:
    d = dict(d)
    d['change'] = Change(**d['change'])
    for key in ('target', 'marker'):
        if d.get(key) is not None:
            d[key] = tuple(d[key])
    return StimulusSpec(**d)


def _other(rng: np.random.Generator, n: int, exclude: int) -> int:
    return int(rng.choice([k for k in range(n) if k != exclude]))


def make_gen_pairs(condition: str, count: int, seed: int) -> list[Pair]:
    """Pairs for 2.5c.  'move' keeps the other coordinate fixed so that the answers share a prefix."""
    cfg = GEN_CONDITIONS[condition]
    g = cfg['grid']
    rng = np.random.default_rng([seed, int(hashlib.sha256(condition.encode()).hexdigest()[:8], 16)])
    pairs = []
    for i in range(count):
        template = DEVELOPMENT_TEMPLATES[i % len(DEVELOPMENT_TEMPLATES)]
        site = int(rng.integers(0, 3)) if cfg['change'][0] == 'gap' else 0
        change = Change(cfg['change'][0], cfg['change'][1], site)
        r, c = int(rng.integers(0, g)), int(rng.integers(0, g))
        common = dict(template=template, rows=g, cols=g, cell=64, seed=seed)
        clean = StimulusSpec(change=change, target=(r, c), **common)
        if cfg['pairing'] == 'move':
            dst = (r, _other(rng, g, c)) if cfg['axis'] == 'col' else (_other(rng, g, r), c)
            corrupt = StimulusSpec(change=change, target=dst, **common)
        else:
            dst, corrupt = None, StimulusSpec(**common)
        prompt = locate_prompt(cfg['query'], cfg['fmt'])
        pairs.append(Pair(f'{condition}_{i:03d}', condition, 'gen', cfg['fmt'], prompt, clean, corrupt,
                          (r, c), dst, cfg['axis']))
    return pairs


def make_yesno_pairs(condition: str, count: int, seed: int) -> list[Pair]:
    """Revised-T3 pairs for 2.5b: the designated cell holds the anomaly (clean) or the template."""
    cfg = YESNO_CONDITIONS[condition]
    g = cfg['grid']
    targets = stimuli.uniform_targets(g, g, count, seed)
    sites = np.random.default_rng(seed + 1).integers(0, 3, count).tolist()
    pairs = []
    for i, (rc, site) in enumerate(zip(targets, sites)):
        template = DEVELOPMENT_TEMPLATES[i % len(DEVELOPMENT_TEMPLATES)]
        change = Change(cfg['change'][0], cfg['change'][1], int(site) if cfg['change'][0] == 'gap' else 0)
        common = dict(template=template, rows=g, cols=g, cell=64, seed=seed)
        clean = StimulusSpec(change=change, target=rc, **common)
        pairs.append(Pair(f'{condition}_{i:03d}', condition, 'yesno', 'yesno', t3_prompt(rc), clean,
                          StimulusSpec(**common), rc, None))
    return pairs


def make_fail_pairs(condition: str, decomposition_dir: Path, grids: tuple[int, ...], max_per_grid: int,
                    seed: int) -> list[Pair]:
    """Experiment-1.3 images where T2 succeeded and T5 (row/column) failed in content.

    A content failure is a parsable {"row": r, "column": c} pair (malformed closing brackets are
    tolerated here) that differs from the truth.  Answers with the correct row and column but a
    malformed JSON closing bracket are format failures and are excluded; their count is reported
    by `fail_census`.  The corrupted image is the same specification without the anomaly.
    """
    census = fail_census(decomposition_dir, grids)
    pairs = []
    rng = np.random.default_rng(seed)
    for g in grids:
        rows = [r for r in census['records'] if r['grid'] == g and r['class'] == 'content_wrong']
        if max_per_grid and len(rows) > max_per_grid:
            rows = [rows[k] for k in sorted(rng.choice(len(rows), max_per_grid, replace=False))]
        for row in rows:
            clean = spec_from_dict(row['spec'])
            corrupt = replace(clean, change=NONE, target=None)
            rc = tuple(row['target_rc'])
            if condition == 't5_fail':
                kind, fmt, prompt = 'gen', 'rowcol', locate_prompt(T5_QUERY, 'rowcol')
            else:
                kind, fmt, prompt = 'yesno', 'yesno', t3_prompt(rc)
            pairs.append(Pair(f"{condition}_{row['record_id']}", condition, kind, fmt, prompt, clean, corrupt,
                              rc, None, None, source=row))
    return pairs


ROWCOL_RE = re.compile(r'"row":\s*(\d+),\s*"column":\s*(\d+)')


def fail_census(decomposition_dir: Path, grids: tuple[int, ...]) -> dict:
    """Classify 1.3 primary-wording T5 row/column answers on images whose T2 pair was correct."""
    d = Path(decomposition_dir)
    per_image = json.load(open(d / 'analysis' / 'per_image_records.json'))
    images = {}
    for line in open(d / 'images.jsonl'):
        rec = json.loads(line)
        images[rec['image_id']] = rec
    responses = {}
    for line in open(d / 'results_main.jsonl'):
        rec = json.loads(line)
        responses[rec['id']] = rec['response']
    records, counts = [], {}
    for line in open(d / 'cases.jsonl'):
        case = json.loads(line)
        if not (case['task'] == 'T5' and case['format'] == 'row_column' and case['wording'] == 'primary'
                and case['stage'] == 'main' and case['grid'] in grids):
            continue
        flags = per_image[case['record_id']]
        if not flags.get('T2') or flags.get('T5 row/col') is not False:
            continue
        m = ROWCOL_RE.search(responses[case['id']])
        truth = tuple(case['truth']['target_rc'])
        if m is None:
            cls = 'unparsable'
        else:
            pred = (int(m[1]) - 1, int(m[2]) - 1)
            cls = 'format_only' if pred == truth else 'content_wrong'
        counts.setdefault(case['grid'], {}).setdefault(cls, 0)
        counts[case['grid']][cls] += 1
        image = images[case['image_id']]
        records.append({'record_id': case['record_id'], 'grid': case['grid'], 'kind': case['kind'],
                        'level': case['level'], 'template': case['template'], 'target_rc': list(truth),
                        'class': cls, 'response_13': responses[case['id']], 'spec': image['spec'],
                        'image_sha256': image['sha256']})
    return {'counts': counts, 'records': records}


def render_pair(pair: Pair):
    """Render both images and check that pixels differ only inside the anomaly cells."""
    clean_img, clean_meta = generate(pair.clean)
    corrupt_img, corrupt_meta = generate(pair.corrupt)
    if pair.source.get('image_sha256') and clean_meta['sha256'] != pair.source['image_sha256']:
        raise ValueError(f'{pair.pair_id}: regenerated image differs from the experiment-1.3 image')
    diff = (np.asarray(clean_img) != np.asarray(corrupt_img)).any(-1)
    if not diff.any():
        raise ValueError(f'{pair.pair_id}: clean and corrupted images are identical')
    allowed = np.zeros_like(diff)
    for rc in (pair.clean_target, pair.corrupt_target):
        if rc is not None:
            x0, y0, x1, y1 = cell_box(clean_meta, rc)
            allowed[y0:y1, x0:x1] = True
    if (diff & ~allowed).any():
        raise ValueError(f'{pair.pair_id}: pixel difference outside the anomaly cells')
    return clean_img, clean_meta, corrupt_img, corrupt_meta


def cell_box(meta: dict, rc: tuple[int, int]) -> list[int]:
    return [int(v) for v in meta['instances'][rc[0] * meta['spec']['cols'] + rc[1]]['cell_box']]


def cell_tokens(meta: dict, rc: tuple[int, int]) -> list[int]:
    """Raster indices (within the image) of the merged tokens covered by one instance cell."""
    x0, y0, x1, y1 = cell_box(meta, rc)
    if any(v % TOKEN_PIXELS for v in (x0, y0, x1, y1)):
        raise ValueError('Cell is not aligned to the 32-pixel token grid')
    width = meta['merged_grid'][1]
    return [r * width + c for r in range(y0 // TOKEN_PIXELS, y1 // TOKEN_PIXELS)
            for c in range(x0 // TOKEN_PIXELS, x1 // TOKEN_PIXELS)]


# ---------------------------------------------------------------------------- answers

def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing token, or None if one sequence is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def parse_rowcol_loose(text: str) -> tuple[int, int] | None:
    """One-based (row, column) from the first '"row": r, "column": c' match, tolerating a bad bracket."""
    m = ROWCOL_RE.search(text)
    return (int(m[1]), int(m[2])) if m else None


def coordinate_spans(text: str, fmt: str) -> list[tuple[int, int]]:
    """Character spans of the coordinate values in answer order (rowcol: row, column; point: x, y)."""
    if fmt == 'rowcol':
        m = ROWCOL_RE.search(text)
        return [m.span(1), m.span(2)] if m else []
    m = re.search(r'"point_2d":\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    return [m.span(1), m.span(2)] if m else []


def token_char_spans(decode, ids: list[int]) -> list[tuple[int, int]]:
    ends = [len(decode(ids[:i + 1])) for i in range(len(ids))]
    return list(zip([0] + ends[:-1], ends))


def coordinate_of_token(decode, ids: list[int], fmt: str, index: int) -> int | None:
    """Which coordinate (0 or 1) the token at `index` belongs to, or None if it is not a coordinate digit."""
    text = decode(ids)
    spans = coordinate_spans(text, fmt)
    a, b = token_char_spans(decode, ids)[index]
    for k, (s, e) in enumerate(spans):
        if a >= s and b <= e and b > a:
            return k
    return None


def point_cell(coords: tuple[float, float], meta: dict) -> tuple[int, int] | None:
    """Grid cell hit by a 0-1000 point answer, or None if it falls outside the grid."""
    w, h = meta['canvas']
    ox, oy = meta['grid_origin']
    cell = meta['spec']['cell']
    x, y = coords[0] * w / 1000, coords[1] * h / 1000
    c, r = math.floor((x - ox) / cell), math.floor((y - oy) / cell)
    if 0 <= r < meta['spec']['rows'] and 0 <= c < meta['spec']['cols']:
        return r, c
    return None


def answer_cell(text: str, fmt: str, meta: dict) -> tuple[int, int] | None:
    """Zero-based cell named by an answer (row/column or point), None when unparsable."""
    if fmt == 'rowcol':
        rc = parse_rowcol_loose(text)
        return None if rc is None else (rc[0] - 1, rc[1] - 1)
    m = re.search(r'"point_2d":\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    return None if m is None else point_cell((float(m[1]), float(m[2])), meta)


# ---------------------------------------------------------------------------- ratios and summaries

def recovery(s_patched, s_corrupt, s_clean):
    s_patched, s_corrupt, s_clean = (np.asarray(v, float) for v in (s_patched, s_corrupt, s_clean))
    if np.any(s_clean == s_corrupt):
        raise ValueError('Recovery ratio undefined for identical clean and corrupted scores')
    return (s_patched - s_corrupt) / (s_clean - s_corrupt)


def damage(s_patched, s_corrupt, s_clean):
    s_patched, s_corrupt, s_clean = (np.asarray(v, float) for v in (s_patched, s_corrupt, s_clean))
    if np.any(s_clean == s_corrupt):
        raise ValueError('Damage ratio undefined for identical clean and corrupted scores')
    return (s_clean - s_patched) / (s_clean - s_corrupt)


def bootstrap_summary(numerators, denominators, n_boot: int, seed: int) -> dict:
    """Pair-level bootstrap of the median ratio and of the ratio of means."""
    num, den = np.asarray(numerators, float), np.asarray(denominators, float)
    ratios = num / den
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratios), (n_boot, len(ratios)))
    med = np.median(ratios[idx], axis=1)
    rom = num[idx].mean(1) / den[idx].mean(1)
    return {'n': int(len(ratios)), 'median': float(np.median(ratios)),
            'median_lo': float(np.percentile(med, 2.5)), 'median_hi': float(np.percentile(med, 97.5)),
            'ratio_of_means': float(num.mean() / den.mean()),
            'rom_lo': float(np.percentile(rom, 2.5)), 'rom_hi': float(np.percentile(rom, 97.5)),
            'mean_shift': float(num.mean()), 'mean_total': float(den.mean())}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def paired_probe(deltas: np.ndarray, folds: int, seed: int) -> dict:
    """Cross-validated difference-of-means probe on paired differences.

    deltas: (n_pairs, D) = h(clean) - h(corrupt) for one site and token group.  The direction is the
    mean difference of the training folds; a held-out pair counts as correct when its difference has
    a positive projection.  Chance level is 0.5.  d' is mean / std of held-out projections.
    """
    n = len(deltas)
    order = np.random.default_rng(seed).permutation(n)
    proj = np.zeros(n)
    for f in range(folds):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        w = deltas[train].mean(0)
        norm = np.linalg.norm(w)
        proj[test] = deltas[test] @ (w / norm) if norm > 0 else 0.0
    sd = proj.std()
    return {'acc': float((proj > 0).mean()), 'dprime': float(proj.mean() / sd) if sd > 0 else 0.0,
            'proj': proj.tolist()}


# ---------------------------------------------------------------------------- engine

def _tensor(output):
    return output[0] if isinstance(output, tuple) else output


@dataclass
class Sequence:
    """Teacher-forced receiver sequence of one input.

    x:   (1, L, D) decoder input embeddings; pos: (4, 1, L) position ids.
    prompt_len: number of prompt tokens P; positions >= P are teacher-forced answer tokens.
    """
    x: torch.Tensor
    pos: torch.Tensor
    ids: list[int]
    prompt_len: int
    span: dict


def build_sequence(rt, image, prompt: str, prefix: list[int]) -> Sequence:
    from analysis_tools.position import coordinate_readout as cr
    prep = cr.prepare(rt, image, prompt)
    x = cr.embeds(rt, prep)
    pos = prep.pos
    if prefix:
        x, pos = cr.teacher_forced(rt, prep, prefix, x, pos)
    return Sequence(x, pos, prep.ids.tolist() + list(prefix), len(prep.ids), dict(prep.span))


class Engine:
    """Batched teacher-forced forward passes with per-row multi-site replacements.

    An edit is (site, positions, source) where source is a (L, D) tensor indexed by positions at hook
    time, so the same cached tensor serves many rows without copies.
    """

    def __init__(self, rt, batch_size: int):
        self.rt, self.B = rt, batch_size

    @torch.inference_mode()
    def run(self, seq: Sequence, rows: list[list[tuple[int, torch.Tensor, torch.Tensor]]],
            token_a: int, token_b: int, capture_all: bool = False,
            capture_rows: dict[int, tuple[int, torch.Tensor]] | None = None, return_logits: bool = False):
        """Score s = z_a - z_b at the last position for every row.

        capture_all: return the full hidden states of row 0 at every site (list of (L, D)).
        capture_rows: row -> (first site, positions); returns {row: {site: (len(positions), D)}}.
        """
        if len(rows) > self.B:
            raise ValueError('More rows than the batch size')
        B = self.B
        x = seq.x.expand(B, -1, -1).clone()
        pos = seq.pos.expand(-1, B, -1).contiguous()
        by_site: dict[int, list] = {}
        for r, edits in enumerate(rows):
            for site, positions, source in edits:
                if site == 0:
                    p = positions.to(x.device)
                    x[r, p] = source[p].to(x.dtype)
                else:
                    by_site.setdefault(site, []).append((r, positions, source))
        full, captured, handles = [], {}, []
        if capture_all:
            full.append(x[0].clone())
        for r, (first, positions) in (capture_rows or {}).items():
            if first == 0:
                captured.setdefault(r, {})[0] = x[r, positions.to(x.device)].clone()
        for layer in range(1, N_LAYERS + 1):
            if layer not in by_site and not capture_all and not capture_rows:
                continue

            def hook(module, inputs, output, layer=layer):
                h = _tensor(output)
                for r, positions, source in by_site.get(layer, []):
                    p = positions.to(h.device)
                    h[r, p] = source[p].to(h.dtype)
                if capture_all:
                    full.append(h[0].clone())
                for r, (first, positions) in (capture_rows or {}).items():
                    if layer >= first:
                        captured.setdefault(r, {})[layer] = h[r, positions.to(h.device)].clone()
                return output
            handles.append(self.rt.layers[layer - 1].register_forward_hook(hook))
        try:
            logits = self.rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1).logits
        finally:
            for handle in handles:
                handle.remove()
        logits = logits[:, -1].float()
        n = max(len(rows), 1)
        s = (logits[:, token_a] - logits[:, token_b]).tolist()[:n]
        top = logits[:n].argmax(-1).tolist()
        if return_logits:
            return s, full, captured, top, logits[:n].cpu()
        return s, full, captured, top


# ---------------------------------------------------------------------------- groups

def token_groups(rt, seq: Sequence, pair: Pair, meta: dict, salt: int) -> dict[str, list[int]]:
    """Sequence positions of every token group (see module docstring and the design document)."""
    s0, s1 = seq.span['start'], seq.span['end']
    n_visual = s1 - s0
    tc = cell_tokens(meta, pair.clean_target)
    td = cell_tokens(meta, pair.corrupt_target) if pair.corrupt_target is not None else []
    targets = sorted(set(tc) | set(td))
    pool = np.setdiff1d(np.arange(n_visual), np.asarray(targets))
    rnd = sorted(np.random.default_rng(salt).choice(pool, len(tc), replace=False).tolist())
    vis = {'target_clean': tc, 'targets': targets, 'random': rnd,
           'nontarget': pool.tolist(), 'all_visual': list(range(n_visual))}
    if td:
        vis['target_corrupt'] = td
    groups = {k: [s0 + t for t in v] for k, v in vis.items()}
    ids, P = seq.ids, seq.prompt_len
    tok = rt.tokenizer
    if ids[s1] != tok.convert_tokens_to_ids('<|vision_end|>'):
        raise ValueError('Image tokens are not followed by <|vision_end|>')
    im_end = ids.index(tok.convert_tokens_to_ids('<|im_end|>'), s1)
    think_end = P - 1 - ids[:P][::-1].index(tok.convert_tokens_to_ids('</think>'))
    question = list(range(s1 + 1, im_end))
    qmarks = [i for i in question if '?' in tok.decode([ids[i]])]
    groups.update({'question': question, 'q_end': [im_end - 1], 'think_end': [think_end],
                   'last_prompt': [P - 1], 'prompt_text': list(range(s1, P))})
    if qmarks:
        groups['qmark'] = [qmarks[-1]]
    if len(ids) > P:
        groups['answer_prefix'] = list(range(P, len(ids) - 1)) if len(ids) - 1 > P else []
        groups['query'] = [len(ids) - 1]
        if not groups['answer_prefix']:
            del groups['answer_prefix']
    return groups


# ---------------------------------------------------------------------------- jobs

@dataclass(frozen=True)
class Job:
    mode: str          # std | direct | direct1 | via | sanity_all | sanity_identity
    site: int
    group: str


# Job sets.  Generation pairs need the path modes and many answer-side groups, so the damage
# direction keeps only the groups needed to confirm the restore-direction conclusions.
JOB_SETS = {
    ('gen', 'restore'): {'std': ('target_clean', 'target_corrupt', 'targets', 'random', 'all_visual', 'q_end',
                                 'think_end', 'last_prompt', 'prompt_text', 'answer_prefix', 'query'),
                         'direct': ('targets', 'all_visual'), 'direct1': ('all_visual',), 'via': ('all_visual',)},
    ('gen', 'damage'): {'std': ('targets', 'all_visual', 'last_prompt', 'prompt_text', 'query'),
                        'direct': ('all_visual',), 'via': ('all_visual',)},
    ('yesno', 'restore'): {'std': ('target_clean', 'nontarget', 'all_visual', 'qmark', 'q_end', 'think_end',
                                   'last_prompt', 'prompt_text')},
    ('yesno', 'damage'): {'std': ('target_clean', 'nontarget', 'all_visual', 'qmark', 'q_end', 'think_end',
                                  'last_prompt', 'prompt_text')},
}
VISUAL_SET = set(VISUAL_GROUPS)


def build_jobs(groups: dict[str, list[int]], kind: str, direction: str, sites: tuple[int, ...],
               sanity_sites: tuple[int, ...]) -> list['Job']:
    """Jobs of one pair and direction.  Text and answer groups exist only at decoder layer outputs."""
    jobs = []
    for site in sites:
        for mode, names in JOB_SETS[(kind, direction)].items():
            for g in names:
                if g in groups and (site > 0 or g in VISUAL_SET):
                    jobs.append(Job(mode, site, g))
    jobs.extend(Job('sanity_all', s, 'all') for s in sanity_sites)
    jobs.append(Job('sanity_identity', sanity_sites[0], 'all'))
    return jobs


def edits_for(job: Job, groups: dict, seq_len: int, prompt_len: int, src: list, rec: list,
              std_capture: dict | None) -> list:
    """Translate one job into a list of (site, positions, source tensor) edits."""
    L = torch.arange(seq_len)
    if job.mode == 'sanity_all':
        return [(job.site, L, src[job.site])]
    if job.mode == 'sanity_identity':
        return [(job.site, L, rec[job.site])]
    g = torch.tensor(groups[job.group], dtype=torch.long)
    if job.mode == 'std':
        return [(job.site, g, src[job.site])]
    prompt = torch.arange(prompt_len)
    others = prompt[~torch.isin(prompt, g)]
    if job.mode == 'direct':
        out = []
        for k in range(job.site, N_LAYERS + 1):
            out += [(k, g, src[k]), (k, others, rec[k])]
        return out
    if job.mode == 'direct1':
        out = [(job.site, g, src[job.site]), (job.site, others, rec[job.site])]
        out += [(k, prompt, rec[k]) for k in range(job.site + 1, N_LAYERS + 1)]
        return out
    if job.mode == 'via':
        text = torch.tensor(groups['prompt_text'], dtype=torch.long)
        visual = torch.tensor(groups['all_visual'], dtype=torch.long)
        out = []
        for k in range(job.site + 1, N_LAYERS + 1):
            # std_capture[k] holds the std-run values at the prompt-text positions; scatter them
            # into a full-length tensor so that the edit indexes by sequence position.
            full = rec[k].clone()
            full[text.to(full.device)] = std_capture[k]
            out += [(k, text, full), (k, visual, rec[k])]
        return out
    raise ValueError(job.mode)


# ---------------------------------------------------------------------------- stages

def code_hashes() -> dict:
    from analysis_tools.common import qwen_runtime
    from analysis_tools.position import coordinate_readout
    files = {'patching_extension': Path(__file__), 'qwen_runtime': Path(qwen_runtime.__file__),
             'stimuli': Path(stimuli.__file__), 'coordinate_readout': Path(coordinate_readout.__file__)}
    return {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


def read_all(out: Path, stem: str) -> list[dict]:
    """Concatenate `<stem>_<tag>.jsonl` files written by parallel processes."""
    rows = []
    for path in sorted(Path(out).glob(f'{stem}_*.jsonl')):
        rows += read_jsonl(path)
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, 'a') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def pairs_for(condition: str, args) -> list[Pair]:
    if condition in GEN_CONDITIONS:
        return make_gen_pairs(condition, args.pairs_per_condition, args.seed)
    if condition in YESNO_CONDITIONS:
        return make_yesno_pairs(condition, args.yesno_pairs, args.seed)
    return make_fail_pairs(condition, Path(args.decomposition_dir), tuple(args.fail_grids), args.fail_max_per_grid,
                           args.seed)


def batch_size_for(pair: Pair, args) -> int:
    g = pair.clean.rows
    return args.batch_size if g <= 6 else (max(1, args.batch_size // 2) if g <= 10 else max(1, args.batch_size // 4))


def screen_pair(rt, pair: Pair, args) -> dict:
    """Decode (gen) or score (yesno) both inputs and decide the scored tokens and the shared prefix."""
    from analysis_tools.common import qwen_runtime as qr
    from analysis_tools.position import coordinate_readout as cr
    clean_img, clean_meta, corrupt_img, _ = render_pair(pair)
    row = {'pair_id': pair.pair_id, 'condition': pair.condition, 'kind': pair.kind, 'fmt': pair.fmt,
           'template': pair.clean.template, 'grid': pair.clean.rows, 'clean_target': list(pair.clean_target),
           'corrupt_target': list(pair.corrupt_target) if pair.corrupt_target else None, 'axis': pair.axis,
           'prompt': pair.prompt, 'clean_sha256': clean_meta['sha256'],
           'clean_spec': {**asdict(pair.clean), 'change': asdict(pair.clean.change)},
           'corrupt_spec': {**asdict(pair.corrupt), 'change': asdict(pair.corrupt.change)}}
    if pair.source:
        row['source'] = {k: pair.source[k] for k in ('record_id', 'kind', 'level', 'response_13')}
    if pair.kind == 'yesno':
        row.update(prefix=[], token_a=qr.token_id(rt, 'Yes'), token_b=qr.token_id(rt, 'No'), status='scan')
        return row
    preps = [cr.prepare(rt, img, pair.prompt) for img in (clean_img, corrupt_img)]
    outs = cr.run_conditions(rt, [cr.embeds(rt, p) for p in preps], [p.pos for p in preps],
                             max_new_tokens=args.max_new_tokens, batch_size=2)
    decode = lambda ids: rt.tokenizer.decode(ids, skip_special_tokens=False)
    texts = [o['text'] for o in outs]
    cells = [answer_cell(t, pair.fmt, clean_meta) for t in texts]
    row.update(clean_text=texts[0], corrupt_text=texts[1], clean_ids=outs[0]['ids'], corrupt_ids=outs[1]['ids'],
               clean_cell=cells[0], corrupt_cell=cells[1],
               finish=[o['finish_reason'] for o in outs])
    clean_ok = cells[0] is not None and tuple(cells[0]) == pair.clean_target
    row['clean_correct'] = clean_ok
    digits_c = coordinate_digits(decode, outs[0]['ids'], pair.fmt)
    if pair.condition == 't5_fail':
        truth = [str(pair.clean_target[0] + 1), str(pair.clean_target[1] + 1)]
        return choose_contrast(row, outs[0]['ids'], digits_c, truth, 'truth', rt)
    corrupt_ok = (cells[1] is not None and tuple(cells[1]) == pair.corrupt_target) if pair.corrupt_target else None
    row['corrupt_correct'] = corrupt_ok
    if not clean_ok:
        row['status'] = 'clean_wrong'
        return row
    if pair.corrupt_target is not None and not corrupt_ok:
        row['status'] = 'corrupt_wrong'
        return row
    digits_k = coordinate_digits(decode, outs[1]['ids'], pair.fmt)
    other = [v for _, v in digits_k] if digits_k else None
    want = None if pair.axis is None else AXIS_INDEX[pair.fmt][pair.axis]
    if other is None and pair.corrupt_target is None and digits_c is not None:
        # Removal pairs: the corrupted answer may be prose.  Score the first row digit against the
        # digit the corrupted input ranks highest at that position under the clean answer prefix.
        i = digits_c[0][0][0]
        seq = build_sequence(rt, corrupt_img, pair.prompt, outs[0]['ids'][:i])
        _, _, _, _, logits = Engine(rt, 1).run(seq, [[]], 0, 0, return_logits=True)
        digit_ids = [rt.tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]
        a = outs[0]['ids'][i]
        b = max((d for d in digit_ids if d != a), key=lambda d: float(logits[0, d]))
        row.update(prefix=outs[0]['ids'][:i], token_a=a, token_b=b, divergence=i, coordinate=0, digit_index=0,
                   b_source='corrupt_top_digit', status='scan')
        return row
    return choose_contrast(row, outs[0]['ids'], digits_c, other, 'corrupt_answer', rt, want)


# Coordinate index of a grid axis in each answer format: rowcol = (row, column), point = (x, y).
AXIS_INDEX = {'rowcol': {'row': 0, 'col': 1}, 'point': {'col': 0, 'row': 1}}


def coordinate_digits(decode, ids: list[int], fmt: str) -> list[tuple[list[int], str]] | None:
    """Per coordinate value: (token indices of its digits, value string); None if not parsable.

    Every digit must be its own token so that a single-token contrast is defined.
    """
    text = decode(ids)
    spans = coordinate_spans(text, fmt)
    if not spans:
        return None
    chars = token_char_spans(decode, ids)
    out = []
    for s, e in spans:
        toks = [i for i, (a, b) in enumerate(chars) if a >= s and b <= e and b > a]
        if ''.join(decode([ids[i]]) for i in toks) != text[s:e] or len(toks) != e - s:
            return None
        out.append((toks, text[s:e]))
    return out


def choose_contrast(row: dict, ids: list[int], digits: list | None, other: list[str] | None, b_source: str,
                    rt, want: int | None = None) -> dict:
    """Select the scored digit: the first digit where the clean answer and `other` values differ.

    The receiver prefix is the clean answer up to that digit.  For 2.5c, a is the clean digit and b
    the digit of the corrupted answer (formats of the two answers may differ; only the values are
    compared).  For failure pairs `other` is the truth and a is the correct digit, b the produced
    one.  `want` restricts the search to one coordinate.
    """
    if digits is None:
        row['status'] = 'unparsable'
        return row
    if other is None:
        row['status'] = 'corrupt_unparsable'
        return row
    for k, (toks, value) in enumerate(digits):
        if want is not None and k != want:
            continue
        if value == other[k]:
            continue
        diff = [j for j in range(min(len(value), len(other[k]))) if value[j] != other[k][j]]
        if not diff:            # e.g. 8 vs 83: no aligned digit differs
            row.update(status='digit_count_differs', coordinate=k)
            return row
        j = diff[0]
        i = toks[j]
        clean_tok = ids[i]
        other_tok = rt.tokenizer.convert_tokens_to_ids(other[k][j])
        a, b = (other_tok, clean_tok) if b_source == 'truth' else (clean_tok, other_tok)
        row.update(prefix=ids[:i], token_a=a, token_b=b, divergence=i, coordinate=k, digit_index=j,
                   b_source=b_source, status='scan')
        return row
    row['status'] = 'clean_correct' if b_source == 'truth' else 'same_coordinate'
    return row


def stage_screen(rt, args, out: Path) -> dict:
    path = out / f'screen_{args.tag}.jsonl'
    done = {r['pair_id'] for r in read_jsonl(path)}
    stats, t0 = {}, time.time()
    for condition in args.conditions:
        if condition in FAIL_CONDITIONS:
            census = fail_census(Path(args.decomposition_dir), tuple(args.fail_grids))
            (out / 'fail_census.json').write_text(json.dumps(census['counts'], indent=1))
        for pair in pairs_for(condition, args):
            if pair.pair_id in done:
                continue
            append_jsonl(path, [screen_pair(rt, pair, args)])
        rows = [r for r in read_jsonl(path) if r['condition'] == condition]
        stats[condition] = {}
        for r in rows:
            stats[condition][r['status']] = stats[condition].get(r['status'], 0) + 1
        print(condition, stats[condition], flush=True)
    return {'screen_status': stats, 'screen_seconds': time.time() - t0}


def scan_pair(rt, row: dict, args, sites: tuple[int, ...]) -> tuple[list[dict], dict, dict | None]:
    pair = pair_from_row(row, args)
    clean_img, meta, corrupt_img, _ = render_pair(pair)
    seq_c = build_sequence(rt, clean_img, pair.prompt, row['prefix'])
    seq_k = build_sequence(rt, corrupt_img, pair.prompt, row['prefix'])
    if seq_c.ids != seq_k.ids:
        raise ValueError(f'{pair.pair_id}: token sequences differ between clean and corrupted inputs')
    engine = Engine(rt, batch_size_for(pair, args))
    a, b = row['token_a'], row['token_b']
    (s_clean,), cache_c, _, (top_c,) = engine.run(seq_c, [[]], a, b, capture_all=True)
    (s_corrupt,), cache_k, _, (top_k,) = engine.run(seq_k, [[]], a, b, capture_all=True)
    base = {'pair_id': pair.pair_id, 's_clean': s_clean, 's_corrupt': s_corrupt, 'top_clean': top_c,
            'top_corrupt': top_k, 'batch_size': engine.B, 'seq_len': len(seq_c.ids), 'prompt_len': seq_c.prompt_len}
    salt = int(hashlib.sha256(pair.pair_id.encode()).hexdigest()[:8], 16)
    groups = token_groups(rt, seq_c, pair, meta, salt)
    base['group_sizes'] = {k: len(v) for k, v in groups.items()}
    # Probe deltas (clean - corrupt) at every site: mean over the clean target cell and single text tokens.
    probe = None
    if pair.kind == 'yesno':
        rows_ = []
        for g in PROBE_GROUPS:
            if g not in groups:
                rows_.append(torch.zeros(len(SITES), cache_c[0].shape[-1]))
                continue
            idx = torch.tensor(groups[g], device=cache_c[0].device)
            rows_.append(torch.stack([(cache_c[k][idx].float() - cache_k[k][idx].float()).mean(0) for k in SITES]).cpu())
        probe = torch.stack(rows_, 1).half().numpy()            # (sites, groups, D)
    gap = s_clean - s_corrupt
    base['gap'] = gap
    if gap == 0:
        base['status'] = 'zero_gap'
        return [], base, probe
    records = []
    L, P = len(seq_c.ids), seq_c.prompt_len
    for direction, seq, src, rec, s_rec in (('restore', seq_k, cache_c, cache_k, s_corrupt),
                                             ('damage', seq_c, cache_k, cache_c, s_clean)):
        jobs = build_jobs(groups, pair.kind, direction, sites, tuple(args.sanity_sites))
        std_jobs = [j for j in jobs if j.mode != 'via']
        via_jobs = [j for j in jobs if j.mode == 'via']
        captured_std = {}
        text_pos = torch.tensor(groups['prompt_text'], dtype=torch.long)
        need = {(j.site, j.group) for j in via_jobs}
        results = []
        for start in range(0, len(std_jobs), engine.B):
            chunk = std_jobs[start:start + engine.B]
            rows_ = [edits_for(j, groups, L, P, src, rec, None) for j in chunk]
            cap = {r: (j.site + 1, text_pos) for r, j in enumerate(chunk)
                   if j.mode == 'std' and (j.site, j.group) in need}
            scores, _, captured, _ = engine.run(seq, rows_, a, b, capture_rows=cap)
            for r, j in enumerate(chunk):
                if r in captured:
                    captured_std[(j.site, j.group)] = captured[r]
            results += list(zip(chunk, scores))
        for start in range(0, len(via_jobs), engine.B):
            chunk = via_jobs[start:start + engine.B]
            rows_ = [edits_for(j, groups, L, P, src, rec, captured_std.get((j.site, j.group), {})) for j in chunk]
            scores, _, _, _ = engine.run(seq, rows_, a, b)
            results += list(zip(chunk, scores))
        for j, s in results:
            ratio = recovery(s, s_corrupt, s_clean) if direction == 'restore' else damage(s, s_corrupt, s_clean)
            records.append({'pair_id': pair.pair_id, 'condition': pair.condition, 'direction': direction,
                            'mode': j.mode, 'site': j.site, 'group': j.group, 's_patched': s,
                            's_clean': s_clean, 's_corrupt': s_corrupt, 'ratio': float(ratio)})
        del captured_std
    base['status'] = 'ok'
    return records, base, probe


_PAIR_CACHE: dict[str, list[Pair]] = {}


def pair_from_row(row: dict, args) -> Pair:
    if row['condition'] not in _PAIR_CACHE:
        _PAIR_CACHE[row['condition']] = pairs_for(row['condition'], args)
    for pair in _PAIR_CACHE[row['condition']]:
        if pair.pair_id == row['pair_id']:
            if pair.prompt != row['prompt'] or list(pair.clean_target) != row['clean_target']:
                raise ValueError(f"{row['pair_id']} cannot be regenerated consistently")
            return pair
    raise ValueError(f"{row['pair_id']} not found")


def stage_scan(rt, args, out: Path) -> dict:
    screen = [r for r in read_jsonl(out / f'screen_{args.tag}.jsonl')
              if r['condition'] in args.conditions and r['status'] == 'scan']
    rec_path, base_path = out / f'records_{args.tag}.jsonl', out / f'baselines_{args.tag}.jsonl'
    probe_dir = out / 'probe_deltas'
    probe_dir.mkdir(exist_ok=True)
    done = {r['pair_id'] for r in read_jsonl(base_path)}
    sites = tuple(args.sites)
    t0, n = time.time(), 0
    if args.reverse:
        screen = screen[::-1]
    for row in screen:
        # Parallel processes (different tags) may scan the same conditions from opposite ends;
        # re-read every baselines file so that a pair finished elsewhere is skipped.
        done |= {b['pair_id'] for b in read_all(out, 'baselines')}
        if row['pair_id'] in done:
            continue
        t = time.time()
        records, base, probe = scan_pair(rt, row, args, sites if row['kind'] == 'gen' else tuple(args.yesno_sites))
        if probe is not None:
            np.save(probe_dir / f"{row['pair_id']}.npy", probe)
        append_jsonl(rec_path, records)
        append_jsonl(base_path, [{**base, 'condition': row['condition'], 'seconds': time.time() - t}])
        n += 1
        print(f"{row['pair_id']}: {len(records)} records, gap {base['gap']:.2f}, {time.time() - t:.1f}s", flush=True)
    return {'scan_pairs': n, 'scan_seconds': time.time() - t0, 'records_file': rec_path.name}


def stage_detail(rt, args, out: Path) -> dict:
    """Single answer-token patches (restore direction, std mode) on scanned generation pairs.

    Locates which generated token carries the information read from the image: every answer
    position P..L-1 is replaced alone at each site in args.detail_sites.
    """
    screen = {r['pair_id']: r for r in read_all(out, 'screen')}
    bases = {b['pair_id']: b for b in read_all(out, 'baselines')}
    path = out / f'detail_{args.tag}.jsonl'
    done = {r['pair_id'] for r in read_jsonl(path)}
    t0, n = time.time(), 0
    for pid, base in bases.items():
        row = screen[pid]
        if row['condition'] not in args.conditions or row['kind'] != 'gen' or pid in done:
            continue
        pair = pair_from_row(row, args)
        clean_img, _, corrupt_img, _ = render_pair(pair)
        seq_c = build_sequence(rt, clean_img, pair.prompt, row['prefix'])
        seq_k = build_sequence(rt, corrupt_img, pair.prompt, row['prefix'])
        engine = Engine(rt, batch_size_for(pair, args))
        a, b = row['token_a'], row['token_b']
        (s_clean,), cache_c, _, _ = engine.run(seq_c, [[]], a, b, capture_all=True)
        (s_corrupt,), _, _, _ = engine.run(seq_k, [[]], a, b)
        L, P = len(seq_c.ids), seq_c.prompt_len
        jobs = [(site, t) for site in args.detail_sites for t in range(P, L)]
        recs = []
        for start in range(0, len(jobs), engine.B):
            chunk = jobs[start:start + engine.B]
            rows_ = [[(site, torch.tensor([t]), cache_c[site])] for site, t in chunk]
            scores, _, _, _ = engine.run(seq_k, rows_, a, b)
            for (site, t), sc in zip(chunk, scores):
                recs.append({'pair_id': pid, 'condition': row['condition'], 'site': site, 'offset': t - (L - 1),
                             'token': rt.tokenizer.decode([seq_c.ids[t]]), 's_patched': sc, 's_clean': s_clean,
                             's_corrupt': s_corrupt, 'ratio': float(recovery(sc, s_corrupt, s_clean))})
        append_jsonl(path, recs)
        n += 1
        print(f'{pid}: detail {len(recs)}', flush=True)
    return {'detail_pairs': n, 'detail_seconds': time.time() - t0}


def stage_spotcheck(rt, args, out: Path) -> dict:
    """Answer-level agreement between the manual greedy decoder and qwen_runtime.generate."""
    from analysis_tools.common import qwen_runtime as qr
    rows = [r for r in read_jsonl(out / f'screen_{args.tag}.jsonl') if r['kind'] == 'gen'][:args.spotcheck]
    agree, details = 0, []
    for r in rows:
        pair = pair_from_row(r, args)
        img = render_pair(pair)[0]
        ref = qr.generate(rt, qr.build(rt, [img], pair.prompt), args.max_new_tokens)['text']
        same = ref.strip() == r['clean_text'].strip()
        agree += same
        details.append({'pair_id': r['pair_id'], 'same': same, 'generate': ref, 'greedy': r['clean_text']})
    (out / f'spotcheck_{args.tag}.json').write_text(json.dumps(details, indent=1))
    return {'spotcheck_agree': agree, 'spotcheck_n': len(rows)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('screen', 'scan', 'spotcheck', 'detail', 'analyze'))
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--conditions', nargs='+', default=list(ALL_CONDITIONS))
    p.add_argument('--pairs-per-condition', type=int, default=24, help='pairs per 2.5c generation condition')
    p.add_argument('--yesno-pairs', type=int, default=40, help='pairs per 2.5b revised-T3 condition')
    p.add_argument('--decomposition-dir', default='')
    p.add_argument('--fail-grids', type=int, nargs='+', default=[3, 6, 10])
    p.add_argument('--fail-max-per-grid', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--sites', type=int, nargs='+', default=list(SITES))
    p.add_argument('--yesno-sites', type=int, nargs='+', default=list(SITES))
    p.add_argument('--detail-sites', type=int, nargs='+', default=[12, 16, 19, 20, 22, 24, 28, 31])
    p.add_argument('--sanity-sites', type=int, nargs='+', default=[1, 16])
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reverse', action='store_true', help='scan the screened pairs in reverse order')
    p.add_argument('--tag', default='main', help='suffix of the screen/records/baselines files of this process')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'analyze':
        from analysis_tools.representation import patching_extension_analysis as an
        an.analyze(out, args)
        return
    from analysis_tools.common import qwen_runtime as qr
    rt = qr.load(args.model, device='cuda:0', attn='sdpa')
    t0 = time.strftime('%Y-%m-%dT%H:%M:%S')
    fn = {'screen': stage_screen, 'scan': stage_scan, 'spotcheck': stage_spotcheck, 'detail': stage_detail}[args.stage]
    info = fn(rt, args, out)
    meta = {'stage': args.stage, 'started': t0, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'args': vars(args), 'code_sha256': code_hashes(), 'runtime': qr.describe(rt),
            'generator': stimuli.GENERATOR_VERSION, **info}
    with open(out / f'metadata_{args.stage}_{args.tag}.jsonl', 'a') as f:
        f.write(json.dumps(meta) + '\n')


if __name__ == '__main__':
    main()
