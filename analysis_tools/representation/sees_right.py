"""Experiment 2.5e: verification of "sees right but answers wrong" on Qwen3.5-9B.

Design and results: docs/analysis/representation/results_2026-10-16_sees_right.md.

Stages (--stage), all resumable (finished record ids are skipped):
  cases      build the stimulus set on CPU: single red disc (point); T6 red-target grids 6/10/20 in the
             row/column and point formats; T6 20x20 with a think-then-answer prompt; T5 find-the-different
             images of experiment 1.3 (3x3/6x6/10x10, row/column and point; every 2.5b content failure plus
             sampled successes); RefCOCO point questions on the experiment-3.4 test split.
  gen        greedy answers with the manual equal-length batched decoder of experiment 3.5 (cr.greedy).
  spotcheck  answer-level agreement of the manual decoder with qwen_runtime.generate.
  readout    one teacher-forced forward per answer: softmax rows of all 8 full-attention layers x 16 heads
             over the visual tokens at the coordinate-digit queries, layer-20 effective contributions, and
             residual states at the digit queries for the probes.
  causal     (a) output patching at the answer positions from a run whose target is moved by a fixed
             offset (L20 H11, all layer-20 heads, all 128 full-attention heads, all token mixers);
             (b) KV-cache edits at the answer positions: value copy V[T] <- V[T+o], key swap K[T] <-> K[T+o],
             key+value swap (a permutation, hence a no-op sanity check).

Query convention (fixed for the whole experiment): for coordinate value k the query 'pred' is the
input position whose next-token prediction is the first digit of value k (the token before the
digit, e.g. ' [' or '": '); the query 'tok' is the position of that first digit itself (it predicts
the second digit or the closing punctuation).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.common.stimuli import DEVELOPMENT_TEMPLATES, Change, StimulusSpec, generate, uniform_targets
from analysis_tools.position import coordinate_readout as cr

FULL_LAYERS = cr.FULL_LAYERS
RESID_LAYERS = (8, 12, 16, 20, 24, 28)
READER = (20, 11)                         # causal reader head of experiment 3.5 C4
T6_QUERY = 'One figure is drawn in red.'
T5_QUERY = 'One figure differs from all the others.'
LOCATE_SUFFIX = {
    'rowcol': ' Give its position. Return only {"row": r, "column": c}, counting rows from top to bottom '
              'and columns from left to right, starting at 1.',
    'point': ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].'
             ' All coordinates use 0-1000 normalization; x increases rightward and y increases downward.',
}
JSON_ONLY = ' Output only the JSON, with no explanation.'
# Think-then-answer wording of experiment 1.3 (analysis_tools/behavior/decomposition.py, EXPLICIT_LOCATE).
EXPLICIT_LOCATE = (' First describe step by step what you observe and how you find this figure, including how many '
                   'rows and columns of figures there are. Then give the final answer on the last line as '
                   '{"row": r, "column": c}, counting rows from top to bottom and columns from left to right, '
                   'starting at 1.')
REFCOCO_QUERY = 'Locate the object described as: {expr}.'
ROWCOL_RE = re.compile(r'"row":\s*(\d+),\s*"column":\s*(\d+)')


# ----------------------------------------------------------------------------- io

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def done_ids(path: Path) -> set[str]:
    return {r['id'] for r in read_jsonl(path)}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- stimuli

def grid_truth(meta: dict, rc: tuple[int, int]) -> dict:
    """Ground truth of a grid stimulus: target box (tight, pixels), center, cell geometry."""
    inst = next(i for i in meta['instances'] if (i['row'], i['col']) == tuple(rc))
    spec = meta['spec']
    return {'canvas': meta['canvas'], 'box': inst['tight_box'], 'center': inst['center'],
            'cell_box': inst['cell_box'], 'rows': spec['rows'], 'cols': spec['cols'], 'cell': spec['cell'],
            'grid_origin': meta['grid_origin'], 'target_rc': list(rc)}


def spec_dict(spec: StimulusSpec) -> dict:
    from dataclasses import asdict
    return {**asdict(spec), 'change': asdict(spec.change)}


def spec_from_dict(d: dict) -> StimulusSpec:
    d = dict(d)
    d['change'] = Change(**d['change'])
    for key in ('target', 'marker'):
        if d.get(key) is not None:
            d[key] = tuple(d[key])
    return StimulusSpec(**d)


def locate_prompt(query: str, fmt: str, style: str = 'direct') -> str:
    if style == 'think':
        if fmt != 'rowcol':
            raise ValueError('The think-then-answer prompt exists for the row/column format only.')
        return query + EXPLICIT_LOCATE
    return query + LOCATE_SUFFIX[fmt] + JSON_ONLY


def disc_cases(n: int, seed: int, out: Path) -> list[dict]:
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        d = (32, 64)[i % 2]
        lo, hi = d / 2 + 8, cr.CANVAS - d / 2 - 8
        center = (float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi)))
        image, truth = cr.disc_image(center, d, cr.CANVAS)
        path = out / 'images' / f'disc_{i:03d}.png'
        image.save(path)
        cases.append({'id': f'disc_point_{i:03d}', 'cond': 'disc_point', 'set': 'disc', 'format': 'point',
                      'style': 'direct', 'grid': None, 'image_key': f'disc_{i:03d}', 'image_path': str(path),
                      'image_sha256': sha256(path), 'prompt': cr.DISC_QUERY + cr.SUFFIX['point'],
                      'truth': truth, 'center_input': list(center)})
    return cases


def t6_cases(grids: tuple[int, ...], n: int, n_think: int, seed: int, out: Path) -> list[dict]:
    """Fresh stimuli-v1 T6 grids (red target, development templates, uniform targets)."""
    cases = []
    for g in grids:
        targets = uniform_targets(g, g, n, seed + g)
        for i, rc in enumerate(targets):
            spec = StimulusSpec(template=DEVELOPMENT_TEMPLATES[i % len(DEVELOPMENT_TEMPLATES)], rows=g, cols=g,
                                change=Change('color', 4), target=tuple(rc), seed=seed * 1000 + g * 100 + i)
            image, meta = generate(spec)
            key = f't6g{g:02d}_{i:03d}'
            path = out / 'images' / f'{key}.png'
            image.save(path)
            base = {'set': 't6', 'grid': g, 'image_key': key, 'image_path': str(path), 'image_sha256': sha256(path),
                    'truth': grid_truth(meta, rc), 'spec': spec_dict(spec)}
            styles = [('rowcol', 'direct'), ('point', 'direct')] + ([('rowcol', 'think')] if g == 20 and i < n_think else [])
            for fmt, style in styles:
                cond = f't6g{g:02d}_{fmt}' + ('_think' if style == 'think' else '')
                cases.append({**base, 'id': f'{cond}_{i:03d}', 'cond': cond, 'format': fmt, 'style': style,
                              'prompt': locate_prompt(T6_QUERY, fmt, style)})
    return cases


def t5_cases(decomposition_dir: Path, grids: tuple[int, ...], n_correct: int, seed: int, out: Path) -> list[dict]:
    """Experiment-1.3 T5 images whose T2 pair was answered correctly.

    Wrong: every primary-wording row/column answer that parses and differs from the truth (the 2.5b
    content failures). Correct: n_correct images per grid sampled from those whose T5 row/column answer
    was correct. Each image is asked in the row/column and in the point format.
    """
    d = Path(decomposition_dir)
    flags = json.loads((d / 'analysis' / 'per_image_records.json').read_text())
    images = {r['image_id']: r for r in read_jsonl(d / 'images.jsonl')}
    responses = {r['id']: r['response'] for r in read_jsonl(d / 'results_main.jsonl')}
    rng = np.random.default_rng(seed)
    cases = []
    for g in grids:
        wrong, right = [], []
        for c in read_jsonl(d / 'cases.jsonl'):
            if not (c['task'] == 'T5' and c['format'] == 'row_column' and c['wording'] == 'primary'
                    and c['stage'] == 'main' and c['grid'] == g):
                continue
            f = flags[c['record_id']]
            if not f.get('T2'):
                continue
            if f.get('T5 row/col'):
                right.append(c)
                continue
            m = ROWCOL_RE.search(responses[c['id']])
            if m and (int(m[1]) - 1, int(m[2]) - 1) != tuple(c['truth']['target_rc']):
                wrong.append(c)
        pick = [right[k] for k in sorted(rng.choice(len(right), min(n_correct, len(right)), replace=False))]
        for c, origin in [(c, 'wrong_13') for c in wrong] + [(c, 'correct_13') for c in pick]:
            img = images[c['image_id']]
            spec = spec_from_dict(img['spec'])
            image, meta = generate(spec)
            if meta['sha256'] != img['sha256']:
                raise ValueError(f"Regenerated image differs from experiment 1.3: {c['record_id']}")
            key = f"t5g{g:02d}_{c['record_id']}"
            path = out / 'images' / f'{key}.png'
            image.save(path)
            base = {'set': 't5', 'grid': g, 'image_key': key, 'image_path': str(path), 'image_sha256': sha256(path),
                    'truth': grid_truth(meta, tuple(c['truth']['target_rc'])), 'spec': spec_dict(spec),
                    'kind': c['kind'], 'level': c['level'], 'origin_13': origin, 'record_13': c['record_id']}
            for fmt in ('rowcol', 'point'):
                cond = f't5g{g:02d}_{fmt}'
                cases.append({**base, 'id': f"{cond}_{c['record_id']}", 'cond': cond, 'format': fmt, 'style': 'direct',
                              'prompt': locate_prompt(T5_QUERY, fmt)})
    return cases


def refcoco_cases(source_cases: Path) -> list[dict]:
    """Point questions of experiment 3.5 on the experiment-3.4 RefCOCO test split (512-pixel copies)."""
    out = []
    for c in json.loads(Path(source_cases).read_text()):
        if c['format'] != 'point':
            continue
        out.append({'id': f"rc_point_{c['source_case']:04d}", 'cond': 'rc_point', 'set': 'refcoco', 'format': 'point',
                    'style': 'direct', 'grid': None, 'image_key': c['image_key'], 'image_path': c['image_path'],
                    'image_sha256': c['image_sha256'], 'prompt': c['prompt'], 'expression': c['expression'],
                    'truth': c['truth']})
    return out


def load_image(case: dict) -> Image.Image:
    p = Path(case['image_path'])
    if sha256(p) != case['image_sha256']:
        raise ValueError(f'Image hash mismatch for {p}')
    return Image.open(p).convert('RGB')


# ----------------------------------------------------------------------------- scoring

def last_rowcol(text: str) -> tuple[int, int] | None:
    """Row/column answer: strict JSON for direct answers, last {"row": r, "column": c} for reasoning."""
    m = list(ROWCOL_RE.finditer(text))
    return (int(m[-1][1]), int(m[-1][2])) if m else None


def cell_center_units(truth: dict, rc: tuple[int, int]) -> list[float]:
    """Center of a (0-based) grid cell in 0-1000 units; cells outside the grid extrapolate linearly."""
    ox, oy = truth['grid_origin']
    s = truth['cell']
    W, H = truth['canvas']
    return [1000 * (ox + (rc[1] + 0.5) * s) / W, 1000 * (oy + (rc[0] + 0.5) * s) / H]


def point_cell(truth: dict, center_units: np.ndarray) -> tuple[int, int]:
    W, H = truth['canvas']
    px = np.asarray(center_units) * np.asarray([W, H]) / 1000
    ox, oy = truth['grid_origin']
    return int((px[1] - oy) // truth['cell']), int((px[0] - ox) // truth['cell'])


def score(case: dict, text: str) -> dict:
    """Unified score: validity, correctness, output position (0-1000) and, for grids, the output cell.

    Correct: row/column exact match; point inside the target cell (grids), inside the disc (disc),
    inside the RefCOCO box (strict 0-1000 convention).
    """
    t = case['truth']
    W, H = t['canvas']
    true_units = [1000 * t['center'][0] / W, 1000 * t['center'][1] / H]
    out = {'true_units': true_units}
    if case['format'] == 'rowcol':
        rc = last_rowcol(text) if case['style'] == 'think' else cr.parse_rowcol(text)
        if rc is None:
            return {**out, 'valid': False, 'correct': False}
        pred = (rc[0] - 1, rc[1] - 1)
        tr, tc = t['target_rc']
        return {**out, 'valid': True, 'pred_rc': list(pred), 'correct': pred == (tr, tc), 'd_row': pred[0] - tr,
                'd_col': pred[1] - tc, 'out_units': cell_center_units(t, pred)}
    coords = cr.parse_coords(text, 'point')
    if coords is None:
        return {**out, 'valid': False, 'correct': False}
    c = np.asarray(coords, float)
    out.update({'valid': True, 'out_units': c.tolist(), 'error_units': float(np.linalg.norm(c - true_units))})
    if case['set'] in ('t6', 't5'):
        pred = point_cell(t, c)
        tr, tc = t['target_rc']
        out.update({'pred_rc': list(pred), 'correct': pred == (tr, tc), 'd_row': pred[0] - tr, 'd_col': pred[1] - tc})
    elif case['set'] == 'disc':
        px = c * np.asarray([W, H]) / 1000
        out['correct'] = bool(np.linalg.norm(px - np.asarray(t['center'])) <= t['diameter'] / 2)
    else:
        b = t['box']
        px = c * np.asarray([W, H]) / 1000
        out['correct'] = bool(b[0] <= px[0] <= b[2] and b[1] <= px[1] <= b[3])
    return out


def coord_tokens(rt, ids: list[int], fmt: str, last: bool) -> list[list[int]]:
    """Generated-token indices of the digits of each coordinate value ([x, y] or [row, column]).

    Direct answers use the first match (identical to cr.coordinate_tokens); think-then-answer
    responses use the last {"row": r, "column": c} so that numbers inside the reasoning are skipped.
    """
    if not last:
        return cr.coordinate_tokens(rt, ids, fmt)
    text = rt.tokenizer.decode(ids, skip_special_tokens=False)
    ends = [len(rt.tokenizer.decode(ids[:i + 1], skip_special_tokens=False)) for i in range(len(ids))]
    starts = [0] + ends[:-1]
    ms = list(ROWCOL_RE.finditer(text))
    if not ms:
        return []
    spans = [ms[-1].span(1), ms[-1].span(2)]
    out = []
    for a, b in spans:
        toks = [i for i in range(len(ids)) if starts[i] >= a and ends[i] <= b]
        if not toks or ''.join(rt.tokenizer.decode([ids[i]]) for i in toks) != text[a:b]:
            return []
        out.append(toks)
    return out


# ----------------------------------------------------------------------------- stages

def stage_cases(out: Path, args) -> None:
    (out / 'images').mkdir(parents=True, exist_ok=True)
    cases = (disc_cases(args.n_disc, args.seed, out)
             + t6_cases(tuple(args.t6_grids), args.n_t6, args.n_think, args.seed, out)
             + t5_cases(Path(args.decomposition_dir), tuple(args.t5_grids), args.n_t5_correct, args.seed, out)
             + refcoco_cases(Path(args.refcoco_cases)))
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case ids')
    (out / 'cases.json').write_text(json.dumps(cases, indent=1))
    counts = {}
    for c in cases:
        counts[c['cond']] = counts.get(c['cond'], 0) + 1
    print(json.dumps(counts, indent=1))


def load_cases(out: Path) -> dict:
    return {c['id']: c for c in json.loads((out / 'cases.json').read_text())}


def stage_gen(rt, out: Path, args) -> None:
    path = out / 'gen.jsonl'
    finished = done_ids(path)
    cases = [c for c in load_cases(out).values() if c['id'] not in finished and c['cond'] in args.conditions]
    by_cond: dict[str, list] = {}
    for c in cases:
        by_cond.setdefault(c['cond'], []).append(c)
    for cond, group in by_cond.items():
        style = group[0]['style']
        max_new = args.max_new_think if style == 'think' else args.max_new_tokens
        for i in range(0, len(group), args.prep_chunk):
            chunk = group[i:i + args.prep_chunk]
            preps = [cr.prepare(rt, load_image(c), c['prompt']) for c in chunk]
            by_len: dict[int, list] = {}
            for c, p in zip(chunk, preps):
                by_len.setdefault(len(p.ids), []).append((c, p))
            for L, items in by_len.items():
                bs = args.batch_size if L < 800 else max(1, args.batch_size // 2)
                for j in range(0, len(items), bs):
                    part = items[j:j + bs]
                    t0 = time.time()
                    res = cr.run_conditions(rt, [cr.embeds(rt, p) for _, p in part], [p.pos for _, p in part],
                                            max_new, batch_size=len(part))
                    rows = [{'id': c['id'], 'cond': cond, 'text': r['text'], 'ids': r['ids'],
                             'finish_reason': r['finish_reason'], 'prompt_len': L, 'batch_size': len(part),
                             'score': score(c, r['text'])} for (c, _), r in zip(part, res)]
                    append_jsonl(path, rows)
                    print(f'gen {cond} L={L} +{len(part)} {time.time() - t0:.1f}s', flush=True)
            del preps
            torch.cuda.empty_cache()


def stage_spotcheck(rt, out: Path, args) -> None:
    path = out / 'spotcheck.jsonl'
    finished = done_ids(path)
    cases = load_cases(out)
    gen = [g for g in read_jsonl(out / 'gen.jsonl') if cases[g['id']]['style'] == 'direct']
    rng = np.random.default_rng(args.seed)
    for k in sorted(rng.choice(len(gen), min(args.spotcheck, len(gen)), replace=False)):
        g = gen[k]
        if g['id'] in finished:
            continue
        c = cases[g['id']]
        ref = qr.generate(rt, qr.build(rt, [load_image(c)], c['prompt']), args.max_new_tokens)
        s_ref, s_man = score(c, ref['text']), g['score']
        key = 'pred_rc' if c['format'] == 'rowcol' else 'out_units'
        append_jsonl(path, [{'id': g['id'], 'reference': ref['text'], 'manual': g['text'],
                             'text_equal': ref['text'].strip() == g['text'].strip(),
                             'answer_equal': s_ref.get(key) == s_man.get(key),
                             'correct_equal': s_ref.get('correct') == s_man.get('correct')}])


def stage_readout(rt, out: Path, args) -> None:
    """Attention rows and residual states at the coordinate-digit queries of every valid answer."""
    root = out / 'readout'
    (root / 'maps').mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    cases = load_cases(out)
    for g in read_jsonl(out / 'gen.jsonl'):
        c = cases[g['id']]
        if g['id'] in finished or c['cond'] not in args.conditions:
            continue
        t0 = time.time()
        if not g['score'].get('valid'):
            append_jsonl(path, [{'id': g['id'], 'status': 'invalid_answer'}])
            continue
        prep = cr.prepare(rt, load_image(c), c['prompt'])
        coord = coord_tokens(rt, g['ids'], c['format'], last=c['style'] == 'think')
        if len(coord) != 2:
            append_jsonl(path, [{'id': g['id'], 'status': 'no_coordinate_tokens'}])
            continue
        x, pos = cr.teacher_forced(rt, prep, g['ids'][:-1])
        L = len(prep.ids)
        queries = [L + coord[0][0] - 1, L + coord[1][0] - 1, L + coord[0][0], L + coord[1][0], L - 1]
        names = ['pred0', 'pred1', 'tok0', 'tok1', 'last']
        s, e = prep.span['start'], prep.span['end']
        qpos = torch.tensor([queries[0], queries[1], queries[4]], device=x.device)
        with qr.capture(rt, list(RESID_LAYERS), qpos) as store:
            ro = cr.attention_readout(rt, x, pos, queries, (s, e))
        a = np.stack([ro.attn[l][:, :, s:e].numpy() for l in FULL_LAYERS])          # (8, 16, 5, Nv)
        vis_mass = a.sum(-1)
        resid = np.stack([store[l].numpy() for l in RESID_LAYERS])                   # (6, 3, D)
        np.savez_compressed(root / 'maps' / f"{g['id']}.npz", attn=a[:, :, :4].astype(np.float16),
                            contrib20=ro.contrib[20][:, :2].numpy().astype(np.float16),
                            resid=resid.astype(np.float16))
        append_jsonl(path, [{'id': g['id'], 'status': 'ok', 'grid': list(prep.grid), 'span': [s, e],
                             'query_names': names, 'query_positions': queries, 'coord_tokens': coord,
                             'target_tokens': cr.tokens_overlapping(c['truth']['box'], prep.grid),
                             'vis_mass_last': vis_mass[:, :, 4].tolist(), 'check': ro.check,
                             'seconds': round(time.time() - t0, 2)}])
        print(f"readout {g['id']} {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------- causal tools

def kv_layers(rt) -> dict[int, int]:
    """1-based full-attention layer -> self_attn.layer_idx used by the cache."""
    return {l: rt.layers[l - 1].self_attn.layer_idx for l in FULL_LAYERS}


def cache_kv(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(keys, values) tensors of shape (B, kv_heads, L, head_dim) of one full-attention cache layer."""
    if hasattr(cache, 'layers'):
        lay = cache.layers[layer_idx]
        return lay.keys, lay.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


class AttentionProbe:
    """Records, at every decoding step, the softmax row of one full-attention layer over the image span.

    Used during KV-edited decoding to verify where the edited model attends; the keys come from the
    (edited) cache plus the current token. Stores (B, heads, Nv) float32 per step.
    """

    def __init__(self, rt, layer: int, span: tuple[int, int]):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
        self.steps, self.cache = [], None
        mod = rt.layers[layer - 1].self_attn
        s, e = span

        def pre(module, args, kwargs):
            h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            if h.shape[1] != 1 or self.cache is None:
                return
            cos, sin = kwargs['position_embeddings']
            hd = mod.head_dim
            q, _ = torch.chunk(mod.q_proj(h).view(h.shape[0], 1, -1, hd * 2), 2, dim=-1)
            q = mod.q_norm(q).transpose(1, 2)
            k_cur = mod.k_norm(mod.k_proj(h).view(h.shape[0], 1, -1, hd)).transpose(1, 2)
            q, k_cur = apply_rotary_pos_emb(q, k_cur, cos, sin)
            keys, _ = cache_kv(self.cache, mod.layer_idx)
            k = torch.cat([keys, k_cur], 2)
            k = repeat_kv(k, mod.num_key_value_groups)
            sc = (q.float() @ k.float().transpose(2, 3)) * mod.scaling               # (B, heads, 1, Lk)
            self.steps.append(sc.softmax(-1)[:, :, 0, s:e].cpu())
        self.handle = mod.register_forward_pre_hook(pre, with_kwargs=True)

    def remove(self):
        self.handle.remove()


def _lin_states(layer) -> list[torch.Tensor]:
    """State tensors (batch-first) of one Gated DeltaNet cache layer."""
    out = []
    for name in ('conv_states', 'recurrent_states'):
        box = getattr(layer, name, None)
        vals = box.values() if isinstance(box, dict) else (box or [])
        out += [t for t in vals if isinstance(t, torch.Tensor)]
    if not out:
        raise ValueError(f'No recurrent state tensors found in cache layer {type(layer).__name__}')
    return out


@torch.inference_mode()
def greedy_kv_edit(rt, x: torch.Tensor, pos: torch.Tensor, edits: list[list[tuple]], max_new_tokens: int,
                   probe: AttentionProbe | None = None) -> list[dict]:
    """Greedy decoding in which the cache of the prompt (all but its last token) is edited per row.

    edits[b]: list of (layer_idx, kind, src, dst, src_row). kind 'v': values[b, :, dst] <- values[src_row, :, src];
    'k': the same for keys; 'lin': every Gated DeltaNet state of layer_idx is copied from src_row (src, dst
    unused). src_row None means the row itself. src/dst are equal-length LongTensors of absolute sequence
    positions. All edits read the unedited cache. The last prompt token and every generated token are the
    answer positions that see the edited cache.
    """
    B, L, _ = x.shape
    eos = cr.eos_ids(rt)
    out = rt.model(inputs_embeds=x[:, :-1], position_ids=pos[:, :, :-1], use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    originals = {}
    for b, row in enumerate(edits):
        for li, kind, src, dst, src_row in row:
            sr_ = b if src_row is None else src_row
            if kind == 'lin':
                tensors = _lin_states(cache.layers[li])
                if ('lin', li) not in originals:
                    originals[('lin', li)] = [t.clone() for t in tensors]
                for t, o in zip(tensors, originals[('lin', li)]):
                    t[b] = o[sr_]
                continue
            keys, values = cache_kv(cache, li)
            target = keys if kind == 'k' else values
            if (li, kind) not in originals:
                originals[(li, kind)] = target.clone()
            target[b, :, dst.to(target.device)] = originals[(li, kind)][sr_, :, src.to(target.device)]
    if probe is not None:
        probe.cache = cache
    embed = rt.model.model.get_input_embeddings()
    step_out = rt.model(inputs_embeds=x[:, -1:], position_ids=pos[:, :, -1:], past_key_values=cache, use_cache=True)
    cache = step_out.past_key_values
    logits = step_out.logits[:, -1].float()
    nxt = pos[1:, :, -1] + 1
    tokens = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool)
    for step in range(max_new_tokens):
        tok = logits.argmax(-1).cpu()
        for b in range(B):
            if not done[b]:
                tokens[b].append(int(tok[b]))
        done |= torch.tensor([int(t) in eos for t in tok.tolist()])
        if bool(done.all()) or step == max_new_tokens - 1:
            break
        p = torch.cat([torch.full((1, B), L + step, device=x.device, dtype=pos.dtype), nxt + step], 0)[:, :, None]
        o = rt.model(inputs_embeds=embed(tok.to(x.device)[:, None]), position_ids=p, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        logits = o.logits[:, -1].float()
    return [{'text': rt.tokenizer.decode(t, skip_special_tokens=True), 'ids': t} for t in tokens]


def shift_tokens(tokens: list[int], grid: tuple[int, int], dr: int, dc: int) -> list[int] | None:
    H, W = grid
    out = []
    for t in tokens:
        r, c = t // W + dr, t % W + dc
        if not (0 <= r < H and 0 <= c < W):
            return None
        out.append(r * W + c)
    return out


def moves(case: dict, grid: tuple[int, int], target: list[int], k_cells: int, k_tokens: int) -> list[dict]:
    """One horizontal and one vertical displacement of the target, positive direction when it fits.

    Grids: the target instance moves by k_cells cells (run B) and the token offset is k_cells * cell / 32.
    Disc: the disc moves by k_tokens merged tokens. RefCOCO: token offset only (no run B).
    """
    out = []
    for axis in ('h', 'v'):
        for sign in (1, -1):
            if case['set'] in ('t6', 't5'):
                t = case['truth']
                n = t['cols'] if axis == 'h' else t['rows']
                cur = t['target_rc'][1] if axis == 'h' else t['target_rc'][0]
                if not 0 <= cur + sign * k_cells < n:
                    continue
                off = sign * k_cells * t['cell'] // cr.TOKEN_PIXELS
                cells = sign * k_cells
            else:
                off, cells = sign * k_tokens, None
            dr, dc = (0, off) if axis == 'h' else (off, 0)
            dst = shift_tokens(target, grid, dr, dc)
            if dst is None:
                continue
            if case['set'] == 'disc':
                c = case['truth']['center']
                moved = (c[0] + 32 * dc, c[1] + 32 * dr)
                r = case['truth']['diameter'] / 2
                if not (r <= moved[0] <= cr.CANVAS - r and r <= moved[1] <= cr.CANVAS - r):
                    continue
            out.append({'axis': axis, 'sign': sign, 'dr': dr, 'dc': dc, 'cells': cells, 'shifted': dst,
                        'overlap': bool(set(dst) & set(target))})
            break
    return out


def run_b_image(case: dict, mv: dict) -> Image.Image | None:
    if case['set'] in ('t6', 't5'):
        spec = spec_from_dict(case['spec'])
        r, c = case['truth']['target_rc']
        new = (r + (mv['cells'] if mv['axis'] == 'v' else 0), c + (mv['cells'] if mv['axis'] == 'h' else 0))
        from dataclasses import replace
        image, _ = generate(replace(spec, target=new))
        return image
    if case['set'] == 'disc':
        cx, cy = case['truth']['center']
        image, _ = cr.disc_image((cx + 32 * mv['dc'], cy + 32 * mv['dr']), case['truth']['diameter'], cr.CANVAS)
        return image
    return None


def patch_sets(rt) -> dict[str, list[tuple[int, int]]]:
    lin = [i + 1 for i, b in enumerate(rt.layers) if b.block_type == 'linear_attention']
    full = [(l, h) for l in FULL_LAYERS for h in range(16)]
    return {'none': [], 'L20H11': [READER], 'L20': [(20, h) for h in range(16)], 'full_all': full,
            'mixers_all': full + [(l, -1) for l in lin]}


def kv_conditions(rt, s: int, e: int, L: int, target: list[int], shifted: list[int], overlap: bool,
                  b_row: int | None) -> dict[str, list[tuple]]:
    """Cache edits at the answer positions.

    Target-local edits (row A only): value copy V[T] <- V[T+o] at layer 20, layers 16/20/24 or all 8 full
    layers; key swap K[T] <-> K[T+o] (attention moves to T+o, values stay); key+value swap (a permutation
    of key slots, a no-op for softmax attention). Source decomposition (needs run B in row b_row): image
    values, image keys+values, post-image text keys+values (positions e..L-2), every Gated DeltaNet state,
    text KV plus DeltaNet states, and the whole cache (reproduces run B).
    """
    idx = kv_layers(rt)
    lin_idx = [i for i, b in enumerate(rt.layers) if b.block_type == 'linear_attention']
    T = torch.tensor([s + t for t in target])
    T2 = torch.tensor([s + t for t in shifted])
    def v(layers):
        return [(idx[l], 'v', T2, T, None) for l in layers]
    conds = {'kv_none': [], 'v_L20': v([20]), 'v_L16_20_24': v([16, 20, 24]), 'v_full': v(FULL_LAYERS)}
    if not overlap:
        swap_k = [(idx[l], 'k', T2, T, None) for l in FULL_LAYERS] + [(idx[l], 'k', T, T2, None) for l in FULL_LAYERS]
        swap_v = [(idx[l], 'v', T2, T, None) for l in FULL_LAYERS] + [(idx[l], 'v', T, T2, None) for l in FULL_LAYERS]
        conds['k_swap_full'] = swap_k
        conds['kv_swap_full'] = swap_k + swap_v
    if b_row is not None:
        img = torch.arange(s, e)
        text = torch.arange(e, L - 1)
        allp = torch.arange(0, L - 1)
        def kv(pos_, kinds):
            return [(idx[l], k, pos_, pos_, b_row) for l in FULL_LAYERS for k in kinds]
        lin = [(i, 'lin', None, None, b_row) for i in lin_idx]
        conds.update({'B_img_v': kv(img, 'v'), 'B_img_kv': kv(img, 'kv'), 'B_text_kv': kv(text, 'kv'),
                      'B_lin': lin, 'B_text_kv_lin': kv(text, 'kv') + lin, 'B_all': kv(allp, 'kv') + lin})
    return conds


def stage_causal(rt, out: Path, args) -> None:
    from analysis_tools.position.coordinate_readout_supp import run_units
    path = out / 'causal' / 'records.jsonl'
    finished = done_ids(path)
    cases = load_cases(out)
    gen = {g['id']: g for g in read_jsonl(out / 'gen.jsonl')}
    rng = np.random.default_rng(args.seed)
    sets = patch_sets(rt)
    for cond in args.causal_conditions:
        ids = [i for i, c in cases.items() if c['cond'] == cond and gen.get(i, {}).get('score', {}).get('valid')]
        wrong = [i for i in ids if not gen[i]['score']['correct']]
        right = [i for i in ids if gen[i]['score']['correct']]
        n_total = args.n_causal_long if cases[ids[0]]['grid'] == 20 else args.n_causal
        nw = min(len(wrong), n_total // 2)
        pick = ([wrong[k] for k in rng.choice(len(wrong), nw, replace=False)] if nw else []) + \
               [right[k] for k in rng.choice(len(right), min(len(right), n_total - nw), replace=False)]
        for cid in sorted(pick):
            if cid in finished:
                continue
            t0 = time.time()
            c = cases[cid]
            prep_a = cr.prepare(rt, load_image(c), c['prompt'])
            xa = cr.embeds(rt, prep_a)
            target = cr.tokens_overlapping(c['truth']['box'], prep_a.grid)
            s, e = prep_a.span['start'], prep_a.span['end']
            rows = []
            for mv in moves(c, prep_a.grid, target, args.k_cells, args.k_tokens):
                img_b = run_b_image(c, mv)
                tag = f"{mv['axis']}{mv['sign']:+d}"
                xb = pos_b = None
                if img_b is not None:
                    prep_b = cr.prepare(rt, img_b, c['prompt'])
                    if len(prep_b.ids) != len(prep_a.ids):
                        raise ValueError(f'Run B length differs for {cid}')
                    xb = cr.embeds(rt, prep_b)
                    units = [{'x': [xb], 'pos': [prep_b.pos], 'follow': {}, 'ops': []}]
                    names = ['B_free']
                    for name, heads in sets.items():
                        units.append({'x': [xb, xa], 'pos': [prep_b.pos, prep_a.pos], 'follow': {0: 1},
                                      'ops': [(1, l, h, 0) for l, h in heads]})
                        names.append(f'patch_{name}')
                    rows_cap = args.max_rows if len(prep_a.ids) < 1000 else args.max_rows_long
                    res = run_units(rt, units, rows_cap, len(prep_a.ids) - 1, max_new_tokens=args.max_new_tokens)
                    for name, r in zip(names, res):
                        r = r[-1]
                        rows.append({'move': tag, **{k: v for k, v in mv.items() if k != 'shifted'},
                                     'condition': name, 'text': r['text'], 'score': score(c, r['text'])})
                    pos_b = prep_b.pos
                L = len(prep_a.ids)
                kv = kv_conditions(rt, s, e, L, target, mv['shifted'], mv['overlap'], None if xb is None else 0)
                names = list(kv)
                # Long prompts (20x20 grids) are decoded in chunks of conditions to bound memory; every
                # chunk repeats run B in row 0 as the edit source.
                chunk = len(names) if L < 1000 else args.kv_chunk
                res, steps = [], []
                for i in range(0, len(names), chunk):
                    part = names[i:i + chunk]
                    xs, ps, edits = [xa] * len(part), [prep_a.pos] * len(part), [kv[n] for n in part]
                    if xb is not None:
                        xs, ps, edits = [xb] + xs, [pos_b] + ps, [[]] + edits
                    probe = AttentionProbe(rt, 20, (s, e))
                    try:
                        r_ = greedy_kv_edit(rt, torch.cat(xs, 0), torch.cat(ps, 1), edits, args.max_new_tokens, probe)
                    finally:
                        probe.remove()
                    off = 0 if xb is None else 1
                    res += r_[off:]
                    steps.append([st[off:] for st in probe.steps])
                n_steps = min(len(st) for st in steps)
                probe.steps = [torch.cat([st[j] for st in steps], 0) for j in range(n_steps)]
                att = torch.stack(probe.steps, 0) if probe.steps else None              # (steps, B, heads, Nv)
                for b, (name, r) in enumerate(zip(names, res)):
                    sc = score(c, r['text'])
                    att_info = {}
                    coord = cr.coordinate_tokens(rt, r['ids'], c['format']) if sc.get('valid') else []
                    if att is not None and len(coord) == 2:
                        tmask = torch.zeros(e - s, dtype=torch.bool)
                        tmask[target] = True
                        smask = torch.zeros(e - s, dtype=torch.bool)
                        smask[mv['shifted']] = True
                        for k in range(2):
                            st = coord[k][0]                     # step 0 = last prompt token, predicts generated token 0
                            if 0 <= st < att.shape[0]:
                                a = att[st, b]                   # (heads, Nv)
                                p = a / a.sum(-1, keepdim=True)
                                att_info[f'v{k}'] = {'T_H11': float(p[11, tmask].sum()), 'S_H11': float(p[11, smask].sum()),
                                                     'T_mean': float(p[:, tmask].sum(-1).mean()),
                                                     'S_mean': float(p[:, smask].sum(-1).mean())}
                    rows.append({'move': tag, **{k: v for k, v in mv.items() if k != 'shifted'}, 'condition': name,
                                 'text': r['text'], 'score': sc, 'attn20': att_info})
            append_jsonl(path, [{'id': cid, 'cond': cond, 'baseline_correct': bool(gen[cid]['score']['correct']),
                                 'baseline_score': gen[cid]['score'], 'target_tokens': target, 'rows': rows,
                                 'seconds': round(time.time() - t0, 1)}])
            print(f'causal {cid} {len(rows)} rows {time.time() - t0:.1f}s', flush=True)
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- main

STAGES = {'gen': stage_gen, 'spotcheck': stage_spotcheck, 'readout': stage_readout, 'causal': stage_causal}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', required=True, choices=['cases', *STAGES])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--conditions', nargs='+', default=[])
    p.add_argument('--causal-conditions', nargs='+', default=[])
    p.add_argument('--n-disc', type=int, default=240)
    p.add_argument('--n-t6', type=int, default=160)
    p.add_argument('--n-think', type=int, default=96)
    p.add_argument('--t6-grids', type=int, nargs='+', default=[6, 10, 20])
    p.add_argument('--t5-grids', type=int, nargs='+', default=[3, 6, 10])
    p.add_argument('--n-t5-correct', type=int, default=60)
    p.add_argument('--decomposition-dir', type=Path)
    p.add_argument('--refcoco-cases', type=Path)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--prep-chunk', type=int, default=64)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--max-new-think', type=int, default=1024)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--n-causal', type=int, default=64)
    p.add_argument('--k-cells', type=int, default=2)
    p.add_argument('--k-tokens', type=int, default=4)
    p.add_argument('--max-rows', type=int, default=24)
    p.add_argument('--max-rows-long', type=int, default=6, help='rows per patching batch for prompts of 1000+ tokens')
    p.add_argument('--kv-chunk', type=int, default=4, help='KV-edit conditions per batch for prompts of 1000+ tokens')
    p.add_argument('--n-causal-long', type=int, default=40, help='causal samples for 20x20 grids')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'cases':
        stage_cases(args.output, args)
        return
    rt = qr.load(args.model)
    meta = {'stage': args.stage, 'args': {k: str(v) for k, v in vars(args).items()}, 'runtime': qr.describe(rt),
            'code_sha256': sha256(Path(__file__)), 'started': datetime.now(timezone.utc).isoformat(),
            'device': torch.cuda.get_device_name(0)}
    append_jsonl(args.output / 'run_log.jsonl', [meta])
    STAGES[args.stage](rt, args.output, args)


if __name__ == '__main__':
    main()
