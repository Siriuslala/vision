"""Experiment 2.5a: activation patching between paired anomalous / normal grids.

Design: docs/analysis/representation/design_causal.md.  A pair consists of a
clean image (the designated instance is altered) and a corrupted image (the same
instance is the template); all other pixels and the question are identical, so
the two token sequences align position by position.  For every extraction site
(ViT block outputs, merger output, decoder layer outputs) and every token group,
the hidden states of one input are replaced by those of the other and the
first-answer-token score s = z_Yes - z_No is re-measured.

Stages (run in this order, each writes into the same output directory):
  screen   build all pairs, measure s_clean / s_corrupt, select qualified pairs
  scan     patch every (site, token group) in both directions for qualified pairs
  detail   high-norm group overlap and single text-token patches on the scanned pairs
  analyze  bootstrap summaries and figures
The scan can be re-run with --dtype float32 on a subset of pairs (fp32 check).

All forward passes use a fixed batch size; rows of one batch share the same
receiving input and differ only in the patch that is applied.  Baselines are
measured in batches of the same size so that every score comes from the same
kernel configuration (bfloat16 results depend slightly on the batch size).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.common import stimuli
from analysis_tools.common.stimuli import (
    DEVELOPMENT_TEMPLATES, TOKEN_PIXELS, Change, StimulusSpec, generate, uniform_targets,
)

ANSWER_SUFFIX = ' Answer Yes or No.'
GRID = 6
CELL = 64
# Each condition: change, designation ('rowcol' or 'box'), question, and the correct answer on the clean image.
CONDITIONS: dict[str, dict] = {
    'red_rowcol': {'kind': 'color', 'level': 4, 'designation': 'rowcol', 'clean_answer': 'No',
                   'question': 'Is the figure in row {row}, column {col} exactly identical to the other figures?'},
    'red_box': {'kind': 'color', 'level': 4, 'designation': 'box', 'clean_answer': 'No',
                'question': 'Is the figure inside the blue box exactly identical to the other figures?'},
    'gap_rowcol': {'kind': 'gap', 'level': 4, 'designation': 'rowcol', 'clean_answer': 'No',
                   'question': 'Is the figure in row {row}, column {col} exactly identical to the other figures?'},
    'gap_box': {'kind': 'gap', 'level': 4, 'designation': 'box', 'clean_answer': 'No',
                'question': 'Is the figure inside the blue box exactly identical to the other figures?'},
    'gap_box_attr': {'kind': 'gap', 'level': 4, 'designation': 'box', 'clean_answer': 'Yes',
                     'question': 'Does the figure in the blue box have a gap?'},
}
VISUAL_GROUPS = ('target', 'neighbors', 'high_norm', 'all_visual')
TEXT_GROUPS = ('question', 'template', 'last', 'text_all')
HIGH_NORM_FRACTION = 0.05
FULL_ATTENTION_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MIN_SCORE_GAP = 1.0


# ---------------------------------------------------------------- pairs

@dataclass(frozen=True)
class Pair:
    pair_id: str
    condition: str
    template: str
    target: tuple[int, int]
    site: int
    question: str
    clean: StimulusSpec
    corrupt: StimulusSpec

    @property
    def clean_sign(self) -> int:
        """Sign of s that counts as correct on the clean image (+1 for Yes)."""
        return 1 if CONDITIONS[self.condition]['clean_answer'] == 'Yes' else -1


def make_pairs(condition: str, count: int, seed: int) -> list[Pair]:
    """Pairs share targets, templates and gap sites across conditions for a given seed.

    Templates cycle through the four development templates; targets are uniform over
    the 6x6 grid (stimuli.uniform_targets); gap sites are drawn uniformly from 0..2.
    """
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown condition {condition!r}')
    cfg = CONDITIONS[condition]
    targets = uniform_targets(GRID, GRID, count, seed)
    sites = np.random.default_rng(seed + 1).integers(0, 3, count).tolist()
    pairs = []
    for i, (target, site) in enumerate(zip(targets, sites)):
        template = DEVELOPMENT_TEMPLATES[i % len(DEVELOPMENT_TEMPLATES)]
        site = int(site) if cfg['kind'] == 'gap' else 0
        marker = target if cfg['designation'] == 'box' else None
        common = dict(template=template, rows=GRID, cols=GRID, cell=CELL, marker=marker, seed=seed)
        clean = StimulusSpec(change=Change(cfg['kind'], cfg['level'], site), target=target, **common)
        corrupt = StimulusSpec(**common)
        question = cfg['question'].format(row=target[0] + 1, col=target[1] + 1) + ANSWER_SUFFIX
        pairs.append(Pair(f'{condition}_{i:03d}', condition, template, target, site, question, clean, corrupt))
    return pairs


def render_pair(pair: Pair):
    """Render both images and verify that they differ only inside the target cell."""
    clean_img, clean_meta = generate(pair.clean)
    corrupt_img, corrupt_meta = generate(pair.corrupt)
    diff = (np.asarray(clean_img) != np.asarray(corrupt_img)).any(-1)
    if not diff.any():
        raise ValueError(f'{pair.pair_id}: clean and corrupted images are identical')
    x0, y0, x1, y1 = cell_box(clean_meta, pair.target)
    outside = diff.copy()
    outside[y0:y1, x0:x1] = False
    if outside.any():
        raise ValueError(f'{pair.pair_id}: pixel difference outside the target cell')
    return clean_img, clean_meta, corrupt_img, corrupt_meta


def cell_box(meta: dict, rc: tuple[int, int]) -> list[int]:
    index = rc[0] * meta['spec']['cols'] + rc[1]
    return [int(v) for v in meta['instances'][index]['cell_box']]


# ---------------------------------------------------------------- token groups

def cell_tokens(meta: dict, rc: tuple[int, int]) -> list[int]:
    """Row-major merged-token indices (within the image) covered by one instance cell."""
    x0, y0, x1, y1 = cell_box(meta, rc)
    if any(v % TOKEN_PIXELS for v in (x0, y0, x1, y1)):
        raise ValueError('Cell is not aligned to the 32-pixel token grid')
    width = meta['merged_grid'][1]
    return [r * width + c for r in range(y0 // TOKEN_PIXELS, y1 // TOKEN_PIXELS)
            for c in range(x0 // TOKEN_PIXELS, x1 // TOKEN_PIXELS)]


def neighbor_tokens(meta: dict, rc: tuple[int, int]) -> list[int]:
    """Tokens of the up to four instances sharing an edge with the target."""
    rows, cols = meta['spec']['rows'], meta['spec']['cols']
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = rc[0] + dr, rc[1] + dc
        if 0 <= r < rows and 0 <= c < cols:
            out.extend(cell_tokens(meta, (r, c)))
    return sorted(out)


def random_tokens(n_visual: int, exclude: list[int], size: int, seed: int) -> list[int]:
    pool = np.setdiff1d(np.arange(n_visual), np.asarray(exclude))
    return sorted(np.random.default_rng(seed).choice(pool, size, replace=False).tolist())


def high_norm_tokens(norms: torch.Tensor, fraction: float = HIGH_NORM_FRACTION) -> list[int]:
    """Indices of the ceil(fraction * n) visual tokens with the largest norm."""
    count = max(1, math.ceil(fraction * norms.numel()))
    return sorted(norms.topk(count).indices.tolist())


def text_groups(input_ids: list[int], image_end: int, im_end_id: int) -> dict[str, list[int]]:
    """Sequence positions of the post-image text groups.

    question: user-question tokens between <|vision_end|> and <|im_end|>.
    template: remaining post-image tokens except the last one (<|vision_end|>, <|im_end|>,
              assistant header and the empty thinking block).
    last:     the final prompt token, where the first answer token is predicted.
    """
    n = len(input_ids)
    vision_end = image_end              # first position after the image tokens
    try:
        im_end = input_ids.index(im_end_id, vision_end)
    except ValueError as error:
        raise ValueError('No <|im_end|> after the image') from error
    question = list(range(vision_end + 1, im_end))
    template = [vision_end] + list(range(im_end, n - 1))
    if not question:
        raise ValueError('Empty question span')
    return {'question': question, 'template': template, 'last': [n - 1],
            'text_all': list(range(vision_end, n))}


# ---------------------------------------------------------------- scores

def recovery(s_patched, s_corrupt, s_clean):
    """Fraction of the clean-corrupt score difference restored by patching corrupt <- clean."""
    s_patched, s_corrupt, s_clean = (np.asarray(v, float) for v in (s_patched, s_corrupt, s_clean))
    denominator = s_clean - s_corrupt
    if np.any(denominator == 0):
        raise ValueError('Recovery ratio undefined for identical clean and corrupted scores')
    return (s_patched - s_corrupt) / denominator


def damage(s_patched, s_corrupt, s_clean):
    """Fraction of the score difference destroyed by patching clean <- corrupt."""
    s_patched, s_corrupt, s_clean = (np.asarray(v, float) for v in (s_patched, s_corrupt, s_clean))
    denominator = s_clean - s_corrupt
    if np.any(denominator == 0):
        raise ValueError('Damage ratio undefined for identical clean and corrupted scores')
    return (s_clean - s_patched) / denominator


def qualify(s_clean: float, s_corrupt: float, clean_sign: int, min_gap: float = MIN_SCORE_GAP) -> str:
    """'ok' or the exclusion reason; correctness follows the sign of s."""
    clean_ok = s_clean * clean_sign > 0
    corrupt_ok = s_corrupt * -clean_sign > 0
    if not clean_ok and not corrupt_ok:
        return 'both_wrong'
    if not clean_ok:
        return 'clean_wrong'
    if not corrupt_ok:
        return 'corrupt_wrong'
    if abs(s_clean - s_corrupt) < min_gap:
        return 'small_gap'
    return 'ok'


def bootstrap_summary(numerators: np.ndarray, denominators: np.ndarray, n_boot: int, seed: int) -> dict:
    """Pair-level bootstrap of the median ratio and of the ratio of means."""
    numerators, denominators = np.asarray(numerators, float), np.asarray(denominators, float)
    ratios = numerators / denominators
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratios), (n_boot, len(ratios)))
    medians = np.median(ratios[idx], axis=1)
    rom = numerators[idx].mean(1) / denominators[idx].mean(1)
    return {'n': int(len(ratios)), 'median': float(np.median(ratios)),
            'median_lo': float(np.percentile(medians, 2.5)), 'median_hi': float(np.percentile(medians, 97.5)),
            'ratio_of_means': float(numerators.mean() / denominators.mean()),
            'rom_lo': float(np.percentile(rom, 2.5)), 'rom_hi': float(np.percentile(rom, 97.5))}


# ---------------------------------------------------------------- batched forward with patches

def site_names(vit_layers: list[int], decoder_layers: list[int]) -> list[str]:
    return [f'vit{k}' for k in vit_layers] + ['merger'] + [f'dec{k}' for k in decoder_layers]


def _module(rt, site: str) -> torch.nn.Module:
    visual = rt.model.model.visual
    if site.startswith('vit'):
        return visual.blocks[int(site[3:]) - 1]
    if site == 'merger':
        return visual.merger
    if site.startswith('dec'):
        return rt.layers[int(site[3:]) - 1]
    raise ValueError(f'Unknown site {site!r}')


def _tensor(output):
    return output[0] if isinstance(output, tuple) else output


class Patcher:
    """Batched prefill with per-row replacements at arbitrary sites.

    Visual sites produce tensors of shape (batch * tokens_per_image, dim) with images
    concatenated in batch order; decoder sites produce (batch, seq, dim).
    """

    def __init__(self, rt, batch_size: int, n_patches: int):
        from analysis_tools.common import qwen_runtime as q
        self.rt, self.B = rt, batch_size
        self.rows_per_image = {'vit': n_patches, 'merger': n_patches // 4}
        self.yes, self.no = q.token_id(rt, 'Yes'), q.token_id(rt, 'No')

    def _rows(self, site: str, row: int, positions: torch.Tensor) -> tuple:
        if site.startswith('dec'):
            return (row, positions)
        n = self.rows_per_image['vit' if site.startswith('vit') else 'merger']
        return (row * n + positions,)

    def replicate(self, inputs: list[dict]) -> dict:
        """Stack single-row model inputs (length B, padded by repeating the last one)."""
        inputs = inputs + [inputs[-1]] * (self.B - len(inputs))
        out = {}
        for key in inputs[0]:
            out[key] = torch.cat([x[key] for x in inputs], 0)
        return out

    @torch.inference_mode()
    def run(self, batch: dict, edits: list[list[tuple[str, torch.Tensor, torch.Tensor]]],
            capture: dict[str, list[int]] | None = None) -> tuple[list[float], dict]:
        """edits[row] = [(site, positions, values)]; capture: site -> rows to record."""
        if len(edits) > self.B:
            raise ValueError('More edit rows than the batch size')
        by_site: dict[str, list] = {}
        for row, row_edits in enumerate(edits):
            for site, positions, values in row_edits:
                by_site.setdefault(site, []).append((row, positions, values))
        store, handles = {}, []
        for site in set(by_site) | set(capture or {}):
            def hook(module, inputs, output, site=site):
                h = _tensor(output)
                if site.startswith('dec') and h.shape[0] != self.B:
                    raise RuntimeError('Unexpected decoder batch size')
                for row, positions, values in by_site.get(site, []):
                    h[self._rows(site, row, positions.to(h.device))] = values.to(h.device, h.dtype)
                for row in (capture or {}).get(site, []):
                    if site.startswith('dec'):
                        store[(site, row)] = h[row].detach().clone()
                    else:
                        n = self.rows_per_image['vit' if site.startswith('vit') else 'merger']
                        store[(site, row)] = h[row * n:(row + 1) * n].detach().clone()
                return output
            handles.append(_module(self.rt, site).register_forward_hook(hook))
        try:
            logits = self.rt.model(**batch).logits[:, -1].float()
        finally:
            for handle in handles:
                handle.remove()
        s = (logits[:, self.yes] - logits[:, self.no]).tolist()
        return s[:len(edits)], store


# ---------------------------------------------------------------- stages

def code_hashes() -> dict:
    from analysis_tools.common import qwen_runtime
    files = {'activation_patching': Path(__file__), 'qwen_runtime': Path(qwen_runtime.__file__),
             'stimuli': Path(stimuli.__file__)}
    return {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}


def build_inputs(rt, pair: Pair, images=None) -> tuple[dict, dict, dict, dict]:
    from analysis_tools.common import qwen_runtime as q
    clean_img, clean_meta, corrupt_img, corrupt_meta = images or render_pair(pair)
    clean = dict(q.build(rt, [clean_img], pair.question))
    corrupt = dict(q.build(rt, [corrupt_img], pair.question))
    if not torch.equal(clean['input_ids'], corrupt['input_ids']):
        raise ValueError(f'{pair.pair_id}: token sequences differ between clean and corrupted inputs')
    # Patch rows of merged token m are 4m..4m+3; check the pixel difference respects this layout.
    changed = (clean['pixel_values'] != corrupt['pixel_values']).any(-1).nonzero().flatten() // 4
    if not set(changed.tolist()) <= set(cell_tokens(clean_meta, pair.target)):
        raise ValueError(f'{pair.pair_id}: changed patches fall outside the target tokens')
    return clean, corrupt, clean_meta, corrupt_meta


def stage_screen(rt, args, out: Path) -> dict:
    """Measure baseline scores of all pairs and record qualification."""
    from analysis_tools.common import qwen_runtime as q
    patcher = Patcher(rt, args.batch_size, n_patches=(GRID * CELL // 16) ** 2)
    image_dir = out / 'pairs'
    image_dir.mkdir(parents=True, exist_ok=True)
    rows, stats, t0 = [], {}, time.time()
    for condition in args.conditions:
        pairs = make_pairs(condition, args.pairs_per_condition, args.seed)
        for start in range(0, len(pairs), args.batch_size // 2):
            chunk = pairs[start:start + args.batch_size // 2]
            inputs, metas = [], []
            for pair in chunk:
                images = render_pair(pair)
                images[0].save(image_dir / f'{pair.pair_id}_clean.png')
                images[2].save(image_dir / f'{pair.pair_id}_corrupt.png')
                clean, corrupt, clean_meta, _ = build_inputs(rt, pair, images)
                inputs += [clean, corrupt]
                metas.append(clean_meta)
            scores, _ = patcher.run(patcher.replicate(inputs), [[] for _ in inputs])
            for k, pair in enumerate(chunk):
                s_clean, s_corrupt = scores[2 * k], scores[2 * k + 1]
                status = qualify(s_clean, s_corrupt, pair.clean_sign)
                rows.append({'pair_id': pair.pair_id, 'condition': condition, 'template': pair.template,
                             'target': list(pair.target), 'site': pair.site, 'question': pair.question,
                             's_clean': s_clean, 's_corrupt': s_corrupt, 'status': status,
                             'clean_sha256': metas[k]['sha256'],
                             'clean_spec': {**asdict(pair.clean), 'change': asdict(pair.clean.change)},
                             'corrupt_spec': {**asdict(pair.corrupt), 'change': asdict(pair.corrupt.change)}})
        counts = {}
        for row in rows:
            if row['condition'] == condition:
                counts[row['status']] = counts.get(row['status'], 0) + 1
        stats[condition] = counts
        print(condition, counts, flush=True)
    with open(out / 'screen.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    n_forward = math.ceil(len(rows) * 2 / args.batch_size)
    return {'qualification': stats, 'screen_seconds': time.time() - t0, 'screen_batches': n_forward,
            'runtime': q.describe(rt)}


def pair_from_row(row: dict, count: int, seed: int) -> Pair:
    pair = make_pairs(row['condition'], count, seed)[int(row['pair_id'].rsplit('_', 1)[1])]
    if pair.pair_id != row['pair_id'] or list(pair.target) != row['target'] or pair.site != row['site']:
        raise ValueError(f'Pair {row["pair_id"]} cannot be regenerated consistently')
    return pair


def scan_pair(rt, patcher: Patcher, pair: Pair, args) -> tuple[list[dict], tuple[float, float]]:
    """All (direction, site, group) patches for one pair."""
    clean, corrupt, meta, _ = build_inputs(rt, pair)
    from analysis_tools.common import qwen_runtime as q
    span = q.image_spans(rt, clean)[0]
    n_visual = span['end'] - span['start']
    sites = site_names(args.vit_layers, args.decoder_layers)
    capture_rows = {site: [0, 1] for site in sites}
    scores, cache = patcher.run(patcher.replicate([clean, corrupt]), [[], []], capture_rows)
    s_clean, s_corrupt = scores
    if qualify(s_clean, s_corrupt, pair.clean_sign) != 'ok':
        # Reported in the scan metadata; the pair stays out of every summary.
        print(f'{pair.pair_id}: baseline no longer qualifies ({s_clean:.3f}, {s_corrupt:.3f})', flush=True)
        return [], (s_clean, s_corrupt)
    target = cell_tokens(meta, pair.target)
    groups_visual = {'target': target, 'neighbors': neighbor_tokens(meta, pair.target),
                     'all_visual': list(range(n_visual))}
    salt = int(hashlib.sha256(pair.pair_id.encode()).hexdigest()[:8], 16)
    for seed in range(args.random_seeds):
        groups_visual[f'random{seed}'] = random_tokens(n_visual, target, len(target), salt + seed)
    ids = clean['input_ids'][0].tolist()
    if ids[span['end']] != rt.tokenizer.convert_tokens_to_ids('<|vision_end|>'):
        raise ValueError('Image tokens are not followed by <|vision_end|>')
    texts = text_groups(ids, span['end'], rt.tokenizer.convert_tokens_to_ids('<|im_end|>'))

    def groups_for(site: str) -> dict[str, list[int]]:
        corrupt_h = cache[(site, 1)].float()
        if site.startswith('vit'):
            norms = corrupt_h.reshape(n_visual, -1).norm(dim=-1)
        elif site == 'merger':
            norms = corrupt_h.norm(dim=-1)
        else:
            norms = corrupt_h[span['start']:span['end']].norm(dim=-1)
        groups = dict(groups_visual, high_norm=high_norm_tokens(norms))
        if site.startswith('dec'):
            groups = {k: [span['start'] + t for t in v] for k, v in groups.items()}
            groups.update(texts)
        return groups

    jobs = []   # (site, group, positions)
    for site in sites:
        for group, tokens in groups_for(site).items():
            jobs.append((site, group, tokens))
    # Sanity rows: identity replacement must reproduce the receiver score; replacing the
    # whole sequence at a decoder layer must reproduce the source score.
    seq_len = clean['input_ids'].shape[1]
    for layer in args.sanity_layers:
        jobs.append((f'dec{layer}', 'sanity_all_tokens', list(range(seq_len))))
    jobs.append((f'dec{args.sanity_layers[0]}', 'sanity_identity', list(range(seq_len))))

    def positions_for(site, tokens):
        tokens = torch.tensor(tokens, dtype=torch.long)
        if site.startswith('vit'):
            return (tokens[:, None] * 4 + torch.arange(4)).flatten()
        return tokens

    records = []
    for direction, receiver, source_row, receiver_row in (('restore', corrupt, 0, 1), ('damage', clean, 1, 0)):
        batch = patcher.replicate([receiver])
        for start in range(0, len(jobs), patcher.B):
            chunk = jobs[start:start + patcher.B]
            edits = []
            for site, group, tokens in chunk:
                positions = positions_for(site, tokens)
                row = receiver_row if group == 'sanity_identity' else source_row
                edits.append([(site, positions, cache[(site, row)][positions.to(cache[(site, row)].device)])])
            scores, _ = patcher.run(batch, edits)
            for (site, group, tokens), s in zip(chunk, scores):
                ratio = (recovery(s, s_corrupt, s_clean) if direction == 'restore'
                         else damage(s, s_corrupt, s_clean))
                records.append({'pair_id': pair.pair_id, 'condition': pair.condition, 'direction': direction,
                                'site': site, 'group': group, 'n_tokens': len(tokens), 's_patched': s,
                                's_clean': s_clean, 's_corrupt': s_corrupt, 'ratio': float(ratio)})
    return records, (s_clean, s_corrupt)


def select_pairs(out: Path, args) -> list[dict]:
    """Qualified screen rows of the requested conditions, optionally capped per condition."""
    screen = [json.loads(line) for line in open(out / 'screen.jsonl')]
    selected = [row for row in screen if row['status'] == 'ok' and row['condition'] in args.conditions]
    if args.max_pairs_per_condition:
        kept, counts = [], {}
        for row in selected:
            counts[row['condition']] = counts.get(row['condition'], 0) + 1
            if counts[row['condition']] <= args.max_pairs_per_condition:
                kept.append(row)
        selected = kept
    return selected


def stage_scan(rt, args, out: Path) -> dict:
    from analysis_tools.common import qwen_runtime as q
    selected = select_pairs(out, args)
    path = out / f'records_{args.tag}.jsonl'
    done = set()
    if path.exists():
        done = {json.loads(line)['pair_id'] for line in open(path)}
    patcher = Patcher(rt, args.batch_size, n_patches=(GRID * CELL // 16) ** 2)
    t0, n_pairs, skipped, baseline_shift = time.time(), 0, [], 0.0
    with open(path, 'a') as f:
        for row in selected:
            if row['pair_id'] in done:
                continue
            pair = pair_from_row(row, args.pairs_per_condition, args.seed)
            t = time.time()
            records, (s_clean, s_corrupt) = scan_pair(rt, patcher, pair, args)
            if args.dtype == 'bfloat16':
                baseline_shift = max(baseline_shift, abs(s_clean - row['s_clean']), abs(s_corrupt - row['s_corrupt']))
            if not records:
                skipped.append({'pair_id': pair.pair_id, 's_clean': s_clean, 's_corrupt': s_corrupt})
                continue
            for rec in records:
                f.write(json.dumps(rec) + '\n')
            f.flush()
            n_pairs += 1
            print(f'{pair.pair_id}: {len(records)} patches in {time.time() - t:.1f}s', flush=True)
    return {'scan_pairs': n_pairs, 'scan_seconds': time.time() - t0, 'records_file': path.name,
            'skipped_pairs': skipped, 'max_baseline_shift_vs_screen': baseline_shift,
            'runtime': q.describe(rt)}


def stage_detail(rt, args, out: Path) -> dict:
    """Follow-ups on the scanned pairs.

    1. Overlap of the high-norm group with the target and neighbor groups at every site.
    2. Both directions for the complement of the target group (all non-target visual tokens).
    3. Single-token patches at every post-image text position for args.token_layers,
       locating which text token receives the anomaly information.
    """
    from analysis_tools.common import qwen_runtime as q
    screen = {row['pair_id']: row for row in select_pairs(out, args)}
    overlap_path = out / f'high_norm_overlap_{args.tag}.jsonl'
    done = {json.loads(line)['pair_id'] for line in open(overlap_path)} if overlap_path.exists() else set()
    scanned = [pid for pid in screen if pid not in done]
    patcher = Patcher(rt, args.batch_size, n_patches=(GRID * CELL // 16) ** 2)
    sites = site_names(args.vit_layers, args.decoder_layers)
    token_path = out / f'text_tokens_{args.tag}.jsonl'
    t0 = time.time()
    complement_path = out / f'complement_{args.tag}.jsonl'
    # Files are appended pair by pair; the overlap record is written last, so it marks completion.
    with open(token_path, 'a') as ft, open(complement_path, 'a') as fc:
        for pid in scanned:
            row = screen[pid]
            pair = pair_from_row(row, args.pairs_per_condition, args.seed)
            clean, corrupt, meta, _ = build_inputs(rt, pair)
            span = q.image_spans(rt, clean)[0]
            n_visual = span['end'] - span['start']
            scores, cache = patcher.run(patcher.replicate([clean, corrupt]), [[], []],
                                        {site: [0, 1] for site in sites})
            s_clean, s_corrupt = scores
            target, neighbors = set(cell_tokens(meta, pair.target)), set(neighbor_tokens(meta, pair.target))
            overlap_lines = []
            for site in sites:
                h = cache[(site, 1)].float()
                if site.startswith('vit'):
                    norms = h.reshape(n_visual, -1).norm(dim=-1)
                elif site == 'merger':
                    norms = h.norm(dim=-1)
                else:
                    norms = h[span['start']:span['end']].norm(dim=-1)
                high = high_norm_tokens(norms)
                overlap_lines.append(json.dumps({'pair_id': pid, 'condition': pair.condition, 'site': site, 'high_norm': high,
                                     'n_target': len(target & set(high)),
                                     'n_neighbors': len(neighbors & set(high))}) + '\n')

            # Complement of the target group: every visual token except the target instance.
            others = [t for t in range(n_visual) if t not in target]
            for direction, receiver, source_row in (('restore', corrupt, 0), ('damage', clean, 1)):
                batch = patcher.replicate([receiver])
                for start in range(0, len(sites), patcher.B):
                    chunk = sites[start:start + patcher.B]
                    edits = []
                    for site in chunk:
                        tokens = torch.tensor(others)
                        if site.startswith('vit'):
                            positions = (tokens[:, None] * 4 + torch.arange(4)).flatten()
                        else:
                            positions = tokens + (span['start'] if site.startswith('dec') else 0)
                        source = cache[(site, source_row)]
                        edits.append([(site, positions, source[positions.to(source.device)])])
                    patched, _ = patcher.run(batch, edits)
                    for site, s in zip(chunk, patched):
                        ratio = recovery(s, s_corrupt, s_clean) if direction == 'restore' else damage(s, s_corrupt, s_clean)
                        fc.write(json.dumps({'pair_id': pid, 'condition': pair.condition, 'direction': direction,
                                             'site': site, 'group': 'non_target_visual', 'n_tokens': len(others),
                                             's_patched': s, 's_clean': s_clean, 's_corrupt': s_corrupt,
                                             'ratio': float(ratio)}) + '\n')
            ids = clean['input_ids'][0].tolist()
            positions = list(range(span['end'], len(ids)))
            jobs = [(layer, t) for layer in args.token_layers for t in positions]
            batch = patcher.replicate([corrupt])
            for start in range(0, len(jobs), patcher.B):
                chunk = jobs[start:start + patcher.B]
                edits = [[(f'dec{layer}', torch.tensor([t]), cache[(f'dec{layer}', 0)][[t]])] for layer, t in chunk]
                patched, _ = patcher.run(batch, edits)
                for (layer, t), s in zip(chunk, patched):
                    ft.write(json.dumps({'pair_id': pid, 'condition': pair.condition, 'layer': layer,
                                         'offset': t - len(ids), 'token': rt.tokenizer.decode([ids[t]]),
                                         's_patched': s, 's_clean': s_clean, 's_corrupt': s_corrupt,
                                         'ratio': float(recovery(s, s_corrupt, s_clean))}) + '\n')
            ft.flush()
            fc.flush()
            with open(overlap_path, 'a') as fo:
                fo.writelines(overlap_lines)
            print(f'{pid}: detail done', flush=True)
    return {'detail_pairs': len(scanned), 'detail_seconds': time.time() - t0, 'runtime': q.describe(rt)}


# ---------------------------------------------------------------- analysis and figures

def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def summarize(records: list[dict], n_boot: int, seed: int, random_seeds: int) -> list[dict]:
    """Per condition x direction x site x group summaries; random groups are averaged per pair."""
    table: dict[tuple, dict[str, tuple[float, float]]] = {}
    for rec in records:
        group = 'random' if rec['group'].startswith('random') else rec['group']
        key = (rec['condition'], rec['direction'], rec['site'], group)
        if rec['direction'] == 'restore':
            num = rec['s_patched'] - rec['s_corrupt']
        else:
            num = rec['s_clean'] - rec['s_patched']
        den = rec['s_clean'] - rec['s_corrupt']
        table.setdefault(key, {}).setdefault(rec['pair_id'], []).append((num, den))
    rows = []
    for key, per_pair in sorted(table.items()):
        nums = np.array([np.mean([v[0] for v in vals]) for vals in per_pair.values()])
        dens = np.array([vals[0][1] for vals in per_pair.values()])
        if key[3] == 'random' and any(len(v) != random_seeds for v in per_pair.values()):
            raise ValueError(f'Incomplete random seeds for {key}')
        rows.append({'condition': key[0], 'direction': key[1], 'site': key[2], 'group': key[3],
                     **bootstrap_summary(nums, dens, n_boot, seed)})
    return rows


def transfer_layer(summary: list[dict], condition: str, direction: str = 'restore') -> int | None:
    """Shallowest decoder layer at which the last prompt token's median ratio exceeds the target group's."""
    med = {(r['site'], r['group']): r['median'] for r in summary
           if r['condition'] == condition and r['direction'] == direction}
    layers = sorted(int(s[3:]) for s, g in med if s.startswith('dec') and g == 'last')
    for layer in layers:
        if med[(f'dec{layer}', 'last')] > med.get((f'dec{layer}', 'target'), math.inf):
            return layer
    return None


INK, MUTED, GRIDLINE = '#0b0b0b', '#898781', '#e1e0d9'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GROUP_ORDER = ['target', 'neighbors', 'high_norm', 'random', 'non_target_visual', 'all_visual', 'question', 'template', 'last', 'text_all']
GROUP_LABEL = {'target': 'target instance', 'neighbors': 'neighbor instances', 'high_norm': 'high-norm (top 5%)',
               'random': 'random visual (mean of 5)', 'non_target_visual': 'all visual except target',
               'all_visual': 'all visual', 'question': 'question text',
               'template': 'chat template', 'last': 'last prompt token', 'text_all': 'all post-image text'}


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('blue_gray_red', ['#1c5cab', '#86b6ef', '#f0efec', '#ec835a', '#b8322f'])


def plot_heatmaps(summary: list[dict], conditions: list[str], direction: str, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sites = sorted({r['site'] for r in summary}, key=_site_order)
    fig, axes = plt.subplots(len(conditions), 1, figsize=(13, 3.2 * len(conditions)), squeeze=False)
    fig.subplots_adjust(hspace=0.45)
    for ax, condition in zip(axes[:, 0], conditions):
        med = {(r['site'], r['group']): r['median'] for r in summary
               if r['condition'] == condition and r['direction'] == direction}
        n = max((r['n'] for r in summary if r['condition'] == condition and r['direction'] == direction), default=0)
        grid = np.full((len(GROUP_ORDER), len(sites)), np.nan)
        for i, group in enumerate(GROUP_ORDER):
            for j, site in enumerate(sites):
                grid[i, j] = med.get((site, group), np.nan)
        im = ax.imshow(grid, cmap=_diverging_cmap(), vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
        ax.set_yticks(range(len(GROUP_ORDER)), [GROUP_LABEL[g] for g in GROUP_ORDER], fontsize=8, color=INK)
        ax.set_xticks(range(len(sites)), [_site_label(s) for s in sites], fontsize=7, rotation=90, color=INK)
        for j, site in enumerate(sites):
            if site.startswith('dec') and int(site[3:]) in FULL_ATTENTION_LAYERS:
                ax.get_xticklabels()[j].set_fontweight('bold')
        ax.axvline(sites.index('merger') + 0.5, color=INK, lw=1)
        ax.set_title(f'{condition}  (n = {n} pairs)', fontsize=10, loc='left', color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    name = 'recovery ratio r (corrupt <- clean)' if direction == 'restore' else 'damage ratio (clean <- corrupt)'
    cbar = fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.015, pad=0.01)
    cbar.set_label(f'median {name}', fontsize=9)
    fig.suptitle(f'Activation patching, {direction} direction; bold decoder labels = full attention layers',
                 fontsize=10, x=0.01, y=0.93 + 0.012 * len(conditions), ha='left', color=INK)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_transfer(summary: list[dict], conditions: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ['target', 'non_target_visual', 'all_visual', 'question', 'template', 'last', 'random']
    fig, axes = plt.subplots(2, len(conditions), figsize=(3.6 * len(conditions), 6), squeeze=False, sharey=True)
    for col, condition in enumerate(conditions):
        for row, direction in enumerate(('restore', 'damage')):
            ax = axes[row, col]
            for layer in FULL_ATTENTION_LAYERS:
                ax.axvline(layer, color=GRIDLINE, lw=1, zorder=0)
            for color, group in zip(SERIES, groups):
                pts = sorted((int(r['site'][3:]), r['median'], r['median_lo'], r['median_hi']) for r in summary
                             if r['condition'] == condition and r['direction'] == direction
                             and r['group'] == group and r['site'].startswith('dec'))
                if not pts:
                    continue
                x, y, lo, hi = map(np.array, zip(*pts))
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
                ax.plot(x, y, color=color, lw=2, label=GROUP_LABEL[group])
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_ylim(-0.3, 1.2)
            ax.set_xlim(1, 32)
            ax.set_title(f'{condition}, {direction}', fontsize=9, loc='left')
            ax.tick_params(labelsize=8)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if row == 1:
                ax.set_xlabel('decoder layer (grey lines: full attention)', fontsize=8)
            if col == 0:
                ax.set_ylabel('median ratio (95% CI)', fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(groups), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_example(out: Path, screen: list[dict], pair_ids: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    rows = {r['pair_id']: r for r in screen}
    fig, axes = plt.subplots(len(pair_ids), 2, figsize=(6.4, 3.4 * len(pair_ids)), squeeze=False)
    for k, pid in enumerate(pair_ids):
        row = rows[pid]
        for j, kind in enumerate(('clean', 'corrupt')):
            ax = axes[k, j]
            ax.imshow(Image.open(out / 'pairs' / f'{pid}_{kind}.png'))
            r, c = row['target']
            ax.add_patch(plt.Rectangle((c * CELL - 0.5, r * CELL - 0.5), CELL, CELL, fill=False,
                                       ec='#eda100', lw=1.5, ls='--'))
            ax.set_xticks([])
            ax.set_yticks([])
            s = row[f's_{kind}']
            ax.set_title(f'{pid} {kind}\ns = {s:+.2f}', fontsize=8)
        axes[k, 0].set_xlabel(row['question'], fontsize=6.5, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_text_tokens(rows: list[dict], conditions: list[str], path: Path) -> None:
    """Median single-token recovery ratio over decoder layer x post-image text position."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(conditions), 1, figsize=(12, 3.4 * len(conditions)), squeeze=False)
    fig.subplots_adjust(hspace=0.75)
    for ax, condition in zip(axes[:, 0], conditions):
        sub = [r for r in rows if r['condition'] == condition]
        layers = sorted({r['layer'] for r in sub})
        offsets = sorted({r['offset'] for r in sub})
        labels = {}
        values: dict[tuple, list] = {}
        for r in sub:
            values.setdefault((r['layer'], r['offset']), []).append(r['ratio'])
            labels.setdefault(r['offset'], set()).add(r['token'])
        grid = np.array([[np.median(values[(layer, o)]) for o in offsets] for layer in layers])
        im = ax.imshow(grid, cmap=_diverging_cmap(), vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
        names = [next(iter(labels[o])) if len(labels[o]) == 1 else '<digit>' for o in offsets]
        ax.set_xticks(range(len(offsets)), [repr(n)[1:-1] for n in names], rotation=90, fontsize=7)
        ax.set_yticks(range(len(layers)), [f'L{layer}' for layer in layers], fontsize=7)
        n = len({r['pair_id'] for r in sub})
        ax.set_title(f'{condition}  (n = {n} pairs)', fontsize=9, loc='left')
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.015, pad=0.01)
    cbar.set_label('median recovery ratio r, single text token', fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def high_norm_split(records: list[dict], overlap: list[dict], n_boot: int, seed: int) -> list[dict]:
    """Restore-direction high-norm ratios restricted to pairs whose high-norm group excludes the target."""
    clean_pairs = {(o['pair_id'], o['site']) for o in overlap if o['n_target'] == 0}
    table: dict[tuple, list] = {}
    for rec in records:
        if rec['group'] != 'high_norm' or rec['direction'] != 'restore':
            continue
        if (rec['pair_id'], rec['site']) in clean_pairs:
            table.setdefault((rec['condition'], rec['site']), []).append(
                (rec['s_patched'] - rec['s_corrupt'], rec['s_clean'] - rec['s_corrupt']))
    out = []
    for (condition, site), vals in sorted(table.items()):
        nums, dens = map(np.array, zip(*vals))
        out.append({'condition': condition, 'site': site, **bootstrap_summary(nums, dens, n_boot, seed)})
    return out


def compare_tags(records: list[dict], reference: list[dict]) -> dict:
    """Compare per-patch ratios and per-cell medians with a reference run on the shared pairs."""
    key = lambda r: (r['pair_id'], r['direction'], r['site'], r['group'])
    ref = {key(r): r for r in reference}
    shared = [(r, ref[key(r)]) for r in records if key(r) in ref]
    if not shared:
        raise ValueError('No shared patches between the two runs')
    diffs = np.array([a['ratio'] - b['ratio'] for a, b in shared])
    cells: dict[tuple, tuple[list, list]] = {}
    for a, b in shared:
        group = 'random' if a['group'].startswith('random') else a['group']
        cell = cells.setdefault((a['condition'], a['direction'], a['site'], group), ([], []))
        cell[0].append(a['ratio'])
        cell[1].append(b['ratio'])
    median_diff = np.array([np.median(x) - np.median(y) for x, y in cells.values()])
    baselines = {(a['pair_id']): (a['s_clean'] - b['s_clean'], a['s_corrupt'] - b['s_corrupt']) for a, b in shared}
    return {'n_pairs': len(baselines), 'n_patches': len(shared),
            'ratio_abs_diff_median': float(np.median(np.abs(diffs))),
            'ratio_abs_diff_p99': float(np.percentile(np.abs(diffs), 99)),
            'ratio_abs_diff_max': float(np.abs(diffs).max()),
            'cell_median_abs_diff_max': float(np.abs(median_diff).max()),
            'cell_median_abs_diff_mean': float(np.abs(median_diff).mean()),
            'baseline_abs_diff_max': float(np.abs(np.array(list(baselines.values()))).max())}


def _site_order(site: str) -> tuple:
    if site.startswith('vit'):
        return (0, int(site[3:]))
    if site == 'merger':
        return (1, 0)
    return (2, int(site[3:]))


def _site_label(site: str) -> str:
    if site.startswith('vit'):
        return f'ViT {site[3:]}'
    if site == 'merger':
        return 'merger'
    return site[3:]


def stage_analyze(args, out: Path) -> dict:
    records = load_records(out / f'records_{args.tag}.jsonl')
    scanned_ids = {r['pair_id'] for r in records}
    if (out / f'complement_{args.tag}.jsonl').exists():
        records += [r for r in load_records(out / f'complement_{args.tag}.jsonl') if r['pair_id'] in scanned_ids]
    sanity = [r for r in records if r['group'].startswith('sanity')]
    records = [r for r in records if not r['group'].startswith('sanity')]
    summary = summarize(records, args.n_boot, args.seed, args.random_seeds)
    with open(out / f'summary_{args.tag}.json', 'w') as f:
        json.dump(summary, f, indent=1)
    conditions = [c for c in args.conditions if any(r['condition'] == c for r in summary)]
    for direction in ('restore', 'damage'):
        plot_heatmaps(summary, conditions, direction, out / f'heatmap_{direction}_{args.tag}.png')
    plot_transfer(summary, conditions, out / f'transfer_{args.tag}.png')
    screen = [json.loads(line) for line in open(out / 'screen.jsonl')]
    scanned = {r['pair_id'] for r in records}
    examples = [next(r['pair_id'] for r in screen if r['condition'] == c and r['pair_id'] in scanned)
                for c in conditions]
    plot_example(out, screen, examples, out / f'examples_{args.tag}.png')
    extra = {}
    token_path, overlap_path = out / f'text_tokens_{args.tag}.jsonl', out / f'high_norm_overlap_{args.tag}.jsonl'
    if token_path.exists():
        plot_text_tokens([r for r in load_records(token_path) if r['pair_id'] in scanned_ids], conditions, out / f'text_tokens_{args.tag}.png')
    if overlap_path.exists():
        overlap = [r for r in load_records(overlap_path) if r['pair_id'] in scanned_ids]
        split = high_norm_split(records, overlap, args.n_boot, args.seed)
        with open(out / f'high_norm_no_overlap_{args.tag}.json', 'w') as f:
            json.dump(split, f, indent=1)
        extra['high_norm_target_overlap_fraction'] = {
            c: {site: float(np.mean([o['n_target'] > 0 for o in overlap if o['condition'] == c and o['site'] == site]))
                for site in sorted({o['site'] for o in overlap}, key=_site_order)} for c in conditions}
    if args.reference_tag:
        extra['precision_check'] = compare_tags(records, load_records(out / f'records_{args.reference_tag}.jsonl'))
    sanity_err = {}
    for r in sanity:
        expected = (1.0 if r['group'] == 'sanity_all_tokens' else 0.0)
        sanity_err[r['group']] = max(sanity_err.get(r['group'], 0.0), abs(r['ratio'] - expected))
    return {**extra, 'examples': examples, 'transfer_layer': {c: transfer_layer(summary, c) for c in conditions},
            'sanity_max_abs_error': sanity_err, 'n_records': len(records)}


# ---------------------------------------------------------------- entry point

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=('screen', 'scan', 'detail', 'analyze'), required=True)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--conditions', nargs='+', default=list(CONDITIONS))
    p.add_argument('--pairs-per-condition', type=int, default=64)
    p.add_argument('--max-pairs-per-condition', type=int, default=0, help='0 scans every qualified pair')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--vit-layers', type=int, nargs='+', default=[3, 9, 18, 27])
    p.add_argument('--decoder-layers', type=int, nargs='+', default=list(range(1, 33)))
    p.add_argument('--sanity-layers', type=int, nargs='+', default=[1, 16])
    p.add_argument('--token-layers', type=int, nargs='+', default=[12, 14, 15, 16, 17, 20, 24, 28])
    p.add_argument('--random-seeds', type=int, default=5)
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    p.add_argument('--tag', default='bf16', help='suffix of the records / summary files')
    p.add_argument('--reference-tag', default='', help='analyze: compare ratios with this run on shared pairs')
    args = p.parse_args(argv)
    if args.batch_size % 2:
        raise ValueError('batch size must be even (screen batches hold clean/corrupt pairs)')
    for condition in args.conditions:
        if condition not in CONDITIONS:
            raise ValueError(f'Unknown condition {condition!r}')
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime('%Y-%m-%d %H:%M:%S')
    meta = {'stage': args.stage, 'started': started, 'args': {k: str(v) if isinstance(v, Path) else v
                                                             for k, v in vars(args).items()},
            'generator': stimuli.GENERATOR_VERSION, 'code_sha256': code_hashes(),
            'conditions': {c: CONDITIONS[c] for c in args.conditions}, 'answer_suffix': ANSWER_SUFFIX,
            'grid': GRID, 'cell': CELL, 'high_norm_fraction': HIGH_NORM_FRACTION, 'min_score_gap': MIN_SCORE_GAP}
    if args.stage == 'analyze':
        meta.update(stage_analyze(args, out))
    else:
        from analysis_tools.common import qwen_runtime as q
        rt = q.load(args.model, dtype=getattr(torch, args.dtype))
        stage = {'screen': stage_screen, 'scan': stage_scan, 'detail': stage_detail}[args.stage]
        meta.update(stage(rt, args, out))
    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    name = f'metadata_{args.stage}' + ('' if args.stage == 'screen' else f'_{args.tag}') + '.json'
    with open(out / name, 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ('code_sha256', 'runtime')}, indent=1)[:3000])


if __name__ == '__main__':
    main()
