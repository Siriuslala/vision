"""Experiment 3.4: position probe refinement (data, feature extraction, moment-based ridge).

Pipeline (see docs/analysis/position/design_probe_refinement.md):

1. ``prepare``  selects deduplicated RefCOCO images, splits them by whole image, resizes them
   (512 px, plus a 768 px copy of the test split), and renders the synthetic stimuli.
2. ``extract --phase trainval`` runs prefill on training and validation images, accumulates
   the ridge sufficient statistics of every extraction position, and stores 64 sampled tokens
   per image at the feature positions used by MLP and relative-position probes.
3. ``fit-ridge`` solves RP256 / RP1024 / full ridge from the statistics for every lambda,
   selects lambda on validation MSE, and fits learning-curve subsets.
4. ``extract --phase test`` runs prefill on test images (512 px and 768 px) and synthetic
   stimuli, applying every fitted ridge probe on the fly to all visual tokens, and stores
   sampled (natural) or full (synthetic) features at the feature positions.

MLP, relative-position probes, synthetic-internal probes and metrics live in
``position_probe_refinement_probes.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from analysis_tools.position.position_real_probe import patch_grid

PROMPT = 'Describe the image briefly.'
TOKENS_PER_IMAGE = 64
SPLIT_SEED = 20261014
SAMPLE_SEED = 30261014
SHUFFLE_SEED = 40261014
RP_SEEDS = {256: 20261001, 1024: 20261002}     # RP256 reuses the projection seed of experiment 3.3
LAMBDAS = np.logspace(-2, 6, 17)
LC_GROUP_IMAGES = 75
LC_SIZES = (75, 150, 300, 600, 1200, 2400)
LC_ORDERS = 3
# Label columns accumulated for every token: image coordinates (0-1000), grid coordinates,
# and the same two labels after a global permutation of training tokens (shuffled control).
LABEL_COLUMNS = ('image_x', 'image_y', 'grid_x', 'grid_y', 'shuf_image_x', 'shuf_image_y', 'shuf_grid_x', 'shuf_grid_y')
SYNTH_GRIDS = ((6, 6, 64), (10, 10, 64), (20, 20, 32))
SYNTH_OFFSETS = (('phase', 0), ('phase', 8), ('phase', 16), ('jitter', 4))
SYNTH_PADS = (0, 64, 128, 192)
UNIFORM_COLORS = {'white': 255, 'gray': 128}
UNIFORM_SIZES = {
    'square_small': [(v, v) for v in (384, 448, 512, 576)],
    'square_large': [(v, v) for v in (640, 704, 768, 832)],
    'nonsquare': [(384, 640), (448, 768), (512, 832), (576, 704), (640, 448), (704, 512), (768, 384), (832, 576)],
}


def stage_names(vit_depth: int = 27, decoder_types: list[str] | None = None) -> list[str]:
    decoder_types = decoder_types or []
    return (['patch_content', 'vit_00_position'] + [f'vit_{i:02d}' for i in range(1, vit_depth + 1)] + ['merger']
            + [f'decoder_{i + 1:02d}_{t}' for i, t in enumerate(decoder_types)])


def is_vision(stage: str) -> bool:
    return stage == 'patch_content' or stage.startswith('vit_')


def decoder_layer(stage: str) -> int | None:
    return int(stage.split('_')[1]) if stage.startswith('decoder_') else None


def token_labels(patch_hw: tuple[int, int], vision: bool) -> tuple[np.ndarray, np.ndarray]:
    """Grid (col, row) and image (0-1000) labels of every token in model sequence order."""
    h, w = patch_hw
    if vision:
        xy = patch_grid(h, w).astype(np.float64)
    else:
        h, w = h // 2, w // 2
        k = np.arange(h * w)
        xy = np.stack([k % w, k // w], -1).astype(np.float64)
    return xy, (xy + .5) / np.array([w, h], np.float64) * 1000


def sample_tokens(patch_hw: tuple[int, int], case_id: int, count: int = TOKENS_PER_IMAGE) -> tuple[np.ndarray, np.ndarray]:
    """Uniform merged-token sample without replacement and one patch inside each merged token."""
    merged = patch_hw[0] * patch_hw[1] // 4
    if merged < count:
        raise ValueError(f'Case {case_id} has {merged} merged tokens, fewer than {count}.')
    rng = np.random.default_rng(SAMPLE_SEED + case_id)
    mids = np.sort(rng.choice(merged, count, replace=False))
    return mids, mids * 4 + rng.integers(0, 4, count)


def resize_to(image: Image.Image, longest: int) -> Image.Image:
    ratio = longest / max(image.size)
    size = tuple(max(32, int(round(v * ratio / 32)) * 32) for v in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


# ----------------------------------------------------------------------------- prepare

def synthetic_specs() -> list[dict]:
    """Repeated patterns (all six templates) and uniform canvases with matched sizes."""
    from analysis_tools.common import stimuli
    rows = []
    for template in stimuli.TEMPLATES:
        for r, c, cell in SYNTH_GRIDS:
            for o, (kind, value) in enumerate(SYNTH_OFFSETS):
                for p, pad in enumerate(SYNTH_PADS):
                    margin = 32
                    if kind == 'phase' and value == 0:
                        margin = 0
                    canvas_min = max(stimuli.MIN_CANVAS, max(r, c) * cell + margin + pad)
                    spec = dict(template=template, rows=r, cols=c, cell=cell, canvas_min=canvas_min,
                                phase=value if kind == 'phase' else 0, jitter=value if kind == 'jitter' else 0,
                                seed=1000 * o + p)
                    rows.append(dict(kind='pattern', template=template, held_out=template in stimuli.HELD_OUT_TEMPLATES,
                                     grid=f'{r}x{c}', offset=f'{kind}{value}', pad=pad, spec=spec))
    # Uniform canvases cover the pattern canvas sizes in three disjoint size groups; 'grid' holds the group name.
    for group, sizes in UNIFORM_SIZES.items():
        for width, height in sizes:
            for color in UNIFORM_COLORS:
                rows.append(dict(kind='uniform', template=f'uniform_{color}', held_out=False, grid=group,
                                 offset='none', pad=0, color=color, size=[width, height]))
    return rows


def render_synthetic(row: dict) -> tuple[Image.Image, dict]:
    from analysis_tools.common import stimuli
    if row['kind'] == 'pattern':
        return stimuli.generate(stimuli.StimulusSpec(**row['spec']))
    width, height = row['size']
    value = UNIFORM_COLORS[row['color']]
    return Image.new('RGB', (width, height), (value,) * 3), {'canvas': [width, height]}


def write_synthetic(out: Path) -> list[dict]:
    rows = []
    for j, row in enumerate(synthetic_specs()):
        image, _ = render_synthetic(row)
        image.save(out / 'images/synthetic' / f'{j:04d}.png')
        rows.append(dict(row, id=j, input_size=[image.width, image.height], patch_hw=[image.height // 16, image.width // 16]))
    return rows


def prepare_synthetic(out: Path) -> None:
    """Re-render the synthetic set of an existing run, allowed only before any synthetic extraction."""
    if (out / 'predictions' / 'synthetic.done').exists():
        raise FileExistsError('Synthetic features were already extracted for this run.')
    cases = json.loads((out / 'cases.json').read_text())
    for f in (out / 'images/synthetic').glob('*.png'):
        f.unlink()
    cases['synthetic'] = write_synthetic(out)
    (out / 'cases.json').write_text(json.dumps(cases, indent=1) + '\n')
    print(f'Rendered {len(cases["synthetic"])} synthetic images.', flush=True)


def prepare(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; prepare never overwrites a run.')
    for sub in ('images/natural512', 'images/natural768', 'images/synthetic'):
        (out / sub).mkdir(parents=True)
    pool, seen = [], set()
    shards = sorted((args.refcoco / 'data').glob('*.parquet'))
    if not shards:
        raise FileNotFoundError(f'No RefCOCO parquet shards under {args.refcoco}/data')
    for shard in shards:
        table = pq.read_table(shard, columns=['image', 'file_name'])
        for idx, (img, name) in enumerate(zip(table.column('image').to_pylist(), table.column('file_name').to_pylist())):
            raw = img['bytes']
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            pool.append((digest, shard.name, idx, name))
    print(f'RefCOCO unique images: {len(pool)}', flush=True)
    n_total = sum(args.split)
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(pool))
    previous = set()
    if args.previous_cases and args.previous_cases.exists():
        previous = {r['source_sha256'] for r in json.loads(args.previous_cases.read_text())}
    chosen, rejected_aspect = [], 0
    # Keep every shard's image column in memory (about 2 GB) so random-order access stays cheap.
    cache = {s.name: pq.read_table(s, columns=['image']).column('image') for s in shards}
    for k in order:
        digest, shard, idx, name = pool[k]
        image = Image.open(io.BytesIO(cache[shard][idx].as_py()['bytes'])).convert('RGB')
        if max(image.size) / min(image.size) > 3:      # same aspect filter as experiment 3.3
            rejected_aspect += 1
            continue
        chosen.append((digest, shard, idx, name, image))
        if len(chosen) == n_total:
            break
    if len(chosen) != n_total:
        raise ValueError(f'Only {len(chosen)} usable unique RefCOCO images, need {n_total}.')
    names = ['train'] * args.split[0] + ['validation'] * args.split[1] + ['test'] * args.split[2]
    cases = []
    for i, ((digest, shard, idx, name, image), split) in enumerate(zip(chosen, names)):
        r512 = resize_to(image, 512)
        r512.save(out / 'images/natural512' / f'{i:04d}.png')
        row = dict(id=i, split=split, source='RefCOCO', shard=shard, row=idx, file_name=name, source_sha256=digest,
                   source_size=list(image.size), input_size=list(r512.size), patch_hw=[r512.height // 16, r512.width // 16],
                   in_experiment_3_3=digest in previous)
        if split == 'test':
            r768 = resize_to(image, 768)
            r768.save(out / 'images/natural768' / f'{i:04d}.png')
            row.update(input_size_768=list(r768.size), patch_hw_768=[r768.height // 16, r768.width // 16])
        cases.append(row)
    synthetic = write_synthetic(out)
    (out / 'cases.json').write_text(json.dumps(dict(natural=cases, synthetic=synthetic), indent=1) + '\n')
    (out / 'prepare.json').write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(), refcoco_unique_images=len(pool),
        rejected_aspect_over_3=rejected_aspect, split_seed=SPLIT_SEED, split=list(args.split),
        overlap_with_experiment_3_3=sum(c['in_experiment_3_3'] for c in cases),
        overlap_by_split={s: sum(c['in_experiment_3_3'] for c in cases if c['split'] == s) for s in ('train', 'validation', 'test')},
        previous_cases_found=bool(previous), synthetic_images=len(synthetic), prompt=PROMPT), indent=2) + '\n')
    print(f'Prepared {len(cases)} natural and {len(synthetic)} synthetic images.', flush=True)


# ----------------------------------------------------------------------------- statistics

class Moments:
    """Running float64 sums for ridge: n, sum x, X^T X, X^T Y, sum y, Y^T Y."""

    def __init__(self, dim: int, labels: int, device, dtype: str = 'float64'):
        import torch
        z = lambda *s: torch.zeros(*s, dtype=getattr(torch, dtype), device=device)
        self.n, self.s, self.c, self.u, self.sy, self.yy = z(()), z(dim), z(dim, dim), z(dim, labels), z(labels), z(labels, labels)

    def add(self, x, y) -> None:
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.c.addmm_(x.T, x)
        self.u.addmm_(x.T, y)
        self.sy += y.sum(0)
        self.yy.addmm_(y.T, y)

    def merge(self, other: 'Moments') -> None:
        for k in ('n', 's', 'c', 'u', 'sy', 'yy'):
            getattr(self, k).add_(getattr(other, k))

    def zero(self) -> None:
        for k in ('n', 's', 'c', 'u', 'sy', 'yy'):
            getattr(self, k).zero_()

    def state(self) -> dict:
        return {k: getattr(self, k).cpu() for k in ('n', 's', 'c', 'u', 'sy', 'yy')}


class ShiftedMoments:
    """GPU float32 moments of shifted data, flushed into exact float64 raw moments on the CPU.

    Inputs are shifted by a fixed reference (x - m0, y - c0) before float32 accumulation to avoid
    cancellation; ``flush`` maps the shifted sums back to raw sums in float64 and adds them to ``total``.
    Raw float64 moments are the stored format consumed by ``ridge_path``.
    """

    def __init__(self, dim: int, labels: int, device, y_shift: float = 500.):
        self.gpu = Moments(dim, labels, device, dtype='float32')
        self.total = Moments(dim, labels, 'cpu')
        self.m0, self.c0 = None, y_shift
        self.pending = 0

    def add(self, x, y) -> None:
        x = x.float()
        if self.m0 is None:
            self.m0 = x.mean(0)
        self.gpu.add(x - self.m0, y.float() - self.c0)
        self.pending += 1

    def flush(self) -> None:
        import torch
        if not self.pending:
            return
        g = {k: v.double().cpu() for k, v in self.gpu.__dict__.items()}
        m0 = self.m0.double().cpu()
        c0 = torch.full_like(g['sy'], self.c0)
        n = g['n']
        t = self.total
        t.n += n
        t.s += g['s'] + n * m0
        t.c += g['c'] + torch.outer(m0, g['s']) + torch.outer(g['s'], m0) + n * torch.outer(m0, m0)
        t.u += g['u'] + torch.outer(g['s'], c0) + torch.outer(m0, g['sy']) + n * torch.outer(m0, c0)
        t.sy += g['sy'] + n * c0
        t.yy += g['yy'] + torch.outer(c0, g['sy']) + torch.outer(g['sy'], c0) + n * torch.outer(c0, c0)
        self.gpu.zero()
        self.pending = 0


def load_moments(path: Path, device='cpu') -> dict:
    import torch
    return {k: v.to(device) for k, v in torch.load(path, map_location='cpu').items()}


def sum_moments(items: list[dict]) -> dict:
    return {k: sum(m[k] for m in items) for k in items[0]}


def project_moments(m: dict, proj) -> dict:
    """Moments of x @ P computed exactly from moments of x."""
    return dict(n=m['n'], s=proj.T @ m['s'], c=proj.T @ m['c'] @ proj, u=proj.T @ m['u'], sy=m['sy'], yy=m['yy'])


def ridge_path(train: dict, val: dict, columns: list[int], lambdas=LAMBDAS, sigma_floor: float = 1e-6) -> dict:
    """Closed-form ridge on standardized inputs for every lambda (one eigendecomposition).

    Returns raw-space affine maps W (len(lambdas), D, k), b (len(lambdas), k) such that
    prediction = x @ W + b, together with validation MSE per lambda computed from validation moments.
    """
    import torch
    n = train['n']
    mu = train['s'] / n
    var = (torch.diagonal(train['c']) / n - mu ** 2).clamp_min(0)
    sigma = var.sqrt().clamp_min(sigma_floor * float(var.sqrt().max()))
    ybar = train['sy'][columns] / n
    cc = train['c'] - n * torch.outer(mu, mu)
    uc = train['u'][:, columns] - n * torch.outer(mu, ybar)
    cz = cc / torch.outer(sigma, sigma)
    uz = uc / sigma[:, None]
    evals, q = torch.linalg.eigh((cz + cz.T) / 2)
    qu = q.T @ uz
    lam = torch.as_tensor(lambdas, dtype=cz.dtype, device=cz.device)
    bz = torch.einsum('dk,lk,kc->ldc', q, 1 / (evals.clamp_min(0)[None] + lam[:, None]), qu)
    w = bz / sigma[None, :, None]
    b = ybar[None] - torch.einsum('d,ldc->lc', mu, w)
    # Validation MSE from moments: sum ||x W + b - y||^2 over validation tokens, per output column.
    vn = val['n']
    xwx = torch.einsum('ldc,de,lec->lc', w, val['c'], w)
    cross = 2 * b * torch.einsum('d,ldc->lc', val['s'], w)
    xy = torch.einsum('ldc,dc->lc', w, val['u'][:, columns])
    yv = torch.diagonal(val['yy'])[columns]
    sse = xwx + cross + vn * b ** 2 - 2 * xy - 2 * b * val['sy'][columns][None] + yv[None]
    return dict(w=w, b=b, val_mse=(sse / vn), eig_min=float(evals.min()), eig_max=float(evals.max()))


def fit_ridge(args: argparse.Namespace) -> None:
    import torch
    out = args.output
    meta = json.loads((out / 'extract_trainval.json').read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    (out / 'ridge').mkdir(exist_ok=True)
    summary = {}
    lc_orders = [np.arange(meta['lc_groups'])] + [np.random.default_rng(50261014 + k).permutation(meta['lc_groups'])
                                                   for k in range(1, LC_ORDERS)]
    projections = {}

    def projection(dim: int, k: int):
        if (dim, k) not in projections:
            projections[dim, k] = torch.as_tensor(np.random.default_rng(RP_SEEDS[k]).normal(size=(dim, k)).astype(np.float32)
                                                  / np.float32(np.sqrt(dim)), dtype=torch.float64, device=device)
        return projections[dim, k]

    sizes = [s for s in LC_SIZES if s <= meta['lc_groups'] * LC_GROUP_IMAGES]
    for stage in meta['stages']:
        target = out / 'ridge' / f'{stage}.pt'
        if target.exists():
            summary[stage] = torch.load(target, map_location='cpu')['summary']
            continue
        train = load_moments(out / 'stats' / 'train' / f'{stage}.pt', device)
        val = load_moments(out / 'stats' / 'validation' / f'{stage}.pt', device)
        dim = train['s'].shape[0]
        cols, names, info = [], [], {}
        for probe in ('rp256', 'rp1024', 'full'):
            if probe == 'full':
                tr, va = train, val
            else:
                proj = projection(dim, int(probe[2:]))
                tr, va = project_moments(train, proj), project_moments(val, proj)
            for label, real, shuf in (('image', [0, 1], [4, 5]), ('grid', [2, 3], [6, 7])):
                path = ridge_path(tr, va, real + shuf)
                mse = path['val_mse'][:, :2].mean(1)
                best = int(torch.argmax(-mse))
                w, b = path['w'][best], path['b'][best]
                if probe != 'full':
                    w = proj @ w
                cols.append((w, b))
                names += [f'{probe}_{label}_x', f'{probe}_{label}_y', f'{probe}_{label}_shuf_x', f'{probe}_{label}_shuf_y']
                info[f'{probe}_{label}'] = dict(lambda_=float(LAMBDAS[best]), at_boundary=best in (0, len(LAMBDAS) - 1),
                                                val_mse=path['val_mse'][:, :2].mean(1).tolist(),
                                                eig_min=path['eig_min'], eig_max=path['eig_max'])
        if stage in meta['lc_stages']:
            groups = [load_moments(out / 'stats' / 'lc' / stage / f'group_{g:02d}.pt', device) for g in range(meta['lc_groups'])]
            for o, order in enumerate(lc_orders):
                for size in sizes:
                    if size == sizes[-1] and o:
                        continue                  # the full training set is identical for every order
                    subset = sum_moments([groups[g] for g in order[:size // LC_GROUP_IMAGES]])
                    for probe in ('full', 'rp256'):
                        if probe == 'full':
                            tr, va = subset, val
                        else:
                            proj = projection(dim, 256)
                            tr, va = project_moments(subset, proj), project_moments(val, proj)
                        path = ridge_path(tr, va, [0, 1])
                        best = int(torch.argmax(-path['val_mse'].mean(1)))
                        w = path['w'][best] if probe == 'full' else proj @ path['w'][best]
                        cols.append((w, path['b'][best]))
                        names += [f'lc_{probe}_n{size}_o{o}_x', f'lc_{probe}_n{size}_o{o}_y']
                        info[f'lc_{probe}_n{size}_o{o}'] = dict(lambda_=float(LAMBDAS[best]),
                                                                at_boundary=best in (0, len(LAMBDAS) - 1))
            del groups
        w = torch.cat([c[0] for c in cols], 1).float().cpu()
        b = torch.cat([c[1] for c in cols], 0).float().cpu()
        torch.save(dict(w=w, b=b, columns=names, summary=info), target)
        summary[stage] = info
        print(f'Fitted {stage}: full image lambda={info["full_image"]["lambda_"]:.3g}', flush=True)
        del train, val
        torch.cuda.empty_cache()
    (out / 'ridge_summary.json').write_text(json.dumps(summary, indent=1) + '\n')


# ----------------------------------------------------------------------------- extraction

def extract(args: argparse.Namespace) -> None:
    import torch
    from analysis_tools.common import qwen_runtime as qr
    out = args.output
    cases = json.loads((out / 'cases.json').read_text())
    natural = cases['natural']
    rt = qr.load(args.model, device='cuda:0')
    # Prefill of the inner model never uses the LM head; release it to leave GPU headroom.
    rt.model.lm_head = None
    torch.cuda.empty_cache()
    torch.set_num_threads(8)
    visual = rt.model.model.visual
    if visual.config.spatial_merge_size != 2:
        raise ValueError('This experiment requires the Qwen 2x2 spatial merger.')
    stages = stage_names(len(visual.blocks), [l.block_type for l in rt.layers])
    feature_stages = [s for s in stages if s in args.feature_stages or (decoder_layer(s) in args.feature_decoder_layers)]
    lc_stages = [s for s in stages if s == 'merger' or decoder_layer(s) in args.lc_decoder_layers]
    if len(feature_stages) != len(args.feature_stages) + len(args.feature_decoder_layers):
        raise ValueError(f'Unknown feature stage requested: {args.feature_stages} {args.feature_decoder_layers}')
    dev = torch.device('cuda:0')
    ctx: dict = {}

    def handle(stage: str, value) -> None:
        vision = is_vision(stage)
        value = value.detach()
        if ctx['mode'] == 'trainval':
            ctx['acc'][stage].add(value.double(), ctx['y_vision' if vision else 'y_merged'])
        else:
            w, b = ctx['ridge'][stage]
            ctx['pred'][stage] = (value.float() @ w + b).cpu().numpy()
        if stage in feature_stages:
            if ctx['store_full']:
                ctx['feat'][stage] = value.to(torch.float16).cpu().numpy()
            else:
                ids = ctx['pid' if vision else 'mid']
                ctx['feat'][stage] = value[ids].to(torch.float16).cpu().numpy()

    def make_hook(stage: str, use_input: bool = False, decoder: bool = False):
        def hook(module, inputs, output):
            value = inputs[0] if use_input else output
            if not isinstance(value, torch.Tensor):
                value = value[0]
            if decoder:
                value = value[0, ctx['start']:ctx['end']]
            handle(stage, value)
        return hook

    handles = [visual.patch_embed.register_forward_hook(make_hook('patch_content')),
               visual.blocks[0].register_forward_hook(make_hook('vit_00_position', use_input=True))]
    handles += [blk.register_forward_hook(make_hook(f'vit_{i + 1:02d}')) for i, blk in enumerate(visual.blocks)]
    handles.append(visual.merger.register_forward_hook(make_hook('merger')))
    handles += [layer.register_forward_hook(make_hook(stages[30 + i], decoder=True)) for i, layer in enumerate(rt.layers)]
    if stages[30] != 'decoder_01_' + stages[30].split('_', 2)[2] or len(stages) != 30 + len(rt.layers):
        raise ValueError('Unexpected stage layout.')

    def run(image: Image.Image, patch_hw: tuple[int, int]) -> None:
        batch = qr.build(rt, [image], PROMPT)
        span = qr.image_spans(rt, batch)[0]
        if tuple(batch['image_grid_thw'][0, 1:].tolist()) != tuple(patch_hw):
            raise ValueError('Processor patch grid differs from the recorded grid.')
        ctx.update(start=span['start'], end=span['end'], pred={}, feat={})
        with torch.inference_mode():
            rt.model.model(**batch, use_cache=False)
        expected = set(stages) if ctx['mode'] != 'trainval' else set()
        if expected and set(ctx['pred']) != expected:
            raise RuntimeError('Not every stage produced predictions.')

    started = datetime.now(timezone.utc).isoformat()
    if args.phase == 'trainval':
        train = [c for c in natural if c['split'] == 'train']
        val = [c for c in natural if c['split'] == 'validation']
        if len(train) % LC_GROUP_IMAGES:
            raise ValueError('Training images must form whole learning-curve groups.')
        rng = np.random.default_rng(SHUFFLE_SEED)
        shuffled = {}
        for family, vision in (('vision', True), ('merged', False)):
            labs = [np.concatenate(token_labels(tuple(c['patch_hw']), vision)[::-1], 1) for c in train]
            pooled = np.concatenate(labs)
            perm = rng.permutation(len(pooled))
            offsets = np.cumsum([0] + [len(l) for l in labs])
            shuffled[family] = [pooled[perm[offsets[i]:offsets[i + 1]]] for i in range(len(train))]
        dims = {s: (visual.config.hidden_size if is_vision(s) else rt.model.config.text_config.hidden_size) for s in stages}
        acc = {s: ShiftedMoments(dims[s], len(LABEL_COLUMNS), dev) for s in stages}
        # Learning-curve stages accumulate each 75-image group separately (CPU float64), then merge.
        train_total = {s: Moments(dims[s], len(LABEL_COLUMNS), 'cpu') for s in lc_stages}
        (out / 'stats' / 'train').mkdir(parents=True, exist_ok=True)
        (out / 'stats' / 'validation').mkdir(parents=True, exist_ok=True)
        for s in lc_stages:
            (out / 'stats' / 'lc' / s).mkdir(parents=True, exist_ok=True)
        (out / 'features').mkdir(exist_ok=True)
        feats = {s: np.lib.format.open_memmap(out / 'features' / f'natural_{s}.npy', mode='w+', dtype=np.float16,
                                              shape=(len(natural), TOKENS_PER_IMAGE, dims[s]))
                 for s in feature_stages}
        for split, rows in (('train', train), ('validation', val)):
            for k, c in enumerate(rows):
                hw = tuple(c['patch_hw'])
                ys = {}
                for family, vision in (('vision', True), ('merged', False)):
                    grid, img = token_labels(hw, vision)
                    sh = shuffled[family][k] if split == 'train' else np.concatenate([img, grid], 1)
                    ys[family] = torch.as_tensor(np.concatenate([img, grid, sh], 1), dtype=torch.float64, device=dev)
                mid, pid = sample_tokens(hw, c['id'])
                ctx.update(mode='trainval', y_vision=ys['vision'], y_merged=ys['merged'], store_full=False,
                           mid=torch.as_tensor(mid, device=dev), pid=torch.as_tensor(pid, device=dev),
                           acc=acc)
                run(Image.open(out / 'images/natural512' / f'{c["id"]:04d}.png').convert('RGB'), hw)
                for s in feature_stages:
                    feats[s][c['id']] = ctx['feat'][s]
                if (k + 1) % args.flush_every == 0 or (split == 'train' and (k + 1) % LC_GROUP_IMAGES == 0):
                    for s in stages:
                        acc[s].flush()
                if split == 'train' and (k + 1) % LC_GROUP_IMAGES == 0:
                    g = k // LC_GROUP_IMAGES
                    for s in lc_stages:
                        torch.save(acc[s].total.state(), out / 'stats' / 'lc' / s / f'group_{g:02d}.pt')
                        train_total[s].merge(acc[s].total)
                        acc[s].total.zero()
                if (k + 1) % 100 == 0:
                    print(f'{split}: {k + 1}/{len(rows)}', flush=True)
            for s in stages:
                acc[s].flush()
                if split == 'train' and s in lc_stages:
                    if float(acc[s].total.n):
                        raise RuntimeError('Learning-curve statistics left outside a group.')
                    torch.save(train_total[s].state(), out / 'stats' / split / f'{s}.pt')
                else:
                    torch.save(acc[s].total.state(), out / 'stats' / split / f'{s}.pt')
                acc[s].total.zero()
        for m in feats.values():
            m.flush()
        meta = dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(), **qr.describe(rt),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), thinking='off', prompt=PROMPT,
                    stages=stages, feature_stages=feature_stages, lc_stages=lc_stages,
                    lc_groups=len(train) // LC_GROUP_IMAGES, lc_group_images=LC_GROUP_IMAGES, label_columns=LABEL_COLUMNS,
                    tokens_per_image=TOKENS_PER_IMAGE, sample_seed=SAMPLE_SEED, shuffle_seed=SHUFFLE_SEED,
                    split_seed=SPLIT_SEED, statistics_dtype='float64 (float32 shifted GPU partial sums)',
                    flush_every=args.flush_every, feature_dtype='float16', batch_size=1,
                    feature_dims={s: dims[s] for s in feature_stages})
        (out / 'extract_trainval.json').write_text(json.dumps(meta, indent=1) + '\n')
        return

    # Test phase: apply ridge maps on the fly; store features for MLP / relative probes.
    meta = json.loads((out / 'extract_trainval.json').read_text())
    if meta['stages'] != stages or meta['feature_stages'] != feature_stages:
        raise ValueError('Stage layout differs from the train/validation extraction.')
    ridge, columns = {}, None
    for s in stages:
        blob = torch.load(out / 'ridge' / f'{s}.pt', map_location='cpu')
        ridge[s] = (blob['w'].to(dev), blob['b'].to(dev))
        columns = columns or {}
        columns[s] = blob['columns']
    (out / 'predictions').mkdir(exist_ok=True)
    test = [c for c in natural if c['split'] == 'test']
    conditions = [('test512', test, 'images/natural512', 'patch_hw'), ('test768', test, 'images/natural768', 'patch_hw_768'),
                  ('synthetic', cases['synthetic'], 'images/synthetic', 'patch_hw')]
    nat_feats = {s: np.load(out / 'features' / f'natural_{s}.npy', mmap_mode='r+') for s in feature_stages}
    for name, rows, folder, key in conditions:
        done = out / 'predictions' / f'{name}.done'
        if done.exists():
            continue
        preds = {s: [] for s in stages}
        full = {}
        if name == 'synthetic':
            counts = {fam: np.cumsum([0] + [c[key][0] * c[key][1] // (1 if fam else 4) for c in rows]) for fam in (True, False)}
            full = {s: np.lib.format.open_memmap(out / 'features' / f'synthetic_{s}.npy', mode='w+', dtype=np.float16,
                                                 shape=(int(counts[is_vision(s)][-1]), meta['feature_dims'][s]))
                    for s in feature_stages}
        sampled = {} if name == 'synthetic' else {
            s: np.zeros((len(rows), TOKENS_PER_IMAGE, meta['feature_dims'][s]), np.float16) for s in feature_stages}
        for k, c in enumerate(rows):
            hw = tuple(c[key])
            mid, pid = sample_tokens(hw, c['id'] + (100000 if name == 'synthetic' else 0))
            ctx.update(mode='test', ridge=ridge, store_full=name == 'synthetic',
                       mid=torch.as_tensor(mid, device=dev), pid=torch.as_tensor(pid, device=dev))
            run(Image.open(out / folder / f'{c["id"]:04d}.png').convert('RGB'), hw)
            for s in stages:
                preds[s].append(ctx['pred'][s].astype(np.float32))
            for s in feature_stages:
                if name == 'synthetic':
                    off = counts[is_vision(s)]
                    if len(ctx['feat'][s]) != off[k + 1] - off[k]:
                        raise RuntimeError(f'Unexpected token count at {s}.')
                    full[s][off[k]:off[k + 1]] = ctx['feat'][s]
                else:
                    sampled[s][k] = ctx['feat'][s]
            if (k + 1) % 50 == 0:
                print(f'{name}: {k + 1}/{len(rows)}', flush=True)
        np.savez(out / 'predictions' / f'{name}.npz', **{s: np.concatenate(v) for s, v in preds.items()},
                 **{f'offsets_{fam}': np.cumsum([0] + [len(v) for v in preds[st]])
                    for fam, st in (('vision', 'patch_content'), ('merged', 'merger'))})
        for s in feature_stages:
            if name == 'test512':
                for k, c in enumerate(rows):
                    nat_feats[s][c['id']] = sampled[s][k]
                nat_feats[s].flush()
            elif name == 'test768':
                np.save(out / 'features' / f'test768_{s}.npy', sampled[s])
            else:
                full[s].flush()
        done.write_text(datetime.now(timezone.utc).isoformat() + '\n')
    (out / 'predictions' / 'columns.json').write_text(json.dumps(columns, indent=0) + '\n')
    (out / 'extract_test.json').write_text(json.dumps(dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                                                           **qr.describe(rt), batch_size=1,
                                                           code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=1) + '\n')
    for h in handles:
        h.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--refcoco', type=Path, required=True)
    p.add_argument('--previous-cases', type=Path)
    p.add_argument('--split', type=int, nargs=3, default=[2400, 300, 300])
    q = sub.add_parser('prepare-synthetic')
    q.add_argument('--output', type=Path, required=True)
    e = sub.add_parser('extract')
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--model', default='Qwen3.5-9B')
    e.add_argument('--phase', choices=['trainval', 'test'], required=True)
    e.add_argument('--feature-stages', nargs='+', default=['patch_content', 'vit_00_position', 'vit_09', 'vit_18', 'vit_27', 'merger'])
    e.add_argument('--feature-decoder-layers', type=int, nargs='+', default=[3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 20, 24, 28, 32])
    e.add_argument('--lc-decoder-layers', type=int, nargs='+', default=[8, 14, 24, 32])
    e.add_argument('--flush-every', type=int, default=25, help='images between float32 GPU -> float64 CPU flushes')
    f = sub.add_parser('fit-ridge')
    f.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'prepare-synthetic':
        prepare_synthetic(args.output)
    elif args.command == 'extract':
        extract(args)
    else:
        fit_ridge(args)


if __name__ == '__main__':
    main()
