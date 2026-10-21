"""Stage drivers of experiment 2.2 (high-norm tokens). GPU stages only; analysis is in high_norm_analysis.py.

Stages (selected with --stages; artifacts under <output>/<tag>/):
  neurons    register neurons of the high-norm-creating block: MLP activations, attention vs MLP update norms,
             massive channels (blank, synthetic, natural images).
  survey     per-block norm/median maps of every ViT block, merger and decoder layer on blank, noise,
             random-placement (probe set), grid, blue-box-marked and natural images; probe features of
             high-norm, canonical-position and random tokens at selected blocks.
  source     ViT position-source interventions (PE zero / shuffle / flip / outlier removal / phase window,
             RoPE zero / shuffle, content placed on canonical positions, canvas aspect) and in-network
             ablations; norm maps after each intervention.
  attention  attention mass received by high-norm tokens in every ViT block; decoder full-attention mass
             on prefix, high-norm-derived and other visual tokens (Qwen3.5 only).
  posvar     split-half position-explained variance R^2_pos at ViT, merger and decoder sites, and the 2.1
             single-image distance readout, under base / high-norm ablations / zero PE.
  ablation   task effects of high-norm ablations: disc coordinates (3.5), T2, T3 (revised), T4 row/column
             and point (direct answer), T4 row/column with free-form reasoning, RefCOCO point and bbox.
  spot       answer-level agreement of batched generation with single-sample generation (base condition).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.behavior import decomposition as dec
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.common.stimuli import DEVELOPMENT_TEMPLATES, Change, StimulusSpec, generate, render_instance
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import vit_layout as vl
from analysis_tools.position.coordinate_readout_run import append_jsonl, done_ids, read_jsonl
from analysis_tools.position.coordinate_readout_supp import score_refcoco
from analysis_tools.representation import high_norm as hn
from analysis_tools.representation.visual_detail import figure_image

PROBE_SITES = (6, 12, 20, 26)            # ViT blocks whose token features are stored for probes
DECODER_SITES = (1, 4, 8, 16, 24, 32)
POSVAR_VIT = (3, 9, 10, 12, 18, 26, 27)


def code_digest() -> dict:
    here = Path(__file__).parent
    return {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in ('high_norm.py', 'high_norm_run.py')}


def write_meta(out: Path, rt, args, stage: str, extra: dict | None = None) -> None:
    meta = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **qr.describe(rt), 'code': code_digest(),
            'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, **(extra or {})}
    (out / f'{stage}_meta.json').write_text(json.dumps(meta, indent=1) + '\n')


# ----------------------------------------------------------------------------- image sources

def blank(color: str, w: int, h: int | None = None, seed: int = 0) -> Image.Image:
    h = w if h is None else h
    if color == 'noise':
        rng = np.random.default_rng(seed)
        return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    fill = {'white': (255, 255, 255), 'gray': (128, 128, 128), 'black': (0, 0, 0)}[color]
    return Image.new('RGB', (w, h), fill)


def probe_case(i: int, seed: int, canvas: int) -> dict:
    """Random-placement image: random count (2-16), template, offset and optional anomalous instance."""
    rng = np.random.default_rng(seed * 7919 + i * 13 + canvas)
    off = int(rng.choice([0, 16, 32]))
    slots = (canvas - off) // 64
    count = int(rng.integers(2, 17))
    cells = rng.choice(slots * slots, size=count, replace=False).tolist()
    template = str(rng.choice(DEVELOPMENT_TEMPLATES))
    anomaly = bool(rng.integers(0, 2))
    return {'id': f'p{canvas}_{i:04d}', 'canvas': canvas, 'offset': off, 'count': count, 'cells': cells, 'slots': slots,
            'template': template, 'anomaly': anomaly, 'anomaly_index': int(rng.integers(0, count)) if anomaly else -1}


def probe_render(case: dict) -> Image.Image:
    n, off, slots = case['canvas'], case['offset'], case['slots']
    img = np.full((n, n, 3), 255, np.uint8)
    normal = render_instance(case['template'], cell=64)
    odd = render_instance(case['template'], Change('color', 4), cell=64)
    for j, cell in enumerate(case['cells']):
        r, c = divmod(cell, slots)
        tile = odd if j == case['anomaly_index'] else normal
        y, x = off + r * 64, off + c * 64
        img[y:y + 64, x:x + 64] = np.minimum(img[y:y + 64, x:x + 64], tile)
    return Image.fromarray(img)


def natural_cases(args) -> list[dict]:
    """Unique RefCOCO test images (512-pixel copies of experiment 3.4) with the 3.5 RefCOCO cases."""
    cases = json.loads((args.refcoco_cases / 'cases.json').read_text())
    for c in cases:
        p = Path(c['image_path'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != c['image_sha256']:
            raise ValueError(f'Image hash mismatch for {p}')
    return cases


def natural_images(args) -> list[tuple[str, Path]]:
    seen, out = set(), []
    for c in natural_cases(args):
        if c['image_key'] not in seen:
            seen.add(c['image_key'])
            out.append((c['image_key'], Path(c['image_path'])))
    return out


def decomposition_root(args) -> Path:
    return args.decomposition


def load_case_image(args, case: dict) -> Image.Image:
    p = decomposition_root(args) / case['image']
    if hashlib.sha256(p.read_bytes()).hexdigest() != case['image_sha256']:
        raise ValueError(f'Image hash mismatch for {p}')
    return Image.open(p).convert('RGB')


def marked_pairs(args, n: int, seed: int) -> list[dict]:
    """Blue-box-marked normal grids of T3 and their re-rendered unmarked twins (same spec, marker=None)."""
    specs = {r['path']: r for r in read_jsonl(decomposition_root(args) / 'images.jsonl')}
    marked = [p for p in specs if p.startswith('images/marked/') and p.endswith('_normal.png')]
    rng = np.random.default_rng(seed)
    out = []
    by_grid = defaultdict(list)
    for p in marked:
        by_grid[specs[p]['rows']].append(p)
    for g in sorted(by_grid):
        pick = rng.choice(len(by_grid[g]), size=min(n // len(by_grid), len(by_grid[g])), replace=False)
        for k in pick:
            p = by_grid[g][int(k)]
            s = dict(specs[p]['spec'])
            s['change'] = Change(**s['change'])
            for key in ('target', 'marker'):
                if s.get(key) is not None:
                    s[key] = tuple(s[key])
            out.append({'path': p, 'spec': s, 'marker': s['marker'], 'rows': specs[p]['rows'],
                        'grid_origin': specs[p]['grid_origin'], 'cell': specs[p]['cell']})
    return out


def marked_render(args, pair: dict) -> tuple[Image.Image, Image.Image]:
    marked = Image.open(decomposition_root(args) / pair['path']).convert('RGB')
    plain, _ = generate(StimulusSpec(**{**pair['spec'], 'marker': None}))
    remark, _ = generate(StimulusSpec(**pair['spec']))
    if not np.array_equal(np.asarray(remark), np.asarray(marked)):
        raise ValueError(f"Re-rendered marked image differs from {pair['path']}")
    return marked, plain


# ----------------------------------------------------------------------------- helpers

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def canonical_positions(rt, hp: int, wp: int, cache: dict, block: int = 12) -> torch.Tensor:
    """Block-order indices of high-norm patches on a white canvas of the same patch grid."""
    key = (hp, wp)
    if key not in cache:
        rec = hn.vit_forward_record(rt, [blank('white', wp * 16, hp * 16)], [], keep=[block])[0]
        mask = hn.hn_mask(rec[f'feat_{block}'], [hp * wp])
        cache[key] = mask.nonzero().flatten().cpu()
    return cache[key]


def register_neurons(out_root: Path) -> list[int]:
    path = out_root / 'neurons' / 'neurons.json'
    if not path.exists():
        raise FileNotFoundError('Run the neurons stage first.')
    return json.loads(path.read_text())['register_neurons']


# ----------------------------------------------------------------------------- stage: neurons

def stage_neurons(rt, out: Path, args) -> None:
    root = out / 'neurons'
    root.mkdir(parents=True, exist_ok=True)
    kb = hn.REGISTER_BLOCK
    images = [(f'{c}{s}', blank(c, s)) for c in ('white', 'gray', 'black') for s in (384, 512, 768)]
    images += [(f'noise512_{k}', blank('noise', 512, seed=k)) for k in range(4)]
    images += [(c['id'], probe_render(c)) for c in [probe_case(i, args.seed, 512) for i in range(16)]]
    images += [(k, Image.open(p).convert('RGB')) for k, p in natural_images(args)[:16]]
    acts_hn, acts_nm, n_hn_total = [], [], 0
    rows = []
    for name, im in images:
        rec = hn.vit_forward_record(rt, [im], [], keep=[kb - 1, kb], mlp_blocks=[kb])[0]
        x = rec[f'feat_{kb}']
        mask = hn.hn_mask(x, [len(x)])
        act = rec[f'act_{kb}']
        acts_hn.append(act[mask].cpu())
        acts_nm.append(act[~mask][torch.randperm(int((~mask).sum()))[:256]].cpu())
        n_hn_total += int(mask.sum())
        mlp, att, xin = rec[f'mlp_{kb}'], rec[f'attn_{kb}'], rec[f'feat_{kb - 1}']
        rows.append({'image': name, 'grid': rec['grid'], 'n_hn': int(mask.sum()),
                     'hn': {'in': float(xin[mask].norm(dim=1).mean()) if mask.any() else None,
                            'attn': float(att[mask].norm(dim=1).mean()) if mask.any() else None,
                            'mlp': float(mlp[mask].norm(dim=1).mean()) if mask.any() else None,
                            'out': float(x[mask].norm(dim=1).mean()) if mask.any() else None},
                     'normal': {'in': float(xin[~mask].norm(dim=1).mean()), 'attn': float(att[~mask].norm(dim=1).mean()),
                                'mlp': float(mlp[~mask].norm(dim=1).mean()), 'out': float(x[~mask].norm(dim=1).mean())}})
    A, B = torch.cat(acts_hn), torch.cat(acts_nm)
    mean_hn, mean_nm = A.mean(0), B.mean(0)
    frac_on = (A > 5).float().mean(0)
    order = torch.argsort(mean_hn - mean_nm, descending=True)
    # Register neurons: large on high-norm tokens, essentially silent elsewhere.
    reg = [int(i) for i in order[:64] if mean_hn[i] > 10 and mean_nm[i].abs() < 0.1]
    w2 = rt.model.model.visual.blocks[kb - 1].mlp.linear_fc2.weight.float().cpu()
    contrib = (A[:, reg] @ w2[:, reg].T).norm(dim=1).mean() if reg else torch.tensor(0.)
    # Massive channels of the high-norm tokens at the block output: top-10 |value| channels per token.
    stats = {'register_neurons': reg, 'n_hn_tokens': n_hn_total, 'block': kb,
             'top_neurons': [{'neuron': int(i), 'mean_hn': float(mean_hn[i]), 'mean_normal': float(mean_nm[i]),
                              'frac_hn_above5': float(frac_on[i])} for i in order[:30]],
             'register_contrib_norm': float(contrib),
             'fc2_out_channels': {int(i): torch.topk(w2[:, i].abs(), 5).indices.tolist() for i in reg},
             'images': rows}
    np.save(root / 'act_hn_top64.npy', A[:, order[:64]].numpy().astype(np.float16))
    np.save(root / 'act_normal_top64.npy', B[:, order[:64]].numpy().astype(np.float16))
    np.save(root / 'neuron_order.npy', order[:64].numpy())
    np.save(root / 'mean_act.npy', torch.stack([mean_hn, mean_nm]).numpy())
    (root / 'neurons.json').write_text(json.dumps(stats, indent=1))
    write_meta(root, rt, args, 'neurons')
    print('register neurons', reg, 'n_hn', n_hn_total, flush=True)


# ----------------------------------------------------------------------------- stage: survey

def survey_sets(args) -> dict[str, list[tuple[str, callable, dict]]]:
    """name -> list of (image id, loader, labels)."""
    sets = {}
    sets['blank'] = [(f'{c}{s}', (lambda c=c, s=s: blank(c, s)), {'color': c, 'canvas': s})
                     for c in ('white', 'gray', 'black') for s in (256, 384, 512, 640, 768, 1024)]
    sets['blank'] += [(f'white{w}x{h}', (lambda w=w, h=h: blank('white', w, h)), {'color': 'white', 'canvas': f'{w}x{h}'})
                      for w, h in ((768, 512), (512, 768), (1024, 512))]
    sets['noise'] = [(f'noise{s}_{k}', (lambda s=s, k=k: blank('noise', s, seed=k)), {'canvas': s, 'seed': k})
                     for s in (512, 768) for k in range(8)]
    probe = []
    for canvas in args.probe_canvases:
        for i in range(args.probe_n):
            c = probe_case(i, args.seed, canvas)
            probe.append((c['id'], (lambda c=c: probe_render(c)), c))
    sets['probe'] = probe
    cases = [c for c in read_jsonl(decomposition_root(args) / 'images.jsonl') if c['role'] == 'grid']
    rng = np.random.default_rng(args.seed)
    by_g = defaultdict(list)
    for c in cases:
        by_g[c['rows']].append(c)
    grid = []
    for g in sorted(by_g):
        for k in rng.choice(len(by_g[g]), size=min(args.grid_n, len(by_g[g])), replace=False):
            c = by_g[g][int(k)]
            grid.append((c['image_id'], (lambda c=c: load_case_image(args, {'image': c['path'], 'image_sha256': c['sha256']})),
                         {'rows': c['rows'], 'canvas': c['canvas'], 'target_rc': c['target_rc'], 'grid_origin': c['grid_origin']}))
    sets['grid'] = grid
    sets['natural'] = [(k, (lambda p=p: Image.open(p).convert('RGB')), {}) for k, p in natural_images(args)[:args.natural_n]]
    return sets


def stage_survey(rt, out: Path, args) -> None:
    root = out / 'survey'
    root.mkdir(parents=True, exist_ok=True)
    nb = len(rt.model.model.visual.blocks)
    blocks = list(range(1, nb + 1))
    canon_cache = {}
    sets = survey_sets(args)
    marked = marked_pairs(args, args.marked_n, args.seed)
    for p in marked:
        sets.setdefault('marked', []).append((p['path'], None, p))
    for name, items in sets.items():
        path = root / f'{name}.pt'
        if path.exists():
            continue
        t0 = time.time()
        recs = []
        for k, (iid, loader, labels) in enumerate(items):
            if name == 'marked':
                pair_imgs = marked_render(args, labels)
                variants = [('marked', pair_imgs[0]), ('plain', pair_imgs[1])]
            else:
                variants = [('', loader())]
            for vname, im in variants:
                rec = hn.vit_forward_record(rt, [im], blocks, keep=list(PROBE_SITES))[0]
                hp, wp = rec['grid']
                r = {'id': iid + (f'|{vname}' if vname else ''), 'labels': labels, 'grid': (hp, wp),
                     'ratio': rec['ratio'], 'median': rec['median'],
                     'merger_ratio': (lambda n: (n / n.median()).view(hp // 2, wp // 2).half().cpu().numpy())(
                         rec['merger'].norm(dim=1))}
                # massive channels of high-norm tokens at block 12
                f12 = rec['feat_12']
                m12 = hn.hn_mask(f12, [hp * wp])
                if m12.any():
                    r['hn_top_channels'] = torch.topk(f12[m12].abs(), 10, dim=1).indices.cpu().numpy().astype(np.int16)
                if name in ('probe', 'natural', 'marked'):
                    canon = canonical_positions(rt, hp, wp, canon_cache)
                    g = torch.Generator().manual_seed(args.seed + k)
                    nonhn = (~m12).nonzero().flatten().cpu()
                    rand = nonhn[torch.randperm(len(nonhn), generator=g)[:args.n_random]]
                    hn_idx = m12.nonzero().flatten().cpu()
                    if len(hn_idx) > args.max_hn:
                        hn_idx = hn_idx[torch.topk(f12[hn_idx.to(f12.device)].norm(dim=1), args.max_hn).indices.cpu()]
                    sel = torch.cat([hn_idx, canon, rand])
                    kinds = np.array([0] * len(hn_idx) + [1] * len(canon) + [2] * len(rand), np.int8)
                    content = hn.patch_content(im, hp, wp)                         # raster
                    rr = vl.to_raster(torch.arange(hp * wp)[:, None], hp, wp)[:, 0]   # raster -> block index
                    block_to_raster = torch.empty_like(rr)
                    block_to_raster[rr] = torch.arange(hp * wp)
                    ras = block_to_raster[sel].numpy()
                    r['probe'] = {'sel_block': sel.numpy(), 'kind': kinds, 'raster': ras,
                                  'row': ras // wp, 'col': ras % wp, 'content': content[ras].astype(np.float16)}
                    for s in PROBE_SITES:
                        f = rec[f'feat_{s}']
                        ms = hn.hn_mask(f, [hp * wp])
                        r['probe'][f'feat_{s}'] = f[sel.to(f.device)].half().cpu().numpy()
                        r['probe'][f'is_hn_{s}'] = ms[sel.to(ms.device)].cpu().numpy()
                        r['probe'][f'pool_{s}'] = f[~ms].mean(0).half().cpu().numpy()
                    # merged tokens containing the selected patches (raster merged index)
                    mi = (ras // wp // 2) * (wp // 2) + (ras % wp) // 2
                    r['probe']['feat_merger'] = rec['merger'][torch.from_numpy(mi).to(rec['merger'].device)].half().cpu().numpy()
                    r['probe']['pool_merger'] = rec['merger'].mean(0).half().cpu().numpy()
                if name in ('blank', 'natural', 'marked') or \
                        (name == 'probe' and int(iid.split('_')[1]) < args.decoder_n // len(args.probe_canvases)) or \
                        (name == 'grid' and k % max(1, len(items) // args.decoder_n) == 0):
                    r['decoder'] = hn.decoder_norms(rt, im)
                recs.append(r)
            if k % 50 == 0:
                print(f'survey {name} {k}/{len(items)} {time.time() - t0:.0f}s', flush=True)
        torch.save(recs, path)
        print(f'survey {name} done {len(recs)} records {time.time() - t0:.0f}s', flush=True)
    write_meta(root, rt, args, 'survey', {'probe_sites': PROBE_SITES})


# ----------------------------------------------------------------------------- stage: source

def source_conditions(rt, out_root: Path) -> dict:
    """name -> (kind, payload). kind: 'edit' (vit_layout.ViTEdit), 'pe' (pe_override kwargs), 'ablation'."""
    E = vl.ViTEdit
    conds = {'base': ('edit', E()), 'pe_zero': ('edit', E(pe='zero')), 'pe_mean': ('edit', E(pe='mean')),
             'pe_shuffle0': ('edit', E(pe='shuffle', seed=0)), 'pe_shuffle1': ('edit', E(pe='shuffle', seed=1)),
             'rope_zero': ('edit', E(rope='zero')), 'rope_shuffle': ('edit', E(rope='shuffle')),
             'pe_x0.5': ('edit', E(pe_scale=0.5)), 'pe_x2': ('edit', E(pe_scale=2.0)),
             'pe_hflip': ('pe', {'pe_fn': hn.pe_flip('h')}), 'pe_vflip': ('pe', {'pe_fn': hn.pe_flip('v')}),
             'pe_outlier_shrink': ('pe', {'table_fn': hn.table_outlier_fn(3.0, 'shrink')})}
    neurons = tuple(register_neurons(out_root))
    for mode in ('hn_mean', 'hn_zero', 'rand_mean'):
        conds[f'ab_{mode}'] = ('ablation', hn.Ablation(mode))
    conds['ab_neuron_zero'] = ('ablation', hn.Ablation('neuron_zero', neurons=neurons))
    return conds


def run_condition(rt, cond, fn):
    kind, payload = cond
    if kind == 'edit':
        with vl.vit_edit(rt, payload) if not payload.is_default else _null():
            return fn()
    if kind == 'pe':
        with hn.pe_override(rt, **payload):
            return fn()
    with hn.vit_ablation(rt, payload):
        return fn()


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def stage_source(rt, out: Path, args) -> None:
    root = out / 'source'
    root.mkdir(parents=True, exist_ok=True)
    nb = len(rt.model.model.visual.blocks)
    blocks = list(range(1, nb + 1))
    images = [('white512', blank('white', 512)), ('white768', blank('white', 768)), ('gray512', blank('gray', 512))]
    images += [(f'noise512_{k}', blank('noise', 512, seed=100 + k)) for k in range(4)]
    images += [(c['id'], probe_render(c)) for c in [probe_case(i, args.seed + 1, 512) for i in range(args.source_n)]]
    images += [(k, Image.open(p).convert('RGB')) for k, p in natural_images(args)[-args.source_n:]]
    conds = source_conditions(rt, out)
    path = root / 'records.pt'
    recs = torch.load(path, weights_only=False) if path.exists() else {}
    for cname, cond in conds.items():
        if cname in recs:
            continue
        rows = []
        for iid, im in images:
            rec = run_condition(rt, cond, lambda: hn.vit_forward_record(rt, [im], blocks))[0]
            rows.append({'id': iid, 'grid': rec['grid'], 'ratio': rec['ratio'][[8, 9, 11, 17, 25, nb - 1]],
                         'blocks': [9, 10, 12, 18, 26, nb], 'n_hn': (rec['ratio'] > hn.TAU).reshape(nb, -1).sum(1),
                         'median': rec['median']})
        recs[cname] = rows
        torch.save(recs, path)
        print('source', cname, flush=True)
    # PE phase window: a 32x32 window of a virtual 40x40 canvas at several offsets (white content).
    win = {}
    for r0, c0 in ((0, 0), (2, 2), (4, 4), (8, 8), (0, 8), (8, 0)):
        e = vl.ViTEdit(pe='window', window_vh=40, window_vw=40, window_r0=r0, window_c0=c0)
        with vl.vit_edit(rt, e):
            rec = hn.vit_forward_record(rt, [blank('white', 512)], [12])[0]
        win[f'{r0}_{c0}'] = rec['ratio'][0]
    full = hn.vit_forward_record(rt, [blank('white', 640)], [12])[0]['ratio'][0]
    # Content on canonical positions: a 64x64 noise tile centred on each canonical high-norm patch (white 512).
    base = hn.vit_forward_record(rt, [blank('white', 512)], [], keep=[12])[0]
    canon = hn.hn_mask(base['feat_12'], [1024]).nonzero().flatten().cpu()
    rr = vl.to_raster(torch.arange(1024)[:, None], 32, 32)[:, 0]
    b2r = torch.empty_like(rr)
    b2r[rr] = torch.arange(1024)
    content = []
    rng = np.random.default_rng(args.seed)
    tile = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    targets = [int(b2r[i]) for i in canon] + [5 * 32 + 5, 26 * 32 + 10, 12 * 32 + 20]
    for ras in targets:
        r, c = divmod(ras, 32)
        img = np.full((512, 512, 3), 255, np.uint8)
        y, x = min(max(r * 16 - 24, 0), 448), min(max(c * 16 - 24, 0), 448)
        img[y:y + 64, x:x + 64] = tile
        rec = hn.vit_forward_record(rt, [Image.fromarray(img)], [12])[0]
        content.append({'target_raster': ras, 'is_canonical': ras in [int(b2r[i]) for i in canon], 'box': [x, y, x + 64, y + 64],
                        'ratio': rec['ratio'][0]})
    torch.save({'window': win, 'full640': full, 'content': content, 'canonical_raster': [int(b2r[i]) for i in canon]},
               root / 'phase_content.pt')
    write_meta(root, rt, args, 'source', {'conditions': list(conds)})


# ----------------------------------------------------------------------------- stage: attention

def stage_attention(rt, out: Path, args) -> None:
    root = out / 'attention'
    root.mkdir(parents=True, exist_ok=True)
    nb = len(rt.model.model.visual.blocks)
    blocks = [3, 6, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 27][: nb]
    images = [('white512', blank('white', 512)), ('noise512_0', blank('noise', 512, seed=100))]
    images += [(c['id'], probe_render(c)) for c in [probe_case(i, args.seed + 2, 512) for i in range(args.attention_n)]]
    images += [(k, Image.open(p).convert('RGB')) for k, p in natural_images(args)[:args.attention_n]]
    vit_rows = []
    for iid, im in images:
        rec = hn.vit_forward_record(rt, [im], [], keep=[12])[0]
        hp, wp = rec['grid']
        m = hn.hn_mask(rec['feat_12'], [hp * wp]).cpu()
        g = torch.Generator().manual_seed(7)
        cand = (~m).nonzero().flatten()
        rnd = torch.zeros_like(m)
        rnd[cand[torch.randperm(len(cand), generator=g)[:max(int(m.sum()), 1)]]] = True
        res = hn.vit_attention_mass(rt, im, blocks, {'hn': m, 'rand': rnd})
        vit_rows.append({'id': iid, 'grid': (hp, wp), 'n_hn': int(m.sum()), 'hn_block': m.numpy(),
                         'blocks': blocks, 'res': {k: {kk: vv.numpy() if torch.is_tensor(vv) else vv for kk, vv in v.items()}
                                                   for k, v in res.items()}})
        print('attention vit', iid, flush=True)
    torch.save(vit_rows, root / 'vit.pt')
    if rt.family != 'qwen3_5':
        write_meta(root, rt, args, 'attention', {'decoder': 'skipped (Qwen3.5-only recompute)'})
        return
    # Decoder: full-attention rows of the last prompt token and of sampled visual tokens.
    dec_rows = []
    prompts = {'describe': 'Describe the image.', 'point': cr.DISC_QUERY + cr.SUFFIX['point']}
    for iid, im in images[2:]:
        for pname, text in prompts.items():
            if pname == 'point' and not iid.startswith('p'):
                continue
            prep = cr.prepare(rt, im, text)
            rec = hn.vit_forward_record(rt, [im], [], keep=[12])[0]
            hp, wp = rec['grid']
            m = hn.hn_mask(rec['feat_12'], [hp * wp]).cpu()
            ras = torch.empty(hp * wp, dtype=torch.long)
            ras[vl.to_raster(torch.arange(hp * wp)[:, None], hp, wp)[:, 0]] = torch.arange(hp * wp)
            hn_ras = ras[m]
            merged = sorted({int((r // wp // 2) * (wp // 2) + (r % wp) // 2) for r in hn_ras.tolist()})
            s, e = prep.span['start'], prep.span['end']
            L = prep.ids.shape[0]
            g = torch.Generator().manual_seed(11)
            vis_q = (s + torch.randperm(e - s, generator=g)[:16]).tolist()
            queries = [L - 1] + vis_q
            ro = cr.attention_readout(rt, cr.embeds(rt, prep), prep.pos, queries, (s, e))
            row = {'id': iid, 'prompt': pname, 'span': (s, e), 'L': L, 'merged_hn': merged, 'queries': queries,
                   'prefix_tokens': [rt.tokenizer.decode([t]) for t in prep.ids[:s].tolist()], 'check': ro.check, 'mass': {}}
            for layer, a in ro.attn.items():
                a = a.numpy()                                   # (heads, Q, L)
                vis = a[:, :, s:e]
                mh = np.zeros(e - s, bool)
                mh[merged] = True
                row['mass'][layer] = {'prefix': a[:, :, :s].sum(-1), 'first': a[:, :, 0], 'visual_all': vis.sum(-1),
                                      'visual_hn': vis[:, :, mh].sum(-1), 'visual_00': vis[:, :, 0],
                                      'suffix': a[:, :, e:].sum(-1), 'n_visual': e - s, 'n_hn': int(mh.sum())}
            dec_rows.append(row)
            print('attention decoder', iid, pname, flush=True)
    torch.save(dec_rows, root / 'decoder.pt')
    write_meta(root, rt, args, 'attention')


# ----------------------------------------------------------------------------- stage: posvar

def posvar_conditions(rt, out_root: Path) -> dict:
    neurons = tuple(register_neurons(out_root))
    return {'base': ('edit', vl.ViTEdit()), 'ab_hn_mean': ('ablation', hn.Ablation('hn_mean')),
            'ab_rand_mean': ('ablation', hn.Ablation('rand_mean')),
            'ab_neuron_zero': ('ablation', hn.Ablation('neuron_zero', neurons=neurons)),
            'pe_zero': ('edit', vl.ViTEdit(pe='zero'))}


@torch.inference_mode()
def posvar_forward(rt, im: Image.Image, vit_blocks, dec_layers):
    """ViT block features (block order), merger and decoder visual hidden states of one prefill."""
    visual = rt.model.model.visual
    store, handles = {}, []
    for k in vit_blocks:
        handles.append(visual.blocks[k - 1].register_forward_hook(
            lambda m, i, o, k=k: store.__setitem__(f'vit_{k}', (o[0] if isinstance(o, tuple) else o).float())))
    handles.append(visual.merger.register_forward_hook(lambda m, i, o: store.__setitem__('merger', o.float())))
    batch = qr.build(rt, [im], 'Describe the image.')
    sp = qr.image_spans(rt, batch)[0]
    try:
        with qr.capture(rt, list(dec_layers), to_cpu=False) as st:
            rt.model(**batch, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    for l in dec_layers:
        store[f'dec_{l}'] = st[l][sp['start']:sp['end']]
    return store


def square512(im: Image.Image) -> Image.Image:
    """Center square crop resized to 512 x 512 (Lanczos), so that all natural images share one token grid."""
    s = min(im.size)
    x0, y0 = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((x0, y0, x0 + s, y0 + s)).resize((512, 512), Image.LANCZOS)


def stage_posvar(rt, out: Path, args) -> None:
    root = out / 'posvar'
    root.mkdir(parents=True, exist_ok=True)
    conds = posvar_conditions(rt, out)
    vit_blocks = [b for b in POSVAR_VIT if b <= len(rt.model.model.visual.blocks)]
    dec_layers = [l for l in DECODER_SITES if l <= len(rt.layers)]
    groups = {'probe512': [probe_render(probe_case(i, args.seed + 3, 512)) for i in range(args.posvar_n)],
              'natural512': [square512(Image.open(p).convert('RGB')) for _, p in natural_images(args)[:args.posvar_n]]}
    canon = canonical_positions(rt, 32, 32, {})
    canon_mask = torch.zeros(1024, dtype=torch.bool)
    canon_mask[canon] = True
    ras = vl.to_raster(torch.arange(1024)[:, None], 32, 32)[:, 0]
    canon_ras = canon_mask[ras]                                        # raster order mask of canonical patches
    merged_mask = canon_ras.view(16, 2, 16, 2).any(3).any(1).flatten()  # merged tokens containing them
    path = root / 'r2.json'
    results = json.loads(path.read_text()) if path.exists() else {}
    for gname, imgs in groups.items():
        for cname, cond in conds.items():
            key = f'{gname}|{cname}'
            if key in results:
                continue
            acc = {}
            t0 = time.time()
            for n, im in enumerate(imgs):
                st = run_condition(rt, cond, lambda: posvar_forward(rt, im, vit_blocks, dec_layers))
                for site, x in st.items():
                    if site not in acc:
                        acc[site] = hn.PosVar(*x.shape, device=x.device)
                    acc[site].add(x, n)
            res = {}
            for site, pv in acc.items():
                excl = canon_mask if site.startswith('vit') else merged_mask
                res[site] = {'all': pv.r2(), 'excl_canonical': pv.r2(excl)}
            results[key] = res
            path.write_text(json.dumps(results, indent=1))
            print('posvar', key, f'{time.time() - t0:.0f}s', flush=True)
    # 2.1 single-image distance readout: RMS distance of each cell to the median cell (6x6 person grid).
    rpath = root / 'readout.json'
    readout = json.loads(rpath.read_text()) if rpath.exists() else {}
    from analysis_tools.representation.visual_detail import cell_features
    for cname, cond in conds.items():
        if cname in readout:
            continue
        rows = []
        for gap in (4, 8):
            for target in [None] + list(range(36)):
                im = figure_image(target, gap if target is not None else 0)
                st = run_condition(rt, cond, lambda: posvar_forward(rt, im, [18], [16]))
                rec = {'gap': gap, 'target': target}
                for site, x in (('vit_18', st['vit_18']), ('merger', st['merger']), ('dec_16', st['dec_16'])):
                    if site == 'vit_18':
                        grid = vl.to_raster(x, 24, 24).view(24, 24, -1)
                    else:
                        grid = x.view(12, 12, -1)
                    cells = cell_features(grid.cpu())
                    score = (cells - cells.median(0).values).square().mean(-1).sqrt()
                    rec[site] = score.tolist()
                rows.append(rec)
                if target is None and gap == 8:
                    pass
        readout[cname] = rows
        rpath.write_text(json.dumps(readout))
        print('readout', cname, flush=True)
    write_meta(root, rt, args, 'posvar', {'vit_blocks': vit_blocks, 'decoder_layers': dec_layers,
                                          'canonical_patches': int(canon_mask.sum())})


# ----------------------------------------------------------------------------- stage: ablation

def ablation_cases(args) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    out = []
    # 3.5 single red disc, 36 grid positions x 2 diameters x point/bbox.
    for d in (32, 64):
        for i, center in enumerate(cr.grid_centers(6)):
            for fmt in ('point', 'bbox'):
                out.append({'id': f'disc_d{d}_p{i:02d}_{fmt}', 'task': 'disc', 'format': fmt, 'center': center, 'diameter': d,
                            'prompt': cr.DISC_QUERY + cr.SUFFIX[fmt], 'kind': 'generate'})
    main = [c for c in read_jsonl(decomposition_root(args) / 'cases.jsonl')
            if c['stage'] == 'main' and c['wording'] == 'primary']

    def strat(pool, n, key):
        groups = defaultdict(list)
        for c in pool:
            groups[key(c)].append(c)
        per = max(1, n // len(groups))
        picked = []
        for g in sorted(groups):
            idx = rng.choice(len(groups[g]), size=min(per, len(groups[g])), replace=False)
            picked += [groups[g][int(i)] for i in idx]
        return picked
    t2 = strat([c for c in main if c['task'] == 'T2'], args.n_binary, lambda c: (c['variant'], c['grid']))
    t4rc = strat([c for c in main if c['task'] == 'T4' and c['format'] == 'row_column'], args.n_locate, lambda c: c['grid'])
    t4pt = strat([c for c in main if c['task'] == 'T4' and c['format'] == 'point'], args.n_locate, lambda c: c['grid'])
    t3 = [c for c in read_jsonl(args.decomposition_supplement / 'cases_t3full.jsonl')]
    t3box = strat([c for c in t3 if c['mode'] == 'box'], args.n_binary, lambda c: (c['variant'], c['grid']))
    t3rc = strat([c for c in t3 if c['mode'] == 'rowcol'], args.n_binary, lambda c: (c['variant'], c['grid']))
    for c in t2 + t3box + t3rc:
        tag = 'T2' if c['task'] == 'T2' else f"T3_{c['mode']}"
        out.append({**c, 'task_tag': tag, 'kind': 'yes_no', 'stage': 'main'})
    for c in t4rc + t4pt:
        out.append({**c, 'task_tag': f"T4_{c['format']}", 'kind': 'generate'})
    if args.with_explicit:
        ex = [c for c in read_jsonl(decomposition_root(args) / 'cases.jsonl') if c['stage'] == 'explicit'
              and c['task'] == 'T4' and c['format'] == 'row_column']
        for c in ex:
            out.append({**c, 'task_tag': 'T4_row_column_explicit', 'kind': 'generate_long'})
    for c in natural_cases(args):
        out.append({**c, 'task': 'refcoco', 'task_tag': f"refcoco_{c['format']}", 'kind': 'generate'})
    for c in out:
        c.setdefault('task_tag', c.get('task'))
    return out


def case_image(args, c: dict) -> Image.Image:
    if c.get('task') == 'disc':
        return cr.disc_image(tuple(c['center']), c['diameter'])[0]
    if c.get('task') == 'refcoco':
        return Image.open(c['image_path']).convert('RGB')
    return load_case_image(args, c)


def score_record(c: dict, rec: dict) -> dict:
    if c.get('task') == 'disc':
        _, truth = cr.disc_image(tuple(c['center']), c['diameter'])
        return cr.score_disc(truth, c['format'], rec['response'])
    if c.get('task') == 'refcoco':
        return score_refcoco(c['truth'], c['format'], rec['response'])
    return dec.score_case(c, rec, diagnostic=False)


def ablation_conditions(rt, out_root: Path, names: list[str]) -> dict:
    neurons = tuple(register_neurons(out_root))
    all_c = {'base': hn.Ablation(), 'hn_mean': hn.Ablation('hn_mean'), 'hn_zero': hn.Ablation('hn_zero'),
             'rand_mean0': hn.Ablation('rand_mean', seed=0), 'rand_mean1': hn.Ablation('rand_mean', seed=1),
             'neuron_zero': hn.Ablation('neuron_zero', neurons=neurons),
             'hn_mean_persist': hn.Ablation('hn_mean', persist=True)}
    return {n: all_c[n] for n in names}


def pixel_cost(im: Image.Image) -> int:
    return im.width * im.height


def stage_ablation(rt, out: Path, args) -> None:
    root = out / 'ablation'
    root.mkdir(parents=True, exist_ok=True)
    cpath = root / 'cases.jsonl'
    if not cpath.exists():
        cases = ablation_cases(args)
        cpath.write_text(''.join(json.dumps(c, sort_keys=True) + '\n' for c in cases))
    cases = read_jsonl(cpath)
    if args.task_filter:
        cases = [c for c in cases if c['task_tag'] in args.task_filter]
    conds = ablation_conditions(rt, out, args.conditions)
    sizes = {}
    for c in cases:
        im = case_image(args, c)
        sizes[c['id']] = pixel_cost(im)
    for cname, ab in conds.items():
        path = root / f'records_{cname}.jsonl'
        done = done_ids(path)
        todo = [c for c in cases if c['id'] not in done and
                (c['kind'] != 'generate_long' or cname in args.explicit_conditions)]
        todo.sort(key=lambda c: (c['kind'], sizes[c['id']]))
        t0 = time.time()
        i = 0
        while i < len(todo):
            kind = todo[i]['kind']
            budget = args.pixel_budget if kind != 'generate_long' else args.pixel_budget // 2
            batch = [todo[i]]
            j = i + 1
            while j < len(todo) and todo[j]['kind'] == kind and len(batch) < args.batch_size and \
                    sizes[todo[j]['id']] * (len(batch) + 1) <= budget:
                batch.append(todo[j])
                j += 1
            items = [([case_image(args, c)], c['prompt']) for c in batch]
            with hn.vit_ablation(rt, ab) as log:
                if kind == 'yes_no':
                    res = hn.yes_no_batch(rt, items)
                else:
                    mnt = args.max_new_tokens if kind == 'generate' else args.max_new_tokens_long
                    res = qr.generate_batch(rt, items, mnt)
                    for r in res:
                        r['response'] = r.pop('text')
            per_img = defaultdict(int)
            for e in log:
                if e['block'] == ab.block:
                    per_img[e['image']] = e['n_hn']
            rows = []
            for k, (c, r) in enumerate(zip(batch, res)):
                rows.append({'id': c['id'], 'task_tag': c['task_tag'], 'condition': cname, **r, 'n_ablated': per_img.get(k, 0),
                             'batch_size': len(batch), 'score': score_record(c, r)})
            append_jsonl(path, rows)
            i = j
            if (i // max(1, len(batch))) % 10 == 0:
                print(f'ablation {cname} {i}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
        print(f'ablation {cname} done {time.time() - t0:.0f}s', flush=True)
    write_meta(root, rt, args, 'ablation', {'conditions': {k: v.as_dict() for k, v in conds.items()},
                                            'pixel_budget': args.pixel_budget, 'batch_size': args.batch_size})


def stage_spot(rt, out: Path, args) -> None:
    root = out / 'ablation'
    cases = {c['id']: c for c in read_jsonl(root / 'cases.jsonl')}
    base = {r['id']: r for r in read_jsonl(root / 'records_base.jsonl')}
    gen = [i for i in base if cases[i]['kind'] == 'generate']
    rng = np.random.default_rng(args.seed)
    pick = [gen[int(k)] for k in rng.choice(len(gen), size=min(args.n_spot, len(gen)), replace=False)]
    path = root / 'spotcheck.jsonl'
    done = done_ids(path)
    for i in pick:
        if i in done:
            continue
        c = cases[i]
        r = qr.generate(rt, qr.build(rt, [case_image(args, c)], c['prompt']), args.max_new_tokens)
        rec = {'response': r['text'], 'finish_reason': r['finish_reason']}
        s1, s0 = score_record(c, rec), base[i]['score']
        append_jsonl(path, [{'id': i, 'single': r['text'], 'batched': base[i]['response'],
                             'text_equal': r['text'].strip() == base[i]['response'].strip(),
                             'correct_equal': bool(s1.get('correct')) == bool(s0.get('correct'))}])
    write_meta(root, rt, args, 'spot', {'n': len(pick)})


STAGES = {'neurons': stage_neurons, 'survey': stage_survey, 'source': stage_source, 'attention': stage_attention,
          'posvar': stage_posvar, 'ablation': stage_ablation, 'spot': stage_spot}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--decomposition', type=Path, required=True, help='experiment 1.3 run directory')
    p.add_argument('--decomposition-supplement', type=Path, required=True, help='experiment 1.3 supplement (T3 revised)')
    p.add_argument('--refcoco-cases', type=Path, required=True, help='experiment 3.5 refcoco directory (cases.json)')
    p.add_argument('--probe-canvases', type=lambda s: [int(x) for x in s.split(',')], default=[384, 512, 640])
    p.add_argument('--probe-n', type=int, default=160)
    p.add_argument('--grid-n', type=int, default=24)
    p.add_argument('--natural-n', type=int, default=300)
    p.add_argument('--marked-n', type=int, default=48)
    p.add_argument('--decoder-n', type=int, default=48)
    p.add_argument('--n-random', type=int, default=16)
    p.add_argument('--max-hn', type=int, default=16)
    p.add_argument('--source-n', type=int, default=16)
    p.add_argument('--attention-n', type=int, default=6)
    p.add_argument('--posvar-n', type=int, default=160)
    p.add_argument('--n-binary', type=int, default=240)
    p.add_argument('--n-locate', type=int, default=200)
    p.add_argument('--with-explicit', type=int, default=1)
    p.add_argument('--conditions', type=lambda s: s.split(','), default=['base'])
    p.add_argument('--explicit-conditions', type=lambda s: s.split(','), default=['base'],
                   help='conditions that also run the long think-then-answer cases')
    p.add_argument('--task-filter', type=lambda s: [x for x in s.split(',') if x], default=[])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--pixel-budget', type=int, default=16 * 512 * 512)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--max-new-tokens-long', type=int, default=1024)
    p.add_argument('--n-spot', type=int, default=20)
    args = p.parse_args()
    out = args.output / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    for stage in args.stages.split(','):
        t0 = time.time()
        STAGES[stage](rt, out, args)
        print(f'stage {stage} finished in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
