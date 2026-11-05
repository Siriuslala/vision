"""Experiment 7.1 stage drivers. Design: docs/analysis/application/design_interp_guided_optimization.md.

Stages (comma list via --stages):
  prepare  (CPU) training mixture (odd-entity search on trained feature pairs + TextVQA train) and evaluation
           sets (odd-entity probe with held-out pairs, grid row/column counting, 3.13 coordinate subset);
  train    LoRA (all decoder linear layers, the 3.13 recipe) under one training condition;
  eval     probe / count / coord answers and the general benchmarks (POPE, TextVQA, MMVP, GQA, text-only) for
           one evaluation condition;
  mech     mechanism read-outs on the odd-entity probe: attention mass of the target tokens on raster-earlier
           and raster-later entities in the full-attention layers, and the contrast-channel norm map;
  spotcheck batched decoding vs `generate` on 20 probe items.

Conditions (CONDITIONS): base, base_bidir (mask only, no training), lora_causal (matched control),
lora_bidir (A), lora_bidir_causaleval (A trained, evaluated with the causal mask), lora_contrast (E),
vcd_all / vcd_digit (C, training-free).
"""
from __future__ import annotations

import argparse
import ast
from argparse import Namespace
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.application import interp_opt as io_
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_run as tfr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry as sa
from analysis_tools.behavior.decomposition import T0_QUERY, T0_SUFFIX, JSON_ONLY

SEED = 20261103
TRAIN_PAIRS = ('tilt', 'length', 'curve', 'sat', 'size')     # qo and closure are held out for evaluation
PROBE_PAIRS = ('qo', 'closure', 'tilt', 'sat')
N_TRAIN = {'odd': 1200, 'vqa': 1200}
TRAIN_GRIDS = (4, 5, 6, 7, 8)
PROBE_GRID = 6
PROBE_EXTRA_FIRST_ROW = 12
N_COUNT = 120
VIT_BLOCKS = tuple(range(6, 17))                              # 1-based ViT blocks for the lora_vit condition
COORD_PICK = {'sweep_h': 2, 'grid10': 1, 'grid20': 1, 'refcoco': 4, 'chart_ui': 2}   # keep every k-th item

CONDITIONS = {
    'base': {},
    'base_bidir': {'mask': 'bidir'},
    'lora_causal': {'train': True},
    'lora_bidir': {'train': True, 'mask': 'bidir'},
    'lora_bidir_causaleval': {'adapter': 'lora_bidir'},
    'lora_contrast': {'train': True, 'contrast': True},
    # Mechanism-located alternative: LoRA only on the ViT MLPs of blocks 6-16, where experiment 2.7 located the
    # rectified presence detectors and the formation of the anomaly code.
    'lora_vit': {'train': True, 'scope': 'vit_mlp'},
    # Second training seeds (LoRA initialisation and sample order) for the two main contrasts.
    'lora_causal_s2': {'train': True, 'seed': 2},
    'lora_vit_s2': {'train': True, 'scope': 'vit_mlp', 'seed': 2},
    'lora_bidir_s2': {'train': True, 'mask': 'bidir', 'seed': 2},
    'lora_causal_s3': {'train': True, 'seed': 3},
    'lora_vit_s3': {'train': True, 'scope': 'vit_mlp', 'seed': 3},
    'lora_vit_bidireval': {'adapter': 'lora_vit', 'mask': 'bidir'},           # ViT adapter, bidirectional mask
    'lora_vit_bidir': {'train': True, 'scope': 'vit_mlp', 'mask': 'bidir'},   # ViT adapter trained under the mask
    'vcd_all': {'vcd': 'all'},
    'vcd_digit': {'vcd': 'digit'},
}


# ----------------------------------------------------------------------------- prepare

def _save(img: Image.Image, root: Path, rel: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return str(p)


def odd_item(root: Path, split: str, k: int, pair: str, level: str, g: int, target: tuple[int, int], variant: str,
             rng: np.random.Generator) -> io_.Item:
    orient = int(rng.integers(0, 4))
    jit = sa.jitter_map(g, g, rng)
    image, meta = sa.pair_images(pair, level, g, target, orient, jit, (variant,))[variant]
    path = _save(image, root, f'images/odd_{split}/{pair}_{level}_g{g}_{k:05d}_{variant}.png')
    ans = io_.ROWCOL_TEMPLATE.format(r=target[0] + 1, c=target[1] + 1) if split == 'train' else ''
    return io_.Item(f'odd_{split}_{pair}_{level}_g{g}_{k:05d}_{variant}', 'odd', split, path, sa.DIRECT_PROMPT, ans,
                    {'target': list(target), 'grid': g}, {'pair': pair, 'level': level, 'variant': variant,
                                                          'orient': orient, 'meta': meta})


def textvqa_items(root: Path, data_dir: Path, n: int, rng) -> list[io_.Item]:
    import pyarrow.parquet as pq
    files = sorted((data_dir / 'TextVQA' / 'data').glob('train-*.parquet'))
    if not files:
        raise FileNotFoundError('TextVQA train parquet files are missing')
    per = math.ceil(n / len(files))
    items = []
    for f in files:
        rows = pq.read_table(f, columns=['image_id', 'question_id', 'question', 'image', 'answers']).to_pylist()
        seen, taken = set(), 0
        for i in rng.permutation(len(rows)):
            r = rows[int(i)]
            if r['image_id'] in seen:
                continue
            answers = ast.literal_eval(r['answers']) if isinstance(r['answers'], str) else list(r['answers'])
            ans, cnt = Counter(a.strip().lower() for a in answers).most_common(1)[0]
            if cnt < 3 or not ans:          # keep questions with a clear majority answer
                continue
            seen.add(r['image_id'])
            img = io_.fit_area(Image.open(io.BytesIO(r['image']['bytes'])))
            path = _save(img, root, f"images/textvqa_train/{r['image_id']}.png")
            items.append(io_.Item(f"vqa_train_{r['question_id']}", 'vqa', 'train', path, r['question'] + io_.SHORT_TRAIN,
                                  ans, {'answers': answers}))
            taken += 1
            if taken >= per:
                break
        del rows
        if len(items) >= n:
            break
    if len(items) < n:
        raise RuntimeError(f'only {len(items)} TextVQA training items')
    return items[:n]


def stage_prepare(out: Path, args) -> None:
    rng = np.random.default_rng(SEED)
    train: list[io_.Item] = []
    for k in range(N_TRAIN['odd']):
        pair = TRAIN_PAIRS[k % len(TRAIN_PAIRS)]
        g = int(rng.choice(TRAIN_GRIDS))
        level = ('subtle', 'salient')[int(rng.integers(2))]
        variant = ('std', 'swap')[(k // len(TRAIN_PAIRS)) % 2]
        target = (int(rng.integers(g)), int(rng.integers(g)))
        train.append(odd_item(out, 'train', k, pair, level, g, target, variant, rng))
    train += textvqa_items(out, Path(args.data_dir), N_TRAIN['vqa'], rng)
    order = rng.permutation(len(train))
    io_.dump_items(out / 'train.jsonl', [train[int(i)] for i in order])
    # Probe: every 6x6 cell once plus extra first-row targets, for std and swap of every probe pair (subtle level,
    # the 2.7 main level); the same target / orientation / jitter for std and swap.
    probe, k = [], 0
    for pair in PROBE_PAIRS:
        targets = [(r, c) for r in range(PROBE_GRID) for c in range(PROBE_GRID)]
        targets += [(0, int(c)) for c in rng.integers(0, PROBE_GRID, PROBE_EXTRA_FIRST_ROW)]
        for target in targets:
            state = int(rng.integers(2**31))
            for variant in ('std', 'swap'):
                probe.append(odd_item(out, 'probe', k, pair, 'subtle', PROBE_GRID, target, variant,
                                      np.random.default_rng(state)))
            k += 1
    io_.dump_items(out / 'probe.jsonl', probe)
    # Grid counting (T0 variation set of experiment 1.6).
    t0 = Path(args.data_dir) / 'stimuli_t0_count_variation_v1'
    rows = [json.loads(l) for l in (t0 / 'metadata.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['canvas'][0] * r['canvas'][1] <= 768 * 768]
    count = []
    for i in sorted(rng.choice(len(rows), N_COUNT, replace=False)):
        r = rows[int(i)]
        count.append(io_.Item(f"count_{r['id']}", 'count', 'test', str(t0 / r['image']), T0_QUERY[0] + T0_SUFFIX + JSON_ONLY,
                              '', {'rows': r['rows'], 'cols': r['cols']}))
    io_.dump_items(out / 'count.jsonl', count)
    # Coordinate subset of the 3.13 held-out test set (ids only; the samples are read from 3.13).
    coord = []
    by_family: dict = {}
    for s in tf.read_samples(Path(args.finetune_run) / 'test.jsonl'):
        by_family.setdefault(s.family, []).append(s)
    for fam, step in COORD_PICK.items():
        coord += [s.id for s in by_family[fam][::step]]
    (out / 'coord_ids.json').write_text(json.dumps(coord))
    info = {'train': Counter(f'{it.task}' for it in train), 'probe': len(probe), 'count': len(count), 'coord': len(coord),
            'seed': SEED, 'train_pairs': TRAIN_PAIRS, 'probe_pairs': PROBE_PAIRS}
    (out / 'prepare.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


# ----------------------------------------------------------------------------- condition context

def _adapter_name(cond: str) -> str | None:
    spec = CONDITIONS[cond]
    if spec.get('train'):
        return cond
    return spec.get('adapter')


def lora_modules(rt: qr.Runtime, scope: str) -> dict:
    """'all_linear': every decoder linear layer (3.13); 'vit_mlp': linear_fc1 / linear_fc2 of ViT blocks VIT_BLOCKS."""
    if scope == 'all_linear':
        return tf.target_modules(rt.layers, 'all_linear')
    if scope == 'vit_mlp':
        blocks = rt.model.model.visual.blocks
        return {f'V{b}.mlp.{n}': getattr(blocks[b - 1].mlp, n) for b in VIT_BLOCKS for n in ('linear_fc1', 'linear_fc2')}
    raise ValueError(scope)


def install(rt: qr.Runtime, out: Path, cond: str, stack: ExitStack, trainable: bool = False) -> dict:
    """Enter every hook of a condition; returns {'lora': LoRA | None, 'contrast': ContrastAdapter | None}."""
    spec = CONDITIONS[cond]
    name = _adapter_name(cond)
    res = {'lora': None, 'contrast': None}
    if spec.get('mask') == 'bidir':
        stack.enter_context(io_.image_bidirectional(rt))
    train_spec = CONDITIONS[name] if name else {}
    if train_spec.get('contrast'):
        d = rt.model.config.text_config.hidden_size
        ad = io_.ContrastAdapter(d, device=rt.device)
        if not trainable:
            ad.load(torch.load(out / 'adapters' / f'{name}_contrast.pt', map_location='cpu'))
        stack.enter_context(io_.contrast_channel(rt, ad))
        res['contrast'] = ad
    if name and not trainable:
        state = torch.load(out / 'adapters' / f'{name}.pt', map_location='cpu', weights_only=False)
        scope = train_spec.get('scope', 'all_linear')
        if state['condition'] != scope:
            raise ValueError(f"adapter {name} was trained on {state['condition']}, expected {scope}")
        lora = tf.LoRA(lora_modules(rt, scope), state['rank'], state['alpha'])
        lora.load(state)
        lora.attach()
        res['lora'] = lora
        stack.callback(lora.detach)
    return res


# ----------------------------------------------------------------------------- train

def stage_train(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    if not CONDITIONS[cond].get('train'):
        raise ValueError(f'{cond} is not a training condition')
    (out / 'adapters').mkdir(exist_ok=True)
    apath = out / 'adapters' / f'{cond}.pt'
    if apath.exists():
        print(f'{apath} exists; skipping training', flush=True)
        return
    items = io_.read_items(out / 'train.jsonl')
    seed = CONDITIONS[cond].get('seed', 0)
    torch.manual_seed(SEED + seed)
    if seed:
        items = [items[int(i)] for i in np.random.default_rng(SEED + seed).permutation(len(items))]
    with ExitStack() as stack:
        hooks = install(rt, out, cond, stack, trainable=True)
        scope = CONDITIONS[cond].get('scope', 'all_linear')
        lora = tf.LoRA(lora_modules(rt, scope), args.rank, args.alpha)
        lora.attach()
        stack.callback(lora.detach)
        params = list(lora.parameters()) + (list(hooks['contrast'].parameters()) if hooks['contrast'] else [])
        model = rt.model
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        model.config.use_cache = False
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        n_steps = min(args.steps, len(items) // args.step_samples)
        warm = max(1, int(0.05 * n_steps))

        def lr_at(s):
            if s < warm:
                return args.lr * (s + 1) / warm
            return args.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_steps - warm)))

        prompt_fn = lambda text: qr.prompt_text(rt, 1, text)
        log = out / 'adapters' / f'{cond}_log.jsonl'
        if log.exists():
            log.unlink()
        t0 = time.time()
        for step in range(n_steps):
            chunk = items[step * args.step_samples:(step + 1) * args.step_samples]
            sizes = [Image.open(it.image).size for it in chunk]
            tokens = [(w // 32) * (h // 32) + 120 for w, h in sizes]
            micro, cur = [], []
            for i in np.argsort(tokens):
                if cur and (len(cur) + 1) * max(tokens[j] for j in cur + [int(i)]) > args.micro_tokens:
                    micro.append(cur)
                    cur = []
                cur.append(int(i))
            micro.append(cur)
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            tot = {'odd': [0.0, 0], 'vqa': [0.0, 0]}
            for mb in micro:
                ss = [chunk[i] for i in mb]
                batch = io_.encode_batch(rt.processor, prompt_fn, ss, [Image.open(s.image).convert('RGB') for s in ss])
                per, _, _ = tf.batch_loss(model, batch, rt.device)
                (per.sum() / len(chunk)).backward()
                for s, v in zip(ss, per.tolist()):
                    tot[s.task][0] += v
                    tot[s.task][1] += 1
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            row = {'step': step, 'loss_odd': tot['odd'][0] / max(1, tot['odd'][1]),
                   'loss_vqa': tot['vqa'][0] / max(1, tot['vqa'][1]), 'lr': lr_at(step), 'grad_norm': float(gn),
                   'micro': len(micro), 'time': time.time() - t0}
            append_jsonl(log, [row])
            if step % 10 == 0:
                print(json.dumps(row), flush=True)
        state = lora.state()
        state.update({'condition': scope, 'train_condition': cond, 'lr': args.lr, 'steps': n_steps,
                      'step_samples': args.step_samples, 'n_params': lora.n_params(), 'delta_norms': lora.delta_norms(),
                      'train_seconds': time.time() - t0, 'runtime': qr.describe(rt)})
        if hooks['contrast'] is not None:
            torch.save(hooks['contrast'].state(), out / 'adapters' / f'{cond}_contrast.pt')
            state['contrast_params'] = sum(p.numel() for p in hooks['contrast'].parameters())
        torch.save(state, apath)
        model.gradient_checkpointing_disable()
        model.eval()
    print(f'saved {apath} ({time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- generation

def vcd_generate_batch(rt: qr.Runtime, items: list, max_new_tokens: int, thinking: bool = False, *,
                       processor_obj: io_.PriorContrastProcessor) -> list[dict]:
    """`qwen_runtime.generate_batch` with a doubled [real; blank] batch and the prior-contrast logits processor.
    Items without an image are decoded normally (the contrast needs an image)."""
    from transformers import LogitsProcessorList
    if any(len(imgs) != 1 for imgs, _ in items):
        if all(len(imgs) == 0 for imgs, _ in items):
            return _orig_generate_batch(rt, items, max_new_tokens, thinking)
        raise ValueError('vcd_generate_batch expects exactly one image per item')
    doubled = list(items) + [([io_.blank_like(imgs[0])], text) for imgs, text in items]
    texts = [qr.prompt_text(rt, 1, text, thinking) for _, text in doubled]
    images = [imgs[0] for imgs, _ in doubled]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    with torch.inference_mode():
        out = rt.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                top_k=None, logits_processor=LogitsProcessorList([processor_obj]))
    eos_ids = rt.model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    pad_id = rt.tokenizer.pad_token_id
    results = []
    for row in out[:len(items), batch['input_ids'].shape[1]:]:
        ids = row.tolist()
        end = next((i + 1 for i, t in enumerate(ids) if t in eos_ids), None)
        new = ids[:end] if end is not None else [t for t in ids if t != pad_id]
        results.append({'text': rt.tokenizer.decode(new, skip_special_tokens=True), 'n_tokens': len(new),
                        'finish_reason': 'eos' if end is not None else 'length'})
    return results


_orig_generate_batch = qr.generate_batch


def make_generator(rt: qr.Runtime, cond: str, args):
    spec = CONDITIONS[cond]
    if not spec.get('vcd'):
        return _orig_generate_batch
    digits = [rt.tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)] if spec['vcd'] == 'digit' else None
    proc = io_.PriorContrastProcessor(args.vcd_alpha, args.vcd_beta, digits)
    return lambda rt_, items, max_new, thinking=False: vcd_generate_batch(rt_, items, max_new, thinking, processor_obj=proc)


def run_items(rt, gen, items: list, budget: int, max_new: int) -> dict:
    """Batched greedy answers grouped by image size; returns {id: output}."""
    groups: dict = {}
    for it in items:
        groups.setdefault(Image.open(it.image).size, []).append(it)
    res = {}
    for size, group in sorted(groups.items()):
        bs = max(1, budget // ((size[0] // 32) * (size[1] // 32) + 200))
        for i in range(0, len(group), bs):
            chunk = group[i:i + bs]
            outs = gen(rt, [([Image.open(it.image).convert('RGB')], it.prompt) for it in chunk], max_new)
            for it, o in zip(chunk, outs):
                res[it.id] = {**o, 'batch_size': len(chunk)}
    return res


def stage_eval(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    d = out / 'eval' / cond
    d.mkdir(parents=True, exist_ok=True)
    gen = make_generator(rt, cond, args)
    budget = args.token_budget // (2 if CONDITIONS[cond].get('vcd') else 1)
    sets = args.sets.split(',')
    with ExitStack() as stack:
        install(rt, out, cond, stack)
        t0 = time.time()
        if 'probe' in sets:
            items = io_.read_items(out / 'probe.jsonl')
            path = d / 'probe.jsonl'
            done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
            todo = [it for it in items if it.id not in done]
            res = run_items(rt, gen, todo, budget, 32)
            append_jsonl(path, [{'id': it.id, 'pair': it.meta['pair'], 'variant': it.meta['variant'],
                                 'target': it.truth['target'], 'text': res[it.id]['text'],
                                 'finish_reason': res[it.id]['finish_reason'], 'batch_size': res[it.id]['batch_size'],
                                 **oe.score({'target': it.truth['target']}, res[it.id])} for it in todo])
            print(f'probe {cond} {time.time() - t0:.0f}s', flush=True)
        if 'count' in sets:
            items = io_.read_items(out / 'count.jsonl')
            path = d / 'count.jsonl'
            if not path.exists():
                res = run_items(rt, gen, items, budget, 32)
                append_jsonl(path, [{'id': it.id, 'truth': it.truth, 'text': res[it.id]['text'],
                                     'finish_reason': res[it.id]['finish_reason'],
                                     **io_.score_count(res[it.id]['text'], it.truth)} for it in items])
            print(f'count {cond} {time.time() - t0:.0f}s', flush=True)
        if 'coord' in sets:
            ids = set(json.loads((out / 'coord_ids.json').read_text()))
            samples = [s for s in tf.read_samples(Path(args.finetune_run) / 'test.jsonl') if s.id in ids]
            path = d / 'coord.jsonl'
            if not path.exists():
                items = [io_.Item(s.id, 'coord', 'test', s.image, s.prompt) for s in samples]
                res = run_items(rt, gen, items, budget, 64)
                append_jsonl(path, [{'id': s.id, 'family': s.family, 'fmt': s.fmt, 'text': res[s.id]['text'],
                                     'finish_reason': res[s.id]['finish_reason'], **tf.score(s, res[s.id]['text'])}
                                    for s in samples])
            print(f'coord {cond} {time.time() - t0:.0f}s', flush=True)
        if 'general' in sets:
            ns = Namespace(**{**vars(args), 'condition': cond})
            qr.generate_batch = gen            # stage_general looks the function up on the module at call time
            try:
                tfr.stage_general(rt, out, ns)
            finally:
                qr.generate_batch = _orig_generate_batch
            print(f'general {cond} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- mechanism read-outs

def target_attention(rt, layers: list[int], captured: dict, tgt: list[int], earlier: list[int], later: list[int],
                     mask: torch.Tensor, below: list[int] | None = None) -> dict:
    """Attention mass (mean over heads and target query tokens) on entity tokens earlier / later in entity raster
    order, on entities in grid rows below the target ('below': all their tokens follow every target token, so the
    causal mask gives them exactly zero) and on the target itself.

    captured[l] = (hidden_states (1, L, d) entering self_attn, (cos, sin)); attention is recomputed from the
    module's own projections (LoRA hooks included), the given (L, L) boolean mask and softmax.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    out = {}
    for l in layers:
        attn = rt.layers[l - 1].self_attn
        h, (cos, sin) = captured[l]
        shp = (*h.shape[:-1], -1, attn.head_dim)
        q, _ = torch.chunk(attn.q_proj(h).view(*h.shape[:-1], -1, attn.head_dim * 2), 2, dim=-1)
        q = attn.q_norm(q.view(shp)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(h).view(shp)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, attn.num_key_value_groups)
        qi = torch.tensor(tgt, device=h.device)
        s = (q[:, :, qi].float() @ k.float().transpose(-1, -2)) * attn.scaling       # (1, heads, |tgt|, L)
        s = s.masked_fill(~mask[qi][None, None].to(s.device), float('-inf'))
        p = s.softmax(-1)[0].mean(0).mean(0)                                           # (L,)
        out[l] = {'earlier': float(p[earlier].sum()), 'later': float(p[later].sum()), 'self': float(p[tgt].sum()),
                  'below': float(p[below].sum()) if below else 0.0}
    return out


def stage_mech(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    d = out / 'eval' / cond
    d.mkdir(parents=True, exist_ok=True)
    items = [it for it in io_.read_items(out / 'probe.jsonl') if it.meta['pair'] in ('qo', 'closure')]
    layers = [4, 8]
    rows = []
    with ExitStack() as stack:
        hooks = install(rt, out, cond, stack)
        bidir = CONDITIONS[cond].get('mask') == 'bidir'
        cap: dict = {}

        def pre(l):
            def f(module, args_, kwargs):
                cap[l] = (kwargs['hidden_states'].detach(), tuple(t.detach() for t in kwargs['position_embeddings']))
            return f
        handles = [rt.layers[l - 1].self_attn.register_forward_pre_hook(pre(l), with_kwargs=True) for l in layers]
        stack.callback(lambda: [h.remove() for h in handles])
        with torch.inference_mode():
            for it in items:
                image = Image.open(it.image).convert('RGB')
                batch = qr.build(rt, [image], it.prompt)
                span = qr.image_spans(rt, batch)[0]
                rt.model(**batch, use_cache=False, logits_to_keep=1)
                ent = oe.entity_tokens(it.meta['meta'])
                t_idx = it.truth['target'][0] * it.truth['grid'] + it.truth['target'][1]
                tgt = [span['start'] + j for j in ent[t_idx]]
                earlier = [span['start'] + j for e in ent[:t_idx] for j in e]
                later = [span['start'] + j for e in ent[t_idx + 1:] for j in e]
                g = it.truth['grid']
                below = [span['start'] + j for e in ent[(it.truth['target'][0] + 1) * g:] for j in e]
                mask = io_.prefill_mask(batch['input_ids'], None, rt.model.config.image_token_id, bidir)[0, 0]
                att = target_attention(rt, layers, cap, tgt, earlier, later, mask, below)
                row = {'id': it.id, 'pair': it.meta['pair'], 'variant': it.meta['variant'], 'target': it.truth['target'],
                       'n_earlier': t_idx, 'attn': {str(k): v for k, v in att.items()}}
                if hooks['contrast'] is not None and hooks['contrast'].last:
                    norms = hooks['contrast'].last[0].numpy()
                    ent_norm = np.array([norms[e].mean() for e in ent])
                    others = np.delete(ent_norm, t_idx)
                    row['contrast_z'] = float((ent_norm[t_idx] - others.mean()) / (others.std() + 1e-8))
                rows.append(row)
    (d / 'mech.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'mech {cond}: {len(rows)} items', flush=True)


def stage_spotcheck(rt: qr.Runtime, out: Path, args) -> None:
    cond = args.condition
    d = out / 'eval' / cond
    rows = {r['id']: r for r in read_jsonl(d / 'probe.jsonl')}
    items = [it for it in io_.read_items(out / 'probe.jsonl') if it.id in rows]
    pick = [items[i] for i in np.random.default_rng(0).choice(len(items), 20, replace=False)]
    res = []
    with ExitStack() as stack:
        install(rt, out, cond, stack)
        for it in pick:
            ref = qr.generate(rt, qr.build(rt, [Image.open(it.image).convert('RGB')], it.prompt), 32)
            a, b = oe.parse_answer(ref['text']), oe.parse_answer(rows[it.id]['text'])
            res.append({'id': it.id, 'batched': rows[it.id]['text'], 'reference': ref['text'], 'same_answer': a == b})
    (d / 'spotcheck.json').write_text(json.dumps(res, indent=1))
    print(f"spotcheck {cond}: same answer {sum(r['same_answer'] for r in res)}/20", flush=True)


VIT_PROBE_BLOCKS = (9, 12, 16)


def stage_vitsal(rt: qr.Runtime, out: Path, args) -> None:
    """Pop-out of the odd entity inside the vision tower (the 2.7 measures, with the condition's ViT adapter).

    For each held-out probe image: per-neuron standardised deviation of the target from the other entities of
    the ViT MLP activations (input of linear_fc2, averaged over the 4 patches of each merged token and over the
    entity's tokens) at VIT_PROBE_BLOCKS, summarised as the mean of the 10 largest positive values; and the
    Mahalanobis salience of the target among the entities at the merger output.
    """
    from analysis_tools.representation import search_asymmetry as sa_
    cond = args.condition
    d = out / 'eval' / cond
    d.mkdir(parents=True, exist_ok=True)
    items = [it for it in io_.read_items(out / 'probe.jsonl') if it.meta['pair'] in ('qo', 'closure')]
    blocks = rt.model.model.visual.blocks
    cap: dict = {}
    rows = []
    with ExitStack() as stack:
        install(rt, out, cond, stack)
        hs = [blocks[b - 1].mlp.linear_fc2.register_forward_pre_hook(
            lambda m, a, b=b: cap.__setitem__(b, a[0].detach().float())) for b in VIT_PROBE_BLOCKS]
        hs.append(rt.model.model.visual.register_forward_hook(
            lambda m, a, o: cap.__setitem__('merger', o.pooler_output.detach().float())))
        stack.callback(lambda: [h.remove() for h in hs])
        with torch.inference_mode():
            for it in items:
                batch = qr.build(rt, [Image.open(it.image).convert('RGB')], it.prompt)
                rt.model.model.get_image_features(batch['pixel_values'], batch['image_grid_thw'])
                ent = oe.entity_tokens(it.meta['meta'])
                t_idx = it.truth['target'][0] * it.truth['grid'] + it.truth['target'][1]
                row = {'id': it.id, 'pair': it.meta['pair'], 'variant': it.meta['variant'], 'target': it.truth['target']}
                for b in VIT_PROBE_BLOCKS:
                    a = cap[b].reshape(-1, 4, cap[b].shape[-1]).mean(1)          # merged-token level
                    h = torch.stack([a[e].mean(0) for e in ent]).cpu().numpy()
                    z = sa_.target_dprime(h, t_idx)
                    row[f'popout_b{b}'] = float(np.sort(z)[-10:].mean())
                m = cap['merger']
                h = torch.stack([m[e].mean(0) for e in ent]).cpu().numpy()
                row['salience_merger'] = sa_.mahalanobis_salience(h, t_idx)
                rows.append(row)
    (d / 'vitsal.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'vitsal {cond}: {len(rows)} items', flush=True)


ODD_LAYERS = (8, 16)
N_HOM = 24


def stage_oddz(rt: qr.Runtime, out: Path, args) -> None:
    """Projection of entity states on the experiment-2.7 r2 oddness direction u (prediction check for (A)).

    u is the leave-one-out direction of 2.7 r2 for the pair (directions.npz, key 'u|dec{L}|{pair}', estimated on
    the base model).  For each image the entity score s_e = (h_e - mean_e' h_e')^T u is standardised within the
    image (z).  Rows are written for the held-out probe images (target z and the maximum z of the other
    entities) and for N_HOM homogeneous images per held-out pair (z of every entity; the start-of-sequence effect
    of 2.7 r2 is the excess z of the first entities).
    """
    cond = args.condition
    d = out / 'eval' / cond
    d.mkdir(parents=True, exist_ok=True)
    dirs = np.load(Path(args.oddness_dirs))
    items = [it for it in io_.read_items(out / 'probe.jsonl') if it.meta['pair'] in ('qo', 'closure')]
    rng = np.random.default_rng(SEED + 7)
    hom = []
    for pair in ('qo', 'closure'):
        for k in range(N_HOM):
            v = ('homA', 'homB')[k % 2]
            orient = int(rng.integers(0, 4))
            img, meta = sa.pair_images(pair, 'subtle', PROBE_GRID, (0, 0), orient, sa.jitter_map(PROBE_GRID, PROBE_GRID, rng),
                                       (v,))[v]
            path = _save(img, out, f'images/hom/{pair}_{k:03d}_{v}.png')
            hom.append(io_.Item(f'hom_{pair}_{k:03d}_{v}', 'hom', 'test', path, sa.DIRECT_PROMPT, '',
                                {'target': None, 'grid': PROBE_GRID}, {'pair': pair, 'variant': v, 'meta': meta}))
    rows = []
    with ExitStack() as stack:
        install(rt, out, cond, stack)
        with torch.inference_mode():
            for it in items + hom:
                batch = qr.build(rt, [Image.open(it.image).convert('RGB')], it.prompt)
                span = qr.image_spans(rt, batch)[0]
                ent = oe.entity_tokens(it.meta['meta'])
                with qr.capture(rt, list(ODD_LAYERS)) as store:
                    rt.model(**batch, use_cache=False, logits_to_keep=1)
                row = {'id': it.id, 'pair': it.meta['pair'], 'variant': it.meta['variant'], 'target': it.truth['target']}
                for l in ODD_LAYERS:
                    u = torch.tensor(dirs[f"u|dec{l}|{it.meta['pair']}"])
                    h = torch.stack([store[l][[span['start'] + j for j in e]].mean(0) for e in ent])
                    sc = ((h - h.mean(0)) @ u).numpy()
                    z = (sc - sc.mean()) / (sc.std() + 1e-8)
                    row[f'z{l}'] = z.tolist()
                rows.append(row)
    (d / 'oddz.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'oddz {cond}: {len(rows)} images', flush=True)


STAGES = {'oddz': stage_oddz, 'vitsal': stage_vitsal, 'train': stage_train, 'eval': stage_eval, 'mech': stage_mech, 'spotcheck': stage_spotcheck}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--condition', default='base', choices=list(CONDITIONS))
    p.add_argument('--sets', default='probe,count,coord,general')
    p.add_argument('--finetune-run', required=True, help='3.13 run directory (coordinate test samples)')
    p.add_argument('--data-dir', default='')
    p.add_argument('--oddness-dirs', default='', help='2.7 r2 directions.npz')
    p.add_argument('--vcot-gqa', default='')
    p.add_argument('--mmvp-csv', default='')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=32.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--step-samples', type=int, default=16)
    p.add_argument('--micro-tokens', type=int, default=3072)
    p.add_argument('--token-budget', type=int, default=16000)
    p.add_argument('--vcd-alpha', type=float, default=1.0)
    p.add_argument('--vcd-beta', type=float, default=0.1)
    p.add_argument('--general-n', type=int, default=200)
    p.add_argument('--general-seed', type=int, default=20261020)
    p.add_argument('--general-batch', type=int, default=16)
    p.add_argument('--general-max-new', type=int, default=96)
    p.add_argument('--max-memory', default='')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(',')
    if stages == ['prepare']:
        stage_prepare(out, args)
        return
    rt = tfr.load_model(args)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages, 'condition': args.condition,
                         'runtime': qr.describe(rt), 'args': vars(args),
                         'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]})
    meta.write_text(json.dumps(info, indent=1))
    for s in stages:
        t0 = time.time()
        STAGES[s](rt, out, args)
        print(f'stage {s} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
