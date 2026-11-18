"""Experiment 7.2 round 2 stage drivers (results: docs/analysis/application/results_2026-11-11_located_finetune_r2.md).

Stages (comma list via --stages):
  link      (CPU) run directory: links to the 7.2 training / test files and to the cited 7.2 adapters;
  restore   per-layer restoration: in a run with a trained adapter, the residual at one token group (question tokens,
            the first image token, all image tokens, the image tokens except the first, the target tokens) is
            replaced after layer l by its value in the base-model run of the same input; teacher-forced digits;
  qswap     donor swap: the residual at one token group after layer l is replaced by its value for another input
            whose coordinate has the same hundreds digit and a tens digit shifted by 5 (same adapter);
  prompts   greedy coordinate answers under prompt variants (paraphrase, Chinese instruction, question placed before
            the image) on the horizontal sweep, 20x20 grids and chart/UI items of the 3.13 test split;
  train / coord / digits / ordinal / general   the 7.2 stages (coord_locate_run) for the round-2 conditions.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.application import coord_locate as cl
from analysis_tools.application import coord_locate_run as clr
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_value_r2 as c2
from analysis_tools.position import readout_prior_r2_run as r2r
from analysis_tools.position import targeted_finetune as tf
from analysis_tools.position import targeted_finetune_r2 as t2
from analysis_tools.position import targeted_finetune_r2_run as tr2
from analysis_tools.position import targeted_finetune_run as tfr
from analysis_tools.position.coordinate_readout_run import append_jsonl, read_jsonl

ROUND1_LINKS = ('train.jsonl', 'test.jsonl', 'train_fix.jsonl')
CITED_ADAPTERS = ('mlp1_4_img', 'mlp1_4_first', 'mlp1_4_text', 'mlp1_4_nofirst', 'down1_4_first_r1', 'mlp1_4_img_fix')
RESTORE_LAYERS = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20, 22, 24, 28)
SWAP_LAYERS = (4, 8, 12, 16, 19, 20, 24)
# Round-2 conditions are registered in addition to the 7.2 table (coord_locate.CONDITIONS_72). Their definitions are
# written in the pre-registration file of the round-2 run directory before training (see the results document).
CONDITIONS_R2: dict[str, dict] = {
    # Pre-registered round-2 LoRA condition: layer-1 MLP, first image token only (393 k parameters).
    'mlp1_first': {'layers': (1,), 'modules': cl.MLP_NAMES, 'positions': 'first'},
}
VECTOR_CONDITIONS = {'vec_first_L1': 1, 'vec_first_L1_lr1e2': 1}
VECTOR_LR = {'vec_first_L1_lr1e2': 1e-2}            # per-condition Adam lr (default --vec-lr)        # learned residual vector at the first image token after layer <value>


def register_r2() -> None:
    cl.register(CONDITIONS_R2)


# ----------------------------------------------------------------------------- link

def stage_link(out: Path, args) -> None:
    src = Path(args.round1_run)
    (out / 'adapters').mkdir(parents=True, exist_ok=True)
    for name in ROUND1_LINKS:
        dst = out / name
        if not dst.exists():
            os.symlink((src / name).resolve(), dst)
    for cond in CITED_ADAPTERS:
        dst = out / 'adapters' / f'{cond}.pt'
        if not dst.exists():
            os.symlink((src / 'adapters' / f'{cond}.pt').resolve(), dst)
    print('linked', flush=True)


# ----------------------------------------------------------------------------- restore / swap

def site_positions(site: str, r: dict) -> list[int]:
    s, e = r['span']
    if site == 'question':
        return r['question']
    if site == 'system':
        return list(range(0, s))
    if site == 'first':
        return [s]
    if site == 'image':
        return list(range(s, e))
    if site == 'image_nofirst':
        return list(range(s + 1, e))
    if site == 'target':
        return list(r['target'])
    raise ValueError(site)


@torch.inference_mode()
def capture_sites(rt: qr.Runtime, chunk: list[dict], site: str, layers: tuple[int, ...]) -> dict:
    """{layer: list over rows of (n_site, D) bf16 GPU tensors} of the residual after each layer at the site."""
    X, P = c2.pad_batch([r['x'] for r in chunk], [r['p'] for r in chunk])
    store, handles = {}, []
    for l in layers:
        def hook(m, a, o, l=l):
            h = o[0] if isinstance(o, tuple) else o
            store[l] = [h[i, site_positions(site, r)].clone() for i, r in enumerate(chunk)]
        handles.append(rt.layers[l - 1].register_forward_hook(hook))
    try:
        rt.model(inputs_embeds=X, position_ids=P, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return store


def set_edit(site: str, values: list[torch.Tensor]):
    def fn(h, chunk):
        h = h.clone()
        for i, r in enumerate(chunk):
            h[i, site_positions(site, r)] = values[i].to(h.device, h.dtype)
        return h
    return fn


RESTORE_PLAN = {'mlp1_4_text': ('question', 'system'), 'mlp1_4_first': ('first', 'image_nofirst', 'target'),
                'mlp1_4_img': ('first', 'image_nofirst', 'image', 'target')}
SWAP_PLAN = {'mlp1_4_text': ('question',), 'mlp1_4_first': ('first', 'image_nofirst'),
             'mlp1_4_img': ('first', 'image_nofirst')}


def stage_restore(rt: qr.Runtime, out: Path, args) -> None:
    d = out / 'restore'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'restore.jsonl'
    done = {(r['adapter'], r['config']) for r in read_jsonl(path)} if path.exists() else set()
    ua = Namespace(round1=args.round1, output=args.finetune_run)
    rows = r2r.build(rt, tr2.disc_units(ua, 2, ('sweep_h', 'sweep_v')))
    t0 = time.time()
    for cond in args.conditions.split(','):
        lora, kind = r2r.load_adapter(rt, Namespace(adapter_run=str(out)), cond)
        try:
            results: dict = {}
            configs = [('none', None, None)] + [(f'{site}@{l}', site, l) for site in RESTORE_PLAN[cond]
                                                 for l in RESTORE_LAYERS]
            configs = [c for c in configs if (cond, c[0]) not in done]
            if not configs:
                continue
            for chunk in tr2.batches(rows, args.batch):
                base_vals = {}
                lora.pos_mask = torch.zeros(len(chunk), max(r['length'] for r in chunk))
                for site in RESTORE_PLAN[cond]:
                    base_vals[site] = capture_sites(rt, chunk, site, RESTORE_LAYERS)
                r2r.set_mask(lora, kind, chunk)
                for name, site, l in configs:
                    edits = [] if site is None else [(l, set_edit(site, base_vals[site][l]))]
                    results.setdefault(name, []).append(tr2.light_forward(rt, chunk, edits=edits)['logp'])
            for name, parts in results.items():
                lp = np.concatenate(parts)
                append_jsonl(path, [{'adapter': cond, 'config': name, 'unit': r['unit'].id, 'family': r['unit'].family,
                                     'truth': r['unit'].truth, 'logp0': lp[i, 0].tolist(), 'logp1': lp[i, 1].tolist()}
                                    for i, r in enumerate(rows)])
            print(f'restore {cond}: {len(results)} configs ({time.time() - t0:.0f}s)', flush=True)
        finally:
            lora.detach()


def stage_qswap(rt: qr.Runtime, out: Path, args) -> None:
    d = out / 'qswap'
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'qswap.jsonl'
    done = {(r['adapter'], r['config']) for r in read_jsonl(path)} if path.exists() else set()
    ua = Namespace(round1=args.round1, output=args.finetune_run)
    rows = r2r.build(rt, tr2.disc_units(ua, 2, ('sweep_h',)))
    truth = np.array([r['unit'].truth for r in rows])
    donor = r2r.donor_index(truth)
    t0 = time.time()
    for cond in args.conditions.split(','):
        lora, kind = r2r.load_adapter(rt, Namespace(adapter_run=str(out)), cond)
        try:
            for site in SWAP_PLAN[cond]:
                if all((cond, f'{site}@{l}') in done for l in SWAP_LAYERS):
                    continue
                vals = {l: [] for l in SWAP_LAYERS}
                for chunk in tr2.batches(rows, args.batch):
                    r2r.set_mask(lora, kind, chunk)
                    c = capture_sites(rt, chunk, site, SWAP_LAYERS)
                    for l in SWAP_LAYERS:
                        vals[l] += [v.cpu() for v in c[l]]
                for l in SWAP_LAYERS:
                    name = f'{site}@{l}'
                    if (cond, name) in done:
                        continue
                    lps = []
                    for bi, chunk in enumerate(tr2.batches(rows, args.batch)):
                        r2r.set_mask(lora, kind, chunk)
                        idx = donor[bi * args.batch: bi * args.batch + len(chunk)]
                        lps.append(tr2.light_forward(rt, chunk, edits=[(l, set_edit(site, [vals[l][j] for j in idx]))])['logp'])
                    lp = np.concatenate(lps)
                    append_jsonl(path, [{'adapter': cond, 'config': name, 'unit': r['unit'].id, 'truth': float(truth[i]),
                                         'donor_truth': float(truth[donor[i]]), 'logp0': lp[i, 0].tolist(),
                                         'logp1': lp[i, 1].tolist()} for i, r in enumerate(rows)])
                print(f'qswap {cond} {site} ({time.time() - t0:.0f}s)', flush=True)
        finally:
            lora.detach()


# ----------------------------------------------------------------------------- prompt variants

SPLIT = ' Give the center of its bounding box.'
VARIANTS = {
    'para': ' Where is its center? Answer in the form [{"point_2d": [x, y]}], with x and y scaled to the range 0-1000 '
            '(x grows to the right, y grows downward). Reply with the JSON and nothing else.',
    'zh': ' 请给出它的边界框中心点。坐标归一化到 0-1000（x 向右增大，y 向下增大），只输出 [{"point_2d": [x, y]}]，不要解释。',
    'qfirst': None,          # original wording, question placed before the image in the user turn
}
PROMPT_FAMILIES = ('sweep_h', 'grid20', 'chart_ui')


def variant_prompt(prompt: str, variant: str) -> str:
    if variant == 'qfirst':
        return prompt
    if SPLIT not in prompt:
        raise ValueError(f'Unexpected prompt: {prompt[:80]}')
    return prompt.split(SPLIT)[0] + VARIANTS[variant]


@torch.inference_mode()
def generate_ordered(rt: qr.Runtime, items: list[tuple[Image.Image, str]], max_new: int, text_first: bool) -> list[dict]:
    """qr.generate_batch with the text placed before the image when text_first (one image per conversation)."""
    if not text_first:
        return qr.generate_batch(rt, [([im], t) for im, t in items], max_new)
    texts = []
    for _, t in items:
        chat = [{'role': 'user', 'content': [{'type': 'text', 'text': t}, {'type': 'image'}]}]
        texts.append(rt.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False))
    images = [im for im, _ in items]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist()):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError('Processor resized an image')
    o = rt.model.generate(**batch, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    res = []
    for row in o[:, batch['input_ids'].shape[1]:]:
        ids = row.tolist()
        end = next((i + 1 for i, t in enumerate(ids) if t in eos), None)
        new = ids[:end] if end is not None else [t for t in ids if t != rt.tokenizer.pad_token_id]
        res.append({'text': rt.tokenizer.decode(new, skip_special_tokens=True),
                    'finish_reason': 'eos' if end is not None else 'length'})
    return res


def stage_prompts(rt: qr.Runtime, out: Path, args, vector=None) -> None:
    samples = [s for s in tf.read_samples(out / 'test.jsonl') if s.family in PROMPT_FAMILIES and s.fmt == 'point']
    for cond in args.conditions.split(','):
        d = out / 'eval' / cond
        d.mkdir(parents=True, exist_ok=True)
        path = d / 'prompts.jsonl'
        done = {(r['variant'], r['id']) for r in read_jsonl(path)} if path.exists() else set()
        lora = clr.attach_trained(rt, out, cond) if cond != 'base' and vector is None else None
        t0 = time.time()
        try:
            for variant in VARIANTS:
                todo = [s for s in samples if (variant, s.id) not in done]
                groups: dict = {}
                for s in todo:
                    groups.setdefault(Image.open(s.image).size, []).append(s)
                for size, group in sorted(groups.items()):
                    bs = max(1, args.token_budget // ((size[0] // 32) * (size[1] // 32) + 200))
                    for i in range(0, len(group), bs):
                        chunk = group[i:i + bs]
                        outs = generate_ordered(rt, [(Image.open(s.image).convert('RGB'), variant_prompt(s.prompt, variant))
                                                     for s in chunk], args.max_new_tokens, variant == 'qfirst')
                        append_jsonl(path, [{'variant': variant, 'id': s.id, 'family': s.family, 'text': o['text'],
                                             'finish_reason': o['finish_reason'], 'batch_size': len(chunk),
                                             **tf.score(s, o['text'])} for s, o in zip(chunk, outs)])
                print(f'prompts {cond} {variant} ({time.time() - t0:.0f}s)', flush=True)
        finally:
            if lora is not None:
                lora.detach()


# ----------------------------------------------------------------------------- learned vector at the first image token

class FirstTokenVector(torch.nn.Module):
    """A learned vector v (D,) added to the residual after decoder layer `layer` at the first image token of every
    row. With input_ids the position is found from the image-token id (prefill only: decoding steps contain no image
    token); with inputs_embeds callers use tr2.light_forward edits (r2r.add_edit('first', v))."""

    def __init__(self, rt: qr.Runtime, layer: int):
        super().__init__()
        self.layer = layer
        self.v = torch.nn.Parameter(torch.zeros(rt.model.config.text_config.hidden_size, device=rt.device))
        self._pos = None
        self._handles = []
        self._rt = rt

    def attach(self) -> None:
        img = self._rt.model.config.image_token_id

        def pre(module, args, kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            if ids is None:
                self._pos = None
                return
            m = ids == img
            self._pos = torch.where(m.any(1), m.float().argmax(1), torch.full_like(m[:, 0], -1, dtype=torch.long))

        def post(module, a, o):
            if self._pos is None or (self._pos < 0).all():
                return o
            h = o[0] if isinstance(o, tuple) else o
            add = torch.zeros_like(h)
            rows = torch.nonzero(self._pos >= 0).flatten()
            add[rows, self._pos[rows]] = self.v.to(h.dtype)
            h = h + add
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        self._handles = [self._rt.model.model.register_forward_pre_hook(pre, with_kwargs=True),
                         self._rt.layers[self.layer - 1].register_forward_hook(post)]

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def load_vector(rt: qr.Runtime, out: Path, cond: str) -> FirstTokenVector:
    st = torch.load(out / 'adapters' / f'{cond}.pt', map_location='cpu', weights_only=False)
    vec = FirstTokenVector(rt, st['layer'])
    vec.v.data.copy_(st['v'].to(rt.device))
    return vec


def stage_vtrain(rt: qr.Runtime, out: Path, args) -> None:
    """7.2 trainer loop (same data, steps, batch, schedule) for a FirstTokenVector (Adam, lr args.vec_lr)."""
    import math
    for cond in args.conditions.split(','):
        apath = out / 'adapters' / f'{cond}.pt'
        if apath.exists():
            print(f'{apath} exists', flush=True)
            continue
        samples = tf.read_samples(out / 'train.jsonl')
        torch.manual_seed(tfr.SEED)
        vec = FirstTokenVector(rt, VECTOR_CONDITIONS[cond])
        vec.attach()
        model = rt.model
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        model.config.use_cache = False
        lr0 = VECTOR_LR.get(cond, args.vec_lr)
        opt = torch.optim.Adam([vec.v], lr=lr0)
        rng = np.random.default_rng(tfr.SEED)
        order = rng.permutation(len(samples))
        n_steps = len(order) // 16
        warm = max(1, int(0.05 * n_steps))
        lr_at = lambda st: lr0 * (st + 1) / warm if st < warm else \
            lr0 * 0.5 * (1 + math.cos(math.pi * (st - warm) / max(1, n_steps - warm)))
        prompt_fn = lambda text: qr.prompt_text(rt, 1, text)
        log = out / 'adapters' / f'{cond}_log.jsonl'
        t0 = time.time()
        try:
            for step in range(n_steps):
                items = [samples[int(i)] for i in order[step * 16:(step + 1) * 16]]
                sizes = [Image.open(x.image).size for x in items]
                tokens = [(w // 32) * (h // 32) + 150 for w, h in sizes]
                micro, cur = [], []
                for i in np.argsort(tokens):
                    if cur and (len(cur) + 1) * max(tokens[j] for j in cur + [int(i)]) > 4096:
                        micro.append(cur)
                        cur = []
                    cur.append(int(i))
                micro.append(cur)
                for g in opt.param_groups:
                    g['lr'] = lr_at(step)
                opt.zero_grad(set_to_none=True)
                tot = 0.0
                for mb in micro:
                    ss = [items[i] for i in mb]
                    batch = tf.encode_batch(rt.processor, prompt_fn, ss, [Image.open(x.image).convert('RGB') for x in ss])
                    per, dce, _ = tf.batch_loss(model, batch, rt.device)
                    (per.sum() / len(items)).backward()
                    tot += float(per.sum())
                opt.step()
                append_jsonl(log, [{'step': step, 'loss': tot / len(items), 'v_norm': float(vec.v.norm()),
                                    'lr': lr_at(step), 'time': time.time() - t0}])
                if step % 20 == 0:
                    print(f'vtrain {cond} step {step} loss {tot / len(items):.3f} |v| {float(vec.v.norm()):.2f}', flush=True)
        finally:
            model.gradient_checkpointing_disable()
            model.eval()
            vec.detach()
        torch.save({'condition': cond, 'layer': vec.layer, 'v': vec.v.detach().cpu(), 'n_params': vec.v.numel(),
                    'lr': lr0, 'n_steps': n_steps, 'train_seconds': time.time() - t0,
                    'gpu': torch.cuda.get_device_name(0)}, apath)
        print(f'saved {apath} ({time.time() - t0:.0f}s)', flush=True)


def stage_veval(rt: qr.Runtime, out: Path, args) -> None:
    """Digits (teacher-forced, as coord_locate_run.stage_digits), coordinate answers and row / column answers for
    vector conditions."""
    ua = Namespace(round1=args.round1, output=args.finetune_run)
    for cond in args.conditions.split(','):
        vec = load_vector(rt, out, cond)
        d = tfr.cond_dir(out, cond)
        path = d / 'digits.jsonl'
        if not path.exists():
            rows = r2r.build(rt, tr2.disc_units(ua, 2, ('sweep_h', 'sweep_v', 'refcoco')))
            res = []
            for chunk in tr2.batches(rows, args.batch):
                lp = tr2.light_forward(rt, chunk, edits=[(vec.layer, r2r.add_edit('first', vec.v.detach()))])['logp']
                res += [{'unit': r['unit'].id, 'family': r['unit'].family, 'truth': r['unit'].truth, 'k': r['unit'].ks[0],
                         'logp0': lp[i, 0].tolist(), 'logp1': lp[i, 1].tolist()} for i, r in enumerate(chunk)]
            append_jsonl(path, res)
        rpath = d / 'resid.npz'
        if not rpath.exists():
            rows = r2r.build(rt, tr2.disc_units(ua, 1, ('sweep_h',)) + tr2.disc_units(ua, 2, ('sweep_v',)))
            layers = (8, 12, 16, 20, 24, 28)
            parts = []
            for chunk in tr2.batches(rows, args.batch):
                o = tr2.light_forward(rt, chunk, layers, edits=[(vec.layer, r2r.add_edit('first', vec.v.detach()))])
                parts.append(np.stack([o['resid'][l].numpy() for l in layers], 1).astype(np.float16))   # (B, nL, 3, D)
            np.savez(rpath, resid=np.concatenate(parts), layers=np.array(layers), sites=np.array(['q0', 'q1', 'target']),
                     truth=np.array([r['unit'].truth for r in rows]), family=np.array([r['unit'].family for r in rows]))
        dp = argparse.Namespace(**vars(clr_defaults()), output=str(out), finetune_run=args.finetune_run,
                                round1=args.round1, features=args.features, data_dir=args.data_dir,
                                vcot_gqa=args.vcot_gqa, mmvp_csv=args.mmvp_csv, condition=cond)
        vec.attach()
        try:
            tfr.stage_coord(rt, out, dp)
            clr._ordinal(rt, out, dp)
            stage_prompts(rt, out, argparse.Namespace(**{**vars(args), 'conditions': cond}), vector=vec)
        finally:
            vec.detach()
        print(f'veval {cond} done', flush=True)


# ----------------------------------------------------------------------------- main

LOCAL = {'restore': stage_restore, 'qswap': stage_qswap, 'prompts': stage_prompts, 'vtrain': stage_vtrain,
         'veval': stage_veval}
DELEGATED = ('train', 'coord', 'digits', 'ordinal', 'general')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--conditions', default='mlp1_4_text')
    p.add_argument('--round1-run', required=True, help='7.2 round-1 run directory')
    p.add_argument('--finetune-run', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--token-budget', type=int, default=16000)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--data-dir', default='')
    p.add_argument('--vcot-gqa', default='')
    p.add_argument('--mmvp-csv', default='')
    p.add_argument('--general-n', type=int, default=200)
    p.add_argument('--vec-lr', type=float, default=1e-3)
    p.add_argument('--limit', type=int, default=0, help='smoke test: keep the first N units of every family')
    args, rest = p.parse_known_args()
    if args.limit:
        r2r.limit_units(args.limit)
    register_r2()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in args.stages.split(',') if s]
    if stages == ['link']:
        stage_link(out, args)
        return
    rt = qr.load(args.model)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['model'] = qr.describe(rt)
    info['runs'].append({'stages': stages, 'conditions': args.conditions, 'batch': args.batch,
                         'started': datetime.now(timezone.utc).isoformat(), 'gpu': torch.cuda.get_device_name(0)})
    meta.write_text(json.dumps(info, indent=1))
    for s in stages:
        t0 = time.time()
        if s in LOCAL:
            LOCAL[s](rt, out, args)
        elif s in DELEGATED:
            # Re-use the 7.2 stage functions with their default arguments (3.13 protocol).
            dp = argparse.Namespace(**vars(clr_defaults()), output=str(out), finetune_run=args.finetune_run,
                                    round1=args.round1, features=args.features, data_dir=args.data_dir,
                                    vcot_gqa=args.vcot_gqa, mmvp_csv=args.mmvp_csv)
            for cond in args.conditions.split(','):
                dp.condition = cond
                clr.STAGES[s](rt, out, dp)
        else:
            raise ValueError(f'Unknown stage {s}')
        append_jsonl(out / 'stage_seconds.jsonl', [{'stage': s, 'conditions': args.conditions, 'seconds': time.time() - t0,
                                                    'gpu': torch.cuda.get_device_name(0),
                                                    'finished': datetime.now(timezone.utc).isoformat()}])


def clr_defaults() -> argparse.Namespace:
    """Default arguments of coord_locate_run (training and evaluation settings of 7.2)."""
    return argparse.Namespace(model='Qwen3.5-9B', bv_annotation='', rank=8, alpha=32.0, lr=1e-4, epochs=1,
                              step_samples=16, micro_tokens=4096, token_budget=16000, max_new_tokens=64, general_n=200,
                              general_seed=20261020, general_batch=16, general_max_new=96, batch=24, bv_batch=8,
                              bv_max_new=1024, eval_positions='')


if __name__ == '__main__':
    main()
